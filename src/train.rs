use rand::prelude::*;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

use crate::matrix::{Matrix, concat_heads, get_head, merge_qkv, softmax, softmax_backward};
use crate::mlp::Mlp;
use crate::optim::{AdamW, clip_grad_norm};
use crate::param::{Param, linear, linear_backward};

const INIT_STD: f32 = 0.02;
const NORM_EPS: f32 = 1e-5;

// ============================================================================
// Config
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub dimensions: usize,
    pub n_heads: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub mlp_hidden: usize,
    pub n_layers: usize,
    /// Longest context the model is fed (training window / generation window).
    pub max_seq_len: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dimensions: 512,
            n_heads: 8,
            vocab_size: 8000,
            head_dim: 64,
            mlp_hidden: 1380,
            n_layers: 4,
            max_seq_len: 256,
        }
    }
}

impl Config {
    pub fn qkv_size(&self) -> usize {
        self.dimensions * 3
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_heads == 0 || self.n_heads * self.head_dim != self.dimensions {
            return Err(format!(
                "n_heads ({}) * head_dim ({}) must equal dimensions ({})",
                self.n_heads, self.head_dim, self.dimensions
            ));
        }
        if self.head_dim % 2 != 0 {
            return Err(format!(
                "head_dim ({}) must be even for RoPE",
                self.head_dim
            ));
        }
        if self.vocab_size == 0 || self.vocab_size > u16::MAX as usize + 1 {
            return Err(format!(
                "vocab_size ({}) must be in 1..=65536",
                self.vocab_size
            ));
        }
        if self.n_layers == 0 || self.mlp_hidden == 0 || self.max_seq_len == 0 {
            return Err("n_layers, mlp_hidden and max_seq_len must be > 0".to_string());
        }
        Ok(())
    }
}

// ============================================================================
// RMSNorm (with learnable gain)
// ============================================================================

#[derive(Clone)]
pub struct RmsNorm {
    pub gain: Param, // [1, dimensions]

    pub cache: Option<RmsNormCache>,
}

#[derive(Clone)]
pub struct RmsNormCache {
    pub x_hat: Matrix,
    pub rms: Vec<f32>,
}

impl RmsNorm {
    pub fn new(dimensions: usize) -> Self {
        Self {
            gain: Param::new(
                Matrix::with_vector(1, dimensions, vec![1.0; dimensions]),
                false,
            ),
            cache: None,
        }
    }

    /// y = x / rms(x) * gain, row-wise.
    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        let cols = x.cols;
        let mut x_hat = x.clone();
        let mut rms = Vec::with_capacity(x.rows);

        for row in x_hat.data.chunks_mut(cols) {
            let sq_sum: f32 = row.iter().map(|v| v * v).sum();
            let r = (sq_sum / cols as f32 + NORM_EPS).sqrt();
            for v in row.iter_mut() {
                *v /= r;
            }
            rms.push(r);
        }

        let mut out = x_hat.clone();
        for row in out.data.chunks_mut(cols) {
            for (v, g) in row.iter_mut().zip(&self.gain.value.data) {
                *v *= g;
            }
        }

        self.cache = Some(RmsNormCache { x_hat, rms });
        out
    }

    /// dx = (d_x_hat - x_hat * mean(d_x_hat * x_hat)) / rms
    pub fn backward(&mut self, d_out: &Matrix) -> Matrix {
        let cache = self
            .cache
            .take()
            .expect("RmsNorm::backward called before forward");
        let cols = d_out.cols;
        let mut dx = Matrix::new(d_out.rows, cols);

        for i in 0..d_out.rows {
            let range = i * cols..(i + 1) * cols;
            let dy = &d_out.data[range.clone()];
            let x_hat = &cache.x_hat.data[range.clone()];

            let mut dot = 0.0;
            for j in 0..cols {
                self.gain.grad.data[j] += dy[j] * x_hat[j];
                dot += dy[j] * self.gain.value.data[j] * x_hat[j];
            }
            let mean = dot / cols as f32;

            let dx_row = &mut dx.data[range];
            for j in 0..cols {
                let d_x_hat = dy[j] * self.gain.value.data[j];
                dx_row[j] = (d_x_hat - x_hat[j] * mean) / cache.rms[i];
            }
        }

        dx
    }

    pub fn clear_cache(&mut self) {
        self.cache = None;
    }
}

// ============================================================================
// Embedding
// ============================================================================

#[derive(Clone)]
pub struct Embedding {
    pub table: Param, // [vocab_size, dimensions]

    pub tokens: Option<Vec<u16>>,
}

impl Embedding {
    pub fn new(vocab_size: usize, dimensions: usize) -> Self {
        Self {
            table: Param::new(Matrix::init(vocab_size, dimensions, INIT_STD), true),
            tokens: None,
        }
    }

    /// Look up embeddings for a sequence of tokens.
    /// Input: tokens of length [seq_len]
    /// Output: matrix of shape [seq_len, dimensions]
    pub fn forward(&mut self, tokens: &[u16]) -> Matrix {
        let table = &self.table.value;
        let seq_len = tokens.len();
        let mut data = Vec::with_capacity(seq_len * table.cols);

        for &token in tokens {
            assert!(
                (token as usize) < table.rows,
                "token id {} out of range for vocab_size {}",
                token,
                table.rows
            );
            data.extend(table.get_row(token as usize));
        }

        self.tokens = Some(tokens.to_vec());
        Matrix::with_vector(seq_len, table.cols, data)
    }

    /// Scatter-add d_out rows into the rows of the looked-up tokens.
    pub fn backward(&mut self, d_out: &Matrix) {
        let tokens = self
            .tokens
            .take()
            .expect("Embedding::backward called before forward");
        let cols = self.table.grad.cols;

        for (i, &token) in tokens.iter().enumerate() {
            let src = &d_out.data[i * cols..(i + 1) * cols];
            let dst = &mut self.table.grad.data[token as usize * cols..(token as usize + 1) * cols];
            for (d, s) in dst.iter_mut().zip(src) {
                *d += s;
            }
        }
    }

    pub fn clear_cache(&mut self) {
        self.tokens = None;
    }
}

// ============================================================================
// Attention
// ============================================================================

#[derive(Clone)]
pub struct Attention {
    pub config: Config,
    pub w_qkv: Param, // [dimensions, dimensions * 3]
    pub w_o: Param,   // [dimensions, dimensions]

    // Cache for backward pass
    pub cache: Option<AttentionCache>,
}

#[derive(Clone)]
pub struct AttentionCache {
    pub input: Matrix,
    pub q_heads: Vec<Matrix>, // after RoPE
    pub k_heads: Vec<Matrix>, // after RoPE
    pub v_heads: Vec<Matrix>,
    pub attn_weights: Vec<Matrix>, // softmax probabilities per head [seq_len, seq_len]
    pub concat: Matrix,
}

impl Attention {
    pub fn new(config: &Config) -> Self {
        let out_std = INIT_STD / (2.0 * config.n_layers as f32).sqrt();
        Self {
            config: config.clone(),
            w_qkv: Param::new(
                Matrix::init(config.dimensions, config.qkv_size(), INIT_STD),
                true,
            ),
            w_o: Param::new(
                Matrix::init(config.dimensions, config.dimensions, out_std),
                true,
            ),
            cache: None,
        }
    }

    /// Forward pass through attention layer.
    /// Input: x [seq_len, dimensions]
    /// Output: attention output [seq_len, dimensions]
    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        let config = &self.config;

        // ---- QKV projection ----
        let qkv = linear(x, &self.w_qkv);
        let (q, k, v) = qkv.split_qkv();

        // ---- Multi-head attention ----
        let mut attention_heads: Vec<Matrix> = Vec::with_capacity(config.n_heads);
        let mut q_heads: Vec<Matrix> = Vec::with_capacity(config.n_heads);
        let mut k_heads: Vec<Matrix> = Vec::with_capacity(config.n_heads);
        let mut v_heads: Vec<Matrix> = Vec::with_capacity(config.n_heads);
        let mut attn_weights_cache: Vec<Matrix> = Vec::with_capacity(config.n_heads);

        for head in 0..config.n_heads {
            // Extract head
            let mut q_head = get_head(&q, head, config.head_dim);
            let mut k_head = get_head(&k, head, config.head_dim);
            let v_head = get_head(&v, head, config.head_dim);

            // ---- RoPE (Rotary Position Embedding) ----
            q_head.rope();
            k_head.rope();

            // ---- Attention scores ----
            let qk = q_head.mul_transpose(&k_head);
            let qk_scaled = qk.dv_scalar((config.head_dim as f32).sqrt());

            let mut qk_masked = qk_scaled;
            qk_masked.causal_mask();

            let attn_weights = softmax(&qk_masked);
            let attention = attn_weights.mul_transpose(&v_head.transpose());

            q_heads.push(q_head);
            k_heads.push(k_head);
            v_heads.push(v_head);
            attn_weights_cache.push(attn_weights);
            attention_heads.push(attention);
        }

        // ---- Concat heads and project ----
        let concat = concat_heads(attention_heads);
        let output = linear(&concat, &self.w_o);

        // Store cache for backward pass
        self.cache = Some(AttentionCache {
            input: x.clone(),
            q_heads,
            k_heads,
            v_heads,
            attn_weights: attn_weights_cache,
            concat,
        });

        output
    }

    /// Backward pass. Accumulates weight gradients and returns d_x.
    /// Input: d_out [seq_len, dimensions]
    pub fn backward(&mut self, d_out: &Matrix) -> Matrix {
        let cache = self
            .cache
            .take()
            .expect("Attention::backward called before forward");
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim;
        let scale = (head_dim as f32).sqrt();

        // output = concat @ W_o
        let d_concat = linear_backward(&cache.concat, &mut self.w_o, d_out);

        let mut d_q_heads = Vec::with_capacity(n_heads);
        let mut d_k_heads = Vec::with_capacity(n_heads);
        let mut d_v_heads = Vec::with_capacity(n_heads);

        for head in 0..n_heads {
            let q = &cache.q_heads[head];
            let k = &cache.k_heads[head];
            let v = &cache.v_heads[head];
            let p = &cache.attn_weights[head];

            // attention = P @ V
            let d_attention = get_head(&d_concat, head, head_dim);
            let d_p = d_attention.mul_transpose(v); // [seq, seq]
            let d_v = p.transpose().mul_transpose(&d_attention.transpose()); // [seq, head_dim]

            // P = softmax(mask(Q @ K^T / sqrt(head_dim)))
            // masked positions have P = 0, so their gradient is already 0
            let d_scores = softmax_backward(p, &d_p).dv_scalar(scale);

            let mut d_q = d_scores.mul_transpose(&k.transpose()); // dS @ K
            let mut d_k = d_scores.transpose().mul_transpose(&q.transpose()); // dS^T @ Q

            // Undo RoPE rotation
            d_q.rope_backward();
            d_k.rope_backward();

            d_q_heads.push(d_q);
            d_k_heads.push(d_k);
            d_v_heads.push(d_v);
        }

        let d_qkv = merge_qkv(
            &concat_heads(d_q_heads),
            &concat_heads(d_k_heads),
            &concat_heads(d_v_heads),
        );

        // qkv = x @ W_qkv
        linear_backward(&cache.input, &mut self.w_qkv, &d_qkv)
    }

    pub fn clear_cache(&mut self) {
        self.cache = None;
    }
}

// ============================================================================
// Transformer Block (pre-norm)
// ============================================================================

#[derive(Clone)]
pub struct TransformerBlock {
    pub config: Config,
    pub attn_norm: RmsNorm,
    pub attention: Attention,
    pub mlp_norm: RmsNorm,
    pub mlp: Mlp,
}

impl TransformerBlock {
    pub fn new(config: &Config) -> Self {
        let out_std = INIT_STD / (2.0 * config.n_layers as f32).sqrt();
        Self {
            config: config.clone(),
            attn_norm: RmsNorm::new(config.dimensions),
            attention: Attention::new(config),
            mlp_norm: RmsNorm::new(config.dimensions),
            mlp: Mlp::new(config.dimensions, config.mlp_hidden, INIT_STD, out_std),
        }
    }

    /// Forward pass through transformer block.
    ///   h = x + Attention(RMSNorm(x))
    ///   out = h + MLP(RMSNorm(h))
    /// Input: x [seq_len, dimensions]
    /// Output: [seq_len, dimensions]
    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        // ---- Self-attention with residual ----
        let norm_x = self.attn_norm.forward(x);
        let attn_out = self.attention.forward(&norm_x);
        let h = x.add(&attn_out);

        // ---- MLP with residual ----
        let norm_h = self.mlp_norm.forward(&h);
        let mlp_out = self.mlp.forward(norm_h);
        h.add(&mlp_out)
    }

    /// Backward pass. Residual connections pass the gradient straight through
    /// and add the gradient flowing through each sub-layer.
    pub fn backward(&mut self, d_out: &Matrix) -> Matrix {
        let d_norm_h = self.mlp.backward(d_out);
        let mut d_h = d_out.clone();
        d_h.add_assign(&self.mlp_norm.backward(&d_norm_h));

        let d_norm_x = self.attention.backward(&d_h);
        let mut d_x = d_h;
        d_x.add_assign(&self.attn_norm.backward(&d_norm_x));
        d_x
    }

    pub fn clear_cache(&mut self) {
        self.attn_norm.clear_cache();
        self.attention.clear_cache();
        self.mlp_norm.clear_cache();
        self.mlp.clear_cache();
    }
}

// ============================================================================
// Language Model Head
// ============================================================================

#[derive(Clone)]
pub struct LanguageModelHead {
    pub weights: Param, // [dimensions, vocab_size]

    pub input: Option<Matrix>,
}

impl LanguageModelHead {
    pub fn new(dimensions: usize, vocab_size: usize) -> Self {
        Self {
            weights: Param::new(Matrix::init(dimensions, vocab_size, INIT_STD), true),
            input: None,
        }
    }

    /// Project to vocabulary probabilities.
    /// Input: x [seq_len, dimensions]
    /// Output: probs [seq_len, vocab_size]
    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        let logits = linear(x, &self.weights);
        self.input = Some(x.clone());
        softmax(&logits)
    }

    /// Input: d_logits [seq_len, vocab_size] (softmax already folded in, see `cross_entropy`)
    /// Output: d_x [seq_len, dimensions]
    pub fn backward(&mut self, d_logits: &Matrix) -> Matrix {
        let input = self
            .input
            .take()
            .expect("LanguageModelHead::backward called before forward");
        linear_backward(&input, &mut self.weights, d_logits)
    }

    pub fn clear_cache(&mut self) {
        self.input = None;
    }
}

// ============================================================================
// Transformer (Full Model)
// ============================================================================

#[derive(Clone)]
pub struct Transformer {
    pub config: Config,
    pub embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub final_norm: RmsNorm,
    pub lm_head: LanguageModelHead,
}

impl Transformer {
    pub fn new(config: Config) -> Self {
        config.validate().expect("invalid config");
        Self {
            embedding: Embedding::new(config.vocab_size, config.dimensions),
            blocks: (0..config.n_layers)
                .map(|_| TransformerBlock::new(&config))
                .collect(),
            final_norm: RmsNorm::new(config.dimensions),
            lm_head: LanguageModelHead::new(config.dimensions, config.vocab_size),
            config,
        }
    }

    /// Full forward pass.
    /// Input: token IDs [seq_len]
    /// Output: vocabulary probabilities [seq_len, vocab_size]
    pub fn forward(&mut self, tokens: &[u16]) -> Matrix {
        // 1. Embedding lookup
        let mut x = self.embedding.forward(tokens);

        // 2. Transformer blocks (attention + MLP)
        for block in self.blocks.iter_mut() {
            x = block.forward(&x);
        }

        // 3. Final RMSNorm
        let x = self.final_norm.forward(&x);

        // 4. Language model head (projection to vocab)
        self.lm_head.forward(&x)
    }

    /// Backward pass from d_logits. Gradients are accumulated into each
    /// parameter's `grad`, so call `zero_grad` before a new batch.
    pub fn backward(&mut self, d_logits: &Matrix) {
        let mut d_x = self.lm_head.backward(d_logits);
        d_x = self.final_norm.backward(&d_x);

        for block in self.blocks.iter_mut().rev() {
            d_x = block.backward(&d_x);
        }

        self.embedding.backward(&d_x);
    }

    /// All trainable parameters, in a fixed order.
    pub fn params(&self) -> Vec<&Param> {
        let mut params = vec![&self.embedding.table];
        for block in &self.blocks {
            params.push(&block.attn_norm.gain);
            params.push(&block.attention.w_qkv);
            params.push(&block.attention.w_o);
            params.push(&block.mlp_norm.gain);
            params.push(&block.mlp.w1);
            params.push(&block.mlp.w2);
            params.push(&block.mlp.w3);
        }
        params.push(&self.final_norm.gain);
        params.push(&self.lm_head.weights);
        params
    }

    /// Same order as `params`.
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params = vec![&mut self.embedding.table];
        for block in self.blocks.iter_mut() {
            params.push(&mut block.attn_norm.gain);
            params.push(&mut block.attention.w_qkv);
            params.push(&mut block.attention.w_o);
            params.push(&mut block.mlp_norm.gain);
            params.push(&mut block.mlp.w1);
            params.push(&mut block.mlp.w2);
            params.push(&mut block.mlp.w3);
        }
        params.push(&mut self.final_norm.gain);
        params.push(&mut self.lm_head.weights);
        params
    }

    pub fn num_params(&self) -> usize {
        self.params().iter().map(|p| p.value.data.len()).sum()
    }

    pub fn zero_grad(&mut self) {
        for p in self.params_mut() {
            p.zero_grad();
        }
    }

    pub fn clear_caches(&mut self) {
        self.embedding.clear_cache();
        for block in self.blocks.iter_mut() {
            block.clear_cache();
        }
        self.final_norm.clear_cache();
        self.lm_head.clear_cache();
    }

    /// Mean next-token loss on `inputs` -> `targets` without keeping caches.
    pub fn loss(&mut self, inputs: &[u16], targets: &[u16]) -> f32 {
        let probs = self.forward(inputs);
        self.clear_caches();
        cross_entropy(&probs, targets).0
    }

    // ---- Checkpointing ----
    //
    // Format (little endian):
    //   b"GLLM", version: u32, 7 x u64 config fields, n_params: u64,
    //   then per param: rows: u64, cols: u64, rows * cols x f32

    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(CHECKPOINT_MAGIC)?;
        w.write_all(&CHECKPOINT_VERSION.to_le_bytes())?;

        let c = &self.config;
        for field in [
            c.dimensions,
            c.n_heads,
            c.vocab_size,
            c.head_dim,
            c.mlp_hidden,
            c.n_layers,
            c.max_seq_len,
        ] {
            w.write_all(&(field as u64).to_le_bytes())?;
        }

        let params = self.params();
        w.write_all(&(params.len() as u64).to_le_bytes())?;
        for p in params {
            w.write_all(&(p.value.rows as u64).to_le_bytes())?;
            w.write_all(&(p.value.cols as u64).to_le_bytes())?;
            for v in &p.value.data {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        w.flush()
    }

    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Transformer> {
        let invalid = |msg: String| io::Error::new(io::ErrorKind::InvalidData, msg);
        let mut r = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != CHECKPOINT_MAGIC {
            return Err(invalid("not a ground_llm checkpoint".to_string()));
        }
        let version = read_u32(&mut r)?;
        if version != CHECKPOINT_VERSION {
            return Err(invalid(format!(
                "unsupported checkpoint version {}",
                version
            )));
        }

        let config = Config {
            dimensions: read_u64(&mut r)? as usize,
            n_heads: read_u64(&mut r)? as usize,
            vocab_size: read_u64(&mut r)? as usize,
            head_dim: read_u64(&mut r)? as usize,
            mlp_hidden: read_u64(&mut r)? as usize,
            n_layers: read_u64(&mut r)? as usize,
            max_seq_len: read_u64(&mut r)? as usize,
        };
        config.validate().map_err(invalid)?;

        let mut model = Transformer::new(config);
        let n_params = read_u64(&mut r)? as usize;
        let mut params = model.params_mut();
        if n_params != params.len() {
            return Err(invalid(format!(
                "expected {} params, checkpoint has {}",
                params.len(),
                n_params
            )));
        }

        let mut buf = [0u8; 4];
        for p in params.iter_mut() {
            let rows = read_u64(&mut r)? as usize;
            let cols = read_u64(&mut r)? as usize;
            if rows != p.value.rows || cols != p.value.cols {
                return Err(invalid(format!(
                    "param shape mismatch: expected [{}, {}], got [{}, {}]",
                    p.value.rows, p.value.cols, rows, cols
                )));
            }
            for v in p.value.data.iter_mut() {
                r.read_exact(&mut buf)?;
                *v = f32::from_le_bytes(buf);
            }
        }

        Ok(model)
    }
}

const CHECKPOINT_MAGIC: &[u8; 4] = b"GLLM";
const CHECKPOINT_VERSION: u32 = 1;

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

// ============================================================================
// Loss
// ============================================================================

/// Mean cross-entropy over positions and its gradient w.r.t. the logits.
/// Softmax + cross-entropy collapses to d_logits = (probs - one_hot) / n.
pub fn cross_entropy(probs: &Matrix, targets: &[u16]) -> (f32, Matrix) {
    assert_eq!(probs.rows, targets.len());
    let n = targets.len() as f32;

    let mut loss = 0.0;
    let mut d_logits = probs.clone();
    for (i, &target) in targets.iter().enumerate() {
        let target = target as usize;
        let prob = probs.get_value(i, target).max(1e-9);
        loss += -prob.ln();
        d_logits.set_value(i, target, d_logits.get_value(i, target) - 1.0);
    }

    d_logits /= n;
    (loss / n, d_logits)
}

// ============================================================================
// Training
// ============================================================================

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub weight_decay: f32,
    pub grad_clip: f32,
    /// Fraction of tokens (taken from the end) held out for validation.
    pub val_fraction: f32,
    /// Evaluate on the validation split every N steps (0 = never).
    pub eval_every: usize,
    pub eval_batches: usize,
    /// Save to this path whenever validation loss improves (and at the end if
    /// there is no validation split).
    pub checkpoint_path: Option<String>,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 1000,
            batch_size: 8,
            seq_len: 128,
            lr: 3e-4,
            min_lr: 3e-5,
            warmup_steps: 100,
            weight_decay: 0.1,
            grad_clip: 1.0,
            val_fraction: 0.1,
            eval_every: 100,
            eval_batches: 8,
            checkpoint_path: None,
            seed: 0,
        }
    }
}

impl TrainConfig {
    /// Linear warmup to `lr`, then cosine decay to `min_lr` at `steps`.
    pub fn lr_at(&self, step: usize) -> f32 {
        if self.warmup_steps > 0 && step <= self.warmup_steps {
            return self.lr * step as f32 / self.warmup_steps as f32;
        }
        let decay_steps = self.steps.saturating_sub(self.warmup_steps).max(1);
        let progress = (step - self.warmup_steps) as f32 / decay_steps as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress.min(1.0)).cos());
        self.min_lr + (self.lr - self.min_lr) * cosine
    }
}

#[derive(Clone, Debug)]
pub struct StepStats {
    pub step: usize,
    pub loss: f32,
    pub lr: f32,
    pub grad_norm: f32,
    pub tokens_per_sec: f32,
    pub val_loss: Option<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct TrainHistory {
    pub train_loss: Vec<f32>,
    pub val_loss: Vec<(usize, f32)>,
}

/// Picks a random window of `seq_len + 1` tokens: (inputs, next-token targets).
fn sample_window<'a>(
    tokens: &'a [u16],
    seq_len: usize,
    rng: &mut impl Rng,
) -> (&'a [u16], &'a [u16]) {
    let start = rng.random_range(0..=tokens.len() - (seq_len + 1));
    let window = &tokens[start..start + seq_len + 1];
    (&window[..seq_len], &window[1..])
}

/// Mean loss over `n_batches` evenly spaced windows (deterministic across calls).
pub fn evaluate(model: &mut Transformer, tokens: &[u16], seq_len: usize, n_batches: usize) -> f32 {
    let max_start = tokens.len() - (seq_len + 1);
    let n = n_batches.max(1);
    let mut total = 0.0;
    for i in 0..n {
        let start = if n == 1 { 0 } else { max_start * i / (n - 1) };
        let window = &tokens[start..start + seq_len + 1];
        total += model.loss(&window[..seq_len], &window[1..]);
    }
    total / n as f32
}

/// Trains `model` on a stream of token ids.
/// `on_step` is called after every optimizer step; return `false` to stop early.
pub fn train(
    model: &mut Transformer,
    tokens: &[u16],
    cfg: &TrainConfig,
    mut on_step: impl FnMut(&StepStats) -> bool,
) -> Result<TrainHistory, String> {
    if cfg.seq_len == 0 || cfg.seq_len > model.config.max_seq_len {
        return Err(format!(
            "seq_len ({}) must be in 1..=max_seq_len ({})",
            cfg.seq_len, model.config.max_seq_len
        ));
    }
    if cfg.batch_size == 0 {
        return Err("batch_size must be > 0".to_string());
    }
    if let Some(&bad) = tokens
        .iter()
        .find(|&&t| t as usize >= model.config.vocab_size)
    {
        return Err(format!(
            "token id {} out of range for vocab_size {}",
            bad, model.config.vocab_size
        ));
    }

    let n_val = (tokens.len() as f32 * cfg.val_fraction.clamp(0.0, 0.5)) as usize;
    let (train_tokens, val_tokens) = tokens.split_at(tokens.len() - n_val);
    let has_val = n_val > 0;

    if train_tokens.len() < cfg.seq_len + 1 {
        return Err(format!(
            "need at least {} training tokens, got {}",
            cfg.seq_len + 1,
            train_tokens.len()
        ));
    }
    if has_val && val_tokens.len() < cfg.seq_len + 1 {
        return Err(format!(
            "validation split has {} tokens, need at least {} (lower seq_len or raise val_fraction)",
            val_tokens.len(),
            cfg.seq_len + 1
        ));
    }

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut optimizer = AdamW::new(cfg.weight_decay);
    let mut history = TrainHistory::default();
    let mut best_val = f32::INFINITY;

    for step in 1..=cfg.steps {
        let started = Instant::now();
        let lr = cfg.lr_at(step);

        // ---- Forward + backward over the batch (gradient accumulation) ----
        model.zero_grad();
        let mut loss_sum = 0.0;
        for _ in 0..cfg.batch_size {
            let (inputs, targets) = sample_window(train_tokens, cfg.seq_len, &mut rng);
            let probs = model.forward(inputs);
            let (loss, mut d_logits) = cross_entropy(&probs, targets);
            d_logits /= cfg.batch_size as f32;
            model.backward(&d_logits);
            loss_sum += loss;
        }
        let loss = loss_sum / cfg.batch_size as f32;

        // ---- Optimizer step ----
        let mut params = model.params_mut();
        let grad_norm = clip_grad_norm(&mut params, cfg.grad_clip);
        optimizer.step(&mut params, lr);

        let elapsed = started.elapsed().as_secs_f32();
        history.train_loss.push(loss);

        // ---- Validation + checkpointing ----
        let is_eval_step = cfg.eval_every > 0 && (step % cfg.eval_every == 0 || step == cfg.steps);
        let val_loss = if has_val && is_eval_step {
            let val = evaluate(model, val_tokens, cfg.seq_len, cfg.eval_batches);
            history.val_loss.push((step, val));
            if val < best_val {
                best_val = val;
                if let Some(path) = &cfg.checkpoint_path {
                    model
                        .save(path)
                        .map_err(|e| format!("failed to save checkpoint: {}", e))?;
                }
            }
            Some(val)
        } else {
            None
        };

        let stats = StepStats {
            step,
            loss,
            lr,
            grad_norm,
            tokens_per_sec: (cfg.batch_size * cfg.seq_len) as f32 / elapsed.max(1e-9),
            val_loss,
        };
        if !on_step(&stats) {
            break;
        }
    }

    if !has_val {
        if let Some(path) = &cfg.checkpoint_path {
            model
                .save(path)
                .map_err(|e| format!("failed to save checkpoint: {}", e))?;
        }
    }

    Ok(history)
}

// ============================================================================
// Generation
// ============================================================================

/// Autoregressive sampling. `temperature <= 0` means greedy; `top_k == 0` means no top-k filter.
pub fn generate(
    model: &mut Transformer,
    prompt: &[u16],
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    rng: &mut impl Rng,
) -> Vec<u16> {
    assert!(!prompt.is_empty(), "prompt must contain at least one token");
    let mut tokens = prompt.to_vec();

    for _ in 0..max_new_tokens {
        let start = tokens.len().saturating_sub(model.config.max_seq_len);
        let probs = model.forward(&tokens[start..]);
        model.clear_caches();

        let last = probs.get_row(probs.rows - 1);
        tokens.push(sample_token(&last, temperature, top_k, rng));
    }

    tokens
}

fn sample_token(probs: &[f32], temperature: f32, top_k: usize, rng: &mut impl Rng) -> u16 {
    let mut candidates: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

    if temperature <= 0.0 {
        return candidates[0].0 as u16;
    }
    if top_k > 0 {
        candidates.truncate(top_k);
    }

    // softmax(logits / T) == normalize(p ^ (1 / T))
    let weights: Vec<f32> = candidates
        .iter()
        .map(|&(_, p)| p.max(1e-30).powf(1.0 / temperature))
        .collect();
    let total: f32 = weights.iter().sum();

    let mut threshold = rng.random_range(0.0..total);
    for (&(token, _), w) in candidates.iter().zip(&weights) {
        if threshold < *w {
            return token as u16;
        }
        threshold -= w;
    }
    candidates[candidates.len() - 1].0 as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Config {
        Config {
            dimensions: 8,
            n_heads: 2,
            vocab_size: 13,
            head_dim: 4,
            mlp_hidden: 12,
            n_layers: 2,
            max_seq_len: 16,
        }
    }

    #[test]
    fn test_embedding_lookup() {
        let mut embedding = Embedding::new(100, 64);
        let tokens = vec![1u16, 2, 3];
        let out = embedding.forward(&tokens);

        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 64);
    }

    #[test]
    fn test_attention_shape() {
        let config = Config::default();
        let x = Matrix::random(5, config.dimensions);

        let w_qkv = Matrix::random(config.dimensions, config.qkv_size());
        let qkv = x.mul_transpose(&w_qkv.transpose());
        let (q, k, v) = qkv.split_qkv();

        assert_eq!(q.cols, config.dimensions);
        assert_eq!(k.cols, config.dimensions);
        assert_eq!(v.cols, config.dimensions);
    }

    #[test]
    fn test_transformer_forward() {
        let config = Config::default();
        let vocab_size = config.vocab_size;
        let mut transformer = Transformer::new(config);

        let tokens = vec![1u16, 2, 3, 4, 5];
        let probs = transformer.forward(&tokens);

        assert_eq!(probs.rows, 5);
        assert_eq!(probs.cols, vocab_size);
    }

    #[test]
    fn test_initial_loss_near_uniform() {
        let config = Config::default();
        let vocab_size = config.vocab_size;
        let mut model = Transformer::new(config);

        let tokens: Vec<u16> = (0..33).map(|i| (i * 37 % vocab_size) as u16).collect();
        let loss = model.loss(&tokens[..32], &tokens[1..]);
        let uniform = (vocab_size as f32).ln();

        assert!(
            (loss - uniform).abs() < 0.5,
            "loss {} vs ln(vocab) {}",
            loss,
            uniform
        );
    }

    /// Compares every analytic gradient against central finite differences.
    #[test]
    fn test_gradient_check() {
        let mut model = Transformer::new(tiny_config());

        // Larger random weights than the real init so gradients are well above f32 noise.
        let mut rng = StdRng::seed_from_u64(42);
        for p in model.params_mut() {
            let offset = if p.decay { 0.0 } else { 1.0 }; // norm gains around 1
            for v in p.value.data.iter_mut() {
                *v = offset + rng.random_range(-0.5..0.5);
            }
        }

        let tokens: Vec<u16> = vec![3, 7, 1, 12, 7, 0, 5];
        let (inputs, targets) = (&tokens[..6], &tokens[1..]);

        model.zero_grad();
        let probs = model.forward(inputs);
        let (_, d_logits) = cross_entropy(&probs, targets);
        model.backward(&d_logits);
        let analytic: Vec<Vec<f32>> = model.params().iter().map(|p| p.grad.data.clone()).collect();

        let eps = 1e-2;
        for (pi, grads) in analytic.iter().enumerate() {
            let mut numeric = Vec::with_capacity(grads.len());
            for j in 0..grads.len() {
                let orig = model.params()[pi].value.data[j];

                model.params_mut()[pi].value.data[j] = orig + eps;
                let loss_plus = model.loss(inputs, targets);
                model.params_mut()[pi].value.data[j] = orig - eps;
                let loss_minus = model.loss(inputs, targets);
                model.params_mut()[pi].value.data[j] = orig;

                numeric.push((loss_plus - loss_minus) / (2.0 * eps));
            }

            let diff: f32 = grads
                .iter()
                .zip(&numeric)
                .map(|(a, n)| (a - n).powi(2))
                .sum::<f32>()
                .sqrt();
            let a_norm: f32 = grads.iter().map(|a| a * a).sum::<f32>().sqrt();
            let n_norm: f32 = numeric.iter().map(|n| n * n).sum::<f32>().sqrt();
            let rel = diff / (a_norm + n_norm).max(1e-6);

            assert!(
                rel < 1e-2,
                "param {} gradient mismatch: rel error {} (analytic norm {}, numeric norm {})",
                pi,
                rel,
                a_norm,
                n_norm
            );
        }
    }

    #[test]
    fn test_training_reduces_loss() {
        let mut model = Transformer::new(tiny_config());
        let pattern = [1u16, 4, 9, 2, 7, 11, 3];
        let tokens: Vec<u16> = pattern.iter().cycle().take(200).cloned().collect();

        let cfg = TrainConfig {
            steps: 300,
            batch_size: 4,
            seq_len: 8,
            lr: 1e-2,
            min_lr: 1e-3,
            warmup_steps: 10,
            weight_decay: 0.0,
            val_fraction: 0.1,
            eval_every: 100,
            eval_batches: 2,
            ..TrainConfig::default()
        };

        let history = train(&mut model, &tokens, &cfg, |_| true).unwrap();
        let first = history.train_loss[0];
        let last = *history.train_loss.last().unwrap();

        assert!(last < 0.2, "loss went from {} to {}", first, last);
        assert!(history.val_loss.last().unwrap().1 < 0.2);
    }

    #[test]
    fn test_generate_greedy_follows_pattern() {
        let mut model = Transformer::new(tiny_config());
        let pattern = [1u16, 4, 9, 2, 7, 11, 3];
        let tokens: Vec<u16> = pattern.iter().cycle().take(200).cloned().collect();
        let cfg = TrainConfig {
            steps: 300,
            batch_size: 4,
            seq_len: 8,
            lr: 1e-2,
            min_lr: 1e-3,
            warmup_steps: 10,
            weight_decay: 0.0,
            val_fraction: 0.0,
            eval_every: 0,
            ..TrainConfig::default()
        };
        train(&mut model, &tokens, &cfg, |_| true).unwrap();

        let mut rng = StdRng::seed_from_u64(0);
        let out = generate(&mut model, &[1, 4, 9], 7, 0.0, 0, &mut rng);
        assert_eq!(out, vec![1, 4, 9, 2, 7, 11, 3, 1, 4, 9]);
    }

    #[test]
    fn test_checkpoint_round_trip() {
        let mut model = Transformer::new(tiny_config());
        let path = std::env::temp_dir().join(format!("ground_llm_test_{}.bin", std::process::id()));

        model.save(&path).unwrap();
        let mut loaded = Transformer::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded.config, model.config);
        let tokens = [1u16, 2, 3, 4];
        let a = model.forward(&tokens);
        let b = loaded.forward(&tokens);
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn test_lr_schedule() {
        let cfg = TrainConfig {
            steps: 100,
            warmup_steps: 10,
            lr: 1.0,
            min_lr: 0.1,
            ..TrainConfig::default()
        };
        assert!((cfg.lr_at(5) - 0.5).abs() < 1e-6);
        assert!((cfg.lr_at(10) - 1.0).abs() < 1e-6);
        assert!((cfg.lr_at(100) - 0.1).abs() < 1e-6);
    }
}
