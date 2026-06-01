use rand::prelude::*;

use crate::matrix::{concat_heads, get_head, softmax, Matrix};
use crate::mlp::{Mlp, MlpCache};

// ============================================================================
// Config
// ============================================================================

#[derive(Clone, Debug)]
pub struct Config {
    pub dimensions: usize,
    pub n_heads: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub mlp_hidden: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dimensions: 512,
            n_heads: 8,
            vocab_size: 8000,
            head_dim: 64,
            mlp_hidden: 1380,
        }
    }
}

impl Config {
    pub fn qkv_size(&self) -> usize {
        self.dimensions * 3
    }
}

// ============================================================================
// Embedding
// ============================================================================

#[derive(Clone)]
pub struct Embedding {
    pub table: Matrix, // [vocab_size, dimensions]
}

impl Embedding {
    pub fn new(vocab_size: usize, dimensions: usize) -> Self {
        Self {
            table: Matrix::random(vocab_size, dimensions),
        }
    }

    /// Look up embeddings for a sequence of tokens.
    /// Input: tokens of length [seq_len]
    /// Output: matrix of shape [seq_len, dimensions]
    pub fn forward(&self, tokens: &[u16]) -> Matrix {
        let seq_len = tokens.len();
        let mut data = Vec::with_capacity(seq_len * self.table.cols);

        for &token in tokens {
            data.extend(self.table.get_row(token as usize));
        }

        Matrix::with_vector(seq_len, self.table.cols, data)
    }
}

// ============================================================================
// Attention
// ============================================================================

#[derive(Clone)]
pub struct Attention {
    pub config: Config,
    pub w_qkv: Matrix, // [dimensions, dimensions * 3]
    pub w_o: Matrix,   // [dimensions, dimensions]

    // Cache for backward pass
    pub cache: Option<AttentionCache>,
}

#[derive(Clone, Default)]
pub struct AttentionCache {
    pub qkv: Option<Matrix>,
    pub q: Option<Matrix>,
    pub k: Option<Matrix>,
    pub v: Option<Matrix>,
    pub q_heads: Vec<Matrix>,
    pub k_heads: Vec<Matrix>,
    pub v_heads: Vec<Matrix>,
    pub attention_heads: Vec<Matrix>,
    pub concat: Option<Matrix>,
    pub output: Option<Matrix>,
}

impl Attention {
    pub fn new(config: &Config) -> Self {
        Self {
            config: config.clone(),
            w_qkv: Matrix::random(config.dimensions, config.qkv_size()),
            w_o: Matrix::random(config.dimensions, config.dimensions),
            cache: None,
        }
    }

    /// Forward pass through attention layer.
    /// Input: x [seq_len, dimensions]
    /// Output: attention output [seq_len, dimensions]
    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        let config = &self.config;

        // ---- QKV projection ----
        let qkv = x.mul_transpose(&self.w_qkv.transpose());
        let (q, k, v) = qkv.split_qkv();

        // ---- Multi-head attention ----
        let mut attention_heads: Vec<Matrix> = Vec::with_capacity(config.n_heads);
        let mut q_heads: Vec<Matrix> = Vec::with_capacity(config.n_heads);
        let mut k_heads: Vec<Matrix> = Vec::with_capacity(config.n_heads);
        let mut v_heads_cache: Vec<Matrix> = Vec::with_capacity(config.n_heads);

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
            v_heads_cache.push(v_head);
            attention_heads.push(attention);
        }

        // ---- Concat heads and project ----
        let concat = concat_heads(attention_heads);
        let output = concat.mul_transpose(&self.w_o.transpose());

        // Store cache for backward pass
        self.cache = Some(AttentionCache {
            qkv: Some(qkv),
            q: Some(q),
            k: Some(k),
            v: Some(v),
            q_heads,
            k_heads,
            v_heads: v_heads_cache,
            attention_heads: Vec::new(),
            concat: Some(concat),
            output: Some(output.clone()),
        });

        output
    }

    pub fn clear_cache(&mut self) {
        self.cache = None;
    }
}

// ============================================================================
// Transformer Block
// ============================================================================

#[derive(Clone)]
pub struct TransformerBlock {
    pub config: Config,
    pub attention: Attention,
    pub mlp: Mlp,

    // Cache for backward pass
    pub cache: Option<TransformerBlockCache>,
}

#[derive(Clone, Default)]
pub struct TransformerBlockCache {
    pub attn_out: Option<Matrix>,
    pub x_after_attn: Option<Matrix>,
    pub norm_x: Option<Matrix>,
    pub mlp_out: Option<Matrix>,
    pub output: Option<Matrix>,
}

impl TransformerBlock {
    pub fn new(config: &Config) -> Self {
        Self {
            config: config.clone(),
            attention: Attention::new(config),
            mlp: Mlp::new(config.dimensions, config.mlp_hidden),
            cache: None,
        }
    }

    /// Forward pass through transformer block.
    /// Input: x [seq_len, dimensions]
    /// Output: [seq_len, dimensions]
    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        // ---- Self-attention with residual ----
        let attn_out = self.attention.forward(x);
        let x = x.add(&attn_out);

        // ---- MLP with residual ----
        let norm_x = x.rms_norm();
        let mlp_out = self.mlp.forward(norm_x.clone());
        let output = x.add(&mlp_out);

        self.cache = Some(TransformerBlockCache {
            attn_out: Some(attn_out),
            x_after_attn: Some(x),
            norm_x: Some(norm_x),
            mlp_out: Some(mlp_out),
            output: Some(output.clone()),
        });

        output
    }

    pub fn clear_cache(&mut self) {
        self.cache = None;
    }
}

// ============================================================================
// Language Model Head
// ============================================================================

#[derive(Clone)]
pub struct LanguageModelHead {
    pub weights: Matrix, // [dimensions, vocab_size]

    pub cache: Option<LmHeadCache>,
}

#[derive(Clone, Default)]
pub struct LmHeadCache {
    pub input: Option<Matrix>,
    pub logits: Option<Matrix>,
    pub probs: Option<Matrix>,
    pub d_w: Option<Matrix>,
}

impl LanguageModelHead {
    pub fn new(dimensions: usize, vocab_size: usize) -> Self {
        Self {
            weights: Matrix::random(dimensions, vocab_size),
            cache: None,
        }
    }

    /// Project to vocabulary logits.
    /// Input: x [seq_len, dimensions]
    /// Output: logits [seq_len, vocab_size]
    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        let logits = x.mul_transpose(&self.weights.transpose());
        let probs = softmax(&logits);

        self.cache = Some(LmHeadCache {
            input: Some(x.clone()),
            logits: Some(logits.clone()),
            probs: Some(probs.clone()),
            d_w: None,
        });

        probs
    }
    pub fn backprop(&mut self, d_logits: &Matrix) -> Matrix {
        let d_w_lm = self
            .cache
            .as_ref()
            .unwrap()
            .input
            .as_ref()
            .unwrap()
            .mul_transpose(&d_logits.transpose());
        let d_x = d_logits.mul_transpose(&self.weights);
        self.cache.as_mut().unwrap().d_w = Some(d_w_lm);
        d_x
    }

    pub fn clear_cache(&mut self) {
        self.cache = None;
    }
}

// ============================================================================
// Transformer (Full Model)
// ============================================================================

#[derive(Clone)]
pub struct Transformer {
    pub config: Config,
    pub embedding: Embedding,
    pub block: TransformerBlock,
    pub lm_head: LanguageModelHead,
}

impl Transformer {
    pub fn new(config: Config) -> Self {
        Self {
            config: config.clone(),
            embedding: Embedding::new(config.vocab_size, config.dimensions),
            block: TransformerBlock::new(&config),
            lm_head: LanguageModelHead::new(config.dimensions, config.vocab_size),
        }
    }

    /// Full forward pass.
    /// Input: token IDs [seq_len]
    /// Output: vocabulary probabilities [seq_len, vocab_size]
    pub fn forward(&mut self, tokens: &[u16]) -> Matrix {
        // 1. Embedding lookup
        let x = self.embedding.forward(tokens);

        // 2. RMSNorm before attention
        let x = x.rms_norm();

        // 3. Transformer block (attention + MLP)
        let x = self.block.forward(&x);

        // 4. Final RMSNorm
        let x = x.rms_norm();

        // 5. Language model head (projection to vocab)
        self.lm_head.forward(&x)
    }

    pub fn backprop(&mut self, d_logits: &Matrix) {
        //backprop for the language model head
        let x = self.lm_head.backprop(d_logits);
        
    }

    pub fn clear_caches(&mut self) {
        self.block.attention.clear_cache();
        self.block.mlp.clear_cache();
        self.lm_head.clear_cache();
    }
}

// ============================================================================
// Training Entry Point
// ============================================================================

pub fn train_model(data: Vec<u16>) {
    let config = Config::default();
    let mut transformer = Transformer::new(config);

    // ---- Forward pass ----
    let probs = transformer.forward(&data);
    let mut loss = 0.0;
    for i in 0..data.len() - 1 {
        let target = data[i + 1] as usize;

        let prob = probs.get_value(i, target).max(1e-9);

        loss += -prob.ln();
    }

    loss /= (data.len() - 1) as f32;

    // Probs now contains softmax probabilities over vocabulary
    // Next step: implement backward pass to compute gradients
    // backprop start
    let mut d_logits = probs.clone();

    for i in 0..data.len() - 1 {
        let target = data[i + 1] as usize;

        d_logits.set_value(i, target, d_logits.get_value(i, target) - 1.0);
    }

    d_logits /= (data.len() - 1) as f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_lookup() {
        let embedding = Embedding::new(100, 64);
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
}
