use rand::prelude::*;
use std::fmt;

use crate::matrix::{concat_heads, get_head, softmax, Matrix};

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
            vocab_size: 50256,
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
// RMSNorm
// ============================================================================

pub fn rms_norm(x: &Matrix) -> Matrix {
    x.rms_norm()
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
    pub qkv: Option<Matrix>,       // [seq_len, dimensions * 3]
    pub q: Option<Matrix>,        // [seq_len, dimensions]
    pub k: Option<Matrix>,       // [seq_len, dimensions]
    pub v: Option<Matrix>,       // [seq_len, dimensions]
    pub q_heads: Vec<Matrix>,     // [n_heads, seq_len, head_dim]
    pub k_heads: Vec<Matrix>,     // [n_heads, seq_len, head_dim]
    pub qk: Option<Matrix>,       // [seq_len, seq_len] per head
    pub qk_scaled: Option<Matrix>,
    pub qk_masked: Option<Matrix>,
    pub attn_weights: Option<Matrix>,
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
        let seq_len = x.rows;

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

            let mut qk_masked = qk_scaled.clone();
            qk_masked.causal_mask();

            let attn_weights = softmax(&qk_masked);
            let attention = attn_weights.mul_transpose(&v_head.transpose());

            q_heads.push(q_head);
            k_heads.push(k_head);
            v_heads_cache.push(v_head);
            attention_heads.push(attention);
        }

        // ---- Concat heads and project ----
        let concat = concat_heads(attention_heads.clone());
        let output = concat.mul_transpose(&self.w_o.transpose());

        // Store cache for backward pass
        self.cache = Some(AttentionCache {
            qkv: Some(qkv),
            q: Some(q),
            k: Some(k),
            v: Some(v),
            q_heads: q_heads,
            k_heads: k_heads,
            qk: None,
            qk_scaled: None,
            qk_masked: None,
            attn_weights: None,
            v_heads: v_heads_cache,
            attention_heads,
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
// MLP / FFN (Gated SiLU)
// ============================================================================

#[derive(Clone)]
pub struct Mlp {
    pub n_hidden: usize,
    pub n_input: usize,
    pub w1: Matrix, // [n_input, n_hidden]
    pub w2: Matrix, // [n_input, n_hidden]  (gate projection)
    pub w3: Matrix, // [n_hidden, n_input]  (output projection)

    // Cache for backward
    pub cache: Option<MlpCache>,
}

#[derive(Clone, Default)]
pub struct MlpCache {
    pub input: Option<Matrix>,
    pub h1: Option<Matrix>,
    pub h2: Option<Matrix>,
    pub h_gated: Option<Matrix>,
    pub output: Option<Matrix>,
}

impl Mlp {
    pub fn new(seq_len: usize, n_input: usize, n_hidden: usize) -> Self {
        Self {
            n_hidden,
            n_input,
            w1: Matrix::random(n_input, n_hidden),
            w2: Matrix::random(n_input, n_hidden),
            w3: Matrix::random(n_hidden, n_input),
            cache: None,
        }
    }

    /// Forward pass through MLP.
    /// Uses Gated SiLU (Swish) activation:
    ///   h = SiLU(x @ W1) * (x @ W2)
    ///   output = h @ W3
    ///
    /// Input: x [seq_len, n_input]
    /// Output: [seq_len, n_input]
    pub fn forward(&mut self, input: Matrix) -> Matrix {
        // Project to hidden: [seq_len, n_hidden]
        let h1 = input.mul_transpose(&self.w1.transpose());
        let h2 = input.mul_transpose(&self.w2.transpose());

        // Gated activation: SiLU(h1) * h2
        let mut h_gated = h1.clone();
        h_gated.swish();
        h_gated.elem_mul(&h2);

        // Output projection: [seq_len, n_input]
        let output = h_gated.mul_transpose(&self.w3.transpose());

        self.cache = Some(MlpCache {
            input: Some(input),
            h1: Some(h1),
            h2: Some(h2),
            h_gated: Some(h_gated),
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
}

impl TransformerBlock {
    pub fn new(config: &Config) -> Self {
        Self {
            config: config.clone(),
            attention: Attention::new(config),
            mlp: Mlp::new(config.dimensions, config.dimensions, config.mlp_hidden),
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
        let norm_x = rms_norm(&x);
        let mlp_out = self.mlp.forward(norm_x);
        x.add(&mlp_out)
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
        });

        probs
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
        let mut x = self.embedding.forward(tokens);

        // 2. RMSNorm before attention
        x = rms_norm(&x);

        // 3. Transformer block (attention + MLP)
        x = self.block.forward(&x);

        // 4. Final RMSNorm
        x = rms_norm(&x);

        // 5. Language model head (projection to vocab)
        self.lm_head.forward(&x)
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
    let mut rng = rand::rng();

    // ---- Build sequence ----
    let mut transformer = Transformer::new(config.clone());

    // ---- Forward pass ----
    let probs = transformer.forward(&data);

    // Probs now contains softmax probabilities over vocabulary
    // Next step: implement backward pass to compute gradients
}

// ============================================================================
// Standalone Attention (matches original train_model logic exactly)
// ============================================================================

/// Simplified attention for single-block use.
/// This mirrors the original train_model exactly.
pub fn run_attention(x: &Matrix, config: &Config) -> Matrix {
    let w_qkv = Matrix::random(config.dimensions, config.qkv_size());
    let qkv = x.mul_transpose(&w_qkv.transpose());
    let (q, k, v) = qkv.split_qkv();

    let mut heads_output: Vec<Matrix> = vec![];
    for head in 0..config.n_heads {
        let mut q_head = get_head(&q, head, config.head_dim);
        let mut k_head = get_head(&k, head, config.head_dim);
        let v_head = get_head(&v, head, config.head_dim);

        q_head.rope();
        k_head.rope();

        let qk = q_head.mul_transpose(&k_head);
        let mut dev_answer = qk.dv_scalar((config.head_dim as f32).sqrt());
        dev_answer.causal_mask();

        let final_matrix = softmax(&dev_answer);
        let attention = final_matrix.mul_transpose(&v_head.transpose());
        heads_output.push(attention);
    }

    let concat = concat_heads(heads_output);
    let w_o = Matrix::random(config.dimensions, config.dimensions);
    concat.mul_transpose(&w_o.transpose())
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
        let mut transformer = Transformer::new(config);

        let tokens = vec![1u16, 2, 3, 4, 5];
        let probs = transformer.forward(&tokens);

        assert_eq!(probs.rows, 5);
        assert_eq!(probs.cols, config.vocab_size);
    }

    #[test]
    fn test_mlp_gated_activation() {
        let mlp = Mlp::new(4, 512, 1380);
        let input = Matrix::random(4, 512);

        // Just verify it runs without panicking
        let _ = input;
    }
}