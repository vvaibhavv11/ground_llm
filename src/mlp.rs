use crate::matrix::Matrix;

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
    pub fn new(n_input: usize, n_hidden: usize) -> Self {
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