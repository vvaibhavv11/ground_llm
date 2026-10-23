use crate::matrix::Matrix;
use crate::param::{Param, linear, linear_backward};

// ============================================================================
// MLP / FFN (Gated SiLU)
// ============================================================================

#[derive(Clone)]
pub struct Mlp {
    pub n_hidden: usize,
    pub n_input: usize,
    pub w1: Param, // [n_input, n_hidden]
    pub w2: Param, // [n_input, n_hidden]  (gate projection)
    pub w3: Param, // [n_hidden, n_input]  (output projection)

    // Cache for backward
    pub cache: Option<MlpCache>,
}

#[derive(Clone)]
pub struct MlpCache {
    pub input: Matrix,
    pub h1: Matrix,
    pub h2: Matrix,
    pub h_gated: Matrix,
}

impl Mlp {
    /// `std` initialises the input projections, `out_std` the output projection
    /// (scaled down so the residual stream doesn't grow with depth).
    pub fn new(n_input: usize, n_hidden: usize, std: f32, out_std: f32) -> Self {
        Self {
            n_hidden,
            n_input,
            w1: Param::new(Matrix::init(n_input, n_hidden, std), true),
            w2: Param::new(Matrix::init(n_input, n_hidden, std), true),
            w3: Param::new(Matrix::init(n_hidden, n_input, out_std), true),
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
        let h1 = linear(&input, &self.w1);
        let h2 = linear(&input, &self.w2);

        // Gated activation: SiLU(h1) * h2
        let mut h_gated = h1.clone();
        h_gated.swish();
        h_gated.elem_mul(&h2);

        // Output projection: [seq_len, n_input]
        let output = linear(&h_gated, &self.w3);

        self.cache = Some(MlpCache {
            input,
            h1,
            h2,
            h_gated,
        });

        output
    }

    /// Backward pass. Accumulates weight gradients and returns d_input.
    /// Input: d_out [seq_len, n_input]
    pub fn backward(&mut self, d_out: &Matrix) -> Matrix {
        let cache = self
            .cache
            .take()
            .expect("Mlp::backward called before forward");

        // output = h_gated @ W3
        let d_gated = linear_backward(&cache.h_gated, &mut self.w3, d_out);

        // h_gated = SiLU(h1) * h2
        let mut d_h1 = cache.h1.clone();
        d_h1.swish_derivation();
        d_h1.elem_mul(&cache.h2);
        d_h1.elem_mul(&d_gated);

        let mut d_h2 = cache.h1;
        d_h2.swish();
        d_h2.elem_mul(&d_gated);

        // h1 = x @ W1, h2 = x @ W2
        let mut d_input = linear_backward(&cache.input, &mut self.w1, &d_h1);
        d_input.add_assign(&linear_backward(&cache.input, &mut self.w2, &d_h2));
        d_input
    }

    pub fn clear_cache(&mut self) {
        self.cache = None;
    }
}
