use crate::param::Param;

// ============================================================================
// AdamW
// ============================================================================

pub struct AdamW {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    t: i32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamW {
    pub fn new(weight_decay: f32) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Applies one update using the gradients currently stored in `params`.
    /// `params` must be passed in the same order on every call.
    pub fn step(&mut self, params: &mut [&mut Param], lr: f32) {
        if self.m.is_empty() {
            self.m = params
                .iter()
                .map(|p| vec![0.0; p.value.data.len()])
                .collect();
            self.v = params
                .iter()
                .map(|p| vec![0.0; p.value.data.len()])
                .collect();
        }
        assert_eq!(self.m.len(), params.len());

        self.t += 1;
        let bias1 = 1.0 - self.beta1.powi(self.t);
        let bias2 = 1.0 - self.beta2.powi(self.t);

        for (i, p) in params.iter_mut().enumerate() {
            let decay = if p.decay { lr * self.weight_decay } else { 0.0 };
            let m = &mut self.m[i];
            let v = &mut self.v[i];

            for j in 0..p.value.data.len() {
                let g = p.grad.data[j];
                m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * g;
                v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * g * g;

                let m_hat = m[j] / bias1;
                let v_hat = v[j] / bias2;

                let w = &mut p.value.data[j];
                *w -= decay * *w;
                *w -= lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}

/// Scales all gradients so their global L2 norm is at most `max_norm`.
/// Returns the norm before clipping.
pub fn clip_grad_norm(params: &mut [&mut Param], max_norm: f32) -> f32 {
    let sq_sum: f32 = params
        .iter()
        .map(|p| p.grad.data.iter().map(|g| g * g).sum::<f32>())
        .sum();
    let norm = sq_sum.sqrt();

    if max_norm > 0.0 && norm > max_norm {
        let scale = max_norm / (norm + 1e-6);
        for p in params.iter_mut() {
            for g in &mut p.grad.data {
                *g *= scale;
            }
        }
    }

    norm
}
