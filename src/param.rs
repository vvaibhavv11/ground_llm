use crate::matrix::Matrix;

// ============================================================================
// Trainable parameter (value + accumulated gradient)
// ============================================================================

#[derive(Clone)]
pub struct Param {
    pub value: Matrix,
    pub grad: Matrix,
    /// Whether AdamW applies weight decay (weight matrices yes, norm gains no).
    pub decay: bool,
}

impl Param {
    pub fn new(value: Matrix, decay: bool) -> Self {
        let grad = Matrix::new(value.rows, value.cols);
        Self { value, grad, decay }
    }

    pub fn zero_grad(&mut self) {
        self.grad.data.fill(0.0);
    }
}

// ============================================================================
// Linear layer helpers (weights stored as [in, out])
// ============================================================================

/// y = x @ W
pub fn linear(x: &Matrix, w: &Param) -> Matrix {
    x.mul_transpose(&w.value.transpose())
}

/// Backward of y = x @ W.
/// Accumulates dW = x^T @ dy into `w.grad` and returns dx = dy @ W^T.
pub fn linear_backward(x: &Matrix, w: &mut Param, dy: &Matrix) -> Matrix {
    let d_w = x.transpose().mul_transpose(&dy.transpose());
    w.grad.add_assign(&d_w);
    dy.mul_transpose(&w.value)
}
