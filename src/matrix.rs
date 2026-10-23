use rand::prelude::*;
use rayon::prelude::*;
use std::{fmt, ops::DivAssign};

const EXPO: f32 = 2.71828;

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = rand::rng();
        let values: Vec<f32> = (0..rows * cols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        Self {
            rows,
            cols,
            data: values,
        }
    }

    /// Uniform init with the given standard deviation (U(-a, a) has std a / sqrt(3)).
    pub fn init(rows: usize, cols: usize, std: f32) -> Matrix {
        let a = std * 3f32.sqrt();
        let mut rng = rand::rng();
        let values: Vec<f32> = (0..rows * cols).map(|_| rng.random_range(-a..a)).collect();
        Matrix::with_vector(rows, cols, values)
    }

    pub fn with_vector(rows: usize, cols: usize, vector: Vec<f32>) -> Matrix {
        if rows * cols != vector.len() {
            panic!("wrong vector")
        }
        Self {
            rows,
            cols,
            data: vector,
        }
    }

    pub fn get_value(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    pub fn get_row(&self, row: usize) -> Vec<f32> {
        let start = row * self.cols;
        let end = start + self.cols;
        self.data[start..end].to_vec()
    }

    pub fn transpose(&self) -> Matrix {
        let mut new_data = Vec::with_capacity(self.rows * self.cols);
        for i in 0..self.cols {
            for j in 0..self.rows {
                new_data.push(self.get_value(j, i));
            }
        }
        return Matrix::with_vector(self.cols, self.rows, new_data);
    }

    pub fn rms_norm(&self) -> Matrix {
        let epsilon = 1e-5;
        let mut new_data = self.data.clone();
        for row in 0..self.rows {
            let start = row * self.cols;
            let end = start + self.cols;

            let slice = &mut new_data[start..end];

            let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
            let rms = (sq_sum / self.cols as f32 + epsilon).sqrt();

            for x in slice.iter_mut() {
                *x /= rms;
            }
        }
        Matrix::with_vector(self.rows, self.cols, new_data)
    }

    pub fn set_value(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    pub fn add(&self, m: &Matrix) -> Matrix {
        assert!(self.rows == m.rows && self.cols == m.cols);

        let data = self
            .data
            .iter()
            .zip(m.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Matrix::with_vector(self.rows, self.cols, data)
    }

    pub fn add_assign(&mut self, m: &Matrix) {
        assert!(self.rows == m.rows && self.cols == m.cols);
        for (a, b) in self.data.iter_mut().zip(m.data.iter()) {
            *a += *b;
        }
    }

    pub fn scale(&self, scalar: f32) -> Matrix {
        let data = self.data.iter().map(|x| x * scalar).collect();
        Matrix::with_vector(self.rows, self.cols, data)
    }

    pub fn elem_mul(&mut self, m: &Matrix) {
        assert_eq!(self.data.len(), m.data.len());

        for (a, b) in self.data.iter_mut().zip(m.data.iter()) {
            *a *= *b;
        }
    }

    pub fn dv_scalar(&self, dv: f32) -> Matrix {
        let data = self.data.iter().map(|x| x / dv).collect();
        Matrix::with_vector(self.rows, self.cols, data)
    }

    pub fn mul(&self, b: &Matrix) -> Matrix {
        assert!(self.cols == b.rows);
        let mut data = vec![0.0; self.rows * b.cols];
        for i in 0..self.rows {
            for j in 0..b.cols {
                let mut sum: f32 = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * b.data[k * b.cols + j];
                }
                data[i * b.cols + j] = sum;
            }
        }
        return Matrix::with_vector(self.rows, b.cols, data);
    }

    pub fn mul_transpose(&self, b_transpose: &Matrix) -> Matrix {
        assert!(self.cols == b_transpose.cols);
        let n = b_transpose.rows;
        let k_len = self.cols;
        let mut data = vec![0.0; self.rows * n];
        data.par_chunks_mut(n.max(1))
            .enumerate()
            .for_each(|(i, out_row)| {
                let a_row = &self.data[i * k_len..(i + 1) * k_len];
                for (j, out) in out_row.iter_mut().enumerate() {
                    let b_row = &b_transpose.data[j * k_len..(j + 1) * k_len];
                    let mut sum: f32 = 0.0;
                    for k in 0..k_len {
                        sum += a_row[k] * b_row[k];
                    }
                    *out = sum;
                }
            });
        return Matrix::with_vector(self.rows, n, data);
    }

    pub fn causal_mask(&mut self) {
        for row in 0..self.rows {
            for col in (row + 1)..self.cols {
                self.data[row * self.cols + col] = f32::NEG_INFINITY;
            }
        }
    }

    pub fn swish(&mut self) {
        for x in &mut self.data {
            let e = (-*x).exp();
            *x = *x / (1.0 + e);
        }
    }

    /// d/dx SiLU(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    pub fn swish_derivation(&mut self) {
        for x in &mut self.data {
            let sig = 1.0 / (1.0 + (-*x).exp());
            *x = sig * (1.0 + *x * (1.0 - sig));
        }
    }

    pub fn split_qkv(&self) -> (Matrix, Matrix, Matrix) {
        let d = self.cols / 3;

        let mut q = Vec::with_capacity(self.rows * d);
        let mut k = Vec::with_capacity(self.rows * d);
        let mut v = Vec::with_capacity(self.rows * d);

        for i in 0..self.rows {
            let row_start = i * self.cols;

            q.extend_from_slice(&self.data[row_start..row_start + d]);
            k.extend_from_slice(&self.data[row_start + d..row_start + 2 * d]);
            v.extend_from_slice(&self.data[row_start + 2 * d..row_start + 3 * d]);
        }

        (
            Matrix::with_vector(self.rows, d, q),
            Matrix::with_vector(self.rows, d, k),
            Matrix::with_vector(self.rows, d, v),
        )
    }

    pub fn rope(&mut self) {
        self.rotate_pairs(1.0);
    }

    /// RoPE is a per-position rotation, so its gradient is the inverse rotation.
    pub fn rope_backward(&mut self) {
        self.rotate_pairs(-1.0);
    }

    fn rotate_pairs(&mut self, direction: f32) {
        let dim = self.cols as f32;

        for pos in 0..self.rows {
            let row_start = pos * self.cols;

            let mut i = 0;
            while i + 1 < self.cols {
                let idx = row_start + i;

                let x = self.data[idx];
                let y = self.data[idx + 1];

                let theta = direction * (pos as f32) / 10000_f32.powf((i as f32) / dim);
                let cos = theta.cos();
                let sin = theta.sin();

                self.data[idx] = x * cos - y * sin;
                self.data[idx + 1] = x * sin + y * cos;

                i += 2;
            }
        }
    }
}

impl DivAssign<f32> for Matrix {
    fn div_assign(&mut self, rhs: f32) {
        for v in &mut self.data {
            *v /= rhs;
        }
    }
}

pub fn softmax(m: &Matrix) -> Matrix {
    let mut result = vec![0.0; m.data.len()];

    for i in 0..m.rows {
        let row_start = i * m.cols;
        let row_end = row_start + m.cols;
        let row = &m.data[row_start..row_end];

        // 1. find max
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // 2. exp
        let exp: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();

        // 3. sum
        let sum: f32 = exp.iter().sum();

        // 4. normalize
        for j in 0..m.cols {
            result[row_start + j] = exp[j] / sum;
        }
    }

    Matrix::with_vector(m.rows, m.cols, result)
}

pub fn derivate_of_softmax(s: &Matrix) -> Matrix {
    let mut result = vec![0.0; s.data.len() * s.cols];

    for i in 0..s.rows {
        let row_start = i * s.cols;
        let row_end = row_start + s.cols;
        let row = &s.data[row_start..row_end];

        for j in 0..s.cols {
            for k in 0..s.cols {
                let delta = if j == k { 1.0 } else { 0.0 };
                let jacobian_start = (row_start + j) * s.cols;
                result[jacobian_start + k] = row[j] * (delta - row[k]);
            }
        }
    }

    Matrix::with_vector(s.rows * s.cols, s.cols, result)
}

/// Backward of row-wise softmax without building the Jacobian:
/// dS = P * (dP - rowsum(dP * P))
pub fn softmax_backward(p: &Matrix, dp: &Matrix) -> Matrix {
    assert!(p.rows == dp.rows && p.cols == dp.cols);
    let mut ds = vec![0.0; p.data.len()];

    for i in 0..p.rows {
        let start = i * p.cols;
        let end = start + p.cols;
        let p_row = &p.data[start..end];
        let dp_row = &dp.data[start..end];

        let dot: f32 = p_row.iter().zip(dp_row).map(|(a, b)| a * b).sum();
        for j in 0..p.cols {
            ds[start + j] = p_row[j] * (dp_row[j] - dot);
        }
    }

    Matrix::with_vector(p.rows, p.cols, ds)
}

/// Inverse of `split_qkv`: [rows, d] x 3 -> [rows, 3d]
pub fn merge_qkv(q: &Matrix, k: &Matrix, v: &Matrix) -> Matrix {
    let d = q.cols;
    let mut data = Vec::with_capacity(q.rows * d * 3);
    for i in 0..q.rows {
        let row = i * d..(i + 1) * d;
        data.extend_from_slice(&q.data[row.clone()]);
        data.extend_from_slice(&k.data[row.clone()]);
        data.extend_from_slice(&v.data[row]);
    }
    Matrix::with_vector(q.rows, d * 3, data)
}

pub fn get_head(q: &Matrix, head: usize, head_dim: usize) -> Matrix {
    let mut data = Vec::with_capacity(q.rows * head_dim);

    for r in 0..q.rows {
        let start = r * q.cols + head * head_dim;
        let end = start + head_dim;

        data.extend_from_slice(&q.data[start..end]);
    }

    Matrix {
        rows: q.rows,
        cols: head_dim,
        data,
    }
}

pub fn concat_heads(v_m: Vec<Matrix>) -> Matrix {
    let rows = v_m[0].rows;
    let cols = v_m[0].cols * v_m.len();
    let mut data = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for m in v_m.iter() {
            let start = row * m.cols;
            let end = start + m.cols;
            data.extend_from_slice(&m.data[start..end]);
        }
    }
    Matrix::with_vector(rows, cols, data)
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let array_string = self
            .data
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        write!(
            f,
            "rows: {} cols: {} n: {} array: [{}]",
            self.rows,
            self.cols,
            self.rows * self.cols,
            array_string
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_transpose_matches_mul() {
        let a = Matrix::random(5, 7);
        let b = Matrix::random(7, 3);
        let expected = a.mul(&b);
        let got = a.mul_transpose(&b.transpose());
        assert_eq!((got.rows, got.cols), (5, 3));
        for (x, y) in got.data.iter().zip(&expected.data) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rope_backward_inverts_rope() {
        let original = Matrix::random(6, 8);
        let mut m = original.clone();
        m.rope();
        m.rope_backward();
        for (x, y) in m.data.iter().zip(&original.data) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn test_merge_qkv_inverts_split() {
        let qkv = Matrix::random(4, 12);
        let (q, k, v) = qkv.split_qkv();
        assert_eq!(merge_qkv(&q, &k, &v).data, qkv.data);
    }

    #[test]
    fn test_swish_derivation() {
        let x = Matrix::with_vector(1, 3, vec![-2.0, 0.0, 1.5]);
        let mut d = x.clone();
        d.swish_derivation();
        let eps = 1e-3;
        for i in 0..3 {
            let mut plus = Matrix::with_vector(1, 1, vec![x.data[i] + eps]);
            let mut minus = Matrix::with_vector(1, 1, vec![x.data[i] - eps]);
            plus.swish();
            minus.swish();
            let numeric = (plus.data[0] - minus.data[0]) / (2.0 * eps);
            assert!((d.data[i] - numeric).abs() < 1e-3);
        }
    }
}
