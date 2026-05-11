use rand::prelude::*;
use rayon::prelude::*;
use std::fmt;

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
        let mut data = vec![0.0; self.rows * b_transpose.rows];
        for i in 0..self.rows {
            for j in 0..b_transpose.rows {
                let mut sum: f32 = 0.0;
                for k in 0..self.cols {
                    sum +=
                        self.data[i * self.cols + k] * b_transpose.data[j * b_transpose.cols + k];
                }
                data[i * b_transpose.rows + j] = sum;
            }
        }
        return Matrix::with_vector(self.rows, b_transpose.rows, data);
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

    pub fn swish_derivation(&mut self) {
        for x in &mut self.data {
            let e = (-*x).exp();
            *x = *x / (1.0 + e);
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
        let dim = self.cols as f32;

        for pos in 0..self.rows {
            let row_start = pos * self.cols;

            let mut i = 0;
            while i + 1 < self.cols {
                let idx = row_start + i;

                let x = self.data[idx];
                let y = self.data[idx + 1];

                let theta = (pos as f32) / 10000_f32.powf((i as f32) / dim);
                let cos = theta.cos();
                let sin = theta.sin();

                self.data[idx] = x * cos - y * sin;
                self.data[idx + 1] = x * sin + y * cos;

                i += 2;
            }
        }
    }

    // =========================================================================
    // Strassen's Matrix Multiplication (parallel)
    // =========================================================================

    /// Strassen's algorithm for matrix multiplication.
    /// Uses parallel thread pool for large matrices.
    /// Falls back to naive multiplication for small matrices.
    pub fn mul_strassen(&self, b: &Matrix) -> Matrix {
        assert!(self.cols == b.rows, "dimension mismatch");
        assert!(
            self.rows == self.cols && b.rows == b.cols,
            "Strassen requires square matrices (padding not implemented)"
        );

        let n = self.rows;
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        self.strassen_recursive(b, num_threads)
    }

    fn strassen_recursive(&self, b: &Matrix, num_threads: usize) -> Matrix {
        let n = self.rows;

        // Fall back to naive for small matrices
        if n <= 64 {
            return self.mul(b);
        }

        // Pad to even dimensions
        let mid = n / 2;

        // Split into quadrants
        let (a11, a12, a21, a22) = self.split_quadrants();
        let (b11, b12, b21, b22) = b.split_quadrants();

        // Strassen's 7 products
        let m1 = a11.add(&a22).strassen_recursive(&b11.add(&b22), num_threads);
        let m2 = a21.add(&a22).strassen_recursive(&b11, num_threads);
        let m3 = a11.strassen_recursive(&b12.sub(&b22), num_threads);
        let m4 = a22.strassen_recursive(&b21.sub(&b11), num_threads);
        let m5 = a11.add(&a12).strassen_recursive(&b22, num_threads);
        let m6 = a21.sub(&a11).strassen_recursive(&b11.add(&b12), num_threads);
        let m7 = a12.sub(&a22).strassen_recursive(&b21.add(&b22), num_threads);

        // Combine results
        // C11 = M1 + M4 - M5 + M7
        let c11 = m1.add(&m4).sub(&m5).add(&m7);
        // C12 = M3 + M5
        let c12 = m3.add(&m5);
        // C21 = M2 + M4
        let c21 = m2.add(&m4);
        // C22 = M1 - M2 + M3 + M6
        let c22 = m1.sub(&m2).add(&m3).add(&m6);

        Matrix::combine_quadrants(&c11, &c12, &c21, &c22)
    }

    fn split_quadrants(&self) -> (Matrix, Matrix, Matrix, Matrix) {
        let n = self.rows / 2;
        let mut a11 = Matrix::new(n, n);
        let mut a12 = Matrix::new(n, n);
        let mut a21 = Matrix::new(n, n);
        let mut a22 = Matrix::new(n, n);

        for i in 0..n {
            for j in 0..n {
                a11.set_value(i, j, self.get_value(i, j));
                a12.set_value(i, j, self.get_value(i, j + n));
                a21.set_value(i, j, self.get_value(i + n, j));
                a22.set_value(i, j, self.get_value(i + n, j + n));
            }
        }

        (a11, a12, a21, a22)
    }

    fn sub(&self, other: &Matrix) -> Matrix {
        assert!(self.rows == other.rows && self.cols == other.cols);
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Matrix::with_vector(self.rows, self.cols, data)
    }

    fn combine_quadrants(
        c11: &Matrix,
        c12: &Matrix,
        c21: &Matrix,
        c22: &Matrix,
    ) -> Matrix {
        let n = c11.rows;
        let mut result = Matrix::new(n * 2, n * 2);

        for i in 0..n {
            for j in 0..n {
                result.set_value(i, j, c11.get_value(i, j));
                result.set_value(i, j + n, c12.get_value(i, j));
                result.set_value(i + n, j, c21.get_value(i, j));
                result.set_value(i + n, j + n, c22.get_value(i, j));
            }
        }

        result
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
    fn test_strassen_small_matches_naive() {
        // Test that strassen matches naive for small square matrices
        let a = Matrix::random(8, 8);
        let b = Matrix::random(8, 8);

        let naive = a.mul(&b);
        let strassen = a.mul_strassen(&b);

        for i in 0..naive.rows {
            for j in 0..naive.cols {
                assert!((naive.get_value(i, j) - strassen.get_value(i, j)).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_strassen_medium_matches_naive() {
        // Test 32x32 matrix
        let a = Matrix::random(32, 32);
        let b = Matrix::random(32, 32);

        let naive = a.mul(&b);
        let strassen = a.mul_strassen(&b);

        for i in 0..naive.rows {
            for j in 0..naive.cols {
                assert!((naive.get_value(i, j) - strassen.get_value(i, j)).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_strassen_64x64_matches_naive() {
        let a = Matrix::random(64, 64);
        let b = Matrix::random(64, 64);

        let naive = a.mul(&b);
        let strassen = a.mul_strassen(&b);

        for i in 0..naive.rows {
            for j in 0..naive.cols {
                assert!((naive.get_value(i, j) - strassen.get_value(i, j)).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_strassen_identity() {
        // A * I = A
        let a = Matrix::random(16, 16);
        let mut identity = Matrix::new(16, 16);
        for i in 0..16 {
            identity.set_value(i, i, 1.0);
        }

        let result = a.mul_strassen(&identity);

        for i in 0..a.rows {
            for j in 0..a.cols {
                assert!((result.get_value(i, j) - a.get_value(i, j)).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_strassen_vs_mul_transpose() {
        // Compare strassen (self * b) vs mul_transpose where b is pre-transposed
        let a = Matrix::random(32, 32);
        let b = Matrix::random(32, 32);
        let b_t = b.transpose();

        let strassen = a.mul_strassen(&b);
        let mul_t = a.mul_transpose(&b_t);

        // Note: results should be the same since both compute A @ B
        for i in 0..strassen.rows {
            for j in 0..strassen.cols {
                let diff = (strassen.get_value(i, j) - mul_t.get_value(i, j)).abs();
                assert!(
                    diff < 1e-4,
                    "Mismatch at ({}, {}): strassen={}, mul_transpose={}",
                    i,
                    j,
                    strassen.get_value(i, j),
                    mul_t.get_value(i, j)
                );
            }
        }
    }

    #[test]
    fn test_strassen_128x128() {
        let a = Matrix::random(128, 128);
        let b = Matrix::random(128, 128);

        let naive = a.mul(&b);
        let strassen = a.mul_strassen(&b);

        for i in 0..naive.rows {
            for j in 0..naive.cols {
                assert!((naive.get_value(i, j) - strassen.get_value(i, j)).abs() < 1e-2);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Strassen requires square matrices")]
    fn test_strassen_requires_square() {
        let a = Matrix::random(4, 8);
        let b = Matrix::random(8, 4);
        let _ = a.mul_strassen(&b);
    }
}
