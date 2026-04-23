use std::fmt;

const EXPO: f32 = 2.71828;

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

    pub fn transpose(&self) -> Matrix {
        let mut new_data = Vec::with_capacity(self.rows * self.cols);
        for i in 0..self.cols {
            for j in 0..self.rows {
                new_data.push(self.get_value(j, i));
            }
        }
        return Matrix::with_vector(self.cols, self.rows, new_data);
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

    pub fn relu(&self) -> Matrix {
        let data = self.data.iter().map(|x| x.max(0.0)).collect();

        Matrix::with_vector(self.rows, self.cols, data)
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
}

pub fn soft_max(m: &Matrix) -> Matrix {
    let sum_vector: f32 = m.data.iter().map(|x| EXPO.powf(*x)).sum();
    let softmax_vector = m.data.iter().map(|x| EXPO.powf(*x) / sum_vector).collect();
    Matrix::with_vector(m.rows, m.cols, softmax_vector)
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
