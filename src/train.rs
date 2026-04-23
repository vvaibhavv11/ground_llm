use std::thread;

use crate::matrix::{get_head, Matrix};
use rand::prelude::*;

const DIMENSIONS: usize = 512;
const NHEAD: usize = 8;
const HEAD_DIMENSIONS: usize = 64;

fn train_model(data: Vec<u16>) {
    let mut rng = rand::rng();
    let values: Vec<f32> = (0..data.len() * DIMENSIONS)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    let x = Matrix::with_vector(data.len(), DIMENSIONS, values);

    let _w_qkv: Vec<f32> = (0..DIMENSIONS * 3 * DIMENSIONS)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    let w_qkv = Matrix::with_vector(DIMENSIONS, 3 * DIMENSIONS, _w_qkv);
    let qkv = x.mul_transpose(&w_qkv.transpose());
    let (q, k, v) = qkv.split_qkv();
    for head in 0..NHEAD {
        let q_head = get_head(&q, head, HEAD_DIMENSIONS);
        let k_head = get_head(&k, head, HEAD_DIMENSIONS);
        let qk = q_head.mul_transpose(&k_head);
        let dev_answer = qk.dv_scalar((HEAD_DIMENSIONS as f32).sqrt());
        

    }
}
