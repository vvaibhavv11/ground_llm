use std::thread;

use crate::matrix::{concat_heads, get_head, softmax, Matrix};
use rand::prelude::*;

const DIMENSIONS: usize = 512;
const NHEAD: usize = 8;
const HEAD_DIMENSIONS: usize = 64;

fn train_model(data: Vec<u16>) {
    let mut rng = rand::rng();
    let values: Vec<f32> = (0..data.len() * DIMENSIONS)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    let mut x = Matrix::with_vector(data.len(), DIMENSIONS, values);

    x.rms_norm();

    let _w_qkv: Vec<f32> = (0..DIMENSIONS * 3 * DIMENSIONS)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    let w_qkv = Matrix::with_vector(DIMENSIONS, 3 * DIMENSIONS, _w_qkv);
    let qkv = x.mul_transpose(&w_qkv.transpose());
    let (q, k, v) = qkv.split_qkv();
    let mut heads_output: Vec<Matrix> = vec![];
    for head in 0..NHEAD {
        let mut q_head = get_head(&q, head, HEAD_DIMENSIONS);
        let mut k_head = get_head(&k, head, HEAD_DIMENSIONS);
        let v_head = get_head(&v, head, HEAD_DIMENSIONS);
        q_head.rope();
        k_head.rope();
        let qk = q_head.mul_transpose(&k_head);
        let dev_answer = qk.dv_scalar((HEAD_DIMENSIONS as f32).sqrt());
        let final_marix = softmax(&dev_answer);
        let attention = final_marix.mul_transpose(&v_head.transpose());
        heads_output.push(attention);
    }
    let concat = concat_heads(heads_output);
    let w_o = Matrix::random(DIMENSIONS, DIMENSIONS);
    let attn_out = concat.mul_transpose(&w_o.transpose());
    x = x.add(&attn_out);
    x.rms_norm();
}
