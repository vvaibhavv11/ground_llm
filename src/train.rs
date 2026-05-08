use std::thread;

use crate::{
    matrix::{concat_heads, get_head, softmax, Matrix},
    mlp::Mlp,
};
use rand::prelude::*;

const DIMENSIONS: usize = 512;
const NHEAD: usize = 8;
const VOCAB_SIZE: usize = 50256;
const HEAD_DIMENSIONS: usize = 64;

fn train_model(data: Vec<u16>) {
    let embedding = Matrix::random(VOCAB_SIZE, DIMENSIONS);
    let mut rng = rand::rng();
    let mut x_data = vec![];

    for token in &data {
        x_data.extend(embedding.get_row(*token as usize));
    }

    let mut x = Matrix::with_vector(data.len(), DIMENSIONS, x_data);

    let norm_x = x.rms_norm();

    let _w_qkv: Vec<f32> = (0..DIMENSIONS * 3 * DIMENSIONS)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    let w_qkv = Matrix::with_vector(DIMENSIONS, 3 * DIMENSIONS, _w_qkv);
    let qkv = norm_x.mul_transpose(&w_qkv.transpose());
    let (q, k, v) = qkv.split_qkv();
    let mut heads_output: Vec<Matrix> = vec![];
    for head in 0..NHEAD {
        let mut q_head = get_head(&q, head, HEAD_DIMENSIONS);
        let mut k_head = get_head(&k, head, HEAD_DIMENSIONS);
        let v_head = get_head(&v, head, HEAD_DIMENSIONS);
        q_head.rope();
        k_head.rope();
        let qk = q_head.mul_transpose(&k_head);
        let mut dev_answer = qk.dv_scalar((HEAD_DIMENSIONS as f32).sqrt());
        dev_answer.causal_mask();
        let final_marix = softmax(&dev_answer);
        let attention = final_marix.mul_transpose(&v_head.transpose());
        heads_output.push(attention);
    }
    let concat = concat_heads(heads_output);
    let w_o = Matrix::random(DIMENSIONS, DIMENSIONS);
    let attn_out = concat.mul_transpose(&w_o.transpose());
    x = x.add(&attn_out);
    let norm_x2 = x.rms_norm();
    let mut ffn = Mlp::new(x.rows, DIMENSIONS);
    let ffn_result = ffn.feedforward(norm_x2);
    x = x.add(&ffn_result);
    let lm_head = Matrix::random(DIMENSIONS, VOCAB_SIZE);
    let logits = x.mul_transpose(&lm_head.transpose());
    let logits_softmax = softmax(&logits);
}
