// In-place RoPE over the first `n_pairs * 2` columns of each row (Q and K of
// a fused QKV row). Position = row % seq_len; direction -1 undoes the rotation
// (its own backward pass).

struct Params {
    rows: u32, seq_len: u32, row_stride: u32, n_pairs: u32,
    head_dim: u32, direction: f32, _pad0: u32, _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= p.rows * p.n_pairs) {
        return;
    }
    let row = i / p.n_pairs;
    let col = (i % p.n_pairs) * 2u;
    let pos = f32(row % p.seq_len);
    let in_head = f32(col % p.head_dim);

    let theta = p.direction * pos / pow(10000.0, in_head / f32(p.head_dim));
    let cos_t = cos(theta);
    let sin_t = sin(theta);

    let idx = row * p.row_stride + col;
    let x = buf[idx];
    let y = buf[idx + 1u];
    buf[idx] = x * cos_t - y * sin_t;
    buf[idx + 1u] = x * sin_t + y * cos_t;
}
