// grad[token, d] = sum of d_x[pos, d] over every position holding `token`.
// The CPU groups positions by token (CSR: offsets into `positions`), so each
// thread owns one (token, d) output and no atomics are needed.

struct Params {
    n_unique: u32, dim: u32, table_off: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> unique_tokens: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> positions: array<u32>;
@group(0) @binding(4) var<storage, read> d_x: array<f32>;
@group(0) @binding(5) var<storage, read_write> grads: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= p.n_unique * p.dim) {
        return;
    }
    let u = i / p.dim;
    let d = i % p.dim;
    var sum = 0.0;
    for (var s = offsets[u]; s < offsets[u + 1u]; s++) {
        sum += d_x[positions[s] * p.dim + d];
    }
    grads[p.table_off + unique_tokens[u] * p.dim + d] = sum;
}
