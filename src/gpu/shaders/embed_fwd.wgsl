// out[r, d] = table[tokens[r], d]

struct Params {
    rows: u32, dim: u32, table_off: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= p.rows * p.dim) {
        return;
    }
    let r = i / p.dim;
    let d = i % p.dim;
    out[i] = params[p.table_off + tokens[r] * p.dim + d];
}
