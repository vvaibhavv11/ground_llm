// d_h1 = SiLU'(h1) * h2 * d_out,  d_h2 = SiLU(h1) * d_out

struct Params {
    n: u32, _pad0: u32, _pad1: u32, _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> h1: array<f32>;
@group(0) @binding(2) var<storage, read> h2: array<f32>;
@group(0) @binding(3) var<storage, read> d_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_h1: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_h2: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= p.n) {
        return;
    }
    let x = h1[i];
    let sig = 1.0 / (1.0 + exp(-x));
    d_h1[i] = sig * (1.0 + x * (1.0 - sig)) * h2[i] * d_out[i];
    d_h2[i] = x * sig * d_out[i];
}
