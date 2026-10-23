// out = SiLU(h1) * h2

struct Params {
    n: u32, _pad0: u32, _pad1: u32, _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> h1: array<f32>;
@group(0) @binding(2) var<storage, read> h2: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= p.n) {
        return;
    }
    let x = h1[i];
    out[i] = x / (1.0 + exp(-x)) * h2[i];
}
