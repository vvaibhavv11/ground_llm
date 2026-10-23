// One workgroup per row: rstd = 1 / sqrt(mean(x^2) + eps), out = x * rstd * gain

struct Params {
    rows: u32, dim: u32, gain_off: u32, eps: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> rstd: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = wg.x + wg.y * 65535u;
    if (row >= p.rows) {
        return;
    }
    let base = row * p.dim;

    var sq = 0.0;
    for (var j = li; j < p.dim; j += 256u) {
        let v = x[base + j];
        sq += v * v;
    }
    red[li] = sq;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) {
            red[li] += red[li + s];
        }
        workgroupBarrier();
    }

    let r = 1.0 / sqrt(red[0] / f32(p.dim) + p.eps);
    for (var j = li; j < p.dim; j += 256u) {
        out[base + j] = x[base + j] * r * params[p.gain_off + j];
    }
    if (li == 0u) {
        rstd[row] = r;
    }
}
