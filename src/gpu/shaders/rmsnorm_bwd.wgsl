// One workgroup per row:
//   dx = (dy * gain - x_hat * mean(dy * gain * x_hat)) * rstd,  x_hat = x * rstd
// With accumulate = 1 the result is added to `dx` (residual branch).

struct Params {
    rows: u32, dim: u32, gain_off: u32, accumulate: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> rstd: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<f32>;
@group(0) @binding(4) var<storage, read> dy: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = wg.x + wg.y * 65535u;
    if (row >= p.rows) {
        return;
    }
    let base = row * p.dim;
    let r = rstd[row];

    var dot = 0.0;
    for (var j = li; j < p.dim; j += 256u) {
        dot += dy[base + j] * params[p.gain_off + j] * x[base + j] * r;
    }
    red[li] = dot;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) {
            red[li] += red[li + s];
        }
        workgroupBarrier();
    }

    let mean = red[0] / f32(p.dim);
    for (var j = li; j < p.dim; j += 256u) {
        let x_hat = x[base + j] * r;
        let v = (dy[base + j] * params[p.gain_off + j] - x_hat * mean) * r;
        if (p.accumulate == 1u) {
            dx[base + j] += v;
        } else {
            dx[base + j] = v;
        }
    }
}
