// One workgroup per column: grad_gain[j] = sum over rows of dy * x * rstd

struct Params {
    rows: u32, dim: u32, gain_off: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> rstd: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> grads: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let col = wg.x;

    var sum = 0.0;
    for (var r = li; r < p.rows; r += 256u) {
        let idx = r * p.dim + col;
        sum += dy[idx] * x[idx] * rstd[r];
    }
    red[li] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) {
            red[li] += red[li + s];
        }
        workgroupBarrier();
    }

    if (li == 0u) {
        grads[p.gain_off + col] = red[0];
    }
}
