// In place: d = scale * P * (dP - rowsum(dP * P)), one workgroup per row.

struct Params {
    rows: u32, cols: u32, scale: f32, _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> d: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = wg.x + wg.y * 65535u;
    if (row >= p.rows) {
        return;
    }
    let base = row * p.cols;

    var dot = 0.0;
    for (var j = li; j < p.cols; j += 256u) {
        dot += probs[base + j] * d[base + j];
    }
    red[li] = dot;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (li < st) {
            red[li] += red[li + st];
        }
        workgroupBarrier();
    }
    let total = red[0];

    for (var j = li; j < p.cols; j += 256u) {
        d[base + j] = p.scale * probs[base + j] * (d[base + j] - total);
    }
}
