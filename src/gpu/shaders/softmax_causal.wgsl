// In-place causal softmax, one workgroup per row of [batch * heads * seq, seq]
// scores. Row t only attends to columns 0..=t; later columns become 0.

struct Params {
    rows: u32, seq_len: u32, _pad0: u32, _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> s: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = wg.x + wg.y * 65535u;
    if (row >= p.rows) {
        return;
    }
    let base = row * p.seq_len;
    let visible = row % p.seq_len + 1u;

    var m = -3.4e38;
    for (var j = li; j < visible; j += 256u) {
        m = max(m, s[base + j]);
    }
    red[li] = m;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (li < st) {
            red[li] = max(red[li], red[li + st]);
        }
        workgroupBarrier();
    }
    let row_max = red[0];
    workgroupBarrier();

    var sum = 0.0;
    for (var j = li; j < visible; j += 256u) {
        sum += exp(s[base + j] - row_max);
    }
    red[li] = sum;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (li < st) {
            red[li] += red[li + st];
        }
        workgroupBarrier();
    }
    let total = red[0];

    for (var j = li; j < p.seq_len; j += 256u) {
        if (j < visible) {
            s[base + j] = exp(s[base + j] - row_max) / total;
        } else {
            s[base + j] = 0.0;
        }
    }
}
