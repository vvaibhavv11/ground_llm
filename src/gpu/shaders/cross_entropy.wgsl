// One workgroup per row of logits. Writes the row's loss -ln(p[target]) and
// replaces the logits in place with d_logits = (softmax - one_hot) * inv_n.

struct Params {
    rows: u32, vocab: u32, inv_n: f32, _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<storage, read_write> row_loss: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = wg.x + wg.y * 65535u;
    if (row >= p.rows) {
        return;
    }
    let base = row * p.vocab;

    var m = -3.4e38;
    for (var j = li; j < p.vocab; j += 256u) {
        m = max(m, logits[base + j]);
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
    for (var j = li; j < p.vocab; j += 256u) {
        sum += exp(logits[base + j] - row_max);
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

    let tgt = targets[row];
    for (var j = li; j < p.vocab; j += 256u) {
        let prob = exp(logits[base + j] - row_max) / total;
        if (j == tgt) {
            row_loss[row] = -log(max(prob, 1e-9));
            logits[base + j] = (prob - 1.0) * p.inv_n;
        } else {
            logits[base + j] = prob * p.inv_n;
        }
    }
}
