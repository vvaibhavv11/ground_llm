// out[workgroup] = sum of input[i]^2 (square = 1) or input[i] (square = 0)
// over a grid-stride slice. Run once over the gradients and once more over the
// partial sums with a single workgroup to get the global total.

struct Params {
    n: u32, square: u32, _pad0: u32, _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    var sum = 0.0;
    for (var i = gid.x; i < p.n; i += nwg.x * 256u) {
        let v = input[i];
        if (p.square == 1u) {
            sum += v * v;
        } else {
            sum += v;
        }
    }
    red[li] = sum;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (li < st) {
            red[li] += red[li + st];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        out[wg.x] = red[0];
    }
}
