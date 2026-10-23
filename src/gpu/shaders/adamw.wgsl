// AdamW update for one parameter tensor, with global-norm gradient clipping
// folded in (grad_norm_sq holds the squared norm of all gradients).

struct Params {
    off: u32, len: u32, decay: u32, _pad: u32,
}

struct Step {
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    weight_decay: f32, bias1: f32, bias2: f32, max_norm: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<uniform> st: Step;
@group(0) @binding(2) var<storage, read> grad_norm_sq: array<f32>;
@group(0) @binding(3) var<storage, read> grads: array<f32>;
@group(0) @binding(4) var<storage, read_write> params: array<f32>;
@group(0) @binding(5) var<storage, read_write> m: array<f32>;
@group(0) @binding(6) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= p.len) {
        return;
    }
    let idx = p.off + i;

    let norm = sqrt(grad_norm_sq[0]);
    var scale = 1.0;
    if (st.max_norm > 0.0 && norm > st.max_norm) {
        scale = st.max_norm / (norm + 1e-6);
    }
    let g = grads[idx] * scale;

    let m_new = st.beta1 * m[idx] + (1.0 - st.beta1) * g;
    let v_new = st.beta2 * v[idx] + (1.0 - st.beta2) * g * g;
    m[idx] = m_new;
    v[idx] = v_new;

    var w = params[idx];
    if (p.decay == 1u) {
        w -= st.lr * st.weight_decay * w;
    }
    w -= st.lr * (m_new / st.bias1) / (sqrt(v_new / st.bias2) + st.eps);
    params[idx] = w;
}
