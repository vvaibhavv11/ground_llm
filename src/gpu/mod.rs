//! Training on the GPU through wgpu (Metal on macOS, Vulkan on Linux).
//!
//! Weights, gradients, AdamW state and every activation stay in GPU memory for
//! the whole run. A step uploads the batch's token ids, replays a fixed list of
//! compute dispatches (forward, backward, optimizer) and reads back the
//! per-position losses and the gradient norm. The whole batch goes through each
//! layer at once as one [batch * seq_len, dim] matrix.

use std::borrow::Cow;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::train::{Backend, NORM_EPS, Transformer};

// ============================================================================
// Device setup
// ============================================================================

/// Describes the GPU training would use, or explains why there is none.
pub fn probe() -> Result<String, String> {
    let adapter = pollster::block_on(request_adapter())?;
    Ok(describe(&adapter.get_info()))
}

async fn request_adapter() -> Result<wgpu::Adapter, String> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("no GPU found: {}", e))?;

    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        return Err(format!("only a software renderer is available ({})", describe(&info)));
    }
    Ok(adapter)
}

fn describe(info: &wgpu::AdapterInfo) -> String {
    format!("{} ({:?})", info.name, info.backend)
}

// ============================================================================
// Kernels
// ============================================================================

struct Kernels {
    matmul: wgpu::ComputePipeline,
    embed_fwd: wgpu::ComputePipeline,
    embed_bwd: wgpu::ComputePipeline,
    rmsnorm_fwd: wgpu::ComputePipeline,
    rmsnorm_bwd: wgpu::ComputePipeline,
    rmsnorm_gain_bwd: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    softmax_causal: wgpu::ComputePipeline,
    softmax_bwd: wgpu::ComputePipeline,
    cross_entropy: wgpu::ComputePipeline,
    swiglu_fwd: wgpu::ComputePipeline,
    swiglu_bwd: wgpu::ComputePipeline,
    sum_squares: wgpu::ComputePipeline,
    adamw: wgpu::ComputePipeline,
}

impl Kernels {
    fn new(device: &wgpu::Device) -> Self {
        let load = |label: &str, src: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        macro_rules! kernel {
            ($name:literal) => {
                load($name, include_str!(concat!("shaders/", $name, ".wgsl")))
            };
        }
        Self {
            matmul: kernel!("matmul"),
            embed_fwd: kernel!("embed_fwd"),
            embed_bwd: kernel!("embed_bwd"),
            rmsnorm_fwd: kernel!("rmsnorm_fwd"),
            rmsnorm_bwd: kernel!("rmsnorm_bwd"),
            rmsnorm_gain_bwd: kernel!("rmsnorm_gain_bwd"),
            add: kernel!("add"),
            rope: kernel!("rope"),
            softmax_causal: kernel!("softmax_causal"),
            softmax_bwd: kernel!("softmax_bwd"),
            cross_entropy: kernel!("cross_entropy"),
            swiglu_fwd: kernel!("swiglu_fwd"),
            swiglu_bwd: kernel!("swiglu_bwd"),
            sum_squares: kernel!("sum_squares"),
            adamw: kernel!("adamw"),
        }
    }
}

/// Workgroups for a 1-D kernel of 256 threads over `n` elements. Large grids
/// spill into y; kernels flatten with `gid.x + gid.y * num_workgroups.x * 256`.
fn grid_1d(n: usize) -> [u32; 3] {
    grid_rows(n.div_ceil(256))
}

/// One workgroup per row; kernels flatten with `wg.x + wg.y * 65535`.
fn grid_rows(rows: usize) -> [u32; 3] {
    const MAX: usize = 65535;
    if rows <= MAX {
        [rows as u32, 1, 1]
    } else {
        [MAX as u32, rows.div_ceil(MAX) as u32, 1]
    }
}

fn f(x: f32) -> u32 {
    x.to_bits()
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ============================================================================
// Recording dispatches
// ============================================================================

struct Dispatch {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    groups: [u32; 3],
}

/// A matmul operand: element (i, j) of batch (outer, inner) lives at
/// `off + outer * bo + inner * bi + i * rs + j * cs`.
#[derive(Clone, Copy)]
struct View<'a> {
    buf: &'a wgpu::Buffer,
    off: usize,
    rs: usize,
    cs: usize,
    bo: usize,
    bi: usize,
}

impl<'a> View<'a> {
    /// Row-major matrix with `row_stride` floats per row.
    fn rows(buf: &'a wgpu::Buffer, off: usize, row_stride: usize) -> Self {
        Self { buf, off, rs: row_stride, cs: 1, bo: 0, bi: 0 }
    }

    /// Transpose of a row-major matrix with `row_stride` floats per row.
    fn transposed(buf: &'a wgpu::Buffer, off: usize, row_stride: usize) -> Self {
        Self { buf, off, rs: 1, cs: row_stride, bo: 0, bi: 0 }
    }

    fn batched(self, outer: usize, inner: usize) -> Self {
        Self { bo: outer, bi: inner, ..self }
    }
}

struct Recorder<'a> {
    device: &'a wgpu::Device,
    k: &'a Kernels,
    params: &'a wgpu::Buffer,
    grads: &'a wgpu::Buffer,
    ops: Vec<Dispatch>,
}

impl<'a> Recorder<'a> {
    fn push(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        uniform: &[u32],
        buffers: &[&wgpu::Buffer],
        groups: [u32; 3],
    ) {
        let uniform = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &u32_bytes(uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        self.push_with_uniform(pipeline, &uniform, buffers, groups);
    }

    /// Like `push`, with a caller-owned uniform buffer (rewritten every step).
    fn push_with_uniform(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        uniform: &wgpu::Buffer,
        buffers: &[&wgpu::Buffer],
        groups: [u32; 3],
    ) {
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }];
        for (i, b) in buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: b.as_entire_binding(),
            });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        self.ops.push(Dispatch {
            pipeline: pipeline.clone(),
            bind_group,
            groups,
        });
    }

    fn take(&mut self) -> Vec<Dispatch> {
        std::mem::take(&mut self.ops)
    }

    /// c = alpha * a @ b + beta * c for `batch` (m x k) @ (k x n) products.
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
        n_inner: usize,
        a: View,
        b: View,
        c: View,
        alpha: f32,
        beta: f32,
    ) {
        let u = [
            m, n, k, n_inner, a.off, a.rs, a.cs, a.bo, a.bi, b.off, b.rs, b.cs, b.bo, b.bi,
            c.off, c.rs, c.cs, c.bo, c.bi,
        ];
        let mut uniform: Vec<u32> = u.iter().map(|&x| x as u32).collect();
        uniform.extend([f(alpha), f(beta), 0, 0, 0]);
        let groups = [n.div_ceil(64) as u32, m.div_ceil(64) as u32, batch as u32];
        let k_ = self.k;
        self.push(&k_.matmul, &uniform, &[a.buf, b.buf, c.buf], groups);
    }

    /// y = x @ W (+ beta * y), W = params[w_off..] stored [d_in, d_out].
    fn linear(&mut self, x: &wgpu::Buffer, rows: usize, d_in: usize, d_out: usize, w_off: usize, y: &wgpu::Buffer) {
        let params = self.params;
        self.matmul(
            rows, d_out, d_in, 1, 1,
            View::rows(x, 0, d_in),
            View::rows(params, w_off, d_out),
            View::rows(y, 0, d_out),
            1.0, 0.0,
        );
    }

    /// dx = dy @ W^T (+ beta * dx)
    #[allow(clippy::too_many_arguments)]
    fn linear_dx(&mut self, dy: &wgpu::Buffer, rows: usize, d_in: usize, d_out: usize, w_off: usize, dx: &wgpu::Buffer, beta: f32) {
        let params = self.params;
        self.matmul(
            rows, d_in, d_out, 1, 1,
            View::rows(dy, 0, d_out),
            View::transposed(params, w_off, d_out),
            View::rows(dx, 0, d_in),
            1.0, beta,
        );
    }

    /// grad(W) = x^T @ dy
    fn linear_dw(&mut self, x: &wgpu::Buffer, dy: &wgpu::Buffer, rows: usize, d_in: usize, d_out: usize, w_off: usize) {
        let grads = self.grads;
        self.matmul(
            d_in, d_out, rows, 1, 1,
            View::transposed(x, 0, d_in),
            View::rows(dy, 0, d_out),
            View::rows(grads, w_off, d_out),
            1.0, 0.0,
        );
    }

    fn rmsnorm_fwd(&mut self, x: &wgpu::Buffer, rows: usize, dim: usize, gain_off: usize, out: &wgpu::Buffer, rstd: &wgpu::Buffer) {
        let (k, params) = (self.k, self.params);
        self.push(
            &k.rmsnorm_fwd,
            &[rows as u32, dim as u32, gain_off as u32, f(NORM_EPS)],
            &[x, params, out, rstd],
            grid_rows(rows),
        );
    }

    /// dx (+)= RMSNorm backward of dy; also writes the gain gradient.
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_bwd(&mut self, x: &wgpu::Buffer, rstd: &wgpu::Buffer, rows: usize, dim: usize, gain_off: usize, dy: &wgpu::Buffer, dx: &wgpu::Buffer, accumulate: bool) {
        let (k, params, grads) = (self.k, self.params, self.grads);
        self.push(
            &k.rmsnorm_bwd,
            &[rows as u32, dim as u32, gain_off as u32, accumulate as u32],
            &[x, rstd, params, dy, dx],
            grid_rows(rows),
        );
        self.push(
            &k.rmsnorm_gain_bwd,
            &[rows as u32, dim as u32, gain_off as u32, 0],
            &[x, rstd, dy, grads],
            [dim as u32, 1, 1],
        );
    }

    fn add(&mut self, a: &wgpu::Buffer, b: &wgpu::Buffer, out: &wgpu::Buffer, n: usize) {
        let k = self.k;
        self.push(&k.add, &[n as u32, 0, 0, 0], &[a, b, out], grid_1d(n));
    }
}

// ============================================================================
// GPU trainer
// ============================================================================

/// Where each parameter of one block lives in the flat parameter buffer.
struct BlockOffsets {
    attn_gain: usize,
    w_qkv: usize,
    w_o: usize,
    mlp_gain: usize,
    w1: usize,
    w2: usize,
    w3: usize,
}

/// Activations of one block, kept for the backward pass.
struct BlockActs {
    x_norm: wgpu::Buffer,
    attn_rstd: wgpu::Buffer,
    qkv: wgpu::Buffer, // Q and K already rotated by RoPE
    probs: wgpu::Buffer,
    concat: wgpu::Buffer,
    h: wgpu::Buffer,
    h_norm: wgpu::Buffer,
    mlp_rstd: wgpu::Buffer,
    h1: wgpu::Buffer,
    h2: wgpu::Buffer,
    gated: wgpu::Buffer,
}

pub struct GpuTrainer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    batch_size: usize,
    seq_len: usize,
    dim: usize,
    embed_off: usize,

    params: wgpu::Buffer,
    /// (offset, len) of each entry of `Transformer::params`, in that order.
    param_ranges: Vec<(usize, usize)>,
    grads: wgpu::Buffer,

    tokens: wgpu::Buffer,
    targets: wgpu::Buffer,
    row_loss: wgpu::Buffer,
    norm_sq: wgpu::Buffer,
    embed_uniform: wgpu::Buffer,
    embed_unique: wgpu::Buffer,
    embed_offsets: wgpu::Buffer,
    embed_positions: wgpu::Buffer,
    step_uniform: wgpu::Buffer,

    forward: Vec<Dispatch>,
    backward: Vec<Dispatch>,
    optimizer: Vec<Dispatch>,

    t: i32,
    weight_decay: f32,
    grad_clip: f32,
}

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.95;
const ADAM_EPS: f32 = 1e-8;
const NORM_PARTIALS: usize = 256;

impl GpuTrainer {
    pub fn new(
        model: &Transformer,
        batch_size: usize,
        seq_len: usize,
        weight_decay: f32,
        grad_clip: f32,
    ) -> Result<Self, String> {
        let adapter = pollster::block_on(request_adapter())?;
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ground_llm"),
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .map_err(|e| format!("failed to open GPU: {}", e))?;

        let c = &model.config;
        let (b, t, d, nh, hd) = (batch_size, seq_len, c.dimensions, c.n_heads, c.head_dim);
        let (ff, vocab, n_layers) = (c.mlp_hidden, c.vocab_size, c.n_layers);
        let n = b * t;

        // ---- Flat parameter layout, same order as Transformer::params ----
        let mut values = Vec::with_capacity(model.num_params());
        let mut param_ranges = Vec::new();
        let mut decays = Vec::new();
        for p in model.params() {
            param_ranges.push((values.len(), p.value.data.len()));
            decays.push(p.decay);
            values.extend_from_slice(&p.value.data);
        }
        let off = |i: usize| param_ranges[i].0;
        let embed_off = off(0);
        let blocks: Vec<BlockOffsets> = (0..n_layers)
            .map(|l| {
                let i = 1 + l * 7;
                BlockOffsets {
                    attn_gain: off(i),
                    w_qkv: off(i + 1),
                    w_o: off(i + 2),
                    mlp_gain: off(i + 3),
                    w1: off(i + 4),
                    w2: off(i + 5),
                    w3: off(i + 6),
                }
            })
            .collect();
        let final_gain = off(1 + n_layers * 7);
        let lm_off = off(2 + n_layers * 7);

        let largest = [values.len(), n * vocab, b * nh * t * t, n * 3 * d, n * ff]
            .into_iter()
            .max()
            .unwrap() as u64
            * 4;
        let max_binding = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        if largest > max_binding {
            return Err(format!(
                "a {} MB GPU buffer is needed but this GPU allows at most {} MB; lower batch_size or seq_len",
                largest >> 20,
                max_binding >> 20
            ));
        }

        // ---- Buffers ----
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let buf = |len: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (len.max(1) * 4) as u64,
                usage: storage,
                mapped_at_creation: false,
            })
        };
        let uniform_buf = |words: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (words * 4) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: &f32_bytes(&values),
            usage: storage,
        });
        let grads = buf(values.len());
        let adam_m = buf(values.len());
        let adam_v = buf(values.len());

        let tokens = buf(n);
        let targets = buf(n);
        let row_loss = buf(n);
        let norm_partials = buf(NORM_PARTIALS);
        let norm_sq = buf(1);
        let embed_uniform = uniform_buf(4);
        let embed_unique = buf(n);
        let embed_offsets = buf(n + 1);
        let embed_positions = buf(n);
        let step_uniform = uniform_buf(8);

        let resid: Vec<wgpu::Buffer> = (0..=n_layers).map(|_| buf(n * d)).collect();
        let acts: Vec<BlockActs> = (0..n_layers)
            .map(|_| BlockActs {
                x_norm: buf(n * d),
                attn_rstd: buf(n),
                qkv: buf(n * 3 * d),
                probs: buf(b * nh * t * t),
                concat: buf(n * d),
                h: buf(n * d),
                h_norm: buf(n * d),
                mlp_rstd: buf(n),
                h1: buf(n * ff),
                h2: buf(n * ff),
                gated: buf(n * ff),
            })
            .collect();
        let final_norm = buf(n * d);
        let final_rstd = buf(n);
        let logits = buf(n * vocab);
        let tmp = buf(n * d);

        let d_res = buf(n * d);
        let d_norm = buf(n * d);
        let d_qkv = buf(n * 3 * d);
        let d_probs = buf(b * nh * t * t);
        let d_concat = buf(n * d);
        let d_gated = buf(n * ff);
        let d_h1 = buf(n * ff);
        let d_h2 = buf(n * ff);

        let kernels = Kernels::new(&device);
        let k = &kernels;
        let mut rec = Recorder {
            device: &device,
            k,
            params: &params,
            grads: &grads,
            ops: Vec::new(),
        };

        let scale = 1.0 / (hd as f32).sqrt();
        let tt = t * t;
        let rope = |rec: &mut Recorder, buf: &wgpu::Buffer, direction: f32| {
            rec.push(
                &k.rope,
                &[n as u32, t as u32, (3 * d) as u32, d as u32, hd as u32, f(direction), 0, 0],
                &[buf],
                grid_1d(n * d),
            );
        };
        // Per-(sequence, head) views into the fused QKV rows and the score matrices.
        let q_view = |buf| View::rows(buf, 0, 3 * d).batched(t * 3 * d, hd);
        let k_view = |buf| View::rows(buf, d, 3 * d).batched(t * 3 * d, hd);
        let v_view = |buf| View::rows(buf, 2 * d, 3 * d).batched(t * 3 * d, hd);
        let kt_view = |buf| View::transposed(buf, d, 3 * d).batched(t * 3 * d, hd);
        let vt_view = |buf| View::transposed(buf, 2 * d, 3 * d).batched(t * 3 * d, hd);
        let s_view = |buf| View::rows(buf, 0, t).batched(nh * tt, tt);
        let st_view = |buf| View::transposed(buf, 0, t).batched(nh * tt, tt);
        let head_view = |buf| View::rows(buf, 0, d).batched(t * d, hd);
        let bh = b * nh;

        // ---- Forward ----
        rec.push(
            &k.embed_fwd,
            &[n as u32, d as u32, embed_off as u32, 0],
            &[&tokens, &params, &resid[0]],
            grid_1d(n * d),
        );
        for (l, (o, a)) in blocks.iter().zip(&acts).enumerate() {
            rec.rmsnorm_fwd(&resid[l], n, d, o.attn_gain, &a.x_norm, &a.attn_rstd);
            rec.linear(&a.x_norm, n, d, 3 * d, o.w_qkv, &a.qkv);
            rope(&mut rec, &a.qkv, 1.0);

            // P = softmax(mask(Q K^T / sqrt(head_dim))), O = P V
            rec.matmul(t, t, hd, bh, nh, q_view(&a.qkv), kt_view(&a.qkv), s_view(&a.probs), scale, 0.0);
            rec.push(&k.softmax_causal, &[(bh * t) as u32, t as u32, 0, 0], &[&a.probs], grid_rows(bh * t));
            rec.matmul(t, hd, t, bh, nh, s_view(&a.probs), v_view(&a.qkv), head_view(&a.concat), 1.0, 0.0);

            rec.linear(&a.concat, n, d, d, o.w_o, &tmp);
            rec.add(&resid[l], &tmp, &a.h, n * d);

            rec.rmsnorm_fwd(&a.h, n, d, o.mlp_gain, &a.h_norm, &a.mlp_rstd);
            rec.linear(&a.h_norm, n, d, ff, o.w1, &a.h1);
            rec.linear(&a.h_norm, n, d, ff, o.w2, &a.h2);
            rec.push(&k.swiglu_fwd, &[(n * ff) as u32, 0, 0, 0], &[&a.h1, &a.h2, &a.gated], grid_1d(n * ff));
            rec.linear(&a.gated, n, ff, d, o.w3, &tmp);
            rec.add(&a.h, &tmp, &resid[l + 1], n * d);
        }
        rec.rmsnorm_fwd(&resid[n_layers], n, d, final_gain, &final_norm, &final_rstd);
        rec.linear(&final_norm, n, d, vocab, lm_off, &logits);
        // Writes per-row losses and turns the logits into d_logits in place.
        rec.push(
            &k.cross_entropy,
            &[n as u32, vocab as u32, f(1.0 / n as f32), 0],
            &[&targets, &logits, &row_loss],
            grid_rows(n),
        );
        let forward = rec.take();

        // ---- Backward ----
        rec.linear_dx(&logits, n, d, vocab, lm_off, &d_norm, 0.0);
        rec.linear_dw(&final_norm, &logits, n, d, vocab, lm_off);
        rec.rmsnorm_bwd(&resid[n_layers], &final_rstd, n, d, final_gain, &d_norm, &d_res, false);

        // d_res holds the gradient of the residual stream as it flows down.
        for (l, (o, a)) in blocks.iter().zip(&acts).enumerate().rev() {
            // MLP: out = h + (SiLU(h_norm W1) * (h_norm W2)) W3
            rec.linear_dx(&d_res, n, ff, d, o.w3, &d_gated, 0.0);
            rec.linear_dw(&a.gated, &d_res, n, ff, d, o.w3);
            rec.push(
                &k.swiglu_bwd,
                &[(n * ff) as u32, 0, 0, 0],
                &[&a.h1, &a.h2, &d_gated, &d_h1, &d_h2],
                grid_1d(n * ff),
            );
            rec.linear_dx(&d_h1, n, d, ff, o.w1, &d_norm, 0.0);
            rec.linear_dx(&d_h2, n, d, ff, o.w2, &d_norm, 1.0);
            rec.linear_dw(&a.h_norm, &d_h1, n, d, ff, o.w1);
            rec.linear_dw(&a.h_norm, &d_h2, n, d, ff, o.w2);
            rec.rmsnorm_bwd(&a.h, &a.mlp_rstd, n, d, o.mlp_gain, &d_norm, &d_res, true);

            // Attention: h = x + concat W_o
            rec.linear_dx(&d_res, n, d, d, o.w_o, &d_concat, 0.0);
            rec.linear_dw(&a.concat, &d_res, n, d, d, o.w_o);
            // dP = dO V^T, dV = P^T dO
            rec.matmul(t, t, hd, bh, nh, head_view(&d_concat), vt_view(&a.qkv), s_view(&d_probs), 1.0, 0.0);
            rec.matmul(t, hd, t, bh, nh, st_view(&a.probs), head_view(&d_concat), v_view(&d_qkv), 1.0, 0.0);
            // dS = P * (dP - rowsum(dP * P)) / sqrt(head_dim), in place
            rec.push(
                &k.softmax_bwd,
                &[(bh * t) as u32, t as u32, f(scale), 0],
                &[&a.probs, &d_probs],
                grid_rows(bh * t),
            );
            // dQ = dS K, dK = dS^T Q
            rec.matmul(t, hd, t, bh, nh, s_view(&d_probs), k_view(&a.qkv), q_view(&d_qkv), 1.0, 0.0);
            rec.matmul(t, hd, t, bh, nh, st_view(&d_probs), q_view(&a.qkv), k_view(&d_qkv), 1.0, 0.0);
            rope(&mut rec, &d_qkv, -1.0);
            rec.linear_dx(&d_qkv, n, d, 3 * d, o.w_qkv, &d_norm, 0.0);
            rec.linear_dw(&a.x_norm, &d_qkv, n, d, 3 * d, o.w_qkv);
            rec.rmsnorm_bwd(&resid[l], &a.attn_rstd, n, d, o.attn_gain, &d_norm, &d_res, true);
        }
        rec.push_with_uniform(
            &k.embed_bwd,
            &embed_uniform,
            &[&embed_unique, &embed_offsets, &embed_positions, &d_res, &grads],
            grid_1d(n * d),
        );
        let backward = rec.take();

        // ---- Gradient norm + AdamW ----
        rec.push(
            &k.sum_squares,
            &[values.len() as u32, 1, 0, 0],
            &[&grads, &norm_partials],
            [NORM_PARTIALS as u32, 1, 1],
        );
        rec.push(
            &k.sum_squares,
            &[NORM_PARTIALS as u32, 0, 0, 0],
            &[&norm_partials, &norm_sq],
            [1, 1, 1],
        );
        for (&(offset, len), &decay) in param_ranges.iter().zip(&decays) {
            rec.push(
                &k.adamw,
                &[offset as u32, len as u32, decay as u32, 0],
                &[&step_uniform, &norm_sq, &grads, &params, &adam_m, &adam_v],
                grid_1d(len),
            );
        }
        let optimizer = rec.take();

        Ok(Self {
            device,
            queue,
            batch_size,
            seq_len,
            dim: d,
            embed_off,
            params,
            param_ranges,
            grads,
            tokens,
            targets,
            row_loss,
            norm_sq,
            embed_uniform,
            embed_unique,
            embed_offsets,
            embed_positions,
            step_uniform,
            forward,
            backward,
            optimizer,
            t: 0,
            weight_decay,
            grad_clip,
        })
    }

    /// Uploads token ids and targets; returns the flat input ids.
    fn upload_batch(&self, batch: &[(&[u16], &[u16])]) -> Vec<u32> {
        assert_eq!(batch.len(), self.batch_size);
        let mut inputs = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(inputs.capacity());
        for (i, t) in batch {
            assert!(i.len() == self.seq_len && t.len() == self.seq_len);
            inputs.extend(i.iter().map(|&x| x as u32));
            targets.extend(t.iter().map(|&x| x as u32));
        }
        self.queue.write_buffer(&self.tokens, 0, &u32_bytes(&inputs));
        self.queue.write_buffer(&self.targets, 0, &u32_bytes(&targets));
        inputs
    }

    /// Groups positions by token so the embedding backward can sum each
    /// token's rows without atomics.
    fn upload_embedding_groups(&self, inputs: &[u32]) {
        let mut order: Vec<u32> = (0..inputs.len() as u32).collect();
        order.sort_by_key(|&p| inputs[p as usize]);

        let mut unique = Vec::new();
        let mut offsets = Vec::new();
        for (s, &p) in order.iter().enumerate() {
            let token = inputs[p as usize];
            if unique.last() != Some(&token) {
                unique.push(token);
                offsets.push(s as u32);
            }
        }
        offsets.push(order.len() as u32);

        self.queue.write_buffer(&self.embed_unique, 0, &u32_bytes(&unique));
        self.queue.write_buffer(&self.embed_offsets, 0, &u32_bytes(&offsets));
        self.queue.write_buffer(&self.embed_positions, 0, &u32_bytes(&order));
        self.queue.write_buffer(
            &self.embed_uniform,
            0,
            &u32_bytes(&[unique.len() as u32, self.dim as u32, self.embed_off as u32, 0]),
        );
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, ops: &[&[Dispatch]]) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        for op in ops.iter().flat_map(|o| o.iter()) {
            pass.set_pipeline(&op.pipeline);
            pass.set_bind_group(0, &op.bind_group, &[]);
            let [x, y, z] = op.groups;
            pass.dispatch_workgroups(x, y, z);
        }
    }

    /// Submits `encoder` and waits for it, returning `buf[..len]` of each `reads`.
    fn finish_and_read(&self, mut encoder: wgpu::CommandEncoder, reads: &[(&wgpu::Buffer, usize)]) -> Vec<Vec<f32>> {
        let total: usize = reads.iter().map(|&(_, len)| len).sum();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (total * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut at = 0;
        for &(buf, len) in reads {
            encoder.copy_buffer_to_buffer(buf, 0, &staging, at, (len * 4) as u64);
            at += (len * 4) as u64;
        }
        self.queue.submit([encoder.finish()]);

        let (tx, rx) = mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        rx.recv()
            .expect("GPU map callback dropped")
            .expect("failed to read GPU buffer");

        let values: Vec<f32> = {
            let bytes = staging.get_mapped_range(..).expect("GPU buffer not mapped");
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        staging.unmap();

        let mut rest = values.as_slice();
        reads
            .iter()
            .map(|&(_, len)| {
                let (head, tail) = rest.split_at(len);
                rest = tail;
                head.to_vec()
            })
            .collect()
    }
}

impl Backend for GpuTrainer {
    fn step(&mut self, _model: &mut Transformer, batch: &[(&[u16], &[u16])], lr: f32) -> (f32, f32) {
        let inputs = self.upload_batch(batch);
        self.upload_embedding_groups(&inputs);

        self.t += 1;
        let step = [
            lr,
            BETA1,
            BETA2,
            ADAM_EPS,
            self.weight_decay,
            1.0 - BETA1.powi(self.t),
            1.0 - BETA2.powi(self.t),
            self.grad_clip,
        ];
        self.queue.write_buffer(&self.step_uniform, 0, &f32_bytes(&step));

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.grads, 0, None);
        self.encode(&mut encoder, &[&self.forward, &self.backward, &self.optimizer]);
        let out = self.finish_and_read(encoder, &[(&self.row_loss, inputs.len()), (&self.norm_sq, 1)]);

        let loss = out[0].iter().sum::<f32>() / out[0].len() as f32;
        (loss, out[1][0].sqrt())
    }

    fn loss(&mut self, _model: &mut Transformer, windows: &[(&[u16], &[u16])]) -> f32 {
        let mut total = 0.0;
        for chunk in windows.chunks(self.batch_size) {
            // Pad a short last chunk; the padding rows are ignored below.
            let mut batch = chunk.to_vec();
            batch.resize(self.batch_size, chunk[0]);
            self.upload_batch(&batch);

            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.encode(&mut encoder, &[&self.forward]);
            let losses = self.finish_and_read(encoder, &[(&self.row_loss, self.batch_size * self.seq_len)]);
            total += losses[0][..chunk.len() * self.seq_len].iter().sum::<f32>();
        }
        total / (windows.len() * self.seq_len) as f32
    }

    fn sync(&mut self, model: &mut Transformer) {
        let total = self.param_ranges.last().map_or(0, |&(o, l)| o + l);
        let encoder = self.device.create_command_encoder(&Default::default());
        let values = self.finish_and_read(encoder, &[(&self.params, total)]).remove(0);
        for (p, &(offset, len)) in model.params_mut().into_iter().zip(&self.param_ranges) {
            p.value.data.copy_from_slice(&values[offset..offset + len]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{Config, cross_entropy};
    use rand::prelude::*;

    impl GpuTrainer {
        fn grads(&self) -> Vec<f32> {
            let total = self.param_ranges.last().map_or(0, |&(o, l)| o + l);
            let encoder = self.device.create_command_encoder(&Default::default());
            self.finish_and_read(encoder, &[(&self.grads, total)]).remove(0)
        }
    }

    /// Sizes chosen so no matmul dimension is a multiple of the 64-wide tile.
    fn config() -> Config {
        Config {
            dimensions: 48,
            n_heads: 4,
            vocab_size: 70,
            head_dim: 12,
            mlp_hidden: 100,
            n_layers: 2,
            max_seq_len: 32,
        }
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let diff: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
        let norm: f32 = a.iter().chain(b).map(|x| x * x).sum::<f32>().sqrt();
        diff / norm.max(1e-12)
    }

    fn gpu_or_skip() -> bool {
        match probe() {
            Ok(_) => true,
            Err(e) => {
                eprintln!("skipping GPU test: {}", e);
                false
            }
        }
    }

    /// One step on the GPU must match the CPU: loss, every gradient, grad norm
    /// and the updated weights.
    #[test]
    fn test_gpu_step_matches_cpu() {
        if !gpu_or_skip() {
            return;
        }
        let mut rng = StdRng::seed_from_u64(7);
        let mut cpu = Transformer::new(config());
        // Bigger weights than the real init so the gradients are far from noise.
        for p in cpu.params_mut() {
            let offset = if p.decay { 0.0 } else { 1.0 };
            for v in p.value.data.iter_mut() {
                *v = offset + rng.random_range(-0.3..0.3);
            }
        }
        let mut gpu_model = cpu.clone();

        let (batch_size, seq_len) = (3, 21);
        let tokens: Vec<u16> = (0..batch_size * (seq_len + 1)).map(|_| rng.random_range(0..70)).collect();
        // Repeat a token inside and across windows to exercise the embedding backward.
        let mut tokens = tokens;
        tokens[5] = 9;
        tokens[6] = 9;
        tokens[30] = 9;
        let batch: Vec<(&[u16], &[u16])> = tokens
            .chunks(seq_len + 1)
            .map(|w| (&w[..seq_len], &w[1..]))
            .collect();

        // ---- CPU reference (no clipping so gradients stay raw) ----
        cpu.zero_grad();
        let mut cpu_loss = 0.0;
        for &(inputs, targets) in &batch {
            let probs = cpu.forward(inputs);
            let (loss, mut d_logits) = cross_entropy(&probs, targets);
            d_logits /= batch_size as f32;
            cpu.backward(&d_logits);
            cpu_loss += loss / batch_size as f32;
        }
        let cpu_grads: Vec<Vec<f32>> = cpu.params().iter().map(|p| p.grad.data.clone()).collect();
        let cpu_norm = cpu_grads.iter().flatten().map(|g| g * g).sum::<f32>().sqrt();
        let mut optimizer = crate::optim::AdamW::new(0.1);
        optimizer.step(&mut cpu.params_mut(), 1e-3);

        // ---- GPU ----
        let mut gpu = GpuTrainer::new(&gpu_model, batch_size, seq_len, 0.1, 0.0).unwrap();
        let (gpu_loss, gpu_norm) = gpu.step(&mut gpu_model, &batch, 1e-3);
        let gpu_grads = gpu.grads();
        gpu.sync(&mut gpu_model);

        assert!((cpu_loss - gpu_loss).abs() < 1e-4, "loss cpu {} gpu {}", cpu_loss, gpu_loss);
        assert!((cpu_norm - gpu_norm).abs() / cpu_norm < 1e-4, "norm cpu {} gpu {}", cpu_norm, gpu_norm);
        for (i, (&(off, len), cg)) in gpu.param_ranges.iter().zip(&cpu_grads).enumerate() {
            let err = rel_err(cg, &gpu_grads[off..off + len]);
            assert!(err < 1e-4, "param {} gradient rel error {}", i, err);
        }
        // Adam's first step is ~lr * sign(g), so gradients that are ~0 can flip
        // sign from rounding alone; hence the looser bound than for gradients.
        for (i, (c, g)) in cpu.params().iter().zip(gpu_model.params()).enumerate() {
            let err = rel_err(&c.value.data, &g.value.data);
            assert!(err < 1e-3, "param {} weights rel error {}", i, err);
        }
    }

    #[test]
    fn test_gpu_loss_matches_cpu() {
        if !gpu_or_skip() {
            return;
        }
        let mut model = Transformer::new(config());
        let mut rng = StdRng::seed_from_u64(3);
        let tokens: Vec<u16> = (0..200).map(|_| rng.random_range(0..70)).collect();
        // 5 windows with batch_size 2 exercises the padded last chunk.
        let windows: Vec<(&[u16], &[u16])> = (0..5)
            .map(|i| (&tokens[i * 30..i * 30 + 16], &tokens[i * 30 + 1..i * 30 + 17]))
            .collect();

        let cpu: f32 = windows.iter().map(|&(i, t)| model.loss(i, t)).sum::<f32>() / 5.0;
        let mut gpu = GpuTrainer::new(&model, 2, 16, 0.1, 1.0).unwrap();
        let gpu_loss = gpu.loss(&mut model, &windows);
        assert!((cpu - gpu_loss).abs() < 1e-4, "cpu {} gpu {}", cpu, gpu_loss);
    }

    #[test]
    fn test_gpu_training_reduces_loss() {
        if !gpu_or_skip() {
            return;
        }
        let mut model = Transformer::new(config());
        let pattern = [1u16, 4, 9, 2, 7, 11, 3];
        let tokens: Vec<u16> = pattern.iter().cycle().take(400).cloned().collect();
        let cfg = crate::train::TrainConfig {
            steps: 300,
            batch_size: 4,
            seq_len: 8,
            lr: 1e-2,
            min_lr: 1e-3,
            warmup_steps: 10,
            weight_decay: 0.0,
            val_fraction: 0.1,
            eval_every: 100,
            eval_batches: 2,
            device: crate::train::Device::Gpu,
            ..Default::default()
        };

        let history = crate::train::train(&mut model, &tokens, &cfg, |_| true).unwrap();
        let last = *history.train_loss.last().unwrap();
        assert!(last < 0.2, "loss went from {} to {}", history.train_loss[0], last);
        // The synced CPU model must have learned too.
        let cpu_loss = model.loss(&tokens[..8], &tokens[1..9]);
        assert!(cpu_loss < 0.2, "CPU copy loss {}", cpu_loss);
    }
}
