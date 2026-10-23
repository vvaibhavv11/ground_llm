// Batched, strided matmul: C = alpha * op(A) @ op(B) + beta * C
//
// Every operand is addressed through (offset, row stride, col stride), so
// transposes and per-head slices of a bigger buffer need no copies:
//   A(i, k) = a[a_base + i * a_rs + k * a_cs]
//   B(k, j) = b[b_base + k * b_rs + j * b_cs]
//   C(i, j) = c[c_base + i * c_rs + j * c_cs]
// The batch index z (workgroup_id.z) is split into (z / n_inner, z % n_inner)
// so attention can walk (sequence, head) pairs, each with its own stride.
//
// Each 16x16 workgroup computes a 64x64 tile of C; each thread a 4x4 block
// spread 16 apart so shared-memory reads and global writes stay contiguous.

struct Params {
    m: u32, n: u32, k: u32, n_inner: u32,
    a_off: u32, a_rs: u32, a_cs: u32, a_bo: u32,
    a_bi: u32, b_off: u32, b_rs: u32, b_cs: u32,
    b_bo: u32, b_bi: u32, c_off: u32, c_rs: u32,
    c_cs: u32, c_bo: u32, c_bi: u32, alpha: f32,
    beta: f32, _pad0: u32, _pad1: u32, _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;

var<workgroup> a_tile: array<f32, 1024>; // [BK][BM]
var<workgroup> b_tile: array<f32, 1024>; // [BK][BN]

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let zo = wg.z / p.n_inner;
    let zi = wg.z % p.n_inner;
    let a_base = p.a_off + zo * p.a_bo + zi * p.a_bi;
    let b_base = p.b_off + zo * p.b_bo + zi * p.b_bi;
    let c_base = p.c_off + zo * p.c_bo + zi * p.c_bi;

    let row0 = wg.y * BM;
    let col0 = wg.x * BN;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }

    for (var k0 = 0u; k0 < p.k; k0 += BK) {
        // Load the tiles, walking whichever axis is contiguous in memory first.
        for (var e = 0u; e < 4u; e++) {
            let idx = li + e * 256u;
            var i: u32;
            var kk: u32;
            if (p.a_cs == 1u) {
                kk = idx % BK;
                i = idx / BK;
            } else {
                i = idx % BM;
                kk = idx / BM;
            }
            let gi = row0 + i;
            let gk = k0 + kk;
            var v = 0.0;
            if (gi < p.m && gk < p.k) {
                v = a[a_base + gi * p.a_rs + gk * p.a_cs];
            }
            a_tile[kk * BM + i] = v;
        }
        for (var e = 0u; e < 4u; e++) {
            let idx = li + e * 256u;
            var j: u32;
            var kk: u32;
            if (p.b_cs == 1u) {
                j = idx % BN;
                kk = idx / BN;
            } else {
                kk = idx % BK;
                j = idx / BK;
            }
            let gj = col0 + j;
            let gk = k0 + kk;
            var v = 0.0;
            if (gj < p.n && gk < p.k) {
                v = b[b_base + gk * p.b_rs + gj * p.b_cs];
            }
            b_tile[kk * BN + j] = v;
        }
        workgroupBarrier();

        for (var kk = 0u; kk < BK; kk++) {
            var av: array<f32, 4>;
            var bv: array<f32, 4>;
            for (var t = 0u; t < 4u; t++) {
                av[t] = a_tile[kk * BM + lid.y + t * 16u];
                bv[t] = b_tile[kk * BN + lid.x + t * 16u];
            }
            for (var i = 0u; i < 4u; i++) {
                for (var j = 0u; j < 4u; j++) {
                    acc[i * 4u + j] += av[i] * bv[j];
                }
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i++) {
        let gi = row0 + lid.y + i * 16u;
        if (gi >= p.m) {
            continue;
        }
        for (var j = 0u; j < 4u; j++) {
            let gj = col0 + lid.x + j * 16u;
            if (gj >= p.n) {
                continue;
            }
            let idx = c_base + gi * p.c_rs + gj * p.c_cs;
            var r = p.alpha * acc[i * 4u + j];
            if (p.beta != 0.0) {
                r += p.beta * c[idx];
            }
            c[idx] = r;
        }
    }
}
