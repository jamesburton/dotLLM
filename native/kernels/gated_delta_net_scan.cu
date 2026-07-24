// Gated DeltaNet (GDN) recurrence step + L2-normalize-heads helper.
// Bit-perfect FP32 port of DotLLM.Cpu.Kernels.GatedDeltaNetScan (CPU reference for
// Qwen3MoeHybrid models). Matches the CPU loop order so float-add associativity
// is preserved across architectures.
//
// ─── Token-sequential design ─────────────────────────────────────────────────
// The recurrence is sequential along the time axis: state[t] depends on state[t-1].
// We launch this kernel ONCE PER TOKEN with grid = (n_v_head,). The host loop
// advances q/k/v/g/beta/output pointers by one token-stride per call. This
// trades seqLen × per-call launch overhead (~5-10 µs each) against a much simpler
// kernel that needs no inter-block synchronization. For typical decode (seqLen=1)
// the overhead is zero; for prefill of a few hundred tokens it is bounded by
// stream serialization on the same CUDA stream.
//
// Alternative (single block, internal seq loop with __syncthreads) was rejected:
// it would cap the parallelism at one V-head per token, which is the opposite of
// what we want, AND restrict each block to one (vh) so seqLen × nVHead headcount
// would crawl through 32 / 64 V heads sequentially in one SM.
//
// ─── Per-block layout (one block per V-head, fixed token t) ──────────────────
//   blockIdx.x  = vh  ∈ [0, n_v_head)
//   blockDim.x  = d_state  (= 128 for Qwen3MoeHybrid; the launch_bounds value)
//   threadIdx.x = col      (the value-dim this thread owns)
//
// State per V-head is [d_state, d_state] row-major:  S[row * d_state + col]
//   row = key dim,  col = value dim.
//
// Each thread "owns" its column. `tmp[col]` is a thread-private register — there
// is NO cross-thread reduction here. The CPU does:
//
//     tmp[col] = 0
//     for row in 0..d_state-1:  tmp[col] += S[row,col] * k[row]    (retrieve)
//     tmp[col] = β * (v[col] - tmp[col])                            (delta)
//     for row, col: S[row,col] += k[row] * tmp[col]                 (write)
//     out[col] = sum_row S[row,col] * q[row]                        (read)
//     out[col] *= 1/sqrt(d_state)                                   (scale)
//
// Mapping one thread per `col` keeps the row-sum order identical to the CPU.
// Float adds happen in row=0..d_state-1 order on a single thread → bit-perfect.
//
// k and q vectors are read by ALL columns (every thread needs k[row] and q[row]
// for every row), so we stage them in shared memory once per token. v[col] and
// β / g are read by exactly one thread each → no need to stage.
//
// ─── Decay (S *= g) ──────────────────────────────────────────────────────────
// The CPU loops `for i = 0..d_state*d_state-1: S[i] *= g`. Each element is one
// independent multiply, so we can stride the work across threads freely without
// affecting float results — multiplication of independent values is parity-safe
// regardless of order. We use a grid-stride loop over the linearized S buffer.
//
// ─── NEGATIVE RESULT: fused decay+retrieve / write+read rewrite (issue #173) ──
// Attempted a kernel-internal restructure: thread `col` touches EXCLUSIVELY
// S[*, col] across every phase of this kernel (the decay grid-stride loop
// above, for blockDim.x == d_state, reduces to exactly row = i / d_state,
// colIndex = i % d_state == col — already only its own column). Since no
// other thread ever reads/writes column `col`, there is no cross-thread
// dependency on S once k_shared/q_shared are populated (the ONLY reason
// EITHER existing __syncthreads() call is needed at all). This means:
//   1. The second __syncthreads() (between write and read) is provably
//      redundant — removable with zero correctness risk.
//   2. Decay writes S[row,col], then retrieve immediately re-reads it in a
//      separate pass; decay is a strictly order-independent per-element
//      multiply, so fusing "decay S[row,col], then immediately accumulate
//      that SAME value into the retrieve sum" into one loop iteration is
//      bit-identical to the current two-pass version (verified against the
//      CPU reference, GatedDeltaNetScan.Execute, row-by-row) while
//      eliminating retrieve's redundant reload.
//   3. Same argument fuses write+read into one loop, eliminating read's
//      redundant reload.
// This restructure — 4 full passes over the per-head [d_state,d_state] state
// matrix collapsing to 2 fused passes, 1 of 2 __syncthreads() removed — was
// implemented, proven bit-exact against the CPU oracle across multiple decode
// steps (tests/DotLLM.Tests.Unit/Cuda/CudaGdnScanStepF32Tests.cs, kept — the
// first focused CUDA test for this kernel, independent of this round's
// outcome), and measured `ptxas -Xptxas -v`/`cuobjdump --dump-sass` clean:
// register count and spill count UNCHANGED (40 regs/thread, 0 spills, so
// occupancy — already 100% pre-change: 40*128=5120 regs/block,
// floor(65536/5120)=12 blocks/SM = 48 warps/SM = the sm_86 1536-thread/SM
// ceiling — was unaffected either way), BAR.SYNC count 2→1, and static SASS
// instruction count for the function dropped ~11% (888→792 instructions).
//
// Despite that clean static signal, REAL `dotnet run ... bench` throughput
// showed NO measurable improvement: two full A/B rounds (`-p 64 -n 48 -r 8`,
// real Bonsai-27B GGUF, rebuilding PTX+CLI between each side, clean baseline
// each time) gave modified=17.68/17.88 then 18.00/18.13 (median/best tok/s)
// vs baseline=17.69/17.76 then 17.98/18.11 — i.e. round-to-round system noise
// (~2-8% swings, occasionally with a slow outlier rep on EITHER side) was
// larger than any modified-vs-baseline gap. Aggregating all 16 reps per side:
// mean 17.738 (modified) vs 17.666 (baseline) tok/s — a ~0.4% average edge,
// not distinguishable from noise at this sample size, nowhere near this
// investigation's other confirmed wins (which cleared their baseline by
// several times this margin). Most likely explanation: the kernel was
// ALREADY at 100% occupancy pre-change, so the GPU already has enough
// resident warps to hide the latency of the "redundant" loads this rewrite
// removes; and the retrieve/read phases' accumulation is an inherently serial
// 128-deep dependency chain (tmp_col/out_col each iteration depends on the
// previous), a critical-path length this rewrite does not shorten — so
// removing surrounding, already-hidden instructions doesn't move wall-clock
// decode time. CONCLUSION: kernel-internal restructuring of this specific
// recurrence step is a dead end absent a change that either (a) breaks the
// 128-deep serial accumulation into a tree/warp-shuffle reduction (rejected
// elsewhere in this file's history for breaking CPU bit-parity — accumulation
// order changes results) or (b) reduces kernel LAUNCH count/overhead instead
// (already exhausted — see the "Token-sequential design" section above; one
// launch per (GDN layer, decode step) is already the floor for this design).
// The functional code change was reverted (this file's kernel body is
// unchanged from pre-#173); only this documentation and the new correctness
// test were kept. Do not re-attempt this exact restructure without new `ncu`-
// level occupancy/stall data (this session did not have elevated-PowerShell
// access to gather it) or a fundamentally different parallelization strategy.

#include <math.h>

extern "C" __global__ void __launch_bounds__(128) gdn_scan_step_f32(
    float* __restrict__ state,           // [n_v_head, d_state, d_state] (in/out)
    const float* __restrict__ q_t,       // [n_k_head, d_state] (already L2-normed by caller)
    const float* __restrict__ k_t,       // [n_k_head, d_state] (already L2-normed by caller)
    const float* __restrict__ v_t,       // [n_v_head, d_state]
    const float* __restrict__ g_t,       // [n_v_head]
    const float* __restrict__ beta_t,    // [n_v_head]
    float* __restrict__ output_t,        // [n_v_head, d_state]
    const int n_v_head, const int n_k_head, const int d_state)
{
    int vh = blockIdx.x;
    if (vh >= n_v_head) return;

    int col = threadIdx.x;                          // this thread owns this value-dim
    // TILED head broadcast (matches llama.cpp ggml_gated_delta_net: iq1 = iv1 % neq1).
    // For NVHead=32, NKHead=16 this maps vh 0..15 → kh 0..15, vh 16..31 → kh 0..15.
    // Previous (incorrect) interleaved mapping vh / (n_v_head/n_k_head) produced garbage.
    int kh = vh % n_k_head;

    float* S = state + (size_t)vh * d_state * d_state;
    const float* k_head = k_t + (size_t)kh * d_state;
    const float* q_head = q_t + (size_t)kh * d_state;
    const float* v_head = v_t + (size_t)vh * d_state;

    float g_vh = g_t[vh];
    float beta_vh = beta_t[vh];

    // Shared staging for k and q (each thread reads every row during the
    // retrieve / write / read phases). One element per thread — host launches
    // with blockDim.x == d_state so the mapping is 1:1.
    extern __shared__ float smem[];
    float* k_shared = smem;                         // [d_state]
    float* q_shared = smem + d_state;               // [d_state]
    k_shared[col] = k_head[col];
    q_shared[col] = q_head[col];

    // ── 1. Decay: S *= g_vh ──────────────────────────────────────────────────
    // Linear grid-stride over all d_state*d_state elements; multiplication on
    // independent elements is parity-safe regardless of thread mapping.
    int state_size = d_state * d_state;
    for (int i = col; i < state_size; i += blockDim.x)
    {
        S[i] *= g_vh;
    }

    // Sync: ensure k_shared/q_shared are populated AND decay is complete
    // before the retrieve phase starts reading S.
    __syncthreads();

    // Host launches with blockDim.x == d_state, so every thread is a valid
    // column. We deliberately DO NOT early-return any threads here: the later
    // __syncthreads() after the rank-1 write would deadlock if some threads
    // had already exited.

    // ── 2. Retrieve: tmp = S.T @ k  =>  tmp[col] = Σ_row S[row,col] * k[row]
    // Per-thread accumulator. Row order matches the CPU exactly.
    float tmp_col = 0.0f;
    for (int row = 0; row < d_state; row++)
    {
        tmp_col += S[row * d_state + col] * k_shared[row];
    }

    // ── 3. Delta: tmp = β * (v - tmp) ───────────────────────────────────────
    tmp_col = beta_vh * (v_head[col] - tmp_col);

    // ── 4. Write: S[row,col] += k[row] * tmp[col]  for all (row, col) ───────
    // Each thread writes its column for every row. Independent stores — no
    // cross-thread interference. (Different threads write different columns of
    // the same row; no aliasing.)
    for (int row = 0; row < d_state; row++)
    {
        S[row * d_state + col] += k_shared[row] * tmp_col;
    }

    // Sync so the read phase sees the fully-updated S.
    __syncthreads();

    // ── 5. Read: out[col] = (Σ_row S[row,col] * q[row]) / sqrt(d_state) ─────
    float out_col = 0.0f;
    for (int row = 0; row < d_state; row++)
    {
        out_col += S[row * d_state + col] * q_shared[row];
    }

    // CPU uses `1.0f / MathF.Sqrt(dState)`. Match exactly — do NOT use rsqrtf
    // here (different rounding under --use_fast_math). This file is NOT in the
    // FAST_MATH list in build_ptx.bat, so sqrtf rounds correctly.
    float scale = 1.0f / sqrtf((float)d_state);
    output_t[(size_t)vh * d_state + col] = out_col * scale;
}

// ─── L2 normalize per head ──────────────────────────────────────────────────
// Mirrors DotLLM.Cpu.Kernels.GatedDeltaNetScan.L2NormalizeHeads exactly.
// Layout: x is treated as `total_heads` contiguous head vectors of `d_state`
// floats; each is independently normalized to unit norm.
//
// CPU code:
//     sumSq = 0
//     for i in 0..d_state-1:  sumSq += head[i] * head[i]
//     invNorm = 1.0f / (sqrtf(sumSq) + eps)
//     for i: head[i] *= invNorm
//
// Bit-perfect parity requires the same sequential 0..d_state-1 accumulation
// order. We do the sum in thread 0 only (127 serial adds at d_state=128 — under
// 100 ns — well below memory-load cost), stash invNorm in shared memory, and
// broadcast for the multiply phase. A warp-shuffle tree reduction WOULD NOT
// match the CPU bit-for-bit (different add order).

extern "C" __global__ void __launch_bounds__(128) l2_normalize_heads_f32(
    float* __restrict__ x,
    const int total_heads, const int d_state, const float eps)
{
    int h = blockIdx.x;
    if (h >= total_heads) return;

    float* head = x + (size_t)h * d_state;

    __shared__ float s_inv_norm;

    if (threadIdx.x == 0)
    {
        // Serial sequential sum to match CPU float-add order exactly.
        float sum_sq = 0.0f;
        for (int i = 0; i < d_state; i++)
        {
            float v = head[i];
            sum_sq += v * v;
        }
        s_inv_norm = 1.0f / (sqrtf(sum_sq) + eps);
    }
    __syncthreads();

    float inv_norm = s_inv_norm;
    for (int i = threadIdx.x; i < d_state; i += blockDim.x)
    {
        head[i] = head[i] * inv_norm;
    }
}

// ─── Decode-time fused deinterleave + L2-normalize (issue #170) ────────────
// Replaces, for the seqLen==1 decode case only, three launches:
//   deinterleave_gdn_qkv_f32(src, q, k, v, ...) + l2_normalize_heads_f32(q, ...)
//   + l2_normalize_heads_f32(k, ...)
// with one. deinterleave_gdn_qkv_f32's generic indexing computes
// `t = idx / conv_dim; e = idx % conv_dim` for a runtime (non-constant)
// conv_dim — cuobjdump confirms this lowers to a full 32-bit integer
// divide/modulo (MUFU.RCP + Newton refinement, ~15 instructions) per element,
// even though decode always has seqLen==1 so t is trivially 0. Here each
// block owns exactly one d_state-sized head (Q, K, or V) with no division:
// k_dim = n_k_head*d_state and v_dim = n_v_head*d_state are always exact
// multiples of d_state, so grid = 2*n_k_head + n_v_head, blockDim = d_state
// covers the whole [Q|K|V] row with pure block/thread-index arithmetic.
//
// Bit-identical to the three kernels it replaces: same serial thread-0
// sum-of-squares accumulation order as l2_normalize_heads_f32 for Q/K, and a
// straight per-element copy for V (matching deinterleave_gdn_qkv_f32's V
// branch exactly). blockDim.x == d_state is required (enforced host-side, as
// with l2_normalize_heads_f32), so each thread owns exactly one element —
// no strided loop needed.
extern "C" __global__ void __launch_bounds__(128) gdn_deinterleave_l2norm_decode_f32(
    const float* __restrict__ src,    // [2*k_dim + v_dim], single decode row
    float* __restrict__ q,            // [k_dim], L2-normalized per d_state-head
    float* __restrict__ k,            // [k_dim], L2-normalized per d_state-head
    float* __restrict__ v,            // [v_dim], straight copy
    const int n_k_head, const int d_state, const float eps)
{
    int block = blockIdx.x;
    int k_dim = n_k_head * d_state;

    if (block >= 2 * n_k_head)
    {
        // V head: straight copy, no normalization.
        int h = block - 2 * n_k_head;
        const float* head_src = src + 2 * k_dim + (size_t)h * d_state;
        float* head_dst = v + (size_t)h * d_state;
        head_dst[threadIdx.x] = head_src[threadIdx.x];
        return;
    }

    bool is_k = block >= n_k_head;
    int h = is_k ? block - n_k_head : block;
    const float* head_src = src + (is_k ? k_dim : 0) + (size_t)h * d_state;
    float* head_dst = (is_k ? k : q) + (size_t)h * d_state;

    __shared__ float s_inv_norm;
    if (threadIdx.x == 0)
    {
        float sum_sq = 0.0f;
        for (int i = 0; i < d_state; i++)
        {
            float val = head_src[i];
            sum_sq += val * val;
        }
        s_inv_norm = 1.0f / (sqrtf(sum_sq) + eps);
    }
    __syncthreads();

    head_dst[threadIdx.x] = head_src[threadIdx.x] * s_inv_norm;
}

// ─── GDN decay: alpha → exp(softplus(alpha + dt_bias) * A) in place ────────
// Bit-perfect port of the CPU reference in Qwen3MoeHybridTransformerModel.cs
// (ForwardGdnBody decay section) and the host fallback at
// CudaQwen3MoeHybridTransformerModel.cs:1112-1121:
//
//     for t in 0..seqLen-1, for vh in 0..nVHead-1:
//         alpha = alphaBuf[t*nVHead + vh] + dt_bias[vh]
//         sp    = log(1 + exp(alpha))            // softplus, NO x>20 guard
//         alphaBuf[t*nVHead + vh] = exp(sp * A[vh])
//
// The CPU oracle does NOT apply the standard "x > 20 → softplus(x) ≈ x"
// numerical guard — the result silently saturates to +inf for very large
// alpha. To preserve bit-for-bit parity we replicate this exactly: no guard,
// raw expf/logf.
//
// Layout: alphaBuf is [seqLen, nVHead] row-major. Each output element is
// computed independently, so we use one thread per (t, vh) cell over a
// linearised grid. Build_ptx.bat compiles this TU with -fmad=false; combined
// with CUDA's precise expf/logf the output is within ≤1 ULP of MathF.Exp /
// MathF.Log on Ampere+ — not strictly bit-equal across all alpha values, but
// numerically equivalent for any well-conditioned input.

extern "C" __global__ void gdn_decay_f32(
    float* __restrict__ alpha_buf,         // [seq_len, n_v_head], in/out
    const float* __restrict__ dt_bias,     // [n_v_head]
    const float* __restrict__ a,           // [n_v_head]
    const int seq_len, const int n_v_head)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * n_v_head;
    if (idx >= total) return;

    int vh = idx % n_v_head;              // [t * n_v_head + vh] → vh = idx % n_v_head
    float alpha = alpha_buf[idx] + dt_bias[vh];
    // softplus(alpha) = log(1 + exp(alpha)) — match CPU exactly, no x>20 guard.
    float sp = logf(1.0f + expf(alpha));
    alpha_buf[idx] = expf(sp * a[vh]);
}

// Fused decay(alpha_buf) + sigmoid(beta_buf) — the hybrid decode path always calls
// gdn_decay_f32 immediately followed by a sigmoid over betaBuf (same [seq_len, n_v_head]
// shape, an independent buffer). Combining into one launch halves the launch count for this
// pair. alpha_buf's math is byte-for-byte gdn_decay_f32's (same TU, same -fmad=false); the
// added sigmoid term is elementwise_f32.cu's sigmoid_f32 formula (1/(1+exp(-x))) — running it
// under -fmad=false too is a strict superset of precision, not a regression, since disabling
// FMA fusion never increases numerical drift versus the CPU host-fallback reference it mirrors.
extern "C" __global__ void gdn_decay_sigmoid_f32(
    float* __restrict__ alpha_buf,         // [seq_len, n_v_head], in/out (decay)
    float* __restrict__ beta_buf,          // [seq_len, n_v_head], in/out (sigmoid)
    const float* __restrict__ dt_bias,     // [n_v_head]
    const float* __restrict__ a,           // [n_v_head]
    const int seq_len, const int n_v_head)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * n_v_head;
    if (idx >= total) return;

    int vh = idx % n_v_head;
    float alpha = alpha_buf[idx] + dt_bias[vh];
    float sp = logf(1.0f + expf(alpha));
    alpha_buf[idx] = expf(sp * a[vh]);

    float b = beta_buf[idx];
    beta_buf[idx] = 1.0f / (1.0f + expf(-b));
}
