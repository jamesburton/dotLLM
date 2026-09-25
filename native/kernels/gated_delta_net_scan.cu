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
#include <cooperative_groups.h>

// ─── OPT-IN row-split cooperative-groups variant (issue #180) ──────────────
// Fresh elevated `ncu --set full` data (2026-07-25) found gdn_scan_step_f32's grid (=n_v_head=48
// for real Bonsai-27B) fills only 0.14 waves/SM (Achieved Occupancy 14.8% vs 100% theoretical) —
// NOT a per-block resource problem (already 40 regs/thread, 0 spills, 12 blocks/SM = the sm_86
// ceiling, confirmed by issue #173's ptxas-only analysis), but a genuine grid-too-small problem:
// 28 SMs × up to 12 resident blocks/SM = 336 possible resident blocks, only 48 ever launch.
// ~85-86% of warp stall cycles are L1TEX scoreboard-dependency (too few resident warps to hide
// load latency). The layer stack is strictly sequential (layer L+1's GDN input depends on layer
// L's full residual output), so batching the launch ACROSS the model's GDN layers is impossible —
// see this file's git history / issue #180 for the full architectural argument. This section
// documents what WAS investigated within one layer's one recurrence step.
//
// MEASURED (not modeled — real cudaEvent/CUevent timing on RTX 3060, both via a standalone
// microbenchmark AND via the actual production PTX-JIT + driver-API path, `cuModuleLoadData` +
// `cuLaunchCooperativeKernel`, to rule out any nvcc-runtime-API artifact):
//   - Splitting each V-head's [d_state,d_state] state-matrix row range across `split` cooperating
//     blocks (grid = n_v_head*split instead of n_v_head, using CUDA Cooperative Groups
//     `grid.sync()` in place of the 2 block-local `__syncthreads()`, so this is STILL ONE KERNEL
//     LAUNCH, not launch-count-multiplied) measurably reduces per-launch device time:
//       split=1 (current, baseline):  ~65-68 us/launch
//       split=2 (grid=96):             ~64-65 us/launch  (grid.sync overhead ~cancels the gain)
//       split=4 (grid=192):            ~48-49 us/launch  (~26-27% real reduction, reproducible)
//   - split=7/8 (grid=336/384) FAIL at launch: `cuOccupancyMaxActiveBlocksPerMultiprocessor` on
//     the cooperative kernel reports only 12 blocks/SM x 28 SMs = 336 max co-resident — but the
//     cooperative-launch reservation overhead can reduce that further in practice (observed 10-12
//     blocks/SM depending on `-rdc` flags; `-rdc=true` is NOT needed for single-TU grid.sync() and
//     COSTS occupancy — do not add it). split=4 (192 blocks) fits comfortably; this is the largest
//     verified-safe split for this exact shape (nVHead=48, dState=128) on this GPU.
//   - REAL end-to-end `dotnet run ... bench` on the real Bonsai-27B GGUF (`-p 64 -n 48 -r 8`, 5
//     independent A/B rounds spanning a long session with visible thermal drift — baseline medians
//     ranged 17.21-18.08 tok/s round to round): baseline medians 18.08/17.80/17.92/17.80/17.21
//     (mean 17.76), split4-enabled medians 18.31/18.26/18.13/17.78/17.92 (mean 18.08) — a
//     reproducible **+1.8% average real decode throughput gain** (median AND best both +1.79%
//     aggregate). split4 beat baseline's median in 4/5 rounds and TIED in round 4 (17.78 vs 17.80,
//     effectively noise) while its BEST still edged baseline's best in all 5/5 rounds — it never
//     lost a round. On the same order as this investigation's SMALLEST confirmed launch-fusion win
//     (#170, deinterleave+L2norm, +0.7-1%), not the larger wins (conv1d fusion +2.5%, F32-native
//     GEMV +8%), but a real, reproducible signal — this machine's typical run-to-run noise floor is
//     2-8%, yet split4 was never the loser across 5 independent rounds.
//
// THE CATCH — bit-exactness is fundamentally, not just practically, incompatible with this split:
// gdn_scan_step_f32's retrieve/read phases are `Σ_row S[row,col]*k[row]` accumulated in STRICT
// row=0..d_state-1 order (a design goal documented at the top of this file, "Float adds happen in
// row=0..d_state-1 order on a single thread → bit-perfect"). A real parallel split necessarily
// computes INDEPENDENT partial sums per row-range block, then combines them
// (partial_0 + partial_1 + ... + partial_{split-1}) — this is mathematically equal but NOT
// bit-identical to the flat sequential accumulation, because IEEE-754 float addition is not
// associative. Measured on a single fresh-state step with random inputs: max abs diff ~2e-6, max
// relative diff ~4.6e-4 vs the CPU oracle, only ~8-9% of output elements bit-matched by
// coincidence. The ONLY way to keep this bit-exact is a sequential hand-off chain (block N may not
// start until block N-1's grid.sync()'d partial arrives) — which has ZERO parallelism benefit, it
// just relocates the same serial chain across extra grid.sync() barriers. This is the same
// "rejected for breaking CPU bit-parity" conclusion this file's history has reached before for
// warp-shuffle tree reductions (see the decay+retrieve/write+read rewrite note above) — it is NOT
// specific to this split idea, it is a fundamental property of parallel floating-point reduction.
//
// COMPOUNDING DRIFT over many decode steps (the GDN state is RECURRENT — this step's slightly-
// different S feeds directly into next step's decay/retrieve/write): a naive single-step
// measurement badly underestimates real drift. `CudaGdnScanStepF32CoopSplit4Tests` ran 500
// consecutive decode steps (real Bonsai-27B shape, nVHead=48/dState=128) tracking GPU-vs-CPU
// diff every step: max ABSOLUTE diff stayed bounded at ~1e-6 to 2.7e-3 across the entire 500-step
// run — no runaway/unbounded growth, no NaN/Inf, consistent with the per-step decay g_vh<1 acting
// as a leaky-integrator forgetting mechanism that caps drift accumulation. The RELATIVE diff metric
// is much noisier and occasionally spikes to double digits (once to ~91% over 500 steps) — this is
// a near-zero-denominator artifact (relative diff = absDiff/(|cpuOut|+1e-8); when the CPU reference
// output for a position is itself near zero, a tiny ~1e-5 absolute difference divides up into a
// large percentage), NOT evidence of instability — the absolute magnitude at every one of those
// spikes was still tiny compared to normal activation magnitudes elsewhere in the network. This
// characterization (bounded absolute drift, noisy-but-explained relative metric, no blowup over
// 500 steps) is what made shipping this as an opt-in defensible — see DECISION below.
//
// Literature check (2026-07-25): official Mamba/Mamba-2 (`selective_scan_fwd_kernel.cuh`),
// FlashLinearAttention's DeltaNet/GLA fused-recurrent Triton kernel, and llama.cpp's
// `ggml-cuda/ssm-scan.cu` were all checked. None expand grid size beyond batch x head/dim at
// single-token decode — FLA's kernel has an actual UNFINISHED extension point for splitting the
// value dimension (`NV = cdiv(V, BV)`) but explicitly asserts `NK == 1` (the state/key-dim split
// this file explores is not supported there either). General LLM-serving literature treats
// batch=1 decode as an accepted <20%-MFU ceiling whose standard remedy is cross-REQUEST continuous
// batching (which this investigation's sequential-layer-dependency constraint also rules out for
// this specific recurrence). No published precedent for this exact intra-step trick was found —
// this appears to be genuinely unexplored territory, not a known-solved or known-rejected idea.
//
// DECISION: real, measured, ~26-27% kernel-level speedup and a reproducible ~1.8% average real
// end-to-end decode gain, validated end-to-end through the actual PTX-JIT +
// `cuLaunchCooperativeKernel` driver-API path this codebase uses (not just a standalone
// executable) — but per this project's stated priority order (CLAUDE.md: "Correctness then
// Performance then Extensibility"), and because the GDN state is the model's
// persistent-across-the-entire-generation recurrent memory (unlike a stateless elementwise
// fusion), this does NOT replace the default kernel. It IS wired in as an explicit, clearly-
// labelled, default-OFF opt-in (`DOTLLM_GDN_SCAN_APPROX_SPLIT4=1`,
// `CudaKernels.EnableGdnScanApproxSplit4`) for anyone who wants the throughput in exchange for
// giving up bit-exact CPU/GPU parity on this one kernel — see `gdn_scan_step_f32_coop_split4`
// below, `CudaKernels.LaunchGdnScanStepF32CoopSplit4`, and
// `CudaKernels.IsGdnScanCoopSplit4Safe` (mandatory per-shape/per-GPU cooperative-launch
// co-residency check — exceeding it is a hard CUDA error, not a soft fallback; only verified
// against Bonsai-27B's nVHead=48/dState=128 shape on this RTX 3060 so far, though the safety check
// itself is shape/GPU-generic). A future session wanting to make this the DEFAULT (not just
// available opt-in) would need a much longer real-generation validation (thousands of steps, ideally
// on real prompts rather than random per-step inputs) to further build confidence beyond the
// 500-step synthetic characterization done here — see `CudaGdnScanStepF32CoopSplit4Tests.cs`.
extern "C" __global__ void gdn_scan_step_f32_coop_split4(
    float* __restrict__ state,           // [n_v_head, d_state, d_state] (in/out)
    const float* __restrict__ q_t,       // [n_k_head, d_state] (already L2-normed by caller)
    const float* __restrict__ k_t,       // [n_k_head, d_state] (already L2-normed by caller)
    const float* __restrict__ v_t,       // [n_v_head, d_state]
    const float* __restrict__ g_t,       // [n_v_head]
    const float* __restrict__ beta_t,    // [n_v_head]
    float* __restrict__ output_t,        // [n_v_head, d_state]
    float* __restrict__ partial_tmp,     // [n_v_head, 4, d_state] scratch (retrieve partials)
    float* __restrict__ partial_out,     // [n_v_head, 4, d_state] scratch (read partials)
    const int n_v_head, const int n_k_head, const int d_state)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int SPLIT = 4;
    int vh = blockIdx.x;
    int half = blockIdx.y;              // row-range index in [0, SPLIT)
    int col = threadIdx.x;
    int kh = vh % n_k_head;

    float* S = state + (size_t)vh * d_state * d_state;
    const float* k_head = k_t + (size_t)kh * d_state;
    const float* q_head = q_t + (size_t)kh * d_state;
    const float* v_head = v_t + (size_t)vh * d_state;
    float g_vh = g_t[vh];
    float beta_vh = beta_t[vh];

    extern __shared__ float smem[];
    float* k_shared = smem;
    float* q_shared = smem + d_state;
    k_shared[col] = k_head[col];
    q_shared[col] = q_head[col];

    int row_count = d_state / SPLIT;
    int row_start = half * row_count;

    // Decay: row-range-local, no cross-block dependency (same argument as the non-split kernel).
    int state_size = row_count * d_state;
    for (int i = col; i < state_size; i += blockDim.x)
    {
        int local_row = i / d_state;
        int c = i % d_state;
        S[(row_start + local_row) * d_state + c] *= g_vh;
    }
    __syncthreads();

    // Retrieve partial: sequential over THIS block's row range only (preserves CPU order
    // *within* the range; combining across ranges below is where associativity is lost — see
    // the header comment above).
    float tmp_partial = 0.0f;
    for (int r = 0; r < row_count; r++)
    {
        int row = row_start + r;
        tmp_partial += S[row * d_state + col] * k_shared[row];
    }
    partial_tmp[((size_t)vh * SPLIT + half) * d_state + col] = tmp_partial;

    grid.sync();   // ALL blocks (every vh, every half) must have written their partial before any read.

    float tmp_col = 0.0f;
    for (int s = 0; s < SPLIT; s++)
        tmp_col += partial_tmp[((size_t)vh * SPLIT + s) * d_state + col];
    tmp_col = beta_vh * (v_head[col] - tmp_col);

    for (int r = 0; r < row_count; r++)
    {
        int row = row_start + r;
        S[row * d_state + col] += k_shared[row] * tmp_col;
    }

    float out_partial = 0.0f;
    for (int r = 0; r < row_count; r++)
    {
        int row = row_start + r;
        out_partial += S[row * d_state + col] * q_shared[row];
    }
    partial_out[((size_t)vh * SPLIT + half) * d_state + col] = out_partial;

    grid.sync();

    if (half == 0)
    {
        float out_col = 0.0f;
        for (int s = 0; s < SPLIT; s++)
            out_col += partial_out[((size_t)vh * SPLIT + s) * d_state + col];
        float scale = 1.0f / sqrtf((float)d_state);
        output_t[(size_t)vh * d_state + col] = out_col * scale;
    }
}

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
