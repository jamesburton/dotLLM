# MlpUp GEMV bandwidth gap vs llama.cpp — research note

**Branch:** `cuda-mlpup-research` (forked from `feature/mamba-3-cuda` @ `a247909`)
**Author:** dotnet-perf-expert agent, 2026-04-25
**Status:** Research only. No kernel changes in this commit.
**Issue:** dotLLM Qwen3-8B Q4_K_M decode at 10.5 tok/s on RTX 3060;
llama.cpp CUDA reports ~30–42 tok/s on the same model + GPU. The gap is
concentrated in `MlpUp` (fused gate+up Q4_K GEMV at n=24576, k=4096), which
consumes 41% of GPU time and runs at ~50 GB/s effective HBM bandwidth — only
14% of the RTX 3060's 360 GB/s peak.

## TL;DR

The dotLLM MMQ Q4_K kernel (`native/kernels/quantized_gemv_mmq.cu`) was tuned
for **small models** (SmolLM-135M at k=576), where each row covers only 2
super-blocks and the natural "1 thread per super-block" parallelism leaves 254/256
threads idle. The 4-rows-per-block tiling fixed that.

**For large models (k≥4096), that tiling is now actively hurting.** Each block
processes 4 rows × 16 super-blocks = 64 work units across 256 threads (a
4-units-per-thread stride loop), then performs a 256-wide cross-warp shared-memory
reduction per row × 4 rows. Most of the 256 threads do tiny work then sit on
`__syncthreads()` waiting for the reduction — and the reduction itself adds
shared-memory traffic and a barrier that aren't on the critical path of a
well-saturated GEMV.

llama.cpp's `mul_mat_vec_q` kernel (the GEMV/batch=1 path) takes the exact
opposite approach for ncols_dst=1 on Ampere:
- **1 row per CUDA block** (not 4)
- **4 warps × 32 = 128 threads per block** (not 256)
- **No cross-warp shared-memory tree** — each warp handles its own dot product
  slice, and the final reduction is `warp_reduce_sum` after a small shmem fan-in
- All 128 threads contribute to the same row's dot product, walking
  super-blocks via `blocks_per_iter = vdr * nwarps * warp_size / qi = 8`
  super-blocks per iteration

**Recommended single change:** restructure the MMQ Q4_K kernel for the
large-k regime to mirror llama.cpp's mmvq: 1 row per block, 4 warps (128
threads), warp-coherent super-block stride, single warp-shuffle reduction at
the end. Keep the 4-rows-per-block variant as a separate kernel selected when
`k ≤ 1024` (≤ 4 super-blocks/row), where the small-row amortization still
helps. Estimated win: **2–4× on MlpUp** (50 → 100–200 GB/s), which is
**~5 tok/s → ~12–20 tok/s** on Qwen3-8B at the system level.

---

## 1. Setup verification — the 50 GB/s number

The bandwidth claim is derivable from numbers already in `.continue-here.md`
without re-running the profiler. Showing the math here:

**Per-call work (one MlpUp GEMV per layer):**
- n = 24576 (= 2 × intermediate_size 12288, fused gate+up)
- k = 4096
- Q4_K_M weight bytes per row: `(k / 256) × 144 = 16 × 144 = 2304 bytes`
- Total weight bytes read: `n × 2304 = 24576 × 2304 = 56.6 MiB`
- Activation read: `k × 2 = 8 KiB` FP16 (negligible vs weight)
- Output write: `n × 2 = 48 KiB` FP16 (negligible)

**Per-call timing** (from profiler median, eager path, Qwen3-8B Q4_K_M):
- MlpUp category total: 40.111 ms / token / 36 layers = **1.1142 ms / call**

**Effective HBM bandwidth:**
- 56.6 MiB / 1.1142 ms = **50.8 GB/s**

**RTX 3060 12GB peak HBM bandwidth:** 360 GB/s (192-bit GDDR6 @ 15 Gbps).

**Achieved fraction:** 50.8 / 360 = **14.1%**.

This matches the number in `.continue-here.md` §3c. Confirmed without a fresh
benchmark run.

> **Operational note:** the box has background VRAM hogs (RustDesk, Edge, Ollama)
> that consume ~6 GiB. Qwen3-8B fits with cleared background processes but
> Q4_K_M loads at the very edge of the available 12 GiB after weight dedup
> (commit `15becbf` brought resident weights from 7.8 → 5.3 GiB). HBM
> contention with these background apps will manifest as further bandwidth
> drop, but the 50 GB/s number is from a profiling run with cleared
> background — the kernel is leaving headroom on the GPU side.

---

## 2. llama.cpp structure summary — what they actually do for batch=1 Q4_K

Source files analyzed (master branch, accessed 2026-04-25):
- `ggml/src/ggml-cuda/mmvq.cu` — the GEMV (batch=1, "matrix×vector") path
- `ggml/src/ggml-cuda/mmq.cu` and `mmq.cuh` — the **batched** GEMM path
  (only used at `ne11 ≥ MMQ_DP4A_MAX_BATCH_SIZE`, not for token-by-token decode)
- `ggml/src/ggml-cuda/vecdotq.cuh` — per-quant-type vec_dot kernels (the
  `vec_dot_q4_K_q8_1_impl_vmmq` function below)
- `ggml/src/ggml-common.h` — `QK_K=256`, `QR4_K=2`, `QI4_K=32`, `QK8_1=32`

### 2a. MMQ vs MMVQ — which one does decode use?

**Both kernels use dp4a on Ampere for Q4_K.** They differ in batch dimension:
- **`mul_mat_vec_q` (mmvq.cu)** — batch=1 (and small batches up to 8). This
  is what decode-step GEMV actually goes through. It does **not** use tensor
  cores — only `dp4a` via `ggml_cuda_dp4a`.
- **`mul_mat_q` (mmq.cu)** — batch ≥ ~8. On Ampere with Q4_K it uses
  **tensor cores (mma.sync m16n8k16 INT8)** via the `MMQ_MMA_TILE_X_K_Q8_1`
  path. This is the prefill/batched path.

Implication for the dotLLM gap: **tensor cores are NOT in the decode-path
critical kernel for llama.cpp either.** Hypothesis 2 ("mma.m16n16k16 INT8")
in the original task brief is therefore lower-priority than initially
estimated — llama.cpp itself doesn't pull the tensor-core lever for
batch=1 GEMV on Ampere. The 30–42 tok/s they hit on this hardware comes
from a well-tuned dp4a kernel, not from tensor cores.

### 2b. The mmvq launch parameters (Q4_K, ncols_dst=1, Ampere)

From `mmvq.cu` `calc_nwarps` and `calc_rows_per_block` (both `constexpr` —
fully resolved at compile time per `<type, ncols_dst, table_id>` template):

```cuda
// MMVQ_PARAMETERS_GENERIC (= NVIDIA non-MMA, NVIDIA Ampere, NVIDIA Hopper, ...)
case ncols_dst == 1: nwarps = 4
case ncols_dst == 1: rows_per_cuda_block = small_k ? nwarps : 1
```

`small_k` is a runtime path that dotLLM's k=4096 does not trigger. So at
**k=4096, ncols_dst=1**:
- `rows_per_cuda_block = 1`
- `nwarps = 4`
- block size = `(warp_size, nwarps, 1) = (32, 4, 1)` = **128 threads/block**
- grid = `(nrows_x / 1, 1, 1) = (24576, 1, 1)` blocks for our MlpUp shape

For Q4_K specifically, `vdr = VDR_Q4_K_Q8_1_MMVQ = 2`, `qi = QI4_K = 32`,
so `blocks_per_iter = vdr × nwarps × warp_size / qi = 2 × 4 × 32 / 32 = 8`
super-blocks per iteration. With `blocks_per_row_x = 4096/256 = 16`
super-blocks/row, each block does **2 iterations** total to cover one row.

### 2c. The mul_mat_vec_q outer loop (cited; not copied)

The structural pattern from `mmvq.cu` `mul_mat_vec_q<Q4_K, 1, false>`:

```
tid = warp_size * threadIdx.y + threadIdx.x   // 0..127
row0 = blockIdx.x * 1                          // one row per block
for (kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter):
    kqs = vdr * (tid % (qi/vdr))
    // single dot-product per row, vec_dot_q_cuda walks one super-block chunk
    tmp[0][0] += vec_dot_q4_K_q8_1(vx, &y[kby], row0*stride + kbx, kqs)

// final reduction: warps 1..3 dump tmp into shared memory,
// warp 0 sums across the (nwarps-1) staged values + warp_reduce_sum
```

The key points to compare with dotLLM:
1. **No on-the-fly input quantization in this kernel.** The `vy` argument is
   already Q8_1-quantized (32 INT8 + half scale + half partial sum, packed
   into `block_q8_1`). llama.cpp runs a separate `quantize_row_q8_1_cuda`
   kernel earlier in the forward pass that converts FP16 activations to Q8_1
   *once per matmul group*.
2. **One row per block.** A given dp4a result lives in registers from
   start to finish on a single warp's lanes — no per-row shmem accumulator.
3. **Block size is 128, not 256.** Fewer threads, so each thread does
   more work (better arithmetic intensity per thread, less reduction
   overhead).

### 2d. The Q4_K vec_dot inner kernel (from `vecdotq.cuh`)

```cuda
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u,
    const uint8_t * __restrict__ sc, const uint8_t * __restrict__ m,
    const half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    #pragma unroll
    for (int i = 0; i < QR4_K; ++i) {  // QR4_K = 2 → 2-iter unroll
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        // Two dp4a chained per i (8 INT4×INT8 mac per dp4a → 32 macs / inner unroll)
        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1],
                            ggml_cuda_dp4a(v0i, u[2*i+0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1],
                            ggml_cuda_dp4a(0x01010101, u[2*i+0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}
```

The `dot2` term computes `Σ xq[j]` cheaply via `dp4a(0x01010101, xq, 0)` —
4 INT8 sums per dp4a — instead of pre-computing it during input quantization.
This is a different design choice from dotLLM (which precomputes `s_sx[c]` in
Stage 1) but the cost is similar (2 extra dp4a per super-block ≈ negligible).

> **Licensing:** llama.cpp source quoted under MIT for *citation* (research
> reference). Not redistributed. dotLLM's GPLv3 kernel will be a re-derivation
> from the GGUF format spec, not a copy.

---

## 3. Hypothesis ranking

### H1 — Block-shape mismatch (recommended target)

dotLLM's MMQ Q4_K block has 256 threads × 4 rows. At Qwen3-8B's k=4096:
- 16 super-blocks × 4 rows = 64 work units distributed across 256 threads
  via `for (unit = tid; unit < total_units; unit += blockDim.x)`.
- Each thread does ~1 work unit, then writes to `s_acc[r*256 + tid]` and
  hits `__syncthreads()`.
- Stage 3 reduction reads back 4 × 256 = 1024 floats from shmem and reduces
  per row across 4 warps (the other 4 warps idle).

llama.cpp's mmvq Q4_K block has 128 threads × 1 row:
- 16 super-blocks × 1 row = 16 work units, but they're processed in the
  loop `kbx += 8` — so 4 lanes × 2 iterations × 1 row = each warp owns
  a stride-8 slice of super-blocks.
- All 128 threads contribute to the same accumulator, reduced by
  `warp_reduce_sum + 3 warp-shfls fan-in across nwarps-1=3`.
- No cross-row shmem, no per-thread `s_acc` slot.

**Estimated impact:** +2–4× on MlpUp throughput at k=4096. Reasoning:
- The 4-rows-per-block tiling was the *fix* for SmolLM's 254/256-idle
  problem. At k=4096 the natural problem is that **rows are huge**
  (16 super-blocks each), not small. We'd be paying the cross-row reduction
  cost without the tile amortization paying for itself in shared work.
- Smaller block (128 vs 256) doubles the number of resident blocks per SM
  (`__launch_bounds__(256, 2)` → 2 blocks of 256, `(128, 4)` could give 4–8
  blocks of 128), which helps hide HBM latency.
- Removing the per-row shmem accumulator + `__syncthreads()` removes a
  serializing barrier that buys nothing at this k.

**Implementation effort:** 1–2 days of focused work.
- New kernel `quantized_gemv_q4_k_mmvq_large` in
  `native/kernels/quantized_gemv_mmq.cu` modeled on llama.cpp's mmvq pattern.
- Keep input-quantization-on-the-fly (don't yet require a separate Q8_1
  pre-pass — that's H4 below). Stage 1 already produces a usable INT8 stream
  in shmem; the new kernel can read from `s_xq` exactly as today.
- Routing in `CudaKernels.LaunchQuantizedGemvMmq`: pick mmvq-large when
  `k ≥ 1024` (≥4 super-blocks/row); pick the existing 4-rows-per-block
  kernel otherwise.
- Block dim = (128, 1, 1), grid dim = (n, 1, 1).

**Risk:**
- Correctness: the same `vec_dot_q4_K` math, just re-tiled. Existing
  `CudaMmqKernelTests` will gate it. Add a (n=24576, k=4096) test case to
  cover Qwen3-8B's MlpUp shape.
- Compatibility: kernel selection by k means SmolLM (k=576) keeps the old
  kernel, so no SmolLM regression. New kernel only kicks in for Qwen3-class
  models.
- The existing `DOTLLM_DISABLE_MMQ_Q4K=1` env var still bypasses everything
  to the legacy FP-fmuladd path — A/B comparison still possible.

**Recommendation:** **DO THIS FIRST.** Single largest expected lift, well
within the kernel author's existing skill set, full A/B test infrastructure
already in place.

### H2 — Tensor cores via mma.m16n8k16 INT8

**Estimated impact:** uncertain, possibly 0–1.5× over a well-tuned dp4a.

**Reasoning against (this is the surprise):**
- llama.cpp does not use tensor cores in their batch=1 GEMV path on Ampere.
  They have full mma machinery (`mmq.cuh`) but it's gated to batch ≥ ~8.
- For a single token, GEMV is bandwidth-bound (we're at 14% of peak HBM,
  not 14% of peak FLOPS). Tensor cores are FLOP accelerators. They wouldn't
  help if the bottleneck is memory.
- mma.sync requires gathering input fragments into a 16×16 tile shape that
  doesn't naturally exist in a single-token GEMV — you'd be pre-replicating
  the activation across the M dimension just to feed the tensor unit, which
  is wasted work.

**Estimated impact: likely zero or modestly negative for batch=1.** Skip
unless and until decode batching is on the roadmap (then it becomes
batch ≥ 8 prefill, and the existing cuBLAS HGEMM path is already the right
answer there).

**Recommendation:** **SKIP for batch=1 decode.** Revisit when adding
multi-request decode batching (multi-week future work).

### H3 — Memory layout / weight repack

**Estimated impact:** small (~1.1–1.3×) at best, possibly zero.

**Reasoning:** Q4_K rows are already row-major and contiguous in HBM. Each
super-block is 144 bytes; with `__ldg` reads of 4-byte qpacked words, a
warp reading 32 contiguous super-blocks reads 32 × 144 = 4.5 KiB per
super-block-stride iteration — that should be fine for L2/L1 prefetching.
What might help:
- Repack so all 8 super-blocks of a row's `qs[]` bytes are contiguous as a
  single 1024-byte stream rather than interleaved with the 16-byte
  scale/min headers. This would let `vec_dot` read `qpacked` with denser
  coalescing.
- Pre-reorder Q4_K to a tensor-core-friendly tile layout — but H2 says
  tensor cores aren't worth chasing for batch=1.

**Implementation effort:** medium (1 week — touches `CudaWeights.LoadFromGguf`
to add a repack-on-load step, plus matching dequant).

**Risk:** moderate — repack has to match an updated kernel one-to-one, easy
to introduce subtle alignment bugs.

**Recommendation:** **SKIP for now.** Try after H1 lands and re-measure
bandwidth — if H1 takes us to ~150–200 GB/s, the layout opt would have
diminishing returns relative to the time investment.

### H4 — Pre-quantize activations to Q8_1 once, share across QKV+GateUp+Down

**Estimated impact:** +5–10% on MlpUp specifically (low), more meaningful
for the *non-fused* projections (would compound across QkvProj and the
attention-block path).

**Reasoning:** dotLLM's MMQ kernel re-runs the input-quantization Stage 1
on every call. Activation `x` for Qwen3-8B MlpUp is 4096 elements — the
quantization pass itself costs maybe 5–10 µs. dotLLM's QkvProj/GateUp are
already fused (one call quantizes once, reused across all output rows in
the same kernel), so within MlpUp this hypothesis only saves the time it
would take to quantize 4096 floats. That's not the dominant cost.

**Bigger lever:** if we *also* split the kernel into two stages —
`quantize_x_to_q8_1` (one launch) + `mmvq_q4_k` (one launch reusing the
quantized x) — then the second-stage kernel has access to a precomputed
INT8 stream and doesn't need shared memory for it. That frees up the
9 KiB currently held by `s_xq + s_dx + s_sx` and might bump occupancy
beyond 2 blocks/SM. But adding launches re-introduces WDDM overhead
(~22 µs per launch on this box), which would eat the gain on Qwen3-8B's
36 layers × 5 GEMV groups = 180 launches.

**Recommendation:** **SKIP unless graph capture is on for this model.**
Once on the graph path, launch overhead is amortized — but Qwen3-8B
currently runs eager (graph hurts at this scale per `.continue-here.md`).
Defer to after H1 lands and graph perf is re-evaluated.

### H5 — Multiple GEMVs in flight

**Estimated impact:** ≤ 0.8 ms/token (≤ 1%).

**Reasoning:** 36 layers × 1 GEMV = 36 launches at 22 µs each = 792 µs
total launch overhead for MlpUp specifically. Out of 40 ms MlpUp time
that's 2%. Not worth chasing.

**Recommendation:** **SKIP.**

### H6 — KV-cache bandwidth contention

Not relevant to MlpUp specifically. The MlpUp kernel reads weights and
activations only — KV cache isn't touched. (KV bandwidth contention would
show up in the Attention category, which is at 1.3% — not a problem.)

### Hypothesis summary table

| # | Hypothesis | Est. impact on MlpUp | Effort | Risk | Verdict |
|---|---|---|---|---|---|
| 1 | Block-shape: 128 thr × 1 row vs 256 × 4 | **+100–300%** | 1–2 days | Low | **DO** |
| 2 | Tensor cores (mma INT8) | ~0% (bandwidth-bound) | 2–3 weeks | High | SKIP |
| 3 | Weight repack | +10–30% maybe | 1 week | Medium | SKIP for now |
| 4 | Pre-quantized activations | +5–10% | 2 days | Medium | SKIP for now |
| 5 | Launch coalescing | +2% | 1 day | Low | SKIP |
| 6 | KV bandwidth | N/A | — | — | N/A |

---

## 4. Recommended single next change

**Implement a 1-row-per-block, 4-warp (128-thread) MMVQ-style Q4_K kernel
for k ≥ 1024, modeled on llama.cpp's `mul_mat_vec_q<Q4_K, 1>`.**

### Files to touch

1. **`native/kernels/quantized_gemv_mmq.cu`** — add new kernel
   `quantized_gemv_q4_k_mmvq_large` with these properties:
   - `blockDim = (128, 1, 1)`; `gridDim = (n, 1, 1)` — one row per block.
   - Stage 1 (input quant) restructured so the warp owning the row's
     dot product also handles input quant for *its* super-block stride.
     Or: keep a small one-time Stage 1 across all 128 threads (k=4096 →
     128 chunks of 32 = 4096 elements covered in one warp-stride pass per
     warp), then __syncthreads(), then dot-product loop.
   - Dot-product loop: `for (sb = warp_id*4 + (tid & 31); ...)` so each
     warp owns 4 of the 16 super-blocks; lanes within a warp split each
     super-block 8-ways (the 8 dp4a calls go 1-per-lane-pair). Final per-warp
     `warp_reduce_sum`, then 4 warp partial sums combined via shmem +
     one more `warp_reduce_sum` in warp 0.
   - Write `y[row]` from threadIdx.x==0, threadIdx.y==0.
   - **No per-row shmem accumulator** — the partial lives in registers
     until the final reduction.

2. **`src/DotLLM.Cuda/CudaKernels.cs`** lines ~880-980 — register
   `_quantizedGemvQ4_KMmvqLargeFunc`, expose `HasMmvqLargeQ4K`, add
   `LaunchQuantizedGemvMmqLarge` with the new (128, 1, 1) launch shape.
   Update `LaunchQuantizedGemvMmq` to pick the `_large` variant when
   `k >= 1024`, fall back to existing 4-rows-per-block kernel otherwise.
   Both still gated by `HasMmqQ4K` and `DOTLLM_DISABLE_MMQ_Q4K`.

3. **`src/DotLLM.Cuda/Tests/CudaMmqKernelTests.cs`** — add test cases at
   (n=4096, k=4096), (n=11008, k=4096), (n=24576, k=4096) to cover
   Qwen3-8B-class shapes. Validate against the legacy
   `quantized_gemv_q4_k` kernel within the existing K-quant tolerance.

4. **`benchmarks/DotLLM.Benchmarks/...`** — add a per-kernel microbench
   `Q4KGemvVariants` that hits MMQ-old, MMQ-mmvq-large, legacy at the
   three shapes above, prints GB/s.

### Validation strategy

1. Build PTX, run unit tests: `dotnet test tests/DotLLM.Tests.Unit/ --filter "Category=GPU"`
   — must stay 47/47 green.
2. Microbench: `Q4KGemvVariants` must show mmvq-large ≥ 2× MMQ-old at
   (24576, 4096) on RTX 3060.
3. End-to-end: `profile-cuda-decode --compare` on Qwen3-8B Q4_K_M.
   Target: MlpUp category drops from 40 ms/token to ≤ 20 ms/token.
   Total decode wall ≤ 60 ms/token (≥ 17 tok/s, vs 10.5 today).
4. Run with `DOTLLM_DISABLE_MMQ_Q4K=1` to confirm the legacy path is
   unchanged (control).
5. SmolLM-135M sanity: confirm 217 graph / 117 eager tok/s unchanged
   (small-k path keeps using the old MMQ kernel via the k-threshold).

---

## 5. (Optional) Tiny prototype — not implemented in this commit

The recommended change is **not** a one-line tile-factor bump. The original
task brief's hypothesis 1 ("bump `MMQ_ROWS_PER_BLOCK` from 4 to 8 or 16")
turns out to be the **wrong direction** — llama.cpp evidence is that for
batch=1 GEMV at large k, you want **1** row per block, not 8 or 16.
Bumping MMQ_ROWS_PER_BLOCK higher would make the cross-row shmem reduction
more expensive without adding work-per-thread (you'd have *more* idle
threads per block, not fewer).

For completeness, one could try `MMQ_ROWS_PER_BLOCK = 2` as a 5-minute
micro-experiment. But the architectural problem is the cross-row shmem
reduction itself, not its width — going from 4 rows to 2 rows just halves
a small constant. The real win requires removing the cross-row shmem
accumulator entirely (the H1 restructure).

So this commit is **research-only**: the note above + the branch.
The implementation is a follow-up issue.

---

## 6. References

### llama.cpp source (MIT — cited, not redistributed)

- [`ggml/src/ggml-cuda/mmvq.cu`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu)
  — the GEMV (batch=1) path. `mul_mat_vec_q` template + `calc_nwarps` /
  `calc_rows_per_block`.
- [`ggml/src/ggml-cuda/vecdotq.cuh`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh)
  — `vec_dot_q4_K_q8_1_impl_vmmq` and constants.
- [`ggml/src/ggml-cuda/mmq.cuh`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh)
  — the *batched* MMQ path with mma tensor cores. **Not** used for batch=1.
- [`ggml/src/ggml-common.h`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-common.h)
  — `QK_K=256`, `QR4_K=2`, `QI4_K=32`, `QK8_1=32`.
- [PR #5394](https://github.com/ggerganov/llama.cpp/pull/5394) — the
  original NVIDIA MMVQ tuning that introduced 4-warp `nwarps` for
  ncols_dst=1: "use of more warps seems to be beneficial in scenarios where
  the compute per memory bandwidth is relatively high." 1.6× win on
  RTX 3090 with Q6_K at the time.

### dotLLM source

- `native/kernels/quantized_gemv_mmq.cu` — current MMQ Q4_K kernel.
- `src/DotLLM.Cuda/CudaKernels.cs` lines 880-980 — dispatcher and tile
  constant `MmqRowsPerBlock = 4`.
- `src/DotLLM.Cuda/CudaTransformerModel.cs` line 382 — fused QKV/GateUp
  call site.
- `.continue-here.md` §3c — the gap framing this note addresses.

### General references

- llama.cpp benchmark scoreboard ([discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013))
  — RTX 3060 with Llama-2 7B Q4_0: ~75 tok/s (this is dense FP16 attention,
  smaller weight than Q4_K_M Qwen3-8B, but anchors achievable order of
  magnitude on this GPU).
- 8B-class on RTX 3060 reported in third-party benchmarks at
  ~22–42 tok/s for Q4_K_M depending on backend (CUDA vs Vulkan) and
  flash-attention setting. dotLLM at 10.5 is at the bottom of that range.
