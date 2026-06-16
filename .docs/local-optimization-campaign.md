# Local optimisation campaign (2026-06-15) — results log

Hardware: Intel Core Ultra 7 155H (Meteor Lake, 16 logical: 6P+8E+2LP-E, AVX2+AVX-VNNI, no AVX-512),
Intel Arc iGPU, NVIDIA RTX 3060 (Ampere CC8.6, eGPU). Models cached: SmolLM2-135M, Qwen2.5-0.5B,
Llama-3.2-1B (Q8_0), Bielik-1.5B (Q8_0/Q4_K_M), SmolLM-135M (Q4_K_M/Q8_0).

Discipline: profile before building; serialize hardware benchmarks (no parallel contention); no single-draw
conclusions (multi-rep, report distribution); accuracy changes gate on perplexity. Each confirmed win committed.

## Status board
| # | Target | Spike | Status | Result |
|---|--------|-------|--------|--------|
| C1 | CPU | AVX-VNNI `vpdpbusd` Q8 dot (port #321) | pending | — |
| C2 | CPU | GFNI nibble unpack (Q4/Q2 dequant) | pending | — |
| C3 | CPU | Hybrid P/E-core thread tuning | pending | — |
| G0 | eGPU | Profile prefill/decode/dequant split | pending | — |
| G1 | eGPU | cuBLAS COMPUTE_16F prefill GEMM | pending | — |
| G2 | eGPU | INT8 cuBLASLt IMMA GEMM | pending | — |
| V0 | iGPU | Vulkan baseline on Intel Arc | pending | — |
| V1 | iGPU | Vulkan shader/subgroup + unified-memory | pending | — |

## Baselines (SmolLM-135M Q8_0, Vulkan harness, "The capital of France is" + 32 decode)
| Device | load | prefill(5tok) | decode avg | decode min | tok/s |
|---|---|---|---|---|---|
| Intel Arc iGPU (Vulkan) | **42 s** | 919 ms | 121 ms | 88.6 ms | **8.25** |
| NVIDIA RTX 3060 (Vulkan) | 7.8 s | 357 ms | 41 ms | 25.6 ms | **24.16** |

- **iGPU vs dGPU (same Vulkan backend): ~2.9× gap** (8.25 vs 24.16 tok/s) — override confirmed working.
- **Cross-cutting V0-c:** even the dGPU does only 24 tok/s on a 135M model via Vulkan (CUDA ~27 tok/s eager on
  Llama-1B too) → small-model decode is **launch/dispatch-overhead-bound on all three backends**, not compute or
  bandwidth. Lever = fewer kernel launches per token (command-buffer batching / CUDA graphs / kernel fusion).

- **iGPU finding V0-a:** 42 s cold load = Vulkan pipeline/shader compilation with **no VkPipelineCache** → every
  start recompiles all pipelines. A persisted pipeline cache (V1) would slash iGPU (and dGPU) startup.
- **iGPU finding V0-b:** decode second-half (steps 22-31, 160-186 ms) slower than first-half (88-138 ms) →
  shared CPU/iGPU power/thermal budget (no growing-KV explanation at this size).

### V1 — Q8_0 GEMV workgroup sweep (decode-weighted, vs production wg128)
| device | best variant | weighted speedup | per-shape highlights |
|---|---|---|---|
| Intel Arc iGPU | **`sg`** (subgroup-add) | **1.05×** | attn_q 1.23×, ffn_gu 1.20×, attn_kv 1.14× |
| NVIDIA RTX 3060 | **`wg64`** | **1.07×** | wg64 1.06-1.11× most shapes; attn_kv→wg256 2.03× |

- **Optimal GEMV variant is VENDOR-dependent** — `sg` wins on Intel but is *no help* on NVIDIA (0.99×); `wg64`
  wins on NVIDIA. Both currently use wg128 → ~1.05-1.07× decode GEMV left on the table on BOTH GPUs.
- **✅ IMPLEMENTED:** `MatMulQ8_0Kernel.Create` now picks per-vendor (Intel→sg, NVIDIA→wg64, else wg128) with
  a `DOTLLM_VULKAN_Q8_GEMV_VARIANT` override + .spv-exists fallback. (Note: per-SHAPE optimum also varies, e.g.
  attn_kv→wg256 2.03× on NVIDIA — a future per-shape dispatch could capture more.)
- **iGPU finding V1-b:** 42 s cold start = no `VkPipelineCache`; persisting one is a concrete startup win (also
  helps the dGPU's 7.8 s). **✅ IMPLEMENTED + verified** — `VulkanDevice` creates a per-device `VkPipelineCache`
  (`%APPDATA%/dotllm/vulkan_pipeline_cache_{vendor}_{device}.bin`), seeds it from disk on create, passes it to
  every `vkCreateComputePipelines` (see `VulkanModule`), and saves it atomically on dispose. Both device caches
  present on this box (Arc 1.25 MB, RTX 733 KB). **Cold-vs-warm measurement (deleting only the app cache):** ~0.2 s
  delta (6.1 s vs 5.9 s full-model build+decode) — small here because the **Intel driver's own shader cache is
  warm and dominates**. The VkPipelineCache is the portable, app-controlled mitigation that delivers the real
  cold-start win in fresh-install / CI / container environments where the driver cache is absent (the original
  42 s scenario); a fully-cold driver state can't be reproduced on this dev box without clearing system driver
  caches.

## Findings
(chronological)

- **2026-06-15 setup:** campaign branch `spike/local-platform-opt` off the bf16 branch (worktree
  `C:\Development\dotllm-bf16e2e`). PTX rebuilt at `compute_86` (38 kernels) for the RTX 3060. Both GPUs
  Vulkan-visible (Intel Arc integrated + RTX 3060 discrete). Stock .NET 10.0.103 SDK present.
- **C1 implemented (pending bench):** ported `OuterProductQ8_0Vnni_4x3` (VPDPBUSD) from #321 into MatMul.cs +
  dispatch branches (`AvxVnni.IsSupported` above the AVX2 branch, both single-thread + worker) + ported the 4
  VNNI parity tests + added `OuterProductVnni_VsAvx2_MicroBench`. DotLLM.Cpu builds clean.
- **G0 in progress:** added `CudaForwardProfileHarness` (per-stage GPU ms split), running on Llama-3.2-1B Q8_0,
  pp256 + decode burst.
- **✅ C1 DONE (committed `70c85c0`): AVX-VNNI is a real Meteor Lake win.** Pinned kernel A/B (VNNI/AVX2 ratio):
  K=8192 **1.97×** (core1)/1.34× (core0); K=2048 1.05×/**1.42×**. 14/14 parity pass. Dispatched only on
  AVX2+VNNI-without-AVX512 (Meteor Lake) — Strix/AVX-512 path untouched.
- **⚠ C3 lead found:** the microbench absolute times are INVERTED — `NumaTopology` labelled core 1 a P-core but
  it runs ~6× slower than core 0 (labelled E). Strongly implies **`NumaTopology.PerformanceCoreIds` misclassifies
  P/E on Meteor Lake** → `--pcore-only` would pin to the weak cores here. Investigate under C3.
- **✅ G0 DONE — warmed Llama-3.2-1B Q8_0 on RTX 3060 (profiling disables CUDA graphs → eager path):**
  - **PREFILL pp256 (103ms): Attention 43.1% (biggest!), GEMM 48.6%** (MlpUp 24%, MlpDown 13%, Qkv 7%, O 4.5%).
    Attention is O(seq²) → dominates further at longer prompts.
  - **DECODE (~31ms eager): GEMM(GEMV) 70.8%** (MlpUp 36%, MlpDown 18%, LmHead 17%), attn 4.3%. ~30 GB/s of
    360 → not bandwidth-saturated (eager/launch-overhead; real decode uses CUDA graphs, suppressed here).
  - (unwarmed first run was contaminated — 71% in the first category from lazy cuBLAS/PTX init; fixed by warmup.)
- **🔑 Top GPU prefill finding:** `native/kernels/attention.cu` (`attention_f16`) is a tiled online-softmax
  (flash-style) kernel but **FP32 on CUDA cores — no tensor cores**. The 43% prefill attention bypasses the
  3060's tensor cores entirely → biggest structural prefill prize (cuBLAS strided-batched QK/​PV GEMM, or mma
  flash attention). GEMM half is addressed by G1/G2.
- **G1 measured (minor): cuBLAS COMPUTE_16F vs _32F, interleaved (thermally fair).** Whole-prefill speedup
  **1.06× median / 1.22× best** (GEMM is ~half of prefill & only it benefits → ~1.1-1.45× on the GEMM portion).
  Real but modest, and carries an FP16-accumulation accuracy risk (needs a perplexity gate before adoption).
  Wired behind `DOTLLM_CUDA_GEMM_16F` / runtime `CudaGemm.Use16FCompute`. **Note: consumer-3060 clocks drift
  ~2× across consecutive heavy runs — a fresh-process GPU A/B is invalid; must interleave + use min.**
- **✅ C1 RE-MEASURED (reliable): VNNI = 1.55× on P-cores (flat across K=2048/4096/8192), 1.2-1.35× on
  E-cores.** Earlier swings (0.56-1.97×) were turbo/thermal confound from dividing separately-measured mins;
  the paired per-round-ratio method (time AVX2+VNNI back-to-back each round, median ratio) is stable. C1 commit
  stands; the true figure is ~1.55× P-core. **Methodology rule: pair A/B per round, median ratio — never
  divide separate mins on this thermally-unstable laptop (applies to CPU and GPU).**
- **✅ C3 BUG FOUND + FIXED: `NumaTopology` P/E inversion on Meteor Lake.** `EfficiencyClass` convention is
  HIGHER = higher-performing (P-core), but the code took the MIN as "Performance", so PerformanceCoreIds held the
  E/LP-E cores and `--pcore-only` pinned to the slow cores (the 6× microbench inversion). Fixed min→max
  (NumaTopology.cs:234-239). Non-hybrid (AMD/Strix) unaffected (all-equal → Unknown → all cores).

## G3 — Tensor-core prefill-attention GEMM-only ceiling check (go/no-go)

Before committing to a fused FP16 flash kernel (correctness-heavy), measured the **lower bound** of a cuBLAS
tensor-core attention: only the two strided-batched GEMMs (QK^T → scores in global mem, then scores·V → out),
**no softmax, no correctness**. GQA = one strided-batched GEMM per KV head (batch = group size; K/V stride 0 →
shared KV head broadcasts over its 4 query heads). 16 cuBLAS calls/attention (8 QK + 8 PV), FP16 in/out,
COMPUTE_32F accumulate. Shapes = Llama-3.2-1B (heads 32 / kv 8 / headDim 64). RTX 3060, CUDA events, 30 reps
interleaved, min. Harness: `tests/DotLLM.Tests.Unit/Cuda/CudaTensorCoreAttentionBench.cs`
(`DOTLLM_CUDA_ATTN_BENCH=1`). P/Invoke `cublasGemmStridedBatchedEx` added to `CublasApi.cs`.

| seq | attn_f16 (current, CUDA-core FP32) | gemm-only floor (tensor-core) | ceiling |
|-----|-----------------------------------|-------------------------------|---------|
| 256  | 1.939 ms  | 0.285 ms | **6.81×** |
| 512  | 7.722 ms  | 0.502 ms | **15.39×** |
| 1024 | 28.886 ms | 1.253 ms | **23.05×** |
| 2048 | 120.103 ms| 4.345 ms | **27.64×** |

- **Verdict: decisive GO at every tested prefill length** — even seq=256 (the worst case, low arithmetic
  intensity at headDim=64) clears 6.8×. Both backends scale cleanly O(S²) (attn ~4× / doubling; gemm-only
  1.76×→3.47× as the 16 small launches stop dominating), so the curve is trustworthy.
- The floor is missing only the softmax pass (causal mask + scale + row-normalise over scores). That is a
  memory-bound sweep of numHeads·S² FP16 (~536 MB R+W at S=2048 ≈ ~1.5 ms on the 3060). So a *real* cuBLAS
  attention at S=2048 ≈ 4.3 + ~1.5 + launch overhead ≈ **~6–7 ms vs 120 ms ≈ ~17–20×** — still enormous.
  The score round-trip through global memory IS already included in the floor (QK writes, PV reads).
- Why so large even at seq 256: the margin is "tensor-core GEMM vs a naive FP32 CUDA-core kernel", and the
  current kernel massively underperforms (~0.28 TFLOP/s at S=2048 vs ~7.9 for the GEMM floor; 3060 FP16-TC
  peak ~51). This re-confirms finding #1 with a concrete number and crossover (GO at all S ≥ 256).
- **Next increment (the actual prototype):** insert a causal-softmax kernel between the two GEMMs, verify
  numerics against `attention_f16`, then time the *complete* path to convert this ceiling into a measured
  real-world win. A hand-fused mma flash kernel (scores never leave shared/registers) would beat even this,
  but the cuBLAS path is far less code and already ~17–20× — likely the right first ship.

## C4 — Zero-point 256-bit AVX-VNNI Q8_0 kernel (counterpart to strix #322 AVX-512 ZP) — SHELVED

Strix's #322 built AVX-512 VNNI **zero-point** Q8_0 outer-product kernels (`OuterProductQ8_0Avx512VnniZp_4x6`)
that drop the per-cell `VPSIGNB` sign trick: feed `VPDPBUSD` an unsigned operand `u = x ^ 0x80 (= x+128)`
directly, then remove the `128·Σw` bias. Those AVX-512 variants are **bench-only on strix — never wired into
`OuterProductGemm.cs` dispatch** (consistent with the Zen5 "512-bit quant kernels are net-negative" lesson).
The 256-bit counterpart was the one genuinely transferable element to Meteor Lake (which has AVX-VNNI-256 but
no AVX-512), so it was built + measured here: `OuterProductQ8_0VnniZp_4x3` in `MatMul.cs`.

- **Accuracy: PASS.** Parity vs scalar at the same 1e-2 bar as the sign-trick path, through K=8192 (the worst
  case for the deferred `−128·Σw` compensation's fp32 cancellation), and ZP-vs-sign-trick within 5e-3. The
  cancellation the deferred form risks is a non-issue at prefill K. (`OuterProductVnniZp_4x3_MatchesScalar`, 8/8.)
- **Performance: REGRESSION — ~2× slower.** Paired per-round median ratio (sign-trick / ZP), pinned, 15 rounds:

  | core | K2048 | K4096 | K8192 |
  |------|-------|-------|-------|
  | P-core (0) | 0.50× | 0.54× | 0.56× |
  | E-core (1) | 0.50× | 0.61× | 0.54× |

  (ratio <1 ⇒ ZP slower; e.g. P-core K8192: sign-trick 10.7 µs vs ZP 20.0 µs.)
- **Why:** ZP removes the per-cell `VPSIGNB` but must compute `Σw` per (row, block) via two 256-bit
  **horizontal reductions** (`Vector256.Sum`) in the hot loop. A cross-lane horizontal sum is far more
  expensive than the `VPSIGNB` it replaces, so the trade is net-negative — the same reason strix shelved its
  AVX-512 ZP kernel. **Verdict: do NOT dispatch.** Kept as a benchmarked-and-shelved artifact + gated tests.
- **Real potential (the actual gap → next issue):** ZP only pays if `Σw` is **precomputed once** (it's a static
  property of the quantized weights, reused across the entire token/N dimension) and read as a cheap scalar in
  the loop — not recomputed per call. That needs the R4 weight repacking to also emit per-block signed weight
  sums (a layout change touching `WeightRepacking` + the GEMM driver), which is a larger change than a drop-in
  microkernel. That is the documented "potential" worth a tracked issue; the drop-in form is a confirmed dead end.

### C4 probe — precompute-Σw (the only ZP form that could win): measured, still doesn't pay on P-cores

Per advisor, before claiming "precompute-Σw is the potential" in the issue, measured it directly:
`OuterProductQ8_0VnniZpPre_4x3` reads `128·Σw` from a buffer filled ONCE outside the timed loop — a faithful
model of the amortized real case (Σw is a static weight property reused across all of N). Paired median ratio
(sign-trick / ZP-pre), pinned, 15 rounds:

| core | K2048 | K4096 | K8192 |
|------|-------|-------|-------|
| P-core (0) | 0.98× | 0.98× | 0.97× |
| "E-core" (1) | 1.48× | 1.53× | 1.64× |

- **P-core: ≤1× — no benefit (0.82–0.98× across two runs).** Even with the in-loop `Σw` cost fully removed,
  dropping `VPSIGNB` never wins on the P-core. **Root cause: the kernel is `VPDPBUSD`-throughput-bound on the
  P-core** — the sign instruction is already hidden under VPDPBUSD issue. Any ZP form keeps ≥3 `VPDPBUSD`, so
  the P-core ceiling is parity, not a win.
- **E-core: a real win, ~1.1–1.6× (direction-robust, magnitude thermally noisy).** Topology dump confirms the
  pinned core (logical 1, phys 1, `CoreType.Efficiency`) is a **genuine Crestmont E-core, not a P-core SMT
  sibling** (HT is disabled on this box: 16 logical = 16 physical; P={0,9,10,11,12,13}, E={1–8}, LP-E={14,15}).
  Across two runs × 3 K the E-core ratio is >1 in all 6 cells (run1 1.48–1.64×, run2 1.07–1.21×); magnitude
  varies because the laptop throttled under repeated benches (absolute times rose ~2.5× run-to-run). The
  E-cores (Gracemont/Crestmont-class) *are* `VPSIGNB`-limited where the P-cores are not, so removing it helps.
- **Net for the issue (corrected):** drop-in ZP = 2× slower everywhere (dead). Idealized precompute-ZP = no
  P-core benefit, but a **confirmed E-core-only win (~1.1–1.6×)**. Capturing it would need ALL of: (1) the
  `WeightRepacking` Σw-precompute layout change, (2) **per-core-type dispatch** (sign-trick on P, ZP on E), and
  (3) an end-to-end prefill bench to size the aggregate (E-cores do part of the parallel GEMM; the net win
  depends on the P/E work split and is bounded by E-core thermal noise). That is a real but contained future
  optimization — decision deferred to the maintainer; the drop-in form is a confirmed dead end.

## C5 — Hybrid per-core ZP dispatch (the full end-to-end build) — REGRESSION, shelved

Per the user's request to make the confirmed E-core ZP win real end-to-end, built the full integration (NOT
just the kernel): per-group core-type dispatch that routes Efficiency-core workers to the ZP kernel while
P-core workers keep the sign trick. Design avoided a weight-layout change: Σw is precomputed once per R4 group
into per-worker scratch and reused across all the group's token tiles within a single GEMM call.

Pieces (all on `spike/local-platform-opt`):
- `OuterProductQ8_0VnniZpPre_4x3` kernel (reads precomputed Σw).
- `CpuAffinity.GetCurrentProcessorId()` (Win `GetCurrentProcessorNumber` / Linux `sched_getcpu`).
- `MatMul`: lazy logical-core→Efficiency map, `ComputeGroupRowBlock128Sw`, `UseZpForCurrentCore()`, runtime
  toggle `ZpOnEfficiencyCoresEnabled` (env `DOTLLM_CPU_ZP_ECORE`, default OFF), + a `ForceZpAllCoresForTesting`
  hook. Wired into both the serial and parallel-worker AVX-VNNI branches of `OuterProductGemmQ8_0`.

Go/no-go first (advisor-prescribed): real CPU prefill, all-16-unpinned vs pcore-6, both sign-trick, in-process
interleaved → all-16 ≥ pcore-6 (min 1.02×, median 1.26×), i.e. **E-cores help prefill** — favorable branch, so
no heterogeneous scheduler needed (E-cores already pull equal shares acceptably).

Correctness: `ZpDispatch_ForwardMatchesBaseline` — full prefill forward with ZP forced on all cores (outer-
product path enabled) matches the sign-trick baseline to **0.4% of logit scale** (maxAbs 8.81, maxDiff 0.036),
guarded by an `OuterProductGemmQ8_0InvocationCount` assertion (which caught that the outer-product prefill path
is itself opt-in via `UseOuterProductQ8Prefill` — an earlier vacuous maxDiff=0 pass).

End-to-end A/B (one all-16 model, outer-product path on, ZP-on-E OFF vs ON, interleaved, 11 reps):

| | min ms | tok/s |
|---|---|---|
| ZP-on-E OFF (baseline) | 35743 | 14.3 |
| ZP-on-E ON | 37410 | 13.7 |

**speedup(ON vs OFF) = 0.96× — no demonstrable win (likely indistinguishable from 1.0 at this box's noise).**
The isolated microbench E-core win (1.1–1.6×) does not translate. Given the prefill A/B noise floor (the
threading A/B alone swung 1.02–1.26×), 0.962× is best read as **"no measurable effect,"** not a precise 4%
regression.

**Mechanism (corrected — the earlier "dispatch overhead" claim was wrong):** at the 35,000 ms prefill scale,
per-group `GetCurrentProcessorNumber` + Σw precompute is negligible (~200K group-dispatches × ~50 ns ≈ ~10 ms ≈
0.03%) — it cannot explain a 1,400 ms gap. The plausible real cause is **unpinned-worker migration**: a group
sampled on an E-core can execute ZP while the thread has since moved to a P-core, where ZP *loses* (0.82–0.98×
per C4). A fraction of groups running the slower-on-P kernel is a clean source of a small consistent slowdown.
That is an artifact of *this* implementation — **unpinned + per-group-sampled dispatch** — not a property of
"ZP on E-cores."

**Not pursued further:** a pinned-static dispatch (one core-type lookup per worker, no migration) would isolate
that hypothesis and was NOT built. Its expected ceiling is still only ~ZP-gain × E-core-share ≈ ~1.05×, which is
below this box's prefill noise floor — so even if it reached the ceiling the win would be unmeasurable here.

**Verdict: shelved, gated OFF by default** (zero production impact) — a documented, tested artifact like strix's
AVX-512 ZP. What IS established on Meteor Lake: ZP is dead as a drop-in (P+E 2× slower); has no P-core benefit
even idealized (VPDPBUSD-bound); has a real-but-isolated E-core microbench win (~1.1–1.6×) that did NOT yield a
measurable end-to-end gain in the unpinned per-group-dispatch build. **Caveat if ever un-shelved:** the ZP
forward drifts ~0.4% from the sign trick on logits (deferred-compensation fp32 cancellation) — enough to move
argmax, so it needs a perplexity gate (as G1 does), not just the kernel parity tests. Issue #22 carries the
full chain.

---

## V2 — DP4a INT8 Q8_0 GEMV on Intel Arc (Xe-LPG) — partial win, prequant pass needed

Counterpart to strix's INT8 ZP work, on the iGPU. Adds a `VK_KHR_shader_integer_dot_product`
(`dotPacked4x8AccSatEXT`, hardware-accelerated on Xe-LPG: `integerDotProduct4x8BitPackedSignedAccelerated=true`)
variant of the Q8_0 decode GEMV. Quantizes the FP32 activation slice to INT8 per 32-block and combines two
scales (`d_weight · dx_act · int32_dot`), mirroring the CUDA MMQ convention (`native/kernels/quantized_gemv_mmq.cu`).
Files: `matmul_q8_0_dp4a.comp`, `MatMulQ8_0Dp4aKernel`, `VulkanDevice.HasIntegerDotProduct` probe/enable.
Commit `0e50967` (spike branch).

**Pre-checks (all green):** Vulkan SDK installed (`glslc` shaderc v2026.2); existing `matmul_q8_0` recompiled
from source and its Vulkan parity test passed on Arc against the fresh `.spv`; GLSL builtins compile under
vulkan1.2; Arc advertises the extension + feature + 4×8 signed *accelerated*.

**Correctness (Arc, 10 shapes):** validated against a CPU reference that quantizes `x` identically — **exact**
(maxRel ≤ 2e-4, mostly 0). The drift vs an FP32-activation reference is the expected INT8-activation error
(near-zero-denominator outputs inflate the relative metric on random data; not a kernel bug).

**Microbench (Arc, `DOTLLM_VULKAN_DP4A_BENCH=1`, min over 5 rounds, scalar baseline = best per-vendor variant `sg`):**

| Shape (M×K) | Scalar ms | DP4a ms | Speedup |
|---|---|---|---|
| 576×576 | 8.34 | 1.40 | 5.98× (scalar first-shape outlier — unreliable) |
| 1536×576 | 3.59 | 1.41 | 2.55× |
| 576×1536 | 3.95 | 1.44 | 2.75× |
| 49152×576 (lm_head) | 26.3 | 41.9 | **0.63× (regression)** |
| 4096×4096 | 9.00 | 9.64 | 0.93× |
| 4096×14336 | 57.0 | 36.2 | 1.57× |

(For contrast, on the RTX 3060 the same bench is ~0.96–0.99× across large shapes — memory-bound GEMV, DP4a
accelerates MACs not weight reads.)

**Reading:** the win tracks K/M. DP4a wins when K ≫ M (activation re-quant amortized over deep K); it **regresses
on high-M lm_head** because the shader re-quantizes the *same* `x` once per workgroup = M times (49152×). Laptop
thermal/turbo noise makes the small-shape numbers unreliable (the 576×576 scalar 8.34 ms is a first-shape
artifact — it is *faster* at 1536 rows).

**Verdict: correct and promising but not an unambiguous win as the inline-requant form.** The fix is a **shared
x-prequantization pass** (quantize `x` to INT8 + per-block scales once into a side buffer, à la llama.cpp
`quantize_q8_1`, then the GEMV reads `xq`), removing the M× redundancy — expected to convert the lm_head/large-M
regressions into wins. That pass (new shader + buffer + 2-dispatch wiring) is the next step before an upstream PR.

**Strip-x-quant probe (decisive, advisor-prompted):** re-benched with all activation processing removed
(constant `xpack`, fixed scale — numerically wrong, weight-read + dp4a only). Same-run comparison (absolute ms
are thermal-noisy between runs; within-run is the signal):

| Shape | Scalar | DP4a inline | DP4a probe (no x-proc) |
|---|---|---|---|
| 49152×576 (lm_head) | ~19–26 | 41.9 | **9.37 (2.03× vs scalar)** |
| 4096×4096 | ~9–15 | 9.64 | 5.14 (2.97×) |
| 4096×14336 | ~38–57 | 36.2 | 16.3 (2.32×) |

lm_head drops 41.9 → 9.37 ms when x-processing is removed — the **redundant per-workgroup x-requant is the
bottleneck, not the weight-read floor.** (Contrast the 3060, which sits at ~1.0× because it *is* weight-read
bound; Arc's scalar Q8_0 GEMV is ALU/dequant-bound, so the packed-int path genuinely wins once x-quant is
amortized.) **Conclusion: the shared x-prequantization pass is evidence-backed to deliver ~2–3× on the reliable
large shapes including the decode-critical lm_head** — worth building. The inline kernel as committed (`0e50967`)
is correct but a net decode regression; it is the validated building block, not the shippable form.

### V2 — shared-prequant DP4a: clean win (batched-fence instrument)

Built the shared activation prequantization pass (`quantize_q8_act.comp` once → packed INT8 `xq` + per-block
`dx`; `matmul_q8_0_dp4a_pq.comp` GEMV reads them), orchestrated by `MatMulQ8_0Dp4aPqKernel` as two dispatches +
compute→compute barrier behind one submit. Parity 10/10 on Arc. Commit `96a9e31`.

**Instrument fixed (advisor):** the per-Launch `vkQueueWaitIdle` floor (~1.4 ms) made the first 3-way bench
unreadable on small shapes and double-counted nothing real but masked the win. Re-benched with a **batched
fence** (N=100 iters recorded into one command buffer, one submit/wait, ÷N; a barrier after each iter serializes
to true per-matmul GPU time) and **paired per-round median ratios** (3 kernels interleaved within each of 7
rounds; median of the per-round ratio — cancels turbo/thermal drift, vs the earlier independent-min-per-side
confound).

| Shape (M×K) | Scalar ms | Inline ms | Prequant ms | inline× | **prequant×** |
|---|---|---|---|---|---|
| 576×576 | 0.408 | 0.389 | 0.347 | 1.01× | **1.18×** |
| 1536×576 | 0.744 | 0.827 | 0.464 | 0.90× | **1.44×** |
| 576×1536 | 0.647 | 0.452 | 0.421 | 1.48× | **1.68×** |
| 49152×576 (lm_head) | 16.21 | 12.15 | 7.05 | 1.74× | **3.39×** |
| 4096×4096 | 5.91 | 4.27 | 3.03 | 1.60× | **2.35×** |
| 4096×14336 | 23.78 | 7.10 | 6.62 | 2.16× | **2.23×** |

**Verdict: shared-prequant DP4a Q8_0 GEMV is a clean win on Arc Xe-LPG — 1.18×–3.39× across all decode shapes,
including the decode-critical lm_head (3.39×) and deep-K FFN (2.23×).** lm_head prequant 7.05 ms beats the
strip-x-quant probe's 9.37 ms → the quantize→gemv barrier is cheap/amortized, NOT a per-matmul tax; the earlier
2-dispatch "regression" (17.7 ms) was per-Launch sync that batched inference does not pay. The inline kernel is
the validated building block but a mixed performer (0.90–2.16×); the **prequant kernel is the shippable form.**

**Productionisation note (not blocking the win):** in a real layer Q/K/V share one input activation and gate/up
share another, so the `quantize_q8_act` cost can be shared across the 3/2 same-input projections — the shaders
are already split for this; it's a forward-integration concern that would widen the small-shape margins further.
Shippable form should shape-gate (prequant for these decode GEMVs; scalar would only matter for tiny isolated
matmuls, which here are still ≥1.18×). The DP4a path is decode-GEMV; the compute-bound prefill/GEMM DP4a (the
original "4–8×" headline) remains separate future work.

---

## V5 — DP4a wired into the Vulkan forward pass + shared K/V activation quant

Took the V2 shippable kernel (`MatMulQ8_0Dp4aPqKernel`) from "validated building block" to "integrated into
`VulkanTransformerModel`'s decode path", opt-in behind `DOTLLM_VULKAN_ENABLE_DP4A=1`. Commits on
`spike/local-platform-opt`: `3412f78` (Step A — route Q8_0 seqLen==1 GEMVs in `RecordMatmul` through DP4a; shared
xq/dx scratch in `VulkanForwardState`, sized to the largest contraction dim, decode-only/seqLen-independent),
`316048d` (Step B — split the PQ kernel into `RecordQuantizeActivation` + `RecordGemvPrequant` and add
`RecordSharedInputMatmulPair` so the K/V pair quantizes the shared input once), `489cb1a` + `2a09738` (real-model
tests + `DOTLLM_VULKAN_DP4A_NO_SHARE` gate).

**Structural finding — fusion limits the shareable groups.** The default rmsnorm+matmul fusion
(`RmsNormMatmulQ8_0FusedKernel`, FP32-activation) already produces Q (attn) and Gate (FFN), so under fusion the
only genuine same-input DP4a *pair* is **K/V**; FFN `Up` is a lone DP4a GEMV and Q/Gate don't use DP4a at all.
Shared-quant is wired at K/V; gate/up sharing would only apply with fusion off.

**Validation (all on Arc Xe-LPG):**
- Kernel parity + new `SharedQuant_TwoGemvs` (one-quant + two-gemv **bit-identical** to two full Launches).
- MoE Q8_0 forward oracle parity green with DP4a on (abs 5e-3 / rel 1e-3 vs FP32-activation CPU oracle).
- Default-path regression 22/22 (opt-in ⇒ zero behaviour change when off; the K/V pair-helper falls back to the
  pre-existing two-`RecordMatmul` form).
- Real-model SmolLM-135M Q8_0 (`HybridPrefillDecodeTests`): DP4a **engages** (first-decode logits differ off-vs-on),
  wiring **correct** (first-decode top-10 overlap 10/10, max|Δ|≈0.32, rms≈0.076 on ~23-magnitude logits), and the
  **shared-K/V path is bit-identical** to the per-matmul fallback (`Decode_Dp4aShareVsNoShare_RealModel_BitIdentical`
  via `DOTLLM_VULKAN_DP4A_NO_SHARE`) — direct verification, not "identical by construction".

**Performance — Release end-to-end decode (SmolLM-135M Q8_0, 16 steps, Arc):** off **1791 ms** → on **1126 ms**
= **~1.6× decode speedup**. (Debug build showed a *slowdown* — build overhead masks the GEMV win; single-run wall
clock, indicative. The rigorous per-kernel figure remains the V2 batched-fence bench, 1.18–3.39×.) The V2 per-kernel
win **does** translate to a net batch=1 decode win at this scale once out of debug.

**Quality caveat:** DP4a's per-block INT8 activation quant perturbs decode logits ~0.3% rms, enough to flip a
top-1/top-2 **near-tie** (this prompt's top two differ by <0.2), so the *greedy trajectory* diverges off-vs-on
(`Decode_Dp4aOnVsOff_RealModel_PreservesTopKRanking` — ranking preserved, so it's quant drift not a bug). A token
flip on a near-tie is not a quality loss, so the greedy divergence is the wrong gate.

**Quality gate — decode-mode perplexity off-vs-on (the right measure): PASS, ~lossless.** Standard perplexity
scores the prefill path (seqLen>1) where DP4a never engages; instead measured *teacher-forced decode* perplexity —
feed the corpus one token at a time through the KV cache so every forward is seqLen==1 and DP4a runs on every
projection of every step (`DecodePerplexity_Dp4aOnVsOff_RealModel_WithinTolerance`). SmolLM-135M Q8_0, 111-token
English corpus, Arc:

| | decode-mode PPL |
|---|---|
| DP4a off (FP32 activation) | 15.715 |
| DP4a on (INT8 activation) | 15.742 |

**ratio on/off = 1.00166 (+0.166%)** — far below the 1% gate. The INT8 activation quant is effectively lossless on
decode quality; the greedy-token divergence is purely near-tie reordering. CUDA's Q8_0 decode keeps both an
FP32-activation (`quantized_gemv_q8_0_f32in`) and INT8 path, consistent precedent.

**Default-on — DONE (vendor-gated to Intel/Arc).** The quality gate being cleared, `BuildModel` now defaults DP4a
**on when `device.VendorId == 0x8086` (Intel)** and off elsewhere — V2 measured ~0.96–0.99× on the RTX 3060
(memory-bound GEMV; DP4a accelerates MACs not weight reads), so a blanket default would add prequant dispatch for
no NVIDIA benefit. `DOTLLM_VULKAN_ENABLE_DP4A` overrides either way (`"1"` forces on any vendor, `"0"` forces off).
Verified on Arc by `Decode_Dp4aDefault_IsOnForIntel_OffOverridable` (env-unset trajectory == forced-on, ≠ forced-off).
Deep-model descriptor pool: **fixed** (commit `e0a6bf6`) — `MatMulQ8_0Dp4aPqKernel.Create` now takes a pool-size
param and `BuildModel` sizes it from `config.NumLayers` (16/layer + 128, floored at 1024), so DP4a-on no longer
risks `VK_ERROR_OUT_OF_POOL_MEMORY` on deep models.

**Commits (spike `spike/local-platform-opt`):** `3412f78` (Step A), `316048d` (Step B shared K/V), `489cb1a`
(real-model characterization), `2a09738` (share-vs-no-share verify + Release timing), perplexity gate test (this
section).

---

## Round 2 — remaining targets (G1 gate, C2, G3, C4/C5 recompare, G2 assessment)

All on `spike/local-platform-opt`, all independently verified on this box's hardware (RTX 3060 / Arc / Meteor Lake).

### G1 perplexity gate — ✅ ADOPT (commits `7be7250`, `2858971`)
`COMPUTE_16F` (`DOTLLM_CUDA_GEMM_16F` / `CudaGemm.Use16FCompute`) is **prefill-only for quantized models** —
Q8_0 `seqLen==1` decode routes through the integer GEMV and never reaches `GemvF16` (confirmed empirically).
Measured on Llama-3.2-1B-Instruct-Q8_0 (hidden 2048, worst case for FP16-accumulate): growing-prefix teacher-forced
**prefill PPL off=15.563 → on=15.551 (−0.079%)**, **decode bit-identical**. Engagement confirmed (prefill logits
differ max|Δ|=0.039; decode bit-identical = integer path). −0.079% is noise, well under the 1% gate.
**Recommendation: default-on for GeForce Ampere + quantized weights, keep the env override** (the ~1.06×/1.22×
prefill win is consumer-Ampere-specific; datacenter cards don't throttle FP32 tensor accumulate).

### C2 GFNI nibble unpack — ✅ measured WASH, shelved (commit `706d6bd`)
GFNI `VGF2P8AFFINEQB` high-nibble extract for Q4_K integer `vec_dot` (low nibble = plain `vpand`, no GFNI gain;
Q2_K is 2-bit scalar, no AVX2 baseline). **Bit-identical parity** (exhaustive matrix check + `SingleToInt32Bits`
equality, 19/19 tests). **Paired-median GFNI/AVX2: P-core 0.99× (wash), E-core 1.05× (slower).** Root cause:
`VGF2P8AFFINEQB` is a ~3-cycle op on **ports 0/1 — the same ports as the `vpmaddubsw`/`vpmaddwd` that bound the
loop**, so it displaces rather than offloads. Same port-contention family as C4/C5. Gated `EnableGfniQ4K` off, AVX2
stays default.

### G3 tensor-core prefill attention — ✅ SHIP coalesced cuBLAS+softmax (commits `5dfed4e`, `d2cfb71`, `490f1d8`)
The decisive-GO follow-through: a causal-softmax CUDA kernel (`attention_softmax_causal.cu`, two variants;
**coalesced one-thread-per-row is the default/shipping form**) between the two cuBLAS strided-batched GEMMs.
FP32-internal softmax over the column-major FP16 scores, causal-prefix max/exp/normalize + non-causal tail zeroing.
**Parity vs `attention_f16`: passes s=128–2048** (abs/rel 5e-3, coopmat precedent; bug-injection confirmed the bar
discriminates — disabling tail-zeroing turns it red). **Complete-path speedup (RTX 3060, interleaved min/30):**

| seq | attn_f16 | cuBLAS+softmax (coalesced) | speedup | GEMM-only floor |
|----:|---------:|---------------------------:|--------:|----------------:|
| 256  | ~1.95 ms | ~0.6–0.75 ms | **2.6–3.2×** | 0.24 ms (8.2×) |
| 512  | ~7.0 ms  | ~0.85–0.91 ms | **7.8–8.2×** | 0.44 ms (15.9×) |
| 1024 | ~27.9 ms | ~2.40 ms | **11.6×** | 1.24 ms (22.6×) |
| 2048 | ~113 ms  | ~8.6 ms | **13.1×** | 4.32 ms (26.3×) |

Coalescing the softmax was decisive (3.8×→8.2× at s=512); at s=2048 the path is ~2× the GEMM-only floor (the
unavoidable s×s score round-trip). **Verdict:** ship coalesced cuBLAS now — at pp256 (s=256, attention 43% of the
step) ~3× attention ≈ **~1.4× of the ~1.75× Amdahl ceiling**, most of the available pp256 win at low risk. A
hand-fused flash kernel (never materializes scores → approaches the ~25× floor) retains upside **only at long
context (s ≥ 1024)** and is not needed for pp256. Toolchain note: `native/build_ptx.bat` has stale MSVC paths on
this box; PTX was built via direct `nvcc -arch=compute_86`; the committed `.ptx` works and the `.cu` auto-globs.

### C4/C5 recompare — unchanged, still dead
Re-ran `OuterProductVnniZp_VsVnni_MicroBench` (Release, paired median, fixed NumaTopology): drop-in ZP
**0.53–0.57× (≈2× slower)** on both P and E cores — identical to the original 0.50–0.56×. The Round-1/Round-2 fixes
were GPU/Vulkan/CUDA + CPU dot kernels; none change the CPU execution-port economics that make ZP port-bound, and
C2 independently re-confirmed the same mechanism. **No change — ZP stays shelved.**

### G2 (INT8 cuBLASLt IMMA prefill GEMM) — assessment: NOT worth building now
No IMMA scaffolding exists; this is a re-assessment in light of G1+G3, not a measurement. **G1 is the decisive
datum:** FP16 tensor-core GEMM lifted whole-prefill only ~1.06–1.22× ⇒ ~1.13–1.5× on the GEMM portion itself —
i.e. the 3060 prefill GEMM is **not tensor-throughput-bound** at these shapes (FP16 tensor cores would give
multiples if it were; it's memory/launch-bound). INT8 IMMA's ~2× tensor-throughput advantage therefore won't
convert, while it carries large cost (IMMA's interleaved-layout constraints) **and** an INT8-activation accuracy
gate (prefill, larger seqLen). After G3, attention is no longer the prefill bottleneck (GEMM dominates more), which
*raises* the relative target — but G1 caps the achievable GEMM speedup low enough that IMMA isn't justified.
**Recommendation: do not build G2; the prefill prize was attention (G3, shipped).** If revisited, gate on a proper
cuBLASLt IMMA GEMM-only ceiling probe first (mirror the G3 go/no-go).
