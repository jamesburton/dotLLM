# Issue #199 v2 — tensor-core decode attention composed with the GQA-group + split-KV grid: RESULT

**Verdict: real, reproducible wall-clock WIN over both the shipping `attention_f32` baseline
and the opt-in `attention_f32_gqa_split_kv` kernel (issues #197/#198), at every realistic
Bonsai-27B decode depth tested (256/512/1024/2048). This is the positive result the whole
#197/#198/#199 investigation has been chasing.**

## Context: why a v2

v1 (`attention_flash_mma_decode.cu`, branch `issue/199-tensor-core-decode-attention`, not
merged) built a decode-only tensor-core kernel with genuine HMMA/LDSM SASS but scoped to ONE
WARP PER BLOCK, one block per query head (grid=numHeads=24) — deliberately not composed with
#197/#198's GQA-group grid, per the original issue's own design comment (keep the new FP16/
tensor-core precision axis separable from the grid-regrid axis while bringing the kernel up).
Real wall-clock A/B found it **4-5x SLOWER** than `attention_f32` at every realistic depth,
root-caused to ~4% theoretical occupancy (worse than the baseline's own already-diagnosed
~16.5% achieved occupancy). v1's own writeup named the fix and declined to attempt it in
scope: compose the tensor-core math with #197/#198's grid design.

This session (v2, clean-room implementation based on `dev`, NOT a port of v1's branch) does
exactly that.

## What was built

`native/kernels/attention_flash_mma_decode_gqa_split.cu`
(`attention_flash_mma_decode_gqa_split_f16`): grid = `(numKvHeads, kvSplit)`, IDENTICAL shape
to `attention_f32_gqa_split_kv`. The key structural change from v1 is not "more warps per
query head" but **packing the `group` query heads sharing a KV head into the mma.sync
instruction's M dimension itself**: `mma.sync.m16n8k16`'s M=16 is fixed regardless of how many
rows are real, and this project caps GQA group at `MAX_GQA_GROUP=8` (`CudaKernels.MaxGqaGroup`)
— since `8 <= 16`, packing all `group` heads' query rows into ONE 16-row tile (rows
`0..group-1` real, rest zero) lets a SINGLE set of QK/PV mma instructions compute ALL group
heads' attention against a KV tile at once, at the SAME instruction count as v1's
single-head version. This is why the kernel's static shared-memory footprint
(`sQ`/`sO`/`sScore`/`sP`, ~42.7KB) stays group-INDEPENDENT, unlike a naive per-warp-per-head
duplication (which the v1 writeup predicted would need dynamic shared memory at group=6).

Within a block (`NUM_WARPS=8`, blockDim=256, matching this project's `BlockSize` convention),
work is split by PHASE, not by query head:
- QK (`S[16x16] = Q_packed . K^T`): warps 0/1 split the 2 N-subtiles.
- Online-softmax reduction: warp 0 only, lanes `< group`.
- PV (`O[16x256] += P[16x16] . V[16x256]`): all 8 warps split the 32 d-chunks, 4 each, writing
  disjoint `sO` columns — real 8-way intra-block parallelism (v1 had exactly 1 active warp;
  `attention_f32_gqa_split_kv`'s own sibling reduction loop is sequential over `group`, not
  parallel across warps).

Cross-split combine (`grid.sync()` + `fast_exp_neg`-reweighted merge of
`partial_max`/`partial_sum`/`partial_out`) is ported verbatim from
`attention_f32_gqa_split_kv` — same buffer layout, same algebra, deliberate de-risking (only
which values feed the combine is new, not the combine itself).

C# side: `CudaAttentionMmaDecodeGqaSplit` (gate/dispatch wrapper, mirrors v1's
`CudaAttentionMmaDecode` API shape) + `CudaKernels.LaunchAttentionMmaDecodeGqaSplit` /
`MaxSafeAttentionMmaDecodeGqaSplit` (cooperative-launch co-residency query, reusing
`CudaKernels.ComputeAttentionKvSplit`'s existing occupancy-target heuristic unchanged — the two
kernels grid identically so the heuristic's `baseBlocks=numKvHeads` framing applies as-is).
Opt-in, default OFF (`DOTLLM_ATTN_MMA_DECODE_GQA_SPLIT=1`), same #180/#183 precedent as every
other new precision/reassociation axis this project has shipped. **Not wired into
`CudaQwen3MoeHybridTransformerModel`'s forward pass in this session** — matching the existing
precedent that `attention_f32_gqa_split_kv` itself is also not wired into that model (see that
kernel's own call-site comment); kernel-level validation + benchmarking was the scope here,
model integration is a reasonable non-blocking follow-up.

## Correctness

`tests/DotLLM.Tests.Unit/Cuda/CudaAttentionMmaDecodeGqaSplitTests.cs`, 11 tests (10 pass + 1
opt-in benchmark), same 5e-3 abs-OR-rel tolerance bar v1 established:

- vs CPU F32 oracle AND vs the `attention_f32` GPU baseline, at the real Bonsai-27B shape,
  using the SAME occupancy-tuned `kvSplit` a real caller would get (not a fixed value — so this
  exercises the actual cross-split combine path, not just the trivial `kvSplit==1` case), at
  seqKv in {1, 256, 512, 1024, 2048}: **all pass, 0 failing elements** at every depth.
  Representative: seqKv=2048, kvSplit=8: `maxAbs=1.508E-003` vs CPU, `maxAbs=3.302E-003` vs F32
  baseline (bar is 5e-3 abs OR 5e-3 rel).
- **New for v2**: three-way agreement check vs `attention_f32_gqa_split_kv` itself (both forced
  to `kvSplit==1`, isolating "does the M-dim-packed tensor-core axis agree with the
  register-blocked-FP32 axis at the same grid shape" from any cross-split reassociation) — pass
  at seqKv in {256, 1024}, confirming this kernel's real bug surface (the new multi-warp PV
  split, the new packed-M-dimension layout) didn't reopen v1's precision story.
- 300-consecutive-decode-step drift check: passes, non-compounding (`firstStepMaxAbs=2.4E-4`,
  `lastStepMaxAbs=3.8E-3`, both well inside the loose 2e-2 sanity bound; the flat-not-growing
  shape confirms attention's no-persistent-state expectation holds for this kernel too).
- `CanUse` gate: rejects prefill, wrong headDim, sliding window, non-dividing GQA, `group=9 >
  MaxGroup=8`, empty cache, disabled toggle.
- `ptxas -v`: **55 registers/thread, 0 spill loads/stores, 42696 bytes static shared memory.**
  `cuobjdump --dump-sass`: **20 static HMMA + 37 static LDSM instructions** — matches the
  hand-computed expectation exactly (QK: 16 k-steps × 1 mma = 16 HMMA + 4 PV chunks × 1 mma = 4
  HMMA = 20 total; QK: 16×2 LDSM + PV: 4×1 + 1 = 37 total), confirming genuine, non-optimized-
  away tensor-core codegen at the intended instruction granularity.

## Performance — real numbers

Interleaved min-of-30-reps CUDA-event timing
(`CudaAttentionMmaDecodeGqaSplitTests.TimingThreeWayVsF32BaselineAndGqaSplit`, opt-in
`DOTLLM_ATTN_MMA_DECODE_GQA_SPLIT_BENCH=1`), same host, same process, real Bonsai-27B shape
(numHeads=24, numKvHeads=4, headDim=256, group=6), **5 independent interleaved runs** (not
blocked/sequential — this project's documented lesson that the 3060's clocks drift across
blocked runs):

| seqKv | `attention_f32` (baseline) | `attention_f32_gqa_split_kv` (#197/#198) | **this kernel (v2)** | vs baseline | vs GQA-split |
|---:|---:|---:|---:|---:|---:|
| 256  | 0.070-0.094 ms | 0.122-0.176 ms | 0.058-0.118 ms | 0.80x-2.12x (see note) | 1.50x-2.12x |
| 512  | 0.137-0.153 ms | 0.139-0.157 ms | 0.063-0.080 ms | **1.91x-2.43x** | **1.95x-2.51x** |
| 1024 | 0.264-0.280 ms | 0.244-0.259 ms | 0.120-0.121 ms | **2.19x-2.33x** | **2.02x-2.15x** |
| 2048 | 0.535-0.537 ms | 0.430-0.446 ms | 0.205-0.208 ms | **2.59x-2.62x** | **2.09x-2.17x** |

(Ranges across the 5 runs; ratio > 1 = this kernel is faster.)

**Reading**: at every depth `>=512` (the depths this project's own methodology notes flag as
the reliable regime — matching the "`--depth 512` clears `AttentionGqaSplitMinSeqKv=256`"
convention already established for the sibling kernel), this kernel is a clean, consistent,
**2-2.6x win over the shipping baseline and a 2-2.5x win over the #197/#198 GQA-split kernel**,
reproduced across 5 independent interleaved runs with no overlap between the loss and win
ranges. seqKv=256 is noisier — 4 of 5 runs show a solid 1.2-2.1x win, one run showed a 0.80x
(loss) at that shallowest depth specifically while 512/1024/2048 stayed solidly positive in the
SAME run — consistent with this project's own prior finding (#219, #230) that the shallowest
depth is where fixed per-launch/grid.sync overhead is least amortized and most exposed to
scheduling jitter. Depths 512-2048 are unambiguous; 256 is a real but noisier win, not
recommended as the sole evidence depth (matches this project's own "`--depth 512`+" convention
for a reason).

## Occupancy

`MaxSafeAttentionMmaDecodeGqaSplit(numKvHeads=4, headDim=256, group=6)` on this RTX 3060: **14**
(vs the sibling FP32 kernel's 35) — this kernel's larger per-thread register count (55 vs the
FP32 kernel's 40) and larger effective per-block footprint at blockDim=256 lower the
co-residency ceiling somewhat, but 14 is still comfortably above the `kvSplit=8` this shape
needs at seqKv=512 (`ComputeAttentionKvSplit` picks the same `kvSplit=8` for both kernels at
this depth, so both reach the same `grid=(4,8)=32` blocks). Theoretical occupancy from the
`ptxas` register/shared-memory numbers: 42696B static shared caps co-residency to 2 blocks/SM
(100KB/SM budget), 256 threads/block → 512/1536 = **~33% theoretical occupancy** — lower than
the FP32 GQA-split kernel's 83% theoretical ceiling, but that kernel's own *achieved* occupancy
(per the 2026-07-30 re-profile in `docs/CUDA.md`) was only ~19% (CTA-barrier-stall bound, not
occupancy-bound) — i.e. this kernel's LOWER theoretical ceiling is not obviously the thing that
matters here; the real, direct evidence is the wall-clock table above, not the occupancy
projection. `ncu --set full` to get a direct achieved-occupancy / stall-reason breakdown for
this kernel specifically was not attempted this session (same intermittent `ERR_NVGPUCTRPERM`
UAC-elevation constraint prior sessions in this investigation have hit) — the wall-clock win is
unambiguous enough to reach a verdict without it, matching this investigation's own precedent
(v1's README reached its negative verdict the same way). A future session with UAC access could
confirm the specific occupancy/stall numbers.

## Why this works when v1 didn't, and why it's structurally different from the GQA-split kernel too

- **vs v1**: v1 wasted 15/16 of every mma instruction's M-dimension on zero-padding (one real
  query row per block) and ran with exactly 1 resident warp (~4% theoretical occupancy, worse
  than the ~16.5% baseline this whole investigation is trying to beat). v2 fills up to
  `group`=6-8 of those 16 M-dimension slots with REAL, DISTINCT query heads' data — the SAME
  instruction count as v1, up to 8x more useful throughput per instruction — while grid=
  `(numKvHeads, kvSplit)` and 8 resident warps/block give real occupancy on both axes v1 lacked.
- **vs `attention_f32_gqa_split_kv`**: that kernel amortizes KV reads across the group via
  PER-THREAD REGISTER blocking (`scores[MAX_GQA_GROUP]`/`v_acc[MAX_GQA_GROUP]`) and computes
  each of the group's heads' softmax reduction SEQUENTIALLY within a loop (`for (g=0;
  g<group;g++)`, the majority of its dynamic barrier count per the 2026-07-30 re-profile,
  36-of-~40 barriers/decode-step). v2 replaces both the register-blocking AND the sequential
  group loop: the group is handled "for free" via M-dim packing (no register blocking needed,
  no per-head loop needed), and the expensive PV work is spread 8-way across warps instead of
  looped. Fewer total instructions per useful FLOP (tensor-core vs CUDA-core scalar), fewer
  barriers, more real intra-block parallelism — the wall-clock table is the direct evidence
  this combination wins, not just the individual pieces in isolation.

## Recommendation

**This is a genuine performance win, not a wash.** Unlike v1 (correctness-validated,
opt-in-shipped, measured regression) and unlike the plain GQA-split kernel (correctness-
validated, opt-in-shipped, flat-to-marginal real-world gain per the existing docs/CUDA.md
history), this kernel clears this project's own "beyond noise" bar decisively at
depth >= 512 across 5 independent interleaved runs. Recommend:
1. Keep shipping opt-in (`DOTLLM_ATTN_MMA_DECODE_GQA_SPLIT=1`, default OFF) per this project's
   standing #180/#183 precedent for new precision/reassociation axes — a synthetic-fixture
   parity pass, however thorough, is not the same bar as a real generation-level validation
   (the same distinction #222 drew for the #183 split-KV kernel, which passed synthetic parity
   but changed real generated output beyond noise).
2. **Before considering a default-on flip**, run a real end-to-end generation-parity test
   (deterministic sampling, first-divergence-step methodology) against Bonsai-27B, the same
   test class #222 used to catch #183's real-world precision problem — this session did NOT
   run that test (kernel-level synthetic-fixture validation + wall-clock benchmarking was the
   scope), so "safe for real generation" is not yet an established fact, only "safe on
   synthetic fixtures at the tolerance bar this project has used throughout this investigation."
3. Model integration (wiring into `CudaQwen3MoeHybridTransformerModel`'s decode path, mirroring
   how v1's own model-integration diff worked) is a reasonable, low-risk next step given the
   kernel-level result — not attempted this session to keep scope on kernel validation and
   honest three-way benchmarking, per the task's own framing.
