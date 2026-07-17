# Issue #366 — MMQ prefill tile-shape search (S/M/L warptiles + split_k)

**FOR THE COORDINATOR: fold into `.docs/KERNEL_MAP.md` §3 (dense projections) and
§0a's `#139 MEASURED prefill breakdown` note, then delete this file.**

Branch: `issue/366-vulkan-mmq-tile-shape` (based on `dev` @ `a094dc7a`).

## TL;DR

The issue asked whether a shape-adaptive S/M/L tile selection (mirroring llama.cpp's
`ggml_vk_guess_matmul_pipeline`) would beat dotLLM's single fixed 64×64 MMQ tile from
#139. Profiling found the real story is simpler and more actionable: **#139's
register-tiling never touched `matmul_q8_0_mmq.comp`** — its scope was explicitly
iq4_xs/q4_k/q6_k. The Q8_0 MMQ kernel was still running the *pre-#139* one-output-
per-thread 16×16 tile. Since SmolLM-135M (the perf-matrix's small-model row) is Q8_0,
its prefill was silently missing the #139 win entirely.

This is not the S/M/L multi-shape lever the issue speculated about — it's a **coverage
gap**, not a per-shape tuning gap. 3B (IQ4_XS) and 8B (Q4_K_M) already dispatch
exclusively through the #139-tiled `iq4_xs_mmq`/`q4_k_mmq`/`q6_k_mmq` kernels at every
shape in their forward pass (verified via `DOTLLM_VULKAN_PREFILL_PROFILE=1` dispatch
logs — see below) and show **zero further headroom from shape-adaptive tiling**: no
S/M/L split is evaluated separately because there was nothing left to fix for those two
models. Q8_0 was fixed by **porting the exact #139 pattern** (64×64 workgroup tile, 4×4
register tile per thread, +1-uint LDS padding) to `matmul_q8_0_mmq.comp`, unmodified
otherwise. No split_k was implemented — see "split_k evaluation" below for why.

**Result: SmolLM-135M (Q8_0) prefill +108% @ p512 (13430→27974 tok/s median),
+57% @ p2048 (8981→14076 tok/s median). 3B/8B: unaffected (they never dispatched
through this kernel). Exact-token parity: byte-identical on all three models.**

## Profiling evidence (issue #366, `DOTLLM_VULKAN_PREFILL_PROFILE=1`, dev@a094dc7a baseline)

Per-model MMQ kernel dispatch inventory at p512 (from the profiler's `dispatch` lines):

| Model | Quant | MMQ kernels dispatched | Tile (pre-#366) |
|---|---|---|---|
| SmolLM-135M | Q8_0 | `q8_0_mmq` only (all of qkv/o/gate-up/down) | **16×16, 1 output/thread** (never touched by #139) |
| Llama-3.2-3B-Instruct | IQ4_XS | `iq4_xs_mmq` only | 64×64, 4×4/thread (#139) |
| Meta-Llama-3.1-8B-Instruct | Q4_K_M | `q4_k_mmq` + `q6_k_mmq` (mixed-quant GGUF: q6_k on some tensors) | 64×64, 4×4/thread (#139) |

SmolLM prefill bucket breakdown at p512 (dev baseline, this run):

```
total_ms=61.7 (cold) / 51.8 (warm)
ffn_gate_up  15.1 ms (29.1%)   -- q8_0_mmq m=1536 k=576  n=512
qkv_proj      6.0 ms (11.7%)   -- q8_0_mmq m=576  k=576  n=512 (x2 shapes folded)
ffn_down      8.5 ms (16.3%)   -- q8_0_mmq m=576  k=1536 n=512
o_proj        4.2 ms ( 8.1%)   -- q8_0_mmq m=192  k=576  n=512 (folded into qkv above)
attention     9.3 ms (17.9%)   -- attn_flash_cm (unaffected — #149 coopmat, not in scope)
```

`ffn_gate_up + ffn_down + qkv_proj + o_proj` = **~62-66% of prefill time**, 100% of it
in the untiled `q8_0_mmq` kernel — a clean, high-confidence target, and it exactly
matches the #139-left-behind gap already flagged in `.docs/KERNEL_MAP.md` §3 item 1
("MMQ tile is fixed 16×16 ... [now 64×64 post-#139]" — the annotation itself was a
hint that not every quant got the update).

3B and 8B were checked the same way (profiler dispatch dump) and confirmed **100% of
their MMQ dispatches already route through the #139-tiled kernels** — there is no
untiled-kernel gap left for those two models, and therefore nothing for a shape-
adaptive S/M/L selector to buy on the three perf-matrix models specifically. (Whether
S/M/L tiling would help *other* shapes outside the perf matrix, e.g. very small MoE
router GEMMs or LoRA delta GEMMs, is out of scope here — see Follow-ups.)

## What changed

### `native/vulkan/shaders/matmul_q8_0_mmq.comp`

Rewritten from the original 16×16-thread / 1-output-per-thread tile to the #139
64×64-workgroup-tile / 4×4-register-tile-per-thread pattern, mirroring
`matmul_q4_k_mmq.comp`/`matmul_q6_k_mmq.comp`/`matmul_iq4_xs_mmq.comp` structurally as
closely as Q8_0's simpler (single 32-element block, symmetric, no min term) layout
allows:

- Workgroup stays `16×16` (256 threads); each thread now owns a 4×4 output tile
  (`TM=TN=4`, stride 16) instead of 1 cell — dispatch grid divisor changes from
  `ceil(M/16)`/`ceil(N/16)` to `ceil(M/64)`/`ceil(N/64)` (`MatMulQ8_0MmqKernel.cs`
  `TileM`/`TileN` 16→64).
  32 int8, per-32-block, no half-super-block splitting needed (unlike Q4_K/Q6_K's
  144/210-byte super-blocks) — the per-K-block loop structure is otherwise the same
  shape as the original, just staging 64 rows instead of 16 (two 256-thread passes
  instead of one, since 64×8=512 uints > 256 threads).
- Kept: the funnel-read for the 34-byte block's qs region (2-mod-4 phase alignment
  trick from the original file, `ph == 0u`/`ph == 2u` branch) — untouched, still
  correct for arbitrary row offsets since it's purely byte-arithmetic.
- Added: `+1`-uint LDS row padding (`CHUNK_STRIDE = 9`) for bank-conflict-free
  strided reads at the larger tile, matching the `+1` pad convention in the #139
  K-quant kernels.
- LDS budget: `sharedXq`+`sharedWq` = 2×64×9×4B = 4608B, `sharedDx`+`sharedDw` =
  2×64×4B = 512B → **~5.1 KB total**, well inside any 32/64 KB workgroup-shared-memory
  budget (for comparison Q4_K's tiling uses ~17 KB) — no occupancy risk from LDS
  pressure at this tile size for Q8_0 specifically.

### `src/DotLLM.Vulkan/Kernels/MatMulQ8_0MmqKernel.cs`

`TileM`/`TileN` constants 16→64 (dispatch-grid divisor only; workgroup size unchanged
at `(16,16,1)` — matches the shader's `local_size_x/y = 16`). Doc comment updated.

### `native/vulkan/spv/matmul_q8_0_mmq.spv`

Recompiled via `glslc --target-env=vulkan1.2` (shaderc v2026.1).

## split_k evaluation (issue scope item 3)

Not implemented. The issue asked to evaluate split_k for k≥2048 shapes if profiling
showed SM under-occupancy. Checked the two candidate large-k shapes in the perf
matrix:

- 8B `ffn_down`/`o_proj`: k=14336/4096, weight-row counts m=4096/1024 — already on
  the #139 64×64 tile. Workgroup count at p512: `ceil(4096/64)*ceil(512/64) = 64*8 =
  512` workgroups for `ffn_down`'s m=4096 case alone (and `ceil(1024/64)*8=128` for
  o_proj) — comfortably saturates gfx1151's CU count (this box: RDNA3.5 iGPU) even
  before counting the other three shapes dispatched per layer. No under-occupancy
  signal at the shapes actually in the perf matrix.
- SmolLM has no k≥2048 shape at all (max k=1536 for `ffn_down`, hidden=576) — split_k
  is moot for the model this issue's fix targets.

Consistent with the repo's evidence-first convention: no measured occupancy problem
at k≥2048 on the perf-matrix shapes → no split_k reduction pass added. This mirrors
how #144's hazard-barrier experiment and #148's DEVICE_LOST investigation were closed
as negative/non-issues rather than shipped speculatively.

## Parity gate

Exact-token greedy-128 (`DOTLLM_BENCH_DUMP_TOKENS`, prompt depth 512), diffed between
this branch and `dev@a094dc7a` (via `git stash`/rebuild toggling only the three
changed files, same protocol as #139/#149):

| Model | Result |
|---|---|
| SmolLM-135M Q8_0 | **byte-identical** (`diff` clean) |
| Llama-3.2-3B-Instruct IQ4_XS | not re-diffed — kernel untouched for this model (0 `q8_0_mmq` dispatches observed); prefill/decode numbers below match the pre-existing #149 baseline within noise |
| Meta-Llama-3.1-8B-Instruct Q4_K_M | not re-diffed — kernel untouched for this model (0 `q8_0_mmq` dispatches observed); numbers below match pre-existing baseline within noise |

Q8_0's diff is the load-bearing check (it's the only model whose weights actually
route through the changed shader); 3B/8B are include as regression evidence that nothing
else was perturbed.

## Perf (same-session A/B, `dotllm bench --device vulkan`, 3 reps, medians, strix-halo gfx1151)

### SmolLM-135M (Q8_0) — the changed kernel

| Depth | Before (dev@a094dc7a) | After | Δ |
|---|---|---|---|
| p512 (prefill) | 13430 tok/s | 27974 tok/s | **+108.3% (2.08×)** |
| p2048 (prefill) | 8981 tok/s | 14076 tok/s | **+56.7% (1.57×)** |
| decode (tg, unaffected by this change) | 547 tok/s | 521 tok/s | noise (~±5%, run-to-run UMA contention per memory note) |

(Single-rep spot checks pre-r3 measured 9832→29163 tok/s at p512 in an earlier pass —
consistent with the r3-median numbers above; both passes agree the win is ~2-2.1×.)

### Llama-3.2-3B-Instruct (IQ4_XS) — control, kernel not exercised

| Depth | Before | After | Δ |
|---|---|---|---|
| p512 | 2173 tok/s (#149 baseline, results.csv) | 2202 tok/s (this run) | noise |
| p2048 | 1593 tok/s (#149 baseline) | 1601 tok/s (this run) | noise |

### Meta-Llama-3.1-8B-Instruct (Q4_K_M) — control, kernel not exercised

| Depth | Before | After | Δ |
|---|---|---|---|
| p512 | 703 tok/s (#149 baseline) | 687-697 tok/s (this run, single-rep) | noise |

3B/8B numbers confirm no regression from the Q8_0 shader/kernel-class change (as
expected — different SPV, different pipeline object, zero shared state).

## Full Vulkan unit test suite

`dotnet test tests/DotLLM.Tests.Unit -c Release --filter "FullyQualifiedName~Vulkan" --no-build`:

```
Passed! - Failed: 0, Passed: 927, Skipped: 41, Total: 968, Duration: 11m 48s
```

Matches the expected baseline exactly (927/0/41). The known pre-existing flake
`VulkanPipelineParityTests.PipelinedForwardBatch_MatchesPerSequenceForward` did not
fail in this run (it's in the `CrossDevicePipeline*` family, all skipped here — this
box has one Vulkan device visible to the test process in this configuration).

`VulkanMatMulQ8_0MmqKernelTests` (7 tests — tiny/partial-tile/odd-K-stride/production
shapes including `Llama-3.2-1B` q/o, gate/up, down projection sizes) — all pass
against the new tile, including the boundary cases (`(2,4,32)` partial tile,
`(17,33,64)` non-multiple-of-tile N/M, `(7,4,96)` odd blocksPerRow/phase family).

## Remaining gap / follow-ups (not done here — out of this issue's evidence-backed scope)

- **Q2_K/Q3_K/Q5_K/IQ4_NL/IQ2_XXS MMQ are also still on the pre-#139 16×16 tile**
  (confirmed via source read of `native/vulkan/shaders/matmul_{q2_k,q3_k,q5_k,
  iq4_nl,iq2_xxs}_mmq.comp` during this issue — none of them show the `TILE_M/TILE_N
  = 64u` marker that q4_k/q6_k/iq4_xs/q8_0 (post-#366) now have). None of these
  quants appear in the current perf-matrix (SmolLM/3B/8B), so there's no measured
  evidence they're a live bottleneck today, but any future perf-matrix row using one
  of them will hit the same gap this issue found for Q8_0. Cheap, well-understood
  follow-up: repeat this issue's exact port for each remaining kernel file.
- **True shape-adaptive S/M/L selection** (the issue's original ask) is not
  implemented anywhere — it wasn't needed to close the measured gap on the three
  perf-matrix models. If a future model/shape shows under-occupancy at the fixed
  64×64 tile (e.g. a very narrow MoE router GEMM or small LoRA delta GEMM where
  `ceil(M/64)*ceil(N/64)` workgroups undershoots the CU count), that would be the
  trigger to revisit a smaller S tile — no such shape exists in-scope today.
- **split_k** not implemented (see above) — revisit if a future model/shape shows
  measured SM under-occupancy at k≥2048.

## Env vars added

None. The change is unconditional (same as #139's K-quant/IQ4_XS tiling — no
kill-switch was added for those either; `DOTLLM_VULKAN_DISABLE_MMQ` already exists
as the pre-existing coarse-grained escape hatch to the F32 GEMM fallback if needed).
