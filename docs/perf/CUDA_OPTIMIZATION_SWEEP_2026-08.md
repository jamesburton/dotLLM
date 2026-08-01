# CUDA Optimization Sweep — 2026-07-31 / 2026-08-01

Progress note for the "optimise for CUDA, particularly BitNet" thrust. Covers: reconciling a
large backlog of unmerged branches, closing stale issues whose bodies no longer matched
reality, a new BitNet-ternary MoE CUDA port, and a real `ncu`-driven fix to the prefill
flash-attention kernel. Written up per request so progress survives a context reset.

## Status summary

| Item | State |
|---|---|
| Branch-reconciliation sweep (6 orphan branches + #237) | DONE, merged to `dev`, pushed |
| BitNet-ternary (I2_S) MoE CUDA port (#246) | DONE, merged to `dev`, pushed |
| G-flash prefill-attention bank-conflict fix (#248) | DONE, merged to `dev`, pushed |
| Stale-issue verification sweep (#60, #61, #70, #229) | DONE, all confirmed already-shipped, closed |
| MSBuild per-file CUDA arch-override gap | DONE, fixed in `DotLLM.Cuda.csproj` |
| CUDA 13.1 install | Confirmed: does NOT remove the MSVC-compat workaround (still C1189 without `-allow-unsupported-compiler`); no new capability for this sm_86 GPU. See "Toolchain note" below. |

## 1. Branch-reconciliation sweep

Six branches sat unmerged on `origin` with real, tested work: #232 (CPU I2_S 4x4
register-blocked GEMM), #233/#236 (Vulkan PQ2_0 register-blocked + coopmat GEMM), #239/#241
(Vulkan I2_S coopmat+wave32 production dispatch), #402 (CPU balanced work partitioning), #413
(BPE tokenizer allocation fix). All merged, with two real merge conflicts resolved (not just
auto-merged): `MatMul.cs`'s `GemmTiledQ8Worker` (kept dev's newer 2D partition scheme) and
`Gpt2TiktokenEncoding.cs`'s `ByteMapIntoSpan` (dev's pre-tokenizer pipeline had already
restructured the function; rethreaded #413's one-queue-per-call fix into the current
`Encode()` call sites instead). Also fixed a real crash (`I2SRegisterBlockedGemmTests` missing
an `Avx2.IsSupported` guard, crashing the whole test host on this box's non-AVX2 CPU) and
verified/closed #237 (BPE pre-tokenizer contraction-splitting bug) against the real production
path via `dotllm perplexity --dump-tokens`.

Along the way, reconciled **four separate rounds** of concurrent pushes from another session
working `dev` in parallel — including one genuine semantic conflict (this session's AMD
coopmat32+wave32 I2_S dispatch vs. a concurrent session's Intel Xe-LPG F16-weight-tile
dispatch, both touching `MatMulI2SGemmF32Kernel.cs`'s `Create()` entry point) resolved by
composing both vendor-specific tracks rather than picking one.

## 2. BitNet-ternary (I2_S) MoE CUDA port — issue #246

`CudaMoeFfn.cs` already had a full, tested standard-SwiGLU MoE forward on GPU (the
`docs/CUDA.md` note calling this "not yet ported" was stale, corrected this session) — but zero
I2_S/PQ2_0 branches. Added `CudaMoeFfn.ForwardBitNetI2S`, porting
`MoeSwiGluMlp.BitNet.cs`'s CPU forward (relu² gate, per-expert FFN Sub-LN before down_proj,
ternary I2_S experts, not SwiGLU) to GPU.

**Composed from already-proven pieces rather than new kernel design**: the existing
`LaunchI2_SGemvF32In` GEMV kernel (one launch per assigned row — no batched I2_S GEMM this
pass, fine since decode is batch=1 per expert), the existing MoE routing/bucketing dispatch,
one new `relu2glu_f32` kernel, and the existing `rmsnorm_f32` kernel reused unmodified for the
per-expert Sub-LN. The per-expert-scale mismatch (CPU bundle keeps scales in a separate array;
the GEMV kernel expects one trailing scale baked into its buffer) turned out to need no
device-side repack kernel at all — a load-time two-copy HtoD upload (payload + that expert's
own scale into one pre-sized buffer) suffices, since it's attaching a scalar to an
already-contiguous slice, not reordering interleaved bytes like PQ2_0's repack problem was.

Verified: CPU-reference parity tests (decode top-1/top-2+bias, prefill multi-token-per-expert,
skip-expert exact-zero) at 1e-3 tolerance, all pass. Full CUDA suite 398-400/400+ passed, 0
failed, independently re-run (not just trusted from the implementing agent's self-report).

**Known limitation**: no real BitNet-MoE checkpoint exists anywhere locally to validate
end-to-end — identity-MoTE (issue #117) is this project's own experimental architecture, not a
public checkpoint family. Synthetic CPU-reference parity is the validation ceiling until one is
trained/exported.

## 3. G-flash prefill-attention bank-conflict fix — issue #248

`ncu --set full --kernel-name attention_flash_mma_f16` at s=1024 (real Llama-3.2-1B shape,
`.perf-runs/flash_mma_s1024_details.txt`) found the kernel's own header's "next step" guess
(double-buffered K/V loads for latency hiding) was pointing at the wrong bottleneck. Real
numbers:

| Metric | Value |
|---|---|
| DRAM Throughput | 2.04% (rules out bandwidth) |
| Memory Throughput (L1/TEX) | 86.35% |
| Compute (SM) Throughput | 26.87% |
| Shared-load bank conflicts | **9.7-way**, 84% of load wavefronts, Est. Speedup 77.4% |
| Shared-store bank conflicts | **4.0-way**, 72% of store wavefronts, Est. Speedup 66.4% |

Root cause: `HEAD_DIM` (64 halves = 128 bytes = exactly 32 banks) and `KV_TILE` (16
halves/floats = 8/16 banks) both divide the 32-bank cycle evenly, so every row of
`sK`/`sQ`/`sVt`/`sP`/`sScore` lands in the same bank phase as the row before it.

**Fix**: pad every shared-array row stride, mirroring this codebase's I2_S GEMM kernel
precedent (128-word stride → 129). Real subtlety hit on the way: `ldmatrix` requires
16-byte-aligned per-thread row addresses. A naive +1-element pad (the exact I2_S technique)
immediately produced **CUDA error 716 "misaligned address"** on real hardware, because
`(lane & 15) * stride` must stay a multiple of 8 halfwords for every lane value — only true if
the stride itself is a multiple of 8. Fixed: the four `ldmatrix`-accessed arrays
(`sK`/`sQ`/`sVt`/`sP`) pad by **+8 elements**; `sScore` (float, plain per-lane scalar access,
no `ldmatrix`) safely pads by the naive +1.

**Real measured result** (`CudaTensorCoreAttentionParityTests.CompletePathTimingVsAttentionF16`,
interleaved min, RTX 3060, Llama-3.2-1B shape):

| seq | G3/flash pre-fix | G3/flash post-fix | floor/flash pre-fix (implied) | floor/flash post-fix |
|---|---|---|---|---|
| 1024 | 1.3-1.69x | **3.66x** | ~3x | 1.75x |
| 2048 | 1.3-1.69x | **3.49x** | ~3x | 1.73x |
| 4096 | 1.3-1.69x | **2.91x** | ~3x | 1.41x |

The kernel moved from ~33% to ~58-70% of cuBLAS's raw per-FLOP efficiency at this shape,
**purely from the shared-memory access-pattern fix, zero algorithmic change.**

Correctness: `FlashMmaPath_MatchesAttentionF16` clean at s=512/1024/2048 (5e-3 tolerance), full
CUDA suite (400+ tests) green, zero regressions.

### Bonus: MSBuild arch-override gap (found + fixed)

This file (and its decode sibling `attention_flash_mma_decode_gqa_split.cu`) needs
`compute_86` for its `mma.sync` instructions. `native/build_ptx.bat` has a per-file `ARCH_86`
override for exactly these two files — but the MSBuild `CompileCudaPtx` target (used by plain
`dotnet build`) had no equivalent mechanism, always compiling at the global `$(CudaArch)`
default (`compute_75`). A plain `dotnet build` after editing either file silently regenerated
**wrong-architecture PTX** — this does not fail the build; it fails **silently at runtime**
(`CudaKernels.cs`'s best-effort module load catches the resulting `CudaException` and just
leaves the capability flag false, falling back to G3 with zero visible symptom). Root-caused
via `ptxas -arch=sm_86` on the regenerated PTX directly (`Feature '.m16n8k16' requires .target
sm_80 or higher` — the PTX header said `.target sm_75`).

**Fixed**: added the same per-file `%(CudaKernel.Arch)` metadata override to
`DotLLM.Cuda.csproj`, so plain `dotnet build` now matches `build_ptx.bat`'s output. Verified by
touching the source and rebuilding via plain `dotnet build` alone (no `build_ptx.bat`): PTX
`.target` line now correctly reads `sm_86`.

**Anyone editing either of these two files going forward**: the csproj fix should now make
plain `dotnet build` produce correct output, but if the flash kernel silently stops dispatching
after an edit, check the PTX header's `.target` line first (`sm_86`, not `sm_75`) before
assuming a logic bug.

## 4. Stale-issue verification sweep

The whole "local-platform-opt" CUDA prefill campaign (G1 COMPUTE_16F prefill GEMM, G3
cuBLAS+softmax tensor-core attention, G-flash hand-fused mma.sync attention) turned out to
already be fully shipped and default-on for GeForce Ampere — issues #60, #61, #70 all had
stale bodies describing prototype/not-wired-in states that no longer matched the code. Verified
each against the actual source before closing (not just assumed):

- **#60 (G1)**: `CudaGemm.Use16FCompute`/`ConfigureDefault`, wired into `CudaTransformerModel`
  at load, plus its exact 1%-perplexity-move quality gate
  (`tests/DotLLM.Tests.Integration/Cuda/CudaGemm16FPerplexityTests.cs`) already exists.
- **#61 (G3)**: `CudaG3Attention`, default-on for GeForce Ampere/Turing, already wired in.
- **#70 (G-flash)**: `CudaFlashAttention`, default-on for GeForce Ampere, dispatched ahead of G3
  for long-context prefill, already wired in — and now further tuned via #248 above.
- **#229 (Vulkan I2_S GEMM ladder, Xe-LPG)**: already concluded (F16 weight-tile shipped, int8
  measured no further gain, production fallback chain in place) — merged commits existed but
  the issue itself wasn't closed.

This matches a recurring pattern across this whole project's history (see
`[[dotllm-issue-cleanup-sweep]]`): work gets done and merged without the corresponding issue
ever being closed, so the issue tracker understates real progress. Worth another general sweep
of the open-issue backlog if a slow moment comes up.

## 5. Toolchain note: CUDA 13.1

Confirmed installed and functional on this box (`nvcc: release 13.1, V13.1.80`/`V13.1.115`
depending on which sub-tool). Empirically tested: **still requires
`-allow-unsupported-compiler`/`DOTLLM_NVCC_ALLOW_UNSUPPORTED_COMPILER=1`** with this box's MSVC
14.50 (VS "18") — CUDA 13.1's `host_config.h` caps at VS2022-supported MSVC same as 12.8.1, so
this doesn't remove the existing build-time workaround. No new GPU-architecture capability
matters for this box's RTX 3060 (sm_86, Ampere — CUDA 13.1's headline additions target newer
architectures this hardware doesn't have).

**Real risk worth flagging**: `CUDA_PATH` now resolves to `C:\Program Files\NVIDIA GPU
Computing Toolkit\CUDA\v13.1` by default on this box, one version ahead of what the committed
`native/ptx/*.ptx` tree was built with (12.8.93, `.version 8.7`). Running the *full*
`native/build_ptx.bat` without explicitly overriding `CUDA_PATH` regenerates the **entire PTX
tree** at `.version 9.1` — a pure toolchain-version diff across every file, not a real source
change, that would bloat any future commit if not caught. Hit this directly this session
(build_ptx.bat swept ~50 unrelated `.ptx` files; reverted all but the one file actually being
worked on, then regenerated that one file alone with `CUDA_PATH` explicitly pointed at
`E:\CUDA_v12.8.1` to keep the diff minimal and version-consistent).

**Recommendation**: either (a) always explicitly set `CUDA_PATH=E:\CUDA_v12.8.1` before running
`build_ptx.bat` on this box until a deliberate, single, verified migration to 13.1 is done for
the whole tree, or (b) do that migration deliberately (rebuild + fully retest every kernel,
one commit, not an incidental side effect of an unrelated fix). Not done here — out of scope
for this session's actual goal.

## 6. Where to look next

Genuinely open CUDA-relevant items remaining, roughly in priority order:

1. **A fresh `ncu` capture confirming the #248 bank-conflict metric itself dropped**, not just
   the wall-clock win (the wall-clock evidence is already decisive and reproducible without
   this, but direct confirmation of the mechanism would be a nice close-out). Needs elevated
   `ncu` — see `[[ncu-elevation-workflow]]` / `[[gflash-bank-conflict-fix]]` for the working
   command pattern.
2. **G-flash tuning has more headroom** per the corrected floor/flash ratio (still 1.4-1.75x
   off the theoretical ceiling, down from ~3x) — a fresh `ncu` pass on the fixed kernel would
   show what the NEW dominant bottleneck is (may no longer be memory-bound at all; could be
   genuinely compute/MMA-utilization-bound now, in which case the remaining lever is different
   — e.g. warp/tile shape, not shared-memory layout).
3. **No batched/grouped I2_S GEMM for BitNet-MoE prefill** (#246's known limitation) — real
   perf concern only if BitNet-MoE prefill becomes a hot path; no current checkpoint exercises
   it, so low urgency.
4. **BitNet-ternary MoE end-to-end validation** blocked on a real checkpoint not existing yet
   — training/exporting one (via the identity-MoTE pipeline, issue #117) would unlock real
   generation-parity testing beyond synthetic CPU-reference parity.
5. Issue #200 (native paged-KV decode kernel) and #125 (Q4_K MMVQ coalesced kernel) remain
   open but are lower-confidence / bigger scope (paged KV-cache doesn't exist on CUDA at all
   yet; MMVQ is parked pending a Kaggle T4 A/B per prior session notes) — not pursued this
   session, flagged for explicit prioritization if wanted.
