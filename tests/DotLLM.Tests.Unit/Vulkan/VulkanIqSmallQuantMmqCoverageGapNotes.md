# Research note: IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S prefill MMQ coverage gap

Scoping/feasibility investigation only. No kernel or GPU-requiring test written.
Ref: `.docs/KERNEL_MAP.md` §3 item 5 ("K-quant coverage gap: IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S
prefill = dequant→F32 scalar GEMM (16×16), no MMQ at all").

## 1. Coverage gap confirmed

`Glob src/DotLLM.Vulkan/Kernels/*Mmq*` returns exactly 9 files:

```
MatMulIq2XxsMmqKernel.cs   MatMulIq4NlMmqKernel.cs   MatMulIq4XsMmqKernel.cs
MatMulQ2KMmqKernel.cs      MatMulQ3KMmqKernel.cs     MatMulQ4KMmqKernel.cs
MatMulQ5KMmqKernel.cs      MatMulQ6KMmqKernel.cs     MatMulQ8_0MmqKernel.cs
```

No `MatMulIq2SMmqKernel.cs`, `MatMulIq2XsMmqKernel.cs`, `MatMulIq3SMmqKernel.cs`,
`MatMulIq3XxsMmqKernel.cs`, or `MatMulIq1SMmqKernel.cs` exist. For all 5 target types only
`*GemmF32Kernel.cs` / `*GemvF32Kernel.cs` (prefill/decode F32 fallback) and `*MmvqKernel.cs`
(decode-only dp4a) exist. Confirmed via directory listing:

- IQ2 family: `MatMulIq2SGemmF32Kernel.cs`, `MatMulIq2SGemvF32Kernel.cs`, `MatMulIq2SMmvqKernel.cs`,
  `MatMulIq2XsGemmF32Kernel.cs`, `MatMulIq2XsGemvF32Kernel.cs`, `MatMulIq2XsMmvqKernel.cs` — no MMQ.
  (IQ2_XXS is the outlier: it *does* have `MatMulIq2XxsMmqKernel.cs`.)
- IQ3 family: `MatMulIq3SGemmF32Kernel.cs`, `MatMulIq3SGemvF32Kernel.cs`, `MatMulIq3SMmvqKernel.cs`,
  `MatMulIq3XxsGemmF32Kernel.cs`, `MatMulIq3XxsGemvF32Kernel.cs`, `MatMulIq3XxsMmvqKernel.cs` — no MMQ.
- IQ1 family: `MatMulIq1SGemmF32Kernel.cs`, `MatMulIq1SGemvF32Kernel.cs`, `MatMulIq1SMmvqKernel.cs` — no MMQ.

Shader side matches: `Glob native/vulkan/shaders/*mmq*` returns only `matmul_iq2_xxs_mmq.comp`,
`matmul_iq4_xs_mmq.comp`, `matmul_iq4_nl_mmq.comp`, plus the K-quant/Q8_0 MMQ shaders
(`matmul_q{2,3,4,5,6}_k_mmq.comp`, `matmul_q8_0_mmq.comp`). No `matmul_iq2_xs_mmq.comp`,
`matmul_iq2_s_mmq.comp`, `matmul_iq3_xxs_mmq.comp`, `matmul_iq3_s_mmq.comp`, or
`matmul_iq1_s_mmq.comp`. Gap is real on both the C# kernel-class side and the `.comp` shader side.

## 2. Codebook infra is already shared — porting looks mechanical

`src/DotLLM.Vulkan/Kernels/Iq2Codebooks.cs`, `Iq3Codebooks.cs`, `Iq1Codebooks.cs` each own
GPU-resident grid/sign SSBOs **shared across their whole family, not per-quant-type**:

- `Iq2Codebooks`: uploads `iq2xxs_grid` (256×8B), `iq2xs_grid` (512×8B), `iq2s_grid` (1024×8B),
  and `ksigns_iq2xs` (128B) — i.e. IQ2_XXS (which already has MMQ), IQ2_XS, and IQ2_S all share
  one `Iq2Codebooks` instance and the same `ksigns` table.
- `Iq3Codebooks`: uploads `iq3xxs_grid` (256×4B), `iq3s_grid` (512×4B), and the IQ2-family
  `ksigns_iq2xs` table (reused, not duplicated) — IQ3_XXS and IQ3_S share one instance.
- `Iq1Codebooks`: uploads the 2048-entry `iq1s_grid` (uint64 ternary-lane packing).

Because IQ2_XXS's MMQ kernel already binds `Iq2XxsGrid` + `Ksigns` from this same shared
`Iq2Codebooks` object (see doc comment in `Iq2Codebooks.cs`: "Uploaded once and bound as readonly
SSBOs alongside the per-row weight bytes" — shared, not per-kernel), adding IQ2_XS/IQ2_S MMQ
kernels means binding `Iq2XsGrid`/`Iq2SGrid` from the *already-resident* buffer — no new host-side
upload/allocation plumbing needed. Same story for IQ3_S off `Iq3Codebooks.Iq3SGrid` and IQ1_S off
`Iq1Codebooks.Iq1SGrid`.

Read `native/vulkan/shaders/matmul_iq2_xxs_mmq.comp` (the closest existing template — grid-codebook
dp4a MMQ, TILE_M/TILE_N=16, shared-memory `sharedWq`/`sharedXq`/`sharedWdb`/`sharedXd`, per-sub-block
scale reconstruction via `read4Bytes`/`readHalf` helpers) confirms the pattern: each super-block is
decoded into shared memory as sign·grid int8 lanes + a per-sub-block `db` scale, then dp4a'd against
the int8-quantized activations. IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S already have this exact decode
arithmetic proven correct in their respective `*_mmvq.comp` (decode) shaders (per KERNEL_MAP: "The
seqLen>1 analogue of matmul_iq2_xxs_mmvq.comp" is literally how the XXS MMQ shader's own header
describes itself). So each new MMQ shader is: take the existing MMVQ per-block decode math (already
validated) + the IQ2_XXS MMQ shader's tiling/shared-memory/dispatch scaffold, splice together. This
is the same "port a proven pattern to a new quant" mechanical lift the #372 Q5_K indexed-MoE kernel
agent report references — not a from-scratch design problem. CPU-side reference dequant logic for
block-structure cross-check lives in `src/DotLLM.Cpu/Kernels/Dequantize.cs` (`Iq2XsGrid`, `Iq2SGrid`,
`Iq3XxsGrid`, `Iq3SGrid`, `Iq1SGrid`, `KsignsIq2Xs` — the exact same tables the GPU codebook buffers
are sourced from, per the `Iq*Codebooks.cs` doc comments).

## 3. Driver-fault risk — was the blocker, now explicitly de-risked (as of 2026-06-29)

IQ2_XXS's existing MMQ kernel is disabled by default (`DOTLLM_VULKAN_ENABLE_IQ2XXS_MMQ`, opt-in)
because of a confirmed **AMD gfx1151 LLPC shader-compiler miscompile** at scale — GPU fault,
per-submit-cumulative (#344; user memory `vulkan-iq2xxs-mmq-driver-fault.md`). This was the
headline risk for extending MMQ to more IQ types.

However, `.docs/HANDOFF.md` records this was already investigated and **resolved as a
scoping question**, not merely flagged:

- Item 4 (DONE 2026-06-29): a cross-vendor test (`VulkanMatMulIq2XxsMmqKernelTests
  .Mmq_LargeShape_NonAmdCrossCheck`, `8×2048×2048`, gated on
  `DOTLLM_IQ2XXS_LARGE_CROSSCHECK=1`) ran on the Framework box and **passed on both RTX 3060
  (NVIDIA) and Intel Arc**. The fault reproduces **only** on AMD gfx1151. This confirms the
  shader logic is correct and the bug is AMD's compiler, not a structural issue with
  grid-codebook MMQ shaders.
- Item 3 (still open, not yet built): "IQ-family prefill MMQ (IQ3/IQ2_XS/IQ2_S/IQ1_S) — the
  gfx1151 fault is now confirmed AMD-specific, so this is unblocked on NVIDIA/Intel; still gate
  opt-in + keep the F32 GEMM default on AMD until the driver is fixed."

So the recorded plan is exactly what this investigation would recommend: build these 5 kernels,
ship them gated opt-in (mirroring `DOTLLM_VULKAN_ENABLE_IQ2XXS_MMQ`), default to F32 GEMM on AMD,
and let non-AMD Vulkan users (NVIDIA/Intel) get the MMQ prefill speedup by default or via the
same opt-in flag. **Risk is real but bounded and already has a working mitigation pattern** — it
does not block building the kernels, only blocks flipping them on by default on gfx1151 (this
project's own primary dev box).

Also worth noting: user memory records a *second*, distinct suspected-same-class device-lost
fault in the MoE indexed-matmul path (#373, `expandedRows > 16`), separate from the dense IQ2_XXS
MMQ fault — both currently bucketed as "AMD gfx1151 LLPC miscompile class" but with only the IQ2_XXS
one cross-vendor-confirmed. Any new IQ MMQ kernel should get the same NVIDIA/Intel cross-check
treatment before being trusted, even though it's opt-in.

## 4. Real-world usage: none of the 5 target quant types appear in any tested model

`benchmarks/perf-matrix/results.csv` — grepped all `IQ[0-9]_[A-Z]+`-shaped tokens across every row:
only **`IQ2_XXS`**, **`IQ4_NL`**, **`IQ4_XS`** appear (12 total IQ-quant rows). **Zero rows** for
IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, or IQ1_S. `.docs/HANDOFF.md` has no mentions of these 5 quant
strings either (checked directly — no matches). No evidence any perf-matrix model, benchmark, or
HANDOFF-tracked real-model validation run has exercised these 5 quant types on this project at all.

These are also the *most aggressively lossy* GGUF quant types in the llama.cpp quant family
(sub-2.5-bit average, mostly used for extreme-memory-constrained huge models, e.g. >70B on
consumer hardware) — a niche use case relative to the Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS quants that
dominate this project's tested model set.

## 5. Priority recommendation: LOW

- (a) Coverage gap is real and confirmed.
- (b) Codebook infra is fully shared already — this is a mechanical "port the MMVQ decode math +
  IQ2_XXS MMQ shader scaffold" job per type (5 kernels + 5 shaders + test wiring), not a design
  problem. Feasibility is high.
- (c) The driver-fault risk that would have been the strongest reason to deprioritize this has
  already been narrowed to "AMD-specific, gate opt-in" by the existing #344/#365 cross-vendor
  work — it's a known, bounded, already-templated risk, not an open unknown.
- (d) **No model in this project's tested/benchmarked set uses any of these 5 quant types** —
  this is the dominant factor. Even a fully mechanical, zero-risk port produces zero measured
  benefit until a real IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S model shows up in the perf matrix.
- (e) This note: `tests/DotLLM.Tests.Unit/Vulkan/VulkanIqSmallQuantMmqCoverageGapNotes.md`.

**Recommendation: LOW priority.** The lift is cheap and low-risk *if* someone needs it, but
nothing currently tested needs it. Don't schedule this ahead of any lever that touches a quant
type actually present in `benchmarks/perf-matrix/results.csv` (Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS/IQ4_NL/
IQ2_XXS) or a real HANDOFF-tracked model. Revisit if/when a specific IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/
IQ1_S model is added to the test/benchmark set (e.g. a very large model quantized that low to fit
this box's UMA budget) — at that point this note's §2 sizing (mechanical port, ~5 kernel/shader
pairs, shared codebook infra already resident) should make it a fast follow.
