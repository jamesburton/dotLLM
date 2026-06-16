# Local-hardware operator & kernel-optimisation analysis (2026-06-15)

Scope per request: scan operators supported on the **local** box, find dotnet-runtime support gaps, and
identify kernel optimisations to test here to improve known/supported models. Strix is explicitly out of scope
for this pass. Two targets: **Intel Core Ultra 7 155H (Meteor Lake)** CPU and **NVIDIA RTX 3060 (Ampere
CC 8.6)** GPU.

Ground truth from `.claude/strix/isaprobe-full` (reflection over every `System.Runtime.Intrinsics.X86` class +
raw CPUID), `nvidia-smi`, and a kernel-surface survey.

---

## 1. Device inventory

### CPU — Intel Core Ultra 7 155H (Meteor Lake), 16 logical, .NET 10.0.9
| Feature | Hardware (CPUID) | .NET exposes | dotLLM uses |
|---|---|---|---|
| AVX2 / FMA | ✅ | ✅ | ✅ (dominant: ~1200 Vector256 / 560 Avx2 sites) |
| **AVX-VNNI (256, VEX `vpdpbusd`)** | ✅ | ✅ `AvxVnni` | ❌ **NOT used** (0 sites) |
| GFNI (256) | ✅ | ✅ `Gfni.V256` | ❌ |
| VAES / VPCLMULQDQ (256) | ✅ | ✅ `Aes` / `Pclmulqdq.V256` | ❌ (n/a to inference) |
| SHA | ✅ | ❌ no .NET class | — (n/a to inference) |
| AVX-512 (any) | ❌ absent | (false) | — |
| AVX10 / AVX-VNNI-INT8 / -INT16 / AVX-IFMA | ❌ absent (CPUID EDX[4]/[10], EAX[23] = 0) | (false) | — |

### GPU — NVIDIA RTX 3060, Ampere CC 8.6, 12 GB, driver 591.86, CUDA 13.1
- 3rd-gen tensor cores: FP16 / BF16 / TF32 / **INT8** (`mma.sync m16n8k32`). **No FP8** (Ada/Hopper only).
- `__dp4a` INT8 dot on CUDA cores (sm_61+). `cp.async`. ~112 GB/s… (12 GB GDDR6, 360 GB/s BW).

---

## 2. dotnet runtime support gaps — essentially none that matter

**There are no material dotnet-runtime exposure gaps on this box for inference.** Everything useful is exposed:
AVX2, AVX-VNNI(256), FMA, GFNI, VAES, PCLMULQDQ, BMI1/2, POPCNT, LZCNT, AES.

- The **only** hardware feature with no .NET class is **SHA** — irrelevant to LLM inference.
- **AVX-512, AVX10v1/v2, AVX-VNNI-INT8/INT16, AVX-IFMA, AVX512-BF16/FP16** all report false because they are
  **hardware-absent on Meteor Lake** (verified against raw CPUID), **not** runtime gaps. (Intel disabled
  AVX-512 on consumer Core since Alder Lake; VNNI-INT8/AVX10 debut on Arrow/Lunar Lake+.)
- CUDA toolkit 13.1 / driver 591.86 are current; no runtime/toolkit gap.

**The opportunity is utilisation of already-exposed ISA, not missing exposure.** (Contrast the Strix work, where
the gaps were genuinely runtime-side: `Avx512Bf16` / `AvxVnni.V512` needed dev-runtime builds.)

---

## 3. Current kernel utilisation vs. what the hardware offers

**CPU Q8 integer matmul (`src/DotLLM.Cpu/Kernels/MatMul.cs`)** uses the AVX2 3-instruction dot:
`vpmaddubsw` (`MultiplyAddAdjacent` byte→i16) → `vpmaddwd`-with-ones (i16→i32) → accumulate. **AVX-VNNI's
single `vpdpbusd`** (byte→i32 dot-accumulate in one op) is not used anywhere. This is exactly the lever that
PR #321/#323 measured at **1.3–1.6× on Meteor Lake** (and only parity on Zen5, whose maddubs throughput is
strong — so it's specifically an Intel-side win). That code lives on branch `issue/321-q8-vnni-outer-product`.

**GPU:** decode GEMV is `quantized_gemv_mmq.cu` using `__dp4a` (good — GEMV is memory-bound, tensor cores
wouldn't help). Prefill GEMM is `CudaGemm.LinearF16`: **dequant quantized weights → FP16 scratch → `cublasGemmEx`
FP16 in/out with `CUBLAS_COMPUTE_32F`**. So prefill already hits FP16 tensor cores via cuBLAS — but (a) on a
**GeForce** Ampere card, FP16-TC with FP32-accumulate runs at ~half the FP16-accumulate rate (documented consumer
throttle), and (b) the INT8 tensor cores (2× FP16 TC rate) and the avoid-the-FP16-dequant-detour win are unused.
No hand-written `mma.sync`/`wmma`; PTX build floor is `compute_75` (Turing) — fine for dp4a, but cannot express
sm_80+ INT8 `mma` shapes (only relevant if we ever hand-write tensor-core kernels).

---

## 4. Optimisations to test — cheapest first, profile-gated

> **Discipline (same as the CPU decode work): profile before building.** On the 3060, first measure where time
> goes — prefill GEMM vs attention vs decode-GEMV vs the per-forward dequant. The GEMM wins below are
> **prefill-only** (decode-GEMV is already dp4a + memory-bound); size that fraction (long-prompt / batched) before
> investing. All accuracy-affecting changes gate on the existing perplexity harness.

### CPU (Meteor Lake)
**C1 — AVX-VNNI(256) `vpdpbusd` for the Q8 integer dot. (highest-confidence local win)**
Port the `issue/321-q8-vnni-outer-product` path rather than reimplementing; dispatch it under
`AvxVnni.IsSupported`, keeping the AVX2 maddubs path as fallback. Frame as a **prefill** win (compute-bound dot,
3 ops → 1); do not promise decode gains (n=1 GEMV is bandwidth-bound). Expected ~1.3–1.6× on the Q8 prefill
matmul on this uarch. Validate byte-near vs the AVX2/scalar oracle + perplexity.

**C2 — GFNI(256) `GF2P8AffineQB` for low-bit nibble unpack (speculative, cheap to spike).**
Q4/Q2 dequant currently unpacks nibbles with shifts/masks; a GF2P8 affine transform can spread 4-bit nibbles to
bytes in one op. Untested; worth a microbench spike against the current unpack before committing.

### GPU (RTX 3060) — laddered, do NOT lead with hand-written mma
**G0 — Profile prefill vs decode vs dequant share.** (gates everything below)

**G1 — One-line: `CUBLAS_COMPUTE_32F` → `CUBLAS_COMPUTE_16F` in `CudaGemm.LinearF16` (CudaGemm.cs:38).**
On GeForce Ampere this can ~2× the prefill GEMM (lifts the consumer FP32-accumulate throttle) at some accuracy
cost. Trivial to try; gate on perplexity. May capture much of the prize with near-zero effort.

**G2 — INT8 prefill GEMM via `cublasGemmEx` (`CUDA_R_8I` / `CUBLAS_COMPUTE_32I`) or cuBLASLt IMMA.**
2× the FP16-TC rate *and* halves weight bytes (skips the FP16 dequant detour — quantize activations to INT8,
keep weights INT8). Consistent with the existing cuBLAS P/Invoke design (not a new dependency). Budget for IMMA
layout/alignment constraints (leading dims multiple of 4/16; cuBLASLt is the cleaner surface). Gate on perplexity.

**G3 — Hand-written INT8 `mma.sync` (m16n8k32) fused dequant-GEMM + `compute_80/86` PTX target.**
Only if G2 wins *and* cuBLAS library overhead is shown to be the limiter. This is the MMQ-on-tensor-cores path
(thousands of lines, llama.cpp-proven on Ampere). The `compute_75` PTX floor would need raising for this path
only — don't frame it as a primary gap.

### Model-fixture mapping (which optimisation helps which "known/supported" model)
- Dense GEMM prefill wins (C1, G1, G2) scale with model size → most visible on **Llama-3.2-3B** (cached locally),
  then Llama-3.2-1B / Bielik-1.5B; marginal on SmolLM-135M.
- **DeepSeek-V2-Lite** (cached) is MoE → the relevant GPU path is `moe_ffn.cu` / `moe_grouped_gemv.cu`, not the
  dense cuBLAS GEMM; treat separately.

---

## 5. Bottom line
- **Runtime gaps:** none material on this box; the actionable gap is *utilisation*.
- **Surest near-term win:** C1 (AVX-VNNI Q8 dot, port #321) on CPU; **G1 (one-line compute-type)** on GPU — both
  cheap, both prefill, both gate on perplexity.
- **Profile the 3060 first** to size the prefill fraction before any GEMM effort (G2/G3).
