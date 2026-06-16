# [API Proposal]: Add support for AVX-512 BF16 hardware intrinsics

> **FILED: dotnet/runtime#129323** (issue) · **PR #129326** (implementation). Authoring/CI monitored elsewhere.
> Modeled on #86849 (AVX-512 VNNI → `AvxVnni.V512`, PR #128365).
> Strix build: use the `dev` worktree at `C:/dotnet-runtime-dev` (has both `AvxVnni.V512` + `Avx512Bf16`).

### Background and motivation

The `AVX512_BF16` instruction set adds bfloat16 support to AVX-512: a bf16 dot-product accumulating into FP32 (`VDPBF16PS`) and FP32→bf16 round-to-nearest-even conversions (`VCVTNE2PS2BF16`, `VCVTNEPS2BF16`). It is supported on Intel Cooper Lake / Ice Lake-SP / Sapphire Rapids+ and on **AMD Zen 4 and Zen 5**, but there is currently **no managed surface for it** in `System.Runtime.Intrinsics.X86`.

This is distinct from two things already (partly) represented in the BCL:
- **`Avx10v1`/`Avx10v2`** — AVX10 also carries BF16, but is gated on the AVX10 feature, which current AMD Zen and Intel Sapphire Rapids parts do **not** report. Exactly as with VNNI (where `AvxVnni.V512` was added against the classic `AVX512_VNNI` gate rather than relying on AVX10), the classic `AVX512_BF16` gate needs its own surface to be usable on shipping hardware.
- **AVX-512 FP16** (IEEE half, `AVX512_FP16`, tracked separately e.g. #62416) — a different extension; BF16 here is bfloat16.

**Motivating use case — quantized / mixed-precision ML inference.** `VDPBF16PS` accumulates bf16×bf16 products directly into an FP32 accumulator. For block-scaled integer quantization (e.g. GGUF `Q8_0`: 32 int8 + one fp16 scale per block), an integer-VNNI dot must convert+rescale after *every* block, which prevents long accumulation. A BF16 path can dequantize int8→bf16 **folding the per-block scale into the value**, then accumulate the already-scaled products across all blocks in one FP32 accumulator via `VDPBF16PS` — removing the per-block reduction entirely. (Concrete data: in an A/B on Zen 5, a 512-bit integer-VNNI Q8_0 GEMM kernel reached only *parity* with the existing `maddubs` path precisely because of these per-block reductions; BF16 long-accumulation is the path past that ceiling.) BF16 GEMM and attention more generally also benefit.

### API Proposal

Mirrors the established `Avx512*`/`AvxVnni` pattern (final shape subject to API review — see Open questions).

```csharp
namespace System.Runtime.Intrinsics.X86
{
    /// <summary>Provides access to X86 AVX512-BF16 hardware instructions via intrinsics.</summary>
    [Intrinsic]
    public abstract class Avx512Bf16 : Avx512F
    {
        public static new bool IsSupported { get; }

        /// <summary>__m512 _mm512_dpbf16_ps (__m512 src, __m512bh a, __m512bh b) — VDPBF16PS zmm, zmm, zmm/m512</summary>
        public static Vector512<float> MultiplyWideningAndAdd(Vector512<float> addend, Vector512<bfloat16> left, Vector512<bfloat16> right);

        /// <summary>__m512bh _mm512_cvtne2ps_pbh (__m512 a, __m512 b) — VCVTNE2PS2BF16 zmm, zmm, zmm/m512 (packs two fp32 vectors → one bf16 vector)</summary>
        public static Vector512<bfloat16> ConvertToBFloat16(Vector512<float> lower, Vector512<float> upper);

        /// <summary>__m256bh _mm512_cvtneps_pbh (__m512 a) — VCVTNEPS2BF16 ymm, zmm/m512</summary>
        public static Vector256<bfloat16> ConvertToBFloat16(Vector512<float> value);

        [Intrinsic] public new abstract class X64 : Avx512F.X64 { public static new bool IsSupported { get; } }

        [Intrinsic]
        public new abstract class VL : Avx512F.VL
        {
            public static new bool IsSupported { get; }
            // 128/256-bit VL forms of the three operations above
            public static Vector128<float> MultiplyWideningAndAdd(Vector128<float> addend, Vector128<bfloat16> left, Vector128<bfloat16> right);
            public static Vector256<float> MultiplyWideningAndAdd(Vector256<float> addend, Vector256<bfloat16> left, Vector256<bfloat16> right);
            public static Vector128<bfloat16> ConvertToBFloat16(Vector128<float> lower, Vector128<float> upper);
            public static Vector256<bfloat16> ConvertToBFloat16(Vector256<float> lower, Vector256<float> upper);
            public static Vector128<bfloat16> ConvertToBFloat16(Vector128<float> value);
            public static Vector128<bfloat16> ConvertToBFloat16(Vector256<float> value);
        }
    }
}
```

### Open questions (for API review)
- **bf16 element type.** Shown above as `Vector512<bfloat16>`. If a `System.Numerics.BFloat16` primitive is not available, represent the bf16 operands as `Vector512<ushort>` (raw bits) as an interim — consistent with how the VNNI proposal used integer vectors. The `ConvertToBFloat16` intrinsics are the natural producers of these operands.
- **Naming.** `MultiplyWideningAndAdd` is used here for consistency with `AvxVnni` (multiply pairs, widen to fp32, add to addend). `ConvertToBFloat16` for the two/one-source conversions. Open to the review team's convention.
- **Preview gating.** Likely `[RequiresPreviewFeatures]` initially, consistent with `AvxVnni`.

### Alternatives considered
- AVX10 BF16 (`Avx10v*`) — gated on AVX10, unavailable on current Zen4/5 and Sapphire Rapids; does not serve those parts.
- Software bf16 dot — no hardware acceleration.

### References
- #86849 / PR #128365 — AVX-512 VNNI (`AvxVnni.V512`), the direct precedent for adding a classic-AVX512-gated intrinsic.
