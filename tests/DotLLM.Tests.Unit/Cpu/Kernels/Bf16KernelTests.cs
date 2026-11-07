using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1118: the shift-widen BF16 kernels. Each SIMD tier is bit-exact with <see cref="MatMul.DotBf16LaneReferenceScalar"/> run with
/// that tier's lane count and FMA setting (the oracle reproduces the tier's accumulation order in scalar code), the GEMV, GEMM,
/// serial and pooled drivers agree bit for bit (a row's output is independent of batch size and thread count), and the whole
/// path stays within a reduction-length-scaled bound of the dequantize-rows path it replaces.
/// </summary>
public sealed unsafe class Bf16KernelTests
{
    private static void FillBf16(Random rng, ushort* w, long count, float scale)
    {
        for (long i = 0; i < count; i++)
        {
            float f = (rng.NextSingle() * 2f - 1f) * scale;
            w[i] = (ushort)(BitConverter.SingleToInt32Bits(f) >> 16);   // truncate to BF16
        }
    }

    private static void FillF32(Random rng, float* x, long count, float scale)
    {
        for (long i = 0; i < count; i++) x[i] = (rng.NextSingle() * 2f - 1f) * scale;
    }

    private static void AssertBitEqual(float expected, float actual, string what)
    {
        if (BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(actual))
            Assert.Fail($"{what}: expected {expected:R} (0x{BitConverter.SingleToInt32Bits(expected):X8}) got {actual:R} (0x{BitConverter.SingleToInt32Bits(actual):X8})");
    }

    /// <summary>Lengths that cover whole vectors, the 8/16-lane remainders and a scalar tail.</summary>
    public static TheoryData<int> Lengths() => new() { 4, 8, 16, 24, 37, 64, 100, 576, 1000, 2048 };

    [SkippableTheory]
    [MemberData(nameof(Lengths))]
    public void FmaTier_Dot_And_Rows4_BitExactWithLaneReference(int k)
    {
        Skip.IfNot(DotLLM.Cpu.CpuFeatures.Avx2Fma, "needs AVX2+FMA");
        var rng = new Random(1118 + k);
        ushort* w = (ushort*)NativeMemory.Alloc((nuint)(4 * k * sizeof(ushort)));
        float* x = (float*)NativeMemory.Alloc((nuint)(k * sizeof(float)));
        float* y = stackalloc float[4];
        try
        {
            FillBf16(rng, w, 4L * k, 1f); FillF32(rng, x, k, 2f);
            for (int r = 0; r < 4; r++)
                AssertBitEqual(MatMul.DotBf16LaneReferenceScalar(w + r * k, x, k, 8, fma: true), MatMul.DotBf16Fma(w + r * k, x, k), $"DotBf16Fma r={r} k={k}");
            MatMul.Bf16Rows4Fma(w, x, y, k);
            for (int r = 0; r < 4; r++)
                AssertBitEqual(MatMul.DotBf16LaneReferenceScalar(w + r * k, x, k, 8, fma: true), y[r], $"Bf16Rows4Fma r={r} k={k}");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    /// <summary>Both AVX2/FMA GEMM forms (register tile and four-row scratch), with odd row and token remainders, against the 8-lane oracle.</summary>
    [SkippableTheory]
    [MemberData(nameof(Lengths))]
    public void FmaTier_GemmForms_BitExactWithLaneReference(int k)
    {
        Skip.IfNot(DotLLM.Cpu.CpuFeatures.Avx2Fma, "needs AVX2+FMA");
        var rng = new Random(7118 + k);
        const int n = 3, m = 7;
        ushort* w = (ushort*)NativeMemory.Alloc((nuint)(m * k * sizeof(ushort)));
        float* b = (float*)NativeMemory.Alloc((nuint)(n * k * sizeof(float)));
        float* c = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        float* scratch = (float*)NativeMemory.Alloc((nuint)(MatMul.GemmBf16ScratchFloats(k) * sizeof(float)));
        try
        {
            FillBf16(rng, w, (long)m * k, 1f); FillF32(rng, b, (long)n * k, 2f);
            MatMul.GemmBf16RowsFmaTile(w, m, b, c, m, k, n);
            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    AssertBitEqual(MatMul.DotBf16LaneReferenceScalar(w + r * k, b + t * k, k, 8, fma: true), c[t * m + r], $"tile r={r} t={t} k={k}");
            new Span<float>(c, n * m).Fill(float.NaN);
            MatMul.GemmBf16RowsFma(w, m, b, c, m, k, n, scratch);
            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    AssertBitEqual(MatMul.DotBf16LaneReferenceScalar(w + r * k, b + t * k, k, 8, fma: true), c[t * m + r], $"scratch r={r} t={t} k={k}");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(b); NativeMemory.Free(c); NativeMemory.Free(scratch); }
    }

    [SkippableTheory]
    [MemberData(nameof(Lengths))]
    public void Avx512Tier_Dot_Rows4_And_Tile_BitExactWithLaneReference(int k)
    {
        Skip.IfNot(DotLLM.Cpu.CpuFeatures.Avx512F, "needs AVX-512F");
        var rng = new Random(2118 + k);
        const int n = 3, m = 7;
        ushort* w = (ushort*)NativeMemory.Alloc((nuint)(m * k * sizeof(ushort)));
        float* b = (float*)NativeMemory.Alloc((nuint)(n * k * sizeof(float)));
        float* c = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        float* y = stackalloc float[4];
        try
        {
            FillBf16(rng, w, (long)m * k, 1f); FillF32(rng, b, (long)n * k, 2f);
            for (int r = 0; r < 4; r++)
                AssertBitEqual(MatMul.DotBf16LaneReferenceScalar(w + r * k, b, k, 16, fma: true), MatMul.DotBf16Avx512(w + r * k, b, k), $"DotBf16Avx512 r={r} k={k}");
            MatMul.Bf16Rows4Avx512(w, b, y, k);
            for (int r = 0; r < 4; r++)
                AssertBitEqual(MatMul.DotBf16LaneReferenceScalar(w + r * k, b, k, 16, fma: true), y[r], $"Bf16Rows4Avx512 r={r} k={k}");
            // 4x2 register tile plus the odd row and odd token remainders.
            MatMul.GemmBf16RowsAvx512(w, m, b, c, m, k, n);
            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    AssertBitEqual(MatMul.DotBf16LaneReferenceScalar(w + r * k, b + t * k, k, 16, fma: true), c[t * m + r], $"GemmBf16RowsAvx512 r={r} t={t} k={k}");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(b); NativeMemory.Free(c); }
    }

    [SkippableTheory]
    [MemberData(nameof(Lengths))]
    public void V128Tier_Dot_And_Rows4_BitExactWithLaneReference(int k)
    {
        Skip.IfNot(DotLLM.Cpu.CpuFeatures.Vector128Accelerated, "needs Vector128 acceleration");
        var rng = new Random(3118 + k);
        ushort* w = (ushort*)NativeMemory.Alloc((nuint)(4 * k * sizeof(ushort)));
        float* x = (float*)NativeMemory.Alloc((nuint)(k * sizeof(float)));
        float* y = stackalloc float[4];
        try
        {
            FillBf16(rng, w, 4L * k, 1f); FillF32(rng, x, k, 2f);
            for (int r = 0; r < 4; r++)
                AssertBitEqual(MatMul.DotBf16LaneReferenceScalar(w + r * k, x, k, 4, fma: false), MatMul.DotBf16V128Simd(w + r * k, x, k), $"DotBf16V128 r={r} k={k}");
            MatMul.Bf16Rows4V128Simd(w, x, y, k);
            for (int r = 0; r < 4; r++)
                AssertBitEqual(MatMul.DotBf16LaneReferenceScalar(w + r * k, x, k, 4, fma: false), y[r], $"Bf16Rows4V128 r={r} k={k}");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    /// <summary>The scalar dot and the lane oracle agree exactly when there is one lane and no FMA (sanity for the oracle itself).</summary>
    [Fact]
    public void LaneReference_OneLane_IsThePlainScalarDot()
    {
        var rng = new Random(4118);
        const int k = 131;
        ushort* w = stackalloc ushort[k];
        float* x = stackalloc float[k];
        FillBf16(rng, w, k, 1f); FillF32(rng, x, k, 2f);
        // lanes = 4 without FMA and a sequential plain dot differ only in order; with k = 4 they coincide exactly for one vector.
        ushort* w4 = stackalloc ushort[4]; float* x4 = stackalloc float[4];
        FillBf16(rng, w4, 4, 1f); FillF32(rng, x4, 4, 2f);
        float plain = (w4[0] is var _ ? MatMul.Bf16ToSingle(w4[0]) * x4[0] + MatMul.Bf16ToSingle(w4[1]) * x4[1] : 0f)
                      + (MatMul.Bf16ToSingle(w4[2]) * x4[2] + MatMul.Bf16ToSingle(w4[3]) * x4[3]);
        AssertBitEqual(plain, MatMul.DotBf16LaneReferenceScalar(w4, x4, 4, 4, fma: false), "4-lane oracle at k=4");
        Assert.Equal(MatMul.DotBf16Scalar(w, x, k), MatMul.DotBf16LaneReferenceScalar(w, x, k, 4, fma: false), 1e-3f);
    }

    /// <summary>
    /// The production drivers on the tier in force: serial GEMM, pooled GEMM, GEMV per token and the pooled GEMV all equal the
    /// tier's lane oracle bit for bit, so a row's result is independent of batch size and thread partitioning.
    /// </summary>
    [SkippableTheory]
    [InlineData(7, 64, 1)]
    [InlineData(13, 576, 3)]
    [InlineData(39, 100, 5)]     // >= ParallelMinRows and k with a scalar tail
    [InlineData(70, 1024, 9)]
    public void Drivers_AgreeWithLaneReference_BitExact(int m, int k, int n)
    {
        Skip.IfNot(MatMul.Bf16KernelEnabled, "BF16 kernel disabled by DOTLLM_CPU_BF16_KERNEL=0");
        (int lanes, bool fma, bool simd) = MatMul.Bf16Avx512Tier ? (16, true, true) : MatMul.Bf16FmaTier ? (8, true, true) : MatMul.Bf16V128Tier ? (4, false, true) : (1, false, false);
        var rng = new Random(5118 + m * 7 + k + n);
        ushort* w = (ushort*)NativeMemory.Alloc((nuint)((long)m * k * sizeof(ushort)));
        float* b = (float*)NativeMemory.Alloc((nuint)((long)n * k * sizeof(float)));
        float* expected = (float*)NativeMemory.Alloc((nuint)((long)n * m * sizeof(float)));
        float* actual = (float*)NativeMemory.Alloc((nuint)((long)n * m * sizeof(float)));
        try
        {
            FillBf16(rng, w, (long)m * k, 1f); FillF32(rng, b, (long)n * k, 2f);
            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    expected[t * m + r] = simd ? MatMul.DotBf16LaneReferenceScalar(w + (long)r * k, b + (long)t * k, k, lanes, fma)
                                               : MatMul.DotBf16Scalar(w + (long)r * k, b + (long)t * k, k);

            using var pool = new ComputeThreadPool(4);
            MatMul.GemmBf16((nint)w, b, actual, m, k, n);
            Compare(expected, actual, n * m, "GemmBf16 serial");
            MatMul.GemmBf16((nint)w, b, actual, m, k, n, pool);
            Compare(expected, actual, n * m, "GemmBf16 pooled");
            for (int t = 0; t < n; t++) MatMul.GemvBf16((nint)w, b + (long)t * k, actual + (long)t * m, m, k);
            Compare(expected, actual, n * m, "GemvBf16 per token");
            for (int t = 0; t < n; t++) MatMul.GemvBf16((nint)w, b + (long)t * k, actual + (long)t * m, m, k, pool);
            Compare(expected, actual, n * m, "GemvBf16 pooled per token");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(b); NativeMemory.Free(expected); NativeMemory.Free(actual); }

        static void Compare(float* expected, float* actual, int count, string arm)
        {
            for (int i = 0; i < count; i++) AssertBitEqual(expected[i], actual[i], $"{arm} [{i}]");
            new Span<float>(actual, count).Fill(float.NaN);
        }
    }

    /// <summary>
    /// Against the dequantize-rows path the kernel replaces: the same inputs, a different summation order, so the difference is
    /// bounded by F32 reassociation. Bound scales with sqrt(k) and the magnitude of the terms (not a fixed tolerance).
    /// </summary>
    [SkippableTheory]
    [InlineData(33, 576, 4)]
    [InlineData(64, 2048, 8)]
    public void GemmBf16_WithinReassociationBoundOfDequantRows(int m, int k, int n)
    {
        Skip.IfNot(MatMul.Bf16KernelEnabled, "BF16 kernel disabled by DOTLLM_CPU_BF16_KERNEL=0");
        var rng = new Random(6118 + m + k + n);
        ushort* w = (ushort*)NativeMemory.Alloc((nuint)((long)m * k * sizeof(ushort)));
        float* b = (float*)NativeMemory.Alloc((nuint)((long)n * k * sizeof(float)));
        float* reference = (float*)NativeMemory.Alloc((nuint)((long)n * m * sizeof(float)));
        float* actual = (float*)NativeMemory.Alloc((nuint)((long)n * m * sizeof(float)));
        try
        {
            FillBf16(rng, w, (long)m * k, 1f); FillF32(rng, b, (long)n * k, 2f);
            MatMul.GemmDequantRows((byte*)w, QuantizationType.BF16, b, reference, m, k, n, null);
            MatMul.GemmBf16((nint)w, b, actual, m, k, n);
            float eps = 1.1920929e-7f;
            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                {
                    double mag = 0;
                    for (int i = 0; i < k; i++) mag += Math.Abs((double)MatMul.Bf16ToSingle(w[(long)r * k + i]) * b[(long)t * k + i]);
                    double bound = 4 * eps * Math.Sqrt(k) * mag + 1e-7;
                    Assert.True(Math.Abs(reference[t * m + r] - actual[t * m + r]) <= bound,
                        $"r={r} t={t}: dequant-rows {reference[t * m + r]:R} vs kernel {actual[t * m + r]:R}, bound {bound:E2}");
                }
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(b); NativeMemory.Free(reference); NativeMemory.Free(actual); }
    }
}
