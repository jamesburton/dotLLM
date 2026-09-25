using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Regression tests for issue #257: the fused decode dispatcher used to throw
/// <see cref="NotSupportedException"/> ("Fused decode does not support {qt}. Use standard Gemm
/// path.") for every quant format without a dedicated <c>ComputeRows</c> kernel, which made CPU
/// text generation impossible for 14 of the 24 supported quantization types.
/// </summary>
/// <remarks>
/// Two distinct defects are covered:
/// <list type="number">
/// <item>the hard throw when no pre-quantized input was available (the reported failure), and</item>
/// <item>a silent <c>return</c> that left the result buffer untouched when a pre-quantized input
/// <em>was</em> supplied but in a format the projection could not consume — producing zeros
/// instead of an exception.</item>
/// </list>
/// BF16 is the representative unsupported format: it is one of the 14 affected types and its
/// layout is simple enough to decode independently inside the test, so the expected values do not
/// come from the same code path under test. It replaced Q4_1 in issue #489, which gave Q4_0, Q4_1,
/// Q5_1 and IQ4_NL packed x Q8_1 fused kernels — they are no longer "unsupported", and their
/// packed path quantizes the activations, so they would also no longer match an F32 reference to
/// 1e-3.
/// </remarks>
public sealed unsafe class MatMulFusedDecodeFallbackTests : IDisposable
{
    private const int Q8_0GroupSize = 32;
    private const int Q8_0BlockBytes = 34;

    private readonly ComputeThreadPool _pool = new(4);

    public void Dispose() => _pool.Dispose();

    // ──────────────────── Capability contract ────────────────────

    /// <summary>
    /// The eight formats that have always taken the fused decode path must keep reporting as
    /// supported — a capability predicate that answered <see langword="false"/> here would
    /// silently route every decode through the slower standard GEMM.
    /// </summary>
    [Theory]
    [InlineData(QuantizationType.F32)]
    [InlineData(QuantizationType.F16)]
    [InlineData(QuantizationType.Q8_0)]
    [InlineData(QuantizationType.Q5_0)]
    [InlineData(QuantizationType.Q4_K)]
    [InlineData(QuantizationType.Q5_K)]
    [InlineData(QuantizationType.Q6_K)]
    // Gained a packed x Q8_1 ComputeRows kernel in issue #489, so they now fuse too.
    [InlineData(QuantizationType.Q4_0)]
    [InlineData(QuantizationType.Q4_1)]
    [InlineData(QuantizationType.Q5_1)]
    [InlineData(QuantizationType.IQ4_NL)]
    public void SupportsFusedDecode_ReturnsTrue_ForFormatsWithAFusedKernel(QuantizationType qt)
    {
        Assert.True(MatMul.SupportsFusedDecode(qt));
    }

    /// <summary>
    /// Every format from issue #257 that still lacks a fused kernel must report as unsupported so
    /// callers route it to the standard GEMM path. Formats leave this list as kernels are written
    /// for them — Q4_0/Q4_1/Q5_1/IQ4_NL in #489, Q2_K/Q3_K in #497.
    /// </summary>
    [Theory]
    [InlineData(QuantizationType.BF16)]
    [InlineData(QuantizationType.MXFP4)]
    // Q2_K and Q3_K left this list in #497: they now have packed x Q8_K ComputeRows kernels,
    // so SupportsFusedDecode is true for them and the fused path is the right route.
    [InlineData(QuantizationType.IQ4_XS)]
    [InlineData(QuantizationType.IQ3_S)]
    [InlineData(QuantizationType.IQ3_XXS)]
    [InlineData(QuantizationType.IQ2_S)]
    [InlineData(QuantizationType.IQ2_XS)]
    [InlineData(QuantizationType.IQ2_XXS)]
    [InlineData(QuantizationType.IQ1_S)]
    [InlineData(QuantizationType.I2_S)]
    [InlineData(QuantizationType.PQ2_0)]
    public void SupportsFusedDecode_ReturnsFalse_ForFormatsWithoutAFusedKernel(QuantizationType qt)
    {
        Assert.False(MatMul.SupportsFusedDecode(qt));
    }

    // ──────────────────── Kernel-level fallback ────────────────────

    /// <summary>
    /// Q/K/V all BF16 with no pre-quantized input. Before the fix this threw
    /// <see cref="NotSupportedException"/> — the exact failure reported in issue #257.
    /// </summary>
    [Fact]
    public void FusedDecodeGemv3_UnsupportedType_NoPreQuant_FallsBackInsteadOfThrowing()
    {
        const int m = 64, k = 256;
        var rng = new Random(1234);

        byte* w0 = AllocBF16Weights(m, k, rng);
        byte* w1 = AllocBF16Weights(m, k, rng);
        byte* w2 = AllocBF16Weights(m, k, rng);
        float* input = AllocFloats(k, rng);
        float* r0 = AllocResult(m);
        float* r1 = AllocResult(m);
        float* r2 = AllocResult(m);

        try
        {
            MatMul.FusedDecodeGemv3(
                w0, QuantizationType.BF16, r0, m,
                w1, QuantizationType.BF16, r1, m,
                w2, QuantizationType.BF16, r2, m,
                input, preQuantInput: null, k, _pool);

            AssertMatchesReference(w0, input, r0, m, k, "Proj0");
            AssertMatchesReference(w1, input, r1, m, k, "Proj1");
            AssertMatchesReference(w2, input, r2, m, k, "Proj2");
        }
        finally
        {
            FreeAll(w0, w1, w2, input, r0, r1, r2);
        }
    }

    /// <summary>
    /// Gate/Up both BF16 with no pre-quantized input — the FFN half of the same defect.
    /// </summary>
    [Fact]
    public void FusedDecodeGemv2_UnsupportedType_NoPreQuant_FallsBackInsteadOfThrowing()
    {
        const int m = 64, k = 256;
        var rng = new Random(4321);

        byte* w0 = AllocBF16Weights(m, k, rng);
        byte* w1 = AllocBF16Weights(m, k, rng);
        float* input = AllocFloats(k, rng);
        float* r0 = AllocResult(m);
        float* r1 = AllocResult(m);

        try
        {
            MatMul.FusedDecodeGemv2(
                w0, QuantizationType.BF16, r0, m,
                w1, QuantizationType.BF16, r1, m,
                input, preQuantInput: null, k, _pool);

            AssertMatchesReference(w0, input, r0, m, k, "Proj0");
            AssertMatchesReference(w1, input, r1, m, k, "Proj1");
        }
        finally
        {
            FreeAll(w0, w1, input, r0, r1);
        }
    }

    /// <summary>
    /// BF16 leading projection with a non-null pre-quantized input. The old dispatcher matched the
    /// "pre-quantized input available" branch, found no <c>ComputeRows</c> function pointer and
    /// returned without writing anything, leaving the caller's buffer at its previous contents —
    /// silently wrong output rather than an exception. Result buffers are poisoned with a sentinel
    /// so "never written" is distinguishable from "computed to zero".
    /// </summary>
    [Fact]
    public void FusedDecodeGemv3_UnsupportedType_WithForeignPreQuant_ComputesInsteadOfSkipping()
    {
        const int m = 64, k = 256;
        var rng = new Random(777);

        byte* w0 = AllocBF16Weights(m, k, rng);
        byte* w1 = AllocBF16Weights(m, k, rng);
        byte* w2 = AllocBF16Weights(m, k, rng);
        float* input = AllocFloats(k, rng);
        float* r0 = AllocResult(m);
        float* r1 = AllocResult(m);
        float* r2 = AllocResult(m);

        // A Q8_0-encoded activation buffer — valid bytes, but not a format BF16 weights can consume.
        int blockCount = k / Q8_0GroupSize;
        byte* preQuant = (byte*)NativeMemory.AlignedAlloc((nuint)(blockCount * Q8_0BlockBytes), 64);
        MatMul.QuantizeF32ToQ8_0(input, preQuant, k);

        try
        {
            MatMul.FusedDecodeGemv3(
                w0, QuantizationType.BF16, r0, m,
                w1, QuantizationType.BF16, r1, m,
                w2, QuantizationType.BF16, r2, m,
                input, preQuant, k, _pool);

            AssertMatchesReference(w0, input, r0, m, k, "Proj0");
            AssertMatchesReference(w1, input, r1, m, k, "Proj1");
            AssertMatchesReference(w2, input, r2, m, k, "Proj2");
        }
        finally
        {
            NativeMemory.AlignedFree(preQuant);
            FreeAll(w0, w1, w2, input, r0, r1, r2);
        }
    }

    /// <summary>
    /// Mixed layer: a supported Q8_0 query projection alongside unsupported BF16 K/V. The supported
    /// projection must still be computed correctly while the unsupported ones fall back.
    /// </summary>
    [Fact]
    public void FusedDecodeGemv3_MixedSupportedAndUnsupported_ComputesAllProjections()
    {
        const int m = 64, k = 256;
        var rng = new Random(2024);

        byte* wQ = AllocQ8_0Weights(m, k, rng);
        byte* wK = AllocBF16Weights(m, k, rng);
        byte* wV = AllocBF16Weights(m, k, rng);
        float* input = AllocFloats(k, rng);
        float* rQ = AllocResult(m);
        float* rK = AllocResult(m);
        float* rV = AllocResult(m);
        float* rQRef = AllocResult(m);

        int blockCount = k / Q8_0GroupSize;
        byte* preQuant = (byte*)NativeMemory.AlignedAlloc((nuint)(blockCount * Q8_0BlockBytes), 64);
        MatMul.QuantizeF32ToQ8_0(input, preQuant, k);

        try
        {
            MatMul.FusedDecodeGemv3(
                wQ, QuantizationType.Q8_0, rQ, m,
                wK, QuantizationType.BF16, rK, m,
                wV, QuantizationType.BF16, rV, m,
                input, preQuant, k, _pool);

            // Q keeps taking the fused Q8_0 kernel — compare against the standard GEMM (n=1).
            MatMul.GemmQ8_0(wQ, input, rQRef, m, k, 1, preQuantizedInput: preQuant);
            for (int i = 0; i < m; i++)
                Assert.Equal(rQRef[i], rQ[i]);

            AssertMatchesReference(wK, input, rK, m, k, "K");
            AssertMatchesReference(wV, input, rV, m, k, "V");
        }
        finally
        {
            NativeMemory.AlignedFree(preQuant);
            NativeMemory.AlignedFree(rQRef);
            FreeAll(wQ, wK, wV, input, rQ, rK, rV);
        }
    }

    // ──────────────────── Helpers ────────────────────

    /// <summary>
    /// Independently decodes the BF16 weight matrix and asserts the kernel output matches
    /// <c>W · x</c> within float tolerance.
    /// </summary>
    private static void AssertMatchesReference(byte* weights, float* input, float* actual,
                                               int m, int k, string label)
    {
        int rowBytes = k * 2;
        var row = new float[k];
        bool anyNonZero = false;

        for (int i = 0; i < m; i++)
        {
            DecodeBF16Row(weights + (long)i * rowBytes, k, row);

            double expected = 0;
            for (int j = 0; j < k; j++)
                expected += row[j] * input[j];

            Assert.False(float.IsNaN(actual[i]), $"{label}[{i}] is NaN");
            Assert.True(Math.Abs(expected - actual[i]) <= 1e-3 * Math.Max(1.0, Math.Abs(expected)),
                $"{label}[{i}]: expected {expected}, got {actual[i]}");

            if (actual[i] != 0f) anyNonZero = true;
        }

        // Guards against the "silently returned without writing" defect: an all-zero result would
        // otherwise sail through the tolerance check only if the reference were also zero.
        Assert.True(anyNonZero, $"{label}: every output element is zero — the projection never ran");
    }

    /// <summary>Decodes one BF16 row: each element is the top 16 bits of an IEEE-754 float.</summary>
    private static void DecodeBF16Row(byte* rowPtr, int k, float[] dest)
    {
        ushort* src = (ushort*)rowPtr;
        for (int j = 0; j < k; j++)
            dest[j] = BitConverter.Int32BitsToSingle(src[j] << 16);
    }

    private static byte* AllocBF16Weights(int m, int k, Random rng)
    {
        byte* ptr = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * k * 2), 64);
        ushort* v = (ushort*)ptr;
        for (long i = 0; i < (long)m * k; i++)
            v[i] = (ushort)(BitConverter.SingleToInt32Bits((float)(rng.NextDouble() * 0.2 - 0.1)) >> 16);
        return ptr;
    }

    private static byte* AllocQ8_0Weights(int m, int k, Random rng)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        int totalBytes = m * blocksPerRow * Q8_0BlockBytes;
        byte* ptr = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);

        for (int i = 0; i < m * blocksPerRow; i++)
        {
            byte* block = ptr + i * Q8_0BlockBytes;
            *(ushort*)block = BitConverter.HalfToUInt16Bits((Half)(0.01f + (float)rng.NextDouble() * 0.05f));
            for (int j = 0; j < Q8_0GroupSize; j++)
                block[2 + j] = (byte)(sbyte)(rng.Next(-127, 128));
        }

        return ptr;
    }

    private static float* AllocFloats(int count, Random rng)
    {
        float* ptr = (float*)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
        for (int i = 0; i < count; i++)
            ptr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return ptr;
    }

    /// <summary>Allocates a result buffer poisoned with NaN so an unwritten buffer fails loudly.</summary>
    private static float* AllocResult(int count)
    {
        float* ptr = (float*)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
        for (int i = 0; i < count; i++)
            ptr[i] = float.NaN;
        return ptr;
    }

    private static void FreeAll(params nint[] ptrs)
    {
        foreach (nint p in ptrs)
            NativeMemory.AlignedFree((void*)p);
    }

    private static void FreeAll(byte* a, byte* b, byte* c, float* d, float* e, float* f, float* g)
        => FreeAll((nint)a, (nint)b, (nint)c, (nint)d, (nint)e, (nint)f, (nint)g);

    private static void FreeAll(byte* a, byte* b, float* c, float* d, float* e)
        => FreeAll((nint)a, (nint)b, (nint)c, (nint)d, (nint)e);
}
