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
/// Q4_1 is used as the representative unsupported format: it is one of the 14 affected types and
/// its block layout is simple enough to decode independently inside the test, so the expected
/// values do not come from the same code path under test.
/// </remarks>
public sealed unsafe class MatMulFusedDecodeFallbackTests : IDisposable
{
    private const int Q4_1GroupSize = 32;
    private const int Q4_1BlockBytes = 20; // d(Half) + m(Half) + 16 packed nibble bytes
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
    public void SupportsFusedDecode_ReturnsTrue_ForFormatsWithAFusedKernel(QuantizationType qt)
    {
        Assert.True(MatMul.SupportsFusedDecode(qt));
    }

    /// <summary>
    /// Every format listed in issue #257 as failing must report as unsupported so callers route it
    /// to the standard GEMM path instead of the fused kernels.
    /// </summary>
    [Theory]
    [InlineData(QuantizationType.BF16)]
    [InlineData(QuantizationType.Q4_1)]
    [InlineData(QuantizationType.Q5_1)]
    [InlineData(QuantizationType.IQ4_NL)]
    [InlineData(QuantizationType.MXFP4)]
    [InlineData(QuantizationType.Q2_K)]
    [InlineData(QuantizationType.Q3_K)]
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
    /// Q/K/V all Q4_1 with no pre-quantized input. Before the fix this threw
    /// <see cref="NotSupportedException"/> — the exact failure reported in issue #257.
    /// </summary>
    [Fact]
    public void FusedDecodeGemv3_UnsupportedType_NoPreQuant_FallsBackInsteadOfThrowing()
    {
        const int m = 64, k = 256;
        var rng = new Random(1234);

        byte* w0 = AllocQ4_1Weights(m, k, rng);
        byte* w1 = AllocQ4_1Weights(m, k, rng);
        byte* w2 = AllocQ4_1Weights(m, k, rng);
        float* input = AllocFloats(k, rng);
        float* r0 = AllocResult(m);
        float* r1 = AllocResult(m);
        float* r2 = AllocResult(m);

        try
        {
            MatMul.FusedDecodeGemv3(
                w0, QuantizationType.Q4_1, r0, m,
                w1, QuantizationType.Q4_1, r1, m,
                w2, QuantizationType.Q4_1, r2, m,
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
    /// Gate/Up both Q4_1 with no pre-quantized input — the FFN half of the same defect.
    /// </summary>
    [Fact]
    public void FusedDecodeGemv2_UnsupportedType_NoPreQuant_FallsBackInsteadOfThrowing()
    {
        const int m = 64, k = 256;
        var rng = new Random(4321);

        byte* w0 = AllocQ4_1Weights(m, k, rng);
        byte* w1 = AllocQ4_1Weights(m, k, rng);
        float* input = AllocFloats(k, rng);
        float* r0 = AllocResult(m);
        float* r1 = AllocResult(m);

        try
        {
            MatMul.FusedDecodeGemv2(
                w0, QuantizationType.Q4_1, r0, m,
                w1, QuantizationType.Q4_1, r1, m,
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
    /// Q4_1 leading projection with a non-null pre-quantized input. The old dispatcher matched the
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

        byte* w0 = AllocQ4_1Weights(m, k, rng);
        byte* w1 = AllocQ4_1Weights(m, k, rng);
        byte* w2 = AllocQ4_1Weights(m, k, rng);
        float* input = AllocFloats(k, rng);
        float* r0 = AllocResult(m);
        float* r1 = AllocResult(m);
        float* r2 = AllocResult(m);

        // A Q8_0-encoded activation buffer — valid bytes, but not a format Q4_1 weights can consume.
        int blockCount = k / Q8_0GroupSize;
        byte* preQuant = (byte*)NativeMemory.AlignedAlloc((nuint)(blockCount * Q8_0BlockBytes), 64);
        MatMul.QuantizeF32ToQ8_0(input, preQuant, k);

        try
        {
            MatMul.FusedDecodeGemv3(
                w0, QuantizationType.Q4_1, r0, m,
                w1, QuantizationType.Q4_1, r1, m,
                w2, QuantizationType.Q4_1, r2, m,
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
    /// Mixed layer: a supported Q8_0 query projection alongside unsupported Q4_1 K/V. The supported
    /// projection must still be computed correctly while the unsupported ones fall back.
    /// </summary>
    [Fact]
    public void FusedDecodeGemv3_MixedSupportedAndUnsupported_ComputesAllProjections()
    {
        const int m = 64, k = 256;
        var rng = new Random(2024);

        byte* wQ = AllocQ8_0Weights(m, k, rng);
        byte* wK = AllocQ4_1Weights(m, k, rng);
        byte* wV = AllocQ4_1Weights(m, k, rng);
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
                wK, QuantizationType.Q4_1, rK, m,
                wV, QuantizationType.Q4_1, rV, m,
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
    /// Independently decodes the Q4_1 weight matrix and asserts the kernel output matches
    /// <c>W · x</c> within float tolerance.
    /// </summary>
    private static void AssertMatchesReference(byte* weights, float* input, float* actual,
                                               int m, int k, string label)
    {
        int blocksPerRow = k / Q4_1GroupSize;
        int rowBytes = blocksPerRow * Q4_1BlockBytes;
        var row = new float[k];
        bool anyNonZero = false;

        for (int i = 0; i < m; i++)
        {
            DecodeQ4_1Row(weights + (long)i * rowBytes, blocksPerRow, row);

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

    /// <summary>Decodes one Q4_1 row (<c>d(Half) + m(Half) + 16 nibble bytes</c> per 32 elements).</summary>
    private static void DecodeQ4_1Row(byte* rowPtr, int blocksPerRow, float[] dest)
    {
        for (int b = 0; b < blocksPerRow; b++)
        {
            byte* block = rowPtr + b * Q4_1BlockBytes;
            float d = (float)BitConverter.UInt16BitsToHalf(*(ushort*)block);
            float min = (float)BitConverter.UInt16BitsToHalf(*(ushort*)(block + 2));
            byte* qs = block + 4;

            for (int j = 0; j < Q4_1GroupSize / 2; j++)
            {
                dest[b * Q4_1GroupSize + j] = d * (qs[j] & 0x0F) + min;
                dest[b * Q4_1GroupSize + j + Q4_1GroupSize / 2] = d * (qs[j] >> 4) + min;
            }
        }
    }

    private static byte* AllocQ4_1Weights(int m, int k, Random rng)
    {
        int blocksPerRow = k / Q4_1GroupSize;
        int totalBytes = m * blocksPerRow * Q4_1BlockBytes;
        byte* ptr = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);

        for (int i = 0; i < m * blocksPerRow; i++)
        {
            byte* block = ptr + i * Q4_1BlockBytes;
            *(ushort*)block = BitConverter.HalfToUInt16Bits((Half)(0.01f + (float)rng.NextDouble() * 0.05f));
            *(ushort*)(block + 2) = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.2f - 0.1f));
            for (int j = 0; j < Q4_1GroupSize / 2; j++)
                block[4 + j] = (byte)rng.Next(256);
        }

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
