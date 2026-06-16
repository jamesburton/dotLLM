using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Parity tests for the AVX VNNI (<c>vpdpbusd</c>) fused integer Q8_0×Q8_0 dot path.
///
/// <para>
/// The VNNI path biases the signed activation to unsigned (+128) and subtracts a
/// <c>128·Σ qw</c> correction term. That correction is non-zero precisely when the weight
/// block has a non-zero signed sum, so the random-signed-weight fixtures here
/// <b>discriminate</b> a missing/incorrect bias correction (a degenerate all-positive or
/// symmetric fixture would let such a bug hide). int8 integer accumulation is exact, so the
/// only divergence from the scalar reference is fp32 scale rounding — tolerances are tight.
/// </para>
/// </summary>
public sealed unsafe class MatMulVnniTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]   // odd, exercises the single-row scalar tail of ComputeRowsVnni
    [InlineData(7)]
    [InlineData(16)]
    [InlineData(128)]
    [InlineData(344)]
    public void VecDotQ8_0Vnni_MatchesScalar(int blockCount)
    {
        if (!MatMul.IsQ8_0VnniSupported) return;

        var rng = new Random(1234);
        nuint totalBytes = (nuint)(blockCount * Q8_0BlockBytes);
        nint aPtr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        nint bPtr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            FillRandomQ8_0Blocks((byte*)aPtr, blockCount, rng);
            FillRandomQ8_0Blocks((byte*)bPtr, blockCount, rng);

            float scalar = MatMul.VecDotQ8_0Scalar((byte*)aPtr, (byte*)bPtr, blockCount);
            float vnni = MatMul.VecDotQ8_0Vnni((byte*)aPtr, (byte*)bPtr, blockCount);

            Assert.Equal(scalar, vnni, MathF.Abs(scalar) * 1e-4f + 1e-3f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)aPtr);
            NativeMemory.AlignedFree((void*)bPtr);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(18)]   // SmolLM hidden dim 576 / 32
    [InlineData(128)]
    public void VecDotQ8_0Vnni_4Row_MatchesSingleRow(int blockCount)
    {
        if (!MatMul.IsQ8_0VnniSupported) return;

        var rng = new Random(99);
        nint w0 = (nint)NativeMemory.AlignedAlloc((nuint)(blockCount * Q8_0BlockBytes), 64);
        nint w1 = (nint)NativeMemory.AlignedAlloc((nuint)(blockCount * Q8_0BlockBytes), 64);
        nint w2 = (nint)NativeMemory.AlignedAlloc((nuint)(blockCount * Q8_0BlockBytes), 64);
        nint w3 = (nint)NativeMemory.AlignedAlloc((nuint)(blockCount * Q8_0BlockBytes), 64);
        nint x = (nint)NativeMemory.AlignedAlloc((nuint)(blockCount * Q8_0BlockBytes), 64);
        try
        {
            FillRandomQ8_0Blocks((byte*)w0, blockCount, rng);
            FillRandomQ8_0Blocks((byte*)w1, blockCount, rng);
            FillRandomQ8_0Blocks((byte*)w2, blockCount, rng);
            FillRandomQ8_0Blocks((byte*)w3, blockCount, rng);
            FillRandomQ8_0Blocks((byte*)x, blockCount, rng);

            float r0 = MatMul.VecDotQ8_0Vnni((byte*)w0, (byte*)x, blockCount);
            float r1 = MatMul.VecDotQ8_0Vnni((byte*)w1, (byte*)x, blockCount);
            float r2 = MatMul.VecDotQ8_0Vnni((byte*)w2, (byte*)x, blockCount);
            float r3 = MatMul.VecDotQ8_0Vnni((byte*)w3, (byte*)x, blockCount);

            float* results = stackalloc float[4];
            MatMul.VecDotQ8_0Vnni_4Rows((byte*)w0, (byte*)w1, (byte*)w2, (byte*)w3,
                (byte*)x, blockCount, results);

            Assert.Equal(r0, results[0], MathF.Abs(r0) * 1e-5f + 1e-4f);
            Assert.Equal(r1, results[1], MathF.Abs(r1) * 1e-5f + 1e-4f);
            Assert.Equal(r2, results[2], MathF.Abs(r2) * 1e-5f + 1e-4f);
            Assert.Equal(r3, results[3], MathF.Abs(r3) * 1e-5f + 1e-4f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)w0);
            NativeMemory.AlignedFree((void*)w1);
            NativeMemory.AlignedFree((void*)w2);
            NativeMemory.AlignedFree((void*)w3);
            NativeMemory.AlignedFree((void*)x);
        }
    }

    /// <summary>
    /// End-to-end: <see cref="MatMul.GemvQ8_0(byte*, float*, float*, int, int)"/> routes through
    /// <c>ComputeRowsVnni</c> on this hardware. Cross-verify against the scalar
    /// <c>VecDotQ8_0Scalar</c> path on the <em>same</em> Q8_0-quantized activation, so the
    /// comparison isolates the integer dot (both sides see identical quantized inputs;
    /// activation-quant cancellation error is excluded from the comparison).
    /// </summary>
    [Theory]
    [InlineData(64, 576)]    // m, k — exercises 4-row groups
    [InlineData(67, 256)]    // m not a multiple of 4 — exercises scalar row tail
    public void GemvQ8_0_Vnni_MatchesScalarPath(int m, int k)
    {
        if (!MatMul.IsQ8_0VnniSupported) return;

        var rng = new Random(7);
        int blockCount = k / Q8_0GroupSize;
        int rowBytes = blockCount * Q8_0BlockBytes;
        nint weights = (nint)NativeMemory.AlignedAlloc((nuint)((long)m * rowBytes), 64);
        for (int row = 0; row < m; row++)
            FillRandomQ8_0Blocks((byte*)weights + (long)row * rowBytes, blockCount, rng);

        float[] input = new float[k];
        for (int i = 0; i < k; i++) input[i] = (rng.NextSingle() - 0.5f) * 2f;

        float[] gemv = new float[m];
        nint xQ8 = (nint)NativeMemory.AlignedAlloc((nuint)rowBytes, 64);
        try
        {
            fixed (float* inPtr = input, outPtr = gemv)
            {
                MatMul.GemvQ8_0((byte*)weights, inPtr, outPtr, m, k);
                // Reference quantizes the activation identically, then scalar dot per row.
                MatMul.QuantizeF32ToQ8_0(inPtr, (byte*)xQ8, k);
            }

            for (int row = 0; row < m; row++)
            {
                float scalar = MatMul.VecDotQ8_0Scalar(
                    (byte*)weights + (long)row * rowBytes, (byte*)xQ8, blockCount);
                Assert.Equal(scalar, gemv[row], MathF.Abs(scalar) * 1e-4f + 1e-3f);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weights);
            NativeMemory.AlignedFree((void*)xQ8);
        }
    }

    /// <summary>All-zero weight scales must yield exactly 0 (no spurious bias-correction leak).</summary>
    [Fact]
    public void VecDotQ8_0Vnni_AllZeroScales_IsZero()
    {
        if (!MatMul.IsQ8_0VnniSupported) return;

        const int blockCount = 8;
        var rng = new Random(5);
        nint aPtr = (nint)NativeMemory.AlignedAlloc((nuint)(blockCount * Q8_0BlockBytes), 64);
        nint bPtr = (nint)NativeMemory.AlignedAlloc((nuint)(blockCount * Q8_0BlockBytes), 64);
        try
        {
            FillRandomQ8_0Blocks((byte*)aPtr, blockCount, rng);
            FillRandomQ8_0Blocks((byte*)bPtr, blockCount, rng);
            // Zero all weight scales.
            for (int b = 0; b < blockCount; b++)
                *(Half*)((byte*)aPtr + b * Q8_0BlockBytes) = (Half)0f;

            float vnni = MatMul.VecDotQ8_0Vnni((byte*)aPtr, (byte*)bPtr, blockCount);
            Assert.Equal(0f, vnni, 1e-6f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)aPtr);
            NativeMemory.AlignedFree((void*)bPtr);
        }
    }

    private static void FillRandomQ8_0Blocks(byte* ptr, int blockCount, Random rng)
    {
        for (int b = 0; b < blockCount; b++)
        {
            byte* block = ptr + b * Q8_0BlockBytes;
            *(Half*)block = (Half)(rng.NextSingle() * 0.1f);
            for (int i = 0; i < Q8_0GroupSize; i++)
                ((sbyte*)(block + 2))[i] = (sbyte)rng.Next(-127, 128);
        }
    }
}
