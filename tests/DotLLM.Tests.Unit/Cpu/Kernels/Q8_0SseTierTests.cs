using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #477: the 128-bit (SSSE3) Q8_0 x Q8_0 tier (<c>MatMul.Q8_0Sse.cs</c>) — the path every
/// Q8_0 GEMV/GEMM takes on Westmere-class hardware.
///
/// <para>The tier is designed to be <b>bit-exact with the scalar tier</b>, so the kernel tests
/// compare float bit patterns, not tolerances: any lane-order, row-offset, stride, scale or
/// half-block bug changes bits. An independent exact integer reference (all scales 1.0) pins the
/// integer reduction itself, independent of the scalar kernel.</para>
///
/// <para>The kernels are called directly and run natively on any SSSE3 box. The
/// <c>*_Dispatch_*</c> tests go through the public drivers; run the suite with
/// <c>DOTNET_EnableAVX=0</c> (Westmere's ISA: SSE2..SSE4.2 + SSSE3, no AVX/AVX2/FMA) and they take
/// the 128-bit tier, where they additionally assert bit-exactness with scalar.</para>
/// </summary>
public sealed unsafe class Q8_0SseTierTests
{
    private const int BlockBytes = 34;
    private readonly ITestOutputHelper _output;
    public Q8_0SseTierTests(ITestOutputHelper output) => _output = output;

    private static bool SseTierDispatched => Ssse3.IsSupported && !Avx2.IsSupported;

    // ──────────────────── Half → Single ────────────────────

    [SkippableFact]
    public void HalfToSingleSse2_MatchesSoftwareHalf_AllBitPatterns()
    {
        Skip.IfNot(Sse2.IsSupported, "needs SSE2");
        for (int h = 0; h < 65536; h += 4)
        {
            Vector128<float> v = MatMul.HalfToSingleSse2(Vector128.Create(h, h + 1, h + 2, h + 3));
            for (int l = 0; l < 4; l++)
            {
                float expected = (float)BitConverter.UInt16BitsToHalf((ushort)(h + l));
                float actual = v.GetElement(l);
                if (float.IsNaN(expected))
                    Assert.True(float.IsNaN(actual), $"half 0x{h + l:X4}: expected NaN, got {actual}");
                else
                    Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(actual),
                        $"half 0x{h + l:X4}: expected {expected} (0x{BitConverter.SingleToInt32Bits(expected):X8}), " +
                        $"got {actual} (0x{BitConverter.SingleToInt32Bits(actual):X8})");
            }
        }
    }

    // ──────────────────── Kernels (bit-exact) ────────────────────

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(18)]   // k = 576 (SmolLM-135M hidden)
    [InlineData(64)]   // k = 2048
    [InlineData(137)]  // odd, 4-block remainder of 1
    public void VecDotQ8_0Sse_BitExactWithScalar(int blockCount)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(4770 + blockCount);
        byte* w = RandomQ8Row(rng, blockCount, allowMinus128: true);
        byte* x = RandomQ8Row(rng, blockCount, allowMinus128: false);
        try
        {
            float expected = MatMul.VecDotQ8_0Scalar(w, x, blockCount);
            float actual = MatMul.VecDotQ8_0Sse(w, x, blockCount);
            AssertBitEqual(expected, actual, $"blockCount={blockCount}");

            if (Avx2.IsSupported)
            {
                // The AVX2 tier sums in a different order (8 float lanes), so only near-equal —
                // and it wraps -128 weights, so compare on a row without them.
                byte* w2 = RandomQ8Row(rng, blockCount, allowMinus128: false);
                try
                {
                    float s = MatMul.VecDotQ8_0Sse(w2, x, blockCount);
                    float a = MatMul.VecDotQ8_0Avx2(w2, x, blockCount);
                    Assert.True(Math.Abs(s - a) <= 1e-5f * MathF.Max(1f, MathF.Abs(a)), $"SSE {s} vs AVX2 {a}");
                }
                finally { NativeMemory.Free(w2); }
            }
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    /// <summary>
    /// Exact integer reference, independent of every kernel: with all scales 1.0 the dot is the
    /// integer Σ w·x, exactly representable while |Σ| &lt; 2^24.
    /// </summary>
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(18)]
    [InlineData(33)]
    public void Q8_0Sse_UnitScales_EqualsExactIntegerDot(int blockCount)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(4780 + blockCount);
        const int m = 4;
        byte* w = (byte*)NativeMemory.Alloc((nuint)(m * blockCount * BlockBytes));
        byte* x = RandomQ8Row(rng, blockCount, allowMinus128: false);
        float* res = stackalloc float[m];
        try
        {
            for (int r = 0; r < m; r++)
            {
                byte* row = w + r * blockCount * BlockBytes;
                FillQ8Row(rng, row, blockCount, allowMinus128: true);
                for (int b = 0; b < blockCount; b++) *(Half*)(row + b * BlockBytes) = (Half)1f;
            }
            for (int b = 0; b < blockCount; b++)
            {
                *(Half*)(x + b * BlockBytes) = (Half)1f;
                // Force extremes into the first block: -128 weight against a negative activation,
                // +127 x -127, so the saturation / sign-wrap corners are always present.
                x[b * BlockBytes + 2] = unchecked((byte)(sbyte)-127);
                x[b * BlockBytes + 3] = 127;
            }
            w[2] = unchecked((byte)(sbyte)-128);
            w[3] = unchecked((byte)(sbyte)-127);

            MatMul.VecDotQ8_0Sse_4Rows(w, blockCount * BlockBytes, BlockBytes, x, blockCount, res);
            for (int r = 0; r < m; r++)
            {
                byte* row = w + r * blockCount * BlockBytes;
                long exact = 0;
                for (int b = 0; b < blockCount; b++)
                for (int i = 0; i < 32; i++)
                    exact += (sbyte)row[b * BlockBytes + 2 + i] * (sbyte)x[b * BlockBytes + 2 + i];
                Assert.True(Math.Abs(exact) < (1 << 24));
                Assert.Equal((float)exact, res[r]);
                Assert.Equal((float)exact, MatMul.VecDotQ8_0Sse(row, x, blockCount));
            }
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(18)]
    [InlineData(137)]
    public void VecDotQ8_0Sse_4Rows_RowMajorAndR4_BitExactWithScalar(int blockCount)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(4790 + blockCount);
        int rowBytes = blockCount * BlockBytes;
        byte* w = (byte*)NativeMemory.Alloc((nuint)(4 * rowBytes));
        byte* r4 = (byte*)NativeMemory.Alloc((nuint)(4 * rowBytes));
        byte* x = RandomQ8Row(rng, blockCount, allowMinus128: false);
        float* res = stackalloc float[4];
        try
        {
            for (int r = 0; r < 4; r++) FillQ8Row(rng, w + r * rowBytes, blockCount, allowMinus128: true);
            // R4 interleave by hand: [r0_b0][r1_b0][r2_b0][r3_b0][r0_b1]...
            for (int b = 0; b < blockCount; b++)
            for (int r = 0; r < 4; r++)
                Buffer.MemoryCopy(w + r * rowBytes + b * BlockBytes, r4 + (b * 4 + r) * BlockBytes, BlockBytes, BlockBytes);

            MatMul.VecDotQ8_0Sse_4Rows(w, rowBytes, BlockBytes, x, blockCount, res);
            for (int r = 0; r < 4; r++)
                AssertBitEqual(MatMul.VecDotQ8_0Scalar(w + r * rowBytes, x, blockCount), res[r], $"row-major r={r}");

            MatMul.VecDotQ8_0Sse_4RowsR4(r4, x, blockCount, res);
            for (int r = 0; r < 4; r++)
            {
                AssertBitEqual(MatMul.VecDotQ8_0Scalar(w + r * rowBytes, x, blockCount), res[r], $"R4 r={r}");
                AssertBitEqual(MatMul.VecDotQ8_0ScalarR4(r4, r, x, blockCount), res[r], $"R4 vs ScalarR4 r={r}");
            }
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(r4); NativeMemory.Free(x); }
    }

    // ──────────────────── Dispatch (public drivers) ────────────────────

    /// <summary>
    /// Row-major <c>ComputeRows</c>, R4 <c>ComputeRowsQ8_0Interleaved</c> (single-threaded and
    /// pooled, full groups + tail rows) and <c>GemmR4TiledQ8_0</c>, against the scalar dot per row.
    /// Native: AVX2+ tiers, tolerance. Under <c>DOTNET_EnableAVX=0</c>: the SSSE3 tier, bit-exact.
    /// </summary>
    [SkippableTheory]
    [InlineData(7, 5)]
    [InlineData(13, 18)]
    [InlineData(39, 18)]   // >= ParallelMinRows: pooled path, 9 groups + 3 tail rows
    [InlineData(67, 64)]
    public void Q8_0_Dispatch_MatchesScalarPerRow(int m, int blockCount)
    {
        _output.WriteLine($"ISA: SSSE3={Ssse3.IsSupported} AVX={Avx.IsSupported} AVX2={Avx2.IsSupported} " +
                          $"AVX512BW={Avx512BW.IsSupported} -> SSE tier dispatched: {SseTierDispatched}");
        var rng = new Random(4800 + m * 131 + blockCount);
        int k = blockCount * 32, rowBytes = blockCount * BlockBytes;
        const int n = 3;
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        byte* xs = (byte*)NativeMemory.Alloc((nuint)(n * rowBytes));
        float* expected = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        float* actual = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        float* scale = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        try
        {
            for (int r = 0; r < m; r++) FillQ8Row(rng, w + r * rowBytes, blockCount, allowMinus128: false);
            for (int t = 0; t < n; t++) FillQ8Row(rng, xs + t * rowBytes, blockCount, allowMinus128: false);
            for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                expected[t * m + r] = MatMul.VecDotQ8_0Scalar(w + r * rowBytes, xs + t * rowBytes, blockCount);
                scale[t * m + r] = AbsTermSum(w + r * rowBytes, xs + t * rowBytes, blockCount);
            }

            MatMul.ComputeRows(w, xs, actual, m, blockCount);
            AssertRows(expected, actual, scale, m, "ComputeRows");

            using var rw = WeightRepacking.RepackR4((nint)w, QuantizationType.Q8_0, m, k);
            MatMul.ComputeRowsQ8_0Interleaved((byte*)rw.Ptr, xs, actual, rw.FullGroupCount, rw.TailRows, blockCount);
            AssertRows(expected, actual, scale, m, "Interleaved");

            using (var pool = new ComputeThreadPool(4))
            {
                new Span<float>(actual, m).Clear();
                MatMul.ComputeRowsQ8_0Interleaved((byte*)rw.Ptr, xs, actual, rw.FullGroupCount, rw.TailRows, blockCount, pool);
                AssertRows(expected, actual, scale, m, "Interleaved pooled");

                new Span<float>(actual, n * m).Clear();
                MatMul.GemmR4TiledQ8_0((byte*)rw.Ptr, xs, actual, rw.FullGroupCount, rw.TailRows, blockCount, m, n, pool);
                AssertRows(expected, actual, scale, n * m, "GemmR4Tiled pooled");
            }

            new Span<float>(actual, n * m).Clear();
            MatMul.GemmR4TiledQ8_0((byte*)rw.Ptr, xs, actual, rw.FullGroupCount, rw.TailRows, blockCount, m, n);
            AssertRows(expected, actual, scale, n * m, "GemmR4Tiled");
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.Free(xs);
            NativeMemory.Free(expected); NativeMemory.Free(actual); NativeMemory.Free(scale);
        }
    }

    // ──────────────────── helpers ────────────────────

    /// <summary>Σ_b |dw·dx·sumi_b| — the magnitude that bounds float reassociation error.</summary>
    private static float AbsTermSum(byte* w, byte* x, int blockCount)
    {
        double s = 0;
        for (int b = 0; b < blockCount; b++)
        {
            long sumi = 0;
            for (int i = 0; i < 32; i++) sumi += (sbyte)w[b * BlockBytes + 2 + i] * (sbyte)x[b * BlockBytes + 2 + i];
            s += Math.Abs((double)(float)*(Half*)(w + b * BlockBytes) * (float)*(Half*)(x + b * BlockBytes) * sumi);
        }
        return (float)s;
    }

    private static void AssertRows(float* expected, float* actual, float* scale, int count, string what)
    {
        for (int i = 0; i < count; i++)
        {
            if (SseTierDispatched)
                AssertBitEqual(expected[i], actual[i], $"{what} [{i}]");
            else
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * scale[i] + 1e-30f,
                    $"{what} [{i}]: expected {expected[i]}, got {actual[i]}");
        }
    }

    private static void AssertBitEqual(float expected, float actual, string what) =>
        Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(actual),
            $"{what}: expected {expected:R}, got {actual:R}");

    private static byte* RandomQ8Row(Random rng, int blockCount, bool allowMinus128)
    {
        byte* p = (byte*)NativeMemory.Alloc((nuint)(blockCount * BlockBytes));
        FillQ8Row(rng, p, blockCount, allowMinus128);
        return p;
    }

    /// <summary>
    /// Random Q8_0 row: quants in [-127, 127] (or [-128, 127] for weights), and scales that span
    /// normal magnitudes, both signs, and some Half subnormals.
    /// </summary>
    private static void FillQ8Row(Random rng, byte* p, int blockCount, bool allowMinus128)
    {
        for (int b = 0; b < blockCount; b++)
        {
            byte* blk = p + b * BlockBytes;
            float d = rng.Next(8) == 0
                ? (rng.NextSingle() - 0.5f) * 1e-5f                     // Half subnormal range
                : (rng.NextSingle() * 2f - 1f) * 0.05f;
            *(Half*)blk = (Half)d;
            for (int i = 0; i < 32; i++)
                blk[2 + i] = unchecked((byte)(sbyte)(allowMinus128 ? rng.Next(-128, 128) : rng.Next(-127, 128)));
        }
    }
}
