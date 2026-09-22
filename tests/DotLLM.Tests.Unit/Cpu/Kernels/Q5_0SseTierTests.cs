using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Issue #477: the 128-bit (SSSE3) Q5_0 x Q8_1 tier (<c>MatMul.Q5_0Sse.cs</c>) — the path every
/// Q5_0 GEMV/GEMM takes on Westmere-class hardware (SmolLM's Q4_K_M file stores all of its
/// k = 576 projections as Q5_0).
///
/// <para>The tier is <b>bit-exact with the scalar tier</b>, so the kernel tests compare float bit
/// patterns. An independent exact integer reference (unit scales) pins the 5-bit unpack and the
/// −16 offset handling without relying on the scalar kernel.</para>
///
/// <para>The <c>*_Dispatch_*</c> tests go through the public drivers; under
/// <c>DOTNET_EnableAVX=0</c> (Westmere's ISA) they take the 128-bit tier and additionally assert
/// bit-exactness with scalar. Because SSE and scalar agree bit for bit, a dispatch test alone
/// cannot prove the 128-bit tier ran — mutating the SSE kernel and watching the emulated dispatch
/// tests fail does (see issue #477).</para>
/// </summary>
public sealed unsafe class Q5_0SseTierTests
{
    private const int Q5Bytes = 22;
    private const int Q81Bytes = 36;
    private readonly ITestOutputHelper _output;
    public Q5_0SseTierTests(ITestOutputHelper output) => _output = output;

    private static bool SseTierDispatched => Ssse3.IsSupported && !Avx2.IsSupported;

    // ──────────────────── Kernels (bit-exact) ────────────────────

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(18)]   // k = 576 (SmolLM-135M hidden)
    [InlineData(48)]   // k = 1536 (SmolLM-135M ffn_down)
    [InlineData(137)]  // odd, 4-block remainder of 1
    public void VecDotQ5_0Q8_1Sse_BitExactWithScalar(int blockCount)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(4770 + blockCount);
        byte* w = (byte*)NativeMemory.Alloc((nuint)(blockCount * Q5Bytes));
        byte* x = (byte*)NativeMemory.Alloc((nuint)(blockCount * Q81Bytes));
        byte* xq = (byte*)NativeMemory.Alloc((nuint)(blockCount * Q81Bytes));
        try
        {
            FillQ5Row(rng, w, blockCount);
            FillRawQ8_1Row(rng, x, blockCount);          // raw bytes incl. -128: no precondition
            QuantizedQ8_1Row(rng, xq, blockCount);         // realistic, from the quantizer

            AssertBitEqual(MatMul.VecDotQ5_0Q8_1Scalar(w, x, blockCount),
                MatMul.VecDotQ5_0Q8_1Sse(w, x, blockCount), $"raw blockCount={blockCount}");
            AssertBitEqual(MatMul.VecDotQ5_0Q8_1Scalar(w, xq, blockCount),
                MatMul.VecDotQ5_0Q8_1Sse(w, xq, blockCount), $"quantized blockCount={blockCount}");

            if (Avx2.IsSupported)
            {
                float s = MatMul.VecDotQ5_0Q8_1Sse(w, xq, blockCount);
                float a = MatMul.VecDotQ5_0Q8_1Avx2(w, xq, blockCount);
                float mag = AbsTermSum(w, xq, blockCount);
                Assert.True(Math.Abs(s - a) <= 1e-5f * mag + 1e-30f, $"SSE {s} vs AVX2 {a} (mag {mag})");
            }
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); NativeMemory.Free(xq); }
    }

    /// <summary>
    /// Exact integer reference, independent of every kernel: d5 = d8 = 1 and activations in
    /// [-64, 63] so that <c>s8 = (Half)Σq8</c> is exact (|Σ| ≤ 2048); the dot is then the integer
    /// Σ (q5u − 16)·q8, exactly representable while below 2^24.
    /// </summary>
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(18)]
    [InlineData(33)]
    public void Q5_0Sse_UnitScales_EqualsExactIntegerDot(int blockCount)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(4780 + blockCount);
        const int m = 4;
        int rowBytes = blockCount * Q5Bytes;
        byte* w = (byte*)NativeMemory.Alloc((nuint)(m * rowBytes));
        byte* x = (byte*)NativeMemory.Alloc((nuint)(blockCount * Q81Bytes));
        float* res = stackalloc float[m];
        try
        {
            for (int r = 0; r < m; r++)
            {
                byte* row = w + r * rowBytes;
                FillQ5Row(rng, row, blockCount);
                for (int b = 0; b < blockCount; b++) *(Half*)(row + b * Q5Bytes) = (Half)1f;
            }
            // Corners: element 0 all-ones (31) and element 31 zero (0) in row 0 block 0.
            *(uint*)(w + 2) = (*(uint*)(w + 2) | 1u) & 0x7FFFFFFFu;
            w[6] = (byte)((w[6] & 0xF0) | 0x0F);
            w[6 + 15] = (byte)(w[6 + 15] & 0x0F);
            for (int b = 0; b < blockCount; b++)
            {
                byte* blk = x + b * Q81Bytes;
                int sum = 0;
                for (int i = 0; i < 32; i++)
                {
                    int v = rng.Next(-64, 64);
                    blk[4 + i] = unchecked((byte)(sbyte)v);
                    sum += v;
                }
                *(Half*)blk = (Half)1f;
                *(Half*)(blk + 2) = (Half)sum;
                Assert.Equal((float)sum, (float)*(Half*)(blk + 2));
            }

            MatMul.VecDotQ5_0Q8_1Sse_4Rows(w, rowBytes, Q5Bytes, x, blockCount, res);
            for (int r = 0; r < m; r++)
            {
                byte* row = w + r * rowBytes;
                long exact = 0;
                for (int b = 0; b < blockCount; b++)
                {
                    byte* blk = row + b * Q5Bytes;
                    uint qh = *(uint*)(blk + 2);
                    for (int j = 0; j < 16; j++)
                    {
                        int v0 = (blk[6 + j] & 0xF) | (int)(((qh >> j) & 1) << 4);
                        int v1 = (blk[6 + j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4);
                        exact += (v0 - 16) * (sbyte)x[b * Q81Bytes + 4 + j];
                        exact += (v1 - 16) * (sbyte)x[b * Q81Bytes + 4 + 16 + j];
                    }
                }
                Assert.True(Math.Abs(exact) < (1 << 24));
                Assert.Equal((float)exact, res[r]);
                Assert.Equal((float)exact, MatMul.VecDotQ5_0Q8_1Sse(row, x, blockCount));
            }
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(18)]
    [InlineData(48)]
    [InlineData(137)]
    public void VecDotQ5_0Q8_1Sse_4Rows_RowMajorAndR4_BitExactWithScalar(int blockCount)
    {
        Skip.IfNot(Ssse3.IsSupported, "needs SSSE3");
        var rng = new Random(4790 + blockCount);
        int rowBytes = blockCount * Q5Bytes;
        byte* w = (byte*)NativeMemory.Alloc((nuint)(4 * rowBytes));
        byte* r4 = (byte*)NativeMemory.Alloc((nuint)(4 * rowBytes));
        byte* x = (byte*)NativeMemory.Alloc((nuint)(blockCount * Q81Bytes));
        float* res = stackalloc float[4];
        try
        {
            for (int r = 0; r < 4; r++) FillQ5Row(rng, w + r * rowBytes, blockCount);
            FillRawQ8_1Row(rng, x, blockCount);
            for (int b = 0; b < blockCount; b++)
            for (int r = 0; r < 4; r++)
                Buffer.MemoryCopy(w + r * rowBytes + b * Q5Bytes, r4 + (b * 4 + r) * Q5Bytes, Q5Bytes, Q5Bytes);

            MatMul.VecDotQ5_0Q8_1Sse_4Rows(w, rowBytes, Q5Bytes, x, blockCount, res);
            for (int r = 0; r < 4; r++)
                AssertBitEqual(MatMul.VecDotQ5_0Q8_1Scalar(w + r * rowBytes, x, blockCount), res[r], $"row-major r={r}");

            new Span<float>(res, 4).Clear();
            MatMul.VecDotQ5_0Q8_1Sse_4RowsR4(r4, x, blockCount, res);
            for (int r = 0; r < 4; r++)
            {
                AssertBitEqual(MatMul.VecDotQ5_0Q8_1Scalar(w + r * rowBytes, x, blockCount), res[r], $"R4 r={r}");
                AssertBitEqual(MatMul.VecDotQ5_0Q8_1ScalarR4(r4, r, x, blockCount), res[r], $"R4 vs ScalarR4 r={r}");
            }
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(r4); NativeMemory.Free(x); }
    }

    // ──────────────────── Dispatch (public drivers) ────────────────────

    /// <summary>
    /// Row-major <c>ComputeRowsQ5_0</c>, R4 <c>ComputeRowsQ5_0Interleaved</c> (single-threaded and
    /// pooled, full groups + tail rows), <c>GemmQ5_0</c> (pre-quantized and not, pooled and not)
    /// and <c>OuterProductGemmQ5_0</c>, against the scalar dot per row. Native: AVX2 tier,
    /// tolerance. Under <c>DOTNET_EnableAVX=0</c>: the SSSE3 tier, bit-exact.
    /// </summary>
    [SkippableTheory]
    [InlineData(7, 5)]
    [InlineData(13, 18)]
    [InlineData(39, 18)]   // >= ParallelMinRows: pooled path, 9 groups + 3 tail rows
    [InlineData(67, 48)]
    public void Q5_0_Dispatch_MatchesScalarPerRow(int m, int blockCount)
    {
        _output.WriteLine($"ISA: SSSE3={Ssse3.IsSupported} AVX={Avx.IsSupported} AVX2={Avx2.IsSupported} " +
                          $"AVX512BW={Avx512BW.IsSupported} -> SSE tier dispatched: {SseTierDispatched}");
        var rng = new Random(4800 + m * 131 + blockCount);
        int k = blockCount * 32, rowBytes = blockCount * Q5Bytes, xRowBytes = blockCount * Q81Bytes;
        const int n = 4; // 3-token outer-product batch + 1 remainder token
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        float* xf = (float*)NativeMemory.Alloc((nuint)(n * k * sizeof(float)));
        byte* xs = (byte*)NativeMemory.Alloc((nuint)(n * xRowBytes));
        float* expected = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        float* actual = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        float* scale = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        try
        {
            for (int r = 0; r < m; r++) FillQ5Row(rng, w + r * rowBytes, blockCount);
            for (int i = 0; i < n * k; i++) xf[i] = (rng.NextSingle() * 2f - 1f) * 3f;
            for (int t = 0; t < n; t++) MatMul.QuantizeF32ToQ8_1(xf + t * k, xs + t * xRowBytes, k);
            for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                expected[t * m + r] = MatMul.VecDotQ5_0Q8_1Scalar(w + r * rowBytes, xs + t * xRowBytes, blockCount);
                scale[t * m + r] = AbsTermSum(w + r * rowBytes, xs + t * xRowBytes, blockCount);
            }

            MatMul.ComputeRowsQ5_0(w, xs, actual, m, blockCount);
            AssertRows(expected, actual, scale, m, "ComputeRowsQ5_0");

            new Span<float>(actual, m).Clear();
            MatMul.GemvQ5_0(w, xf, actual, m, k);
            AssertRows(expected, actual, scale, m, "GemvQ5_0");

            new Span<float>(actual, n * m).Clear();
            MatMul.GemmQ5_0(w, xf, actual, m, k, n);
            AssertRows(expected, actual, scale, n * m, "GemmQ5_0");

            new Span<float>(actual, n * m).Clear();
            MatMul.GemmQ5_0(w, xf, actual, m, k, n, xs);
            AssertRows(expected, actual, scale, n * m, "GemmQ5_0 pre-quantized");

            using var rw = WeightRepacking.RepackR4((nint)w, QuantizationType.Q5_0, m, k);
            new Span<float>(actual, m).Clear();
            MatMul.ComputeRowsQ5_0Interleaved((byte*)rw.Ptr, xs, actual, rw.FullGroupCount, rw.TailRows, blockCount);
            AssertRows(expected, actual, scale, m, "Interleaved");

            new Span<float>(actual, n * m).Clear();
            MatMul.OuterProductGemmQ5_0((byte*)rw.Ptr, xs, actual, rw.FullGroupCount, rw.TailRows, blockCount, m, n);
            AssertRows(expected, actual, scale, n * m, "OuterProductGemm");

            using (var pool = new ComputeThreadPool(4))
            {
                new Span<float>(actual, m).Clear();
                MatMul.ComputeRowsQ5_0Interleaved((byte*)rw.Ptr, xs, actual, rw.FullGroupCount, rw.TailRows, blockCount, pool);
                AssertRows(expected, actual, scale, m, "Interleaved pooled");

                new Span<float>(actual, m).Clear();
                MatMul.GemvQ5_0(w, xf, actual, m, k, pool);
                AssertRows(expected, actual, scale, m, "GemvQ5_0 pooled");

                new Span<float>(actual, n * m).Clear();
                MatMul.GemmQ5_0(w, xf, actual, m, k, n, pool);
                AssertRows(expected, actual, scale, n * m, "GemmQ5_0 pooled");

                new Span<float>(actual, n * m).Clear();
                MatMul.OuterProductGemmQ5_0((byte*)rw.Ptr, xs, actual, rw.FullGroupCount, rw.TailRows, blockCount, m, n, pool);
                AssertRows(expected, actual, scale, n * m, "OuterProductGemm pooled");
            }
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.Free(xf); NativeMemory.Free(xs);
            NativeMemory.Free(expected); NativeMemory.Free(actual); NativeMemory.Free(scale);
        }
    }

    // ──────────────────── helpers ────────────────────

    /// <summary>Σ_b |d5·d8·sumi_b| + 16·|d5·s8| — the magnitude that bounds float reassociation error.</summary>
    private static float AbsTermSum(byte* w, byte* x, int blockCount)
    {
        double s = 0;
        for (int b = 0; b < blockCount; b++)
        {
            byte* blk = w + b * Q5Bytes;
            byte* xb = x + b * Q81Bytes;
            uint qh = *(uint*)(blk + 2);
            long sumi = 0;
            for (int j = 0; j < 16; j++)
            {
                int v0 = (blk[6 + j] & 0xF) | (int)(((qh >> j) & 1) << 4);
                int v1 = (blk[6 + j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4);
                sumi += v0 * (sbyte)xb[4 + j] + v1 * (sbyte)xb[4 + 16 + j];
            }
            double d5 = (float)*(Half*)blk;
            s += Math.Abs(d5 * (float)*(Half*)xb * sumi) + 16 * Math.Abs(d5 * (float)*(Half*)(xb + 2));
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

    /// <summary>Random Q5_0 row: random qh/qs bits, scales spanning normals, both signs and Half subnormals.</summary>
    private static void FillQ5Row(Random rng, byte* p, int blockCount)
    {
        for (int b = 0; b < blockCount; b++)
        {
            byte* blk = p + b * Q5Bytes;
            float d = rng.Next(8) == 0
                ? (rng.NextSingle() - 0.5f) * 1e-5f
                : (rng.NextSingle() * 2f - 1f) * 0.05f;
            *(Half*)blk = (Half)d;
            *(uint*)(blk + 2) = (uint)rng.Next() ^ ((uint)rng.Next() << 1);
            for (int i = 0; i < 16; i++) blk[6 + i] = (byte)rng.Next(256);
        }
    }

    /// <summary>Raw Q8_1 row: arbitrary quant bytes (incl. -128) and finite d/s, not self-consistent.</summary>
    private static void FillRawQ8_1Row(Random rng, byte* p, int blockCount)
    {
        for (int b = 0; b < blockCount; b++)
        {
            byte* blk = p + b * Q81Bytes;
            *(Half*)blk = (Half)((rng.NextSingle() * 2f - 1f) * 0.05f);
            *(Half*)(blk + 2) = (Half)((rng.NextSingle() * 2f - 1f) * 20f);
            for (int i = 0; i < 32; i++) blk[4 + i] = (byte)rng.Next(256);
        }
    }

    private static void QuantizedQ8_1Row(Random rng, byte* p, int blockCount)
    {
        int k = blockCount * 32;
        float* xf = (float*)NativeMemory.Alloc((nuint)(k * sizeof(float)));
        try
        {
            for (int i = 0; i < k; i++) xf[i] = (rng.NextSingle() * 2f - 1f) * 3f;
            MatMul.QuantizeF32ToQ8_1(xf, p, k);
        }
        finally { NativeMemory.Free(xf); }
    }
}
