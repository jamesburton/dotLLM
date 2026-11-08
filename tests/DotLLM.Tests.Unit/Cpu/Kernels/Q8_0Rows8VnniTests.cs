using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1119: the eight-row transposed AVX-VNNI Q8_0 kernels and their dispatch. Every assertion is bit-exact against
/// <see cref="MatMul.VecDotQ8_0Scalar"/>: the kernels run the scalar float recurrence (one accumulator per row, multiply then
/// add, block order) over exact integer block sums, so a tolerance would only hide a bug. The previous AVX2/VNNI kernels keep
/// eight float lane partials per row and are therefore NOT bit-exact with the scalar dot, which is what makes the dispatch
/// test discriminate "the new tier ran" from "an old kernel ran" (checked by mutation: forcing the tier off fails it).
/// </summary>
public sealed unsafe class Q8_0Rows8VnniTests
{
    private const int BlockBytes = 34;

    private static void FillQ8Row(Random rng, byte* p, int blockCount)
    {
        for (int b = 0; b < blockCount; b++)
        {
            byte* blk = p + b * BlockBytes;
            float d = rng.Next(8) == 0
                ? (rng.NextSingle() - 0.5f) * 1e-5f                     // Half subnormal range
                : (rng.NextSingle() * 2f - 1f) * 0.05f;
            *(Half*)blk = (Half)d;
            for (int i = 0; i < 32; i++)
            {
                // Mostly full-range values so the int32 bias correction and the 8-lane fold are exercised at their extremes.
                int q = rng.Next(4) == 0 ? (rng.Next(2) == 0 ? -127 : 127) : rng.Next(-127, 128);
                blk[2 + i] = unchecked((byte)(sbyte)q);
            }
        }
    }

    private static void AssertBitEqual(float expected, float actual, string what)
    {
        if (BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(actual))
            Assert.Fail($"{what}: expected {expected:R} (0x{BitConverter.SingleToInt32Bits(expected):X8}) got {actual:R} (0x{BitConverter.SingleToInt32Bits(actual):X8})");
    }

    public static TheoryData<int> BlockCounts() => new() { 1, 2, 3, 7, 18, 64, 97 };

    /// <summary>The eight-row GEMV kernel, including repeated (clamped) row offsets, against the scalar dot per row.</summary>
    [SkippableTheory]
    [MemberData(nameof(BlockCounts))]
    public void Gemv8Rows_BitExactWithScalarPerRow(int blockCount)
    {
        Skip.IfNot(DotLLM.Cpu.CpuFeatures.AvxVnni, "needs AVX-VNNI");
        var rng = new Random(1119 + blockCount);
        int rowBytes = blockCount * BlockBytes;
        byte* w = (byte*)NativeMemory.Alloc((nuint)(8 * rowBytes));
        byte* x = (byte*)NativeMemory.Alloc((nuint)rowBytes);
        int* rowOff = stackalloc int[8];
        float* res = stackalloc float[8];
        try
        {
            for (int r = 0; r < 8; r++) FillQ8Row(rng, w + r * rowBytes, blockCount);
            FillQ8Row(rng, x, blockCount);

            for (int r = 0; r < 8; r++) rowOff[r] = r * rowBytes;
            MatMul.VecDotQ8_0Vnni_8Rows(w, rowOff, x, blockCount, res);
            for (int r = 0; r < 8; r++)
                AssertBitEqual(MatMul.VecDotQ8_0Scalar(w + r * rowBytes, x, blockCount), res[r], $"8-rows r={r} bc={blockCount}");

            // A clamped tail: rows 5..7 point at row 4; lanes 0..4 must still be exact.
            for (int r = 0; r < 8; r++) rowOff[r] = Math.Min(r, 4) * rowBytes;
            MatMul.VecDotQ8_0Vnni_8Rows(w, rowOff, x, blockCount, res);
            for (int r = 0; r < 8; r++)
                AssertBitEqual(MatMul.VecDotQ8_0Scalar(w + Math.Min(r, 4) * rowBytes, x, blockCount), res[r], $"clamped r={r} bc={blockCount}");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(x); }
    }

    /// <summary>
    /// The multi-column kernel through <see cref="MatMul.GemmQ8_0Rows8Vnni"/>: m not a multiple of 8 (clamped tail group), n across
    /// the 32-token chunk boundary and n = 1, serial and pooled, bit-exact per element with the scalar dot.
    /// </summary>
    [SkippableTheory]
    [InlineData(8, 5, 1)]
    [InlineData(13, 7, 3)]
    [InlineData(37, 18, 33)]
    [InlineData(64, 16, 70)]
    [InlineData(70, 3, 64)]
    public void Gemm8Rows_BitExactWithScalarPerElement(int m, int blockCount, int n)
    {
        Skip.IfNot(DotLLM.Cpu.CpuFeatures.AvxVnni, "needs AVX-VNNI");
        var rng = new Random(21119 + m * 131 + blockCount * 7 + n);
        int rowBytes = blockCount * BlockBytes;
        byte* w = (byte*)NativeMemory.Alloc((nuint)(m * rowBytes));
        byte* xs = (byte*)NativeMemory.Alloc((nuint)(n * rowBytes));
        float* c = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        try
        {
            for (int r = 0; r < m; r++) FillQ8Row(rng, w + r * rowBytes, blockCount);
            for (int t = 0; t < n; t++) FillQ8Row(rng, xs + t * rowBytes, blockCount);

            MatMul.GemmQ8_0Rows8Vnni(w, xs, c, m, n, blockCount, pool: null);
            Check(w, xs, c, m, n, blockCount, rowBytes, "serial");

            using var pool = new ComputeThreadPool(4);
            new Span<float>(c, n * m).Clear();
            MatMul.GemmQ8_0Rows8Vnni(w, xs, c, m, n, blockCount, pool);
            Check(w, xs, c, m, n, blockCount, rowBytes, "pooled");
        }
        finally { NativeMemory.Free(w); NativeMemory.Free(xs); NativeMemory.Free(c); }

        static void Check(byte* w, byte* xs, float* c, int m, int n, int blockCount, int rowBytes, string arm)
        {
            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    AssertBitEqual(MatMul.VecDotQ8_0Scalar(w + r * rowBytes, xs + t * rowBytes, blockCount), c[t * m + r],
                        $"{arm} m={m} n={n} bc={blockCount} r={r} t={t}");
        }
    }

    /// <summary>
    /// Every production Q8_0 driver on the eight-row tier is bit-exact with the scalar dot: row-major <c>ComputeRows</c>, serial and
    /// pooled <c>GemmQ8_0</c> (pre-quantized and F32 input), the R4-view GEMV (serial and pooled) and <c>GemmR4TiledQ8_0</c>. This is
    /// the "the new tier ran" check: the previous kernels accumulate eight float lane partials per row and fail it (mutation-verified).
    /// </summary>
    [SkippableTheory]
    [InlineData(7, 5, 3)]
    [InlineData(39, 18, 5)]    // >= ParallelMinRows: pooled paths; 9 groups + 3 tail rows in the R4 view
    [InlineData(67, 64, 40)]   // across the 32-token chunk
    public void Dispatch_OnRows8Tier_BitExactWithScalar(int m, int blockCount, int n)
    {
        Skip.IfNot(MatMul.Q8_0Rows8Tier, "eight-row Q8_0 tier not in force (needs AVX-VNNI and DOTLLM_CPU_Q8_ROWS8 != 0)");
        var rng = new Random(31119 + m * 131 + blockCount + n);
        int k = blockCount * 32, rowBytes = blockCount * BlockBytes;
        byte* w = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        byte* xs = (byte*)NativeMemory.Alloc((nuint)(n * rowBytes));
        float* bf = (float*)NativeMemory.Alloc((nuint)(n * k * sizeof(float)));
        float* expected = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        float* actual = (float*)NativeMemory.Alloc((nuint)(n * m * sizeof(float)));
        try
        {
            for (int r = 0; r < m; r++) FillQ8Row(rng, w + r * rowBytes, blockCount);
            for (int i = 0; i < n * k; i++) bf[i] = (rng.NextSingle() * 2f - 1f) * 3f;
            for (int t = 0; t < n; t++) MatMul.QuantizeF32ToQ8_0(bf + t * k, xs + t * rowBytes, k);
            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    expected[t * m + r] = MatMul.VecDotQ8_0Scalar(w + r * rowBytes, xs + t * rowBytes, blockCount);

            using var pool = new ComputeThreadPool(4);

            MatMul.ComputeRows(w, xs, actual, m, blockCount);
            Compare(expected, actual, m, "ComputeRows");
            MatMul.ComputeRows(w, xs, actual, m, blockCount, pool);
            Compare(expected, actual, m, "ComputeRows pooled");

            MatMul.GemmQ8_0(w, null, actual, m, k, n, xs);
            Compare(expected, actual, n * m, "GemmQ8_0 serial preQ");
            MatMul.GemmQ8_0(w, bf, actual, m, k, n);
            Compare(expected, actual, n * m, "GemmQ8_0 serial f32");
            MatMul.GemmQ8_0(w, null, actual, m, k, n, pool, xs);
            Compare(expected, actual, n * m, "GemmQ8_0 pooled preQ");
            MatMul.GemmQ8_0(w, bf, actual, m, k, n, pool);
            Compare(expected, actual, n * m, "GemmQ8_0 pooled f32");

            using var rw = WeightRepacking.RepackR4((nint)w, QuantizationType.Q8_0, m, k);
            MatMul.ComputeRowsQ8_0Interleaved((byte*)rw.Ptr, xs, actual, rw.FullGroupCount, rw.TailRows, blockCount);
            Compare(expected, actual, m, "Interleaved");
            MatMul.ComputeRowsQ8_0Interleaved((byte*)rw.Ptr, xs, actual, rw.FullGroupCount, rw.TailRows, blockCount, pool);
            Compare(expected, actual, m, "Interleaved pooled");
            MatMul.GemmR4TiledQ8_0((byte*)rw.Ptr, xs, actual, rw.FullGroupCount, rw.TailRows, blockCount, m, n);
            Compare(expected, actual, n * m, "GemmR4Tiled serial");
            MatMul.GemmR4TiledQ8_0((byte*)rw.Ptr, xs, actual, rw.FullGroupCount, rw.TailRows, blockCount, m, n, pool);
            Compare(expected, actual, n * m, "GemmR4Tiled pooled");
            MatMul.GemmR4TiledQ8_0((byte*)rw.Ptr, bf, null, actual, rw.FullGroupCount, rw.TailRows, blockCount, m, k, n, pool);
            Compare(expected, actual, n * m, "GemmR4Tiled pooled f32");
        }
        finally
        {
            NativeMemory.AlignedFree(w); NativeMemory.Free(xs); NativeMemory.Free(bf);
            NativeMemory.Free(expected); NativeMemory.Free(actual);
        }

        static void Compare(float* expected, float* actual, int count, string arm)
        {
            for (int i = 0; i < count; i++)
                AssertBitEqual(expected[i], actual[i], $"{arm} [{i}]");
            new Span<float>(actual, count).Fill(float.NaN);
        }
    }

    /// <summary>The activation tables feed the GEMM's bias correction: <c>128·Σx</c> exactly, and the scale as <c>(float)Half</c>.</summary>
    [SkippableFact]
    public void ActivationTables_MatchScalar()
    {
        Skip.IfNot(DotLLM.Cpu.CpuFeatures.AvxVnni, "needs AVX-VNNI");
        var rng = new Random(41119);
        const int n = 5, blockCount = 9, rowBytes = blockCount * BlockBytes;
        byte* xs = (byte*)NativeMemory.Alloc((nuint)(n * rowBytes));
        float* dx = stackalloc float[n * blockCount];
        int* corr = stackalloc int[n * blockCount];
        try
        {
            for (int t = 0; t < n; t++) FillQ8Row(rng, xs + t * rowBytes, blockCount);
            MatMul.Q8_0ActivationTablesVnni(xs, rowBytes, n, blockCount, dx, corr);
            for (int t = 0; t < n; t++)
                for (int b = 0; b < blockCount; b++)
                {
                    byte* blk = xs + t * rowBytes + b * BlockBytes;
                    int sum = 0;
                    for (int i = 0; i < 32; i++) sum += (sbyte)blk[2 + i];
                    Assert.Equal(128 * sum, corr[t * blockCount + b]);
                    AssertBitEqual((float)*(Half*)blk, dx[t * blockCount + b], $"dx t={t} b={b}");
                }
        }
        finally { NativeMemory.Free(xs); }
    }
}
