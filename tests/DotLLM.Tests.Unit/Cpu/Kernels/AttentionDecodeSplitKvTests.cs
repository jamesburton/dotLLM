using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1927 E2c: split-KV decode attention (fixed 256-key ranges, F16 or F32 partial outputs, F32 max/sum, combine). Not bit-identical
/// to the per-head tiled kernel, so the oracle is that kernel (single-threaded span path) within a tolerance per partial type; the
/// result must be bit-identical across thread counts; and the kernel must engage only where its gate says.
/// </summary>
[Collection("AttentionStaticSwitches")]
public sealed unsafe class AttentionDecodeSplitKvTests
{
    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    /// <summary>
    /// Fixture: random K/V per key and KV head, peaked queries (x4), and the keys of the LAST range scaled up so the running max rises
    /// after range 0 (a combine that ignores the per-range max correction cannot pass: an all-equal-max fixture could).
    /// </summary>
    private static (float[] q, float[] k, float[] v) Fixture(int numHeads, int numKvHeads, int headDim, int seqKv)
    {
        float[] q = Random(numHeads * headDim, 11), k = Random(seqKv * numKvHeads * headDim, 12), v = Random(seqKv * numKvHeads * headDim, 13);
        for (int i = 0; i < q.Length; i++) q[i] *= 4f;
        int lastRange = (seqKv - 1) / 256 * 256;
        for (int i = lastRange * numKvHeads * headDim; i < k.Length; i++) k[i] *= 1.6f;
        return (q, k, v);
    }

    private static float[] Reference(float[] q, float[] k, float[] v, int numHeads, int numKvHeads, int headDim, int seqKv)
    {
        var expected = new float[q.Length];
        Attention.Execute(q, k, v, expected, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1);   // single-threaded per-head TiledRow
        return expected;
    }

    private static float[] Run(float[] q, float[] k, float[] v, int numHeads, int numKvHeads, int headDim, int seqKv, int threads,
                               DecodeSplitKvPartials mode, out long engaged)
    {
        var saved = Attention.DecodeSplitKv;
        var got = new float[q.Length];
        try
        {
            Attention.DecodeSplitKv = mode;
            using var pool = new ComputeThreadPool(threads);
            long before = Attention.DecodeSplitKvCount;
            fixed (float* qp = q, kp = k, vp = v, op = got)
                Attention.Execute(qp, kp, vp, op, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, pool);
            engaged = Attention.DecodeSplitKvCount - before;
        }
        finally { Attention.DecodeSplitKv = saved; }
        return got;
    }

    [Theory]
    [InlineData(32, 8, 128, 2049)]   // Qwen3-4B; 9 ranges, the last holds one key
    [InlineData(32, 8, 128, 7600)]   // 30 ranges, last partial
    [InlineData(14, 2, 64, 4129)]    // GQA group 7, hd64
    [InlineData(8, 8, 128, 3000)]    // MHA
    public void SplitKv_MatchesTiledReference_WithinPartialPrecision(int numHeads, int numKvHeads, int headDim, int seqKv)
    {
        var (q, k, v) = Fixture(numHeads, numKvHeads, headDim, seqKv);
        float[] expected = Reference(q, k, v, numHeads, numKvHeads, headDim, seqKv);
        // F32 partials differ from the online recurrence only by reduction order; F16 partials add one F16 rounding (2^-11 relative)
        // of each range's normalised output (|o| <= 1 here). Measured max |d| on these shapes: F32 1.6-2.2e-7, F16 2.7-8.0e-5.
        foreach (var (mode, tol) in new[] { (DecodeSplitKvPartials.F32, 1e-6f), (DecodeSplitKvPartials.F16, 3e-4f) })
        {
            float[] got = Run(q, k, v, numHeads, numKvHeads, headDim, seqKv, 7, mode, out long engaged);
            Assert.True(engaged >= 1, "the split-KV kernel did not run");   // >=: the counter is process-wide
            float maxAbs = 0; int worst = -1;
            for (int i = 0; i < got.Length; i++)
            {
                float d = MathF.Abs(got[i] - expected[i]);
                if (!(d <= maxAbs)) { maxAbs = d; worst = i; }
            }
            Assert.True(maxAbs <= tol, $"{mode}: max |d| {maxAbs:E3} at {worst} (head {worst / headDim}) exceeds {tol:E1}");
        }
    }

    [Theory]
    [InlineData(DecodeSplitKvPartials.F16)]
    [InlineData(DecodeSplitKvPartials.F32)]
    public void SplitKv_IsThreadCountIndependent(DecodeSplitKvPartials mode)
    {
        const int numHeads = 32, numKvHeads = 8, headDim = 128, seqKv = 5000;
        var (q, k, v) = Fixture(numHeads, numKvHeads, headDim, seqKv);
        float[] baseline = Run(q, k, v, numHeads, numKvHeads, headDim, seqKv, 2, mode, out _);
        foreach (int threads in new[] { 7, 14, 16 })
        {
            float[] got = Run(q, k, v, numHeads, numKvHeads, headDim, seqKv, threads, mode, out _);
            for (int i = 0; i < got.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(baseline[i]) == BitConverter.SingleToInt32Bits(got[i]),
                    $"{mode} threads={threads}: output {i} differs from 2 threads");
        }
    }

    [Fact]
    public void SplitKv_EngagesOnlyWhereGated()
    {
        const int numHeads = 16, numKvHeads = 4, headDim = 64, seqKv = 3000;
        var (q, k, v) = Fixture(numHeads, numKvHeads, headDim, seqKv);
        var saved = Attention.DecodeSplitKv;
        try
        {
            Attention.DecodeSplitKv = DecodeSplitKvPartials.F16;
            using var pool = new ComputeThreadPool(4);
            var outA = new float[q.Length];
            long before = Attention.DecodeSplitKvCount;
            fixed (float* qp = q, kp = k, vp = v, op = outA)
            {
                Attention.Execute(qp, kp, vp, op, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, pool, slidingWindowSize: 4096);
                Attention.Execute(qp, kp, vp, op, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, pool, softCap: 30f);
                Attention.Execute(qp, kp, vp, op, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, pool, sinks: new float[numHeads]);
                Attention.Execute(qp, kp, vp, op, 1, 2048, numHeads, numKvHeads, headDim, 2047, pool);   // one-shot bound
                Assert.Equal(before, Attention.DecodeSplitKvCount);
                Attention.Execute(qp, kp, vp, op, 1, 2049, numHeads, numKvHeads, headDim, 2048, pool);
                Assert.Equal(before + 1, Attention.DecodeSplitKvCount);
                Attention.DecodeSplitKv = DecodeSplitKvPartials.Off;
                Attention.Execute(qp, kp, vp, op, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, pool);
                Assert.Equal(before + 1, Attention.DecodeSplitKvCount);
            }
        }
        finally { Attention.DecodeSplitKv = saved; }
    }
}
