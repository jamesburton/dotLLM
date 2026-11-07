using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1570: the blocked causal prefill kernel (KV head x query-row block x key tile, dynamic work items) must be bit-identical to the
/// single-threaded per-row kernel. Shapes are the real LFM3 ones (GQA 16/4, hd128) plus MHA and hd256 (tile 128), with the query
/// block straddling the one-shot / tiled threshold (2048 visible keys), a row count that is not a multiple of the 32-row block, a
/// causal end that is not a multiple of the tile, and a cache longer than the last query position.
/// </summary>
public sealed unsafe class AttentionBlockedPrefillTests : IDisposable
{
    private readonly ComputeThreadPool _pool = new(6);   // not a divisor of the item count: uneven dynamic scheduling

    public void Dispose() => _pool.Dispose();

    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    [Theory]
    [InlineData(16, 4, 128, 300, 1900, 2200)]   // LFM3 shape; rows 1901..2200 visible: straddles the 2048 one-shot bound and a 256-tile edge
    [InlineData(16, 4, 128, 77, 2500, 2577)]    // 77 rows: 3 blocks, the last partial
    [InlineData(16, 4, 128, 2300, 0, 2300)]     // prefill from position 0 crossing the bound inside the forward
    [InlineData(16, 4, 128, 130, 4000, 4500)]   // cache longer than the last query position
    [InlineData(8, 8, 64, 200, 2048, 2248)]     // MHA
    [InlineData(6, 2, 256, 150, 3000, 3150)]    // hd256: tile size 128
    public void Blocked_IsBitIdentical_ToPerRowKernel(int numHeads, int numKvHeads, int headDim, int seqQ, int positionOffset, int seqKv)
    {
        float[] q = Random(seqQ * numHeads * headDim, 1), k = Random(seqKv * numKvHeads * headDim, 2), v = Random(seqKv * numKvHeads * headDim, 3);
        // Scale the queries so the softmax is peaked (large score range): a wrong rescale or boundary moves the output far.
        for (int i = 0; i < q.Length; i++) q[i] *= 4f;
        float[] expected = new float[q.Length];
        Attention.Execute(q, k, v, expected, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset);

        float* outP = (float*)NativeMemory.AlignedAlloc((nuint)(q.Length * sizeof(float)), 64);
        try
        {
            new Span<float>(outP, q.Length).Fill(float.NaN);
            long before = Attention.BlockedPrefillCount;
            fixed (float* qp = q, kp = k, vp = v)
                Attention.Execute(qp, kp, vp, outP, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, _pool);
            Assert.True(Attention.BlockedPrefillCount > before, "the blocked kernel did not run");

            int mismatches = 0; float maxAbs = 0;
            for (int i = 0; i < q.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(outP[i]))
                {
                    mismatches++;
                    maxAbs = MathF.Max(maxAbs, MathF.Abs(expected[i] - outP[i]));
                }
            }
            Assert.True(mismatches == 0, $"{mismatches} of {q.Length} outputs differ (max |d| {maxAbs:E3})");
        }
        finally
        {
            NativeMemory.AlignedFree(outP);
        }
    }

    [Fact]
    public void Blocked_DoesNotRun_ForSlidingWindowOrShortRows()
    {
        const int numHeads = 16, numKvHeads = 4, headDim = 128, seqQ = 64;
        float[] q = Random(seqQ * numHeads * headDim, 1), k = Random(3000 * numKvHeads * headDim, 2), v = Random(3000 * numKvHeads * headDim, 3);
        float* outP = (float*)NativeMemory.AlignedAlloc((nuint)(q.Length * sizeof(float)), 64);
        try
        {
            long before = Attention.BlockedPrefillCount;
            fixed (float* qp = q, kp = k, vp = v)
            {
                Attention.Execute(qp, kp, vp, outP, seqQ, 3000, numHeads, numKvHeads, headDim, 2936, _pool, slidingWindowSize: 256);
                Attention.Execute(qp, kp, vp, outP, seqQ, 2000, numHeads, numKvHeads, headDim, 1936, _pool);
            }
            Assert.Equal(before, Attention.BlockedPrefillCount);
        }
        finally
        {
            NativeMemory.AlignedFree(outP);
        }
    }
}
