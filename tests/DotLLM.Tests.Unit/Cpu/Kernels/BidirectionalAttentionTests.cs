using DotLLM.Core.Attention;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1402: the parallel encoder attention must equal the single-threaded span kernel it replaced. Each query row runs
/// the same arithmetic, so the comparison is bit-exact (a cosine gate would also pass a wrong head/row mapping at
/// these sizes only if it were tiny; exact equality removes that doubt). Shapes are discriminating: nh != hd, a
/// sequence length that does not divide the block size, and a length above the one-shot softmax limit.
/// </summary>
public sealed class BidirectionalAttentionTests
{
    [Theory]
    [InlineData(12, 32, 7)]
    [InlineData(12, 32, 64)]
    [InlineData(12, 64, 301)]
    [InlineData(16, 64, 129)]
    [InlineData(3, 32, 1)]
    [InlineData(4, 16, 1500)]   // > one-shot limit: exercises the tiled (online softmax) row path
    public unsafe void Parallel_equals_single_threaded_bit_exact(int nh, int hd, int seq)
    {
        int h = nh * hd;
        var rng = new Random(1402 + seq);
        float[] q = new float[seq * h], k = new float[seq * h], v = new float[seq * h];
        for (int i = 0; i < q.Length; i++)
        {
            q[i] = (float)(rng.NextDouble() * 2 - 1);
            k[i] = (float)(rng.NextDouble() * 2 - 1);
            v[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        float scale = 1f / MathF.Sqrt(hd);

        var expected = new float[seq * h];
        Attention.Execute(q, k, v, expected, seq, seq, nh, nh, hd, 0, scale, default, null, 0f,
                          AttentionMaskMode.Bidirectional, 0, default);

        foreach (int threads in new[] { 1, 3, 8 })
        {
            using var pool = threads > 1 ? new ComputeThreadPool(threads) : null;
            var actual = new float[seq * h];
            fixed (float* pq = q, pk = k, pv = v, po = actual)
                Attention.ExecuteBidirectional(pq, pk, pv, po, seq, nh, hd, scale, pool);
            for (int i = 0; i < actual.Length; i++)
                Assert.True(expected[i] == actual[i], $"threads={threads} nh={nh} hd={hd} seq={seq}: element {i} {actual[i]} != {expected[i]}");
        }
    }
}
