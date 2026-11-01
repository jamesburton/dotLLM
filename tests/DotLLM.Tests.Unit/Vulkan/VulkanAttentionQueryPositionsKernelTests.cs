using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Per-query-row-position causal attention (#1531 step 3) against a double-precision reference. Positions are non-contiguous and
/// lie inside a KV range that extends past the last query position, the shape every contiguous-row kernel mis-masks.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanAttentionQueryPositionsKernelTests
{
    public static IEnumerable<object[]> Cases()
    {
        // (storage, numHeads, numKvHeads, headDim, slidingWindow, positions)
        foreach (var storage in new[] { VulkanKvStorage.F32, VulkanKvStorage.F16 })
        {
            yield return new object[] { storage, 8, 2, 64, 0, new[] { 3, 4, 17, 40, 41, 59 } };      // scattered subset, GQA 4:1
            yield return new object[] { storage, 4, 4, 32, 0, new[] { 0, 31, 32 } };                  // first row, tile-ish boundary
            yield return new object[] { storage, 8, 2, 64, 12, new[] { 20, 21, 50 } };               // sliding window
            yield return new object[] { storage, 4, 2, 128, 0, new[] { 63, 5, 5 + 20 } };            // unordered rows (row order != position order)
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void AttendsEachRowUpToItsOwnPosition(VulkanKvStorage storage, int numHeads, int numKvHeads, int headDim, int window, int[] positions)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int seqKv = 64;
        int seqQ = positions.Length;
        var rng = new Random(1531 + headDim);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var q = Rand(seqQ * numHeads * headDim);
        var k = Rand(seqKv * numKvHeads * headDim);
        var v = Rand(seqKv * numKvHeads * headDim);
        if (storage == VulkanKvStorage.F16)   // reference reads exactly what the kernel reads
        {
            for (int i = 0; i < k.Length; i++) { k[i] = (float)(Half)k[i]; v[i] = (float)(Half)v[i]; }
        }

        using var dev = VulkanDevice.Create();
        using var kernel = AttentionQueryPositionsKernel.Create(dev, spvDir, storage);
        using var bq = dev.Allocate(q.Length * 4L);
        using var bo = dev.Allocate(q.Length * 4L);
        using var bk = dev.Allocate(k.Length * (long)storage.ElementBytes());
        using var bv = dev.Allocate(v.Length * (long)storage.ElementBytes());
        using var bp = dev.Allocate(seqQ * 4L);
        dev.Upload(q, bq);
        if (storage == VulkanKvStorage.F16)
        {
            var kh = new Half[k.Length]; var vh = new Half[v.Length];
            for (int i = 0; i < k.Length; i++) { kh[i] = (Half)k[i]; vh[i] = (Half)v[i]; }
            dev.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(kh.AsSpan()), bk);
            dev.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(vh.AsSpan()), bv);
        }
        else { dev.Upload(k, bk); dev.Upload(v, bv); }
        dev.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(positions.AsSpan()), bp);

        kernel.Launch(bq, bk, bv, bo, bp, seqQ, seqKv, numHeads, numKvHeads, headDim, slidingWindow: window);
        var got = new float[q.Length];
        dev.Download(bo, got);

        double worst = 0;
        int group = numHeads / numKvHeads;
        double scale = 1.0 / Math.Sqrt(headDim);
        for (int t = 0; t < seqQ; t++)
            for (int h = 0; h < numHeads; h++)
            {
                int hk = h / group, pq = positions[t];
                var scores = new List<(int j, double s)>();
                for (int j = 0; j <= pq; j++)
                {
                    if (window > 0 && pq - j >= window) continue;
                    double s = 0;
                    for (int d = 0; d < headDim; d++) s += (double)q[(t * numHeads + h) * headDim + d] * k[(j * numKvHeads + hk) * headDim + d];
                    scores.Add((j, s * scale));
                }
                double mx = scores.Max(x => x.s), sum = 0;
                foreach (var (_, s) in scores) sum += Math.Exp(s - mx);
                for (int d = 0; d < headDim; d++)
                {
                    double acc = 0;
                    foreach (var (j, s) in scores) acc += Math.Exp(s - mx) / sum * v[(j * numKvHeads + hk) * headDim + d];
                    worst = Math.Max(worst, Math.Abs(acc - got[(t * numHeads + h) * headDim + d]));
                }
            }
        Assert.True(worst < 2e-5, $"max |gpu - reference| = {worst:E2}");
    }
}
