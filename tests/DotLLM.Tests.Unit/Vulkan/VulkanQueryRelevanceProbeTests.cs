using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Query-relevance probe kernels (#1531 step 4) against a double-precision reference of the CPU definition
/// (<c>QueryAttention.Probabilities</c> in the chunk-reuse lab): score(j) = mean over layers, query rows and heads of the causal softmax
/// probability of key j. Query rows sit AFTER the scored context, so their own keys are in the softmax denominator but not in the output.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQueryRelevanceProbeTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var storage in new[] { VulkanKvStorage.F32, VulkanKvStorage.F16 })
        {
            // (storage, numHeads, numKvHeads, headDim, contextKeys, queryPositions)
            yield return new object[] { storage, 8, 2, 64, 40, new[] { 40, 41, 42, 43 } };   // GQA 4:1, rows right after the context
            yield return new object[] { storage, 4, 4, 32, 25, new[] { 25, 26 } };           // MHA
            yield return new object[] { storage, 8, 2, 128, 33, new[] { 10, 33, 50 } };      // rows inside / after the context, unordered caps
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Scores_MatchReference_AcrossAccumulatedLayers(VulkanKvStorage storage, int numHeads, int numKvHeads, int headDim, int contextKeys, int[] positions)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int layers = 3;
        int rows = positions.Length;
        int cacheRows = positions.Max() + 1;
        float scale = 1f / MathF.Sqrt(headDim);
        var rng = new Random(1531 + headDim);
        float[] Rand(int n, float s) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * s); return a; }

        using var dev = VulkanDevice.Create();
        using var probe = VulkanQueryRelevanceProbe.Create(dev, spvDir, storage);
        using var bpos = dev.Allocate(rows * 4L);
        dev.Upload(MemoryMarshal.AsBytes(positions.AsSpan()), bpos);
        probe.Begin(contextKeys, rows, numHeads);

        var expected = new double[contextKeys];
        for (int layer = 0; layer < layers; layer++)
        {
            var q = Rand(rows * numHeads * headDim, 2f);
            var k = Rand(cacheRows * numKvHeads * headDim, 2f);
            if (storage == VulkanKvStorage.F16) for (int i = 0; i < k.Length; i++) k[i] = (float)(Half)k[i];
            using var bq = dev.Allocate(q.Length * 4L);
            using var bk = dev.Allocate(k.Length * (long)storage.ElementBytes());
            dev.Upload(q, bq);
            if (storage == VulkanKvStorage.F16)
            {
                var kh = new Half[k.Length];
                for (int i = 0; i < k.Length; i++) kh[i] = (Half)k[i];
                dev.Upload(MemoryMarshal.AsBytes(kh.AsSpan()), bk);
            }
            else dev.Upload(k, bk);
            probe.ProbeLayer(bq, bk, bk, bpos, rows, numHeads, numKvHeads, headDim, scale);

            int group = numHeads / numKvHeads;
            for (int r = 0; r < rows; r++)
                for (int h = 0; h < numHeads; h++)
                {
                    int visible = positions[r] + 1, hk = h / group;
                    var s = new double[visible];
                    for (int j = 0; j < visible; j++)
                    {
                        double dot = 0;
                        for (int d = 0; d < headDim; d++) dot += (double)q[(r * numHeads + h) * headDim + d] * k[(j * numKvHeads + hk) * headDim + d];
                        s[j] = dot * scale;
                    }
                    double mx = s.Max(), sum = s.Sum(x => Math.Exp(x - mx));
                    for (int j = 0; j < Math.Min(visible, contextKeys); j++)
                        expected[j] += Math.Exp(s[j] - mx) / sum / numHeads;
                }
        }
        float[] got = probe.ReadScores();
        double worst = 0;
        for (int j = 0; j < contextKeys; j++)
            worst = Math.Max(worst, Math.Abs(got[j] - expected[j] / (layers * rows)));
        Assert.True(worst < 2e-6, $"max |gpu - reference| = {worst:E2} (scores ~ {1.0 / contextKeys:E1})");
    }
}
