using DotLLM.Core.Attention;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1572: the KV start LFM3's Vulkan model hands the single-row split-KV pass for a sliding-window layer. A start that is too large drops live
/// keys; one that is too small only wastes work, so the tests pin the exact value.
/// </summary>
public sealed class Lfm3SlidingKvStartTests
{
    [Theory]
    [InlineData(0, 1, 256, 0)]        // first token
    [InlineData(254, 1, 256, 0)]      // pos 254: 255 keys, fits
    [InlineData(255, 1, 256, 0)]      // pos 255: 256 keys == window, still all of them
    [InlineData(256, 1, 256, 1)]      // pos 256: key 0 just left the window
    [InlineData(1000, 1, 256, 745)]   // keys 745..1000 = 256
    [InlineData(1000, 1, 0, 0)]       // global layer
    [InlineData(1000, 5, 256, 749)]   // last row of a 5-row forward is pos 1004
    public void KvStart_IsLastRowWindowStart(int posOffset, int tokens, int window, int expected)
        => Assert.Equal(expected, VulkanLfm3TransformerModel.SlidingKvStart(posOffset, tokens, window));

    [Theory]
    [InlineData(1000, 256)]
    [InlineData(300, 256)]
    [InlineData(5000, 100)]
    public void KvStart_KeepsExactlyWindowKeys(int pos, int window)
    {
        int start = VulkanLfm3TransformerModel.SlidingKvStart(pos, 1, window);
        Assert.Equal(window, pos + 1 - start);
    }
}

/// <summary>#1572: single-row split-KV decode with the window-clipped range (kvStart) against the CPU sliding-window reference, LFM3 geometry.</summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Lfm3SplitKvWindowTests
{
    [SkippableTheory]
    [InlineData(100, false)]    // pos < window: kvStart 0
    [InlineData(256, false)]    // seqKv == window
    [InlineData(257, false)]    // key 0 just dropped
    [InlineData(1000, false)]
    [InlineData(4001, false)]   // deep
    [InlineData(1000, true)]    // GQA-grouped split pass
    [InlineData(4001, true)]
    public void SplitKvWithKvStart_MatchesCpuSlidingWindow(int seqKv, bool gqa)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int numHeads = 16, numKvHeads = 8, headDim = 128, window = 256;
        using var device = VulkanDevice.Create();
        using var split = VulkanSplitKvAttentionKernel.Create(device, spvDir);
        split.PreferGqa = gqa;
        Skip.If(gqa && !split.UsesGqa(numHeads, numKvHeads, headDim), "GQA-grouped split pass unavailable");

        var rng = new Random(0x1572 + seqKv);
        float[] Rand(int n) { var r = new float[n]; for (int i = 0; i < n; i++) r[i] = (float)(rng.NextDouble() * 2 - 1); return r; }
        float[] q = Rand(numHeads * headDim), k = Rand(seqKv * numKvHeads * headDim), v = Rand(seqKv * numKvHeads * headDim);
        int pos = seqKv - 1;
        int kvStart = VulkanLfm3TransformerModel.SlidingKvStart(pos, 1, window);
        float scale = 1f / MathF.Sqrt(headDim);
        float[] expected = new float[q.Length];
        Attention.ExecuteScalar(q, k, v, expected, 1, seqKv, numHeads, numKvHeads, headDim, pos, scale, default, window, 0f, AttentionMaskMode.Causal, 0);
        if (seqKv > window)
        {
            // Discrimination: the window must change the answer, else an off-by-one in the clipped range is invisible.
            float[] noWin = new float[q.Length];
            Attention.ExecuteScalar(q, k, v, noWin, 1, seqKv, numHeads, numKvHeads, headDim, pos, scale, default, null, 0f, AttentionMaskMode.Causal, 0);
            float gap = 0;
            for (int i = 0; i < noWin.Length; i++) gap = MathF.Max(gap, MathF.Abs(noWin[i] - expected[i]));
            Assert.True(gap > 1e-3f, $"window does not change the answer (gap {gap})");
        }

        using var qB = device.Allocate(q.Length * 4L);
        using var kB = device.Allocate(k.Length * 4L);
        using var vB = device.Allocate(v.Length * 4L);
        using var oB = device.Allocate(q.Length * 4L);
        device.Upload(q, qB); device.Upload(k, kB); device.Upload(v, vB);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            split.Record(ctx.CommandBuffer, qB, kB, vB, oB, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: pos, slidingWindow: window, kvStart: kvStart);
            ctx.SubmitAndWait();
        }
        float[] actual = new float[q.Length];
        device.Download(oB, actual);
        float maxAbs = expected.Max(MathF.Abs);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= 1e-4f * maxAbs + 1e-5f, $"i={i} (head {i / headDim}): cpu {expected[i]:G7} vs split {actual[i]:G7}");
    }
}
