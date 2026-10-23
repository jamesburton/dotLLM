using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #915: gpt-oss decode attention on the split-KV flash-decoding kernel with the attention sink folded into the merge pass and the KV range
/// clipped to the sliding window, versus the per-token sink kernel it replaces (<c>attention_sinks_f32.comp</c>, the reference). Same inputs, same
/// sink logits; only the summation order differs, so agreement is tight (1e-4 relative). Cases cover a windowed layer deep in the context (the
/// clipped range is a strict suffix), a dense layer, a window longer than the context, and a sink so large it dominates the softmax denominator.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanSinkSplitKvAttentionTests
{
    [SkippableTheory]
    [InlineData(700, 128, 64, 8, 64, 0.5f)]      // windowed layer, gpt-oss head geometry
    [InlineData(513, 0, 64, 8, 64, 0.0f)]        // dense layer
    [InlineData(100, 128, 64, 8, 64, -1.0f)]     // window longer than the context: kvStart = 0
    [InlineData(1025, 128, 64, 8, 64, 4.0f)]     // sink dominates the denominator (e^4 x the typical key weight)
    [InlineData(300, 0, 12, 4, 128, 1.0f)]       // other geometry (GQA 3, headDim 128)
    [InlineData(40, 0, 64, 8, 64, 0.0f)]         // short context, small split count
    public void SplitKvWithSinks_MatchesSinkKernel(int seqKv, int window, int numHeads, int numKvHeads, int headDim, float sinkBias)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var gpt = VulkanGptOssKernels.Create(device, spvDir);
        using var split = VulkanSplitKvAttentionKernel.Create(device, spvDir);

        var rng = new Random(0x915 + seqKv + window);
        float[] q = Rand(rng, numHeads * headDim, 1f);
        float[] k = Rand(rng, seqKv * numKvHeads * headDim, 1f);
        float[] v = Rand(rng, seqKv * numKvHeads * headDim, 1f);
        float[] sinks = new float[numHeads];
        for (int h = 0; h < numHeads; h++) sinks[h] = sinkBias + (float)(rng.NextDouble() * 2 - 1);
        int pos = seqKv - 1;
        int kvStart = window > 0 ? Math.Max(0, pos - window + 1) : 0;

        using var qB = device.Allocate(q.Length * 4L);
        using var kB = device.Allocate(k.Length * 4L);
        using var vB = device.Allocate(v.Length * 4L);
        using var sB = device.Allocate(sinks.Length * 4L);
        using var refOut = device.Allocate(q.Length * 4L);
        using var splitOut = device.Allocate(q.Length * 4L);
        device.Upload(q, qB); device.Upload(k, kB); device.Upload(v, vB); device.Upload(sinks, sB);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            gpt.RecordAttentionSinks(ctx.CommandBuffer, qB, kB, vB, refOut, sB, 1, seqKv, numHeads, numKvHeads, headDim, pos, window);
            split.Record(ctx.CommandBuffer, qB, kB, vB, splitOut, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: pos, slidingWindow: window,
                sinks: sB, kvStart: kvStart);
            ctx.SubmitAndWait();
        }
        float[] a = new float[q.Length], b = new float[q.Length];
        device.Download(refOut, a);
        device.Download(splitOut, b);
        Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(seqKv - kvStart, numHeads) || seqKv < 64, "case does not exercise a real split");
        float maxAbs = a.Max(MathF.Abs);
        Assert.True(maxAbs > 1e-3f, "reference degenerate");
        for (int i = 0; i < a.Length; i++)
            Assert.True(MathF.Abs(a[i] - b[i]) <= 1e-4f * maxAbs + 1e-6f, $"i={i} (head {i / headDim}): sink kernel {a[i]:G7} vs split+sink {b[i]:G7}");
    }

    /// <summary>Control: the sink must matter (a sink-less split-KV launch differs from the sink kernel), so the comparison above cannot pass by ignoring sinks.</summary>
    [SkippableFact]
    public void WithoutSinks_SplitKvDiffersFromSinkKernel()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var gpt = VulkanGptOssKernels.Create(device, spvDir);
        using var split = VulkanSplitKvAttentionKernel.Create(device, spvDir);
        const int seqKv = 400, numHeads = 64, numKvHeads = 8, headDim = 64;
        var rng = new Random(77);
        float[] q = Rand(rng, numHeads * headDim, 1f), k = Rand(rng, seqKv * numKvHeads * headDim, 1f), v = Rand(rng, seqKv * numKvHeads * headDim, 1f);
        float[] sinks = Enumerable.Repeat(8.0f, numHeads).ToArray();   // large sink: denominators are visibly inflated
        using var qB = device.Allocate(q.Length * 4L);
        using var kB = device.Allocate(k.Length * 4L);
        using var vB = device.Allocate(v.Length * 4L);
        using var sB = device.Allocate(sinks.Length * 4L);
        using var refOut = device.Allocate(q.Length * 4L);
        using var noSink = device.Allocate(q.Length * 4L);
        device.Upload(q, qB); device.Upload(k, kB); device.Upload(v, vB); device.Upload(sinks, sB);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            gpt.RecordAttentionSinks(ctx.CommandBuffer, qB, kB, vB, refOut, sB, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, 0);
            split.Record(ctx.CommandBuffer, qB, kB, vB, noSink, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: seqKv - 1);
            ctx.SubmitAndWait();
        }
        float[] a = new float[q.Length], b = new float[q.Length];
        device.Download(refOut, a);
        device.Download(noSink, b);
        float worst = 0;
        for (int i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        Assert.True(worst > 0.01f * a.Max(MathF.Abs), $"sink-less output is indistinguishable from the sink output (worst diff {worst})");
    }

    private static float[] Rand(Random rng, int n, float amp)
    {
        float[] r = new float[n];
        for (int i = 0; i < n; i++) r[i] = (float)((rng.NextDouble() * 2 - 1) * amp);
        return r;
    }
}
