using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>Loads the NVFP4 target on Vulkan once for a class (17.6 GB; the caller holds the Strix GPU lock).</summary>
public sealed class VkInklingFixture : IDisposable
{
    public string? TargetPath { get; } = Inkling2RealModelTests.FindGguf();
    public string? DraftPath { get; } = Inkling2DSparkTests.FindDraft();
    public GgufFile? File { get; private set; }
    public VulkanDevice? Device { get; private set; }
    public VulkanInkling2TransformerModel? Model { get; private set; }
    public DotLLM.Core.Models.ModelConfig? Config { get; private set; }

    public VulkanInkling2TransformerModel Get()
    {
        if (Model is not null) return Model;
        File = GgufFile.Open(TargetPath!);
        Config = GgufModelConfigExtractor.Extract(File.Metadata);
        Device = VulkanDevice.Create();
        Model = VulkanInkling2TransformerModel.BuildFromGguf(Device, File, Config, Inkling2VulkanTests.SpvDir());
        return Model;
    }

    public void Dispose() { Model?.Dispose(); Device?.Dispose(); File?.Dispose(); }
}

/// <summary>Speculative verify needs an N-row forward to equal N single-row forwards bit for bit (#2046; CPU: #2039 section 14).</summary>
public sealed class Inkling2VulkanRowInvarianceTests : IClassFixture<VkInklingFixture>
{
    private readonly ITestOutputHelper _out;
    private readonly VkInklingFixture _fx;
    public Inkling2VulkanRowInvarianceTests(ITestOutputHelper o, VkInklingFixture fx) { _out = o; _fx = fx; }

    internal static unsafe float[] Copy(ITensor t) { var a = new float[t.Shape[0] * t.Shape[1]]; new ReadOnlySpan<float>((void*)t.DataPointer, a.Length).CopyTo(a); return a; }

    internal static int[] Prompt(int n)
    {
        var rnd = new Random(7);
        return Enumerable.Range(0, n).Select(_ => rnd.Next(1000, 100000)).ToArray();
    }

    [SkippableTheory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public void NRowForward_EqualsSingleRowForwards_BitForBit(int rows)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        Skip.If(_fx.TargetPath is null, "model missing");
        var m = _fx.Get();
        int V = _fx.Config!.VocabSize;
        var ids = Prompt(20 + rows);
        var pre = ids[..20]; var tail = ids[20..];
        using (m.Forward(pre, Enumerable.Range(0, 20).ToArray(), -1)) { }
        float[] batch;
        using (var t = m.Forward(tail, Enumerable.Range(20, rows).ToArray(), -1)) batch = Copy(t);
        m.ResetSequenceState();
        using (m.Forward(pre, Enumerable.Range(0, 20).ToArray(), -1)) { }
        var single = new float[rows * V];
        for (int r = 0; r < rows; r++)
            using (var t = m.Forward([tail[r]], [20 + r], -1)) Copy(t).CopyTo(single, r * V);
        int differing = 0, rowsDiff = 0; double maxd = 0;
        for (int r = 0; r < rows; r++)
        {
            int d = 0;
            for (int i = 0; i < V; i++) { float a = batch[r * V + i], b = single[r * V + i]; if (BitConverter.SingleToInt32Bits(a) != BitConverter.SingleToInt32Bits(b)) { d++; maxd = Math.Max(maxd, Math.Abs(a - b)); } }
            if (d > 0) rowsDiff++;
            differing += d;
            _out.WriteLine($"row {r}: {d} of {V} logits differ");
        }
        _out.WriteLine($"rows={rows}: rows differing {rowsDiff}, logits differing {differing}, max |diff| {maxd:E2}");
        Assert.Equal(0, differing);
    }
}
