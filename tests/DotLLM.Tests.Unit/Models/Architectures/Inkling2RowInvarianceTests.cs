using DotLLM.Core.Configuration;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Speculative verify only reproduces plain greedy decoding if an N-row forward equals N single-row forwards bit for bit (issue #2039).
/// </summary>
public sealed class Inkling2RowInvarianceTests
{
    private readonly ITestOutputHelper _output;
    public Inkling2RowInvarianceTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public unsafe void BatchedRows_EqualSingleRowForwards_BitForBit()
    {
        string? gguf = Inkling2RealModelTests.FindGguf();
        Skip.If(gguf is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        using var file = GgufFile.Open(gguf!);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var model = Inkling2TransformerModel.LoadFromGguf(file, cfg, ThreadingConfig.Auto, Inkling2Options.FromEnvironment());
        int[] prompt = Enumerable.Range(3000, 20).Select(x => x * 7 % 50000 + 100).ToArray();
        int[] more = [1234, 5678, 91011, 2222, 33333, 4444, 55555, 666];
        using var a = model.CreateState();
        using var b = model.CreateState();
        using (model.ForwardWithState(prompt, Enumerable.Range(0, 20).ToArray(), -1, a)) { }
        using (model.ForwardWithState(prompt, Enumerable.Range(0, 20).ToArray(), -1, b)) { }
        using var batched = model.ForwardWithState(more, Enumerable.Range(20, 8).ToArray(), -1, a);
        int V = batched.Shape[1];
        int worst = -1; float maxDiff = 0;
        for (int r = 0; r < 8; r++)
        {
            using var single = model.ForwardWithState([more[r]], [20 + r], -1, b);
            var x = new ReadOnlySpan<float>((float*)batched.DataPointer + (long)r * V, V);
            var y = new ReadOnlySpan<float>((void*)single.DataPointer, V);
            int diffs = 0;
            for (int i = 0; i < V; i++) if (BitConverter.SingleToInt32Bits(x[i]) != BitConverter.SingleToInt32Bits(y[i])) { diffs++; maxDiff = Math.Max(maxDiff, Math.Abs(x[i] - y[i])); }
            _output.WriteLine($"row {r}: {diffs} differing logits");
            if (diffs > 0 && worst < 0) worst = r;
        }
        Assert.True(worst < 0, $"first differing row {worst}, max |diff| {maxDiff}");
    }
}
