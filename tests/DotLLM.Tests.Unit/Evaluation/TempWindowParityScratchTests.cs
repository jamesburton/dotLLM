using DotLLM.Core.Configuration;
using DotLLM.Core.Evaluation;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Evaluation;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Evaluation;

// THROWAWAY verification harness for issue #395 — delete after reporting.
public sealed class TempWindowParityScratchTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _scratch = CreateScratch();

    private static string CreateScratch()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"dotllm-395-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return dir;
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void SplitWindows_MatchWholeModel()
    {
        string path = SyntheticQwen35MoeGguf.Write(
            Path.Combine(_scratch, "qwen35moe-4layer.gguf"), blockCount: 4);

        using GgufFile gguf = GgufFile.Open(path);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using IModel model = ModelLoader.CreateCpuModelFromGguf(gguf, config);

        Assert.Equal(4, config.NumLayers);
        Assert.True(model.RequiresPerSequenceState);

        int[] tokens = [1, 2, 3, 4, 5, 6];
        int[] positions = [0, 1, 2, 3, 4, 5];
        int seqLen = tokens.Length;
        int hidden = config.HiddenSize;
        int vocab = config.VocabSize;

        // Whole-model reference.
        model.ResetSequenceState();
        using ITensor reference = model.Forward(tokens, positions, -1);
        var refLogits = new float[seqLen * vocab];
        unsafe
        {
            new ReadOnlySpan<float>((void*)reference.DataPointer, refLogits.Length).CopyTo(refLogits);
        }

        // Cycled: [0..2) then [2..4). Layer 0 and layer 2 are both GDN (full_attention_interval=2),
        // so a cut at 2 puts GDN ordinal 0 on one side and ordinal 1 on the other.
        using var windowModel = new CpuLayerWindowModel(model, config);
        var h1 = new float[seqLen * hidden];
        var h2 = new float[seqLen * hidden];

        using (ILayerWindowExecutor a = windowModel.CreateWindow(0, 2))
        {
            a.ResetState();
            a.Run(tokens, ReadOnlySpan<float>.Empty, positions, h1);
        }

        using (ILayerWindowExecutor b = windowModel.CreateWindow(2, 2))
        {
            b.ResetState();
            b.Run(ReadOnlySpan<int>.Empty, h1, positions, h2);
        }

        using ITensor cycled = windowModel.ApplyOutputHead(h2, seqLen);
        var cycLogits = new float[seqLen * vocab];
        unsafe
        {
            new ReadOnlySpan<float>((void*)cycled.DataPointer, cycLogits.Length).CopyTo(cycLogits);
        }

        double maxAbs = 0, maxRel = 0;
        for (int i = 0; i < refLogits.Length; i++)
        {
            double abs = Math.Abs(refLogits[i] - cycLogits[i]);
            double rel = abs / Math.Max(1e-9, Math.Abs(refLogits[i]));
            if (abs > maxAbs) maxAbs = abs;
            if (rel > maxRel) maxRel = rel;
        }

        output.WriteLine($"layers={config.NumLayers} seqLen={seqLen} vocab={vocab} " +
                         $"maxAbs={maxAbs:E6} maxRel={maxRel:E6}");
        Assert.True(maxRel < 1e-4, $"maxAbs={maxAbs:E6} maxRel={maxRel:E6}");
    }

    [Fact]
    public void WrongGdnOrdinal_WouldChangeTheAnswer()
    {
        // Discriminating control: if the second window's GDN layer read state slot 0 instead of its
        // own slot 1, the result would differ. Prove the two slots are actually distinguishable by
        // checking that a 3-layer-ish alternative cut (1..4) — which routes layer 2's GDN through a
        // window whose start is an attention layer — still reproduces the whole-model logits.
        string path = SyntheticQwen35MoeGguf.Write(
            Path.Combine(_scratch, "qwen35moe-4layer-b.gguf"), blockCount: 4);

        using GgufFile gguf = GgufFile.Open(path);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using IModel model = ModelLoader.CreateCpuModelFromGguf(gguf, config);

        int[] tokens = [2, 5, 1, 3];
        int[] positions = [0, 1, 2, 3];
        int seqLen = tokens.Length;
        int hidden = config.HiddenSize;
        int vocab = config.VocabSize;

        model.ResetSequenceState();
        using ITensor reference = model.Forward(tokens, positions, -1);
        var refLogits = new float[seqLen * vocab];
        unsafe
        {
            new ReadOnlySpan<float>((void*)reference.DataPointer, refLogits.Length).CopyTo(refLogits);
        }

        using var windowModel = new CpuLayerWindowModel(model, config);
        var buf = new float[seqLen * hidden];
        var buf2 = new float[seqLen * hidden];

        using (ILayerWindowExecutor a = windowModel.CreateWindow(0, 1))
        {
            a.ResetState();
            a.Run(tokens, ReadOnlySpan<float>.Empty, positions, buf);
        }

        using (ILayerWindowExecutor b = windowModel.CreateWindow(1, 3))
        {
            b.ResetState();
            b.Run(ReadOnlySpan<int>.Empty, buf, positions, buf2);
        }

        using ITensor cycled = windowModel.ApplyOutputHead(buf2, seqLen);
        var cycLogits = new float[seqLen * vocab];
        unsafe
        {
            new ReadOnlySpan<float>((void*)cycled.DataPointer, cycLogits.Length).CopyTo(cycLogits);
        }

        double maxAbs = 0, maxRel = 0;
        for (int i = 0; i < refLogits.Length; i++)
        {
            double abs = Math.Abs(refLogits[i] - cycLogits[i]);
            double rel = abs / Math.Max(1e-9, Math.Abs(refLogits[i]));
            if (abs > maxAbs) maxAbs = abs;
            if (rel > maxRel) maxRel = rel;
        }

        output.WriteLine($"cut@1 maxAbs={maxAbs:E6} maxRel={maxRel:E6}");
        Assert.True(maxRel < 1e-4, $"maxAbs={maxAbs:E6} maxRel={maxRel:E6}");
    }
}
