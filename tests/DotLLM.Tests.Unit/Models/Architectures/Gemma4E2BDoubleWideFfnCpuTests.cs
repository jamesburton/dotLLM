using DotLLM.Core.Models;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// #917: the real Gemma-4 E2B GGUF has a per-layer <c>feed_forward_length</c> array (the trailing shared-KV
/// layers carry a double-wide MLP). Before the fix the extractor read a missing scalar =&gt; IntermediateSize 0
/// =&gt; zero-sized FFN scratch, and the first CPU forward died with an access violation inside
/// <c>Gemma4DenseFfn</c> (Vulkan: "Parameter 'bytes'" at load).
/// </summary>
public sealed class Gemma4E2BDoubleWideFfnCpuTests
{
    private static readonly int[] Ids = [2, 7, 8, 9, 5, 6, 3];
    private static readonly int[] Pos = [0, 1, 2, 3, 4, 5, 6];

    [Fact]
    public void CpuForward_MixedWidthDenseFfn_IsFiniteAndSeesTheWideLayers()
    {
        float[] narrow = Run(doubleWide: false, out int narrowInterm);
        float[] wide = Run(doubleWide: true, out int wideInterm);

        Assert.Equal(64, narrowInterm);
        Assert.Equal(128, wideInterm);   // widest layer, not layer 0's 64
        Assert.All(wide, v => Assert.True(float.IsFinite(v)));
        Assert.True(wide.Max() - wide.Min() > 0.5f, "logits degenerate");

        // Discrimination: the wide shared-KV layers must change the output (same seed => the only
        // difference is the extra FFN rows), i.e. their weights were really consumed.
        float worst = 0;
        for (int i = 0; i < wide.Length; i++) worst = MathF.Max(worst, MathF.Abs(wide[i] - narrow[i]));
        Assert.True(worst > 1e-3f, $"double-wide FFN had no effect (worst diff {worst:E2})");
    }

    private static unsafe float[] Run(bool doubleWide, out int intermediateSize)
    {
        var syn = SyntheticGemma4Gguf.E4BLike with
        {
            RopeFreqsProportionalPairs = 2,
            DoubleWideSharedKvFfn = doubleWide,
        };
        string path = Path.Combine(Path.GetTempPath(), $"syn_g4e2b_cpu_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, syn);
        try
        {
            using var gguf = GgufFile.Open(path);
            ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            intermediateSize = cfg.IntermediateSize;
            using var model = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded);
            using ITensor logits = model.Forward(Ids, Pos, -1);
            int total = 1;
            for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
            var all = new float[total];
            new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(all);
            return all[(total - cfg.VocabSize)..];
        }
        finally { try { File.Delete(path); } catch { } }
    }
}
