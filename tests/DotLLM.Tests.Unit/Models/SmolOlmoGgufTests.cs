using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// GGUF <c>smollm3</c>, <c>olmo2</c> and <c>olmoe</c> on the CPU backend (#765), against the independent double-precision
/// <see cref="GraniteGgufReference"/> (llama.cpp <c>smollm3.cpp</c> / <c>olmo2.cpp</c> / <c>olmoe.cpp</c> semantics), each
/// feature with an ablation arm proving the fixture is sensitive to it: NoPE layers, the post-norm-only layout, the
/// whole-projection Q/K norm (vs per-head), and OLMoE's non-renormalised top-k gating.
/// </summary>
public sealed class SmolOlmoGgufTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-smololmo-{Guid.NewGuid():N}");

    public SmolOlmoGgufTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private static readonly int[] Prompt = [1, 17, 5, 44, 9, 71, 30, 8, 55, 12, 63, 21];

    // No Granite scalars for these arches.
    private static SyntheticGraniteConfig Cfg(string arch, int layers = 4) => new()
    {
        Arch = arch, Layers = layers, EmbeddingScale = 0, AttentionScale = 0, ResidualScale = 0, LogitScale = 0,
    };

    private static unsafe float[] LastRow(ITensor logits)
    {
        int rows = logits.Shape[0], vocab = logits.Shape[1];
        var row = new float[vocab];
        new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * vocab, vocab).CopyTo(row);
        return row;
    }

    private (float[] Logits, ModelConfig Config) RunCpu(SyntheticGraniteGguf.Weights wts)
    {
        string path = Path.Combine(_scratch, $"{wts.Config.Arch}-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, SyntheticGraniteGguf.Serialize(wts));
        var (model, gguf, config) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
        using (gguf)
        using (model)
        {
            using ITensor logits = model.Forward(Prompt, Enumerable.Range(0, Prompt.Length).ToArray(), deviceId: -1);
            return (LastRow(logits), config);
        }
    }

    private static double MaxAbsDiff(float[] a, double[] b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    [Fact]
    public void Extract_SmolLM3_NoPeEveryFourthLayer_RopeNorm()
    {
        var (_, cfg) = RunCpu(SyntheticGraniteGguf.BuildWeights(Cfg("smollm3", 8)));
        Assert.Equal(Architecture.SmolLM3, cfg.Architecture);
        Assert.Equal(new[] { 3, 7 }, cfg.NoRopeLayers!.ToArray());
        Assert.True(cfg.IsNoRopeLayer(3));
        Assert.False(cfg.IsNoRopeLayer(2));
        Assert.Equal(RoPEType.Norm, cfg.RoPEConfig!.Value.Type);   // llama converter permutes Q/K
        Assert.False(cfg.QkNormWholeProjection);
    }

    [Fact]
    public void Extract_Olmo2_AndOlmoe_MapLlamaCppSemantics()
    {
        var (_, o2) = RunCpu(SyntheticGraniteGguf.BuildWeights(Cfg("olmo2")));
        Assert.Equal(Architecture.Olmo2, o2.Architecture);
        Assert.True(o2.QkNormWholeProjection);
        Assert.Equal(RoPEType.NeoX, o2.RoPEConfig!.Value.Type);
        Assert.Equal(ActivationFunction.SiLU, o2.ActivationFunction);

        var (_, oe) = RunCpu(SyntheticGraniteGguf.BuildWeights(Cfg("olmoe")));
        Assert.Equal(Architecture.QwenMoe, oe.Architecture);
        Assert.NotNull(oe.Moe);
        Assert.False(oe.Moe!.NormTopKProb);              // llama.cpp olmoe: norm_w = false
        Assert.Null(oe.Moe.SharedExpertIntermediateSize);
        Assert.True(oe.QkNormWholeProjection);
        Assert.Equal(RoPEType.NeoX, oe.RoPEConfig!.Value.Type);
    }

    [Fact]
    public void Olmo3SlidingWindow_IsRefusedNotSilentlyRunAsOlmo2()
    {
        var wts = SyntheticGraniteGguf.BuildWeights(Cfg("olmo2") with { SlidingWindow = 3 });
        var ex = Assert.Throws<NotSupportedException>(() => RunCpu(wts));
        Assert.Contains("OLMo 3", ex.Message);
    }

    [Theory]
    [InlineData("smollm3", 8)]
    [InlineData("olmo2", 4)]
    [InlineData("olmoe", 4)]
    public void Cpu_ForwardMatchesIndependentReference(string arch, int layers)
    {
        var wts = SyntheticGraniteGguf.BuildWeights(Cfg(arch, layers));
        var (logits, _) = RunCpu(wts);
        double d = MaxAbsDiff(logits, GraniteGgufReference.LastTokenLogits(wts, Prompt));
        Assert.True(d < 2e-3, $"{arch}: max|cpu - reference| = {d}");
    }

    [Theory]
    [InlineData("smollm3", 8)]
    [InlineData("olmo2", 4)]
    [InlineData("olmoe", 4)]
    public void FixtureIsSensitiveToEveryFeature_AblatedReferencesDisagree(string arch, int layers)
    {
        var wts = SyntheticGraniteGguf.BuildWeights(Cfg(arch, layers));
        var (logits, _) = RunCpu(wts);
        var ablations = new List<(string Name, GraniteGgufReference.Options Opt)>();
        if (arch == "smollm3") ablations.Add(("NoPE layers", new() { NoPe = false }));
        if (arch == "olmo2")
        {
            ablations.Add(("post-norm-only layout", new() { Olmo2Layout = false }));
        }
        if (arch is "olmo2" or "olmoe")
        {
            ablations.Add(("Q/K norm", new() { QkNorm = false }));
            ablations.Add(("whole-projection (vs per-head) Q/K norm", new() { QkNormWhole = false }));
        }
        if (arch == "olmoe") ablations.Add(("no top-k renormalisation", new() { OlmoeNoRenorm = false }));
        foreach (var (name, opt) in ablations)
        {
            double d = MaxAbsDiff(logits, GraniteGgufReference.LastTokenLogits(wts, Prompt, opt));
            Assert.True(d > 0.01, $"{arch}: fixture is insensitive to '{name}' (ablated reference differs by only {d}).");
        }
    }

    [Theory]
    [InlineData("smaug-bpe")]   // SmolLM3
    [InlineData("dbrx")]        // OLMo 2 1B
    [InlineData("olmo")]        // OLMoE
    public void Tokenizer_PreTypesShippedByTheseModelsAreKnown(string pre)
    {
        // Pre-#765 these threw "Unknown tokenizer.ggml.pre type" (and loading the real GGUFs failed in the CLI).
        var regexes = DotLLM.Tokenizers.Bpe.TiktokenPreTokenizer.GetRegexes(pre);
        Assert.NotEmpty(regexes);
    }
}
