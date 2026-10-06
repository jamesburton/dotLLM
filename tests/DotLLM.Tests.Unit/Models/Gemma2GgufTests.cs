using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// GGUF <c>gemma</c> (Gemma 1 / CodeGemma) and <c>gemma2</c> support on the CPU backend (#736).
/// The forward-pass tests compare against <see cref="Gemma2GgufReference"/> — an independent
/// double-precision implementation of llama.cpp's graph — and each is paired with ablation
/// arms that PROVE the fixture is sensitive to the feature (a soft-cap or window that never
/// bites would let an unwired implementation pass).
/// </summary>
public sealed class Gemma2GgufTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma2-{Guid.NewGuid():N}");

    public Gemma2GgufTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // 12 tokens against a window of 3: the window, the soft-caps and RoPE position all matter.
    private static readonly int[] Prompt = [2, 17, 5, 44, 9, 71, 30, 8, 55, 12, 63, 21];

    private static unsafe float[] LastRow(ITensor logits)
    {
        int rows = logits.Shape[0], vocab = logits.Shape[1];
        var row = new float[vocab];
        new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * vocab, vocab).CopyTo(row);
        return row;
    }

    private (float[] Logits, ModelConfig Config) RunCpu(SyntheticGemma2Gguf.Weights wts)
    {
        string path = Path.Combine(_scratch, $"{wts.Config.Arch}-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, SyntheticGemma2Gguf.Serialize(wts));
        var (model, gguf, config) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
        using (gguf)
        using (model)
        {
            int[] positions = Enumerable.Range(0, Prompt.Length).ToArray();
            using ITensor logits = model.Forward(Prompt, positions, deviceId: -1);
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
    public void Extract_Gemma2_MapsEveryGemmaField()
    {
        var wts = SyntheticGemma2Gguf.BuildWeights(new SyntheticGemma2Config());
        var (_, cfg) = RunCpu(wts);

        Assert.Equal(Architecture.Gemma2, cfg.Architecture);
        Assert.Equal(ActivationFunction.GELUTanh, cfg.ActivationFunction);
        Assert.Equal(MathF.Sqrt(64), cfg.EmbeddingScale);
        Assert.Equal(2.0f, cfg.AttnLogitSoftcap);
        Assert.Equal(3.0f, cfg.FinalLogitSoftcap);
        Assert.Equal(3, cfg.SlidingWindowSize);
        Assert.Equal(2, cfg.SlidingWindowPattern);
        // 46 layers => the 27B rule: n_embd / n_head = 16, NOT head_dim = 8.
        Assert.Equal(16.0f, cfg.QueryPreAttnScalar);
        Assert.Equal(8, cfg.HeadDim);
        Assert.Equal(2, cfg.NumKvHeads);
        Assert.NotNull(cfg.RoPEConfig);
        Assert.Equal(10000f, cfg.RoPEConfig!.Value.Theta);
        Assert.Equal(RoPEType.NeoX, cfg.RoPEConfig.Value.Type);
        Assert.Equal(8, cfg.RoPEConfig.Value.DimensionCount);
    }

    [Theory]
    [InlineData(46, 4608, 32, 128, 144f)]   // 27B: hidden/heads, not head_dim
    [InlineData(42, 3584, 16, 256, 256f)]   // 9B
    [InlineData(26, 2304, 8, 256, 256f)]    // 2B
    public void QueryPreAttnScalar_FollowsLlamaCppModelSizeRule(int layers, int hidden, int heads, int headDim, float expected)
        => Assert.Equal(expected, GgufModelConfigExtractor.ResolveGemma2QueryPreAttnScalar(layers, hidden, heads, headDim));

    [Fact]
    public void Cpu_Gemma2_ForwardMatchesIndependentReference()
    {
        var wts = SyntheticGemma2Gguf.BuildWeights(new SyntheticGemma2Config());
        var (logits, _) = RunCpu(wts);
        double[] reference = Gemma2GgufReference.LastTokenLogits(wts, Prompt);
        Assert.True(MaxAbsDiff(logits, reference) < 2e-3, $"max|cpu - reference| = {MaxAbsDiff(logits, reference)}");
    }

    [Fact]
    public void Gemma2_FixtureIsSensitiveToEveryFeature_AblatedReferencesDisagree()
    {
        // Control arms: if turning a feature OFF in the reference does not move the logits, the fixture
        // cannot distinguish an implementation that forgot the feature, and the test above is vacuous.
        var wts = SyntheticGemma2Gguf.BuildWeights(new SyntheticGemma2Config());
        var (logits, _) = RunCpu(wts);
        var ablations = new (string Name, Gemma2GgufReference.Options Opt)[]
        {
            ("attn softcap", new() { AttnSoftcap = false }),
            ("final softcap", new() { FinalSoftcap = false }),
            ("sliding window", new() { SlidingWindow = false }),
            ("post norms", new() { PostNorms = false }),
            ("query_pre_attn_scalar (hidden/heads vs head_dim)", new() { QueryPreAttnScalarFromHidden = false }),
            ("embedding scale", new() { EmbedScale = false }),
        };
        foreach (var (name, opt) in ablations)
        {
            double d = MaxAbsDiff(logits, Gemma2GgufReference.LastTokenLogits(wts, Prompt, opt));
            Assert.True(d > 0.05, $"fixture is insensitive to '{name}' (ablated reference differs by only {d}).");
        }
    }

    [Fact]
    public void Cpu_Gemma1_ForwardMatchesIndependentReference()
    {
        var wts = SyntheticGemma2Gguf.BuildWeights(new SyntheticGemma2Config { Arch = "gemma", Layers = 4 });
        var (logits, cfg) = RunCpu(wts);
        Assert.Equal(Architecture.Gemma, cfg.Architecture);
        Assert.Null(cfg.AttnLogitSoftcap);
        Assert.Null(cfg.FinalLogitSoftcap);
        Assert.Null(cfg.SlidingWindowSize);
        Assert.NotNull(cfg.RoPEConfig);   // gemma GGUFs have no rope keys; RoPE must still be on
        double[] reference = Gemma2GgufReference.LastTokenLogits(wts, Prompt);
        Assert.True(MaxAbsDiff(logits, reference) < 2e-3, $"max|cpu - reference| = {MaxAbsDiff(logits, reference)}");
        // And embedding scaling is genuinely load-bearing for Gemma 1 too.
        Assert.True(MaxAbsDiff(logits, Gemma2GgufReference.LastTokenLogits(wts, Prompt, new() { EmbedScale = false })) > 0.05);
    }

    [Fact]
    public void Tokenizer_Gemma_PrependsBosOnEncode_ButNotOnEncodeRaw()
    {
        string path = Path.Combine(_scratch, "tok.gguf");
        SyntheticGemma2Gguf.Write(path, new SyntheticGemma2Config { Layers = 2 });
        using var gguf = DotLLM.Models.Gguf.GgufFile.Open(path);
        var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int[] raw = tok.EncodeRaw("tok7tok9");
        int[] enc = tok.Encode("tok7tok9");
        Assert.NotEmpty(raw);
        Assert.NotEqual(tok.BosTokenId, raw[0]);
        Assert.Equal(tok.BosTokenId, enc[0]);
        Assert.Equal(raw, enc.Skip(1).ToArray());
        // A chat-template-rendered prompt that already starts with <bos> must not get a second one.
        int[] already = tok.Encode("<bos>tok7");
        Assert.Equal(tok.BosTokenId, already[0]);
        Assert.NotEqual(tok.BosTokenId, already[1]);
    }
}
