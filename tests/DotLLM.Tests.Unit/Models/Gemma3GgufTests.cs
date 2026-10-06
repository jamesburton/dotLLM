using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// GGUF <c>gemma3</c> support on the CPU backend (#763). The forward-pass tests compare against
/// <see cref="Gemma2GgufReference"/> — an independent double-precision implementation of llama.cpp's
/// <c>gemma3.cpp</c> graph — and every Gemma-3-specific feature (QK-norm, dual RoPE base, linear scale on
/// the global layers only, the 1-in-N global pattern, the 27B score-scale rule) is paired with an
/// ablation arm that PROVES the fixture is sensitive to it. The fixture uses a pattern of 3 (not 6) so a
/// 12-layer model has four global layers, and distinct RoPE bases (7000 / 50000) so a single-table
/// implementation cannot pass.
/// </summary>
public sealed class Gemma3GgufTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma3-{Guid.NewGuid():N}");

    public Gemma3GgufTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private static readonly int[] Prompt = [2, 17, 5, 44, 9, 71, 30, 8, 55, 12, 63, 21];

    private static SyntheticGemma2Config Cfg(int layers = 12) => new() { Arch = "gemma3", Layers = layers };

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
    public void Extract_Gemma3_MapsEveryGemmaField()
    {
        var wts = SyntheticGemma2Gguf.BuildWeights(Cfg());
        var (_, cfg) = RunCpu(wts);

        Assert.Equal(Architecture.Gemma3, cfg.Architecture);
        Assert.Equal(ActivationFunction.GELUTanh, cfg.ActivationFunction);
        Assert.Equal(MathF.Sqrt(64), cfg.EmbeddingScale);
        Assert.True(cfg.TiedEmbeddings);
        Assert.Null(cfg.AttnLogitSoftcap);
        Assert.Null(cfg.FinalLogitSoftcap);
        Assert.Equal(8.0f, cfg.QueryPreAttnScalar);      // 12 layers => head_dim (not the 27B rule)
        Assert.Equal(3, cfg.SlidingWindowSize);

        // Pattern 3: layers 2, 5, 8, 11 are full-attention (null window), the rest windowed.
        Assert.NotNull(cfg.PerLayerSlidingWindow);
        for (int l = 0; l < 12; l++)
        {
            bool expectGlobal = (l % 3) == 2;
            Assert.Equal(expectGlobal, cfg.IsFullAttentionLayer(l));
            Assert.Equal(expectGlobal ? (int?)null : 3, cfg.PerLayerSlidingWindow![l]);
        }

        // Local table = freq_base_swa, no scaling; global table = freq_base + linear factor 8.
        Assert.NotNull(cfg.RoPEConfig);
        Assert.Equal(7000f, cfg.RoPEConfig!.Value.Theta);
        Assert.Equal(RoPEType.NeoX, cfg.RoPEConfig.Value.Type);
        Assert.Equal(RoPEScalingType.None, cfg.RoPEConfig.Value.ScalingType);
        Assert.NotNull(cfg.GlobalRoPEConfig);
        Assert.Equal(50000f, cfg.GlobalRoPEConfig!.Value.Theta);
        Assert.Equal(RoPEType.NeoX, cfg.GlobalRoPEConfig.Value.Type);
        Assert.Equal(RoPEScalingType.Linear, cfg.GlobalRoPEConfig.Value.ScalingType);
        Assert.Equal(8f, cfg.GlobalRoPEConfig.Value.ScalingFactor);
    }

    [Fact]
    public void Extract_Gemma3_DefaultsWhenOptionalKeysAreOmitted()
    {
        // llama.cpp: freq_base_swa defaults to 10000, the pattern to 6, scaling to none.
        var wts = SyntheticGemma2Gguf.BuildWeights(Cfg(12) with { RopeBaseSwa = 0, SlidingPattern = 0, RopeLinearFactor = 0 });
        var (_, cfg) = RunCpu(wts);
        Assert.Equal(10000f, cfg.RoPEConfig!.Value.Theta);
        Assert.Equal(RoPEScalingType.None, cfg.GlobalRoPEConfig!.Value.ScalingType);
        for (int l = 0; l < 12; l++)
            Assert.Equal((l % 6) == 5, cfg.IsFullAttentionLayer(l));
    }

    [Theory]
    [InlineData(62, 5376, 32, 128, 168f)]   // 27B: hidden/heads, NOT head_dim
    [InlineData(48, 3840, 16, 256, 256f)]   // 12B
    [InlineData(34, 2560, 8, 256, 256f)]    // 4B
    [InlineData(26, 1152, 4, 256, 256f)]    // 1B (hidden/heads = 288 != head_dim)
    public void QueryPreAttnScalar_FollowsLlamaCppModelSizeRule(int layers, int hidden, int heads, int headDim, float expected)
        => Assert.Equal(expected, GgufModelConfigExtractor.ResolveGemma3QueryPreAttnScalar(layers, hidden, heads, headDim));

    [Fact]
    public void Cpu_Gemma3_ForwardMatchesIndependentReference()
    {
        var wts = SyntheticGemma2Gguf.BuildWeights(Cfg());
        var (logits, _) = RunCpu(wts);
        double[] reference = Gemma2GgufReference.LastTokenLogits(wts, Prompt);
        Assert.True(MaxAbsDiff(logits, reference) < 2e-3, $"max|cpu - reference| = {MaxAbsDiff(logits, reference)}");
    }

    [Fact]
    public void Cpu_Gemma3_62LayerModelUsesHiddenOverHeadsScale()
    {
        // The 62-layer 27B scales scores by 1/sqrt(n_embd/n_head) = 1/4 here; head_dim = 8 would give 1/sqrt(8).
        var wts = SyntheticGemma2Gguf.BuildWeights(Cfg(62));
        var (logits, cfg) = RunCpu(wts);
        Assert.Equal(16.0f, cfg.QueryPreAttnScalar);
        Assert.True(MaxAbsDiff(logits, Gemma2GgufReference.LastTokenLogits(wts, Prompt)) < 2e-3);
        Assert.True(MaxAbsDiff(logits, Gemma2GgufReference.LastTokenLogits(wts, Prompt,
            new() { QueryPreAttnScalarFromHidden = false })) > 0.01, "fixture insensitive to the 27B score-scale rule");
    }

    [Fact]
    public void Gemma3_FixtureIsSensitiveToEveryFeature_AblatedReferencesDisagree()
    {
        var wts = SyntheticGemma2Gguf.BuildWeights(Cfg());
        var (logits, _) = RunCpu(wts);
        var ablations = new (string Name, Gemma2GgufReference.Options Opt)[]
        {
            ("QK-norm", new() { QkNorm = false }),
            ("dual rope base (local theta)", new() { DualRope = false }),
            ("linear scale on global layers", new() { LinearScale = false }),
            ("1-in-N global pattern", new() { Pattern6 = false }),
            ("sliding window", new() { SlidingWindow = false }),
            ("post norms", new() { PostNorms = false }),
            ("embedding scale", new() { EmbedScale = false }),
        };
        foreach (var (name, opt) in ablations)
        {
            double d = MaxAbsDiff(logits, Gemma2GgufReference.LastTokenLogits(wts, Prompt, opt));
            Assert.True(d > 0.01, $"fixture is insensitive to '{name}' (ablated reference differs by only {d}).");
        }
    }

    [Fact]
    public void Cpu_Gemma3_SlidingPatternArrayEqualsScalar()
    {
        // llama.cpp load_swa_pattern accepts an explicit per-layer array OR the scalar N; both must agree.
        var scalar = SyntheticGemma2Gguf.BuildWeights(Cfg());
        var array = SyntheticGemma2Gguf.BuildWeights(Cfg() with { SlidingPatternAsArray = true });
        var (a, ca) = RunCpu(scalar);
        var (b, cb) = RunCpu(array);
        Assert.Equal(ca.PerLayerSlidingWindow, cb.PerLayerSlidingWindow);
        Assert.Equal(a, b);
    }

    [Fact]
    public void Tokenizer_Gemma3_PrependsBosOnEncode()
    {
        string path = Path.Combine(_scratch, "tok.gguf");
        SyntheticGemma2Gguf.Write(path, Cfg(2));
        using var gguf = DotLLM.Models.Gguf.GgufFile.Open(path);
        var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] raw = tok.EncodeRaw("tok7tok9");
        int[] enc = tok.Encode("tok7tok9");
        Assert.NotEqual(tok.BosTokenId, raw[0]);
        Assert.Equal(tok.BosTokenId, enc[0]);
    }

    [Fact]
    public void HfConfig_Gemma3_UsesLocalAndGlobalRopeBases()
    {
        // Safetensors path (pre-#763 it ran EVERY layer on rope_theta): local layers must use rope_local_base_freq.
        string dir = Path.Combine(_scratch, "hf");
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "config.json"), """
        {
          "architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text",
          "hidden_size": 64, "num_hidden_layers": 12, "num_attention_heads": 4, "num_key_value_heads": 2,
          "head_dim": 8, "intermediate_size": 96, "vocab_size": 96, "max_position_embeddings": 128,
          "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "rope_local_base_freq": 10000.0,
          "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
          "sliding_window": 3, "sliding_window_pattern": 6, "query_pre_attn_scalar": 8
        }
        """);
        var cfg = DotLLM.Models.SafeTensors.HfConfigExtractor.Extract(File.ReadAllText(Path.Combine(dir, "config.json")));
        Assert.Equal(Architecture.Gemma3, cfg.Architecture);
        Assert.Equal(10000f, cfg.RoPEConfig!.Value.Theta);
        Assert.Equal(1000000f, cfg.GlobalRoPEConfig!.Value.Theta);
        Assert.Equal(RoPEScalingType.Linear, cfg.GlobalRoPEConfig.Value.ScalingType);
        Assert.Equal(8f, cfg.GlobalRoPEConfig.Value.ScalingFactor);
        Assert.Equal(RoPEScalingType.None, cfg.RoPEConfig.Value.ScalingType);
    }
}
