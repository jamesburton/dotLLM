using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// GGUF <c>granite</c> / <c>granitemoe</c> on the CPU backend and the four Granite scalars (#764, #313).
/// Forward tests compare against <see cref="GraniteGgufReference"/> (independent, double precision); each scalar has
/// an ablation arm proving the fixture is sensitive to it. The scalars are != 1 and mutually distinct
/// (embedding 3.0, attention 0.07, residual 0.4, logit 5.0) so a swapped, missing or doubly-applied scalar
/// cannot pass. NB the logit scale does not change the argmax: only logit VALUES can detect a missing one.
/// </summary>
public sealed class GraniteGgufTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-granite-{Guid.NewGuid():N}");

    public GraniteGgufTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private static readonly int[] Prompt = [1, 17, 5, 44, 9, 71, 30, 8, 55, 12];

    private static unsafe float[] LastRow(ITensor logits)
    {
        int rows = logits.Shape[0], vocab = logits.Shape[1];
        var row = new float[vocab];
        new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * vocab, vocab).CopyTo(row);
        return row;
    }

    private string Write(SyntheticGraniteGguf.Weights wts)
    {
        string path = Path.Combine(_scratch, $"{wts.Config.Arch}-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, SyntheticGraniteGguf.Serialize(wts));
        return path;
    }

    private (float[] Logits, ModelConfig Config) RunCpu(SyntheticGraniteGguf.Weights wts)
    {
        var (model, gguf, config) = ModelLoader.LoadFromGguf(Write(wts), ThreadingConfig.SingleThreaded);
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
    public void Extract_Granite_MapsAllFourScalarsAndRopeNorm()
    {
        var wts = SyntheticGraniteGguf.BuildWeights(new SyntheticGraniteConfig());
        var (_, cfg) = RunCpu(wts);
        Assert.Equal(Architecture.Granite, cfg.Architecture);
        Assert.Equal(3.0f, cfg.EmbeddingScale);
        Assert.Equal(0.07f, cfg.AttentionScale);
        Assert.Equal(0.4f, cfg.ResidualScale);
        Assert.Equal(5.0f, cfg.LogitScale);
        Assert.True(cfg.HasGraniteScalars);
        Assert.Equal(32, cfg.HeadDim);                       // hidden / heads: Granite has no head_dim key
        Assert.Equal(0.07f, cfg.AttentionScoreScale(cfg.HeadDim));
        Assert.Equal(RoPEType.Norm, cfg.RoPEConfig!.Value.Type);   // converter permutes Q/K
        Assert.Null(cfg.Moe);
    }

    [Fact]
    public void Extract_Granite_ScalarsOmittedMeanOff_AndLogitScaleIsRequired()
    {
        var off = SyntheticGraniteGguf.BuildWeights(new SyntheticGraniteConfig
            { EmbeddingScale = 0, AttentionScale = 0, ResidualScale = 0 });
        var (_, cfg) = RunCpu(off);
        Assert.Null(cfg.EmbeddingScale);
        Assert.Null(cfg.AttentionScale);
        Assert.Null(cfg.ResidualScale);
        Assert.Equal(1.0f / MathF.Sqrt(32), cfg.AttentionScoreScale(32));

        // llama.cpp reads logit_scale as REQUIRED: a file without it must not silently load with scale 1.
        var missing = SyntheticGraniteGguf.BuildWeights(new SyntheticGraniteConfig { LogitScale = 0 });
        Assert.Throws<InvalidDataException>(() => RunCpu(missing));
    }

    [Fact]
    public void Extract_GraniteMoe_MapsMoeAndRejectsSharedExpert()
    {
        var wts = SyntheticGraniteGguf.BuildWeights(new SyntheticGraniteConfig { Arch = "granitemoe" });
        var (_, cfg) = RunCpu(wts);
        Assert.Equal(Architecture.GraniteMoe, cfg.Architecture);
        Assert.NotNull(cfg.Moe);
        Assert.Equal(4, cfg.Moe!.NumExperts);
        Assert.Equal(2, cfg.Moe.NumExpertsPerTok);
        Assert.Equal(96, cfg.Moe.MoeIntermediateSize);
        Assert.True(cfg.Moe.NormTopKProb);
        Assert.Null(cfg.Moe.SharedExpertIntermediateSize);
        Assert.Equal(5.0f, cfg.LogitScale);

        // Granite MoE-shared adds an UNGATED shared FFN branch we do not implement: refuse, never drop it.
        var shared = SyntheticGraniteGguf.BuildWeights(new SyntheticGraniteConfig { Arch = "granitemoe", SharedFeedForward = 64 });
        Assert.Throws<NotSupportedException>(() => RunCpu(shared));
    }

    [Theory]
    [InlineData("granite")]
    [InlineData("granitemoe")]
    public void Cpu_Granite_ForwardMatchesIndependentReference(string arch)
    {
        var wts = SyntheticGraniteGguf.BuildWeights(new SyntheticGraniteConfig { Arch = arch });
        var (logits, _) = RunCpu(wts);
        double d = MaxAbsDiff(logits, GraniteGgufReference.LastTokenLogits(wts, Prompt));
        Assert.True(d < 2e-3, $"max|cpu - reference| = {d}");
    }

    [Theory]
    [InlineData("granite")]
    [InlineData("granitemoe")]
    public void Granite_FixtureIsSensitiveToEveryScalar_AblatedReferencesDisagree(string arch)
    {
        var wts = SyntheticGraniteGguf.BuildWeights(new SyntheticGraniteConfig { Arch = arch });
        var (logits, _) = RunCpu(wts);
        var ablations = new (string Name, GraniteGgufReference.Options Opt)[]
        {
            ("embedding scale", new() { EmbeddingScale = false }),
            ("attention scale", new() { AttentionScale = false }),
            ("residual scale", new() { ResidualScale = false }),
            ("logit scale", new() { LogitScale = false }),
        };
        foreach (var (name, opt) in ablations)
        {
            double d = MaxAbsDiff(logits, GraniteGgufReference.LastTokenLogits(wts, Prompt, opt));
            Assert.True(d > 0.02, $"fixture is insensitive to '{name}' ({arch}; ablated reference differs by only {d}).");
        }
    }

    [Fact]
    public void Cpu_Granite_TiedLmHead()
    {
        var wts = SyntheticGraniteGguf.BuildWeights(new SyntheticGraniteConfig { Tied = true });
        var (logits, _) = RunCpu(wts);
        Assert.True(MaxAbsDiff(logits, GraniteGgufReference.LastTokenLogits(wts, Prompt)) < 2e-3);
    }

    /// <summary>
    /// The three KV-cache entry points must all honour the attention multiplier: the cacheless forward (above),
    /// the FP32 KV-cache decode, the Q8_0 quantized-KV decode (whose kernel hard-codes 1/sqrt(head_dim) and is
    /// pre-scaled), and the fused multi-sequence ForwardBatch path.
    /// </summary>
    [Fact]
    public void Cpu_Granite_CachedDecodeHonoursScalars_Fp32Q8AndBatch()
    {
        var wts = SyntheticGraniteGguf.BuildWeights(new SyntheticGraniteConfig());
        var (model, gguf, cfg) = ModelLoader.LoadFromGguf(Write(wts), ThreadingConfig.SingleThreaded);
        using (gguf)
        using (model)
        {
            int split = Prompt.Length - 1;
            double[] reference = GraniteGgufReference.LastTokenLogits(wts, Prompt);
            var geometry = KvGeometry.FromConfig(cfg);

            // FP32 cache: prefill + one decode step.
            using (var fp32 = new SimpleKvCache(geometry, Prompt.Length + 1))
            {
                using (model.Forward(Prompt[..split], Enumerable.Range(0, split).ToArray(), -1, fp32)) { }
                using ITensor l = model.Forward([Prompt[split]], [split], -1, fp32);
                Assert.True(MaxAbsDiff(LastRow(l), reference) < 2e-3, "fp32 KV decode diverged from the reference");
            }

            // Q8_0 KV cache (no window): the quantized kernel has no scale parameter.
            using (var q8 = new QuantizedKvCache(geometry, Prompt.Length + 1, KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0))
            {
                using (model.Forward(Prompt[..split], Enumerable.Range(0, split).ToArray(), -1, q8)) { }
                using ITensor l = model.Forward([Prompt[split]], [split], -1, q8);
                double d = MaxAbsDiff(LastRow(l), reference);
                Assert.True(d < 0.08, $"Q8_0 KV decode diverged from the reference by {d} (attention scale not honoured?)");
                // Control: a model that ignored attention.scale would sit near the ablated reference.
                double wrong = MaxAbsDiff(LastRow(l), GraniteGgufReference.LastTokenLogits(wts, Prompt, new() { AttentionScale = false }));
                Assert.True(wrong > 3 * d, $"Q8_0 result is not closer to the correct reference ({d}) than to the unscaled one ({wrong}).");
            }

            // Batched: two sequences through ForwardBatch (FP32 caches).
            int[] other = Prompt.Select((t, i) => i == 0 ? 1 : (t * 7 + 3) % 90 + 4).ToArray();
            var otherRef = GraniteGgufReference.LastTokenLogits(wts, other);
            using var cA = new SimpleKvCache(geometry, Prompt.Length + 1);
            using var cB = new SimpleKvCache(geometry, Prompt.Length + 1);
            using (model.Forward(Prompt[..split], Enumerable.Range(0, split).ToArray(), -1, cA)) { }
            using (model.Forward(other[..split], Enumerable.Range(0, split).ToArray(), -1, cB)) { }
            var results = model.ForwardBatch(
            [
                new SequenceForwardRequest { TokenIds = new[] { Prompt[split] }, Positions = new[] { split }, KvCache = cA },
                new SequenceForwardRequest { TokenIds = new[] { other[split] }, Positions = new[] { split }, KvCache = cB },
            ], deviceId: -1);
            Assert.True(MaxAbsDiff(LastRow(results[0]), reference) < 2e-3, "ForwardBatch seq A diverged");
            Assert.True(MaxAbsDiff(LastRow(results[1]), otherRef) < 2e-3, "ForwardBatch seq B diverged");
            foreach (var r in results) r.Dispose();
        }
    }

    [Fact]
    public void HfConfig_Granite_ReadsTheFourMultipliers()
    {
        const string json = """
        {
          "architectures": ["GraniteForCausalLM"], "model_type": "granite",
          "hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
          "intermediate_size": 96, "vocab_size": 96, "max_position_embeddings": 128, "rms_norm_eps": 1e-5,
          "rope_theta": 10000.0, "embedding_multiplier": 12.0, "attention_multiplier": 0.0078125,
          "residual_multiplier": 0.22, "logits_scaling": 8.0, "tie_word_embeddings": true
        }
        """;
        var cfg = DotLLM.Models.SafeTensors.HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.Granite, cfg.Architecture);
        Assert.Equal(12.0f, cfg.EmbeddingScale);
        Assert.Equal(0.0078125f, cfg.AttentionScale);
        Assert.Equal(0.22f, cfg.ResidualScale);
        Assert.Equal(8.0f, cfg.LogitScale);

        // GraniteMoe carries the same four scalars.
        var moe = DotLLM.Models.SafeTensors.HfConfigExtractor.Extract(json
            .Replace("GraniteForCausalLM", "GraniteMoeForCausalLM").Replace("\"granite\"", "\"granitemoe\"")
            .Replace("\"hidden_size\"", "\"num_local_experts\": 4, \"num_experts_per_tok\": 2, \"hidden_size\""));
        Assert.Equal(Architecture.GraniteMoe, moe.Architecture);
        Assert.Equal(0.22f, moe.ResidualScale);
        Assert.Equal(8.0f, moe.LogitScale);

        // Shared-expert / hybrid Granite must be refused, not silently mapped to GraniteMoe.
        Assert.Throws<NotSupportedException>(() => DotLLM.Models.SafeTensors.HfConfigExtractor.Extract(
            json.Replace("GraniteForCausalLM", "GraniteMoeSharedForCausalLM").Replace("\"granite\"", "\"granitemoeshared\"")));
    }
}
