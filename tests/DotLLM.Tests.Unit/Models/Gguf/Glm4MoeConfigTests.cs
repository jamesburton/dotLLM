using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Vulkan;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// Issue #1598 (S0 of the GLM-4.5-Air plan): <c>glm4moe</c> config extraction (GGUF + HF), the tensor census and the loud-refusal paths.
/// The synthetic fixture uses values that are pairwise distinct (experts 8 / used 3 / shared 2 / trunk 3 / dense-lead 1 / nextn 1 /
/// heads 6 / kv 2) so a swapped or dropped field cannot hide behind a coincidence. CPU-only: no device is created anywhere.
/// </summary>
public sealed class Glm4MoeConfigTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-glm4moe-" + Guid.NewGuid().ToString("N"));

    public Glm4MoeConfigTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort cleanup */ }
    }

    private (GgufFile File, ModelConfig Config) Open(int experts = SyntheticGlm4MoeGguf.DefaultExperts, bool nextn = true, bool qkNorm = false)
    {
        string path = SyntheticGlm4MoeGguf.Write(Path.Combine(_dir, $"g-{experts}-{nextn}-{qkNorm}.gguf"), experts, withNextn: nextn, withQkNorm: qkNorm);
        var file = GgufFile.Open(path);
        return (file, GgufModelConfigExtractor.Extract(file.Metadata));
    }

    // ───────────────────────── GGUF config ─────────────────────────

    [Fact]
    public void Gguf_Extracts_Glm4MoeShape()
    {
        var (file, c) = Open();
        using (file)
        {
            Assert.Equal(Architecture.Glm4Moe, c.Architecture);
            Assert.Equal(3, c.NumLayers);                  // block_count 4 minus the NextN block
            Assert.Equal(1, c.NextnPredictLayers);
            Assert.Equal(SyntheticGlm4MoeGguf.HiddenSize, c.HiddenSize);
            Assert.Equal(6, c.NumAttentionHeads);
            Assert.Equal(2, c.NumKvHeads);
            Assert.Equal(16, c.HeadDim);
            Assert.Equal(SyntheticGlm4MoeGguf.DenseIntermediate, c.IntermediateSize);
            Assert.Equal(AttentionType.GQA, c.AttentionType);
            Assert.False(c.TiedEmbeddings);
            Assert.Null(c.UseQkNorm);                      // GGUF carries no such key: the census discovers it from the tensors

            Assert.True(c.RoPEConfig.HasValue);
            var rope = c.RoPEConfig!.Value;
            Assert.Equal(RoPEType.NeoX, rope.Type);
            Assert.Equal(8, rope.DimensionCount);          // partial rotary: head_dim 16 x 0.5
            Assert.Equal(1_000_000f, rope.Theta);
            Assert.Equal(0.5f, c.PartialRotaryFactor);

            Assert.NotNull(c.Moe);
            var m = c.Moe!;
            Assert.Equal(8, m.NumExperts);
            Assert.Equal(3, m.NumExpertsPerTok);
            Assert.Equal(24, m.MoeIntermediateSize);
            Assert.Equal(2, m.NumSharedExperts);
            Assert.Equal(48, m.SharedExpertIntermediateSize);   // fused: moe * shared
            Assert.False(m.HasSharedExpertGate);
            Assert.True(m.SigmoidGating);
            Assert.True(m.HasSelectionBias);
            Assert.True(m.NormalizeExpertWeights);
            Assert.False(m.NormTopKProb);
            Assert.Equal(1.0f, m.ExpertWeightsScale);
            Assert.Equal(new[] { 0 }, m.MlpOnlyLayers);         // layer 0 is the dense SwiGLU layer
            Assert.Equal(1, m.DecoderSparseStep);
        }
    }

    [Theory]
    [InlineData(128)]   // GLM-4.5-Air
    [InlineData(96)]    // Cerebras REAP-82B-A12B (25% pruned)
    public void Gguf_ExpertCount_FollowsTheFile_ForAirAndReapShapes(int experts)
    {
        var (file, c) = Open(experts);
        using (file)
        {
            Assert.Equal(experts, c.Moe!.NumExperts);
            Assert.Equal(SyntheticGlm4MoeGguf.ExpertsUsed, c.Moe.NumExpertsPerTok);
        }
    }

    [Fact]
    public void Gguf_WithoutNextnBlock_IsTrunkOnly()
    {
        var (file, c) = Open(nextn: false);
        using (file)
        {
            Assert.Equal(3, c.NumLayers);
            Assert.Equal(0, c.NextnPredictLayers);
        }
    }

    [Fact]
    public void Gguf_AbsentGatingFunc_DefaultsToSigmoid_LikeLlamaCpp()
    {
        var c = ExtractMutated(d => d.Remove("glm4moe.expert_gating_func"));
        Assert.True(c.Moe!.SigmoidGating);
        Assert.True(c.Moe.HasSelectionBias);
    }

    [Fact]
    public void Gguf_RejectsInvalidMoeMetadata()
    {
        Assert.Throws<KeyNotFoundException>(() => ExtractMutated(d => d.Remove("glm4moe.expert_shared_count")));
        Assert.Throws<InvalidDataException>(() => ExtractMutated(d => d["glm4moe.expert_used_count"] = U32(99)));
        Assert.Throws<InvalidDataException>(() => ExtractMutated(d => d["glm4moe.expert_gating_func"] = U32(7)));
    }

    private static GgufMetadataValue U32(uint v) => new(GgufValueType.UInt32, v);

    private ModelConfig ExtractMutated(Action<Dictionary<string, GgufMetadataValue>> mutate)
    {
        var (file, _) = Open();
        using (file)
        {
            var dict = new Dictionary<string, GgufMetadataValue>(StringComparer.Ordinal);
            foreach (string key in file.Metadata.Keys)
            {
                file.Metadata.TryGetValue(key, out var v);
                dict[key] = v;
            }
            mutate(dict);
            return GgufModelConfigExtractor.Extract(new GgufMetadata(dict));
        }
    }

    // ───────────────────────── HF config ─────────────────────────

    /// <summary>Verbatim zai-org/GLM-4.5-Air config.json (2026-11-03).</summary>
    private const string AirConfigJson = """
    {
      "architectures": ["Glm4MoeForCausalLM"], "attention_bias": true, "attention_dropout": 0.0, "pad_token_id": 151329,
      "eos_token_id": [151329, 151336, 151338], "head_dim": 128, "hidden_act": "silu", "hidden_size": 4096,
      "partial_rotary_factor": 0.5, "initializer_range": 0.02, "intermediate_size": 10944, "max_position_embeddings": 131072,
      "model_type": "glm4_moe", "moe_intermediate_size": 1408, "norm_topk_prob": true, "num_attention_heads": 96,
      "n_group": 1, "topk_group": 1, "n_routed_experts": 128, "n_shared_experts": 1, "routed_scaling_factor": 1.0,
      "num_experts_per_tok": 8, "first_k_dense_replace": 1, "num_hidden_layers": 46, "num_key_value_heads": 8,
      "rms_norm_eps": 1e-05, "rope_scaling": null, "rope_theta": 1000000, "num_nextn_predict_layers": 1,
      "tie_word_embeddings": false, "torch_dtype": "bfloat16", "transformers_version": "4.54.0", "use_cache": true,
      "use_qk_norm": false, "vocab_size": 151552
    }
    """;

    private static string Reap(string json) => json.Replace("\"n_routed_experts\": 128", "\"n_routed_experts\": 96");

    [Fact]
    public void Hf_Air_Extracts_Glm4MoeShape()
    {
        var c = HfConfigExtractor.Extract(AirConfigJson);
        Assert.Equal(Architecture.Glm4Moe, c.Architecture);
        Assert.Equal(46, c.NumLayers);
        Assert.Equal(1, c.NextnPredictLayers);
        Assert.Equal(4096, c.HiddenSize);
        Assert.Equal(96, c.NumAttentionHeads);
        Assert.Equal(8, c.NumKvHeads);
        Assert.Equal(128, c.HeadDim);
        Assert.Equal(10944, c.IntermediateSize);
        Assert.Equal(151552, c.VocabSize);
        Assert.False(c.TiedEmbeddings);
        Assert.False(c.UseQkNorm);
        Assert.Equal(0.5f, c.PartialRotaryFactor);
        Assert.True(c.RoPEConfig.HasValue);
            var rope = c.RoPEConfig!.Value;
        Assert.Equal(64, rope.DimensionCount);
        Assert.Equal(RoPEType.NeoX, rope.Type);
        Assert.Equal(1_000_000f, rope.Theta);
        Assert.NotNull(c.Moe);
            var m = c.Moe!;
        Assert.Equal(128, m.NumExperts);
        Assert.Equal(8, m.NumExpertsPerTok);
        Assert.Equal(1408, m.MoeIntermediateSize);
        Assert.Equal(1, m.NumSharedExperts);
        Assert.Equal(1408, m.SharedExpertIntermediateSize);
        Assert.True(m.SigmoidGating);
        Assert.True(m.HasSelectionBias);
        Assert.True(m.NormalizeExpertWeights);
        Assert.False(m.NormTopKProb);
        Assert.Equal(1.0f, m.ExpertWeightsScale);
        Assert.Equal(new[] { 0 }, m.MlpOnlyLayers);
    }

    [Fact]
    public void Hf_Reap_HasNinetySixExperts_OtherwiseIdentical()
    {
        var air = HfConfigExtractor.Extract(AirConfigJson);
        var reap = HfConfigExtractor.Extract(Reap(AirConfigJson));
        Assert.Equal(96, reap.Moe!.NumExperts);
        Assert.Equal(air.Moe!.NumExpertsPerTok, reap.Moe.NumExpertsPerTok);
        Assert.Equal(air.NumLayers, reap.NumLayers);
        Assert.Equal(air.HeadDim, reap.HeadDim);
    }

    [Fact]
    public void Hf_QkNormFlag_And_RopeParametersSpelling_AreHonoured()
    {
        // GLM-4.5 355B: use_qk_norm true. transformers >= 5 nests partial_rotary_factor / rope_theta under rope_parameters.
        string json = AirConfigJson
            .Replace("\"use_qk_norm\": false", "\"use_qk_norm\": true")
            .Replace("\"partial_rotary_factor\": 0.5,", "")
            .Replace("\"rope_theta\": 1000000,", "\"rope_parameters\": {\"rope_theta\": 500000, \"partial_rotary_factor\": 0.25},");
        var c = HfConfigExtractor.Extract(json);
        Assert.True(c.UseQkNorm);
        Assert.Equal(32, c.RoPEConfig!.Value.DimensionCount);
        Assert.Equal(0.25f, c.PartialRotaryFactor);
        Assert.Equal(500_000f, c.RoPEConfig!.Value.Theta);
    }

    [Fact]
    public void Hf_DenseGlm4_IsStillUnsupported_NotMisroutedToGlm4Moe()
    {
        string dense = "{\"architectures\":[\"Glm4ForCausalLM\"],\"model_type\":\"glm4\",\"hidden_size\":8,\"num_hidden_layers\":1,\"num_attention_heads\":2,\"intermediate_size\":8,\"vocab_size\":8}";
        Assert.Throws<InvalidDataException>(() => HfConfigExtractor.Extract(dense));
    }

    [Fact]
    public void Gguf_And_Hf_AgreeOnEveryMoeAndTrunkField_ForTheSameModel()
    {
        // The HF twin of the synthetic fixture: same dims, spelled the HF way.
        string json = $$"""
        {"architectures":["Glm4MoeForCausalLM"],"model_type":"glm4_moe","hidden_size":{{SyntheticGlm4MoeGguf.HiddenSize}},
         "num_hidden_layers":3,"num_attention_heads":6,"num_key_value_heads":2,"head_dim":16,"partial_rotary_factor":0.5,
         "intermediate_size":{{SyntheticGlm4MoeGguf.DenseIntermediate}},"moe_intermediate_size":24,"n_routed_experts":8,"n_shared_experts":2,
         "num_experts_per_tok":3,"first_k_dense_replace":1,"norm_topk_prob":true,"routed_scaling_factor":1.0,"num_nextn_predict_layers":1,
         "rope_theta":1000000,"rms_norm_eps":1e-05,"vocab_size":{{SyntheticGlm4MoeGguf.VocabSize}},"max_position_embeddings":{{SyntheticGlm4MoeGguf.ContextLength}},
         "tie_word_embeddings":false,"use_qk_norm":false}
        """;
        var hf = HfConfigExtractor.Extract(json);
        var (file, g) = Open();
        using (file)
        {
            Assert.Equal(g.NumLayers, hf.NumLayers);
            Assert.Equal(g.NextnPredictLayers, hf.NextnPredictLayers);
            Assert.Equal(g.HeadDim, hf.HeadDim);
            Assert.Equal(g.NumKvHeads, hf.NumKvHeads);
            Assert.Equal(g.PartialRotaryFactor, hf.PartialRotaryFactor);
            Assert.Equal(g.RoPEConfig!.Value.DimensionCount, hf.RoPEConfig!.Value.DimensionCount);
            Assert.Equal(g.RoPEConfig.Value.Type, hf.RoPEConfig.Value.Type);
            var a = g.Moe!; var b = hf.Moe!;
            Assert.Equal(a.NumExperts, b.NumExperts);
            Assert.Equal(a.NumExpertsPerTok, b.NumExpertsPerTok);
            Assert.Equal(a.MoeIntermediateSize, b.MoeIntermediateSize);
            Assert.Equal(a.NumSharedExperts, b.NumSharedExperts);
            Assert.Equal(a.SharedExpertIntermediateSize, b.SharedExpertIntermediateSize);
            Assert.Equal(a.SigmoidGating, b.SigmoidGating);
            Assert.Equal(a.HasSelectionBias, b.HasSelectionBias);
            Assert.Equal(a.NormalizeExpertWeights, b.NormalizeExpertWeights);
            Assert.Equal(a.NormTopKProb, b.NormTopKProb);
            Assert.Equal(a.ExpertWeightsScale, b.ExpertWeightsScale);
            Assert.Equal(a.MlpOnlyLayers, b.MlpOnlyLayers);
        }
    }

    // ───────────────────────── Loud failure ─────────────────────────

    [Fact]
    public void Cpu_Dispatch_Refuses_NamingTheArchitecture()
    {
        var (file, c) = Open();
        using (file)
        {
            var ex = Assert.Throws<NotSupportedException>(() => ModelLoader.CreateCpuModelFromGguf(file, c));
            Assert.Contains("glm4moe", ex.Message, StringComparison.Ordinal);
            // The dispatcher's own arm (not just the generic loader's backstop) must be what refuses.
            Assert.Contains("backend: CPU", ex.Message, StringComparison.Ordinal);
            Assert.Throws<NotSupportedException>(() => TransformerModel.LoadFromGguf(file, c));
            Assert.Throws<NotSupportedException>(() => TransformerWeights.LoadFromGguf(file, c));
        }
    }

    [Fact]
    public void ModelLoader_LoadFromGguf_Refuses_FromPath()
    {
        string path = SyntheticGlm4MoeGguf.Write(Path.Combine(_dir, "p.gguf"));
        var ex = Assert.Throws<NotSupportedException>(() => ModelLoader.LoadFromGguf(path));
        Assert.Contains("glm4moe", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Cuda_And_Vulkan_Refuse_BeforeTouchingADevice()
    {
        var (file, c) = Open();
        using (file)
        {
            var cuda = Assert.Throws<NotSupportedException>(() => CudaModelLoader.CreateFromGguf(file, c));
            Assert.Contains("glm4moe", cuda.Message, StringComparison.Ordinal);
            Assert.Throws<NotSupportedException>(() => CudaTransformerModel.RejectUnsupportedArchitecture(c));
            var vk = Assert.Throws<NotSupportedException>(() => VulkanTransformerModel.RejectUnsupportedArchitecture(c));
            Assert.Contains("glm4moe", vk.Message, StringComparison.Ordinal);
        }
    }

    // ───────────────────────── Tensor census ─────────────────────────

    [Theory]
    [InlineData(8, true, false)]
    [InlineData(128, true, false)]
    [InlineData(96, true, false)]
    [InlineData(8, false, false)]      // trunk-only conversion (ggml-org style): NextN absent is NOT a mismatch
    [InlineData(8, true, true)]        // GLM-4.5 355B: per-head Q/K norm
    public void Census_IsClean_ForAWellFormedFile(int experts, bool nextn, bool qkNorm)
    {
        var (file, c) = Open(experts, nextn, qkNorm);
        using (file)
        {
            var r = Glm4MoeTensorCensus.Check(c, file);
            Assert.True(r.IsClean, r + "\n" + string.Join("\n", r.Missing.Concat(r.Unexpected).Concat(r.ShapeMismatches)));
            Assert.Equal(nextn, r.NextnPresent);
            Assert.Equal(qkNorm, r.HasQkNorm);
        }
    }

    [Fact]
    public void Census_TensorCount_MatchesTheRealAirLayoutFormula()
    {
        // Real REAP-82B IQ4_XS: 803 tensors = 3 global + 12 (dense layer) + 17 per MoE layer + (17 + 6) NextN. Same formula here.
        var (file, c) = Open();
        using (file)
        {
            int expected = 3 + 12 + 17 * (c.NumLayers - 1) + (17 + 6);
            Assert.Equal(expected, Glm4MoeTensorCensus.Check(c, file).TensorCount);
        }
    }

    [Fact]
    public void Census_Detects_MissingTensor()
    {
        foreach (string victim in new[] { "blk.1.exp_probs_b.bias", "blk.2.ffn_up_shexp.weight", "blk.0.attn_q.bias", "blk.3.nextn.eh_proj.weight" })
        {
            string path = Path.Combine(_dir, "omit.gguf");
            File.WriteAllBytes(path, SyntheticGlm4MoeGguf.Build(omitTensor: victim));
            using var file = GgufFile.Open(path);
            var r = Glm4MoeTensorCensus.Check(GgufModelConfigExtractor.Extract(file.Metadata), file);
            Assert.False(r.IsClean, victim);
            Assert.Contains(victim, r.Missing);
        }
    }

    [Fact]
    public void Census_Detects_ShapeMismatch()
    {
        foreach (string victim in new[] { "blk.1.ffn_down_exps.weight", "blk.2.attn_k.weight", "output_norm.weight" })
        {
            string path = Path.Combine(_dir, "shape.gguf");
            File.WriteAllBytes(path, SyntheticGlm4MoeGguf.Build(wrongShapeTensor: victim));
            using var file = GgufFile.Open(path);
            var r = Glm4MoeTensorCensus.Check(GgufModelConfigExtractor.Extract(file.Metadata), file);
            Assert.Contains(r.ShapeMismatches, m => m.StartsWith(victim + ":", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void Census_Detects_ExpertCountDisagreeingWithTheMetadata()
    {
        // 8-expert tensors, metadata claims 6: every router/bank shape is off.
        var (file, c) = Open(experts: 8);
        using (file)
        {
            var wrong = c with { Moe = c.Moe! with { NumExperts = 6 } };
            var r = Glm4MoeTensorCensus.Check(wrong, file);
            Assert.False(r.IsClean);
            Assert.Contains(r.ShapeMismatches, m => m.StartsWith("blk.1.ffn_gate_inp.weight:", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void Census_Flags_UnexpectedAndMlaStyleTensors()
    {
        var (file, c) = Open();
        using (file)
        {
            var listing = file.Tensors.Select(t => (t.Name, t.Shape.Dimensions)).ToList();
            listing.Add(("blk.1.attn_kv_a_mqa.weight", [4, 4]));            // an MLA tensor: wrong family
            var r = Glm4MoeTensorCensus.Check(c, listing);
            Assert.Contains("blk.1.attn_kv_a_mqa.weight", r.Unexpected);
        }
    }

    [Fact]
    public void Census_DenseFirstKLayers_AreEnforced()
    {
        // Layer 0 must be dense and layer 1 MoE: a file whose layer 0 carries a router instead is wrong.
        var (file, c) = Open();
        using (file)
        {
            var listing = file.Tensors.Where(t => !t.Name.StartsWith("blk.0.ffn_", StringComparison.Ordinal))
                .Select(t => (t.Name, t.Shape.Dimensions)).ToList();
            listing.Add(("blk.0.ffn_gate_inp.weight", [SyntheticGlm4MoeGguf.HiddenSize, 8]));
            var r = Glm4MoeTensorCensus.Check(c, listing);
            Assert.Contains("blk.0.ffn_gate.weight", r.Missing);
            Assert.Contains("blk.0.ffn_gate_inp.weight", r.Unexpected);
        }
    }
}
