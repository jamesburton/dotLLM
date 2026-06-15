using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Unit + integration tests for <see cref="DiffusionGemmaConfigExtractor"/> and the
/// DiffusionGemma <see cref="ModelLoader"/> dispatch (issue #29, DiffusionGemma PR-8).
/// CPU-only; no model downloads — every fixture is written to a scratch directory.
/// </summary>
/// <remarks>
/// Two structural cases are covered:
/// <list type="bullet">
///   <item><b>REAL-structure config</b> — the verified <c>google/diffusiongemma-26B-A4B-it</c>
///     shape (hidden 2816, 30 layers, global_head_dim 512 != head_dim 256, 128 experts top-8,
///     full attention at layers 5/11/17/23/29, per-attention-type RoPE). Asserts the extractor
///     PARSES it into a complete <see cref="ModelConfig"/> without throwing, even though the
///     CPU forward gates the distinct global_head_dim on issue #36.</item>
///   <item><b>TINY SYNTHETIC config + weights with a UNIFORM head_dim</b> — global_head_dim ==
///     head_dim so the #36 guard is not hit; loaded end-to-end through <see cref="ModelLoader"/>
///     and driven through <see cref="DotLLM.Engine.DiffusionTextGenerator"/> for a few denoise
///     steps; returns tokens with no surviving mask token.</item>
/// </list>
/// </remarks>
public sealed class DiffusionGemmaConfigExtractorTests : IDisposable
{
    private readonly string _scratch;

    public DiffusionGemmaConfigExtractorTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-dg-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // ───────────────────────── REAL-structure config parse (no forward) ─────────────────────────

    /// <summary>
    /// The verified DiffusionGemma 26B config (text_config hoisted, top-level
    /// canvas_length, vision_config present-but-skipped, global_head_dim 512 !=
    /// head_dim 256) must parse into a fully-populated Gemma-4 MoE ModelConfig
    /// with a non-null DiffusionConfig and the resolved mask token id. This is
    /// the #36 case: extraction succeeds, forward would be gated.
    /// </summary>
    [Fact]
    public void ExtractFromDirectory_RealStructureConfig_ParsesCompleteModelConfig()
    {
        File.WriteAllText(Path.Combine(_scratch, "config.json"), RealConfigJson);
        File.WriteAllText(Path.Combine(_scratch, "generation_config.json"), RealGenerationConfigJson);
        WriteTokenizerFixture(_scratch, maskTokenId: 262144);

        using var doc = JsonDocument.Parse(RealConfigJson);
        ModelConfig cfg = DiffusionGemmaConfigExtractor.ExtractFromDirectory(doc.RootElement, _scratch);

        // Architecture + backbone scalars.
        Assert.Equal(Architecture.DiffusionGemma, cfg.Architecture);
        Assert.True(cfg.IsGemmaArchitecture);
        Assert.Equal(2816, cfg.HiddenSize);
        Assert.Equal(30, cfg.NumLayers);
        Assert.Equal(16, cfg.NumAttentionHeads);
        Assert.Equal(8, cfg.NumKvHeads);            // sliding KV heads
        Assert.Equal(2, cfg.NumGlobalKvHeads);      // full-attn KV heads (dual KV)
        Assert.Equal(256, cfg.HeadDim);             // sliding head_dim
        Assert.Equal(512, cfg.GlobalHeadDim);       // full-attn head_dim (#36 case, != HeadDim)
        Assert.Equal(2112, cfg.IntermediateSize);
        Assert.Equal(262144, cfg.VocabSize);
        Assert.Equal(1024, cfg.SlidingWindowSize);
        Assert.Equal(30.0f, cfg.FinalLogitSoftcap);
        Assert.Equal(ActivationFunction.GELUTanh, cfg.ActivationFunction);
        Assert.Equal(MathF.Sqrt(2816), cfg.EmbeddingScale);

        // MoE.
        Assert.NotNull(cfg.Moe);
        Assert.Equal(128, cfg.Moe!.NumExperts);
        Assert.Equal(8, cfg.Moe.NumExpertsPerTok);
        Assert.Equal(704, cfg.Moe.MoeIntermediateSize);

        // Per-attention-type RoPE.
        Assert.NotNull(cfg.RoPEConfig);
        Assert.NotNull(cfg.GlobalRoPEConfig);
        Assert.Equal(10_000.0f, cfg.RoPEConfig!.Value.Theta);        // sliding theta
        Assert.Equal(1_000_000.0f, cfg.GlobalRoPEConfig!.Value.Theta); // full theta
        Assert.Equal(0.25f, cfg.PartialRotaryFactor);

        // Full-attention layer pattern: layers 5,11,17,23,29 (0-based) are full.
        int[] expectedFull = [5, 11, 17, 23, 29];
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            bool shouldBeFull = Array.IndexOf(expectedFull, i) >= 0;
            Assert.Equal(shouldBeFull, cfg.IsFullAttentionLayer(i));
        }

        // Diffusion decode config + tokenizer-resolved mask token.
        Assert.NotNull(cfg.DiffusionConfig);
        Assert.Equal(256, cfg.DiffusionConfig!.CanvasLength);
        Assert.Equal(48, cfg.DiffusionConfig.MaxDenoisingSteps);
        Assert.Equal(0.1f, cfg.DiffusionConfig.EntropyBound);
        Assert.Equal(262144, cfg.DiffusionConfig.MaskTokenId);
    }

    /// <summary>
    /// The distinct global_head_dim (512) vs head_dim (256) on the real config is
    /// faithfully carried; building a CPU model from it is what trips the #36 gate
    /// (NotSupportedException at model-build time), not config extraction.
    /// Documents that real-26B forward awaits issue #36.
    /// </summary>
    [Fact]
    public void RealStructureConfig_DistinctGlobalHeadDim_GatesForwardOn36NotExtraction()
    {
        using var doc = JsonDocument.Parse(RealConfigJson);
        // Extraction-only core (no tokenizer needed): must not throw.
        ModelConfig cfg = DiffusionGemmaConfigExtractor.ExtractTextConfig(doc.RootElement);

        Assert.Equal(512, cfg.GlobalHeadDim);
        Assert.Equal(256, cfg.HeadDim);
        Assert.NotEqual(cfg.HeadDim, cfg.GlobalHeadDim!.Value);
        // The forward path (model build) is the gated layer — documented here so a
        // future #36 fix flips this expectation. We assert the config alone is sound.
        Assert.NotNull(cfg.GlobalRoPEConfig);
    }

    // ───────────────────────── ModelLoader dispatch routes diffusion_gemma ─────────────────────────

    /// <summary>
    /// A tiny synthetic diffusion_gemma checkpoint (uniform head_dim) loads through
    /// the top-level <see cref="ModelLoader.LoadFromSafetensors(string, DotLLM.Core.Configuration.ThreadingConfig?)"/>
    /// dispatch without throwing, returning a model whose ModelConfig carries the
    /// diffusion configuration and a resolved mask token id.
    /// </summary>
    [Fact]
    public void ModelLoader_DiffusionGemma_LoadsWithoutThrowingAndCarriesDiffusionConfig()
    {
        SyntheticCheckpoint chk = WriteSyntheticCheckpoint(seed: 7);

        var (model, source, config) = ModelLoader.LoadFromSafetensors(_scratch);
        using (model)
        using (source)
        {
            Assert.Equal(Architecture.DiffusionGemma, config.Architecture);
            Assert.NotNull(config.DiffusionConfig);
            Assert.Equal(chk.MaskTokenId, config.DiffusionConfig!.MaskTokenId);
            Assert.Equal(chk.CanvasLength, config.DiffusionConfig.CanvasLength);
            // Uniform head_dim: the forward path is NOT gated, so the model built.
            Assert.IsType<TransformerModel>(model);
        }
    }

    // ───────────────────────── End-to-end diffusion decode on synthetic weights ─────────────────────────

    /// <summary>
    /// Integration: a tiny synthetic diffusion-gemma checkpoint (uniform head_dim,
    /// small expert count) loaded end-to-end through <see cref="ModelLoader"/> and
    /// driven through <see cref="DotLLM.Engine.DiffusionTextGenerator"/> for a few
    /// denoise steps on CPU. Returns the requested number of tokens; no mask token
    /// survives into the finished canvas.
    /// </summary>
    [Fact]
    public void EndToEnd_SyntheticDiffusionGemma_DenoisesAndReturnsTokensWithNoMask()
    {
        SyntheticCheckpoint chk = WriteSyntheticCheckpoint(seed: 123);

        var (model, source, config) = ModelLoader.LoadFromSafetensors(_scratch);
        using (model)
        using (source)
        {
            // EOS = -1 so no argmax token id (0..vocab-1) is ever read as EOS — the
            // canvas always fills to its full length, isolating the denoise loop.
            var tokenizer = new StubTokenizer(config.VocabSize, maskTokenId: chk.MaskTokenId, eosTokenId: -1);
            var gen = new DotLLM.Engine.DiffusionTextGenerator(model, tokenizer);

            DotLLM.Engine.DiffusionResult result = gen.Generate([1, 2, 3]);

            Assert.Equal(chk.CanvasLength, result.GeneratedTokenCount);
            Assert.Equal(chk.CanvasLength, result.GeneratedTokenIds.Length);
            Assert.Equal(1, result.CanvasCount);
            Assert.True(result.TotalDenoisingSteps >= 1);
            Assert.DoesNotContain(chk.MaskTokenId, result.GeneratedTokenIds);
            // Every committed token is a valid vocab id.
            Assert.All(result.GeneratedTokenIds, t => Assert.InRange(t, 0, config.VocabSize - 1));
        }
    }

    // ───────────────────────── synthetic checkpoint writer ─────────────────────────

    // Tiny synthetic diffusion-gemma dimensions. UNIFORM head_dim (global == sliding)
    // so the #36 head-dim guard is NOT hit; small expert count for a fast forward.
    private const int SynHidden = 16;
    private const int SynLayers = 4;
    private const int SynHeads = 4;
    private const int SynHeadDim = SynHidden / SynHeads; // 4
    private const int SynKvHeads = 2;        // sliding KV heads
    private const int SynGlobalKvHeads = 1;  // full-attn KV heads (dual KV count, same head_dim)
    private const int SynExperts = 6;
    private const int SynTopK = 2;
    private const int SynMoeInter = 12;
    private const int SynInter = 12;
    private const int SynVocab = 16;
    private const int SynSlidingWindow = 2;
    private const int SynCanvas = 4;
    private const int SynMaskToken = 15; // within [0, SynVocab)

    private readonly record struct SyntheticCheckpoint(int MaskTokenId, int CanvasLength);

    /// <summary>
    /// Writes a complete tiny diffusion_gemma checkpoint to the scratch dir:
    /// a single-shard Gemma-4 MoE safetensors file (Qwen-MoE expert naming), a
    /// top-level diffusion_gemma config.json with a nested text_config + uniform
    /// head_dim, a generation_config.json, and tokenizer metadata declaring the
    /// mask token. Full-attention layers are 1,3 (every 2nd, 1-indexed).
    /// </summary>
    private SyntheticCheckpoint WriteSyntheticCheckpoint(int seed)
    {
        WriteSyntheticWeights(Path.Combine(_scratch, "model.safetensors"), seed);

        // layer_types: layers 1 and 3 full, 0 and 2 sliding.
        string layerTypes = "[\"sliding_attention\",\"full_attention\",\"sliding_attention\",\"full_attention\"]";

        string config = $$"""
            {
                "model_type": "diffusion_gemma",
                "architectures": ["DiffusionGemmaForBlockDiffusion"],
                "canvas_length": {{SynCanvas}},
                "tie_word_embeddings": false,
                "vision_config": { "hidden_size": 99, "num_hidden_layers": 1 },
                "text_config": {
                    "model_type": "diffusion_gemma_text",
                    "hidden_size": {{SynHidden}},
                    "num_hidden_layers": {{SynLayers}},
                    "num_attention_heads": {{SynHeads}},
                    "num_key_value_heads": {{SynKvHeads}},
                    "num_global_key_value_heads": {{SynGlobalKvHeads}},
                    "head_dim": {{SynHeadDim}},
                    "global_head_dim": {{SynHeadDim}},
                    "intermediate_size": {{SynInter}},
                    "moe_intermediate_size": {{SynMoeInter}},
                    "num_experts": {{SynExperts}},
                    "top_k_experts": {{SynTopK}},
                    "vocab_size": {{SynVocab}},
                    "hidden_activation": "gelu_pytorch_tanh",
                    "rms_norm_eps": 1e-6,
                    "final_logit_softcapping": 30.0,
                    "sliding_window": {{SynSlidingWindow}},
                    "layer_types": {{layerTypes}},
                    "max_position_embeddings": 64,
                    "rope_parameters": {
                        "full_attention": { "rope_theta": 1000000.0, "partial_rotary_factor": 0.25, "rope_type": "proportional" },
                        "sliding_attention": { "rope_theta": 10000.0 }
                    }
                }
            }
            """;
        File.WriteAllText(Path.Combine(_scratch, "config.json"), config);
        File.WriteAllText(Path.Combine(_scratch, "generation_config.json"), RealGenerationConfigJson);
        WriteTokenizerFixture(_scratch, SynMaskToken);

        return new SyntheticCheckpoint(SynMaskToken, SynCanvas);
    }

    private static void WriteSyntheticWeights(string path, int seed)
    {
        var b = new SafetensorsFixtureBuilder();
        int qStride = SynHeads * SynHeadDim; // = SynHidden

        AddRand(b, "model.embed_tokens.weight", [SynVocab, SynHidden], 0.1f, seed + 0);
        AddRand(b, "model.norm.weight", [SynHidden], 0.05f, seed + 1);
        AddRand(b, "lm_head.weight", [SynVocab, SynHidden], 0.1f, seed + 2);

        for (int i = 0; i < SynLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            string prefix = $"model.layers.{i}";
            bool isFull = (i % 2) == 1; // layers 1,3 full
            int layerKvHeads = isFull ? SynGlobalKvHeads : SynKvHeads;
            int kvStride = layerKvHeads * SynHeadDim;

            AddRand(b, $"{prefix}.input_layernorm.weight", [SynHidden], 0.05f, s + 0);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [SynHidden], 0.10f, s + 1);
            AddRand(b, $"{prefix}.pre_feedforward_layernorm.weight", [SynHidden], 0.05f, s + 9);
            AddRand(b, $"{prefix}.post_feedforward_layernorm.weight", [SynHidden], 0.10f, s + 10);

            AddRand(b, $"{prefix}.self_attn.q_proj.weight", [qStride, SynHidden], 0.1f, s + 2);
            AddRand(b, $"{prefix}.self_attn.k_proj.weight", [kvStride, SynHidden], 0.1f, s + 3);
            AddRand(b, $"{prefix}.self_attn.v_proj.weight", [kvStride, SynHidden], 0.1f, s + 4);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight", [SynHidden, qStride], 0.1f, s + 5);
            AddRand(b, $"{prefix}.self_attn.q_norm.weight", [SynHeadDim], 0.05f, s + 11);
            AddRand(b, $"{prefix}.self_attn.k_norm.weight", [SynHeadDim], 0.05f, s + 12);

            AddRand(b, $"{prefix}.mlp.gate.weight", [SynExperts, SynHidden], 0.2f, s + 6);
            for (int e = 0; e < SynExperts; e++)
            {
                int es = s + 100 + 7 * e;
                AddRand(b, $"{prefix}.mlp.experts.{e}.gate_proj.weight", [SynMoeInter, SynHidden], 0.10f, es + 0);
                AddRand(b, $"{prefix}.mlp.experts.{e}.up_proj.weight", [SynMoeInter, SynHidden], 0.10f, es + 1);
                AddRand(b, $"{prefix}.mlp.experts.{e}.down_proj.weight", [SynHidden, SynMoeInter], 0.05f, es + 2);
            }
        }

        b.WriteTo(path);
    }

    private static void AddRand(SafetensorsFixtureBuilder b, string name, int[] shape, float amplitude, int seed)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
            values[i] = amplitude * MathF.Cos(0.61803398875f * (i + 1) + seed * 0.37f);
        b.AddFloat32(name, shape, values);
    }

    /// <summary>
    /// Writes the minimum tokenizer metadata that
    /// <see cref="DiffusionConfigExtractor.ResolveMaskTokenId(string)"/> needs:
    /// a special_tokens_map declaring the mask content and a tokenizer_config
    /// whose added_tokens_decoder maps the id to that content.
    /// </summary>
    private static void WriteTokenizerFixture(string dir, int maskTokenId)
    {
        File.WriteAllText(Path.Combine(dir, "special_tokens_map.json"),
            """ { "mask_token": "[MASK]" } """);
        File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"),
            $$"""
            {
                "mask_token": "[MASK]",
                "added_tokens_decoder": {
                    "0": { "content": "<pad>" },
                    "{{maskTokenId}}": { "content": "[MASK]", "special": true }
                }
            }
            """);
    }

    // ───────────────────────── verified real-26B config fixtures ─────────────────────────

    // Mirrors the verified google/diffusiongemma-26B-A4B-it config.json structure
    // (docs/diffusiongemma/README.md). global_head_dim 512 != head_dim 256 → #36.
    private const string RealConfigJson = """
        {
            "model_type": "diffusion_gemma",
            "architectures": ["DiffusionGemmaForBlockDiffusion"],
            "canvas_length": 256,
            "dtype": "bfloat16",
            "tie_word_embeddings": true,
            "use_bidirectional_attention": "vision",
            "vision_config": { "hidden_size": 1152, "num_hidden_layers": 27 },
            "text_config": {
                "model_type": "diffusion_gemma_text",
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 2,
                "head_dim": 256,
                "global_head_dim": 512,
                "intermediate_size": 2112,
                "moe_intermediate_size": 704,
                "num_experts": 128,
                "top_k_experts": 8,
                "vocab_size": 262144,
                "hidden_activation": "gelu_pytorch_tanh",
                "rms_norm_eps": 1e-6,
                "final_logit_softcapping": 30.0,
                "sliding_window": 1024,
                "sliding_window_pattern": 6,
                "layer_types": [
                    "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"
                ],
                "max_position_embeddings": 262144,
                "rope_parameters": {
                    "full_attention": { "rope_theta": 1000000.0, "partial_rotary_factor": 0.25, "rope_type": "proportional" },
                    "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
                }
            }
        }
        """;

    private const string RealGenerationConfigJson = """
        {
            "max_denoising_steps": 48,
            "entropy_bound": 0.1,
            "confidence_threshold": 0.005,
            "stability_threshold": 1,
            "t_max": 0.8,
            "t_min": 0.4
        }
        """;

    // ───────────────────────── stub tokenizer ─────────────────────────

    /// <summary>Trivial identity tokenizer for the integration test: EOS = 0, mask within vocab.</summary>
    private sealed class StubTokenizer(int vocabSize, int maskTokenId, int eosTokenId) : DotLLM.Tokenizers.ITokenizer
    {
        public int VocabSize => vocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => eosTokenId;
        public int MaskTokenId => maskTokenId;

        public int[] Encode(string text) => string.IsNullOrEmpty(text) ? [] : [1];
        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            var sb = new System.Text.StringBuilder();
            foreach (int id in tokenIds) sb.Append('t').Append(id).Append(' ');
            return sb.ToString().TrimEnd();
        }
        public string DecodeToken(int tokenId) => $"t{tokenId}";
        public int CountTokens(string text) => Encode(text).Length;
    }
}
