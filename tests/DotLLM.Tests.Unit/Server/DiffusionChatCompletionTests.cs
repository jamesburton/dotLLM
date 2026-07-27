using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tests.Unit.Models.SafeTensors;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Server-integration tests for the diffusion decode routing added in DiffusionGemma PR-10
/// (issue #34). All CPU-only and download-free: a tiny synthetic diffusion_gemma checkpoint
/// (uniform head_dim, reusing the <see cref="DotLLM.Tests.Unit.Models.SafeTensors"/> fixture pattern)
/// is loaded through <see cref="ModelLoader"/> and wired into a <see cref="ServerState"/> exactly as
/// <see cref="ServerStartup.LoadModel"/> does for a diffusion model. The endpoint routing helpers are
/// exercised directly (the private HTTP <c>HandleAsync</c> is covered indirectly through the same
/// helpers it delegates to). Full HTTP round-trip and real-weight end-to-end are issue #32.
/// </summary>
public sealed class DiffusionChatCompletionTests : IDisposable
{
    // Tiny synthetic diffusion-gemma dimensions — UNIFORM head_dim so the #36 head-dim gate is not hit.
    private const int SynHidden = 16;
    private const int SynLayers = 4;
    private const int SynHeads = 4;
    private const int SynHeadDim = SynHidden / SynHeads; // 4
    private const int SynKvHeads = 2;
    private const int SynGlobalKvHeads = 1;
    private const int SynExperts = 6;
    private const int SynTopK = 2;
    private const int SynMoeInter = 12;
    private const int SynInter = 12;
    private const int SynVocab = 16;
    private const int SynSlidingWindow = 2;
    private const int SynCanvas = 4;
    private const int SynMaskToken = 15; // within [0, SynVocab)

    private readonly string _scratch;
    private readonly List<IDisposable> _sources = new();

    public DiffusionChatCompletionTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-dg-server-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        foreach (var s in _sources)
        {
            try { s.Dispose(); } catch { /* best-effort */ }
        }
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // ───────────────────────── Routing: diffusion vs autoregressive ─────────────────────────

    /// <summary>
    /// A diffusion model wired the way <see cref="ServerStartup.LoadModel"/> wires it carries a
    /// non-null <see cref="ServerState.DiffusionGenerator"/> alongside the AR <c>Generator</c>; an
    /// autoregressive ServerState leaves <c>DiffusionGenerator</c> null. This is the routing switch the
    /// endpoint branches on.
    /// </summary>
    [Fact]
    public void DiffusionModel_PopulatesDiffusionGenerator_ArModelLeavesItNull()
    {
        using var diffusionState = BuildDiffusionServerState(out _);
        Assert.NotNull(diffusionState.DiffusionGenerator);
        Assert.NotNull(diffusionState.Generator);          // AR generator still constructed
        Assert.NotNull(diffusionState.Config!.DiffusionConfig);

        // A non-diffusion ServerState (no model / no diffusion config) never sets the diffusion gen.
        using var arState = new ServerState { Options = new ServerOptions { Model = "ar" } };
        Assert.Null(arState.DiffusionGenerator);
    }

    // ───────────────────────── Non-streaming decode ─────────────────────────

    /// <summary>
    /// Non-streaming chat completion on the diffusion model produces decoded text and a usage count
    /// matching the generated tokens. Exercises the same <see cref="DiffusionTextGenerator.Generate(string, int?, System.Action{DiffusionCanvasState}?)"/>
    /// call the non-streaming endpoint handler makes, and asserts the OpenAI-shaped response fields.
    /// </summary>
    [Fact]
    public void NonStreaming_DiffusionModel_ReturnsDecodedTextAndUsage()
    {
        using var state = BuildDiffusionServerState(out _);
        var gen = state.DiffusionGenerator!;

        DiffusionResult result = gen.Generate("hello", targetLength: SynCanvas);

        Assert.Equal(SynCanvas, result.GeneratedTokenCount);
        Assert.Equal(SynCanvas, result.GeneratedTokenIds.Length);
        Assert.DoesNotContain(SynMaskToken, result.GeneratedTokenIds);
        Assert.False(string.IsNullOrEmpty(result.Text));

        // Mirror the response the endpoint builds.
        var response = new ChatCompletionResponse
        {
            Id = "test",
            Model = "diffusion",
            Choices = [new ChatChoiceDto
            {
                Index = 0,
                Message = new ChatMessageDto { Role = "assistant", Content = result.Text },
                FinishReason = RequestConverter.ToFinishReasonString(result.FinishReason),
            }],
            Usage = new UsageDto
            {
                PromptTokens = result.PromptTokenCount,
                CompletionTokens = result.GeneratedTokenCount,
                TotalTokens = result.PromptTokenCount + result.GeneratedTokenCount,
            },
        };

        Assert.Equal(result.Text, response.Choices[0].Message.Content);
        Assert.Equal(SynCanvas, response.Usage.CompletionTokens);
        Assert.True(response.Usage.TotalTokens >= response.Usage.CompletionTokens);
    }

    // ───────────────────────── Streaming SSE delta mapping ─────────────────────────

    /// <summary>
    /// The streaming routing helper maps the canvas-streaming hook to progressive decoded-text deltas:
    /// concatenating every emitted delta reconstructs the final decoded text, and at least one delta is
    /// observed (the canvas reveals at least one committed-prefix growth as it denoises).
    /// </summary>
    [Fact]
    public void Streaming_DiffusionModel_EmitsProgressiveDeltasThatReconstructFinalText()
    {
        using var state = BuildDiffusionServerState(out _);
        var gen = state.DiffusionGenerator!;

        var deltas = new List<string>();
        DiffusionResult result = ChatCompletionEndpoint.RunDiffusionStreaming(gen, "hello", SynCanvas, deltas);

        // The progressive deltas are the growth of the leading committed run as the canvas denoises.
        Assert.NotEmpty(deltas);
        // Production-contract invariant: the streamed prefix is always a prefix of the final text, and
        // the handler's final tail chunk emits exactly the remainder — together they reconstruct it.
        string streamed = string.Concat(deltas);
        Assert.StartsWith(streamed, result.Text, StringComparison.Ordinal);
        string tail = result.Text.Length > streamed.Length ? result.Text[streamed.Length..] : string.Empty;
        Assert.Equal(result.Text, streamed + tail);
        Assert.False(string.IsNullOrEmpty(result.Text));
    }

    // ───────────────────────── Per-request diffusion overrides ─────────────────────────

    /// <summary>
    /// With no diffusion overrides the load-time generator is reused as-is; with an override present a
    /// fresh per-request generator is built. Either generator decodes a valid canvas.
    /// </summary>
    [Fact]
    public void Overrides_AbsentReuseLoadTime_PresentBuildFreshGenerator()
    {
        using var state = BuildDiffusionServerState(out _);
        var loadTime = state.DiffusionGenerator!;

        var same = ChatCompletionEndpoint.ResolveDiffusionGenerator(loadTime, state, overrides: null);
        Assert.Same(loadTime, same);

        var noFields = ChatCompletionEndpoint.ResolveDiffusionGenerator(
            loadTime, state, new DiffusionOptionsDto());
        Assert.Same(loadTime, noFields);

        var overridden = ChatCompletionEndpoint.ResolveDiffusionGenerator(
            loadTime, state, new DiffusionOptionsDto { MaxDenoisingSteps = 3, CanvasLength = SynCanvas });
        Assert.NotSame(loadTime, overridden);

        // The overridden generator still decodes a full canvas.
        DiffusionResult result = overridden.Generate("hello", targetLength: SynCanvas);
        Assert.Equal(SynCanvas, result.GeneratedTokenCount);
        Assert.DoesNotContain(SynMaskToken, result.GeneratedTokenIds);
    }

    // ───────────────────────── Warm-up (CPU, no GPU) ─────────────────────────

    /// <summary>
    /// The diffusion warm-up path runs a small canvas on CPU with no GPU dependency and does not throw.
    /// </summary>
    [Fact]
    public void Warmup_DiffusionModel_RunsCanvasWithoutGpu()
    {
        using var state = BuildDiffusionServerState(out var tokenizer);
        var ex = Record.Exception(() =>
            WarmupRunner.RunDiffusion(state.DiffusionGenerator!, tokenizer,
                new WarmupOptions { Enabled = true, Iterations = 1, MaxTokens = SynCanvas }));
        Assert.Null(ex);
    }

    // ───────────────────────── fixture wiring ─────────────────────────

    /// <summary>
    /// Loads the tiny synthetic diffusion_gemma checkpoint through <see cref="ModelLoader"/> and wires a
    /// <see cref="ServerState"/> exactly as <see cref="ServerStartup.LoadModel"/> does for a diffusion
    /// model (AR generator + diffusion generator). The caller owns disposal (disposes the ServerState).
    /// </summary>
    private ServerState BuildDiffusionServerState(out ITokenizer tokenizer)
    {
        WriteSyntheticCheckpoint(seed: 123);

        var (model, source, config) = ModelLoader.LoadFromSafetensors(_scratch);
        _sources.Add(source); // mmap source kept alive for the model's lifetime; disposed in test teardown
        Assert.Equal(Architecture.DiffusionGemma, config.Architecture);
        Assert.NotNull(config.DiffusionConfig);

        // EOS = -1 so no committed token id is ever read as EOS — the canvas always fills fully,
        // isolating the denoise/routing path under test (matches the engine tests' stub).
        var tok = new StubTokenizer(config.VocabSize, maskTokenId: config.DiffusionConfig!.MaskTokenId, eosTokenId: -1);
        tokenizer = tok;

        var diffusionGenerator = new DiffusionTextGenerator(model, tok, sampler: null, config.DiffusionConfig);
        var arGenerator = new TextGenerator(model, tok, kvCacheFactory: null, prefixCache: null);

        return new ServerState
        {
            Options = new ServerOptions { Model = "synthetic-diffusion-gemma", ModelId = "diffusion" },
            Config = config,
            IsReady = true,
            Model = model,
            Tokenizer = tok,
            Generator = arGenerator,
            DiffusionGenerator = diffusionGenerator,
        };
    }

    // ───────────────────────── synthetic checkpoint writer (mirrors PR-8 fixture) ─────────────────────────

    private void WriteSyntheticCheckpoint(int seed)
    {
        WriteSyntheticWeights(Path.Combine(_scratch, "model.safetensors"), seed);

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
        File.WriteAllText(Path.Combine(_scratch, "generation_config.json"),
            """
            { "max_denoising_steps": 6, "entropy_bound": 0.1, "confidence_threshold": 0.005, "stability_threshold": 1, "t_max": 0.8, "t_min": 0.4 }
            """);
        WriteTokenizerFixture(_scratch, SynMaskToken);
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
            bool isFull = (i % 2) == 1;
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

    /// <summary>Trivial identity tokenizer: ids ↔ "t{id}". Mask within vocab; EOS configurable.</summary>
    private sealed class StubTokenizer(int vocabSize, int maskTokenId, int eosTokenId) : ITokenizer
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
