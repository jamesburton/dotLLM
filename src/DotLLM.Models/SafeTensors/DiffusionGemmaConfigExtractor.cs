using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Parses a DiffusionGemma checkpoint's <c>config.json</c>
/// (<c>model_type=diffusion_gemma</c>,
/// <c>architectures=["DiffusionGemmaForBlockDiffusion"]</c>) into a populated
/// Gemma-4 MoE <see cref="ModelConfig"/> carrying
/// <see cref="Architecture.DiffusionGemma"/> and a non-null
/// <see cref="ModelConfig.DiffusionConfig"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a dedicated extractor.</b> The DiffusionGemma wrapper houses the
/// Gemma-4 text tower under a <c>text_config</c> sub-object (mirroring Gemma 3's
/// multimodal hoist in <see cref="HfConfigExtractor"/>) and adds a top-level
/// <c>canvas_length</c> + a <c>vision_config</c> we deliberately skip (text-only;
/// multimodal is out of scope). Crucially, the generic
/// <see cref="HfConfigExtractor.Extract(JsonElement)"/> path populates only the
/// dense Gemma-3 fields for <see cref="Architecture.Gemma4"/> — it does NOT read
/// the Gemma-4-specific <c>num_global_key_value_heads</c>, <c>global_head_dim</c>,
/// per-attention-type <c>rope_parameters</c>, MoE block, soft-cap, or embedding
/// scale. This extractor therefore builds the full Gemma-4 MoE
/// <see cref="ModelConfig"/> directly from the (hoisted) text-tower config.
/// </para>
/// <para>
/// <b>Per-attention-type RoPE.</b> Real DiffusionGemma ships <c>rope_parameters</c>
/// split by attention type: full-attention layers use base <c>rope_theta = 1e6</c>
/// with a <c>partial_rotary_factor = 0.25</c> (<c>rope_type = proportional</c>);
/// sliding-window layers use base <c>rope_theta = 1e4</c> with full rotation. The
/// sliding schedule lands on <see cref="ModelConfig.RoPEConfig"/>; the full schedule
/// on <see cref="ModelConfig.GlobalRoPEConfig"/> + <see cref="ModelConfig.PartialRotaryFactor"/>.
/// </para>
/// <para>
/// <b>Mask token.</b> The diffusion mask token id is not in any config file; it is
/// resolved from the checkpoint tokenizer metadata via
/// <see cref="DiffusionConfigExtractor.ResolveMaskTokenId(string)"/>.
/// </para>
/// <para>
/// <b>Forward gate (#36).</b> The real 26B checkpoint has
/// <c>global_head_dim = 512 != head_dim = 256</c>; this extractor parses that into a
/// faithful <see cref="ModelConfig"/> (it does NOT throw), but the CPU forward path
/// rejects a non-uniform head_dim at model-build time
/// (<c>ValidateGemma4UniformHeadDim</c>, issue #36). Real-weight forward therefore
/// awaits #36; synthetic fixtures with a uniform head_dim run end-to-end today.
/// </para>
/// </remarks>
public static class DiffusionGemmaConfigExtractor
{
    /// <summary>
    /// Parses a DiffusionGemma <c>config.json</c> from a checkpoint directory:
    /// builds the Gemma-4 MoE <see cref="ModelConfig"/> and attaches a
    /// <see cref="DiffusionConfig"/> (reading <c>generation_config.json</c> and
    /// resolving the mask token id from the directory's tokenizer metadata).
    /// </summary>
    /// <param name="root">Parsed <c>config.json</c> root (top-level diffusion_gemma object).</param>
    /// <param name="weightsDir">Directory holding the checkpoint JSON metadata
    /// (<c>generation_config.json</c>, tokenizer files).</param>
    /// <returns>A populated Gemma-4 MoE <see cref="ModelConfig"/> with a non-null
    /// <see cref="ModelConfig.DiffusionConfig"/>.</returns>
    /// <exception cref="InvalidDataException">The config is malformed, declares an
    /// unexpected <c>model_type</c>, or the mask token id cannot be resolved.</exception>
    public static ModelConfig ExtractFromDirectory(JsonElement root, string weightsDir)
    {
        ArgumentNullException.ThrowIfNull(weightsDir);

        // canvas_length lives at the TOP level (outside text_config). Read it
        // before hoisting so it is taken from the wrapper, not the text tower.
        DiffusionConfig diffusion = DiffusionConfigExtractor.ExtractFromDirectory(weightsDir, root);

        ModelConfig textConfig = ExtractTextConfig(root);
        return textConfig with { DiffusionConfig = diffusion };
    }

    /// <summary>
    /// Parses a DiffusionGemma <c>config.json</c> root into a Gemma-4 MoE
    /// <see cref="ModelConfig"/> WITHOUT attaching a <see cref="DiffusionConfig"/>.
    /// The pure, file-system-free core — the mask token id and denoising schedule
    /// are supplied by the caller (see <see cref="ExtractFromDirectory"/>) or, in
    /// tests, attached directly. Hoists <c>text_config</c> and reads the top-level
    /// <c>canvas_length</c> (default 256) but does not consult the tokenizer.
    /// </summary>
    /// <param name="root">Parsed <c>config.json</c> root (top-level diffusion_gemma object).</param>
    /// <returns>A populated Gemma-4 MoE <see cref="ModelConfig"/> with
    /// <see cref="ModelConfig.DiffusionConfig"/> left null.</returns>
    /// <exception cref="InvalidDataException">The config is malformed or declares an
    /// unexpected <c>model_type</c>.</exception>
    public static ModelConfig ExtractTextConfig(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("DiffusionGemma config.json root must be a JSON object.");

        // Validate the wrapper discriminators when present (don't hard-require
        // them — tiny synthetic fixtures may omit architectures[]).
        string? modelType = GetStringOrNull(root, "model_type");
        if (modelType is not null
            && !string.Equals(modelType, "diffusion_gemma", StringComparison.Ordinal)
            && !string.Equals(modelType, "diffusion_gemma_text", StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"DiffusionGemma config extractor requires model_type='diffusion_gemma' / "
                + $"'diffusion_gemma_text', got '{modelType}'.");
        }

        // Hoist the Gemma-4 text tower. Real diffusion_gemma houses the tower
        // (hidden_size, layers, rope_parameters, MoE, …) under `text_config`; the
        // top level carries canvas_length + vision_config (skipped). When the key
        // is absent (a flat text-only config / synthetic fixture) treat the root
        // itself as the text tower.
        JsonElement text = root;
        if (root.TryGetProperty("text_config", out var tc) && tc.ValueKind == JsonValueKind.Object)
            text = tc;

        int hiddenSize = GetInt32(text, "hidden_size");
        int numLayers = GetInt32(text, "num_hidden_layers");
        int numAttentionHeads = GetInt32(text, "num_attention_heads");
        int numKvHeads = GetInt32OrDefault(text, "num_key_value_heads", numAttentionHeads);
        int intermediateSize = GetInt32(text, "intermediate_size");
        int vocabSize = GetInt32(text, "vocab_size");
        int maxSeqLen = GetInt32OrDefault(text, "max_position_embeddings", 2048);
        int headDim = GetInt32OrDefault(text, "head_dim", hiddenSize / numAttentionHeads);

        // Dual KV-head count / head dim for the FULL-attention layers. global_head_dim
        // defaults to head_dim (uniform) when omitted. The real 26B ships 512 here vs
        // head_dim 256 — that distinct value is carried faithfully; the forward path
        // gates on it (#36), config extraction never throws on it.
        int? numGlobalKvHeads = GetInt32NullableIfPositive(text, "num_global_key_value_heads");
        int globalHeadDim = GetInt32OrDefault(text, "global_head_dim", headDim);

        float normEps = GetFloatOrDefault(text, "rms_norm_eps",
            GetFloatOrDefault(text, "layer_norm_eps", 1e-6f));
        bool tieEmbeddings = GetBoolOrDefault(text, "tie_word_embeddings",
            // diffusion_gemma ties by default (Gemma convention); the wrapper top
            // level also carries the flag — prefer the text-tower value, fall back
            // to the wrapper, then to the Gemma default (true).
            GetBoolOrDefault(root, "tie_word_embeddings", true));

        // Sliding-window pattern. Real diffusion_gemma ships sliding_window=1024 and
        // a 30-entry layer_types list with full attention at layers 5,11,17,23,29
        // (every 6th, 1-indexed). Default sliding_window 1024 + pattern 6 when absent.
        int slidingWindow = GetInt32OrDefault(text, "sliding_window", 1024);
        if (slidingWindow <= 0) slidingWindow = 1024;
        IReadOnlyList<int?> perLayerSlidingWindow = BuildPerLayerSlidingWindow(text, numLayers, slidingWindow);

        // Final-logit soft-cap (30.0 on real diffusion_gemma; no attn soft-cap).
        float? attnLogitSoftcap = GetFloatNullableIfPositive(text, "attn_logit_softcapping");
        float? finalLogitSoftcap = GetFloatNullableIfPositive(text, "final_logit_softcapping");
        int qpas = GetInt32OrDefault(text, "query_pre_attn_scalar", 0);
        float? queryPreAttnScalar = qpas > 0 ? qpas : null;

        // GeGLU activation (gelu_pytorch_tanh). Match HF naming variants defensively;
        // Gemma defaults to GeGLU-tanh.
        ActivationFunction activation = ResolveGemmaActivation(text);

        // Per-attention-type RoPE. Sliding layers → RoPEConfig; full layers →
        // GlobalRoPEConfig + PartialRotaryFactor.
        (RoPEConfig slidingRope, RoPEConfig globalRope, float? partialRotaryFactor) =
            ExtractPerAttnTypeRope(text, headDim);

        // MoE FFN. DiffusionGemma uses num_experts + top_k_experts (NOT the
        // Mixtral num_local_experts / num_experts_per_tok keys), with a per-expert
        // width moe_intermediate_size. Dense-only configs (no experts) leave Moe null.
        MoeConfig? moe = ExtractGemmaMoe(text, intermediateSize);

        return new ModelConfig
        {
            Architecture = Architecture.DiffusionGemma,
            VocabSize = vocabSize,
            HiddenSize = hiddenSize,
            IntermediateSize = intermediateSize,
            NumLayers = numLayers,
            NumAttentionHeads = numAttentionHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            MaxSequenceLength = maxSeqLen,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = slidingRope,
            GlobalRoPEConfig = globalRope,
            PartialRotaryFactor = partialRotaryFactor,
            NumGlobalKvHeads = numGlobalKvHeads,
            GlobalHeadDim = globalHeadDim,
            ActivationFunction = activation,
            NormType = NormType.RMSNorm,
            NormEpsilon = normEps,
            TiedEmbeddings = tieEmbeddings,
            SlidingWindowSize = slidingWindow,
            PerLayerSlidingWindow = perLayerSlidingWindow,
            AttnLogitSoftcap = attnLogitSoftcap,
            FinalLogitSoftcap = finalLogitSoftcap,
            QueryPreAttnScalar = queryPreAttnScalar,
            // Gemma scales input embeddings by sqrt(hidden_size).
            EmbeddingScale = MathF.Sqrt(hiddenSize),
            Moe = moe,
            ChatTemplate = null,
            DiffusionConfig = null,
        };
    }

    /// <summary>
    /// Builds the per-layer sliding-window override (full-attention layers carry a
    /// null window). Prefers the explicit <c>layer_types</c> array ("sliding_attention"
    /// / "full_attention" per layer) when present and correctly sized; otherwise
    /// falls back to the <c>sliding_window_pattern</c> formula (every Nth layer,
    /// 1-indexed, is full attention).
    /// </summary>
    private static IReadOnlyList<int?> BuildPerLayerSlidingWindow(
        JsonElement text, int numLayers, int slidingWindow)
    {
        var layerTypes = new int?[numLayers];

        if (text.TryGetProperty("layer_types", out var lt) && lt.ValueKind == JsonValueKind.Array
            && lt.GetArrayLength() == numLayers)
        {
            int i = 0;
            foreach (var el in lt.EnumerateArray())
            {
                string? s = el.ValueKind == JsonValueKind.String ? el.GetString() : null;
                bool isFull = string.Equals(s, "full_attention", StringComparison.Ordinal);
                layerTypes[i] = isFull ? null : slidingWindow;
                i++;
            }
            return layerTypes;
        }

        int swPattern = GetInt32OrDefault(text, "sliding_window_pattern", 6);
        if (swPattern <= 0) swPattern = 1;
        for (int i = 0; i < numLayers; i++)
        {
            bool isFull = ((i + 1) % swPattern) == 0; // HF Gemma formula
            layerTypes[i] = isFull ? null : slidingWindow;
        }
        return layerTypes;
    }

    /// <summary>
    /// Extracts the per-attention-type RoPE schedule. Reads the modern
    /// <c>rope_parameters</c> block (with <c>full_attention</c> / <c>sliding_attention</c>
    /// sub-objects, each carrying <c>rope_theta</c> + optional
    /// <c>partial_rotary_factor</c>) when present; otherwise falls back to the flat
    /// <c>rope_theta</c> (+ optional top-level <c>partial_rotary_factor</c>) with the
    /// verified DiffusionGemma defaults (full theta 1e6 / partial 0.25, sliding theta 1e4).
    /// </summary>
    private static (RoPEConfig Sliding, RoPEConfig Global, float? PartialRotaryFactor)
        ExtractPerAttnTypeRope(JsonElement text, int headDim)
    {
        // Verified DiffusionGemma defaults.
        float slidingTheta = 10_000.0f;
        float globalTheta = 1_000_000.0f;
        float? partialRotaryFactor = 0.25f;

        // Flat fallbacks first (so a single rope_theta still moves both, but the
        // per-type defaults win where a flat key is absent).
        float flatTheta = GetFloatOrDefault(text, "rope_theta", float.NaN);
        if (!float.IsNaN(flatTheta))
        {
            // A single flat theta typically describes the sliding (local) base on
            // Gemma; keep the verified global default unless rope_parameters overrides.
            slidingTheta = flatTheta;
        }
        float flatPrf = GetFloatOrDefault(text, "partial_rotary_factor", float.NaN);
        if (!float.IsNaN(flatPrf))
            partialRotaryFactor = flatPrf > 0f ? flatPrf : null;

        if (text.TryGetProperty("rope_parameters", out var rp) && rp.ValueKind == JsonValueKind.Object)
        {
            // Sliding sub-object.
            if (TryGetObject(rp, "sliding_attention", out var sl)
                || TryGetObject(rp, "local_attention", out sl)
                || TryGetObject(rp, "sliding", out sl))
            {
                float t = GetFloatOrDefault(sl, "rope_theta", slidingTheta);
                if (t > 0f) slidingTheta = t;
            }

            // Full sub-object: theta + partial rotary factor.
            if (TryGetObject(rp, "full_attention", out var fu)
                || TryGetObject(rp, "global_attention", out fu)
                || TryGetObject(rp, "full", out fu))
            {
                float t = GetFloatOrDefault(fu, "rope_theta", globalTheta);
                if (t > 0f) globalTheta = t;
                float prf = GetFloatOrDefault(fu, "partial_rotary_factor", float.NaN);
                if (!float.IsNaN(prf))
                    partialRotaryFactor = prf > 0f ? prf : null;
            }
        }

        var slidingRope = new RoPEConfig(
            Theta: slidingTheta,
            DimensionCount: headDim,
            Type: RoPEType.NeoX);

        var globalRope = new RoPEConfig(
            Theta: globalTheta,
            DimensionCount: headDim,
            Type: RoPEType.NeoX);

        return (slidingRope, globalRope, partialRotaryFactor);
    }

    /// <summary>
    /// Detects the DiffusionGemma MoE FFN. Recognises the Gemma key names
    /// <c>num_experts</c> + <c>top_k_experts</c> (and the Mixtral/Qwen aliases as a
    /// fallback) plus the per-expert width <c>moe_intermediate_size</c>. Returns null
    /// when no expert count is declared (a dense Gemma-4 variant).
    /// </summary>
    private static MoeConfig? ExtractGemmaMoe(JsonElement text, int defaultIntermediateSize)
    {
        int numExperts = GetInt32OrDefault(text, "num_experts", 0);
        if (numExperts <= 0)
            numExperts = GetInt32OrDefault(text, "num_local_experts", 0);
        if (numExperts <= 0)
            return null;

        int topK = GetInt32OrDefault(text, "top_k_experts", 0);
        if (topK <= 0)
            topK = GetInt32OrDefault(text, "num_experts_per_tok", 0);
        if (topK <= 0)
            throw new InvalidDataException(
                $"DiffusionGemma config declares {numExperts} MoE experts but is missing or has "
                + "an invalid 'top_k_experts' / 'num_experts_per_tok'.");
        if (topK > numExperts)
            throw new InvalidDataException(
                $"DiffusionGemma config has top_k_experts={topK} > num_experts={numExperts}.");

        int moeIntermediate = GetInt32OrDefault(text, "moe_intermediate_size", defaultIntermediateSize);
        bool normTopKProb = GetBoolOrDefault(text, "norm_topk_prob", true);

        return new MoeConfig
        {
            NumExperts = numExperts,
            NumExpertsPerTok = topK,
            MoeIntermediateSize = moeIntermediate,
            NormTopKProb = normTopKProb,
        };
    }

    private static ActivationFunction ResolveGemmaActivation(JsonElement text)
    {
        string? hiddenAct = GetStringOrNull(text, "hidden_activation")
                          ?? GetStringOrNull(text, "hidden_act");
        return (hiddenAct?.ToLowerInvariant()) switch
        {
            "gelu_pytorch_tanh" or "gelu_new" or "gelu_tanh" or "gelu_fast" => ActivationFunction.GELUTanh,
            "gelu" => ActivationFunction.GELU,
            "silu" or "swish" or null => ActivationFunction.GELUTanh, // Gemma default
            _ => ActivationFunction.GELUTanh,
        };
    }

    // -------------------------------------------------------------------------
    // Low-level JSON helpers (kept local — HfConfigExtractor's are private).
    // -------------------------------------------------------------------------

    private static bool TryGetObject(JsonElement root, string key, out JsonElement value)
    {
        if (root.TryGetProperty(key, out value) && value.ValueKind == JsonValueKind.Object)
            return true;
        value = default;
        return false;
    }

    private static int GetInt32(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            throw new InvalidDataException($"DiffusionGemma config.json missing required integer key '{key}'.");
        if (!prop.TryGetInt32(out int value))
            throw new InvalidDataException($"DiffusionGemma config.json key '{key}' is not a 32-bit integer.");
        return value;
    }

    private static int GetInt32OrDefault(JsonElement root, string key, int fallback)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            return fallback;
        return prop.TryGetInt32(out int value) ? value : fallback;
    }

    private static int? GetInt32NullableIfPositive(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            return null;
        if (!prop.TryGetInt32(out int v)) return null;
        return v > 0 ? v : null;
    }

    private static float GetFloatOrDefault(JsonElement root, string key, float fallback)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            return fallback;
        return prop.TryGetSingle(out float value) ? value : fallback;
    }

    private static float? GetFloatNullableIfPositive(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            return null;
        if (!prop.TryGetSingle(out float v)) return null;
        return v > 0f ? v : null;
    }

    private static bool GetBoolOrDefault(JsonElement root, string key, bool fallback)
    {
        if (!root.TryGetProperty(key, out var prop)) return fallback;
        return prop.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => fallback,
        };
    }

    private static string? GetStringOrNull(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.String)
            return null;
        return prop.GetString();
    }
}
