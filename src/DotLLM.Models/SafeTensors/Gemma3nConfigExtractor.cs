using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Parses a Gemma-3n checkpoint's <c>config.json</c> (<c>model_type=gemma3n</c>,
/// <c>architectures=["Gemma3nForConditionalGeneration"]</c>) into a populated
/// <see cref="ModelConfig"/> carrying <see cref="Architecture.Gemma3n"/> and a
/// non-null <see cref="ModelConfig.Gemma3n"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a dedicated extractor.</b> Gemma-3n's real config (verified against
/// <c>google/gemma-3n-E4B-it</c>, mirrored on <c>unsloth/gemma-3n-E4B-it</c>)
/// houses the text tower under a <c>text_config</c> sub-object
/// (<c>model_type=gemma3n_text</c>) alongside <c>audio_config</c> /
/// <c>vision_config</c> we deliberately skip (text-only). It carries the
/// Gemma-3n-only AltUp/Laurel/activation-sparsity fields the generic
/// <see cref="HfConfigExtractor"/> knows nothing about, plus <b>flat</b>
/// per-attention-type RoPE bases (<c>rope_theta</c> for full-attention,
/// <c>rope_local_base_freq</c> for sliding — unlike Gemma-4/DiffusionGemma's
/// nested <c>rope_parameters</c> block) and, unlike Gemma-4, NO distinct
/// global head-dim / global KV-head count / partial-rotary factor — every
/// layer uses the same <c>head_dim</c> / <c>num_key_value_heads</c>, and both
/// RoPE schedules rotate the full head.
/// </para>
/// <para>
/// <b>MatFormer.</b> <c>intermediate_size</c> is a per-layer JSON array (the
/// real E4B ships 35 identical entries — MatFormer elastic-width slicing is
/// out of scope, matching the earlier Gemma-4 E2B PLE punt); a heterogeneous
/// array is rejected rather than silently averaged/truncated.
/// </para>
/// </remarks>
public static class Gemma3nConfigExtractor
{
    /// <summary>
    /// Parses a Gemma-3n <c>config.json</c> root into a <see cref="ModelConfig"/>.
    /// </summary>
    /// <param name="root">Parsed <c>config.json</c> root (top-level gemma3n object,
    /// or a flat text-only <c>gemma3n_text</c> config).</param>
    /// <exception cref="InvalidDataException">The config is malformed or declares
    /// an unexpected <c>model_type</c>.</exception>
    public static ModelConfig Extract(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("Gemma-3n config.json root must be a JSON object.");

        string? modelType = GetStringOrNull(root, "model_type");
        if (modelType is not null
            && !string.Equals(modelType, "gemma3n", StringComparison.Ordinal)
            && !string.Equals(modelType, "gemma3n_text", StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Gemma3nConfigExtractor requires model_type='gemma3n' / 'gemma3n_text', got '{modelType}'.");
        }

        // Hoist the text tower. Real gemma3n houses it under `text_config`
        // (audio_config / vision_config skipped — text-only); a flat
        // gemma3n_text config (or a synthetic fixture) treats the root itself
        // as the text tower.
        JsonElement text = root;
        if (root.TryGetProperty("text_config", out var tc) && tc.ValueKind == JsonValueKind.Object)
            text = tc;

        int hiddenSize = GetInt32(text, "hidden_size");
        int numLayers = GetInt32(text, "num_hidden_layers");
        int numAttentionHeads = GetInt32(text, "num_attention_heads");
        int numKvHeads = GetInt32OrDefault(text, "num_key_value_heads", numAttentionHeads);
        int headDim = GetInt32OrDefault(text, "head_dim", hiddenSize / numAttentionHeads);
        int intermediateSize = GetIntermediateSize(text, numLayers);
        int vocabSize = GetInt32(text, "vocab_size");
        int maxSeqLen = GetInt32OrDefault(text, "max_position_embeddings", 32768);

        float normEps = GetFloatOrDefault(text, "rms_norm_eps", 1e-6f);
        bool tieEmbeddings = GetBoolOrDefault(text, "tie_word_embeddings", true);

        int slidingWindow = GetInt32OrDefault(text, "sliding_window", 512);
        if (slidingWindow <= 0) slidingWindow = 512;
        IReadOnlyList<int?> perLayerSlidingWindow = BuildPerLayerSlidingWindow(text, numLayers, slidingWindow);

        float? finalLogitSoftcap = GetFloatNullableIfPositive(text, "final_logit_softcapping");
        float? attnLogitSoftcap = GetFloatNullableIfPositive(text, "attn_logit_softcapping");
        int qpas = GetInt32OrDefault(text, "query_pre_attn_scalar", 0);
        float? queryPreAttnScalar = qpas > 0 ? qpas : null;

        ActivationFunction activation = ResolveGemmaActivation(text);

        // Per-attention-type RoPE — FLAT bases (unlike Gemma-4's nested
        // rope_parameters): rope_theta (global/full), rope_local_base_freq
        // (local/sliding). No partial-rotary factor and no distinct
        // global head-dim / KV-head count on real Gemma-3n.
        float globalTheta = GetFloatOrDefault(text, "rope_theta", 1_000_000.0f);
        float localTheta = GetFloatOrDefault(text, "rope_local_base_freq", 10_000.0f);
        var slidingRope = new RoPEConfig(Theta: localTheta, DimensionCount: headDim, Type: RoPEType.NeoX);
        var globalRope = new RoPEConfig(Theta: globalTheta, DimensionCount: headDim, Type: RoPEType.NeoX);

        // Per-Layer Embeddings (PLE) — reuses the same config shape as the dense
        // Gemma-4 text tower (PLE originated in Gemma 3n).
        PerLayerEmbeddingConfig? perLayerEmbedding = null;
        if (text.TryGetProperty("hidden_size_per_layer_input", out var hsPle)
            && hsPle.ValueKind == JsonValueKind.Number && hsPle.GetInt32() > 0)
        {
            perLayerEmbedding = new PerLayerEmbeddingConfig
            {
                PerLayerDim = hsPle.GetInt32(),
                VocabSize = GetInt32OrDefault(text, "vocab_size_per_layer_input", vocabSize),
            };
        }

        // Trailing KV-shared layers (llama.cpp / Gemma-4 reuse rule — same
        // ModelConfig.SharedKvDonorLayer donor formula).
        int numSharedKvLayers = GetInt32OrDefault(text, "num_kv_shared_layers", 0);

        // AltUp / Laurel / activation sparsity — the genuinely new Gemma-3n pieces.
        int altupNumInputs = GetInt32OrDefault(text, "altup_num_inputs", 4);
        int altupActiveIdx = GetInt32OrDefault(text, "altup_active_idx", 0);
        float? altupCoefClip = GetFloatNullableIfPositive(text, "altup_coef_clip");
        bool altupCorrectScale = GetBoolOrDefault(text, "altup_correct_scale", true);
        int laurelRank = GetInt32OrDefault(text, "laurel_rank", 64);
        IReadOnlyList<float> activationSparsity = GetActivationSparsityPattern(text, numLayers);

        var gemma3nConfig = new Gemma3nConfig
        {
            NumInputs = altupNumInputs,
            ActiveIdx = altupActiveIdx,
            CoefClip = altupCoefClip,
            CorrectOutputScale = altupCorrectScale,
            LaurelRank = laurelRank,
            ActivationSparsityPattern = activationSparsity,
        };

        return new ModelConfig
        {
            Architecture = Architecture.Gemma3n,
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
            // No partial-rotary factor / distinct global head-dim / KV-head count
            // on real Gemma-3n — both stay null and every layer collapses to the
            // uniform HeadDim / NumKvHeads.
            PartialRotaryFactor = null,
            NumGlobalKvHeads = null,
            GlobalHeadDim = null,
            NumSharedKvLayers = numSharedKvLayers,
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
            PerLayerEmbedding = perLayerEmbedding,
            Gemma3n = gemma3nConfig,
            ChatTemplate = null,
        };
    }

    /// <summary>
    /// Reads <c>intermediate_size</c>, which HF ships as either a single int
    /// (dense-uniform configs) or a per-layer array (real Gemma-3n — MatFormer-
    /// capable, but every released SKU's array is uniform). Rejects a
    /// heterogeneous array rather than silently picking one entry.
    /// </summary>
    private static int GetIntermediateSize(JsonElement text, int numLayers)
    {
        if (!text.TryGetProperty("intermediate_size", out var prop))
            throw new InvalidDataException("Gemma-3n config.json missing required key 'intermediate_size'.");

        if (prop.ValueKind == JsonValueKind.Number)
            return prop.GetInt32();

        if (prop.ValueKind == JsonValueKind.Array)
        {
            int count = prop.GetArrayLength();
            if (count != numLayers)
                throw new InvalidDataException(
                    $"Gemma-3n config.json 'intermediate_size' array has {count} entries, expected {numLayers} (num_hidden_layers).");

            int first = -1;
            int i = 0;
            foreach (var el in prop.EnumerateArray())
            {
                int v = el.GetInt32();
                if (i == 0) first = v;
                else if (v != first)
                    throw new InvalidDataException(
                        "Gemma-3n config.json 'intermediate_size' is a heterogeneous per-layer array "
                        + "(MatFormer elastic width) — not supported; every released SKU ships a uniform array.");
                i++;
            }
            return first;
        }

        throw new InvalidDataException("Gemma-3n config.json 'intermediate_size' must be a number or an array.");
    }

    /// <summary>
    /// Reads <c>activation_sparsity_pattern</c> (per-layer float array; the real
    /// E4B/E2B ship 0.95 on the first 10 layers, 0.0 thereafter). Defaults to
    /// all-zero (sparsity disabled — plain GeGLU) when absent, so a synthetic
    /// fixture without the key still forwards correctly.
    /// </summary>
    private static IReadOnlyList<float> GetActivationSparsityPattern(JsonElement text, int numLayers)
    {
        var pattern = new float[numLayers];
        if (text.TryGetProperty("activation_sparsity_pattern", out var prop))
        {
            if (prop.ValueKind == JsonValueKind.Array && prop.GetArrayLength() == numLayers)
            {
                int i = 0;
                foreach (var el in prop.EnumerateArray())
                    pattern[i++] = el.ValueKind == JsonValueKind.Number && el.TryGetSingle(out float v) ? v : 0f;
            }
            else if (prop.ValueKind == JsonValueKind.Number && prop.TryGetSingle(out float flat))
            {
                for (int i = 0; i < numLayers; i++) pattern[i] = flat;
            }
        }
        return pattern;
    }

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

        int swPattern = GetInt32OrDefault(text, "sliding_window_pattern", 5);
        if (swPattern <= 0) swPattern = 1;
        for (int i = 0; i < numLayers; i++)
        {
            bool isFull = ((i + 1) % swPattern) == 0; // HF Gemma formula
            layerTypes[i] = isFull ? null : slidingWindow;
        }
        return layerTypes;
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

    private static int GetInt32(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            throw new InvalidDataException($"Gemma-3n config.json missing required integer key '{key}'.");
        if (!prop.TryGetInt32(out int value))
            throw new InvalidDataException($"Gemma-3n config.json key '{key}' is not a 32-bit integer.");
        return value;
    }

    private static int GetInt32OrDefault(JsonElement root, string key, int fallback)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            return fallback;
        return prop.TryGetInt32(out int value) ? value : fallback;
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
