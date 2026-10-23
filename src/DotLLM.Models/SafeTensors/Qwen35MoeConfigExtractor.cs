using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Parses a Qwen3.6-35B-A3B-family checkpoint's <c>config.json</c>
/// (<c>model_type=qwen3_5_moe</c>, <c>architectures=
/// ["Qwen3_5MoeForConditionalGeneration"]</c>) into a populated
/// <see cref="ModelConfig"/> carrying <see cref="Architecture.Qwen3MoeHybrid"/>
/// plus a non-null <see cref="ModelConfig.GdnConfig"/>, <see cref="ModelConfig.Moe"/>,
/// and <see cref="ModelConfig.HybridLayout"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a dedicated extractor.</b> Real Qwen3.6-35B-A3B checkpoints
/// (verified against <c>SyzygyResearch/Mach-1-Additive-35B</c>'s
/// <c>config.json</c>, which is a verbatim upstream Qwen config) house the
/// text tower under a <c>text_config</c> sub-object (<c>model_type=
/// qwen3_5_moe_text</c>) alongside a <c>vision_config</c> we deliberately
/// skip (text-only — the multimodal vision tower is a separate, non-goal
/// issue). This mirrors <see cref="Gemma3nConfigExtractor"/>'s nested-config
/// hoist pattern.
/// </para>
/// <para>
/// <b>Gated DeltaNet + full-attention hybrid.</b> Unlike a plain Qwen-MoE
/// transformer, this architecture alternates Gated DeltaNet (GDN) recurrence
/// layers with full GQA attention layers every <c>full_attention_interval</c>
/// steps (<c>layer_types</c> confirms the pattern but the numeric interval
/// key is authoritative — mirrors <c>GgufModelConfigExtractor.
/// BuildQwen3MoeHybridLayout</c>, reimplemented here since that helper is
/// private to the GGUF extractor). The GDN shapes come from
/// <c>linear_key_head_dim</c>/<c>linear_value_head_dim</c> (equal on every
/// released SKU — <see cref="GatedDeltaNetConfig.DState"/> requires K and V
/// head dims to match), <c>linear_num_key_heads</c>, <c>linear_num_value_heads</c>,
/// and <c>linear_conv_kernel_dim</c>.
/// </para>
/// <para>
/// <b>Partial rotary.</b> <c>head_dim</c> (256) is the FULL per-head Q/K
/// width; only <c>head_dim * partial_rotary_factor</c> (64, at
/// <c>partial_rotary_factor=0.25</c>) is actually rotated. Mirrors
/// <c>Qwen3MoeHybridTransformerModel.LoadFromGguf</c>'s convention: the
/// rotated width is baked directly into <see cref="RoPEConfig.DimensionCount"/>
/// (NOT into <see cref="ModelConfig.PartialRotaryFactor"/>, which the GDN
/// hybrid forward path does not consult).
/// </para>
/// <para>
/// <b>mRoPE (non-goal).</b> <c>rope_parameters.mrope_interleaved</c> /
/// <c>mrope_section</c> describe a 3-axis (text/height/width) RoPE for
/// vision token positions. This extractor collapses to the single-axis
/// <c>rope_theta</c> schedule — correct for text-only generation, wrong for
/// vision positions — matching the existing GGUF path's documented
/// limitation (<c>Qwen3MoeHybridTransformerModel.cs:1056-1057</c>) and
/// issue #266's explicit non-goals list.
/// </para>
/// </remarks>
public static class Qwen35MoeConfigExtractor
{
    /// <summary>
    /// Parses a Qwen3.6-35B-A3B-family <c>config.json</c> root into a
    /// <see cref="ModelConfig"/>.
    /// </summary>
    /// <param name="root">Parsed <c>config.json</c> root (top-level
    /// <c>qwen3_5_moe</c> object, or a flat text-only <c>qwen3_5_moe_text</c>
    /// config).</param>
    /// <exception cref="InvalidDataException">The config is malformed, declares
    /// an unexpected <c>model_type</c>, or is missing the GDN-defining keys.</exception>
    public static ModelConfig Extract(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("Qwen3.6-MoE config.json root must be a JSON object.");

        string? modelType = GetStringOrNull(root, "model_type");
        if (modelType is not null
            && !string.Equals(modelType, "qwen3_5_moe", StringComparison.Ordinal)
            && !string.Equals(modelType, "qwen3_5_moe_text", StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Qwen35MoeConfigExtractor requires model_type='qwen3_5_moe' / 'qwen3_5_moe_text', got '{modelType}'.");
        }

        // Hoist the text tower. Real checkpoints house it under `text_config`
        // (vision_config skipped — text-only); a flat qwen3_5_moe_text config
        // (or a synthetic fixture) treats the root itself as the text tower.
        JsonElement text = root;
        if (root.TryGetProperty("text_config", out var tc) && tc.ValueKind == JsonValueKind.Object)
            text = tc;

        int hiddenSize = GetInt32(text, "hidden_size");
        int numLayers = GetInt32(text, "num_hidden_layers");
        int numAttentionHeads = GetInt32(text, "num_attention_heads");
        int headDim = GetInt32OrDefault(text, "head_dim", hiddenSize / numAttentionHeads);
        int vocabSize = GetInt32(text, "vocab_size");
        int maxSeqLen = GetInt32OrDefault(text, "max_position_embeddings", 262144);
        float normEps = GetFloatOrDefault(text, "rms_norm_eps", 1e-6f);
        bool tieEmbeddings = GetBoolOrDefault(text, "tie_word_embeddings", false);

        // ── GDN (Gated DeltaNet) config ───────────────────────────────────
        int fullAttnInterval = GetInt32OrDefault(text, "full_attention_interval", 4);
        int nKHead = GetInt32(text, "linear_num_key_heads");
        int nVHead = GetInt32(text, "linear_num_value_heads");
        int keyHeadDim = GetInt32(text, "linear_key_head_dim");
        int valueHeadDim = GetInt32(text, "linear_value_head_dim");
        if (keyHeadDim != valueHeadDim)
            throw new InvalidDataException(
                $"Qwen3.6-MoE GDN requires linear_key_head_dim == linear_value_head_dim " +
                $"(GatedDeltaNetConfig.DState is shared); got {keyHeadDim} vs {valueHeadDim}.");
        int dState = keyHeadDim;
        int dConv = GetInt32OrDefault(text, "linear_conv_kernel_dim", 4);
        // DInner: width of the combined input projection driving Q, K, V (GGUF
        // ssm.inner_size) — not an independent HF config key, derived from the
        // per-head dims: 2*nKHead*dState (Q+K) + nVHead*dState (V).
        int dInner = 2 * nKHead * dState + nVHead * dState;
        var gdnConfig = new GatedDeltaNetConfig(fullAttnInterval, nVHead, nKHead, dState, dInner, dConv);

        // numKvHeads (top-level scalar) mirrors the GGUF path: the *attention-layer*
        // KV-head count (num_key_value_heads), used for the shared scalar ModelConfig
        // fields even though most layers are GDN (no discrete KV heads at all).
        int numKvHeads = GetInt32OrDefault(text, "num_key_value_heads", numAttentionHeads);

        HybridLayerLayout hybridLayout = BuildHybridLayout(text, numLayers, fullAttnInterval, numKvHeads);

        // ── MoE config ─────────────────────────────────────────────────────
        int numExperts = GetInt32(text, "num_experts");
        int numExpertsPerTok = GetInt32(text, "num_experts_per_tok");
        int moeIntermediate = GetInt32(text, "moe_intermediate_size");
        // qwen35moe convention (mirrors GgufModelConfigExtractor.TryExtractQwenMoeConfig):
        // shared-expert count is implicit (1), intermediate width is a direct HF key,
        // and the branch is always paired with a sigmoid gate.
        int? sharedIntermediate = GetInt32OrDefaultNullable(text, "shared_expert_intermediate_size");
        var moeConfig = new MoeConfig
        {
            NumExperts = numExperts,
            NumExpertsPerTok = numExpertsPerTok,
            MoeIntermediateSize = moeIntermediate,
            NormTopKProb = true,
            SharedExpertIntermediateSize = sharedIntermediate,
            NumSharedExperts = sharedIntermediate is > 0 ? 1 : 0,
            HasSharedExpertGate = sharedIntermediate is > 0,
            DecoderSparseStep = 1,
        };

        // ── RoPE (partial rotary; mRoPE collapsed to single-axis — see remarks) ──
        JsonElement ropeParams = text.TryGetProperty("rope_parameters", out var rp) && rp.ValueKind == JsonValueKind.Object
            ? rp
            : text;
        float ropeTheta = GetFloatOrDefault(ropeParams, "rope_theta", GetFloatOrDefault(text, "rope_theta", 10_000_000.0f));
        float partialRotaryFactor = GetFloatOrDefault(ropeParams, "partial_rotary_factor",
            GetFloatOrDefault(text, "partial_rotary_factor", 1.0f));
        int ropeDim = (int)Math.Round(headDim * partialRotaryFactor);
        if ((ropeDim & 1) != 0)
            throw new InvalidDataException(
                $"Qwen3.6-MoE rope_dim={ropeDim} (head_dim={headDim} * partial_rotary_factor={partialRotaryFactor}) " +
                "must be even for pair-wise rotation.");
        var ropeConfig = new RoPEConfig(Theta: ropeTheta, DimensionCount: ropeDim, Type: RoPEType.NeoX);

        return new ModelConfig
        {
            Architecture = Architecture.Qwen3MoeHybrid,
            VocabSize = vocabSize,
            HiddenSize = hiddenSize,
            // No dense IntermediateSize on this architecture (every layer is MoE) —
            // mirrors GgufModelConfigExtractor's MaxNonZero(hybridLayout.FeedForwardLength, 0)
            // fallback, which is 0 here since FeedForwardLength is all-zero for MoE-tracked layers.
            IntermediateSize = 0,
            NumLayers = numLayers,
            NumAttentionHeads = numAttentionHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            MaxSequenceLength = maxSeqLen,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = ropeConfig,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = normEps,
            TiedEmbeddings = tieEmbeddings,
            HybridLayout = hybridLayout,
            GdnConfig = gdnConfig,
            Moe = moeConfig,
            ChatTemplate = null,
        };
    }

    /// <summary>
    /// Builds a <see cref="HybridLayerLayout"/> from <c>layer_types</c> when
    /// present (authoritative — real checkpoints ship it), falling back to the
    /// <c>full_attention_interval</c> formula (layer <c>i</c>, 1-indexed, is
    /// full attention when <c>i % fullAttnInterval == 0</c>) otherwise. Mirrors
    /// <c>GgufModelConfigExtractor.BuildQwen3MoeHybridLayout</c> (private to
    /// that type, so reimplemented here for the HF config path).
    /// </summary>
    private static HybridLayerLayout BuildHybridLayout(
        JsonElement text, int numLayers, int fullAttnInterval, int numKvHeads)
    {
        var kinds = new HybridLayerKind[numLayers];
        var headCountKv = new int[numLayers];
        var feedForwardLength = new int[numLayers]; // all zero — MoE tracked via ModelConfig.Moe

        if (text.TryGetProperty("layer_types", out var lt) && lt.ValueKind == JsonValueKind.Array
            && lt.GetArrayLength() == numLayers)
        {
            int i = 0;
            foreach (JsonElement el in lt.EnumerateArray())
            {
                string? s = el.ValueKind == JsonValueKind.String ? el.GetString() : null;
                bool isFullAttn = string.Equals(s, "full_attention", StringComparison.Ordinal);
                kinds[i] = isFullAttn ? HybridLayerKind.Attention : HybridLayerKind.GatedDeltaNet;
                headCountKv[i] = isFullAttn ? numKvHeads : 0;
                i++;
            }
        }
        else
        {
            for (int i = 0; i < numLayers; i++)
            {
                bool isFullAttn = (i + 1) % fullAttnInterval == 0;
                kinds[i] = isFullAttn ? HybridLayerKind.Attention : HybridLayerKind.GatedDeltaNet;
                headCountKv[i] = isFullAttn ? numKvHeads : 0;
            }
        }

        return new HybridLayerLayout
        {
            LayerKind = kinds,
            HeadCountKv = headCountKv,
            FeedForwardLength = feedForwardLength,
        };
    }

    // -------------------------------------------------------------------------
    // Low-level JSON helpers (kept local — mirrors Gemma3nConfigExtractor's).
    // -------------------------------------------------------------------------

    private static int GetInt32(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            throw new InvalidDataException($"Qwen3.6-MoE config.json missing required integer key '{key}'.");
        if (!prop.TryGetInt32(out int value))
            throw new InvalidDataException($"Qwen3.6-MoE config.json key '{key}' is not a 32-bit integer.");
        return value;
    }

    private static int GetInt32OrDefault(JsonElement root, string key, int fallback)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            return fallback;
        return prop.TryGetInt32(out int value) ? value : fallback;
    }

    private static int? GetInt32OrDefaultNullable(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            return null;
        return prop.TryGetInt32(out int value) && value > 0 ? value : null;
    }

    private static float GetFloatOrDefault(JsonElement root, string key, float fallback)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            return fallback;
        return prop.TryGetSingle(out float value) ? value : fallback;
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
