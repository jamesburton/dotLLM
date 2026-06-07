using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Parses a HuggingFace <c>config.json</c> for a dense-transformer checkpoint
/// (Llama, Mistral, Phi, Qwen) into a populated <see cref="ModelConfig"/>.
/// </summary>
/// <remarks>
/// <para>
/// Mirrors <see cref="DotLLM.Models.Gguf.GgufModelConfigExtractor"/> but reads
/// JSON rather than GGUF metadata KV pairs. The source of truth for the field
/// names is the <c>transformers</c> per-architecture <c>configuration_*.py</c>
/// file — e.g. <c>LlamaConfig</c> (<c>hidden_size</c>, <c>num_hidden_layers</c>,
/// <c>num_attention_heads</c>, <c>num_key_value_heads</c>, <c>intermediate_size</c>,
/// <c>vocab_size</c>, <c>max_position_embeddings</c>, <c>rope_theta</c>,
/// <c>rms_norm_eps</c>, <c>tie_word_embeddings</c>, <c>architectures[0]</c>).
/// </para>
/// <para>
/// Defensive about common HF quirks: <c>num_key_value_heads</c> may be absent
/// (implies MHA: equal to <c>num_attention_heads</c>), <c>head_dim</c> may be
/// stored explicitly (Qwen3/some Llamas) or implied by <c>hidden_size /
/// num_attention_heads</c>, and the top-level <c>architectures</c> array
/// carries the class name (e.g. <c>LlamaForCausalLM</c>) which disambiguates
/// Llama vs Mistral vs Phi3 vs Qwen2 when <c>model_type</c> alone is ambiguous.
/// </para>
/// </remarks>
public static class HfConfigExtractor
{
    /// <summary>
    /// Parses a HF <c>config.json</c> payload (raw string) into a
    /// <see cref="ModelConfig"/>.
    /// </summary>
    public static ModelConfig Extract(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        using var doc = JsonDocument.Parse(json);
        return Extract(doc.RootElement);
    }

    /// <summary>
    /// Parses a HF <c>config.json</c> already deserialised into a
    /// <see cref="JsonElement"/> into a <see cref="ModelConfig"/>.
    /// </summary>
    /// <exception cref="InvalidDataException">
    /// Required fields missing / illegal values / unsupported architecture.
    /// </exception>
    public static ModelConfig Extract(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("HF config.json root must be a JSON object.");

        Architecture architecture = ResolveArchitecture(root);

        int hiddenSize = GetInt32(root, "hidden_size");
        int numLayers = GetInt32(root, "num_hidden_layers");
        int numAttentionHeads = GetInt32(root, "num_attention_heads");
        int numKvHeads = GetInt32OrDefault(root, "num_key_value_heads", numAttentionHeads);
        int intermediateSize = GetInt32(root, "intermediate_size");
        int vocabSize = GetInt32(root, "vocab_size");
        int maxSeqLen = GetInt32OrDefault(root, "max_position_embeddings", 2048);
        int headDim = GetInt32OrDefault(root, "head_dim", hiddenSize / numAttentionHeads);

        float normEps = GetFloatOrDefault(root, "rms_norm_eps",
            GetFloatOrDefault(root, "layer_norm_eps", 1e-5f));
        float ropeTheta = GetFloatOrDefault(root, "rope_theta", 10000.0f);
        bool tieEmbeddings = GetBoolOrDefault(root, "tie_word_embeddings", DefaultTieForArch(architecture));

        int? slidingWindow = GetInt32NullableIfPositive(root, "sliding_window");

        // RoPE element-pairing convention — identical to GgufModelConfigExtractor.
        // Llama/Mistral use interleaved (Norm); Qwen/Phi use non-interleaved
        // (NeoX). SmolLM3 inherits rotate_half from Llama in modeling_smollm3.py
        // (HF rotate_half = halves-pair = our NeoX), so it diverges from the
        // Llama default for this extractor.
        RoPEType ropeType = architecture switch
        {
            Architecture.Qwen or Architecture.Phi or Architecture.SmolLM3 => RoPEType.NeoX,
            _ => RoPEType.Norm,
        };

        // Dense-path YaRN scaling. MLA handles its own rope_scaling extraction
        // (carried inside MlaConfig); for non-MLA architectures we surface
        // YaRN into RoPEConfig so TransformerModel can rebuild the cos/sin
        // tables via PrecomputeFrequencyTableYarn. The base SmolLM3-3B ships
        // rope_scaling=null — for the long-context 128k SKUs the same checkpoint
        // family ships {"rope_type":"yarn","factor":...,"original_max_position_embeddings":...}.
        (RoPEScalingType ropeScalingType, float ropeScalingFactor,
         int ropeScalingOrigMax, float ropeScalingBetaFast, float ropeScalingBetaSlow,
         float ropeScalingAttnFactor) = ExtractDenseRopeScaling(root);

        // SmolLM3 NoPE mask: per-layer 0/1 array where the layer skips RoPE
        // when the value is 0 (HF naming is counterintuitive — see
        // modeling_smollm3.py: self.use_rope = config.no_rope_layers[layer_idx]
        // then `if self.use_rope: apply_rotary_pos_emb(...)`). Convert to a
        // list of layer INDICES where RoPE is skipped so the forward pass can
        // gate via ModelConfig.IsNoRopeLayer.
        IReadOnlyList<int>? noRopeLayers = ExtractNoRopeLayers(root);

        var ropeConfig = new RoPEConfig(
            Theta: ropeTheta,
            DimensionCount: headDim,
            Type: ropeType,
            ScalingType: ropeScalingType,
            ScalingFactor: ropeScalingFactor,
            OrigMaxSeqLen: ropeScalingOrigMax,
            AttnFactor: ropeScalingAttnFactor,
            BetaFast: ropeScalingBetaFast,
            BetaSlow: ropeScalingBetaSlow);

        return new ModelConfig
        {
            Architecture = architecture,
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
            RoPEConfig = ropeConfig,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = normEps,
            TiedEmbeddings = tieEmbeddings,
            SlidingWindowSize = slidingWindow,
            ChatTemplate = null,
            NoRopeLayers = noRopeLayers,
        };
    }

    /// <summary>
    /// Pulls the optional <c>rope_scaling</c> block for non-MLA architectures
    /// (SmolLM3, Llama 3.1+, ...). Returns defaults
    /// (<see cref="RoPEScalingType.None"/>, factor=1) when the block is
    /// absent or declares <c>rope_type=default</c>. Recognises HF's modern
    /// keys (<c>rope_type</c>, <c>type</c>) alongside the legacy
    /// <c>linear</c> / <c>dynamic</c> family.
    /// </summary>
    private static (RoPEScalingType Type, float Factor, int OrigMax, float BetaFast, float BetaSlow, float AttnFactor)
        ExtractDenseRopeScaling(JsonElement root)
    {
        if (!root.TryGetProperty("rope_scaling", out var rs) || rs.ValueKind != JsonValueKind.Object)
            return (RoPEScalingType.None, 1.0f, 0, 32.0f, 1.0f, 1.0f);

        // HF rope_scaling sometimes uses `rope_type`, sometimes `type`.
        string? typeName = null;
        if (rs.TryGetProperty("rope_type", out var rt) && rt.ValueKind == JsonValueKind.String)
            typeName = rt.GetString();
        else if (rs.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String)
            typeName = t.GetString();

        RoPEScalingType scalingType = typeName?.ToLowerInvariant() switch
        {
            "linear" => RoPEScalingType.Linear,
            "yarn" => RoPEScalingType.YaRN,
            "ntk" => RoPEScalingType.NTK,
            "dynamic" or "dynamic_ntk" => RoPEScalingType.DynamicNTK,
            "su" or "longrope" => RoPEScalingType.Su,
            _ => RoPEScalingType.None,
        };

        float factor = 1.0f;
        if (rs.TryGetProperty("factor", out var f) && f.ValueKind == JsonValueKind.Number
            && f.TryGetSingle(out float fv))
            factor = fv;

        int origMax = 0;
        if (rs.TryGetProperty("original_max_position_embeddings", out var om)
            && om.ValueKind == JsonValueKind.Number
            && om.TryGetInt32(out int omv))
            origMax = omv;

        float betaFast = 32.0f;
        if (rs.TryGetProperty("beta_fast", out var bf) && bf.ValueKind == JsonValueKind.Number
            && bf.TryGetSingle(out float bfv))
            betaFast = bfv;

        float betaSlow = 1.0f;
        if (rs.TryGetProperty("beta_slow", out var bs) && bs.ValueKind == JsonValueKind.Number
            && bs.TryGetSingle(out float bsv))
            betaSlow = bsv;

        // attention_factor (HF) — softmax scale multiplier folded into cos/sin.
        // Defaults to 1.0 (SmolLM3 doesn't ship it).
        float attnFactor = 1.0f;
        if (rs.TryGetProperty("attention_factor", out var af) && af.ValueKind == JsonValueKind.Number
            && af.TryGetSingle(out float afv))
            attnFactor = afv;

        return (scalingType, factor, origMax, betaFast, betaSlow, attnFactor);
    }

    /// <summary>
    /// Parses SmolLM3's <c>no_rope_layers</c> per-layer mask. HF encodes
    /// <c>1 = apply RoPE</c>, <c>0 = skip RoPE</c> — we invert and return
    /// the indices that SKIP RoPE so downstream code reads naturally
    /// (<c>IsNoRopeLayer(i)</c> matches the field name). Returns null when
    /// the key is absent or every layer applies RoPE.
    /// </summary>
    private static IReadOnlyList<int>? ExtractNoRopeLayers(JsonElement root)
    {
        if (!root.TryGetProperty("no_rope_layers", out var prop) || prop.ValueKind != JsonValueKind.Array)
            return null;
        int len = prop.GetArrayLength();
        if (len == 0) return null;

        List<int>? skipped = null;
        int i = 0;
        foreach (var el in prop.EnumerateArray())
        {
            // 1 = apply RoPE; 0 = skip RoPE (NoPE).
            if (el.ValueKind == JsonValueKind.Number && el.TryGetInt32(out int v) && v == 0)
            {
                skipped ??= new List<int>();
                skipped.Add(i);
            }
            i++;
        }
        return skipped;
    }

    /// <summary>
    /// Peeks at <c>model_type</c> / <c>architectures[0]</c> so the caller
    /// (e.g. <c>ModelLoader.LoadFromSafetensors</c>) can pre-dispatch before
    /// running the full extractor.
    /// </summary>
    public static Architecture ResolveArchitecture(JsonElement root)
    {
        string? archName = null;
        if (root.TryGetProperty("architectures", out var archArr)
            && archArr.ValueKind == JsonValueKind.Array
            && archArr.GetArrayLength() > 0)
        {
            var first = archArr[0];
            if (first.ValueKind == JsonValueKind.String)
                archName = first.GetString();
        }

        string? modelType = GetStringOrDefault(root, "model_type", null);

        return (archName?.ToLowerInvariant(), modelType?.ToLowerInvariant()) switch
        {
            // SmolLM3 — `SmolLM3ForCausalLM` / `model_type=smollm3`. Llama-shaped
            // tensors but carries `no_rope_layers` mask + (optional) YaRN scaling.
            (var a, _) when a is not null && a.Contains("smollm3") => Architecture.SmolLM3,
            (_, "smollm3") => Architecture.SmolLM3,
            (var a, _) when a is not null && a.Contains("llama") => Architecture.Llama,
            (var a, _) when a is not null && a.Contains("mistral") => Architecture.Mistral,
            (var a, _) when a is not null && a.StartsWith("phi") => Architecture.Phi,
            (var a, _) when a is not null && a.Contains("qwen") => Architecture.Qwen,
            (_, "llama") => Architecture.Llama,
            (_, "mistral") => Architecture.Mistral,
            (_, "phi" or "phi3" or "phi2") => Architecture.Phi,
            (_, "qwen" or "qwen2" or "qwen3") => Architecture.Qwen,
            _ => throw new InvalidDataException(
                $"Unsupported HF architecture: architectures[0]='{archName}', model_type='{modelType}'.")
        };
    }

    /// <summary>
    /// Default tie-embeddings behaviour for architectures where HF typically
    /// omits the key. Gemma/Phi3 tie by default; Llama/Mistral/Qwen don't.
    /// Safest behaviour is "don't tie unless declared", which matches the
    /// spec for Llama/Mistral/Qwen. Phi's config almost always states it
    /// explicitly so this fallback rarely fires.
    /// </summary>
    private static bool DefaultTieForArch(Architecture arch) => arch switch
    {
        Architecture.Phi => true,
        _ => false,
    };

    private static int GetInt32(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            throw new InvalidDataException($"HF config.json missing required integer key '{key}'.");
        if (!prop.TryGetInt32(out int value))
            throw new InvalidDataException($"HF config.json key '{key}' is not a 32-bit integer.");
        return value;
    }

    private static int GetInt32OrDefault(JsonElement root, string key, int fallback)
    {
        if (!root.TryGetProperty(key, out var prop)) return fallback;
        // HF sometimes stores None as JSON null (e.g. num_key_value_heads) —
        // defensively coerce that to the fallback.
        if (prop.ValueKind != JsonValueKind.Number) return fallback;
        return prop.TryGetInt32(out int value) ? value : fallback;
    }

    private static int? GetInt32NullableIfPositive(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop)) return null;
        if (prop.ValueKind != JsonValueKind.Number) return null;
        if (!prop.TryGetInt32(out int v)) return null;
        return v > 0 ? v : null;
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

    private static string? GetStringOrDefault(JsonElement root, string key, string? fallback)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.String)
            return fallback;
        return prop.GetString() ?? fallback;
    }
}
