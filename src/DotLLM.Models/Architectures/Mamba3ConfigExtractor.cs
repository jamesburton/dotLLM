using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Parses a HuggingFace <c>config.json</c> for a Mamba-3 checkpoint
/// (e.g. <c>ib-ssm/mamba3-370M-10BT</c>) into a populated
/// <see cref="ModelConfig"/> with an attached <see cref="Mamba3Config"/>.
/// </summary>
/// <remarks>
/// <para>
/// Stage D1 only — this does not load weights. Compare
/// <see cref="DotLLM.Models.Gguf.GgufModelConfigExtractor"/> for the
/// Nemotron-H (GGUF-first) precedent. Mamba-3 has no upstream GGUF support,
/// so <see cref="Architecture.Mamba3"/> is parsed from HF JSON directly.
/// </para>
/// <para>
/// Field semantics match the reference <c>VikramKarLex/mamba3-minimal</c>
/// Mamba3Config dataclass where the HF JSON keys overlap; new HF-only
/// keys are carried verbatim (see <see cref="Mamba3Config"/>).
/// </para>
/// </remarks>
public static class Mamba3ConfigExtractor
{
    /// <summary>
    /// Parses a HuggingFace <c>config.json</c> payload (as a raw string)
    /// into a <see cref="ModelConfig"/>.
    /// </summary>
    /// <param name="json">Raw JSON text of a HF Mamba-3 <c>config.json</c>.</param>
    /// <returns>A populated <see cref="ModelConfig"/> with <see cref="Architecture.Mamba3"/>.</returns>
    /// <exception cref="InvalidDataException">
    /// <paramref name="json"/> is not a valid Mamba-3 HF config, or required
    /// fields are missing / have illegal values.
    /// </exception>
    public static ModelConfig Extract(string json)
    {
        ArgumentNullException.ThrowIfNull(json);

        using var doc = JsonDocument.Parse(json);
        return Extract(doc.RootElement);
    }

    /// <summary>
    /// Parses a HuggingFace <c>config.json</c> already deserialised into a
    /// <see cref="JsonElement"/> into a <see cref="ModelConfig"/>.
    /// </summary>
    /// <param name="root">Root JSON object of the HF config.</param>
    /// <returns>A populated <see cref="ModelConfig"/> with <see cref="Architecture.Mamba3"/>.</returns>
    /// <exception cref="InvalidDataException">
    /// Required fields are missing or have illegal values, or the <c>model_type</c>
    /// is not <c>mamba3</c>.
    /// </exception>
    public static ModelConfig Extract(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("Mamba-3 config.json root must be a JSON object.");

        string modelType = GetString(root, "model_type");
        if (!string.Equals(modelType, "mamba3", StringComparison.Ordinal))
            throw new InvalidDataException(
                $"Mamba-3 config extractor requires model_type='mamba3', got '{modelType}'.");

        int hiddenSize = GetInt32(root, "hidden_size");
        int numLayers = GetInt32(root, "num_hidden_layers");
        int numHeads = GetInt32(root, "num_heads");
        int headDim = GetInt32(root, "head_dim");
        int expand = GetInt32OrDefault(root, "expand", 2);
        int numGroups = GetInt32OrDefault(root, "n_groups", 1);
        int stateSize = GetInt32(root, "state_size");
        int vocabSize = GetInt32(root, "vocab_size");
        int chunkSize = GetInt32OrDefault(root, "chunk_size", 64);
        int mimoRank = GetInt32OrDefault(root, "mimo_rank", 4);

        bool isMimo = GetBoolOrDefault(root, "is_mimo", false);
        bool isOutProjNorm = GetBoolOrDefault(root, "is_outproj_norm", false);
        bool useL2Warp = GetBoolOrDefault(root, "use_l2warp", false);
        bool tieEmbeddings = GetBoolOrDefault(root, "tie_word_embeddings", false);
        bool rescalePrenormResidual = GetBoolOrDefault(root, "rescale_prenorm_residual", true);
        bool residualInFp32 = GetBoolOrDefault(root, "residual_in_fp32", true);

        float aFloor = GetFloatOrDefault(root, "A_floor", 1e-4f);
        float dtInitFloor = GetFloatOrDefault(root, "dt_init_floor", 1e-4f);
        float dtMin = GetFloatOrDefault(root, "dt_min", 1e-3f);
        float dtMax = GetFloatOrDefault(root, "dt_max", 0.1f);
        float normEps = GetFloatOrDefault(root, "norm_eps", 1e-5f);
        float ropeFraction = GetFloatOrDefault(root, "rope_fraction", 1.0f);

        if (stateSize <= 0 || (stateSize & 1) != 0)
            throw new InvalidDataException(
                $"Mamba-3 state_size must be positive and even (got {stateSize}).");
        if (numHeads <= 0)
            throw new InvalidDataException($"Mamba-3 num_heads must be positive (got {numHeads}).");
        if (headDim <= 0)
            throw new InvalidDataException($"Mamba-3 head_dim must be positive (got {headDim}).");
        if (numHeads * headDim != expand * hiddenSize)
            throw new InvalidDataException(
                $"Mamba-3 num_heads*head_dim ({numHeads * headDim}) must equal " +
                $"expand*hidden_size ({expand * hiddenSize}). Checkpoint is likely misconfigured.");
        if (numGroups <= 0)
            throw new InvalidDataException($"Mamba-3 n_groups must be positive (got {numGroups}).");
        if (numHeads % numGroups != 0)
            throw new InvalidDataException(
                $"Mamba-3 num_heads ({numHeads}) must be divisible by n_groups ({numGroups}).");
        if (ropeFraction is < 0.0f or > 1.0f)
            throw new InvalidDataException(
                $"Mamba-3 rope_fraction must be in [0,1] (got {ropeFraction}).");
        if (mimoRank <= 0)
            throw new InvalidDataException($"Mamba-3 mimo_rank must be positive (got {mimoRank}).");

        var mamba3Config = new Mamba3Config
        {
            StateSize = stateSize,
            NumHeads = numHeads,
            HeadDim = headDim,
            Expand = expand,
            NumGroups = numGroups,
            ChunkSize = chunkSize,
            IsMimo = isMimo,
            MimoRank = mimoRank,
            AFloor = aFloor,
            DtInitFloor = dtInitFloor,
            DtMin = dtMin,
            DtMax = dtMax,
            UseL2Warp = useL2Warp,
            RopeFraction = ropeFraction,
            IsOutProjNorm = isOutProjNorm,
            RescalePrenormResidual = rescalePrenormResidual,
            ResidualInFp32 = residualInFp32,
        };

        return new ModelConfig
        {
            Architecture = Architecture.Mamba3,
            VocabSize = vocabSize,
            HiddenSize = hiddenSize,
            // Mamba-3 has no attention and no MLP per-layer on the known HF
            // checkpoint; IntermediateSize is meaningless here. Carry 0.
            IntermediateSize = 0,
            NumLayers = numLayers,
            // Same for attention-related fields: retained for ModelConfig
            // shape compatibility, but semantically "no attention path".
            NumAttentionHeads = numHeads,
            NumKvHeads = numHeads,
            HeadDim = headDim,
            MaxSequenceLength = GetInt32OrDefault(root, "max_position_embeddings", 2048),
            AttentionType = AttentionType.GQA, // placeholder — no attention layers exist
            PositionEncodingType = PositionEncodingType.None, // RoPE is data-dependent, not positional
            RoPEConfig = null,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = normEps,
            TiedEmbeddings = tieEmbeddings,
            SlidingWindowSize = null,
            MlaConfig = null,
            HybridLayout = null,
            SsmConfig = null,
            Mamba3Config = mamba3Config,
            ChatTemplate = null,
        };
    }

    private static string GetString(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.String)
            throw new InvalidDataException($"Mamba-3 config.json missing required string key '{key}'.");
        return prop.GetString() ?? throw new InvalidDataException(
            $"Mamba-3 config.json key '{key}' is null.");
    }

    private static int GetInt32(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            throw new InvalidDataException($"Mamba-3 config.json missing required integer key '{key}'.");
        if (!prop.TryGetInt32(out int value))
            throw new InvalidDataException(
                $"Mamba-3 config.json key '{key}' is not a 32-bit integer.");
        return value;
    }

    private static int GetInt32OrDefault(JsonElement root, string key, int fallback)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            return fallback;
        return prop.TryGetInt32(out int value) ? value : fallback;
    }

    private static bool GetBoolOrDefault(JsonElement root, string key, bool fallback)
    {
        if (!root.TryGetProperty(key, out var prop))
            return fallback;
        return prop.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => fallback,
        };
    }

    private static float GetFloatOrDefault(JsonElement root, string key, float fallback)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            return fallback;
        return prop.TryGetSingle(out float value) ? value : fallback;
    }
}
