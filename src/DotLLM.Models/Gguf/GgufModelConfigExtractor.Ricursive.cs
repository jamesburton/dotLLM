using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;

namespace DotLLM.Models.Gguf;

public static partial class GgufModelConfigExtractor
{
    /// <summary>
    /// Builds the <see cref="ModelConfig"/> of a <c>ricursive</c> (Ripple) GGUF (issue #1643).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The contract is the union of the two community ports (AtomicChat <c>ricursive-port.patch</c>, prometheusAIR
    /// <c>llama.cpp-ricursive.patch</c>) and llama.cpp PR #31692, and was checked against the real AtomicChat Q4_K_M and prometheusAIR
    /// Q6_K headers (identical keys and values; the only differences are the value-residual tensor name, AtomicChat's extra
    /// <c>ricursive.vocab_size</c> / <c>tokenizer.ggml.eot_token_id</c> and prometheusAIR's <c>tokenizer.ggml.add_bos_token = false</c>).
    /// Keys: <c>block_count</c>, <c>context_length</c>, <c>embedding_length</c>, <c>feed_forward_length</c>, <c>attention.head_count</c>,
    /// <c>attention.head_count_kv</c>, <c>attention.key_length</c> = <c>attention.value_length</c>, <c>attention.layer_norm_rms_epsilon</c>,
    /// <c>rope.freq_base</c> (optional <c>rope.freq_base_swa</c>), <c>rope.dimension_count</c> (= head_dim: full rotation on the sliding
    /// layers), <c>attention.sliding_window</c> (required, &gt; 0, as in PR #31692), <c>attention.sliding_window_pattern</c> (bool per layer,
    /// true = sliding + RoPE, false = full attention + NoPE; absent = every 5th layer full), optional <c>attention.rope_pattern</c> (PR #31692:
    /// must equal the sliding pattern), <c>embedding_scale</c> (absent = folded) and <c>swiglu_clamp</c> (scalar or per-layer array; absent or
    /// 0 = no clamp).
    /// </para>
    /// <para>
    /// The value residual and the output gate have no keys: llama.cpp loads them as optional tensors (all layers or none), so
    /// <see cref="RicursiveConfig.UseValueResidual"/> and <see cref="RicursiveConfig.UseGatedAttention"/> stay <see langword="null"/> here and
    /// <see cref="RicursiveTensorCensus"/> resolves them from the tensor table.
    /// </para>
    /// </remarks>
    private static ModelConfig BuildRicursiveConfig(GgufMetadata metadata, string arch)
    {
        int hiddenSize = RequiredPositiveU32(metadata, $"{arch}.embedding_length");
        int numLayers = RequiredPositiveU32(metadata, $"{arch}.block_count");
        int numHeads = RequiredPositiveU32(metadata, $"{arch}.attention.head_count");
        int numKvHeads = RequiredPositiveU32(metadata, $"{arch}.attention.head_count_kv");
        int ffn = RequiredPositiveU32(metadata, $"{arch}.feed_forward_length");
        if (numHeads % numKvHeads != 0)
            throw new InvalidDataException($"{arch}: head_count {numHeads} is not a multiple of head_count_kv {numKvHeads}.");

        int headDim = (int)metadata.GetUInt32OrDefault($"{arch}.attention.key_length", (uint)(hiddenSize / numHeads));
        int valueDim = (int)metadata.GetUInt32OrDefault($"{arch}.attention.value_length", (uint)headDim);
        if (headDim <= 0 || valueDim != headDim)
            throw new InvalidDataException($"{arch}: attention.key_length ({headDim}) and attention.value_length ({valueDim}) must be equal and positive.");

        float normEps = metadata.GetFloat32($"{arch}.attention.layer_norm_rms_epsilon");

        uint window = metadata.GetUInt32OrDefault($"{arch}.attention.sliding_window", 0);
        if (window == 0)
            throw new InvalidDataException(
                $"{arch}: attention.sliding_window must be > 0 (RoPE is only applied on the sliding layers, so a model without them has no positional encoding).");

        bool[] sliding = ReadRicursiveSlidingPattern(metadata, arch, numLayers);
        if (Array.IndexOf(sliding, true) < 0)
            throw new InvalidDataException($"{arch}: the sliding_window_pattern has no sliding layer, the model would have no positional encoding.");

        // PR #31692: optional rope_pattern must coincide with the sliding pattern (RoPE exactly on the sliding layers).
        int[] ropePattern = GetIntArrayOrEmpty(metadata, $"{arch}.attention.rope_pattern");
        if (ropePattern.Length > 0)
        {
            if (ropePattern.Length != numLayers)
                throw new InvalidDataException($"{arch}.attention.rope_pattern has {ropePattern.Length} entries, expected {numLayers}.");
            for (int l = 0; l < numLayers; l++)
                if ((ropePattern[l] != 0) != sliding[l])
                    throw new InvalidDataException(
                        $"{arch}: layer {l} has {(ropePattern[l] != 0 ? "RoPE" : "no RoPE")} but {(sliding[l] ? "a sliding window" : "no sliding window")}.");
        }

        float theta = metadata.GetFloat32OrDefault($"{arch}.rope.freq_base", 10000.0f);
        // Only the sliding layers rotate, so a freq_base_swa (none of the published files carry one) is THE theta.
        theta = metadata.GetFloat32OrDefault($"{arch}.rope.freq_base_swa", theta);
        int ropeDims = (int)metadata.GetUInt32OrDefault($"{arch}.rope.dimension_count", (uint)headDim);
        if (ropeDims != headDim)
            throw new NotSupportedException(
                $"{arch}: rope.dimension_count {ropeDims} != head_dim {headDim}; every Ripple port rotates the full head (partial rotary is not part of the contract).");

        var perLayer = new int?[numLayers];
        var noRope = new List<int>();
        for (int l = 0; l < numLayers; l++)
        {
            perLayer[l] = sliding[l] ? (int)window : null;
            if (!sliding[l]) noRope.Add(l);
        }

        float? embedScale = null;
        string embKey = $"{arch}.embedding_scale";
        if (metadata.ContainsKey(embKey))
        {
            float es = metadata.GetFloat32(embKey);
            if (!float.IsFinite(es) || es <= 0f)
                throw new InvalidDataException(
                    $"{embKey} = {es} is invalid: a declared embedding scale must be finite and > 0 (an absent key means the scale is folded into token_embd).");
            embedScale = es;
        }

        var (clamp, clampDeclared) = ReadRicursiveClamp(metadata, arch, numLayers);

        string? chatTemplate = metadata.GetStringOrDefault("tokenizer.chat_template", null!);
        if (string.IsNullOrEmpty(chatTemplate)) chatTemplate = null;

        var rope = new RoPEConfig(Theta: theta, DimensionCount: ropeDims, Type: RoPEType.NeoX);
        return new ModelConfig
        {
            Architecture = Architecture.Ricursive,
            VocabSize = ResolveVocabSize(metadata, arch),
            HiddenSize = hiddenSize,
            IntermediateSize = ffn,
            NumLayers = numLayers,
            NumAttentionHeads = numHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            MaxSequenceLength = (int)metadata.GetUInt32OrDefault($"{arch}.context_length", 2048),
            NormEpsilon = normEps,
            AttentionType = AttentionType.GQA,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            EmbeddingScale = embedScale,
            TiedEmbeddings = false,               // untied lm_head (output.weight); a file without it falls back to token_embd like llama.cpp
            RoPEConfig = rope,
            PositionEncodingType = PositionEncodingType.RoPE,
            SlidingWindowSize = (int)window,
            PerLayerSlidingWindow = perLayer,
            NoRopeLayers = noRope,
            UseQkNorm = true,                     // attn_q_norm / attn_k_norm are required tensors in every port
            ChatTemplate = chatTemplate,
            Ricursive = new RicursiveConfig
            {
                UseGatedAttention = null,
                GateFullPrecision = true,
                UseValueResidual = null,
                FfnClamp = clamp,
                FfnClampDeclared = clampDeclared,
                EmbeddingScaleFolded = embedScale is null,
            },
        };
    }

    /// <summary>
    /// The per-layer sliding flags (true = sliding window + RoPE). An explicit array must have exactly one entry per layer (PR #31692:
    /// a short array would leave the last layers dense and RoPE-free); absent = llama.cpp <c>load_swa_pattern(ml, 5)</c> / the vendor
    /// default <c>layer_types</c>: layer <c>il</c> is sliding iff <c>il % 5 &lt; 4</c>.
    /// </summary>
    private static bool[] ReadRicursiveSlidingPattern(GgufMetadata metadata, string arch, int numLayers)
    {
        string key = $"{arch}.attention.sliding_window_pattern";
        var sliding = new bool[numLayers];
        if (!metadata.TryGetValue(key, out var entry))
        {
            for (int l = 0; l < numLayers; l++)
                sliding[l] = l % RicursiveConfig.DefaultFullAttentionPeriod < RicursiveConfig.DefaultFullAttentionPeriod - 1;
            return sliding;
        }
        if (entry.Type != GgufValueType.Array)
            throw new InvalidDataException($"{key} must be a per-layer bool array for {arch}, got a {entry.Type} scalar.");
        int[] flags = GetIntArrayOrEmpty(metadata, key);
        if (flags.Length != numLayers)
            throw new InvalidDataException($"{key} has {flags.Length} entries, expected {numLayers} (one per layer).");
        for (int l = 0; l < numLayers; l++) sliding[l] = flags[l] != 0;
        return sliding;
    }

    /// <summary>
    /// <c>{arch}.swiglu_clamp</c>: a Float32 scalar (AtomicChat / prometheusAIR files) applying to every layer, or a per-layer Float32
    /// array (PR #31692's current converter). Absent = no clamp and not declared. Every value must be finite and &gt;= 0 (0 = no clamp).
    /// </summary>
    private static (float[] Clamp, bool Declared) ReadRicursiveClamp(GgufMetadata metadata, string arch, int numLayers)
    {
        string key = $"{arch}.swiglu_clamp";
        var clamp = new float[numLayers];
        if (!metadata.TryGetValue(key, out var entry))
            return (clamp, false);
        if (entry.Type == GgufValueType.Array)
        {
            float[] values = metadata.GetFloat32Array(key);
            if (values.Length != numLayers)
                throw new InvalidDataException($"{key} has {values.Length} entries, expected {numLayers}.");
            values.CopyTo(clamp, 0);
        }
        else
        {
            Array.Fill(clamp, metadata.GetFloat32(key));
        }
        for (int l = 0; l < numLayers; l++)
            if (!float.IsFinite(clamp[l]) || clamp[l] < 0f)
                throw new InvalidDataException($"{key} for layer {l} must be finite and non-negative, got {clamp[l]}.");
        return (clamp, true);
    }

    private static int RequiredPositiveU32(GgufMetadata metadata, string key)
    {
        if (!metadata.ContainsKey(key))
            throw new InvalidDataException($"Required GGUF key '{key}' is missing.");
        uint v = metadata.GetUInt32(key);
        if (v == 0 || v > int.MaxValue)
            throw new InvalidDataException($"GGUF key '{key}' = {v} must be a positive 32-bit value.");
        return (int)v;
    }
}
