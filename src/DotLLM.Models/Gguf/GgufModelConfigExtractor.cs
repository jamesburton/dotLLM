using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Extracts a <see cref="ModelConfig"/> from GGUF metadata following standard GGUF key conventions.
/// </summary>
public static class GgufModelConfigExtractor
{
    /// <summary>
    /// Builds a <see cref="ModelConfig"/> from the given GGUF metadata.
    /// </summary>
    /// <param name="metadata">Parsed GGUF metadata.</param>
    /// <returns>A fully populated <see cref="ModelConfig"/>.</returns>
    /// <exception cref="InvalidDataException">Required metadata keys are missing or have invalid values.</exception>
    public static ModelConfig Extract(GgufMetadata metadata)
    {
        string archString = metadata.GetString("general.architecture");
        Architecture architecture = ParseArchitecture(archString);
        string arch = archString.ToLowerInvariant();

        int hiddenSize = (int)metadata.GetUInt32($"{arch}.embedding_length");
        int numLayers = (int)metadata.GetUInt32($"{arch}.block_count");
        int numAttentionHeads = (int)metadata.GetUInt32($"{arch}.attention.head_count");

        // Hybrid models (Nemotron-H) store head_count_kv and feed_forward_length as
        // per-layer Int32 arrays whose entries are zero for layers of the wrong kind.
        // Build a HybridLayerLayout in that case; for pure-Transformer architectures
        // both keys are scalar UInt32.
        HybridLayerLayout? hybridLayout = TryExtractHybridLayout(metadata, arch, numLayers);

        int intermediateSize;
        int numKvHeads;
        if (hybridLayout is not null)
        {
            // Use the *attention-layer* values as the canonical scalar config so existing
            // attention/KV-cache code paths see meaningful sizes. Fall back to zeros only
            // when the model has no attention layers at all (unsupported here).
            numKvHeads = MaxNonZero(hybridLayout.HeadCountKv, numAttentionHeads);
            intermediateSize = MaxNonZero(hybridLayout.FeedForwardLength, 0);
        }
        else
        {
            intermediateSize = (int)metadata.GetUInt32($"{arch}.feed_forward_length");
            numKvHeads = (int)metadata.GetUInt32OrDefault($"{arch}.attention.head_count_kv", (uint)numAttentionHeads);
        }

        // Head dimension: prefer explicit GGUF key (needed for models like Qwen3 where
        // head_dim != hidden_size / num_heads), fall back to derived value.
        // For DeepSeek-V2/V3 MLA, key_length is the qk_nope_head_dim only — total
        // qk_head_dim is qk_nope + qk_rope; HeadDim is fixed up after MLA config
        // extraction below.
        int headDim = (int)metadata.GetUInt32OrDefault($"{arch}.attention.key_length",
                                                        (uint)(hiddenSize / numAttentionHeads));
        int maxSeqLen = (int)metadata.GetUInt32OrDefault($"{arch}.context_length", 2048);

        float normEps = metadata.GetFloat32OrDefault($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f);

        int? slidingWindowSize = null;
        uint swValue = metadata.GetUInt32OrDefault($"{arch}.attention.sliding_window", 0);
        if (swValue > 0)
            slidingWindowSize = (int)swValue;

        int vocabSize = ResolveVocabSize(metadata, arch);

        string? chatTemplate = metadata.GetStringOrDefault("tokenizer.chat_template", null!);
        if (string.IsNullOrEmpty(chatTemplate))
            chatTemplate = null;

        RoPEConfig? ropeConfig = ExtractRoPEConfig(metadata, arch, headDim, architecture);
        MambaSsmConfig? ssmConfig = TryExtractSsmConfig(metadata, arch);

        // DeepSeek-V2/V3: extract MLA + MoE config and patch HeadDim to the full
        // qk_head_dim (key_length stores qk_nope only; total = qk_nope + qk_rope).
        MlaConfig? mlaConfig = null;
        MoeConfig? moeConfig = null;
        AttentionType attentionType = AttentionType.GQA;
        if (architecture is Architecture.DeepSeekV2 or Architecture.DeepSeekV3)
        {
            mlaConfig = ExtractMlaConfig(metadata, arch, ropeConfig);
            moeConfig = TryExtractDeepseekMoeConfig(metadata, arch, intermediateSize, numLayers);
            attentionType = AttentionType.MLA;
            // GGUF's attention.key_length is qk_nope only. Total per-head dim
            // for MLA attention is qk_nope + qk_rope — patch HeadDim so the
            // GQA-shaped pieces of the model (cache stride etc.) see the full
            // value.
            headDim = mlaConfig.QkNopeHeadDim + mlaConfig.QkRopeHeadDim;
        }

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
            NormEpsilon = normEps,
            AttentionType = attentionType,
            ActivationFunction = architecture == Architecture.NemotronH
                ? ActivationFunction.ReluSquared
                : ActivationFunction.SiLU,
            RoPEConfig = ropeConfig,
            PositionEncodingType = ropeConfig.HasValue ? PositionEncodingType.RoPE : PositionEncodingType.None,
            SlidingWindowSize = slidingWindowSize,
            HybridLayout = hybridLayout,
            SsmConfig = ssmConfig,
            MlaConfig = mlaConfig,
            Moe = moeConfig,
            ChatTemplate = chatTemplate,
        };
    }

    /// <summary>
    /// Extracts an <see cref="MlaConfig"/> from DeepSeek-V2/V3 GGUF metadata.
    /// Required keys (per llama.cpp's gguf_writer):
    /// <list type="bullet">
    ///   <item><c>{arch}.attention.q_lora_rank</c> — Q LoRA bottleneck (0 = monolithic, V2-Lite default)</item>
    ///   <item><c>{arch}.attention.kv_lora_rank</c> — KV LoRA bottleneck (typically 512)</item>
    ///   <item><c>{arch}.attention.key_length</c> — qk_nope_head_dim (NOT including rope half)</item>
    ///   <item><c>{arch}.attention.value_length</c> — v_head_dim (may differ from qk_nope_head_dim)</item>
    ///   <item><c>{arch}.rope.dimension_count</c> — qk_rope_head_dim (must be even)</item>
    /// </list>
    /// </summary>
    private static MlaConfig ExtractMlaConfig(GgufMetadata metadata, string arch, RoPEConfig? ropeConfig)
    {
        // q_lora_rank may be absent or zero on V2-Lite (monolithic-Q variant).
        int qLoraRank = (int)metadata.GetUInt32OrDefault($"{arch}.attention.q_lora_rank", 0);
        int kvLoraRank = (int)metadata.GetUInt32($"{arch}.attention.kv_lora_rank");
        int qkNope = (int)metadata.GetUInt32($"{arch}.attention.key_length");
        int vHead = (int)metadata.GetUInt32($"{arch}.attention.value_length");
        int qkRope = (int)metadata.GetUInt32($"{arch}.rope.dimension_count");

        if (kvLoraRank <= 0)
            throw new InvalidDataException(
                $"DeepSeek-V2 MLA requires '{arch}.attention.kv_lora_rank' > 0; got {kvLoraRank}.");
        if (qkRope <= 0 || (qkRope & 1) != 0)
            throw new InvalidDataException(
                $"DeepSeek-V2 MLA requires '{arch}.rope.dimension_count' (qk_rope) to be a positive even number; got {qkRope}.");

        float ropeTheta = ropeConfig?.Theta ?? 10000.0f;

        // YaRN params (when rope.scaling.type=yarn). Already extracted into
        // ropeConfig but MLA carries its own copy for the standalone MLA
        // softmax-scale correction (see MlaConfig.ComputeYarnSoftmaxScaleMultiplier).
        float? ropeScalingFactor = null;
        float? ropeScalingMscale = null;
        float? ropeScalingMscaleAllDim = null;
        int? ropeScalingOrigCtx = null;
        if (ropeConfig is { ScalingType: RoPEScalingType.YaRN } yarn)
        {
            ropeScalingFactor = yarn.ScalingFactor;
            ropeScalingMscale = metadata.GetFloat32OrDefault($"{arch}.rope.scaling.yarn_log_multiplier", 0.0f);
            ropeScalingMscaleAllDim = metadata.GetFloat32OrDefault($"{arch}.rope.scaling.attn_factor", 1.0f);
            ropeScalingOrigCtx = yarn.OrigMaxSeqLen > 0 ? yarn.OrigMaxSeqLen : null;
        }

        return new MlaConfig
        {
            KvLoraRank = kvLoraRank,
            QLoraRank = qLoraRank,
            QkNopeHeadDim = qkNope,
            QkRopeHeadDim = qkRope,
            VHeadDim = vHead,
            RopeTheta = ropeTheta,
            RopeScalingFactor = ropeScalingFactor,
            RopeScalingMscale = ropeScalingMscale,
            RopeScalingMscaleAllDim = ropeScalingMscaleAllDim,
            RopeScalingOriginalMaxPositionEmbeddings = ropeScalingOrigCtx,
        };
    }

    /// <summary>
    /// Extracts a <see cref="MoeConfig"/> from DeepSeek-V2/V3 GGUF metadata when
    /// the model declares MoE FFN (<c>{arch}.expert_count</c> &gt; 0). Returns null
    /// for non-MoE checkpoints (e.g. dense-only V2 fine-tunes).
    /// </summary>
    /// <remarks>
    /// Per llama.cpp's gguf_writer: <c>{arch}.expert_count</c> = total routed
    /// experts; <c>{arch}.expert_used_count</c> = top-k; <c>{arch}.expert_shared_count</c>
    /// = N shared experts (V2-Lite=2, V2-full=2, V3=1); <c>{arch}.expert_feed_forward_length</c>
    /// = moe_intermediate_size per expert; <c>{arch}.leading_dense_block_count</c>
    /// = first_k_dense_replace (number of leading layers that stay dense FFN).
    /// </remarks>
    private static MoeConfig? TryExtractDeepseekMoeConfig(GgufMetadata metadata, string arch,
                                                           int denseIntermediate, int numLayers)
    {
        uint expertCount = metadata.GetUInt32OrDefault($"{arch}.expert_count", 0);
        if (expertCount == 0) return null;

        int expertUsed = (int)metadata.GetUInt32($"{arch}.expert_used_count");
        int expertShared = (int)metadata.GetUInt32OrDefault($"{arch}.expert_shared_count", 0);
        int moeIntermediate = (int)metadata.GetUInt32OrDefault(
            $"{arch}.expert_feed_forward_length", (uint)denseIntermediate);
        int leadingDense = (int)metadata.GetUInt32OrDefault($"{arch}.leading_dense_block_count", 0);

        // DeepSeek convention: leading_dense_block_count = N means layers
        // [0, N) are dense FFN, [N, numLayers) are MoE. Map this to MoeConfig's
        // MlpOnlyLayers (the explicit per-index dense override) so the existing
        // IsMoeLayer dispatcher works without extra plumbing.
        int[]? mlpOnlyLayers = null;
        if (leadingDense > 0)
        {
            mlpOnlyLayers = new int[leadingDense];
            for (int i = 0; i < leadingDense; i++) mlpOnlyLayers[i] = i;
        }

        // Shared-expert intermediate: DeepSeek-V2/V3 fuses N shared experts into
        // a single MLP of width (moe_intermediate * n_shared_experts) on disk
        // (HfConfigExtractor docs the same convention). The CudaMoe loader / CPU
        // path consume `SharedExpertIntermediateSize` as the *total* width and
        // `NumSharedExperts` as the count.
        int? sharedIntermediate = null;
        if (expertShared > 0)
            sharedIntermediate = moeIntermediate * expertShared;

        return new MoeConfig
        {
            NumExperts = (int)expertCount,
            NumExpertsPerTok = expertUsed,
            MoeIntermediateSize = moeIntermediate,
            NormTopKProb = true,   // V2 + V3 both renormalize
            SharedExpertIntermediateSize = sharedIntermediate,
            NumSharedExperts = expertShared,
            HasSharedExpertGate = false,  // DeepSeek convention: no per-token sigmoid gate
            DecoderSparseStep = 1,
            MlpOnlyLayers = mlpOnlyLayers,
        };
    }

    private static HybridLayerLayout? TryExtractHybridLayout(GgufMetadata metadata, string arch, int numLayers)
    {
        string kvKey = $"{arch}.attention.head_count_kv";
        string ffKey = $"{arch}.feed_forward_length";

        if (!metadata.TryGetValue(kvKey, out var kvEntry) || kvEntry.Type != GgufValueType.Array) return null;
        if (!metadata.TryGetValue(ffKey, out var ffEntry) || ffEntry.Type != GgufValueType.Array) return null;

        // Both keys are per-layer Int32 arrays in hybrid models (Nemotron-H).
        int[] headCountKv = metadata.GetInt32Array(kvKey);
        int[] feedForwardLength = metadata.GetInt32Array(ffKey);

        if (headCountKv.Length != numLayers)
            throw new InvalidDataException(
                $"'{kvKey}' array length {headCountKv.Length} does not match block_count {numLayers}.");
        if (feedForwardLength.Length != numLayers)
            throw new InvalidDataException(
                $"'{ffKey}' array length {feedForwardLength.Length} does not match block_count {numLayers}.");

        var kinds = new HybridLayerKind[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            bool hasAttn = headCountKv[i] > 0;
            bool hasFfn = feedForwardLength[i] > 0;
            kinds[i] = (hasAttn, hasFfn) switch
            {
                (true, false) => HybridLayerKind.Attention,
                (false, true) => HybridLayerKind.Ffn,
                (false, false) => HybridLayerKind.Ssm,
                (true, true) => throw new InvalidDataException(
                    $"Layer {i} has both non-zero head_count_kv and feed_forward_length; hybrid block kinds must be exclusive.")
            };
        }

        return new HybridLayerLayout
        {
            LayerKind = kinds,
            HeadCountKv = headCountKv,
            FeedForwardLength = feedForwardLength,
        };
    }

    private static MambaSsmConfig? TryExtractSsmConfig(GgufMetadata metadata, string arch)
    {
        string innerKey = $"{arch}.ssm.inner_size";
        if (!metadata.ContainsKey(innerKey)) return null;

        int dConv = (int)metadata.GetUInt32($"{arch}.ssm.conv_kernel");
        int dInner = (int)metadata.GetUInt32(innerKey);
        int dState = (int)metadata.GetUInt32($"{arch}.ssm.state_size");
        int nGroup = (int)metadata.GetUInt32OrDefault($"{arch}.ssm.group_count", 1);
        int nHead = (int)metadata.GetUInt32($"{arch}.ssm.time_step_rank");

        if (dInner % nHead != 0)
            throw new InvalidDataException(
                $"SSM inner_size {dInner} not divisible by time_step_rank {nHead}.");
        if (dInner % nGroup != 0)
            throw new InvalidDataException(
                $"SSM inner_size {dInner} not divisible by group_count {nGroup}.");
        if (nHead % nGroup != 0)
            throw new InvalidDataException(
                $"SSM time_step_rank {nHead} not divisible by group_count {nGroup}.");

        return new MambaSsmConfig(dConv, dInner, dState, nGroup, nHead);
    }

    private static int MaxNonZero(int[] values, int fallback)
    {
        int max = 0;
        foreach (int v in values) if (v > max) max = v;
        return max > 0 ? max : fallback;
    }

    private static Architecture ParseArchitecture(string archString)
    {
        return archString.ToLowerInvariant() switch
        {
            "llama" => Architecture.Llama,
            "mistral" or "mistral3" => Architecture.Mistral,
            "phi" or "phi2" or "phi3" => Architecture.Phi,
            "qwen" or "qwen2" or "qwen3" => Architecture.Qwen,
            // Pre-V2 DeepSeek (legacy placeholder — never actually loaded by us).
            "deepseek" => Architecture.DeepSeek,
            // V2 / V2-Lite — MLA + MoE per <c>convert_hf_to_gguf.py</c>'s
            // <c>DeepseekV2Model</c>. Distinct from V3 only in routing details.
            "deepseek2" => Architecture.DeepSeekV2,
            // V3 / V3-MoE — MLA + sigmoid-gated routing + group-norm experts.
            "deepseek3" => Architecture.DeepSeekV3,
            "nemotron_h" => Architecture.NemotronH,
            _ => throw new InvalidDataException($"Unsupported GGUF architecture: '{archString}'.")
        };
    }

    private static int ResolveVocabSize(GgufMetadata metadata, string arch)
    {
        uint vocabSize = metadata.GetUInt32OrDefault($"{arch}.vocab_size", 0);
        if (vocabSize > 0)
            return (int)vocabSize;

        // Fallback: count entries in the tokenizer vocabulary array.
        if (metadata.ContainsKey("tokenizer.ggml.tokens"))
        {
            string[] tokens = metadata.GetStringArray("tokenizer.ggml.tokens");
            return tokens.Length;
        }

        throw new InvalidDataException(
            "Cannot determine vocabulary size: neither '{arch}.vocab_size' nor 'tokenizer.ggml.tokens' found.");
    }

    private static RoPEConfig? ExtractRoPEConfig(GgufMetadata metadata, string arch, int headDim,
        Architecture architecture)
    {
        // If no rope keys exist at all, this model may not use RoPE.
        string freqBaseKey = $"{arch}.rope.freq_base";
        string dimCountKey = $"{arch}.rope.dimension_count";
        if (!metadata.ContainsKey(freqBaseKey) && !metadata.ContainsKey(dimCountKey))
            return null;

        float theta = metadata.GetFloat32OrDefault(freqBaseKey, 10000.0f);
        int dimCount = (int)metadata.GetUInt32OrDefault(dimCountKey, (uint)headDim);

        // Determine RoPE element-pairing convention. Must match the GGUF Q/K weight layout:
        // - Llama/Mistral: converter permutes Q/K weights → interleaved (Norm)
        // - Qwen/Phi: weights kept in HuggingFace order → non-interleaved (NeoX)
        RoPEType ropeType = architecture switch
        {
            Architecture.Qwen or Architecture.Phi => RoPEType.NeoX,
            _ => RoPEType.Norm,
        };

        RoPEScalingType scalingType = RoPEScalingType.None;
        float scalingFactor = 1.0f;
        int origMaxSeqLen = 0;
        float attnFactor = 1.0f;
        float betaFast = 32.0f;
        float betaSlow = 1.0f;

        string scalingTypeKey = $"{arch}.rope.scaling.type";
        if (metadata.ContainsKey(scalingTypeKey))
        {
            string scalingTypeStr = metadata.GetString(scalingTypeKey);
            scalingType = scalingTypeStr.ToLowerInvariant() switch
            {
                "linear" => RoPEScalingType.Linear,
                "yarn" => RoPEScalingType.YaRN,
                "ntk" => RoPEScalingType.NTK,
                "dynamic" or "dynamic_ntk" => RoPEScalingType.DynamicNTK,
                "su" or "longrope" => RoPEScalingType.Su,
                _ => RoPEScalingType.None
            };

            scalingFactor = metadata.GetFloat32OrDefault($"{arch}.rope.scaling.factor", 1.0f);
            origMaxSeqLen = (int)metadata.GetUInt32OrDefault($"{arch}.rope.scaling.original_context_length", 0);
            attnFactor = metadata.GetFloat32OrDefault($"{arch}.rope.scaling.attn_factor", 1.0f);
            betaFast = metadata.GetFloat32OrDefault($"{arch}.rope.scaling.beta_fast", 32.0f);
            betaSlow = metadata.GetFloat32OrDefault($"{arch}.rope.scaling.beta_slow", 1.0f);
        }

        return new RoPEConfig(
            Theta: theta,
            DimensionCount: dimCount,
            Type: ropeType,
            ScalingType: scalingType,
            ScalingFactor: scalingFactor,
            OrigMaxSeqLen: origMaxSeqLen,
            AttnFactor: attnFactor,
            BetaFast: betaFast,
            BetaSlow: betaSlow);
    }
}
