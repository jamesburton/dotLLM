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

        // Gemma 4 / DiffusionGemma have a fundamentally different per-layer shape
        // (dual head_dim, dual KV-head count stored as a per-layer array, dual
        // RoPE per attention type, MoE experts) that does not fit the generic
        // scalar path below — in particular `attention.head_count_kv` is a
        // per-layer Int32 ARRAY here, which the generic GetUInt32 path cannot
        // read. Build it via the dedicated extractor.
        if (architecture is Architecture.Gemma4 or Architecture.DiffusionGemma)
            return BuildGemma4Config(metadata, arch, architecture);

        int hiddenSize = (int)metadata.GetUInt32($"{arch}.embedding_length");
        int numLayers = (int)metadata.GetUInt32($"{arch}.block_count");
        int numAttentionHeads = (int)metadata.GetUInt32($"{arch}.attention.head_count");

        // Multi-Token Prediction (MTP / "NextN") head: llama.cpp PR #22673 stores the trailing
        // MTP block(s) as extra entries appended to block_count (confirmed against the merged
        // convert_hf_to_gguf.py: `block_count = num_hidden_layers + mtp_num_hidden_layers`).
        // Only Qwen3.5/3.6 (Qwen3HybridDense / Qwen3MoeHybrid) ship this key today — every other
        // architecture defaults to 0 and numTrunkLayers == numLayers, so nothing changes for them.
        int nextnPredictLayers = architecture is Architecture.Qwen3MoeHybrid or Architecture.Qwen3HybridDense
            ? (int)metadata.GetUInt32OrDefault($"{arch}.nextn_predict_layers", 0)
            : 0;
        int numTrunkLayers = numLayers - nextnPredictLayers;

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
            // Pure-MoE GGUFs (Qwen-MoE, Mixtral) omit feed_forward_length and store the
            // per-expert width in expert_feed_forward_length instead.
            uint ffLen = metadata.GetUInt32OrDefault($"{arch}.feed_forward_length", 0);
            if (ffLen == 0)
                ffLen = metadata.GetUInt32OrDefault($"{arch}.expert_feed_forward_length", 0);
            intermediateSize = (int)ffLen;
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

        // Interleaved SWA pattern (gpt-oss: window on even layers, dense on odd —
        // llama.cpp set_swa_pattern(2, dense_first=false); the metadata key is
        // optional and defaults to 2 for gpt-oss).
        int slidingWindowPattern = 0;
        if (architecture == Architecture.GptOss && slidingWindowSize is not null)
            slidingWindowPattern = (int)metadata.GetUInt32OrDefault(
                $"{arch}.attention.sliding_window_pattern", 2);

        int vocabSize = ResolveVocabSize(metadata, arch);

        string? chatTemplate = metadata.GetStringOrDefault("tokenizer.chat_template", null!);
        if (string.IsNullOrEmpty(chatTemplate))
            chatTemplate = null;

        RoPEConfig? ropeConfig = ExtractRoPEConfig(metadata, arch, headDim, architecture);

        // GDN models reuse the same {arch}.ssm.* key names as Mamba-2 but with
        // different semantics — skip Mamba-2 SSM config extraction for them.
        MambaSsmConfig? ssmConfig = architecture is Architecture.Qwen3MoeHybrid or Architecture.Qwen3HybridDense
            ? null
            : TryExtractSsmConfig(metadata, arch);

        // DeepSeek-V2/V3: extract MLA + MoE config and patch HeadDim to the full
        // qk_head_dim (key_length stores qk_nope only; total = qk_nope + qk_rope).
        MlaConfig? mlaConfig = null;
        MoeConfig? moeConfig = null;
        GatedDeltaNetConfig? gdnConfig = null;
        AttentionType attentionType = AttentionType.GQA;
        if (architecture is Architecture.QwenMoe or Architecture.Mixtral)
        {
            moeConfig = TryExtractQwenMoeConfig(metadata, arch, numLayers);
        }
        else if (architecture is Architecture.DeepSeekV2 or Architecture.DeepSeekV3)
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
        else if (architecture is Architecture.Qwen3MoeHybrid or Architecture.Qwen3HybridDense)
        {
            gdnConfig = TryExtractGdnConfig(metadata, arch);
            // Dense hybrid (Qwen3HybridDense, e.g. Bonsai) has no MoE sublayer at all —
            // don't call TryExtractQwenMoeConfig for it. It would return null anyway
            // (no {arch}.expert_count key), but calling it only for the MoE variant
            // keeps the intent explicit rather than relying on that null-return.
            if (architecture is Architecture.Qwen3MoeHybrid)
                moeConfig = TryExtractQwenMoeConfig(metadata, arch, numLayers);
            // Build per-layer layout from full_attention_interval (not stored as
            // per-layer arrays like Nemotron-H, so TryExtractHybridLayout returned null).
            // Use numTrunkLayers, not raw numLayers: an MTP checkpoint's block_count
            // includes the trailing nextn_predict_layers MTP block(s), which are always
            // full-attention and must not be interleaved into the GDN/full-attn pattern.
            if (gdnConfig is { } gdn)
                hybridLayout = BuildQwen3MoeHybridLayout(numTrunkLayers, gdn.FullAttnInterval, numKvHeads);
        }
        else if (architecture == Architecture.GptOss)
        {
            moeConfig = ExtractGptOssMoeConfig(metadata, arch, intermediateSize);
        }

        return new ModelConfig
        {
            Architecture = architecture,
            VocabSize = vocabSize,
            HiddenSize = hiddenSize,
            IntermediateSize = intermediateSize,
            NumLayers = numTrunkLayers,
            NextnPredictLayers = nextnPredictLayers,
            NumAttentionHeads = numAttentionHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            MaxSequenceLength = maxSeqLen,
            NormEpsilon = normEps,
            AttentionType = attentionType,
            ActivationFunction = architecture is Architecture.NemotronH or Architecture.BitNet
                ? ActivationFunction.ReluSquared
                : ActivationFunction.SiLU,
            RoPEConfig = ropeConfig,
            PositionEncodingType = ropeConfig.HasValue ? PositionEncodingType.RoPE : PositionEncodingType.None,
            SlidingWindowSize = slidingWindowSize,
            SlidingWindowPattern = slidingWindowPattern,
            HybridLayout = hybridLayout,
            SsmConfig = ssmConfig,
            MlaConfig = mlaConfig,
            Moe = moeConfig,
            GdnConfig = gdnConfig,
            ChatTemplate = chatTemplate,
        };
    }

    /// <summary>
    /// Applies user-supplied RoPE overrides (CLI <c>--rope-scaling</c>/<c>--rope-freq-base</c>/etc.,
    /// or server <c>ModelLoadRequest</c> equivalents) on top of the GGUF-derived <see cref="ModelConfig.RoPEConfig"/>
    /// (and <see cref="ModelConfig.GlobalRoPEConfig"/> for dual-RoPE architectures like Gemma 4).
    /// No new scaling math — every override field is a straight replacement of the corresponding
    /// <see cref="RoPEConfig"/> field, everything else derived from GGUF metadata is left as-is.
    /// A no-op (returns <paramref name="config"/> unchanged) when <paramref name="overrides"/> is
    /// null, has no fields set, or the model has no RoPE config to override (non-RoPE architectures).
    /// </summary>
    public static ModelConfig ApplyRoPEOverride(ModelConfig config, RoPEOverrideOptions? overrides)
    {
        if (overrides is null || !overrides.HasAnyOverride)
            return config;

        ModelConfig result = config;
        if (config.RoPEConfig is { } rope)
            result = result with { RoPEConfig = ApplyOverride(rope, overrides) };
        if (config.GlobalRoPEConfig is { } globalRope)
            result = result with { GlobalRoPEConfig = ApplyOverride(globalRope, overrides) };
        return result;
    }

    private static RoPEConfig ApplyOverride(RoPEConfig rope, RoPEOverrideOptions overrides) => rope with
    {
        Theta = overrides.FreqBase ?? rope.Theta,
        ScalingType = overrides.ScalingType ?? rope.ScalingType,
        ScalingFactor = overrides.ScalingFactor ?? rope.ScalingFactor,
        OrigMaxSeqLen = overrides.OrigMaxSeqLen ?? rope.OrigMaxSeqLen,
        AttnFactor = overrides.AttnFactor ?? rope.AttnFactor,
        BetaFast = overrides.BetaFast ?? rope.BetaFast,
        BetaSlow = overrides.BetaSlow ?? rope.BetaSlow,
    };

    /// <summary>
    /// Extracts an <see cref="MlaConfig"/> from DeepSeek-V2/V3 GGUF metadata.
    /// Required keys (per llama.cpp's gguf_writer):
    /// <list type="bullet">
    ///   <item><c>{arch}.attention.q_lora_rank</c> — Q LoRA bottleneck (0 = monolithic, V2-Lite default)</item>
    ///   <item><c>{arch}.attention.kv_lora_rank</c> — KV LoRA bottleneck (typically 512)</item>
    ///   <item><c>{arch}.attention.key_length</c> — TOTAL per-head qk dim (qk_nope + qk_rope; 192 on V2-Lite)</item>
    ///   <item><c>{arch}.attention.value_length</c> — v_head_dim (may differ from qk_nope_head_dim)</item>
    ///   <item><c>{arch}.rope.dimension_count</c> — qk_rope_head_dim (must be even; 64 on V2-Lite)</item>
    /// </list>
    /// <para>
    /// <b>qk_nope_head_dim is derived</b> as <c>key_length - rope.dimension_count</c>.
    /// Confirmed against the bartowski DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf
    /// (key_length=192, rope.dimension_count=64 ⇒ qk_nope=128).
    /// </para>
    /// </summary>
    private static MlaConfig ExtractMlaConfig(GgufMetadata metadata, string arch, RoPEConfig? ropeConfig)
    {
        // q_lora_rank may be absent or zero on V2-Lite (monolithic-Q variant).
        int qLoraRank = (int)metadata.GetUInt32OrDefault($"{arch}.attention.q_lora_rank", 0);
        int kvLoraRank = (int)metadata.GetUInt32($"{arch}.attention.kv_lora_rank");
        int qkTotal = (int)metadata.GetUInt32($"{arch}.attention.key_length");
        int vHead = (int)metadata.GetUInt32($"{arch}.attention.value_length");
        int qkRope = (int)metadata.GetUInt32($"{arch}.rope.dimension_count");
        int qkNope = qkTotal - qkRope;

        if (kvLoraRank <= 0)
            throw new InvalidDataException(
                $"DeepSeek-V2 MLA requires '{arch}.attention.kv_lora_rank' > 0; got {kvLoraRank}.");
        if (qkRope <= 0 || (qkRope & 1) != 0)
            throw new InvalidDataException(
                $"DeepSeek-V2 MLA requires '{arch}.rope.dimension_count' (qk_rope) to be a positive even number; got {qkRope}.");
        if (qkNope <= 0)
            throw new InvalidDataException(
                $"DeepSeek-V2 MLA requires '{arch}.attention.key_length' > '{arch}.rope.dimension_count' " +
                $"(qk_nope = key_length - rope.dimension_count); got {qkTotal} and {qkRope}.");

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
            // Phase C latent cache by default — matches HfConfigExtractor's
            // default at 4b54a72. DeepSeek-V2/V3 are designed around the
            // latent KV cache; Phase A's expanded cache scales as O(numLayers
            // × maxSeqLen × numHeads × (qkNope + v) × 4 bytes) which for
            // V2-Lite at max_position_embeddings=163840 is ~68 GB and OOMs.
            // Phase C ≈ 9 GB for the same model.
            UseHybridMlaCache = true,
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

    /// <summary>
    /// Extracts a <see cref="MoeConfig"/> for Qwen-MoE and Mixtral GGUF checkpoints.
    /// </summary>
    /// <remarks>
    /// Per llama.cpp's GGUF convention: <c>{arch}.expert_count</c> = total routed experts,
    /// <c>{arch}.expert_used_count</c> = top-k, <c>{arch}.expert_feed_forward_length</c> = per-expert
    /// intermediate width, <c>{arch}.expert_shared_count</c> = shared experts (Qwen1.5-MoE),
    /// <c>{arch}.decoder_sparse_step</c> = Qwen3-MoE alternating-layer stride (default 1 = all MoE).
    /// </remarks>
    private static MoeConfig? TryExtractQwenMoeConfig(GgufMetadata metadata, string arch, int numLayers)
    {
        uint expertCount = metadata.GetUInt32OrDefault($"{arch}.expert_count", 0);
        if (expertCount == 0) return null;

        int expertUsed = (int)metadata.GetUInt32($"{arch}.expert_used_count");
        int moeIntermediate = (int)metadata.GetUInt32($"{arch}.expert_feed_forward_length");
        int expertShared = (int)metadata.GetUInt32OrDefault($"{arch}.expert_shared_count", 0);

        // qwen35moe convention: shared-expert count is implicit (1) and only the intermediate
        // width is stored as `expert_shared_feed_forward_length`. Detect this case so we still
        // mark the layer as having a shared branch + sigmoid gate (ffn_gate_inp_shexp.weight).
        int sharedFfl = (int)metadata.GetUInt32OrDefault($"{arch}.expert_shared_feed_forward_length", 0);
        bool sharedFromFfl = expertShared == 0 && sharedFfl > 0;
        bool hasSharedExpertGate = false;
        if (sharedFromFfl)
        {
            expertShared = 1;
            hasSharedExpertGate = true; // qwen35moe always pairs the shared branch with a sigmoid gate.
        }

        // decoder_sparse_step: Qwen3-MoE alternates dense/MoE every N layers (typically 2).
        // Qwen3.6 / Mixtral have all layers as MoE so this key is absent and defaults to 1.
        int decoderSparseStep = (int)metadata.GetUInt32OrDefault($"{arch}.decoder_sparse_step", 1);

        int[]? mlpOnlyLayers = null;
        if (decoderSparseStep > 1)
        {
            // Layers 0, step, 2*step, … are dense; the rest are MoE.
            var denseLayers = new List<int>();
            for (int i = 0; i < numLayers; i += decoderSparseStep)
                denseLayers.Add(i);
            mlpOnlyLayers = denseLayers.ToArray();
        }

        // qwen35moe: per-expert width (sharedFfl), NOT multiplied by count (count is implicit 1).
        // DeepSeek path: count × moeIntermediate (legacy convention, kept for the historic V2/V3 path).
        int? sharedIntermediate = sharedFromFfl
            ? sharedFfl
            : (expertShared > 0 ? moeIntermediate * expertShared : null);

        return new MoeConfig
        {
            NumExperts = (int)expertCount,
            NumExpertsPerTok = expertUsed,
            MoeIntermediateSize = moeIntermediate,
            NormTopKProb = true,   // Qwen-MoE and Mixtral always renormalize top-k weights
            SharedExpertIntermediateSize = sharedIntermediate,
            NumSharedExperts = expertShared,
            HasSharedExpertGate = hasSharedExpertGate,
            DecoderSparseStep = decoderSparseStep,
            MlpOnlyLayers = mlpOnlyLayers,
        };
    }

    /// <summary>
    /// Extracts the gpt-oss MoE configuration. Every layer is a routed-MoE
    /// layer (no dense lead, no shared experts). Router gating is
    /// softmax-after-top-k over raw (bias-added) logits; experts use the
    /// clamped <c>swiglu_oai</c> activation; router and expert projections all
    /// carry biases. Per llama.cpp's <c>LLM_ARCH_OPENAI_MOE</c>.
    /// </summary>
    private static MoeConfig ExtractGptOssMoeConfig(GgufMetadata metadata, string arch,
                                                     int denseIntermediate)
    {
        int expertCount = (int)metadata.GetUInt32($"{arch}.expert_count");
        int expertUsed = (int)metadata.GetUInt32($"{arch}.expert_used_count");
        int moeIntermediate = (int)metadata.GetUInt32OrDefault(
            $"{arch}.expert_feed_forward_length", (uint)denseIntermediate);

        return new MoeConfig
        {
            NumExperts = expertCount,
            NumExpertsPerTok = expertUsed,
            MoeIntermediateSize = moeIntermediate,
            SoftmaxAfterTopK = true,
            UseSwiGluOai = true,
            HasExpertBiases = true,
            DecoderSparseStep = 1,
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

    /// <summary>
    /// Extracts a <see cref="GatedDeltaNetConfig"/> from Qwen3MoeHybrid GGUF metadata.
    /// Returns null if the defining key (<c>{arch}.ssm.inner_size</c>) is absent.
    /// </summary>
    /// <remarks>
    /// Key mapping (confirmed from llama.cpp <c>src/llama.cpp</c> qwen35moe case):
    /// <list type="bullet">
    ///   <item><c>{arch}.full_attention_interval</c> → <see cref="GatedDeltaNetConfig.FullAttnInterval"/></item>
    ///   <item><c>{arch}.ssm.inner_size</c>          → <see cref="GatedDeltaNetConfig.DInner"/></item>
    ///   <item><c>{arch}.ssm.state_size</c>           → <see cref="GatedDeltaNetConfig.DState"/> (= head_k_dim = head_v_dim)</item>
    ///   <item><c>{arch}.ssm.time_step_rank</c>       → <see cref="GatedDeltaNetConfig.NVHead"/> (num_v_heads)</item>
    ///   <item><c>{arch}.ssm.group_count</c>          → <see cref="GatedDeltaNetConfig.NKHead"/> (num_k_heads)</item>
    ///   <item><c>{arch}.ssm.conv_kernel</c>          → <see cref="GatedDeltaNetConfig.DConv"/></item>
    /// </list>
    /// </remarks>
    private static GatedDeltaNetConfig? TryExtractGdnConfig(GgufMetadata metadata, string arch)
    {
        string innerKey = $"{arch}.ssm.inner_size";
        if (!metadata.ContainsKey(innerKey)) return null;

        int fullAttnInterval = (int)metadata.GetUInt32OrDefault($"{arch}.full_attention_interval", 4);
        int dInner = (int)metadata.GetUInt32(innerKey);
        int dState = (int)metadata.GetUInt32($"{arch}.ssm.state_size");
        int nVHead = (int)metadata.GetUInt32($"{arch}.ssm.time_step_rank");
        int nKHead = (int)metadata.GetUInt32OrDefault($"{arch}.ssm.group_count", 1);
        int dConv = (int)metadata.GetUInt32($"{arch}.ssm.conv_kernel");

        if (nVHead <= 0)
            throw new InvalidDataException(
                $"GDN config requires '{arch}.ssm.time_step_rank' (num_v_heads) > 0; got {nVHead}.");
        if (nVHead % nKHead != 0)
            throw new InvalidDataException(
                $"GDN time_step_rank (nVHead={nVHead}) must be divisible by group_count (nKHead={nKHead}).");

        return new GatedDeltaNetConfig(fullAttnInterval, nVHead, nKHead, dState, dInner, dConv);
    }

    /// <summary>
    /// Constructs a <see cref="HybridLayerLayout"/> for Qwen3MoeHybrid models using
    /// the <paramref name="fullAttnInterval"/> formula rather than per-layer GGUF arrays.
    /// Layer <c>i</c> (0-based) is full attention when <c>(i+1) % fullAttnInterval == 0</c>;
    /// all other layers are <see cref="HybridLayerKind.GatedDeltaNet"/>.
    /// All layers carry a MoE FFN tracked separately via <see cref="ModelConfig.Moe"/>,
    /// so <c>FeedForwardLength</c> is zeroed throughout.
    /// </summary>
    private static HybridLayerLayout BuildQwen3MoeHybridLayout(
        int numLayers, int fullAttnInterval, int numKvHeads)
    {
        var kinds = new HybridLayerKind[numLayers];
        var headCountKv = new int[numLayers];
        var feedForwardLength = new int[numLayers]; // all zero — MoE tracked via ModelConfig.Moe

        for (int i = 0; i < numLayers; i++)
        {
            bool isFullAttn = (i + 1) % fullAttnInterval == 0;
            kinds[i] = isFullAttn ? HybridLayerKind.Attention : HybridLayerKind.GatedDeltaNet;
            headCountKv[i] = isFullAttn ? numKvHeads : 0;
        }

        return new HybridLayerLayout
        {
            LayerKind = kinds,
            HeadCountKv = headCountKv,
            FeedForwardLength = feedForwardLength,
        };
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
            // LLaDA is a Llama-backbone masked-diffusion LLM (same transformer; the
            // diffusion behaviour comes from the decode loop, not the weights). Its
            // GGUFs declare general.architecture = "llada" and store hyperparameters
            // under "llada.*" — read via the dynamic arch-prefix path, so mapping to
            // the Llama transformer is sufficient.
            "llada" => Architecture.Llama,
            "mistral" or "mistral3" => Architecture.Mistral,
            "phi" or "phi2" or "phi3" => Architecture.Phi,
            "qwen" or "qwen2" or "qwen3" => Architecture.Qwen,
            // Qwen dense MoE variants: Qwen1.5-MoE, Qwen2-MoE, Qwen3-MoE.
            "qwen2moe" or "qwen3moe" or "qwenmoe" => Architecture.QwenMoe,
            // Qwen3.6-35B-A3B: Gated DeltaNet hybrid — NOT a plain Qwen-MoE transformer.
            "qwen35moe" => Architecture.Qwen3MoeHybrid,
            // Dense Qwen3.5 hybrid (no MoE suffix) — same GDN/attention alternation as
            // qwen35moe, dense SwiGLU FFN instead of sparse MoE. First seen in PrismML's
            // Bonsai-27B (distilled from Qwen/Qwen3.6-27B).
            "qwen35" => Architecture.Qwen3HybridDense,
            "mixtral" => Architecture.Mixtral,
#pragma warning disable CS0618 // Preserve legacy GGUF metadata mapping for compatibility diagnostics.
            // Pre-V2 DeepSeek (legacy placeholder — never actually loaded by us).
            "deepseek" => Architecture.DeepSeek,
#pragma warning restore CS0618
            // V2 / V2-Lite — MLA + MoE per <c>convert_hf_to_gguf.py</c>'s
            // <c>DeepseekV2Model</c>. Distinct from V3 only in routing details.
            "deepseek2" => Architecture.DeepSeekV2,
            // V3 / V3-MoE — MLA + sigmoid-gated routing + group-norm experts.
            "deepseek3" => Architecture.DeepSeekV3,
            "nemotron_h" => Architecture.NemotronH,
            // Gemma 4 MoE text tower (llama.cpp `gemma4` arch). Dual head-dim /
            // dual KV-head / dual-RoPE-per-attention-type + MoE experts. See
            // BuildGemma4Config for the full GGUF → ModelConfig mapping.
            "gemma4" => Architecture.Gemma4,
            // DiffusionGemma — the Gemma 4 MoE tower run as a masked-canvas block
            // diffusion model. Same `gemma4` backbone; the `diffusion-gemma` GGUF
            // adds `diffusion.canvas_length` + a `<mask>` token, which
            // BuildGemma4Config turns into a non-null DiffusionConfig.
            "diffusion-gemma" or "diffusion_gemma" => Architecture.DiffusionGemma,
            "bitnet" or "bitnet-b1.58" or "bitnet-25" => Architecture.BitNet,
            // OpenAI gpt-oss (llama.cpp LLM_ARCH_OPENAI_MOE).
            "gpt-oss" => Architecture.GptOss,
            _ => throw new InvalidDataException($"Unsupported GGUF architecture: '{archString}'.")
        };
    }

    /// <summary>
    /// Builds a Gemma-4 MoE <see cref="ModelConfig"/> from a llama.cpp
    /// <c>gemma4</c> / <c>diffusion-gemma</c> GGUF. This is the GGUF counterpart
    /// of <see cref="DotLLM.Models.SafeTensors.DiffusionGemmaConfigExtractor"/>
    /// and produces an identically-shaped config (dual head dim, dual KV-head
    /// count, dual RoPE per attention type, MoE experts, soft-cap, embedding
    /// scale, per-layer sliding-window pattern).
    /// </summary>
    /// <remarks>
    /// <para><b>Per-layer arrays.</b> Unlike Gemma 3, the gemma4 GGUF stores
    /// <c>attention.head_count_kv</c> as a per-layer Int32 array (e.g.
    /// <c>[8,8,8,8,8,2, …]</c>) and <c>attention.sliding_window_pattern</c> as a
    /// per-layer 0/1 array (<c>1</c> = sliding/local layer, <c>0</c> = full/global
    /// layer). The two arrays are correlated: sliding layers carry the larger KV
    /// count (<c>num_key_value_heads</c>, e.g. 8) and the SWA head dim / local RoPE;
    /// full layers carry the smaller KV count (<c>num_global_key_value_heads</c>,
    /// e.g. 2) and the global head dim / global RoPE. The sliding values land on
    /// <see cref="ModelConfig.NumKvHeads"/>/<see cref="ModelConfig.HeadDim"/>/
    /// <see cref="ModelConfig.RoPEConfig"/>; the full values on
    /// <see cref="ModelConfig.NumGlobalKvHeads"/>/<see cref="ModelConfig.GlobalHeadDim"/>/
    /// <see cref="ModelConfig.GlobalRoPEConfig"/>.</para>
    /// <para><b>Partial rotary.</b> llama.cpp bakes the partial-rotary factor into
    /// <c>rope.dimension_count(_swa)</c> directly (it stores the actual rotated
    /// dim, not a fraction). We therefore set the RoPE <c>DimensionCount</c> from
    /// those keys and leave <see cref="ModelConfig.PartialRotaryFactor"/> null — the
    /// forward path uses <c>DimensionCount</c> as the rotated span when the factor
    /// is null, which is exactly the GGUF representation.</para>
    /// <para><b>PLE / shared KV (dense E2B/E4B).</b> The dense text-tower GGUFs
    /// (<c>unsloth/gemma-4-E4B-it</c>) report
    /// <c>embedding_length_per_layer_input &gt; 0</c> and ship the Gemma-3n-style
    /// <c>per_layer_*</c> PLE tensors, plus <c>attention.shared_kv_layers</c>
    /// (trailing layers reuse an earlier layer's KV) and a <c>rope_freqs</c>
    /// proportional-rope factor tensor for the full-attention layers — all wired
    /// below. The MoE 26B GGUFs report 0 / omit these and are unaffected.</para>
    /// <para><b>AltUp / Laurel / activation sparsity.</b> Gemma-3n-only components
    /// that Gemma 4 dropped: no released <c>gemma4</c> GGUF carries
    /// <c>altup_*</c>/<c>laurel_*</c> tensors and llama.cpp's <c>gemma4.cpp</c>
    /// graph has no gaussian-topk sparsity, so none are wired here.</para>
    /// </remarks>
    private static ModelConfig BuildGemma4Config(GgufMetadata metadata, string arch, Architecture architecture)
    {
        int hiddenSize = (int)metadata.GetUInt32($"{arch}.embedding_length");
        int numLayers = (int)metadata.GetUInt32($"{arch}.block_count");
        int numAttentionHeads = (int)metadata.GetUInt32($"{arch}.attention.head_count");
        int maxSeqLen = (int)metadata.GetUInt32OrDefault($"{arch}.context_length", 2048);
        float normEps = metadata.GetFloat32OrDefault($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);
        int vocabSize = ResolveVocabSize(metadata, arch);

        // Dense FFN width (the non-expert ffn_up/gate/down on every layer — Gemma 4
        // keeps a dense MLP alongside the routed experts). Stored under
        // feed_forward_length; expert width is separate (expert_feed_forward_length).
        int intermediateSize = (int)metadata.GetUInt32OrDefault($"{arch}.feed_forward_length", 0);

        // Sliding-window pattern: per-layer 1 (sliding/local) / 0 (full/global).
        int slidingWindow = (int)metadata.GetUInt32OrDefault($"{arch}.attention.sliding_window", 1024);
        if (slidingWindow <= 0) slidingWindow = 1024;
        int[] swPattern = GetIntArrayOrEmpty(metadata, $"{arch}.attention.sliding_window_pattern");
        var perLayerSliding = new int?[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            bool isSliding = i < swPattern.Length ? swPattern[i] != 0 : false;
            perLayerSliding[i] = isSliding ? slidingWindow : (int?)null;
        }

        // Per-layer KV-head array → (sliding count, full count). When a layer is
        // sliding its entry is the local KV count; full layers carry the global
        // KV count. Derive each by scanning for the first layer of each kind.
        int[] kvArray = GetIntArrayOrEmpty(metadata, $"{arch}.attention.head_count_kv");
        int slidingKv = numAttentionHeads, globalKv = numAttentionHeads;
        bool sawSliding = false, sawGlobal = false;
        for (int i = 0; i < numLayers && i < kvArray.Length; i++)
        {
            bool isSliding = perLayerSliding[i] is not null;
            if (isSliding && !sawSliding) { slidingKv = kvArray[i]; sawSliding = true; }
            if (!isSliding && !sawGlobal) { globalKv = kvArray[i]; sawGlobal = true; }
        }
        if (kvArray.Length == 0)
            slidingKv = globalKv = (int)metadata.GetUInt32OrDefault($"{arch}.attention.head_count_kv", (uint)numAttentionHeads);
        int? numGlobalKvHeads = sawGlobal && globalKv != slidingKv ? globalKv : null;

        // Dual head dim: key_length_swa = sliding head dim, key_length = full head dim.
        int slidingHeadDim = (int)metadata.GetUInt32OrDefault($"{arch}.attention.key_length_swa",
            metadata.GetUInt32OrDefault($"{arch}.attention.key_length", (uint)(hiddenSize / numAttentionHeads)));
        int globalHeadDim = (int)metadata.GetUInt32OrDefault($"{arch}.attention.key_length",
            (uint)slidingHeadDim);
        int? globalHeadDimNullable = globalHeadDim != slidingHeadDim ? globalHeadDim : null;

        // Dual RoPE: *_swa keys are the sliding (local) schedule; the bare keys are
        // the full (global) schedule. llama.cpp bakes the partial-rotary factor into
        // rope.dimension_count(_swa) — we use those dims directly and leave
        // PartialRotaryFactor null.
        float slidingTheta = metadata.GetFloat32OrDefault($"{arch}.rope.freq_base_swa",
            metadata.GetFloat32OrDefault($"{arch}.rope.freq_base", 10000.0f));
        int slidingRopeDim = (int)metadata.GetUInt32OrDefault($"{arch}.rope.dimension_count_swa",
            (uint)slidingHeadDim);
        float globalTheta = metadata.GetFloat32OrDefault($"{arch}.rope.freq_base", 1_000_000.0f);
        int globalRopeDim = (int)metadata.GetUInt32OrDefault($"{arch}.rope.dimension_count",
            (uint)globalHeadDim);

        var slidingRope = new RoPEConfig(Theta: slidingTheta, DimensionCount: slidingRopeDim, Type: RoPEType.NeoX);
        var globalRope = new RoPEConfig(Theta: globalTheta, DimensionCount: globalRopeDim, Type: RoPEType.NeoX);

        // MoE experts (expert_count / expert_used_count / expert_feed_forward_length).
        MoeConfig? moe = null;
        uint expertCount = metadata.GetUInt32OrDefault($"{arch}.expert_count", 0);
        if (expertCount > 0)
        {
            moe = new MoeConfig
            {
                NumExperts = (int)expertCount,
                NumExpertsPerTok = (int)metadata.GetUInt32($"{arch}.expert_used_count"),
                MoeIntermediateSize = (int)metadata.GetUInt32($"{arch}.expert_feed_forward_length"),
                NormTopKProb = true,
            };
        }

        // ── Per-Layer Embeddings (PLE) — the dense Gemma-4 text tower (E2B/E4B). ──
        // embedding_length_per_layer_input > 0 marks the PLE variant: an auxiliary
        // per-layer token-embedding table (per_layer_token_embd, width
        // pleDim*numLayers) plus a context projection (per_layer_model_proj) feed a
        // gated residual into every layer (llama.cpp gemma4.cpp build_inp_per_layer /
        // project_per_layer_inputs). The released MoE 26B GGUFs report 0 here and
        // carry no per_layer_* tensors, so this stays null for them.
        PerLayerEmbeddingConfig? perLayerEmbedding = null;
        uint pleDim = metadata.GetUInt32OrDefault($"{arch}.embedding_length_per_layer_input", 0);
        if (pleDim > 0)
        {
            perLayerEmbedding = new PerLayerEmbeddingConfig
            {
                PerLayerDim = (int)pleDim,
                // The gemma4 GGUF has a single vocabulary — per_layer_token_embd
                // has the same row count as token_embd.
                VocabSize = vocabSize,
            };
        }

        // ── Shared trailing KV layers (Gemma-4 E2B/E4B). ──
        // attention.shared_kv_layers = number of TRAILING layers that reuse an
        // earlier layer's KV (llama.cpp: n_layer_kv_from_start = n_layer - shared).
        // The reuse rule itself lives on ModelConfig.SharedKvDonorLayer.
        int sharedKvLayers = (int)metadata.GetUInt32OrDefault($"{arch}.attention.shared_kv_layers", 0);
        if (sharedKvLayers < 0 || sharedKvLayers >= numLayers)
            throw new InvalidDataException(
                $"gemma4 GGUF '{arch}.attention.shared_kv_layers' = {sharedKvLayers} must be in [0, {numLayers}).");

        // Partial rotary: the MoE 26B rotates only the leading 0.25 fraction of the
        // 512-dim global head (validated end-to-end). The dense-PLE E2B/E4B variant
        // instead rotates the FULL head dim on both layer kinds
        // (rope.dimension_count == key_length, rope.dimension_count_swa ==
        // key_length_swa; llama.cpp n_rot(il) comes straight from those keys) and
        // modulates the global-layer frequencies via the rope_freqs.weight
        // proportional-rope factor tensor (loaded by TransformerWeights). Gate on
        // the PLE marker, which cleanly separates the two released families.
        float? partialRotaryFactor = perLayerEmbedding is null ? 0.25f : null;

        float? finalLogitSoftcap = null;
        float fls = metadata.GetFloat32OrDefault($"{arch}.final_logit_softcapping", 0f);
        if (fls > 0f) finalLogitSoftcap = fls;
        float? attnLogitSoftcap = null;
        float als = metadata.GetFloat32OrDefault($"{arch}.attn_logit_softcapping", 0f);
        if (als > 0f) attnLogitSoftcap = als;

        string? chatTemplate = metadata.GetStringOrDefault("tokenizer.chat_template", null!);
        if (string.IsNullOrEmpty(chatTemplate)) chatTemplate = null;

        // DiffusionGemma: a non-null DiffusionConfig (canvas length + mask token).
        // The DiffusionGemma backbone is the EXACT gemma4 backbone; the diffusion
        // graph adds three region-aware deltas (region embed rms_noscale on the
        // canvas, region per-layer scalar enc_layer_output_scale/layer_output_scale,
        // region-aware Hybrid(P) mask) wired in TransformerModel, gated on
        // Config.DiffusionConfig. Self-conditioning + the PKV prefill/decode cache +
        // the long-sequence sliding-mask clip remain DEFERRED optimizations.
        DiffusionConfig? diffusion = null;
        if (architecture is Architecture.DiffusionGemma)
        {
            int canvas = (int)metadata.GetUInt32OrDefault("diffusion.canvas_length", 256);
            int maskTokenId = (int)metadata.GetUInt32OrDefault("tokenizer.ggml.mask_token_id", uint.MaxValue);
            if (maskTokenId == int.MaxValue || maskTokenId < 0)
                throw new InvalidDataException(
                    "diffusion-gemma GGUF is missing 'tokenizer.ggml.mask_token_id'; a diffusion model cannot decode without a mask token.");
            diffusion = new DiffusionConfig
            {
                CanvasLength = canvas,
                MaskTokenId = maskTokenId,
                CanvasAttentionMode = DotLLM.Core.Attention.AttentionMaskMode.Hybrid,
            };
        }

        return new ModelConfig
        {
            Architecture = architecture,
            VocabSize = vocabSize,
            HiddenSize = hiddenSize,
            IntermediateSize = intermediateSize,
            NumLayers = numLayers,
            NumAttentionHeads = numAttentionHeads,
            NumKvHeads = slidingKv,
            HeadDim = slidingHeadDim,
            MaxSequenceLength = maxSeqLen,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = slidingRope,
            GlobalRoPEConfig = globalRope,
            // Gemma 4 full-attention (global) layers apply RoPE to only the
            // leading 0.25 fraction of the 512-dim head (first 64 pairs = 128
            // dims; the GGUF rope.dimension_count is the FULL head dim, not the
            // rotated span — partial_rotary_factor 0.25 selects the rotated dims).
            // The forward path multiplies this factor by the global head dim and
            // rounds down to even. Sliding layers keep full rotation (factor n/a).
            // Null on the dense-PLE E2B/E4B variant (full-dim rotation + rope_freqs
            // proportional factors — see above).
            PartialRotaryFactor = partialRotaryFactor,
            NumGlobalKvHeads = numGlobalKvHeads,
            GlobalHeadDim = globalHeadDimNullable,
            // Gemma 4 q_norm/k_norm make Q,K unit so the attention softmax scale
            // is exactly 1.0 (NOT 1/sqrt(head_dim)). QueryPreAttnScalar = 1.0
            // routes the forward's scale computation to 1/sqrt(1.0) == 1.0.
            QueryPreAttnScalar = 1.0f,
            // Gemma 4 MoE dual-FFN graph (V-from-K, weight-less V-norm, dense+MoE
            // parallel FFN, custom router, per-expert down scale, layer_output_scale).
            Gemma4DualFfn = true,
            ActivationFunction = ActivationFunction.GELUTanh,
            NormType = NormType.RMSNorm,
            NormEpsilon = normEps,
            // Gemma ties input/output embeddings.
            TiedEmbeddings = true,
            SlidingWindowSize = slidingWindow,
            PerLayerSlidingWindow = perLayerSliding,
            AttnLogitSoftcap = attnLogitSoftcap,
            FinalLogitSoftcap = finalLogitSoftcap,
            EmbeddingScale = MathF.Sqrt(hiddenSize),
            Moe = moe,
            PerLayerEmbedding = perLayerEmbedding,
            NumSharedKvLayers = sharedKvLayers,
            ChatTemplate = chatTemplate,
            DiffusionConfig = diffusion,
        };
    }

    /// <summary>
    /// Reads a GGUF metadata key as an Int32 array, tolerating the UInt32 array
    /// encoding llama.cpp uses for unsigned per-layer values. Returns an empty
    /// array when the key is absent or not an array.
    /// </summary>
    private static int[] GetIntArrayOrEmpty(GgufMetadata metadata, string key)
    {
        if (!metadata.TryGetValue(key, out var entry) || entry.Type != GgufValueType.Array)
            return Array.Empty<int>();
        // The element type can be Int32, UInt32, or Bool depending on the writer.
        // gemma4 stores attention.sliding_window_pattern as a Bool array
        // ([True,True,…,False] — true = sliding/local layer) and
        // attention.head_count_kv as a (U)Int32 array. Map each to int[].
        if (entry.Value is int[] ints)
            return ints;
        if (entry.Value is uint[] uints)
        {
            var r = new int[uints.Length];
            for (int i = 0; i < uints.Length; i++) r[i] = (int)uints[i];
            return r;
        }
        if (entry.Value is bool[] bools)
        {
            var r = new int[bools.Length];
            for (int i = 0; i < bools.Length; i++) r[i] = bools[i] ? 1 : 0;
            return r;
        }
        return Array.Empty<int>();
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
        // - Llama/Mistral: mainline llama.cpp's converter permutes Q/K weights → interleaved (Norm)
        // - Qwen/Phi: weights kept in HuggingFace order → non-interleaved (NeoX)
        // Qwen family (dense + MoE + GDN hybrid) and Phi keep HF weight order → NeoX pairing.
        //
        // BitNet (issue #247): despite being "Llama-shaped" architecturally, BitNet GGUFs are
        // written by Microsoft's own bitnet.cpp fork of llama.cpp, whose `BitnetModel.modify_tensors`
        // does NOT call the `permute()` step that `LlamaModel.modify_tensors` applies to attn_q/attn_k
        // (verified against bitnet.cpp's `convert-hf-to-gguf-bitnet.py`: `BitnetModel` overrides
        // `modify_tensors` to quantize weights but never invokes `permute`). So BitNet's Q/K weights
        // stay in the original HuggingFace (`rotate_half`) layout, same as Qwen/Phi — NeoX pairing is
        // required, not Norm. Applying Norm (interleaved) pairing to HF-layout weights doesn't corrupt
        // short-range attention (rotation angles are near-identity for small positions/relative
        // distances) but increasingly scrambles the learned phase structure as position grows, which
        // is exactly the "quality degrades with sequence length" symptom this fixes. This is
        // BitNet-specific: other I2_S-quantized GGUFs built on a plain Llama-arch body (e.g.
        // Falcon-E-3B, Falcon3-3B) go through the normal llama.cpp Llama conversion path (with the
        // permute) and correctly resolve to Architecture.Llama, not Architecture.BitNet, so they are
        // unaffected by this case.
        RoPEType ropeType = architecture switch
        {
            Architecture.Qwen or Architecture.QwenMoe
                or Architecture.Qwen3MoeHybrid or Architecture.Qwen3HybridDense or Architecture.Phi
                or Architecture.GptOss or Architecture.BitNet => RoPEType.NeoX,
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
