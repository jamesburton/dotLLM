using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Models.Gguf;

public static partial class GgufModelConfigExtractor
{
    /// <summary>
    /// Builds the <see cref="ModelConfig"/> of an <c>olala</c> GGUF (OVHcloud Olala; issue #1970). Keys are those written by the OVH
    /// llama.cpp fork converter (<c>conversion/olala.py</c>, branch <c>olala-master</c>): the generic <c>block_count</c>, <c>embedding_length</c>,
    /// <c>attention.*</c>, <c>expert_*</c>, <c>ssm.*</c> keys plus <c>olala.layer_types</c>, <c>mamba_mimo_dim</c>, <c>mamba_headdim</c>,
    /// <c>num_signal_heads</c>, <c>num_noise_heads</c>, <c>tpa_rank</c>, <c>num_rope_angles</c> and <c>gate_bias</c>.
    /// </summary>
    /// <remarks>
    /// Anything the forward pass does not implement fails loudly instead of running a different model (as HF's
    /// <c>_check_supported_config</c> does): a <c>final_norm</c>, a non-geodesic residual, more than one B/C group, an unknown layer letter.
    /// <c>attention.sliding_window</c> is the HF <c>slw_wsize</c> (32768) — it bounds both the attention window and the scalable-softmax
    /// <c>log(min(pos+1, window))</c>. The zero-centred RMSNorm gamma is already folded (<c>1 + w</c>) into the stored norm weights by the
    /// converter, so <c>olala.zero_centered_gamma</c> is informational only.
    /// </remarks>
    private static ModelConfig BuildOlalaConfig(GgufMetadata metadata, string arch)
    {
        int hidden = RequiredPositiveU32(metadata, $"{arch}.embedding_length");
        int numLayers = RequiredPositiveU32(metadata, $"{arch}.block_count");
        int numHeads = RequiredPositiveU32(metadata, $"{arch}.attention.head_count");
        int numKvHeads = RequiredPositiveU32(metadata, $"{arch}.attention.head_count_kv");
        int headDim = RequiredPositiveU32(metadata, $"{arch}.attention.key_length");
        int valueDim = (int)metadata.GetUInt32OrDefault($"{arch}.attention.value_length", (uint)headDim);
        if (valueDim != headDim)
            throw new InvalidDataException($"{arch}: key_length {headDim} != value_length {valueDim}.");
        float normEps = metadata.GetFloat32($"{arch}.attention.layer_norm_rms_epsilon");
        int window = RequiredPositiveU32(metadata, $"{arch}.attention.sliding_window");
        float softcap = metadata.GetFloat32OrDefault($"{arch}.attn_logit_softcapping", 0f);

        string layerTypes = metadata.GetStringOrDefault($"{arch}.layer_types", null!) ?? "";
        if (layerTypes.Length != numLayers)
            throw new InvalidDataException($"{arch}.layer_types has {layerTypes.Length} letters, expected block_count {numLayers}.");
        foreach (char c in layerTypes)
            if (c != 'M' && c != 'V')
                throw new NotSupportedException($"{arch}.layer_types contains '{c}'; only M (Mamba-3 MIMO) and V (differential TPA attention) are implemented.");

        if (!metadata.GetBoolOrDefault($"{arch}.geodesic_update", false))
            throw new NotSupportedException($"{arch}: geodesic_update=false (pre-norm residual blocks) is not implemented.");
        if (metadata.GetBoolOrDefault($"{arch}.final_norm", false))
            throw new NotSupportedException($"{arch}: final_norm=true is not implemented (the 7A1B has none).");
        int groups = RequiredPositiveU32(metadata, $"{arch}.ssm.group_count");
        if (groups != 1)
            throw new NotSupportedException($"{arch}: ssm.group_count {groups} is not implemented (the B/C layout is only defined for one group).");

        int signal = RequiredPositiveU32(metadata, $"{arch}.num_signal_heads");
        int noise = RequiredPositiveU32(metadata, $"{arch}.num_noise_heads");
        if (signal + noise != numHeads || noise != numKvHeads || signal % noise != 0)
            throw new InvalidDataException(
                $"{arch}: signal {signal} + noise {noise} heads must equal head_count {numHeads}, noise must equal head_count_kv {numKvHeads}, and signal must be a multiple of noise.");

        int ssmInner = RequiredPositiveU32(metadata, $"{arch}.ssm.inner_size");
        int mambaHead = RequiredPositiveU32(metadata, $"{arch}.mamba_headdim");
        if (ssmInner % mambaHead != 0)
            throw new InvalidDataException($"{arch}: ssm.inner_size {ssmInner} is not a multiple of mamba_headdim {mambaHead}.");
        int stateSize = RequiredPositiveU32(metadata, $"{arch}.ssm.state_size");
        int angles = RequiredPositiveU32(metadata, $"{arch}.num_rope_angles");
        if (angles > stateSize / 2)
            throw new InvalidDataException($"{arch}: {angles} rope angle pairs exceed half the state size {stateSize}.");
        uint gating = metadata.GetUInt32OrDefault($"{arch}.expert_gating_func", 2);
        if (gating != 2)
            throw new NotSupportedException($"{arch}: expert_gating_func {gating} is not implemented (sigmoid = 2 only).");

        int sharedFf = (int)metadata.GetUInt32OrDefault($"{arch}.expert_shared_feed_forward_length", 0);
        string? chatTemplate = metadata.GetStringOrDefault("tokenizer.chat_template", null!);
        if (string.IsNullOrEmpty(chatTemplate)) chatTemplate = null;

        return new ModelConfig
        {
            Architecture = Architecture.Olala,
            VocabSize = ResolveVocabSize(metadata, arch),
            HiddenSize = hidden,
            IntermediateSize = 0,
            NumLayers = numLayers,
            NumAttentionHeads = numHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            MaxSequenceLength = (int)metadata.GetUInt32OrDefault($"{arch}.context_length", 2048),
            NormEpsilon = normEps,
            AttentionType = AttentionType.GQA,
            ActivationFunction = ActivationFunction.ReluSquared,
            NormType = NormType.RMSNorm,
            TiedEmbeddings = false,
            PositionEncodingType = PositionEncodingType.None,   // V layers carry no RoPE (rope_theta = 0)
            SlidingWindowSize = window,
            AttnLogitSoftcap = softcap > 0f ? softcap : null,
            UseQkNorm = true,
            ChatTemplate = chatTemplate,
            Olala = new OlalaConfig
            {
                LayerTypes = layerTypes,
                SsmInner = ssmInner,
                StateSize = stateSize,
                GroupCount = groups,
                MambaHeadDim = mambaHead,
                MimoRank = RequiredPositiveU32(metadata, $"{arch}.mamba_mimo_dim"),
                NumRopeAngles = angles,
                NumSignalHeads = signal,
                NumNoiseHeads = noise,
                TpaRank = RequiredPositiveU32(metadata, $"{arch}.tpa_rank"),
                GateBias = metadata.GetFloat32OrDefault($"{arch}.gate_bias", 0f),
                ExpertCount = RequiredPositiveU32(metadata, $"{arch}.expert_count"),
                ExpertUsedCount = RequiredPositiveU32(metadata, $"{arch}.expert_used_count"),
                ExpertFfLength = RequiredPositiveU32(metadata, $"{arch}.expert_feed_forward_length"),
                SharedExpertFfLength = sharedFf,
                ExpertWeightsScale = metadata.GetFloat32OrDefault($"{arch}.expert_weights_scale", 1f),
                MoeLatentSize = (int)metadata.GetUInt32OrDefault($"{arch}.moe_latent_size", 0),
            },
        };
    }
}
