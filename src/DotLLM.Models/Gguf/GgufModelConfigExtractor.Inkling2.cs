using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Models.Gguf;

public static partial class GgufModelConfigExtractor
{
    /// <summary>
    /// Builds the <see cref="ModelConfig"/> of an <c>inkling2</c> / <c>inkling2_loop</c> GGUF (Thinking Machines Inkling 2; issue #2026). Key names
    /// are those of llama.cpp's <c>llama-arch.cpp</c> and <c>conversion/inkling2.py</c>.
    /// </summary>
    /// <remarks>
    /// Anything the CPU reference does not implement fails loudly instead of running a different model: non-uniform head counts or head
    /// dimensions across layers, a loop that is not aligned to the attention-residual blocks, a RoPE dimension, a varying loop
    /// (<c>loop.varying</c>), or a missing numerics key. The embedding and output tensors are padded past the vocabulary, so
    /// <see cref="ModelConfig.VocabSize"/> is the tokenizer size (unpadded) and <see cref="Inkling2Config.PaddedVocabSize"/> the row count of the embedding / output matrices.
    /// </remarks>
    private static ModelConfig BuildInkling2Config(GgufMetadata metadata, string arch)
    {
        int hidden = RequiredPositiveU32(metadata, $"{arch}.embedding_length");
        int numLayers = RequiredPositiveU32(metadata, $"{arch}.block_count");

        // head_count / head_count_kv are per-layer arrays in the converter's output; accept a scalar too, but every layer must agree.
        int numHeads = UniformLayerValue(metadata, $"{arch}.attention.head_count", numLayers);
        int numKvHeads = UniformLayerValue(metadata, $"{arch}.attention.head_count_kv", numLayers);
        if (numKvHeads <= 0 || numHeads % numKvHeads != 0)
            throw new InvalidDataException($"{arch}: head_count {numHeads} is not a multiple of head_count_kv {numKvHeads}.");
        int headDim = RequiredPositiveU32(metadata, $"{arch}.attention.key_length");
        foreach (string k in new[] { $"{arch}.attention.value_length", $"{arch}.attention.key_length_swa", $"{arch}.attention.value_length_swa" })
            if ((int)metadata.GetUInt32OrDefault(k, (uint)headDim) != headDim)
                throw new NotSupportedException($"{arch}: {k} differs from key_length {headDim}; per-layer head dimensions are not implemented.");
        if (metadata.GetUInt32OrDefault($"{arch}.rope.dimension_count", 0) != 0)
            throw new NotSupportedException($"{arch}: rope.dimension_count != 0; Inkling 2 has no RoPE (relative-position bias carries the positions).");

        bool[] pattern = metadata.GetBoolArray($"{arch}.attention.sliding_window_pattern");
        if (pattern.Length != numLayers)
            throw new InvalidDataException($"{arch}.attention.sliding_window_pattern has {pattern.Length} entries, expected block_count {numLayers}.");

        float[] req = new float[]
        {
            metadata.GetFloat32($"{arch}.attention.layer_norm_rms_epsilon"),
            metadata.GetFloat32($"{arch}.attention.qk_clip"),
            metadata.GetFloat32($"{arch}.attention.relative_clip"),
            metadata.GetFloat32($"{arch}.activation.situ_beta"),
        };
        foreach (float v in req)
            if (!float.IsFinite(v) || v <= 0f)
                throw new InvalidDataException($"{arch}: normalisation, clipping and SITU parameters must be finite and positive.");
        float linearBeta = metadata.GetFloat32($"{arch}.activation.situ_linear_beta");
        float alpha = metadata.GetFloat32($"{arch}.attention.log_scale_alpha");
        if (!float.IsFinite(linearBeta) || linearBeta < 0f || !float.IsFinite(alpha) || alpha < 0f)
            throw new InvalidDataException($"{arch}: invalid situ_linear_beta {linearBeta} or log_scale_alpha {alpha}.");

        int padded = RequiredPositiveU32(metadata, $"{arch}.vocab_size");
        int unpadded = RequiredPositiveU32(metadata, $"{arch}.vocab_size_unpadded");
        if (unpadded > padded)
            throw new InvalidDataException($"{arch}: vocab_size_unpadded {unpadded} exceeds vocab_size {padded}.");

        int convKernel = RequiredPositiveU32(metadata, $"{arch}.shortconv.l_cache");
        if (convKernel < 2 || convKernel > 64)
            throw new InvalidDataException($"{arch}: shortconv.l_cache {convKernel} out of range [2, 64].");
        int block = RequiredPositiveU32(metadata, $"{arch}.attn_res.block_size");
        int window = RequiredPositiveU32(metadata, $"{arch}.attention.sliding_window");
        int relLen = RequiredPositiveU32(metadata, $"{arch}.attention.relative_length");
        int relExtent = RequiredPositiveU32(metadata, $"{arch}.attention.relative_extent");

        int experts = RequiredPositiveU32(metadata, $"{arch}.expert_count");
        int used = RequiredPositiveU32(metadata, $"{arch}.expert_used_count");
        int expertFf = RequiredPositiveU32(metadata, $"{arch}.expert_feed_forward_length");
        if (used > experts)
            throw new InvalidDataException($"{arch}: expert_used_count {used} exceeds expert_count {experts}.");

        // The loop keys exist only in inkling2_loop; a plain inkling2 file is the same graph with one pass.
        int loopStart = 0, loopEnd = 0, passes = 1, rank = 0;
        if (metadata.ContainsKey($"{arch}.loop.count"))
        {
            loopStart = RequiredPositiveU32(metadata, $"{arch}.loop.start");
            loopEnd = RequiredPositiveU32(metadata, $"{arch}.loop.end");
            passes = RequiredPositiveU32(metadata, $"{arch}.loop.count");
            rank = RequiredPositiveU32(metadata, $"{arch}.loop.gate_rank");
            if (metadata.GetBoolOrDefault($"{arch}.loop.varying", false))
                throw new NotSupportedException($"{arch}: loop.varying=true (a variable number of passes) is not implemented.");
            // llama.cpp: a fixed loop needs at least 2 passes; the core must sit on attention-residual block boundaries.
            if (passes < 2 || loopStart >= loopEnd || loopEnd > numLayers || loopStart % block != 0 || loopEnd % block != 0)
                throw new InvalidDataException(
                    $"{arch}: invalid loop core [{loopStart}, {loopEnd}) with block size {block}, {passes} passes for {numLayers} layers.");
        }

        string? chatTemplate = metadata.GetStringOrDefault("tokenizer.chat_template", null!);
        if (string.IsNullOrEmpty(chatTemplate)) chatTemplate = null;

        return new ModelConfig
        {
            Architecture = Architecture.Inkling2,
            VocabSize = unpadded, // logits cover the tokenizer vocabulary only; the padded rows of the output matrix are never computed
            HiddenSize = hidden,
            IntermediateSize = (int)metadata.GetUInt32OrDefault($"{arch}.feed_forward_length", (uint)expertFf),
            NumLayers = numLayers,
            NumAttentionHeads = numHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            MaxSequenceLength = (int)Math.Min(metadata.GetUInt32OrDefault($"{arch}.context_length", 2048), int.MaxValue),
            NormEpsilon = req[0],
            AttentionType = AttentionType.GQA,
            ActivationFunction = ActivationFunction.SiLU, // nominal: the experts use SITU (see Inkling2Config); generic loaders are refused
            NormType = NormType.RMSNorm,
            TiedEmbeddings = false,
            PositionEncodingType = PositionEncodingType.None,
            SlidingWindowSize = window,
            UseQkNorm = false,
            ChatTemplate = chatTemplate,
            Inkling2 = new Inkling2Config
            {
                IsLocalLayer = pattern,
                SlidingWindow = window,
                RelativeLength = relLen,
                RelativeExtent = relExtent,
                QkClip = req[1],
                RelativeClip = req[2],
                LogScaleFloor = RequiredPositiveU32(metadata, $"{arch}.attention.log_scale_floor"),
                LogScaleAlpha = alpha,
                ConvKernel = convKernel,
                AttnResBlockSize = block,
                SituBeta = req[3],
                SituLinearBeta = linearBeta,
                ExpertCount = experts,
                ExpertUsedCount = used,
                ExpertFfLength = expertFf,
                ExpertWeightsScale = metadata.GetFloat32OrDefault($"{arch}.expert_weights_scale", 1f),
                FinalLogitSoftcap = metadata.GetFloat32OrDefault($"{arch}.final_logit_softcapping", 0f),
                PaddedVocabSize = padded,
                LoopStart = loopStart,
                LoopEnd = loopEnd,
                LoopPasses = passes,
                LoopGateRank = rank,
            },
        };
    }

    /// <summary>Reads a per-layer array (or a scalar) and requires every layer to hold the same value.</summary>
    private static int UniformLayerValue(GgufMetadata metadata, string key, int numLayers)
    {
        if (!metadata.TryGetValue(key, out var value))
            throw new InvalidDataException($"Required GGUF key '{key}' is missing (inkling2).");
        if (value.Type != GgufValueType.Array)
            return RequiredPositiveU32(metadata, key);
        int[] arr = value.Value is uint[] u ? Array.ConvertAll(u, x => (int)x) : metadata.GetInt32Array(key);
        if (arr.Length != numLayers)
            throw new InvalidDataException($"{key} has {arr.Length} entries, expected block_count {numLayers}.");
        if (arr.Length == 0 || arr.Any(v => v != arr[0]) || arr[0] <= 0)
            throw new NotSupportedException($"{key} varies across layers or is not positive; per-layer head counts are not implemented for inkling2.");
        return arr[0];
    }
}
