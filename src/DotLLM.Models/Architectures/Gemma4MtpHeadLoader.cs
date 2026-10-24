using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Loads <see cref="Gemma4MtpHeadWeights"/> from a <c>gemma4-assistant</c> GGUF file (issue #959).
/// The file is memory-mapped; weight pointers point into the mapping and are valid for its lifetime.
/// Norm weights are copied to managed float arrays (small, F32-only).
/// </summary>
internal static class Gemma4MtpHeadLoader
{
    /// <summary>
    /// Loads the Gemma 4 MTP drafter from <paramref name="gguf"/>.
    /// </summary>
    /// <param name="gguf">An opened <c>mtp-*.gguf</c> file with <c>general.architecture = gemma4-assistant</c>.</param>
    /// <param name="trunkHiddenSize">
    /// The trunk model's hidden size (e.g. 2560 for E4B). Used to validate <c>embedding_length_out</c>.
    /// Pass 0 to skip validation.
    /// </param>
    /// <returns>The loaded weights. The caller must keep <paramref name="gguf"/> alive for the pointer lifetime.</returns>
    public static Gemma4MtpHeadWeights Load(GgufFile gguf, int trunkHiddenSize = 0)
    {
        var metadata = gguf.Metadata;
        string archString = metadata.GetString("general.architecture");
        if (archString is not "gemma4-assistant" and not "gemma4_assistant")
            throw new InvalidDataException(
                $"Expected gemma4-assistant GGUF, got architecture '{archString}'.");

        string arch = archString;
        int hiddenSize = (int)metadata.GetUInt32($"{arch}.embedding_length");
        int numLayers = (int)metadata.GetUInt32($"{arch}.block_count");
        int numHeads = (int)metadata.GetUInt32($"{arch}.attention.head_count");
        int numKvHeads = (int)metadata.GetUInt32OrDefault($"{arch}.attention.head_count_kv", (uint)numHeads);
        int intermediateSize = (int)metadata.GetUInt32($"{arch}.feed_forward_length");
        int nextnPredictLayers = (int)metadata.GetUInt32OrDefault($"{arch}.nextn_predict_layers", (uint)numLayers);
        int vocabSize = (int)metadata.GetUInt32OrDefault("tokenizer.ggml.tokens_count",
            metadata.GetUInt32OrDefault($"{arch}.vocab_size", 262144));

        int outHiddenSize = (int)metadata.GetUInt32OrDefault($"{arch}.embedding_length_out", 0);
        if (trunkHiddenSize > 0 && outHiddenSize > 0 && outHiddenSize != trunkHiddenSize)
            throw new InvalidDataException(
                $"Gemma 4 MTP drafter embedding_length_out ({outHiddenSize}) does not match trunk hidden size ({trunkHiddenSize}).");
        int actualTrunkHidden = outHiddenSize > 0 ? outHiddenSize : trunkHiddenSize;
        if (actualTrunkHidden <= 0)
            throw new InvalidDataException(
                "Cannot determine trunk hidden size: neither embedding_length_out in the drafter GGUF nor trunkHiddenSize argument is set.");

        // Sliding-window pattern (per-layer).
        int[] swPattern = GgufModelConfigExtractor.GetIntArrayOrEmpty(metadata, $"{arch}.attention.sliding_window_pattern");

        // Dual head dim: SWA layers use key_length_swa, full layers use key_length.
        int swaHeadDim = (int)metadata.GetUInt32OrDefault($"{arch}.attention.key_length_swa",
            metadata.GetUInt32OrDefault($"{arch}.attention.key_length", (uint)(hiddenSize / numHeads)));
        int fullHeadDim = (int)metadata.GetUInt32OrDefault($"{arch}.attention.key_length",
            (uint)swaHeadDim);

        var dataBase = new GgufDataBase(gguf);
        var tensors = gguf.TensorsByName;

        // ── Global tensors ──

        var preDesc = tensors["nextn.pre_projection.weight"];
        nint prePtr = dataBase.Of(preDesc);

        var postDesc = tensors["nextn.post_projection.weight"];
        nint postPtr = dataBase.Of(postDesc);

        var embDesc = tensors["token_embd.weight"];
        nint embPtr = dataBase.Of(embDesc);

        var outNormDesc = tensors["output_norm.weight"];
        float[] outNorm = DequantizeNorm(dataBase, outNormDesc, hiddenSize);

        float[]? ropeFreqs = null;
        if (tensors.TryGetValue("rope_freqs.weight", out var ropeDesc))
            ropeFreqs = DequantizeNorm(dataBase, ropeDesc, (int)ropeDesc.Shape.ElementCount);

        // ── Per-layer tensors ──

        var layers = new Gemma4MtpLayerWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            string prefix = $"blk.{i}";
            bool isSwa = i < swPattern.Length ? swPattern[i] != 0 : false;
            int headDim = isSwa ? swaHeadDim : fullHeadDim;
            int qDim = numHeads * headDim;

            var attnNormDesc = tensors[$"{prefix}.attn_norm.weight"];
            float[] attnNorm = DequantizeNorm(dataBase, attnNormDesc, hiddenSize);

            var qDesc = tensors[$"{prefix}.attn_q.weight"];
            nint qPtr = dataBase.Of(qDesc);

            var qNormDesc = tensors[$"{prefix}.attn_q_norm.weight"];
            float[] qNorm = DequantizeNorm(dataBase, qNormDesc, qDim);

            var oDesc = tensors[$"{prefix}.attn_output.weight"];
            nint oPtr = dataBase.Of(oDesc);

            var postAttnDesc = tensors[$"{prefix}.post_attention_norm.weight"];
            float[] postAttnNorm = DequantizeNorm(dataBase, postAttnDesc, hiddenSize);

            var ffnNormDesc = tensors[$"{prefix}.ffn_norm.weight"];
            float[] ffnNorm = DequantizeNorm(dataBase, ffnNormDesc, hiddenSize);

            var gateDesc = tensors[$"{prefix}.ffn_gate.weight"];
            nint gatePtr = dataBase.Of(gateDesc);

            var upDesc = tensors[$"{prefix}.ffn_up.weight"];
            nint upPtr = dataBase.Of(upDesc);

            var downDesc = tensors[$"{prefix}.ffn_down.weight"];
            nint downPtr = dataBase.Of(downDesc);

            var postFfwDesc = tensors[$"{prefix}.post_ffw_norm.weight"];
            float[] postFfwNorm = DequantizeNorm(dataBase, postFfwDesc, hiddenSize);

            float layerScale = 1.0f;
            if (tensors.TryGetValue($"{prefix}.layer_output_scale.weight", out var scaleDesc))
            {
                float[] scaleArr = DequantizeNorm(dataBase, scaleDesc, 1);
                layerScale = scaleArr[0];
            }

            layers[i] = new Gemma4MtpLayerWeights
            {
                AttnNormWeight = attnNorm,
                AttnQWeight = qPtr,
                AttnQQuantType = qDesc.QuantizationType,
                AttnQOutputDim = qDim,
                AttnQNormWeight = qNorm,
                AttnOutputWeight = oPtr,
                AttnOutputQuantType = oDesc.QuantizationType,
                PostAttentionNormWeight = postAttnNorm,
                FfnNormWeight = ffnNorm,
                FfnGateWeight = gatePtr,
                FfnGateQuantType = gateDesc.QuantizationType,
                FfnUpWeight = upPtr,
                FfnUpQuantType = upDesc.QuantizationType,
                FfnDownWeight = downPtr,
                FfnDownQuantType = downDesc.QuantizationType,
                PostFfwNormWeight = postFfwNorm,
                LayerOutputScale = layerScale,
                IsSlidingWindow = isSwa,
                HeadDim = headDim,
            };
        }

        return new Gemma4MtpHeadWeights
        {
            NumLayers = numLayers,
            HiddenSize = hiddenSize,
            TrunkHiddenSize = actualTrunkHidden,
            IntermediateSize = intermediateSize,
            NumHeads = numHeads,
            NumKvHeads = numKvHeads,
            VocabSize = vocabSize,
            NextnPredictLayers = nextnPredictLayers,
            PreProjectionWeight = prePtr,
            PreProjectionQuantType = preDesc.QuantizationType,
            PostProjectionWeight = postPtr,
            PostProjectionQuantType = postDesc.QuantizationType,
            TokenEmbdWeight = embPtr,
            TokenEmbdQuantType = embDesc.QuantizationType,
            OutputNormWeight = outNorm,
            RopeFreqsWeight = ropeFreqs,
            Layers = layers,
        };
    }

    private static float[] DequantizeNorm(GgufDataBase dataBase, GgufTensorDescriptor desc, int expectedSize)
    {
        nint ptr = dataBase.Of(desc);
        float[] result = new float[expectedSize];
        Dequantize.ToFloat32(ptr, expectedSize, desc.QuantizationType, result);
        return result;
    }
}
