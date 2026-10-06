using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;

namespace DotLLM.Models.Gguf;

public static partial class GgufModelConfigExtractor
{
    /// <summary>
    /// Builds a <see cref="ModelConfig"/> for the BERT-class encoders (<c>bert</c>, <c>nomic-bert</c>).
    /// </summary>
    /// <remarks>
    /// llama.cpp reads the LayerNorm epsilon from <c>{arch}.attention.layer_norm_epsilon</c> (not the
    /// RMS key every decoder uses), MiniLM/BERT ship <c>1e-12</c>. nomic-bert carries
    /// <c>rope.freq_base</c> (1000) and rotates the full head dimension with NeoX pairing
    /// (<c>llama_model_rope_type</c> → <c>LLAMA_ROPE_TYPE_NEOX</c>).
    /// </remarks>
    private static ModelConfig BuildBertConfig(GgufMetadata metadata, string arch, Architecture architecture)
    {
        int hidden = (int)metadata.GetUInt32($"{arch}.embedding_length");
        int layers = (int)metadata.GetUInt32($"{arch}.block_count");
        int heads = (int)metadata.GetUInt32($"{arch}.attention.head_count");
        int ff = (int)metadata.GetUInt32($"{arch}.feed_forward_length");
        int headDim = hidden / heads;
        float eps = metadata.GetFloat32OrDefault($"{arch}.attention.layer_norm_epsilon", 1e-12f);

        bool nomic = architecture == Architecture.NomicBert;
        RoPEConfig? rope = null;
        if (nomic)
        {
            rope = new RoPEConfig(
                Theta: metadata.GetFloat32OrDefault($"{arch}.rope.freq_base", 10000.0f),
                DimensionCount: (int)metadata.GetUInt32OrDefault($"{arch}.rope.dimension_count", (uint)headDim),
                Type: RoPEType.NeoX);
        }

        return new ModelConfig
        {
            Architecture = architecture,
            VocabSize = ResolveVocabSize(metadata, arch),
            HiddenSize = hidden,
            IntermediateSize = ff,
            NumLayers = layers,
            NumAttentionHeads = heads,
            NumKvHeads = heads,
            HeadDim = headDim,
            MaxSequenceLength = (int)metadata.GetUInt32OrDefault($"{arch}.context_length", 512),
            NormEpsilon = eps,
            NormType = NormType.LayerNorm,
            ActivationFunction = nomic ? ActivationFunction.SiLU : ActivationFunction.GELU,
            PositionEncodingType = nomic ? PositionEncodingType.RoPE : PositionEncodingType.Absolute,
            RoPEConfig = rope,
            PoolingType = ExtractPoolingType(metadata, arch),
        };
    }
}
