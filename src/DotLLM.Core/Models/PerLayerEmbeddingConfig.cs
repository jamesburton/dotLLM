namespace DotLLM.Core.Models;

/// <summary>
/// Per-Layer Embeddings (PLE) configuration for the Gemma-4 text tower
/// (<c>gemma4_text</c>, e.g. <c>google/gemma-4-E2B</c>). PLE feeds an auxiliary
/// gated residual into every decoder layer, combining a per-layer token-identity
/// lookup (<c>embed_tokens_per_layer</c>) with a context-aware projection of the
/// scaled input embedding. Non-null only for a Gemma-4 dense text tower that ships
/// the PLE tables; every other architecture leaves
/// <see cref="ModelConfig.PerLayerEmbedding"/> null and is unaffected.
/// </summary>
/// <remarks>
/// Mirrors HF <c>Gemma4TextConfig</c> fields <c>vocab_size_per_layer_input</c> and
/// <c>hidden_size_per_layer_input</c>. The per-layer embedding table has
/// <see cref="VocabSize"/> rows of width <c>NumLayers * <see cref="PerLayerDim"/></c>.
/// </remarks>
public sealed record PerLayerEmbeddingConfig
{
    /// <summary>Vocabulary size of the per-layer embedding table
    /// (<c>vocab_size_per_layer_input</c>; 262144 for E2B). Usually equal to the
    /// main <see cref="ModelConfig.VocabSize"/>.</summary>
    public required int VocabSize { get; init; }

    /// <summary>Per-layer embedding dimension (<c>hidden_size_per_layer_input</c>;
    /// 256 for E2B). Each layer receives a slice of this width.</summary>
    public required int PerLayerDim { get; init; }
}
