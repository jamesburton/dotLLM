using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;

namespace DotLLM.Core.Models;

/// <summary>
/// Complete configuration for a transformer model architecture. Populated from GGUF metadata or explicit construction.
/// A single <see cref="ModelConfig"/> parameterizes the transformer block to handle Llama/Mistral/Phi/Qwen/DeepSeek.
/// </summary>
public record ModelConfig
{
    /// <summary>Model architecture family.</summary>
    public required Architecture Architecture { get; init; }

    /// <summary>Vocabulary size (number of token embeddings).</summary>
    public required int VocabSize { get; init; }

    /// <summary>Hidden size (embedding dimension).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>FFN intermediate dimension.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Number of transformer layers.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of attention heads for queries.</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>Number of KV heads. Equal to <see cref="NumAttentionHeads"/> for MHA, 1 for MQA, between for GQA.</summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Dimension per attention head. Typically <see cref="HiddenSize"/> / <see cref="NumAttentionHeads"/>.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Maximum supported sequence length.</summary>
    public required int MaxSequenceLength { get; init; }

    /// <summary>Attention mechanism type (GQA or MLA).</summary>
    public AttentionType AttentionType { get; init; } = AttentionType.GQA;

    /// <summary>Positional encoding type.</summary>
    public PositionEncodingType PositionEncodingType { get; init; } = PositionEncodingType.RoPE;

    /// <summary>RoPE-specific configuration. Null when not using RoPE.</summary>
    public RoPEConfig? RoPEConfig { get; init; }

    /// <summary>Activation function used in FFN layers.</summary>
    public ActivationFunction ActivationFunction { get; init; } = ActivationFunction.SiLU;

    /// <summary>Normalization layer type.</summary>
    public NormType NormType { get; init; } = NormType.RMSNorm;

    /// <summary>Epsilon for normalization layers.</summary>
    public float NormEpsilon { get; init; } = 1e-5f;

    /// <summary>Whether input and output embeddings share weights.</summary>
    public bool TiedEmbeddings { get; init; }

    /// <summary>Sliding window size for local attention. Null = full attention.</summary>
    public int? SlidingWindowSize { get; init; }

    /// <summary>MLA configuration. Only set for DeepSeek-style MLA attention.</summary>
    public MlaConfig? MlaConfig { get; init; }

    /// <summary>Jinja2 chat template from model metadata. Null if not present.</summary>
    public string? ChatTemplate { get; init; }

    /// <summary>
    /// Layer indices that skip RoPE entirely (NoPE — "no positional encoding").
    /// Null or empty means every layer applies RoPE per the standard
    /// <see cref="RoPEConfig"/>. SmolLM3 ships a sparse pattern
    /// (every 4th layer — indices 3, 7, 11, ... on the 3B SKU). The forward
    /// pass tests <see cref="IsNoRopeLayer(int)"/> per layer and conditionally
    /// skips the RoPE rotation while leaving the rest of the GQA pipeline
    /// (projections, attention, output) intact. Non-RoPE architectures
    /// (Mamba-3, MLA decoupled-rope, ...) ignore this field.
    /// </summary>
    public IReadOnlyList<int>? NoRopeLayers { get; init; }

    /// <summary>
    /// Returns true when <paramref name="layerIdx"/> should skip the per-layer
    /// RoPE rotation (NoPE behaviour). Defaults to false when
    /// <see cref="NoRopeLayers"/> is null or empty — every layer applies RoPE.
    /// </summary>
    public bool IsNoRopeLayer(int layerIdx)
    {
        if (NoRopeLayers is null || NoRopeLayers.Count == 0)
            return false;
        return NoRopeLayers.Contains(layerIdx);
    }
}
