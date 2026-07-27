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

    /// <summary>
    /// Optional multiplier applied to the input embeddings immediately after the
    /// token-embedding lookup. Gemma scales embeddings by <c>sqrt(hidden_size)</c>
    /// (the "normalizer" in HF's <c>Gemma3TextScaledWordEmbedding</c>). Null means
    /// no scaling — every non-Gemma architecture leaves this null and is unaffected.
    /// The value is stored explicitly (rather than derived from the architecture)
    /// so the forward path stays a single multiply with no per-architecture branch.
    /// </summary>
    public float? EmbeddingScale { get; init; }

    /// <summary>Sliding window size for local attention. Null = full attention.</summary>
    public int? SlidingWindowSize { get; init; }

    /// <summary>
    /// Per-layer sliding-window override (length = <see cref="NumLayers"/>). Each
    /// entry is the per-layer window size, or <see langword="null"/> for a full-attention
    /// layer. Null at the model level means "no per-layer override — every layer uses
    /// <see cref="SlidingWindowSize"/>". Populated for Gemma 3's interleaved local/global
    /// attention pattern (<c>sliding_window_pattern</c>); ignored for every architecture
    /// where every layer behaves the same.
    /// </summary>
    public IReadOnlyList<int?>? PerLayerSlidingWindow { get; init; }

    /// <summary>
    /// Optional attention-logit soft-cap (Gemma 2 / Gemma 3 <c>attn_logit_softcapping</c>).
    /// When non-null, attention scores <c>z</c> are transformed in-place between scaling
    /// and softmax as <c>z' = tanh(z / cap) * cap</c>. Gemma 2 sets this to 50.0;
    /// Gemma 3 leaves it null but the field is wired regardless.
    /// </summary>
    public float? AttnLogitSoftcap { get; init; }

    /// <summary>
    /// Optional final-logit soft-cap (Gemma 2 / Gemma 3 <c>final_logit_softcapping</c>).
    /// When non-null, the LM-head logits <c>z</c> are transformed as
    /// <c>z' = tanh(z / cap) * cap</c> after the LM-head projection and before sampling.
    /// Gemma 2 sets this to 30.0; Gemma 3 leaves it null but the field is wired
    /// regardless.
    /// </summary>
    public float? FinalLogitSoftcap { get; init; }

    /// <summary>
    /// Optional attention-score scale multiplier override (Gemma's
    /// <c>query_pre_attn_scalar</c>). When non-null the kernel uses
    /// <c>1 / sqrt(query_pre_attn_scalar)</c> instead of the default
    /// <c>1 / sqrt(<see cref="HeadDim"/>)</c>. Gemma 3 ships this as 256 (matching the
    /// pre-attn-scalar value used when training the 2.6B/9B/27B SKUs).
    /// </summary>
    public float? QueryPreAttnScalar { get; init; }

    /// <summary>
    /// Interleaved sliding-window pattern period. 0 (default) = the sliding
    /// window (when set) applies to every layer (Mistral convention). N &gt; 0 =
    /// layer <c>il</c> uses the sliding window iff <c>il % N &lt; N - 1</c>
    /// (llama.cpp <c>set_swa_pattern(N, dense_first=false)</c>); the remaining
    /// layers use dense full-context attention. gpt-oss uses N=2 (even layers
    /// windowed, odd layers dense).
    /// </summary>
    public int SlidingWindowPattern { get; init; }

    /// <summary>MLA configuration. Only set for DeepSeek-style MLA attention.</summary>
    public MlaConfig? MlaConfig { get; init; }

    /// <summary>
    /// Per-layer sub-layer layout for hybrid SSM+Transformer models (e.g. Nemotron-H).
    /// Null for pure-Transformer architectures.
    /// </summary>
    public HybridLayerLayout? HybridLayout { get; init; }

    /// <summary>
    /// Mamba2 SSM configuration shared by all SSM layers in a hybrid model.
    /// Null when the model has no SSM layers.
    /// </summary>
    public MambaSsmConfig? SsmConfig { get; init; }

    /// <summary>
    /// Mamba-3 architecture configuration. Non-null iff <see cref="Architecture"/>
    /// is <see cref="DotLLM.Core.Configuration.Architecture.Mamba3"/>.
    /// Carries the Mamba-3 specific hyperparameters (state size, head count,
    /// MIMO flag, trapezoidal/RoPE floors, …) that do not map onto the
    /// Mamba-2 <see cref="SsmConfig"/>.
    /// </summary>
    public Mamba3Config? Mamba3Config { get; init; }

    /// <summary>
    /// Mixture-of-Experts configuration. Non-null when the per-layer FFN is
    /// replaced by top-k dense routing over <see cref="MoeConfig.NumExperts"/>
    /// experts. Present on <see cref="DotLLM.Core.Configuration.Architecture.Mixtral"/>
    /// today; extensible to Qwen*-MoE and Phi-3.5-MoE via the same record.
    /// </summary>
    public MoeConfig? Moe { get; init; }

    /// <summary>
    /// Gated DeltaNet (GDN) SSM configuration. Non-null iff
    /// <see cref="Architecture"/> is
    /// <see cref="DotLLM.Core.Configuration.Architecture.Qwen3MoeHybrid"/>.
    /// Carries GDN-specific parameters (state size, head counts, attention
    /// interval) that do not map onto the Mamba-2 <see cref="SsmConfig"/>.
    /// </summary>
    public GatedDeltaNetConfig? GdnConfig { get; init; }

    /// <summary>
    /// Diffusion-decoding configuration for masked-canvas text-diffusion models
    /// (DiffusionGemma). Non-null only for a diffusion checkpoint; null for every
    /// autoregressive architecture — those models are unaffected by this slot.
    /// Carries the canvas length, denoising-step budget, entropy/temperature
    /// schedule, and the tokenizer-resolved mask token id.
    /// </summary>
    public DiffusionConfig? DiffusionConfig { get; init; }

    /// <summary>
    /// Per-Layer Embeddings (PLE) configuration for the Gemma-4 dense text tower
    /// (<c>gemma4_text</c>, e.g. E2B/E4B). Non-null only when the checkpoint ships the
    /// PLE tables (<c>embed_tokens_per_layer</c> + <c>per_layer_model_projection</c> +
    /// the per-layer gate/projection/norm). When set, the forward pass computes the
    /// per-layer input tensor once after the embedding lookup and injects a gated
    /// residual into every decoder layer's output. Null for every other architecture
    /// (including the Gemma-4 MoE backbone and Gemma 3) — those paths are unaffected.
    /// </summary>
    public PerLayerEmbeddingConfig? PerLayerEmbedding { get; init; }

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
    /// True when this configuration targets a Gemma-family architecture. Gemma
    /// requires the four-RMSNorm-per-layer residual layout, the <c>(1+w)</c>
    /// RMSNorm weight convention, GeGLU FFN activation, and <c>sqrt(hidden)</c>
    /// embedding scaling — all gated behind this single predicate so every other
    /// architecture keeps the standard two-norm SwiGLU path untouched.
    /// </summary>
    public bool IsGemmaArchitecture =>
        Architecture is DotLLM.Core.Configuration.Architecture.Gemma3
                     or DotLLM.Core.Configuration.Architecture.Gemma4
                     or DotLLM.Core.Configuration.Architecture.DiffusionGemma;

    /// <summary>
    /// Optional per-attention-type RoPE override for the FULL-attention layers
    /// (Gemma 4 / DiffusionGemma). When non-null, every layer flagged as a
    /// full-attention layer by <see cref="IsFullAttentionLayer(int)"/> applies
    /// this RoPE configuration instead of <see cref="RoPEConfig"/>; sliding-window
    /// layers always use <see cref="RoPEConfig"/>. Gemma 4 ships full layers with a
    /// larger base (<c>rope_theta = 1e6</c>) and a partial-rotary factor (see
    /// <see cref="PartialRotaryFactor"/>) while the sliding layers keep the local
    /// base (<c>rope_theta = 1e4</c>). Null for every architecture that uses a
    /// single RoPE configuration across all layers (including Gemma 3) — the
    /// per-layer dispatch then collapses to the single <see cref="RoPEConfig"/>.
    /// </summary>
    public RoPEConfig? GlobalRoPEConfig { get; init; }

    /// <summary>
    /// Optional partial-rotary factor applied to the FULL-attention layers
    /// (Gemma 4 <c>partial_rotary_factor</c>, e.g. 0.25). When non-null, only the
    /// leading <c>round(PartialRotaryFactor * head_dim)</c> (rounded down to an
    /// even count) dimensions of each head are rotated by the full-attention RoPE;
    /// the remaining dimensions pass through unchanged. Mirrors HF's
    /// <c>partial_rotary_factor</c> on the global attention type. Null means full
    /// rotation (factor 1.0) — the default for every architecture and for Gemma 4's
    /// sliding-window layers.
    /// </summary>
    public float? PartialRotaryFactor { get; init; }

    /// <summary>
    /// Optional KV-head count for the FULL-attention layers (Gemma 4
    /// <c>num_global_key_value_heads</c>, e.g. 2). When non-null, full-attention
    /// layers use this GQA group size instead of <see cref="NumKvHeads"/>;
    /// sliding-window layers always use <see cref="NumKvHeads"/>
    /// (Gemma 4 <c>num_key_value_heads</c>, e.g. 8). Null means a single uniform
    /// KV-head count across all layer types — the default for every architecture
    /// other than Gemma 4.
    /// </summary>
    public int? NumGlobalKvHeads { get; init; }

    /// <summary>
    /// True for the Gemma-4 MoE backbone (and the DiffusionGemma tower that
    /// reuses it), which the forward pass treats with the dedicated Gemma-4 graph:
    /// V projected from the raw K projection on V-less (global) layers, a
    /// weight-less RMSNorm on V, attention softmax scale 1.0, a dual <i>parallel</i>
    /// FFN (a dense GeGLU MLP summed with a 128-expert MoE), a custom router
    /// (<c>rms(attn_out)·1/sqrt(hidden)·ffn_gate_inp_s</c> then <c>ffn_gate_inp·…</c>),
    /// a per-expert down-projection scale, and a per-layer <c>layer_output_scale</c>.
    /// All other Gemma-family configs (Gemma 3) leave this false. Gated as a flag
    /// (not derived from <see cref="Architecture"/>) so the dense-Gemma path stays
    /// untouched and the seam is explicit. See
    /// <c>docs/diffusiongemma/GEMMA4-GRAPH-SPEC.md</c>.
    /// </summary>
    public bool Gemma4DualFfn { get; init; }

    /// <summary>
    /// Optional per-head dimension for the FULL-attention layers (Gemma 4
    /// <c>global_head_dim</c>, e.g. 512 vs the sliding <c>head_dim</c> 256). When
    /// non-null and different from <see cref="HeadDim"/>, the full-attention layers
    /// project Q/K/V at a different per-head width than the sliding layers; the CPU
    /// forward pass resolves the layer head dim per layer via
    /// <see cref="GetLayerHeadDim(int)"/> and sizes its scratch buffers for the
    /// larger of the two. Null, or equal to <see cref="HeadDim"/>, means a uniform
    /// head dimension across all layer types — every non-Gemma-4 architecture
    /// leaves this null and is unaffected.
    /// </summary>
    public int? GlobalHeadDim { get; init; }

    /// <summary>
    /// Returns the per-head dimension for <paramref name="layerIdx"/>. Full-attention
    /// layers use <see cref="GlobalHeadDim"/> when set (Gemma 4 <c>global_head_dim</c>);
    /// sliding-window layers and every other architecture use <see cref="HeadDim"/>.
    /// Collapses to a uniform <see cref="HeadDim"/> when <see cref="GlobalHeadDim"/>
    /// is null, so non-Gemma-4 forwards are unaffected.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    public int GetLayerHeadDim(int layerIdx)
    {
        if (GlobalHeadDim is int g && IsFullAttentionLayer(layerIdx))
            return g;
        return HeadDim;
    }

    /// <summary>
    /// Returns the KV-head count for <paramref name="layerIdx"/>. Full-attention
    /// layers use <see cref="NumGlobalKvHeads"/> when set (Gemma 4); sliding-window
    /// layers and every other architecture use <see cref="NumKvHeads"/>. Collapses
    /// to a uniform <see cref="NumKvHeads"/> when <see cref="NumGlobalKvHeads"/> is
    /// null, so non-Gemma-4 models are unaffected. This is the single source of
    /// truth for per-layer KV-head resolution (backends delegate here rather than
    /// re-deriving it) and feeds <see cref="DotLLM.Core.Attention.KvGeometry.FromConfig"/>.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    public int GetLayerKvHeads(int layerIdx)
    {
        if (NumGlobalKvHeads is int g && IsFullAttentionLayer(layerIdx))
            return g;
        return NumKvHeads;
    }

    /// <summary>
    /// Returns true when <paramref name="layerIdx"/> is a FULL-attention layer.
    /// A layer is full-attention when <see cref="PerLayerSlidingWindow"/> is set
    /// and the entry for that layer is <see langword="null"/> (no sliding window);
    /// it is a sliding-window layer when the entry is a positive window size. When
    /// <see cref="PerLayerSlidingWindow"/> is null the model has no per-layer
    /// attention-type distinction — every layer is treated as full-attention
    /// (returns true), so the per-attention-type RoPE / KV-head overrides collapse
    /// to the model-wide defaults.
    /// </summary>
    public bool IsFullAttentionLayer(int layerIdx)
    {
        var perLayer = PerLayerSlidingWindow;
        if (perLayer is null || (uint)layerIdx >= (uint)perLayer.Count)
            return true;
        return perLayer[layerIdx] is null;
    }

    /// <summary>
    /// Number of trailing layers that REUSE an earlier layer's KV instead of
    /// projecting/caching their own (Gemma-4 E2B/E4B
    /// <c>attention.shared_kv_layers</c>; llama.cpp <c>n_layer_kv_from_start =
    /// n_layer - shared_kv_layers</c>). Zero (the default) means every layer has
    /// its own KV — all non-Gemma-4-PLE architectures are unaffected. When
    /// positive, layers <c>[NumLayers - NumSharedKvLayers, NumLayers)</c> skip
    /// their K/V projections entirely and attend over the KV of
    /// <see cref="SharedKvDonorLayer(int)"/> (the last own-KV layer of the same
    /// attention type: sliding layers borrow the last own-KV sliding layer,
    /// full-attention layers the last own-KV full layer).
    /// </summary>
    public int NumSharedKvLayers { get; init; }

    /// <summary>
    /// Returns true when <paramref name="layerIdx"/> projects and stores its own
    /// K/V. False only for the trailing <see cref="NumSharedKvLayers"/> shared-KV
    /// layers (Gemma-4 E2B/E4B); always true when sharing is off.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    public bool LayerHasOwnKv(int layerIdx)
        => NumSharedKvLayers <= 0 || layerIdx < NumLayers - NumSharedKvLayers;

    /// <summary>
    /// Returns the donor layer whose KV a shared-KV layer reuses. Mirrors
    /// llama.cpp's gemma3n/gemma4 reuse rule
    /// (<c>n_layer_kv_from_start - (is_swa(il) ? 2 : 1)</c>): with
    /// <c>kvFromStart = NumLayers - NumSharedKvLayers</c>, a sliding-window shared
    /// layer borrows layer <c>kvFromStart - 2</c> (the last own-KV sliding layer)
    /// and a full-attention shared layer borrows <c>kvFromStart - 1</c> (the last
    /// own-KV full layer). Only meaningful when <see cref="LayerHasOwnKv(int)"/>
    /// is false for <paramref name="layerIdx"/>.
    /// </summary>
    public int SharedKvDonorLayer(int layerIdx)
    {
        int kvFromStart = NumLayers - NumSharedKvLayers;
        return kvFromStart - (IsFullAttentionLayer(layerIdx) ? 1 : 2);
    }

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
