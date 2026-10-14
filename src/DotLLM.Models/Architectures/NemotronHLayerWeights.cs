using DotLLM.Core.Configuration;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Per-layer weight references for the three possible Nemotron-H sub-layer kinds.
/// Exactly one of <see cref="Ssm"/>, <see cref="Attention"/>, <see cref="Ffn"/>
/// is non-null; the others are null and their memory is not allocated.
/// </summary>
/// <remarks>
/// All linear-projection weights are kept as raw mmap pointers plus their quantization
/// type (matching <see cref="TransformerLayerWeights"/> conventions). Norm weights are
/// dequantized to <c>float[]</c> at load time because they are small and hot.
/// </remarks>
internal sealed class NemotronHLayerWeights
{
    /// <summary>Pre-sublayer RMSNorm weight [hiddenSize]. Always present.</summary>
    public required float[] AttnNormWeight { get; init; }

    /// <summary>SSM weights, non-null only for SSM layers.</summary>
    public NemotronHSsmWeights? Ssm { get; init; }

    /// <summary>Attention weights, non-null only for attention layers.</summary>
    public NemotronHAttentionWeights? Attention { get; init; }

    /// <summary>FFN weights, non-null only for DENSE FFN layers.</summary>
    public NemotronHFfnWeights? Ffn { get; init; }

    /// <summary>Routed-MoE weights, non-null only for MoE FFN layers (nemotron_h_moe, #375).
    /// A layer classified <c>HybridLayerKind.Ffn</c> carries exactly one of
    /// <c>Ffn</c>/<c>Moe</c>.</summary>
    public NemotronHMoeWeights? Moe { get; init; }
}

/// <summary>
/// nemotron_h_moe routed-MoE per-layer weights (issue #375). Anchored to llama.cpp's
/// <c>build_ffn_layer</c> MoE branch in <c>src/models/nemotron-h.cpp</c>: sigmoid router
/// with a per-expert selection bias, expert-weight renorm + scale, UNGATED squared-ReLU
/// experts stored as fused per-expert banks, plus one shared relu² expert added to the
/// routed sum. Expert banks keep the GGUF byte layout: expert e's matrix starts at
/// <c>bank + e * perExpertBytes</c> with rows contiguous, identical to a dense tensor.
/// </summary>
internal sealed class NemotronHMoeWeights
{
    /// <summary>ffn_gate_inp.weight — router [hiddenSize, numExperts]. Runs on the FULL hidden input.</summary>
    public required nint GateInpWeight { get; init; }
    public required QuantizationType GateInpQuantType { get; init; }

    /// <summary>exp_probs_b.bias [numExperts], F32 — added to sigmoid probabilities for
    /// top-k SELECTION ONLY; gating weights come from the unbiased probabilities.</summary>
    public required float[] SelectionBias { get; init; }

    /// <summary>ffn_up_exps.weight fused bank [hiddenSize, moeIntermediate, numExperts].</summary>
    public required nint UpExpsWeight { get; init; }
    public required QuantizationType UpExpsQuantType { get; init; }
    /// <summary>Byte stride between consecutive experts' up matrices in the bank.</summary>
    public required long UpPerExpertBytes { get; init; }

    /// <summary>ffn_down_exps.weight fused bank [moeIntermediate, hiddenSize, numExperts].</summary>
    public required nint DownExpsWeight { get; init; }
    public required QuantizationType DownExpsQuantType { get; init; }
    public required long DownPerExpertBytes { get; init; }

    /// <summary>ffn_up_shexp.weight [hiddenSize, sharedIntermediate] — shared expert up.</summary>
    public required nint UpShexpWeight { get; init; }
    public required QuantizationType UpShexpQuantType { get; init; }

    /// <summary>ffn_down_shexp.weight [sharedIntermediate, hiddenSize] — shared expert down.</summary>
    public required nint DownShexpWeight { get; init; }
    public required QuantizationType DownShexpQuantType { get; init; }

    public required int NumExperts { get; init; }
    public required int NumExpertsPerTok { get; init; }
    public required int MoeIntermediateSize { get; init; }
    public required int SharedIntermediateSize { get; init; }
    /// <summary>llama.cpp <c>expert_weights_norm</c> — renormalise gathered weights by their sum.</summary>
    public required bool NormalizeWeights { get; init; }
    /// <summary>llama.cpp <c>expert_weights_scale</c> — 2.5 in the shipping checkpoint.</summary>
    public required float WeightsScale { get; init; }
}

/// <summary>
/// Mamba2 SSM per-layer weights. Shapes follow the llama.cpp / GGUF convention:
/// GGUF stores tensors as [input_dim, output_dim] row-major.
/// </summary>
internal sealed class NemotronHSsmWeights
{
    /// <summary>ssm_in.weight — input projection [hiddenSize, d_in_proj]. Quantized.</summary>
    public required nint InWeight { get; init; }
    public required QuantizationType InQuantType { get; init; }
    public required int InInputDim { get; init; }
    public required int InOutputDim { get; init; }

    /// <summary>ssm_conv1d.weight [d_conv, conv_dim]. F32.</summary>
    public required float[] Conv1dWeight { get; init; }

    /// <summary>ssm_conv1d.bias [conv_dim]. F32.</summary>
    public required float[] Conv1dBias { get; init; }

    /// <summary>ssm_a [1, n_head] — scalar A per head (Mamba2). F32.</summary>
    public required float[] A { get; init; }

    /// <summary>ssm_d [1, n_head] — scalar D per head (skip connection). F32.</summary>
    public required float[] D { get; init; }

    /// <summary>ssm_dt.bias [n_head] — bias added to dt before softplus. F32.</summary>
    public required float[] DtBias { get; init; }

    /// <summary>ssm_norm.weight [d_inner / n_group, n_group] — group RMSNorm gains. F32.</summary>
    public required float[] NormWeight { get; init; }

    /// <summary>ssm_out.weight [d_inner, hiddenSize]. Quantized.</summary>
    public required nint OutWeight { get; init; }
    public required QuantizationType OutQuantType { get; init; }
    public required int OutInputDim { get; init; }
    public required int OutOutputDim { get; init; }
}

/// <summary>Standard GQA attention per-layer weights (no bias on any projection for Nemotron-H).</summary>
internal sealed class NemotronHAttentionWeights
{
    public required nint QWeight { get; init; }
    public required QuantizationType QQuantType { get; init; }
    public required int QInputDim { get; init; }
    public required int QOutputDim { get; init; }

    public required nint KWeight { get; init; }
    public required QuantizationType KQuantType { get; init; }
    public required int KInputDim { get; init; }
    public required int KOutputDim { get; init; }

    public required nint VWeight { get; init; }
    public required QuantizationType VQuantType { get; init; }
    public required int VInputDim { get; init; }
    public required int VOutputDim { get; init; }

    public required nint OWeight { get; init; }
    public required QuantizationType OQuantType { get; init; }
    public required int OInputDim { get; init; }
    public required int OOutputDim { get; init; }

    /// <summary>Per-layer KV-head count from the hybrid layout (may differ from model-level default).</summary>
    public required int NumKvHeads { get; init; }
}

/// <summary>Squared-ReLU parallel MLP — no gate, just up -> ReLU^2 -> down.</summary>
internal sealed class NemotronHFfnWeights
{
    public required nint UpWeight { get; init; }
    public required QuantizationType UpQuantType { get; init; }
    public required int UpInputDim { get; init; }
    public required int UpOutputDim { get; init; }

    public required nint DownWeight { get; init; }
    public required QuantizationType DownQuantType { get; init; }
    public required int DownInputDim { get; init; }
    public required int DownOutputDim { get; init; }

    /// <summary>Per-layer intermediate dim from the hybrid layout.</summary>
    public required int IntermediateSize { get; init; }
}
