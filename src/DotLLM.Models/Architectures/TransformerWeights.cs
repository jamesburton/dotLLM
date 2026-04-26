using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Per-layer dense-routing MoE weight bundle. Present on a
/// <see cref="TransformerLayerWeights"/> when the layer replaces its FFN
/// with a Mixtral-convention or Qwen-MoE-convention MoE block. All pointers
/// are F32 row-major — bf16 and F16 tensors are upcast at load time so the
/// MoE kernel can feed <see cref="DotLLM.Cpu.Kernels.MoeSwiGluMlp"/>
/// directly without per-call dequant.
/// </summary>
/// <remarks>
/// <para>
/// Qwen-MoE and DeepSeek-V2/V3 add optional shared-expert pointers — each
/// carried as parallel arrays (<see cref="SharedGateProj"/>, <see cref="SharedUpProj"/>,
/// <see cref="SharedDownProj"/>) of length <see cref="NumSharedExperts"/>.
/// Qwen1.5-MoE ships a single shared expert optionally gated by a
/// <see cref="SharedExpertGate"/> sigmoid; DeepSeek-V2/V3 ships
/// <c>n_shared_experts</c> shared experts (often 1 or 2) and does not gate.
/// When <see cref="HasSharedExpert"/> is true, the forward pass runs each
/// shared expert as a dense SwiGLU over the token, sums their outputs, and
/// adds the (optionally gated) sum to the routed top-k sum. The
/// <see cref="NormTopKProb"/> flag controls whether the selected top-k
/// probabilities are renormalised to sum to 1.0 (Mixtral + Qwen3-MoE) or
/// left as raw softmax values (Qwen1.5-MoE-A2.7B).
/// </para>
/// </remarks>
internal sealed class MoeLayerWeights
{
    /// <summary>Router gate.weight as F32 [numExperts, hiddenSize] row-major.</summary>
    public readonly float[] Gate;

    /// <summary>Per-expert <c>w1</c> (gate_proj) F32 pointers [intermediateSize, hiddenSize] row-major.</summary>
    public readonly nint[] W1;

    /// <summary>Per-expert <c>w2</c> (down_proj) F32 pointers [hiddenSize, intermediateSize] row-major.</summary>
    public readonly nint[] W2;

    /// <summary>Per-expert <c>w3</c> (up_proj) F32 pointers [intermediateSize, hiddenSize] row-major.</summary>
    public readonly nint[] W3;

    public readonly int NumExperts;
    public readonly int NumExpertsPerTok;
    public readonly int HiddenSize;
    public readonly int IntermediateSize;

    /// <summary>
    /// When <c>true</c>, the kernel renormalises the selected top-k
    /// probabilities to sum to 1.0 (Mixtral + Qwen3-MoE). When <c>false</c>,
    /// the raw softmax probabilities are used as gating weights (Qwen1.5-MoE).
    /// </summary>
    public readonly bool NormTopKProb;

    /// <summary>
    /// Per-shared-expert <c>gate_proj</c> pointers — F32
    /// [sharedIntermediateSize, hiddenSize] row-major, one per shared expert.
    /// Length equals <see cref="NumSharedExperts"/>; empty when no shared
    /// experts are present.
    /// </summary>
    public readonly nint[] SharedGateProj;
    /// <summary>
    /// Per-shared-expert <c>up_proj</c> pointers — F32
    /// [sharedIntermediateSize, hiddenSize] row-major, one per shared expert.
    /// </summary>
    public readonly nint[] SharedUpProj;
    /// <summary>
    /// Per-shared-expert <c>down_proj</c> pointers — F32
    /// [hiddenSize, sharedIntermediateSize] row-major, one per shared expert.
    /// </summary>
    public readonly nint[] SharedDownProj;
    /// <summary>
    /// Per-shared-expert intermediate width (0 when no shared expert).
    /// Applies uniformly across all shared experts (they share width).
    /// </summary>
    public readonly int SharedIntermediateSize;
    /// <summary>
    /// Number of parallel shared experts whose outputs are summed. 1 for
    /// Qwen1.5-MoE, &gt;=1 for DeepSeek-V2/V3 (<c>n_shared_experts</c>).
    /// Zero only when there is no shared-expert branch.
    /// </summary>
    public readonly int NumSharedExperts;
    /// <summary>
    /// Optional shared-expert sigmoid gate weight — F32 [hiddenSize]. When
    /// present, per-token <c>sigmoid(hidden . SharedExpertGate)</c> scales
    /// the summed shared-expert output before it's added to the routed sum
    /// (Qwen1.5-MoE convention; ALWAYS paired with a single shared expert).
    /// Null = no gate, summed shared-expert output added unscaled
    /// (DeepSeek-V2/V3 convention).
    /// </summary>
    public readonly float[]? SharedExpertGate;

    /// <summary>True iff a shared-expert branch is present on this layer.</summary>
    public bool HasSharedExpert => SharedIntermediateSize > 0 && NumSharedExperts > 0;

    /// <summary>
    /// Raw GGUF mmap base pointer of the fused-experts <c>ffn_gate_exps</c>
    /// tensor, populated alongside the F32 dequants <see cref="W1"/> when the
    /// source is a GGUF-quantized DeepSeek-V2/V3 checkpoint. The CUDA loader
    /// consumes these (zero-copy upload to GPU per-expert slice, on-device
    /// dequant) instead of the F32 host inflation. Zero when the source is
    /// non-GGUF (e.g. safetensors) — in which case only <see cref="W1"/> is
    /// populated. Per-expert byte offset into the raw view is
    /// <c>e * (M * RowByteSize(K, qt))</c> where M = <see cref="GateExpsMDim"/>
    /// and K = <see cref="GateExpsKDim"/>.
    /// </summary>
    public readonly nint GateExpsRaw;
    /// <summary>Quant type of <see cref="GateExpsRaw"/>; <c>F32</c> when raw view absent.</summary>
    public readonly QuantizationType GateExpsRawQt;
    /// <summary>Output dim (M) of the per-expert <c>ffn_gate_exps</c> slice (= moe_intermediate_size).</summary>
    public readonly int GateExpsMDim;
    /// <summary>Input dim (K) of the per-expert <c>ffn_gate_exps</c> slice (= hidden_size).</summary>
    public readonly int GateExpsKDim;

    /// <summary>Raw GGUF mmap base pointer of <c>ffn_up_exps</c>. See <see cref="GateExpsRaw"/>.</summary>
    public readonly nint UpExpsRaw;
    public readonly QuantizationType UpExpsRawQt;
    public readonly int UpExpsMDim;
    public readonly int UpExpsKDim;

    /// <summary>Raw GGUF mmap base pointer of <c>ffn_down_exps</c>. See <see cref="GateExpsRaw"/>.</summary>
    /// <remarks>For down_exps the M/K dims are swapped: M = hidden_size, K = moe_intermediate_size.</remarks>
    public readonly nint DownExpsRaw;
    public readonly QuantizationType DownExpsRawQt;
    public readonly int DownExpsMDim;
    public readonly int DownExpsKDim;

    /// <summary>Raw GGUF mmap pointers for shared experts (parallel to <see cref="SharedGateProj"/>). 0 when raw view absent.</summary>
    public readonly nint[] SharedGateRaw;
    public readonly QuantizationType SharedGateRawQt;
    public readonly nint[] SharedUpRaw;
    public readonly QuantizationType SharedUpRawQt;
    public readonly nint[] SharedDownRaw;
    public readonly QuantizationType SharedDownRawQt;

    /// <summary>
    /// True when the routed-expert raw quant views are populated — the CUDA
    /// loader can take the on-device dequant fast path. False on safetensors
    /// loads where only the F32 dequants are populated.
    /// </summary>
    public bool HasRawQuantView => GateExpsRaw != 0 && UpExpsRaw != 0 && DownExpsRaw != 0;

    /// <summary>Mixtral-convention ctor (no shared expert, always renormalise top-k).</summary>
    public MoeLayerWeights(
        float[] gate,
        nint[] w1, nint[] w2, nint[] w3,
        int numExperts, int numExpertsPerTok, int hiddenSize, int intermediateSize)
        : this(gate, w1, w2, w3, numExperts, numExpertsPerTok, hiddenSize, intermediateSize,
               normTopKProb: true,
               sharedGateProj: Array.Empty<nint>(),
               sharedUpProj: Array.Empty<nint>(),
               sharedDownProj: Array.Empty<nint>(),
               sharedIntermediateSize: 0,
               sharedExpertGate: null)
    {
    }

    /// <summary>
    /// Full ctor covering Qwen-MoE and DeepSeek extensions: per-shared-expert
    /// pointer arrays, <c>norm_topk_prob</c> flag, optional sigmoid gate.
    /// Length of the three shared arrays must agree; a zero-length array set
    /// disables the shared-expert branch. Raw quant views default to absent.
    /// </summary>
    public MoeLayerWeights(
        float[] gate,
        nint[] w1, nint[] w2, nint[] w3,
        int numExperts, int numExpertsPerTok, int hiddenSize, int intermediateSize,
        bool normTopKProb,
        nint[] sharedGateProj, nint[] sharedUpProj, nint[] sharedDownProj,
        int sharedIntermediateSize, float[]? sharedExpertGate)
        : this(gate, w1, w2, w3, numExperts, numExpertsPerTok, hiddenSize, intermediateSize,
               normTopKProb,
               sharedGateProj, sharedUpProj, sharedDownProj,
               sharedIntermediateSize, sharedExpertGate,
               gateExpsRaw: 0, gateExpsRawQt: QuantizationType.F32,
               gateExpsMDim: 0, gateExpsKDim: 0,
               upExpsRaw: 0, upExpsRawQt: QuantizationType.F32,
               upExpsMDim: 0, upExpsKDim: 0,
               downExpsRaw: 0, downExpsRawQt: QuantizationType.F32,
               downExpsMDim: 0, downExpsKDim: 0,
               sharedGateRaw: Array.Empty<nint>(), sharedGateRawQt: QuantizationType.F32,
               sharedUpRaw: Array.Empty<nint>(), sharedUpRawQt: QuantizationType.F32,
               sharedDownRaw: Array.Empty<nint>(), sharedDownRawQt: QuantizationType.F32)
    {
    }

    /// <summary>
    /// Full ctor including raw GGUF mmap views for the routed-expert and
    /// shared-expert tensors. Used by the GGUF MoE loader so the CUDA backend
    /// can upload raw quantized bytes per expert (avoiding the ~57 GB host
    /// F32 inflation at V2-Lite scale).
    /// </summary>
    public MoeLayerWeights(
        float[] gate,
        nint[] w1, nint[] w2, nint[] w3,
        int numExperts, int numExpertsPerTok, int hiddenSize, int intermediateSize,
        bool normTopKProb,
        nint[] sharedGateProj, nint[] sharedUpProj, nint[] sharedDownProj,
        int sharedIntermediateSize, float[]? sharedExpertGate,
        nint gateExpsRaw, QuantizationType gateExpsRawQt, int gateExpsMDim, int gateExpsKDim,
        nint upExpsRaw, QuantizationType upExpsRawQt, int upExpsMDim, int upExpsKDim,
        nint downExpsRaw, QuantizationType downExpsRawQt, int downExpsMDim, int downExpsKDim,
        nint[] sharedGateRaw, QuantizationType sharedGateRawQt,
        nint[] sharedUpRaw, QuantizationType sharedUpRawQt,
        nint[] sharedDownRaw, QuantizationType sharedDownRawQt)
    {
        if (sharedGateProj.Length != sharedUpProj.Length || sharedGateProj.Length != sharedDownProj.Length)
            throw new ArgumentException(
                "Shared-expert pointer arrays must all have the same length (number of shared experts).");

        Gate = gate;
        W1 = w1; W2 = w2; W3 = w3;
        NumExperts = numExperts;
        NumExpertsPerTok = numExpertsPerTok;
        HiddenSize = hiddenSize;
        IntermediateSize = intermediateSize;
        NormTopKProb = normTopKProb;
        SharedGateProj = sharedGateProj;
        SharedUpProj = sharedUpProj;
        SharedDownProj = sharedDownProj;
        SharedIntermediateSize = sharedIntermediateSize;
        NumSharedExperts = sharedGateProj.Length;
        SharedExpertGate = sharedExpertGate;

        GateExpsRaw = gateExpsRaw; GateExpsRawQt = gateExpsRawQt;
        GateExpsMDim = gateExpsMDim; GateExpsKDim = gateExpsKDim;
        UpExpsRaw = upExpsRaw; UpExpsRawQt = upExpsRawQt;
        UpExpsMDim = upExpsMDim; UpExpsKDim = upExpsKDim;
        DownExpsRaw = downExpsRaw; DownExpsRawQt = downExpsRawQt;
        DownExpsMDim = downExpsMDim; DownExpsKDim = downExpsKDim;
        SharedGateRaw = sharedGateRaw; SharedGateRawQt = sharedGateRawQt;
        SharedUpRaw = sharedUpRaw; SharedUpRawQt = sharedUpRawQt;
        SharedDownRaw = sharedDownRaw; SharedDownRawQt = sharedDownRawQt;
    }
}

/// <summary>
/// Holds per-layer weight references for a single transformer layer.
/// Norm weights are dequantized to <c>float[]</c> at load time (small).
/// Linear projection weights remain as mmap pointers with their quantization type.
/// Bias arrays are nullable — null when the model has no biases (e.g. standard Llama/Mistral).
/// </summary>
internal readonly struct TransformerLayerWeights
{
    /// <summary>Pre-attention RMSNorm weight [hiddenSize].</summary>
    public readonly float[] AttnNormWeight;

    /// <summary>Optional QK-norm weight [headDim]. Applied per-head to Q after projection, before RoPE. Null when absent (e.g. Qwen2, Llama).</summary>
    public readonly float[]? QNormWeight;
    /// <summary>Optional QK-norm weight [headDim]. Applied per-head to K after projection, before RoPE. Null when absent.</summary>
    public readonly float[]? KNormWeight;

    /// <summary>Q projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint QWeight;
    public readonly QuantizationType QQuantType;
    public readonly int QOutputDim;
    public readonly int QInputDim;
    /// <summary>Optional Q projection bias [QOutputDim]. Null when absent.</summary>
    public readonly float[]? QBias;

    /// <summary>K projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint KWeight;
    public readonly QuantizationType KQuantType;
    public readonly int KOutputDim;
    public readonly int KInputDim;
    /// <summary>Optional K projection bias [KOutputDim]. Null when absent.</summary>
    public readonly float[]? KBias;

    /// <summary>V projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint VWeight;
    public readonly QuantizationType VQuantType;
    public readonly int VOutputDim;
    public readonly int VInputDim;
    /// <summary>Optional V projection bias [VOutputDim]. Null when absent.</summary>
    public readonly float[]? VBias;

    /// <summary>Output projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint OWeight;
    public readonly QuantizationType OQuantType;
    public readonly int OOutputDim;
    public readonly int OInputDim;
    /// <summary>Optional output projection bias [OOutputDim]. Null when absent.</summary>
    public readonly float[]? OBias;

    /// <summary>Pre-FFN RMSNorm weight [hiddenSize].</summary>
    public readonly float[] FfnNormWeight;

    /// <summary>SwiGLU gate projection.</summary>
    public readonly nint GateWeight;
    public readonly QuantizationType GateQuantType;
    public readonly int GateOutputDim;
    public readonly int GateInputDim;
    /// <summary>Optional gate projection bias [GateOutputDim]. Null when absent.</summary>
    public readonly float[]? GateBias;

    /// <summary>SwiGLU up projection.</summary>
    public readonly nint UpWeight;
    public readonly QuantizationType UpQuantType;
    public readonly int UpOutputDim;
    public readonly int UpInputDim;
    /// <summary>Optional up projection bias [UpOutputDim]. Null when absent.</summary>
    public readonly float[]? UpBias;

    /// <summary>Down projection.</summary>
    public readonly nint DownWeight;
    public readonly QuantizationType DownQuantType;
    public readonly int DownOutputDim;
    public readonly int DownInputDim;
    /// <summary>Optional down projection bias [DownOutputDim]. Null when absent.</summary>
    public readonly float[]? DownBias;

    /// <summary>
    /// MoE FFN bundle for Mixtral-convention layers. When non-null the dense
    /// <see cref="GateWeight"/>/<see cref="UpWeight"/>/<see cref="DownWeight"/>
    /// slots are ignored by the forward pass and MoE routing runs instead.
    /// </summary>
    public readonly MoeLayerWeights? Moe;

    // ──────────────────────────── MLA attention ────────────────────────────
    // DeepSeek-V2/V3 replaces the monolithic Q/K/V/O projections with a
    // low-rank-factorised set. When <see cref="Mla"/> is non-null, the
    // forward pass routes through MlaAttention and ignores the legacy
    // Q/K/V slots above (O is still used as the output projection).

    /// <summary>
    /// Non-null on DeepSeek-V2/V3 MLA layers. Carries all MLA-specific
    /// projection pointers + hyperparameters (qk nope/rope dims, v_head_dim,
    /// q/kv LoRA ranks). When present, <see cref="QWeight"/>/<see cref="KWeight"/>/
    /// <see cref="VWeight"/> are zeroed and the forward pass takes the MLA branch.
    /// </summary>
    public readonly MlaLayerWeights? Mla;

    public TransformerLayerWeights(
        float[] attnNormWeight,
        nint qWeight, QuantizationType qQuantType, int qOutputDim, int qInputDim,
        nint kWeight, QuantizationType kQuantType, int kOutputDim, int kInputDim,
        nint vWeight, QuantizationType vQuantType, int vOutputDim, int vInputDim,
        nint oWeight, QuantizationType oQuantType, int oOutputDim, int oInputDim,
        float[] ffnNormWeight,
        nint gateWeight, QuantizationType gateQuantType, int gateOutputDim, int gateInputDim,
        nint upWeight, QuantizationType upQuantType, int upOutputDim, int upInputDim,
        nint downWeight, QuantizationType downQuantType, int downOutputDim, int downInputDim,
        float[]? qBias = null, float[]? kBias = null, float[]? vBias = null, float[]? oBias = null,
        float[]? gateBias = null, float[]? upBias = null, float[]? downBias = null,
        float[]? qNormWeight = null, float[]? kNormWeight = null,
        MoeLayerWeights? moe = null,
        MlaLayerWeights? mla = null)
    {
        AttnNormWeight = attnNormWeight;
        QNormWeight = qNormWeight;
        KNormWeight = kNormWeight;
        QWeight = qWeight; QQuantType = qQuantType; QOutputDim = qOutputDim; QInputDim = qInputDim; QBias = qBias;
        KWeight = kWeight; KQuantType = kQuantType; KOutputDim = kOutputDim; KInputDim = kInputDim; KBias = kBias;
        VWeight = vWeight; VQuantType = vQuantType; VOutputDim = vOutputDim; VInputDim = vInputDim; VBias = vBias;
        OWeight = oWeight; OQuantType = oQuantType; OOutputDim = oOutputDim; OInputDim = oInputDim; OBias = oBias;
        FfnNormWeight = ffnNormWeight;
        GateWeight = gateWeight; GateQuantType = gateQuantType; GateOutputDim = gateOutputDim; GateInputDim = gateInputDim; GateBias = gateBias;
        UpWeight = upWeight; UpQuantType = upQuantType; UpOutputDim = upOutputDim; UpInputDim = upInputDim; UpBias = upBias;
        DownWeight = downWeight; DownQuantType = downQuantType; DownOutputDim = downOutputDim; DownInputDim = downInputDim; DownBias = downBias;
        Moe = moe;
        Mla = mla;
    }
}

/// <summary>
/// Per-layer MLA (Multi-head Latent Attention) weight bundle for DeepSeek-V2/V3.
/// All projection pointers are F32 row-major — F16 / BF16 tensors are upcast at
/// load time (via <c>ResolveLinearAsF32</c>) so the kernel can consume a uniform
/// F32 layout matching <see cref="DotLLM.Cpu.Kernels.MlaAttention.Execute"/>.
/// </summary>
/// <remarks>
/// <para>
/// Exactly one of the Q paths is populated:
/// <list type="bullet">
///   <item>LoRA-factored Q (<see cref="QLoraRank"/> &gt; 0): <see cref="QAProj"/>,
///     <see cref="QALayernormWeight"/>, <see cref="QBProj"/> are all non-zero;
///     <see cref="QProj"/> is zero.</item>
///   <item>Monolithic Q (<see cref="QLoraRank"/> == 0): <see cref="QProj"/> is
///     non-zero; <see cref="QAProj"/>, <see cref="QBProj"/> are zero and
///     <see cref="QALayernormWeight"/> is null.</item>
/// </list>
/// The KV path is always LoRA-factored (<see cref="KvAProjWithMqa"/>,
/// <see cref="KvALayernormWeight"/>, <see cref="KvBProj"/>).
/// </para>
/// </remarks>
internal sealed class MlaLayerWeights
{
    /// <summary>Q down-projection (F32) [qLoraRank, hidden]. Zero when <see cref="QLoraRank"/>==0.</summary>
    public readonly nint QAProj;
    /// <summary>Q LoRA RMSNorm weight [qLoraRank]. Null when <see cref="QLoraRank"/>==0.</summary>
    public readonly float[]? QALayernormWeight;
    /// <summary>Q up-projection (F32) [numHeads * qkHeadDim, qLoraRank]. Zero when <see cref="QLoraRank"/>==0.</summary>
    public readonly nint QBProj;
    /// <summary>Monolithic Q projection (F32) [numHeads * qkHeadDim, hidden]. Zero when <see cref="QLoraRank"/>&gt;0.</summary>
    public readonly nint QProj;

    /// <summary>KV down-projection with shared-rope-K (F32) [kvLoraRank + qkRopeHeadDim, hidden].</summary>
    public readonly nint KvAProjWithMqa;
    /// <summary>KV LoRA RMSNorm weight [kvLoraRank].</summary>
    public readonly float[] KvALayernormWeight;
    /// <summary>KV up-projection (F32) [numHeads * (qkNopeHeadDim + vHeadDim), kvLoraRank].</summary>
    public readonly nint KvBProj;

    /// <summary>
    /// Raw GGUF mmap views of the projection weights — populated alongside
    /// the F32 dequants when the source is GGUF-quantized. The GPU loader
    /// consumes these directly (zero-copy upload to GPU, on-device dequant)
    /// to avoid the F32 host inflation. Zero / F32 means the F32 dequant
    /// pointer above is the only view (e.g. safetensors source).
    /// </summary>
    public readonly nint QAProjRaw;
    public readonly QuantizationType QAProjRawQt;
    public readonly nint QBProjRaw;
    public readonly QuantizationType QBProjRawQt;
    public readonly nint QProjRaw;
    public readonly QuantizationType QProjRawQt;
    public readonly nint KvAProjWithMqaRaw;
    public readonly QuantizationType KvAProjWithMqaRawQt;
    public readonly nint KvBProjRaw;
    public readonly QuantizationType KvBProjRawQt;

    // Hyperparameters (mirrors MlaConfig, carried on the layer for forward-path convenience).
    public readonly int NumHeads;
    public readonly int QkNopeHeadDim;
    public readonly int QkRopeHeadDim;
    public readonly int VHeadDim;
    public readonly int QLoraRank;
    public readonly int KvLoraRank;
    public readonly int HiddenSize;

    /// <summary>Back-compat ctor — F32 dequants only, no raw quant view (safetensors path).</summary>
    public MlaLayerWeights(
        nint qAProj, float[]? qALayernormWeight, nint qBProj, nint qProj,
        nint kvAProjWithMqa, float[] kvALayernormWeight, nint kvBProj,
        int numHeads, int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
        int qLoraRank, int kvLoraRank,
        int hiddenSize = 0)
        : this(qAProj, qALayernormWeight, qBProj, qProj,
               kvAProjWithMqa, kvALayernormWeight, kvBProj,
               numHeads, qkNopeHeadDim, qkRopeHeadDim, vHeadDim,
               qLoraRank, kvLoraRank, hiddenSize,
               qAProjRaw: 0, qAProjRawQt: QuantizationType.F32,
               qBProjRaw: 0, qBProjRawQt: QuantizationType.F32,
               qProjRaw: 0, qProjRawQt: QuantizationType.F32,
               kvAProjWithMqaRaw: 0, kvAProjWithMqaRawQt: QuantizationType.F32,
               kvBProjRaw: 0, kvBProjRawQt: QuantizationType.F32)
    {
    }

    /// <summary>Full ctor with both F32 dequant views and raw GGUF quant views.</summary>
    public MlaLayerWeights(
        nint qAProj, float[]? qALayernormWeight, nint qBProj, nint qProj,
        nint kvAProjWithMqa, float[] kvALayernormWeight, nint kvBProj,
        int numHeads, int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
        int qLoraRank, int kvLoraRank, int hiddenSize,
        nint qAProjRaw, QuantizationType qAProjRawQt,
        nint qBProjRaw, QuantizationType qBProjRawQt,
        nint qProjRaw, QuantizationType qProjRawQt,
        nint kvAProjWithMqaRaw, QuantizationType kvAProjWithMqaRawQt,
        nint kvBProjRaw, QuantizationType kvBProjRawQt)
    {
        QAProj = qAProj;
        QALayernormWeight = qALayernormWeight;
        QBProj = qBProj;
        QProj = qProj;
        KvAProjWithMqa = kvAProjWithMqa;
        KvALayernormWeight = kvALayernormWeight;
        KvBProj = kvBProj;
        NumHeads = numHeads;
        QkNopeHeadDim = qkNopeHeadDim;
        QkRopeHeadDim = qkRopeHeadDim;
        VHeadDim = vHeadDim;
        QLoraRank = qLoraRank;
        KvLoraRank = kvLoraRank;
        HiddenSize = hiddenSize;
        QAProjRaw = qAProjRaw; QAProjRawQt = qAProjRawQt;
        QBProjRaw = qBProjRaw; QBProjRawQt = qBProjRawQt;
        QProjRaw = qProjRaw; QProjRawQt = qProjRawQt;
        KvAProjWithMqaRaw = kvAProjWithMqaRaw; KvAProjWithMqaRawQt = kvAProjWithMqaRawQt;
        KvBProjRaw = kvBProjRaw; KvBProjRawQt = kvBProjRawQt;
    }

    /// <summary>True when at least one raw quant view is non-trivial — the GPU
    /// loader can take the on-device dequant fast path. False on safetensors
    /// loads where everything is F32 and only the F32 dequants are populated.</summary>
    public bool HasRawQuantView => QAProjRaw != 0 || QBProjRaw != 0 || QProjRaw != 0;
}

/// <summary>
/// Holds R4-interleaved weight buffers for all projections in a single transformer layer.
/// Disposed when the parent <see cref="TransformerWeights"/> is disposed.
/// </summary>
internal sealed class RepackedLayerWeights : IDisposable
{
    public WeightRepacking.RepackedWeight Q, K, V, O, Gate, Up, Down;

    public void Dispose()
    {
        Q.Dispose(); K.Dispose(); V.Dispose(); O.Dispose();
        Gate.Dispose(); Up.Dispose(); Down.Dispose();
    }
}

/// <summary>
/// Organizes all weight tensor references from a loaded GGUF file for a transformer-family model.
/// Norm weights are dequantized to managed <c>float[]</c> at load time.
/// Linear projections remain as raw mmap pointers for zero-copy inference.
/// Optionally holds R4-interleaved weight buffers for improved cache locality in 4-row SIMD kernels.
/// </summary>
internal sealed class TransformerWeights : IDisposable
{
    /// <summary>Token embedding pointer and metadata.</summary>
    public nint TokenEmbedWeight { get; }
    public QuantizationType TokenEmbedQuantType { get; }
    public int VocabSize { get; }
    public int HiddenSize { get; }

    /// <summary>Per-layer weights.</summary>
    public TransformerLayerWeights[] Layers { get; }

    /// <summary>Final RMSNorm weight [hiddenSize].</summary>
    public float[] OutputNormWeight { get; }

    /// <summary>LM head (output projection) pointer and metadata.</summary>
    public nint OutputWeight { get; }
    public QuantizationType OutputQuantType { get; }
    public int OutputOutputDim { get; }
    public int OutputInputDim { get; }

    /// <summary>Per-layer R4-interleaved weights. Null until <see cref="RepackWeights"/> is called.</summary>
    public RepackedLayerWeights[]? RepackedLayers { get; private set; }

    /// <summary>R4-interleaved LM head weights. Null until <see cref="RepackWeights"/> is called or if type is not repackable.</summary>
    public WeightRepacking.RepackedWeight? RepackedOutput { get; private set; }

    /// <summary>
    /// Loader-owned 64-byte-aligned allocations created at load time (e.g.
    /// bf16 → F32 upcasts for the safetensors path). Freed by
    /// <see cref="Dispose"/>. Empty for pure-mmap GGUF loads.
    /// </summary>
    private readonly List<nint>? _ownedAllocations;

    private TransformerWeights(
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType, int vocabSize, int hiddenSize,
        TransformerLayerWeights[] layers,
        float[] outputNormWeight,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim,
        List<nint>? ownedAllocations = null)
    {
        TokenEmbedWeight = tokenEmbedWeight;
        TokenEmbedQuantType = tokenEmbedQuantType;
        VocabSize = vocabSize;
        HiddenSize = hiddenSize;
        Layers = layers;
        OutputNormWeight = outputNormWeight;
        OutputWeight = outputWeight;
        OutputQuantType = outputQuantType;
        OutputOutputDim = outputOutputDim;
        OutputInputDim = outputInputDim;
        _ownedAllocations = ownedAllocations;
    }

    /// <summary>
    /// Factory used by the safetensors loader. Wraps the private constructor
    /// and accepts the list of owned allocations (bf16→F32 upcast buffers)
    /// that must be freed when the weights are disposed.
    /// </summary>
    internal static TransformerWeights CreateFromSafetensors(
        nint tokenEmbedWeight, QuantizationType tokenEmbedQt, int vocabSize, int hiddenSize,
        TransformerLayerWeights[] layers,
        float[] outputNormWeight,
        nint outputWeight, QuantizationType outputQt, int outputM, int outputK,
        List<nint> ownedAllocations)
    {
        return new TransformerWeights(
            tokenEmbedWeight, tokenEmbedQt, vocabSize, hiddenSize,
            layers,
            outputNormWeight,
            outputWeight, outputQt, outputM, outputK,
            ownedAllocations);
    }

    /// <summary>
    /// Loads all weight references from an opened GGUF file.
    /// Norm weights are dequantized to <c>float[]</c>. Linear projections stay as mmap pointers.
    /// </summary>
    public static TransformerWeights LoadFromGguf(GgufFile gguf, ModelConfig config)
    {
        nint dataBase = gguf.DataBasePointer;
        var tensors = gguf.TensorsByName;

        // Token embeddings
        var embDesc = tensors["token_embd.weight"];
        nint embPtr = dataBase + (nint)embDesc.DataOffset;

        // MLA (DeepSeek-V2/V3) loads its projection tensors as F32 dequant
        // buffers since the CPU MlaAttention.Execute oracle is F32-only. Track
        // them on the loader so Dispose can free them. Empty for non-MLA models.
        var owned = config.MlaConfig is not null ? new List<nint>() : null;

        // Per-layer weights
        var layers = new TransformerLayerWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = config.MlaConfig is not null
                ? LoadMlaLayer(i, dataBase, tensors, config, owned!)
                : LoadLayer(i, dataBase, tensors, config);
        }

        // Output norm
        var outNormDesc = tensors["output_norm.weight"];
        float[] outputNormWeight = DequantizeNorm(dataBase, outNormDesc, config.HiddenSize);

        // LM head — may be tied to token embeddings
        nint outputPtr;
        QuantizationType outputQt;
        int outputM, outputK;

        if (tensors.TryGetValue("output.weight", out var outDesc))
        {
            outputPtr = dataBase + (nint)outDesc.DataOffset;
            outputQt = outDesc.QuantizationType;
            // GGUF: Dimensions[0] = input dim (K), Dimensions[1] = output dim (M)
            outputK = outDesc.Shape[0];
            outputM = outDesc.Shape[1];
        }
        else
        {
            // Tied embeddings: alias token_embd.weight
            outputPtr = embPtr;
            outputQt = embDesc.QuantizationType;
            outputK = embDesc.Shape[0];
            outputM = embDesc.Shape[1];
        }

        return new TransformerWeights(
            embPtr, embDesc.QuantizationType, config.VocabSize, config.HiddenSize,
            layers,
            outputNormWeight,
            outputPtr, outputQt, outputM, outputK,
            ownedAllocations: owned);
    }

    /// <summary>
    /// Repacks all linear projection weights into R4 interleaved layout for improved
    /// cache locality in 4-row SIMD kernels. Skips token embeddings (random row access)
    /// and non-block-structured types (F32, F16).
    /// </summary>
    public void RepackWeights()
    {
        var repacked = new RepackedLayerWeights[Layers.Length];
        for (int i = 0; i < Layers.Length; i++)
        {
            ref readonly var lw = ref Layers[i];
            // MoE layers don't populate the dense gate/up/down slots —
            // repack only the attention projections. The MoE FFN path runs
            // without R4 interleaving (the per-expert GEMMs are tiny and
            // the win would be microscopic).
            bool isMoe = lw.Moe is not null;
            // MLA layers don't populate the legacy Q/K/V slots either — the
            // MLA forward takes its weights from lw.Mla and calls the scalar
            // MlaAttention kernel which does not consume R4 repacks.
            bool isMla = lw.Mla is not null;
            repacked[i] = new RepackedLayerWeights
            {
                Q = isMla ? default : TryRepack(lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim),
                K = isMla ? default : TryRepack(lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim),
                V = isMla ? default : TryRepack(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim),
                O = isMla ? default : TryRepack(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim),
                Gate = isMoe ? default : TryRepack(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim),
                Up = isMoe ? default : TryRepack(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim),
                Down = isMoe ? default : TryRepack(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim),
            };
        }
        RepackedLayers = repacked;

        if (WeightRepacking.IsRepackable(OutputQuantType))
            RepackedOutput = WeightRepacking.RepackR4(OutputWeight, OutputQuantType, OutputOutputDim, OutputInputDim);
    }

    private static WeightRepacking.RepackedWeight TryRepack(nint ptr, QuantizationType qt, int m, int k)
    {
        if (!WeightRepacking.IsRepackable(qt))
            return default;
        return WeightRepacking.RepackR4(ptr, qt, m, k);
    }

    /// <summary>Frees all R4-interleaved weight buffers and any owned aligned allocations.</summary>
    public unsafe void Dispose()
    {
        if (RepackedLayers is not null)
        {
            foreach (var rl in RepackedLayers)
                rl.Dispose();
            RepackedLayers = null;
        }
        RepackedOutput?.Dispose();
        RepackedOutput = null;

        if (_ownedAllocations is not null)
        {
            foreach (var ptr in _ownedAllocations)
            {
                if (ptr != nint.Zero)
                    NativeMemory.AlignedFree((void*)ptr);
            }
            _ownedAllocations.Clear();
        }
    }

    private static TransformerLayerWeights LoadLayer(
        int layerIdx,
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config)
    {
        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;

        // Attention norm — dequantize to float[]
        var attnNormDesc = tensors[$"{prefix}.attn_norm.weight"];
        float[] attnNorm = DequantizeNorm(dataBase, attnNormDesc, hiddenSize);

        // Q/K/V projections — check for fused attn_qkv.weight (Phi-3 style)
        nint qPtr, kPtr, vPtr;
        QuantizationType qQt, kQt, vQt;
        int qM, qK, kM, kK, vM, vK;

        if (tensors.TryGetValue($"{prefix}.attn_qkv.weight", out var qkvDesc))
        {
            // Fused QKV — split by row offset
            nint qkvPtr = dataBase + (nint)qkvDesc.DataOffset;
            int inputDim = qkvDesc.Shape[0]; // hidden_size
            long rowBytes = Dequantize.RowByteSize(inputDim, qkvDesc.QuantizationType);

            int qDim = config.NumAttentionHeads * config.HeadDim;
            int kvDim = config.NumKvHeads * config.HeadDim;

            qPtr = qkvPtr; qQt = qkvDesc.QuantizationType; qM = qDim; qK = inputDim;
            kPtr = qkvPtr + (nint)(qDim * rowBytes); kQt = qkvDesc.QuantizationType; kM = kvDim; kK = inputDim;
            vPtr = qkvPtr + (nint)((qDim + kvDim) * rowBytes); vQt = qkvDesc.QuantizationType; vM = kvDim; vK = inputDim;
        }
        else
        {
            // Separate Q/K/V (standard path)
            (qPtr, qQt, qM, qK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_q.weight"]);
            (kPtr, kQt, kM, kK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_k.weight"]);
            (vPtr, vQt, vM, vK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_v.weight"]);
        }

        var (oPtr, oQt, oM, oK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_output.weight"]);

        // Optional biases — check for fused attn_qkv.bias (Phi-3 style)
        float[]? qBias, kBias, vBias;
        if (tensors.TryGetValue($"{prefix}.attn_qkv.bias", out var qkvBiasDesc))
        {
            // Fused QKV bias — split by element offset
            nint biasPtr = dataBase + (nint)qkvBiasDesc.DataOffset;
            int qDim = config.NumAttentionHeads * config.HeadDim;
            int kvDim = config.NumKvHeads * config.HeadDim;

            qBias = new float[qDim];
            kBias = new float[kvDim];
            vBias = new float[kvDim];

            Dequantize.ToFloat32(biasPtr, qDim, qkvBiasDesc.QuantizationType, qBias);
            Dequantize.ToFloat32(biasPtr + qDim * sizeof(float), kvDim, qkvBiasDesc.QuantizationType, kBias);
            Dequantize.ToFloat32(biasPtr + (qDim + kvDim) * sizeof(float), kvDim, qkvBiasDesc.QuantizationType, vBias);
        }
        else
        {
            qBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.attn_q.bias");
            kBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.attn_k.bias");
            vBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.attn_v.bias");
        }
        float[]? oBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.attn_output.bias");

        // Optional QK-norms (Qwen3-style): per-head RMSNorm applied to Q/K after projection, before RoPE
        float[]? qNormWeight = LoadOptionalNorm(dataBase, tensors, $"{prefix}.attn_q_norm.weight", config.HeadDim);
        float[]? kNormWeight = LoadOptionalNorm(dataBase, tensors, $"{prefix}.attn_k_norm.weight", config.HeadDim);

        // FFN norm
        var ffnNormDesc = tensors[$"{prefix}.ffn_norm.weight"];
        float[] ffnNorm = DequantizeNorm(dataBase, ffnNormDesc, hiddenSize);

        // FFN projections — check for fused gate+up (Phi-3 style: ffn_up.weight has 2x intermediate rows)
        nint gatePtr, upPtr, downPtr;
        QuantizationType gateQt, upQt, downQt;
        int gateM, gateK, upM, upK, downM, downK;
        float[]? gateBias, upBias, downBias;

        (downPtr, downQt, downM, downK) = LoadLinear(dataBase, tensors[$"{prefix}.ffn_down.weight"]);
        downBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.ffn_down.bias");

        if (tensors.TryGetValue($"{prefix}.ffn_gate.weight", out var gateDesc))
        {
            // Standard separate gate/up (Llama, Mistral, Qwen)
            (gatePtr, gateQt, gateM, gateK) = LoadLinear(dataBase, gateDesc);
            (upPtr, upQt, upM, upK) = LoadLinear(dataBase, tensors[$"{prefix}.ffn_up.weight"]);
            gateBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.ffn_gate.bias");
            upBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.ffn_up.bias");
        }
        else
        {
            // Fused gate+up in ffn_up.weight (Phi-3 style): output dim = 2 * intermediate_size
            // Split: first intermediate_size rows = gate, next intermediate_size rows = up
            var fusedDesc = tensors[$"{prefix}.ffn_up.weight"];
            nint fusedPtr = dataBase + (nint)fusedDesc.DataOffset;
            int inputDim = fusedDesc.Shape[0]; // hidden_size
            int fusedOutputDim = fusedDesc.Shape[1]; // 2 * intermediate_size
            int halfDim = fusedOutputDim / 2;
            long rowBytes = Dequantize.RowByteSize(inputDim, fusedDesc.QuantizationType);

            gatePtr = fusedPtr; gateQt = fusedDesc.QuantizationType; gateM = halfDim; gateK = inputDim;
            upPtr = fusedPtr + (nint)(halfDim * rowBytes); upQt = fusedDesc.QuantizationType; upM = halfDim; upK = inputDim;

            // Fused bias split (if present)
            if (tensors.TryGetValue($"{prefix}.ffn_up.bias", out var fusedBiasDesc))
            {
                nint biasPtr = dataBase + (nint)fusedBiasDesc.DataOffset;
                gateBias = new float[halfDim];
                upBias = new float[halfDim];
                Dequantize.ToFloat32(biasPtr, halfDim, fusedBiasDesc.QuantizationType, gateBias);
                Dequantize.ToFloat32(biasPtr + halfDim * sizeof(float), halfDim, fusedBiasDesc.QuantizationType, upBias);
            }
            else
            {
                gateBias = null;
                upBias = null;
            }
        }

        return new TransformerLayerWeights(
            attnNorm,
            qPtr, qQt, qM, qK,
            kPtr, kQt, kM, kK,
            vPtr, vQt, vM, vK,
            oPtr, oQt, oM, oK,
            ffnNorm,
            gatePtr, gateQt, gateM, gateK,
            upPtr, upQt, upM, upK,
            downPtr, downQt, downM, downK,
            qBias, kBias, vBias, oBias,
            gateBias, upBias, downBias,
            qNormWeight, kNormWeight);
    }

    private static (nint ptr, QuantizationType qt, int outputDim, int inputDim) LoadLinear(
        nint dataBase, GgufTensorDescriptor desc)
    {
        nint ptr = dataBase + (nint)desc.DataOffset;
        // GGUF: Dimensions[0] = input dim (K), Dimensions[1] = output dim (M)
        int k = desc.Shape[0];
        int m = desc.Shape[1];
        return (ptr, desc.QuantizationType, m, k);
    }

    /// <summary>
    /// Loads a single DeepSeek-V2 / V3 MLA layer's projection tensors from GGUF.
    /// Each MLA-specific tensor is dequantized to a 64-byte-aligned F32 host
    /// buffer (to match the CPU oracle <see cref="DotLLM.Cpu.Kernels.MlaAttention.Execute"/>'s
    /// F32 contract); the returned <see cref="TransformerLayerWeights"/> carries
    /// these F32 pointers in <c>lw.Mla</c> and zeroes the legacy GQA Q/K/V slots.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Tensor naming</b> (per llama.cpp's <c>convert_hf_to_gguf.py</c>
    /// <c>DeepseekV2Model</c>):
    /// <list type="bullet">
    ///   <item><c>blk.{N}.attn_q_a.weight</c> + <c>attn_q_a_norm.weight</c> +
    ///     <c>attn_q_b.weight</c> when <c>q_lora_rank &gt; 0</c></item>
    ///   <item><c>blk.{N}.attn_q.weight</c> when <c>q_lora_rank == 0</c> (V2-Lite)</item>
    ///   <item><c>blk.{N}.attn_kv_a_mqa.weight</c> + <c>attn_kv_a_norm.weight</c> +
    ///     <c>attn_kv_b.weight</c></item>
    ///   <item><c>blk.{N}.attn_output.weight</c> (same name as GQA — reused as o_proj)</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Memory budget.</b> Q4_K_M → F32 dequant inflates ~4× per element.
    /// V2-Lite MLA per-layer footprint ≈ 12 MB raw → 48 MB F32 (×27 layers ≈
    /// 1.3 GB total). Dense FFN (separate path) is the main pressure.
    /// Full-V2 MLA is ~10× this (160 GB) — that needs an on-device dequant
    /// path; flagged as a follow-up.
    /// </para>
    /// </remarks>
    private static unsafe TransformerLayerWeights LoadMlaLayer(
        int layerIdx,
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config,
        List<nint> owned)
    {
        var mla = config.MlaConfig
            ?? throw new InvalidOperationException("LoadMlaLayer called without MlaConfig.");

        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        int qLora = mla.QLoraRank;
        int kvLora = mla.KvLoraRank;
        int qkNope = mla.QkNopeHeadDim;
        int qkRope = mla.QkRopeHeadDim;
        int vHead = mla.VHeadDim;
        int numHeads = config.NumAttentionHeads;
        int qTotal = numHeads * (qkNope + qkRope);
        int kvAOut = kvLora + qkRope;
        int kvBOut = numHeads * (qkNope + vHead);
        int oInput = numHeads * vHead;

        // ── Norms ─────────────────────────────────────────────────────
        float[] attnNorm = DequantizeNorm(dataBase, tensors[$"{prefix}.attn_norm.weight"], hiddenSize);
        float[] ffnNorm = DequantizeNorm(dataBase, tensors[$"{prefix}.ffn_norm.weight"], hiddenSize);

        // ── Q path ─────────────────────────────────────────────────────
        // Populate BOTH the F32 dequant (for the CPU MlaAttention oracle)
        // and the raw GGUF mmap pointer + quant type (for the GPU loader's
        // on-device dequant path). MLA total host F32 footprint at V2-Lite
        // scale is ~1.4 GB across 27 layers — acceptable; the MoE 3D-stacked
        // experts are where the host-RAM blowup lives, and that path is
        // refactored separately in task #10.
        nint qAProj = 0, qBProj = 0, qProj = 0;
        nint qAProjRaw = 0, qBProjRaw = 0, qProjRaw = 0;
        QuantizationType qAProjRawQt = QuantizationType.F32, qBProjRawQt = QuantizationType.F32, qProjRawQt = QuantizationType.F32;
        float[]? qANorm = null;
        if (qLora > 0)
        {
            var qaDesc = tensors[$"{prefix}.attn_q_a.weight"];
            qAProjRaw = dataBase + (nint)qaDesc.DataOffset;
            qAProjRawQt = qaDesc.QuantizationType;
            qAProj = DequantToF32(dataBase, qaDesc, (long)qLora * hiddenSize, owned);
            qANorm = DequantizeNorm(dataBase, tensors[$"{prefix}.attn_q_a_norm.weight"], qLora);

            var qbDesc = tensors[$"{prefix}.attn_q_b.weight"];
            qBProjRaw = dataBase + (nint)qbDesc.DataOffset;
            qBProjRawQt = qbDesc.QuantizationType;
            qBProj = DequantToF32(dataBase, qbDesc, (long)qTotal * qLora, owned);
        }
        else
        {
            var qDesc = tensors[$"{prefix}.attn_q.weight"];
            qProjRaw = dataBase + (nint)qDesc.DataOffset;
            qProjRawQt = qDesc.QuantizationType;
            qProj = DequantToF32(dataBase, qDesc, (long)qTotal * hiddenSize, owned);
        }

        // ── KV path (always factored) ────────────────────────────────
        var kvaDesc = tensors[$"{prefix}.attn_kv_a_mqa.weight"];
        nint kvAProjRaw = dataBase + (nint)kvaDesc.DataOffset;
        QuantizationType kvAProjRawQt = kvaDesc.QuantizationType;
        nint kvAProj = DequantToF32(dataBase, kvaDesc, (long)kvAOut * hiddenSize, owned);
        float[] kvANorm = DequantizeNorm(dataBase, tensors[$"{prefix}.attn_kv_a_norm.weight"], kvLora);

        var kvbDesc = tensors[$"{prefix}.attn_kv_b.weight"];
        nint kvBProjRaw = dataBase + (nint)kvbDesc.DataOffset;
        QuantizationType kvBProjRawQt = kvbDesc.QuantizationType;
        nint kvBProj = DequantToF32(dataBase, kvbDesc, (long)kvBOut * kvLora, owned);

        // ── O projection (same tensor name as GQA: attn_output) ──────
        // O lives in TransformerLayerWeights.OWeight + OQuantType (the existing
        // GQA slot) — the raw quant view comes for free via that field;
        // the F32 dequant here is for the CPU MLA path's o_proj GEMM.
        var oDesc = tensors[$"{prefix}.attn_output.weight"];
        nint oProj = DequantToF32(dataBase, oDesc, (long)hiddenSize * oInput, owned);

        var mlaBundle = new MlaLayerWeights(
            qAProj: qAProj,
            qALayernormWeight: qANorm,
            qBProj: qBProj,
            qProj: qProj,
            kvAProjWithMqa: kvAProj,
            kvALayernormWeight: kvANorm,
            kvBProj: kvBProj,
            numHeads: numHeads,
            qkNopeHeadDim: qkNope,
            qkRopeHeadDim: qkRope,
            vHeadDim: vHead,
            qLoraRank: qLora,
            kvLoraRank: kvLora,
            hiddenSize: hiddenSize,
            qAProjRaw: qAProjRaw, qAProjRawQt: qAProjRawQt,
            qBProjRaw: qBProjRaw, qBProjRawQt: qBProjRawQt,
            qProjRaw: qProjRaw, qProjRawQt: qProjRawQt,
            kvAProjWithMqaRaw: kvAProjRaw, kvAProjWithMqaRawQt: kvAProjRawQt,
            kvBProjRaw: kvBProjRaw, kvBProjRawQt: kvBProjRawQt);

        // ── FFN ────────────────────────────────────────────────────────
        // DeepSeek-V2/V3 layouts:
        //   * Pre-MoE dense layers (layerIdx < leading_dense_block_count) carry
        //     `blk.{N}.ffn_gate.weight` / `ffn_up.weight` / `ffn_down.weight`.
        //   * MoE layers carry instead a 3D-stacked expert block and (optionally)
        //     a single fused shared-expert MLP — see LoadDeepSeekMoeLayer.
        bool layerIsMoe = config.Moe is not null && config.Moe.IsMoeLayer(layerIdx);

        nint gatePtr = 0; QuantizationType gateQt = QuantizationType.F32; int gateM = 0, gateK = 0;
        nint upPtr = 0; QuantizationType upQt = QuantizationType.F32; int upM = 0, upK = 0;
        nint downPtr = 0; QuantizationType downQt = QuantizationType.F32; int downM = 0, downK = 0;
        MoeLayerWeights? moeBundle = null;

        if (layerIsMoe)
        {
            moeBundle = LoadDeepSeekMoeLayer(layerIdx, dataBase, tensors, config, owned);
        }
        else if (tensors.TryGetValue($"{prefix}.ffn_gate.weight", out var gateDesc))
        {
            (gatePtr, gateQt, gateM, gateK) = LoadLinear(dataBase, gateDesc);
            (upPtr, upQt, upM, upK) = LoadLinear(dataBase, tensors[$"{prefix}.ffn_up.weight"]);
            (downPtr, downQt, downM, downK) = LoadLinear(dataBase, tensors[$"{prefix}.ffn_down.weight"]);
        }
        else
        {
            throw new InvalidDataException(
                $"DeepSeek-V2 layer {layerIdx} has neither dense ffn_gate.weight nor MoE ffn_*_exps tensors.");
        }

        // GGUF: Dimensions[0] = input dim (K), Dimensions[1] = output dim (M)
        return new TransformerLayerWeights(
            attnNormWeight: attnNorm,
            qWeight: 0, qQuantType: QuantizationType.F32, qOutputDim: 0, qInputDim: 0,
            kWeight: 0, kQuantType: QuantizationType.F32, kOutputDim: 0, kInputDim: 0,
            vWeight: 0, vQuantType: QuantizationType.F32, vOutputDim: 0, vInputDim: 0,
            oWeight: oProj, oQuantType: QuantizationType.F32,
            oOutputDim: hiddenSize, oInputDim: oInput,
            ffnNormWeight: ffnNorm,
            gateWeight: gatePtr, gateQuantType: gateQt, gateOutputDim: gateM, gateInputDim: gateK,
            upWeight: upPtr, upQuantType: upQt, upOutputDim: upM, upInputDim: upK,
            downWeight: downPtr, downQuantType: downQt, downOutputDim: downM, downInputDim: downK,
            mla: mlaBundle,
            moe: moeBundle);
    }

    /// <summary>
    /// Loads a single DeepSeek-V2 / V3 MoE layer's expert tensors from GGUF.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>3D-stacked expert layout</b> (per llama.cpp's <c>convert_hf_to_gguf.py</c>
    /// <c>DeepseekV2Model</c>):
    /// <list type="bullet">
    ///   <item><c>blk.{N}.ffn_gate_inp.weight</c> — router gate <c>[hidden, num_experts]</c>.</item>
    ///   <item><c>blk.{N}.ffn_gate_exps.weight</c> — fused per-expert gate_proj
    ///     <c>[hidden, intermediate, num_experts]</c>. Each expert is a contiguous
    ///     <c>[hidden, intermediate]</c> slice in GGUF on-disk order.</item>
    ///   <item><c>blk.{N}.ffn_up_exps.weight</c> — fused per-expert up_proj, same layout.</item>
    ///   <item><c>blk.{N}.ffn_down_exps.weight</c> — fused per-expert down_proj
    ///     <c>[intermediate, hidden, num_experts]</c>.</item>
    ///   <item>Optional shared experts: <c>ffn_gate_shexp.weight</c> / <c>ffn_up_shexp.weight</c>
    ///     / <c>ffn_down_shexp.weight</c>. DeepSeek fuses N shared experts into a single
    ///     MLP of width <c>moe_intermediate × n_shared_experts</c>.</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Memory pressure.</b> Each expert is dequantized to a contiguous F32 host
    /// buffer. For V2-Lite (64 experts × 2048 hidden × 1408 intermediate × 3 mats
    /// × 4 bytes ≈ 2.2 GB per layer × 26 MoE layers ≈ 57 GB of F32 host RAM).
    /// This is acknowledged untenable for full-V2 and is what tasks #9/#10
    /// (on-device dequant) replace.
    /// </para>
    /// </remarks>
    internal static unsafe MoeLayerWeights LoadDeepSeekMoeLayer(
        int layerIdx,
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config,
        List<nint> owned)
    {
        var moe = config.Moe
            ?? throw new InvalidOperationException("LoadDeepSeekMoeLayer called without Moe config.");

        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        int numExperts = moe.NumExperts;
        int moeIntermediate = moe.MoeIntermediateSize;

        // Router (2D, F32 — small, dequant inline).
        var routerDesc = tensors[$"{prefix}.ffn_gate_inp.weight"];
        float[] router = new float[numExperts * hiddenSize];
        Dequantize.ToFloat32(
            dataBase + (nint)routerDesc.DataOffset,
            (long)numExperts * hiddenSize,
            routerDesc.QuantizationType,
            router);

        // Per-expert routed projections. Populate BOTH the F32 dequant (for
        // the CPU MoeSwiGluMlp oracle path) AND the raw GGUF mmap pointer +
        // quant type (for the CUDA loader's on-device dequant path). The F32
        // dequant footprint at full V2-Lite scale is ~57 GB (untenable) — the
        // GPU loader takes the raw view and avoids the host inflation; the
        // CPU oracle just isn't run on full V2-Lite for this reason.
        var gateDesc = tensors[$"{prefix}.ffn_gate_exps.weight"];
        var upDesc = tensors[$"{prefix}.ffn_up_exps.weight"];
        var downDesc = tensors[$"{prefix}.ffn_down_exps.weight"];

        var w1 = SliceExpertsToF32(
            dataBase, gateDesc,
            numExperts, M: moeIntermediate, K: hiddenSize, owned);
        var w3 = SliceExpertsToF32(
            dataBase, upDesc,
            numExperts, M: moeIntermediate, K: hiddenSize, owned);
        var w2 = SliceExpertsToF32(
            dataBase, downDesc,
            numExperts, M: hiddenSize, K: moeIntermediate, owned);

        nint gateRaw = dataBase + (nint)gateDesc.DataOffset;
        nint upRaw = dataBase + (nint)upDesc.DataOffset;
        nint downRaw = dataBase + (nint)downDesc.DataOffset;

        // Shared expert (DeepSeek-V2/V3 fuses N shared into a single wider MLP).
        nint[] sharedGate = Array.Empty<nint>();
        nint[] sharedUp = Array.Empty<nint>();
        nint[] sharedDown = Array.Empty<nint>();
        nint[] sharedGateRaw = Array.Empty<nint>();
        nint[] sharedUpRaw = Array.Empty<nint>();
        nint[] sharedDownRaw = Array.Empty<nint>();
        QuantizationType sharedGateRawQt = QuantizationType.F32;
        QuantizationType sharedUpRawQt = QuantizationType.F32;
        QuantizationType sharedDownRawQt = QuantizationType.F32;
        int sharedIntermediate = 0;
        if (moe.SharedExpertIntermediateSize is int sharedI && sharedI > 0
            && tensors.ContainsKey($"{prefix}.ffn_gate_shexp.weight"))
        {
            sharedIntermediate = sharedI;
            var sharedGateDesc = tensors[$"{prefix}.ffn_gate_shexp.weight"];
            var sharedUpDesc = tensors[$"{prefix}.ffn_up_shexp.weight"];
            var sharedDownDesc = tensors[$"{prefix}.ffn_down_shexp.weight"];

            sharedGate = [DequantToF32(dataBase, sharedGateDesc, (long)sharedI * hiddenSize, owned)];
            sharedUp = [DequantToF32(dataBase, sharedUpDesc, (long)sharedI * hiddenSize, owned)];
            sharedDown = [DequantToF32(dataBase, sharedDownDesc, (long)hiddenSize * sharedI, owned)];

            sharedGateRaw = [dataBase + (nint)sharedGateDesc.DataOffset];
            sharedGateRawQt = sharedGateDesc.QuantizationType;
            sharedUpRaw = [dataBase + (nint)sharedUpDesc.DataOffset];
            sharedUpRawQt = sharedUpDesc.QuantizationType;
            sharedDownRaw = [dataBase + (nint)sharedDownDesc.DataOffset];
            sharedDownRawQt = sharedDownDesc.QuantizationType;
        }

        // DeepSeek convention: no per-token sigmoid gate on the shared branch.
        // (HfConfigExtractor + MoeConfig keep HasSharedExpertGate=false here.)
        return new MoeLayerWeights(
            gate: router,
            w1: w1,
            w2: w2,
            w3: w3,
            numExperts: numExperts,
            numExpertsPerTok: moe.NumExpertsPerTok,
            hiddenSize: hiddenSize,
            intermediateSize: moeIntermediate,
            normTopKProb: moe.NormTopKProb,
            sharedGateProj: sharedGate,
            sharedUpProj: sharedUp,
            sharedDownProj: sharedDown,
            sharedIntermediateSize: sharedIntermediate,
            sharedExpertGate: null,
            gateExpsRaw: gateRaw, gateExpsRawQt: gateDesc.QuantizationType,
            gateExpsMDim: moeIntermediate, gateExpsKDim: hiddenSize,
            upExpsRaw: upRaw, upExpsRawQt: upDesc.QuantizationType,
            upExpsMDim: moeIntermediate, upExpsKDim: hiddenSize,
            downExpsRaw: downRaw, downExpsRawQt: downDesc.QuantizationType,
            downExpsMDim: hiddenSize, downExpsKDim: moeIntermediate,
            sharedGateRaw: sharedGateRaw, sharedGateRawQt: sharedGateRawQt,
            sharedUpRaw: sharedUpRaw, sharedUpRawQt: sharedUpRawQt,
            sharedDownRaw: sharedDownRaw, sharedDownRawQt: sharedDownRawQt);
    }

    /// <summary>
    /// Slices a 3D fused-experts tensor and dequantizes each expert's [M, K]
    /// sub-block into its own F32 buffer. Returns the per-expert pointer array.
    /// </summary>
    /// <remarks>
    /// GGUF on-disk layout for <c>ffn_gate_exps</c>/<c>ffn_up_exps</c>:
    /// <c>Shape = [K, M, num_experts]</c> (K innermost). Each expert's slice
    /// has byte size <c>M * RowByteSize(K, qt)</c>. The offset to expert e's
    /// slice is <c>baseOffset + e * (M * RowByteSize(K, qt))</c>. We dequant
    /// each expert as a contiguous run of <c>M*K</c> elements (every Q4_K-family
    /// row aligns on the start of a 256-element super-block when K%256==0,
    /// which holds for every shipping DeepSeek-V2/V3 size).
    /// </remarks>
    private static unsafe nint[] SliceExpertsToF32(
        nint dataBase, GgufTensorDescriptor desc,
        int numExperts, int M, int K, List<nint> owned)
    {
        if (desc.Shape.Rank != 3)
            throw new InvalidDataException(
                $"Expected 3D fused-experts tensor; got rank {desc.Shape.Rank}.");

        // GGUF Shape ordering: [innermost, ..., outermost]. For ffn_*_exps
        // the on-disk shape is [K, M, num_experts] — verify against expected
        // dims so we fail fast on mis-shaped checkpoints.
        if (desc.Shape[0] != K || desc.Shape[1] != M || desc.Shape[2] != numExperts)
            throw new InvalidDataException(
                $"Fused-experts tensor shape {desc.Shape[0]}×{desc.Shape[1]}×{desc.Shape[2]} " +
                $"does not match expected K={K} × M={M} × E={numExperts}.");

        long perExpertBytes = M * Dequantize.RowByteSize(K, desc.QuantizationType);
        long perExpertElements = (long)M * K;
        nint base_ = dataBase + (nint)desc.DataOffset;

        var ptrs = new nint[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            nuint dstBytes = (nuint)(perExpertElements * sizeof(float));
            nint dst = (nint)NativeMemory.AlignedAlloc(dstBytes, 64);
            owned.Add(dst);
            Dequantize.ToFloat32(
                base_ + (nint)(e * perExpertBytes),
                perExpertElements,
                desc.QuantizationType,
                new Span<float>((void*)dst, (int)perExpertElements));
            ptrs[e] = dst;
        }
        return ptrs;
    }

    /// <summary>
    /// Allocates a 64-byte-aligned F32 buffer and dequantizes <paramref name="elementCount"/>
    /// values from the GGUF tensor at <paramref name="desc"/>'s data offset into it.
    /// Tracks the allocation in <paramref name="owned"/> so the loader's Dispose
    /// can free it. Returns the pointer.
    /// </summary>
    private static unsafe nint DequantToF32(nint dataBase, GgufTensorDescriptor desc,
                                            long elementCount, List<nint> owned)
    {
        nuint bytes = (nuint)(elementCount * sizeof(float));
        nint dst = (nint)NativeMemory.AlignedAlloc(bytes, 64);
        owned.Add(dst);
        nint src = dataBase + (nint)desc.DataOffset;
        Dequantize.ToFloat32(src, elementCount, desc.QuantizationType,
                              new Span<float>((void*)dst, (int)elementCount));
        return dst;
    }

    private static float[] DequantizeNorm(nint dataBase, GgufTensorDescriptor desc, int expectedSize)
    {
        nint ptr = dataBase + (nint)desc.DataOffset;
        float[] result = new float[expectedSize];
        Dequantize.ToFloat32(ptr, expectedSize, desc.QuantizationType, result);
        return result;
    }

    /// <summary>
    /// Loads an optional norm weight tensor. Returns null when the tensor is absent.
    /// </summary>
    private static float[]? LoadOptionalNorm(nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, string name, int expectedSize)
    {
        if (!tensors.TryGetValue(name, out var desc)) return null;
        return DequantizeNorm(dataBase, desc, expectedSize);
    }

    /// <summary>
    /// Loads an optional bias tensor (F32 in GGUF). Returns null when the tensor is absent.
    /// </summary>
    private static float[]? LoadOptionalBias(nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, string name)
    {
        if (!tensors.TryGetValue(name, out var desc)) return null;
        int size = (int)desc.Shape.ElementCount;
        float[] result = new float[size];
        Dequantize.ToFloat32(dataBase + (nint)desc.DataOffset, size, desc.QuantizationType, result);
        return result;
    }
}
