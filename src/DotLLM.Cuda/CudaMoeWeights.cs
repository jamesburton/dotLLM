using DotLLM.Core.Configuration;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Numeric precision used by an MoE layer's expert weights. F32 is the
/// existing CPU-oracle-matching path. <c>Quantized</c> keeps raw GGUF Q4_K /
/// Q8_0 bytes on device per expert and dequantizes per call into reused F16
/// scratch — the only viable path for full DeepSeek-V2-Lite (the F32
/// fully-resident expert block doesn't fit a 12 GB GPU).
/// </summary>
public enum MoePrecision
{
    /// <summary>F32 expert weights resident on GPU; cuBLAS LinearF32 directly.</summary>
    F32 = 0,
    /// <summary>Raw GGUF quant bytes per expert; dequant per call.</summary>
    Quantized = 1,
    /// <summary>
    /// Ternary I2_S (BitNet) expert weights (issue #246): per-expert packed-trit banks with
    /// a per-expert tail scale, dispatched through <see cref="CudaKernels.LaunchI2_SGemvF32In"/>
    /// (one GEMV call per assigned row — see <see cref="CudaMoeFfn"/>'s BitNet-MoE forward).
    /// The expert body is <c>down( ffn_sub_norm( relu²(gate(x)) * up(x) ) )</c>, not SwiGLU.
    /// </summary>
    BitNetI2S = 2,
}

/// <summary>
/// Per-layer GPU weight pointers for a Mixture-of-Experts SwiGLU FFN
/// (Mixtral / Qwen-MoE / Phi-3.5-MoE / DeepSeek-V2/V3).
/// Phase 1: every projection is uploaded as F32 row-major to GPU memory and
/// the kernel path operates fully in F32 — matches the CPU oracle
/// <c>DotLLM.Cpu.Kernels.MoeSwiGluMlp.Execute</c>/<c>ExecuteWithSharedExpert</c>
/// byte-for-byte algorithmically. Quantized / FP16 paths are deferred to a
/// follow-up agent.
/// </summary>
/// <remarks>
/// <para>
/// <b>Routed-expert layout.</b> <see cref="GateProj"/> / <see cref="UpProj"/> /
/// <see cref="DownProj"/> are device-resident <c>nint[]</c> arrays of length
/// <see cref="NumExperts"/>. Each entry points at one expert's row-major F32
/// weight matrix:
/// </para>
/// <list type="bullet">
///   <item><c>GateProj[e]</c> — <c>[moeIntermediateSize, hiddenSize]</c></item>
///   <item><c>UpProj[e]</c> — <c>[moeIntermediateSize, hiddenSize]</c></item>
///   <item><c>DownProj[e]</c> — <c>[hiddenSize, moeIntermediateSize]</c></item>
/// </list>
/// <para>
/// <b>Router.</b> <see cref="Router"/> is a single F32 device buffer of
/// <c>[numExperts × hiddenSize]</c> row-major. The forward path runs
/// <c>logits = router @ hidden</c> per token via cuBLAS GEMV/GEMM.
/// </para>
/// <para>
/// <b>Shared experts.</b> When <see cref="NumSharedExperts"/> &gt; 0 (DeepSeek
/// has N≥1 shared experts; Qwen1.5-MoE has 1), each shared expert is a dense
/// SwiGLU MLP run on EVERY token. The pointer arrays
/// <see cref="SharedGateProj"/>/<see cref="SharedUpProj"/>/<see cref="SharedDownProj"/>
/// have length <see cref="NumSharedExperts"/> and follow the same layout as
/// the routed experts but with width <see cref="SharedIntermediateSize"/>.
/// </para>
/// <para>
/// <b>Optional sigmoid gate.</b> Qwen1.5-MoE ships with a
/// <c>shared_expert_gate</c> weight (single hidden-wide vector). When
/// <see cref="SharedExpertGate"/> ≠ 0, the shared-expert contribution is
/// scaled by <c>sigmoid(hidden · gate)</c> per token. DeepSeek does not use
/// this gate (<see cref="SharedExpertGate"/> stays 0).
/// </para>
/// </remarks>
public sealed class CudaMoeLayerWeights
{
    /// <summary>Total number of routed experts per layer (Mixtral=8, DeepSeek-V2-Lite=64).</summary>
    public int NumExperts { get; }

    /// <summary>Number of routed experts activated per token (Mixtral=2, DeepSeek-V2-Lite=6).</summary>
    public int NumExpertsPerTok { get; }

    /// <summary>Model hidden / residual dimension.</summary>
    public int HiddenSize { get; }

    /// <summary>Per-routed-expert intermediate width (Phi-3.5-MoE / DeepSeek surface this).</summary>
    public int MoeIntermediateSize { get; }

    /// <summary>Whether to renormalise top-k weights to sum to 1.0 (Mixtral / Qwen3-MoE).</summary>
    public bool NormTopKProb { get; }

    /// <summary>F32 router weight device pointer <c>[numExperts × hiddenSize]</c>.</summary>
    public nint Router { get; }

    /// <summary>
    /// Per-routed-expert device pointers — length <see cref="NumExperts"/>. Meaning depends on
    /// <see cref="Precision"/>: F32 → row-major F32 <c>gate_proj</c> weight; Quantized → raw GGUF
    /// quant bytes; <see cref="MoePrecision.BitNetI2S"/> → a self-contained per-expert I2_S
    /// buffer of <c>rowBytes + sizeof(float)</c> bytes, packed-trit payload followed by that
    /// expert's own tail scale (issue #246 — see <see cref="CudaMoeWeightsLoader.LoadLayerBitNetI2S"/>
    /// for why the scale is repacked per-expert rather than read from a shared array).
    /// </summary>
    public nint[] GateProj { get; }

    /// <summary>Per-routed-expert <c>up_proj</c> device pointers. See <see cref="GateProj"/> for
    /// the precision-dependent meaning.</summary>
    public nint[] UpProj { get; }

    /// <summary>Per-routed-expert <c>down_proj</c> device pointers. See <see cref="GateProj"/> for
    /// the precision-dependent meaning.</summary>
    public nint[] DownProj { get; }

    /// <summary>
    /// Per-routed-expert F32 BitNet FFN Sub-LN weight device pointers — length
    /// <see cref="NumExperts"/>, each <c>[moeIntermediateSize]</c>. Only populated when
    /// <see cref="Precision"/> == <see cref="MoePrecision.BitNetI2S"/>; empty otherwise.
    /// </summary>
    public nint[] ExpertFfnSubNormF32 { get; }

    /// <summary>
    /// Optional F32 additive router-bias device pointer <c>[numExperts]</c> (identity-MoTE /
    /// Qwen3 aux-loss-free routing). 0 when the router has no bias. Applied to the router
    /// logits before softmax/top-k — see <see cref="CudaKernels.LaunchMoeGateBiasAddF32"/>.
    /// </summary>
    public nint GateBiasF32 { get; }

    /// <summary>
    /// Optional F32 router-bias device pointer <c>[numExperts]</c> for the quantized-expert
    /// gating path (gpt-oss <c>ffn_gate_inp.bias</c>, issue #348). 0 when absent. Distinct
    /// from <see cref="GateBiasF32"/> (the identity-MoTE / <see cref="MoePrecision.BitNetI2S"/>
    /// router bias) so the two features can evolve independently even though both add to the
    /// same router logits before softmax/top-k.
    /// </summary>
    public nint RouterBiasF32 { get; }

    /// <summary>
    /// Optional per-expert gate-projection bias, flat F32 device pointer
    /// <c>[numExperts × moeIntermediateSize]</c> (gpt-oss <c>ffn_gate_exps.bias</c>). 0 when
    /// absent. Expert <c>e</c>'s slice starts at byte offset
    /// <c>e * moeIntermediateSize * sizeof(float)</c>.
    /// </summary>
    public nint GateExpsBiasF32 { get; }

    /// <summary>Optional per-expert up-projection bias. Same layout as <see cref="GateExpsBiasF32"/>.</summary>
    public nint UpExpsBiasF32 { get; }

    /// <summary>
    /// Optional per-expert down-projection bias, flat F32 device pointer
    /// <c>[numExperts × hiddenSize]</c>. 0 when absent. Expert <c>e</c>'s slice starts at byte
    /// offset <c>e * hiddenSize * sizeof(float)</c>.
    /// </summary>
    public nint DownExpsBiasF32 { get; }

    /// <summary>
    /// True = gpt-oss clamped swiglu_oai activation (<see cref="CudaKernels.LaunchSwiGLUOaiF32"/>);
    /// false = plain SwiGLU (<see cref="CudaKernels.LaunchSwiGLUF32"/>). Default false.
    /// </summary>
    public bool UseSwiGluOai { get; }

    /// <summary>
    /// RMSNorm epsilon for the per-expert FFN Sub-LN. Only meaningful when
    /// <see cref="Precision"/> == <see cref="MoePrecision.BitNetI2S"/>.
    /// </summary>
    public float RmsEps { get; }

    /// <summary>Number of shared experts (1+ for DeepSeek + Qwen1.5-MoE; 0 for Mixtral / Phi-3.5-MoE).</summary>
    public int NumSharedExperts { get; }

    /// <summary>Per-shared-expert intermediate width (0 when no shared experts).</summary>
    public int SharedIntermediateSize { get; }

    /// <summary>Per-shared-expert F32 gate_proj device pointers — length <see cref="NumSharedExperts"/>.</summary>
    public nint[] SharedGateProj { get; }

    /// <summary>Per-shared-expert F32 up_proj device pointers — length <see cref="NumSharedExperts"/>.</summary>
    public nint[] SharedUpProj { get; }

    /// <summary>Per-shared-expert F32 down_proj device pointers — length <see cref="NumSharedExperts"/>.</summary>
    public nint[] SharedDownProj { get; }

    /// <summary>
    /// Optional Qwen1.5-MoE per-token sigmoid gate weight <c>[hiddenSize]</c>.
    /// 0 ⇒ no gate (DeepSeek default; shared-expert contribution added with
    /// implicit scale=1).
    /// </summary>
    public nint SharedExpertGate { get; }

    /// <summary>Total expert vector elements per token: <c>numExpertsPerTok</c>.</summary>
    public int TopK => NumExpertsPerTok;

    /// <summary>Precision of the per-expert projections. F32 = direct cuBLAS GEMM; Quantized = dequant-per-call into F16 scratch.</summary>
    public MoePrecision Precision { get; }

    /// <summary>Quant type for <see cref="GateProj"/> when <see cref="Precision"/>==<see cref="MoePrecision.Quantized"/> (raw GGUF bytes per expert).</summary>
    public QuantizationType GateProjQuantType { get; }
    /// <summary>Quant type for <see cref="UpProj"/> on the quantized path.</summary>
    public QuantizationType UpProjQuantType { get; }
    /// <summary>Quant type for <see cref="DownProj"/> on the quantized path.</summary>
    public QuantizationType DownProjQuantType { get; }
    /// <summary>Quant type for <see cref="SharedGateProj"/> on the quantized path.</summary>
    public QuantizationType SharedGateProjQuantType { get; }
    /// <summary>Quant type for <see cref="SharedUpProj"/> on the quantized path.</summary>
    public QuantizationType SharedUpProjQuantType { get; }
    /// <summary>Quant type for <see cref="SharedDownProj"/> on the quantized path.</summary>
    public QuantizationType SharedDownProjQuantType { get; }

    /// <summary>Constructs a fully-populated F32 MoE layer weight bundle (back-compat ctor).</summary>
    public CudaMoeLayerWeights(
        int numExperts, int numExpertsPerTok, int hiddenSize, int moeIntermediateSize,
        bool normTopKProb,
        nint router,
        nint[] gateProj, nint[] upProj, nint[] downProj,
        int numSharedExperts, int sharedIntermediateSize,
        nint[]? sharedGateProj, nint[]? sharedUpProj, nint[]? sharedDownProj,
        nint sharedExpertGate)
        : this(numExperts, numExpertsPerTok, hiddenSize, moeIntermediateSize,
               normTopKProb, router,
               gateProj, upProj, downProj,
               numSharedExperts, sharedIntermediateSize,
               sharedGateProj, sharedUpProj, sharedDownProj, sharedExpertGate,
               precision: MoePrecision.F32,
               gateProjQuantType: QuantizationType.F32,
               upProjQuantType: QuantizationType.F32,
               downProjQuantType: QuantizationType.F32,
               sharedGateProjQuantType: QuantizationType.F32,
               sharedUpProjQuantType: QuantizationType.F32,
               sharedDownProjQuantType: QuantizationType.F32)
    {
    }

    /// <summary>Full ctor with explicit precision + per-projection quant types.</summary>
    public CudaMoeLayerWeights(
        int numExperts, int numExpertsPerTok, int hiddenSize, int moeIntermediateSize,
        bool normTopKProb,
        nint router,
        nint[] gateProj, nint[] upProj, nint[] downProj,
        int numSharedExperts, int sharedIntermediateSize,
        nint[]? sharedGateProj, nint[]? sharedUpProj, nint[]? sharedDownProj,
        nint sharedExpertGate,
        MoePrecision precision,
        QuantizationType gateProjQuantType,
        QuantizationType upProjQuantType,
        QuantizationType downProjQuantType,
        QuantizationType sharedGateProjQuantType,
        QuantizationType sharedUpProjQuantType,
        QuantizationType sharedDownProjQuantType,
        nint[]? expertFfnSubNormF32 = null,
        nint gateBiasF32 = 0,
        float rmsEps = 0f,
        nint routerBiasF32 = 0,
        nint gateExpsBiasF32 = 0,
        nint upExpsBiasF32 = 0,
        nint downExpsBiasF32 = 0,
        bool useSwiGluOai = false)
    {
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (numExpertsPerTok <= 0 || numExpertsPerTok > numExperts)
            throw new ArgumentOutOfRangeException(nameof(numExpertsPerTok));
        if (gateProj.Length != numExperts || upProj.Length != numExperts || downProj.Length != numExperts)
            throw new ArgumentException("Per-expert pointer arrays must have numExperts entries.");
        if (precision == MoePrecision.BitNetI2S
            && (expertFfnSubNormF32 is null || expertFfnSubNormF32.Length != numExperts))
            throw new ArgumentException(
                "expertFfnSubNormF32 must have numExperts entries when precision is BitNetI2S.");
        if (numSharedExperts > 0)
        {
            if (sharedIntermediateSize <= 0)
                throw new ArgumentException("sharedIntermediateSize must be > 0 when numSharedExperts > 0.");
            if (sharedGateProj is null || sharedUpProj is null || sharedDownProj is null
                || sharedGateProj.Length != numSharedExperts
                || sharedUpProj.Length != numSharedExperts
                || sharedDownProj.Length != numSharedExperts)
                throw new ArgumentException("Per-shared-expert pointer arrays must have numSharedExperts entries.");
        }
        NumExperts = numExperts;
        NumExpertsPerTok = numExpertsPerTok;
        HiddenSize = hiddenSize;
        MoeIntermediateSize = moeIntermediateSize;
        NormTopKProb = normTopKProb;
        Router = router;
        GateProj = gateProj;
        UpProj = upProj;
        DownProj = downProj;
        NumSharedExperts = numSharedExperts;
        SharedIntermediateSize = sharedIntermediateSize;
        SharedGateProj = sharedGateProj ?? Array.Empty<nint>();
        SharedUpProj = sharedUpProj ?? Array.Empty<nint>();
        SharedDownProj = sharedDownProj ?? Array.Empty<nint>();
        SharedExpertGate = sharedExpertGate;
        Precision = precision;
        GateProjQuantType = gateProjQuantType;
        UpProjQuantType = upProjQuantType;
        DownProjQuantType = downProjQuantType;
        SharedGateProjQuantType = sharedGateProjQuantType;
        SharedUpProjQuantType = sharedUpProjQuantType;
        SharedDownProjQuantType = sharedDownProjQuantType;
        ExpertFfnSubNormF32 = expertFfnSubNormF32 ?? Array.Empty<nint>();
        GateBiasF32 = gateBiasF32;
        RmsEps = rmsEps;
        RouterBiasF32 = routerBiasF32;
        GateExpsBiasF32 = gateExpsBiasF32;
        UpExpsBiasF32 = upExpsBiasF32;
        DownExpsBiasF32 = downExpsBiasF32;
        UseSwiGluOai = useSwiGluOai;
    }
}
