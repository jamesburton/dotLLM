using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

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

    /// <summary>Per-routed-expert F32 gate_proj device pointers — length <see cref="NumExperts"/>.</summary>
    public nint[] GateProj { get; }

    /// <summary>Per-routed-expert F32 up_proj device pointers — length <see cref="NumExperts"/>.</summary>
    public nint[] UpProj { get; }

    /// <summary>Per-routed-expert F32 down_proj device pointers — length <see cref="NumExperts"/>.</summary>
    public nint[] DownProj { get; }

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

    /// <summary>Constructs a fully-populated MoE layer weight bundle.</summary>
    public CudaMoeLayerWeights(
        int numExperts, int numExpertsPerTok, int hiddenSize, int moeIntermediateSize,
        bool normTopKProb,
        nint router,
        nint[] gateProj, nint[] upProj, nint[] downProj,
        int numSharedExperts, int sharedIntermediateSize,
        nint[]? sharedGateProj, nint[]? sharedUpProj, nint[]? sharedDownProj,
        nint sharedExpertGate)
    {
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (numExpertsPerTok <= 0 || numExpertsPerTok > numExperts)
            throw new ArgumentOutOfRangeException(nameof(numExpertsPerTok));
        if (gateProj.Length != numExperts || upProj.Length != numExperts || downProj.Length != numExperts)
            throw new ArgumentException("Per-expert pointer arrays must have numExperts entries.");
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
    }
}
