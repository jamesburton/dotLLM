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

    // ── Vulkan-only quant overlay ──────────────────────────────────────────────
    // Production loaders today upcast every MoE projection to F32, so these fields are
    // unused on production paths; tests populate them to exercise the Vulkan quantised
    // GEMV/GEMM path for the non-indexed MoE matmuls (router gate + shared-expert
    // gate/up/down + optional Qwen1.5-MoE shared-expert sigmoid gate). The CPU forward
    // continues to consume the corresponding F32 arrays (Gate, SharedGateProj, etc.); the
    // F32 arrays must hold values equivalent to dequantising the raw quant bytes so the
    // Vulkan vs CPU comparison is fair. The per-routed-expert W1/W2/W3 banks deliberately
    // have NO quant overlay here — the Vulkan moe_indexed_matmul_f32 kernel is F32-only
    // in tree, so a quantised indexed variant is future work. Same two-mode storage
    // policy as the standard transformer: when the source is Q8_0 (contraction axis a
    // multiple of 32) or Q4_K / Q5_K / Q6_K (a multiple of 256), raw blocks live on
    // device and dispatch via the matching matmul kernel; otherwise the Vulkan upload
    // dequantises to F32.
    //
    // The overlay slots use the historical "Q8" naming because Q8_0 was the first quant
    // type wired through. They actually carry raw bytes for whichever format the
    // companion `*QuantTypeOverlay` field declares — Q8_0, Q4_K, Q5_K, or Q6_K (Phase 1
    // of the K-quant work, now complete for the Vulkan matmul kernels — coopmat variants
    // and the remaining K-quant formats (Q2_K, Q3_K) remain follow-up tickets).

    /// <summary>Optional raw-quant bytes for the router gate ([numExperts, hiddenSize]).
    /// Zero when the gate stays F32 on device. When non-zero, <see cref="Gate"/> still
    /// holds an F32 array for the CPU oracle (must match the dequant of the raw bytes).
    /// The format is declared by <see cref="GateQuantTypeOverlay"/>.</summary>
    public nint GateQ8Ptr;
    /// <summary>Storage type of the router-gate raw-byte overlay (<see cref="GateQ8Ptr"/>).
    /// One of <see cref="QuantizationType.Q8_0"/>, <see cref="QuantizationType.Q4_K"/>,
    /// <see cref="QuantizationType.Q5_K"/>, or <see cref="QuantizationType.Q6_K"/>;
    /// <see cref="QuantizationType.F32"/> when no overlay is present.</summary>
    public QuantizationType GateQuantTypeOverlay;

    /// <summary>Optional raw-quant byte pointers for the per-shared-expert gate_proj
    /// ([sharedIntermediateSize, hiddenSize]). Null or empty when no overlay; otherwise
    /// length must equal <see cref="NumSharedExperts"/>. Format declared by
    /// <see cref="SharedExpertProjQuantTypeOverlay"/>.</summary>
    public nint[]? SharedGateProjQ8Ptrs;
    /// <summary>Optional raw-quant byte pointers for the per-shared-expert up_proj.</summary>
    public nint[]? SharedUpProjQ8Ptrs;
    /// <summary>Optional raw-quant byte pointers for the per-shared-expert down_proj
    /// ([hiddenSize, sharedIntermediateSize]).</summary>
    public nint[]? SharedDownProjQ8Ptrs;
    /// <summary>Storage type of the shared-expert projection overlay arrays. All three
    /// arrays share one quant type (uniform across the shared-expert branch).
    /// One of <see cref="QuantizationType.Q8_0"/>, <see cref="QuantizationType.Q4_K"/>,
    /// <see cref="QuantizationType.Q5_K"/>, or <see cref="QuantizationType.Q6_K"/>;
    /// <see cref="QuantizationType.F32"/> when no overlay is present.</summary>
    public QuantizationType SharedExpertProjQuantTypeOverlay;

    /// <summary>Optional raw-quant bytes for the Qwen1.5-MoE shared-expert sigmoid gate
    /// ([1, hiddenSize] — the Vulkan side stores it as a one-row matrix). Null when no
    /// overlay is present (matches <see cref="SharedExpertGate"/> being F32-only).
    /// Format declared by <see cref="SharedExpertGateQuantTypeOverlay"/>.</summary>
    public nint SharedExpertGateQ8Ptr;
    /// <summary>Storage type of the shared-expert gate overlay (<see cref="SharedExpertGateQ8Ptr"/>).
    /// One of <see cref="QuantizationType.Q8_0"/>, <see cref="QuantizationType.Q4_K"/>,
    /// <see cref="QuantizationType.Q5_K"/>, or <see cref="QuantizationType.Q6_K"/>;
    /// <see cref="QuantizationType.F32"/> when no overlay is present.</summary>
    public QuantizationType SharedExpertGateQuantTypeOverlay;

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

    // ── BitNet ternary (I2_S) routed-expert banks (CPU) ─────────────────────────
    // Populated by the safetensors BitNet-MoE loader (LoadBitNetMoeLayer). The
    // per-expert {gate,up,down}_proj are ternary I2_S, laid out as CONTIGUOUS
    // packed-trit banks (payload only, NO inline tail scale) with a parallel
    // per-expert absmean scale vector — the exact shape MatMul.MoeIndexedMatmulI2_S
    // consumes. The BitNet expert body differs from SwiGLU: it is
    // down( ffn_sub_norm( relu2(gate(x)) * up(x) ) ), so the per-expert
    // ffn_sub_norm RMSNorm weights live here too. The router (Gate/GateBias) stays
    // F32. Mutable (set post-construction) — same policy as the Vulkan quant overlay.

    /// <summary>
    /// Quant type of the routed experts. <see cref="QuantizationType.F32"/> (default)
    /// selects the SwiGLU path over the <see cref="W1"/>/<see cref="W2"/>/<see cref="W3"/>
    /// F32 pointers. <see cref="QuantizationType.I2_S"/> selects the BitNet-MoE path over
    /// the packed ternary banks below (relu2 + per-expert <see cref="ExpertFfnSubNorm"/>).
    /// <para><b>Q4_K/Q8_0 extension point:</b> add the new quant type here and a matching
    /// indexed-matmul kernel dispatch in the BitNet-MoE forward; the loader and this bundle
    /// already carry per-expert base+stride banks and a per-expert scale/format field.</para>
    /// </summary>
    public QuantizationType RoutedExpertQuantType = QuantizationType.F32;

    /// <summary>Contiguous packed-trit bank base for <c>gate_proj</c> (payload only, no tail
    /// scale). Expert <c>e</c> lives at <c>GateExpsI2SBase + e*GateExpsI2SRowBytes</c>. 0 when
    /// the routed experts are not I2_S.</summary>
    public nint GateExpsI2SBase;
    /// <summary>Byte stride between consecutive I2_S <c>gate_proj</c> expert banks (= <c>I·H/4</c>).</summary>
    public long GateExpsI2SRowBytes;
    /// <summary>Per-expert absmean α for <c>gate_proj</c> [numExperts]. Null when not I2_S.</summary>
    public float[]? GateExpsI2SScales;

    /// <summary>Contiguous packed-trit bank base for <c>up_proj</c>. See <see cref="GateExpsI2SBase"/>.</summary>
    public nint UpExpsI2SBase;
    /// <summary>Byte stride between consecutive I2_S <c>up_proj</c> expert banks (= <c>I·H/4</c>).</summary>
    public long UpExpsI2SRowBytes;
    /// <summary>Per-expert absmean α for <c>up_proj</c> [numExperts].</summary>
    public float[]? UpExpsI2SScales;

    /// <summary>Contiguous packed-trit bank base for <c>down_proj</c>. See <see cref="GateExpsI2SBase"/>.</summary>
    public nint DownExpsI2SBase;
    /// <summary>Byte stride between consecutive I2_S <c>down_proj</c> expert banks (= <c>H·I/4</c>).</summary>
    public long DownExpsI2SRowBytes;
    /// <summary>Per-expert absmean α for <c>down_proj</c> [numExperts].</summary>
    public float[]? DownExpsI2SScales;

    /// <summary>Per-expert BitNet FFN Sub-LN weight, <c>[numExperts][moeIntermediateSize]</c>.
    /// Applied as an RMSNorm over the gated intermediate <c>relu2(gate)*up</c> before
    /// <c>down_proj</c>, per the expert that produced the row. Null for non-BitNet MoE.</summary>
    public float[][]? ExpertFfnSubNorm;

    /// <summary>Optional router bias <c>[numExperts]</c> added to the gate logits before
    /// softmax/top-k. Required for identity-MoTE (top-1 selection is bias-shifted) and
    /// Qwen3 aux-loss-free routing; harmless (null) elsewhere.</summary>
    public float[]? GateBias;

    /// <summary>True when the routed experts are ternary I2_S (BitNet-MoE forward path).</summary>
    public bool IsBitNetI2S => RoutedExpertQuantType == QuantizationType.I2_S;

    // ── Quantized-expert CPU path (gpt-oss) ───────────────────────────────
    // When UseQuantExperts is true the CPU forward runs
    // DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp directly on the raw GGUF views
    // (GateExpsRaw / UpExpsRaw / DownExpsRaw + their quant types) instead of
    // the F32 W1/W2/W3 banks — no F32 host inflation. The W1/W2/W3 arrays are
    // zero-filled placeholders in this mode.

    /// <summary>True = CPU forward consumes the raw quantized expert banks via
    /// <c>MoeQuantSwiGluMlp</c> (gpt-oss). False = classic F32 W1/W2/W3 path.</summary>
    public bool UseQuantExperts;

    /// <summary>Optional router bias [NumExperts] (gpt-oss <c>ffn_gate_inp.bias</c>).</summary>
    public float[]? RouterBias;

    /// <summary>Optional per-expert gate bias, flat [NumExperts × IntermediateSize].</summary>
    public float[]? GateExpsBias;

    /// <summary>Optional per-expert up bias, flat [NumExperts × IntermediateSize].</summary>
    public float[]? UpExpsBias;

    /// <summary>Optional per-expert down bias, flat [NumExperts × HiddenSize].</summary>
    public float[]? DownExpsBias;

    /// <summary>True = clamped swiglu_oai activation (gpt-oss); false = plain SwiGLU.</summary>
    public bool UseSwiGluOai;

    /// <summary>True = softmax over the selected top-k raw logits (gpt-oss);
    /// false = Mixtral softmax-then-topk gating.</summary>
    public bool SoftmaxAfterTopK;

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

        // Q8_0 / K-quant overlays default to F32 / null — production loaders never set them; tests
        // populate them post-construction to exercise the Vulkan quant matmul path.
        GateQ8Ptr = 0;
        GateQuantTypeOverlay = QuantizationType.F32;
        SharedGateProjQ8Ptrs = null;
        SharedUpProjQ8Ptrs = null;
        SharedDownProjQ8Ptrs = null;
        SharedExpertProjQuantTypeOverlay = QuantizationType.F32;
        SharedExpertGateQ8Ptr = 0;
        SharedExpertGateQuantTypeOverlay = QuantizationType.F32;

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
/// Per-layer Gemma-4 MoE extras: the dual-<i>parallel</i> FFN's extra norms,
/// the custom router scale, the per-expert down-projection scale, and the
/// per-layer output scale. Present only on a Gemma-4 (<c>gemma4</c> /
/// DiffusionGemma) layer; null on every other architecture. See
/// <c>docs/diffusiongemma/GEMMA4-GRAPH-SPEC.md</c>.
/// </summary>
/// <remarks>
/// The dense MLP's <c>gate</c>/<c>up</c>/<c>down</c> projections live in the
/// standard <see cref="TransformerLayerWeights.GateWeight"/> /
/// <see cref="TransformerLayerWeights.UpWeight"/> /
/// <see cref="TransformerLayerWeights.DownWeight"/> slots, and the MoE experts
/// in <see cref="TransformerLayerWeights.Moe"/>. <see cref="TransformerLayerWeights.FfnNormWeight"/>
/// holds the dense branch's <c>ffn_norm</c>. This bundle carries the four
/// remaining FFN norms plus the scalars.
/// </remarks>
internal sealed class Gemma4LayerWeights
{
    /// <summary>MoE branch pre-norm <c>pre_ffw_norm_2</c> [hiddenSize] — RMSNorm'd attn_out fed to the experts. Null on a dense (non-MoE) Gemma-4 layer (E2B/E4B).</summary>
    public float[]? PreFfwNorm2;
    /// <summary>Dense branch post-norm <c>post_ffw_norm_1</c> [hiddenSize] — applied to the dense MLP output. Null on a dense (non-MoE) layer, whose single FFN output goes straight to <see cref="PostFfwNorm"/>.</summary>
    public float[]? PostFfwNorm1;
    /// <summary>MoE branch post-norm <c>post_ffw_norm_2</c> [hiddenSize] — applied to the MoE output. Null on a dense layer.</summary>
    public float[]? PostFfwNorm2;
    /// <summary>Combined post-norm <c>post_ffw_norm</c> [hiddenSize] — wraps (dense + MoE) before the residual add. On dense layers wraps the single FFN output.</summary>
    public required float[] PostFfwNorm;
    /// <summary>Custom-router channel scale <c>ffn_gate_inp.scale</c> [hiddenSize]. Null on a dense layer.</summary>
    public float[]? RouterScale;
    /// <summary>Per-expert down-projection scale <c>ffn_down_exps.scale</c> [numExperts]. Null on a dense layer.</summary>
    public float[]? DownExpertScale;
    /// <summary>Per-layer output scale <c>layer_output_scale</c> — single scalar applied as the LAST per-layer op (canvas rows on diffusion-gemma; ALL rows on gemma4).</summary>
    public required float LayerOutputScale;
    /// <summary>
    /// DiffusionGemma-only per-layer encoder output scale <c>enc_layer_output_scale</c> —
    /// single scalar applied as the LAST per-layer op to the PROMPT rows [0, P) of the
    /// unified [prompt | canvas] forward. <see langword="null"/> on autoregressive gemma4
    /// (the tensor is absent), in which case every row uses <see cref="LayerOutputScale"/>.
    /// </summary>
    public float? EncLayerOutputScale;
    /// <summary>True on a V-less (global/full-attention) layer where V branches off the RAW K projection.</summary>
    public required bool VFromK;

    /// <summary>
    /// Per-expert byte stride for the fused <c>ffn_gate_up_exps</c> bank (= the full
    /// <c>2*Ie</c>-row slab). Both the gate raw view (offset 0) and the up raw view
    /// (offset <c>Ie</c> rows, stored on <see cref="MoeLayerWeights.UpExpsRaw"/>)
    /// step by this stride. NOT the kernel-default <c>Ie*rowBytes</c> — the two
    /// projections are interleaved per expert in one tensor. Zero on a dense layer.
    /// </summary>
    public long GateUpExpsRowBytes;
    /// <summary>Per-expert byte stride for the <c>ffn_down_exps</c> bank (= <c>hidden</c> rows). Zero on a dense layer.</summary>
    public long DownExpsRowBytes;
}

/// <summary>
/// Model-level DiffusionGemma self-conditioning (SC) weights. Present ONLY on the
/// diffusion-gemma GGUF (absent on autoregressive gemma4, where SC is meaningless).
/// SC feeds the PREVIOUS denoise step's canvas logits back into the canvas region
/// embedding via a gated GeGLU MLP over a soft token-embedding of those logits
/// (<c>diffusion-gemma.cpp</c> <c>dg_canvas_embed</c>):
/// <code>
/// soft   = sqrt(n_embd) * Σ_v softmax(prev_logits)[v] * tok_embd[v]   // [n_embd] per canvas col
/// normed = rms_norm(soft, eps) * self_cond_pre_norm                   // [n_embd]
/// sc_sig = self_cond_down( gelu_tanh(self_cond_gate·normed) * (self_cond_up·normed) )  // [n_embd]
/// canvas = rms_noscale(canvas + sc_sig)
/// </code>
/// The gate/up projections widen to the DENSE feed-forward width (n_ff = 2112), and
/// down projects back to n_embd. Weights stay as raw mmap pointers + quant type
/// (the same dequant GEMV primitive serves them); the pre-norm is dequantized to
/// <c>float[]</c> at load. Null on every non-diffusion-gemma model.
/// </summary>
internal sealed class Gemma4SelfCondWeights
{
    /// <summary>RMSNorm weight <c>self_cond_pre_norm.weight</c> [n_embd] (F32) — applied (with scale) to the soft-embedding.</summary>
    public required float[] PreNorm;

    /// <summary>Gate projection <c>self_cond_gate.weight</c>: ptr, quant, out dim (n_ff), in dim (n_embd).</summary>
    public required nint GatePtr;
    public required QuantizationType GateQt;
    public required int GateOut;
    public required int GateIn;

    /// <summary>Up projection <c>self_cond_up.weight</c>: ptr, quant, out dim (n_ff), in dim (n_embd).</summary>
    public required nint UpPtr;
    public required QuantizationType UpQt;
    public required int UpOut;
    public required int UpIn;

    /// <summary>Down projection <c>self_cond_down.weight</c>: ptr, quant, out dim (n_embd), in dim (n_ff).</summary>
    public required nint DownPtr;
    public required QuantizationType DownQt;
    public required int DownOut;
    public required int DownIn;
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

    /// <summary>Optional attention sub-norm weight [hiddenSize]. Applied to the attention output before the output projection (BitNet Sub-LN). Null when absent.</summary>
    public readonly float[]? AttnSubNormWeight;
    /// <summary>Optional FFN sub-norm weight [intermediateSize]. Applied to the gated FFN intermediate before the down projection (BitNet Sub-LN). Null when absent.</summary>
    public readonly float[]? FfnSubNormWeight;

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

    /// <summary>
    /// Pre-FFN RMSNorm weight [hiddenSize]. For the standard two-norm layout
    /// (Llama/Mistral/Qwen/…) this is the <c>post_attention_layernorm</c>. For
    /// Gemma's four-norm layout this is the <c>pre_feedforward_layernorm</c> (the
    /// <c>post_attention_layernorm</c> moves to <see cref="PostAttnNormWeight"/>).
    /// </summary>
    public readonly float[] FfnNormWeight;

    /// <summary>
    /// Optional Gemma <c>post_attention_layernorm</c> weight [hiddenSize], applied
    /// to the attention sublayer output BEFORE the residual add. Null on every
    /// non-Gemma architecture (which keeps the standard two-norm residual layout).
    /// </summary>
    public readonly float[]? PostAttnNormWeight;

    /// <summary>
    /// Optional Gemma <c>post_feedforward_layernorm</c> weight [hiddenSize], applied
    /// to the FFN sublayer output BEFORE the residual add. Null on every non-Gemma
    /// architecture.
    /// </summary>
    public readonly float[]? PostFfnNormWeight;

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

    /// <summary>
    /// Optional per-head attention-sink logits [numHeads] (gpt-oss
    /// <c>attn_sinks.weight</c>). When non-null each head's attention softmax
    /// denominator additionally includes <c>exp(sink[h] - max)</c>. Null for
    /// architectures without sinks (zero overhead).
    /// </summary>
    public readonly float[]? AttnSinks;

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

    /// <summary>
    /// Non-null on a Gemma-4 MoE layer — the dual-parallel-FFN extra norms,
    /// custom-router scale, per-expert down scale, and per-layer output scale.
    /// Null on every other architecture (the standard FFN/MoE path runs). See
    /// <see cref="Gemma4LayerWeights"/>.
    /// </summary>
    public readonly Gemma4LayerWeights? Gemma4;

    // ──────────────────── Per-Layer Embeddings (PLE) ────────────────────
    // Gemma-4 dense text tower (E2B/E4B) only. Non-zero/non-null iff the model
    // config carries PerLayerEmbedding. The forward pass, after the MLP residual
    // add, computes gate→gelu_tanh→(× per-layer input)→proj→post-norm→+residual.
    // All F32 (upcast at load); dims derive from the config (pleDim / hidden).

    /// <summary>PLE <c>per_layer_input_gate.weight</c> [pleDim, hidden] F32. Zero when absent.</summary>
    public readonly nint PleGateWeight;
    /// <summary>PLE <c>per_layer_projection.weight</c> [hidden, pleDim] F32. Zero when absent.</summary>
    public readonly nint PleProjWeight;
    /// <summary>PLE <c>post_per_layer_input_norm.weight</c> [hidden] ((1+w) absorbed). Null when absent.</summary>
    public readonly float[]? PlePostNormWeight;

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
        MlaLayerWeights? mla = null,
        float[]? postAttnNormWeight = null, float[]? postFfnNormWeight = null,
        Gemma4LayerWeights? gemma4 = null,
        float[]? attnSubNormWeight = null, float[]? ffnSubNormWeight = null,
        nint pleGateWeight = 0, nint pleProjWeight = 0, float[]? plePostNormWeight = null,
        float[]? attnSinks = null)
    {
        AttnNormWeight = attnNormWeight;
        QNormWeight = qNormWeight;
        KNormWeight = kNormWeight;
        AttnSubNormWeight = attnSubNormWeight;
        FfnSubNormWeight = ffnSubNormWeight;
        QWeight = qWeight; QQuantType = qQuantType; QOutputDim = qOutputDim; QInputDim = qInputDim; QBias = qBias;
        KWeight = kWeight; KQuantType = kQuantType; KOutputDim = kOutputDim; KInputDim = kInputDim; KBias = kBias;
        VWeight = vWeight; VQuantType = vQuantType; VOutputDim = vOutputDim; VInputDim = vInputDim; VBias = vBias;
        OWeight = oWeight; OQuantType = oQuantType; OOutputDim = oOutputDim; OInputDim = oInputDim; OBias = oBias;
        FfnNormWeight = ffnNormWeight;
        PostAttnNormWeight = postAttnNormWeight;
        PostFfnNormWeight = postFfnNormWeight;
        GateWeight = gateWeight; GateQuantType = gateQuantType; GateOutputDim = gateOutputDim; GateInputDim = gateInputDim; GateBias = gateBias;
        UpWeight = upWeight; UpQuantType = upQuantType; UpOutputDim = upOutputDim; UpInputDim = upInputDim; UpBias = upBias;
        DownWeight = downWeight; DownQuantType = downQuantType; DownOutputDim = downOutputDim; DownInputDim = downInputDim; DownBias = downBias;
        Moe = moe;
        Mla = mla;
        Gemma4 = gemma4;
        PleGateWeight = pleGateWeight;
        PleProjWeight = pleProjWeight;
        PlePostNormWeight = plePostNormWeight;
        AttnSinks = attnSinks;
    }
}

/// <summary>
/// Model-level Per-Layer Embeddings (PLE) weight bundle for the Gemma-4 dense text
/// tower (E2B/E4B). Non-null on <see cref="TransformerWeights.PerLayerEmbedding"/>
/// only when the checkpoint ships the PLE tables. The per-layer gate/projection/norm
/// live on <see cref="TransformerLayerWeights"/>; this holds the two model-level
/// tensors used once per forward to build the per-layer input tensor.
/// </summary>
internal sealed class PerLayerEmbeddingWeights
{
    /// <summary><c>embed_tokens_per_layer.weight</c> pointer [vocabPle, numLayers*pleDim].
    /// Kept at its native quant type + gathered per token (the full table is huge —
    /// never upcast wholesale).</summary>
    public required nint EmbedTokensPerLayer { get; init; }
    /// <summary>Quant type of <see cref="EmbedTokensPerLayer"/>.</summary>
    public required QuantizationType EmbedTokensPerLayerQt { get; init; }

    /// <summary><c>per_layer_model_projection.weight</c> [numLayers*pleDim, hidden] F32.</summary>
    public required nint ModelProjection { get; init; }

    /// <summary><c>per_layer_projection_norm.weight</c> [pleDim] ((1+w) absorbed).</summary>
    public required float[] ProjectionNorm { get; init; }

    /// <summary>Per-layer embedding dimension (<c>hidden_size_per_layer_input</c>).</summary>
    public required int PerLayerDim { get; init; }
    /// <summary>Per-layer embedding vocabulary (<c>vocab_size_per_layer_input</c>).</summary>
    public required int VocabSize { get; init; }
    /// <summary>Number of decoder layers (row width of the PLE table = NumLayers*PerLayerDim).</summary>
    public required int NumLayers { get; init; }
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

    /// <summary>
    /// Model-level DiffusionGemma self-conditioning weights. Non-null ONLY when the
    /// GGUF carries the <c>self_cond_*</c> tensors (diffusion-gemma); null otherwise.
    /// </summary>
    public Gemma4SelfCondWeights? SelfCond { get; }

    /// <summary>
    /// Model-level Per-Layer Embeddings (PLE) weights (Gemma-4 dense text tower,
    /// E2B/E4B). Non-null only when the checkpoint ships the PLE tables; null for
    /// every other architecture. When set, the forward pass builds the per-layer
    /// input tensor once after the embedding lookup and injects a gated residual
    /// into each decoder layer via the per-layer PLE slots on
    /// <see cref="TransformerLayerWeights"/>.
    /// </summary>
    public PerLayerEmbeddingWeights? PerLayerEmbedding { get; }

    /// <summary>
    /// Optional proportional-rope per-pair frequency factors (<c>rope_freqs.weight</c>,
    /// length = global rotated dim / 2). Gemma-4 E2B/E4B applies them on the
    /// full-attention layers (ggml <c>theta / freq_factors[i]</c>); the sliding
    /// layers and every other architecture leave this null. Folded into the
    /// global cos/sin table at model construction.
    /// </summary>
    public float[]? RopeFreqFactors { get; private set; }

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

    /// <summary>
    /// Live subset of <see cref="_ownedAllocations"/> — the owned host buffers that
    /// have NOT yet been freed. Built once from <see cref="_ownedAllocations"/> at
    /// construction. The direct-to-device GPU upload path removes entries as it
    /// frees them per-tensor (<see cref="TryReleaseOwnedHostAllocation"/>); anything
    /// still present is freed by <see cref="Dispose"/>. Using a set (not the list)
    /// as the free source makes early release and final disposal mutually exclusive,
    /// so a streamed buffer is never double-freed. Null iff <see cref="_ownedAllocations"/>
    /// is null (pure-mmap GGUF load with nothing to own).
    /// </summary>
    private readonly HashSet<nint>? _liveOwnedAllocations;

    private TransformerWeights(
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType, int vocabSize, int hiddenSize,
        TransformerLayerWeights[] layers,
        float[] outputNormWeight,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim,
        List<nint>? ownedAllocations = null,
        Gemma4SelfCondWeights? selfCond = null,
        PerLayerEmbeddingWeights? perLayerEmbedding = null)
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
        _liveOwnedAllocations = ownedAllocations is not null
            ? new HashSet<nint>(ownedAllocations)
            : null;
        SelfCond = selfCond;
        PerLayerEmbedding = perLayerEmbedding;
    }

    /// <summary>
    /// Number of loader-owned host allocations still live (not yet freed). Used by
    /// the direct-to-device upload path and tests to observe streamed releases. Zero
    /// for a pure-mmap GGUF load (nothing owned).
    /// </summary>
    internal int LiveOwnedAllocationCount => _liveOwnedAllocations?.Count ?? 0;

    /// <summary>
    /// Releases a single loader-owned host allocation early — used by the GPU
    /// weight-upload path (direct-to-device streaming) to free each host scratch
    /// buffer immediately after its synchronous host→device copy has completed,
    /// instead of holding the entire host weight set resident until
    /// <see cref="Dispose"/>. This roughly halves the transient CPU-RAM peak on the
    /// GPU load path, where the host copy and the device copy would otherwise coexist
    /// for the whole model.
    /// <para>
    /// <paramref name="hostPtr"/> is freed ONLY when it is a tracked owned allocation
    /// (a bf16/f16 → F32 upcast, or an I2_S ternary-packed buffer). Memory-mapped
    /// zero-copy views (F32 safetensors tensors, GGUF mmap pointers) are NOT owned and
    /// are silently ignored — freeing them would corrupt the mmap and is never done.
    /// </para>
    /// <para>
    /// Idempotent and memory-safe: a pointer already released (or never owned) returns
    /// <c>false</c> without touching memory, so a duplicate call cannot double-free, and
    /// <see cref="Dispose"/> will not re-free a streamed buffer (it drains the same live
    /// set). The caller MUST guarantee the host bytes have already been fully consumed by
    /// the device copy before calling — for CUDA that is the synchronous
    /// <c>cuMemcpyHtoD_v2</c>, which blocks until the transfer completes; the subsequent
    /// on-device dequant kernels read device memory only and never the freed host buffer.
    /// </para>
    /// </summary>
    /// <param name="hostPtr">A host pointer previously handed to the upload path.</param>
    /// <returns><c>true</c> if this call freed an owned allocation; otherwise <c>false</c>.</returns>
    internal unsafe bool TryReleaseOwnedHostAllocation(nint hostPtr)
    {
        if (hostPtr == nint.Zero || _liveOwnedAllocations is null)
            return false;
        if (!_liveOwnedAllocations.Remove(hostPtr))
            return false;
        NativeMemory.AlignedFree((void*)hostPtr);
        return true;
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
        List<nint> ownedAllocations,
        PerLayerEmbeddingWeights? perLayerEmbedding = null)
    {
        return new TransformerWeights(
            tokenEmbedWeight, tokenEmbedQt, vocabSize, hiddenSize,
            layers,
            outputNormWeight,
            outputWeight, outputQt, outputM, outputK,
            ownedAllocations,
            selfCond: null,
            perLayerEmbedding: perLayerEmbedding);
    }

    /// <summary>
    /// Loads all weight references from an opened GGUF file.
    /// Norm weights are dequantized to <c>float[]</c>. Linear projections stay as mmap pointers.
    /// </summary>
    /// <param name="gguf">Opened GGUF file.</param>
    /// <param name="config">Model configuration.</param>
    /// <param name="skipF32MoeDequant">When true, skip the per-expert F32 host
    /// dequant of the MoE 3D-stacked tensors (still populates raw GGUF mmap
    /// views so the GPU loader can take its zero-copy path). Used by GPU-only
    /// callers — the F32 dequants are dead weight when the CPU MoE oracle
    /// isn't called, and they blow ~2.2 GB host RAM per V2-Lite Q4_K_M layer.
    /// CPU-only callers leave this false to keep <see cref="DotLLM.Cpu.Kernels.MoeSwiGluMlp.Execute"/>
    /// callable.</param>
    public static TransformerWeights LoadFromGguf(GgufFile gguf, ModelConfig config,
                                                    bool skipF32MoeDequant = false)
    {
        nint dataBase = gguf.DataBasePointer;
        var tensors = gguf.TensorsByName;

        // Token embeddings
        var embDesc = tensors["token_embd.weight"];
        nint embPtr = dataBase + (nint)embDesc.DataOffset;

        // MLA (DeepSeek-V2/V3) loads its projection tensors as F32 dequant
        // buffers since the CPU MlaAttention.Execute oracle is F32-only. Track
        // them on the loader so Dispose can free them. Empty for non-MLA models.
        // Gemma 4 dequantizes its small per-layer scalar tensors (router scale,
        // per-expert down scale, layer_output_scale) into managed float[] and
        // keeps the big expert banks as raw mmap views; its dense-PLE variant
        // (E2B/E4B) additionally owns F32 upcasts of the PLE projections when
        // they are not stored as F32 (per_layer_model_proj ships BF16).
        var owned = config.MlaConfig is not null || config.PerLayerEmbedding is not null
            ? new List<nint>()
            : null;

        // Per-layer weights
        var layers = new TransformerLayerWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = config.MlaConfig is not null
                ? LoadMlaLayer(i, dataBase, tensors, config, owned!, skipF32MoeDequant)
                : config.Gemma4DualFfn
                    ? LoadGemma4Layer(i, dataBase, tensors, config, owned)
                    : LoadLayer(i, dataBase, tensors, config);
        }

        // Output norm
        var outNormDesc = tensors["output_norm.weight"];
        float[] outputNormWeight = DequantizeNorm(dataBase, outNormDesc, config.HiddenSize);

        // DiffusionGemma self-conditioning (model-level). On denoise steps > 0 the
        // canvas region embed becomes rms_noscale(scaled_embed + sc_sig(prev_logits)),
        // where sc_sig is a gated GeGLU MLP over a soft token-embedding of the previous
        // step's canvas logits (diffusion-gemma.cpp dg_canvas_embed). Present ONLY on
        // the diffusion-gemma GGUF — absent ⇒ SelfCond stays null and the forward keeps
        // the byte-identical zero-SC path. The pre-norm dequantizes to float[]; the
        // gate/up/down projections stay as raw mmap pointers (same dequant GEMV serves them).
        Gemma4SelfCondWeights? selfCond = null;
        if (tensors.TryGetValue("self_cond_pre_norm.weight", out var scPreDesc)
            && tensors.TryGetValue("self_cond_gate.weight", out var scGateDesc)
            && tensors.TryGetValue("self_cond_up.weight", out var scUpDesc)
            && tensors.TryGetValue("self_cond_down.weight", out var scDownDesc))
        {
            float[] scPreNorm = DequantizeNorm(dataBase, scPreDesc, config.HiddenSize);
            var (scGatePtr, scGateQt, scGateM, scGateK) = LoadLinear(dataBase, scGateDesc);
            var (scUpPtr, scUpQt, scUpM, scUpK) = LoadLinear(dataBase, scUpDesc);
            var (scDownPtr, scDownQt, scDownM, scDownK) = LoadLinear(dataBase, scDownDesc);
            selfCond = new Gemma4SelfCondWeights
            {
                PreNorm = scPreNorm,
                GatePtr = scGatePtr, GateQt = scGateQt, GateOut = scGateM, GateIn = scGateK,
                UpPtr = scUpPtr, UpQt = scUpQt, UpOut = scUpM, UpIn = scUpK,
                DownPtr = scDownPtr, DownQt = scDownQt, DownOut = scDownM, DownIn = scDownK,
            };
        }

        // Per-Layer Embeddings (PLE) — Gemma-4 dense text tower (E2B/E4B) GGUF.
        // Model-level tensors mirror llama.cpp gemma4.cpp load_arch_tensors:
        //   per_layer_token_embd.weight [pleDim*L, vocab]  — kept at native quant,
        //     gathered per token (huge table; never upcast wholesale);
        //   per_layer_model_proj.weight [hidden, pleDim*L] — F32 for the CPU
        //     GemmF32 kernel (BF16 in the released E4B → owned upcast);
        //   per_layer_proj_norm.weight  [pleDim]           — plain weights (Gemma-4
        //     GGUF stores final norm weights, no +1).
        PerLayerEmbeddingWeights? perLayerEmbedding = null;
        if (config.PerLayerEmbedding is PerLayerEmbeddingConfig pleCfg)
        {
            int lp = config.NumLayers * pleCfg.PerLayerDim;
            var pleTokDesc = tensors["per_layer_token_embd.weight"];
            if (pleTokDesc.Shape[0] != lp)
                throw new InvalidDataException(
                    $"per_layer_token_embd.weight row width {pleTokDesc.Shape[0]} != numLayers*pleDim ({lp}).");
            var pleProjDesc = tensors["per_layer_model_proj.weight"];
            nint pleProjPtr = pleProjDesc.QuantizationType == QuantizationType.F32
                ? dataBase + (nint)pleProjDesc.DataOffset      // zero-copy mmap view
                : DequantToF32(dataBase, pleProjDesc, (long)lp * config.HiddenSize, owned!);
            float[] pleProjNorm = DequantizeNorm(
                dataBase, tensors["per_layer_proj_norm.weight"], pleCfg.PerLayerDim);

            perLayerEmbedding = new PerLayerEmbeddingWeights
            {
                EmbedTokensPerLayer = dataBase + (nint)pleTokDesc.DataOffset,
                EmbedTokensPerLayerQt = pleTokDesc.QuantizationType,
                ModelProjection = pleProjPtr,
                ProjectionNorm = pleProjNorm,
                PerLayerDim = pleCfg.PerLayerDim,
                VocabSize = pleCfg.VocabSize,
                NumLayers = config.NumLayers,
            };
        }

        // Proportional-rope frequency factors (rope_freqs.weight, Gemma-4 E2B/E4B
        // full-attention layers; also Llama-3.1-style GGUFs). Optional — absent on
        // every other released model. Length = global rotated dim / 2.
        float[]? ropeFreqFactors = null;
        if (tensors.TryGetValue("rope_freqs.weight", out var ropeFreqsDesc))
        {
            ropeFreqFactors = new float[ropeFreqsDesc.Shape[0]];
            Dequantize.ToFloat32(dataBase + (nint)ropeFreqsDesc.DataOffset,
                ropeFreqsDesc.Shape[0], ropeFreqsDesc.QuantizationType, ropeFreqFactors);
        }

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

        var weights = new TransformerWeights(
            embPtr, embDesc.QuantizationType, config.VocabSize, config.HiddenSize,
            layers,
            outputNormWeight,
            outputPtr, outputQt, outputM, outputK,
            ownedAllocations: owned,
            selfCond: selfCond,
            perLayerEmbedding: perLayerEmbedding);
        weights.RopeFreqFactors = ropeFreqFactors;
        return weights;
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

        // Free only the STILL-LIVE owned allocations. The direct-to-device upload
        // path may have already freed (and removed) some of these per-tensor via
        // TryReleaseOwnedHostAllocation; draining the live set here guarantees each
        // owned buffer is freed exactly once whether or not streaming ran.
        if (_liveOwnedAllocations is not null)
        {
            foreach (var ptr in _liveOwnedAllocations)
            {
                if (ptr != nint.Zero)
                    NativeMemory.AlignedFree((void*)ptr);
            }
            _liveOwnedAllocations.Clear();
        }
        _ownedAllocations?.Clear();
    }

    private static TransformerLayerWeights LoadLayer(
        int layerIdx,
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config)
    {
        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        // Per-attention-type head dim + KV-head count (Gemma 4): full-attention
        // layers may use a distinct GlobalHeadDim / NumGlobalKvHeads. Collapses to
        // the uniform config.HeadDim / config.NumKvHeads for every other
        // architecture, so the fused-QKV split and q/k-norm lengths are unchanged.
        int layerHeadDim = config.GetLayerHeadDim(layerIdx);
        int layerKvHeads = config.NumGlobalKvHeads is int gkv && config.IsFullAttentionLayer(layerIdx)
            ? gkv
            : config.NumKvHeads;

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

            int qDim = config.NumAttentionHeads * layerHeadDim;
            int kvDim = layerKvHeads * layerHeadDim;

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
            int qDim = config.NumAttentionHeads * layerHeadDim;
            int kvDim = layerKvHeads * layerHeadDim;

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
        float[]? qNormWeight = LoadOptionalNorm(dataBase, tensors, $"{prefix}.attn_q_norm.weight", layerHeadDim);
        float[]? kNormWeight = LoadOptionalNorm(dataBase, tensors, $"{prefix}.attn_k_norm.weight", layerHeadDim);

        // Optional attention sub-norm (BitNet Sub-LN): RMSNorm over the attention output [hiddenSize] before o_proj.
        float[]? attnSubNormWeight = LoadOptionalNorm(dataBase, tensors, $"{prefix}.attn_sub_norm.weight", hiddenSize);

        // Optional per-head attention sinks (gpt-oss): F32 [numHeads] scalar logits.
        float[]? attnSinks = LoadOptionalBias(dataBase, tensors, $"{prefix}.attn_sinks.weight");

        // FFN norm — gpt-oss names its pre-FFN norm "post_attention_norm"
        // (llama.cpp LLM_TENSOR_ATTN_POST_NORM); it plays the same role as
        // ffn_norm (applied to the post-attention residual before the FFN/MoE).
        var ffnNormDesc = tensors.TryGetValue($"{prefix}.ffn_norm.weight", out var ffnNormD)
            ? ffnNormD
            : tensors[$"{prefix}.post_attention_norm.weight"];
        float[] ffnNorm = DequantizeNorm(dataBase, ffnNormDesc, hiddenSize);

        // Optional FFN sub-norm (BitNet Sub-LN): RMSNorm over the gated intermediate [intermediateSize] before ffn_down.
        float[]? ffnSubNormWeight = LoadOptionalNorm(dataBase, tensors, $"{prefix}.ffn_sub_norm.weight", config.IntermediateSize);

        // Routed-MoE layer with quantized experts (gpt-oss): the dense
        // ffn_gate/up/down tensors are absent; a 3D-stacked expert block with
        // per-expert biases is loaded instead and consumed by
        // MoeQuantSwiGluMlp straight from the mmap (no F32 inflation).
        if (config.Moe is not null && config.Moe.IsMoeLayer(layerIdx)
            && tensors.ContainsKey($"{prefix}.ffn_gate_exps.weight"))
        {
            MoeLayerWeights quantMoe = LoadQuantExpertMoeLayer(layerIdx, dataBase, tensors, config);
            return new TransformerLayerWeights(
                attnNorm,
                qPtr, qQt, qM, qK,
                kPtr, kQt, kM, kK,
                vPtr, vQt, vM, vK,
                oPtr, oQt, oM, oK,
                ffnNorm,
                gateWeight: 0, gateQuantType: QuantizationType.F32, gateOutputDim: 0, gateInputDim: 0,
                upWeight: 0, upQuantType: QuantizationType.F32, upOutputDim: 0, upInputDim: 0,
                downWeight: 0, downQuantType: QuantizationType.F32, downOutputDim: 0, downInputDim: 0,
                qBias, kBias, vBias, oBias,
                gateBias: null, upBias: null, downBias: null,
                qNormWeight, kNormWeight,
                moe: quantMoe,
                mla: null,
                attnSinks: attnSinks);
        }

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
            qNormWeight, kNormWeight,
            attnSubNormWeight: attnSubNormWeight, ffnSubNormWeight: ffnSubNormWeight,
            attnSinks: attnSinks);
    }

    /// <summary>
    /// Loads one Gemma-4 MoE layer from GGUF. Differs from <see cref="LoadLayer"/>
    /// in five ways (see <c>docs/diffusiongemma/GEMMA4-GRAPH-SPEC.md</c>):
    /// <list type="number">
    ///   <item><b>V-from-K.</b> Full-attention (global) layers carry no
    ///     <c>attn_v.weight</c>; V branches off the raw K projection. <c>wv</c> is
    ///     loaded only when present and the <see cref="Gemma4LayerWeights.VFromK"/>
    ///     flag records the V-less layers.</item>
    ///   <item><b>Dual parallel FFN.</b> A dense GeGLU MLP (<c>ffn_gate</c>/
    ///     <c>ffn_up</c>/<c>ffn_down</c>, width 2112) loaded into the standard
    ///     gate/up/down slots, summed with a 128-expert MoE.</item>
    ///   <item><b>Fused gate_up experts.</b> <c>ffn_gate_up_exps</c> [K, 2*Ie, E]
    ///     split by row offset — gate = rows [0, Ie), up = rows [Ie, 2*Ie) — via
    ///     two raw views into one tensor (per-expert stride = 2*Ie rows).</item>
    ///   <item><b>Custom router + scalars.</b> <c>ffn_gate_inp.scale</c> [hidden],
    ///     <c>ffn_down_exps.scale</c> [E], <c>layer_output_scale</c> [1].</item>
    ///   <item><b>Five FFN norms.</b> <c>ffn_norm</c> (dense), <c>pre_ffw_norm_2</c>
    ///     (MoE in), <c>post_ffw_norm_1</c> (dense out), <c>post_ffw_norm_2</c>
    ///     (MoE out), <c>post_ffw_norm</c> (combined), plus <c>post_attention_norm</c>.</item>
    /// </list>
    /// Norm weights are dequantized plain (NO +1 — Gemma 4 overrides Gemma 3's
    /// <c>(1+w)</c> convention; the GGUF stores the final weights directly).
    /// </summary>
    private static TransformerLayerWeights LoadGemma4Layer(
        int layerIdx,
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config,
        List<nint>? owned)
    {
        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        int layerHeadDim = config.GetLayerHeadDim(layerIdx);
        // MoE is per-layer optional (llama.cpp gemma4.cpp keys off ffn_gate_inp
        // presence): the 26B backbone routes experts on every layer; the dense
        // E2B/E4B tower has none. A layer without the router runs dense-only.
        bool hasExperts = config.Moe is not null
            && tensors.ContainsKey($"{prefix}.ffn_gate_inp.weight");
        var moeCfg = config.Moe;

        // ── Attention ────────────────────────────────────────────────────
        float[] attnNorm = DequantizeNorm(dataBase, tensors[$"{prefix}.attn_norm.weight"], hiddenSize);

        var (qPtr, qQt, qM, qK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_q.weight"]);

        // Shared-KV layers (trailing NumSharedKvLayers on E2B/E4B) never project
        // K/V — they attend over an earlier donor layer's KV. llama.cpp marks
        // their wk/wv TENSOR_NOT_REQUIRED; the released E4B GGUF still ships them
        // but we neither require nor use them.
        bool ownKv = config.LayerHasOwnKv(layerIdx);

        nint kPtr = 0; QuantizationType kQt = QuantizationType.F32; int kM = 0, kK = 0;
        if (ownKv)
            (kPtr, kQt, kM, kK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_k.weight"]);

        // V-from-K: full-attention layers have NO attn_v.weight. When absent we
        // record the flag and zero the V slot; the forward projects V from the
        // raw K projection (gemma4.cpp:241-272).
        bool vFromK;
        nint vPtr = 0; QuantizationType vQt = QuantizationType.F32; int vM = kM, vK = kK;
        if (ownKv && tensors.TryGetValue($"{prefix}.attn_v.weight", out var vDesc))
        {
            (vPtr, vQt, vM, vK) = LoadLinear(dataBase, vDesc);
            vFromK = false;
        }
        else
        {
            vFromK = ownKv; // shared-KV layers have neither their own K nor V
        }

        var (oPtr, oQt, oM, oK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_output.weight"]);

        // QK-norms (head_dim sized; 256 sliding / 512 global). Plain weights (no +1).
        float[]? qNormWeight = LoadOptionalNorm(dataBase, tensors, $"{prefix}.attn_q_norm.weight", layerHeadDim);
        float[]? kNormWeight = LoadOptionalNorm(dataBase, tensors, $"{prefix}.attn_k_norm.weight", layerHeadDim);

        float[] postAttnNorm = DequantizeNorm(dataBase, tensors[$"{prefix}.post_attention_norm.weight"], hiddenSize);

        // ── Dense FFN branch (shared expert) ─────────────────────────────
        float[] ffnNorm = DequantizeNorm(dataBase, tensors[$"{prefix}.ffn_norm.weight"], hiddenSize);
        var (gatePtr, gateQt, gateM, gateK) = LoadLinear(dataBase, tensors[$"{prefix}.ffn_gate.weight"]);
        var (upPtr, upQt, upM, upK) = LoadLinear(dataBase, tensors[$"{prefix}.ffn_up.weight"]);
        var (downPtr, downQt, downM, downK) = LoadLinear(dataBase, tensors[$"{prefix}.ffn_down.weight"]);

        // ── MoE branch (only when this layer routes experts) ──────────────
        MoeLayerWeights? moe = null;
        float[]? preFfwNorm2 = null, postFfwNorm1 = null, postFfwNorm2 = null;
        float[]? routerScale = null, downExpertScale = null;
        long gateUpStride = 0, downStride = 0;
        if (hasExperts)
        {
        int numExperts = moeCfg!.NumExperts;
        int moeIntermediate = moeCfg.MoeIntermediateSize; // 704 (n_ff_exp)

        // Router gate ffn_gate_inp.weight: GGUF [K=hidden, M=numExperts] stored
        // row-major (M outer) == [numExperts, hidden] — exactly the kernel's gate
        // layout. Small → dequant inline to F32.
        var routerDesc = tensors[$"{prefix}.ffn_gate_inp.weight"];
        float[] router = new float[(long)numExperts * hiddenSize];
        Dequantize.ToFloat32(dataBase + (nint)routerDesc.DataOffset,
            (long)numExperts * hiddenSize, routerDesc.QuantizationType, router);

        // Fused gate_up experts ffn_gate_up_exps [K=hidden, 2*Ie, E]: per expert a
        // [2*Ie, hidden] slab, gate = rows [0, Ie), up = rows [Ie, 2*Ie). Expose as
        // two raw views sharing the per-expert stride (= 2*Ie rows). The up view's
        // base is offset by Ie rows.
        var gateUpDesc = tensors[$"{prefix}.ffn_gate_up_exps.weight"];
        if (gateUpDesc.Shape.Rank != 3
            || gateUpDesc.Shape[0] != hiddenSize
            || gateUpDesc.Shape[1] != 2 * moeIntermediate
            || gateUpDesc.Shape[2] != numExperts)
            throw new InvalidDataException(
                $"{prefix}.ffn_gate_up_exps.weight shape {gateUpDesc.Shape[0]}×{gateUpDesc.Shape[1]}×{gateUpDesc.Shape[2]} "
                + $"does not match expected hidden={hiddenSize} × 2*Ie={2 * moeIntermediate} × E={numExperts}.");
        long gateUpRowBytes = Dequantize.RowByteSize(hiddenSize, gateUpDesc.QuantizationType);
        gateUpStride = (long)(2 * moeIntermediate) * gateUpRowBytes; // per-expert slab
        nint gateUpBase = dataBase + (nint)gateUpDesc.DataOffset;
        nint gateExpsRaw = gateUpBase;                                       // gate rows [0, Ie)
        nint upExpsRaw = gateUpBase + (nint)((long)moeIntermediate * gateUpRowBytes); // up rows [Ie, 2*Ie)

        // Down experts ffn_down_exps [K=Ie, M=hidden, E]: per expert [hidden, Ie].
        var downExpsDesc = tensors[$"{prefix}.ffn_down_exps.weight"];
        if (downExpsDesc.Shape.Rank != 3
            || downExpsDesc.Shape[0] != moeIntermediate
            || downExpsDesc.Shape[1] != hiddenSize
            || downExpsDesc.Shape[2] != numExperts)
            throw new InvalidDataException(
                $"{prefix}.ffn_down_exps.weight shape {downExpsDesc.Shape[0]}×{downExpsDesc.Shape[1]}×{downExpsDesc.Shape[2]} "
                + $"does not match expected Ie={moeIntermediate} × hidden={hiddenSize} × E={numExperts}.");
        long downRowBytes = Dequantize.RowByteSize(moeIntermediate, downExpsDesc.QuantizationType);
        downStride = (long)hiddenSize * downRowBytes;
        nint downExpsRaw = dataBase + (nint)downExpsDesc.DataOffset;

        // Empty F32 per-expert pointer arrays — the kernel uses the raw strided
        // views (Q4_K gate/up, Q5_1 down) and never the F32 fallback array.
        var emptyF32 = new nint[numExperts];

        moe = new MoeLayerWeights(
            gate: router,
            w1: emptyF32, w2: emptyF32, w3: emptyF32,
            numExperts: numExperts,
            numExpertsPerTok: moeCfg.NumExpertsPerTok,
            hiddenSize: hiddenSize,
            intermediateSize: moeIntermediate,
            normTopKProb: moeCfg.NormTopKProb,
            sharedGateProj: Array.Empty<nint>(),
            sharedUpProj: Array.Empty<nint>(),
            sharedDownProj: Array.Empty<nint>(),
            sharedIntermediateSize: 0,
            sharedExpertGate: null,
            gateExpsRaw: gateExpsRaw, gateExpsRawQt: gateUpDesc.QuantizationType,
            gateExpsMDim: moeIntermediate, gateExpsKDim: hiddenSize,
            upExpsRaw: upExpsRaw, upExpsRawQt: gateUpDesc.QuantizationType,
            upExpsMDim: moeIntermediate, upExpsKDim: hiddenSize,
            downExpsRaw: downExpsRaw, downExpsRawQt: downExpsDesc.QuantizationType,
            downExpsMDim: hiddenSize, downExpsKDim: moeIntermediate,
            sharedGateRaw: Array.Empty<nint>(), sharedGateRawQt: QuantizationType.F32,
            sharedUpRaw: Array.Empty<nint>(), sharedUpRawQt: QuantizationType.F32,
            sharedDownRaw: Array.Empty<nint>(), sharedDownRawQt: QuantizationType.F32);

        // ── MoE-only extras (dual-FFN split norms + router/expert scales) ──
        preFfwNorm2 = DequantizeNorm(dataBase, tensors[$"{prefix}.pre_ffw_norm_2.weight"], hiddenSize);
        postFfwNorm1 = DequantizeNorm(dataBase, tensors[$"{prefix}.post_ffw_norm_1.weight"], hiddenSize);
        postFfwNorm2 = DequantizeNorm(dataBase, tensors[$"{prefix}.post_ffw_norm_2.weight"], hiddenSize);
        routerScale = DequantizeNorm(dataBase, tensors[$"{prefix}.ffn_gate_inp.scale"], hiddenSize);
        downExpertScale = DequantizeNorm(dataBase, tensors[$"{prefix}.ffn_down_exps.scale"], numExperts);
        }

        // ── Gemma-4 extras (all layer kinds) ─────────────────────────────
        float[] postFfwNorm = DequantizeNorm(dataBase, tensors[$"{prefix}.post_ffw_norm.weight"], hiddenSize);
        // layer_output_scale [1] — TENSOR_NOT_REQUIRED in llama.cpp; identity 1.0
        // when absent.
        float[]? layerOutScaleArr = LoadOptionalNorm(dataBase, tensors, $"{prefix}.layer_output_scale.weight", 1);

        // DiffusionGemma-only: enc_layer_output_scale [1]. Present only on the
        // diffusion-gemma GGUF (TENSOR_NOT_REQUIRED on gemma4). When present the
        // unified forward uses it as the per-layer scalar for the PROMPT region;
        // when absent (autoregressive gemma4) every row uses LayerOutputScale.
        float[]? encLayerOutScaleArr =
            LoadOptionalNorm(dataBase, tensors, $"{prefix}.enc_layer_output_scale.weight", 1);

        var gemma4 = new Gemma4LayerWeights
        {
            PreFfwNorm2 = preFfwNorm2,
            PostFfwNorm1 = postFfwNorm1,
            PostFfwNorm2 = postFfwNorm2,
            PostFfwNorm = postFfwNorm,
            RouterScale = routerScale,
            DownExpertScale = downExpertScale,
            LayerOutputScale = layerOutScaleArr is null ? 1.0f : layerOutScaleArr[0],
            EncLayerOutputScale = encLayerOutScaleArr is null ? (float?)null : encLayerOutScaleArr[0],
            VFromK = vFromK,
            GateUpExpsRowBytes = gateUpStride,
            DownExpsRowBytes = downStride,
        };

        // ── Per-layer PLE slots (Gemma-4 dense E2B/E4B) ──────────────────
        // inp_gate [pleDim, hidden], proj [hidden, pleDim] — F32 for the
        // GemmF32 injection kernel (owned upcast when the file quantizes them);
        // post_norm [hidden] plain weights. Names per llama.cpp llama-arch.cpp
        // (LLM_TENSOR_PER_LAYER_INP_GATE / _PROJ / _POST_NORM).
        nint pleGatePtr = 0, pleProjPtr = 0;
        float[]? plePostNorm = null;
        if (config.PerLayerEmbedding is PerLayerEmbeddingConfig layerPle)
        {
            var pleGateDesc = tensors[$"{prefix}.inp_gate.weight"];
            var pleProjDesc = tensors[$"{prefix}.proj.weight"];
            pleGatePtr = pleGateDesc.QuantizationType == QuantizationType.F32
                ? dataBase + (nint)pleGateDesc.DataOffset
                : DequantToF32(dataBase, pleGateDesc, (long)layerPle.PerLayerDim * hiddenSize, owned!);
            pleProjPtr = pleProjDesc.QuantizationType == QuantizationType.F32
                ? dataBase + (nint)pleProjDesc.DataOffset
                : DequantToF32(dataBase, pleProjDesc, (long)hiddenSize * layerPle.PerLayerDim, owned!);
            plePostNorm = DequantizeNorm(dataBase, tensors[$"{prefix}.post_norm.weight"], hiddenSize);
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
            qBias: null, kBias: null, vBias: null, oBias: null,
            gateBias: null, upBias: null, downBias: null,
            qNormWeight: qNormWeight, kNormWeight: kNormWeight,
            moe: moe,
            mla: null,
            postAttnNormWeight: postAttnNorm, postFfnNormWeight: null,
            gemma4: gemma4,
            pleGateWeight: pleGatePtr, pleProjWeight: pleProjPtr, plePostNormWeight: plePostNorm);
    }

    /// <summary>
    /// Loads a routed-MoE layer whose experts stay in their on-disk
    /// quantization (gpt-oss convention: MXFP4 experts + F32 biases on the
    /// router and every expert projection). Populates the raw GGUF views and
    /// per-expert bias arrays consumed by
    /// <see cref="DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp"/>; the F32 W1/W2/W3
    /// banks are zero-filled placeholders.
    /// </summary>
    /// <remarks>
    /// Tensor naming (llama.cpp <c>LLM_ARCH_OPENAI_MOE</c>):
    /// <c>ffn_gate_inp.{weight,bias}</c> — router [hidden, E] + [E];
    /// <c>ffn_{gate,up}_exps.{weight,bias}</c> — [hidden, I, E] + [I, E];
    /// <c>ffn_down_exps.{weight,bias}</c> — [I, hidden, E] + [hidden, E].
    /// Bias arrays are stored expert-major on disk ([inner, E] → expert e's
    /// slice is a contiguous run at <c>e * inner</c>), matching the flat
    /// layout the CPU kernel indexes.
    /// </remarks>
    internal static MoeLayerWeights LoadQuantExpertMoeLayer(
        int layerIdx,
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config)
    {
        var moe = config.Moe
            ?? throw new InvalidOperationException("LoadQuantExpertMoeLayer called without Moe config.");

        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        int numExperts = moe.NumExperts;
        int moeIntermediate = moe.MoeIntermediateSize;

        // Router (2D, F32 — small, dequant inline) + optional bias.
        var routerDesc = tensors[$"{prefix}.ffn_gate_inp.weight"];
        float[] router = new float[numExperts * hiddenSize];
        Dequantize.ToFloat32(
            dataBase + (nint)routerDesc.DataOffset,
            (long)numExperts * hiddenSize,
            routerDesc.QuantizationType,
            router);
        float[]? routerBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.ffn_gate_inp.bias");

        var gateDesc = tensors[$"{prefix}.ffn_gate_exps.weight"];
        var upDesc = tensors[$"{prefix}.ffn_up_exps.weight"];
        var downDesc = tensors[$"{prefix}.ffn_down_exps.weight"];

        // Validate 3D shapes: [K, M, E] with K innermost.
        ValidateExpertShape(gateDesc, K: hiddenSize, M: moeIntermediate, E: numExperts);
        ValidateExpertShape(upDesc, K: hiddenSize, M: moeIntermediate, E: numExperts);
        ValidateExpertShape(downDesc, K: moeIntermediate, M: hiddenSize, E: numExperts);

        // Optional per-expert biases, kept flat ([E × inner], expert-major).
        float[]? gateBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.ffn_gate_exps.bias");
        float[]? upBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.ffn_up_exps.bias");
        float[]? downBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.ffn_down_exps.bias");

        // Zero-filled placeholder banks (satisfy MoeLayerWeights validation;
        // the CPU forward never dereferences them in quant-expert mode).
        var w1 = new nint[numExperts];
        var w2 = new nint[numExperts];
        var w3 = new nint[numExperts];

        var bundle = new MoeLayerWeights(
            gate: router,
            w1: w1, w2: w2, w3: w3,
            numExperts: numExperts,
            numExpertsPerTok: moe.NumExpertsPerTok,
            hiddenSize: hiddenSize,
            intermediateSize: moeIntermediate,
            normTopKProb: moe.NormTopKProb,
            sharedGateProj: Array.Empty<nint>(),
            sharedUpProj: Array.Empty<nint>(),
            sharedDownProj: Array.Empty<nint>(),
            sharedIntermediateSize: 0,
            sharedExpertGate: null,
            gateExpsRaw: dataBase + (nint)gateDesc.DataOffset, gateExpsRawQt: gateDesc.QuantizationType,
            gateExpsMDim: moeIntermediate, gateExpsKDim: hiddenSize,
            upExpsRaw: dataBase + (nint)upDesc.DataOffset, upExpsRawQt: upDesc.QuantizationType,
            upExpsMDim: moeIntermediate, upExpsKDim: hiddenSize,
            downExpsRaw: dataBase + (nint)downDesc.DataOffset, downExpsRawQt: downDesc.QuantizationType,
            downExpsMDim: hiddenSize, downExpsKDim: moeIntermediate,
            sharedGateRaw: Array.Empty<nint>(), sharedGateRawQt: QuantizationType.F32,
            sharedUpRaw: Array.Empty<nint>(), sharedUpRawQt: QuantizationType.F32,
            sharedDownRaw: Array.Empty<nint>(), sharedDownRawQt: QuantizationType.F32)
        {
            UseQuantExperts = true,
            RouterBias = routerBias,
            GateExpsBias = gateBias,
            UpExpsBias = upBias,
            DownExpsBias = downBias,
            UseSwiGluOai = moe.UseSwiGluOai,
            SoftmaxAfterTopK = moe.SoftmaxAfterTopK,
        };
        return bundle;
    }

    private static void ValidateExpertShape(GgufTensorDescriptor desc, int K, int M, int E)
    {
        if (desc.Shape.Rank != 3 || desc.Shape[0] != K || desc.Shape[1] != M || desc.Shape[2] != E)
            throw new InvalidDataException(
                $"Fused-experts tensor '{desc.Name}' shape does not match expected " +
                $"[{K}, {M}, {E}] (got rank {desc.Shape.Rank}: " +
                $"[{string.Join(", ", Enumerable.Range(0, desc.Shape.Rank).Select(i => desc.Shape[i]))}]).");
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
        List<nint> owned,
        bool skipF32MoeDequant = false)
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
            moeBundle = LoadDeepSeekMoeLayer(layerIdx, dataBase, tensors, config, owned, skipF32MoeDequant);
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
        List<nint> owned,
        bool skipF32Dequant = false,
        bool skipRoutedF32Only = false)
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

        // skipRoutedF32Only is a Qwen3MoeHybrid CPU optimization: skip the F32 dequant of routed
        // experts (~120 GB at Qwen3.6-35B-A3B scale) but keep shared-expert F32 (always small,
        // ~10 MB per layer). The caller dequantizes the routed experts on-demand per layer using
        // the raw-quant view.
        bool skipRoutedDequant = skipF32Dequant || skipRoutedF32Only;
        bool skipSharedDequant = skipF32Dequant; // shared stays unless caller asks for full skip

        nint[] w1, w3, w2;
        if (skipRoutedDequant)
        {
            // GPU-only callers skip the F32 host dequant of the per-expert
            // 3D tensors — saves ~2.2 GB host RAM per V2-Lite Q4_K_M MoE
            // layer. Zero-filled arrays of size numExperts (to satisfy the
            // MoeLayerWeights ctor's length validation); the GPU loader uses
            // the raw views (gateRaw/upRaw/downRaw below). The CPU
            // MoeSwiGluMlp oracle is not callable on this layer in this mode.
            w1 = new nint[numExperts];
            w3 = new nint[numExperts];
            w2 = new nint[numExperts];
        }
        else
        {
            w1 = SliceExpertsToF32(
                dataBase, gateDesc,
                numExperts, M: moeIntermediate, K: hiddenSize, owned);
            w3 = SliceExpertsToF32(
                dataBase, upDesc,
                numExperts, M: moeIntermediate, K: hiddenSize, owned);
            w2 = SliceExpertsToF32(
                dataBase, downDesc,
                numExperts, M: hiddenSize, K: moeIntermediate, owned);
        }

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

            // Single fused shared MLP — small enough that skipping the F32 dequant only saves
            // ~10 MB host RAM per layer (vs ~2.2 GB for routed experts). With skipRoutedF32Only
            // we still dequant shared so the CPU MoE kernel can use them directly without a
            // separate on-the-fly path for the (small) shared branch.
            if (skipSharedDequant)
            {
                sharedGate = [(nint)0];
                sharedUp = [(nint)0];
                sharedDown = [(nint)0];
            }
            else
            {
                sharedGate = [DequantToF32(dataBase, sharedGateDesc, (long)sharedI * hiddenSize, owned)];
                sharedUp = [DequantToF32(dataBase, sharedUpDesc, (long)sharedI * hiddenSize, owned)];
                sharedDown = [DequantToF32(dataBase, sharedDownDesc, (long)hiddenSize * sharedI, owned)];
            }

            sharedGateRaw = [dataBase + (nint)sharedGateDesc.DataOffset];
            sharedGateRawQt = sharedGateDesc.QuantizationType;
            sharedUpRaw = [dataBase + (nint)sharedUpDesc.DataOffset];
            sharedUpRawQt = sharedUpDesc.QuantizationType;
            sharedDownRaw = [dataBase + (nint)sharedDownDesc.DataOffset];
            sharedDownRawQt = sharedDownDesc.QuantizationType;
        }

        // Qwen1.5-MoE / qwen35moe convention: a per-token sigmoid gate scales the shared-expert
        // output. The tensor key is "ffn_gate_inp_shexp.weight" (shape [hiddenSize], F32-dequant).
        // DeepSeek-V2/V3 GGUFs don't carry this tensor — the TryGetValue returns null and the
        // shared branch is added unscaled, preserving the DeepSeek code path unchanged.
        float[]? sharedExpertGate = null;
        if (sharedIntermediate > 0
            && tensors.TryGetValue($"{prefix}.ffn_gate_inp_shexp.weight", out var shGateDesc))
        {
            sharedExpertGate = new float[hiddenSize];
            Dequantize.ToFloat32(
                dataBase + (nint)shGateDesc.DataOffset,
                hiddenSize,
                shGateDesc.QuantizationType,
                sharedExpertGate);
        }

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
            sharedExpertGate: sharedExpertGate,
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
