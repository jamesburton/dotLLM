using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

using QuantType = DotLLM.Core.Configuration.QuantizationType;

namespace DotLLM.Vulkan;

/// <summary>
/// End-to-end Vulkan forward pass for Llama-family transformer models.
/// Implements <see cref="IModel"/> using the wave-1/wave-2 Vulkan compute
/// kernels: <see cref="MatMulF32Kernel"/> plus the Q8_0 matmul kernels
/// (<see cref="MatMulQ8_0Kernel"/> for decode-path GEMV,
/// <see cref="MatMulQ8_0GemmKernel"/> for batched prefill),
/// <see cref="RmsNormF32Kernel"/>, <see cref="RopeF32Kernel"/>,
/// <see cref="AttentionF32Kernel"/>, <see cref="SwiGluF32Kernel"/>, and
/// <see cref="AddKernel"/> for residuals.
/// </summary>
/// <remarks>
/// <para>
/// Supported quantized weights stay on device in their source byte layout
/// and are consumed directly by the matching matmul kernels. This covers
/// Q8_0, Q4_K, Q5_K, Q6_K, F16, and BF16 where shape constraints permit;
/// unsupported shapes fall back to F32 upload. All non-matmul kernels remain
/// F32. Transformer-family MLA and MoE layers are handled by dedicated Vulkan
/// branches; SSM / Mamba models use their separate Vulkan model type.
/// </para>
/// <para>
/// Forward pass is fence-pipelined: a single persistent command buffer
/// records every kernel dispatch + inter-kernel pipeline barrier for the
/// whole forward, submits once per forward, and waits on a single fence
/// before downloading logits. Legacy synchronous kernel launches (one
/// <c>vkQueueWaitIdle</c> per kernel) are only used by the standalone
/// unit tests.
/// </para>
/// <para>
/// Architectural parallel with <c>DotLLM.Cuda.CudaTransformerModel</c>:
/// upload weights once at construction, reuse a single
/// <see cref="VulkanForwardState"/> for scratch. Each linear projection
/// dispatches through <see cref="RecordMatmul"/> which picks
/// <c>matmul_q8_0</c> / <c>matmul_q8_0_gemm</c> / <c>matmul_f32</c> based on
/// the weight's device-side quant type and <c>seqLen</c>. Logits come back
/// as a single <see cref="UnmanagedTensor"/> of shape <c>[1, vocabSize]</c>
/// matching the CUDA return convention.
/// </para>
/// </remarks>
public sealed class VulkanTransformerModel : IModel
{
    private readonly VulkanDevice _device;
    private readonly VulkanWeights _weights;
    private readonly VulkanForwardState _state;

    // Kernels — one instance each, pipelines are reused across all launches.
    private readonly MatMulF32Kernel _matmul;
    private readonly MatMulQ8_0Kernel _matmulQ8;
    private readonly MatMulQ8_0GemmKernel _matmulQ8Gemm;
    // Optional: coopmat Q8_0 GEMM for prefill (seqLen>1) on devices that
    // advertise VK_KHR_cooperative_matrix. ~3.8× over the scalar GEMM on AMD
    // RDNA3.5 iGPU at Llama-3 4096² N=64 (790 vs 209 GFLOPS). Null on devices
    // without coopmat — the router falls back to _matmulQ8Gemm then.
    private readonly MatMulQ8_0GemmCoopmatKernel? _matmulQ8GemmCoopmat;
    // Q2_K + Q3_K matmul kernels — completes the K-quant family on Vulkan.
    // Always created; the dispatcher in RecordMatmul branches on the
    // device-side QuantizationType per call. No coopmat variants — follow-up
    // ticket sibling of the Q4_K / Q5_K / Q6_K coopmat work.
    private readonly MatMulQ2KGemvF32Kernel _matmulQ2K;
    private readonly MatMulQ2KGemmF32Kernel _matmulQ2KGemm;
    private readonly MatMulQ3KGemvF32Kernel _matmulQ3K;
    private readonly MatMulQ3KGemmF32Kernel _matmulQ3KGemm;
    // Q4_K_M matmul kernels — Phase 1 of K-quant work. Always created; the
    // dispatcher in RecordMatmul branches on the device-side QuantizationType
    // per call. Coopmat Q4_K is a follow-up ticket.
    private readonly MatMulQ4KGemvF32Kernel _matmulQ4K;
    private readonly MatMulQ4KGemmF32Kernel _matmulQ4KGemm;
    // Q5_K_M matmul kernels — Phase 1 sibling of Q4_K. Same dispatch shape
    // as Q4_K (just one extra qh-byte read per element); always created.
    private readonly MatMulQ5KGemvF32Kernel _matmulQ5K;
    private readonly MatMulQ5KGemmF32Kernel _matmulQ5KGemm;
    // IQ4_NL / IQ4_XS matmul kernels — IQ-family follow-up to the K-quant
    // Phase 1 work. Same dispatch shape as Q4_K/Q5_K/Q6_K (one workgroup per
    // output row for GEMV, 16x16 cell tile for GEMM). IQ4_NL uses 32-element
    // blocks (alignment 32) while IQ4_XS uses 256-element super-blocks.
    private readonly MatMulIq4NlGemvF32Kernel _matmulIq4Nl;
    private readonly MatMulIq4NlGemmF32Kernel _matmulIq4NlGemm;
    private readonly MatMulIq4XsGemvF32Kernel _matmulIq4Xs;
    private readonly MatMulIq4XsGemmF32Kernel _matmulIq4XsGemm;
    // IQ1_S matmul kernels — smallest GGUF quant (~1.5-1.7 bpw). Same dispatch
    // shape as IQ4_XS: one workgroup per output row for GEMV, 16x16 cell tile
    // for GEMM. 256-element super-block alignment.
    private readonly MatMulIq1SGemvF32Kernel _matmulIq1S;
    private readonly MatMulIq1SGemmF32Kernel _matmulIq1SGemm;
    // I2_S matmul kernels — BitNet b1.58 ternary (~1.6 bpw). 128-element block
    // alignment; raw on-device layout is m·(K/4) packed bytes + one trailing
    // per-tensor float32 scale. Decode via GEMV, prefill via 16x16-tiled GEMM.
    private readonly MatMulI2SGemvF32Kernel _matmulI2S;
    private readonly MatMulI2SGemmF32Kernel _matmulI2SGemm;
    // Q6_K_M matmul kernels — Phase 1 sibling of Q4_K / Q5_K, completing the
    // K-quant matmul kernel coverage. Q6_K is structurally simpler on the
    // metadata side (no dmin / 6-bit-packed scales) but has a more intricate
    // (ql, qh) byte-extraction; always created.
    private readonly MatMulQ6KGemvF32Kernel _matmulQ6K;
    private readonly MatMulQ6KGemmF32Kernel _matmulQ6KGemm;
    // IQ2 family (XXS / XS / S) matmul kernels — IQ-family follow-up to the
    // K-quant / IQ4 work. Same dispatch shape as Q4_K/IQ4_XS GEMV/GEMM. The
    // three variants share one Iq2Codebooks instance (4 SSBOs: 3 grids +
    // ksigns) created on the first kernel and threaded through via
    // CreateWithCodebooks. IQ2_S only needs its own grid (no ksigns — sign
    // mask is stored explicitly per pair) but reuses the shared host for
    // disposal symmetry.
    private readonly Iq2Codebooks _iq2Codebooks;
    private readonly MatMulIq2XxsGemvF32Kernel _matmulIq2Xxs;
    private readonly MatMulIq2XxsGemmF32Kernel _matmulIq2XxsGemm;
    private readonly MatMulIq2XsGemvF32Kernel _matmulIq2Xs;
    private readonly MatMulIq2XsGemmF32Kernel _matmulIq2XsGemm;
    private readonly MatMulIq2SGemvF32Kernel _matmulIq2S;
    private readonly MatMulIq2SGemmF32Kernel _matmulIq2SGemm;
    // IQ3 family (XXS / S) matmul kernels — IQ-family follow-up. Shares the
    // same SSBO codebook pattern as IQ2: Iq3Codebooks owns the IQ3_XXS grid
    // (256 × 4 bytes) + IQ3_S grid (512 × 4 bytes), shared across all 4 kernels
    // via CreateWithCodebooks. Bit-perfect against the CPU oracle (Iq3Fixture).
    private readonly Iq3Codebooks _iq3Codebooks;
    private readonly MatMulIq3XxsGemvF32Kernel _matmulIq3Xxs;
    private readonly MatMulIq3XxsGemmF32Kernel _matmulIq3XxsGemm;
    private readonly MatMulIq3SGemvF32Kernel _matmulIq3S;
    private readonly MatMulIq3SGemmF32Kernel _matmulIq3SGemm;
    // F16 / BF16 native matmul kernels — Phase 8. Always created; the
    // RecordMatmul dispatcher routes per device-side QuantizationType. The
    // F16 GEMM coopmat path is optional (null when the device does not
    // advertise VK_KHR_cooperative_matrix); the scalar GEMM picks up the
    // slack on those devices. BF16 has no coopmat path — KHR_coopmat exposes
    // F16 / Sint8 operands on mainstream drivers, not BF16.
    private readonly MatMulF16GemvF32Kernel _matmulF16;
    private readonly MatMulF16GemmF32Kernel _matmulF16Gemm;
    private readonly MatMulF16GemmCoopmatKernel? _matmulF16GemmCoopmat;
    private readonly MatMulBf16GemvF32Kernel _matmulBf16;
    private readonly MatMulBf16GemmF32Kernel _matmulBf16Gemm;
    // Optional decode-path fusion of rmsnorm + Q8_0 GEMV. Eliminates one
    // dispatch + one barrier per attn-norm/Q proj and per ffn-norm/Gate proj
    // (60 dispatches per decode at 30 layers). Null when the SPV is missing
    // or when the model's hidden size exceeds the shader's on-chip cap;
    // router falls back to the standalone (rmsnorm + matmul_q8_0) pair.
    private readonly RmsNormMatmulQ8_0FusedKernel? _rmsnormMatmulQ8Fused;
    // dp4a MMVQ decode path (issue #46) — both null when the device lacks
    // integer-dot-product support, the SPVs are missing, or the env-var
    // opt-out is set; RecordMatmul then falls back to the F32-in Q8_0 GEMV.
    private readonly QuantizeQ8_1Kernel? _quantizeQ8_1;
    private readonly MatMulQ8_0MmvqKernel? _matmulQ8Mmvq;
    // dp4a MMVQ decode path for Q4_K (issue #52) — null under the same
    // conditions as _matmulQ8Mmvq; reuses _quantizeQ8_1 + the Q8_1Xq/Xds
    // activation scratch. RecordMatmul falls back to the F32-in Q4_K GEMV.
    private readonly MatMulQ4KMmvqKernel? _matmulQ4KMmvq;
    // dp4a MMVQ decode path for Q6_K (issue #338) — null under the same
    // conditions as _matmulQ8Mmvq; reuses _quantizeQ8_1 + the Q8_1Xq/Xds
    // activation scratch. RecordMatmul falls back to the F32-in Q6_K GEMV.
    // On 8B-class Q4_K_M models ffn_down/attn_v are Q6_K, so this is the
    // dominant remaining decode GEMV there.
    private readonly MatMulQ6KMmvqKernel? _matmulQ6KMmvq;
    // dp4a MMVQ decode path for Q5_K (issue #338) — sibling of Q4_K/Q6_K MMVQ,
    // completing K-quant decode MMVQ coverage. RecordMatmul falls back to the
    // F32-in Q5_K GEMV when not wired.
    private readonly MatMulQ5KMmvqKernel? _matmulQ5KMmvq;
    // dp4a MMVQ decode path for Q2_K / Q3_K (issue #339) — sibling of the #338
    // K-quant MMVQ kernels. RecordMatmul falls back to the F32-in GEMV otherwise.
    private readonly MatMulQ2KMmvqKernel? _matmulQ2KMmvq;
    private readonly MatMulQ3KMmvqKernel? _matmulQ3KMmvq;
    // dp4a MMVQ decode path for IQ4_NL / IQ4_XS (issue #339) — codebook-lookup
    // weights become int8 for dp4a. RecordMatmul falls back to the F32-in GEMV.
    private readonly MatMulIq4NlMmvqKernel? _matmulIq4NlMmvq;
    private readonly MatMulIq4XsMmvqKernel? _matmulIq4XsMmvq;
    // dp4a MMVQ decode path for the IQ2/IQ3/IQ1 codebook-grid quants (issue #339).
    // Reuse the shared Iq2Codebooks/Iq3Codebooks SSBOs (6 bindings each).
    private readonly MatMulIq2XxsMmvqKernel? _matmulIq2XxsMmvq;
    private readonly MatMulIq2XsMmvqKernel? _matmulIq2XsMmvq;
    private readonly MatMulIq2SMmvqKernel? _matmulIq2SMmvq;
    private readonly MatMulIq3XxsMmvqKernel? _matmulIq3XxsMmvq;
    private readonly MatMulIq3SMmvqKernel? _matmulIq3SMmvq;
    private readonly MatMulIq1SMmvqKernel? _matmulIq1SMmvq;
    // When true (default whenever the MMVQ decode path is wired),
    // RecordSharedInputMmvqGroup quantizes the shared activation once for a group
    // of same-input Q8_0 projections (Q/K/V share the post-attn-norm input;
    // gate/up share the post-ffn-norm input) instead of re-running quantize_q8_1
    // per projection. DOTLLM_VULKAN_MMVQ_NO_SHARE=1 forces the per-projection
    // path (used by the share-vs-no-share bit-identical parity test).
    private readonly bool _mmvqShareActivation = !IsMmvqShareDisabled();
    // Reusable scratch for RecordSharedInputMmvqGroup's projection list — sized
    // for the largest same-input group (Q/K/V = 3). Reused across layers and
    // decode steps to keep command-buffer recording allocation-free (the element
    // type holds a managed VulkanDevice.Buffer, so a Span collection-expression
    // would heap-allocate per call). Recording is single-threaded per model.
    private readonly MmvqGroupProjection[] _mmvqGroupScratch = new MmvqGroupProjection[3];
    // dp4a MMQ prefill path (issue #50) — both null when the device lacks
    // integer-dot-product support, the SPVs are missing, or the MMQ env-var
    // opt-out is set; RecordMatmul then falls back to the coopmat / scalar
    // F32-in Q8_0 GEMM for seqLen>1.
    private readonly QuantizeQ8_1RowsKernel? _quantizeQ8_1Rows;
    private readonly MatMulQ8_0MmqKernel? _matmulQ8Mmq;
    private readonly MatMulQ4KMmqKernel? _matmulQ4KMmq;
    private readonly MatMulQ6KMmqKernel? _matmulQ6KMmq;
    private readonly MatMulQ5KMmqKernel? _matmulQ5KMmq;
    private readonly MatMulIq4XsMmqKernel? _matmulIq4XsMmq;
    private readonly MatMulIq4NlMmqKernel? _matmulIq4NlMmq;
    private readonly MatMulQ2KMmqKernel? _matmulQ2KMmq;
    private readonly MatMulQ3KMmqKernel? _matmulQ3KMmq;
    private readonly RmsNormF32Kernel _rmsnorm;
    private readonly RopeF32Kernel _rope;
    private readonly AttentionF32Kernel _attention;
    /// <summary>
    /// Flash-Attention F32 kernel for the GQA prefill path (seqQ &gt; 1). Null
    /// when the SPV is missing (older builds), when the env-var opt-out is
    /// set, or when the model's head_dim exceeds the shader's MAX_HEAD_DIM.
    /// When null, every dispatch falls through to <see cref="_attention"/>.
    /// </summary>
    private readonly VulkanFlashAttentionF32Kernel? _flashAttention;
    private readonly SwiGluF32Kernel _swiglu;
    // GeGLU (tanh-approximate GELU) FFN activation — created only when the
    // model's ActivationFunction is GELUTanh (Gemma 2 / Gemma 3). Null on
    // every SwiGLU architecture so the standard path is byte-identical.
    private readonly GeGluTanhF32Kernel? _geglu;
    // Gated squared-ReLU (relu(gate)²·up) FFN activation — created only when the
    // model's ActivationFunction is ReluSquared (BitNet b1.58). Null on every
    // SwiGLU/GeGLU architecture so the standard path is byte-identical.
    private readonly ReLU2GluF32Kernel? _relu2glu;
    // In-place scalar multiply for the Gemma sqrt(hidden) embedding scale —
    // created only when Config.EmbeddingScale is set. Null otherwise. Also
    // reused for the Gemma-4 per-layer output scale (layer_output_scale).
    private readonly ScaleInplaceF32Kernel? _embedScale;
    // Unit-gamma (all-ones) [maxHeadDim] vector for Gemma-4's weight-less V-norm
    // (per-kv-head RMSNorm with no scale). Lazily allocated on first use.
    private VulkanDevice.Buffer? _gemma4OnesVec;
    private readonly AddKernel _add;
    // Per-feature bias add. Replaces the host-mapped fallback that used to
    // split the forward into multiple submits whenever Phi-3 / Qwen3 /
    // DeepSeek-V2 layers carried biases — now the whole forward stays in
    // one submit regardless of bias presence.
    private readonly BiasAddF32Kernel _biasAdd;
    // MLA (DeepSeek-V2/V3) — null when the model carries no MLA layer.
    // The post-projection attention loop (per-head SDPA with Q_nope/Q_pe
    // split + MQA-shared K_pe), the decoupled-rope rotation on Q_pe + K_pe,
    // and the per-head split of kv_b_proj's fused output into K_nope/V.
    private readonly AttentionMlaF32Kernel? _mlaAttention;
    private readonly RopeMlaF32Kernel? _mlaRope;
    private readonly MlaKvSplitF32Kernel? _mlaKvSplit;
    // MLA softmax scale = (YaRN mscale²) / sqrt(qk_head_dim). Folded once at
    // construction since the kernel takes scale as a push constant.
    private readonly float _mlaScale;
    private readonly int _mlaQkNopeHeadDim;
    private readonly int _mlaQkRopeHeadDim;
    private readonly int _mlaVHeadDim;
    private readonly int _mlaNumHeads;
    private readonly float _mlaRopeTheta;
    // MoE (Mixtral / Qwen-MoE) — null when the model carries no MoE layer.
    private readonly MoeTopKSoftmaxF32Kernel? _moeTopkSoftmax;
    private readonly MoeIndexedMatmulF32Kernel? _moeIndexedMatmul;
    private readonly MoeIndexedMatmulQ8_0F32Kernel? _moeIndexedMatmulQ8;
    // Gemma-4 quantized experts: Q4_K (fused gate_up → split W1/W3) and Q5_1
    // (down, with the per-expert ffn_down_exps.scale folded in-shader). Keep the
    // real 26B's experts quantized on device. Null when the model has no gemma4
    // quantized-MoE layer.
    private readonly MoeIndexedMatmulQ4_KF32Kernel? _moeIndexedMatmulQ4K;
    private readonly MoeIndexedMatmulQ5_1F32Kernel? _moeIndexedMatmulQ5_1;
    // Tiled (shared-memory) variant of the indexed matmul. Wins on prefill at
    // large N (seqLen * topK ≥ 32) by amortising the x-row load across a
    // TILE_M-wide output tile; the scalar variant remains for decode (small N)
    // where the GEMV-style scalar dispatch wins.
    private readonly MoeIndexedMatmulTiledF32Kernel? _moeIndexedMatmulTiled;
    private readonly MoeIndexedLoraDeltaF32Kernel? _moeIndexedLoraDelta;
    private readonly MoeExpertOffsetsKernel? _moeExpertOffsets;
    private readonly MoeExpandGroupByExpertF32Kernel? _moeExpandGroupByExpert;
    private readonly MoeGroupedMatmulF16CoopmatKernel? _moeGroupedMatmulF16Coopmat;
    private readonly MoeUngroupScatterF32Kernel? _moeUngroupScatter;
    private readonly MoeWeightedScatterF32Kernel? _moeWeightedScatter;
    private readonly MoeBroadcastF32Kernel? _moeBroadcast;
    // Optional Qwen1.5-MoE per-token sigmoid gate fold for the shared-expert
    // branch. Null when no MoE layer exists OR when no MoE layer carries a
    // SharedExpertGate weight (DeepSeek-V2/V3, Mixtral). Allocated alongside
    // the other MoE kernels so the gated path is wired wherever it might
    // fire across the per-layer mix.
    private readonly MoeSigmoidGatedAddF32Kernel? _moeSigmoidGatedAdd;

    // Persistent command buffer + fence used by Forward. One SubmitContext
    // per model — reset+begin at the start of each forward, submit+wait at
    // the end. Bias host-side steps split the forward into multiple submits
    // but each submit still batches many dispatches behind one fence.
    private readonly VulkanDevice.SubmitContext _submit;

    private readonly TransformerWeights _cpuWeights; // retained for embedding lookup
    private readonly GgufFile? _gguf;

    // ── DiffusionGemma per-forward state ────────────────────────────────────
    // Mirrors the CPU TransformerModel diffusion seam. _diffusionMaskMode /
    // _diffusionPrefixLen carry the region split (Hybrid prefix = prompt length P);
    // the canvas region is rows [P, seqLen). Default Causal/0 ⇒ the AR gemma4 path
    // and every non-diffusion forward are byte-identical (region deltas inert).
    // Single-threaded per generation, like _currentLora.
    private AttentionMaskMode _diffusionMaskMode = AttentionMaskMode.Causal;
    private int _diffusionPrefixLen;
    // Self-conditioning (set by the diffusion generator each denoise step). _scUse
    // is the gate (0 on step 0 ⇒ zero-SC; 1 thereafter); _scPrevLogits holds the
    // previous step's canvas-region logits [_scCanvasLen × vocab] (post-softcap).
    private float[]? _scPrevLogits;
    private int _scCanvasLen;
    private float _scUse;
    // Lazily-allocated all-position logits buffer [maxSeqLen × vocab] for the
    // diffusion forward (the AR Forward returns only the last row; diffusion needs
    // every canvas row). Grows monotonically; null until the first diffusion forward.
    private VulkanDevice.Buffer? _diffusionLogits;
    private int _diffusionLogitsCapacityRows;
    // Host-visible scratch holding the host-computed self-conditioning signal
    // [canvasLen × hidden] (uploaded then device-added into the canvas embedding).
    // Lazy; grows monotonically; null until the first SC step.
    private VulkanDevice.Buffer? _diffusionScSig;
    private int _diffusionScSigCapacityElems;
    // ── DiffusionGemma prompt-KV (PKV) phase state ──────────────────────────
    // Drives the two-phase PKV optimisation inside RecordGemma4Attention. None:
    // normal cacheless forward (DEFAULT). Prefill: capture each layer's final K/V
    // into _pkvStore. Decode: read cached prompt K/V and attend [prompt|canvas] under
    // a rectangular Bidirectional mask with positionOffset = promptLen. Single-threaded
    // per generation, like _diffusionMaskMode.
    private DiffusionKvPhase _pkvPhase = DiffusionKvPhase.None;
    private VulkanDiffusionPromptKv? _pkvStore;
    private int _pkvPromptLen;
    // Device-local K/V concat scratch [maxKvCtx × maxKvStride] for the decode phase
    // (cached prompt K/V | fresh canvas K/V). Lazy; grows monotonically.
    private VulkanDevice.Buffer? _pkvKConcat;
    private VulkanDevice.Buffer? _pkvVConcat;
    private long _pkvConcatCapacityBytes;

    private enum DiffusionKvPhase { None, Prefill, Decode }
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly RopeF32Kernel.Variant _ropeVariant;
    private readonly int _slidingWindow;
    private readonly bool _ownsDevice;

    // LoRA (Phase 4b) — device-side cache of uploaded adapters keyed by
    // ILoraAdapter reference identity. Lazy: zero VRAM when no LoRA Forward
    // is ever invoked. _currentLora is set/cleared in the try/finally
    // surrounding the inner Forward and is checked at every projection
    // site in RecordMatmulWithLora to decide whether to dispatch the
    // LoRA delta on top of the base projection.
    private readonly VulkanLoraAdapterCache _loraCache;
    private VulkanLoraAdapter? _currentLora;

    // Fused LoRA delta-GEMV (single dispatch in place of the four-step
    // matmul(B) → matmul(A) → add → vkCmdCopyBuffer chain). Null when the
    // .spv is missing (older builds); router falls back to the un-fused
    // path. Used only when the adapter's rank ≤ LoraDeltaGemvFusedF32Kernel.MaxRank.
    private readonly LoraDeltaGemvFusedF32Kernel? _loraDeltaGemvFused;

    // Phase 5f — ForwardBatch intra-block matmul fusion scratch. Lazy-allocated
    // on first batched call (zero VRAM cost when only Forward is used). The
    // scratch holds per-seq Q / attention-output staging buffers (attention is
    // dispatched per-seq because each sequence has its own VulkanKvCache + own
    // positionOffset) plus a stacked [N_simple, hidden] last-row buffer + a
    // [N_simple, vocab] batched lm_head output. See ForwardBatch + the
    // VulkanForwardBatchScratch summary for the full data-flow.
    private VulkanForwardBatchScratch? _batchScratch;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _weights.AllocatedBytes;

    /// <summary>Creates a <see cref="VulkanKvCache"/> sized for this model.</summary>
    /// <remarks>
    /// Per-layer geometry comes from the single Core helper
    /// <see cref="KvGeometry.FromConfig"/>: uniform <c>NumKvHeads × HeadDim</c> for every
    /// dense/GQA/MoE model, and distinct per-layer strides for Gemma-4 (whose sliding and
    /// global layers carry different KV-head counts AND head dims).
    /// </remarks>
    public VulkanKvCache CreateKvCache(int maxSeqLen)
        => new(_device, KvGeometry.FromConfig(Config), maxSeqLen);

    /// <summary>
    /// Creates a GPU-resident TurboQuant (MSE-stage) KV-cache on this model's device. The caller
    /// supplies the codec constants (the Vulkan project does not depend on the Engine codec):
    /// <paramref name="centroids"/> (2^mseBits, scaled by 1/√d), per-K/V RHT sign sets (length
    /// headDim, ±1), and <paramref name="invSqrtD"/>. Uniform geometry only (headDim a power of two
    /// ≤ 256); used on the single-sequence autoregressive forward path.
    /// </summary>
    public VulkanTurboQuantKvCache CreateTurboQuantKvCache(
        string spvDir, int maxSeqLen, int mseBits,
        ReadOnlySpan<float> centroids, ReadOnlySpan<float> signsK, ReadOnlySpan<float> signsV, float invSqrtD)
        => new(_device, spvDir, Config.NumLayers, Config.NumKvHeads, Config.HeadDim,
               maxSeqLen, mseBits, centroids, signsK, signsV, invSqrtD);

    /// <summary>
    /// Creates a per-layer MLA (DeepSeek-V2/V3) KV-cache sized for this
    /// model. Throws when the model is not an MLA model (no <c>MlaConfig</c>
    /// at construction).
    /// </summary>
    public MlaVulkanKvCache CreateMlaKvCache(int maxSeqLen)
    {
        if (Config.MlaConfig is null)
            throw new InvalidOperationException(
                "CreateMlaKvCache requires a model with MlaConfig set; this model has none.");
        return new MlaVulkanKvCache(_device, Config.NumLayers, maxSeqLen,
            _mlaNumHeads, _mlaQkNopeHeadDim, _mlaVHeadDim, _mlaQkRopeHeadDim);
    }

    private VulkanTransformerModel(
        VulkanDevice device, bool ownsDevice,
        ModelConfig config, VulkanWeights weights, TransformerWeights cpuWeights,
        VulkanForwardState state,
        MatMulF32Kernel matmul, MatMulQ8_0Kernel matmulQ8, MatMulQ8_0GemmKernel matmulQ8Gemm,
        MatMulQ8_0GemmCoopmatKernel? matmulQ8GemmCoopmat,
        MatMulQ2KGemvF32Kernel matmulQ2K, MatMulQ2KGemmF32Kernel matmulQ2KGemm,
        MatMulQ3KGemvF32Kernel matmulQ3K, MatMulQ3KGemmF32Kernel matmulQ3KGemm,
        MatMulQ4KGemvF32Kernel matmulQ4K, MatMulQ4KGemmF32Kernel matmulQ4KGemm,
        MatMulQ5KGemvF32Kernel matmulQ5K, MatMulQ5KGemmF32Kernel matmulQ5KGemm,
        MatMulQ6KGemvF32Kernel matmulQ6K, MatMulQ6KGemmF32Kernel matmulQ6KGemm,
        MatMulIq4NlGemvF32Kernel matmulIq4Nl, MatMulIq4NlGemmF32Kernel matmulIq4NlGemm,
        MatMulIq4XsGemvF32Kernel matmulIq4Xs, MatMulIq4XsGemmF32Kernel matmulIq4XsGemm,
        Iq2Codebooks iq2Codebooks,
        MatMulIq2XxsGemvF32Kernel matmulIq2Xxs, MatMulIq2XxsGemmF32Kernel matmulIq2XxsGemm,
        MatMulIq2XsGemvF32Kernel matmulIq2Xs, MatMulIq2XsGemmF32Kernel matmulIq2XsGemm,
        MatMulIq2SGemvF32Kernel matmulIq2S, MatMulIq2SGemmF32Kernel matmulIq2SGemm,
        Iq3Codebooks iq3Codebooks,
        MatMulIq3XxsGemvF32Kernel matmulIq3Xxs, MatMulIq3XxsGemmF32Kernel matmulIq3XxsGemm,
        MatMulIq3SGemvF32Kernel matmulIq3S, MatMulIq3SGemmF32Kernel matmulIq3SGemm,
        MatMulIq1SGemvF32Kernel matmulIq1S, MatMulIq1SGemmF32Kernel matmulIq1SGemm,
        MatMulI2SGemvF32Kernel matmulI2S, MatMulI2SGemmF32Kernel matmulI2SGemm,
        MatMulF16GemvF32Kernel matmulF16, MatMulF16GemmF32Kernel matmulF16Gemm,
        MatMulF16GemmCoopmatKernel? matmulF16GemmCoopmat,
        MatMulBf16GemvF32Kernel matmulBf16, MatMulBf16GemmF32Kernel matmulBf16Gemm,
        RmsNormMatmulQ8_0FusedKernel? rmsnormMatmulQ8Fused,
        QuantizeQ8_1Kernel? quantizeQ8_1, MatMulQ8_0MmvqKernel? matmulQ8Mmvq,
        MatMulQ4KMmvqKernel? matmulQ4KMmvq,
        MatMulQ6KMmvqKernel? matmulQ6KMmvq,
        MatMulQ5KMmvqKernel? matmulQ5KMmvq,
        MatMulQ2KMmvqKernel? matmulQ2KMmvq,
        MatMulQ3KMmvqKernel? matmulQ3KMmvq,
        MatMulIq4NlMmvqKernel? matmulIq4NlMmvq,
        MatMulIq4XsMmvqKernel? matmulIq4XsMmvq,
        MatMulIq2XxsMmvqKernel? matmulIq2XxsMmvq,
        MatMulIq2XsMmvqKernel? matmulIq2XsMmvq,
        MatMulIq2SMmvqKernel? matmulIq2SMmvq,
        MatMulIq3XxsMmvqKernel? matmulIq3XxsMmvq,
        MatMulIq3SMmvqKernel? matmulIq3SMmvq,
        MatMulIq1SMmvqKernel? matmulIq1SMmvq,
        QuantizeQ8_1RowsKernel? quantizeQ8_1Rows, MatMulQ8_0MmqKernel? matmulQ8Mmq,
        MatMulQ4KMmqKernel? matmulQ4KMmq,
        MatMulQ6KMmqKernel? matmulQ6KMmq,
        MatMulQ5KMmqKernel? matmulQ5KMmq,
        MatMulIq4XsMmqKernel? matmulIq4XsMmq,
        MatMulIq4NlMmqKernel? matmulIq4NlMmq,
        MatMulQ2KMmqKernel? matmulQ2KMmq,
        MatMulQ3KMmqKernel? matmulQ3KMmq,
        RmsNormF32Kernel rmsnorm, RopeF32Kernel rope,
        AttentionF32Kernel attention, VulkanFlashAttentionF32Kernel? flashAttention,
        SwiGluF32Kernel swiglu, GeGluTanhF32Kernel? geglu, ReLU2GluF32Kernel? relu2glu,
        ScaleInplaceF32Kernel? embedScale, AddKernel add,
        BiasAddF32Kernel biasAdd,
        AttentionMlaF32Kernel? mlaAttention, RopeMlaF32Kernel? mlaRope, MlaKvSplitF32Kernel? mlaKvSplit,
        MoeTopKSoftmaxF32Kernel? moeTopkSoftmax, MoeIndexedMatmulF32Kernel? moeIndexedMatmul,
        MoeIndexedMatmulQ8_0F32Kernel? moeIndexedMatmulQ8,
        MoeIndexedMatmulQ4_KF32Kernel? moeIndexedMatmulQ4K,
        MoeIndexedMatmulQ5_1F32Kernel? moeIndexedMatmulQ5_1,
        MoeIndexedMatmulTiledF32Kernel? moeIndexedMatmulTiled,
        MoeIndexedLoraDeltaF32Kernel? moeIndexedLoraDelta,
        MoeExpertOffsetsKernel? moeExpertOffsets,
        MoeExpandGroupByExpertF32Kernel? moeExpandGroupByExpert,
        MoeGroupedMatmulF16CoopmatKernel? moeGroupedMatmulF16Coopmat,
        MoeUngroupScatterF32Kernel? moeUngroupScatter,
        MoeWeightedScatterF32Kernel? moeWeightedScatter, MoeBroadcastF32Kernel? moeBroadcast,
        MoeSigmoidGatedAddF32Kernel? moeSigmoidGatedAdd,
        LoraDeltaGemvFusedF32Kernel? loraDeltaGemvFused,
        VulkanDevice.SubmitContext submit,
        GgufFile? gguf,
        float ropeTheta, int ropeDim, RopeF32Kernel.Variant ropeVariant, int slidingWindow,
        int mlaNumHeads, int mlaQkNopeHeadDim, int mlaQkRopeHeadDim, int mlaVHeadDim,
        float mlaScale, float mlaRopeTheta)
    {
        _device = device;
        _ownsDevice = ownsDevice;
        Config = config;
        _weights = weights;
        _cpuWeights = cpuWeights;
        _state = state;
        _matmul = matmul;
        _matmulQ8 = matmulQ8;
        _matmulQ8Gemm = matmulQ8Gemm;
        _matmulQ8GemmCoopmat = matmulQ8GemmCoopmat;
        _matmulQ2K = matmulQ2K;
        _matmulQ2KGemm = matmulQ2KGemm;
        _matmulQ3K = matmulQ3K;
        _matmulQ3KGemm = matmulQ3KGemm;
        _matmulQ4K = matmulQ4K;
        _matmulQ4KGemm = matmulQ4KGemm;
        _matmulQ5K = matmulQ5K;
        _matmulQ5KGemm = matmulQ5KGemm;
        _matmulQ6K = matmulQ6K;
        _matmulQ6KGemm = matmulQ6KGemm;
        _matmulIq4Nl = matmulIq4Nl;
        _matmulIq4NlGemm = matmulIq4NlGemm;
        _matmulIq4Xs = matmulIq4Xs;
        _matmulIq4XsGemm = matmulIq4XsGemm;
        _iq2Codebooks = iq2Codebooks;
        _matmulIq2Xxs = matmulIq2Xxs;
        _matmulIq2XxsGemm = matmulIq2XxsGemm;
        _matmulIq2Xs = matmulIq2Xs;
        _matmulIq2XsGemm = matmulIq2XsGemm;
        _matmulIq2S = matmulIq2S;
        _matmulIq2SGemm = matmulIq2SGemm;
        _iq3Codebooks = iq3Codebooks;
        _matmulIq3Xxs = matmulIq3Xxs;
        _matmulIq3XxsGemm = matmulIq3XxsGemm;
        _matmulIq3S = matmulIq3S;
        _matmulIq3SGemm = matmulIq3SGemm;
        _matmulIq1S = matmulIq1S;
        _matmulIq1SGemm = matmulIq1SGemm;
        _matmulI2S = matmulI2S;
        _matmulI2SGemm = matmulI2SGemm;
        _matmulF16 = matmulF16;
        _matmulF16Gemm = matmulF16Gemm;
        _matmulF16GemmCoopmat = matmulF16GemmCoopmat;
        _matmulBf16 = matmulBf16;
        _matmulBf16Gemm = matmulBf16Gemm;
        _rmsnormMatmulQ8Fused = rmsnormMatmulQ8Fused;
        _quantizeQ8_1 = quantizeQ8_1;
        _matmulQ8Mmvq = matmulQ8Mmvq;
        _matmulQ4KMmvq = matmulQ4KMmvq;
        _matmulQ6KMmvq = matmulQ6KMmvq;
        _matmulQ5KMmvq = matmulQ5KMmvq;
        _matmulQ2KMmvq = matmulQ2KMmvq;
        _matmulQ3KMmvq = matmulQ3KMmvq;
        _matmulIq4NlMmvq = matmulIq4NlMmvq;
        _matmulIq4XsMmvq = matmulIq4XsMmvq;
        _matmulIq2XxsMmvq = matmulIq2XxsMmvq;
        _matmulIq2XsMmvq = matmulIq2XsMmvq;
        _matmulIq2SMmvq = matmulIq2SMmvq;
        _matmulIq3XxsMmvq = matmulIq3XxsMmvq;
        _matmulIq3SMmvq = matmulIq3SMmvq;
        _matmulIq1SMmvq = matmulIq1SMmvq;
        _quantizeQ8_1Rows = quantizeQ8_1Rows;
        _matmulQ8Mmq = matmulQ8Mmq;
        _matmulQ4KMmq = matmulQ4KMmq;
        _matmulQ6KMmq = matmulQ6KMmq;
        _matmulQ5KMmq = matmulQ5KMmq;
        _matmulIq4XsMmq = matmulIq4XsMmq;
        _matmulIq4NlMmq = matmulIq4NlMmq;
        _matmulQ2KMmq = matmulQ2KMmq;
        _matmulQ3KMmq = matmulQ3KMmq;
        _rmsnorm = rmsnorm;
        _rope = rope;
        _attention = attention;
        _flashAttention = flashAttention;
        _swiglu = swiglu;
        _geglu = geglu;
        _relu2glu = relu2glu;
        _embedScale = embedScale;
        _add = add;
        _biasAdd = biasAdd;
        _mlaAttention = mlaAttention;
        _mlaRope = mlaRope;
        _mlaKvSplit = mlaKvSplit;
        _moeTopkSoftmax = moeTopkSoftmax;
        _moeIndexedMatmul = moeIndexedMatmul;
        _moeIndexedMatmulQ8 = moeIndexedMatmulQ8;
        _moeIndexedMatmulQ4K = moeIndexedMatmulQ4K;
        _moeIndexedMatmulQ5_1 = moeIndexedMatmulQ5_1;
        _moeIndexedMatmulTiled = moeIndexedMatmulTiled;
        _moeIndexedLoraDelta = moeIndexedLoraDelta;
        _moeExpertOffsets = moeExpertOffsets;
        _moeExpandGroupByExpert = moeExpandGroupByExpert;
        _moeGroupedMatmulF16Coopmat = moeGroupedMatmulF16Coopmat;
        _moeUngroupScatter = moeUngroupScatter;
        _moeWeightedScatter = moeWeightedScatter;
        _moeBroadcast = moeBroadcast;
        _moeSigmoidGatedAdd = moeSigmoidGatedAdd;
        _loraDeltaGemvFused = loraDeltaGemvFused;
        _submit = submit;
        _gguf = gguf;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _ropeVariant = ropeVariant;
        _slidingWindow = slidingWindow;
        _mlaNumHeads = mlaNumHeads;
        _mlaQkNopeHeadDim = mlaQkNopeHeadDim;
        _mlaQkRopeHeadDim = mlaQkRopeHeadDim;
        _mlaVHeadDim = mlaVHeadDim;
        _mlaScale = mlaScale;
        _mlaRopeTheta = mlaRopeTheta;
        _loraCache = new VulkanLoraAdapterCache(device);
    }

    /// <summary>
    /// Loads a model from an opened GGUF file onto a new Vulkan device.
    /// The caller owns the returned model; disposing it tears down the
    /// device, pipelines, and weight buffers.
    /// </summary>
    /// <param name="gguf">Opened GGUF file. Must remain alive for the model's lifetime.</param>
    /// <param name="config">Model configuration extracted from the GGUF metadata.</param>
    /// <param name="spvDir">
    /// Directory containing the compiled Vulkan SPIR-V blobs. When null,
    /// falls back to <c>spv/</c> next to the running assembly (matches the
    /// MSBuild <c>Content</c> copy pattern used by the Vulkan project).
    /// </param>
    public static VulkanTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);

        RejectUnsupportedArchitecture(config);

        var device = VulkanDevice.Create();
        try
        {
            spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");
            var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);
            return BuildModel(device, ownsDevice: true, config, cpuWeights, spvDir, gguf);
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Loads a model onto an existing <see cref="VulkanDevice"/>. The device
    /// is NOT disposed when the model is disposed — the caller retains
    /// ownership. Useful when the device is shared with other Vulkan
    /// components (e.g. a diagnostic hook that wants to launch its own
    /// kernels on the same queue).
    /// </summary>
    public static VulkanTransformerModel LoadFromGguf(
        VulkanDevice device, GgufFile gguf, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);

        RejectUnsupportedArchitecture(config);

        spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");
        var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);
        return BuildModel(device, ownsDevice: false, config, cpuWeights, spvDir, gguf);
    }

    /// <summary>
    /// Loads a model from a HuggingFace-convention safetensors source onto a
    /// new Vulkan device. Mirrors <see cref="TransformerModel.LoadFromSafetensors(ISafetensorsTensorSource, ModelConfig)"/>
    /// but produces a Vulkan-backed model. Used by tests and tooling that
    /// build synthetic fixtures (no GGUF roundtrip).
    /// </summary>
    public static VulkanTransformerModel LoadFromSafetensors(
        ISafetensorsTensorSource file, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);

        RejectUnsupportedArchitecture(config);

        var device = VulkanDevice.Create();
        try
        {
            spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");
            var cpuWeights = TransformerWeightsSafetensorsLoader.Load(file, config);
            return BuildModel(device, ownsDevice: true, config, cpuWeights, spvDir, gguf: null);
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Test-only factory that loads a model from already-built CPU
    /// <see cref="TransformerWeights"/>. Used by parity tests that need to inject
    /// Q8_0 overlays onto <see cref="MoeLayerWeights"/> before the Vulkan upload —
    /// the safetensors loader currently upcasts every MoE projection to F32, so
    /// production code paths can't carry a Q8_0 MoE projection through to Vulkan
    /// without this hook.
    /// </summary>
    internal static VulkanTransformerModel BuildFromPrebuiltWeights(
        VulkanDevice device, ModelConfig config, TransformerWeights cpuWeights, string spvDir)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(cpuWeights);
        ArgumentNullException.ThrowIfNull(spvDir);

        RejectUnsupportedArchitecture(config);
        return BuildModel(device, ownsDevice: false, config, cpuWeights, spvDir, gguf: null);
    }

    private static VulkanTransformerModel BuildModel(
        VulkanDevice device, bool ownsDevice, ModelConfig config,
        TransformerWeights cpuWeights, string spvDir, GgufFile? gguf)
    {
        // Q8_0 matrices stay on device as 34-byte blocks — the forward pass
        // below dispatches them through the Q8_0 GEMV / GEMM kernels. Other
        // quant types are still dequantised to FP32 at upload.
        var weights = VulkanWeights.Upload(device, cpuWeights, config.NumLayers);

        // MoE detection: any layer with non-null Moe in CPU weights. We
        // don't gate on config.Moe because Mixtral/Qwen-MoE configs may
        // mark "MoE everywhere" while DeepSeek-V2 first_k_dense_replace
        // makes only the tail layers MoE — any-layer check is the
        // conservative trigger.
        bool hasMoe = false;
        int moeNumExperts = 0, moeTopK = 0, moeIntermediate = 0;
        int moeSharedIntermediate = 0, moeNumSharedExperts = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            ref readonly var lwTmp = ref cpuWeights.Layers[i];
            if (lwTmp.Moe is not null)
            {
                hasMoe = true;
                moeNumExperts = Math.Max(moeNumExperts, lwTmp.Moe.NumExperts);
                moeTopK = Math.Max(moeTopK, lwTmp.Moe.NumExpertsPerTok);
                moeIntermediate = Math.Max(moeIntermediate, lwTmp.Moe.IntermediateSize);
                if (lwTmp.Moe.HasSharedExpert)
                {
                    moeSharedIntermediate = Math.Max(moeSharedIntermediate, lwTmp.Moe.SharedIntermediateSize);
                    moeNumSharedExperts = Math.Max(moeNumSharedExperts, lwTmp.Moe.NumSharedExperts);
                }
            }
        }

        bool hasMla = config.MlaConfig is not null;
        int mlaNumHeads = hasMla ? config.NumAttentionHeads : 0;
        int mlaQkNope = hasMla ? config.MlaConfig!.QkNopeHeadDim : 0;
        int mlaQkRope = hasMla ? config.MlaConfig!.QkRopeHeadDim : 0;
        int mlaVHead = hasMla ? config.MlaConfig!.VHeadDim : 0;
        int mlaQLora = hasMla ? config.MlaConfig!.QLoraRank : 0;
        int mlaKvLora = hasMla ? config.MlaConfig!.KvLoraRank : 0;
        float mlaScale = 0f, mlaRopeTheta = 0f;
        if (hasMla)
        {
            int qkHeadDim = mlaQkNope + mlaQkRope;
            float yarnMul = config.MlaConfig!.ComputeYarnSoftmaxScaleMultiplier();
            mlaScale = yarnMul / MathF.Sqrt(qkHeadDim);
            mlaRopeTheta = config.MlaConfig!.RopeTheta;
        }

        // dp4a MMVQ decode path (issue #46). Quantize the F32 activation to
        // Q8_1 on the device, then run an integer-dot (dotPacked4x8AccSatEXT)
        // GEMV against the int8 Q8_0 weights. Created only when the device
        // advertises VK_KHR_shader_integer_dot_product AND the SPVs are present
        // AND the env-var opt-out is unset. Both null => the router falls back
        // to the F32-in Q8_0 GEMV (MatMulQ8_0Kernel). The activation scratch
        // (Q8_1Xq / Q8_1Xds) is only allocated when both kernels are live.
        QuantizeQ8_1Kernel? quantizeQ8_1 = null;
        MatMulQ8_0MmvqKernel? matmulQ8Mmvq = null;
        MatMulQ4KMmvqKernel? matmulQ4KMmvq = null;
        MatMulQ6KMmvqKernel? matmulQ6KMmvq = null;
        MatMulQ5KMmvqKernel? matmulQ5KMmvq = null;
        MatMulQ2KMmvqKernel? matmulQ2KMmvq = null;
        MatMulQ3KMmvqKernel? matmulQ3KMmvq = null;
        MatMulIq4NlMmvqKernel? matmulIq4NlMmvq = null;
        MatMulIq4XsMmvqKernel? matmulIq4XsMmvq = null;
        if (!IsMmvqDisabled() && device.HasIntegerDotProduct)
        {
            quantizeQ8_1 = QuantizeQ8_1Kernel.TryCreate(device, spvDir);
            matmulQ8Mmvq = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir);
            // Q4_K MMVQ (issue #52) reuses quantizeQ8_1; it is an independent
            // weight-format path, so a missing Q4_K SPV must not disable Q8_0
            // MMVQ (and vice versa).
            matmulQ4KMmvq = MatMulQ4KMmvqKernel.TryCreate(device, spvDir);
            // Q6_K / Q5_K MMVQ (issue #338), Q2_K / Q3_K MMVQ (issue #339) — same
            // independent-path policy.
            matmulQ6KMmvq = MatMulQ6KMmvqKernel.TryCreate(device, spvDir);
            matmulQ5KMmvq = MatMulQ5KMmvqKernel.TryCreate(device, spvDir);
            matmulQ2KMmvq = MatMulQ2KMmvqKernel.TryCreate(device, spvDir);
            matmulQ3KMmvq = MatMulQ3KMmvqKernel.TryCreate(device, spvDir);
            // IQ4_NL / IQ4_XS MMVQ (issue #339) — codebook-lookup quants.
            matmulIq4NlMmvq = MatMulIq4NlMmvqKernel.TryCreate(device, spvDir);
            matmulIq4XsMmvq = MatMulIq4XsMmvqKernel.TryCreate(device, spvDir);
            // The activation quantizer is shared. Keep it alive if ANY MMVQ
            // weight kernel is present; tear the whole path down only when the
            // quantizer is missing or no weight kernel loaded.
            if (quantizeQ8_1 is null
                || (matmulQ8Mmvq is null && matmulQ4KMmvq is null
                    && matmulQ6KMmvq is null && matmulQ5KMmvq is null
                    && matmulQ2KMmvq is null && matmulQ3KMmvq is null
                    && matmulIq4NlMmvq is null && matmulIq4XsMmvq is null))
            {
                quantizeQ8_1?.Dispose();
                matmulQ8Mmvq?.Dispose();
                matmulQ4KMmvq?.Dispose();
                matmulQ6KMmvq?.Dispose();
                matmulQ5KMmvq?.Dispose();
                matmulQ2KMmvq?.Dispose();
                matmulQ3KMmvq?.Dispose();
                matmulIq4NlMmvq?.Dispose();
                matmulIq4XsMmvq?.Dispose();
                quantizeQ8_1 = null;
                matmulQ8Mmvq = null;
                matmulQ4KMmvq = null;
                matmulQ6KMmvq = null;
                matmulQ5KMmvq = null;
                matmulQ2KMmvq = null;
                matmulQ3KMmvq = null;
                matmulIq4NlMmvq = null;
                matmulIq4XsMmvq = null;
            }
        }
        // The decode activation scratch (Q8_1Xq/Xds) is shared by all MMVQ
        // weight paths; allocate it when the quantizer plus at least one weight
        // kernel are live.
        bool mmvqEnabled = quantizeQ8_1 is not null
            && (matmulQ8Mmvq is not null || matmulQ4KMmvq is not null
                || matmulQ6KMmvq is not null || matmulQ5KMmvq is not null
                || matmulQ2KMmvq is not null || matmulQ3KMmvq is not null
                || matmulIq4NlMmvq is not null || matmulIq4XsMmvq is not null);

        // dp4a MMQ prefill path (issue #50). The compute-bound seqLen>1 analogue
        // of MMVQ: quantize the F32 activation B-matrix to Q8_1 row-wise, then
        // run an integer-dot (dotPacked4x8AccSatEXT) GEMM against the int8 Q8_0
        // weights instead of the dequant→FP GEMM. Same gate as MMVQ (device
        // integer-dot support + SPVs present) plus its own env-var opt-out. Both
        // null => the router falls back to the coopmat / scalar Q8_0 GEMM.
        QuantizeQ8_1RowsKernel? quantizeQ8_1Rows = null;
        MatMulQ8_0MmqKernel? matmulQ8Mmq = null;
        MatMulQ4KMmqKernel? matmulQ4KMmq = null;
        MatMulQ6KMmqKernel? matmulQ6KMmq = null;
        MatMulQ5KMmqKernel? matmulQ5KMmq = null;
        MatMulIq4XsMmqKernel? matmulIq4XsMmq = null;
        MatMulIq4NlMmqKernel? matmulIq4NlMmq = null;
        MatMulQ2KMmqKernel? matmulQ2KMmq = null;
        MatMulQ3KMmqKernel? matmulQ3KMmq = null;
        if (!IsMmqDisabled() && device.HasIntegerDotProduct)
        {
            quantizeQ8_1Rows = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir);
            if (quantizeQ8_1Rows is not null)
            {
                // Each MMQ weight format is independent; they all share the one
                // row-wise Q8_1 activation quantizer. Keep the quantizer if any
                // MMQ kernel loaded (issue #340 adds Q4_K alongside Q8_0).
                matmulQ8Mmq = MatMulQ8_0MmqKernel.TryCreate(device, spvDir);
                matmulQ4KMmq = MatMulQ4KMmqKernel.TryCreate(device, spvDir);
                matmulQ6KMmq = MatMulQ6KMmqKernel.TryCreate(device, spvDir);
                matmulQ5KMmq = MatMulQ5KMmqKernel.TryCreate(device, spvDir);
                matmulIq4XsMmq = MatMulIq4XsMmqKernel.TryCreate(device, spvDir);
                matmulIq4NlMmq = MatMulIq4NlMmqKernel.TryCreate(device, spvDir);
                matmulQ2KMmq = MatMulQ2KMmqKernel.TryCreate(device, spvDir);
                matmulQ3KMmq = MatMulQ3KMmqKernel.TryCreate(device, spvDir);
                if (matmulQ8Mmq is null && matmulQ4KMmq is null && matmulQ6KMmq is null
                    && matmulQ5KMmq is null && matmulIq4XsMmq is null && matmulIq4NlMmq is null
                    && matmulQ2KMmq is null && matmulQ3KMmq is null)
                {
                    quantizeQ8_1Rows.Dispose();
                    quantizeQ8_1Rows = null;
                }
            }
        }
        // The prefill MMQ activation scratch (Q8_1XqRows / Q8_1XdsRows) shares
        // the mmvqEnabled allocation gate — both pairs need the same device
        // integer-dot support. Enable the rows scratch when any MMQ path is live.
        bool mmqEnabled = quantizeQ8_1Rows is not null
            && (matmulQ8Mmq is not null || matmulQ4KMmq is not null || matmulQ6KMmq is not null
                || matmulQ5KMmq is not null || matmulIq4XsMmq is not null
                || matmulIq4NlMmq is not null || matmulQ2KMmq is not null || matmulQ3KMmq is not null);

        // Gemma-4 has a dual head dim (sliding 256 / global 512) and dual KV-head
        // count (sliding 8 / global 2). Size the Q/K/V/AttnOutput scratch for the
        // MAX over the two so global layers fit; the matmul writes each layer's
        // actual (smaller-or-equal) packed dims and the attention/rope kernels
        // use the per-layer dims. No-op for non-Gemma-4 (global dims null → max
        // with 0 = the base dim).
        int stateHeadDim = Math.Max(config.HeadDim, config.GlobalHeadDim ?? 0);
        int stateKvHeads = Math.Max(config.NumKvHeads, config.NumGlobalKvHeads ?? 0);
        var state = new VulkanForwardState(device,
            config.HiddenSize, config.NumAttentionHeads, stateKvHeads,
            stateHeadDim, config.IntermediateSize, config.VocabSize,
            initialSeqLen: 1,
            mlaNumHeads: mlaNumHeads,
            mlaQkNopeHeadDim: mlaQkNope,
            mlaQkRopeHeadDim: mlaQkRope,
            mlaVHeadDim: mlaVHead,
            mlaQLoraRank: mlaQLora,
            mlaKvLoraRank: mlaKvLora,
            moeNumExperts: moeNumExperts,
            moeTopK: moeTopK,
            moeIntermediateSize: moeIntermediate,
            moeSharedIntermediateSize: moeSharedIntermediate,
            moeNumSharedExperts: moeNumSharedExperts,
            // The forward state's mmvqEnabled gate allocates BOTH the decode
            // (Q8_1Xq/Xds) and prefill (Q8_1XqRows/XdsRows) activation scratch.
            // Enable it when either the decode MMVQ or prefill MMQ path is live.
            mmvqEnabled: mmvqEnabled || mmqEnabled,
            gemma4DualFfn: config.Gemma4DualFfn);

        var matmul = MatMulF32Kernel.Create(device, spvDir);
        var matmulQ8 = MatMulQ8_0Kernel.Create(device, spvDir);
        var matmulQ8Gemm = MatMulQ8_0GemmKernel.Create(device, spvDir);
        // Q2_K + Q3_K GEMV + GEMM — completes the K-quant family on Vulkan.
        // Always created; the dispatcher routes per device-side QuantizationType.
        var matmulQ2K = MatMulQ2KGemvF32Kernel.Create(device, spvDir);
        var matmulQ2KGemm = MatMulQ2KGemmF32Kernel.Create(device, spvDir);
        var matmulQ3K = MatMulQ3KGemvF32Kernel.Create(device, spvDir);
        var matmulQ3KGemm = MatMulQ3KGemmF32Kernel.Create(device, spvDir);
        // Q4_K_M GEMV + GEMM — Phase 1 of K-quant work. Always created; the
        // RecordMatmul dispatcher routes per device-side QuantizationType.
        var matmulQ4K = MatMulQ4KGemvF32Kernel.Create(device, spvDir);
        var matmulQ4KGemm = MatMulQ4KGemmF32Kernel.Create(device, spvDir);
        // Q5_K_M GEMV + GEMM — Phase 1 sibling of Q4_K. Always created.
        var matmulQ5K = MatMulQ5KGemvF32Kernel.Create(device, spvDir);
        var matmulQ5KGemm = MatMulQ5KGemmF32Kernel.Create(device, spvDir);
        // Q6_K_M GEMV + GEMM — Phase 1 sibling of Q4_K / Q5_K. Always created.
        var matmulQ6K = MatMulQ6KGemvF32Kernel.Create(device, spvDir);
        var matmulQ6KGemm = MatMulQ6KGemmF32Kernel.Create(device, spvDir);
        // IQ4_NL / IQ4_XS GEMV + GEMM — IQ-family follow-up. Always created;
        // dispatcher routes per device-side QuantizationType. Most-used IQ
        // quants in production (Llama-3.1 / Qwen2.5 IQ4_XS).
        var matmulIq4Nl = MatMulIq4NlGemvF32Kernel.Create(device, spvDir);
        var matmulIq4NlGemm = MatMulIq4NlGemmF32Kernel.Create(device, spvDir);
        var matmulIq4Xs = MatMulIq4XsGemvF32Kernel.Create(device, spvDir);
        var matmulIq4XsGemm = MatMulIq4XsGemmF32Kernel.Create(device, spvDir);
        // IQ2 (XXS / XS / S) GEMV + GEMM — IQ-family follow-up. Always
        // created. Codebooks (3 grid SSBOs + ksigns) uploaded once and
        // shared across all 6 IQ2 matmul kernels.
        var iq2Codebooks = Iq2Codebooks.Create(device);
        // dp4a MMVQ decode path for the IQ codebook quants (issue #339). Reuses the
        // shared codebooks + the Q8_1 activation scratch (allocated above whenever any
        // K-quant MMVQ kernel is live). null when integer-dot is unavailable / SPV
        // missing / env opt-out → RecordMatmul falls back to the F32-in GEMV.
        var matmulIq2XxsMmvq = IsMmvqDisabled() ? null : MatMulIq2XxsMmvqKernel.TryCreate(device, spvDir, iq2Codebooks);
        var matmulIq2XsMmvq  = IsMmvqDisabled() ? null : MatMulIq2XsMmvqKernel.TryCreate(device, spvDir, iq2Codebooks);
        var matmulIq2SMmvq   = IsMmvqDisabled() ? null : MatMulIq2SMmvqKernel.TryCreate(device, spvDir, iq2Codebooks);
        var matmulIq2Xxs     = MatMulIq2XxsGemvF32Kernel.CreateWithCodebooks(device, spvDir, iq2Codebooks);
        var matmulIq2XxsGemm = MatMulIq2XxsGemmF32Kernel.CreateWithCodebooks(device, spvDir, iq2Codebooks);
        var matmulIq2Xs      = MatMulIq2XsGemvF32Kernel.CreateWithCodebooks(device, spvDir, iq2Codebooks);
        var matmulIq2XsGemm  = MatMulIq2XsGemmF32Kernel.CreateWithCodebooks(device, spvDir, iq2Codebooks);
        var matmulIq2S       = MatMulIq2SGemvF32Kernel.CreateWithCodebooks(device, spvDir, iq2Codebooks);
        var matmulIq2SGemm   = MatMulIq2SGemmF32Kernel.CreateWithCodebooks(device, spvDir, iq2Codebooks);
        var iq3Codebooks = Iq3Codebooks.Create(device);
        var matmulIq3XxsMmvq = IsMmvqDisabled() ? null : MatMulIq3XxsMmvqKernel.TryCreate(device, spvDir, iq3Codebooks);
        var matmulIq3Xxs     = MatMulIq3XxsGemvF32Kernel.CreateWithCodebooks(device, spvDir, iq3Codebooks);
        var matmulIq3XxsGemm = MatMulIq3XxsGemmF32Kernel.CreateWithCodebooks(device, spvDir, iq3Codebooks);
        var matmulIq3SMmvq   = IsMmvqDisabled() ? null : MatMulIq3SMmvqKernel.TryCreate(device, spvDir, iq3Codebooks);
        var matmulIq3S       = MatMulIq3SGemvF32Kernel.CreateWithCodebooks(device, spvDir, iq3Codebooks);
        var matmulIq3SGemm   = MatMulIq3SGemmF32Kernel.CreateWithCodebooks(device, spvDir, iq3Codebooks);
        // IQ1_S GEMV + GEMM — smallest GGUF quant (~1.5-1.7 bpw). Always
        // created; closes the IQ-family Vulkan matmul coverage.
        var matmulIq1SMmvq = IsMmvqDisabled() ? null : MatMulIq1SMmvqKernel.TryCreate(device, spvDir);
        var matmulIq1S = MatMulIq1SGemvF32Kernel.Create(device, spvDir);
        var matmulIq1SGemm = MatMulIq1SGemmF32Kernel.Create(device, spvDir);
        // I2_S GEMV + GEMM — BitNet b1.58 ternary. Always created; the dispatcher
        // routes per device-side QuantizationType.
        var matmulI2S = MatMulI2SGemvF32Kernel.Create(device, spvDir);
        var matmulI2SGemm = MatMulI2SGemmF32Kernel.Create(device, spvDir);
        // F16 / BF16 native matmul kernels — Phase 8. Always created; the
        // dispatcher routes per device-side QuantizationType. The F16 GEMM
        // coopmat path is opportunistic (null on devices without coopmat).
        var matmulF16 = MatMulF16GemvF32Kernel.Create(device, spvDir);
        var matmulF16Gemm = MatMulF16GemmF32Kernel.Create(device, spvDir);
        MatMulF16GemmCoopmatKernel? matmulF16GemmCoopmat = null;
        if (device.HasCooperativeMatrix)
        {
            try { matmulF16GemmCoopmat = MatMulF16GemmCoopmatKernel.Create(device, spvDir); }
            catch (InvalidOperationException) { /* No usable F16 tile shape; stay on scalar. */ }
        }
        var matmulBf16 = MatMulBf16GemvF32Kernel.Create(device, spvDir);
        var matmulBf16Gemm = MatMulBf16GemmF32Kernel.Create(device, spvDir);
        // Optional coopmat prefill GEMM — 3.8× over scalar on AMD RDNA3.5 at
        // Llama-3 4096² N=64. Null on devices without KHR_cooperative_matrix;
        // router falls back to the scalar GEMM. Tolerance: abs 5e-3 / rel 5e-3
        // end-to-end (looser than the 1e-4 / 1e-3 of the scalar path because
        // KHR_coopmat only offers F16 operands — see the coopmat kernel tests).
        MatMulQ8_0GemmCoopmatKernel? matmulQ8GemmCoopmat = null;
        if (device.HasCooperativeMatrix)
        {
            try { matmulQ8GemmCoopmat = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir); }
            catch (InvalidOperationException) { /* Kernel threw: no usable tile shape. Stay on scalar. */ }
        }
        // Optional decode-path fusion of rmsnorm + Q8_0 GEMV. Older builds
        // without the fused SPV stay working — TryCreate returns null and
        // the router falls back to the standalone pair.
        RmsNormMatmulQ8_0FusedKernel? rmsnormMatmulQ8Fused =
            RmsNormMatmulQ8_0FusedKernel.TryCreate(device, spvDir);

        var rmsnorm = RmsNormF32Kernel.Create(device, spvDir);
        var rope = RopeF32Kernel.Create(device, spvDir);
        var attention = AttentionF32Kernel.Create(device, spvDir);
        // Optional Flash-Attention kernel for the GQA prefill path. Disabled by
        // env-var opt-out, by missing SPV (older builds), or when the model's
        // head_dim exceeds the shader bound — every gate falls back to the
        // legacy per-token attention kernel.
        VulkanFlashAttentionF32Kernel? flashAttention =
            IsFlashAttentionDisabled() || config.HeadDim > VulkanFlashAttentionF32Kernel.MaxHeadDim
                ? null
                : VulkanFlashAttentionF32Kernel.TryCreate(device, spvDir);
        var swiglu = SwiGluF32Kernel.Create(device, spvDir);
        // Gemma GeGLU-tanh FFN activation — created only for GELUTanh models so
        // the SwiGLU path stays untouched. The embedding-scale kernel is
        // created only when Config.EmbeddingScale is set (Gemma sqrt(hidden)).
        GeGluTanhF32Kernel? geglu =
            config.ActivationFunction == ActivationFunction.GELUTanh
                ? GeGluTanhF32Kernel.Create(device, spvDir)
                : null;
        // BitNet gated squared-ReLU FFN activation — created only for ReluSquared
        // models so the SwiGLU/GeGLU paths stay untouched.
        ReLU2GluF32Kernel? relu2glu =
            config.ActivationFunction == ActivationFunction.ReluSquared
                ? ReLU2GluF32Kernel.Create(device, spvDir)
                : null;
        ScaleInplaceF32Kernel? embedScale =
            config.EmbeddingScale is float es && es != 1.0f
                ? ScaleInplaceF32Kernel.Create(device, spvDir)
                : null;
        var add = AddKernel.Create(device, spvDir);
        var biasAdd = BiasAddF32Kernel.Create(device, spvDir);

        AttentionMlaF32Kernel? mlaAttention = null;
        RopeMlaF32Kernel? mlaRope = null;
        MlaKvSplitF32Kernel? mlaKvSplit = null;
        if (hasMla)
        {
            mlaAttention = AttentionMlaF32Kernel.Create(device, spvDir);
            mlaRope = RopeMlaF32Kernel.Create(device, spvDir);
            mlaKvSplit = MlaKvSplitF32Kernel.Create(device, spvDir);
        }

        MoeTopKSoftmaxF32Kernel? moeTopkSoftmax = null;
        MoeIndexedMatmulF32Kernel? moeIndexedMatmul = null;
        MoeIndexedMatmulQ8_0F32Kernel? moeIndexedMatmulQ8 = null;
        MoeIndexedMatmulQ4_KF32Kernel? moeIndexedMatmulQ4K = null;
        MoeIndexedMatmulQ5_1F32Kernel? moeIndexedMatmulQ5_1 = null;
        MoeIndexedMatmulTiledF32Kernel? moeIndexedMatmulTiled = null;
        MoeIndexedLoraDeltaF32Kernel? moeIndexedLoraDelta = null;
        MoeExpertOffsetsKernel? moeExpertOffsets = null;
        MoeExpandGroupByExpertF32Kernel? moeExpandGroupByExpert = null;
        MoeGroupedMatmulF16CoopmatKernel? moeGroupedMatmulF16Coopmat = null;
        MoeUngroupScatterF32Kernel? moeUngroupScatter = null;
        MoeWeightedScatterF32Kernel? moeWeightedScatter = null;
        MoeBroadcastF32Kernel? moeBroadcast = null;
        MoeSigmoidGatedAddF32Kernel? moeSigmoidGatedAdd = null;
        if (hasMoe)
        {
            moeTopkSoftmax = MoeTopKSoftmaxF32Kernel.Create(device, spvDir);
            moeIndexedMatmul = MoeIndexedMatmulF32Kernel.Create(device, spvDir);
            moeIndexedMatmulQ8 = MoeIndexedMatmulQ8_0F32Kernel.Create(device, spvDir);
            if (config.Gemma4DualFfn)
            {
                // Gemma-4 quantized experts: fused gate_up Q4_K + down Q5_1 stay
                // quantized on device (the real 26B's F32 experts are tens of GB).
                moeIndexedMatmulQ4K = MoeIndexedMatmulQ4_KF32Kernel.Create(device, spvDir);
                moeIndexedMatmulQ5_1 = MoeIndexedMatmulQ5_1F32Kernel.Create(device, spvDir);
            }
            moeIndexedMatmulTiled = MoeIndexedMatmulTiledF32Kernel.Create(device, spvDir);
            moeIndexedLoraDelta = MoeIndexedLoraDeltaF32Kernel.Create(device, spvDir);
            moeExpertOffsets = MoeExpertOffsetsKernel.Create(device, spvDir);
            moeExpandGroupByExpert = MoeExpandGroupByExpertF32Kernel.Create(device, spvDir);
            if (device.HasCooperativeMatrix)
            {
                try { moeGroupedMatmulF16Coopmat = MoeGroupedMatmulF16CoopmatKernel.Create(device, spvDir); }
                catch (InvalidOperationException) { /* Stay on indexed MoE when no usable coopmat tile is exposed. */ }
            }
            moeUngroupScatter = MoeUngroupScatterF32Kernel.Create(device, spvDir);
            moeWeightedScatter = MoeWeightedScatterF32Kernel.Create(device, spvDir);
            moeBroadcast = MoeBroadcastF32Kernel.Create(device, spvDir);
            moeSigmoidGatedAdd = MoeSigmoidGatedAddF32Kernel.Create(device, spvDir);
        }

        // Optional fused LoRA delta-GEMV — TryCreate so older builds without
        // the .spv blob fall back to the un-fused 4-dispatch path. Always
        // attempted (no MoE/MLA gating) because LoRA can target any standard
        // q/k/v/o + gate/up/down projection on the dense path.
        LoraDeltaGemvFusedF32Kernel? loraDeltaGemvFused =
            LoraDeltaGemvFusedF32Kernel.TryCreate(device, spvDir);

        var submit = device.CreateSubmitContext();

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        RoPEType ropeType = config.RoPEConfig?.Type ?? RoPEType.Norm;
        var ropeVariant = ropeType == RoPEType.NeoX ? RopeF32Kernel.Variant.NeoX : RopeF32Kernel.Variant.Norm;

        int slidingWindow = config.SlidingWindowSize ?? 0;

        return new VulkanTransformerModel(
            device, ownsDevice,
            config, weights, cpuWeights, state,
            matmul, matmulQ8, matmulQ8Gemm, matmulQ8GemmCoopmat,
            matmulQ2K, matmulQ2KGemm,
            matmulQ3K, matmulQ3KGemm,
            matmulQ4K, matmulQ4KGemm,
            matmulQ5K, matmulQ5KGemm,
            matmulQ6K, matmulQ6KGemm,
            matmulIq4Nl, matmulIq4NlGemm,
            matmulIq4Xs, matmulIq4XsGemm,
            iq2Codebooks,
            matmulIq2Xxs, matmulIq2XxsGemm,
            matmulIq2Xs, matmulIq2XsGemm,
            matmulIq2S, matmulIq2SGemm,
            iq3Codebooks,
            matmulIq3Xxs, matmulIq3XxsGemm,
            matmulIq3S, matmulIq3SGemm,
            matmulIq1S, matmulIq1SGemm,
            matmulI2S, matmulI2SGemm,
            matmulF16, matmulF16Gemm, matmulF16GemmCoopmat,
            matmulBf16, matmulBf16Gemm,
            rmsnormMatmulQ8Fused,
            quantizeQ8_1, matmulQ8Mmvq,
            matmulQ4KMmvq,
            matmulQ6KMmvq,
            matmulQ5KMmvq,
            matmulQ2KMmvq,
            matmulQ3KMmvq,
            matmulIq4NlMmvq,
            matmulIq4XsMmvq,
            matmulIq2XxsMmvq,
            matmulIq2XsMmvq,
            matmulIq2SMmvq,
            matmulIq3XxsMmvq,
            matmulIq3SMmvq,
            matmulIq1SMmvq,
            quantizeQ8_1Rows, matmulQ8Mmq,
            matmulQ4KMmq,
            matmulQ6KMmq,
            matmulQ5KMmq,
            matmulIq4XsMmq,
            matmulIq4NlMmq,
            matmulQ2KMmq,
            matmulQ3KMmq,
            rmsnorm, rope, attention, flashAttention, swiglu, geglu, relu2glu, embedScale, add,
            biasAdd,
            mlaAttention, mlaRope, mlaKvSplit,
            moeTopkSoftmax, moeIndexedMatmul, moeIndexedMatmulQ8,
            moeIndexedMatmulQ4K, moeIndexedMatmulQ5_1,
            moeIndexedMatmulTiled, moeIndexedLoraDelta,
            moeExpertOffsets, moeExpandGroupByExpert, moeGroupedMatmulF16Coopmat, moeUngroupScatter,
            moeWeightedScatter, moeBroadcast,
            moeSigmoidGatedAdd,
            loraDeltaGemvFused,
            submit,
            gguf,
            ropeTheta, ropeDim, ropeVariant, slidingWindow,
            mlaNumHeads, mlaQkNope, mlaQkRope, mlaVHead,
            mlaScale, mlaRopeTheta);
    }

    /// <summary>
    /// Env-var opt-out for the Flash-Attention prefill path. Set
    /// <c>DOTLLM_VULKAN_DISABLE_FLASH_ATTENTION=1</c> to force every dispatch
    /// onto the legacy per-token <see cref="AttentionF32Kernel"/>.
    /// </summary>
    internal const string DisableFlashAttentionEnvVar = "DOTLLM_VULKAN_DISABLE_FLASH_ATTENTION";

    internal static bool IsFlashAttentionDisabled() =>
        Environment.GetEnvironmentVariable(DisableFlashAttentionEnvVar) == "1";

    /// <summary>
    /// Env-var opt-out for the dp4a MMVQ decode path (issue #46). Set
    /// <c>DOTLLM_VULKAN_DISABLE_MMVQ=1</c> to force decode Q8_0 GEMV onto the
    /// F32-in <see cref="MatMulQ8_0Kernel"/> (e.g. to A/B benchmark or to work
    /// around a driver integer-dot bug). When unset, the path is used whenever
    /// the device advertises integer-dot-product support and the SPVs exist.
    /// </summary>
    internal const string DisableMmvqEnvVar = "DOTLLM_VULKAN_DISABLE_MMVQ";

    internal static bool IsMmvqDisabled() =>
        Environment.GetEnvironmentVariable(DisableMmvqEnvVar) == "1";

    /// <summary>
    /// Env-var opt-out for the shared-activation-quant optimisation on the MMVQ
    /// decode path. Set <c>DOTLLM_VULKAN_MMVQ_NO_SHARE=1</c> to quantize the
    /// activation per projection (the original per-call behaviour) instead of
    /// once per same-input group (Q/K/V, gate/up). Used by the share-vs-no-share
    /// bit-identical parity test and for A/B benchmarking. When unset, sharing is
    /// on whenever the MMVQ decode path is wired.
    /// </summary>
    internal const string MmvqNoShareEnvVar = "DOTLLM_VULKAN_MMVQ_NO_SHARE";

    internal static bool IsMmvqShareDisabled() =>
        Environment.GetEnvironmentVariable(MmvqNoShareEnvVar) == "1";

    /// <summary>
    /// Env-var opt-out for the dp4a MMQ prefill path (issue #50). Set
    /// <c>DOTLLM_VULKAN_DISABLE_MMQ=1</c> to force the seqLen&gt;1 Q8_0 GEMM onto
    /// the dequant→FP GEMM (coopmat where available, else scalar
    /// <see cref="MatMulQ8_0GemmKernel"/>) — e.g. to A/B benchmark MMQ vs the FP
    /// GEMM or to work around a driver integer-dot bug. When unset, MMQ is used
    /// for prefill Q8_0 whenever the device advertises integer-dot support and
    /// the SPVs exist.
    /// </summary>
    internal const string DisableMmqEnvVar = "DOTLLM_VULKAN_DISABLE_MMQ";

    internal static bool IsMmqDisabled() =>
        Environment.GetEnvironmentVariable(DisableMmqEnvVar) == "1";

    /// <summary>
    /// Records the attention dispatch using Flash-Attention when the kernel
    /// is available, head_dim fits the shader bound, and the sequence is in
    /// the prefill regime (seqQ &gt; 1); falls back to the legacy per-token
    /// kernel for decode (seqQ == 1). Decode keeps the legacy path because
    /// Flash-Attention's amortisation factor only kicks in across multiple
    /// Q-rows. Both paths now honour the Gemma-2/3 <paramref name="softCap"/>
    /// and the Gemma-3 <paramref name="scaleOverride"/> (QPAS) — see
    /// <c>attention_f32.comp</c> / <c>attention_flash_f32.comp</c> for the
    /// shared push-constant layout.
    /// </summary>
    private void RecordAttention(
        nint cmdBuf,
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset, int slidingWindow,
        float softCap = 0.0f, float scaleOverride = 0.0f,
        AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0)
    {
        if (_flashAttention is not null && seqQ > 1 && headDim <= VulkanFlashAttentionF32Kernel.MaxHeadDim)
        {
            _flashAttention.Record(cmdBuf, q, k, v, output,
                seqQ: seqQ, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: positionOffset, slidingWindow: slidingWindow,
                softCap: softCap, scaleOverride: scaleOverride,
                maskMode: maskMode, prefixLen: prefixLen);
            return;
        }
        _attention.Record(cmdBuf, q, k, v, output,
            seqQ: seqQ, seqKv: seqKv,
            numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
            positionOffset: positionOffset, slidingWindow: slidingWindow,
            softCap: softCap, scaleOverride: scaleOverride,
            maskMode: maskMode, prefixLen: prefixLen);
    }

    /// <summary>
    /// Returns the effective sliding-window size for <paramref name="layer"/>.
    /// Honours <see cref="ModelConfig.PerLayerSlidingWindow"/> when set (each
    /// entry may be null for full attention or a positive int for sliding);
    /// otherwise falls back to the model-wide <see cref="_slidingWindow"/>.
    /// Used for Gemma 3's interleaved local/global pattern. Mirrors the CPU
    /// <c>TransformerModel.GetLayerSlidingWindow</c> helper.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int GetLayerSlidingWindow(int layer)
    {
        var perLayer = Config.PerLayerSlidingWindow;
        if (perLayer is not null && (uint)layer < (uint)perLayer.Count)
            return perLayer[layer] ?? 0;
        return _slidingWindow;
    }

    /// <summary>
    /// Returns the effective <c>scaleOverride</c> push-constant value for the
    /// attention dispatch. <c>0</c> = use the shader default <c>1/sqrt(headDim)</c>;
    /// when <see cref="ModelConfig.QueryPreAttnScalar"/> is set (Gemma 3),
    /// returns <c>1/sqrt(QPAS)</c> so the shader uses Gemma's alternative scale.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float GetAttentionScaleOverride()
    {
        return Config.QueryPreAttnScalar is float qpas && qpas > 0.0f
            ? 1.0f / MathF.Sqrt(qpas)
            : 0.0f;
    }

    /// <summary>Gemma-4 per-layer KV-head count. Delegates to the single source of truth
    /// <see cref="ModelConfig.GetLayerKvHeads"/> (full layers use <see cref="ModelConfig.NumGlobalKvHeads"/>).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int GemmaLayerKvHeads(int layer) => Config.GetLayerKvHeads(layer);

    /// <summary>
    /// Gemma-4 per-layer RoPE (theta, rotated-dim count). Full-attention layers use the
    /// global table with a partial-rotary factor applied to the full head; sliding layers
    /// use the local table at full rotation. Mirrors the CPU <c>GetLayerRope</c> /
    /// global-rope-dim derivation exactly.
    /// </summary>
    private (float Theta, int RopeDim) GemmaLayerRope(int layer)
    {
        if (Config.GlobalRoPEConfig is RoPEConfig gcfg && Config.IsFullAttentionLayer(layer))
        {
            int baseDim = gcfg.DimensionCount > 0 ? gcfg.DimensionCount : (Config.GlobalHeadDim ?? Config.HeadDim);
            float prf = Config.PartialRotaryFactor ?? 1.0f;
            int rotated = (int)MathF.Floor(prf * baseDim) & ~1; // round down to even
            if (rotated < 2) rotated = 2;
            return (gcfg.Theta, Math.Min(rotated, baseDim));
        }
        var s = Config.RoPEConfig!.Value;
        return (s.Theta, s.DimensionCount > 0 ? s.DimensionCount : Config.HeadDim);
    }

    /// <summary>
    /// Lazily-allocated unit-gamma (all-ones) [maxHeadDim] vector for Gemma-4's
    /// weight-less V-norm (per-kv-head RMSNorm with no learned scale). Host-visible
    /// so the single tiny upload is trivial; read-only thereafter.
    /// </summary>
    private VulkanDevice.Buffer Gemma4OnesVec()
    {
        if (_gemma4OnesVec is null)
        {
            // Sized for the widest unit-gamma RMSNorm consumer: the per-kv-head V-norm
            // (n = headDim) AND the DiffusionGemma canvas weight-less rms_noscale (n = hidden).
            int len = Math.Max(Math.Max(Config.HeadDim, Config.GlobalHeadDim ?? 0), Config.HiddenSize);
            var ones = new float[len];
            Array.Fill(ones, 1.0f);
            var buf = _device.Allocate((long)len * sizeof(float));
            _device.Upload(ones.AsSpan(), buf);
            _gemma4OnesVec = buf;
        }
        return _gemma4OnesVec;
    }

    /// <summary>
    /// DiffusionGemma region split P (prompt length) for the current forward: the
    /// Hybrid prefix length clamped to [0, seqLen]. Canvas rows are [P, seqLen).
    /// Returns 0 when not a diffusion-region forward (AR gemma4 / Causal / Bidirectional)
    /// — so every region delta is inert and the AR path is byte-identical.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int DiffusionRegionPrefix(int seqLen)
    {
        if (Config.DiffusionConfig is null) return 0;
        if (_diffusionMaskMode != AttentionMaskMode.Hybrid) return 0;
        int p = _diffusionPrefixLen;
        if (p < 0) p = 0;
        if (p > seqLen) p = seqLen;
        return p;
    }

    private static void RejectUnsupportedArchitecture(ModelConfig config)
    {
        if (config.HybridLayout is not null || config.SsmConfig is not null || config.Mamba3Config is not null)
            throw new NotSupportedException("Hybrid SSM / Mamba architectures are not supported on the Vulkan backend yet.");
        // MLA: latent / hybrid cache modes are CPU-only for now; the Vulkan
        // path runs the Phase A expanded cache (MlaVulkanKvCache) which
        // matches the CPU expanded path. Reject the latent flags so callers
        // don't silently get a different attention math.
        if (config.MlaConfig is { UseLatentCache: true } or { UseHybridMlaCache: true })
            throw new NotSupportedException(
                "MLA latent / hybrid KV-cache modes are not supported on the Vulkan backend yet; use the default expanded cache.");
        // Gemma-4 MoE autoregressive forward (gemma4) IS now supported on the
        // Vulkan backend (RecordGemma4Attention + RecordGemma4Ffn; experts
        // host-dequantised to F32 at load). DiffusionGemma generation additionally
        // needs the non-causal canvas attention + region-aware embed / self-cond,
        // which are not wired here yet — but the cacheless gemma4 backbone runs.
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    /// <remarks>
    /// Causal (the default) routes to the byte-identical autoregressive forward. The non-causal
    /// modes (Bidirectional / Hybrid) are supported ONLY for DiffusionGemma (a gemma4 backbone with
    /// a non-null <see cref="ModelConfig.DiffusionConfig"/>): the canvas region attends bidirectionally
    /// (Hybrid keeps the prompt prefix causal), the canvas embedding gets the region-aware
    /// rms_noscale (+ self-conditioning), and the LM head runs over every position — returning
    /// <c>[seqLen, vocab]</c> for the denoise generator. A non-null KV-cache with a non-causal mask
    /// is rejected (PR-3): the diffusion forward is cacheless (the PKV seam carries its own cache).
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
        IKvCache? kvCache, ILoraAdapter? adapter, AttentionMaskSpec maskSpec)
    {
        if (maskSpec.IsCausal)
            return Forward(tokenIds, positions, deviceId, kvCache, adapter);

        if (Config.DiffusionConfig is null || !Config.Gemma4DualFfn)
            throw new NotSupportedException(
                $"{nameof(VulkanTransformerModel)} supports non-causal attention only for DiffusionGemma "
                + $"(gemma4 backbone with a diffusion config); mask mode {maskSpec.Mode}.");
        if (kvCache is not null)
            throw new NotSupportedException(
                "A non-null KV-cache with a non-causal mask is not supported (the diffusion forward is cacheless).");
        if (adapter is not null)
            throw new NotSupportedException("LoRA adapters are not supported on the DiffusionGemma forward.");

        _diffusionMaskMode = maskSpec.Mode;
        _diffusionPrefixLen = maskSpec.Mode == AttentionMaskMode.Hybrid ? maskSpec.PrefixLength : 0;
        try
        {
            return Forward(tokenIds, positions, deviceId, kvCache: null);
        }
        finally
        {
            _diffusionMaskMode = AttentionMaskMode.Causal;
            _diffusionPrefixLen = 0;
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Supported on the DiffusionGemma backbone (gemma4 tower + diffusion config). PKV reuses the
    /// per-layer prompt K/V across denoise steps; the cacheless diffusion forward remains correct
    /// and is unaffected when PKV is off.
    /// </remarks>
    public bool SupportsDiffusionPromptKv => Config.DiffusionConfig is not null && Config.Gemma4DualFfn;

    /// <inheritdoc/>
    public void DiffusionPrefillPromptKv(
        ReadOnlySpan<int> promptTokens, ReadOnlySpan<int> positions, DiffusionPromptKvStore store)
    {
        ArgumentNullException.ThrowIfNull(store);
        if (!SupportsDiffusionPromptKv)
            throw new NotSupportedException("DiffusionPrefillPromptKv requires a DiffusionGemma (gemma4) model.");
        if (promptTokens.Length == 0)
            throw new ArgumentException("Prompt must be non-empty for PKV prefill.", nameof(promptTokens));
        if (promptTokens.Length != positions.Length)
            throw new ArgumentException("promptTokens and positions length mismatch.", nameof(positions));

        int p = promptTokens.Length;
        int numLayers = Config.NumLayers;
        // Per-layer KV block width (nKvHead*headDim) — sliding vs global layers differ.
        Span<int> kvBlockElems = numLayers <= 256 ? stackalloc int[numLayers] : new int[numLayers];
        for (int l = 0; l < numLayers; l++)
            kvBlockElems[l] = GemmaLayerKvHeads(l) * Config.GetLayerHeadDim(l);
        // The CPU-side store mirrors PromptLen for the generator's bookkeeping.
        store.BeginPrefill(p, kvBlockElems);
        // The device-resident store holds the actual K/V (allocated/grown OUTSIDE recording).
        _pkvStore ??= new VulkanDiffusionPromptKv(_device, numLayers);
        _pkvStore.BeginPrefill(p, kvBlockElems);
        if (_pkvStore.LastBeginReallocated) InvalidateKernelCaches();

        _pkvPromptLen = p;
        _pkvPhase = DiffusionKvPhase.Prefill;
        // Hybrid(P) with P == seqLen ⇒ every prompt row is in the causal prefix (the region
        // deltas are inert exactly as the unified forward's prompt rows). No SC on the prompt.
        _diffusionMaskMode = AttentionMaskMode.Hybrid;
        _diffusionPrefixLen = p;
        try
        {
            // Run the layer stack only — the prefill needs the captured K/V, not the LM head.
            Forward(promptTokens, positions, deviceId: -1, kvCache: null);
        }
        finally
        {
            _pkvPhase = DiffusionKvPhase.None;
            _diffusionMaskMode = AttentionMaskMode.Causal;
            _diffusionPrefixLen = 0;
        }
    }

    /// <inheritdoc/>
    public ITensor DiffusionDecodeWithPromptKv(
        ReadOnlySpan<int> canvasTokens, ReadOnlySpan<int> positions, int deviceId,
        DiffusionPromptKvStore store)
    {
        ArgumentNullException.ThrowIfNull(store);
        if (!SupportsDiffusionPromptKv)
            throw new NotSupportedException("DiffusionDecodeWithPromptKv requires a DiffusionGemma (gemma4) model.");
        if (_pkvStore is null || _pkvStore.PromptLen <= 0)
            throw new InvalidOperationException("PKV store is empty — run DiffusionPrefillPromptKv first.");
        if (canvasTokens.Length == 0)
            throw new ArgumentException("Canvas must be non-empty for PKV decode.", nameof(canvasTokens));
        if (canvasTokens.Length != positions.Length)
            throw new ArgumentException("canvasTokens and positions length mismatch.", nameof(positions));

        int c = canvasTokens.Length;
        int p = _pkvStore.PromptLen;
        // Pre-allocate the K/V concat scratch (max per-layer KV width × (P+C)) BEFORE recording —
        // growth may invalidate descriptor caches, which is unsafe mid-command-buffer.
        int maxKvStride = 0;
        for (int l = 0; l < Config.NumLayers; l++)
            maxKvStride = Math.Max(maxKvStride, GemmaLayerKvHeads(l) * Config.GetLayerHeadDim(l));
        EnsurePkvConcat((long)(p + c) * maxKvStride * sizeof(float));

        _pkvPromptLen = p;
        _pkvPhase = DiffusionKvPhase.Decode;
        // Bidirectional spec so the region embed (p == 0 ⇒ all C rows are canvas) and region
        // scalar treat every decode row as canvas; the actual [prompt|canvas] attention is built
        // from the cached prompt K/V inside RecordGemma4Attention.
        _diffusionMaskMode = AttentionMaskMode.Bidirectional;
        _diffusionPrefixLen = 0;
        try
        {
            return Forward(canvasTokens, positions, deviceId, kvCache: null);
        }
        finally
        {
            _pkvPhase = DiffusionKvPhase.None;
            _diffusionMaskMode = AttentionMaskMode.Causal;
            _diffusionPrefixLen = 0;
        }
    }

    /// <summary>Lazily (re)allocates the device-local PKV K/V concat scratch to <paramref name="bytes"/> each.</summary>
    private (VulkanDevice.Buffer k, VulkanDevice.Buffer v) EnsurePkvConcat(long bytes)
    {
        if (_pkvKConcat is null || _pkvConcatCapacityBytes < bytes)
        {
            _pkvKConcat?.Dispose();
            _pkvVConcat?.Dispose();
            _pkvKConcat = _device.AllocateDeviceLocal(bytes);
            _pkvVConcat = _device.AllocateDeviceLocal(bytes);
            _pkvConcatCapacityBytes = bytes;
            InvalidateKernelCaches();
        }
        return (_pkvKConcat!, _pkvVConcat!);
    }

    /// <inheritdoc/>
    public void SetDiffusionSelfCond(ReadOnlySpan<float> prevCanvasLogits, int canvasLen, float scUse)
    {
        if (scUse > 0f && !prevCanvasLogits.IsEmpty && canvasLen > 0)
        {
            int need = canvasLen * Config.VocabSize;
            if (prevCanvasLogits.Length < need)
                throw new ArgumentException(
                    $"prevCanvasLogits length {prevCanvasLogits.Length} < canvasLen*vocab {need}.",
                    nameof(prevCanvasLogits));
            if (_scPrevLogits is null || _scPrevLogits.Length < need)
                _scPrevLogits = new float[need];
            prevCanvasLogits[..need].CopyTo(_scPrevLogits);
            _scCanvasLen = canvasLen;
            _scUse = scUse;
        }
        else
        {
            _scCanvasLen = 0;
            _scUse = 0f;
        }
    }

    /// <summary>
    /// LoRA-aware forward. When <paramref name="adapter"/> is non-null, each
    /// adapted projection (q/k/v/o + gate/up/down on the standard transformer
    /// path) adds <c>scale × (x · B) · A</c> on top of the base projection.
    /// When null, this is byte-equivalent to the 4-arg overload.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mirrors the CPU <c>TransformerModel.Forward</c> 5-arg overload: a
    /// per-call <see cref="_currentLora"/> field is set/cleared via
    /// try/finally around the inner forward; <see cref="MaybeApplyLoraDelta"/>
    /// at every standard projection site in the inner forward checks the
    /// field and applies the LoRA delta as an extra dispatch chain.
    /// </para>
    /// <para>
    /// MLA-attention (DeepSeek-V2/V3) and MoE-FFN adapter targets are
    /// rejected at validation time — they are deferred follow-ups.
    /// </para>
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, ILoraAdapter? adapter)
    {
        if (adapter is null)
            return Forward(tokenIds, positions, deviceId, kvCache);

        ValidateAdapterForModel(adapter);

        // Resolve / lazy-upload device-side LoRA buffers. Subsequent forwards
        // with the same adapter hit the cache and pay zero upload cost.
        var vkLora = _loraCache.GetOrAdd(adapter);

        // Size LoRA scratch for this adapter's largest output dim. seqLen is
        // honoured via _state._capacitySeqLen (set in EnsureCapacity below
        // when the inner Forward runs); EnsureLoraScratch is called from
        // here AFTER EnsureCapacity has run by routing through the inner
        // Forward — but the inner Forward needs the scratch to exist
        // already. So we EnsureCapacity ourselves first, then EnsureLora.
        int seqLen = tokenIds.Length;
        if (seqLen == 0) throw new ArgumentException("tokenIds must be non-empty.", nameof(tokenIds));
        bool resized = _state.EnsureCapacity(seqLen);
        bool loraResized = _state.EnsureLoraScratch(vkLora.Rank, vkLora.MaxOutputDim);
        if (resized || loraResized)
            InvalidateKernelCaches();

        _currentLora = vkLora;
        try
        {
            return Forward(tokenIds, positions, deviceId, kvCache);
        }
        finally
        {
            _currentLora = null;
        }
    }

    /// <summary>
    /// Validates that <paramref name="adapter"/> is compatible with this
    /// model. Projection support is governed by <see cref="ILoraAdapter.IsCompatible"/>;
    /// unsupported runtime sites are no-ops rather than validation failures.
    /// </summary>
    private void ValidateAdapterForModel(ILoraAdapter adapter)
    {
        if (!adapter.IsCompatible(Config))
            throw new InvalidOperationException(
                $"LoRA adapter '{adapter.Name}' is not compatible with the loaded model "
                + "(layer count, hidden size, or per-projection dimensions mismatch).");
    }

    /// <summary>
    /// Phase 5f — Vulkan ForwardBatch override. Fuses the intra-block matmuls
    /// (RMSNorm + Q/K/V/O + gate/up/down + lm_head) across <c>N</c> in-flight
    /// sequences into single dispatches at <c>seqLen = Σ N_i</c>, while keeping
    /// attention per-seq (each sequence has its own <see cref="VulkanKvCache"/>
    /// + own positionOffset). The win amortises ~30 layers × 7 dispatch / submit
    /// overheads × N sequences for the per-iter overhead, on top of the GEMM
    /// vs. <c>N</c> GEMV throughput edge.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Sequences that target features the batched path doesn't yet cover are
    /// peeled off and run through the existing per-seq <c>Forward(... adapter)</c>
    /// loop. The fall-through set is: MLA layers (<c>_weights.Layers[*].Mla != null</c>),
    /// MoE layers, LoRA-active sequences, and sequences whose KV-cache is not a
    /// <see cref="VulkanKvCache"/>. The remaining "simple" sequences run through
    /// the batched path below.
    /// </para>
    /// <para>
    /// The per-seq attention sub-loop copies each seq's Q-slice from the batched
    /// <see cref="VulkanForwardState.Q"/> into <see cref="VulkanForwardBatchScratch.PerSeqQ"/>,
    /// updates that seq's KV-cache from the batched <see cref="VulkanForwardState.K"/>/
    /// <see cref="VulkanForwardState.V"/> slices, runs <see cref="AttentionF32Kernel.Record"/>
    /// with the seq's positionOffset, then copies the per-seq attention output
    /// back into the batched <see cref="VulkanForwardState.AttnOutput"/> slot.
    /// The PerSeqQ / PerSeqAttn scratch is reused across both layers and
    /// sequences within a layer — barriers serialise them.
    /// </para>
    /// <para>
    /// For the lm_head: only the last hidden row of each simple sequence is
    /// fed into the head (matching <c>Forward</c>'s contract that the Vulkan
    /// return is <c>[1, vocab]</c>). The last rows are gathered into
    /// <see cref="VulkanForwardBatchScratch.LastRowHidden"/> at slot <c>i</c>,
    /// a single batched RMSNorm + lm_head matmul produces
    /// <c>[N_simple, vocab]</c> in <see cref="VulkanForwardBatchScratch.BatchedLogits"/>,
    /// and per-seq <c>[1, vocab]</c> host tensors are split out post-submit.
    /// </para>
    /// </remarks>
    public IReadOnlyList<ITensor> ForwardBatch(
        IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();
        if (requests.Count == 1)
        {
            var r0 = requests[0];
            return new[] { Forward(r0.TokenIds.Span, r0.Positions.Span,
                                   deviceId, r0.KvCache, r0.Adapter) };
        }

        // Partition into "simple" (fused batched path) and "complex" (per-seq fallback).
        // Simple = no LoRA adapter, KvCache is VulkanKvCache, and the model itself
        // carries no MLA / MoE layer (dense VulkanTransformerModel does not support
        // MoE — that's VulkanQwen3MoeHybridTransformerModel — but MLA can appear
        // in DeepSeek-V2/V3 dense hosts and falls through to per-seq for now).
        bool modelHasMlaOrMoe = false;
        for (int layer = 0; layer < Config.NumLayers && !modelHasMlaOrMoe; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];
            if (lw.Mla is not null || lw.Moe is not null) modelHasMlaOrMoe = true;
        }

        // Build the simple / complex index lists. Preserve input order in the result.
        var simpleIdx = new List<int>(requests.Count);
        var complexIdx = new List<int>(requests.Count);
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            bool simple = !modelHasMlaOrMoe
                       && r.Adapter is null
                       && r.KvCache is VulkanKvCache;
            (simple ? simpleIdx : complexIdx).Add(i);
        }

        var results = new ITensor[requests.Count];

        // Complex / fallback — execute via existing per-seq Forward (which correctly
        // handles MLA / MoE / LoRA via its own dispatch paths).
        foreach (int i in complexIdx)
        {
            var r = requests[i];
            results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.Adapter);
        }

        // Fewer than 2 simple seqs: no batching benefit; just run through per-seq Forward.
        if (simpleIdx.Count < 2)
        {
            foreach (int i in simpleIdx)
            {
                var r = requests[i];
                results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.Adapter);
            }
            return results;
        }

        ForwardBatchSimple(requests, simpleIdx, deviceId, results);
        return results;
    }

    /// <summary>
    /// Inner batched dispatch for the subset of <paramref name="requests"/>
    /// pre-classified as "simple" (no LoRA adapter, VulkanKvCache,
    /// non-MLA / non-MoE model). Mirrors the layer-loop structure of
    /// <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?)"/>
    /// but dispatches every intra-block matmul + RMSNorm + RoPE + residual at
    /// <c>seqLen = Σ N_i</c> against the existing <see cref="VulkanForwardState"/>
    /// scratch. Attention is per-seq via the <see cref="VulkanForwardBatchScratch"/>
    /// staging buffers because the kernel API binds whole buffers — see the
    /// class summary on <see cref="VulkanForwardBatchScratch"/>.
    /// </summary>
    private unsafe void ForwardBatchSimple(
        IReadOnlyList<SequenceForwardRequest> requests,
        List<int> simpleIdx, int deviceId, ITensor[] results)
    {
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;
        int qDim = numHeads * headDim;
        int kvDim = numKvHeads * headDim;
        int maxSeq = Config.MaxSequenceLength;

        int simpleCount = simpleIdx.Count;
        int totalTokens = 0;
        int maxSingleSeq = 0;
        for (int s = 0; s < simpleCount; s++)
        {
            int n = requests[simpleIdx[s]].TokenIds.Length;
            if (n <= 0) throw new ArgumentException("Per-seq tokenIds must be non-empty.", nameof(requests));
            totalTokens += n;
            if (n > maxSingleSeq) maxSingleSeq = n;
        }

        // Resize state scratch for the batched seqLen and ensure batch-scratch buffers exist.
        bool scratchResized = _state.EnsureCapacity(totalTokens);

        _batchScratch ??= new VulkanForwardBatchScratch(_device, hiddenSize, qDim, vocabSize);
        bool batchResized = _batchScratch.EnsureCapacity(
            maxSingleSeqTokens: maxSingleSeq, batchSeqs: simpleCount);

        if (scratchResized || batchResized)
            InvalidateKernelCaches();

        // Build packed tokenIds + positions for the batched dispatch. Per-seq
        // positions are honoured by RoPE (one rotation angle per row); the
        // batched RoPE dispatch reads PositionsBuffer[totalTokens] and rotates
        // each row independently. Validate everything host-side first — a bad
        // id throws cleanly without leaving the submit context half-written.
        int[] packedTokens = new int[totalTokens];
        int[] packedPositions = new int[totalTokens];
        int off = 0;
        for (int s = 0; s < simpleCount; s++)
        {
            var r = requests[simpleIdx[s]];
            int n = r.TokenIds.Length;
            if (r.Positions.Length != n)
                throw new ArgumentException("Per-seq tokenIds and positions must have the same length.", nameof(requests));
            var idsSpan = r.TokenIds.Span;
            var posSpan = r.Positions.Span;
            for (int t = 0; t < n; t++)
            {
                int id = idsSpan[t];
                int pos = posSpan[t];
                if ((uint)id >= (uint)vocabSize)
                    throw new ArgumentOutOfRangeException(nameof(requests), $"Token id {id} is out of range.");
                if ((uint)pos >= (uint)maxSeq)
                    throw new ArgumentOutOfRangeException(nameof(requests), $"Position {pos} exceeds max sequence length {maxSeq}.");
                packedTokens[off + t] = id;
                packedPositions[off + t] = pos;
            }
            off += n;
        }

        // Upload packed positions to PositionsBuffer (sized for totalTokens by EnsureCapacity above).
        var posBytes = MemoryMarshal.AsBytes(packedPositions.AsSpan(0, totalTokens));
        _device.Upload(posBytes, _state.PositionsBuffer);

        // Begin the single per-batch command buffer.
        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);

        _state.ResetHiddenSlot();

        // Embedding gather — batched: one vkCmdCopyBuffer per token writes into
        // HiddenState[t, :] for t in [0, totalTokens). Order matches packedTokens
        // (= per-seq concatenation in simpleIdx order).
        RecordEmbeddingGather(cmdBuf, packedTokens.AsSpan(0, totalTokens));
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        // Per-seq token offset into the batched buffer. Used inside the layer loop
        // to slice Q/K/V/AttnOutput per sequence (computed once, reused per layer).
        int[] seqOffsets = new int[simpleCount];
        off = 0;
        for (int s = 0; s < simpleCount; s++)
        {
            seqOffsets[s] = off;
            off += requests[simpleIdx[s]].TokenIds.Length;
        }

        long qRowBytes = (long)qDim * sizeof(float);
        long kvRowBytes = (long)kvDim * sizeof(float);
        long hiddenRowBytes = (long)hiddenSize * sizeof(float);

        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];

            // ── Attention block ────────────────────────────────────────────
            // Batched RMSNorm → Q/K/V — all at seqLen=totalTokens against the
            // existing state scratch.
            _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.AttnNormWeight, _state.NormOutput,
                rowCount: totalTokens, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            RecordMatmul(cmdBuf, lw.Q, lw.QDeviceQuantType, _state.NormOutput, _state.Q,
                lw.QOutputDim, lw.QInputDim, totalTokens);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            RecordMatmul(cmdBuf, lw.K, lw.KDeviceQuantType, _state.NormOutput, _state.K,
                lw.KOutputDim, lw.KInputDim, totalTokens);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            RecordMatmul(cmdBuf, lw.V, lw.VDeviceQuantType, _state.NormOutput, _state.V,
                lw.VOutputDim, lw.VInputDim, totalTokens);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Optional Q/K/V biases — add across all totalTokens rows.
            if (lw.QBias is not null) _biasAdd.Record(cmdBuf, _state.Q, lw.QBias, totalTokens, lw.QOutputDim);
            if (lw.KBias is not null) _biasAdd.Record(cmdBuf, _state.K, lw.KBias, totalTokens, lw.KOutputDim);
            if (lw.VBias is not null) _biasAdd.Record(cmdBuf, _state.V, lw.VBias, totalTokens, lw.VOutputDim);
            if (lw.QBias is not null || lw.KBias is not null || lw.VBias is not null)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Batched RoPE — reads packed positions [totalTokens] and rotates each
            // row independently. Per-seq position semantics are preserved by the
            // packed positions array.
            _rope.Record(cmdBuf, _state.Q, _state.K, _state.PositionsBuffer,
                seqLen: totalTokens, numHeads: numHeads, numKvHeads: numKvHeads,
                headDim: headDim, ropeDim: _ropeDim, theta: _ropeTheta,
                variant: _ropeVariant);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Per-seq attention sub-loop. Each seq:
            //   (1) copy this seq's K/V slice from _state.K/V into its VulkanKvCache
            //       at positions[s]. The existing RecordUpdate assumes srcOffset=0, so
            //       we issue the vkCmdCopyBuffer commands inline with the per-seq offset.
            //   (2) copy this seq's Q slice from _state.Q into PerSeqQ (offset 0).
            //   (3) run attention with PerSeqQ + cache K/V into PerSeqAttn.
            //   (4) copy PerSeqAttn back into _state.AttnOutput at this seq's offset.
            // PerSeqQ / PerSeqAttn are shared across seqs within the layer — barriers
            // serialise consecutive attention dispatches.
            var perSeqQ = _batchScratch.PerSeqQ!;
            var perSeqAttn = _batchScratch.PerSeqAttn!;

            for (int s = 0; s < simpleCount; s++)
            {
                var req = requests[simpleIdx[s]];
                int nS = req.TokenIds.Length;
                int seqOff = seqOffsets[s];
                var vkCache = (VulkanKvCache)req.KvCache;

                // (1) KV-cache update — copy this seq's contiguous slice from
                // _state.K / _state.V into the cache at positions[s][0]..[N_s-1].
                // The simple case is when positions are contiguous-ascending —
                // single 2-region copy. We restrict to contiguous-ascending for
                // batched-mode simplicity; the scheduler invariant is that per-seq
                // positions are always either decode (1 token at currentLength) or
                // prefill (contiguous from 0). If we encounter a non-contiguous
                // seq we fall back: copy row-by-row. We assert and rely on the
                // scheduler emitting contiguous positions per seq.
                int basePos = req.Positions.Span[0];
                bool contiguous = true;
                for (int t = 1; t < nS; t++)
                {
                    if (req.Positions.Span[t] != basePos + t) { contiguous = false; break; }
                }

                if (contiguous)
                {
                    var kRegion = new VkBufferCopy
                    {
                        srcOffset = (ulong)((long)seqOff * kvRowBytes),
                        dstOffset = (ulong)((long)basePos * kvRowBytes),
                        size = (ulong)((long)nS * kvRowBytes),
                    };
                    VulkanApi.vkCmdCopyBuffer(cmdBuf, _state.K.Handle,
                        vkCache.GetKeysBuffer(layer).Handle, 1, kRegion);
                    VulkanApi.vkCmdCopyBuffer(cmdBuf, _state.V.Handle,
                        vkCache.GetValuesBuffer(layer).Handle, 1, kRegion);
                }
                else
                {
                    for (int t = 0; t < nS; t++)
                    {
                        int pos = req.Positions.Span[t];
                        var region = new VkBufferCopy
                        {
                            srcOffset = (ulong)((long)(seqOff + t) * kvRowBytes),
                            dstOffset = (ulong)((long)pos * kvRowBytes),
                            size = (ulong)kvRowBytes,
                        };
                        VulkanApi.vkCmdCopyBuffer(cmdBuf, _state.K.Handle,
                            vkCache.GetKeysBuffer(layer).Handle, 1, region);
                        VulkanApi.vkCmdCopyBuffer(cmdBuf, _state.V.Handle,
                            vkCache.GetValuesBuffer(layer).Handle, 1, region);
                    }
                }
                // Advance cache visible length on the LAST layer only — the cache's
                // CurrentLength is a single counter shared across layers, and each
                // layer's RecordUpdate normally advances it. Match the existing
                // per-layer Forward semantics: advance every layer (idempotent
                // because each layer sets the same maxPos+1 value).
                int maxPosThisSeq = basePos;
                for (int t = 1; t < nS; t++)
                {
                    int p = req.Positions.Span[t];
                    if (p > maxPosThisSeq) maxPosThisSeq = p;
                }
                vkCache.SetCurrentLength(Math.Max(vkCache.CurrentLength, maxPosThisSeq + 1));

                // (2) Copy this seq's Q slice into PerSeqQ.
                var qRegion = new VkBufferCopy
                {
                    srcOffset = (ulong)((long)seqOff * qRowBytes),
                    dstOffset = 0,
                    size = (ulong)((long)nS * qRowBytes),
                };
                VulkanApi.vkCmdCopyBuffer(cmdBuf, _state.Q.Handle, perSeqQ.Handle, 1, qRegion);

                // TRANSFER → COMPUTE before the attention dispatch — attention reads
                // PerSeqQ (compute, just-written by vkCmdCopyBuffer = TRANSFER) AND
                // cache K/V (compute, just-written by vkCmdCopyBuffer = TRANSFER).
                KernelSupport.TransferToComputeBarrier(cmdBuf);

                // (3) Attention dispatch — honour Gemma-3 per-layer sliding,
                // attn soft-cap, and query-pre-attn scalar (no-op on every
                // other architecture).
                int seqKv = vkCache.CurrentLength;
                int positionOffset = basePos;
                int layerSlidingWindow = GetLayerSlidingWindow(layer);
                float attnScaleOverride = GetAttentionScaleOverride();
                float attnSoftCap = Config.AttnLogitSoftcap ?? 0.0f;
                RecordAttention(cmdBuf, perSeqQ, vkCache.GetKeysBuffer(layer), vkCache.GetValuesBuffer(layer),
                    perSeqAttn,
                    seqQ: nS, seqKv: seqKv,
                    numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                    positionOffset: positionOffset, slidingWindow: layerSlidingWindow,
                    softCap: attnSoftCap, scaleOverride: attnScaleOverride);
                KernelSupport.ComputeToTransferBarrier(cmdBuf);

                // (4) Copy PerSeqAttn back into _state.AttnOutput at this seq's offset.
                var attnRegion = new VkBufferCopy
                {
                    srcOffset = 0,
                    dstOffset = (ulong)((long)seqOff * qRowBytes),
                    size = (ulong)((long)nS * qRowBytes),
                };
                VulkanApi.vkCmdCopyBuffer(cmdBuf, perSeqAttn.Handle, _state.AttnOutput.Handle, 1, attnRegion);
            }
            // All per-seq attention dispatches done — TRANSFER → COMPUTE so the
            // batched O projection reads the freshly-scattered _state.AttnOutput.
            KernelSupport.TransferToComputeBarrier(cmdBuf);

            // BitNet Sub-LN: in-place RMSNorm over the attention output before the
            // batched output projection. No-op for non-BitNet (AttnSubNormWeight null).
            if (lw.AttnSubNormWeight is { } attnSubNorm)
            {
                _rmsnorm.Record(cmdBuf, _state.AttnOutput, attnSubNorm, _state.AttnOutput,
                    rowCount: totalTokens, n: lw.OInputDim, eps: eps);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // Batched O projection → NormOutput.
            RecordMatmul(cmdBuf, lw.O, lw.ODeviceQuantType, _state.AttnOutput, _state.NormOutput,
                lw.OOutputDim, lw.OInputDim, totalTokens);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            if (lw.OBias is not null)
            {
                _biasAdd.Record(cmdBuf, _state.NormOutput, lw.OBias, totalTokens, lw.OOutputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // Residual add #1: AddScratch = Residual + NormOutput at totalTokens × hidden.
            _add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch, totalTokens * hiddenSize);
            _state.RotateHiddenSlot();
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // ── FFN block (dense — model has no MoE layer in the simple-batched path) ──
            _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.FfnNormWeight, _state.NormOutput,
                rowCount: totalTokens, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            RecordMatmul(cmdBuf, lw.Gate, lw.GateDeviceQuantType, _state.NormOutput, _state.FfnGate,
                lw.GateOutputDim, lw.GateInputDim, totalTokens);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            RecordMatmul(cmdBuf, lw.Up, lw.UpDeviceQuantType, _state.NormOutput, _state.FfnUp,
                lw.UpOutputDim, lw.UpInputDim, totalTokens);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            if (lw.GateBias is not null) _biasAdd.Record(cmdBuf, _state.FfnGate, lw.GateBias, totalTokens, lw.GateOutputDim);
            if (lw.UpBias is not null) _biasAdd.Record(cmdBuf, _state.FfnUp, lw.UpBias, totalTokens, lw.UpOutputDim);
            if (lw.GateBias is not null || lw.UpBias is not null)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // FFN gate activation: gated squared-ReLU (BitNet) when _relu2glu is
            // non-null, otherwise the standard SwiGLU. (No GeGLU in this path — the
            // simple-batched path is never taken by Gemma's four-norm layout.)
            if (_relu2glu is not null)
                _relu2glu.Record(cmdBuf, _state.FfnGate, _state.FfnUp, _state.SiluOutput, totalTokens * intermediateSize);
            else
                _swiglu.Record(cmdBuf, _state.FfnGate, _state.FfnUp, _state.SiluOutput, totalTokens * intermediateSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // BitNet Sub-LN: in-place RMSNorm over the gated FFN intermediate before
            // the batched down projection. No-op for non-BitNet (FfnSubNormWeight null).
            if (lw.FfnSubNormWeight is { } ffnSubNorm)
            {
                _rmsnorm.Record(cmdBuf, _state.SiluOutput, ffnSubNorm, _state.SiluOutput,
                    rowCount: totalTokens, n: lw.DownInputDim, eps: eps);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            RecordMatmul(cmdBuf, lw.Down, lw.DownDeviceQuantType, _state.SiluOutput, _state.NormOutput,
                lw.DownOutputDim, lw.DownInputDim, totalTokens);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            if (lw.DownBias is not null)
            {
                _biasAdd.Record(cmdBuf, _state.NormOutput, lw.DownBias, totalTokens, lw.DownOutputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // Residual add #2 + rotate.
            _add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch, totalTokens * hiddenSize);
            _state.RotateHiddenSlot();

            if (layer < Config.NumLayers - 1)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }

        // ── lm_head fan-out ────────────────────────────────────────────────
        // Each simple seq's LAST hidden row → LastRowHidden[s, :]. Matches the
        // Vulkan Forward contract (lm_head only on the last token, returns
        // [1, vocab]). Batched lm_head: one RMSNorm + matmul at seqLen=N_simple
        // against LastRowHidden, producing BatchedLogits[N_simple, vocab].
        var lastRowHidden = _batchScratch.LastRowHidden!;
        var batchedLogits = _batchScratch.BatchedLogits!;

        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        for (int s = 0; s < simpleCount; s++)
        {
            int nS = requests[simpleIdx[s]].TokenIds.Length;
            int seqOff = seqOffsets[s];
            int lastRowAbs = seqOff + nS - 1;
            var region = new VkBufferCopy
            {
                srcOffset = (ulong)((long)lastRowAbs * hiddenRowBytes),
                dstOffset = (ulong)((long)s * hiddenRowBytes),
                size = (ulong)hiddenRowBytes,
            };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, _state.HiddenState.Handle, lastRowHidden.Handle, 1, region);
        }
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        _rmsnorm.Record(cmdBuf, lastRowHidden, _weights.OutputNormWeight, lastRowHidden,
            rowCount: simpleCount, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        RecordMatmul(cmdBuf, _weights.OutputWeight, _weights.OutputDeviceQuantType,
            lastRowHidden, batchedLogits,
            _weights.OutputOutputDim, _weights.OutputInputDim, seqLen: simpleCount);

        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        // Download per-seq vocab rows and split into individual [1, vocab] host tensors.
        // BatchedLogits is host-visible (Allocate, not AllocateDeviceLocal — see
        // VulkanForwardBatchScratch.EnsureCapacity), so the device.Download path
        // works as for the per-seq Forward.
        unsafe
        {
            // Stage the full batch into a managed buffer once, then split per-seq.
            // simpleCount × vocabSize fits int — Vulkan vocabSize cap is ~256k and
            // simpleCount is bounded by the scheduler's MaxActiveSequences (64 today).
            int totalLogits = checked(simpleCount * vocabSize);
            float[] hostBuf = new float[totalLogits];
            _device.Download(batchedLogits, hostBuf.AsSpan());
            // Gemma 2/3 final-logit soft-cap over the full batched block.
            // No-op when Config.FinalLogitSoftcap is null.
            ApplyFinalLogitSoftcapHost(hostBuf.AsSpan(0, totalLogits));
            for (int s = 0; s < simpleCount; s++)
            {
                int reqIdx = simpleIdx[s];
                var shape = new TensorShape(1, vocabSize);
                var tensor = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
                var src = hostBuf.AsSpan(s * vocabSize, vocabSize);
                src.CopyTo(new Span<float>((void*)tensor.DataPointer, vocabSize));
                results[reqIdx] = tensor;
            }
        }
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        if (tokenIds.Length != positions.Length)
            throw new ArgumentException("tokenIds and positions must have the same length.");

        int seqLen = tokenIds.Length;
        if (seqLen == 0) throw new ArgumentException("tokenIds must be non-empty.", nameof(tokenIds));

        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;

        bool scratchResized = _state.EnsureCapacity(seqLen);

        // Descriptor sets cache buffer handles. When scratch is re-allocated
        // every cached set becomes stale and must be dropped — otherwise the
        // next dispatch binds a dangling VkBuffer. In steady-state decode
        // (seqLen = 1 after the initial prefill) scratch never grows, so the
        // cache stays warm across forwards.
        if (scratchResized)
            InvalidateKernelCaches();

        // Diffusion forward: pre-allocate the all-position logits buffer and (when
        // self-conditioning is active this step) the SC-signal buffer BEFORE recording
        // begins — both may grow and invalidate descriptor caches, which is only safe
        // outside an open command buffer. No-op on the AR / Causal path.
        bool diffusionForward = _diffusionMaskMode != AttentionMaskMode.Causal && Config.DiffusionConfig is not null;
        if (diffusionForward)
        {
            EnsureDiffusionLogits(seqLen, vocabSize);
            int regionP0 = DiffusionRegionPrefix(seqLen);
            int canvasLen0 = seqLen - regionP0;
            bool scThisStep = _scUse > 0f && _cpuWeights.SelfCond is not null
                && _scPrevLogits is not null && _scCanvasLen == canvasLen0 && canvasLen0 > 0;
            if (scThisStep)
                EnsureScSigBuffer(canvasLen0 * hiddenSize);
        }

        // 1. Validate token IDs (done host-side; cheap), then upload only
        //    positions host→device. The embedding table is device-local and
        //    populated once at construction; per-token rows are gathered into
        //    HiddenState via vkCmdCopyBuffer recorded on the same command
        //    buffer (see RecordEmbeddingGather below).
        ValidateTokenIds(tokenIds);
        UploadPositions(positions);

        // 2. Begin the single per-forward command buffer and record the
        //    whole transformer. Bias-add host steps split the forward into
        //    multiple submits (one per distinct set of biases we need to
        //    pause for); everything else stays inside the pipelined path.
        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);

        // Canonicalise the hidden-slot rotation to slot 0 so the embedding
        // gather below writes into the same physical buffer every forward
        // (keeps kernel descriptor-set caches warm across decode steps).
        _state.ResetHiddenSlot();

        // Gather one embedding row per token from the device-local
        // TokenEmbedding buffer into HiddenState[t, :]. The first consumer
        // is the first RMSNorm's COMPUTE read on HiddenState — hidden/residual
        // now alias (no TRANSFER copy in between) so a TRANSFER→COMPUTE
        // barrier is all we need.
        RecordEmbeddingGather(cmdBuf, tokenIds);
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        // Gemma sqrt(hidden) embedding scaling — multiply the gathered
        // embedding rows in-place before the first layer. No-op on every
        // architecture that leaves Config.EmbeddingScale null (_embedScale is
        // null), so non-Gemma output is byte-identical.
        if (_embedScale is not null)
        {
            _embedScale.Record(cmdBuf, _state.HiddenState, seqLen * hiddenSize,
                Config.EmbeddingScale!.Value);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }

        // DiffusionGemma region embedding: the canvas rows [P, seqLen) get an
        // EXTRA weight-less rms_norm (no scale) — the zero-self-conditioning path
        // (diffusion-gemma.cpp dg_canvas_embed). With self-conditioning ON (step > 0)
        // the canvas rows first receive sc_sig before the rms_noscale. Gated on a
        // diffusion-region forward (Hybrid) ⇒ AR gemma4 / non-diffusion are
        // byte-identical (regionP == 0 ⇒ no-op). Implemented host-light: the
        // contiguous canvas tail is copied into NormOutput[0..], normed there with
        // a unit-gamma weight, and copied back — reuses the shared kernels with no
        // offset-binding or new shader.
        // A diffusion forward (Hybrid OR Bidirectional) normalises its canvas rows. For
        // Hybrid the canvas is the tail [P, seqLen); for Bidirectional (PKV decode / LLaDA-
        // style) every row is canvas (regionP == 0). Mirrors CPU: p = Hybrid ? prefix : 0.
        if (_diffusionMaskMode != AttentionMaskMode.Causal && _pkvPhase != DiffusionKvPhase.Prefill)
        {
            int regionP = _diffusionMaskMode == AttentionMaskMode.Hybrid ? DiffusionRegionPrefix(seqLen) : 0;
            int canvasLen = seqLen - regionP;
            if (canvasLen > 0)
                RecordDiffusionCanvasEmbed(cmdBuf, regionP, canvasLen, hiddenSize, eps);
        }

        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];
            ref readonly var cpuLw = ref _cpuWeights.Layers[layer];

            // Pre-attention residual snapshot: Residual aliases HiddenState
            // (same physical buffer), so no copy is needed. The barrier from
            // the previous layer's final residual add (or the embedding
            // gather's TRANSFER→COMPUTE on layer 0) has already made the
            // hidden-state writes visible to this rmsnorm.

            if (lw.Mla is { } mlaW)
            {
                // MLA (DeepSeek-V2/V3) attention block — projection ladder +
                // decoupled RoPE + per-head SDPA. Writes the post-o_proj
                // result into _state.NormOutput (mirrors the GQA path's
                // contract so the shared residual-add code below works
                // unchanged).
                RecordMlaLayer(cmdBuf, layer, mlaW, lw, seqLen, eps,
                    positions, kvCache);
            }
            else if (lw.Gemma4 is not null)
            {
                // Gemma-4 attention (V-from-K, weight-less V-norm, per-layer dual
                // head dim / rope, attn scale 1.0). Writes o_proj into NormOutput;
                // the shared post-attn-norm + residual #1 follow. When a
                // VulkanKvCache is supplied (autoregressive generation) the post-
                // norm/post-RoPE K and weight-less-normed V are appended to the
                // per-layer-strided cache and attention reads the full window;
                // otherwise (diffusion / single-shot) it is cacheless.
                RecordGemma4Attention(cmdBuf, layer, lw, seqLen, eps, positions, kvCache);
            }
            else
            {

            // Attn RMSNorm + Q projection — fused into one dispatch when
            // available (decode + Q8_0 + hidden ≤ shader cap). The fused
            // shader writes BOTH the normalised hidden state (for K/V to
            // read) AND the Q matmul output. Falls back to the standalone
            // pair on prefill, non-Q8_0 weights, or oversized hidden.
            if (TryRecordFusedRmsNormMatmul(cmdBuf,
                    _state.HiddenState, lw.AttnNormWeight,
                    lw.Q, lw.QDeviceQuantType,
                    _state.NormOutput, _state.Q,
                    lw.QOutputDim, lw.QInputDim, seqLen, eps))
            {
                // Fused path wrote NormOutput + Q; K/V follow over the same input.
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                RecordMatmul(cmdBuf, lw.K, lw.KDeviceQuantType, _state.NormOutput, _state.K,
                    lw.KOutputDim, lw.KInputDim, seqLen);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                RecordMatmul(cmdBuf, lw.V, lw.VDeviceQuantType, _state.NormOutput, _state.V,
                    lw.VOutputDim, lw.VInputDim, seqLen);
            }
            else
            {
                _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.AttnNormWeight, _state.NormOutput,
                    rowCount: seqLen, n: hiddenSize, eps: eps);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);

                // Q/K/V all read the post-attn-norm hidden state. On the MMVQ
                // decode path this shares one Q8_1 activation-quant across the
                // three GEMVs (issue #46 follow-up). Non-qualifying groups fall
                // back to per-projection RecordMatmul with the same barriers.
                _mmvqGroupScratch[0] = new(lw.Q, lw.QDeviceQuantType, _state.Q, lw.QOutputDim);
                _mmvqGroupScratch[1] = new(lw.K, lw.KDeviceQuantType, _state.K, lw.KOutputDim);
                _mmvqGroupScratch[2] = new(lw.V, lw.VDeviceQuantType, _state.V, lw.VOutputDim);
                RecordSharedInputMmvqGroup(cmdBuf, _state.NormOutput, lw.QInputDim, seqLen,
                    _mmvqGroupScratch.AsSpan(0, 3));
            }

            // Optional QKV biases — kernel path keeps the whole forward in
            // one submit. Each bias add writes a different output buffer
            // (Q / K / V are independent), so no inter-bias barrier needed.
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            if (lw.QBias is not null) _biasAdd.Record(cmdBuf, _state.Q, lw.QBias, seqLen, lw.QOutputDim);
            if (lw.KBias is not null) _biasAdd.Record(cmdBuf, _state.K, lw.KBias, seqLen, lw.KOutputDim);
            if (lw.VBias is not null) _biasAdd.Record(cmdBuf, _state.V, lw.VBias, seqLen, lw.VOutputDim);
            if (lw.QBias is not null || lw.KBias is not null || lw.VBias is not null)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // LoRA delta (q/k/v) — applied AFTER bias and BEFORE QK-norm /
            // RoPE so the delta contributes to the same downstream pipeline
            // as the base projection. The fused rmsnorm+matmul path above
            // still writes F32 normOut (its shader's contract), so the LoRA
            // matmul's input is materialised in either branch — no need to
            // bypass the fused path.
            if (_currentLora is not null)
            {
                MaybeApplyLoraDelta(cmdBuf, layer, "q_proj", _state.NormOutput, _state.Q,
                    seqLen, lw.QInputDim, lw.QOutputDim);
                MaybeApplyLoraDelta(cmdBuf, layer, "k_proj", _state.NormOutput, _state.K,
                    seqLen, lw.KInputDim, lw.KOutputDim);
                MaybeApplyLoraDelta(cmdBuf, layer, "v_proj", _state.NormOutput, _state.V,
                    seqLen, lw.VInputDim, lw.VOutputDim);
            }

            // RoPE on Q and K
            _rope.Record(cmdBuf, _state.Q, _state.K, _state.PositionsBuffer,
                seqLen: seqLen, numHeads: numHeads, numKvHeads: numKvHeads,
                headDim: headDim, ropeDim: _ropeDim, theta: _ropeTheta,
                variant: _ropeVariant);

            // Attention input buffers: either the uncached K/V window or the full KV cache.
            VulkanDevice.Buffer kSrc, vSrc;
            int seqKv;
            int positionOffset;
            if (kvCache is VulkanKvCache vkCache)
            {
                // RoPE writes K; attention (via the cache buffers) reads K.
                // Barrier the RoPE → KV copy, then the KV copy → attention.
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                vkCache.RecordUpdate(cmdBuf, _state.K, _state.V, positions, seqLen, layer);
                KernelSupport.TransferToComputeBarrier(cmdBuf);
                kSrc = vkCache.GetKeysBuffer(layer);
                vSrc = vkCache.GetValuesBuffer(layer);
                seqKv = vkCache.CurrentLength;
                positionOffset = positions[0];
            }
            else if (kvCache is VulkanTurboQuantKvCache tqCache)
            {
                // TurboQuant: encode the freshly-projected K/V into compressed codes, then dequant
                // the live range into the shared fp32 scratch the attention kernel reads. RoPE(K) →
                // encode, encode(codes) → dequant, dequant(scratch) → attention are all COMPUTE; the
                // same COMPUTE barriers also order the previous layer's attention reads of the shared
                // scratch before this layer's dequant overwrites it (WAR).
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                tqCache.RecordUpdate(cmdBuf, _state.K, _state.V, positions, seqLen, layer);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                tqCache.RecordDequant(cmdBuf, layer);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                kSrc = tqCache.GetKeysBuffer();
                vSrc = tqCache.GetValuesBuffer();
                seqKv = tqCache.CurrentLength;
                positionOffset = positions[0];
            }
            else
            {
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                kSrc = _state.K;
                vSrc = _state.V;
                seqKv = seqLen;
                positionOffset = 0;
            }

            // Gemma 3 family extras (no-op on every other architecture):
            //  - PerLayerSlidingWindow[layer]: per-layer override for the
            //    interleaved local/global pattern (e.g. layers 0,2 sliding,
            //    1,3 full). Resolved via GetLayerSlidingWindow.
            //  - QueryPreAttnScalar: optional override of the default
            //    1/sqrt(headDim) attention scale, passed via the shader's
            //    scaleOverride push constant.
            //  - AttnLogitSoftcap: pre-softmax tanh soft-cap (Gemma 2 sets
            //    50.0; Gemma 3 leaves it null). Forwarded via the shader's
            //    softCap push constant.
            int layerSlidingWindow = GetLayerSlidingWindow(layer);
            float attnScaleOverride = GetAttentionScaleOverride();
            float attnSoftCap = Config.AttnLogitSoftcap ?? 0.0f;
            RecordAttention(cmdBuf, _state.Q, kSrc, vSrc, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: positionOffset, slidingWindow: layerSlidingWindow,
                softCap: attnSoftCap, scaleOverride: attnScaleOverride);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // BitNet Sub-LN: in-place RMSNorm over the attention output before the
            // output projection. No-op for non-BitNet (AttnSubNormWeight null).
            if (lw.AttnSubNormWeight is { } attnSubNorm)
            {
                _rmsnorm.Record(cmdBuf, _state.AttnOutput, attnSubNorm, _state.AttnOutput,
                    rowCount: seqLen, n: lw.OInputDim, eps: eps);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // Output projection → NormOutput (reuse slot).
            RecordMatmul(cmdBuf, lw.O, lw.ODeviceQuantType, _state.AttnOutput, _state.NormOutput,
                lw.OOutputDim, lw.OInputDim, seqLen);

            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            if (lw.OBias is not null)
            {
                _biasAdd.Record(cmdBuf, _state.NormOutput, lw.OBias, seqLen, lw.OOutputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // LoRA delta (o_proj): y += scale * (attnOut · B) · A.
            if (_currentLora is not null)
            {
                MaybeApplyLoraDelta(cmdBuf, layer, "o_proj", _state.AttnOutput, _state.NormOutput,
                    seqLen, lw.OInputDim, lw.OOutputDim);
            }
            }  // end of GQA branch (else of MLA)

            // Gemma four-norm layout: post-attention RMSNorm applied to the
            // attention sublayer output (NormOutput) BEFORE the residual add.
            // In-place row-wise norm. No-op for non-Gemma (PostAttnNormWeight
            // null) — the standard two-norm residual is byte-identical.
            if (lw.PostAttnNormWeight is { } postAttnNorm1)
            {
                _rmsnorm.Record(cmdBuf, _state.NormOutput, postAttnNorm1, _state.NormOutput,
                    rowCount: seqLen, n: hiddenSize, eps: eps);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // Residual add #1: AddScratch = Residual + NormOutput. The add
            // reads from HiddenState (which aliases Residual — same slot)
            // and writes to AddScratch (the alternate slot). After the
            // rotate, HiddenState = old AddScratch and AddScratch = old
            // HiddenState — no copies, just a label swap. The single
            // ComputeToComputeBarrier covers the shader_write→shader_read
            // ordering the FFN rmsnorm needs to see the new hidden state.
            _add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch, seqLen * hiddenSize);
            _state.RotateHiddenSlot();
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Pre-FFN residual snapshot: Residual aliases HiddenState (same
            // slot); no copy needed.

            if (lw.Moe is { } moeW)
            {
                if (lw.Gemma4 is not null)
                {
                    // Gemma-4 dual parallel dense + MoE FFN. Writes the combined
                    // post_ffw_norm'd result into NormOutput so the shared
                    // residual-add below works unchanged; the per-layer output
                    // scale is applied after that add.
                    RecordGemma4Ffn(cmdBuf, layer, moeW, lw, seqLen, eps);
                }
                else
                {
                    // MoE FFN replaces the dense Gate/Up/Down with a sparse
                    // top-k expert dispatch. Writes the post-MoE result into
                    // _state.NormOutput so the shared residual-add below
                    // works unchanged.
                    RecordMoeLayer(cmdBuf, layer, moeW, lw, seqLen, eps);
                }
            }
            else
            {

            // FFN RMSNorm + Gate projection — fused when available
            // (mirrors the attn-norm + Q fusion above). Up reads the
            // normalised hidden state written by the fused dispatch.
            if (TryRecordFusedRmsNormMatmul(cmdBuf,
                    _state.HiddenState, lw.FfnNormWeight,
                    lw.Gate, lw.GateDeviceQuantType,
                    _state.NormOutput, _state.FfnGate,
                    lw.GateOutputDim, lw.GateInputDim, seqLen, eps))
            {
                // Fused path wrote NormOutput + Gate; Up follows over the same input.
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                RecordMatmul(cmdBuf, lw.Up, lw.UpDeviceQuantType, _state.NormOutput, _state.FfnUp,
                    lw.UpOutputDim, lw.UpInputDim, seqLen);
            }
            else
            {
                _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.FfnNormWeight, _state.NormOutput,
                    rowCount: seqLen, n: hiddenSize, eps: eps);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);

                // Gate/Up both read the post-ffn-norm hidden state. On the MMVQ
                // decode path this shares one Q8_1 activation-quant across the two
                // GEMVs. Non-qualifying groups fall back to per-projection
                // RecordMatmul with the same barriers.
                _mmvqGroupScratch[0] = new(lw.Gate, lw.GateDeviceQuantType, _state.FfnGate, lw.GateOutputDim);
                _mmvqGroupScratch[1] = new(lw.Up, lw.UpDeviceQuantType, _state.FfnUp, lw.UpOutputDim);
                RecordSharedInputMmvqGroup(cmdBuf, _state.NormOutput, lw.GateInputDim, seqLen,
                    _mmvqGroupScratch.AsSpan(0, 2));
            }

            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            if (lw.GateBias is not null) _biasAdd.Record(cmdBuf, _state.FfnGate, lw.GateBias, seqLen, lw.GateOutputDim);
            if (lw.UpBias is not null) _biasAdd.Record(cmdBuf, _state.FfnUp, lw.UpBias, seqLen, lw.UpOutputDim);
            if (lw.GateBias is not null || lw.UpBias is not null)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // LoRA delta (gate/up): y += scale * (normOut · B) · A.
            if (_currentLora is not null)
            {
                MaybeApplyLoraDelta(cmdBuf, layer, "gate_proj", _state.NormOutput, _state.FfnGate,
                    seqLen, lw.GateInputDim, lw.GateOutputDim);
                MaybeApplyLoraDelta(cmdBuf, layer, "up_proj", _state.NormOutput, _state.FfnUp,
                    seqLen, lw.UpInputDim, lw.UpOutputDim);
            }

            // FFN gate activation: GeGLU-tanh (Gemma) when _geglu is non-null,
            // gated squared-ReLU (BitNet) when _relu2glu is non-null, otherwise
            // the standard SwiGLU. All fuse gate*act + up into SiluOutput; only
            // the gate non-linearity differs.
            if (_geglu is not null)
                _geglu.Record(cmdBuf, _state.FfnGate, _state.FfnUp, _state.SiluOutput, seqLen * intermediateSize);
            else if (_relu2glu is not null)
                _relu2glu.Record(cmdBuf, _state.FfnGate, _state.FfnUp, _state.SiluOutput, seqLen * intermediateSize);
            else
                _swiglu.Record(cmdBuf, _state.FfnGate, _state.FfnUp, _state.SiluOutput, seqLen * intermediateSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // BitNet Sub-LN: in-place RMSNorm over the gated FFN intermediate before
            // the down projection. No-op for non-BitNet (FfnSubNormWeight null).
            if (lw.FfnSubNormWeight is { } ffnSubNorm)
            {
                _rmsnorm.Record(cmdBuf, _state.SiluOutput, ffnSubNorm, _state.SiluOutput,
                    rowCount: seqLen, n: lw.DownInputDim, eps: eps);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // Down projection
            RecordMatmul(cmdBuf, lw.Down, lw.DownDeviceQuantType, _state.SiluOutput, _state.NormOutput,
                lw.DownOutputDim, lw.DownInputDim, seqLen);

            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            if (lw.DownBias is not null)
            {
                _biasAdd.Record(cmdBuf, _state.NormOutput, lw.DownBias, seqLen, lw.DownOutputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // LoRA delta (down_proj): y += scale * (siluOut · B) · A.
            // Input is post-SwiGLU (siluOut), not normOut. The base GEMM
            // already wrote into normOut, so we accumulate delta in place.
            if (_currentLora is not null)
            {
                MaybeApplyLoraDelta(cmdBuf, layer, "down_proj", _state.SiluOutput, _state.NormOutput,
                    seqLen, lw.DownInputDim, lw.DownOutputDim);
            }
            }  // end of dense-FFN branch (else of MoE)

            // Gemma four-norm layout: post-FFN RMSNorm applied to the FFN
            // sublayer output (NormOutput) BEFORE the residual add. In-place
            // row-wise norm. No-op for non-Gemma (PostFfnNormWeight null).
            if (lw.PostFfnNormWeight is { } postFfnNorm1)
            {
                _rmsnorm.Record(cmdBuf, _state.NormOutput, postFfnNorm1, _state.NormOutput,
                    rowCount: seqLen, n: hiddenSize, eps: eps);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // Residual add #2: AddScratch = Residual + NormOutput; then rotate
            // the slot so the new hidden state lives in the buffer we just
            // wrote. See residual add #1 comment above for why no copy is
            // needed.
            _add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch, seqLen * hiddenSize);
            _state.RotateHiddenSlot();

            // Gemma-4 per-layer output scale (layer_output_scale) — the LAST
            // per-layer op, an in-place scalar multiply on the post-residual
            // hidden state. Reuses the embedding-scale kernel (a generic scalar
            // multiply). No-op on every other architecture (lw.Gemma4 null).
            if (lw.Gemma4 is { } g4scale)
            {
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                // AR gemma4: every row uses layer_output_scale. DiffusionGemma: the
                // canvas rows [P, seqLen) use layer_output_scale, the PROMPT rows
                // [0, P) use enc_layer_output_scale (same backbone weights — the
                // encoder contributes ONLY the scalar). layer_output_scale is applied
                // to ALL rows first (the canvas-correct value); then the contiguous
                // HEAD prompt rows [0, P) are corrected by the enc/layer ratio. Prompt
                // rows start at offset 0, so the whole-buffer kernel (which processes
                // P*hidden elements from the start) hits exactly them — no offset binding.
                _embedScale!.Record(cmdBuf, _state.HiddenState, seqLen * hiddenSize, g4scale.LayerOutputScale);
                int regionP = DiffusionRegionPrefix(seqLen);
                float? encScale = _cpuWeights.Layers[layer].Gemma4?.EncLayerOutputScale;
                if (regionP > 0 && encScale is float enc && g4scale.LayerOutputScale != 0f)
                {
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    float ratio = enc / g4scale.LayerOutputScale;
                    _embedScale!.Record(cmdBuf, _state.HiddenState, regionP * hiddenSize, ratio);
                }
            }

            // COMPUTE→COMPUTE between layers — next iteration's first op is
            // the attention RMSNorm, which reads the freshly-rotated
            // HiddenState written by the add.
            if (layer < Config.NumLayers - 1)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }

        // 3a. PKV prefill: the captured per-layer K/V is all we need — skip the final
        //     norm + LM head entirely (the generator discards the prefill result).
        if (_pkvPhase == DiffusionKvPhase.Prefill)
        {
            _submit.SubmitAndWait();
            return UnmanagedTensor.Allocate(new TensorShape(1, vocabSize), DType.Float32, deviceId: -1);
        }

        // 3b. DiffusionGemma: final RMSNorm + LM head over ALL positions (the
        //     diffusion generator gathers logit rows for the masked canvas
        //     positions, not just the last token). Returns [seqLen, vocab].
        if (_diffusionMaskMode != AttentionMaskMode.Causal && Config.DiffusionConfig is not null)
            return FinishDiffusionForward(cmdBuf, seqLen, hiddenSize, vocabSize, eps, deviceId);

        // 3. Final RMSNorm on the last token only, then LM head.
        //    The last hidden state was just written by the final layer's
        //    residual add (compute shader). The following single-row copy
        //    runs in TRANSFER, so we need a compute→transfer barrier — a
        //    plain ComputeToComputeBarrier does NOT synchronise transfer
        //    reads against prior compute writes.
        long rowBytes = (long)hiddenSize * sizeof(float);
        long lastRowOffset = (long)(seqLen - 1) * rowBytes;
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.NormOutput,
            srcOffset: (ulong)lastRowOffset, dstOffset: 0, size: (ulong)rowBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        _rmsnorm.Record(cmdBuf, _state.NormOutput, _weights.OutputNormWeight, _state.NormOutput,
            rowCount: 1, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        RecordMatmul(cmdBuf, _weights.OutputWeight, _weights.OutputDeviceQuantType,
            _state.NormOutput, _state.Logits,
            _weights.OutputOutputDim, _weights.OutputInputDim, seqLen: 1);

        // 4. COMPUTE→HOST barrier for the vocab-row download that follows, submit, wait.
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        // 5. Return logits as a host-resident UnmanagedTensor [1, vocabSize].
        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        unsafe
        {
            var dest = new Span<float>((void*)result.DataPointer, vocabSize);
            _device.Download(_state.Logits, dest);
            // Gemma 2/3 final-logit soft-cap: z' = cap * tanh(z / cap). The
            // lm_head download already brings logits to host, so this is the
            // cheaper option vs an extra device-side dispatch + readback.
            // Mirrors the CPU TransformerModel.ApplyFinalLogitSoftcap; no-op
            // when Config.FinalLogitSoftcap is null or non-positive.
            ApplyFinalLogitSoftcapHost(dest);
        }
        return result;
    }

    /// <summary>
    /// DiffusionGemma forward tail: final RMSNorm + LM head over ALL <paramref name="seqLen"/>
    /// positions (vs the AR path's last-token-only head), returning a host-resident
    /// <c>[seqLen, vocab]</c> logits tensor (per-row final soft-cap applied). Used only by the
    /// diffusion forward — the canvas generator gathers the masked-position rows.
    /// </summary>
    private ITensor FinishDiffusionForward(nint cmdBuf, int seqLen, int hiddenSize, int vocabSize, float eps, int deviceId)
    {
        // Final RMSNorm over every row, in place on HiddenState → NormOutput.
        _rmsnorm.Record(cmdBuf, _state.HiddenState, _weights.OutputNormWeight, _state.NormOutput,
            rowCount: seqLen, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        VulkanDevice.Buffer logitsBuf = EnsureDiffusionLogits(seqLen, vocabSize);
        RecordMatmul(cmdBuf, _weights.OutputWeight, _weights.OutputDeviceQuantType,
            _state.NormOutput, logitsBuf,
            _weights.OutputOutputDim, _weights.OutputInputDim, seqLen: seqLen);

        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        var shape = new TensorShape(seqLen, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        unsafe
        {
            var dest = new Span<float>((void*)result.DataPointer, seqLen * vocabSize);
            _device.Download(logitsBuf, dest);
            // Per-row Gemma final-logit soft-cap (no-op when FinalLogitSoftcap is null).
            if (Config.FinalLogitSoftcap is float cap && cap > 0f)
                for (int r = 0; r < seqLen; r++)
                    ApplyFinalLogitSoftcapHost(dest.Slice(r * vocabSize, vocabSize));
        }
        return result;
    }

    /// <summary>Lazily (re)allocates the all-position diffusion logits buffer for <paramref name="rows"/> × <paramref name="vocab"/>.</summary>
    private VulkanDevice.Buffer EnsureDiffusionLogits(int rows, int vocab)
    {
        if (_diffusionLogits is null || _diffusionLogitsCapacityRows < rows)
        {
            _diffusionLogits?.Dispose();
            _diffusionLogits = _device.Allocate((long)rows * vocab * sizeof(float));
            _diffusionLogitsCapacityRows = rows;
            // The new handle invalidates any cached lm-head matmul descriptor set.
            InvalidateKernelCaches();
        }
        return _diffusionLogits;
    }

    /// <summary>
    /// Downloads the post-transformer hidden state (all <paramref name="seqLen"/> rows)
    /// from device-local memory to a freshly-allocated host CPU tensor. Must be called
    /// immediately after <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?)"/>
    /// while the device buffers are still valid.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Used by the M1 Vulkan/CUDA hybrid model to extract the hidden state after the
    /// Vulkan-resident layers complete and pass it to the CUDA backend for the
    /// remaining layers. The <c>Forward</c> call on a model loaded with
    /// <c>numVulkanLayers</c> (via <c>config with {{ NumLayers = numVulkanLayers }}</c>)
    /// runs only those layers; the returned logits from <c>Forward</c> are discarded.
    /// </para>
    /// <para>
    /// Two synchronous operations: <c>vkCmdCopyBuffer</c> DEVICE_LOCAL → HOST_VISIBLE
    /// staging (one fence wait), then a host-side <c>MemoryCopy</c>. Both are off
    /// the per-token hot path for the M1 proof-of-concept; async overlap is M3 scope.
    /// </para>
    /// </remarks>
    /// <param name="seqLen">Number of tokens whose hidden states to download. Must match the last Forward call's seqLen.</param>
    /// <returns>
    /// A newly allocated CPU tensor [<paramref name="seqLen"/>, hiddenSize], FP32, deviceId=-1.
    /// Caller owns disposal.
    /// </returns>
    public unsafe ITensor DownloadHiddenState(int seqLen)
    {
        if (seqLen <= 0)
            throw new ArgumentOutOfRangeException(nameof(seqLen));

        int hiddenSize = Config.HiddenSize;
        long bytes = (long)seqLen * hiddenSize * sizeof(float);
        var shape = new TensorShape(seqLen, hiddenSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);

        // Copy DEVICE_LOCAL → HOST_VISIBLE staging via a separate synchronous command buffer.
        // The caller's Forward call already did SubmitAndWait so the compute writes are visible.
        using var staging = _device.Allocate(bytes);
        _device.CopyBufferRangeSynchronous(_state.HiddenState, staging,
            srcOffset: 0, dstOffset: 0, size: (ulong)bytes);
        _device.Download(staging, new Span<float>((void*)result.DataPointer, seqLen * hiddenSize));

        return result;
    }

    /// <summary>
    /// Host-side Gemma 2/3 final-logit soft-cap: <c>z' = cap * tanh(z / cap)</c>
    /// in-place. Uses <see cref="TensorPrimitives.Tanh"/> for the SIMD path.
    /// No-op when <see cref="ModelConfig.FinalLogitSoftcap"/> is null or
    /// non-positive. Mirrors <c>TransformerModel.ApplyFinalLogitSoftcap</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ApplyFinalLogitSoftcapHost(Span<float> logits)
    {
        if (Config.FinalLogitSoftcap is not float cap || cap <= 0.0f) return;
        float inv = 1.0f / cap;
        TensorPrimitives.Multiply(logits, inv, logits);
        TensorPrimitives.Tanh(logits, logits);
        TensorPrimitives.Multiply(logits, cap, logits);
    }

    private void InvalidateKernelCaches()
    {
        _matmul.InvalidateDescriptorCache();
        _matmulQ8.InvalidateDescriptorCache();
        _matmulQ8Gemm.InvalidateDescriptorCache();
        _matmulQ8GemmCoopmat?.InvalidateDescriptorCache();
        _matmulQ2K.InvalidateDescriptorCache();
        _matmulQ2KGemm.InvalidateDescriptorCache();
        _matmulQ3K.InvalidateDescriptorCache();
        _matmulQ3KGemm.InvalidateDescriptorCache();
        _matmulQ4K.InvalidateDescriptorCache();
        _matmulQ4KGemm.InvalidateDescriptorCache();
        _matmulQ5K.InvalidateDescriptorCache();
        _matmulQ5KGemm.InvalidateDescriptorCache();
        _matmulQ6K.InvalidateDescriptorCache();
        _matmulQ6KGemm.InvalidateDescriptorCache();
        _matmulIq4Nl.InvalidateDescriptorCache();
        _matmulIq4NlGemm.InvalidateDescriptorCache();
        _matmulIq4Xs.InvalidateDescriptorCache();
        _matmulIq4XsGemm.InvalidateDescriptorCache();
        _matmulIq2Xxs.InvalidateDescriptorCache();
        _matmulIq2XxsGemm.InvalidateDescriptorCache();
        _matmulIq2Xs.InvalidateDescriptorCache();
        _matmulIq2XsGemm.InvalidateDescriptorCache();
        _matmulIq2S.InvalidateDescriptorCache();
        _matmulIq2SGemm.InvalidateDescriptorCache();
        _matmulIq3Xxs.InvalidateDescriptorCache();
        _matmulIq3XxsGemm.InvalidateDescriptorCache();
        _matmulIq3S.InvalidateDescriptorCache();
        _matmulIq3SGemm.InvalidateDescriptorCache();
        _matmulIq1S.InvalidateDescriptorCache();
        _matmulIq1SGemm.InvalidateDescriptorCache();
        _matmulI2S.InvalidateDescriptorCache();
        _matmulI2SGemm.InvalidateDescriptorCache();
        _matmulF16.InvalidateDescriptorCache();
        _matmulF16Gemm.InvalidateDescriptorCache();
        _matmulF16GemmCoopmat?.InvalidateDescriptorCache();
        _matmulBf16.InvalidateDescriptorCache();
        _matmulBf16Gemm.InvalidateDescriptorCache();
        _rmsnormMatmulQ8Fused?.InvalidateDescriptorCache();
        _quantizeQ8_1?.InvalidateDescriptorCache();
        _matmulQ8Mmvq?.InvalidateDescriptorCache();
        _matmulQ4KMmvq?.InvalidateDescriptorCache();
        _matmulQ6KMmvq?.InvalidateDescriptorCache();
        _matmulQ5KMmvq?.InvalidateDescriptorCache();
        _matmulQ2KMmvq?.InvalidateDescriptorCache();
        _matmulQ3KMmvq?.InvalidateDescriptorCache();
        _matmulIq4NlMmvq?.InvalidateDescriptorCache();
        _matmulIq4XsMmvq?.InvalidateDescriptorCache();
        _matmulIq2XxsMmvq?.InvalidateDescriptorCache();
        _matmulIq2XsMmvq?.InvalidateDescriptorCache();
        _matmulIq2SMmvq?.InvalidateDescriptorCache();
        _matmulIq3XxsMmvq?.InvalidateDescriptorCache();
        _matmulIq3SMmvq?.InvalidateDescriptorCache();
        _matmulIq1SMmvq?.InvalidateDescriptorCache();
        _quantizeQ8_1Rows?.InvalidateDescriptorCache();
        _matmulQ8Mmq?.InvalidateDescriptorCache();
        _matmulQ4KMmq?.InvalidateDescriptorCache();
        _matmulQ6KMmq?.InvalidateDescriptorCache();
        _matmulQ5KMmq?.InvalidateDescriptorCache();
        _matmulIq4XsMmq?.InvalidateDescriptorCache();
        _matmulIq4NlMmq?.InvalidateDescriptorCache();
        _matmulQ2KMmq?.InvalidateDescriptorCache();
        _matmulQ3KMmq?.InvalidateDescriptorCache();
        _rmsnorm.InvalidateDescriptorCache();
        _rope.InvalidateDescriptorCache();
        _attention.InvalidateDescriptorCache();
        _flashAttention?.InvalidateDescriptorCache();
        _swiglu.InvalidateDescriptorCache();
        _geglu?.InvalidateDescriptorCache();
        _relu2glu?.InvalidateDescriptorCache();
        _embedScale?.InvalidateDescriptorCache();
        _add.InvalidateDescriptorCache();
        _biasAdd.InvalidateDescriptorCache();
        _mlaAttention?.InvalidateDescriptorCache();
        _mlaRope?.InvalidateDescriptorCache();
        _mlaKvSplit?.InvalidateDescriptorCache();
        _moeTopkSoftmax?.InvalidateDescriptorCache();
        _moeIndexedMatmul?.InvalidateDescriptorCache();
        _moeIndexedMatmulQ8?.InvalidateDescriptorCache();
        _moeIndexedMatmulQ4K?.InvalidateDescriptorCache();
        _moeIndexedMatmulQ5_1?.InvalidateDescriptorCache();
        _moeIndexedMatmulTiled?.InvalidateDescriptorCache();
        _moeExpertOffsets?.InvalidateDescriptorCache();
        _moeExpandGroupByExpert?.InvalidateDescriptorCache();
        _moeGroupedMatmulF16Coopmat?.InvalidateDescriptorCache();
        _moeUngroupScatter?.InvalidateDescriptorCache();
        _moeWeightedScatter?.InvalidateDescriptorCache();
        _moeBroadcast?.InvalidateDescriptorCache();
        _moeIndexedLoraDelta?.InvalidateDescriptorCache();
        _moeSigmoidGatedAdd?.InvalidateDescriptorCache();
        _loraDeltaGemvFused?.InvalidateDescriptorCache();
    }

    /// <summary>
    /// Dispatches a matmul for a single linear projection: chooses
    /// <see cref="MatMulQ8_0Kernel"/> (decode-path GEMV) when the device-side
    /// weight is Q8_0 and <paramref name="seqLen"/>==1, the batched
    /// <see cref="MatMulQ8_0GemmKernel"/> when Q8_0 and <paramref name="seqLen"/>&gt;1,
    /// and <see cref="MatMulF32Kernel"/> for every non-Q8_0 weight.
    /// </summary>
    /// <remarks>
    /// All Q8_0 kernels require <paramref name="inputDim"/> to be a multiple
    /// of 32 (the Q8_0 group size). Llama-family projections satisfy this by
    /// construction; the Q8_0 kernels still validate at dispatch so a
    /// surprise non-aligned model fails loud.
    /// </remarks>
    /// <summary>
    /// Attempts to dispatch a fused (rmsnorm → Q8_0 matmul) pair as a single
    /// dispatch with one barrier instead of two. Returns false when the fast
    /// path is unavailable (no fused SPV, non-Q8_0 weight, prefill, or hidden
    /// size beyond the shader's on-chip cap) — the caller must record the
    /// standalone (rmsnorm + matmul) pair as a fallback.
    /// </summary>
    /// <remarks>
    /// On success the fused dispatch:
    ///   1. Computes rmsnorm of <paramref name="hidden"/> with
    ///      <paramref name="normWeight"/>, writing the normalised values to
    ///      <paramref name="normOutput"/> (so downstream non-fused matmuls
    ///      like K, V, Up still see the normalised hidden state).
    ///   2. Computes <c>matmulOutput[m] = sum_k weight[m,k] * normalised[k]</c>
    ///      using on-chip shared memory for the dot product.
    /// Caller is responsible for the post-dispatch barrier — same shape as a
    /// standalone matmul.
    /// </remarks>
    private bool TryRecordFusedRmsNormMatmul(
        nint cmdBuf,
        VulkanDevice.Buffer hidden, VulkanDevice.Buffer normWeight,
        VulkanDevice.Buffer weights, QuantType weightQt,
        VulkanDevice.Buffer normOutput, VulkanDevice.Buffer matmulOutput,
        int outputDim, int inputDim, int seqLen, float eps)
    {
        if (_rmsnormMatmulQ8Fused is null) return false;
        // When the dp4a MMVQ decode path is wired, prefer it over the fused
        // rmsnorm+Q8_0 GEMV: MMVQ's integer-dot inner loop is the optimized
        // decode path we want on the hot projections (issue #46). Returning
        // false here routes the (separate rmsnorm + RecordMatmul) pair, and
        // RecordMatmul picks MMVQ for the Q8_0 GEMV. The fused F32-in GEMV only
        // ever fires for models below its 1024-hidden cap; deferring keeps the
        // decode quant path consistent across all Q8_0 projections.
        if (seqLen == 1 && weightQt == QuantType.Q8_0
            && _matmulQ8Mmvq is not null && _quantizeQ8_1 is not null)
            return false;
        // Opt-out switch — default is fused-on. On RDNA3.5 the fused path
        // wins by ~3-5% in median paired-run min latency and is more
        // resilient to dispatch-time contention. Set the env var to "1"
        // to bypass on hardware where fusion regresses (vendor A/B).
        if (Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_FUSED_RMSNORM_MATMUL") == "1") return false;
        if (seqLen != 1) return false;
        if (weightQt != QuantType.Q8_0) return false;
        if (!RmsNormMatmulQ8_0FusedKernel.SupportsHiddenSize(inputDim)) return false;

        _rmsnormMatmulQ8Fused.Record(cmdBuf, hidden, normWeight, weights,
            normOutput, matmulOutput,
            m: outputDim, k: inputDim, eps: eps);
        return true;
    }

    /// <summary>
    /// Records the MLA (DeepSeek-V2/V3) attention block for one layer:
    /// rmsnorm → Q path (LoRA-factored or monolithic) → KV path (latent
    /// rmsnorm + kv_b expansion + per-head split) → decoupled-rope on
    /// Q_pe + shared K_pe → optional KV-cache write → per-head SDPA →
    /// o_proj. Writes the post-o_proj result into <c>_state.NormOutput</c>
    /// so the shared residual-add downstream sees the same contract as the
    /// GQA path.
    /// </summary>
    /// <remarks>
    /// All MLA projections are F32 (no Q8_0 path on MLA today; the loader
    /// upcasts F16/BF16 at load). The matmul router still uses
    /// <see cref="RecordMatmul"/> and lands on <c>matmul_f32</c> uniformly.
    /// </remarks>
    private void RecordMlaLayer(
        nint cmdBuf, int layer, VulkanWeights.MlaLayerBuffers mlaW,
        in VulkanWeights.LayerBuffers lw, int seqLen, float eps,
        ReadOnlySpan<int> positions, IKvCache? kvCache)
    {
        int qkHeadDim = mlaW.QkHeadDim;
        int hidden = mlaW.HiddenSize;

        // Pre-attention RMSNorm: HiddenState → NormOutput.
        _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.AttnNormWeight, _state.NormOutput,
            rowCount: seqLen, n: hidden, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── Q path ────────────────────────────────────────────────────
        // LoRA: NormOutput → MlaQLatent → (rmsnorm with QALayernormWeight)
        //       → MlaQLatentNorm → MlaQ.
        // Monolithic: NormOutput → MlaQ.
        if (mlaW.QLoraRank > 0)
        {
            _matmul.Record(cmdBuf, mlaW.QAProj!, _state.NormOutput, _state.MlaQLatent!,
                m: mlaW.QLoraRank, k: hidden, n: seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            MaybeApplyLoraDelta(cmdBuf, layer, "q_a_proj", _state.NormOutput, _state.MlaQLatent!,
                seqLen, hidden, mlaW.QLoraRank);
            _rmsnorm.Record(cmdBuf, _state.MlaQLatent!, mlaW.QALayernormWeight!, _state.MlaQLatentNorm!,
                rowCount: seqLen, n: mlaW.QLoraRank, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            _matmul.Record(cmdBuf, mlaW.QBProj!, _state.MlaQLatentNorm!, _state.MlaQ!,
                m: mlaW.QTotal, k: mlaW.QLoraRank, n: seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            MaybeApplyLoraDelta(cmdBuf, layer, "q_b_proj", _state.MlaQLatentNorm!, _state.MlaQ!,
                seqLen, mlaW.QLoraRank, mlaW.QTotal);
        }
        else
        {
            _matmul.Record(cmdBuf, mlaW.QProj!, _state.NormOutput, _state.MlaQ!,
                m: mlaW.QTotal, k: hidden, n: seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            MaybeApplyLoraDelta(cmdBuf, layer, "q_proj", _state.NormOutput, _state.MlaQ!,
                seqLen, hidden, mlaW.QTotal);
        }

        // ── KV path (latent + rope-K split) ──────────────────────────
        // Two parallel matmuls off the same NormOutput: first kvLoraRank
        // rows of kv_a_proj_with_mqa → MlaKvLatent (independent), last
        // qkRopeHeadDim rows → MlaKPe (independent — also independent from
        // the Q matmul above). All three writes complete before the
        // single barrier below.
        _matmul.Record(cmdBuf, mlaW.KvALatentProj, _state.NormOutput, _state.MlaKvLatent!,
            m: mlaW.KvLoraRank, k: hidden, n: seqLen);
        _matmul.Record(cmdBuf, mlaW.KvAKPeProj, _state.NormOutput, _state.MlaKPe!,
            m: mlaW.QkRopeHeadDim, k: hidden, n: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // RMSNorm the latent slice (rope-K is left untouched).
        _rmsnorm.Record(cmdBuf, _state.MlaKvLatent!, mlaW.KvALayernormWeight, _state.MlaKvLatentNorm!,
            rowCount: seqLen, n: mlaW.KvLoraRank, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // kv_b expansion: latent_norm → MlaKvBExpanded
        // Then split per-head into MlaKNope and MlaV.
        _matmul.Record(cmdBuf, mlaW.KvBProj, _state.MlaKvLatentNorm!, _state.MlaKvBExpanded!,
            m: mlaW.KvBOutputDim, k: mlaW.KvLoraRank, n: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        MaybeApplyLoraDelta(cmdBuf, layer, "kv_b_proj", _state.MlaKvLatentNorm!, _state.MlaKvBExpanded!,
            seqLen, mlaW.KvLoraRank, mlaW.KvBOutputDim);

        _mlaKvSplit!.Record(cmdBuf, _state.MlaKvBExpanded!, _state.MlaKNope!, _state.MlaV!,
            seqLen: seqLen, numHeads: mlaW.NumHeads,
            qkNopeHeadDim: mlaW.QkNopeHeadDim, vHeadDim: mlaW.VHeadDim);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── Decoupled RoPE on Q_pe (per head) and shared K_pe ────────
        _mlaRope!.Record(cmdBuf, _state.MlaQ!, _state.MlaKPe!, _state.PositionsBuffer,
            seqLen: seqLen, numHeads: mlaW.NumHeads,
            qkNopeHeadDim: mlaW.QkNopeHeadDim, qkRopeHeadDim: mlaW.QkRopeHeadDim,
            theta: _mlaRopeTheta);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── KV-cache update + attention ──────────────────────────────
        VulkanDevice.Buffer kNopeSrc, vSrc, kPeSrc;
        int seqKv;
        int positionOffset;
        if (kvCache is MlaVulkanKvCache mlaCache)
        {
            // Cache the new K_nope / V / K_pe rows; attention then reads
            // the full cached window.
            mlaCache.RecordUpdate(cmdBuf, _state.MlaKNope!, _state.MlaV!, _state.MlaKPe!,
                positions, seqLen, layer);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
            kNopeSrc = mlaCache.GetKNopeBuffer(layer);
            vSrc = mlaCache.GetVBuffer(layer);
            kPeSrc = mlaCache.GetKPeBuffer(layer);
            seqKv = mlaCache.CurrentLength;
            positionOffset = positions[0];
        }
        else
        {
            kNopeSrc = _state.MlaKNope!;
            vSrc = _state.MlaV!;
            kPeSrc = _state.MlaKPe!;
            seqKv = seqLen;
            positionOffset = 0;
        }

        _mlaAttention!.Record(cmdBuf, _state.MlaQ!, kNopeSrc, vSrc, kPeSrc, _state.MlaAttnOutput!,
            seqQ: seqLen, seqKv: seqKv, numHeads: mlaW.NumHeads,
            qkNopeHeadDim: mlaW.QkNopeHeadDim, qkRopeHeadDim: mlaW.QkRopeHeadDim,
            vHeadDim: mlaW.VHeadDim,
            positionOffset: positionOffset, scale: _mlaScale);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── o_proj → NormOutput (mirrors GQA contract for residual add) ─
        RecordMatmul(cmdBuf, lw.O, lw.ODeviceQuantType, _state.MlaAttnOutput!, _state.NormOutput,
            lw.OOutputDim, lw.OInputDim, seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        if (lw.OBias is not null)
        {
            _biasAdd.Record(cmdBuf, _state.NormOutput, lw.OBias, seqLen, lw.OOutputDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
        MaybeApplyLoraDelta(cmdBuf, layer, "o_proj", _state.MlaAttnOutput!, _state.NormOutput,
            seqLen, lw.OInputDim, lw.OOutputDim);
    }

    // ─────────────────────────────────────────────────────────────────────
    // DIAGNOSTIC HARNESS (bug-#2 bisection). Run ONE Gemma-4 layer on the GPU
    // against a host-supplied residual stream so a working CPU forward can swap
    // individual layers to Vulkan one at a time and re-test ("still Paris?").
    // NOT a production path (one submit per call). RecordGemma4LayerBody mirrors
    // the Forward loop's gemma4 path verbatim so the swap is numerically faithful.
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Diagnostic: compute one Gemma-4 layer on the GPU given a host residual
    /// stream <paramref name="hidden"/> (<c>[seqLen × hiddenSize]</c>, row-major
    /// F32 — the same layout as the CPU forward), overwriting it in place with
    /// the post-layer residual stream. Cacheless / AR. For the hybrid CPU↔Vulkan
    /// bisection harness only; do not use on the hot path.
    /// </summary>
    public unsafe void RunGemma4LayerOnHost(Span<float> hidden, int layer, int seqLen, ReadOnlySpan<int> positions)
    {
        int hiddenSize = Config.HiddenSize;
        float eps = Config.NormEpsilon;
        if (hidden.Length != (long)seqLen * hiddenSize)
            throw new ArgumentException(
                $"hidden length {hidden.Length} != seqLen*hidden {(long)seqLen * hiddenSize}.", nameof(hidden));
        if ((uint)layer >= (uint)Config.NumLayers) throw new ArgumentOutOfRangeException(nameof(layer));
        if (_weights.Layers[layer].Gemma4 is null)
            throw new InvalidOperationException($"Layer {layer} is not a Gemma-4 layer.");

        bool resized = _state.EnsureCapacity(seqLen);
        if (resized) InvalidateKernelCaches();
        UploadPositions(positions);

        // HiddenState is device-local (not host-mappable on this UMA path), so
        // bridge host↔device through a host-visible staging buffer + synchronous
        // copy. The buffer is tiny (seqLen*hidden*4) and diagnostic-only.
        long bytes = (long)hidden.Length * sizeof(float);
        using var staging = _device.Allocate(bytes);

        // Canonicalise the hidden slot (label op), then stage the host residual
        // stream into HiddenState before recording begins.
        _state.ResetHiddenSlot();
        _device.Upload(hidden, staging);
        _device.CopyBufferSynchronous(staging, _state.HiddenState, (ulong)bytes);

        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        RecordGemma4LayerBody(cmdBuf, layer, seqLen, eps, positions);
        _submit.SubmitAndWait();

        _device.CopyBufferSynchronous(_state.HiddenState, staging, (ulong)bytes);
        _device.Download(staging, hidden);
    }

    /// <summary>
    /// Records exactly the per-layer op sequence the Forward loop runs for a
    /// Gemma-4 layer (attention → post-attn-norm → residual → dual FFN →
    /// residual → layer_output_scale), so the diagnostic single-layer path stays
    /// numerically faithful to the full forward. AR / cacheless only (regionP 0).
    /// </summary>
    private void RecordGemma4LayerBody(nint cmdBuf, int layer, int seqLen, float eps, ReadOnlySpan<int> positions)
    {
        ref readonly var lw = ref _weights.Layers[layer];
        int hiddenSize = Config.HiddenSize;

        // Attention → o_proj into NormOutput.
        RecordGemma4Attention(cmdBuf, layer, lw, seqLen, eps, positions, kvCache: null);

        // Post-attention RMSNorm (Gemma four-norm) on NormOutput, then residual #1.
        if (lw.PostAttnNormWeight is { } postAttnNorm1)
        {
            _rmsnorm.Record(cmdBuf, _state.NormOutput, postAttnNorm1, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
        _add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch, seqLen * hiddenSize);
        _state.RotateHiddenSlot();
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Dual parallel dense + MoE FFN → combined post_ffw_norm'd result in NormOutput.
        var moeW = lw.Moe!.Value;
        RecordGemma4Ffn(cmdBuf, layer, moeW, lw, seqLen, eps);

        // Post-FFN RMSNorm is null for gemma4 (its post_ffw_norm lives inside the
        // FFN); mirror the loop's guarded check for fidelity anyway.
        if (lw.PostFfnNormWeight is { } postFfnNorm1)
        {
            _rmsnorm.Record(cmdBuf, _state.NormOutput, postFfnNorm1, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
        _add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch, seqLen * hiddenSize);
        _state.RotateHiddenSlot();

        // layer_output_scale (AR: all rows; regionP 0 so no enc correction).
        if (lw.Gemma4 is { } g4scale)
        {
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            _embedScale!.Record(cmdBuf, _state.HiddenState, seqLen * hiddenSize, g4scale.LayerOutputScale);
        }
    }

    /// <summary>
    /// Diagnostic: run the EMBEDDING (gather + Gemma sqrt(hidden) scale) on the
    /// GPU for <paramref name="tokenIds"/>, returning the host residual stream
    /// <c>[seqLen × hiddenSize]</c>. Mirrors the prologue of Forward. For the
    /// hybrid bisection harness (embedding-vs-pipeline localisation) only.
    /// </summary>
    public unsafe float[] RunGemma4EmbeddingOnHost(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions)
    {
        int seqLen = tokenIds.Length;
        int hiddenSize = Config.HiddenSize;
        bool resized = _state.EnsureCapacity(seqLen);
        if (resized) InvalidateKernelCaches();
        ValidateTokenIds(tokenIds);
        UploadPositions(positions);

        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        _state.ResetHiddenSlot();
        RecordEmbeddingGather(cmdBuf, tokenIds);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        if (_embedScale is not null)
        {
            _embedScale.Record(cmdBuf, _state.HiddenState, seqLen * hiddenSize, Config.EmbeddingScale!.Value);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
        _submit.SubmitAndWait();

        long bytes = (long)seqLen * hiddenSize * sizeof(float);
        using var staging = _device.Allocate(bytes);
        _device.CopyBufferSynchronous(_state.HiddenState, staging, (ulong)bytes);
        var result = new float[seqLen * hiddenSize];
        _device.Download(staging, result);
        return result;
    }

    /// <summary>
    /// Diagnostic: run the FINAL head (last-row copy → output RMSNorm → lm_head →
    /// final-logit soft-cap) on the GPU given a host residual stream
    /// <paramref name="hidden"/> (<c>[seqLen × hiddenSize]</c>), returning the
    /// host last-token logits <c>[vocabSize]</c>. Mirrors the tail of Forward.
    /// For the hybrid bisection harness (embedding-vs-head localisation) only.
    /// </summary>
    public unsafe float[] RunGemma4FinalHeadOnHost(ReadOnlySpan<float> hidden, int seqLen)
    {
        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;
        if (hidden.Length != (long)seqLen * hiddenSize)
            throw new ArgumentException($"hidden length {hidden.Length} != seqLen*hidden {(long)seqLen * hiddenSize}.", nameof(hidden));

        bool resized = _state.EnsureCapacity(seqLen);
        if (resized) InvalidateKernelCaches();

        long bytes = (long)hidden.Length * sizeof(float);
        using var staging = _device.Allocate(bytes);
        _state.ResetHiddenSlot();
        _device.Upload(hidden, staging);
        _device.CopyBufferSynchronous(staging, _state.HiddenState, (ulong)bytes);

        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);

        long rowBytes = (long)hiddenSize * sizeof(float);
        long lastRowOffset = (long)(seqLen - 1) * rowBytes;
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.NormOutput,
            srcOffset: (ulong)lastRowOffset, dstOffset: 0, size: (ulong)rowBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        _rmsnorm.Record(cmdBuf, _state.NormOutput, _weights.OutputNormWeight, _state.NormOutput,
            rowCount: 1, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        RecordMatmul(cmdBuf, _weights.OutputWeight, _weights.OutputDeviceQuantType,
            _state.NormOutput, _state.Logits,
            _weights.OutputOutputDim, _weights.OutputInputDim, seqLen: 1);
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        var logits = new float[vocabSize];
        _device.Download(_state.Logits, logits);
        ApplyFinalLogitSoftcapHost(logits);
        return logits;
    }

    /// <summary>
    /// Records the Gemma-4 attention block for one layer (cacheless / single
    /// forward — the AR validation + diffusion paths are cacheless). Mirrors the
    /// CPU <c>RunGemma4Layer</c> attention: attn_norm → Q/K(/V) proj (V branches
    /// off the RAW K projection on V-less global layers) → per-head Q/K-norm +
    /// WEIGHT-LESS V-norm → partial/dual RoPE → softmax(QᵀK·1.0)·V → o_proj.
    /// Writes o_proj into <c>NormOutput</c>; the shared post-attn-norm
    /// (<c>lw.PostAttnNormWeight</c>) + residual #1 in the layer loop then produce
    /// attn_out, exactly as for the Gemma-3 four-norm path.
    /// </summary>
    private void RecordGemma4Attention(nint cmdBuf, int layer, in VulkanWeights.LayerBuffers lw, int seqLen, float eps,
        ReadOnlySpan<int> positions, IKvCache? kvCache)
    {
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int headDim = Config.GetLayerHeadDim(layer);
        int numKvHeads = GemmaLayerKvHeads(layer);
        var g4 = lw.Gemma4!.Value;
        var (ropeTheta, ropeDim) = GemmaLayerRope(layer);

        // attn_norm → NormOutput
        _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.AttnNormWeight, _state.NormOutput,
            rowCount: seqLen, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Q, K projections (raw — K is captured before k-norm/rope for V-from-K).
        RecordMatmul(cmdBuf, lw.Q, lw.QDeviceQuantType, _state.NormOutput, _state.Q,
            lw.QOutputDim, lw.QInputDim, seqLen);
        RecordMatmul(cmdBuf, lw.K, lw.KDeviceQuantType, _state.NormOutput, _state.K,
            lw.KOutputDim, lw.KInputDim, seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        if (g4.VFromK)
        {
            // V-less global layer: V = raw K projection (no attn_v weight).
            long kvBytes = (long)seqLen * numKvHeads * headDim * sizeof(float);
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            RecordCopyBufferRange(cmdBuf, _state.K, _state.V, 0, 0, (ulong)kvBytes);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
        }
        else
        {
            RecordMatmul(cmdBuf, lw.V, lw.VDeviceQuantType, _state.NormOutput, _state.V,
                lw.VOutputDim, lw.VInputDim, seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }

        // Per-head Q/K RMSNorm (× learned weight); weight-less V RMSNorm (unit gamma).
        _rmsnorm.Record(cmdBuf, _state.Q, lw.QNormWeight!, _state.Q,
            rowCount: seqLen * numHeads, n: headDim, eps: eps);
        _rmsnorm.Record(cmdBuf, _state.K, lw.KNormWeight!, _state.K,
            rowCount: seqLen * numKvHeads, n: headDim, eps: eps);
        _rmsnorm.Record(cmdBuf, _state.V, Gemma4OnesVec(), _state.V,
            rowCount: seqLen * numKvHeads, n: headDim, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // RoPE(Q, K) — per-layer theta / rotated dims, NeoX. V is NOT roped.
        // Gemma-4 global (full-attention) layers use a NON-STANDARD partial-rotary
        // NeoX (matches CPU RoPE.ExecutePartialNeoX → ApplyRotationNeoXPartial): only
        // the leading `ropeDim` dims rotate, but BOTH (a) the per-pair frequency
        // denominator AND (b) the rotate-half pairing offset are over the FULL head
        // dim (freqDim = headDim, neoxPairOffset = headDim/2) — e.g. dims [0,64) ↔
        // [256,320). Sliding / full-rope layers keep the standard convention
        // (freqDim = ropeDim, pairing = ropeDim/2 — kernel defaults), which for those
        // layers (ropeDim == headDim) coincides anyway. This differs from EVERY OTHER
        // partial-rotary NeoX model (Qwen3 / NemotronH / Llama) which pairs and scales
        // within the rotated block (CPU RoPE.Execute → ApplyRotationNeoX) — those must
        // NOT receive the Gemma overrides.
        bool partialGlobal = Config.IsFullAttentionLayer(layer)
            && Config.PartialRotaryFactor is float prf && prf > 0f && prf < 1f
            && ropeDim < headDim;
        _rope.Record(cmdBuf, _state.Q, _state.K, _state.PositionsBuffer,
            seqLen: seqLen, numHeads: numHeads, numKvHeads: numKvHeads,
            headDim: headDim, ropeDim: ropeDim, theta: ropeTheta,
            variant: RopeF32Kernel.Variant.NeoX,
            freqDim: partialGlobal ? headDim : 0,
            neoxPairOffset: partialGlobal ? headDim / 2 : (int?)null);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Attention K/V source: either this forward's freshly-projected window
        // (cacheless — diffusion / single-shot) or the per-layer-strided KV cache
        // (autoregressive generation). The cache stores the FINAL K (post per-head
        // norm + RoPE) and V (post weight-less norm; on V-from-K global layers this
        // is the normed copy of the raw K projection) — exactly the buffers the
        // attention kernel reads, so appending them here mirrors the GQA path.
        int kvStride = numKvHeads * headDim;
        VulkanDevice.Buffer kSrc, vSrc;
        int seqKv;
        int positionOffset;
        AttentionMaskMode maskMode;
        int prefixLen;
        if (kvCache is VulkanKvCache vkCache)
        {
            vkCache.RecordUpdate(cmdBuf, _state.K, _state.V, positions, seqLen, layer);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
            kSrc = vkCache.GetKeysBuffer(layer);
            vSrc = vkCache.GetValuesBuffer(layer);
            seqKv = vkCache.CurrentLength;
            positionOffset = positions[0];
            maskMode = AttentionMaskMode.Causal;
            prefixLen = 0;
        }
        else if (kvCache is VulkanTurboQuantKvCache tqCache)
        {
            // See the GQA path for the barrier rationale (encode → dequant → attention, all COMPUTE).
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            tqCache.RecordUpdate(cmdBuf, _state.K, _state.V, positions, seqLen, layer);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            tqCache.RecordDequant(cmdBuf, layer);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            kSrc = tqCache.GetKeysBuffer();
            vSrc = tqCache.GetValuesBuffer();
            seqKv = tqCache.CurrentLength;
            positionOffset = positions[0];
            maskMode = AttentionMaskMode.Causal;
            prefixLen = 0;
        }
        else if (_pkvPhase == DiffusionKvPhase.Prefill)
        {
            // Capture this layer's final K/V (post per-head norm + RoPE; V post
            // weight-less norm, incl. V-from-K) for reuse across denoise steps, then
            // run the normal causal prompt attention (Hybrid(P) with P == seqLen).
            var store = _pkvStore!;
            long kvBytes = (long)seqLen * kvStride * sizeof(float);
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            RecordCopyBufferRange(cmdBuf, _state.K, store.Keys(layer), 0, 0, (ulong)kvBytes);
            RecordCopyBufferRange(cmdBuf, _state.V, store.Values(layer), 0, 0, (ulong)kvBytes);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
            kSrc = _state.K;
            vSrc = _state.V;
            seqKv = seqLen;
            positionOffset = 0;
            maskMode = _diffusionMaskMode;       // Hybrid(P==seqLen) ⇒ all-causal prompt
            prefixLen = _diffusionPrefixLen;
        }
        else if (_pkvPhase == DiffusionKvPhase.Decode)
        {
            // Attend the C canvas queries over [cached prompt K/V | fresh canvas K/V]
            // (length P+C) under a rectangular Bidirectional mask: a canvas query at
            // logical position P+i attends every prompt key + every canvas key, clipped
            // by the per-layer sliding window via positionOffset = P.
            int p = _pkvPromptLen;
            int c = seqLen;
            int kvCtx = p + c;
            var store = _pkvStore!;
            (VulkanDevice.Buffer kCat, VulkanDevice.Buffer vCat) = EnsurePkvConcat((long)kvCtx * kvStride * sizeof(float));
            long promptBytes = (long)p * kvStride * sizeof(float);
            long canvasBytes = (long)c * kvStride * sizeof(float);
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            RecordCopyBufferRange(cmdBuf, store.Keys(layer), kCat, 0, 0, (ulong)promptBytes);
            RecordCopyBufferRange(cmdBuf, store.Values(layer), vCat, 0, 0, (ulong)promptBytes);
            RecordCopyBufferRange(cmdBuf, _state.K, kCat, 0, (ulong)promptBytes, (ulong)canvasBytes);
            RecordCopyBufferRange(cmdBuf, _state.V, vCat, 0, (ulong)promptBytes, (ulong)canvasBytes);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
            kSrc = kCat;
            vSrc = vCat;
            seqKv = kvCtx;
            positionOffset = p;
            maskMode = AttentionMaskMode.Bidirectional;
            prefixLen = 0;
        }
        else
        {
            kSrc = _state.K;
            vSrc = _state.V;
            seqKv = seqLen;
            positionOffset = 0;
            // Diffusion: the cacheless canvas forward uses the region-aware mask
            // (Hybrid(P): prompt prefix causal, canvas bidirectional). On the AR path
            // _diffusionMaskMode is Causal/0 ⇒ byte-identical.
            maskMode = _diffusionMaskMode;
            prefixLen = _diffusionPrefixLen;
        }

        // Attention: scale = 1/sqrt(QPAS) = 1.0 (q/k-norm make Q,K unit); no attn softcap.
        RecordAttention(cmdBuf, _state.Q, kSrc, vSrc, _state.AttnOutput,
            seqQ: seqLen, seqKv: seqKv,
            numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
            positionOffset: positionOffset, slidingWindow: GetLayerSlidingWindow(layer),
            softCap: 0.0f, scaleOverride: GetAttentionScaleOverride(),
            maskMode: maskMode, prefixLen: prefixLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // o_proj → NormOutput. Shared post-attn-norm + residual #1 follow.
        RecordMatmul(cmdBuf, lw.O, lw.ODeviceQuantType, _state.AttnOutput, _state.NormOutput,
            lw.OOutputDim, lw.OInputDim, seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
    }

    /// <summary>
    /// Records the Gemma-4 dual parallel FFN for one layer. Reads attn_out
    /// (<c>HiddenState</c>, after residual #1), runs a dense GeGLU MLP and a
    /// 128-expert routed GeGLU MoE IN PARALLEL (each over its own pre-norm of
    /// attn_out), post-norms each branch, sums them, applies the combined
    /// post_ffw_norm, and writes the result into <c>NormOutput</c> — so the
    /// shared residual #2 adds attn_out and the per-layer output scale applies
    /// after. Mirrors CPU <c>Gemma4DenseFfn</c> + <c>Gemma4Moe</c>. The custom
    /// router input is <c>rms(attn_out)·(ffn_gate_inp.scale·1/√hidden)</c> (the
    /// 1/√hidden is pre-folded into RouterScale at upload); the per-expert down
    /// scale is pre-folded into the W2 bank.
    /// </summary>
    private unsafe void RecordGemma4Ffn(
        nint cmdBuf, int layer, VulkanWeights.MoeLayerBuffers moeW,
        in VulkanWeights.LayerBuffers lw, int seqLen, float eps)
    {
        int hidden = Config.HiddenSize;
        int interm = moeW.IntermediateSize;        // expert FF width (Ie)
        int denseInterm = lw.GateOutputDim;        // dense ("shared expert") FF width
        int numE = moeW.NumExperts;
        int topK = moeW.NumExpertsPerTok;
        int expandedRows = seqLen * topK;
        var g4 = lw.Gemma4!.Value;

        // ── Dense ("shared expert") branch: cur_mlp = rms(rms(attn_out)*ffn_norm GeGLU) * post_ffw_norm_1 ──
        _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.FfnNormWeight, _state.NormOutput,
            rowCount: seqLen, n: hidden, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        RecordMatmul(cmdBuf, lw.Gate, lw.GateDeviceQuantType, _state.NormOutput, _state.FfnGate,
            lw.GateOutputDim, lw.GateInputDim, seqLen);
        RecordMatmul(cmdBuf, lw.Up, lw.UpDeviceQuantType, _state.NormOutput, _state.FfnUp,
            lw.UpOutputDim, lw.UpInputDim, seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        _geglu!.Record(cmdBuf, _state.FfnGate, _state.FfnUp, _state.SiluOutput, seqLen * denseInterm);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        RecordMatmul(cmdBuf, lw.Down, lw.DownDeviceQuantType, _state.SiluOutput, _state.Gemma4DenseResult!,
            lw.DownOutputDim, lw.DownInputDim, seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        _rmsnorm.Record(cmdBuf, _state.Gemma4DenseResult!, g4.PostFfwNorm1, _state.Gemma4DenseResult!,
            rowCount: seqLen, n: hidden, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── MoE branch ──
        // Custom router: logits = ffn_gate_inp · (rms(attn_out) · RouterScale·1/√H).
        _rmsnorm.Record(cmdBuf, _state.HiddenState, g4.RouterScale, _state.NormOutput,
            rowCount: seqLen, n: hidden, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        RecordMatmul(cmdBuf, moeW.Gate, moeW.GateDeviceQuantType, _state.NormOutput, _state.MoeRouterLogits!,
            outputDim: numE, inputDim: hidden, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        _moeTopkSoftmax!.Record(cmdBuf, _state.MoeRouterLogits!, _state.MoeTopkIndices!, _state.MoeTopkWeights!,
            seqLen: seqLen, numExperts: numE, k: topK, normTopKProb: moeW.NormTopKProb);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        // Expert input = rms(attn_out) * pre_ffw_norm_2 (overwrites the router-input temp).
        _rmsnorm.Record(cmdBuf, _state.HiddenState, g4.PreFfwNorm2, _state.NormOutput,
            rowCount: seqLen, n: hidden, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        _moeBroadcast!.Record(cmdBuf, _state.NormOutput, _state.MoeExpandedInput!,
            seqLen: seqLen, topK: topK, hidden: hidden);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        RecordMoeIndexedMatmul(cmdBuf, moeW.W1Bank, _state.MoeExpandedInput!, _state.MoeTopkIndices!, _state.MoeGateInter!,
            moeW.W1DeviceQuantType, m: interm, k: hidden, n: expandedRows, numExperts: numE);
        RecordMoeIndexedMatmul(cmdBuf, moeW.W3Bank, _state.MoeExpandedInput!, _state.MoeTopkIndices!, _state.MoeUpInter!,
            moeW.W3DeviceQuantType, m: interm, k: hidden, n: expandedRows, numExperts: numE);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        _geglu!.Record(cmdBuf, _state.MoeGateInter!, _state.MoeUpInter!, _state.MoeSiluInter!, expandedRows * interm);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        if (moeW.W2DeviceQuantType == QuantType.Q5_1)
        {
            // Quantized down path: the Q5_1 indexed-matmul shader folds the per-expert
            // ffn_down_exps.scale (op #14) into its output, so the weighted-scatter below
            // carries only the routing weight — exactly matching the CPU fold order.
            if (_moeIndexedMatmulQ5_1 is null)
                throw new InvalidOperationException("Q5_1 MoE indexed matmul kernel was not created.");
            _moeIndexedMatmulQ5_1.Record(cmdBuf, moeW.W2Bank, _state.MoeSiluInter!, _state.MoeTopkIndices!,
                _state.MoeDownRows!, g4.DownExpertScale!, m: hidden, k: interm, n: expandedRows, numExperts: numE);
        }
        else
        {
            // F32 host-dequant path: per-expert down scale was pre-folded into W2 at upload.
            RecordMoeIndexedMatmul(cmdBuf, moeW.W2Bank, _state.MoeSiluInter!, _state.MoeTopkIndices!, _state.MoeDownRows!,
                moeW.W2DeviceQuantType, m: hidden, k: interm, n: expandedRows, numExperts: numE);
        }
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        // Weighted scatter (routing weights; per-expert down scale folded by the Q5_1 shader
        // or pre-folded into the F32 W2) → Gemma4MoeResult.
        _moeWeightedScatter!.Record(cmdBuf, _state.MoeDownRows!, _state.MoeTopkWeights!, _state.Gemma4MoeResult!,
            seqLen: seqLen, topK: topK, hiddenSize: hidden);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        _rmsnorm.Record(cmdBuf, _state.Gemma4MoeResult!, g4.PostFfwNorm2, _state.Gemma4MoeResult!,
            rowCount: seqLen, n: hidden, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // ── Combine: cur = rms(dense + moe) * post_ffw_norm → NormOutput ──
        _add.Record(cmdBuf, _state.Gemma4DenseResult!, _state.Gemma4MoeResult!, _state.NormOutput, seqLen * hidden);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        _rmsnorm.Record(cmdBuf, _state.NormOutput, g4.PostFfwNorm, _state.NormOutput,
            rowCount: seqLen, n: hidden, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
    }

    /// <summary>
    /// Records the DiffusionGemma canvas region embedding for the contiguous canvas
    /// tail rows <c>[regionP, regionP+canvasLen)</c> of <see cref="VulkanForwardState.HiddenState"/>:
    /// (optionally) adds the self-conditioning signal, then applies the weight-less
    /// (unit-gamma) <c>rms_norm</c> — mirroring CPU <c>TransformerModel</c>'s region-embed
    /// block (diffusion-gemma.cpp <c>dg_canvas_embed</c>). The canvas rows are a contiguous
    /// tail, so the work is done in <see cref="VulkanForwardState.NormOutput"/> (a device
    /// scratch, free before the layer loop) and copied back — reusing the shared rmsnorm/add
    /// kernels with no offset binding or new shader.
    /// </summary>
    private void RecordDiffusionCanvasEmbed(nint cmdBuf, int regionP, int canvasLen, int hiddenSize, float eps)
    {
        long rowBytes = (long)hiddenSize * sizeof(float);
        ulong canvasOffset = (ulong)((long)regionP * rowBytes);
        ulong canvasBytes = (ulong)((long)canvasLen * rowBytes);

        // Copy the canvas tail of HiddenState into NormOutput[0..] (device scratch).
        KernelSupport.ComputeToTransferBarrier(cmdBuf);
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.NormOutput,
            srcOffset: canvasOffset, dstOffset: 0, size: canvasBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        // Self-conditioning (steps > 0): add the host-computed sc_sig to the canvas
        // embedding BEFORE the weight-less rms_norm. sc_sig is computed entirely from
        // the previous step's canvas logits (already on host) — independent of device
        // state — so it is uploaded into a host-visible buffer and device-added here.
        bool applySc = _scUse > 0f
            && _cpuWeights.SelfCond is not null
            && _scPrevLogits is not null
            && _scCanvasLen == canvasLen
            && canvasLen > 0;
        if (applySc)
        {
            VulkanDevice.Buffer scSig = EnsureScSigBuffer(canvasLen * hiddenSize);
            ComputeSelfConditioningSignalHost(canvasLen, hiddenSize, eps, scSig);
            _add.Record(cmdBuf, _state.NormOutput, scSig, _state.NormOutput, canvasLen * hiddenSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }

        // Weight-less rms_norm (unit gamma) over the canvasLen rows in place.
        _rmsnorm.Record(cmdBuf, _state.NormOutput, Gemma4OnesVec(), _state.NormOutput,
            rowCount: canvasLen, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToTransferBarrier(cmdBuf);

        // Copy the normalised canvas rows back into HiddenState's canvas tail.
        RecordCopyBufferRange(cmdBuf, _state.NormOutput, _state.HiddenState,
            srcOffset: 0, dstOffset: canvasOffset, size: canvasBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
    }

    /// <summary>Lazily (re)allocates the host-visible self-conditioning signal buffer to hold <paramref name="elems"/> floats.</summary>
    private VulkanDevice.Buffer EnsureScSigBuffer(int elems)
    {
        if (_diffusionScSig is null || _diffusionScSigCapacityElems < elems)
        {
            _diffusionScSig?.Dispose();
            _diffusionScSig = _device.Allocate((long)elems * sizeof(float));
            _diffusionScSigCapacityElems = elems;
            // The new handle invalidates any cached add-kernel descriptor set.
            InvalidateKernelCaches();
        }
        return _diffusionScSig;
    }

    /// <summary>
    /// Computes the DiffusionGemma self-conditioning signal <c>sc_sig[canvasLen × hidden]</c>
    /// entirely on the host (mirrors CPU <c>TransformerModel.ApplySelfConditioning</c>) and
    /// uploads it into <paramref name="dst"/>. SC feeds the previous step's canvas logits back
    /// via a gated GeGLU MLP over a soft token-embedding:
    /// <code>
    /// soft[c]   = sqrt(n_embd) * Σ_v softmax(prev_logits[c])[v] * tok_embd[v]
    /// normed[c] = rms_norm(soft[c]) * self_cond_pre_norm
    /// sc_sig[c] = self_cond_down( gelu_tanh(self_cond_gate·normed) * (self_cond_up·normed) )
    /// </code>
    /// Source-confirmed against diffusion-gemma.cpp; runs only on the (small) canvas region,
    /// once per denoise step. The soft-embed sweeps the tied token-embedding table once.
    /// </summary>
    private unsafe void ComputeSelfConditioningSignalHost(int canvasLen, int hiddenSize, float eps, VulkanDevice.Buffer dst)
    {
        var sc = _cpuWeights.SelfCond!;
        int vocab = Config.VocabSize;
        int ff = sc.GateOut;
        float embScale = Config.EmbeddingScale ?? 1.0f;
        float[] prev = _scPrevLogits!;

        var soft = new float[canvasLen * hiddenSize];
        var probs = new float[vocab];
        var row = new float[hiddenSize];
        var normed = new float[canvasLen * hiddenSize];
        var gate = new float[canvasLen * ff];
        var up = new float[canvasLen * ff];
        var gelu = new float[canvasLen * ff];
        var sig = new float[canvasLen * hiddenSize];

        // soft[c] = Σ_v softmax(prev_logits[c])[v] * tok_embd[v]. Single vocab sweep:
        // each embedding row is dequantized once and scatter-accumulated into every
        // canvas soft-vector weighted by that token's probability.
        // (Compute per-column probs lazily to avoid a [canvasLen × vocab] buffer.)
        var allProbs = new float[canvasLen * vocab];
        for (int c = 0; c < canvasLen; c++)
            Softmax.Execute(prev.AsSpan(c * vocab, vocab), allProbs.AsSpan(c * vocab, vocab));
        for (int v = 0; v < vocab; v++)
        {
            DequantTokenEmbedRow(v, row, hiddenSize);
            for (int c = 0; c < canvasLen; c++)
            {
                float w = allProbs[c * vocab + v];
                if (w == 0f) continue;
                TensorPrimitives.MultiplyAdd(row, w, soft.AsSpan(c * hiddenSize, hiddenSize),
                    soft.AsSpan(c * hiddenSize, hiddenSize));
            }
        }
        _ = probs;

        // soft *= sqrt(n_embd); normed = rms_norm(soft) * self_cond_pre_norm.
        for (int c = 0; c < canvasLen; c++)
        {
            var softC = soft.AsSpan(c * hiddenSize, hiddenSize);
            if (embScale != 1.0f) TensorPrimitives.Multiply(softC, embScale, softC);
            RmsNorm.Execute(softC, sc.PreNorm, eps, normed.AsSpan(c * hiddenSize, hiddenSize));
        }

        // g = gate·normed ; u = up·normed (batched). gelu = gelu_tanh(g) * u.
        HostGemm(sc.GatePtr, sc.GateQt, normed, gate, sc.GateOut, sc.GateIn, canvasLen);
        HostGemm(sc.UpPtr, sc.UpQt, normed, up, sc.UpOut, sc.UpIn, canvasLen);
        for (int c = 0; c < canvasLen; c++)
            FusedOps.GeGLUTanh(gate.AsSpan(c * ff, ff), up.AsSpan(c * ff, ff), gelu.AsSpan(c * ff, ff));

        // sc_sig = down·(gelu*u).
        HostGemm(sc.DownPtr, sc.DownQt, gelu, sig, sc.DownOut, sc.DownIn, canvasLen);

        _device.Upload(sig.AsSpan(0, canvasLen * hiddenSize), dst);
    }

    /// <summary>
    /// Host F32 GEMM mirroring the CPU model's row-major projection contract:
    /// <c>out[r*M + m] = Σ_k W[m,k] * in[r*K + k]</c>, dequantizing each weight row of
    /// <paramref name="weightPtr"/> (quant <paramref name="qt"/>) on the fly. Used only for
    /// the small self-conditioning gate/up/down projections.
    /// </summary>
    private static unsafe void HostGemm(
        nint weightPtr, QuantizationType qt, ReadOnlySpan<float> input, Span<float> output,
        int m, int k, int rows)
    {
        long rowBytes = qt == QuantizationType.F32 ? (long)k * sizeof(float)
            : qt == QuantizationType.F16 ? (long)k * sizeof(Half)
            : Dequantize.RowByteSize(k, qt);
        Span<float> wRow = k <= 4096 ? stackalloc float[k] : new float[k];
        for (int mi = 0; mi < m; mi++)
        {
            nint rp = weightPtr + (nint)((long)mi * rowBytes);
            if (qt == QuantizationType.F32)
                new ReadOnlySpan<float>((float*)rp, k).CopyTo(wRow);
            else if (qt == QuantizationType.F16)
                TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>((Half*)rp, k), wRow);
            else
                Dequantize.ToFloat32(rp, k, qt, wRow);
            for (int r = 0; r < rows; r++)
                output[r * m + mi] = TensorPrimitives.Dot(wRow, input.Slice(r * k, k));
        }
    }

    /// <summary>Dequantizes one token-embedding row (raw, no embedding scale) into <paramref name="dest"/> [hidden].</summary>
    private unsafe void DequantTokenEmbedRow(int tokenId, Span<float> dest, int hiddenSize)
    {
        nint embPtr = _cpuWeights.TokenEmbedWeight;
        var qt = _cpuWeights.TokenEmbedQuantType;
        if (qt == QuantizationType.F32)
            new ReadOnlySpan<float>((float*)embPtr + (long)tokenId * hiddenSize, hiddenSize).CopyTo(dest);
        else if (qt == QuantizationType.F16)
            TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>((Half*)embPtr + (long)tokenId * hiddenSize, hiddenSize), dest);
        else
        {
            long rowBytes = Dequantize.RowByteSize(hiddenSize, qt);
            Dequantize.ToFloat32(embPtr + (nint)((long)tokenId * rowBytes), hiddenSize, qt, dest);
        }
    }

    /// <summary>
    /// Records the MoE (Mixtral / Qwen-MoE) FFN block for one layer:
    /// rmsnorm → router gate matmul → top-k softmax → broadcast hidden
    /// to per-(token, slot) → indexed gate / up matmuls (W1, W3) →
    /// SwiGLU → indexed down matmul (W2) → weighted scatter back to
    /// per-token output. Writes the post-MoE result into <c>_state.NormOutput</c>
    /// so the shared residual-add downstream sees the same contract as
    /// the dense FFN path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The per-routed-expert <c>W1</c>/<c>W2</c>/<c>W3</c> banks are always F32 — the
    /// indexed-matmul kernel (<c>moe_indexed_matmul_f32</c>) is F32-only in tree, no Q8_0
    /// indexed variant exists yet. The router gate and the shared-expert
    /// <c>W1</c>/<c>W2</c>/<c>W3</c> + sigmoid gate are dispatched through the standard
    /// <see cref="RecordMatmul"/> router so they pick the Q8_0 GEMV/GEMM kernels when the
    /// upload kept the source as raw Q8_0 blocks (and fall back to <c>matmul_f32</c>
    /// otherwise — the production loader path today).
    /// </para>
    /// </remarks>
    private unsafe void RecordMoeLayer(
        nint cmdBuf, int layer, VulkanWeights.MoeLayerBuffers moeW,
        in VulkanWeights.LayerBuffers lw, int seqLen, float eps)
    {
        int hidden = moeW.HiddenSize;
        int interm = moeW.IntermediateSize;
        int numE = moeW.NumExperts;
        int topK = moeW.NumExpertsPerTok;
        int expandedRows = seqLen * topK;

        // 1. Pre-FFN RMSNorm: HiddenState → NormOutput.
        _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.FfnNormWeight, _state.NormOutput,
            rowCount: seqLen, n: hidden, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 2. Router gate matmul: Gate @ NormOutput → MoeRouterLogits.
        //    Q8_0 dispatch via RecordMatmul (matmul_q8_0 GEMV at seqLen==1 / GEMM at >1)
        //    when the upload kept the gate as Q8_0; otherwise the standard F32 kernel.
        RecordMatmul(cmdBuf, moeW.Gate, moeW.GateDeviceQuantType,
            _state.NormOutput, _state.MoeRouterLogits!,
            outputDim: numE, inputDim: hidden, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 3. Top-k softmax: writes MoeTopkIndices (int) and MoeTopkWeights.
        _moeTopkSoftmax!.Record(cmdBuf,
            _state.MoeRouterLogits!, _state.MoeTopkIndices!, _state.MoeTopkWeights!,
            seqLen: seqLen, numExperts: numE, k: topK, normTopKProb: moeW.NormTopKProb);
        // Broadcast (compute) reads NormOutput, writes MoeExpandedInput; the
        // indexed matmul downstream reads MoeExpandedInput plus topk
        // indices/weights. A single compute→compute barrier covers both
        // RMSNorm-output → broadcast-read on NormOutput and topk-write →
        // matmul-read on the indices/weights.
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 4. Broadcast NormOutput[seqLen, hidden] → MoeExpandedInput[seqLen*topK, hidden].
        //    Each token's row gets replicated topK times so each (t, slot)
        //    shares the same input but consumes a different expert. One
        //    compute dispatch replaces the seqLen × topK loop of
        //    vkCmdCopyBuffer regions the previous implementation issued —
        //    same math, no transfer↔compute stage transition, dispatch
        //    count drops from O(seqLen·topK) to 1 per MoE layer.
        _moeBroadcast!.Record(cmdBuf,
            _state.NormOutput, _state.MoeExpandedInput!,
            seqLen: seqLen, topK: topK, hidden: hidden);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        if (CanUseGroupedF16Moe(moeW, hidden, interm))
        {
            RecordMoeGroupedF16Layer(cmdBuf, moeW, expandedRows, hidden, interm, numE);
        }
        else
        {
            // 5. Indexed expert matmuls: gate (W1) and up (W3) project the
            //    expanded input through the experts selected by topk indices.
            RecordMoeIndexedMatmul(cmdBuf,
                moeW.W1Bank, _state.MoeExpandedInput!, _state.MoeTopkIndices!, _state.MoeGateInter!,
                moeW.W1DeviceQuantType,
                m: interm, k: hidden, n: expandedRows, numExperts: numE);
            RecordMoeIndexedMatmul(cmdBuf,
                moeW.W3Bank, _state.MoeExpandedInput!, _state.MoeTopkIndices!, _state.MoeUpInter!,
                moeW.W3DeviceQuantType,
                m: interm, k: hidden, n: expandedRows, numExperts: numE);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            MaybeApplyMoeIndexedLoraDeltas(cmdBuf, layer, "gate_proj",
                _state.MoeExpandedInput!, _state.MoeTopkIndices!, _state.MoeGateInter!,
                rows: expandedRows, inputDim: hidden, outputDim: interm, numExperts: numE);
            MaybeApplyMoeIndexedLoraDeltas(cmdBuf, layer, "up_proj",
                _state.MoeExpandedInput!, _state.MoeTopkIndices!, _state.MoeUpInter!,
                rows: expandedRows, inputDim: hidden, outputDim: interm, numExperts: numE);
            if (_currentLora is not null)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // 6. SwiGLU pointwise: silu(gate) * up.
            _swiglu.Record(cmdBuf, _state.MoeGateInter!, _state.MoeUpInter!, _state.MoeSiluInter!,
                n: expandedRows * interm);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // 7. Indexed down matmul (W2): silu_intermediate → MoeDownRows.
            RecordMoeIndexedMatmul(cmdBuf,
                moeW.W2Bank, _state.MoeSiluInter!, _state.MoeTopkIndices!, _state.MoeDownRows!,
                moeW.W2DeviceQuantType,
                m: hidden, k: interm, n: expandedRows, numExperts: numE);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            MaybeApplyMoeIndexedLoraDeltas(cmdBuf, layer, "down_proj",
                _state.MoeSiluInter!, _state.MoeTopkIndices!, _state.MoeDownRows!,
                rows: expandedRows, inputDim: interm, outputDim: hidden, numExperts: numE);
            if (_currentLora is not null)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }

        // 8. Weighted scatter: combine each token's topK expert outputs into
        //    NormOutput, scaled by the routing weights.
        _moeWeightedScatter!.Record(cmdBuf,
            _state.MoeDownRows!, _state.MoeTopkWeights!, _state.NormOutput,
            seqLen: seqLen, topK: topK, hiddenSize: hidden);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 9. Shared-expert branch (DeepSeek-V2/V3 ungated). Each shared expert
        //    runs a dense SwiGLU MLP on the per-token hidden state and the
        //    outputs are summed into the routed result. Skipped when the
        //    layer has no shared experts (Mixtral / Qwen3-MoE without shared).
        if (moeW.NumSharedExperts > 0)
        {
            RecordMoeSharedExperts(cmdBuf, moeW, lw.FfnNormWeight, seqLen, hidden, eps);
        }
    }

    private bool CanUseGroupedF16Moe(
        in VulkanWeights.MoeLayerBuffers moeW, int hidden, int interm)
        => _currentLora is null
        && _moeExpertOffsets is not null
        && _moeExpandGroupByExpert is not null
        && _moeGroupedMatmulF16Coopmat is not null
        && _moeUngroupScatter is not null
        && moeW.W1DeviceQuantType == QuantType.F16
        && moeW.W2DeviceQuantType == QuantType.F16
        && moeW.W3DeviceQuantType == QuantType.F16
        && (hidden % MoeGroupedMatmulF16CoopmatKernel.KChunk) == 0
        && (interm % MoeGroupedMatmulF16CoopmatKernel.KChunk) == 0;

    private void RecordMoeGroupedF16Layer(
        nint cmdBuf, VulkanWeights.MoeLayerBuffers moeW,
        int expandedRows, int hidden, int interm, int numExperts)
    {
        _moeExpertOffsets!.Record(cmdBuf,
            _state.MoeTopkIndices!, _state.MoeExpertCounts!,
            _state.MoeExpertOffsets!, _state.MoeExpertCounters!,
            rows: expandedRows, numExperts: numExperts);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _moeExpandGroupByExpert!.Record(cmdBuf,
            _state.MoeExpandedInput!, _state.MoeTopkIndices!, _state.MoeExpertOffsets!,
            _state.MoeExpertCounters!, _state.MoeGroupedHidden!, _state.MoePermutation!,
            rows: expandedRows, hidden: hidden, numExperts: numExperts);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _moeGroupedMatmulF16Coopmat!.Record(cmdBuf,
            moeW.W1Bank, _state.MoeGroupedHidden!, _state.MoeExpertOffsets!, _state.MoeGroupedGateInter!,
            m: interm, k: hidden, rows: expandedRows, numExperts: numExperts);
        _moeGroupedMatmulF16Coopmat.Record(cmdBuf,
            moeW.W3Bank, _state.MoeGroupedHidden!, _state.MoeExpertOffsets!, _state.MoeGroupedUpInter!,
            m: interm, k: hidden, rows: expandedRows, numExperts: numExperts);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _moeUngroupScatter!.Record(cmdBuf,
            _state.MoeGroupedGateInter!, _state.MoePermutation!, _state.MoeGateInter!,
            rows: expandedRows, hidden: interm);
        _moeUngroupScatter.Record(cmdBuf,
            _state.MoeGroupedUpInter!, _state.MoePermutation!, _state.MoeUpInter!,
            rows: expandedRows, hidden: interm);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _swiglu.Record(cmdBuf, _state.MoeGateInter!, _state.MoeUpInter!, _state.MoeSiluInter!,
            n: expandedRows * interm);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Re-run the count/prefix kernel to reset group counters before grouping the
        // post-SwiGLU rows for W2. Offsets/counts are deterministic for the same indices.
        _moeExpertOffsets.Record(cmdBuf,
            _state.MoeTopkIndices!, _state.MoeExpertCounts!,
            _state.MoeExpertOffsets!, _state.MoeExpertCounters!,
            rows: expandedRows, numExperts: numExperts);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _moeExpandGroupByExpert.Record(cmdBuf,
            _state.MoeSiluInter!, _state.MoeTopkIndices!, _state.MoeExpertOffsets!,
            _state.MoeExpertCounters!, _state.MoeGroupedGateInter!, _state.MoePermutation!,
            rows: expandedRows, hidden: interm, numExperts: numExperts);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _moeGroupedMatmulF16Coopmat.Record(cmdBuf,
            moeW.W2Bank, _state.MoeGroupedGateInter!, _state.MoeExpertOffsets!, _state.MoeGroupedHidden!,
            m: hidden, k: interm, rows: expandedRows, numExperts: numExperts);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _moeUngroupScatter.Record(cmdBuf,
            _state.MoeGroupedHidden!, _state.MoePermutation!, _state.MoeDownRows!,
            rows: expandedRows, hidden: hidden);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
    }

    /// <summary>
    /// Records the shared-expert branch of a DeepSeek-V2/V3-style MoE layer:
    /// for each shared expert run a dense SwiGLU MLP over the per-token
    /// normalised hidden state, sum the outputs, and add the sum into the
    /// routed-MoE result already in <c>_state.NormOutput</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The routed-MoE scatter has overwritten <c>NormOutput</c> with the
    /// routed sum already, so we re-derive the normalised hidden state from
    /// <c>HiddenState</c> via a fresh rmsnorm into
    /// <see cref="VulkanForwardState.MoeSharedInput"/> — a dedicated buffer
    /// that pins the shared-expert input across every iteration. That keeps
    /// the SumA / SumB pair available as pure ping-pong accumulator slots.
    /// </para>
    /// <para>
    /// Accumulation: shared expert 0's down-projection writes directly into
    /// SumA. For each subsequent expert we matmul into MoeSharedDown and add
    /// (running-sum, MoeSharedDown) into the alternating ping-pong side.
    /// After all shared experts the running sum is folded into NormOutput via
    /// the unused ping-pong slot and a device-to-device copy lands the result
    /// back in <c>NormOutput</c> so the caller's residual-add contract is
    /// preserved.
    /// </para>
    /// </remarks>
    private unsafe void RecordMoeSharedExperts(
        nint cmdBuf, VulkanWeights.MoeLayerBuffers moeW,
        VulkanDevice.Buffer ffnNormWeight, int seqLen, int hidden, float eps)
    {
        int numShared = moeW.NumSharedExperts;
        int sharedI = moeW.SharedIntermediateSize;
        int hiddenElems = seqLen * hidden;
        int sharedInterElems = seqLen * sharedI;

        // Re-derive the normalised hidden state. NormOutput is occupied by
        // the routed-MoE result; HiddenState still holds the pre-FFN residual.
        var sharedInput = _state.MoeSharedInput!;
        _rmsnorm.Record(cmdBuf, _state.HiddenState, ffnNormWeight, sharedInput,
            rowCount: seqLen, n: hidden, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // SumA / SumB ping-pong; activeSum tracks the slot currently holding
        // the running shared-expert sum. Expert 0 writes directly into SumA;
        // subsequent experts compute their down-output into MoeSharedDown and
        // we add (activeSum + MoeSharedDown) → the OTHER side, alternating.
        VulkanDevice.Buffer activeSum = _state.MoeSharedSumA!;

        for (int s = 0; s < numShared; s++)
        {
            // gate / up matmuls share sharedInput; SwiGLU then fuses them. Each
            // dispatch goes through RecordMatmul so Q8_0 shared-expert weights pick
            // the matmul_q8_0[_gemm[_coopmat]] kernels when the upload kept them
            // Q8_0; F32 dispatches stay on matmul_f32 (production path).
            RecordMatmul(cmdBuf, moeW.SharedW1![s], moeW.SharedW1DeviceQuantType,
                sharedInput, _state.MoeSharedGate!,
                outputDim: sharedI, inputDim: hidden, seqLen: seqLen);
            RecordMatmul(cmdBuf, moeW.SharedW3![s], moeW.SharedW3DeviceQuantType,
                sharedInput, _state.MoeSharedUp!,
                outputDim: sharedI, inputDim: hidden, seqLen: seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            _swiglu.Record(cmdBuf, _state.MoeSharedGate!, _state.MoeSharedUp!, _state.MoeSharedSilu!,
                n: sharedInterElems);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            if (s == 0)
            {
                // First expert seeds the running sum directly into SumA.
                RecordMatmul(cmdBuf, moeW.SharedW2![s], moeW.SharedW2DeviceQuantType,
                    _state.MoeSharedSilu!, _state.MoeSharedSumA!,
                    outputDim: hidden, inputDim: sharedI, seqLen: seqLen);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                activeSum = _state.MoeSharedSumA!;
            }
            else
            {
                // Per-expert down output → MoeSharedDown, then ping-pong add
                // into the slot opposite activeSum.
                RecordMatmul(cmdBuf, moeW.SharedW2![s], moeW.SharedW2DeviceQuantType,
                    _state.MoeSharedSilu!, _state.MoeSharedDown!,
                    outputDim: hidden, inputDim: sharedI, seqLen: seqLen);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);

                var sumDst = activeSum.Handle == _state.MoeSharedSumA!.Handle
                    ? _state.MoeSharedSumB!
                    : _state.MoeSharedSumA!;
                _add.Record(cmdBuf, activeSum, _state.MoeSharedDown!, sumDst, hiddenElems);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                activeSum = sumDst;
            }
        }

        // Fold the running shared sum into NormOutput. Two paths:
        //  - DeepSeek-V2/V3 (no gate): plain add via a ping-pong destination,
        //    then a vkCmdCopyBuffer back into NormOutput (the existing add
        //    kernel cannot self-write).
        //  - Qwen1.5-MoE (sigmoid gate): compute per-token gate logits via a
        //    1×hidden matmul against SharedExpertGate, then apply sigmoid +
        //    weighted-add into NormOutput in place via the fused kernel.
        //    No ping-pong / copy needed — the gated kernel writes NormOutput
        //    directly, with sigmoid(logit_t) folded into the per-token scale.
        if (moeW.SharedExpertGate is not null)
        {
            // gateLogits[t] = SharedExpertGate[1, hidden] @ MoeSharedInput[t, :].
            // The post-FFN-RMSNorm hidden state is the right input here — it
            // mirrors MoeSwiGluMlp.ExecuteCoreGrouped which receives the
            // already-RMSNormed hidden as `hidden` and computes the gate
            // logit against that same buffer. RecordMatmul picks Q8_0 vs F32
            // based on the upload-time storage choice.
            RecordMatmul(cmdBuf, moeW.SharedExpertGate, moeW.SharedExpertGateDeviceQuantType,
                sharedInput, _state.MoeSharedGateLogits!,
                outputDim: 1, inputDim: hidden, seqLen: seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            _moeSigmoidGatedAdd!.Record(cmdBuf,
                output: _state.NormOutput, b: activeSum, gateLogits: _state.MoeSharedGateLogits!,
                seqLen: seqLen, hiddenSize: hidden);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
        else
        {
            VulkanDevice.Buffer foldDst = activeSum.Handle == _state.MoeSharedSumA!.Handle
                ? _state.MoeSharedSumB!
                : _state.MoeSharedSumA!;
            _add.Record(cmdBuf, _state.NormOutput, activeSum, foldDst, hiddenElems);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            var foldRegion = new VkBufferCopy
            {
                srcOffset = 0,
                dstOffset = 0,
                size = (ulong)hiddenElems * sizeof(float),
            };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, foldDst.Handle, _state.NormOutput.Handle, 1, foldRegion);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
        }
    }

    /// <summary>
    /// Routes between the scalar and tiled (shared-memory) variants of the
    /// MoE indexed expert matmul based on dispatch shape. Tiled wins on
    /// prefill at large N (each token's x-row is reloaded TILE_M times by
    /// the scalar variant — the tile amortises that), scalar wins on decode
    /// where N is tiny and a TILE_M-wide cooperative load is mostly idle.
    /// </summary>
    /// <remarks>
    /// Heuristic: use the tiled kernel when <c>n ≥ TiledMinRows</c> AND
    /// <c>m % TILE_M == 0</c>. The first guard keeps decode (N ≤ 16, e.g.
    /// 1 token × topK=8 = 8 expanded rows) on the scalar fast path. The
    /// second avoids the shader's tail-bounds path on prefill — the tile
    /// kernel handles ragged m correctly via its in-shader bounds checks,
    /// but on a divisible m we get the cleanest dispatch shape with no
    /// branching in the inner loop. The threshold is conservative; a
    /// future perf-wave should re-tune from device benchmarks.
    /// </remarks>
    private const int TiledMinRows = 32;

    private void RecordMoeIndexedMatmul(
        nint cmdBuf,
        VulkanDevice.Buffer bank, VulkanDevice.Buffer x,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        QuantizationType weightQt,
        int m, int k, int n, int numExperts)
    {
        if (weightQt == QuantizationType.Q8_0)
        {
            if (_moeIndexedMatmulQ8 is null)
                throw new InvalidOperationException("Q8_0 MoE indexed matmul kernel was not created.");
            _moeIndexedMatmulQ8.Record(cmdBuf,
                bank, x, indices, y,
                m: m, k: k, n: n, numExperts: numExperts);
            return;
        }

        if (weightQt == QuantizationType.Q4_K)
        {
            if (_moeIndexedMatmulQ4K is null)
                throw new InvalidOperationException("Q4_K MoE indexed matmul kernel was not created.");
            _moeIndexedMatmulQ4K.Record(cmdBuf,
                bank, x, indices, y,
                m: m, k: k, n: n, numExperts: numExperts);
            return;
        }

        if (weightQt != QuantizationType.F32)
            throw new NotSupportedException($"MoE indexed matmul does not support {weightQt} banks.");

        bool useTiled = _moeIndexedMatmulTiled is not null
            && n >= TiledMinRows
            && (m % MoeIndexedMatmulTiledF32Kernel.TileM) == 0;

        if (useTiled)
        {
            _moeIndexedMatmulTiled!.Record(cmdBuf,
                bank, x, indices, y,
                m: m, k: k, n: n, numExperts: numExperts);
        }
        else
        {
            _moeIndexedMatmul!.Record(cmdBuf,
                bank, x, indices, y,
                m: m, k: k, n: n, numExperts: numExperts);
        }
    }

    private void MaybeApplyMoeIndexedLoraDeltas(
        nint cmdBuf,
        int layer,
        string projection,
        VulkanDevice.Buffer x,
        VulkanDevice.Buffer indices,
        VulkanDevice.Buffer y,
        int rows,
        int inputDim,
        int outputDim,
        int numExperts)
    {
        var lora = _currentLora;
        if (lora is null || _moeIndexedLoraDelta is null) return;

        for (int expert = 0; expert < numExperts; expert++)
        {
            string projName = $"mlp.experts.{expert}.{projection}";
            var lb = lora.Get(layer, projName);
            if (lb is not { } w) continue;

            if (w.InputDim != inputDim || w.OutputDim != outputDim)
                throw new InvalidOperationException(
                    $"LoRA adapter '{lora.Source.Name}' layer={layer} proj='{projName}' shape "
                    + $"({w.InputDim}x{w.OutputDim}) does not match MoE projection ({inputDim}x{outputDim}).");

            _moeIndexedLoraDelta.Record(cmdBuf,
                x, indices, w.B, w.A, y,
                rows: rows, inputDim: inputDim, outputDim: outputDim,
                rank: w.Rank, expert: expert);
        }
    }

    /// <summary>
    /// Dispatches the LoRA delta for <paramref name="projName"/> at
    /// <paramref name="layer"/> when an adapter is active and targets that
    /// site. No-op when there is no active adapter or no entry.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Fast path (rank ≤ <see cref="LoraDeltaGemvFusedF32Kernel.MaxRank"/>
    /// and the fused .spv blob is present): a single dispatch of
    /// <see cref="LoraDeltaGemvFusedF32Kernel"/> performs
    /// <c>y[t, m] += sum_r A[m, r] · dot(B[r, :], x[t, :])</c> in place.
    /// One workgroup per token row keeps the rank-sized inner reduction in
    /// shared memory and reuses it across the full output dim.
    /// </para>
    /// <para>
    /// Fallback path (rank &gt; 32 or older builds without the fused .spv):
    /// the original 4-dispatch chain
    /// <list type="number">
    ///   <item><c>tmp[seqLen, rank] = matmul_f32(B_scaled, x)</c> via <see cref="MatMulF32Kernel"/>.</item>
    ///   <item><c>delta[seqLen, outputDim] = matmul_f32(A, tmp)</c> via <see cref="MatMulF32Kernel"/>.</item>
    ///   <item><c>deltaSum[seqLen, outputDim] = AddKernel(y, delta)</c> via <see cref="AddKernel"/>.</item>
    ///   <item><c>vkCmdCopyBuffer(deltaSum -> y)</c>.</item>
    /// </list>
    /// </para>
    /// <para>
    /// The <c>scale = alpha / rank</c> factor is folded into <c>B</c> at
    /// upload time (see <see cref="VulkanLoraAdapter.Upload"/>), so neither
    /// path needs a separate scale parameter.
    /// </para>
    /// </remarks>
    private void MaybeApplyLoraDelta(
        nint cmdBuf, int layer, string projName,
        VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int seqLen, int inputDim, int outputDim)
    {
        var lora = _currentLora;
        if (lora is null) return;
        var lb = lora.Get(layer, projName);
        if (lb is not { } w) return;

        if (w.InputDim != inputDim || w.OutputDim != outputDim)
            throw new InvalidOperationException(
                $"LoRA adapter '{lora.Source.Name}' layer={layer} proj='{projName}' shape "
                + $"({w.InputDim}x{w.OutputDim}) does not match base projection ({inputDim}x{outputDim}).");

        var tmp = _state.LoraTmp ?? throw new InvalidOperationException(
            "LoraTmp scratch is null — EnsureLoraScratch was not called before a LoRA-active Forward.");

        // Fused fast path: two dispatches (B-reduce + A-accumulate-in-place)
        // in place of the original four. Gated by SPV availability + rank cap.
        if (_loraDeltaGemvFused is not null && w.Rank <= LoraDeltaGemvFusedF32Kernel.MaxRank
            && Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_FUSED_LORA_DELTA") != "1")
        {
            _loraDeltaGemvFused.Record(cmdBuf, x, w.B, w.A, y, tmp,
                seqLen: seqLen, inputDim: inputDim, outputDim: outputDim, rank: w.Rank);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            return;
        }

        var delta = _state.LoraDelta ?? throw new InvalidOperationException("LoraDelta scratch is null.");
        var deltaSum = _state.LoraDeltaSum ?? throw new InvalidOperationException("LoraDeltaSum scratch is null.");

        _matmul.Record(cmdBuf, w.B, x, tmp, m: w.Rank, k: inputDim, n: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _matmul.Record(cmdBuf, w.A, tmp, delta, m: outputDim, k: w.Rank, n: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _add.Record(cmdBuf, y, delta, deltaSum, seqLen * outputDim);
        KernelSupport.ComputeToTransferBarrier(cmdBuf);

        var region = new VkBufferCopy
        {
            srcOffset = 0,
            dstOffset = 0,
            size = (ulong)((long)seqLen * outputDim * sizeof(float)),
        };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, deltaSum.Handle, y.Handle, 1, region);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
    }

    /// <summary>
    /// One projection in a same-input MMVQ group: the Q8_0 weight blob, its
    /// device-side quant type, the output buffer, and the output dimension.
    /// </summary>
    private readonly record struct MmvqGroupProjection(
        VulkanDevice.Buffer Weights, QuantType WeightQt,
        VulkanDevice.Buffer Output, int OutputDim);

    /// <summary>
    /// Records a group of decode (<c>seqLen==1</c>) projections that all read the
    /// same activation <paramref name="input"/> (e.g. Q/K/V over the post-attn-norm
    /// hidden state, or gate/up over the post-ffn-norm hidden state), sharing one
    /// Q8_1 activation-quant across the group's MMVQ GEMVs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When the MMVQ decode path is wired, sharing is enabled
    /// (<see cref="_mmvqShareActivation"/>), and every projection is Q8_0 with a
    /// qualifying shape, the activation is quantized to Q8_1 once into the shared
    /// <c>_state.Q8_1Xq</c>/<c>_state.Q8_1Xds</c> scratch, then each projection's
    /// integer-dot GEMV runs against that one quantized copy. Recording is:
    /// <c>quantize → barrier → GEMV_0, GEMV_1, …</c> with NO barrier between the
    /// GEMVs (they only read the shared scratch; their outputs are disjoint). The
    /// caller is responsible for the barrier that precedes the next dispatch which
    /// WRITES the shared scratch (e.g. the o-proj / down-proj quantize), exactly
    /// as for a standalone <see cref="RecordMatmul"/>.
    /// </para>
    /// <para>
    /// Results are bit-identical to the per-projection path
    /// (<c>DOTLLM_VULKAN_MMVQ_NO_SHARE=1</c>): the same quantized activation feeds
    /// every GEMV, so sharing only removes redundant quantize dispatches. When the
    /// group does not qualify (sharing off, a non-Q8_0 member, an unsupported
    /// shape, or scratch too small) each projection falls back to a standalone
    /// <see cref="RecordMatmul"/> with a separating barrier — identical semantics
    /// to recording them individually.
    /// </para>
    /// </remarks>
    private void RecordSharedInputMmvqGroup(
        nint cmdBuf, VulkanDevice.Buffer input, int inputDim, int seqLen,
        ReadOnlySpan<MmvqGroupProjection> projections)
    {
        if (CanShareMmvqQuant(inputDim, seqLen, projections))
        {
            // One activation quant shared by every projection in the group.
            _quantizeQ8_1!.Record(cmdBuf, input, _state.Q8_1Xq!, _state.Q8_1Xds!, inputDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            foreach (var p in projections)
            {
                // No inter-GEMV barrier: all GEMVs only read the shared scratch
                // and write disjoint outputs (read-after-write on the scratch is
                // ordered by the single barrier above).
                _matmulQ8Mmvq!.Record(cmdBuf, p.Weights, _state.Q8_1Xq!, _state.Q8_1Xds!, p.Output,
                    m: p.OutputDim, k: inputDim);
            }
            return;
        }

        // Fallback: dispatch each projection independently, with a barrier between
        // them, at the caller's real seqLen. RecordMatmul routes each to the
        // appropriate kernel (decode MMVQ / prefill MMQ / coopmat / scalar). The
        // barrier orders each projection's activation-quant after the prior GEMV's
        // shared-scratch read (WAR on _state.Q8_1Xq); non-shared paths are
        // order-independent anyway.
        for (int i = 0; i < projections.Length; i++)
        {
            if (i > 0) KernelSupport.ComputeToComputeBarrier(cmdBuf);
            var p = projections[i];
            RecordMatmul(cmdBuf, p.Weights, p.WeightQt, input, p.Output,
                p.OutputDim, inputDim, seqLen);
        }
    }

    /// <summary>
    /// Returns true when every projection in <paramref name="projections"/>
    /// qualifies for the shared MMVQ activation-quant: decode step
    /// (<paramref name="seqLen"/>==1), sharing enabled, the MMVQ decode path
    /// wired, the shared scratch present and large enough for
    /// <paramref name="inputDim"/>, and every member is Q8_0 with a 32-aligned
    /// input. The group's input dim is identical for all members (they read the
    /// same buffer) — asserted via the shared-scratch size check on
    /// <paramref name="inputDim"/>. Prefill (seqLen&gt;1) never shares: the
    /// quantize_q8_1 / MMVQ kernels are single-row decode kernels, so each
    /// projection must fall back to the row-wise MMQ / GEMM path.
    /// </summary>
    private bool CanShareMmvqQuant(int inputDim, int seqLen, ReadOnlySpan<MmvqGroupProjection> projections)
    {
        if (seqLen != 1) return false;
        if (!_mmvqShareActivation || projections.Length < 2) return false;
        if (_matmulQ8Mmvq is null || _quantizeQ8_1 is null
            || _state.Q8_1Xq is null || _state.Q8_1Xds is null)
            return false;
        if ((inputDim % QuantizeQ8_1Kernel.GroupSize) != 0) return false;
        if (QuantizeQ8_1Kernel.PackedBytes(inputDim) > _state.Q8_1Xq.Size) return false;
        if (QuantizeQ8_1Kernel.ScaleBytes(inputDim) > _state.Q8_1Xds.Size) return false;
        foreach (var p in projections)
            if (p.WeightQt != QuantType.Q8_0) return false;
        return true;
    }

    private void RecordMatmul(
        nint cmdBuf,
        VulkanDevice.Buffer weights, QuantType weightQt,
        VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int outputDim, int inputDim, int seqLen)
    {
        if (weightQt == QuantType.Q8_0)
        {
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #46): quantize the F32
                // activation row to Q8_1, then run the integer-dot GEMV.
                // Falls back to the F32-in GEMV when the path isn't wired
                // (no integer-dot support / SPV missing / env opt-out) or the
                // shapes don't qualify.
                if (_matmulQ8Mmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % QuantizeQ8_1Kernel.GroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulQ8Mmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulQ8.Record(cmdBuf, weights, input, output,
                        m: outputDim, k: inputDim);
                }
            }
            else if (_matmulQ8Mmq is not null && _quantizeQ8_1Rows is not null
                && _state.Q8_1XqRows is not null && _state.Q8_1XdsRows is not null
                && (inputDim % QuantizeQ8_1RowsKernel.GroupSize) == 0
                && QuantizeQ8_1RowsKernel.PackedBytes(seqLen, inputDim) <= _state.Q8_1XqRows.Size
                && QuantizeQ8_1RowsKernel.ScaleBytes(seqLen, inputDim) <= _state.Q8_1XdsRows.Size)
            {
                // dp4a MMQ prefill path (issue #50): quantize the F32 activation
                // B-matrix to Q8_1 row-wise, then run the integer-dot GEMM. The
                // compute-bound seqLen>1 analogue of the MMVQ decode GEMV; the
                // integer-dot inner loop replaces the per-element dequant FMA of
                // the FP GEMM. Falls through to coopmat / scalar when the path
                // isn't wired or the activation scratch can't hold [N, K].
                _quantizeQ8_1Rows.Record(cmdBuf, input, _state.Q8_1XqRows, _state.Q8_1XdsRows,
                    n: seqLen, k: inputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                _matmulQ8Mmq.Record(cmdBuf, weights, _state.Q8_1XqRows, _state.Q8_1XdsRows, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else if (_matmulQ8GemmCoopmat is not null)
            {
                // Prefill path on coopmat-capable devices — ~3.8× over scalar
                // at Llama-3 prefill shapes. See MatMulQ8_0GemmCoopmatKernel.
                _matmulQ8GemmCoopmat.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else
            {
                _matmulQ8Gemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantType.Q2_K)
        {
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #339); F32-in fallback otherwise.
                if (_matmulQ2KMmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % MatMulQ2KMmvqKernel.Q2KGroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulQ2KMmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulQ2K.Record(cmdBuf, weights, input, output,
                        m: outputDim, k: inputDim);
                }
            }
            else if (_matmulQ2KMmq is not null && _quantizeQ8_1Rows is not null
                && _state.Q8_1XqRows is not null && _state.Q8_1XdsRows is not null
                && (inputDim % MatMulQ2KMmqKernel.Q2KGroupSize) == 0
                && QuantizeQ8_1RowsKernel.PackedBytes(seqLen, inputDim) <= _state.Q8_1XqRows.Size
                && QuantizeQ8_1RowsKernel.ScaleBytes(seqLen, inputDim) <= _state.Q8_1XdsRows.Size)
            {
                // dp4a MMQ prefill path (issue #344).
                _quantizeQ8_1Rows.Record(cmdBuf, input, _state.Q8_1XqRows, _state.Q8_1XdsRows,
                    n: seqLen, k: inputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                _matmulQ2KMmq.Record(cmdBuf, weights, _state.Q8_1XqRows, _state.Q8_1XdsRows, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else
            {
                _matmulQ2KGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantType.Q3_K)
        {
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #339); F32-in fallback otherwise.
                if (_matmulQ3KMmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % MatMulQ3KMmvqKernel.Q3KGroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulQ3KMmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulQ3K.Record(cmdBuf, weights, input, output,
                        m: outputDim, k: inputDim);
                }
            }
            else if (_matmulQ3KMmq is not null && _quantizeQ8_1Rows is not null
                && _state.Q8_1XqRows is not null && _state.Q8_1XdsRows is not null
                && (inputDim % MatMulQ3KMmqKernel.Q3KGroupSize) == 0
                && QuantizeQ8_1RowsKernel.PackedBytes(seqLen, inputDim) <= _state.Q8_1XqRows.Size
                && QuantizeQ8_1RowsKernel.ScaleBytes(seqLen, inputDim) <= _state.Q8_1XdsRows.Size)
            {
                // dp4a MMQ prefill path (issue #344).
                _quantizeQ8_1Rows.Record(cmdBuf, input, _state.Q8_1XqRows, _state.Q8_1XdsRows,
                    n: seqLen, k: inputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                _matmulQ3KMmq.Record(cmdBuf, weights, _state.Q8_1XqRows, _state.Q8_1XdsRows, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else
            {
                _matmulQ3KGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantType.Q4_K)
        {
            // Q4_K_M decode-path GEMV (seqLen==1) or prefill-path tiled GEMM.
            // No coopmat variant in Phase 1 — follow-up ticket.
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #52): quantize the F32 activation
                // row to Q8_1, then run the integer-dot Q4_K GEMV. Reuses the
                // same Q8_1 activation scratch as the Q8_0 MMVQ path (the
                // activation quant is weight-format-independent). Falls back to
                // the F32-in Q4_K GEMV when the path isn't wired (no integer-dot
                // support / SPV missing / env opt-out) or the shapes don't
                // qualify (Q4_K needs inputDim % 256 == 0).
                if (_matmulQ4KMmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % MatMulQ4KMmvqKernel.Q4KGroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulQ4KMmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulQ4K.Record(cmdBuf, weights, input, output,
                        m: outputDim, k: inputDim);
                }
            }
            else if (_matmulQ4KMmq is not null && _quantizeQ8_1Rows is not null
                && _state.Q8_1XqRows is not null && _state.Q8_1XdsRows is not null
                && (inputDim % MatMulQ4KMmqKernel.Q4KGroupSize) == 0
                && QuantizeQ8_1RowsKernel.PackedBytes(seqLen, inputDim) <= _state.Q8_1XqRows.Size
                && QuantizeQ8_1RowsKernel.ScaleBytes(seqLen, inputDim) <= _state.Q8_1XdsRows.Size)
            {
                // dp4a MMQ prefill path (issue #340): quantize the F32 activation
                // B-matrix to Q8_1 row-wise, then run the integer-dot Q4_K GEMM.
                // The compute-bound seqLen>1 analogue of the Q4_K MMVQ decode GEMV;
                // replaces the dequant→FP GEMM. Falls through to the F32-in GEMM
                // when not wired or the activation scratch can't hold [N, K].
                _quantizeQ8_1Rows.Record(cmdBuf, input, _state.Q8_1XqRows, _state.Q8_1XdsRows,
                    n: seqLen, k: inputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                _matmulQ4KMmq.Record(cmdBuf, weights, _state.Q8_1XqRows, _state.Q8_1XdsRows, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else
            {
                _matmulQ4KGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantType.Q5_K)
        {
            // Q5_K_M decode-path GEMV (seqLen==1) or prefill-path tiled GEMM.
            // Same dispatch shape as Q4_K — same alignment requirement
            // (inputDim % 256 == 0, enforced by the upload path).
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #338) — sibling of Q4_K/Q6_K;
                // falls back to the F32-in Q5_K GEMV when not wired or unaligned.
                if (_matmulQ5KMmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % MatMulQ5KMmvqKernel.Q5KGroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulQ5KMmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulQ5K.Record(cmdBuf, weights, input, output,
                        m: outputDim, k: inputDim);
                }
            }
            else if (_matmulQ5KMmq is not null && _quantizeQ8_1Rows is not null
                && _state.Q8_1XqRows is not null && _state.Q8_1XdsRows is not null
                && (inputDim % MatMulQ5KMmqKernel.Q5KGroupSize) == 0
                && QuantizeQ8_1RowsKernel.PackedBytes(seqLen, inputDim) <= _state.Q8_1XqRows.Size
                && QuantizeQ8_1RowsKernel.ScaleBytes(seqLen, inputDim) <= _state.Q8_1XdsRows.Size)
            {
                // dp4a MMQ prefill path (issue #342): sibling of the Q4_K MMQ +
                // the qh 5th bit. Quantize the F32 activation B-matrix to Q8_1
                // row-wise, then run the integer-dot Q5_K GEMM.
                _quantizeQ8_1Rows.Record(cmdBuf, input, _state.Q8_1XqRows, _state.Q8_1XdsRows,
                    n: seqLen, k: inputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                _matmulQ5KMmq.Record(cmdBuf, weights, _state.Q8_1XqRows, _state.Q8_1XdsRows, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else
            {
                _matmulQ5KGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantType.Q6_K)
        {
            // Q6_K_M decode-path GEMV (seqLen==1) or prefill-path tiled GEMM.
            // Same dispatch shape as Q4_K / Q5_K — same alignment requirement
            // (inputDim % 256 == 0, enforced by the upload path). No coopmat
            // variant in Phase 1 — follow-up ticket sibling of the Q4_K /
            // Q5_K coopmat work.
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #338): quantize the F32 activation
                // row to Q8_1, then run the integer-dot Q6_K GEMV. Reuses the same
                // Q8_1 activation scratch as the Q8_0 / Q4_K MMVQ paths. Falls back
                // to the F32-in Q6_K GEMV when not wired (no integer-dot / SPV
                // missing) or the shapes don't qualify (Q6_K needs inputDim % 256 == 0).
                if (_matmulQ6KMmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % MatMulQ6KMmvqKernel.Q6KGroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulQ6KMmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulQ6K.Record(cmdBuf, weights, input, output,
                        m: outputDim, k: inputDim);
                }
            }
            else if (_matmulQ6KMmq is not null && _quantizeQ8_1Rows is not null
                && _state.Q8_1XqRows is not null && _state.Q8_1XdsRows is not null
                && (inputDim % MatMulQ6KMmqKernel.Q6KGroupSize) == 0
                && QuantizeQ8_1RowsKernel.PackedBytes(seqLen, inputDim) <= _state.Q8_1XqRows.Size
                && QuantizeQ8_1RowsKernel.ScaleBytes(seqLen, inputDim) <= _state.Q8_1XdsRows.Size)
            {
                // dp4a MMQ prefill path (issue #341): quantize the F32 activation
                // B-matrix to Q8_1 row-wise, then run the integer-dot Q6_K GEMM —
                // the seqLen>1 analogue of the Q6_K MMVQ decode GEMV. Completes the
                // Q4_K_M prefill win (#340 left ffn_down/attn_v on dequant→FP).
                _quantizeQ8_1Rows.Record(cmdBuf, input, _state.Q8_1XqRows, _state.Q8_1XdsRows,
                    n: seqLen, k: inputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                _matmulQ6KMmq.Record(cmdBuf, weights, _state.Q8_1XqRows, _state.Q8_1XdsRows, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else
            {
                _matmulQ6KGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantType.IQ4_NL)
        {
            // IQ4_NL: 32-element block alignment (inputDim % 32 == 0, enforced
            // by the upload path's KeepIq4NlOnDevice predicate).
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #339); F32-in fallback otherwise.
                if (_matmulIq4NlMmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % MatMulIq4NlMmvqKernel.Iq4NlGroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulIq4NlMmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulIq4Nl.Record(cmdBuf, weights, input, output,
                        m: outputDim, k: inputDim);
                }
            }
            else if (_matmulIq4NlMmq is not null && _quantizeQ8_1Rows is not null
                && _state.Q8_1XqRows is not null && _state.Q8_1XdsRows is not null
                && (inputDim % MatMulIq4NlMmqKernel.Iq4NlGroupSize) == 0
                && QuantizeQ8_1RowsKernel.PackedBytes(seqLen, inputDim) <= _state.Q8_1XqRows.Size
                && QuantizeQ8_1RowsKernel.ScaleBytes(seqLen, inputDim) <= _state.Q8_1XdsRows.Size)
            {
                // dp4a MMQ prefill path (issue #344).
                _quantizeQ8_1Rows.Record(cmdBuf, input, _state.Q8_1XqRows, _state.Q8_1XdsRows,
                    n: seqLen, k: inputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                _matmulIq4NlMmq.Record(cmdBuf, weights, _state.Q8_1XqRows, _state.Q8_1XdsRows, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else
            {
                _matmulIq4NlGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantType.IQ4_XS)
        {
            // IQ4_XS: 256-element super-block alignment, mirrors Q4_K_M shape.
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #339); F32-in fallback otherwise.
                if (_matmulIq4XsMmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % MatMulIq4XsMmvqKernel.Iq4XsGroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulIq4XsMmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulIq4Xs.Record(cmdBuf, weights, input, output,
                        m: outputDim, k: inputDim);
                }
            }
            else if (_matmulIq4XsMmq is not null && _quantizeQ8_1Rows is not null
                && _state.Q8_1XqRows is not null && _state.Q8_1XdsRows is not null
                && (inputDim % MatMulIq4XsMmqKernel.Iq4XsGroupSize) == 0
                && QuantizeQ8_1RowsKernel.PackedBytes(seqLen, inputDim) <= _state.Q8_1XqRows.Size
                && QuantizeQ8_1RowsKernel.ScaleBytes(seqLen, inputDim) <= _state.Q8_1XdsRows.Size)
            {
                // dp4a MMQ prefill path (issue #343): the first IQ-family prefill
                // MMQ — codebook-decode in shared mem, no min term. Quantize the F32
                // activation B-matrix to Q8_1 row-wise, then run the integer-dot GEMM.
                _quantizeQ8_1Rows.Record(cmdBuf, input, _state.Q8_1XqRows, _state.Q8_1XdsRows,
                    n: seqLen, k: inputDim);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                _matmulIq4XsMmq.Record(cmdBuf, weights, _state.Q8_1XqRows, _state.Q8_1XdsRows, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else
            {
                _matmulIq4XsGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantType.IQ2_XXS)
        {
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #339); F32-in fallback otherwise.
                if (_matmulIq2XxsMmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % MatMulIq2XxsMmvqKernel.Iq2XxsGroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulIq2XxsMmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulIq2Xxs.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                }
            }
            else
                _matmulIq2XxsGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
        }
        else if (weightQt == QuantType.IQ2_XS)
        {
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #339); F32-in fallback otherwise.
                if (_matmulIq2XsMmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % MatMulIq2XsMmvqKernel.Iq2XsGroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulIq2XsMmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulIq2Xs.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                }
            }
            else
                _matmulIq2XsGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
        }
        else if (weightQt == QuantType.IQ2_S)
        {
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #339); F32-in fallback otherwise.
                if (_matmulIq2SMmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % MatMulIq2SMmvqKernel.Iq2SGroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulIq2SMmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulIq2S.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                }
            }
            else
                _matmulIq2SGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
        }
        else if (weightQt == QuantType.IQ3_XXS)
        {
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #339); F32-in fallback otherwise.
                if (_matmulIq3XxsMmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % MatMulIq3XxsMmvqKernel.Iq3XxsGroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulIq3XxsMmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulIq3Xxs.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                }
            }
            else
                _matmulIq3XxsGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
        }
        else if (weightQt == QuantType.IQ3_S)
        {
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #339); F32-in fallback otherwise.
                if (_matmulIq3SMmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % MatMulIq3SMmvqKernel.Iq3SGroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulIq3SMmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulIq3S.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim);
                }
            }
            else
                _matmulIq3SGemm.Record(cmdBuf, weights, input, output, m: outputDim, k: inputDim, n: seqLen);
        }
        else if (weightQt == QuantType.I2_S)
        {
            // I2_S (BitNet b1.58 ternary): 128-element block alignment (enforced by
            // KeepI2SOnDevice). Decode via GEMV, prefill via 16x16-tiled GEMM. The
            // per-tensor scale lives at the weight-buffer tail; both kernels read it.
            if (seqLen == 1)
            {
                _matmulI2S.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim);
            }
            else
            {
                _matmulI2SGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantType.IQ1_S)
        {
            // IQ1_S: 256-element super-block alignment, ~1.5-1.7 bpw smallest GGUF quant.
            if (seqLen == 1)
            {
                // dp4a MMVQ decode path (issue #339); F32-in fallback otherwise.
                if (_matmulIq1SMmvq is not null && _quantizeQ8_1 is not null
                    && _state.Q8_1Xq is not null && _state.Q8_1Xds is not null
                    && (inputDim % MatMulIq1SMmvqKernel.Iq1SGroupSize) == 0
                    && QuantizeQ8_1Kernel.PackedBytes(inputDim) <= _state.Q8_1Xq.Size
                    && QuantizeQ8_1Kernel.ScaleBytes(inputDim) <= _state.Q8_1Xds.Size)
                {
                    _quantizeQ8_1.Record(cmdBuf, input, _state.Q8_1Xq, _state.Q8_1Xds, inputDim);
                    KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    _matmulIq1SMmvq.Record(cmdBuf, weights, _state.Q8_1Xq, _state.Q8_1Xds, output,
                        m: outputDim, k: inputDim);
                }
                else
                {
                    _matmulIq1S.Record(cmdBuf, weights, input, output,
                        m: outputDim, k: inputDim);
                }
            }
            else
            {
                _matmulIq1SGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantType.F16)
        {
            // Phase 8: native F16 weights stay 2 bytes / element on device.
            // Decode (seqLen==1) -> GEMV; prefill -> coopmat GEMM when
            // available, scalar tiled GEMM otherwise. Alignment: inputDim %
            // 32 == 0 for the GEMM path (K-chunk = 32); inputDim % 2 == 0 for
            // the GEMV. Both enforced by the upload path's KeepF16OnDevice
            // predicate (which itself requires inputDim % 2 == 0; layers with
            // inputDim % 32 != 0 are decode-only and stay on the GEMV).
            if (seqLen == 1)
            {
                _matmulF16.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim);
            }
            else if (_matmulF16GemmCoopmat is not null)
            {
                _matmulF16GemmCoopmat.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else
            {
                _matmulF16Gemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantType.BF16)
        {
            // Phase 8: native BF16 weights stay 2 bytes / element on device.
            // No coopmat path for BF16 (KHR_cooperative_matrix exposes F16 /
            // Sint8 operands on mainstream drivers, not BF16) — decode goes
            // through the scalar GEMV, prefill through the scalar tiled GEMM.
            if (seqLen == 1)
            {
                _matmulBf16.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim);
            }
            else
            {
                _matmulBf16Gemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else
        {
            _matmul.Record(cmdBuf, weights, input, output,
                outputDim, inputDim, seqLen);
        }
    }

    /// <summary>
    /// Records a single-region device-to-device <c>vkCmdCopyBuffer</c>. The
    /// residual-shuffle copies that used to run here were eliminated via
    /// hidden-slot rotation (<see cref="VulkanForwardState.RotateHiddenSlot"/>).
    /// Remaining callers: the last-row extraction before the LM-head RMSNorm
    /// (offset copy of one row), the embedding gather, and the KV-cache
    /// update — none of which can be turned into a label swap.
    /// </summary>
    private static void RecordCopyBufferRange(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst,
        ulong srcOffset, ulong dstOffset, ulong size)
    {
        var region = new VkBufferCopy { srcOffset = srcOffset, dstOffset = dstOffset, size = size };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, src.Handle, dst.Handle, 1, region);
    }

    /// <summary>
    /// Validates every token id is in range <c>[0, vocabSize)</c>. Separated
    /// from <see cref="RecordEmbeddingGather"/> so the check happens before
    /// we begin recording the command buffer — a bad id throws cleanly
    /// without leaving the submit context half-written.
    /// </summary>
    private void ValidateTokenIds(ReadOnlySpan<int> tokenIds)
    {
        int vocab = Config.VocabSize;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int id = tokenIds[t];
            if ((uint)id >= (uint)vocab)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {id} is out of range");
        }
    }

    /// <summary>
    /// Records N device-local <c>vkCmdCopyBuffer</c> calls (one per input
    /// token) that gather per-token rows from the already-resident
    /// <see cref="VulkanWeights.TokenEmbedding"/> buffer into
    /// <see cref="VulkanForwardState.HiddenState"/>. The embedding table was
    /// dequantised to F32 and uploaded to device-local VRAM at construction
    /// time (see <see cref="VulkanWeights.Upload"/>), so the only
    /// per-forward cost here is <c>seqLen</c> cheap on-device copy commands
    /// — no host-mapped write, no host→device transfer bandwidth.
    /// </summary>
    /// <remarks>
    /// Vulkan's <c>vkCmdCopyBuffer</c> does accept a regions array, but the
    /// current P/Invoke surface takes a single region (matching the
    /// KV-cache-update path in <see cref="VulkanKvCache.RecordUpdate"/>).
    /// For <c>seqLen=1</c> decode this is one call; for prefill it's
    /// <c>promptLen</c> calls, still dwarfed by the per-layer matmul cost.
    /// </remarks>
    private void RecordEmbeddingGather(nint cmdBuf, ReadOnlySpan<int> tokenIds)
    {
        int hiddenSize = Config.HiddenSize;
        long rowBytes = (long)hiddenSize * sizeof(float);
        var srcBuf = _weights.TokenEmbedding.Handle;
        var dstBuf = _state.HiddenState.Handle;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int id = tokenIds[t];
            var region = new VkBufferCopy
            {
                srcOffset = (ulong)((long)id * rowBytes),
                dstOffset = (ulong)((long)t * rowBytes),
                size = (ulong)rowBytes,
            };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, srcBuf, dstBuf, 1, region);
        }
    }

    private unsafe void UploadPositions(ReadOnlySpan<int> positions)
    {
        // The Allocate in EnsureCapacity already sized PositionsBuffer for seqLen;
        // delegate the mapped copy to device.Upload via a raw byte span.
        var posBytes = MemoryMarshal.AsBytes(positions);
        _device.Upload(posBytes, _state.PositionsBuffer);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        // Drop the device-side LoRA cache before tearing down the device —
        // each VulkanLoraAdapter owns VkBuffers that must be freed before
        // the device is disposed.
        _loraCache.Dispose();

        // ForwardBatch scratch — null when batched path was never invoked.
        _batchScratch?.Dispose();

        _diffusionLogits?.Dispose();
        _diffusionScSig?.Dispose();
        _pkvStore?.Dispose();
        _pkvKConcat?.Dispose();
        _pkvVConcat?.Dispose();

        _submit.Dispose();
        _state.Dispose();
        _weights.Dispose();

        _loraDeltaGemvFused?.Dispose();
        _moeSigmoidGatedAdd?.Dispose();
        _moeBroadcast?.Dispose();
        _moeWeightedScatter?.Dispose();
        _moeUngroupScatter?.Dispose();
        _moeGroupedMatmulF16Coopmat?.Dispose();
        _moeExpandGroupByExpert?.Dispose();
        _moeExpertOffsets?.Dispose();
        _moeIndexedLoraDelta?.Dispose();
        _moeIndexedMatmulTiled?.Dispose();
        _moeIndexedMatmulQ5_1?.Dispose();
        _moeIndexedMatmulQ4K?.Dispose();
        _moeIndexedMatmulQ8?.Dispose();
        _moeIndexedMatmul?.Dispose();
        _moeTopkSoftmax?.Dispose();
        _mlaKvSplit?.Dispose();
        _mlaRope?.Dispose();
        _mlaAttention?.Dispose();
        _biasAdd.Dispose();
        _add.Dispose();
        _swiglu.Dispose();
        _geglu?.Dispose();
        _relu2glu?.Dispose();
        _embedScale?.Dispose();
        _gemma4OnesVec?.Dispose();
        _flashAttention?.Dispose();
        _attention.Dispose();
        _rope.Dispose();
        _rmsnorm.Dispose();
        _rmsnormMatmulQ8Fused?.Dispose();
        _matmulBf16Gemm.Dispose();
        _matmulBf16.Dispose();
        _matmulF16GemmCoopmat?.Dispose();
        _matmulF16Gemm.Dispose();
        _matmulF16.Dispose();
        _matmulIq1SGemm.Dispose();
        _matmulIq1S.Dispose();
        _matmulI2SGemm.Dispose();
        _matmulI2S.Dispose();
        _matmulIq4XsGemm.Dispose();
        _matmulIq4Xs.Dispose();
        _matmulIq4NlGemm.Dispose();
        _matmulIq4Nl.Dispose();
        _matmulIq2SGemm.Dispose();
        _matmulIq2S.Dispose();
        _matmulIq2XsGemm.Dispose();
        _matmulIq2Xs.Dispose();
        _matmulIq2XxsGemm.Dispose();
        _matmulIq2Xxs.Dispose();
        _iq2Codebooks.Dispose();
        _matmulIq3SGemm.Dispose();
        _matmulIq3S.Dispose();
        _matmulIq3XxsGemm.Dispose();
        _matmulIq3Xxs.Dispose();
        _iq3Codebooks.Dispose();
        _matmulQ6KGemm.Dispose();
        _matmulQ6K.Dispose();
        _matmulQ5KGemm.Dispose();
        _matmulQ5K.Dispose();
        _matmulQ4KGemm.Dispose();
        _matmulQ4K.Dispose();
        _matmulQ3KGemm.Dispose();
        _matmulQ3K.Dispose();
        _matmulQ2KGemm.Dispose();
        _matmulQ2K.Dispose();
        _matmulQ8Mmq?.Dispose();
        _matmulQ4KMmq?.Dispose();
        _matmulQ6KMmq?.Dispose();
        _matmulQ5KMmq?.Dispose();
        _matmulIq4XsMmq?.Dispose();
        _matmulIq4NlMmq?.Dispose();
        _matmulQ2KMmq?.Dispose();
        _matmulQ3KMmq?.Dispose();
        _quantizeQ8_1Rows?.Dispose();
        _matmulQ4KMmvq?.Dispose();
        _matmulQ6KMmvq?.Dispose();
        _matmulQ5KMmvq?.Dispose();
        _matmulQ2KMmvq?.Dispose();
        _matmulQ3KMmvq?.Dispose();
        _matmulIq4NlMmvq?.Dispose();
        _matmulIq4XsMmvq?.Dispose();
        _matmulIq2XxsMmvq?.Dispose();
        _matmulIq2XsMmvq?.Dispose();
        _matmulIq2SMmvq?.Dispose();
        _matmulIq3XxsMmvq?.Dispose();
        _matmulIq3SMmvq?.Dispose();
        _matmulIq1SMmvq?.Dispose();
        _matmulQ8Mmvq?.Dispose();
        _quantizeQ8_1?.Dispose();
        _matmulQ8GemmCoopmat?.Dispose();
        _matmulQ8Gemm.Dispose();
        _matmulQ8.Dispose();
        _matmul.Dispose();

        _cpuWeights.Dispose();
        if (_ownsDevice)
            _device.Dispose();
    }
}
