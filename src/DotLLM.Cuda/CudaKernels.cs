using DotLLM.Core.Configuration;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Loads all PTX kernel modules and provides typed launch methods for each kernel.
/// Arguments are marshaled as pointer arrays for <see cref="CudaDriverApi.cuLaunchKernel"/>.
/// </summary>
public sealed unsafe class CudaKernels : IDisposable
{
    private const int BlockSize = 256;

    /// <summary>
    /// I2_S GEMV output rows processed per block (warp-per-row scheme): the block's
    /// <see cref="BlockSize"/>/32 = 8 warps each own <c>I2S_ROWS_PER_WARP</c> rows. Grid is sized
    /// ceil(n / this). Must stay in sync with <c>I2S_ROWS_PER_BLOCK</c> (= 8 × I2S_ROWS_PER_WARP)
    /// in native/kernels/i2_s_gemv.cu.
    /// </summary>
    private const int I2sRowsPerBlock = 16;

    /// <summary>
    /// PQ2_0 GEMV output rows processed per block (warp-per-row scheme, mirrors
    /// <see cref="I2sRowsPerBlock"/>): 8 warps each own <c>PQ2_0_ROWS_PER_WARP</c>=2 rows. Grid
    /// is sized ceil(n / this). Must stay in sync with <c>PQ2_0_ROWS_PER_BLOCK</c> in
    /// native/kernels/pq2_0_gemv.cu (ROWS_PER_WARP=4 was tried and measured worse — see that
    /// file's comment).
    /// </summary>
    private const int Pq2_0RowsPerBlock = 16;

    /// <summary>
    /// Upper bound on <c>k</c> for routing to the <c>_small</c> PQ2_0 F16 GEMV kernel variants
    /// (<c>pq2_0_gemv_f16in_small</c>/<c>pq2_0_gemv2_f16in_small</c>) instead of the default
    /// (<c>PQ2_0_MAX_K</c>=17408-sized) ones. Must match <c>PQ2_0_MAX_K_SMALL</c> in
    /// native/kernels/pq2_0_gemv.cu — see that file's "Small-K specialization" header comment for
    /// the occupancy-arithmetic rationale (shrinking the static <c>xs[]</c> shared-memory buffer
    /// raises the shared-mem occupancy limit above the register limit for these smaller-k
    /// launches). Exactly Bonsai-27B's attention/GDN input dim (qwen35.embedding_length); the FFN
    /// call sites (k=17408) always exceed this and stay on the default kernels.
    /// </summary>
    private const int Pq2_0MaxKSmall = 5120;

    /// <summary>
    /// Max CUDA blocks for dequant kernel launches. Kernels use grid-stride loops,
    /// so capping grid size amortizes block launch overhead on GPUs with many SMs
    /// (e.g. RTX 3050 has 20 SMs; launching 65K+ blocks per dequant overwhelms the
    /// hardware block scheduler). Value is ~4x typical consumer SM count.
    /// </summary>
    private const int MaxDequantGridSize = 256;

    private readonly CudaModule _rmsnormModule;
    private readonly CudaModule _ropeModule;
    private readonly CudaModule _swigluModule;
    private readonly CudaModule _addModule;
    private readonly CudaModule _softmaxModule;
    private readonly CudaModule _embeddingModule;
    private readonly CudaModule _attentionModule;
    private readonly CudaModule _kvCacheUpdateModule;
    private readonly CudaModule _biasAddModule;
    private readonly CudaModule _perHeadRmsNormModule;
    private readonly CudaModule _convertModule;
    private readonly CudaModule _dequantModule;
    private readonly CudaModule? _dequantIQuantsModule;
    private readonly CudaModule _quantizedGemvModule;
    private readonly CudaModule _fusedAddRmsNormModule;
    private readonly CudaModule _rmsnormF32InModule;
    private readonly CudaModule _addF32Module;
    private readonly CudaModule _embeddingF32OutModule;
    private readonly CudaModule _ropeF32Module;
    private readonly CudaModule _attentionF32Module;
    private readonly CudaModule _swigluF32Module;
    private readonly CudaModule _biasAddF32Module;
    private readonly CudaModule _perHeadRmsNormF32Module;
    private readonly CudaModule _rmsnormF32Module;
    private readonly CudaModule? _copyRmsNormF32Module;
    private readonly CudaModule _quantizedGemvF32InModule;
    private readonly CudaModule? _quantizedGemvMmqModule;
    private readonly nint _quantizedGemvQ2_KMmqFunc;
    private readonly nint _quantizedGemvQ4_KMmqFunc;
    private readonly nint _quantizedGemvQ5_KMmqFunc;
    private readonly nint _quantizedGemvQ6_KMmqFunc;
    private readonly nint _quantizedGemvIQ4_NLMmqFunc;
    private readonly nint _quantizedGemvIQ4_XSMmqFunc;
    // MMVQ-large variants — 1 row per CUDA block, 128 threads (4 warps).
    // Tuned for k≥1024 (≥4 super-blocks/row); fall back to MMQ-4-rows for smaller k.
    private readonly nint _quantizedGemvQ2_KMmvqLargeFunc;
    private readonly nint _quantizedGemvQ4_KMmvqLargeFunc;
    private readonly nint _quantizedGemvQ4_KMmvqCoalescedFunc;
    private readonly nint _quantizedGemvQ4_KMmvqLlamaCppFunc;
    private readonly nint _quantizedGemvQ5_KMmvqLargeFunc;
    private readonly nint _quantizedGemvQ6_KMmvqLargeFunc;
    private readonly nint _quantizedGemvIQ4_NLMmvqLargeFunc;
    private readonly nint _quantizedGemvIQ4_XSMmvqLargeFunc;
    // Pre-Q8_1 variants. Read INT8/dx/sx2 from device-resident scratch (populated
    // once per fused projection group via _quantizeXToQ8_1Func) instead of
    // re-quantizing the input inside every CUDA block. Eliminates the redundant
    // Stage 1 work that scales with output dim n (n× for MMVQ-large, n/4× for MMQ-4-rows).
    private readonly CudaModule? _quantizeXModule;
    private readonly nint _quantizeXToQ8_1Func;
    private readonly nint _quantizeXToQ8_1BatchedFunc;

    // TurboQuant (MSE-stage) KV codec — optional module (turboquant.ptx). The CUDA port of the
    // Vulkan turboquant_{dequant,encode}_f32 shaders. 0 funcs when the PTX is absent/stale.
    private readonly CudaModule? _turboquantModule;
    private readonly nint _turboquantDequantF32Func;
    private readonly nint _turboquantEncodeF32Func;
    private readonly nint _turboquantDequantF16Func;
    private readonly nint _turboquantEncodeF16Func;
    private readonly nint _quantizedGemvQ2_KMmqPreqFunc;
    private readonly nint _quantizedGemvQ4_KMmqPreqFunc;
    private readonly nint _quantizedGemvQ5_KMmqPreqFunc;
    private readonly nint _quantizedGemvQ6_KMmqPreqFunc;
    private readonly nint _quantizedGemvIQ4_NLMmqPreqFunc;
    private readonly nint _quantizedGemvIQ4_XSMmqPreqFunc;
    private readonly nint _quantizedGemvQ2_KMmvqLargePreqFunc;
    private readonly nint _quantizedGemvQ4_KMmvqLargePreqFunc;
    private readonly nint _quantizedGemvQ5_KMmvqLargePreqFunc;
    private readonly nint _quantizedGemvQ6_KMmvqLargePreqFunc;
    private readonly nint _quantizedGemvIQ4_NLMmvqLargePreqFunc;
    private readonly nint _quantizedGemvIQ4_XSMmvqLargePreqFunc;
    // Batched-M dp4a MMQ prefill kernel (issue #349) — DESIGNED to amortize weight reads
    // across MMQ_BATCH_M_TILE(2) prefill tokens per block via L1/L2 cache reuse (see
    // native/kernels/quantized_gemv_mmq.cu for the kernel body), but measured 2x-51x
    // slower than the dequant->cuBLAS baseline in practice (Task 6 benchmark sweep,
    // RTX 3060) — the 2-token tile is too narrow for real amortization; see issue #367
    // for the redesign. Note MMQ_BATCH_M_TILE (the kernel's per-block tile width) is a
    // different number from MmqBatchedMinSeqLen (the dispatch threshold below, now
    // int.MaxValue) — don't conflate them. Disabled by default via that threshold.
    // Q4_K only for this PoC.
    private readonly nint _quantizedGemvQ4_KMmqBatchedFunc;
    /// <summary>
    /// Device's maximum opt-in dynamic shared-memory bytes per block (queried once at
    /// kernel-load via CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN). The
    /// on-the-fly MMQ kernels are opted into this cap via cuFuncSetAttribute so
    /// arbitrarily large k (up to ~k=53000 on a 100 KB cap) launches succeed without
    /// recompiling. 0 means we couldn't query — fall back to the default 48 KB cap.
    /// </summary>
    private readonly int _maxDynamicSharedBytesOptIn;
    private readonly CudaModule _i2sGemvModule;
    private readonly CudaModule _dequantI2sModule;
    private readonly CudaModule _pq2_0GemvModule;
    private readonly CudaModule _dequantPQ2_0Module;
    private readonly CudaModule _pq2_0RepackModule;
    private readonly CudaModule _relu2Module;
    private readonly CudaModule _relu2F32Module;
    private readonly CudaModule _relu2GluRmsNormModule;
    private readonly CudaModule _fusedAddRmsNormF32ResModule;

    private readonly nint _rmsnormFunc;
    private readonly nint _rmsnormF32Func;
    private readonly nint _copyRmsNormF32Func;
    private readonly nint _quantizedGemvQ8_0F32InFunc;
    private readonly nint _fusedAddRmsNormFunc;
    private readonly nint _rmsnormF32InF16OutFunc;
    private readonly nint _addF32Func;
    private readonly nint _addF32F16Func;
    private readonly nint _embeddingF32OutF32Func;
    private readonly nint _embeddingF32OutF16Func;
    private readonly nint _embeddingF32OutQ8_0Func;
    private readonly nint _ropeF32Func;
    private readonly nint _attentionF32Func;
    private readonly nint _attentionF32SplitKvFunc;
    private int _attnSplitKvMaxCoResidentGrid = -1; // -1 = not yet queried
    private int _attnSplitKvCachedHeadDim = -1;      // headDim the cached query above is valid for
    // Issue #226 spike: fp64-combine variant of attention_f32_split_kv.
    private readonly nint _attentionF32SplitKvHpFunc;
    private int _attnSplitKvHpMaxCoResidentGrid = -1;
    private int _attnSplitKvHpCachedHeadDim = -1;
    // Combined GQA-group + split-KV kernel (issues #197 + #198): grid = (numKvHeads, kv_split).
    private readonly nint _attentionF32GqaSplitKvFunc;
    private int _attnGqaSplitMaxCoResidentGrid = -1; // -1 = not yet queried
    private int _attnGqaSplitCachedHeadDim = -1;      // headDim the cached query above is valid for
    private int _attnGqaSplitCachedGroup = -1;        // group (numHeads/numKvHeads) the cache is valid for
    private readonly nint _swigluF32Func;
    private readonly nint _relu2GluF32Func;
    private readonly nint _biasAddF32Func;
    private readonly nint _perHeadRmsNormF32Func;
    private readonly nint _ropeFunc;
    private readonly nint _swigluFunc;
    private readonly nint _addFunc;
    private readonly nint _softmaxFunc;
    private readonly nint _embeddingF32Func;
    private readonly nint _embeddingF16Func;
    private readonly nint _embeddingQ8_0Func;
    private readonly nint _embeddingQ4_KFunc;
    private readonly nint _embeddingQ5_KFunc;
    private readonly nint _embeddingQ6_KFunc;
    private readonly nint _attentionFunc;
    private readonly nint _attentionPosFunc;
    private readonly nint _kvCacheUpdatePosFunc;
    private readonly nint _biasAddFunc;
    private readonly nint _perHeadRmsNormFunc;
    private readonly nint _convertF16ToF32Func;
    private readonly nint _convertF32ToF16Func;
    private readonly nint _quantizedGemvQ8_0Func;
    private readonly nint _quantizedGemvQ2_KFunc;
    private readonly nint _quantizedGemvQ4_KFunc;
    private readonly nint _quantizedGemvQ5_0Func;
    private readonly nint _quantizedGemvQ5_KFunc;
    private readonly nint _quantizedGemvQ6_KFunc;
    private readonly nint _quantizedGemvIQ4_NLFunc;
    private readonly nint _quantizedGemvIQ4_XSFunc;
    private readonly nint _i2sGemvF16InFunc;
    private readonly nint _i2sGemv2F16InFunc;
    private readonly nint _i2sGemv3F16InFunc;
    private readonly nint _i2sGemvNormF16InFunc;
    private readonly nint _i2sGemvF32InFunc;
    // Batched GEMM (issue #250): decode-row-once-reuse-across-tokens twin of i2_s_gemv_f32in,
    // used by CudaMoeFfn.ForwardBitNetI2S's prefill path (seqLen>1 routes multiple tokens to the
    // same expert). TryGetFunction'd so a stale PTX (not yet recompiled) still loads gracefully —
    // gate any use on HasI2SBatchedGemm.
    private readonly nint _i2sGemmF32InFunc;
    private readonly nint _i2sGemvA8Func;
    private readonly nint _i2sGemvA8DeviceScaleFunc;
    private readonly nint _quantizeF16ToI8AbsMaxFunc;
    private readonly nint _dequantI2sF16Func;
    // Ragged K (k % 128 != 0) — issue #206.
    private readonly nint _i2sGemvF16InRaggedFunc;
    private readonly nint _i2sGemvF32InRaggedFunc;
    private readonly nint _dequantI2sF16RaggedFunc;
    private readonly nint _pq2_0GemvF32InFunc;
    private readonly nint _pq2_0GemvF16InFunc;
    private readonly nint _pq2_0Gemv2F16InFunc;
    private readonly nint _pq2_0GemvF16InSmallFunc;
    private readonly nint _pq2_0Gemv2F16InSmallFunc;
    private readonly nint _pq2_0GemvF32IoFunc;
    private readonly nint _pq2_0Gemv2F32IoFunc;
    private readonly nint _pq2_0GemvF32IoSmallFunc;
    private readonly nint _pq2_0Gemv2F32IoSmallFunc;
    private readonly nint _dequantPQ2_0F16Func;
    private readonly nint _pq2_0RepackSplitF16Func;
    private readonly nint _relu2Func;
    private readonly nint _relu2F32Func;
    private readonly nint _relu2GluRmsNormFunc;
    private readonly nint _fusedAddRmsNormF32ResFunc;
    private readonly nint _copyF16ToF32Func;
    private readonly nint _addF32ResF16Func;
    private readonly nint _rmsNormF32InF16WFunc;
    private readonly nint _dequantQ8_0Func;
    private readonly nint _dequantQ4_0Func;
    private readonly nint _dequantQ4_1Func;
    private readonly nint _dequantQ5_0Func;
    private readonly nint _dequantQ5_1Func;
    private readonly nint _dequantQ2_KFunc;
    private readonly nint _dequantQ3_KFunc;
    private readonly nint _dequantQ4_KFunc;
    private readonly nint _dequantQ5_KFunc;
    private readonly nint _dequantQ5_KF32Func;
    private readonly nint _dequantQ6_KFunc;
    private readonly nint _dequantIQ4_NLFunc;
    private readonly nint _dequantIQ4_XSFunc;
    private readonly nint _dequantIQ4_NLF32Func;
    private readonly nint _dequantIQ4_XSF32Func;

    // IQ2 family (iq2.ptx). Each format ships a F16 dequant, a F32 dequant, and a
    // legacy per-row quantized GEMV. IQ2_S doubles as the on-disk format for the
    // MOSTLY_IQ2_M file-type recipe. MMQ / MMVQ-large / MoE-grouped variants are
    // deferred — prefill falls back to dequant+cuBLAS for IQ2 weights.
    private readonly CudaModule? _iq2Module;
    private readonly nint _dequantIQ2_XXSFunc;
    private readonly nint _dequantIQ2_XSFunc;
    private readonly nint _dequantIQ2_SFunc;
    private readonly nint _dequantIQ2_XXSF32Func;
    private readonly nint _dequantIQ2_XSF32Func;
    private readonly nint _dequantIQ2_SF32Func;
    private readonly nint _quantizedGemvIQ2_XXSFunc;
    private readonly nint _quantizedGemvIQ2_XSFunc;
    private readonly nint _quantizedGemvIQ2_SFunc;

    // IQ3 family (dequant_iq3.ptx) and IQ1_S (dequant_iq1.ptx) — issue #258.
    // Dequant-only: prefill and decode both go through dequant + cuBLAS/dot,
    // matching how the IQ2 family shipped before its GEMVs were added.
    private readonly CudaModule? _dequantIQ3Module;
    private readonly nint _dequantIQ3_XXSFunc;
    private readonly nint _dequantIQ3_SFunc;
    private readonly nint _dequantIQ3_XXSF32Func;
    private readonly nint _dequantIQ3_SF32Func;
    private readonly CudaModule? _dequantIQ1Module;
    private readonly nint _dequantIQ1_SFunc;
    private readonly nint _dequantIQ1_SF32Func;

    // BF16 expansion and MXFP4 dequant (dequant_bf16_mxfp4.ptx) — issue #258.
    // BF16 is a pure bit-shift widening, not a quantization; MXFP4 is the
    // format gpt-oss models ship in exclusively.
    private readonly CudaModule? _dequantBf16Mxfp4Module;
    private readonly nint _dequantBF16Func;
    private readonly nint _dequantBF16F32Func;
    private readonly nint _dequantMXFP4Func;
    private readonly nint _dequantMXFP4F32Func;

    private readonly CudaModule? _quantKvModule;
    private readonly nint _quantKvQ8_0Func;
    private readonly nint _quantKvQ4_0Func;
    // Graph-friendly KV-quant variants: read decode position from a device int
    // and predicate the FP16-row → quantized-row eviction.
    private readonly nint _quantKvQ8_0DynFunc;
    private readonly nint _quantKvQ4_0DynFunc;

    // Graph-friendly attention variant: reads seq_kv / position_offset from
    // 4-byte device buffers. Pointers stay stable across cuGraphLaunch replays;
    // host bumps the underlying ints via cuMemcpyHtoD between launches (~1 µs).
    private readonly nint _attentionDynFunc;
    // Issue #200: direct block-table-read paged decode attention -- reads K/V through a
    // small per-layer array of block base device pointers instead of a contiguous buffer,
    // eliminating CudaPagedKvCache.PrepareAttentionScratch's D2D gather. TryGetFunction so a
    // stale PTX without this symbol still loads gracefully (HasAttentionF16Paged reports
    // false and callers keep using the gather-based path).
    private readonly nint _attentionPagedFunc;
    // Decode-step KV-cache write: dst row is dst_base + posPtr[0] * kv_stride.
    // Replaces a host-side cuMemcpyDtoDAsync where the dst address would be
    // baked into the graph at instantiate time.
    private readonly CudaModule? _kvWriteModule;
    private readonly nint _kvWriteOneF16Func;
    // Graph-friendly KV-quant scratch helpers (live in kv_write.ptx alongside
    // the FP16 ring write so they share device-resident pos_ptr conventions).
    private readonly nint _kvWriteOneF16RingFunc;
    private readonly nint _kvDequantQ8_0DynFunc;
    private readonly nint _kvDequantQ4_0DynFunc;
    private readonly nint _kvWindowToScratchDynFunc;

    // Fused decode-step RoPE + KV-cache write. Replaces three eager launches per
    // layer (rope_f16 + 2× cuMemcpyDtoDAsync) with one. Eager-only; the graph
    // path keeps a separate dyn variant.
    private readonly CudaModule? _fusedRopeKvWriteModule;
    private readonly nint _fusedRopeKvWriteF16Func;
    private readonly nint _fusedRopeKvWriteF16DynFunc;

    // ── MLA (Multi-head Latent Attention) Phase A naive forward kernel ──
    // F32 throughout to match the CPU MlaAttention.Execute oracle byte-for-byte
    // algorithmically. Optional — PTX may not be present on stale builds.
    private readonly CudaModule? _attentionMlaModule;
    private readonly nint _attentionMlaF32Func;

    // #region MLA FP16 — sibling FP16 attention kernel (FP32 softmax accum).
    // Loaded from the same attention_mla.ptx module via TryGetFunction so a
    // stale PTX (F32-only) still loads gracefully and HasMlaAttentionKernelF16
    // reports false.
    private readonly nint _attentionMlaF16Func;
    // #endregion

    // MLA forward-path helpers: per-head split of kv_b expansion, RoPE on
    // the rope sub-dim of Q (per head) and on the shared K_pe, and a
    // (numRows, dim) F32 RMSNorm used by q_a_layernorm / kv_a_layernorm.
    private readonly CudaModule? _mlaHelpersModule;
    private readonly nint _mlaSplitKvBF32Func;
    private readonly nint _mlaRopeQpeF32Func;
    private readonly nint _mlaRopeKpeF32Func;
    private readonly nint _mlaRmsNormF32Func;

    // #region MLA FP16 — sibling FP16 helper kernels (split / RoPE / RMSNorm).
    // Loaded from the same mla_helpers.ptx module via TryGetFunction.
    private readonly nint _mlaSplitKvBF16Func;
    private readonly nint _mlaRopeQpeF16Func;
    private readonly nint _mlaRopeKpeF16Func;
    private readonly nint _mlaRmsNormF16Func;
    // #endregion

    // ── MLA Phase B (latent KV cache + W_UK absorbed attention) ──────────
    // Optional — PTX may not be present on stale builds. Phase B's compact
    // cache is 8-16× smaller than Phase A's expanded form (V2-Lite: 7.22×).
    // The attention kernel reads c_kv directly and outputs into the latent
    // dim; the W_UV expansion happens in a follow-on helper.
    private readonly CudaModule? _attentionMlaLatentModule;
    private readonly nint _attentionMlaLatentF32Func;
    private readonly nint _mlaQAbsorbUkF32Func;
    private readonly nint _mlaVExpandUvF32Func;

    // ── MoE (Mixture-of-Experts) helper kernels (F32) ──
    // Routing softmax + top-k selection, output zero-init, weighted/unweighted
    // axpy accumulators, sigmoid-gate dot product, and per-expert token gather.
    // All optional — TryGetFunction so a stale PTX without the new symbols still
    // loads gracefully (HasMoeKernels reports false and the dispatcher skips MoE).
    private readonly CudaModule? _moeFfnModule;
    private readonly nint _moeSoftmaxTopkF32Func;
    private readonly nint _moeRenormTopkF32Func;
    private readonly nint _moeZeroF32Func;
    private readonly nint _moeAxpyScaledRowF32Func;
    private readonly nint _moeAxpyUnweightedF32Func;
    private readonly nint _moeAxpyScaledPerTokenF32Func;
    private readonly nint _moeSigmoidLogitF32Func;
    private readonly nint _moeGatherTokenRowsF32Func;
    // Issue #246 (BitNet-ternary MoE): additive router bias, applied before softmax/top-k.
    private readonly nint _moeGateBiasAddF32Func;
    // Issue #348 (gpt-oss MoE): OAI-clamped SwiGLU activation.
    private readonly nint _swigluOaiF32Func;

    // ── MoE grouped-GEMV kernels (Phase B — single launch across K_active experts) ──
    // One kernel computes (K_active × M) F16 outputs by walking K_active raw-quant
    // weight pointers + K_active output pointers, sharing a single F16 input row.
    // Reduces dispatch overhead from K_active per-projection launches to 1 per
    // projection. Optional — PTX may be missing on stale builds; HasMoeGroupedGemv
    // reports false and CudaMoeFfn falls back to the per-expert path.
    private readonly CudaModule? _moeGroupedGemvModule;
    private readonly nint _moeGroupedGemvQ2_KFunc;
    private readonly nint _moeGroupedGemvQ4_KFunc;
    private readonly nint _moeGroupedGemvQ5_KFunc;
    private readonly nint _moeGroupedGemvQ6_KFunc;
    private readonly nint _moeGroupedGemvQ8_0Func;
    private readonly nint _moeGroupedGemvIQ4_NLFunc;
    private readonly nint _moeGroupedGemvIQ4_XSFunc;

    // ── Qwen3MoeHybrid recurrence-path kernels (F32) ─────────────────────────
    // Optional kernels backing the Gated DeltaNet (GDN) recurrence and the post-
    // attention sigmoid-gate used by Qwen3MoeHybrid models. Each is a numerically
    // equivalent FP32 port of the CPU reference in DotLLM.Cpu.Kernels
    // (Conv1dCausal, GatedDeltaNetScan) / Qwen3MoeHybridTransformerModel forward
    // body. All call expf/logf — compiled with -fmad=false (see build_ptx.bat
    // NO_FMA list and DotLLM.Cuda.csproj FmadFlag metadata) to disable FMA
    // fusion. Numerical accuracy against the CPU oracle is within ≤1 ULP per
    // element on Ampere+; not strictly bit-equal across all inputs (CUDA's
    // precise expf is not guaranteed bit-identical to MathF.Exp), but matches
    // the CPU oracle's existing host fallback to within negligible tolerance.
    //   • conv1d_causal_f32        — depthwise causal 1-D convolution
    //   • gdn_scan_step_f32        — per-token GDN recurrence step (host loops over seqLen)
    //   • l2_normalize_heads_f32   — in-place per-head L2 normalisation (pre-scan)
    //   • gdn_deinterleave_l2norm_decode_f32 — decode-only (seqLen==1) fusion of
    //     deinterleave_gdn_qkv_f32 + both Q/K l2_normalize_heads_f32 calls
    //   • gdn_decay_f32            — fused softplus + exp for the per-token decay g
    //   • sigmoid_f32              — in-place elementwise sigmoid
    //   • silu_f32                 — in-place elementwise SiLU
    //   • sigmoid_mul_f32          — out[i] *= sigmoid(b[i])
    // PTX may be absent on stale builds; the Has* properties report false and
    // callers fall back to the host-side path or surface an error.
    private readonly CudaModule? _conv1dCausalF32Module;
    private readonly nint _conv1dCausalF32Func;
    private readonly nint _gdnConv1dCausalDecodeF32Func;
    private readonly CudaModule? _gdnScanF32Module;
    private readonly nint _gdnScanStepF32Func;
    private readonly nint _gdnScanStepF32CoopSplit4Func;
    private int _gdnScanCoopSplit4MaxCoResidentGrid = -1; // -1 = not yet queried
    private readonly CudaModule? _l2NormHeadsF32Module;
    private readonly nint _l2NormHeadsF32Func;
    private readonly nint _gdnDeinterleaveL2NormDecodeF32Func;
    private readonly nint _gdnDecayF32Func;
    private readonly nint _gdnDecaySigmoidF32Func;
    private readonly CudaModule? _mamba2ScanF32Module;
    private readonly nint _mamba2ScanF32Func;
    private CudaModule? _groupRmsNormF32Module;
    private nint _groupRmsNormF32Func;
    private CudaModule? _reluSquaredInplaceF32Module;
    private nint _reluSquaredInplaceF32Func;
    private readonly CudaModule? _elementwiseF32Module;
    private readonly nint _sigmoidF32Func;
    private readonly nint _siluF32Func;
    private readonly nint _sigmoidMulF32Func;
    private readonly nint _deinterleaveQGateF32Func;
    private readonly nint _deinterleaveGdnQkvF32Func;
    private readonly nint _dequantQ6_KF32Func;

    // ── Gemma-4 (DiffusionGemma AR) F32 helper kernels ───────────────────────
    // Cover the gemma4 MoE forward ops absent from the generic F32 catalog:
    //   • geglu_tanh_f32         — tanh-approx GELU GeGLU (dense + experts)
    //   • rope_f32_partial_neox  — partial NeoX rope (pair (i, i+head_dim/2), full-head freq base)
    //   • scale_inplace_f32      — in-place scalar multiply (layer_output_scale)
    //   • rmsnorm_weightless_f32 — per-row RMSNorm with unit gamma (weight-less V-norm)
    //   • softcap_inplace_f32    — final-logit soft-capping c*tanh(x/c)
    // Optional — PTX may be absent on stale builds; HasGemma4Kernels reports false
    // and the gemma4 forward path surfaces a clear error instead of an NRE.
    private readonly CudaModule? _gemma4F32Module;
    private readonly nint _gegluTanhF32Func;
    private readonly nint _ropeF32PartialNeoxFunc;
    private readonly nint _scaleInplaceF32Func;
    private readonly nint _rmsnormWeightlessF32Func;
    private readonly nint _softcapInplaceF32Func;
    private readonly nint _moeRenormTopkClampedF32Func;
    private readonly nint _quantizeActQ8_0RoundtripF32Func;

    // Causal softmax over the column-major [s x s] FP16 score buffer produced by the
    // cuBLAS tensor-core prefill-attention path (QK GEMM -> THIS -> P*V GEMM).
    // Optional — PTX may be absent on stale builds; HasAttentionSoftmaxCausal reports
    // false and callers must keep the fused attention_f16 path. Required by a not-yet-
    // built module would throw across the whole shared-CudaKernels GPU test suite.
    private readonly CudaModule? _attentionSoftmaxCausalModule;
    private readonly nint _attentionSoftmaxCausalFunc;
    // Coalesced sibling: one thread per softmax row (consecutive threads → consecutive
    // query rows → consecutive addresses), avoiding the per-block strided-read penalty.
    private readonly nint _attentionSoftmaxCausalCoalescedFunc;
    // FP32-scores variant: reads FP32 QK scores, writes FP16 probs to a separate buffer.
    // Removes the dominant precision loss of rounding pre-softmax scores to FP16 (G3 e2e).
    private readonly nint _attentionSoftmaxCausalCoalescedF32InFunc;

    // Hand-fused FP16 tensor-core (mma.sync) flash-attention prefill kernel — keeps the
    // s x s scores in shared/registers, never materialising them to global memory.
    // Prototype for the long-context fused-kernel go/no-go. Optional PTX.
    private readonly CudaModule? _attentionFlashMmaModule;
    private readonly nint _attentionFlashMmaFunc;

    // Tensor-core (mma.sync) FP16 DECODE attention composed with the #197/#198 GQA-group +
    // split-KV grid design (issue #199 v2 — see native/kernels/attention_flash_mma_decode_gqa_split.cu).
    // v1 (single warp/block, grid=numHeads) was 4-5x slower than the F32 baseline at every
    // realistic depth due to ~4% occupancy; this kernel packs the GQA group into the mma
    // tile's M dimension (free — same instruction count as v1, up to 8x more useful
    // throughput) and grids (numKvHeads, kvSplit) like attention_f32_gqa_split_kv. Optional
    // PTX, Ampere-only (mma.sync is sm_80+).
    private readonly CudaModule? _attentionMmaDecodeGqaSplitModule;
    private readonly nint _attentionMmaDecodeGqaSplitFunc;
    private int _attnMmaDecodeGqaSplitMaxCoResidentGrid = -1; // -1 = not yet queried
    private int _attnMmaDecodeGqaSplitCachedGroup = -1;        // group the cached query above is valid for


    /// <summary>
    /// Loads all PTX modules from the specified directory.
    /// </summary>
    /// <param name="ptxDir">Directory containing compiled .ptx files.</param>
    public CudaKernels(string ptxDir)
    {
        _rmsnormModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "rmsnorm.ptx"));
        _ropeModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "rope.ptx"));
        _swigluModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "swiglu.ptx"));
        _addModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "add.ptx"));
        _softmaxModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "softmax.ptx"));
        _embeddingModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "embedding.ptx"));
        _attentionModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "attention.ptx"));
        _kvCacheUpdateModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "kv_cache_update.ptx"));
        _biasAddModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "bias_add.ptx"));
        _perHeadRmsNormModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "per_head_rmsnorm.ptx"));
        _convertModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "convert.ptx"));
        _dequantModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant.ptx"));
        _quantizedGemvModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "quantized_gemv.ptx"));
        _fusedAddRmsNormModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "fused_add_rmsnorm.ptx"));
        _rmsnormF32InModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "rmsnorm_f32in.ptx"));
        _addF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "add_f32.ptx"));
        _embeddingF32OutModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "embedding_f32out.ptx"));
        _ropeF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "rope_f32.ptx"));
        _attentionF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "attention_f32.ptx"));
        _swigluF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "swiglu_f32.ptx"));
        _biasAddF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "bias_add_f32.ptx"));
        _perHeadRmsNormF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "per_head_rmsnorm_f32.ptx"));
        _rmsnormF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "rmsnorm_f32.ptx"));
        // Optional — TryGetFunction so a stale PTX still loads gracefully and
        // HasCopyRmsNormF32 reports false (caller falls back to a separate D2D
        // copy + LaunchRmsNormF32 pair).
        string copyRmsNormF32Path = Path.Combine(ptxDir, "copy_rmsnorm_f32.ptx");
        if (File.Exists(copyRmsNormF32Path))
        {
            _copyRmsNormF32Module = CudaModule.LoadFromFile(copyRmsNormF32Path);
            _copyRmsNormF32Func = _copyRmsNormF32Module.TryGetFunction("copy_rmsnorm_f32");
        }
        _quantizedGemvF32InModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "quantized_gemv_f32in.ptx"));
        _i2sGemvModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "i2_s_gemv.ptx"));
        _dequantI2sModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant_i2_s.ptx"));
        _pq2_0GemvModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "pq2_0_gemv.ptx"));
        _dequantPQ2_0Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant_pq2_0.ptx"));
        _pq2_0RepackModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "pq2_0_repack.ptx"));
        _relu2Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "relu2.ptx"));
        _relu2F32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "relu2_f32.ptx"));
        _relu2GluRmsNormModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "relu2_glu_rmsnorm.ptx"));
        _fusedAddRmsNormF32ResModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "fused_add_rmsnorm_f32res.ptx"));

        // Query the device's opt-in dynamic shared-memory cap ONCE, unconditionally (not gated
        // on the optional MMQ PTX below) — the always-loaded I2_S GEMV kernels need it too (see
        // issue #207: their x-staging shared buffer used to be a BitNet-2B-4T-sized static array;
        // it is now dynamic and must be opted into >48 KB for large-intermediate non-BitNet models).
        {
            int devForOptIn;
            if (CudaDriverApi.cuCtxGetDevice(out devForOptIn) == 0
                && CudaDriverApi.cuDeviceGetAttribute(out int optIn0,
                    CudaDriverApi.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, devForOptIn) == 0
                && optIn0 > 0)
            {
                _maxDynamicSharedBytesOptIn = optIn0;
            }
        }

        // MMQ-style fused dequant+matmul GEMV (optional — PTX may not be compiled yet).
        // Provides a faster Q4_K decode path via dp4a-packed INT8 multiply-add.
        string mmqPath = Path.Combine(ptxDir, "quantized_gemv_mmq.ptx");
        if (File.Exists(mmqPath))
        {
            _quantizedGemvMmqModule = CudaModule.LoadFromFile(mmqPath);
            _quantizedGemvQ4_KMmqFunc = _quantizedGemvMmqModule.GetFunction("quantized_gemv_q4_k_mmq");
            _quantizedGemvQ2_KMmqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q2_k_mmq");
            _quantizedGemvQ5_KMmqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q5_k_mmq");
            _quantizedGemvQ6_KMmqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q6_k_mmq");
            _quantizedGemvIQ4_NLMmqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_iq4_nl_mmq");
            _quantizedGemvIQ4_XSMmqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_iq4_xs_mmq");
            // MMVQ-large variants (k≥1024). TryGetFunction so a stale PTX without the
            // new kernels still loads — HasMmvqLarge* will report false and the dispatcher
            // will fall back to the MMQ-4-rows path.
            _quantizedGemvQ2_KMmvqLargeFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q2_k_mmvq_large");
            _quantizedGemvQ4_KMmvqLargeFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q4_k_mmvq_large");
            _quantizedGemvQ4_KMmvqCoalescedFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q4_k_mmvq_coalesced");
            _quantizedGemvQ4_KMmvqLlamaCppFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q4_k_mmvq_llamacpp");
            _quantizedGemvQ5_KMmvqLargeFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q5_k_mmvq_large");
            _quantizedGemvQ6_KMmvqLargeFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q6_k_mmvq_large");
            _quantizedGemvIQ4_NLMmvqLargeFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_iq4_nl_mmvq_large");
            _quantizedGemvIQ4_XSMmvqLargeFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_iq4_xs_mmvq_large");
            // Pre-quantized GEMV variants (consume scratch from quantize_x.ptx kernel).
            // TryGetFunction so a stale PTX without the new symbols still loads.
            _quantizedGemvQ2_KMmqPreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q2_k_mmq_preq");
            _quantizedGemvQ4_KMmqPreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q4_k_mmq_preq");
            _quantizedGemvQ5_KMmqPreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q5_k_mmq_preq");
            _quantizedGemvQ6_KMmqPreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q6_k_mmq_preq");
            _quantizedGemvIQ4_NLMmqPreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_iq4_nl_mmq_preq");
            _quantizedGemvIQ4_XSMmqPreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_iq4_xs_mmq_preq");
            _quantizedGemvQ2_KMmvqLargePreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q2_k_mmvq_large_preq");
            _quantizedGemvQ4_KMmvqLargePreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q4_k_mmvq_large_preq");
            _quantizedGemvQ5_KMmvqLargePreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q5_k_mmvq_large_preq");
            _quantizedGemvQ6_KMmvqLargePreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q6_k_mmvq_large_preq");
            _quantizedGemvIQ4_NLMmvqLargePreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_iq4_nl_mmvq_large_preq");
            _quantizedGemvIQ4_XSMmvqLargePreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_iq4_xs_mmvq_large_preq");
            // Batched-M prefill kernel (issue #349) — TryGetFunction so a stale PTX
            // without the new symbol still loads; HasMmqBatchedQ4K reports false.
            _quantizedGemvQ4_KMmqBatchedFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q4_k_mmq_batched");

            // The on-the-fly MMQ kernels size their per-chunk Stage 1 scratch (s_xq/s_dx/s_sx[2])
            // dynamically from `k`. For k up to ~12 KiB-shmem-worth (Qwen3-8B intermediate=12288)
            // we fit under the 48 KB default cap, but Llama-70B-class intermediate=14336 lands at
            // ~15.7 KB and Llama-405B-class intermediate=53248 lands at ~58 KB — past 48 KB. Opt
            // each on-the-fly variant into the device's full optin cap (typically 100+ KB on
            // Ampere/Ada/Hopper) so any in-budget k launches without recompiling.
            if (_maxDynamicSharedBytesOptIn > 0)
            {
                int optIn = _maxDynamicSharedBytesOptIn;
                SetMaxDynamicSharedBytes(_quantizedGemvQ4_KMmqFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ2_KMmqFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ5_KMmqFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ6_KMmqFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvIQ4_NLMmqFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvIQ4_XSMmqFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ2_KMmvqLargeFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ4_KMmvqLargeFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ4_KMmvqLlamaCppFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ5_KMmvqLargeFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ6_KMmvqLargeFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvIQ4_NLMmvqLargeFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvIQ4_XSMmvqLargeFunc, optIn);
            }
        }

        // Pre-Q8_1 input quantization kernel (optional — PTX may be missing on stale builds).
        string quantizeXPath = Path.Combine(ptxDir, "quantize_x.ptx");
        if (File.Exists(quantizeXPath))
        {
            _quantizeXModule = CudaModule.LoadFromFile(quantizeXPath);
            _quantizeXToQ8_1Func = _quantizeXModule.GetFunction("quantize_x_to_q8_1");
            _quantizeXToQ8_1BatchedFunc = _quantizeXModule.TryGetFunction("quantize_x_to_q8_1_batched");
        }

        string turboquantPath = Path.Combine(ptxDir, "turboquant.ptx");
        if (File.Exists(turboquantPath))
        {
            _turboquantModule = CudaModule.LoadFromFile(turboquantPath);
            _turboquantDequantF32Func = _turboquantModule.TryGetFunction("turboquant_dequant_f32");
            _turboquantEncodeF32Func = _turboquantModule.TryGetFunction("turboquant_encode_f32");
            _turboquantDequantF16Func = _turboquantModule.TryGetFunction("turboquant_dequant_f16");
            _turboquantEncodeF16Func = _turboquantModule.TryGetFunction("turboquant_encode_f16");
        }

        _rmsnormFunc = _rmsnormModule.GetFunction("rmsnorm_f16");
        _rmsnormF32Func = _rmsnormF32Module.GetFunction("rmsnorm_f32");
        _quantizedGemvQ8_0F32InFunc = _quantizedGemvF32InModule.GetFunction("quantized_gemv_q8_0_f32in");
        _fusedAddRmsNormFunc = _fusedAddRmsNormModule.GetFunction("fused_add_rmsnorm_f16");
        _rmsnormF32InF16OutFunc = _rmsnormF32InModule.GetFunction("rmsnorm_f32in_f16out");
        _addF32Func = _addF32Module.GetFunction("add_f32");
        _addF32F16Func = _addF32Module.GetFunction("add_f32_f16");
        _embeddingF32OutF32Func = _embeddingF32OutModule.GetFunction("embedding_lookup_f32_f32out");
        _embeddingF32OutF16Func = _embeddingF32OutModule.GetFunction("embedding_lookup_f16_f32out");
        _embeddingF32OutQ8_0Func = _embeddingF32OutModule.GetFunction("embedding_lookup_q8_0_f32out");
        _ropeF32Func = _ropeF32Module.GetFunction("rope_f32");
        _attentionF32Func = _attentionF32Module.GetFunction("attention_f32");
        // Optional (issue #183) — TryGetFunction so a stale PTX without the split-KV kernel
        // still loads gracefully; HasAttentionF32SplitKv reports false and callers fall back to
        // the exact LaunchAttentionF32 path unconditionally.
        _attentionF32SplitKvFunc = _attentionF32Module.TryGetFunction("attention_f32_split_kv");
        // Optional (issue #226 spike) -- fp64-combine variant of attention_f32_split_kv, see that
        // kernel's header for scope. TryGetFunction: stale PTX without it just disables the gate.
        _attentionF32SplitKvHpFunc = _attentionF32Module.TryGetFunction("attention_f32_split_kv_hp");
        // Optional (issues #197+#198) -- TryGetFunction so a stale PTX without the combined
        // GQA-group + split-KV kernel still loads gracefully; HasAttentionF32GqaSplitKv reports
        // false and callers fall back to attention_f32_split_kv / LaunchAttentionF32.
        _attentionF32GqaSplitKvFunc = _attentionF32Module.TryGetFunction("attention_f32_gqa_split_kv");
        _swigluF32Func = _swigluF32Module.GetFunction("swiglu_f32");
        _relu2GluF32Func = _swigluF32Module.GetFunction("relu2glu_f32");
        _biasAddF32Func = _biasAddF32Module.GetFunction("bias_add_f32");
        _perHeadRmsNormF32Func = _perHeadRmsNormF32Module.GetFunction("per_head_rmsnorm_f32");
        _ropeFunc = _ropeModule.GetFunction("rope_f16");
        _swigluFunc = _swigluModule.GetFunction("swiglu_f16");
        _addFunc = _addModule.GetFunction("add_f16");
        _softmaxFunc = _softmaxModule.GetFunction("softmax_f16");
        _embeddingF32Func = _embeddingModule.GetFunction("embedding_lookup_f32");
        _embeddingF16Func = _embeddingModule.GetFunction("embedding_lookup_f16");
        _embeddingQ8_0Func = _embeddingModule.GetFunction("embedding_lookup_q8_0");
        // Per-row K-quant lookups are optional — TryGetFunction so a stale PTX
        // (without the new symbols) still loads gracefully.
        _embeddingQ4_KFunc = _embeddingModule.TryGetFunction("embedding_lookup_q4_k");
        _embeddingQ5_KFunc = _embeddingModule.TryGetFunction("embedding_lookup_q5_k");
        _embeddingQ6_KFunc = _embeddingModule.TryGetFunction("embedding_lookup_q6_k");
        _attentionFunc = _attentionModule.GetFunction("attention_f16");
        _attentionDynFunc = _attentionModule.GetFunction("attention_f16_dyn");
        _attentionPosFunc = _attentionModule.GetFunction("attention_pos_f16");
        // Issue #200 -- optional, see field comment above.
        _attentionPagedFunc = _attentionModule.TryGetFunction("attention_f16_paged");
        _kvCacheUpdatePosFunc = _kvCacheUpdateModule.GetFunction("kv_cache_update_pos_f16");
        _biasAddFunc = _biasAddModule.GetFunction("bias_add_f16");
        _perHeadRmsNormFunc = _perHeadRmsNormModule.GetFunction("per_head_rmsnorm_f16");
        _convertF16ToF32Func = _convertModule.GetFunction("convert_f16_to_f32");
        _convertF32ToF16Func = _convertModule.GetFunction("convert_f32_to_f16");
        _quantizedGemvQ8_0Func = _quantizedGemvModule.GetFunction("quantized_gemv_q8_0");
        _quantizedGemvQ2_KFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q2_k");
        _quantizedGemvQ4_KFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q4_k");
        _quantizedGemvQ5_0Func = _quantizedGemvModule.GetFunction("quantized_gemv_q5_0");
        _quantizedGemvQ5_KFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q5_k");
        _quantizedGemvQ6_KFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q6_k");
        _quantizedGemvIQ4_NLFunc = _quantizedGemvModule.TryGetFunction("quantized_gemv_iq4_nl");
        _quantizedGemvIQ4_XSFunc = _quantizedGemvModule.TryGetFunction("quantized_gemv_iq4_xs");
        _i2sGemvF16InFunc = _i2sGemvModule.GetFunction("i2_s_gemv_f16in");
        _i2sGemv2F16InFunc = _i2sGemvModule.GetFunction("i2_s_gemv2_f16in");
        _i2sGemv3F16InFunc = _i2sGemvModule.GetFunction("i2_s_gemv3_f16in");
        _i2sGemvNormF16InFunc = _i2sGemvModule.GetFunction("i2_s_gemv_norm_f16in");
        _i2sGemvF32InFunc = _i2sGemvModule.GetFunction("i2_s_gemv_f32in");
        _i2sGemmF32InFunc = _i2sGemvModule.TryGetFunction("i2_s_gemm_f32in");
        _i2sGemvA8Func = _i2sGemvModule.GetFunction("i2_s_gemv_a8");
        _i2sGemvA8DeviceScaleFunc = _i2sGemvModule.GetFunction("i2_s_gemv_a8_device_scale");
        _quantizeF16ToI8AbsMaxFunc = _i2sGemvModule.GetFunction("quantize_f16_to_i8_absmax");
        _dequantI2sF16Func = _dequantI2sModule.GetFunction("dequant_i2_s_f16");
        // Ragged K (k % 128 != 0) — issue #206.
        _i2sGemvF16InRaggedFunc = _i2sGemvModule.GetFunction("i2_s_gemv_f16in_ragged");
        _i2sGemvF32InRaggedFunc = _i2sGemvModule.GetFunction("i2_s_gemv_f32in_ragged");
        _dequantI2sF16RaggedFunc = _dequantI2sModule.GetFunction("dequant_i2_s_f16_ragged");
        // Opt the x-staging I2_S GEMV kernels (dynamic shared memory, see i2_s_gemv.cu / issue #207)
        // into the device's full dynamic-shared cap, same as the on-the-fly MMQ kernels above.
        // Covers both the aligned fast-path kernels and the ragged (#206) fallback kernels — both
        // families stage x[k] into the same `extern __shared__ float xs[]` pattern now.
        // i2_s_gemv_a8[_device_scale] read activations from global (no xs[] staging) — no opt-in needed.
        if (_maxDynamicSharedBytesOptIn > 0)
        {
            SetMaxDynamicSharedBytes(_i2sGemvF16InFunc, _maxDynamicSharedBytesOptIn);
            SetMaxDynamicSharedBytes(_i2sGemv2F16InFunc, _maxDynamicSharedBytesOptIn);
            SetMaxDynamicSharedBytes(_i2sGemv3F16InFunc, _maxDynamicSharedBytesOptIn);
            SetMaxDynamicSharedBytes(_i2sGemvNormF16InFunc, _maxDynamicSharedBytesOptIn);
            SetMaxDynamicSharedBytes(_i2sGemvF32InFunc, _maxDynamicSharedBytesOptIn);
            SetMaxDynamicSharedBytes(_i2sGemvF16InRaggedFunc, _maxDynamicSharedBytesOptIn);
            SetMaxDynamicSharedBytes(_i2sGemvF32InRaggedFunc, _maxDynamicSharedBytesOptIn);
            // i2_s_gemm_f32in's shared cache is int8 (rowsPerBlock * k bytes) — strictly smaller
            // than the GEMV kernels' k*sizeof(float) staging buffer for the same k at
            // rowsPerBlock<=4, so this opt-in is a safety margin rather than a hard requirement.
            SetMaxDynamicSharedBytes(_i2sGemmF32InFunc, _maxDynamicSharedBytesOptIn);
        }
        _pq2_0GemvF32InFunc = _pq2_0GemvModule.GetFunction("pq2_0_gemv_f32in");
        _pq2_0GemvF16InFunc = _pq2_0GemvModule.GetFunction("pq2_0_gemv_f16in");
        _pq2_0Gemv2F16InFunc = _pq2_0GemvModule.GetFunction("pq2_0_gemv2_f16in");
        _pq2_0GemvF16InSmallFunc = _pq2_0GemvModule.GetFunction("pq2_0_gemv_f16in_small");
        _pq2_0Gemv2F16InSmallFunc = _pq2_0GemvModule.GetFunction("pq2_0_gemv2_f16in_small");
        _pq2_0GemvF32IoFunc = _pq2_0GemvModule.GetFunction("pq2_0_gemv_f32io");
        _pq2_0Gemv2F32IoFunc = _pq2_0GemvModule.GetFunction("pq2_0_gemv2_f32io");
        _pq2_0GemvF32IoSmallFunc = _pq2_0GemvModule.GetFunction("pq2_0_gemv_f32io_small");
        _pq2_0Gemv2F32IoSmallFunc = _pq2_0GemvModule.GetFunction("pq2_0_gemv2_f32io_small");
        _dequantPQ2_0F16Func = _dequantPQ2_0Module.GetFunction("dequant_pq2_0_f16");
        _pq2_0RepackSplitF16Func = _pq2_0RepackModule.GetFunction("pq2_0_repack_split_f16");
        _relu2Func = _relu2Module.GetFunction("relu2_f16");
        _relu2F32Func = _relu2F32Module.GetFunction("relu2_f32");
        _relu2GluRmsNormFunc = _relu2GluRmsNormModule.GetFunction("relu2_glu_rmsnorm_f16");
        _fusedAddRmsNormF32ResFunc = _fusedAddRmsNormF32ResModule.GetFunction("fused_add_rmsnorm_f32res");
        _copyF16ToF32Func = _fusedAddRmsNormF32ResModule.GetFunction("copy_f16_to_f32");
        _addF32ResF16Func = _fusedAddRmsNormF32ResModule.GetFunction("add_f32res_f16");
        _rmsNormF32InF16WFunc = _fusedAddRmsNormF32ResModule.GetFunction("rmsnorm_f32in_f16w");
        _dequantQ8_0Func = _dequantModule.GetFunction("dequant_q8_0_f16");
        _dequantQ4_0Func = _dequantModule.GetFunction("dequant_q4_0_f16");
        _dequantQ4_1Func = _dequantModule.TryGetFunction("dequant_q4_1_f16");
        _dequantQ5_0Func = _dequantModule.GetFunction("dequant_q5_0_f16");
        _dequantQ5_1Func = _dequantModule.TryGetFunction("dequant_q5_1_f16");
        // Q2_K is optional — older PTX builds may not have it.
        _dequantQ2_KFunc = _dequantModule.TryGetFunction("dequant_q2_k_f16");
        // Q3_K is optional — older PTX builds (pre-Round 12) may not have it.
        _dequantQ3_KFunc = _dequantModule.TryGetFunction("dequant_q3_k_f16");
        _dequantQ4_KFunc = _dequantModule.GetFunction("dequant_q4_k_f16");
        _dequantQ5_KFunc = _dequantModule.GetFunction("dequant_q5_k_f16");
        _dequantQ5_KF32Func = _dequantModule.TryGetFunction("dequant_q5_k_f32");
        _dequantQ6_KFunc = _dequantModule.GetFunction("dequant_q6_k_f16");
        // F32 sibling for Q6_K — optional, falls back to a NotSupportedException
        // in LaunchDequantToF32 if absent on a stale PTX.
        _dequantQ6_KF32Func = _dequantModule.TryGetFunction("dequant_q6_k_f32");

        string dequantIQuantsPath = Path.Combine(ptxDir, "dequant_iquants.ptx");
        if (File.Exists(dequantIQuantsPath))
        {
            _dequantIQuantsModule = CudaModule.LoadFromFile(dequantIQuantsPath);
            _dequantIQ4_NLFunc = _dequantIQuantsModule.TryGetFunction("dequant_iq4_nl_f16");
            _dequantIQ4_XSFunc = _dequantIQuantsModule.TryGetFunction("dequant_iq4_xs_f16");
            _dequantIQ4_NLF32Func = _dequantIQuantsModule.TryGetFunction("dequant_iq4_nl_f32");
            _dequantIQ4_XSF32Func = _dequantIQuantsModule.TryGetFunction("dequant_iq4_xs_f32");
        }

        // IQ2 module bundles the 2-bit dequant + GEMV kernels (iq2_xxs / iq2_xs / iq2_s).
        // Optional: older PTX builds may pre-date this kernel, so the file is allowed to be
        // absent. Each function pointer is independently nullable so partial loads still work.
        string iq2Path = Path.Combine(ptxDir, "iq2.ptx");
        if (File.Exists(iq2Path))
        {
            _iq2Module = CudaModule.LoadFromFile(iq2Path);
            _dequantIQ2_XXSFunc = _iq2Module.TryGetFunction("dequant_iq2_xxs_f16");
            _dequantIQ2_XSFunc = _iq2Module.TryGetFunction("dequant_iq2_xs_f16");
            _dequantIQ2_SFunc = _iq2Module.TryGetFunction("dequant_iq2_s_f16");
            _dequantIQ2_XXSF32Func = _iq2Module.TryGetFunction("dequant_iq2_xxs_f32");
            _dequantIQ2_XSF32Func = _iq2Module.TryGetFunction("dequant_iq2_xs_f32");
            _dequantIQ2_SF32Func = _iq2Module.TryGetFunction("dequant_iq2_s_f32");
            _quantizedGemvIQ2_XXSFunc = _iq2Module.TryGetFunction("quantized_gemv_iq2_xxs");
            _quantizedGemvIQ2_XSFunc = _iq2Module.TryGetFunction("quantized_gemv_iq2_xs");
            _quantizedGemvIQ2_SFunc = _iq2Module.TryGetFunction("quantized_gemv_iq2_s");
        }

        // IQ3 dequant (iq3_xxs / iq3_s) — issue #258. Optional, like the IQ2 module.
        string iq3Path = Path.Combine(ptxDir, "dequant_iq3.ptx");
        if (File.Exists(iq3Path))
        {
            _dequantIQ3Module = CudaModule.LoadFromFile(iq3Path);
            _dequantIQ3_XXSFunc = _dequantIQ3Module.TryGetFunction("dequant_iq3_xxs_f16");
            _dequantIQ3_SFunc = _dequantIQ3Module.TryGetFunction("dequant_iq3_s_f16");
            _dequantIQ3_XXSF32Func = _dequantIQ3Module.TryGetFunction("dequant_iq3_xxs_f32");
            _dequantIQ3_SF32Func = _dequantIQ3Module.TryGetFunction("dequant_iq3_s_f32");
        }

        // IQ1_S dequant — issue #258.
        string iq1Path = Path.Combine(ptxDir, "dequant_iq1.ptx");
        if (File.Exists(iq1Path))
        {
            _dequantIQ1Module = CudaModule.LoadFromFile(iq1Path);
            _dequantIQ1_SFunc = _dequantIQ1Module.TryGetFunction("dequant_iq1_s_f16");
            _dequantIQ1_SF32Func = _dequantIQ1Module.TryGetFunction("dequant_iq1_s_f32");
        }

        // BF16 widening + MXFP4 dequant — issue #258.
        string bf16Mxfp4Path = Path.Combine(ptxDir, "dequant_bf16_mxfp4.ptx");
        if (File.Exists(bf16Mxfp4Path))
        {
            _dequantBf16Mxfp4Module = CudaModule.LoadFromFile(bf16Mxfp4Path);
            _dequantBF16Func = _dequantBf16Mxfp4Module.TryGetFunction("dequant_bf16_f16");
            _dequantBF16F32Func = _dequantBf16Mxfp4Module.TryGetFunction("dequant_bf16_f32");
            _dequantMXFP4Func = _dequantBf16Mxfp4Module.TryGetFunction("dequant_mxfp4_f16");
            _dequantMXFP4F32Func = _dequantBf16Mxfp4Module.TryGetFunction("dequant_mxfp4_f32");
        }

        // KV-cache quantization (optional — PTX may not be compiled yet)
        string quantKvPath = Path.Combine(ptxDir, "quant_kv.ptx");
        if (File.Exists(quantKvPath))
        {
            _quantKvModule = CudaModule.LoadFromFile(quantKvPath);
            _quantKvQ8_0Func = _quantKvModule.GetFunction("quant_f16_to_q8_0");
            _quantKvQ4_0Func = _quantKvModule.GetFunction("quant_f16_to_q4_0");
            _quantKvQ8_0DynFunc = _quantKvModule.GetFunction("quant_f16_to_q8_0_dyn");
            _quantKvQ4_0DynFunc = _quantKvModule.GetFunction("quant_f16_to_q4_0_dyn");
        }

        // Graph-friendly KV write (optional — only present when CUDA Graphs path is in use).
        string kvWritePath = Path.Combine(ptxDir, "kv_write.ptx");
        if (File.Exists(kvWritePath))
        {
            _kvWriteModule = CudaModule.LoadFromFile(kvWritePath);
            _kvWriteOneF16Func = _kvWriteModule.GetFunction("kv_write_one_f16");
            _kvWriteOneF16RingFunc = _kvWriteModule.GetFunction("kv_write_one_f16_ring");
            _kvDequantQ8_0DynFunc = _kvWriteModule.GetFunction("kv_dequant_q8_0_dyn");
            _kvDequantQ4_0DynFunc = _kvWriteModule.GetFunction("kv_dequant_q4_0_dyn");
            _kvWindowToScratchDynFunc = _kvWriteModule.GetFunction("kv_window_to_scratch_dyn");
        }

        // Fused decode-step RoPE + KV-cache write (optional — eager decode optimization).
        string fusedRopeKvWritePath = Path.Combine(ptxDir, "fused_rope_kv_write.ptx");
        if (File.Exists(fusedRopeKvWritePath))
        {
            _fusedRopeKvWriteModule = CudaModule.LoadFromFile(fusedRopeKvWritePath);
            _fusedRopeKvWriteF16Func = _fusedRopeKvWriteModule.GetFunction("fused_rope_kv_write_f16");
            _fusedRopeKvWriteF16DynFunc = _fusedRopeKvWriteModule.GetFunction("fused_rope_kv_write_f16_dyn");
        }

        // MLA Phase A naive forward kernel (F32). Optional — only required when
        // the model is DeepSeek-V2 / V3 with MLA attention.
        string attentionMlaPath = Path.Combine(ptxDir, "attention_mla.ptx");
        if (File.Exists(attentionMlaPath))
        {
            _attentionMlaModule = CudaModule.LoadFromFile(attentionMlaPath);
            _attentionMlaF32Func = _attentionMlaModule.GetFunction("attention_mla_f32");
            _attentionMlaF16Func = _attentionMlaModule.TryGetFunction("attention_mla_f16");
        }

        // MLA forward-path helper kernels (F32 split / RoPE / RMSNorm).
        string mlaHelpersPath = Path.Combine(ptxDir, "mla_helpers.ptx");
        if (File.Exists(mlaHelpersPath))
        {
            _mlaHelpersModule = CudaModule.LoadFromFile(mlaHelpersPath);
            _mlaSplitKvBF32Func = _mlaHelpersModule.GetFunction("mla_split_kv_b_f32");
            _mlaRopeQpeF32Func = _mlaHelpersModule.GetFunction("mla_rope_q_pe_f32");
            _mlaRopeKpeF32Func = _mlaHelpersModule.GetFunction("mla_rope_k_pe_f32");
            _mlaRmsNormF32Func = _mlaHelpersModule.GetFunction("mla_rmsnorm_f32");
            // FP16 siblings — TryGetFunction so a stale PTX (F32-only) still loads.
            _mlaSplitKvBF16Func = _mlaHelpersModule.TryGetFunction("mla_split_kv_b_f16");
            _mlaRopeQpeF16Func = _mlaHelpersModule.TryGetFunction("mla_rope_q_pe_f16");
            _mlaRopeKpeF16Func = _mlaHelpersModule.TryGetFunction("mla_rope_k_pe_f16");
            _mlaRmsNormF16Func = _mlaHelpersModule.TryGetFunction("mla_rmsnorm_f16");
        }

        // MLA Phase B: absorbed-attention kernel + Q absorption + V expansion
        // helpers. Optional — PTX file may not be present on stale builds, in
        // which case HasMlaPhaseB returns false and callers can fall back to
        // Phase A.
        string attentionMlaLatentPath = Path.Combine(ptxDir, "attention_mla_latent.ptx");
        if (File.Exists(attentionMlaLatentPath))
        {
            _attentionMlaLatentModule = CudaModule.LoadFromFile(attentionMlaLatentPath);
            _attentionMlaLatentF32Func = _attentionMlaLatentModule.GetFunction("attention_mla_latent_f32");
            _mlaQAbsorbUkF32Func = _attentionMlaLatentModule.GetFunction("mla_q_absorb_uk_f32");
            _mlaVExpandUvF32Func = _attentionMlaLatentModule.GetFunction("mla_v_expand_uv_f32");
        }

        // MoE (Mixture-of-Experts) forward-path helper kernels (F32). Optional —
        // only required when the model has MoE layers (Mixtral / Qwen-MoE /
        // DeepSeek-V2/V3). TryGetFunction so a stale PTX without the new
        // symbols still loads.
        string moeFfnPath = Path.Combine(ptxDir, "moe_ffn.ptx");
        if (File.Exists(moeFfnPath))
        {
            _moeFfnModule = CudaModule.LoadFromFile(moeFfnPath);
            _moeSoftmaxTopkF32Func = _moeFfnModule.TryGetFunction("moe_softmax_topk_f32");
            _moeRenormTopkF32Func = _moeFfnModule.TryGetFunction("moe_renorm_topk_f32");
            _moeZeroF32Func = _moeFfnModule.TryGetFunction("moe_zero_f32");
            _moeAxpyScaledRowF32Func = _moeFfnModule.TryGetFunction("moe_axpy_scaled_row_f32");
            _moeAxpyUnweightedF32Func = _moeFfnModule.TryGetFunction("moe_axpy_unweighted_f32");
            _moeAxpyScaledPerTokenF32Func = _moeFfnModule.TryGetFunction("moe_axpy_scaled_per_token_f32");
            _moeSigmoidLogitF32Func = _moeFfnModule.TryGetFunction("moe_sigmoid_logit_f32");
            _moeGatherTokenRowsF32Func = _moeFfnModule.TryGetFunction("moe_gather_token_rows_f32");
            _moeGateBiasAddF32Func = _moeFfnModule.TryGetFunction("moe_gate_bias_add_f32");
            _swigluOaiF32Func = _moeFfnModule.TryGetFunction("swiglu_oai_f32");
        }

        // MoE grouped-GEMV (Phase B). One kernel walks K_active raw-quant per-expert
        // pointers in a single launch. Q4_K + Q5_K + Q6_K + Q8_0 supported; per-quant
        // HasMoeGroupedGemv* gates the call so a stale PTX without one entry still
        // routes the others through the fast path and falls back per-expert for the
        // missing one.
        string moeGroupedGemvPath = Path.Combine(ptxDir, "moe_grouped_gemv.ptx");
        if (File.Exists(moeGroupedGemvPath))
        {
            _moeGroupedGemvModule = CudaModule.LoadFromFile(moeGroupedGemvPath);
            _moeGroupedGemvQ2_KFunc = _moeGroupedGemvModule.TryGetFunction("moe_grouped_gemv_q2_k_f16");
            _moeGroupedGemvQ4_KFunc = _moeGroupedGemvModule.TryGetFunction("moe_grouped_gemv_q4_k_f16");
            _moeGroupedGemvQ5_KFunc = _moeGroupedGemvModule.TryGetFunction("moe_grouped_gemv_q5_k_f16");
            _moeGroupedGemvQ6_KFunc = _moeGroupedGemvModule.TryGetFunction("moe_grouped_gemv_q6_k_f16");
            _moeGroupedGemvQ8_0Func = _moeGroupedGemvModule.TryGetFunction("moe_grouped_gemv_q8_0_f16");
            _moeGroupedGemvIQ4_NLFunc = _moeGroupedGemvModule.TryGetFunction("moe_grouped_gemv_iq4_nl_f16");
            _moeGroupedGemvIQ4_XSFunc = _moeGroupedGemvModule.TryGetFunction("moe_grouped_gemv_iq4_xs_f16");
        }

        // Qwen3MoeHybrid recurrence-path kernels (optional — present only when
        // the GDN/conv1d CUDA path has been built). Each .cu file compiles to
        // a separate .ptx; build_ptx.bat auto-globs native/kernels/*.cu so no
        // script change is required.
        string conv1dCausalF32Path = Path.Combine(ptxDir, "conv1d_causal.ptx");
        if (File.Exists(conv1dCausalF32Path))
        {
            _conv1dCausalF32Module = CudaModule.LoadFromFile(conv1dCausalF32Path);
            _conv1dCausalF32Func = _conv1dCausalF32Module.TryGetFunction("conv1d_causal_f32");
            _gdnConv1dCausalDecodeF32Func = _conv1dCausalF32Module.TryGetFunction("gdn_conv1d_causal_decode_f32");
        }

        string gdnScanF32Path = Path.Combine(ptxDir, "gated_delta_net_scan.ptx");
        if (File.Exists(gdnScanF32Path))
        {
            _gdnScanF32Module = CudaModule.LoadFromFile(gdnScanF32Path);
            _gdnScanStepF32Func = _gdnScanF32Module.TryGetFunction("gdn_scan_step_f32");
            // Opt-in, default-off row-split cooperative-groups variant (issue #180) — see
            // gated_delta_net_scan.cu's header for the full measured-speedup-vs-bit-parity-
            // tradeoff writeup. TryGetFunction so a stale PTX (pre-#180) still loads gracefully.
            _gdnScanStepF32CoopSplit4Func = _gdnScanF32Module.TryGetFunction("gdn_scan_step_f32_coop_split4");
            // The L2-norm and decay helpers live in the same .cu translation unit,
            // so they are in the same .ptx module — query each function pointer here.
            _l2NormHeadsF32Module = _gdnScanF32Module;
            _l2NormHeadsF32Func = _gdnScanF32Module.TryGetFunction("l2_normalize_heads_f32");
            _gdnDeinterleaveL2NormDecodeF32Func =
                _gdnScanF32Module.TryGetFunction("gdn_deinterleave_l2norm_decode_f32");
            _gdnDecayF32Func = _gdnScanF32Module.TryGetFunction("gdn_decay_f32");
            _gdnDecaySigmoidF32Func = _gdnScanF32Module.TryGetFunction("gdn_decay_sigmoid_f32");
        }

        string mamba2ScanF32Path = Path.Combine(ptxDir, "mamba2_selective_scan.ptx");
        if (File.Exists(mamba2ScanF32Path))
        {
            _mamba2ScanF32Module = CudaModule.LoadFromFile(mamba2ScanF32Path);
            _mamba2ScanF32Func = _mamba2ScanF32Module.TryGetFunction("mamba2_selective_scan_f32");
        }

        string groupRmsNormF32Path = Path.Combine(ptxDir, "group_rmsnorm.ptx");
        if (File.Exists(groupRmsNormF32Path))
        {
            _groupRmsNormF32Module = CudaModule.LoadFromFile(groupRmsNormF32Path);
            _groupRmsNormF32Func = _groupRmsNormF32Module.TryGetFunction("group_rmsnorm_f32");
        }

        // Plain elementwise squared-ReLU for NemotronH's non-gated FFN activation.
        // Optional — TryGetFunction so a stale PTX still loads gracefully.
        string reluSquaredInplaceF32Path = Path.Combine(ptxDir, "relu_squared_inplace.ptx");
        if (File.Exists(reluSquaredInplaceF32Path))
        {
            _reluSquaredInplaceF32Module = CudaModule.LoadFromFile(reluSquaredInplaceF32Path);
            _reluSquaredInplaceF32Func = _reluSquaredInplaceF32Module.TryGetFunction("relu_squared_inplace_f32");
        }

        // Pointwise FP32 helpers (sigmoid / silu / sigmoid_mul) for the post-
        // attention gating and the Qwen3MoeHybrid GDN body. Optional — TryGetFunction
        // so a stale PTX still loads gracefully and HasElementwiseF32 reports false.
        string elementwiseF32Path = Path.Combine(ptxDir, "elementwise_f32.ptx");
        if (File.Exists(elementwiseF32Path))
        {
            _elementwiseF32Module = CudaModule.LoadFromFile(elementwiseF32Path);
            _sigmoidF32Func = _elementwiseF32Module.TryGetFunction("sigmoid_f32");
            _siluF32Func = _elementwiseF32Module.TryGetFunction("silu_f32");
            _sigmoidMulF32Func = _elementwiseF32Module.TryGetFunction("sigmoid_mul_f32");
            _deinterleaveQGateF32Func = _elementwiseF32Module.TryGetFunction("deinterleave_qgate_f32");
            _deinterleaveGdnQkvF32Func = _elementwiseF32Module.TryGetFunction("deinterleave_gdn_qkv_f32");
        }

        // Gemma-4 (DiffusionGemma AR) F32 helper kernels — optional PTX. Absent on
        // stale builds until native/build.{sh,ps1} regenerates it (no nvcc here on
        // the AMD dev box; regenerated on the CUDA box). HasGemma4Kernels gates the
        // gemma4 forward path so a stale PTX surfaces a clear error, not an NRE.
        string gemma4F32Path = Path.Combine(ptxDir, "gemma4_f32.ptx");
        if (File.Exists(gemma4F32Path))
        {
            _gemma4F32Module = CudaModule.LoadFromFile(gemma4F32Path);
            _gegluTanhF32Func = _gemma4F32Module.TryGetFunction("geglu_tanh_f32");
            _ropeF32PartialNeoxFunc = _gemma4F32Module.TryGetFunction("rope_f32_partial_neox");
            _scaleInplaceF32Func = _gemma4F32Module.TryGetFunction("scale_inplace_f32");
            _rmsnormWeightlessF32Func = _gemma4F32Module.TryGetFunction("rmsnorm_weightless_f32");
            _softcapInplaceF32Func = _gemma4F32Module.TryGetFunction("softcap_inplace_f32");
            _moeRenormTopkClampedF32Func = _gemma4F32Module.TryGetFunction("moe_renorm_topk_clamped_f32");
            _quantizeActQ8_0RoundtripF32Func = _gemma4F32Module.TryGetFunction("quantize_activation_q8_0_roundtrip_f32");
        }

        // Causal softmax for the cuBLAS tensor-core prefill-attention path (G3 prototype).
        // Optional — PTX may be missing on stale builds.
        string attentionSoftmaxCausalPath = Path.Combine(ptxDir, "attention_softmax_causal.ptx");
        if (File.Exists(attentionSoftmaxCausalPath))
        {
            _attentionSoftmaxCausalModule = CudaModule.LoadFromFile(attentionSoftmaxCausalPath);
            _attentionSoftmaxCausalFunc = _attentionSoftmaxCausalModule.TryGetFunction("attention_softmax_causal_f16");
            _attentionSoftmaxCausalCoalescedFunc = _attentionSoftmaxCausalModule.TryGetFunction("attention_softmax_causal_coalesced_f16");
            _attentionSoftmaxCausalCoalescedF32InFunc = _attentionSoftmaxCausalModule.TryGetFunction("attention_softmax_causal_coalesced_f32in_f16out");
        }

        // Hand-fused mma.sync flash-attention prefill kernel (G-flash). Optional, and the
        // ONLY module compiled to compute_86 (mma.sync.m16n8k16 is sm_80+). On a pre-Ampere
        // GPU (e.g. Turing sm_75) the driver cannot JIT this PTX: cuModuleLoadData JITs
        // eagerly and would throw, taking down model construction on hardware that should
        // simply fall back to G3/attention_f16. So the load is best-effort — a failure
        // leaves HasAttentionFlashMma false and the dispatch never selects it (it is also
        // arch-gated off in CudaFlashAttention.ConfigureDefault). This keeps non-Ampere
        // devices unregressed even though the sm_86 PTX ships in every package.
        string attentionFlashMmaPath = Path.Combine(ptxDir, "attention_flash_mma.ptx");
        if (File.Exists(attentionFlashMmaPath))
        {
            try
            {
                var flashModule = CudaModule.LoadFromFile(attentionFlashMmaPath);
                _attentionFlashMmaFunc = flashModule.TryGetFunction("attention_flash_mma_f16");
                _attentionFlashMmaModule = flashModule;
            }
            catch (CudaException)
            {
                // Pre-Ampere driver rejected the sm_86 module — leave flash disabled.
                _attentionFlashMmaModule = null;
                _attentionFlashMmaFunc = 0;
            }
        }

        // Tensor-core FP16 decode attention composed with the GQA-group + split-KV grid
        // (issue #199 v2). Same best-effort load pattern as attention_flash_mma above and for
        // the same reason (mma.sync PTX is sm_80+; cuModuleLoadData JITs eagerly and would
        // throw on Turing) — a failure leaves HasAttentionMmaDecodeGqaSplit false and the
        // gate/dispatch wrapper (CudaAttentionMmaDecodeGqaSplit) never selects it.
        string attentionMmaDecodeGqaSplitPath = Path.Combine(ptxDir, "attention_flash_mma_decode_gqa_split.ptx");
        if (File.Exists(attentionMmaDecodeGqaSplitPath))
        {
            try
            {
                var mmaDecodeGqaSplitModule = CudaModule.LoadFromFile(attentionMmaDecodeGqaSplitPath);
                _attentionMmaDecodeGqaSplitFunc = mmaDecodeGqaSplitModule.TryGetFunction("attention_flash_mma_decode_gqa_split_f16");
                _attentionMmaDecodeGqaSplitModule = mmaDecodeGqaSplitModule;
            }
            catch (CudaException)
            {
                _attentionMmaDecodeGqaSplitModule = null;
                _attentionMmaDecodeGqaSplitFunc = 0;
            }
        }
    }

    /// <summary>
    /// True when all Gemma-4 F32 helper kernels (GeGLU-tanh, partial-NeoX RoPE,
    /// scalar scale, weight-less RMSNorm, softcap, clamped top-k renorm) are
    /// available. Required by the gemma4 AR forward path
    /// (<see cref="CudaTransformerModel"/>).
    /// </summary>
    public bool HasGemma4Kernels =>
        _gegluTanhF32Func != 0 && _ropeF32PartialNeoxFunc != 0
        && _scaleInplaceF32Func != 0 && _rmsnormWeightlessF32Func != 0
        && _softcapInplaceF32Func != 0 && _moeRenormTopkClampedF32Func != 0
        && _quantizeActQ8_0RoundtripF32Func != 0;

    /// <summary>
    /// True when the hand-fused mma.sync flash-attention prefill kernel is loaded
    /// (attention_flash_mma.ptx present). Prototype path, Llama-3.2-1B head shape only.
    /// </summary>
    public bool HasAttentionFlashMma => _attentionFlashMmaFunc != 0;

    /// <summary>
    /// True when the causal-softmax kernel for the cuBLAS tensor-core prefill-attention
    /// path is loaded (attention_softmax_causal.ptx present). When false, callers must
    /// keep the fused <c>attention_f16</c> path.
    /// </summary>
    public bool HasAttentionSoftmaxCausal => _attentionSoftmaxCausalFunc != 0;

    /// <summary>
    /// True when the coalesced (one-thread-per-row) causal-softmax kernel is loaded.
    /// This is the preferred variant — its global reads are coalesced, unlike the
    /// one-block-per-row sibling whose strided reads cap throughput.
    /// </summary>
    public bool HasAttentionSoftmaxCausalCoalesced => _attentionSoftmaxCausalCoalescedFunc != 0;

    /// <summary>
    /// True when the FP32-scores causal-softmax kernel is loaded (reads FP32 QK scores,
    /// writes FP16 probs). The G3 prefill path uses this to hold end-to-end logit parity
    /// at the 5e-3 bar — the all-FP16 variant loses too much precision rounding the
    /// pre-softmax scores at realistic activation magnitudes.
    /// </summary>
    public bool HasAttentionSoftmaxCausalCoalescedF32In => _attentionSoftmaxCausalCoalescedF32InFunc != 0;

    /// <summary>True when the MLA Phase A attention kernel is available on this kernel module.</summary>
    public bool HasMlaAttentionKernel => _attentionMlaF32Func != 0;

    /// <summary>True when all MLA forward-path helper kernels (split, RoPE, RMSNorm) are available.</summary>
    public bool HasMlaHelpers =>
        _mlaSplitKvBF32Func != 0 && _mlaRopeQpeF32Func != 0
        && _mlaRopeKpeF32Func != 0 && _mlaRmsNormF32Func != 0;

    #region MLA FP16
    /// <summary>True when the MLA Phase A attention kernel (FP16 sibling) is available on this kernel module.</summary>
    public bool HasMlaAttentionKernelF16 => _attentionMlaF16Func != 0;

    /// <summary>True when all FP16 MLA forward-path helper kernels (split, RoPE, RMSNorm) are available.</summary>
    public bool HasMlaHelpersF16 =>
        _mlaSplitKvBF16Func != 0 && _mlaRopeQpeF16Func != 0
        && _mlaRopeKpeF16Func != 0 && _mlaRmsNormF16Func != 0;
    #endregion

    /// <summary>True when the MLA Phase B (absorbed attention + helpers) PTX is available.</summary>
    public bool HasMlaPhaseB =>
        _attentionMlaLatentF32Func != 0
        && _mlaQAbsorbUkF32Func != 0 && _mlaVExpandUvF32Func != 0;

    /// <summary>
    /// True when all MoE FFN helper kernels (softmax-topk, renorm, zero, axpy
    /// variants, sigmoid logit, token gather) are available. Required by
    /// <see cref="CudaMoeFfn.Forward"/>.
    /// </summary>
    public bool HasMoeKernels =>
        _moeSoftmaxTopkF32Func != 0 && _moeRenormTopkF32Func != 0
        && _moeZeroF32Func != 0 && _moeAxpyScaledRowF32Func != 0
        && _moeAxpyUnweightedF32Func != 0 && _moeAxpyScaledPerTokenF32Func != 0
        && _moeSigmoidLogitF32Func != 0 && _moeGatherTokenRowsF32Func != 0;

    /// <summary>
    /// True when the additive router-bias kernel (<see cref="LaunchMoeGateBiasAddF32"/>,
    /// issue #246) is available. Required only when the MoE layer's router carries a
    /// non-null <c>gate.bias</c> (identity-MoTE / Qwen3 aux-loss-free routing); harmless
    /// (unused) otherwise.
    /// </summary>
    public bool HasMoeGateBiasAdd => _moeGateBiasAddF32Func != 0;

    /// <summary>
    /// True when the gpt-oss OAI-clamped-SwiGLU activation kernel (issue #348,
    /// <see cref="LaunchSwiGLUOaiF32"/>) is loaded. Optional — a stale PTX build
    /// without this symbol still loads; <see cref="CudaMoeFfn.Forward"/> throws
    /// a descriptive error only if a model actually needs it
    /// (<c>CudaMoeLayerWeights.UseSwiGluOai == true</c>).
    /// </summary>
    public bool HasSwiGluOai => _swigluOaiF32Func != 0;

    /// <summary>
    /// True when all kernels needed by <see cref="CudaMoeFfn"/>'s BitNet-ternary (I2_S)
    /// MoE forward path (issue #246) are available: the shared MoE orchestration kernels
    /// (<see cref="HasMoeKernels"/>), the F32 I2_S GEMV, the relu²·GLU activation, and F32
    /// RMSNorm (per-expert FFN Sub-LN). The additive router-bias kernel is checked
    /// separately via <see cref="HasMoeGateBiasAdd"/> — only required when the layer's
    /// router actually carries a bias.
    /// </summary>
    public bool HasBitNetMoeKernels =>
        HasMoeKernels && _i2sGemvF32InFunc != 0 && _relu2GluF32Func != 0 && _rmsnormF32Func != 0;

    /// <summary>
    /// True when the batched I2_S GEMM kernel (<see cref="LaunchI2_SGemmF32In"/>, issue #250) is
    /// available. Lets <see cref="CudaMoeFfn.ForwardBitNetI2S"/>'s prefill path (seqLen&gt;1, multiple
    /// tokens routed to the same expert) decode each expert's gate/up/down weight row ONCE and reuse
    /// it across all routed tokens, instead of the original per-row-GEMV-call loop (issue #246 scope
    /// note) that re-decoded the weight matrix once per token. Independent of
    /// <see cref="HasBitNetMoeKernels"/> so a stale PTX (not yet recompiled) still runs correctly via
    /// the per-row-loop fallback — this is a pure prefill-throughput optimization, never a correctness
    /// requirement.
    /// </summary>
    public bool HasI2SBatchedGemm => _i2sGemmF32InFunc != 0;

    /// <summary>
    /// True when all Qwen3MoeHybrid recurrence-path FP32 kernels (causal conv1d,
    /// per-token GDN scan step, per-head L2 normalize) are available. Required
    /// by the CUDA Qwen3MoeHybrid forward path.
    /// </summary>
    public bool HasGdnKernels =>
        _conv1dCausalF32Func != 0 && _gdnScanStepF32Func != 0 && _l2NormHeadsF32Func != 0;

    /// <summary>
    /// True when the fused GDN decay kernel (softplus + exp) is available on the
    /// loaded gated_delta_net_scan.ptx module. When false, callers must use the
    /// host-side fallback that D2Hs / H2Ds alpha.
    /// </summary>
    public bool HasGdnDecayF32 => _gdnDecayF32Func != 0;

    /// <summary>
    /// True when all three FP32 pointwise activations (sigmoid, silu, sigmoid_mul)
    /// are loaded from elementwise_f32.ptx. When false, the Qwen3MoeHybrid forward
    /// path must keep its host-side fallbacks.
    /// </summary>
    public bool HasElementwiseF32 =>
        _sigmoidF32Func != 0 && _siluF32Func != 0 && _sigmoidMulF32Func != 0;

    /// <summary>
    /// True when the gather-kernel de-interleave replacements for the hybrid-model decode
    /// host loops (Q+Gate split, GDN Q/K/V split) are loaded. When false, callers must fall
    /// back to the per-head/per-token <c>cuMemcpyDtoDAsync</c> loop.
    /// </summary>
    public bool HasDeinterleaveF32 =>
        _deinterleaveQGateF32Func != 0 && _deinterleaveGdnQkvF32Func != 0;

    /// <summary>True when the Phase-B Q2_K grouped-GEMV kernel is loaded (PTX present).</summary>
    /// <remarks>Set <see cref="DisableMoeGroupedGemv"/> to force the per-expert
    /// fallback for A/B comparison.</remarks>
    public bool HasMoeGroupedGemvQ2K =>
        _moeGroupedGemvQ2_KFunc != 0 && !DisableMoeGroupedGemv;

    /// <summary>True when the Phase-B Q4_K grouped-GEMV kernel is loaded (PTX present).</summary>
    /// <remarks>Set <see cref="DisableMoeGroupedGemv"/> to force the per-expert
    /// fallback for A/B comparison.</remarks>
    public bool HasMoeGroupedGemvQ4K =>
        _moeGroupedGemvQ4_KFunc != 0 && !DisableMoeGroupedGemv;

    /// <summary>True when the Phase-B Q5_K grouped-GEMV kernel is loaded (PTX present).</summary>
    public bool HasMoeGroupedGemvQ5K =>
        _moeGroupedGemvQ5_KFunc != 0 && !DisableMoeGroupedGemv;

    /// <summary>True when the Phase-B Q6_K grouped-GEMV kernel is loaded (PTX present).</summary>
    public bool HasMoeGroupedGemvQ6K =>
        _moeGroupedGemvQ6_KFunc != 0 && !DisableMoeGroupedGemv;

    /// <summary>True when the Phase-B Q8_0 grouped-GEMV kernel is loaded (PTX present).</summary>
    public bool HasMoeGroupedGemvQ8_0 =>
        _moeGroupedGemvQ8_0Func != 0 && !DisableMoeGroupedGemv;

    /// <summary>True when the MoE grouped IQ4_NL GEMV kernel is loaded and enabled.</summary>
    public bool HasMoeGroupedGemvIQ4_NL =>
        _moeGroupedGemvIQ4_NLFunc != 0 && !DisableMoeGroupedGemv;

    /// <summary>True when the MoE grouped IQ4_XS GEMV kernel is loaded and enabled.</summary>
    public bool HasMoeGroupedGemvIQ4_XS =>
        _moeGroupedGemvIQ4_XSFunc != 0 && !DisableMoeGroupedGemv;

    /// <summary>Disable the Phase-B grouped-GEMV path. Forces the per-expert
    /// <see cref="LaunchQuantizedGemv"/> fallback in <see cref="CudaMoeFfn"/>.</summary>
    public static bool DisableMoeGroupedGemv { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MOE_GROUPED_GEMV") == "1";

    /// <summary>Force direct GEMV path, disabling MMQ and MMVQ-Large variants.
    /// Useful for diagnostics to isolate dequant vs advanced GEMV kernels.
    /// Set environment variable DOTLLM_FORCE_DIRECT_GEMV=1 to enable.</summary>
    public static bool ForceDirectGemv { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_FORCE_DIRECT_GEMV") == "1";

    /// <summary>Force dequant-then-cuBLAS fallback instead of quantized GEMV.</summary>
    public static bool DisableQuantizedGemv { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_QUANTIZED_GEMV") == "1";

    /// <summary>
    /// Opt-in gate for persistently expanding a quant type CUDA has no native kernel for into
    /// a larger resident format (FP16/F32) kept for the model's entire lifetime. Default
    /// disabled: a checkpoint that can't actually be run in its compact form on this backend
    /// fails loudly at load time (see <see cref="EnsureQuantExpansionAllowed"/>) instead of
    /// silently ballooning VRAM and running a de-facto bigger, slower model than the one that
    /// was loaded. Set <c>DOTLLM_CUDA_ALLOW_QUANT_EXPANSION=1</c> to restore the old
    /// silent-fallback behavior.
    /// </summary>
    public static bool AllowQuantExpansion { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_CUDA_ALLOW_QUANT_EXPANSION") == "1";

    /// <summary>
    /// Call at every site that is about to fall back to a full, model-lifetime-resident
    /// dequant of <paramref name="qt"/> for <paramref name="tensorLabel"/> because no native
    /// CUDA kernel exists for it. Throws unless <see cref="AllowQuantExpansion"/> is set — see
    /// its remarks for why this defaults to a hard failure rather than a silent fallback.
    /// Not for the transient per-call scratch-buffer dequant used during prefill by types that
    /// DO have a native decode kernel — that path doesn't change the model's resident memory
    /// footprint and is not gated.
    /// </summary>
    /// <param name="qt">The quantization type with no native CUDA kernel.</param>
    /// <param name="tensorLabel">Human-readable tensor/layer identity for the error message.</param>
    /// <param name="compactBytes">Size of the tensor in its native on-disk quant format.</param>
    /// <param name="expandedBytes">Size of the tensor once expanded to the resident fallback format.</param>
    public static void EnsureQuantExpansionAllowed(
        QuantizationType qt, string tensorLabel, long compactBytes, long expandedBytes)
    {
        if (AllowQuantExpansion) return;
        throw new InvalidOperationException(
            $"CUDA has no native kernel for quantization type {qt} on '{tensorLabel}'. Running "
            + $"this model on CUDA as-is would silently expand this tensor from "
            + $"{FormatBytes(compactBytes)} (native quant, what you loaded) to "
            + $"{FormatBytes(expandedBytes)} (dequantized, resident for the model's whole "
            + "lifetime) — you would not be running the model you loaded, but a de-facto "
            + "larger, slower version of it, and every other tensor of this type in the model "
            + "would silently expand the same way. Set DOTLLM_CUDA_ALLOW_QUANT_EXPANSION=1 if "
            + "you explicitly want this fallback, or use a checkpoint requantized to a "
            + "CUDA-native type (F16, F32, BF16, Q8_0, Q2_K, Q4_K, Q5_0, Q5_K, Q6_K, IQ4_NL, "
            + "IQ4_XS, IQ2_XXS, IQ2_XS, IQ2_S, I2_S, or PQ2_0).");
    }

    private static string FormatBytes(long bytes) => bytes >= (1L << 30)
        ? $"{bytes / (double)(1L << 30):F2} GiB"
        : $"{bytes / (double)(1L << 20):F1} MiB";

    /// <summary>MMVQ-large block size (threads). MUST equal the compiled <c>MMVQ_LARGE_THREADS</c> in
    /// quantized_gemv_mmq.cu. Tunable via <c>DOTLLM_CUDA_MMVQ_THREADS</c> (32–256, multiple of 32; default
    /// 128) to sweep occupancy for the decode GEMV — for small-k models (k≤~3072) the default leaves most
    /// threads idle in the dp4a stage. When overriding, recompile the kernel with the matching #define.</summary>
    public static uint MmvqLargeThreads { get; set; } = ParseMmvqLargeThreads();

    private static uint ParseMmvqLargeThreads()
    {
        if (uint.TryParse(Environment.GetEnvironmentVariable("DOTLLM_CUDA_MMVQ_THREADS"), out uint t)
            && t >= 32 && t <= 256 && (t % 32) == 0)
            return t;
        return 128;
    }

    /// <summary>Opt-in: route Q4_K decode GEMV (k ≥ <see cref="MmvqLargeKThreshold"/>) through the
    /// experimental coalesced MMVQ kernel (warp-per-superblock, coalesced 128-byte qs loads) instead of
    /// the default MMVQ-large. On-the-fly only (ignores preq scratch). Set <c>DOTLLM_CUDA_MMVQ_COALESCED=1</c>.</summary>
    public static bool MmvqCoalesced { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_CUDA_MMVQ_COALESCED") == "1";

    /// <summary>Opt-in: route Q4_K decode GEMV (k ≥ <see cref="MmvqLargeKThreshold"/>) through the
    /// llama.cpp-faithful coalesced+accumulate-once MMVQ kernel (<c>quantized_gemv_q4_k_mmvq_llamacpp</c>)
    /// instead of the default MMVQ-large. Unlike <see cref="MmvqCoalesced"/> (#363, warp-per-superblock
    /// with a per-iteration 8-wide reduction — bit-exact but 3x SLOWER on Kaggle T4), this kernel keeps
    /// the coalesced 128-byte qs read per super-block but defers ALL cross-lane combination to a single
    /// end-of-row reduction, matching MMVQ-large's accumulate-once property. On-the-fly only (ignores
    /// preq scratch). Set <c>DOTLLM_CUDA_MMVQ_LLAMACPP=1</c>.</summary>
    public static bool MmvqLlamaCpp { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_CUDA_MMVQ_LLAMACPP") == "1";

    /// <summary>Disable decode-time packed QKV upload/dispatch for diagnostics.</summary>
    public static bool DisablePackedQkv { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_PACKED_QKV") == "1";

    /// <summary>Disable decode-time packed Gate/Up upload/dispatch for diagnostics.</summary>
    public static bool DisablePackedGateUp { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_PACKED_GATEUP") == "1";

    /// <summary>True when a grouped-GEMV kernel is available for the given quant type.</summary>
    public bool HasMoeGroupedGemv(QuantizationType qt) => qt switch
    {
        QuantizationType.Q2_K => HasMoeGroupedGemvQ2K,
        QuantizationType.Q4_K => HasMoeGroupedGemvQ4K,
        QuantizationType.Q5_K => HasMoeGroupedGemvQ5K,
        QuantizationType.Q6_K => HasMoeGroupedGemvQ6K,
        QuantizationType.Q8_0 => HasMoeGroupedGemvQ8_0,
        QuantizationType.IQ4_NL => HasMoeGroupedGemvIQ4_NL,
        QuantizationType.IQ4_XS => HasMoeGroupedGemvIQ4_XS,
        _ => false,
    };

    /// <summary>
    /// Opt a kernel into >48 KB dynamic shared memory (up to the device's optin cap).
    /// Silently skipped when func == 0 (kernel not loaded — TryGetFunction returned 0).
    /// Errors are non-fatal; the kernel will still launch as long as the launch's
    /// requested sharedMemBytes stays within the static 48 KB default.
    /// </summary>
    private static void SetMaxDynamicSharedBytes(nint func, int bytes)
    {
        if (func == 0) return;
        // Best effort — if the driver rejects the attribute (older driver, kernel
        // already too large for occupancy=1), we silently fall back to the default.
        // Launches that need more than the default will fail with a clear CUDA error.
        CudaDriverApi.cuFuncSetAttribute(func,
            CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes);
    }

    /// <summary>RMS normalization. One block per row.</summary>
    public void LaunchRmsNorm(nint input, nint weight, nint output,
                               int hiddenSize, float eps, int rows, nint stream)
    {
        nint inputArg = input, weightArg = weight, outputArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;

        void** args = stackalloc void*[] {&inputArg, &weightArg, &outputArg, &nArg, &epsArg};
        CudaDriverApi.cuLaunchKernel(_rmsnormFunc,
                (uint)rows, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused residual-add + RMS normalization. Avoids FP16 truncation at residual junction.</summary>
    /// <remarks>
    /// Allocates dynamic shared memory: <c>(hiddenSize + 32) * sizeof(float)</c> per block.
    /// hiddenSize rounded up to even is the FP32 sum cache; the trailing 32 floats are warp-reduce scratch (also stores rms_inv).
    /// For hidden=576 (SmolLM-135M), this is 2432 bytes — well within the SM86 default 48 KB budget.
    /// </remarks>
    public void LaunchFusedAddRmsNorm(nint residual, nint x, nint weight, nint output,
                                        int hiddenSize, float eps, int rows, nint stream)
    {
        nint resArg = residual, xArg = x, wArg = weight, outArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;

        void** args = stackalloc void*[] {&resArg, &xArg, &wArg, &outArg, &nArg, &epsArg};
        // Shared memory: n floats (sum cache, padded to even) + 32 floats (warp scratch)
        uint sharedBytes = (uint)((((hiddenSize + 1) & ~1) + 32) * sizeof(float));
        CudaDriverApi.cuLaunchKernel(_fusedAddRmsNormFunc,
                (uint)rows, 1, 1, BlockSize, 1, 1,
                sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Full FP32 RMS normalization: FP32 input, FP32 weight, FP32 output.</summary>
    public void LaunchRmsNormF32(nint input, nint weight, nint output,
                                   int hiddenSize, float eps, int rows, nint stream)
    {
        nint inputArg = input, weightArg = weight, outputArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;

        void** args = stackalloc void*[] {&inputArg, &weightArg, &outputArg, &nArg, &epsArg};
        CudaDriverApi.cuLaunchKernel(_rmsnormF32Func,
                (uint)rows, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// True when the fused copy+RMSNorm kernel (<see cref="LaunchCopyRmsNormF32"/>) is loaded.
    /// When false, callers fall back to a separate D2D copy + <see cref="LaunchRmsNormF32"/> pair.
    /// </summary>
    public bool HasCopyRmsNormF32 => _copyRmsNormF32Func != 0;

    /// <summary>
    /// Fused "copy <paramref name="input"/> to <paramref name="residualOut"/>, then RMSNorm it
    /// into <paramref name="output"/>" — replaces the decode-time pattern of a separate
    /// <c>cuMemcpyDtoDAsync</c> (save the pre-norm value for a later residual add) immediately
    /// followed by <see cref="LaunchRmsNormF32"/> on the same input, halving the launch count for
    /// that pair. Numerics are byte-for-byte identical to <see cref="LaunchRmsNormF32"/> — only
    /// the fused residual store is new. <paramref name="input"/>, <paramref name="residualOut"/>,
    /// and <paramref name="output"/> must be three distinct buffers (no in-place aliasing).
    /// </summary>
    public void LaunchCopyRmsNormF32(nint input, nint residualOut, nint weight, nint output,
                                       int hiddenSize, float eps, int rows, nint stream)
    {
        nint inputArg = input, residualArg = residualOut, weightArg = weight, outputArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;

        void** args = stackalloc void*[] {&inputArg, &residualArg, &weightArg, &outputArg, &nArg, &epsArg};
        CudaDriverApi.cuLaunchKernel(_copyRmsNormF32Func,
                (uint)rows, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Quantized GEMV with FP32 input: y_f32[n] = W_q8_0[n,k] @ x_f32[k].</summary>
    public void LaunchQuantizedGemvF32In(nint quantWeight, nint xF32, nint yF32,
                                            int n, int k, nint stream)
    {
        nint wArg = quantWeight, xArg = xF32, yArg = yF32;
        int nArg = n, kArg = k;

        void** args = stackalloc void*[] {&wArg, &xArg, &yArg, &nArg, &kArg};
        CudaDriverApi.cuLaunchKernel(_quantizedGemvQ8_0F32InFunc,
                (uint)n, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// I2_S ternary GEMV (W2A16): <c>y_f16[n] = scale · ternary(W[n,k]) @ x_f16[k]</c>.
    /// <paramref name="quantWeight"/> must point at the full I2_S tensor including the trailing
    /// per-tensor float32 scale at byte offset n·k/4 (the kernel reads it from the tail).
    /// </summary>
    public void LaunchI2_SGemvF16In(nint quantWeight, nint xF16, nint yF16, int n, int k, nint stream)
    {
        nint wArg = quantWeight, xArg = xF16, yArg = yF16;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] {&wArg, &xArg, &yArg, &nArg, &kArg};
        // Dynamic shared memory: x[k] staged as FP32 (see i2_s_gemv.cu). Sized per-call since k
        // varies by projection (e.g. FFN-down's k = intermediateSize, which can exceed the old
        // BitNet-2B-4T-specific static bound — see issue #207).
        uint dynShmem = (uint)k * sizeof(float);
        CheckDynamicSharedBudget(dynShmem, $"I2_S GEMV k={k}");
        // v2 warp-per-row: I2sRowsPerBlock output rows per 256-thread block → ceil(n / rows) blocks.
        CudaDriverApi.cuLaunchKernel(_i2sGemvF16InFunc,
                (uint)((n + I2sRowsPerBlock - 1) / I2sRowsPerBlock), 1, 1, BlockSize, 1, 1,
                dynShmem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused I2_S ternary GEMV for two projections sharing one FP16 input vector.</summary>
    public void LaunchI2_SGemv2F16In(
        nint quantWeight0, nint quantWeight1, nint xF16,
        nint yF16_0, nint yF16_1, int n0, int n1, int k, nint stream)
    {
        nint w0Arg = quantWeight0, w1Arg = quantWeight1, xArg = xF16;
        nint y0Arg = yF16_0, y1Arg = yF16_1;
        int n0Arg = n0, n1Arg = n1, kArg = k;
        int totalN = n0 + n1;
        uint grid = (uint)((totalN + I2sRowsPerBlock - 1) / I2sRowsPerBlock);

        void** args = stackalloc void*[]
        {
            &w0Arg, &w1Arg, &xArg, &y0Arg, &y1Arg, &n0Arg, &n1Arg, &kArg
        };
        uint dynShmem = (uint)k * sizeof(float);
        CheckDynamicSharedBudget(dynShmem, $"I2_S GEMV2 k={k}");
        CudaDriverApi.cuLaunchKernel(_i2sGemv2F16InFunc,
                grid, 1, 1, BlockSize, 1, 1,
                dynShmem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused I2_S ternary GEMV for three projections sharing one FP16 input vector.</summary>
    public void LaunchI2_SGemv3F16In(
        nint quantWeight0, nint quantWeight1, nint quantWeight2, nint xF16,
        nint yF16_0, nint yF16_1, nint yF16_2, int n0, int n1, int n2, int k, nint stream)
    {
        nint w0Arg = quantWeight0, w1Arg = quantWeight1, w2Arg = quantWeight2, xArg = xF16;
        nint y0Arg = yF16_0, y1Arg = yF16_1, y2Arg = yF16_2;
        int n0Arg = n0, n1Arg = n1, n2Arg = n2, kArg = k;
        int totalN = n0 + n1 + n2;
        uint grid = (uint)((totalN + I2sRowsPerBlock - 1) / I2sRowsPerBlock);

        void** args = stackalloc void*[]
        {
            &w0Arg, &w1Arg, &w2Arg, &xArg, &y0Arg, &y1Arg, &y2Arg,
            &n0Arg, &n1Arg, &n2Arg, &kArg
        };
        uint dynShmem = (uint)k * sizeof(float);
        CheckDynamicSharedBudget(dynShmem, $"I2_S GEMV3 k={k}");
        CudaDriverApi.cuLaunchKernel(_i2sGemv3F16InFunc,
                grid, 1, 1, BlockSize, 1, 1,
                dynShmem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>I2_S ternary GEMV whose FP16 input is RMS-normalized in-kernel before projection.</summary>
    public void LaunchI2_SGemvNormF16In(
        nint quantWeight, nint xF16, nint normWeightF16, nint yF16,
        int n, int k, float eps, nint stream)
    {
        nint wArg = quantWeight, xArg = xF16, normArg = normWeightF16, yArg = yF16;
        int nArg = n, kArg = k;
        float epsArg = eps;
        uint grid = (uint)((n + I2sRowsPerBlock - 1) / I2sRowsPerBlock);

        void** args = stackalloc void*[]
        {
            &wArg, &xArg, &normArg, &yArg, &nArg, &kArg, &epsArg
        };
        // Shared layout (i2_s_gemv.cu): xs[k] (even-aligned) + warp_sums[32] + rms_inv[1].
        uint dynShmem = (uint)(((k + 1) & ~1) + 33) * sizeof(float);
        CheckDynamicSharedBudget(dynShmem, $"I2_S GEMV-norm k={k}");
        CudaDriverApi.cuLaunchKernel(_i2sGemvNormF16InFunc,
                grid, 1, 1, BlockSize, 1, 1,
                dynShmem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>I2_S ternary GEMV with FP32 activations/output. Exact-match twin for CPU-vs-GPU tests.</summary>
    public void LaunchI2_SGemvF32In(nint quantWeight, nint xF32, nint yF32, int n, int k, nint stream)
    {
        nint wArg = quantWeight, xArg = xF32, yArg = yF32;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] {&wArg, &xArg, &yArg, &nArg, &kArg};
        uint dynShmem = (uint)k * sizeof(float);
        CheckDynamicSharedBudget(dynShmem, $"I2_S GEMV-f32 k={k}");
        CudaDriverApi.cuLaunchKernel(_i2sGemvF32InFunc,
                (uint)((n + I2sRowsPerBlock - 1) / I2sRowsPerBlock), 1, 1, BlockSize, 1, 1,
                dynShmem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Weight rows decoded per block for <see cref="LaunchI2_SGemmF32In"/> (issue #250), before
    /// any shared-memory-budget shrink. One warp owns exactly one row (no ROWS_PER_WARP multiplier —
    /// see i2_s_gemv.cu's Variant C remarks for why).</summary>
    private const int I2sGemmMaxRowsPerBlock = 8;

    /// <summary>
    /// Batched I2_S ternary GEMM (issue #250): <c>C[t,row] = scale · dot(ternary(W[row,:]), B[t,:])</c>
    /// for <c>row in [0,n)</c>, <c>t in [0,numTokens)</c>. <paramref name="bF32"/> is <c>[numTokens,k]</c>
    /// row-major; <paramref name="cF32"/> is <c>[numTokens,n]</c> row-major (token-major — matches
    /// <c>CudaMoeScratch.GateBatch</c>/<c>UpBatch</c>/<c>DownBatch</c>'s layout). Decodes each weight row
    /// ONCE (into a per-warp shared int8 cache) and reuses it across every token, instead of the
    /// per-row-GEMV-call loop <see cref="CudaMoeFfn"/> used before this issue — see
    /// <c>i2_s_gemm_f32in</c> in i2_s_gemv.cu for the full rationale.
    /// </summary>
    /// <remarks>
    /// <paramref name="numTokens"/> == 1 degrades to a plain <see cref="LaunchI2_SGemvF32In"/> call: decoding
    /// a row to shared just to dot it once is strictly extra work versus the single-pass GEMV kernel.
    /// Caller must check <see cref="HasI2SBatchedGemm"/> first (this method assumes the kernel is loaded).
    /// </remarks>
    public void LaunchI2_SGemmF32In(nint quantWeight, nint bF32, nint cF32, int n, int k, int numTokens, nint stream)
    {
        if (numTokens <= 0) return;
        if (numTokens == 1)
        {
            LaunchI2_SGemvF32In(quantWeight, bF32, cF32, n, k, stream);
            return;
        }

        // The shared row cache costs rowsPerBlock * k BYTES (int8) — shrink rowsPerBlock so this
        // fits the device's dynamic-shared opt-in cap for large k (e.g. big FFN intermediate sizes)
        // instead of failing the launch. Skipped (keeps the max) if the cap couldn't be queried,
        // matching CheckDynamicSharedBudget's own "unknown cap → don't block" convention below.
        int rowsPerBlock = I2sGemmMaxRowsPerBlock;
        if (_maxDynamicSharedBytesOptIn > 0)
        {
            int maxRowsForBudget = _maxDynamicSharedBytesOptIn / k;
            if (maxRowsForBudget < 1) maxRowsForBudget = 1;
            if (maxRowsForBudget < rowsPerBlock) rowsPerBlock = maxRowsForBudget;
        }

        nint wArg = quantWeight, bArg = bF32, cArg = cF32;
        int nArg = n, kArg = k, tArg = numTokens, rpbArg = rowsPerBlock;
        void** args = stackalloc void*[] {&wArg, &bArg, &cArg, &nArg, &kArg, &tArg, &rpbArg};
        uint dynShmem = (uint)(rowsPerBlock * k); // int8 ternary cache, 1 byte/element
        CheckDynamicSharedBudget(dynShmem, $"I2_S GEMM-f32 k={k} rowsPerBlock={rowsPerBlock}");
        CudaDriverApi.cuLaunchKernel(_i2sGemmF32InFunc,
                (uint)((n + rowsPerBlock - 1) / rowsPerBlock), 1, 1, BlockSize, 1, 1,
                dynShmem, stream, (nint)args, 0).ThrowOnError();
    }

    // Ragged K (k % 128 != 0) — issue #206. i2_s_gemv_{f16,f32}in_ragged use ONE warp per row
    // (8 rows/block, fixed — not I2sRowsPerBlock, which encodes the aligned kernel's
    // ROWS_PER_WARP=2 tuning) since they don't share the aligned kernels' uint4/shared-block
    // launch contract. See i2_s_gemv.cu's issue #206 comment for why a ragged row cannot reuse the
    // aligned addressing even for a "tail cleanup".
    private const int I2sRaggedRowsPerBlock = 8;

    /// <summary>
    /// Ragged-K (<c>k % 128 != 0</c>) twin of <see cref="LaunchI2_SGemvF16In"/>. Scalar
    /// correctness-first fallback — see <c>i2_s_gemv_f16in_ragged</c> in i2_s_gemv.cu.
    /// </summary>
    public void LaunchI2_SGemvF16InRagged(nint quantWeight, nint xF16, nint yF16, int n, int k, nint stream)
    {
        nint wArg = quantWeight, xArg = xF16, yArg = yF16;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] {&wArg, &xArg, &yArg, &nArg, &kArg};
        // Dynamic shared memory (issue #207 fix applied here too — the ragged kernel's x-staging
        // buffer used the same fixed BitNet-2B-4T-sized static array as the aligned kernels).
        uint dynShmem = (uint)k * sizeof(float);
        CheckDynamicSharedBudget(dynShmem, $"I2_S GEMV ragged k={k}");
        CudaDriverApi.cuLaunchKernel(_i2sGemvF16InRaggedFunc,
                (uint)((n + I2sRaggedRowsPerBlock - 1) / I2sRaggedRowsPerBlock), 1, 1, BlockSize, 1, 1,
                dynShmem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Ragged-K (<c>k % 128 != 0</c>) twin of <see cref="LaunchI2_SGemvF32In"/>.</summary>
    public void LaunchI2_SGemvF32InRagged(nint quantWeight, nint xF32, nint yF32, int n, int k, nint stream)
    {
        nint wArg = quantWeight, xArg = xF32, yArg = yF32;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] {&wArg, &xArg, &yArg, &nArg, &kArg};
        uint dynShmem = (uint)k * sizeof(float);
        CheckDynamicSharedBudget(dynShmem, $"I2_S GEMV ragged-f32 k={k}");
        CudaDriverApi.cuLaunchKernel(_i2sGemvF32InRaggedFunc,
                (uint)((n + I2sRaggedRowsPerBlock - 1) / I2sRaggedRowsPerBlock), 1, 1, BlockSize, 1, 1,
                dynShmem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// I2_S ternary GEMV (W2A8, <c>__dp4a</c>): <c>y_f32[n] = scale · invActScale · int8dot(W[n,k], xq[k])</c>.
    /// Activations <paramref name="xqInt8"/> must be quantized per token (symmetric absmax,
    /// s_act = 127/absmax(x), xq_i = round(x_i·s_act)); <paramref name="invActScale"/> = absmax(x)/127
    /// = 1/s_act so x_i ≈ xq_i·invActScale. Output is FP32.
    /// <paramref name="quantWeight"/> must point at the full I2_S tensor including the trailing
    /// per-tensor float32 scale at byte offset n·k/4 (the kernel reads it from the tail).
    /// Requires sm_61+ (dotLLM builds compute_61). Validate against the int8 reference, not the
    /// float CPU path — the dp4a-vs-float diff is expected activation-quant error.
    /// </summary>
    public void LaunchI2_SGemvA8(nint quantWeight, nint xqInt8, nint yF32, int n, int k,
                                  float invActScale, nint stream)
    {
        nint wArg = quantWeight, xArg = xqInt8, yArg = yF32;
        int nArg = n, kArg = k;
        float sArg = invActScale;
        void** args = stackalloc void*[] {&wArg, &xArg, &yArg, &nArg, &kArg, &sArg};
        CudaDriverApi.cuLaunchKernel(_i2sGemvA8Func,
                (uint)((n + I2sRowsPerBlock - 1) / I2sRowsPerBlock), 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Quantizes one FP16 activation vector to int8 with symmetric absmax scaling.</summary>
    public void LaunchQuantizeF16ToI8AbsMax(nint xF16, nint xqInt8, nint invActScale, int k, nint stream)
    {
        nint xArg = xF16, xqArg = xqInt8, scaleArg = invActScale;
        int kArg = k;
        void** args = stackalloc void*[] {&xArg, &xqArg, &scaleArg, &kArg};
        CudaDriverApi.cuLaunchKernel(_quantizeF16ToI8AbsMaxFunc,
                1, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>I2_S W2A8 GEMV using a device-resident activation inverse scale for graph replay.</summary>
    public void LaunchI2_SGemvA8DeviceScale(nint quantWeight, nint xqInt8, nint yF32, int n, int k,
                                             nint invActScale, nint stream)
    {
        nint wArg = quantWeight, xArg = xqInt8, yArg = yF32, scaleArg = invActScale;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] {&wArg, &xArg, &yArg, &nArg, &kArg, &scaleArg};
        CudaDriverApi.cuLaunchKernel(_i2sGemvA8DeviceScaleFunc,
                (uint)((n + I2sRowsPerBlock - 1) / I2sRowsPerBlock), 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Dequantizes an I2_S (BitNet ternary) weight matrix to dense FP16 on the GPU for prefill GEMM.
    /// <c>dst[row·k + col] = (code(W[row,col]) - 1) · scale</c>. Unlike the generic
    /// <see cref="LaunchDequantToF16"/> (keyed on total element count), this needs <paramref name="n"/>
    /// and <paramref name="k"/> to locate the per-tensor float32 scale at the tensor tail (offset n·k/4).
    /// <paramref name="src"/> must point at the full I2_S tensor including the trailing scale.
    /// </summary>
    public void LaunchDequantI2_SToF16(nint src, nint dst, int n, int k, nint stream)
    {
        nint srcArg = src, dstArg = dst;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] {&srcArg, &dstArg, &nArg, &kArg};

        long totalBlocks = (long)n * (k / 128);
        int warpsPerBlock = BlockSize / 32;
        uint gridDim = (uint)Math.Min((totalBlocks + warpsPerBlock - 1) / warpsPerBlock, MaxDequantGridSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_dequantI2sF16Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Ragged-K (<c>k % 128 != 0</c>) twin of <see cref="LaunchDequantI2_SToF16"/>. The aligned
    /// kernel's <c>blocks_per_row = k/128</c> integer division silently drops each row's tail
    /// elements for a ragged k, so this dispatches a one-thread-per-output-element grid-stride
    /// kernel (<c>dequant_i2_s_f16_ragged</c>) that derives each element's block/bit address
    /// directly from its flattened index instead. See issue #206.
    /// </summary>
    public void LaunchDequantI2_SToF16Ragged(nint src, nint dst, int n, int k, nint stream)
    {
        nint srcArg = src, dstArg = dst;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] {&srcArg, &dstArg, &nArg, &kArg};

        long totalElems = (long)n * k;
        uint gridDim = (uint)Math.Min((totalElems + BlockSize - 1) / BlockSize, MaxDequantGridSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_dequantI2sF16RaggedFunc,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// PQ2_0 (PrismML Bonsai ternary) GEMV with FP32 activations/output. Exact-match twin for
    /// CPU-vs-GPU tests (see the CPU reference, MatMul.GemvPQ2_0). Correctness-first: one warp
    /// per output row, grid-strided over rows (unlike I2_S's fixed rows-per-block launch
    /// contract, this kernel loops internally so any grid size is correct).
    /// </summary>
    public void LaunchPQ2_0GemvF32In(nint quantWeight, nint xF32, nint yF32, int n, int k, nint stream)
    {
        nint wArg = quantWeight, xArg = xF32, yArg = yF32;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] {&wArg, &xArg, &yArg, &nArg, &kArg};

        int warpsPerBlock = BlockSize / 32;
        uint gridDim = (uint)Math.Min((n + warpsPerBlock - 1) / warpsPerBlock, MaxDequantGridSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_pq2_0GemvF32InFunc,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// PQ2_0 ternary GEMV with FP16 activations/output — the production decode path. v2:
    /// shared-x staging (half-width) + warp-per-row (<see cref="Pq2_0RowsPerBlock"/> rows/block),
    /// mirroring I2_S's proven scheme — see native/kernels/pq2_0_gemv.cu's file header for the
    /// full rationale. Grid is uncapped (not <see cref="MaxDequantGridSize"/>-limited like the
    /// v1 grid-stride kernel was) since each block now covers a fixed row count.
    /// Routes to the <c>_small</c> kernel variant when <paramref name="k"/> &lt;=
    /// <see cref="Pq2_0MaxKSmall"/> — a smaller static shared-memory footprint raises the
    /// occupancy ceiling for the attention/GDN-shaped calls (see pq2_0_gemv.cu's "Small-K
    /// specialization" comment for the arithmetic). Transparent to callers: same signature,
    /// same grid/block sizing (both variants share <see cref="Pq2_0RowsPerBlock"/>/<see cref="BlockSize"/>).
    /// </summary>
    public void LaunchPQ2_0GemvF16In(nint quantWeight, nint xF16, nint yF16, int n, int k, nint stream)
    {
        nint wArg = quantWeight, xArg = xF16, yArg = yF16;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] {&wArg, &xArg, &yArg, &nArg, &kArg};

        uint gridDim = (uint)((n + Pq2_0RowsPerBlock - 1) / Pq2_0RowsPerBlock);
        if (gridDim == 0) gridDim = 1;

        nint func = k <= Pq2_0MaxKSmall ? _pq2_0GemvF16InSmallFunc : _pq2_0GemvF16InFunc;
        CudaDriverApi.cuLaunchKernel(func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Fused PQ2_0 ternary GEMV for two projections sharing one FP16 input vector (e.g. dense
    /// FFN gate+up, or full-attention K+V) — mirrors <see cref="LaunchI2_SGemv2F16In"/>. Stages
    /// x into shared memory once and reuses it across the virtual row-concatenation of both
    /// weight matrices, instead of two separate launches each re-staging x.
    /// Routes to the <c>_small</c> kernel variant when <paramref name="k"/> &lt;=
    /// <see cref="Pq2_0MaxKSmall"/> — see <see cref="LaunchPQ2_0GemvF16In"/>'s doc comment for the
    /// occupancy rationale (identical here; both fused call sites, alpha/beta and K+V, share k=5120).
    /// </summary>
    public void LaunchPQ2_0Gemv2F16In(
        nint quantWeight0, nint quantWeight1, nint xF16,
        nint yF16_0, nint yF16_1, int n0, int n1, int k, nint stream)
    {
        nint w0Arg = quantWeight0, w1Arg = quantWeight1, xArg = xF16;
        nint y0Arg = yF16_0, y1Arg = yF16_1;
        int n0Arg = n0, n1Arg = n1, kArg = k;
        int totalN = n0 + n1;
        uint grid = (uint)((totalN + Pq2_0RowsPerBlock - 1) / Pq2_0RowsPerBlock);
        if (grid == 0) grid = 1;

        void** args = stackalloc void*[]
        {
            &w0Arg, &w1Arg, &xArg, &y0Arg, &y1Arg, &n0Arg, &n1Arg, &kArg
        };
        nint func = k <= Pq2_0MaxKSmall ? _pq2_0Gemv2F16InSmallFunc : _pq2_0Gemv2F16InFunc;
        CudaDriverApi.cuLaunchKernel(func,
                grid, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// PQ2_0 ternary GEMV with FP32-native activations/output — the decode path used by
    /// <see cref="DotLLM.Cuda.Architectures.CudaQwen3HybridDenseTransformerModel"/> (issue #161).
    /// Converts F32&lt;-&gt;F16 inline in the kernel's own vectorized activation-staging/output-store
    /// steps (see native/kernels/pq2_0_gemv.cu's "F32-native activations" file-header section) —
    /// no surrounding <see cref="LaunchConvertF32ToF16"/>/<see cref="LaunchConvertF16ToF32"/>
    /// launches or F16 scratch-buffer round-trip needed at the call site, unlike
    /// <see cref="LaunchPQ2_0GemvF16In"/>. Same dispatch-by-k routing to the <c>_small</c> variant
    /// as <see cref="LaunchPQ2_0GemvF16In"/> — see that method's doc comment for the occupancy
    /// rationale (identical here; only the activation pointer types/conversion site differ).
    /// </summary>
    public void LaunchPQ2_0GemvF32Native(nint quantWeight, nint xF32, nint yF32, int n, int k, nint stream)
    {
        nint wArg = quantWeight, xArg = xF32, yArg = yF32;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] {&wArg, &xArg, &yArg, &nArg, &kArg};

        uint gridDim = (uint)((n + Pq2_0RowsPerBlock - 1) / Pq2_0RowsPerBlock);
        if (gridDim == 0) gridDim = 1;

        nint func = k <= Pq2_0MaxKSmall ? _pq2_0GemvF32IoSmallFunc : _pq2_0GemvF32IoFunc;
        CudaDriverApi.cuLaunchKernel(func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Fused PQ2_0 ternary GEMV for two projections sharing one FP32-native input vector — the
    /// FP32-native analog of <see cref="LaunchPQ2_0Gemv2F16In"/> (issue #161), used by
    /// <see cref="DotLLM.Cuda.Architectures.CudaQwen3HybridDenseTransformerModel.TryFusedPQ2_0Gemm2"/>.
    /// See <see cref="LaunchPQ2_0GemvF32Native"/>'s doc comment for the no-convert-launch rationale.
    /// </summary>
    public void LaunchPQ2_0Gemv2F32Native(
        nint quantWeight0, nint quantWeight1, nint xF32,
        nint yF32_0, nint yF32_1, int n0, int n1, int k, nint stream)
    {
        nint w0Arg = quantWeight0, w1Arg = quantWeight1, xArg = xF32;
        nint y0Arg = yF32_0, y1Arg = yF32_1;
        int n0Arg = n0, n1Arg = n1, kArg = k;
        int totalN = n0 + n1;
        uint grid = (uint)((totalN + Pq2_0RowsPerBlock - 1) / Pq2_0RowsPerBlock);
        if (grid == 0) grid = 1;

        void** args = stackalloc void*[]
        {
            &w0Arg, &w1Arg, &xArg, &y0Arg, &y1Arg, &n0Arg, &n1Arg, &kArg
        };
        nint func = k <= Pq2_0MaxKSmall ? _pq2_0Gemv2F32IoSmallFunc : _pq2_0Gemv2F32IoFunc;
        CudaDriverApi.cuLaunchKernel(func,
                grid, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Dequantizes a PQ2_0 (PrismML Bonsai ternary) weight matrix to dense FP16 on the GPU for
    /// prefill GEMM. <c>dst[row·k + col] = (code(W[row,col]) - 1) · group_scale</c>. Unlike I2_S,
    /// the scale is per-128-element-group (not per-tensor), so no tensor-tail offset is read —
    /// <paramref name="src"/> points at the packed row-major PQ2_0 payload only. v2: one WARP
    /// decodes each group (coalesced byte reads/writes) — see dequant_pq2_0.cu's v2 comment.
    /// </summary>
    public void LaunchDequantPQ2_0ToF16(nint src, nint dst, int n, int k, nint stream)
    {
        nint srcArg = src, dstArg = dst;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] {&srcArg, &dstArg, &nArg, &kArg};

        long totalGroups = (long)n * (k / 128);
        int warpsPerBlock = BlockSize / 32;
        uint gridDim = (uint)Math.Min((totalGroups + warpsPerBlock - 1) / warpsPerBlock, MaxDequantGridSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_dequantPQ2_0F16Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// One-time PQ2_0 weight repack: reorders a tensor from dotLLM's interleaved on-disk layout
    /// (per-group scale immediately followed by that group's 32 code bytes) into the split layout
    /// <see cref="LaunchPQ2_0GemvF16In"/>/<see cref="LaunchPQ2_0Gemv2F16In"/>/
    /// <see cref="LaunchDequantPQ2_0ToF16"/> now expect (all scales first, then all codes — see
    /// native/kernels/pq2_0_repack.cu's file header for the exact byte layout and the
    /// round-up-to-32 alignment rationale). Load-time only, never called on the decode hot path —
    /// see <c>CudaQwen3HybridDenseTransformerModel.UploadRawTensor</c> for the call site.
    /// <paramref name="split"/> must be sized <c>PQ2_0SplitLayoutBytes(n, k)</c>-worth of bytes
    /// (codes-region start rounded up to 32 — a few bytes larger than the interleaved source in
    /// the worst case, never smaller).
    /// </summary>
    public void LaunchPQ2_0RepackSplitF16(nint interleaved, nint split, int n, int k, nint stream)
    {
        nint srcArg = interleaved, dstArg = split;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] {&srcArg, &dstArg, &nArg, &kArg};

        long totalGroups = (long)n * (k / 128);
        int warpsPerBlock = BlockSize / 32;
        uint gridDim = (uint)Math.Min((totalGroups + warpsPerBlock - 1) / warpsPerBlock, MaxDequantGridSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_pq2_0RepackSplitF16Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Byte size of the split-layout buffer <see cref="LaunchPQ2_0RepackSplitF16"/> writes into,
    /// for a PQ2_0 tensor with <paramref name="n"/> rows and row length <paramref name="k"/>.
    /// Codes region start is rounded up to 32 bytes (see native/kernels/pq2_0_repack.cu's file
    /// header) so total size is at most 31 bytes larger than the interleaved source
    /// (<c>n * (k/128) * 34</c>), never smaller.
    /// </summary>
    public static long PQ2_0SplitLayoutBytes(int n, int k)
    {
        long totalGroups = (long)n * (k / 128);
        long codesBaseOffset = (totalGroups * sizeof(ushort) + 31) & ~31L;
        return codesBaseOffset + totalGroups * 32;
    }

    /// <summary>Fused squared-ReLU GLU (BitNet): <c>out = relu(gate)² · up</c>. FP16, half2 vectorized.</summary>
    public void LaunchReLU2(nint gate, nint up, nint output, int n, int seqLen, nint stream)
    {
        nint gateArg = gate, upArg = up, outArg = output;
        int nArg = n, slArg = seqLen;

        void** args = stackalloc void*[] {&gateArg, &upArg, &outArg, &nArg, &slArg};
        int total = n * seqLen;
        uint gridDim = (uint)((total / 2 + BlockSize - 1) / BlockSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_relu2Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused squared-ReLU GLU (BitNet), all FP32: <c>out = relu(gate)² · up</c>.</summary>
    public void LaunchReLU2F32(nint gate, nint up, nint output, int n, int seqLen, nint stream)
    {
        nint gateArg = gate, upArg = up, outArg = output;
        int nArg = n, slArg = seqLen;

        void** args = stackalloc void*[] {&gateArg, &upArg, &outArg, &nArg, &slArg};
        uint gridDim = (uint)((n * seqLen + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_relu2F32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Fused BitNet FFN: squared-ReLU GLU + Sub-LN RMSNorm. <c>out = rmsnorm(relu(gate)²·up) · weight</c>.
    /// The large pre-norm intermediate (relu² amplifies values past FP16 range) is held in FP32 and
    /// never materialized to FP16; only the normalized O(1) result is stored. One block per token row.
    /// </summary>
    public void LaunchReLU2GluRmsNorm(nint gate, nint up, nint weight, nint output,
                                        int n, float eps, int seqLen, nint stream)
    {
        nint gateArg = gate, upArg = up, wArg = weight, outArg = output;
        int nArg = n, slArg = seqLen;
        float epsArg = eps;

        void** args = stackalloc void*[] {&gateArg, &upArg, &wArg, &outArg, &nArg, &epsArg, &slArg};
        CudaDriverApi.cuLaunchKernel(_relu2GluRmsNormFunc,
                (uint)seqLen, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused residual-add + RMSNorm with an FP32 residual stream (BitNet, whose residual
    /// magnitude exceeds FP16 range). Residual is FP32 in/out; x/weight/output are FP16.</summary>
    public void LaunchFusedAddRmsNormF32Res(nint residualF32, nint xF16, nint weightF16, nint outputF16,
                                              int hiddenSize, float eps, int rows, nint stream)
    {
        nint resArg = residualF32, xArg = xF16, wArg = weightF16, outArg = outputF16;
        int nArg = hiddenSize;
        float epsArg = eps;

        void** args = stackalloc void*[] {&resArg, &xArg, &wArg, &outArg, &nArg, &epsArg};
        CudaDriverApi.cuLaunchKernel(_fusedAddRmsNormF32ResFunc,
                (uint)rows, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Widen an FP16 buffer to FP32 (seeds the FP32 residual from the FP16 embedding).</summary>
    public void LaunchCopyF16ToF32(nint srcF16, nint dstF32, int n, nint stream)
    {
        nint srcArg = srcF16, dstArg = dstF32;
        int nArg = n;
        void** args = stackalloc void*[] {&srcArg, &dstArg, &nArg};
        uint gridDim = (uint)((n + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_copyF16ToF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Final residual add with FP32 residual: <c>output_f16 = FP16(residual_f32 + x_f16)</c>.</summary>
    public void LaunchAddF32ResF16(nint residualF32, nint xF16, nint outputF16, int n, nint stream)
    {
        nint resArg = residualF32, xArg = xF16, outArg = outputF16;
        int nArg = n;
        void** args = stackalloc void*[] {&resArg, &xArg, &outArg, &nArg};
        uint gridDim = (uint)((n + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_addF32ResF16Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>RMSNorm with FP32 input, FP16 weight, FP16 output. For the BitNet final norm over the
    /// FP32 residual (norm weights are uploaded as FP16, unlike <see cref="LaunchRmsNormF32In"/>).</summary>
    public void LaunchRmsNormF32InF16W(nint inputF32, nint weightF16, nint outputF16,
                                         int hiddenSize, float eps, int rows, nint stream)
    {
        nint inputArg = inputF32, weightArg = weightF16, outputArg = outputF16;
        int nArg = hiddenSize;
        float epsArg = eps;
        void** args = stackalloc void*[] {&inputArg, &weightArg, &outputArg, &nArg, &epsArg};
        CudaDriverApi.cuLaunchKernel(_rmsNormF32InF16WFunc,
                (uint)rows, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>RMS normalization with FP32 input, FP32 weight, FP16 output. For FP32 residual stream.</summary>
    public void LaunchRmsNormF32In(nint input, nint weight, nint output,
                                     int hiddenSize, float eps, int rows, nint stream)
    {
        nint inputArg = input, weightArg = weight, outputArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;

        void** args = stackalloc void*[] {&inputArg, &weightArg, &outputArg, &nArg, &epsArg};
        CudaDriverApi.cuLaunchKernel(_rmsnormF32InF16OutFunc,
                (uint)rows, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>FP32 element-wise add: output_f32 = a_f32 + b_f32.</summary>
    public void LaunchAddF32(nint a, nint b, nint output, int n, nint stream)
    {
        nint aArg = a, bArg = b, outArg = output;
        int nArg = n;

        void** args = stackalloc void*[] {&aArg, &bArg, &outArg, &nArg};
        uint gridDim = (uint)((n + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_addF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Mixed add: output_f32 = a_f32 + b_f16. For adding FP16 projection output into FP32 residual.</summary>
    public void LaunchAddF32F16(nint aF32, nint bF16, nint outputF32, int n, nint stream)
    {
        nint aArg = aF32, bArg = bF16, outArg = outputF32;
        int nArg = n;

        void** args = stackalloc void*[] {&aArg, &bArg, &outArg, &nArg};
        uint gridDim = (uint)((n + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_addF32F16Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Embedding lookup with FP32 output for the residual stream.</summary>
    public void LaunchEmbeddingLookupF32(nint embedTable, QuantizationType embedDtype,
                                           nint tokenIds, nint output,
                                           int seqLen, int hiddenSize, nint stream)
    {
        nint tableArg = embedTable, idsArg = tokenIds, outArg = output;
        int slArg = seqLen, hsArg = hiddenSize;

        nint func = embedDtype switch
        {
            QuantizationType.F32 => _embeddingF32OutF32Func,
            QuantizationType.F16 => _embeddingF32OutF16Func,
            QuantizationType.Q8_0 => _embeddingF32OutQ8_0Func,
            _ => throw new NotSupportedException($"FP32 embedding lookup not supported for {embedDtype}.")
        };

        void** args = stackalloc void*[] {&tableArg, &idsArg, &outArg, &slArg, &hsArg};

        CudaDriverApi.cuLaunchKernel(func,
                (uint)seqLen, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Translates the public C# <see cref="DotLLM.Core.Configuration.RoPEType"/>
    /// enum to the integer encoding the CUDA RoPE kernels expect.
    /// </summary>
    /// <remarks>
    /// The C# enum is a public API surface (used by <c>RoPEConfig</c>) and
    /// historically encodes <c>Norm = 0</c>, <c>NeoX = 2</c>. The CUDA kernels
    /// in <c>native/kernels/rope_f32.cu</c>, <c>rope.cu</c>, and
    /// <c>fused_rope_kv_write.cu</c> encode the pair pattern as <c>0 = GPT-J /
    /// Norm interleaved</c>, <c>1 = NeoX / rotate-half</c>. Casting the C#
    /// enum value straight to <see cref="int"/> would feed <c>NeoX == 2</c>
    /// into the kernel which falls into the "anything but 1 → interleaved"
    /// branch — silently running Qwen / Phi (NeoX) models with the GPT-J
    /// rotation pattern. This translator is the single chokepoint that maps
    /// between the two conventions. The Vulkan and CPU paths consume the
    /// <see cref="DotLLM.Core.Configuration.RoPEType"/> enum directly so this
    /// helper is CUDA-only.
    /// </remarks>
    /// <param name="type">The element-pairing convention from <see cref="DotLLM.Core.PositionEncoding.RoPEConfig"/>.</param>
    /// <returns><c>0</c> for <see cref="DotLLM.Core.Configuration.RoPEType.Norm"/>, <c>1</c> for <see cref="DotLLM.Core.Configuration.RoPEType.NeoX"/>.</returns>
    public static int ToCudaRopeType(DotLLM.Core.Configuration.RoPEType type) => type switch
    {
        DotLLM.Core.Configuration.RoPEType.NeoX => 1,
        _ => 0,
    };

    /// <summary>
    /// FP32 RoPE: in-place rotation on FP32 Q and K. <paramref name="freqDim"/> is the
    /// frequency-denominator dim for the exponent <c>2*pair/freqDim</c>: pass 0 (default) to use
    /// <paramref name="ropeDim"/> (full rotation + standard partial NeoX); for Gemma-4 partial
    /// global layers pass the FULL head dim so the exponent matches the CPU/Vulkan oracle.
    /// <paramref name="neoxPairOffset"/> is the NeoX rotate-half pairing offset: pass 0 (default)
    /// for the standard <c>ropeDim/2</c> (Qwen3 / NemotronH / Llama — matches CPU
    /// <c>RoPE.Execute</c>); pass <c>headDim/2</c> for Gemma-4 partial global layers (matches CPU
    /// <c>RoPE.ApplyRotationNeoXPartial</c>).
    /// </summary>
    public void LaunchRoPEF32(nint q, nint k, nint positions,
                                int seqLen, int numHeads, int numKvHeads, int headDim,
                                int ropeDim, float theta, int ropeType, nint stream,
                                int freqDim = 0, int neoxPairOffset = 0)
    {
        nint qArg = q, kArg = k, posArg = positions;
        int slArg = seqLen, nhArg = numHeads, nkvArg = numKvHeads;
        int hdArg = headDim, rdArg = ropeDim, rtArg = ropeType;
        int fdArg = freqDim; // 0 ⇒ kernel falls back to rope_dim
        int npoArg = neoxPairOffset; // 0 ⇒ kernel falls back to rope_dim/2 (standard)
        float thetaArg = theta;

        void** args = stackalloc void*[] {&qArg, &kArg, &posArg, &slArg, &nhArg, &nkvArg,
                        &hdArg, &rdArg, &thetaArg, &rtArg, &fdArg, &npoArg};

        int halfRope = ropeDim / 2;
        int totalPairs = seqLen * Math.Max(numHeads, numKvHeads) * halfRope;
        uint gridDim = (uint)((totalPairs + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_ropeF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>FP32 attention: Q/K/V/output all FP32.</summary>
    public void LaunchAttentionF32(nint q, nint k, nint v, nint output,
                                     int seqQ, int seqKv,
                                     int numHeads, int numKvHeads, int headDim,
                                     int positionOffset, int slidingWindow, nint stream)
    {
        nint qArg = q, kArg = k, vArg = v, outArg = output;
        int sqArg = seqQ, skvArg = seqKv;
        int nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim;
        int poArg = positionOffset, swArg = slidingWindow;

        void** args = stackalloc void*[] {&qArg, &kArg, &vArg, &outArg,
                        &sqArg, &skvArg, &nhArg, &nkvArg, &hdArg,
                        &poArg, &swArg};

        int numBlocks = seqQ * numHeads;
        // Tiled online softmax: q_shared[headDim] + score_tile[256] + out_accum[headDim] + warp_scratch[32]
        const int TileKv = 256;
        uint sharedBytes = (uint)((headDim + TileKv + headDim + 32) * sizeof(float));

        CudaDriverApi.cuLaunchKernel(_attentionF32Func,
                (uint)numBlocks, 1, 1, BlockSize, 1, 1,
                sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Number of cooperating blocks per head in <see cref="LaunchAttentionF32SplitKv"/> — must match
    /// <c>ATTN_KV_SPLIT</c> in <c>attention_f32.cu</c>.</summary>
    public const int AttentionKvSplit = 4;

    /// <summary>
    /// Whether the opt-in split-KV ("Flash-Decoding") <c>attention_f32_split_kv</c> kernel is
    /// present in the loaded PTX (issue #183). Presence alone does not mean it's safe to launch —
    /// call <see cref="IsAttentionSplitKvSafe"/> first.
    /// </summary>
    public bool HasAttentionF32SplitKv => _attentionF32SplitKvFunc != 0;

    /// <summary>
    /// Opt-in (default OFF) env-var gate for <see cref="LaunchAttentionF32SplitKv"/>. <b>Enabling
    /// this trades away attention_f32's bit-near-CPU-equivalent output</b> for a reassociated
    /// (mathematically equal, not bit-identical) cross-block combine — same tradeoff shape as
    /// <see cref="EnableGdnScanApproxSplit4"/> (issue #180), but see
    /// <c>attention_f32.cu</c>'s header for why this kernel's error-compounding story across decode
    /// steps is expected to differ from GDN's recurrent-state case (attention reads the exact,
    /// unperturbed KV cache each step — it does not carry a persistent approximate state forward).
    /// Off by default per this project's Correctness-then-Performance priority order.
    /// </summary>
    public static bool EnableAttentionSplitKv { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_ATTN_SPLIT_KV") == "1";

    /// <summary>
    /// Minimum <c>seqKv</c> (cached KV length) before the split-KV kernel is worth its grid.sync +
    /// combine overhead versus the exact single-block kernel — below this, each split would save
    /// only a few iterations of the per-tile weighted-V accumulation loop, not worth 4x the blocks
    /// plus a grid-wide barrier. Tunable; not read from an env var (only the on/off gate is).
    /// </summary>
    public static int AttentionSplitKvMinSeqKv { get; set; } = 256;

    /// <summary>
    /// Queries (once per distinct <paramref name="headDim"/>, cached) whether split-KV cooperative
    /// launch is safe for this (numHeads, headDim) shape on THIS GPU — i.e. whether
    /// <c>numHeads*4</c> blocks can be simultaneously co-resident, a hard requirement of
    /// <c>cuLaunchCooperativeKernel</c>. Returns false (safe default: caller falls back to
    /// <see cref="LaunchAttentionF32"/>) if the split kernel isn't loaded, cooperative launch isn't
    /// supported on this device/driver, or the shape doesn't fit.
    /// </summary>
    public bool IsAttentionSplitKvSafe(int numHeads, int headDim)
    {
        if (_attentionF32SplitKvFunc == 0) return false;

        if (_attnSplitKvMaxCoResidentGrid < 0 || _attnSplitKvCachedHeadDim != headDim)
        {
            CudaDriverApi.cuCtxGetDevice(out int device);
            int coopSupported = 0;
            CudaDriverApi.cuDeviceGetAttribute(out coopSupported,
                CudaDriverApi.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device);
            if (coopSupported == 0)
            {
                _attnSplitKvMaxCoResidentGrid = 0;
            }
            else
            {
                CudaDriverApi.cuDeviceGetAttribute(out int numSms,
                    CudaDriverApi.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
                const int TileKv = 256;
                uint sharedBytesForQuery = (uint)((headDim + TileKv + headDim + 32) * sizeof(float));
                int rc = CudaDriverApi.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    out int maxBlocksPerSm, _attentionF32SplitKvFunc, BlockSize, sharedBytesForQuery);
                _attnSplitKvMaxCoResidentGrid = rc == 0 ? numSms * maxBlocksPerSm : 0;
            }
            _attnSplitKvCachedHeadDim = headDim;
        }

        return _attnSplitKvMaxCoResidentGrid > 0 && numHeads * AttentionKvSplit <= _attnSplitKvMaxCoResidentGrid;
    }

    /// <summary>
    /// OPT-IN, default-OFF split-KV ("Flash-Decoding") variant of <see cref="LaunchAttentionF32"/>
    /// (issue #183), decode-only (<c>seqQ==1</c> implicit — no seqQ parameter). Splits the KV
    /// dimension across <see cref="AttentionKvSplit"/> cooperating blocks per head (grid =
    /// <c>numHeads * AttentionKvSplit</c> instead of <c>numHeads</c>) using CUDA Cooperative Groups
    /// <c>grid.sync()</c> — still exactly ONE kernel launch. NOT bit-exact vs
    /// <see cref="LaunchAttentionF32"/> (reassociated float reduction across the cross-block
    /// combine — see <see cref="EnableAttentionSplitKv"/>'s doc and attention_f32.cu's header for
    /// the full tradeoff). Callers MUST check <see cref="IsAttentionSplitKvSafe"/> first (exceeding
    /// the cooperative-launch co-residency ceiling is a hard CUDA error) and fall back to
    /// <see cref="LaunchAttentionF32"/> if it returns false.
    /// </summary>
    /// <param name="q">Device pointer to this step's query, <c>[numHeads, headDim]</c> (seqQ=1).</param>
    /// <param name="k">Device pointer to the cached keys, <c>[seqKv, numKvHeads, headDim]</c>.</param>
    /// <param name="v">Device pointer to the cached values, <c>[seqKv, numKvHeads, headDim]</c>.</param>
    /// <param name="output">Device pointer to the output, <c>[numHeads, headDim]</c>. Overwritten.</param>
    /// <param name="seqKv">Cached KV length (causal upper bound is <paramref name="positionOffset"/>).</param>
    /// <param name="numHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of KV heads (GQA broadcast group = numHeads/numKvHeads).</param>
    /// <param name="headDim">Per-head dimension.</param>
    /// <param name="positionOffset">This step's query position (causal mask upper bound).</param>
    /// <param name="slidingWindow">Sliding window size, or 0 for full causal attention.</param>
    /// <param name="partialMax">Scratch, <c>[numHeads, AttentionKvSplit]</c> floats.</param>
    /// <param name="partialSum">Scratch, <c>[numHeads, AttentionKvSplit]</c> floats.</param>
    /// <param name="partialOut">Scratch, <c>[numHeads, AttentionKvSplit, headDim]</c> floats.</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchAttentionF32SplitKv(nint q, nint k, nint v, nint output,
                                     int seqKv, int numHeads, int numKvHeads, int headDim,
                                     int positionOffset, int slidingWindow,
                                     nint partialMax, nint partialSum, nint partialOut, nint stream)
    {
        nint qArg = q, kArg = k, vArg = v, outArg = output;
        int skvArg = seqKv, nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim;
        int poArg = positionOffset, swArg = slidingWindow;
        nint pmArg = partialMax, psArg = partialSum, poutArg = partialOut;

        void** args = stackalloc void*[] {&qArg, &kArg, &vArg, &outArg,
                        &skvArg, &nhArg, &nkvArg, &hdArg,
                        &poArg, &swArg, &pmArg, &psArg, &poutArg};

        const int TileKv = 256;
        uint sharedBytes = (uint)((headDim + TileKv + headDim + 32) * sizeof(float));

        CudaDriverApi.cuLaunchCooperativeKernel(_attentionF32SplitKvFunc,
                (uint)numHeads, AttentionKvSplit, 1, BlockSize, 1, 1,
                sharedBytes, stream, (nint)args).ThrowOnError();
    }

    // ─── Issue #226 spike: fp64-combine variant of attention_f32_split_kv ──────────────────────
    //
    // Same grid/block/shared-memory shape as LaunchAttentionF32SplitKv; the ONLY kernel-side
    // difference is the cross-split combine step accumulates in double instead of float (see
    // attention_f32_split_kv_hp's header in attention_f32.cu). Separate function pointer, separate
    // opt-in gate (DOTLLM_ATTN_SPLIT_KV_HP=1) so it can be A/B'd against the plain split-KV kernel
    // without a rebuild. Whether this actually reduces the #222-documented divergence, and at what
    // perf cost, is exactly what issue #226 measures -- no verdict assumed here.

    /// <summary>Whether the opt-in fp64-combine <c>attention_f32_split_kv_hp</c> kernel (issue
    /// #226) is present in the loaded PTX.</summary>
    public bool HasAttentionF32SplitKvHp => _attentionF32SplitKvHpFunc != 0;

    /// <summary>
    /// Opt-in (default OFF) env-var gate for <see cref="LaunchAttentionF32SplitKvHp"/>. Issue #226
    /// spike: does accumulating the cross-split combine (partial_max/partial_sum/partial_out
    /// merge) in double precision reduce the real-generation argmax divergence #222 found in the
    /// float-combine <see cref="LaunchAttentionF32SplitKv"/>? <c>fast_exp_neg</c> itself is
    /// untouched (still float, still the same approximation) -- only the summation of its already-
    /// computed outputs across the (up to) <see cref="AttentionKvSplit"/> terms changes precision.
    /// </summary>
    public static bool EnableAttentionSplitKvHp { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_ATTN_SPLIT_KV_HP") == "1";

    /// <summary>Same co-residency safety check as <see cref="IsAttentionSplitKvSafe"/>, queried
    /// independently against <c>attention_f32_split_kv_hp</c>'s own function pointer (its register
    /// footprint may differ slightly from the plain kernel's due to the double-precision locals in
    /// the combine block).</summary>
    public bool IsAttentionSplitKvHpSafe(int numHeads, int headDim)
    {
        if (_attentionF32SplitKvHpFunc == 0) return false;

        if (_attnSplitKvHpMaxCoResidentGrid < 0 || _attnSplitKvHpCachedHeadDim != headDim)
        {
            CudaDriverApi.cuCtxGetDevice(out int device);
            int coopSupported = 0;
            CudaDriverApi.cuDeviceGetAttribute(out coopSupported,
                CudaDriverApi.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device);
            if (coopSupported == 0)
            {
                _attnSplitKvHpMaxCoResidentGrid = 0;
            }
            else
            {
                CudaDriverApi.cuDeviceGetAttribute(out int numSms,
                    CudaDriverApi.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
                const int TileKv = 256;
                uint sharedBytesForQuery = (uint)((headDim + TileKv + headDim + 32) * sizeof(float));
                int rc = CudaDriverApi.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    out int maxBlocksPerSm, _attentionF32SplitKvHpFunc, BlockSize, sharedBytesForQuery);
                _attnSplitKvHpMaxCoResidentGrid = rc == 0 ? numSms * maxBlocksPerSm : 0;
            }
            _attnSplitKvHpCachedHeadDim = headDim;
        }

        return _attnSplitKvHpMaxCoResidentGrid > 0 && numHeads * AttentionKvSplit <= _attnSplitKvHpMaxCoResidentGrid;
    }

    /// <summary>fp64-combine variant of <see cref="LaunchAttentionF32SplitKv"/> (issue #226) --
    /// identical signature/scratch-buffer layout, callers can share the same scratch allocation
    /// and safety-check call site pattern.</summary>
    public void LaunchAttentionF32SplitKvHp(nint q, nint k, nint v, nint output,
                                     int seqKv, int numHeads, int numKvHeads, int headDim,
                                     int positionOffset, int slidingWindow,
                                     nint partialMax, nint partialSum, nint partialOut, nint stream)
    {
        nint qArg = q, kArg = k, vArg = v, outArg = output;
        int skvArg = seqKv, nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim;
        int poArg = positionOffset, swArg = slidingWindow;
        nint pmArg = partialMax, psArg = partialSum, poutArg = partialOut;

        void** args = stackalloc void*[] {&qArg, &kArg, &vArg, &outArg,
                        &skvArg, &nhArg, &nkvArg, &hdArg,
                        &poArg, &swArg, &pmArg, &psArg, &poutArg};

        const int TileKv = 256;
        uint sharedBytes = (uint)((headDim + TileKv + headDim + 32) * sizeof(float));

        CudaDriverApi.cuLaunchCooperativeKernel(_attentionF32SplitKvHpFunc,
                (uint)numHeads, AttentionKvSplit, 1, BlockSize, 1, 1,
                sharedBytes, stream, (nint)args).ThrowOnError();
    }

    // ─── OPT-IN combined GQA-group + split-KV kernel (issues #197 + #198) ─────────────────────
    //
    // attention_f32_split_kv (above) grids one block per QUERY head. attention_f32_gqa_split_kv
    // grids one block per KV HEAD instead, register-blocking the QK/PV loops across the GQA
    // group of query heads sharing that KV head, AND composes that with a runtime (not
    // hardcoded) KV-split factor -- see native/kernels/attention_f32.cu's header on the combined
    // kernel for the full design and the two source issues' brainstorm comments for why the two
    // ideas MUST compose (grid=numKvHeads alone would make occupancy ~6x worse for Bonsai-27B's
    // real shape, numKvHeads=4 < numHeads=24).

    /// <summary>Max GQA group (query heads per KV head) the combined kernel supports -- MUST
    /// equal <c>MAX_GQA_GROUP</c> in <c>attention_f32.cu</c> (a compile-time cap on a
    /// runtime-loop-bounded register array; exceeding it is a correctness hazard, not just a
    /// perf one, so callers must gate on this before launch).</summary>
    public const int MaxGqaGroup = 8;

    /// <summary>
    /// Whether the opt-in combined GQA-group + split-KV <c>attention_f32_gqa_split_kv</c> kernel
    /// (issues #197 + #198) is present in the loaded PTX. Presence alone does not mean it's safe
    /// to launch for a given shape -- call <see cref="MaxSafeAttentionGqaSplit"/> first.
    /// </summary>
    public bool HasAttentionF32GqaSplitKv => _attentionF32GqaSplitKvFunc != 0;

    /// <summary>
    /// Opt-in (default OFF) env-var gate for <see cref="LaunchAttentionF32GqaSplit"/>. Same
    /// tradeoff shape as <see cref="EnableAttentionSplitKv"/> (issue #183) for the split-KV
    /// combine's reassociation, PLUS the (bit-exact, validated) GQA-group regrid on top -- see
    /// attention_f32.cu's combined-kernel header for the full correctness story. Off by default
    /// per this project's Correctness-then-Performance priority order.
    /// </summary>
    public static bool EnableAttentionGqaSplit { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_ATTN_GQA_SPLIT") == "1";

    /// <summary>
    /// Minimum <c>seqKv</c> before the combined kernel is worth its grid.sync + combine overhead
    /// -- same rationale as <see cref="AttentionSplitKvMinSeqKv"/>. Tunable; not read from an env
    /// var (only the on/off gate is).
    /// </summary>
    public static int AttentionGqaSplitMinSeqKv { get; set; } = 256;

    private static int EnvIntOrDefault(string envVar, int fallback)
    {
        var raw = Environment.GetEnvironmentVariable(envVar);
        return int.TryParse(raw, out int v) && v > 0 ? v : fallback;
    }

    /// <summary>
    /// Target total co-resident blocks for <see cref="ComputeAttentionKvSplit"/>'s occupancy
    /// term (<c>baseBlocks</c> is <c>numKvHeads</c> for the combined kernel). Default derived
    /// from an RTX-3060 sweep against the real Bonsai-27B shape (numKvHeads=4, headDim=256) --
    /// see the issue #197/#198 PR description for the sweep table. Override via
    /// <c>DOTLLM_CUDA_SPLIT_TARGET_BLOCKS</c> (kept permanently, like Vulkan kept its equivalent
    /// post-#347, so a different CUDA arch/GPU tier can be re-tuned without a native rebuild).
    /// </summary>
    private static readonly int AttnSplitTargetBlocks =
        EnvIntOrDefault("DOTLLM_CUDA_SPLIT_TARGET_BLOCKS", 32);

    /// <summary>
    /// Minimum KV rows per split before further splitting stops paying for its own grid.sync +
    /// combine overhead. Override via <c>DOTLLM_CUDA_SPLIT_MIN_KV</c>.
    /// </summary>
    /// <remarks>
    /// <b>Issue #219 fix (was 128):</b> at 128, this term was the BINDING clamp in
    /// <see cref="ComputeAttentionKvSplit"/> at real Bonsai-27B decode depths (seqKv~258-270),
    /// overriding <see cref="AttnSplitTargetBlocks"/>'s occupancy target entirely -- it forced
    /// <c>kv_split=3</c> (grid=12 blocks) when the occupancy term (<c>byOccupancy=8</c> for
    /// numKvHeads=4) and the co-residency ceiling (<c>maxSafeSplit=35</c>) both allowed far more.
    /// grid=12 is HALF the pre-split baseline's grid=24, i.e. the "fix" made grid fill worse, not
    /// better -- confirmed via <c>ncu --set full</c> (issue #199's finding, root-caused in #219):
    /// duration 174-176us, Achieved Occupancy 16.5-16.7% (statistically identical to the unsplit
    /// baseline's 16.6-16.8%), Waves/SM 0.09 (worse than baseline's 0.14).
    /// <para/>
    /// Lowering to 32 lets <c>byOccupancy</c> become the binding term at this shape/depth as the
    /// heuristic's own doc comment always intended (<c>S = clamp(TargetBlocks/baseBlocks, 1,
    /// ceil(seqKv/MinKvPerSplit))</c>, hard-clamped to <c>maxSafeSplit</c>) -- re-profiled
    /// (<c>ncu --set full</c>, same shape/depth): kv_split=8 (grid=32, now above the unsplit
    /// baseline's grid=24 as designed), duration 144-146us (~17% faster than the pre-fix 174-176us
    /// launches), Achieved Occupancy 18.7-19.2% (up from 16.5-16.7%), Waves/SM 0.23 (up from
    /// 0.09), Compute Throughput 15.7-15.8% (up from 9.3-9.4%).
    /// <para/>
    /// <b>This does NOT make the split kernel beat the plain unsplit path</b> at this shape/depth
    /// -- the fixed 144-146us launches are still ~37% slower than <c>attention_f32</c>'s
    /// 105-106us at the same seqKv (memory throughput only reaches 16.8-20.6% vs baseline's
    /// 44.6-45.0%; the cooperative-launch grid.sync()+combine overhead appears to dominate, not
    /// the intended K/V-read amortization). Pushing further (kv_split=16, grid=64, via
    /// <c>DOTLLM_CUDA_SPLIT_TARGET_BLOCKS=64</c> + this var at 16) raises Achieved Occupancy
    /// further (32-45%, avg ~38%) but duration is FLAT-TO-WORSE (148-150us) and Warp
    /// Cycles/Instruction regresses (28.5 vs 18.6-18.8 at kv_split=8) -- confirms occupancy is no
    /// longer the binding constraint past ~8 splits for this shape, so 32 (giving kv_split=8 at
    /// numKvHeads=4) is kept as the default rather than pushed further. Net: this fixes the
    /// heuristic to behave as documented (a real, ncu-validated improvement whenever
    /// <c>DOTLLM_ATTN_GQA_SPLIT=1</c> IS enabled), but the feature stays opt-in/default-OFF --
    /// even fixed, it does not beat the default path at this depth. Full data: issue #219.
    /// </remarks>
    private static readonly int AttnSplitMinKvPerSplit =
        EnvIntOrDefault("DOTLLM_CUDA_SPLIT_MIN_KV", 32);

    /// <summary>
    /// Occupancy-target split-count heuristic (issue #197), form ported from Vulkan's
    /// <c>VulkanSplitKvAttentionKernel.ComputeSplits</c> (issue #347's sweep) -- but the
    /// CONSTANTS are re-derived for CUDA, not copied: Vulkan's <c>TargetWorkgroups</c>/
    /// <c>MinKvPerSplit</c> were swept entirely on gfx1151 (Strix Halo) against a two-dispatch
    /// execution model with no co-residency constraint. CUDA's combined kernel is a single
    /// cooperative launch where ALL requested blocks must be co-resident simultaneously (a hard
    /// <c>cuLaunchCooperativeKernel</c> requirement), which the Vulkan formula has no equivalent
    /// clamp for -- that's exactly what <paramref name="maxSafeSplit"/> supplies here.
    /// <c>S = clamp(TargetBlocks/baseBlocks, 1, ceil(seqKv/MinKvPerSplit))</c>, then hard-clamped
    /// to <paramref name="maxSafeSplit"/>.
    /// </summary>
    /// <param name="seqKv">Cached KV length for this decode step.</param>
    /// <param name="baseBlocks">Grid.x base block count before splitting -- <c>numKvHeads</c> for
    /// the combined kernel.</param>
    /// <param name="maxSafeSplit">The real, GPU/shape-specific co-residency ceiling from
    /// <see cref="MaxSafeAttentionGqaSplit"/>. Hard clamp -- exceeding it is a CUDA launch error,
    /// not a soft perf regression.</param>
    public static int ComputeAttentionKvSplit(int seqKv, int baseBlocks, int maxSafeSplit)
    {
        if (seqKv <= 0 || baseBlocks <= 0 || maxSafeSplit <= 0) return 1;
        int byKv = (seqKv + AttnSplitMinKvPerSplit - 1) / AttnSplitMinKvPerSplit;
        int byOccupancy = Math.Max(1, AttnSplitTargetBlocks / baseBlocks);
        int s = Math.Min(byOccupancy, byKv);
        s = Math.Min(s, maxSafeSplit);
        return Math.Max(1, s);
    }

    /// <summary>
    /// True when <paramref name="numHeads"/>/<paramref name="numKvHeads"/> is a shape the
    /// combined kernel can handle at all: divides evenly (same convention
    /// <see cref="DotLLM.Cuda.CudaFlashAttention"/>'s <c>CanUse</c> gate already enforces -- no
    /// silent truncating-division mis-index) and the resulting group does not exceed
    /// <see cref="MaxGqaGroup"/>.
    /// </summary>
    public static bool IsGqaGroupShapeSupported(int numHeads, int numKvHeads)
    {
        if (numHeads <= 0 || numKvHeads <= 0 || numHeads % numKvHeads != 0) return false;
        int group = numHeads / numKvHeads;
        return group >= 1 && group <= MaxGqaGroup;
    }

    /// <summary>
    /// Queries (once per distinct (<paramref name="headDim"/>, group) pair, cached) the maximum
    /// safe <c>kv_split</c> for <see cref="LaunchAttentionF32GqaSplit"/> at this
    /// (numKvHeads, headDim) shape on THIS GPU -- i.e. the largest S such that
    /// <c>numKvHeads*S</c> blocks can be co-resident simultaneously (the same hard
    /// <c>cuLaunchCooperativeKernel</c> requirement <see cref="IsAttentionSplitKvSafe"/> already
    /// guards for the fixed-split kernel, generalized: this returns the ceiling itself, not a
    /// bool, so callers can feed it into <see cref="ComputeAttentionKvSplit"/>). Returns 0 if the
    /// kernel isn't loaded, cooperative launch isn't supported, or the shape doesn't fit even at
    /// split=1 -- callers must fall back to <see cref="LaunchAttentionF32SplitKv"/> or
    /// <see cref="LaunchAttentionF32"/> in that case.
    /// </summary>
    public int MaxSafeAttentionGqaSplit(int numKvHeads, int headDim, int group)
    {
        if (_attentionF32GqaSplitKvFunc == 0 || numKvHeads <= 0 || group <= 0) return 0;

        if (_attnGqaSplitMaxCoResidentGrid < 0
            || _attnGqaSplitCachedHeadDim != headDim
            || _attnGqaSplitCachedGroup != group)
        {
            CudaDriverApi.cuCtxGetDevice(out int device);
            int coopSupported = 0;
            CudaDriverApi.cuDeviceGetAttribute(out coopSupported,
                CudaDriverApi.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device);
            if (coopSupported == 0)
            {
                _attnGqaSplitMaxCoResidentGrid = 0;
            }
            else
            {
                CudaDriverApi.cuDeviceGetAttribute(out int numSms,
                    CudaDriverApi.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
                const int TileKv = 256;
                uint sharedBytesForQuery =
                    (uint)((group * headDim * 2 + group * TileKv + 32 + group * 2) * sizeof(float));
                int rc = CudaDriverApi.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    out int maxBlocksPerSm, _attentionF32GqaSplitKvFunc, BlockSize, sharedBytesForQuery);
                _attnGqaSplitMaxCoResidentGrid = rc == 0 ? numSms * maxBlocksPerSm : 0;
            }
            _attnGqaSplitCachedHeadDim = headDim;
            _attnGqaSplitCachedGroup = group;
        }

        if (_attnGqaSplitMaxCoResidentGrid <= 0) return 0;
        return _attnGqaSplitMaxCoResidentGrid / numKvHeads; // integer floor; 0 = unsafe at any split
    }

    /// <summary>
    /// OPT-IN, default-OFF combined GQA-group + split-KV kernel (issues #197 + #198),
    /// decode-only (<c>seqQ==1</c> implicit -- no seqQ parameter). Grid = <c>(numKvHeads,
    /// kvSplit)</c> using CUDA Cooperative Groups <c>grid.sync()</c> -- still exactly ONE kernel
    /// launch. Each block register-blocks the QK/PV loops across the GQA group of query heads
    /// sharing its KV head (see attention_f32.cu's header). Bit-exact per head vs
    /// <see cref="LaunchAttentionF32"/> at <c>kvSplit==1</c> (validated, not just asserted --
    /// see <c>CudaAttentionF32GqaSplitTests.cs</c>); inherits <see cref="LaunchAttentionF32SplitKv"/>'s
    /// existing reassociation tolerance at <c>kvSplit&gt;1</c>. Callers MUST have confirmed
    /// <see cref="IsGqaGroupShapeSupported"/> and clamped <paramref name="kvSplit"/> to
    /// <see cref="MaxSafeAttentionGqaSplit"/>'s result before calling (exceeding the
    /// cooperative-launch co-residency ceiling is a hard CUDA error).
    /// </summary>
    /// <param name="q">Device pointer to this step's query, <c>[numHeads, headDim]</c> (seqQ=1).</param>
    /// <param name="k">Device pointer to the cached keys, <c>[seqKv, numKvHeads, headDim]</c>.</param>
    /// <param name="v">Device pointer to the cached values, <c>[seqKv, numKvHeads, headDim]</c>.</param>
    /// <param name="output">Device pointer to the output, <c>[numHeads, headDim]</c>. Overwritten.</param>
    /// <param name="seqKv">Cached KV length (causal upper bound is <paramref name="positionOffset"/>).</param>
    /// <param name="numHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of KV heads (GQA group = numHeads/numKvHeads).</param>
    /// <param name="headDim">Per-head dimension.</param>
    /// <param name="positionOffset">This step's query position (causal mask upper bound).</param>
    /// <param name="slidingWindow">Sliding window size, or 0 for full causal attention.</param>
    /// <param name="kvSplit">KV-dimension split factor -- see <see cref="ComputeAttentionKvSplit"/>.</param>
    /// <param name="partialMax">Scratch, <c>[numHeads, kvSplit]</c> floats.</param>
    /// <param name="partialSum">Scratch, <c>[numHeads, kvSplit]</c> floats.</param>
    /// <param name="partialOut">Scratch, <c>[numHeads, kvSplit, headDim]</c> floats.</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchAttentionF32GqaSplit(nint q, nint k, nint v, nint output,
                                     int seqKv, int numHeads, int numKvHeads, int headDim,
                                     int positionOffset, int slidingWindow, int kvSplit,
                                     nint partialMax, nint partialSum, nint partialOut, nint stream)
    {
        nint qArg = q, kArg = k, vArg = v, outArg = output;
        int skvArg = seqKv, nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim;
        int poArg = positionOffset, swArg = slidingWindow, ksArg = kvSplit;
        nint pmArg = partialMax, psArg = partialSum, poutArg = partialOut;

        void** args = stackalloc void*[] {&qArg, &kArg, &vArg, &outArg,
                        &skvArg, &nhArg, &nkvArg, &hdArg,
                        &poArg, &swArg, &ksArg, &pmArg, &psArg, &poutArg};

        int group = numHeads / numKvHeads;
        const int TileKv = 256;
        uint sharedBytes =
            (uint)((group * headDim * 2 + group * TileKv + 32 + group * 2) * sizeof(float));

        CudaDriverApi.cuLaunchCooperativeKernel(_attentionF32GqaSplitKvFunc,
                (uint)numKvHeads, (uint)kvSplit, 1, BlockSize, 1, 1,
                sharedBytes, stream, (nint)args).ThrowOnError();
    }

    // ─── OPT-IN tensor-core (mma.sync) FP16 decode attention, composed with the GQA-group +
    // split-KV grid (issue #199 v2) ─────────────────────────────────────────────────────────
    //
    // Same grid shape as attention_f32_gqa_split_kv above -- (numKvHeads, kvSplit) via
    // cuLaunchCooperativeKernel -- but each block packs the `group` query heads sharing its
    // KV head into ONE mma.sync.m16n8k16 tile's M dimension instead of register-blocking them
    // (see attention_flash_mma_decode_gqa_split.cu's header for the full design and why this
    // keeps the kernel's STATIC shared-memory footprint group-independent, unlike the sibling
    // kernel's dynamic per-group shared arrays). Reads Q/K/V as FP16 (K/V straight from the
    // FP16 KV cache) and writes F32 output directly, same convention as v1
    // (attention_flash_mma_decode.cu, issue #199 v1, not merged).

    /// <summary>Block size (NUM_WARPS*32) the composed tensor-core decode kernel is compiled
    /// with -- MUST match <c>NUM_WARPS</c> in <c>attention_flash_mma_decode_gqa_split.cu</c>.</summary>
    public const int AttentionMmaDecodeGqaSplitBlockSize = 256;

    /// <summary>Head dimension the composed tensor-core decode kernel is compiled for
    /// (Bonsai-27B's qwen35moe shape) -- MUST match <c>HEAD_DIM</c> in
    /// <c>attention_flash_mma_decode_gqa_split.cu</c>.</summary>
    public const int AttentionMmaDecodeGqaSplitHeadDim = 256;

    /// <summary>
    /// True when the composed tensor-core decode kernel (issue #199 v2) is loaded
    /// (<c>attention_flash_mma_decode_gqa_split.ptx</c> present and the sm_86 module loaded
    /// successfully on this device). Presence alone does not mean it's safe to launch for a
    /// given shape -- call <see cref="MaxSafeAttentionMmaDecodeGqaSplit"/> first. See
    /// <see cref="DotLLM.Cuda.CudaAttentionMmaDecodeGqaSplit"/> for the gating/dispatch wrapper.
    /// </summary>
    public bool HasAttentionMmaDecodeGqaSplit => _attentionMmaDecodeGqaSplitFunc != 0;

    /// <summary>
    /// Queries (once per distinct <paramref name="group"/>, cached) the maximum safe
    /// <c>kvSplit</c> for <see cref="LaunchAttentionMmaDecodeGqaSplit"/> at this
    /// (numKvHeads, group) shape on THIS GPU -- i.e. the largest S such that
    /// <c>numKvHeads*S</c> blocks can be co-resident simultaneously (the same hard
    /// <c>cuLaunchCooperativeKernel</c> requirement <see cref="MaxSafeAttentionGqaSplit"/>
    /// already guards for the sibling FP32 kernel). Unlike that kernel, this one's shared
    /// memory is STATIC and group-independent (see the .cu file's header), so the occupancy
    /// query passes <c>dynamicSMemSize=0</c> -- the driver already knows the static footprint
    /// from the compiled module. <paramref name="headDim"/> is accepted for API symmetry with
    /// <see cref="MaxSafeAttentionGqaSplit"/> but is not used in the query (the kernel is
    /// compiled for a single fixed <see cref="AttentionMmaDecodeGqaSplitHeadDim"/>). Returns 0
    /// if the kernel isn't loaded, cooperative launch isn't supported, or the shape doesn't
    /// fit even at split=1.
    /// </summary>
    public int MaxSafeAttentionMmaDecodeGqaSplit(int numKvHeads, int headDim, int group)
    {
        if (_attentionMmaDecodeGqaSplitFunc == 0 || numKvHeads <= 0 || group <= 0) return 0;

        if (_attnMmaDecodeGqaSplitMaxCoResidentGrid < 0 || _attnMmaDecodeGqaSplitCachedGroup != group)
        {
            CudaDriverApi.cuCtxGetDevice(out int device);
            int coopSupported = 0;
            CudaDriverApi.cuDeviceGetAttribute(out coopSupported,
                CudaDriverApi.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device);
            if (coopSupported == 0)
            {
                _attnMmaDecodeGqaSplitMaxCoResidentGrid = 0;
            }
            else
            {
                CudaDriverApi.cuDeviceGetAttribute(out int numSms,
                    CudaDriverApi.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
                int rc = CudaDriverApi.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    out int maxBlocksPerSm, _attentionMmaDecodeGqaSplitFunc,
                    AttentionMmaDecodeGqaSplitBlockSize, dynamicSMemSize: 0);
                _attnMmaDecodeGqaSplitMaxCoResidentGrid = rc == 0 ? numSms * maxBlocksPerSm : 0;
            }
            _attnMmaDecodeGqaSplitCachedGroup = group;
        }

        if (_attnMmaDecodeGqaSplitMaxCoResidentGrid <= 0) return 0;
        return _attnMmaDecodeGqaSplitMaxCoResidentGrid / numKvHeads; // integer floor; 0 = unsafe at any split
    }

    /// <summary>
    /// OPT-IN, default-OFF composed tensor-core decode kernel (issue #199 v2), decode-only
    /// (<c>seqQ==1</c> implicit). Grid = <c>(numKvHeads, kvSplit)</c> using CUDA Cooperative
    /// Groups <c>grid.sync()</c> -- still exactly ONE kernel launch, same shape and combine
    /// algebra as <see cref="LaunchAttentionF32GqaSplit"/> (ported verbatim -- see the .cu
    /// file's header). Callers MUST have confirmed <see cref="CudaKernels.IsGqaGroupShapeSupported"/>
    /// and clamped <paramref name="kvSplit"/> to <see cref="MaxSafeAttentionMmaDecodeGqaSplit"/>'s
    /// result before calling (exceeding the cooperative-launch co-residency ceiling is a hard
    /// CUDA error).
    /// </summary>
    /// <param name="q">Device pointer to this step's query, FP16, <c>[numHeads, headDim]</c> (seqQ=1).</param>
    /// <param name="k">Device pointer to the FP16 KV cache's keys, <c>[seqKv, numKvHeads, headDim]</c>.</param>
    /// <param name="v">Device pointer to the FP16 KV cache's values, <c>[seqKv, numKvHeads, headDim]</c>.</param>
    /// <param name="output">Device pointer to the F32 output, <c>[numHeads, headDim]</c>. Overwritten.</param>
    /// <param name="seqKv">Cached KV length.</param>
    /// <param name="numHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of KV heads (GQA group = numHeads/numKvHeads).</param>
    /// <param name="kvSplit">KV-dimension split factor -- see <see cref="ComputeAttentionKvSplit"/>.</param>
    /// <param name="partialMax">Scratch, <c>[numHeads, kvSplit]</c> floats.</param>
    /// <param name="partialSum">Scratch, <c>[numHeads, kvSplit]</c> floats.</param>
    /// <param name="partialOut">Scratch, <c>[numHeads, kvSplit, headDim]</c> floats.</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchAttentionMmaDecodeGqaSplit(nint q, nint k, nint v, nint output,
        int seqKv, int numHeads, int numKvHeads, int kvSplit,
        nint partialMax, nint partialSum, nint partialOut, nint stream)
    {
        nint qArg = q, kArg = k, vArg = v, outArg = output;
        int skvArg = seqKv, nhArg = numHeads, nkvArg = numKvHeads, ksArg = kvSplit;
        float scArg = 1.0f / MathF.Sqrt(AttentionMmaDecodeGqaSplitHeadDim);
        nint pmArg = partialMax, psArg = partialSum, poutArg = partialOut;

        void** args = stackalloc void*[] {&qArg, &kArg, &vArg, &outArg,
                        &skvArg, &nhArg, &nkvArg, &scArg, &ksArg, &pmArg, &psArg, &poutArg};

        CudaDriverApi.cuLaunchCooperativeKernel(_attentionMmaDecodeGqaSplitFunc,
                (uint)numKvHeads, (uint)kvSplit, 1, AttentionMmaDecodeGqaSplitBlockSize, 1, 1,
                0, stream, (nint)args).ThrowOnError();
    }


    /// <summary>FP32 SwiGLU: out = SiLU(gate) * up, all FP32.</summary>
    public void LaunchSwiGLUF32(nint gate, nint up, nint output,
                                  int n, int seqLen, nint stream)
    {
        nint gateArg = gate, upArg = up, outArg = output;
        int nArg = n, slArg = seqLen;

        void** args = stackalloc void*[] {&gateArg, &upArg, &outArg, &nArg, &slArg};
        uint gridDim = (uint)((n * seqLen + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_swigluF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// FP32 BitNet-MoE gate activation: <c>out = relu(gate)^2 * up</c> (issue #246). Replaces
    /// <see cref="LaunchSwiGLUF32"/> for ternary I2_S expert bodies — same launch shape (one
    /// thread per element, grid-strided over <c>n * seqLen</c>).
    /// </summary>
    public void LaunchReLU2GLUF32(nint gate, nint up, nint output,
                                    int n, int seqLen, nint stream)
    {
        nint gateArg = gate, upArg = up, outArg = output;
        int nArg = n, slArg = seqLen;

        void** args = stackalloc void*[] {&gateArg, &upArg, &outArg, &nArg, &slArg};
        uint gridDim = (uint)((n * seqLen + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_relu2GluF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>FP32 bias add: output_f32[i] += bias_f16[i % dim].</summary>
    public void LaunchBiasAddF32(nint output, nint biasF16, int dim, int seqLen, nint stream)
    {
        nint outArg = output, biasArg = biasF16;
        int dimArg = dim, slArg = seqLen;

        void** args = stackalloc void*[] {&outArg, &biasArg, &dimArg, &slArg};
        uint gridDim = (uint)((dim * seqLen + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_biasAddF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>FP32 per-head RmsNorm: FP32 data, FP16 weight.</summary>
    public void LaunchPerHeadRmsNormF32(nint qk, nint weightF16, float eps,
                                          int numHeads, int headDim, int seqLen, nint stream)
    {
        nint qkArg = qk, wArg = weightF16;
        float epsArg = eps;
        int nhArg = numHeads, hdArg = headDim, slArg = seqLen;

        void** args = stackalloc void*[] {&qkArg, &wArg, &epsArg, &nhArg, &hdArg, &slArg};

        CudaDriverApi.cuLaunchKernel(_perHeadRmsNormF32Func,
                (uint)(seqLen * numHeads), 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Per-group F32 RmsNorm (NVIDIA Nemotron-H Mamba2 SSM output norm). Each group has
    /// its own weight slice — unlike <see cref="LaunchPerHeadRmsNormF32"/>'s single shared weight
    /// array. See native/kernels/group_rmsnorm.cu.</summary>
    /// <param name="x">Device pointer, <c>[seqLen, nGroup, groupDim]</c> F32, normalized in place.</param>
    /// <param name="weight">Device pointer, <c>[nGroup, groupDim]</c> F32 (== ssm_norm.weight).</param>
    /// <param name="eps">Variance epsilon added before the reciprocal square root.</param>
    /// <param name="seqLen">Number of tokens (rows) in <paramref name="x"/>.</param>
    /// <param name="nGroup">Number of SSM groups (grid.x == seqLen * nGroup).</param>
    /// <param name="groupDim">Elements per group (== d_inner / nGroup).</param>
    /// <param name="stream">CUDA stream to launch on.</param>
    public void LaunchGroupRmsNormF32(nint x, nint weight, float eps,
                                        int seqLen, int nGroup, int groupDim, nint stream)
    {
        nint xArg = x, wArg = weight;
        float epsArg = eps;
        int slArg = seqLen, ngArg = nGroup, gdArg = groupDim;

        void** args = stackalloc void*[] {&xArg, &wArg, &epsArg, &slArg, &ngArg, &gdArg};

        CudaDriverApi.cuLaunchKernel(_groupRmsNormF32Func,
                (uint)(seqLen * nGroup), 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    // ── Gemma-4 (DiffusionGemma AR) F32 launchers ────────────────────────────

    /// <summary>
    /// In-place Q8_0 round-trip of an F32 activation buffer <c>[rows, k]</c>:
    /// per 32-block, quantize to Q8_0 (FP16 block scale, round-nearest-even,
    /// clamp ±127) then dequantize back to F32. Reproduces the CPU oracle's
    /// on-the-fly activation quantization for Q8_0-weight GEMMs so the gemma4
    /// F32 forward stays within parity tolerance of <c>MatMul.GemmQ8_0</c>.
    /// Requires <c>k % 32 == 0</c> (gemma4 K dims always satisfy this).
    /// </summary>
    public void LaunchQuantizeActivationQ8_0RoundtripF32(nint xF32, int k, int rows, nint stream)
    {
        if ((k & 31) != 0) return; // CPU only quantizes whole 32-blocks
        nint xArg = xF32;
        int kArg = k, rowsArg = rows;
        void** args = stackalloc void*[] {&xArg, &kArg, &rowsArg};
        long nb = (long)rows * (k / 32);
        CudaDriverApi.cuLaunchKernel(_quantizeActQ8_0RoundtripF32Func,
                (uint)nb, 1, 1, 32, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>FP32 GeGLU (tanh-approx GELU): out = gelu_tanh(gate) * up.</summary>
    public void LaunchGeGLUTanhF32(nint gate, nint up, nint output,
                                     int n, int seqLen, nint stream)
    {
        nint gateArg = gate, upArg = up, outArg = output;
        int nArg = n, slArg = seqLen;

        void** args = stackalloc void*[] {&gateArg, &upArg, &outArg, &nArg, &slArg};
        uint gridDim = (uint)(((long)n * seqLen + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_gegluTanhF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Partial NeoX RoPE (FP32, in place on Q and K). Rotates the leading
    /// <paramref name="rotatedPairs"/> pairs of each head, coupling
    /// <c>(vec[i], vec[i + headDim/2])</c>; freq base over the full
    /// <paramref name="headDim"/>. Mirrors <c>RoPE.ExecutePartialNeoX</c>.
    /// </summary>
    public void LaunchRoPEF32PartialNeoX(nint q, nint k, nint positions,
                                           int seqLen, int numHeads, int numKvHeads,
                                           int headDim, int rotatedPairs, float theta, nint stream)
    {
        nint qArg = q, kArg = k, posArg = positions;
        int slArg = seqLen, nhArg = numHeads, nkvArg = numKvHeads;
        int hdArg = headDim, rpArg = rotatedPairs;
        float thetaArg = theta;

        void** args = stackalloc void*[] {&qArg, &kArg, &posArg, &slArg, &nhArg, &nkvArg,
                        &hdArg, &rpArg, &thetaArg};

        int totalPairs = seqLen * Math.Max(numHeads, numKvHeads) * rotatedPairs;
        uint gridDim = (uint)((totalPairs + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_ropeF32PartialNeoxFunc,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>In-place FP32 scalar multiply: x[i] *= scale (layer_output_scale).</summary>
    public void LaunchScaleInplaceF32(nint x, int n, float scale, nint stream)
    {
        nint xArg = x;
        int nArg = n;
        float scaleArg = scale;

        void** args = stackalloc void*[] {&xArg, &nArg, &scaleArg};
        uint gridDim = (uint)((n + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_scaleInplaceF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// FP32 weight-less per-row RMSNorm (unit gamma). One block per row;
    /// <paramref name="rows"/> rows of width <paramref name="n"/>. Used for the
    /// gemma4 weight-less V-norm with one row per (token, kv-head).
    /// </summary>
    public void LaunchRmsNormWeightlessF32(nint input, nint output,
                                             int n, float eps, int rows, nint stream)
    {
        nint inputArg = input, outputArg = output;
        int nArg = n;
        float epsArg = eps;

        void** args = stackalloc void*[] {&inputArg, &outputArg, &nArg, &epsArg};
        CudaDriverApi.cuLaunchKernel(_rmsnormWeightlessF32Func,
                (uint)rows, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>In-place FP32 final-logit soft-capping: x[i] = cap * tanh(x[i] / cap).</summary>
    public void LaunchSoftcapInplaceF32(nint x, int n, float cap, nint stream)
    {
        nint xArg = x;
        int nArg = n;
        float capArg = cap;

        void** args = stackalloc void*[] {&xArg, &nArg, &capArg};
        uint gridDim = (uint)((n + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_softcapInplaceF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Gemma-4 top-k renorm with the 6.1e-5 denominator clamp:
    /// <c>w[i] *= 1 / max(Σ w, 2^-14)</c>. One block per token.
    /// </summary>
    public void LaunchMoeRenormTopkClampedF32(nint topkWeight, int seqLen, int topK, nint stream)
    {
        nint wArg = topkWeight;
        int slArg = seqLen, kArg = topK;

        void** args = stackalloc void*[] {&wArg, &slArg, &kArg};
        CudaDriverApi.cuLaunchKernel(_moeRenormTopkClampedF32Func,
                (uint)seqLen, 1, 1, 32, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Depthwise causal 1-D convolution (FP32). Bit-perfect port of the CPU
    /// <c>Conv1dCausal.ExecuteScalar</c> reference.
    /// </summary>
    /// <param name="input">
    /// Device pointer to <c>[d_conv-1+seqLen, channels]</c> FP32 input, row-major.
    /// Caller must prepend the cached <c>conv_state</c> (d_conv-1 rows) before
    /// the new activations.
    /// </param>
    /// <param name="weight">
    /// Device pointer to <c>[d_conv, channels]</c> FP32 weight in GGUF
    /// channel-major layout: element (k, c) at <c>c * d_conv + k</c>.
    /// </param>
    /// <param name="bias">
    /// Device pointer to per-channel FP32 bias (<c>channels</c> elements). Pass
    /// a zero-filled buffer when the model has no bias — the add is unconditional.
    /// </param>
    /// <param name="output">
    /// Device pointer to <c>[seqLen, channels]</c> FP32 output, row-major.
    /// </param>
    /// <param name="dConv">Convolution kernel width (4 for Qwen3MoeHybrid).</param>
    /// <param name="channels">Number of channels (depthwise width).</param>
    /// <param name="seqLen">Number of output time steps.</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchConv1dCausalF32(nint input, nint weight, nint bias, nint output,
                                        int dConv, int channels, int seqLen, nint stream)
    {
        nint inArg = input, wArg = weight, bArg = bias, outArg = output;
        int dcArg = dConv, chArg = channels, slArg = seqLen;

        void** args = stackalloc void*[] {&inArg, &wArg, &bArg, &outArg,
                        &dcArg, &chArg, &slArg};

        int total = seqLen * channels;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_conv1dCausalF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// True when the fused decode-time causal-conv1d kernel
    /// (<see cref="LaunchGdnConv1dCausalDecodeF32"/>) is loaded. When false, callers must fall
    /// back to the general <see cref="LaunchConv1dCausalF32"/> path (state/qkv concat memcpy,
    /// conv1d, <see cref="LaunchSiluF32"/>, trailing-state extract memcpy — 5 launches).
    /// </summary>
    public bool HasGdnConv1dCausalDecodeF32 => _gdnConv1dCausalDecodeF32Func != 0;

    /// <summary>
    /// Fused decode-time (single new row, <c>seqLen==1</c>) causal conv1d + SiLU + rolling
    /// conv-state update — issue #168. Reads <paramref name="state"/> (the existing
    /// <c>[(dConv-1), channels]</c> rolling history) and <paramref name="qkvIn"/> (the one new
    /// pre-conv row) directly — no physical <c>[state; qkv]</c> concat buffer — computes the
    /// causal conv1d output, applies SiLU in place, and writes the shifted trailing state back
    /// into <paramref name="state"/>. Replaces the general path's 3
    /// <c>cuMemcpyDtoDAsync</c> launches (concat-in ×2, trailing-state-extract ×1) + separate
    /// <see cref="LaunchConv1dCausalF32"/> + <see cref="LaunchSiluF32"/> — 5 launches down to 1
    /// — for the decode (<c>seqLen==1</c>) case only; prefill (<c>seqLen&gt;1</c>) keeps using
    /// the general path.
    /// </summary>
    /// <remarks>
    /// <paramref name="state"/> is read AND written in place (safe — see the kernel's own header
    /// comment in <c>conv1d_causal.cu</c> for the per-channel-ownership aliasing argument).
    /// <paramref name="qkvOut"/> may alias <paramref name="qkvIn"/> (also safe, same argument).
    /// Bit-exact vs. the general path's (memcpy → conv1d_causal_f32 → silu_f32 → memcpy)
    /// sequence for <c>seqLen==1</c> — same accumulation order, same SiLU formula, compiled from
    /// the same <c>-fmad=false</c> translation unit.
    /// </remarks>
    /// <param name="state">Device pointer to <c>[(dConv-1), channels]</c> FP32 rolling conv state, in/out.</param>
    /// <param name="qkvIn">Device pointer to the new <c>[channels]</c> FP32 pre-conv row.</param>
    /// <param name="weight">Device pointer to <c>[channels, dConv]</c> FP32 weight (GGUF channel-major layout).</param>
    /// <param name="bias">Device pointer to per-channel FP32 bias (<c>channels</c> elements).</param>
    /// <param name="qkvOut">Device pointer to <c>[channels]</c> FP32 output (SiLU(conv output)); may alias <paramref name="qkvIn"/>.</param>
    /// <param name="dConv">Convolution kernel width (4 for Qwen3HybridDense/Qwen3MoeHybrid).</param>
    /// <param name="channels">Number of channels (depthwise width).</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchGdnConv1dCausalDecodeF32(nint state, nint qkvIn, nint weight, nint bias,
                                                 nint qkvOut, int dConv, int channels, nint stream)
    {
        nint stArg = state, qiArg = qkvIn, wArg = weight, bArg = bias, qoArg = qkvOut;
        int dcArg = dConv, chArg = channels;

        void** args = stackalloc void*[] {&stArg, &qiArg, &wArg, &bArg, &qoArg, &dcArg, &chArg};

        uint gridDim = (uint)((channels + BlockSize - 1) / BlockSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_gdnConv1dCausalDecodeF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Gated DeltaNet (GDN) recurrence — one token step. Bit-perfect port of the
    /// CPU <c>GatedDeltaNetScan.Execute</c> per-token inner body. The host loops
    /// over <c>seqLen</c> and calls this once per token, advancing the
    /// <paramref name="qT"/>/<paramref name="kT"/>/<paramref name="vT"/>/<paramref name="gT"/>/<paramref name="betaT"/>/<paramref name="outputT"/>
    /// pointers by one token-stride per call. The recurrence on <paramref name="state"/>
    /// flows automatically through the in-place update.
    /// </summary>
    /// <param name="state">
    /// Device pointer to <c>[nVHead, dState, dState]</c> FP32 state matrix
    /// (row-major: <c>S[vh, row, col]</c>, row = key dim, col = value dim).
    /// Updated in place.
    /// </param>
    /// <param name="qT">Device pointer to this token's Q vectors, <c>[nKHead, dState]</c>. Caller must have L2-normalised per head.</param>
    /// <param name="kT">Device pointer to this token's K vectors, <c>[nKHead, dState]</c>. Caller must have L2-normalised per head.</param>
    /// <param name="vT">Device pointer to this token's V vectors, <c>[nVHead, dState]</c>.</param>
    /// <param name="gT">Device pointer to this token's per-head decay scalars, <c>[nVHead]</c>.</param>
    /// <param name="betaT">Device pointer to this token's per-head write-gate scalars, <c>[nVHead]</c>.</param>
    /// <param name="outputT">Device pointer to this token's output, <c>[nVHead, dState]</c>. Overwritten.</param>
    /// <param name="nVHead">Number of value heads.</param>
    /// <param name="nKHead">Number of key heads (must divide <paramref name="nVHead"/> evenly).</param>
    /// <param name="dState">
    /// Per-head state dimension. The kernel launches with <c>blockDim.x == dState</c>
    /// and is compiled with <c>__launch_bounds__(128)</c>, so the upper bound is
    /// 128 — the universal Qwen3MoeHybrid configuration. Larger <c>dState</c> would
    /// fail launch validation; this method throws on out-of-range values rather than
    /// silently failing inside <c>cuLaunchKernel</c>.
    /// </param>
    /// <param name="stream">CUDA stream handle.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="dState"/> is outside (0, 128].
    /// </exception>
    public void LaunchGdnScanStepF32(nint state, nint qT, nint kT, nint vT,
                                       nint gT, nint betaT, nint outputT,
                                       int nVHead, int nKHead, int dState, nint stream)
    {
        if (dState <= 0 || dState > 128)
            throw new ArgumentOutOfRangeException(nameof(dState),
                $"dState={dState}; gdn_scan_step_f32 is compiled with __launch_bounds__(128).");

        nint sArg = state, qArg = qT, kArg = kT, vArg = vT;
        nint gArg = gT, bArg = betaT, oArg = outputT;
        int nvArg = nVHead, nkArg = nKHead, dsArg = dState;

        void** args = stackalloc void*[] {&sArg, &qArg, &kArg, &vArg,
                        &gArg, &bArg, &oArg,
                        &nvArg, &nkArg, &dsArg};

        // Shared memory: k_shared[dState] + q_shared[dState]
        uint sharedBytes = (uint)(2 * dState * sizeof(float));

        CudaDriverApi.cuLaunchKernel(_gdnScanStepF32Func,
                (uint)nVHead, 1, 1, (uint)dState, 1, 1,
                sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Mamba2 selective-scan (NVIDIA Nemotron-H), one launch per SSM-layer forward call —
    /// bit-order-faithful port of <see cref="DotLLM.Cpu.Kernels.Mamba2SelectiveScan.Execute"/>,
    /// fused with the raw-dt + dtBias guarded-softplus decay and the D-skip term. See
    /// native/kernels/mamba2_selective_scan.cu for the full layout and fusion documentation.
    /// </summary>
    /// <param name="state">Device pointer, <c>[nHead, headDim, dState]</c> F32, updated in place.</param>
    /// <param name="x">Device pointer, <c>[seqLen, dInner]</c> F32 (dInner = nHead*headDim).</param>
    /// <param name="dtRaw">Device pointer, <c>[seqLen, nHead]</c> F32, NOT yet bias-added.</param>
    /// <param name="dtBias">Device pointer, <c>[nHead]</c> F32.</param>
    /// <param name="a">Device pointer, <c>[nHead]</c> F32 (stored negative by the GGUF converter).</param>
    /// <param name="d">Device pointer, <c>[nHead]</c> F32 (D skip parameter).</param>
    /// <param name="b">Device pointer, <c>[seqLen, nGroup, dState]</c> F32.</param>
    /// <param name="c">Device pointer, <c>[seqLen, nGroup, dState]</c> F32.</param>
    /// <param name="y">Device pointer, <c>[seqLen, dInner]</c> F32, overwritten (includes D-skip).</param>
    /// <param name="nHead">Number of Mamba2 heads.</param>
    /// <param name="headDim">Channels per head (dInner / nHead). Must be in (0, 256] — the kernel
    /// is compiled with <c>__launch_bounds__(256)</c> and launches with blockDim.x == headDim.</param>
    /// <param name="dState">SSM state width.</param>
    /// <param name="nGroup">Number of B/C groups (must divide nHead evenly).</param>
    /// <param name="seqLen">Number of tokens in this call.</param>
    /// <param name="stream">CUDA stream handle.</param>
    /// <exception cref="ArgumentOutOfRangeException">headDim outside (0, 256].</exception>
    public void LaunchMamba2SelectiveScanF32(nint state, nint x, nint dtRaw, nint dtBias,
                                               nint a, nint d, nint b, nint c, nint y,
                                               int nHead, int headDim, int dState, int nGroup,
                                               int seqLen, nint stream)
    {
        if (headDim <= 0 || headDim > 256)
            throw new ArgumentOutOfRangeException(nameof(headDim),
                $"headDim={headDim}; mamba2_selective_scan_f32 is compiled with __launch_bounds__(256) " +
                "and launches with blockDim.x == headDim.");

        nint sArg = state, xArg = x, dtArg = dtRaw, dtbArg = dtBias;
        nint aArg = a, dArg = d, bArg = b, cArg = c, yArg = y;
        int nhArg = nHead, hdArg = headDim, dsArg = dState, ngArg = nGroup, slArg = seqLen;

        void** args = stackalloc void*[] {&sArg, &xArg, &dtArg, &dtbArg,
                        &aArg, &dArg, &bArg, &cArg, &yArg,
                        &nhArg, &hdArg, &dsArg, &ngArg, &slArg};

        // Shared memory: b_shared[dState] + c_shared[dState]
        uint sharedBytes = (uint)(2 * dState * sizeof(float));

        CudaDriverApi.cuLaunchKernel(_mamba2ScanF32Func,
                (uint)nHead, 1, 1, (uint)headDim, 1, 1,
                sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Whether the opt-in row-split cooperative-groups <c>gdn_scan_step_f32_coop_split4</c> kernel
    /// is present in the loaded PTX (issue #180). Presence alone does not mean it's SAFE to launch
    /// for a given (nVHead, dState) shape on this GPU — call <see cref="IsGdnScanCoopSplit4Safe"/>
    /// first, every time nVHead changes (e.g. a different model), since exceeding the cooperative
    /// launch's co-residency ceiling is a hard CUDA error, not a silent fallback.
    /// </summary>
    public bool HasGdnScanStepF32CoopSplit4 => _gdnScanStepF32CoopSplit4Func != 0;

    /// <summary>
    /// Opt-in (default OFF) env-var gate for <see cref="LaunchGdnScanStepF32CoopSplit4"/>.
    /// <b>Enabling this trades away gdn_scan_step_f32's documented bit-exact CPU/GPU parity</b> —
    /// the row-split reduction reassociates the retrieve/read accumulation (independent partial
    /// sums combined across blocks, not the CPU's strict sequential 0..dState-1 order), which is
    /// mathematically equal but not bit-identical (measured ~2e-6 abs / ~4.6e-4 rel diff on a
    /// single fresh-state step with random inputs). Since the GDN state persists across the ENTIRE
    /// generation, a 500-decode-step characterization found the ABSOLUTE diff stays bounded
    /// (~1e-6 to 2.7e-3, no runaway growth, no NaN/Inf) but not characterized beyond 500 synthetic
    /// steps this session. Real gain when enabled: ~26-27% faster gdn_scan_step_f32 kernel time,
    /// ~1.8% average end-to-end decode throughput across 5 independent A/B rounds, never losing a
    /// round (measured on RTX 3060 / Bonsai-27B — see gated_delta_net_scan.cu's header and issue
    /// #180 for the full writeup). Off by default per this project's stated
    /// Correctness-then-Performance priority order.
    /// </summary>
    public static bool EnableGdnScanApproxSplit4 { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_GDN_SCAN_APPROX_SPLIT4") == "1";

    /// <summary>
    /// Queries (once, cached) whether <c>split=4</c> cooperative launch is safe for this exact
    /// (nVHead, dState) shape on THIS GPU — i.e. whether <c>nVHead*4</c> blocks can be
    /// simultaneously co-resident (a hard requirement of <c>cuLaunchCooperativeKernel</c>; asking
    /// for more than the device can hold is a launch ERROR, not a graceful degrade). Returns false
    /// (safe default: caller falls back to the exact, non-split kernel) if the coop kernel isn't
    /// loaded, cooperative launch isn't supported on this device/driver, or the shape doesn't fit.
    /// </summary>
    public bool IsGdnScanCoopSplit4Safe(int nVHead, int dState)
    {
        if (_gdnScanStepF32CoopSplit4Func == 0) return false;
        if (dState != 128) return false; // kernel hardcodes SPLIT=4 dividing d_state=128 evenly

        if (_gdnScanCoopSplit4MaxCoResidentGrid < 0)
        {
            CudaDriverApi.cuCtxGetDevice(out int device);
            int coopSupported = 0;
            CudaDriverApi.cuDeviceGetAttribute(out coopSupported,
                CudaDriverApi.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device);
            if (coopSupported == 0)
            {
                _gdnScanCoopSplit4MaxCoResidentGrid = 0;
            }
            else
            {
                CudaDriverApi.cuDeviceGetAttribute(out int numSms,
                    CudaDriverApi.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
                uint sharedBytesForQuery = (uint)(2 * dState * sizeof(float));
                int rc = CudaDriverApi.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    out int maxBlocksPerSm, _gdnScanStepF32CoopSplit4Func, dState, sharedBytesForQuery);
                _gdnScanCoopSplit4MaxCoResidentGrid = rc == 0 ? numSms * maxBlocksPerSm : 0;
            }
        }

        return _gdnScanCoopSplit4MaxCoResidentGrid > 0 && nVHead * 4 <= _gdnScanCoopSplit4MaxCoResidentGrid;
    }

    /// <summary>
    /// OPT-IN, default-OFF row-split cooperative-groups variant of <see cref="LaunchGdnScanStepF32"/>
    /// (issue #180) — splits each V-head's [dState,dState] state-matrix row range across 4
    /// cooperating blocks (grid = nVHead*4 instead of nVHead) using CUDA Cooperative Groups
    /// <c>grid.sync()</c>, still exactly ONE kernel launch. Measured ~26-27% faster kernel time on
    /// RTX 3060 (real cudaEvent/CUevent timing, both standalone and via this exact PTX-JIT +
    /// driver-API path) — but is NOT bit-exact vs the CPU oracle (reassociated float reduction;
    /// see <see cref="EnableGdnScanApproxSplit4"/>'s doc and gated_delta_net_scan.cu's header for
    /// the full tradeoff). Callers MUST check <see cref="IsGdnScanCoopSplit4Safe"/> first (exceeding
    /// the cooperative-launch co-residency ceiling is a hard CUDA error) and fall back to
    /// <see cref="LaunchGdnScanStepF32"/> if it returns false.
    /// </summary>
    /// <param name="state">Device pointer to the GDN recurrence state, <c>[nVHead, dState, dState]</c> (in/out).</param>
    /// <param name="qT">Device pointer to this token's Q vectors, <c>[nKHead, dState]</c>, L2-normalised.</param>
    /// <param name="kT">Device pointer to this token's K vectors, <c>[nKHead, dState]</c>, L2-normalised.</param>
    /// <param name="vT">Device pointer to this token's V vectors, <c>[nVHead, dState]</c>.</param>
    /// <param name="gT">Device pointer to this token's per-head decay scalars, <c>[nVHead]</c>.</param>
    /// <param name="betaT">Device pointer to this token's per-head write-gate scalars, <c>[nVHead]</c>.</param>
    /// <param name="outputT">Device pointer to this token's output, <c>[nVHead, dState]</c>. Overwritten.</param>
    /// <param name="partialTmp">Scratch, <c>[nVHead, 4, dState]</c> floats — retrieve-phase partials.</param>
    /// <param name="partialOut">Scratch, <c>[nVHead, 4, dState]</c> floats — read-phase partials.</param>
    /// <param name="nVHead">Number of value heads.</param>
    /// <param name="nKHead">Number of key heads (must divide <paramref name="nVHead"/> evenly).</param>
    /// <param name="dState">Per-head state dimension. Must be exactly 128 (SPLIT=4 hardcoded).</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchGdnScanStepF32CoopSplit4(nint state, nint qT, nint kT, nint vT,
                                       nint gT, nint betaT, nint outputT,
                                       nint partialTmp, nint partialOut,
                                       int nVHead, int nKHead, int dState, nint stream)
    {
        if (dState != 128)
            throw new ArgumentOutOfRangeException(nameof(dState),
                $"dState={dState}; gdn_scan_step_f32_coop_split4 hardcodes SPLIT=4 dividing dState=128 evenly.");

        nint sArg = state, qArg = qT, kArg = kT, vArg = vT;
        nint gArg = gT, bArg = betaT, oArg = outputT;
        nint ptArg = partialTmp, poArg = partialOut;
        int nvArg = nVHead, nkArg = nKHead, dsArg = dState;

        void** args = stackalloc void*[] {&sArg, &qArg, &kArg, &vArg,
                        &gArg, &bArg, &oArg, &ptArg, &poArg,
                        &nvArg, &nkArg, &dsArg};

        // Shared memory: k_shared[dState] + q_shared[dState] (same layout as the non-split kernel).
        uint sharedBytes = (uint)(2 * dState * sizeof(float));

        CudaDriverApi.cuLaunchCooperativeKernel(_gdnScanStepF32CoopSplit4Func,
                (uint)nVHead, 4, 1, (uint)dState, 1, 1,
                sharedBytes, stream, (nint)args).ThrowOnError();
    }

    /// <summary>
    /// In-place per-head L2 normalisation (FP32). Bit-perfect port of the CPU
    /// <c>GatedDeltaNetScan.L2NormalizeHeads</c> reference — the sum-of-squares
    /// is computed serially in thread 0 of each block to preserve the CPU's
    /// 0..dState-1 float-add order exactly.
    /// </summary>
    /// <param name="x">
    /// Device pointer to a flat FP32 buffer of <c>totalHeads × dState</c> elements.
    /// Each <c>dState</c>-element slice is normalised independently to unit norm.
    /// </param>
    /// <param name="totalHeads">Number of head vectors to normalise (e.g. <c>seqLen × nKHead</c>).</param>
    /// <param name="dState">Dimension of each head vector.</param>
    /// <param name="eps">Epsilon added to the L2 norm for numerical stability (default 1e-6 in the CPU code).</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchL2NormalizeHeadsF32(nint x, int totalHeads, int dState, float eps, nint stream)
    {
        if (dState <= 0 || dState > 128)
            throw new ArgumentOutOfRangeException(nameof(dState),
                $"dState={dState}; l2_normalize_heads_f32 is compiled with __launch_bounds__(128).");

        nint xArg = x;
        int thArg = totalHeads, dsArg = dState;
        float epsArg = eps;

        void** args = stackalloc void*[] {&xArg, &thArg, &dsArg, &epsArg};

        // One block per head; dState threads per block. The kernel's serial-sum
        // phase runs in thread 0 only, but the broadcast multiply uses all
        // threads via a grid-stride loop, so blockDim can equal dState exactly.
        CudaDriverApi.cuLaunchKernel(_l2NormHeadsF32Func,
                (uint)totalHeads, 1, 1, (uint)dState, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// True when the fused decode-time GDN deinterleave+L2-normalize kernel
    /// (<see cref="LaunchGdnDeinterleaveL2NormDecodeF32"/>) is loaded. When false,
    /// callers fall back to <see cref="LaunchDeinterleaveGdnQkvF32"/> (or the host-loop
    /// fallback) plus two separate <see cref="LaunchL2NormalizeHeadsF32"/> calls.
    /// </summary>
    public bool HasGdnDeinterleaveL2NormDecodeF32 => _gdnDeinterleaveL2NormDecodeF32Func != 0;

    /// <summary>
    /// Decode-time (seqLen==1) fusion of <see cref="LaunchDeinterleaveGdnQkvF32"/> + two
    /// <see cref="LaunchL2NormalizeHeadsF32"/> calls (Q then K) into one launch. Also removes
    /// the runtime integer division/modulo <c>deinterleave_gdn_qkv_f32</c> pays per element for
    /// the general (seqLen&gt;1) case — SASS-confirmed (issue #170) — since decode's single row
    /// makes every index directly computable from block/thread indices alone. Requires
    /// <c>n_k_head*dState</c> and <c>n_v_head*dState</c> (the actual Q/K and V buffer sizes) to
    /// each be exact multiples of <paramref name="dState"/>, which always holds for GDN.
    /// </summary>
    /// <param name="src">Device pointer to the single decode row: <c>[Q(kDim) | K(kDim) | V(vDim)]</c>.</param>
    /// <param name="q">Destination for the L2-normalized Q heads, <c>[kDim]</c>.</param>
    /// <param name="k">Destination for the L2-normalized K heads, <c>[kDim]</c>.</param>
    /// <param name="v">Destination for the straight-copied V heads, <c>[vDim]</c>.</param>
    /// <param name="nKHead">Number of key/query heads (kDim = nKHead * dState).</param>
    /// <param name="nVHead">Number of value heads (vDim = nVHead * dState).</param>
    /// <param name="dState">Per-head dimension.</param>
    /// <param name="eps">Epsilon added to the L2 norm, matching <see cref="LaunchL2NormalizeHeadsF32"/>.</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchGdnDeinterleaveL2NormDecodeF32(
        nint src, nint q, nint k, nint v, int nKHead, int nVHead, int dState, float eps, nint stream)
    {
        if (dState <= 0 || dState > 128)
            throw new ArgumentOutOfRangeException(nameof(dState),
                $"dState={dState}; gdn_deinterleave_l2norm_decode_f32 is compiled with __launch_bounds__(128).");

        nint srcArg = src, qArg = q, kArg = k, vArg = v;
        int nkArg = nKHead, dsArg = dState;
        float epsArg = eps;
        void** args = stackalloc void*[] {&srcArg, &qArg, &kArg, &vArg, &nkArg, &dsArg, &epsArg};

        uint gridDim = (uint)(2 * nKHead + nVHead);

        CudaDriverApi.cuLaunchKernel(_gdnDeinterleaveL2NormDecodeF32Func,
                gridDim, 1, 1, (uint)dState, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Per-token GDN decay: in place transforms <paramref name="alphaBuf"/> via
    /// <c>g[t, vh] = exp(softplus(alpha[t, vh] + dt_bias[vh]) * A[vh])</c>, with
    /// <c>softplus(x) = log(1 + exp(x))</c> (no <c>x &gt; 20</c> short-circuit —
    /// matches the Qwen3MoeHybrid CPU oracle exactly, see
    /// <c>Qwen3MoeHybridTransformerModel.ForwardGdnBody</c>).
    /// </summary>
    /// <remarks>
    /// Compiled with <c>-fmad=false</c>; CUDA's precise expf/logf are within
    /// ≤1 ULP of <see cref="MathF.Exp(float)"/> / <see cref="MathF.Log(float)"/>
    /// on Ampere+, so the result matches the CPU host fallback to within
    /// negligible tolerance — not strictly bit-equal across every input, but
    /// numerically equivalent for any well-conditioned alpha.
    /// </remarks>
    /// <param name="alphaBuf">Device pointer to <c>[seqLen, nVHead]</c> FP32, in/out.</param>
    /// <param name="dtBias">Device pointer to per-head FP32 bias, <c>[nVHead]</c>.</param>
    /// <param name="a">Device pointer to per-head FP32 decay coefficient, <c>[nVHead]</c>.</param>
    /// <param name="seqLen">Number of tokens (rows of <paramref name="alphaBuf"/>).</param>
    /// <param name="nVHead">Number of value heads (columns of <paramref name="alphaBuf"/>).</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchGdnDecayF32(nint alphaBuf, nint dtBias, nint a,
                                    int seqLen, int nVHead, nint stream)
    {
        nint alphaArg = alphaBuf, dtArg = dtBias, aArg = a;
        int slArg = seqLen, nvArg = nVHead;

        void** args = stackalloc void*[] {&alphaArg, &dtArg, &aArg, &slArg, &nvArg};

        int total = seqLen * nVHead;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_gdnDecayF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// True when the fused decay+sigmoid kernel (<see cref="LaunchGdnDecaySigmoidF32"/>) is
    /// loaded. When false, callers fall back to separate <see cref="LaunchGdnDecayF32"/> +
    /// <see cref="LaunchSigmoidF32"/> calls.
    /// </summary>
    public bool HasGdnDecaySigmoidF32 => _gdnDecaySigmoidF32Func != 0;

    /// <summary>
    /// Fused <see cref="LaunchGdnDecayF32"/> (on <paramref name="alphaBuf"/>) +
    /// in-place sigmoid (on the independent, identically-shaped <paramref name="betaBuf"/>) —
    /// GDN's decode path always calls both back-to-back on two same-size <c>[seqLen, nVHead]</c>
    /// buffers, so one launch handling both halves the launch count. Numerics are byte-for-byte
    /// identical to the two separate calls (same translation unit, same <c>-fmad=false</c>
    /// compile flag).
    /// </summary>
    public void LaunchGdnDecaySigmoidF32(nint alphaBuf, nint betaBuf, nint dtBias, nint a,
                                           int seqLen, int nVHead, nint stream)
    {
        nint alphaArg = alphaBuf, betaArg = betaBuf, dtArg = dtBias, aArg = a;
        int slArg = seqLen, nvArg = nVHead;

        void** args = stackalloc void*[] {&alphaArg, &betaArg, &dtArg, &aArg, &slArg, &nvArg};

        int total = seqLen * nVHead;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_gdnDecaySigmoidF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// In-place elementwise sigmoid: <c>buf[i] = 1 / (1 + exp(-buf[i]))</c>.
    /// Matches the host fallback at
    /// <c>CudaQwen3MoeHybridTransformerModel.LaunchSigmoidHostFallback</c>.
    /// </summary>
    /// <param name="buf">Device pointer to FP32 buffer of length <paramref name="n"/>.</param>
    /// <param name="n">Number of elements.</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchSigmoidF32(nint buf, long n, nint stream)
    {
        if (n <= 0) return;
        if (n > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(n),
                $"n={n}; sigmoid_f32 takes an int count to match CUDA driver param size.");

        nint bufArg = buf;
        int nArg = (int)n;

        void** args = stackalloc void*[] {&bufArg, &nArg};

        uint gridDim = (uint)((n + BlockSize - 1) / BlockSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_sigmoidF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// In-place elementwise SiLU: <c>buf[i] = buf[i] * sigmoid(buf[i])</c>.
    /// Matches the host fallback at
    /// <c>CudaQwen3MoeHybridTransformerModel.LaunchSiluHostFallback</c>.
    /// </summary>
    /// <param name="buf">Device pointer to FP32 buffer of length <paramref name="n"/>.</param>
    /// <param name="n">Number of elements.</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchSiluF32(nint buf, long n, nint stream)
    {
        if (n <= 0) return;
        if (n > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(n),
                $"n={n}; silu_f32 takes an int count to match CUDA driver param size.");

        nint bufArg = buf;
        int nArg = (int)n;

        void** args = stackalloc void*[] {&bufArg, &nArg};

        uint gridDim = (uint)((n + BlockSize - 1) / BlockSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_siluF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Plain elementwise squared-ReLU in place: x = max(0,x)^2. NVIDIA Nemotron-H's
    /// non-gated FFN activation — distinct from <see cref="LaunchReLU2F32"/>'s GLU-fused form.
    /// See native/kernels/relu_squared_inplace.cu.</summary>
    public void LaunchReluSquaredInplaceF32(nint x, int n, nint stream)
    {
        if (n <= 0) return;
        nint xArg = x;
        int nArg = n;
        void** args = stackalloc void*[] {&xArg, &nArg};
        uint gridDim = (uint)((n + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_reluSquaredInplaceF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Elementwise <c>a[i] *= sigmoid(b[i])</c>, in place on <paramref name="a"/>.
    /// Matches the host fallback at
    /// <c>CudaQwen3MoeHybridTransformerModel.LaunchSigmoidMulHostFallback</c>.
    /// </summary>
    /// <param name="a">Device pointer to FP32 in/out buffer of length <paramref name="n"/>.</param>
    /// <param name="b">Device pointer to FP32 gate buffer of length <paramref name="n"/>.</param>
    /// <param name="n">Number of elements.</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchSigmoidMulF32(nint a, nint b, long n, nint stream)
    {
        if (n <= 0) return;
        if (n > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(n),
                $"n={n}; sigmoid_mul_f32 takes an int count to match CUDA driver param size.");

        nint aArg = a, bArg = b;
        int nArg = (int)n;

        void** args = stackalloc void*[] {&aArg, &bArg, &nArg};

        uint gridDim = (uint)((n + BlockSize - 1) / BlockSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_sigmoidMulF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Gather-kernel replacement for the decode-time host loop that split a fused Q+Gate
    /// projection's output into separate Q and Gate tensors via numHeads separate
    /// <c>cuMemcpyDtoDAsync</c> calls. Per-token <paramref name="qg"/> layout:
    /// <c>[Q_h0(headDim), Gate_h0(headDim), Q_h1(headDim), Gate_h1(headDim), ...]</c>.
    /// </summary>
    public void LaunchDeinterleaveQGateF32(nint qg, nint q, nint gate, int numHeads, int headDim, int seqLen, nint stream)
    {
        nint qgArg = qg, qArg = q, gateArg = gate;
        int nhArg = numHeads, hdArg = headDim, slArg = seqLen;
        void** args = stackalloc void*[] {&qgArg, &qArg, &gateArg, &nhArg, &hdArg, &slArg};

        long total = (long)seqLen * numHeads * headDim;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_deinterleaveQGateF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Gather-kernel replacement for the decode-time host loop that split a GDN layer's fused
    /// conv1d output into separate Q/K/V tensors. Per-token <paramref name="src"/> layout:
    /// <c>[Q(kDim) | K(kDim) | V(vDim)]</c>.
    /// </summary>
    public void LaunchDeinterleaveGdnQkvF32(nint src, nint q, nint k, nint v, int kDim, int vDim, int seqLen, nint stream)
    {
        nint srcArg = src, qArg = q, kArg = k, vArg = v;
        int kdArg = kDim, vdArg = vDim, slArg = seqLen;
        void** args = stackalloc void*[] {&srcArg, &qArg, &kArg, &vArg, &kdArg, &vdArg, &slArg};

        long total = (long)seqLen * (2 * kDim + vDim);
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_deinterleaveGdnQkvF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Rotary position embedding. In-place on Q and K.</summary>
    public void LaunchRoPE(nint q, nint k, nint positions,
                            int seqLen, int numHeads, int numKvHeads, int headDim,
                            int ropeDim, float theta, int ropeType, nint stream)
    {
        nint qArg = q, kArg = k, posArg = positions;
        int slArg = seqLen, nhArg = numHeads, nkvArg = numKvHeads;
        int hdArg = headDim, rdArg = ropeDim, rtArg = ropeType;
        float thetaArg = theta;

        void** args = stackalloc void*[] {&qArg, &kArg, &posArg, &slArg, &nhArg, &nkvArg,
                        &hdArg, &rdArg, &thetaArg, &rtArg};

        int halfRope = ropeDim / 2;
        int totalPairs = seqLen * Math.Max(numHeads, numKvHeads) * halfRope;
        uint gridDim = (uint)((totalPairs + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_ropeFunc,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused SwiGLU: out = SiLU(gate) * up. half2 vectorized (2 elements/thread).</summary>
    public void LaunchSwiGLU(nint gate, nint up, nint output,
                              int n, int seqLen, nint stream)
    {
        nint gateArg = gate, upArg = up, outArg = output;
        int nArg = n, slArg = seqLen;

        void** args = stackalloc void*[] {&gateArg, &upArg, &outArg, &nArg, &slArg};
        int total = n * seqLen;
        // half2: each thread processes 2 elements
        uint gridDim = (uint)((total / 2 + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_swigluFunc,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Element-wise add: output = a + b. half2 vectorized (2 elements/thread).</summary>
    public void LaunchAdd(nint a, nint b, nint output, int n, nint stream)
    {
        nint aArg = a, bArg = b, outArg = output;
        int nArg = n;

        void** args = stackalloc void*[] {&aArg, &bArg, &outArg, &nArg};
        // half2: each thread processes 2 elements
        uint gridDim = (uint)((n / 2 + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_addFunc,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Softmax over last dimension. One block per row.</summary>
    /// <remarks>
    /// Not used in the GPU forward pass — softmax is fused into <c>attention.cu</c>.
    /// Available for standalone use and testing.
    /// </remarks>
    public void LaunchSoftmax(nint input, nint output, int rows, int cols, nint stream)
    {
        nint inputArg = input, outputArg = output;
        int rowsArg = rows, colsArg = cols;

        void** args = stackalloc void*[] {&inputArg, &outputArg, &rowsArg, &colsArg};

        CudaDriverApi.cuLaunchKernel(_softmaxFunc,
                (uint)rows, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Causal softmax over the column-major <c>[s x s]</c> FP16 score buffer produced by
    /// the cuBLAS tensor-core prefill-attention path. Operates in place: one plane per
    /// query head (base <c>hq * s * s</c>), element <c>(tq, tk)</c> at <c>tq + tk * s</c>.
    /// Each block softmaxes one query row (fixed <c>tq</c>) over the strided key axis,
    /// applies the causal mask (<c>tk &gt; tq</c> → 0), and normalizes. The QK GEMM
    /// already applied the <c>1/sqrt(headDim)</c> scale; it is not re-applied here.
    /// One block per <c>(query head, query row)</c>; grid = <c>numHeads * s</c>.
    /// </summary>
    /// <param name="scores">Device pointer to the FP16 score buffer (<c>numHeads * s * s</c> elements), modified in place.</param>
    /// <param name="s">Sequence length (rows and columns of each per-head score plane).</param>
    /// <param name="numHeads">Number of query heads (score planes).</param>
    /// <param name="stream">CUDA stream handle.</param>
    /// <param name="coalesced">
    /// When true (default), uses the one-thread-per-row variant whose global reads are
    /// coalesced — the preferred path. When false, uses the one-block-per-row variant
    /// (warp-reduced per row, but strided/uncoalesced reads) for A/B comparison.
    /// </param>
    public void LaunchAttentionSoftmaxCausal(nint scores, int s, int numHeads, nint stream, bool coalesced = true)
    {
        nint scoresArg = scores;
        int sArg = s, nhArg = numHeads;

        void** args = stackalloc void*[] {&scoresArg, &sArg, &nhArg};

        if (coalesced)
        {
            // One thread per row; grid sized to cover numHeads*s rows.
            int totalRows = numHeads * s;
            uint gridDim = (uint)((totalRows + BlockSize - 1) / BlockSize);
            CudaDriverApi.cuLaunchKernel(_attentionSoftmaxCausalCoalescedFunc,
                    gridDim, 1, 1, BlockSize, 1, 1,
                    0, stream, (nint)args, 0).ThrowOnError();
        }
        else
        {
            // One block per row.
            int numBlocks = numHeads * s;
            CudaDriverApi.cuLaunchKernel(_attentionSoftmaxCausalFunc,
                    (uint)numBlocks, 1, 1, BlockSize, 1, 1,
                    0, stream, (nint)args, 0).ThrowOnError();
        }
    }

    /// <summary>
    /// Launches the FP32-scores causal softmax: reads FP32 QK <paramref name="scores"/>
    /// (per-head col-major [s × s]), writes normalized FP16 probs into
    /// <paramref name="probs"/> with the same layout. Coalesced one-thread-per-row.
    /// </summary>
    /// <param name="scores">Device pointer to FP32 QK scores ([numHeads × s × s]).</param>
    /// <param name="probs">Device pointer to FP16 output probs ([numHeads × s × s]).</param>
    /// <param name="s">Sequence length (square scores).</param>
    /// <param name="numHeads">Number of query heads (score planes).</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchAttentionSoftmaxCausalF32In(nint scores, nint probs, int s, int numHeads, nint stream)
    {
        nint scoresArg = scores, probsArg = probs;
        int sArg = s, nhArg = numHeads;
        void** args = stackalloc void*[] {&scoresArg, &probsArg, &sArg, &nhArg};

        int totalRows = numHeads * s;
        uint gridDim = (uint)((totalRows + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_attentionSoftmaxCausalCoalescedF32InFunc,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Launches the hand-fused mma.sync flash-attention prefill kernel. Prototype:
    /// headDim must be 64, causal, position_offset 0, FP16 in/out, output row-major
    /// [seq, numHeads, headDim] (matching <c>attention_f16</c>). One warp per
    /// (query head, 16-query tile).
    /// </summary>
    /// <param name="q">Device pointer to Q [seq, numHeads, headDim] FP16.</param>
    /// <param name="k">Device pointer to K [seq, numKvHeads, headDim] FP16.</param>
    /// <param name="v">Device pointer to V [seq, numKvHeads, headDim] FP16.</param>
    /// <param name="output">Device pointer to output [seq, numHeads, headDim] FP16.</param>
    /// <param name="seq">Sequence length (prefill).</param>
    /// <param name="numHeads">Number of query heads.</param>
    /// <param name="numKvHeads">Number of KV heads (GQA).</param>
    /// <param name="headDim">Head dimension; must be 64 for this prototype.</param>
    /// <param name="scale">QK softmax scale (1/sqrt(headDim)).</param>
    /// <param name="stream">CUDA stream handle.</param>
    public void LaunchAttentionFlashMma(nint q, nint k, nint v, nint output,
        int seq, int numHeads, int numKvHeads, int headDim, float scale, nint stream)
    {
        nint qArg = q, kArg = k, vArg = v, oArg = output;
        int seqArg = seq, nhArg = numHeads, nkvArg = numKvHeads;
        float scArg = scale;

        void** args = stackalloc void*[] { &qArg, &kArg, &vArg, &oArg, &seqArg, &nhArg, &nkvArg, &scArg };

        // TUNED layout: one block per (kv head, 16-query tile); `group` warps per block,
        // one warp per query head sharing that kv head (K/V loaded once, reused group×).
        int group = numHeads / numKvHeads;
        uint gridX = (uint)numKvHeads;
        uint gridY = (uint)((seq + 15) / 16);   // 16-query tiles
        uint blockX = (uint)(group * 32);       // group warps per block
        CudaDriverApi.cuLaunchKernel(_attentionFlashMmaFunc,
                gridX, gridY, 1, blockX, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Embedding lookup with per-format dispatch.</summary>
    /// <remarks>
    /// Per-row K-quant lookups (Q4_K/Q5_K/Q6_K) require <paramref name="hiddenSize"/>
    /// to be a multiple of 256 (the K-quant super-block size). Caller
    /// (<see cref="HasEmbeddingLookup"/>) gates this — types without an available
    /// per-row kernel must be dequant-expanded to FP16 at load.
    /// </remarks>
    public void LaunchEmbeddingLookup(nint embedTable, QuantizationType embedDtype,
                                       nint tokenIds, nint output,
                                       int seqLen, int hiddenSize, nint stream)
    {
        nint tableArg = embedTable, idsArg = tokenIds, outArg = output;
        int slArg = seqLen, hsArg = hiddenSize;

        nint func = embedDtype switch
        {
            QuantizationType.F32 => _embeddingF32Func,
            QuantizationType.F16 => _embeddingF16Func,
            QuantizationType.Q8_0 => _embeddingQ8_0Func,
            QuantizationType.Q4_K => _embeddingQ4_KFunc,
            QuantizationType.Q5_K => _embeddingQ5_KFunc,
            QuantizationType.Q6_K => _embeddingQ6_KFunc,
            _ => 0,
        };

        if (func == 0)
            throw new NotSupportedException($"Embedding type {embedDtype} not supported on GPU.");

        void** args = stackalloc void*[] {&tableArg, &idsArg, &outArg, &slArg, &hsArg};

        CudaDriverApi.cuLaunchKernel(func,
                (uint)seqLen, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// True when a per-row embedding lookup kernel exists for <paramref name="qt"/>
    /// at the given <paramref name="hiddenSize"/>. K-quant variants require
    /// <c>hiddenSize % 256 == 0</c> (the super-block size); other types have no
    /// such constraint.
    /// </summary>
    public bool HasEmbeddingLookup(QuantizationType qt, int hiddenSize) => qt switch
    {
        QuantizationType.F32 or QuantizationType.F16 or QuantizationType.Q8_0 => true,
        QuantizationType.Q4_K => _embeddingQ4_KFunc != 0 && (hiddenSize % 256) == 0,
        QuantizationType.Q5_K => _embeddingQ5_KFunc != 0 && (hiddenSize % 256) == 0,
        QuantizationType.Q6_K => _embeddingQ6_KFunc != 0 && (hiddenSize % 256) == 0,
        _ => false,
    };

    /// <summary>Naive scaled dot-product attention with causal mask and GQA.</summary>
    public void LaunchAttention(nint q, nint k, nint v, nint output,
                                 int seqQ, int seqKv,
                                 int numHeads, int numKvHeads, int headDim,
                                 int positionOffset, int slidingWindow, nint stream)
    {
        nint qArg = q, kArg = k, vArg = v, outArg = output;
        int sqArg = seqQ, skvArg = seqKv;
        int nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim;
        int poArg = positionOffset, swArg = slidingWindow;

        void** args = stackalloc void*[] {&qArg, &kArg, &vArg, &outArg,
                        &sqArg, &skvArg, &nhArg, &nkvArg, &hdArg,
                        &poArg, &swArg};

        int numBlocks = seqQ * numHeads;
        // Tiled online softmax: q_shared[headDim] + score_tile[256] + out_accum[headDim] + warp_scratch[32]
        const int TileKv = 256;
        uint sharedBytes = (uint)((headDim + TileKv + headDim + 32) * sizeof(float));

        CudaDriverApi.cuLaunchKernel(_attentionFunc,
                (uint)numBlocks, 1, 1, BlockSize, 1, 1,
                sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Decode-step attention with device-resident <c>seqKv</c> and <c>positionOffset</c>.
    /// Identical body to <see cref="LaunchAttention"/> but reads the two scalars via
    /// pointer dereferences inside the kernel — letting CUDA Graphs replay the same
    /// instantiated graph across decode steps where only the KV-cache length changes.
    /// <c>seqKvPtr</c> and <c>positionOffsetPtr</c> are device pointers to 4-byte
    /// ints; the host bumps them via <c>cuMemcpyHtoD</c> between <c>cuGraphLaunch</c> calls.
    /// </summary>
    #pragma warning disable CS1573 // match LaunchAttention; params are self-documenting
    public void LaunchAttentionDyn(nint q, nint k, nint v, nint output,
                                    int seqQ, nint seqKvPtr,
                                    int numHeads, int numKvHeads, int headDim,
                                    nint positionOffsetPtr, int slidingWindow, nint stream)
    {
        nint qArg = q, kArg = k, vArg = v, outArg = output;
        int sqArg = seqQ;
        nint skvPtrArg = seqKvPtr;
        int nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim;
        nint poPtrArg = positionOffsetPtr;
        int swArg = slidingWindow;

        void** args = stackalloc void*[] {&qArg, &kArg, &vArg, &outArg,
                        &sqArg, &skvPtrArg, &nhArg, &nkvArg, &hdArg,
                        &poPtrArg, &swArg};

        int numBlocks = seqQ * numHeads;
        const int TileKv = 256;
        uint sharedBytes = (uint)((headDim + TileKv + headDim + 32) * sizeof(float));

        CudaDriverApi.cuLaunchKernel(_attentionDynFunc,
                (uint)numBlocks, 1, 1, BlockSize, 1, 1,
                sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }
    #pragma warning restore CS1573

    /// <summary>
    /// Whether the opt-in direct block-table-read paged decode attention kernel (issue #200,
    /// <c>attention_f16_paged</c>) is available in the loaded PTX. False on a stale build --
    /// callers must keep using <see cref="CudaPagedKvCache"/>'s gather-based
    /// <c>PrepareAttentionScratch</c> + <see cref="LaunchAttention"/> path unconditionally.
    /// </summary>
    public bool HasAttentionF16Paged => _attentionPagedFunc != 0;

    /// <summary>
    /// Enables <see cref="LaunchAttentionPaged"/> in <c>CudaTransformerModel</c>'s paged-cache
    /// decode dispatch, in place of the default gather-into-scratch path. Default OFF (this
    /// kernel has not been measured against the gather path yet -- see issue #200's own
    /// "wire in behind an opt-in flag" requirement and <c>docs/perf/CUDA_PAGED_ATTENTION_DESIGN.md</c>).
    /// Set <c>DOTLLM_ATTN_PAGED_NATIVE=1</c> to opt in.
    /// </summary>
    public static bool EnableNativePagedAttention { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_ATTN_PAGED_NATIVE") == "1";

    /// <summary>
    /// Issue #200: direct block-table-read paged decode attention. Same online-softmax math
    /// and launch geometry as <see cref="LaunchAttention"/> (one block per (query_token,
    /// query_head), same tiled shared-memory layout) -- the only difference is that K/V rows
    /// are resolved through <paramref name="kBlockPtrs"/>/<paramref name="vBlockPtrs"/> (a
    /// small per-layer device array of block base pointers, see
    /// <see cref="CudaPagedKvCache.PrepareNativeBlockPtrs"/>) instead of a flat contiguous
    /// buffer, eliminating the D2D gather <see cref="CudaPagedKvCache.PrepareAttentionScratch"/>
    /// would otherwise need to run before this launch.
    /// </summary>
    /// <param name="kBlockPtrs">Device pointer to an array of <c>ceil(seqKv/blockSize)</c> device pointers (K blocks, this layer).</param>
    /// <param name="vBlockPtrs">Device pointer to an array of <c>ceil(seqKv/blockSize)</c> device pointers (V blocks, this layer).</param>
    /// <param name="blockSize">Tokens per block (<see cref="CudaKvBlockPool.BlockSize"/>).</param>
    #pragma warning disable CS1573 // match LaunchAttention; remaining params are self-documenting
    public void LaunchAttentionPaged(nint q, nint kBlockPtrs, nint vBlockPtrs, nint output,
                                      int seqQ, int seqKv, int blockSize,
                                      int numHeads, int numKvHeads, int headDim,
                                      int positionOffset, int slidingWindow, nint stream)
    {
        if (_attentionPagedFunc == 0)
            throw new InvalidOperationException(
                "attention_f16_paged is not available in the loaded PTX (stale build). " +
                "Check HasAttentionF16Paged before calling LaunchAttentionPaged.");

        nint qArg = q, kbpArg = kBlockPtrs, vbpArg = vBlockPtrs, outArg = output;
        int sqArg = seqQ, skvArg = seqKv, bsArg = blockSize;
        int nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim;
        int poArg = positionOffset, swArg = slidingWindow;

        void** args = stackalloc void*[] {&qArg, &kbpArg, &vbpArg, &outArg,
                        &sqArg, &skvArg, &bsArg, &nhArg, &nkvArg, &hdArg,
                        &poArg, &swArg};

        int numBlocks = seqQ * numHeads;
        // Same shared-memory layout/size as LaunchAttention (q_shared+score_tile+out_accum+warp_scratch).
        const int TileKv = 256;
        uint sharedBytes = (uint)((headDim + TileKv + headDim + 32) * sizeof(float));

        CudaDriverApi.cuLaunchKernel(_attentionPagedFunc,
                (uint)numBlocks, 1, 1, BlockSize, 1, 1,
                sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }
    #pragma warning restore CS1573

    /// <summary>
    /// Diagnostic-only (issue #213): reports register/local-memory usage for the
    /// <c>attention_f16</c> vs <c>attention_f16_dyn</c> compiled functions, to check for an
    /// occupancy-affecting compiled-code difference between the two entry points without
    /// needing Nsight Compute. Returns (numRegsScalar, numRegsDyn, localBytesScalar, localBytesDyn).
    /// </summary>
    internal (int regsScalar, int regsDyn, int localScalar, int localDyn) DebugGetAttentionFuncStats()
    {
        CudaDriverApi.cuFuncGetAttribute(out int regsScalar,
            CudaDriverApi.CU_FUNC_ATTRIBUTE_NUM_REGS, _attentionFunc).ThrowOnError();
        CudaDriverApi.cuFuncGetAttribute(out int regsDyn,
            CudaDriverApi.CU_FUNC_ATTRIBUTE_NUM_REGS, _attentionDynFunc).ThrowOnError();
        CudaDriverApi.cuFuncGetAttribute(out int localScalar,
            CudaDriverApi.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, _attentionFunc).ThrowOnError();
        CudaDriverApi.cuFuncGetAttribute(out int localDyn,
            CudaDriverApi.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, _attentionDynFunc).ThrowOnError();
        return (regsScalar, regsDyn, localScalar, localDyn);
    }

    /// <summary>Attention variant that reads the query position from device memory.</summary>
    public void LaunchAttentionPos(nint q, nint k, nint v, nint output, nint positions,
                                   int seqQ, int seqKv,
                                   int numHeads, int numKvHeads, int headDim,
                                   int slidingWindow, nint stream)
    {
        nint qArg = q, kArg = k, vArg = v, outArg = output, posArg = positions;
        int sqArg = seqQ, skvArg = seqKv;
        int nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim;
        int swArg = slidingWindow;

        void** args = stackalloc void*[] {&qArg, &kArg, &vArg, &outArg, &posArg,
                        &sqArg, &skvArg, &nhArg, &nkvArg, &hdArg, &swArg};

        int numBlocks = seqQ * numHeads;
        const int TileKv = 256;
        uint sharedBytes = (uint)((headDim + TileKv + headDim + 32) * sizeof(float));

        CudaDriverApi.cuLaunchKernel(_attentionPosFunc,
                (uint)numBlocks, 1, 1, BlockSize, 1, 1,
                sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>True when the graph-friendly KV write kernel is loaded (PTX present).</summary>
    public bool HasKvWriteKernel => _kvWriteModule != null;

    /// <summary>
    /// Writes one row of <paramref name="kvStride"/> FP16 elements from
    /// <paramref name="src"/> to <paramref name="dstBase"/><c> + posPtr[0] * kvStride</c>.
    /// Replaces a host-side <c>cuMemcpyDtoDAsync</c> for the decode KV-cache
    /// update so the destination address is computed device-side and stable
    /// across <c>cuGraphLaunch</c> replays.
    /// </summary>
    public void LaunchKvWriteOneF16(nint src, nint dstBase, int kvStride, nint posPtr, nint stream)
    {
        if (_kvWriteModule == null)
            throw new InvalidOperationException(
                "KV write kernel not available. Compile native/kernels/kv_write.cu to PTX.");

        nint srcArg = src, dstArg = dstBase, posArg = posPtr;
        int kvArg = kvStride;
        void** args = stackalloc void*[] { &srcArg, &dstArg, &kvArg, &posArg };
        uint gridDim = (uint)((kvStride + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_kvWriteOneF16Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Writes a single FP16 row into the per-layer ring buffer at slot
    /// <c>posPtr[0] % windowSize</c>. Graph-friendly counterpart to a
    /// host-computed <c>cuMemcpyDtoDAsync</c> into <c>_keysWindow[layer]</c>.
    /// </summary>
    public void LaunchKvWriteOneF16Ring(nint src, nint ringBase, int kvStride, int windowSize,
                                         nint posPtr, nint stream)
    {
        if (_kvWriteModule == null)
            throw new InvalidOperationException(
                "KV write kernel not available. Compile native/kernels/kv_write.cu to PTX.");

        nint srcArg = src, ringArg = ringBase, posArg = posPtr;
        int kvArg = kvStride, wsArg = windowSize;
        void** args = stackalloc void*[] { &srcArg, &ringArg, &kvArg, &wsArg, &posArg };
        uint gridDim = (uint)((kvStride + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_kvWriteOneF16RingFunc,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Dequantizes the <c>[0, max(0, posPtr[0] + 1 - windowSize))</c> prefix of the per-layer
    /// quantized cache into FP16 attention scratch. Predicated: no-op when the FP16 window
    /// hasn't yet started evicting. Grid is sized for the maximum quantized region; the
    /// kernel's grid-stride loop bounds the work to the live prefix.
    /// </summary>
    public void LaunchKvDequantDyn(nint quantBase, nint scratchBase, int kvStride, int windowSize,
                                     int maxSeqLen, KvCacheDType dtype, nint posPtr, nint stream)
    {
        if (_kvWriteModule == null)
            throw new InvalidOperationException(
                "KV write kernel not available. Compile native/kernels/kv_write.cu to PTX.");

        nint qArg = quantBase, sArg = scratchBase, posArg = posPtr;
        int kvArg = kvStride, wsArg = windowSize;
        void** args = stackalloc void*[] { &qArg, &sArg, &kvArg, &wsArg, &posArg };

        // Match LaunchDequantToF16's grid sizing: cap at MaxDequantGridSize CUDA blocks.
        // Each CUDA block has 8 warps (one per quant block of 32 elements).
        int blocksPerRow = kvStride / 32;
        int maxQuantRows = Math.Max(0, maxSeqLen - windowSize);
        int totalBlocks = Math.Max(blocksPerRow, maxQuantRows * blocksPerRow);
        uint gridDim = (uint)Math.Min((totalBlocks + 7) / 8, MaxDequantGridSize);
        if (gridDim == 0) gridDim = 1;

        nint func = dtype switch
        {
            KvCacheDType.Q8_0 => _kvDequantQ8_0DynFunc,
            KvCacheDType.Q4_0 => _kvDequantQ4_0DynFunc,
            _ => throw new NotSupportedException($"Dynamic KV dequant not supported for {dtype}.")
        };

        CudaDriverApi.cuLaunchKernel(func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Scatters the live FP16 window from a per-layer ring buffer into the FP16 attention
    /// scratch starting at row <c>max(0, posPtr[0] + 1 - windowSize)</c>. One CUDA block
    /// per ring slot — each block predicates on whether its slot is currently populated
    /// at the device-side decode position.
    /// </summary>
    public void LaunchKvWindowToScratchDyn(nint ringBase, nint scratchBase, int kvStride,
                                             int windowSize, nint posPtr, nint stream)
    {
        if (_kvWriteModule == null)
            throw new InvalidOperationException(
                "KV write kernel not available. Compile native/kernels/kv_write.cu to PTX.");

        nint rArg = ringBase, sArg = scratchBase, posArg = posPtr;
        int kvArg = kvStride, wsArg = windowSize;
        void** args = stackalloc void*[] { &rArg, &sArg, &kvArg, &wsArg, &posArg };

        // One CUDA block per ring slot; threads-per-block sized to cover kvStride
        // with a per-thread stride loop.
        uint gridDim = (uint)windowSize;
        uint threadsPerBlock = (uint)Math.Min(BlockSize, kvStride);
        if (threadsPerBlock == 0) threadsPerBlock = 32;

        CudaDriverApi.cuLaunchKernel(_kvWindowToScratchDynFunc,
                gridDim, 1, 1, threadsPerBlock, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>True when the fused decode-step RoPE+KV-write kernel is loaded (PTX present).</summary>
    public bool HasFusedRopeKvWriteKernel => _fusedRopeKvWriteModule != null;

    /// <summary>
    /// Fused decode-step (seqLen=1) RoPE + KV-cache write.
    /// Replaces three eager launches per layer (rope_f16 + 2× cuMemcpyDtoDAsync) with one.
    /// Q is rotated in place on <paramref name="qSrc"/>; K is rotated and the rotated row
    /// is written to <paramref name="kCacheBase"/><c> + cachePos * kvStride</c>; V is plain-copied
    /// to <paramref name="vCacheBase"/><c> + cachePos * kvStride</c>.
    /// </summary>
    public void LaunchFusedRopeKvWriteF16(
        nint qSrc, nint kSrc, nint vSrc,
        nint kCacheBase, nint vCacheBase,
        nint positionsDevice, int cachePos,
        int numHeads, int numKvHeads, int headDim,
        int ropeDim, int kvStride, float theta, int ropeType,
        nint stream)
    {
        if (_fusedRopeKvWriteModule == null)
            throw new InvalidOperationException(
                "Fused RoPE+KV-write kernel not available. Compile native/kernels/fused_rope_kv_write.cu to PTX.");

        nint qArg = qSrc, kArg = kSrc, vArg = vSrc;
        nint kCacheArg = kCacheBase, vCacheArg = vCacheBase;
        nint posArg = positionsDevice;
        int cachePosArg = cachePos;
        int nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim;
        int rdArg = ropeDim, kvStrideArg = kvStride;
        float thetaArg = theta;
        int rtArg = ropeType;

        void** args = stackalloc void*[] {
            &qArg, &kArg, &vArg, &kCacheArg, &vCacheArg,
            &posArg, &cachePosArg,
            &nhArg, &nkvArg, &hdArg,
            &rdArg, &kvStrideArg,
            &thetaArg, &rtArg
        };

        int halfRope = ropeDim / 2;
        int tail = headDim - ropeDim;
        int totalThreads = numHeads * halfRope                     // Q rotation pairs
                         + numKvHeads * halfRope                   // K rotation pairs
                         + numKvHeads * tail                       // K tail copy
                         + numKvHeads * headDim;                   // V copy
        uint gridDim = (uint)((totalThreads + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_fusedRopeKvWriteF16Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Graph-friendly variant of <see cref="LaunchFusedRopeKvWriteF16"/>: <c>cachePos</c>
    /// is read from a device pointer (<paramref name="cachePosPtr"/>) so its value can
    /// change between <c>cuGraphLaunch</c> replays without re-instantiating the graph.
    /// </summary>
    public void LaunchFusedRopeKvWriteF16Dyn(
        nint qSrc, nint kSrc, nint vSrc,
        nint kCacheBase, nint vCacheBase,
        nint positionsDevice, nint cachePosPtr,
        int numHeads, int numKvHeads, int headDim,
        int ropeDim, int kvStride, float theta, int ropeType,
        nint stream)
    {
        if (_fusedRopeKvWriteModule == null)
            throw new InvalidOperationException(
                "Fused RoPE+KV-write kernel not available. Compile native/kernels/fused_rope_kv_write.cu to PTX.");

        nint qArg = qSrc, kArg = kSrc, vArg = vSrc;
        nint kCacheArg = kCacheBase, vCacheArg = vCacheBase;
        nint posArg = positionsDevice;
        nint cachePosPtrArg = cachePosPtr;
        int nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim;
        int rdArg = ropeDim, kvStrideArg = kvStride;
        float thetaArg = theta;
        int rtArg = ropeType;

        void** args = stackalloc void*[] {
            &qArg, &kArg, &vArg, &kCacheArg, &vCacheArg,
            &posArg, &cachePosPtrArg,
            &nhArg, &nkvArg, &hdArg,
            &rdArg, &kvStrideArg,
            &thetaArg, &rtArg
        };

        int halfRope = ropeDim / 2;
        int tail = headDim - ropeDim;
        int totalThreads = numHeads * halfRope
                         + numKvHeads * halfRope
                         + numKvHeads * tail
                         + numKvHeads * headDim;
        uint gridDim = (uint)((totalThreads + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_fusedRopeKvWriteF16DynFunc,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Single-token KV-cache update that reads the target position from device memory.</summary>
    public void LaunchKvCacheUpdatePos(nint key, nint value, nint cacheKey, nint cacheValue,
                                       nint positions, int kvStride, nint stream)
    {
        nint keyArg = key, valueArg = value, cacheKeyArg = cacheKey, cacheValueArg = cacheValue;
        nint posArg = positions;
        int strideArg = kvStride;
        int blocks = Math.Min(32, (kvStride + BlockSize - 1) / BlockSize);

        void** args = stackalloc void*[]
        {
            &keyArg, &valueArg, &cacheKeyArg, &cacheValueArg, &posArg, &strideArg
        };
        CudaDriverApi.cuLaunchKernel(_kvCacheUpdatePosFunc,
                (uint)blocks, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Bias add: output[t, :] += bias[:]. half2 vectorized (2 elements/thread).</summary>
    public void LaunchBiasAdd(nint output, nint bias, int dim, int seqLen, nint stream)
    {
        nint outArg = output, biasArg = bias;
        int dimArg = dim, slArg = seqLen;

        void** args = stackalloc void*[] {&outArg, &biasArg, &dimArg, &slArg};
        int total = dim * seqLen;
        // half2: each thread processes 2 elements
        uint gridDim = (uint)((total / 2 + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_biasAddFunc,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Per-head RMS norm (QK-norm, Qwen3-style).</summary>
    public void LaunchPerHeadRmsNorm(nint qk, nint weight, float eps,
                                       int numHeads, int headDim, int seqLen, nint stream)
    {
        nint qkArg = qk, wArg = weight;
        float epsArg = eps;
        int nhArg = numHeads, hdArg = headDim, slArg = seqLen;

        void** args = stackalloc void*[] {&qkArg, &wArg, &epsArg, &nhArg, &hdArg, &slArg};
        int numBlocks = seqLen * numHeads;

        CudaDriverApi.cuLaunchKernel(_perHeadRmsNormFunc,
                (uint)numBlocks, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Convert FP16 → FP32. half2/float2 vectorized (2 elements/thread).</summary>
    public void LaunchConvertF16ToF32(nint src, nint dst, int n, nint stream)
    {
        nint srcArg = src, dstArg = dst;
        int nArg = n;

        void** args = stackalloc void*[] {&srcArg, &dstArg, &nArg};
        // half2/float2: each thread processes 2 elements
        uint gridDim = (uint)((n / 2 + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_convertF16ToF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Convert FP32 → FP16. float2/half2 vectorized (2 elements/thread).</summary>
    public void LaunchConvertF32ToF16(nint src, nint dst, int n, nint stream)
    {
        nint srcArg = src, dstArg = dst;
        int nArg = n;

        void** args = stackalloc void*[] {&srcArg, &dstArg, &nArg};
        // float2/half2: each thread processes 2 elements
        uint gridDim = (uint)((n / 2 + BlockSize - 1) / BlockSize);

        CudaDriverApi.cuLaunchKernel(_convertF32ToF16Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Quantized GEMV: y[n] = W_quant[n,k] @ x[k]. Operates directly on quantized weights.</summary>
    public void LaunchQuantizedGemv(nint quantWeight, QuantizationType qt,
                                      nint x, nint y, int n, int k, nint stream)
    {
        nint wArg = quantWeight, xArg = x, yArg = y;
        int nArg = n, kArg = k;

        nint func = qt switch
        {
            QuantizationType.Q8_0 => _quantizedGemvQ8_0Func,
            QuantizationType.Q2_K => _quantizedGemvQ2_KFunc,
            QuantizationType.Q4_K => _quantizedGemvQ4_KFunc,
            QuantizationType.Q5_0 => _quantizedGemvQ5_0Func,
            QuantizationType.Q5_K => _quantizedGemvQ5_KFunc,
            QuantizationType.Q6_K => _quantizedGemvQ6_KFunc,
            QuantizationType.IQ4_NL => _quantizedGemvIQ4_NLFunc,
            QuantizationType.IQ4_XS => _quantizedGemvIQ4_XSFunc,
            QuantizationType.IQ2_XXS => _quantizedGemvIQ2_XXSFunc,
            QuantizationType.IQ2_XS => _quantizedGemvIQ2_XSFunc,
            QuantizationType.IQ2_S => _quantizedGemvIQ2_SFunc,
            _ => 0
        };

        if (func == 0)
            throw new NotSupportedException($"Quantized GEMV not supported for {qt}.");

        void** args = stackalloc void*[] {&wArg, &xArg, &yArg, &nArg, &kArg};

        CudaDriverApi.cuLaunchKernel(func,
                (uint)n, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Whether a quantization type has a custom quantized GEMV kernel.</summary>
    public static bool HasQuantizedGemv(QuantizationType qt) =>
        qt is QuantizationType.Q8_0 or QuantizationType.Q4_K or QuantizationType.Q5_0
            or QuantizationType.Q5_K or QuantizationType.Q6_K
            or QuantizationType.Q2_K
            or QuantizationType.IQ4_NL or QuantizationType.IQ4_XS
            or QuantizationType.IQ2_XXS or QuantizationType.IQ2_XS or QuantizationType.IQ2_S;

    /// <summary>
    /// True when the legacy per-row quantized GEMV kernel function is loaded for
    /// the given type. Unlike <see cref="HasQuantizedGemv"/>, this is runtime PTX
    /// capability, not static type-support metadata.
    /// </summary>
    public bool HasQuantizedGemvKernel(QuantizationType qt) => qt switch
    {
        QuantizationType.Q8_0 => !DisableQuantizedGemv && _quantizedGemvQ8_0Func != 0,
        QuantizationType.Q2_K => !DisableQuantizedGemv && _quantizedGemvQ2_KFunc != 0,
        QuantizationType.Q4_K => !DisableQuantizedGemv && _quantizedGemvQ4_KFunc != 0,
        QuantizationType.Q5_0 => !DisableQuantizedGemv && _quantizedGemvQ5_0Func != 0,
        QuantizationType.Q5_K => !DisableQuantizedGemv && _quantizedGemvQ5_KFunc != 0,
        QuantizationType.Q6_K => !DisableQuantizedGemv && _quantizedGemvQ6_KFunc != 0,
        QuantizationType.IQ4_NL => !DisableQuantizedGemv && _quantizedGemvIQ4_NLFunc != 0,
        QuantizationType.IQ4_XS => !DisableQuantizedGemv && _quantizedGemvIQ4_XSFunc != 0,
        QuantizationType.IQ2_XXS => !DisableQuantizedGemv && _quantizedGemvIQ2_XXSFunc != 0,
        QuantizationType.IQ2_XS => !DisableQuantizedGemv && _quantizedGemvIQ2_XSFunc != 0,
        QuantizationType.IQ2_S => !DisableQuantizedGemv && _quantizedGemvIQ2_SFunc != 0,
        _ => false,
    };

    /// <summary>
    /// True when any loaded decode-time quantized GEMV implementation can execute
    /// this type (MMQ/MMVQ-large or the legacy per-row kernel).
    /// </summary>
    public bool HasLoadedQuantizedGemv(QuantizationType qt) =>
        HasMmq(qt) || HasQuantizedGemvKernel(qt);

    /// <summary>
    /// Minimum K alignment required by the per-call <see cref="LaunchQuantizedGemv"/>
    /// kernel for the given quant type. Block-32 quants (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0)
    /// require <c>K % 32 == 0</c>; K-quants (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K) require
    /// <c>K % 256 == 0</c>. Caller-side gates use this to decide between the
    /// direct-GEMV fast path and the dequant-then-GEMM fallback.
    /// </summary>
    /// <remarks>
    /// V2-Lite's <c>ffn_down_exps</c> is stored at K=intermediate=1408 with quant
    /// type Q8_0 (Q4_K_M mix) or Q5_0 (Q3_K_M mix). 1408 is a multiple of 32 but
    /// not 256; the previous unconditional <c>K % 256</c> gate locked these
    /// projections out of the GEMV fast path. The block-32 kernels handle K=1408
    /// natively (<c>blocks_per_row = K/32 = 44</c>).
    /// </remarks>
    public static int MinKAlignmentFor(QuantizationType qt) => qt switch
    {
        QuantizationType.Q4_0 or QuantizationType.Q4_1
            or QuantizationType.Q5_0 or QuantizationType.Q5_1
            or QuantizationType.Q8_0 or QuantizationType.IQ4_NL => 32,
        QuantizationType.Q3_K or QuantizationType.Q4_K
            or QuantizationType.Q5_K or QuantizationType.Q6_K
            or QuantizationType.Q2_K or QuantizationType.IQ4_XS
            or QuantizationType.IQ2_XXS or QuantizationType.IQ2_XS
            or QuantizationType.IQ2_S => 256,
        _ => int.MaxValue,  // unsupported types — gate always fails
    };

    /// <summary>
    /// Phase-B MoE grouped quantized GEMV. Computes K_active independent
    /// <c>y_e[m] = W_e[m,k] @ x[k]</c> projections in a single launch where
    /// <c>x</c> (FP16, length K) is shared across all experts and each
    /// <c>W_e</c> / <c>y_e</c> pair is selected from the K_active per-expert
    /// pointer arrays. Supports Q2_K, Q4_K, Q5_K, Q6_K, and Q8_0 — gate the
    /// call with <see cref="HasMoeGroupedGemv"/>.
    /// </summary>
    /// <param name="weightPtrsDevice">Device pointer to <c>K_active</c> contiguous
    /// <c>nint</c>-sized weight pointers (one per active expert, raw quant bytes).</param>
    /// <param name="outputPtrsDevice">Device pointer to <c>K_active</c> contiguous
    /// <c>nint</c>-sized output pointers (one per active expert, FP16 [M] each).</param>
    /// <param name="x">Device pointer to the shared FP16 input row, length K.</param>
    /// <param name="qt">Quantization type. Must be one of Q2_K, Q4_K, Q5_K, Q6_K, Q8_0.</param>
    /// <param name="M">Per-expert output rows.</param>
    /// <param name="K">Input dim. Must satisfy <c>K % 256 == 0</c> for K-quants and
    /// <c>K % 32 == 0</c> for Q8_0. The shared dispatch keeps the 256 alignment so
    /// callers can use the same gate regardless of quant type.</param>
    /// <param name="kActive">Number of active experts.</param>
    /// <param name="stream">CUDA stream.</param>
    public void LaunchMoeGroupedGemv(nint weightPtrsDevice, nint outputPtrsDevice,
                                       nint x, QuantizationType qt,
                                       int M, int K, int kActive, nint stream)
    {
        if (kActive <= 0 || M <= 0 || K <= 0) return;
        if ((K & 255) != 0)
            throw new ArgumentException($"K must be a multiple of 256 (got {K}).", nameof(K));

        nint func = qt switch
        {
            QuantizationType.Q2_K => _moeGroupedGemvQ2_KFunc,
            QuantizationType.Q4_K => _moeGroupedGemvQ4_KFunc,
            QuantizationType.Q5_K => _moeGroupedGemvQ5_KFunc,
            QuantizationType.Q6_K => _moeGroupedGemvQ6_KFunc,
            QuantizationType.Q8_0 => _moeGroupedGemvQ8_0Func,
            QuantizationType.IQ4_NL => _moeGroupedGemvIQ4_NLFunc,
            QuantizationType.IQ4_XS => _moeGroupedGemvIQ4_XSFunc,
            _ => 0,
        };
        if (func == 0)
            throw new NotSupportedException(
                $"MoE grouped GEMV not available for {qt}. Compile native/kernels/moe_grouped_gemv.cu to PTX.");

        nint xArg = x, wArg = weightPtrsDevice, yArg = outputPtrsDevice;
        int mArg = M, kArg = K, kActiveArg = kActive;
        void** args = stackalloc void*[] { &xArg, &wArg, &yArg, &mArg, &kArg, &kActiveArg };

        // Grid: (M output rows, K_active experts, 1). Block: 256 threads.
        CudaDriverApi.cuLaunchKernel(func,
                (uint)M, (uint)kActive, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>True when the MMQ-style Q2_K GEMV kernel is loaded (PTX present).</summary>
    public bool HasMmqQ2K => _quantizedGemvQ2_KMmqFunc != 0 && !DisableQuantizedGemv && !DisableMmqQ2K;

    /// <summary>True when the MMQ-style Q4_K GEMV kernel is loaded (PTX present).</summary>
    public bool HasMmqQ4K => _quantizedGemvMmqModule != null && !DisableQuantizedGemv && !DisableMmqQ4K;

    /// <summary>True when the MMQ-style Q5_K GEMV kernel is loaded (PTX present).</summary>
    public bool HasMmqQ5K => _quantizedGemvQ5_KMmqFunc != 0 && !DisableQuantizedGemv && !DisableMmqQ5K;

    /// <summary>True when the MMQ-style Q6_K GEMV kernel is loaded (PTX present).</summary>
    public bool HasMmqQ6K => _quantizedGemvQ6_KMmqFunc != 0 && !DisableQuantizedGemv && !DisableMmqQ6K;

    /// <summary>True when the MMQ-style IQ4_NL GEMV kernel is loaded (PTX present).</summary>
    public bool HasMmqIQ4_NL => !DisableQuantizedGemv && _quantizedGemvIQ4_NLMmqFunc != 0;

    /// <summary>True when the MMQ-style IQ4_XS GEMV kernel is loaded (PTX present).</summary>
    public bool HasMmqIQ4_XS => !DisableQuantizedGemv && _quantizedGemvIQ4_XSMmqFunc != 0;

    /// <summary>True when the MMVQ-large Q2_K GEMV kernel is loaded and not disabled.</summary>
    public bool HasMmvqLargeQ2K => _quantizedGemvQ2_KMmvqLargeFunc != 0 && !DisableMmvqLargeQ2K;

    /// <summary>True when the MMVQ-large Q4_K GEMV kernel (1 row × 128 threads) is loaded
    /// AND not disabled. Default ON — pre-Q8_1 input quantization (<see cref="HasPreQ8_1"/>)
    /// removes the redundant Stage 1 cost that previously made this kernel regress
    /// (<c>docs/perf/MLPUP_GEMV_GAP.md</c> §H4). Per-quant-type override:
    /// <c>DOTLLM_DISABLE_MMVQ_LARGE_Q4K=1</c>.</summary>
    public bool HasMmvqLargeQ4K => _quantizedGemvQ4_KMmvqLargeFunc != 0 && !DisableMmvqLargeQ4K;

    /// <summary>True when the llama.cpp-faithful coalesced+accumulate-once Q4_K MMVQ kernel
    /// (<c>quantized_gemv_q4_k_mmvq_llamacpp</c>) is loaded (PTX present). Independent of
    /// <see cref="MmvqLlamaCpp"/>, which additionally gates whether the dispatcher routes to it.</summary>
    public bool HasMmvqLlamaCppQ4K => _quantizedGemvQ4_KMmvqLlamaCppFunc != 0;

    /// <summary>True when the MMVQ-large Q5_K GEMV kernel is loaded and not disabled.</summary>
    public bool HasMmvqLargeQ5K => _quantizedGemvQ5_KMmvqLargeFunc != 0 && !DisableMmvqLargeQ5K;

    /// <summary>True when the MMVQ-large Q6_K GEMV kernel is loaded and not disabled.</summary>
    public bool HasMmvqLargeQ6K => _quantizedGemvQ6_KMmvqLargeFunc != 0 && !DisableMmvqLargeQ6K;

    /// <summary>True when the MMVQ-large IQ4_NL GEMV kernel is loaded and not disabled.</summary>
    public bool HasMmvqLargeIQ4_NL => _quantizedGemvIQ4_NLMmvqLargeFunc != 0 && !DisableMmvqLargeIQ4_NL;

    /// <summary>True when the MMVQ-large IQ4_XS GEMV kernel is loaded and not disabled.</summary>
    public bool HasMmvqLargeIQ4_XS => _quantizedGemvIQ4_XSMmvqLargeFunc != 0 && !DisableMmvqLargeIQ4_XS;

    /// <summary>True when the pre-Q8_1 input-quantization kernel is loaded and not disabled.
    /// When this is on (default) and a scratch buffer is provided to the MMQ GEMV launcher,
    /// Stage 1 runs once via <see cref="LaunchQuantizeXToQ8_1"/> and the GEMV uses the
    /// <c>_preq</c> kernel variants — eliminating the per-block redundant input quant.
    /// Override: <c>DOTLLM_DISABLE_PREQ8_1=1</c>.</summary>
    public bool HasPreQ8_1 => _quantizeXToQ8_1Func != 0 && !DisablePreQ8_1;

    /// <summary>
    /// True when the batched-M dp4a MMQ Q4_K prefill kernel AND its batched input-quantization
    /// companion are both loaded (issue #349 proof of concept — Q4_K only). Gates the prefill
    /// dispatcher in <c>CudaTransformerModel.Project</c>: when true (and <see cref="MmqBatchedMinSeqLen"/>
    /// is met), prefill skips dequant→cuBLAS HGEMM entirely for Q4_K weights.
    /// <b>In practice this path is dormant by default</b>: <see cref="MmqBatchedMinSeqLen"/>'s default
    /// is <see cref="int.MaxValue"/> (Task 6 benchmark sweep measured the batched-MMQ kernel 2x-51x
    /// slower than the dequant→cuBLAS fallback at every tested prefill length), so
    /// <c>HasMmqBatchedQ4K</c> being true does NOT mean the batched-MMQ path is actually taken —
    /// see issue #367 for the wider-tile redesign needed before the threshold can be safely lowered.
    /// </summary>
    public bool HasMmqBatchedQ4K => _quantizedGemvQ4_KMmqBatchedFunc != 0 && _quantizeXToQ8_1BatchedFunc != 0
        && !DisableQuantizedGemv && !DisableMmqBatchedQ4K;

    /// <summary>Test/benchmark hook to force the legacy Q2_K GEMV kernel even when MMQ is loaded.</summary>
    public static bool DisableMmqQ2K { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMQ_Q2K") == "1";

    /// <summary>Test/benchmark hook to force the legacy Q4_K GEMV kernel even when MMQ is loaded.</summary>
    public static bool DisableMmqQ4K { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMQ_Q4K") == "1";

    /// <summary>Test/benchmark hook to force the legacy Q5_K GEMV kernel even when MMQ is loaded.</summary>
    public static bool DisableMmqQ5K { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMQ_Q5K") == "1";

    /// <summary>Test/benchmark hook to force the legacy Q6_K GEMV kernel even when MMQ is loaded.</summary>
    public static bool DisableMmqQ6K { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMQ_Q6K") == "1";

    /// <summary>Disable MMVQ-large routing for Q2_K (forces MMQ-4-rows for k ≥ <see cref="MmvqLargeKThreshold"/>).</summary>
    public static bool DisableMmvqLargeQ2K { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMVQ_LARGE_Q2K") == "1";

    /// <summary>Disable MMVQ-large routing (forces MMQ-4-rows for k ≥ <see cref="MmvqLargeKThreshold"/>).</summary>
    public static bool DisableMmvqLargeQ4K { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMVQ_LARGE_Q4K") == "1";

    /// <summary>Disable MMVQ-large routing for Q5_K.</summary>
    public static bool DisableMmvqLargeQ5K { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMVQ_LARGE_Q5K") == "1";

    /// <summary>Disable MMVQ-large routing for Q6_K.</summary>
    public static bool DisableMmvqLargeQ6K { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMVQ_LARGE_Q6K") == "1";

    /// <summary>Disable MMVQ-large routing for IQ4_NL.</summary>
    public static bool DisableMmvqLargeIQ4_NL { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMVQ_LARGE_IQ4_NL") == "1";

    /// <summary>Disable MMVQ-large routing for IQ4_XS.</summary>
    public static bool DisableMmvqLargeIQ4_XS { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMVQ_LARGE_IQ4_XS") == "1";

    /// <summary>Disable pre-Q8_1 input quantization (falls back to on-the-fly Stage 1 in every GEMV).</summary>
    /// <remarks>Useful for A/B comparison or when the model's max k makes the pre-quant scratch
    /// buffer awkward to size. Default off — pre-Q8_1 is the recommended path.</remarks>
    public static bool DisablePreQ8_1 { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_PREQ8_1") == "1";

    /// <summary>Test/benchmark hook to force the dequant→cuBLAS prefill fallback even when the
    /// batched-MMQ Q4_K kernel is loaded. Override: <c>DOTLLM_DISABLE_MMQ_BATCHED_Q4K=1</c>.</summary>
    public static bool DisableMmqBatchedQ4K { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMQ_BATCHED_Q4K") == "1";

    /// <summary>
    /// Minimum prefill seqLen (inclusive) at which the batched-MMQ Q4_K kernel is preferred over
    /// the dequant→cuBLAS HGEMM fallback — mirrors llama.cpp's MMVQ_MAX_BATCH_SIZE /
    /// MMQ_DP4A_MAX_BATCH_SIZE crossover gating (docs/perf/MMA_BATCHED_MMQ.md §1d).
    /// </summary>
    /// <remarks>
    /// <b>KNOWN LIMITATION (Task 6 benchmark sweep, RTX 3060, 2026-08-12) — set to
    /// <see cref="int.MaxValue"/>, effectively disabled for all realistic prefill lengths.</b>
    /// Measured on Qwen3-4B-Q4_K_M and the pure Q4_K
    /// <c>Llama-3.2-1B-pure-Q4_K.gguf</c> ladder fixture (no Q6_K-mixture dilution) across
    /// <c>-p</c> ∈ {4, 8, 16, 32, 64, 128, 256, 512}: the batched-MMQ path was slower than the
    /// dequant→cuBLAS fallback at <b>every</b> seqLen ≥ 8 tested, by a margin that *grows* with
    /// seqLen instead of shrinking (Qwen3-4B: ~2× slower at p=8, ~13x at p=64, ~39x at p=256,
    /// ~51× at p=512; the dilution-free Llama-3.2-1B pure-Q4_K fixture showed ~22× at p=128,
    /// i.e. the regression is not a Q4_K_M-mixture artifact). Reproduced with interleaved
    /// baseline/new-path pairs (ruling out ordering/thermal noise) and confirmed the dispatcher
    /// gate genuinely fires (prefill tok/s differs sharply at p≥8 vs the p=4 control pair, where
    /// both paths are identical because the gate is off in both). Root cause (not fixed here —
    /// out of this task's file scope): <see cref="LaunchQuantizedGemvMmqBatchedQ4K"/> launches
    /// <c>gridY = ceil(m / MmqBatchMTile)</c> with <c>MmqBatchMTile = 2</c>, so each pair of
    /// prefill rows re-reads the entire Q4_K weight matrix from global memory independently —
    /// weight-read volume (and thus wall time) scales ~linearly with seqLen instead of being
    /// amortized the way cuBLAS's HGEMM tiling amortizes it, which is why per-token prefill cost
    /// stays roughly flat at decode-GEMV levels (~19–23 ms/token on the 3060) instead of dropping
    /// as seqLen grows. A real fix needs a larger per-block M-tile (with correspondingly larger
    /// register/shared-memory reuse across the batch) before this default can be safely lowered;
    /// track as a follow-up issue: #367. Until then, prefer <c>DOTLLM_DISABLE_MMQ_BATCHED_Q4K=1</c>'s
    /// effective state (dequant→cuBLAS) for all prefill. Override with
    /// <c>DOTLLM_MMQ_BATCHED_MIN_SEQLEN</c> for A/B comparison once a fix lands.
    /// <para>
    /// <b>Lowering this at runtime is not fully retroactive.</b> <see cref="CudaForwardState.EnsureCapacity"/>
    /// only allocates <c>PreQ8_1BatchedScratch</c> when the capacity it is growing to already
    /// clears whatever this threshold was <i>at that call</i>; lowering the property afterward does
    /// not go back and allocate the buffer for already-grown state (<see cref="CudaTransformerModel.Project"/>
    /// guards against dispatching into the kernel with a null scratch pointer in that case and falls
    /// back to dequant→cuBLAS, but that means the lowered threshold silently has no effect until the
    /// next capacity growth or a freshly constructed forward state). Setting this via the env var
    /// override before process start is therefore the most reliable way to get the lowered threshold
    /// to actually take effect.
    /// </para>
    /// </remarks>
    public static int MmqBatchedMinSeqLen { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_MMQ_BATCHED_MIN_SEQLEN"), out int v) ? v : int.MaxValue;

    /// <summary>True when this MMQ GEMV variant is available for the given quantization type.</summary>
    public bool HasMmq(QuantizationType qt) => qt switch
    {
        QuantizationType.Q2_K => HasMmqQ2K,
        QuantizationType.Q4_K => HasMmqQ4K,
        QuantizationType.Q5_K => HasMmqQ5K,
        QuantizationType.Q6_K => HasMmqQ6K,
        QuantizationType.IQ4_NL => HasMmqIQ4_NL,
        QuantizationType.IQ4_XS => HasMmqIQ4_XS,
        _ => false,
    };

    /// <summary>True when the MMVQ-large variant is available for the given quantization type.</summary>
    public bool HasMmvqLarge(QuantizationType qt) => qt switch
    {
        QuantizationType.Q2_K => HasMmvqLargeQ2K,
        QuantizationType.Q4_K => HasMmvqLargeQ4K,
        QuantizationType.Q5_K => HasMmvqLargeQ5K,
        QuantizationType.Q6_K => HasMmvqLargeQ6K,
        QuantizationType.IQ4_NL => HasMmvqLargeIQ4_NL,
        QuantizationType.IQ4_XS => HasMmvqLargeIQ4_XS,
        _ => false,
    };

    /// <summary>True when the 4-rows pre-Q8_1 MMQ variant is loaded for the given quantization type.</summary>
    public bool HasMmqPreq(QuantizationType qt) => qt switch
    {
        QuantizationType.Q2_K => _quantizedGemvQ2_KMmqPreqFunc != 0,
        QuantizationType.Q4_K => _quantizedGemvQ4_KMmqPreqFunc != 0,
        QuantizationType.Q5_K => _quantizedGemvQ5_KMmqPreqFunc != 0,
        QuantizationType.Q6_K => _quantizedGemvQ6_KMmqPreqFunc != 0,
        QuantizationType.IQ4_NL => _quantizedGemvIQ4_NLMmqPreqFunc != 0,
        QuantizationType.IQ4_XS => _quantizedGemvIQ4_XSMmqPreqFunc != 0,
        _ => false,
    };

    /// <summary>True when the MMVQ-large pre-Q8_1 variant is loaded for the given quantization type.</summary>
    public bool HasMmvqLargePreq(QuantizationType qt) => qt switch
    {
        QuantizationType.Q2_K => _quantizedGemvQ2_KMmvqLargePreqFunc != 0,
        QuantizationType.Q4_K => _quantizedGemvQ4_KMmvqLargePreqFunc != 0,
        QuantizationType.Q5_K => _quantizedGemvQ5_KMmvqLargePreqFunc != 0,
        QuantizationType.Q6_K => _quantizedGemvQ6_KMmvqLargePreqFunc != 0,
        QuantizationType.IQ4_NL => _quantizedGemvIQ4_NLMmvqLargePreqFunc != 0,
        QuantizationType.IQ4_XS => _quantizedGemvIQ4_XSMmvqLargePreqFunc != 0,
        _ => false,
    };

    /// <summary>k threshold (inclusive) below which the MMQ-4-rows kernel is preferred over MMVQ-large.</summary>
    /// <remarks>
    /// At k&lt;1024 (≤3 super-blocks per row) the input-quantization amortization across 4 rows
    /// outweighs the per-row warp parallelism of the MMVQ-large kernel. At k≥1024 the per-row
    /// work saturates 128 threads and the row-coherent register accumulator wins.
    /// SmolLM-135M (k=576) stays on MMQ-4-rows; Qwen3-8B (k=4096) gets MMVQ-large.
    /// </remarks>
    public const int MmvqLargeKThreshold = 1024;

    /// <summary>
    /// Pre-Q8_1 input quantization. Quantizes <paramref name="x"/>[k] to INT8 with one FP16 scale
    /// per 32-element chunk and per-half-chunk FP16 sums. Output layout (single contiguous buffer):
    /// <code>int8_t xq[k] | half dx[k/32] | half sx2[k/16]</code>
    /// Use <see cref="CudaForwardState.PreQ8_1ScratchBytes"/> to size the scratch.
    /// Consumed by the <c>_preq</c> MMQ kernel variants (see the MMQ GEMV launcher overload
    /// taking a <c>preqScratch</c> pointer).
    /// </summary>
    public void LaunchQuantizeXToQ8_1(nint x, nint scratch, int k, nint stream)
    {
        if (_quantizeXToQ8_1Func == 0)
            throw new InvalidOperationException(
                "Pre-Q8_1 quantization kernel not available. Compile native/kernels/quantize_x.cu to PTX.");
        if ((k & 31) != 0)
            throw new ArgumentException($"k must be a multiple of 32 (got {k}).", nameof(k));

        int numChunks = k >> 5;
        // xq starts at offset 0; dx at offset k; sx2 at offset k + 2*numChunks.
        nint xqPtr  = scratch;
        nint dxPtr  = scratch + k;
        nint sx2Ptr = scratch + k + (nint)(numChunks * 2);

        nint xArg = x, xqArg = xqPtr, dxArg = dxPtr, sx2Arg = sx2Ptr;
        int kArg = k;
        void** args = stackalloc void*[] { &xArg, &xqArg, &dxArg, &sx2Arg, &kArg };

        // Must mirror QX_THREADS_X / QX_WARPS_PER_BLOCK in quantize_x.cu (32 × 8 = 256).
        const uint QxThreadsX = 32;
        const uint QxWarpsPerBlock = 8;
        uint gridDim = (uint)((numChunks + (int)QxWarpsPerBlock - 1) / (int)QxWarpsPerBlock);
        CudaDriverApi.cuLaunchKernel(_quantizeXToQ8_1Func,
                gridDim, 1, 1, QxThreadsX, QxWarpsPerBlock, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// MMQ-style fused dequant+matmul GEMV. Quantizes the input activation to
    /// INT8 (per-32-element scale) and accumulates the dot product via __dp4a
    /// (packed 4×INT8 multiply-add) instead of FP fmuladd. Lossy on the input
    /// quantization but matches CPU output within K-quant tolerance.
    /// Routes between two kernel variants based on k:
    /// <list type="bullet">
    /// <item>k ≥ <see cref="MmvqLargeKThreshold"/> (1024): MMVQ-large kernel — 1 row per CUDA block,
    /// 128 threads (4 warps), no cross-row shmem accumulator. Modeled on llama.cpp's
    /// <c>mul_mat_vec_q&lt;Q4_K, 1&gt;</c>. Optimal for Qwen3-8B-class shapes (k=4096).</item>
    /// <item>k &lt; 1024: MMQ-4-rows kernel — 4 rows per block, 256 threads, cross-row reduction.
    /// Optimal for SmolLM-135M-class shapes (k=576) where rows are small (≤3 super-blocks).</item>
    /// </list>
    /// Supports Q2_K, Q4_K, Q5_K, Q6_K, IQ4_NL, and IQ4_XS — gate the call with
    /// <see cref="HasMmq"/>. These quant types have MMQ + MMVQ-large coverage in both
    /// on-the-fly and pre-Q8_1 modes when the corresponding PTX symbols are present.
    /// On-the-fly Stage 1 input quantization. Use the overload taking a <c>preqScratch</c>
    /// pointer for the pre-Q8_1 path (eliminates per-block redundant Stage 1).
    /// </summary>
    public void LaunchQuantizedGemvMmq(nint quantWeight, QuantizationType qt,
                                         nint x, nint y, int n, int k, nint stream)
        => LaunchQuantizedGemvMmq(quantWeight, qt, x, y, n, k, preqScratch: 0, stream);

    /// <summary>
    /// MMQ-style fused dequant+matmul GEMV with optional pre-Q8_1 scratch. When
    /// <paramref name="preqScratch"/> is non-zero, <see cref="HasPreQ8_1"/> is true, and the
    /// <c>_preq</c> variant is loaded for the chosen quant type, the GEMV reads INT8/dx/sx2
    /// from the device-resident scratch (populated upstream by <see cref="LaunchQuantizeXToQ8_1"/>)
    /// and skips Stage 1 entirely. Otherwise falls back to the on-the-fly Stage 1 kernel.
    /// </summary>
    public void LaunchQuantizedGemvMmq(nint quantWeight, QuantizationType qt,
                                         nint x, nint y, int n, int k, nint preqScratch, nint stream)
    {
        if (_quantizedGemvMmqModule == null)
            throw new InvalidOperationException(
                "MMQ GEMV kernel not available. Compile native/kernels/quantized_gemv_mmq.cu to PTX.");

        bool usePreq = preqScratch != 0 && HasPreQ8_1;

        // llama.cpp-faithful coalesced+accumulate-once Q4_K MMVQ (opt-in, DOTLLM_CUDA_MMVQ_LLAMACPP=1):
        // same warp-per-superblock coalesced 128-byte qs load as MmvqCoalesced below, but defers ALL
        // cross-lane reduction to a single end-of-row reduce (MMVQ-large's accumulate-once property).
        // Checked ahead of MmvqCoalesced so it takes priority if both toggles are set. On-the-fly only.
        if (MmvqLlamaCpp && qt == QuantizationType.Q4_K && k >= MmvqLargeKThreshold
            && _quantizedGemvQ4_KMmvqLlamaCppFunc != 0)
        {
            nint wLc = quantWeight, xLc = x, yLc = y;
            int nLc = n, kLc = k;
            void** argsLc = stackalloc void*[] { &wLc, &xLc, &yLc, &nLc, &kLc };
            uint dynShmemLc = (uint)ComputeMmqDynamicSharedBytes(qt, k);
            CheckDynamicSharedBudget(dynShmemLc, qt, k);
            CudaDriverApi.cuLaunchKernel(_quantizedGemvQ4_KMmvqLlamaCppFunc,
                    (uint)n, 1, 1, MmvqLargeThreads, 1, 1,
                    dynShmemLc, stream, (nint)argsLc, 0).ThrowOnError();
            return;
        }

        // Experimental coalesced Q4_K MMVQ (opt-in, DOTLLM_CUDA_MMVQ_COALESCED=1): warp-per-superblock
        // with coalesced 128-byte qs loads. On-the-fly only (does its own Stage-1 from x, ignores preq).
        // Same 1-row-per-block launch as MMVQ-large; gated to Q4_K + k ≥ threshold.
        if (MmvqCoalesced && qt == QuantizationType.Q4_K && k >= MmvqLargeKThreshold
            && _quantizedGemvQ4_KMmvqCoalescedFunc != 0)
        {
            nint wC = quantWeight, xC = x, yC = y;
            int nC = n, kC = k;
            void** argsC = stackalloc void*[] { &wC, &xC, &yC, &nC, &kC };
            uint dynShmemC = (uint)ComputeMmqDynamicSharedBytes(qt, k);
            CheckDynamicSharedBudget(dynShmemC, qt, k);
            CudaDriverApi.cuLaunchKernel(_quantizedGemvQ4_KMmvqCoalescedFunc,
                    (uint)n, 1, 1, MmvqLargeThreads, 1, 1,
                    dynShmemC, stream, (nint)argsC, 0).ThrowOnError();
            return;
        }

        // Prefer MMVQ-large for k ≥ threshold when the variant is loaded and not disabled.
        // The DOTLLM_DISABLE_MMVQ_LARGE_<QT> env vars (separate from DOTLLM_DISABLE_MMQ_<QT>)
        // force the MMQ-4-rows path for A/B comparison without bypassing dp4a entirely.
        if (k >= MmvqLargeKThreshold && HasMmvqLarge(qt))
        {
            nint largePreqFunc = usePreq
                ? qt switch
                {
                    QuantizationType.Q2_K => _quantizedGemvQ2_KMmvqLargePreqFunc,
                    QuantizationType.Q4_K => _quantizedGemvQ4_KMmvqLargePreqFunc,
                    QuantizationType.Q5_K => _quantizedGemvQ5_KMmvqLargePreqFunc,
                    QuantizationType.Q6_K => _quantizedGemvQ6_KMmvqLargePreqFunc,
                    QuantizationType.IQ4_NL => _quantizedGemvIQ4_NLMmvqLargePreqFunc,
                    QuantizationType.IQ4_XS => _quantizedGemvIQ4_XSMmvqLargePreqFunc,
                    _ => 0,
                }
                : 0;
            nint largeOnTheFlyFunc = qt switch
            {
                QuantizationType.Q2_K => _quantizedGemvQ2_KMmvqLargeFunc,
                QuantizationType.Q4_K => _quantizedGemvQ4_KMmvqLargeFunc,
                QuantizationType.Q5_K => _quantizedGemvQ5_KMmvqLargeFunc,
                QuantizationType.Q6_K => _quantizedGemvQ6_KMmvqLargeFunc,
                QuantizationType.IQ4_NL => _quantizedGemvIQ4_NLMmvqLargeFunc,
                QuantizationType.IQ4_XS => _quantizedGemvIQ4_XSMmvqLargeFunc,
                _ => 0,
            };
            bool useLargePreq = largePreqFunc != 0;
            nint largeFunc = useLargePreq ? largePreqFunc : largeOnTheFlyFunc;
            if (largeFunc != 0)
            {
                // Block size mirrors MMVQ_LARGE_THREADS in quantized_gemv_mmq.cu (tunable via the
                // MmvqLargeThreads property / DOTLLM_CUDA_MMVQ_THREADS for occupancy sweeps).
                if (useLargePreq)
                {
                    int numChunks = k >> 5;
                    nint xqPtr  = preqScratch;
                    nint dxPtr  = preqScratch + k;
                    nint sx2Ptr = preqScratch + k + (nint)(numChunks * 2);
                    nint wL = quantWeight, xqL = xqPtr, dxL = dxPtr, sx2L = sx2Ptr, yL = y;
                    int nL = n, kL = k;
                    void** argsL = stackalloc void*[] { &wL, &xqL, &dxL, &sx2L, &yL, &nL, &kL };
                    // _preq variants don't use dynamic shmem — only static s_warp_partials (16 B).
                    CudaDriverApi.cuLaunchKernel(largeFunc,
                            (uint)n, 1, 1, MmvqLargeThreads, 1, 1,
                            0, stream, (nint)argsL, 0).ThrowOnError();
                }
                else
                {
                    nint wL = quantWeight, xL = x, yL = y;
                    int nL = n, kL = k;
                    void** argsL = stackalloc void*[] { &wL, &xL, &yL, &nL, &kL };
                    uint dynShmem = (uint)ComputeMmqDynamicSharedBytes(qt, k);
                    CheckDynamicSharedBudget(dynShmem, qt, k);
                    CudaDriverApi.cuLaunchKernel(largeFunc,
                            (uint)n, 1, 1, MmvqLargeThreads, 1, 1,
                            dynShmem, stream, (nint)argsL, 0).ThrowOnError();
                }
                return;
            }
        }

        nint func = usePreq
            ? qt switch
            {
                QuantizationType.Q2_K => _quantizedGemvQ2_KMmqPreqFunc,
                QuantizationType.Q4_K => _quantizedGemvQ4_KMmqPreqFunc,
                QuantizationType.Q5_K => _quantizedGemvQ5_KMmqPreqFunc,
                QuantizationType.Q6_K => _quantizedGemvQ6_KMmqPreqFunc,
                QuantizationType.IQ4_NL => _quantizedGemvIQ4_NLMmqPreqFunc,
                QuantizationType.IQ4_XS => _quantizedGemvIQ4_XSMmqPreqFunc,
                _ => 0,
            }
            : qt switch
            {
                QuantizationType.Q2_K => _quantizedGemvQ2_KMmqFunc,
                QuantizationType.Q4_K => _quantizedGemvQ4_KMmqFunc,
                QuantizationType.Q5_K => _quantizedGemvQ5_KMmqFunc,
                QuantizationType.Q6_K => _quantizedGemvQ6_KMmqFunc,
                QuantizationType.IQ4_NL => _quantizedGemvIQ4_NLMmqFunc,
                QuantizationType.IQ4_XS => _quantizedGemvIQ4_XSMmqFunc,
                _ => 0,
            };

        if (func == 0)
        {
            // Fallback: requested preq variant missing (stale PTX) — try the on-the-fly path.
            if (usePreq)
            {
                LaunchQuantizedGemvMmq(quantWeight, qt, x, y, n, k, 0, stream);
                return;
            }
            throw new NotSupportedException($"MMQ GEMV not available for {qt}.");
        }

        // Must mirror MMQ_ROWS_PER_BLOCK in quantized_gemv_mmq.cu.
        const int MmqRowsPerBlock = 4;
        uint gridDim = (uint)((n + MmqRowsPerBlock - 1) / MmqRowsPerBlock);

        if (usePreq)
        {
            int numChunks = k >> 5;
            nint xqPtr  = preqScratch;
            nint dxPtr  = preqScratch + k;
            nint sx2Ptr = preqScratch + k + (nint)(numChunks * 2);
            nint wArg = quantWeight, xqArg = xqPtr, dxArg = dxPtr, sx2Arg = sx2Ptr, yArg = y;
            int nArg = n, kArg = k;
            void** args = stackalloc void*[] { &wArg, &xqArg, &dxArg, &sx2Arg, &yArg, &nArg, &kArg };
            // _preq variants don't use dynamic shmem — only static s_acc (4 KB).
            CudaDriverApi.cuLaunchKernel(func,
                    gridDim, 1, 1, BlockSize, 1, 1,
                    0, stream, (nint)args, 0).ThrowOnError();
        }
        else
        {
            nint wArg = quantWeight, xArg = x, yArg = y;
            int nArg = n, kArg = k;
            void** args = stackalloc void*[] { &wArg, &xArg, &yArg, &nArg, &kArg };
            uint dynShmem = (uint)ComputeMmqDynamicSharedBytes(qt, k);
            CheckDynamicSharedBudget(dynShmem, qt, k);
            CudaDriverApi.cuLaunchKernel(func,
                    gridDim, 1, 1, BlockSize, 1, 1,
                    dynShmem, stream, (nint)args, 0).ThrowOnError();
        }
    }

    /// <summary>
    /// Batched pre-Q8_1 input quantization for prefill (issue #349). Quantizes
    /// <paramref name="x"/>[m, k] (row-major, row stride k) to INT8 in ONE launch covering all
    /// m activation rows, instead of m separate <see cref="LaunchQuantizeXToQ8_1"/> calls.
    /// Scratch layout is the single-row layout's sections concatenated by row (NOT interleaved
    /// per-row blocks): <c>int8_t xq[m,k] | half dx[m,k/32] | half sx2[m,k/16]</c>, each section
    /// itself row-major with row stride k, k/32, k/16 respectively. Size the scratch as
    /// <c>m * CudaForwardState.PreQ8_1ScratchBytes(k)</c> bytes. Consumed by
    /// <see cref="LaunchQuantizedGemvMmqBatchedQ4K"/>.
    /// </summary>
    public void LaunchQuantizeXToQ8_1Batched(nint x, nint scratch, int k, int m, nint stream)
    {
        if (_quantizeXToQ8_1BatchedFunc == 0)
            throw new InvalidOperationException(
                "Batched pre-Q8_1 quantization kernel not available. Compile native/kernels/quantize_x.cu to PTX.");
        if ((k & 31) != 0)
            throw new ArgumentException($"k must be a multiple of 32 (got {k}).", nameof(k));

        int numChunks = k >> 5;
        nint xqPtr  = scratch;
        nint dxPtr  = scratch + (nint)((long)m * k);
        nint sx2Ptr = dxPtr + (nint)((long)m * numChunks * 2);

        nint xArg = x, xqArg = xqPtr, dxArg = dxPtr, sx2Arg = sx2Ptr;
        int kArg = k;
        void** args = stackalloc void*[] { &xArg, &xqArg, &dxArg, &sx2Arg, &kArg };

        // Must mirror QX_THREADS_X / QX_WARPS_PER_BLOCK in quantize_x.cu (32 × 8 = 256).
        const uint QxThreadsX = 32;
        const uint QxWarpsPerBlock = 8;
        uint gridX = (uint)((numChunks + QxWarpsPerBlock - 1) / QxWarpsPerBlock);
        CudaDriverApi.cuLaunchKernel(_quantizeXToQ8_1BatchedFunc,
                gridX, (uint)m, 1, QxThreadsX, QxWarpsPerBlock, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Batched-M dp4a MMQ prefill GEMV for Q4_K (issue #349 proof of concept). Computes
    /// <c>Y[m, n] = X[m, k] × W[n, k]^T</c> for Q4_K-quantized W in a single launch — replacing
    /// the dequant→cuBLAS HGEMM prefill fallback. Requires <paramref name="preqScratch"/> to
    /// already hold the output of <see cref="LaunchQuantizeXToQ8_1Batched"/> for the same x/k/m.
    /// Gate calls with <see cref="HasMmqBatchedQ4K"/>.
    /// </summary>
    public void LaunchQuantizedGemvMmqBatchedQ4K(nint quantWeight, nint preqScratch,
                                                   nint y, int n, int k, int m, nint stream)
    {
        if (_quantizedGemvQ4_KMmqBatchedFunc == 0)
            throw new InvalidOperationException(
                "Batched MMQ GEMV kernel not available. Compile native/kernels/quantized_gemv_mmq.cu to PTX.");

        int numChunks = k >> 5;
        nint xqPtr  = preqScratch;
        nint dxPtr  = preqScratch + (nint)((long)m * k);
        nint sx2Ptr = dxPtr + (nint)((long)m * numChunks * 2);

        nint wArg = quantWeight, xqArg = xqPtr, dxArg = dxPtr, sx2Arg = sx2Ptr, yArg = y;
        int nArg = n, kArg = k, mArg = m;
        void** args = stackalloc void*[] { &wArg, &xqArg, &dxArg, &sx2Arg, &yArg, &nArg, &kArg, &mArg };

        // Must mirror MMQ_ROWS_PER_BLOCK / MMQ_BATCH_M_TILE in quantized_gemv_mmq.cu.
        const int MmqRowsPerBlock = 4;
        const int MmqBatchMTile = 2;
        uint gridX = (uint)((n + MmqRowsPerBlock - 1) / MmqRowsPerBlock);
        uint gridY = (uint)((m + MmqBatchMTile - 1) / MmqBatchMTile);

        // No dynamic shmem — pure register + warp-shfl accumulation, no budget check needed.
        CudaDriverApi.cuLaunchKernel(_quantizedGemvQ4_KMmqBatchedFunc,
                gridX, gridY, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>MMQ-GEMV-specific overload of <see cref="CheckDynamicSharedBudget(uint, string)"/>.</summary>
    private void CheckDynamicSharedBudget(uint dynShmem, QuantizationType qt, int k)
        => CheckDynamicSharedBudget(dynShmem, $"MMQ GEMV {qt} k={k}");

    /// <summary>
    /// Pre-launch check that the requested dynamic shmem fits the device budget. Failing
    /// fast here gives a much clearer error than CUDA's generic CUDA_ERROR_INVALID_VALUE
    /// (or, worse, a downstream "illegal memory access" from an out-of-bounds shared-memory
    /// write inside the kernel) when sharedMemBytes exceeds the opt-in cap. Skipped if we
    /// couldn't query the cap.
    /// </summary>
    private void CheckDynamicSharedBudget(uint dynShmem, string label)
    {
        if (_maxDynamicSharedBytesOptIn <= 0) return;
        if (dynShmem <= (uint)_maxDynamicSharedBytesOptIn) return;
        throw new InvalidOperationException(
            $"{label} requires {dynShmem} bytes of dynamic shared memory, " +
            $"exceeding the device cap of {_maxDynamicSharedBytesOptIn} bytes. " +
            "Either route through the dequantize-then-cuBLAS-FP16 fallback or fan the matmul " +
            "across multiple kernel launches.");
    }

    /// <summary>
    /// Compute the dynamic shared-memory bytes needed by an on-the-fly MMQ kernel
    /// for the given quant type and k. Layout per chunk (32 elements):
    /// <list type="bullet">
    /// <item>Q4_K / Q5_K: 32 INT8 (s_xq) + 1 half (s_dx) + 1 half (s_sx) = 36 bytes/chunk</item>
    /// <item>Q2_K / Q6_K: 32 INT8 (s_xq) + 1 half (s_dx) + 2 halves (s_sx2 lo, hi) = 38 bytes/chunk</item>
    /// </list>
    /// Q2_K and Q6_K both use 16-element sub-blocks so the chunk-32 covers two
    /// sub-blocks and the dmin/min term needs per-half-chunk xsum precomputation.
    /// SmolLM-135M k=576 → 18 chunks → 648 B (Q4_K) / 684 B (Q2_K, Q6_K). Qwen3-8B
    /// k=12288 → 384 chunks → 13.5 KB / 14.25 KB. Llama-405B k=53248 → 1664 chunks
    /// → ~58 KB / ~62 KB (still under the 100 KB sm_86 optin cap).
    /// </summary>
    private static int ComputeMmqDynamicSharedBytes(QuantizationType qt, int k)
    {
        int numChunks = k >> 5;
        int bytesPerChunk = (qt == QuantizationType.Q6_K || qt == QuantizationType.Q2_K) ? 38 : 36;
        return numChunks * bytesPerChunk;
    }

    /// <summary>Dequantize a weight matrix to FP32 on the GPU.</summary>
    /// <summary>Whether the TurboQuant KV codec kernels are loaded (turboquant.ptx present).</summary>
    public bool TurboQuantAvailable => _turboquantDequantF32Func != 0 && _turboquantEncodeF32Func != 0;

    /// <summary>
    /// TurboQuant (MSE-stage) dequant: per-head-vector codes + fp32 norm → contiguous fp32 scratch.
    /// One CUDA block per head-vector (<paramref name="numVectors"/> = positions × numKvHeads), 256
    /// threads. Mirrors the Vulkan turboquant_dequant_f32 shader and the CPU codec.
    /// </summary>
    public unsafe void LaunchTurboQuantDequantF32(
        nint codes, nint norms, nint centroids, nint signs, nint dst,
        int numVectors, int headDim, int numKvHeads, int mseBits, int codeUintsPerVec, float invSqrtD, nint stream)
    {
        if (_turboquantDequantF32Func == 0)
            throw new InvalidOperationException("turboquant_dequant_f32 not loaded (turboquant.ptx missing/stale).");
        nint c = codes, n = norms, ce = centroids, s = signs, d = dst;
        int nv = numVectors, hd = headDim, nkv = numKvHeads, mb = mseBits, cu = codeUintsPerVec;
        float isd = invSqrtD;
        void** args = stackalloc void*[] { &c, &n, &ce, &s, &d, &nv, &hd, &nkv, &mb, &cu, &isd };
        CudaDriverApi.cuLaunchKernel(_turboquantDequantF32Func,
                (uint)numVectors, 1, 1, 256, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// TurboQuant (MSE-stage) encode: contiguous fresh fp32 K/V <c>[seqLen, numKvHeads*headDim]</c> →
    /// per-head-vector codes + fp32 norm. One CUDA block per head-vector (<paramref name="seqLen"/> ×
    /// numKvHeads), 256 threads. Destination head-vector index is <c>(startPos + srcRow)*numKvHeads + h</c>.
    /// </summary>
    public unsafe void LaunchTurboQuantEncodeF32(
        nint src, nint centroids, nint signs, nint codes, nint norms,
        int seqLen, int headDim, int numKvHeads, int mseBits, int codeUintsPerVec, int levelCount, int startPos,
        float invSqrtD, nint stream)
    {
        if (_turboquantEncodeF32Func == 0)
            throw new InvalidOperationException("turboquant_encode_f32 not loaded (turboquant.ptx missing/stale).");
        nint sr = src, ce = centroids, sg = signs, co = codes, no = norms;
        int hd = headDim, nkv = numKvHeads, mb = mseBits, cu = codeUintsPerVec, lc = levelCount, sp = startPos;
        float isd = invSqrtD;
        void** args = stackalloc void*[] { &sr, &ce, &sg, &co, &no, &hd, &nkv, &mb, &cu, &lc, &sp, &isd };
        CudaDriverApi.cuLaunchKernel(_turboquantEncodeF32Func,
                (uint)(seqLen * numKvHeads), 1, 1, 256, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Whether the FP16 TurboQuant KV codec kernels are loaded (for the CUDA forward path).</summary>
    public bool TurboQuantF16Available => _turboquantDequantF16Func != 0 && _turboquantEncodeF16Func != 0;

    /// <summary>FP16 TurboQuant dequant: codes + fp32 norm → contiguous <c>half</c> scratch (attention input).</summary>
    public unsafe void LaunchTurboQuantDequantF16(
        nint codes, nint norms, nint centroids, nint signs, nint dst,
        int numVectors, int headDim, int numKvHeads, int mseBits, int codeUintsPerVec, float invSqrtD, nint stream)
    {
        if (_turboquantDequantF16Func == 0)
            throw new InvalidOperationException("turboquant_dequant_f16 not loaded (turboquant.ptx missing/stale).");
        nint c = codes, n = norms, ce = centroids, s = signs, d = dst;
        int nv = numVectors, hd = headDim, nkv = numKvHeads, mb = mseBits, cu = codeUintsPerVec;
        float isd = invSqrtD;
        void** args = stackalloc void*[] { &c, &n, &ce, &s, &d, &nv, &hd, &nkv, &mb, &cu, &isd };
        CudaDriverApi.cuLaunchKernel(_turboquantDequantF16Func,
                (uint)numVectors, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>FP16 TurboQuant encode: fresh <c>half</c> K/V <c>[seqLen, numKvHeads*headDim]</c> → codes + fp32 norm.</summary>
    public unsafe void LaunchTurboQuantEncodeF16(
        nint src, nint centroids, nint signs, nint codes, nint norms,
        int seqLen, int headDim, int numKvHeads, int mseBits, int codeUintsPerVec, int levelCount, int startPos,
        float invSqrtD, nint stream)
    {
        if (_turboquantEncodeF16Func == 0)
            throw new InvalidOperationException("turboquant_encode_f16 not loaded (turboquant.ptx missing/stale).");
        nint sr = src, ce = centroids, sg = signs, co = codes, no = norms;
        int hd = headDim, nkv = numKvHeads, mb = mseBits, cu = codeUintsPerVec, lc = levelCount, sp = startPos;
        float isd = invSqrtD;
        void** args = stackalloc void*[] { &sr, &ce, &sg, &co, &no, &hd, &nkv, &mb, &cu, &lc, &sp, &isd };
        CudaDriverApi.cuLaunchKernel(_turboquantEncodeF16Func,
                (uint)(seqLen * numKvHeads), 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Dequantizes a quantized weight/KV blob to a contiguous fp32 buffer on device.</summary>
    public void LaunchDequantToF32(nint src, QuantizationType srcDtype,
                                     nint dst, int totalElements, nint stream)
    {
        nint srcArg = src, dstArg = dst;

        switch (srcDtype)
        {
            case QuantizationType.F32:
                // (long) widening is load-bearing: `totalElements * 4` overflows int at
                // totalElements > 2^29 (a 2 GiB F32 tensor — reachable for a large
                // embedding / LM head inside a 12–16 GB budget).
                CudaDriverApi.cuMemcpyDtoD_v2(dst, src, (nuint)((long)totalElements * sizeof(float))).ThrowOnError();
                return;

            case QuantizationType.F16:
                LaunchConvertF16ToF32(src, dst, totalElements, stream);
                return;

            case QuantizationType.BF16:
            {
                if (_dequantBF16F32Func == 0)
                    break;
                int teArg = totalElements;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &teArg};
                uint gridDim = (uint)Math.Min((totalElements + (int)BlockSize - 1) / (int)BlockSize,
                    MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantBF16F32Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.MXFP4:
            {
                if (_dequantMXFP4F32Func == 0)
                    break;
                int totalBlocks = totalElements / 32;
                int tbArg = totalBlocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tbArg};
                uint gridDim = (uint)Math.Min((totalBlocks + 7) / 8, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantMXFP4F32Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.IQ3_XXS:
            case QuantizationType.IQ3_S:
            case QuantizationType.IQ1_S:
            {
                nint f32Func = srcDtype switch
                {
                    QuantizationType.IQ3_XXS => _dequantIQ3_XXSF32Func,
                    QuantizationType.IQ3_S => _dequantIQ3_SF32Func,
                    _ => _dequantIQ1_SF32Func
                };
                if (f32Func == 0)
                    break;
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(f32Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.IQ4_NL:
            {
                if (_dequantIQ4_NLF32Func == 0)
                    break;
                int totalBlocks = totalElements / 32;
                int tbArg = totalBlocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tbArg};
                // 8 IQ4_NL blocks (256 elements) per CTA — matches the kernel's
                // thread grouping fixed in issue #265.
                uint gridDim = (uint)Math.Min((totalBlocks + 7) / 8, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantIQ4_NLF32Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.IQ4_XS:
            {
                if (_dequantIQ4_XSF32Func == 0)
                    break;
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantIQ4_XSF32Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.IQ2_XXS:
            case QuantizationType.IQ2_XS:
            case QuantizationType.IQ2_S:
            {
                // All three IQ2 dequant F32 kernels share the same launch shape (one
                // thread per element within a 256-element super-block). Resolve the
                // function by quant type and fall through to break (-> NotSupported)
                // when the PTX hasn't shipped the symbol.
                nint func = srcDtype switch
                {
                    QuantizationType.IQ2_XXS => _dequantIQ2_XXSF32Func,
                    QuantizationType.IQ2_XS => _dequantIQ2_XSF32Func,
                    QuantizationType.IQ2_S => _dequantIQ2_SF32Func,
                    _ => 0
                };
                if (func == 0)
                    break;
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.Q5_K:
            {
                if (_dequantQ5_KF32Func == 0)
                    break;
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantQ5_KF32Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.Q6_K:
            {
                if (_dequantQ6_KF32Func == 0)
                    break;
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantQ6_KF32Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }
        }

        throw new NotSupportedException($"GPU FP32 dequantization not supported directly for {srcDtype}.");
    }

    /// <summary>Dequantize a weight matrix to FP16 on the GPU.</summary>
    /// <param name="src">Device pointer to quantized weight data.</param>
    /// <param name="srcDtype">Source quantization type.</param>
    /// <param name="dst">Device pointer to FP16 output buffer.</param>
    /// <param name="totalElements">Total number of output elements.</param>
    /// <param name="stream">CUDA stream.</param>
    public void LaunchDequantToF16(nint src, QuantizationType srcDtype,
                                     nint dst, int totalElements, nint stream)
    {
        nint srcArg = src, dstArg = dst;

        switch (srcDtype)
        {
            case QuantizationType.F16:
                // Already FP16, just copy.
                // (long) widening is load-bearing: `totalElements * 2` overflows int at
                // totalElements > 2^30 (a 2 GiB F16 tensor).
                CudaDriverApi.cuMemcpyDtoD_v2(dst, src, (nuint)((long)totalElements * 2)).ThrowOnError();
                return;

            case QuantizationType.F32:
                // FP32 → FP16 conversion
                LaunchConvertF32ToF16(src, dst, totalElements, stream);
                return;

            case QuantizationType.Q8_0:
            {
                int totalBlocks = totalElements / 32;
                int tbArg = totalBlocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tbArg};
                // Grid-stride loop: cap grid at MaxDequantGridSize CUDA blocks.
                // Each CUDA block has 8 warps, so natural 1:1 mapping is ceil(totalBlocks/8).
                uint gridDim = (uint)Math.Min((totalBlocks + 7) / 8, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantQ8_0Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.Q4_0:
            {
                int totalBlocks = totalElements / 32;
                int tbArg = totalBlocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tbArg};
                uint gridDim = (uint)Math.Min((totalBlocks + 7) / 8, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantQ4_0Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.Q5_0:
            {
                int totalBlocks = totalElements / 32;
                int tbArg = totalBlocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tbArg};
                uint gridDim = (uint)Math.Min((totalBlocks + 7) / 8, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantQ5_0Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.Q4_1:
            {
                if (_dequantQ4_1Func == 0)
                    throw new InvalidOperationException(
                        "Q4_1 dequant kernel not in dequant.ptx — rebuild PTX from native/kernels/dequant.cu.");
                int totalBlocks = totalElements / 32;
                int tbArg = totalBlocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tbArg};
                uint gridDim = (uint)Math.Min((totalBlocks + 7) / 8, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantQ4_1Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.Q5_1:
            {
                if (_dequantQ5_1Func == 0)
                    throw new InvalidOperationException(
                        "Q5_1 dequant kernel not in dequant.ptx — rebuild PTX from native/kernels/dequant.cu.");
                int totalBlocks = totalElements / 32;
                int tbArg = totalBlocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tbArg};
                uint gridDim = (uint)Math.Min((totalBlocks + 7) / 8, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantQ5_1Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.Q2_K:
            {
                if (_dequantQ2_KFunc == 0)
                    throw new InvalidOperationException(
                        "Q2_K dequant kernel not present in dequant.ptx — rebuild PTX from " +
                        "native/kernels/dequant.cu.");
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantQ2_KFunc,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.IQ4_NL:
            {
                if (_dequantIQ4_NLFunc == 0)
                    throw new InvalidOperationException(
                        "IQ4_NL dequant kernel not present in dequant_iquants.ptx - rebuild PTX from " +
                        "native/kernels/dequant_iquants.cu.");
                int totalBlocks = totalElements / 32;
                int tbArg = totalBlocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tbArg};
                // Issue #265: kernel now groups 8 IQ4_NL blocks (256 elements) per
                // CTA so all 256 threads participate (was 32/256, 1/8 occupancy).
                // Grid dim must match that grouping, same formula already used by
                // LaunchDequantToF32's IQ4_NL case below.
                uint gridDim = (uint)Math.Min((totalBlocks + 7) / 8, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantIQ4_NLFunc,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.IQ4_XS:
            {
                if (_dequantIQ4_XSFunc == 0)
                    throw new InvalidOperationException(
                        "IQ4_XS dequant kernel not present in dequant_iquants.ptx - rebuild PTX from " +
                        "native/kernels/dequant_iquants.cu.");
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantIQ4_XSFunc,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.IQ2_XXS:
            case QuantizationType.IQ2_XS:
            case QuantizationType.IQ2_S:
            {
                // All three IQ2 dequant F16 kernels follow IQ4_XS's super-block shape
                // (one thread per element within a 256-element block). IQ2_S doubles
                // as the on-disk format for IQ2_M file-type GGUFs (e.g. Qwen3.6-A3B-IQ2_M).
                nint func = srcDtype switch
                {
                    QuantizationType.IQ2_XXS => _dequantIQ2_XXSFunc,
                    QuantizationType.IQ2_XS => _dequantIQ2_XSFunc,
                    QuantizationType.IQ2_S => _dequantIQ2_SFunc,
                    _ => 0
                };
                if (func == 0)
                    throw new InvalidOperationException(
                        $"{srcDtype} dequant kernel not present in iq2.ptx — rebuild PTX from " +
                        "native/kernels/iq2.cu.");
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.BF16:
            {
                // BF16 is the high 16 bits of an F32; widening is a bit-shift,
                // so this is elementwise rather than block-structured.
                if (_dequantBF16Func == 0)
                    throw new InvalidOperationException(
                        "BF16 dequant kernel not present in dequant_bf16_mxfp4.ptx — rebuild PTX from " +
                        "native/kernels/dequant_bf16_mxfp4.cu.");
                int teArg = totalElements;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &teArg};
                uint gridDim = (uint)Math.Min((totalElements + (int)BlockSize - 1) / (int)BlockSize,
                    MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantBF16Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.MXFP4:
            {
                if (_dequantMXFP4Func == 0)
                    throw new InvalidOperationException(
                        "MXFP4 dequant kernel not present in dequant_bf16_mxfp4.ptx — rebuild PTX from " +
                        "native/kernels/dequant_bf16_mxfp4.cu.");
                int totalBlocks = totalElements / 32;
                int tbArg = totalBlocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tbArg};
                // Each 256-thread CUDA block covers 8 MXFP4 blocks (32 elements each).
                uint gridDim = (uint)Math.Min((totalBlocks + 7) / 8, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantMXFP4Func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.IQ3_XXS:
            case QuantizationType.IQ3_S:
            case QuantizationType.IQ1_S:
            {
                // Super-block shaped like the IQ2/IQ4_XS kernels: one thread per
                // element within a 256-element super-block, one CUDA block per
                // super-block, grid-strided.
                (nint func, string ptx) = srcDtype switch
                {
                    QuantizationType.IQ3_XXS => (_dequantIQ3_XXSFunc, "dequant_iq3"),
                    QuantizationType.IQ3_S => (_dequantIQ3_SFunc, "dequant_iq3"),
                    _ => (_dequantIQ1_SFunc, "dequant_iq1")
                };
                if (func == 0)
                    throw new InvalidOperationException(
                        $"{srcDtype} dequant kernel not present in {ptx}.ptx — rebuild PTX from " +
                        $"native/kernels/{ptx}.cu.");
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(func,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.Q3_K:
            {
                if (_dequantQ3_KFunc == 0)
                    throw new InvalidOperationException(
                        "Q3_K dequant kernel not present in dequant.ptx — rebuild PTX from " +
                        "native/kernels/dequant.cu (Round 12+ adds Q3_K support).");
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantQ3_KFunc,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.Q4_K:
            {
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                // Grid-stride loop: 1 CUDA block per superblock naturally, capped at MaxDequantGridSize
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantQ4_KFunc,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.Q5_K:
            {
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantQ5_KFunc,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            case QuantizationType.Q6_K:
            {
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantQ6_KFunc,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }

            default:
                throw new NotSupportedException($"GPU dequantization not supported for {srcDtype}.");
        }
    }

    /// <summary>
    /// Quantizes a single row of FP16 KV data to Q8_0 or Q4_0 on the GPU.
    /// Used for KV-cache quantize-on-evict.
    /// </summary>
    /// <param name="src">Device pointer to FP16 input [elementCount].</param>
    /// <param name="dst">Device pointer to quantized output buffer.</param>
    /// <param name="elementCount">Number of elements to quantize (must be multiple of 32).</param>
    /// <param name="dtype">Target quantization type.</param>
    /// <param name="stream">CUDA stream.</param>
    public unsafe void LaunchQuantKv(nint src, nint dst, int elementCount,
                                      Core.Configuration.KvCacheDType dtype, nint stream)
    {
        if (_quantKvModule == null)
            throw new InvalidOperationException(
                "KV-cache quantization kernels not available. Compile native/kernels/quant_kv.cu to PTX.");

        int totalBlocks = elementCount / 32;
        nint srcArg = src, dstArg = dst;
        int tbArg = totalBlocks;
        void** args = stackalloc void*[] { &srcArg, &dstArg, &tbArg };
        uint gridDim = (uint)((totalBlocks + BlockSize - 1) / BlockSize);

        nint func = dtype switch
        {
            Core.Configuration.KvCacheDType.Q8_0 => _quantKvQ8_0Func,
            Core.Configuration.KvCacheDType.Q4_0 => _quantKvQ4_0Func,
            _ => throw new NotSupportedException($"KV quantization not supported for {dtype}")
        };

        CudaDriverApi.cuLaunchKernel(func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Predicated FP16 → Q8_0 / Q4_0 KV-row quantizer for the CUDA Graphs
    /// decode path. Source row is selected from <paramref name="windowBase"/>
    /// (FP16 ring buffer) and destination row in <paramref name="quantBase"/>
    /// (per-layer Q-cache) using the absolute decode position read from
    /// <paramref name="posPtr"/>. Until the FP16 window fills (i.e.
    /// <c>pos &lt; windowSize</c>) the kernel returns immediately, so it is
    /// safe to launch on every decode step. One CUDA block of <c>kvStride/32</c>
    /// threads quantizes a single row.
    /// </summary>
    public void LaunchQuantKvDyn(nint windowBase, nint quantBase, int kvStride, int windowSize,
                                   KvCacheDType dtype, nint posPtr, nint stream)
    {
        if (_quantKvModule == null)
            throw new InvalidOperationException(
                "KV-cache quantization kernels not available. Compile native/kernels/quant_kv.cu to PTX.");

        nint wArg = windowBase, qArg = quantBase, posArg = posPtr;
        int kvArg = kvStride, wsArg = windowSize;
        void** args = stackalloc void*[] { &wArg, &qArg, &kvArg, &wsArg, &posArg };

        int totalBlocksPerRow = kvStride / 32;
        // Single CUDA block per row covers up to 256 quant blocks (kvStride ≤ 8192) at
        // 256 threads/block — typical models stay well under that.
        uint gridDim = (uint)((totalBlocksPerRow + BlockSize - 1) / BlockSize);
        if (gridDim == 0) gridDim = 1;

        nint func = dtype switch
        {
            KvCacheDType.Q8_0 => _quantKvQ8_0DynFunc,
            KvCacheDType.Q4_0 => _quantKvQ4_0DynFunc,
            _ => throw new NotSupportedException($"Dynamic KV quantization not supported for {dtype}.")
        };

        CudaDriverApi.cuLaunchKernel(func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Multi-head Latent Attention (MLA) Phase A naive forward — F32 throughout.
    /// One CUDA block per (query_token, head) pair. Computes the equivalent of
    /// <c>MlaAttention.Execute</c>'s attention loop:
    /// <c>scores = Q_nope · K_nope_per_head + Q_pe · K_pe_shared, scaled by softmax_scale,
    /// causal-masked, softmaxed, then weighted sum over per-head V (which has its own
    /// vHead dim — typically 128 vs the 192-wide attention score dim).</c>
    /// </summary>
    /// <param name="q">F32 [seqQ, numHeads, qkNopeHeadDim + qkRopeHeadDim] (per-head Q with split dims).</param>
    /// <param name="kNope">F32 [seqKv, numHeads, qkNopeHeadDim] (per-head K_nope cache slice).</param>
    /// <param name="kPe">F32 [seqKv, qkRopeHeadDim] (MQA-shared K_pe cache slice, RoPE-applied).</param>
    /// <param name="v">F32 [seqKv, numHeads, vHeadDim] (per-head V cache slice).</param>
    /// <param name="output">F32 [seqQ, numHeads, vHeadDim] (per-head attention output).</param>
    /// <param name="seqQ">Number of query tokens this call produces output for.</param>
    /// <param name="seqKv">Total cached length the queries attend over (= cachedLength + seqQ in autoregression).</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="qkNopeHeadDim">Per-head non-rope Q·K dim.</param>
    /// <param name="qkRopeHeadDim">Per-head rope Q·K dim (must be even).</param>
    /// <param name="vHeadDim">Per-head V dim (may differ from qkHeadDim).</param>
    /// <param name="positionOffset">Absolute position of query token 0 (for the causal mask).</param>
    /// <param name="softmaxScale">Combined softmax scale: <c>(1 / sqrt(qk_head_dim)) * yarn_mscale²</c>.</param>
    /// <param name="stream">CUDA stream.</param>
    public void LaunchAttentionMla(
        nint q, nint kNope, nint kPe, nint v, nint output,
        int seqQ, int seqKv,
        int numHeads, int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
        int positionOffset, float softmaxScale, nint stream)
    {
        if (_attentionMlaF32Func == 0)
            throw new InvalidOperationException(
                "MLA attention kernel not available. Compile native/kernels/attention_mla.cu to PTX.");

        nint qArg = q, kNopeArg = kNope, kPeArg = kPe, vArg = v, outArg = output;
        int sqArg = seqQ, skvArg = seqKv;
        int nhArg = numHeads;
        int nopeArg = qkNopeHeadDim, ropeArg = qkRopeHeadDim, vhArg = vHeadDim;
        int poArg = positionOffset;
        float scaleArg = softmaxScale;

        void** args = stackalloc void*[] {
            &qArg, &kNopeArg, &kPeArg, &vArg, &outArg,
            &sqArg, &skvArg,
            &nhArg, &nopeArg, &ropeArg, &vhArg,
            &poArg, &scaleArg
        };

        int numBlocks = seqQ * numHeads;
        // Shared memory layout: q_nope[qkNope] + q_pe[qkRope] + score_tile[128] + out_accum[vHead] + warp_scratch[32]
        const int TileKv = 128;
        uint sharedBytes = (uint)((qkNopeHeadDim + qkRopeHeadDim + TileKv + vHeadDim + 32) * sizeof(float));
        // Block size is 128 (matches __launch_bounds__ in attention_mla.cu).
        const uint MlaBlockSize = 128;

        CudaDriverApi.cuLaunchKernel(_attentionMlaF32Func,
                (uint)numBlocks, 1, 1, MlaBlockSize, 1, 1,
                sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Per-head split of the kv_b expansion for MLA Phase A. Reads packed
    /// <c>[seqLen, numHeads * (qkNope + vHead)]</c> F32 and writes per-head
    /// K_nope <c>[seqLen, numHeads * qkNope]</c> + per-head V
    /// <c>[seqLen, numHeads * vHead]</c>. One CUDA block per (token, head).
    /// </summary>
    public void LaunchMlaSplitKvB(
        nint kvBExpanded, nint kNopeDst, nint vDst,
        int seqLen, int numHeads, int qkNopeHeadDim, int vHeadDim, nint stream)
    {
        if (_mlaSplitKvBF32Func == 0)
            throw new InvalidOperationException(
                "MLA helper kernels not available. Compile native/kernels/mla_helpers.cu to PTX.");

        nint srcArg = kvBExpanded, kArg = kNopeDst, vArg = vDst;
        int slArg = seqLen, nhArg = numHeads, nopeArg = qkNopeHeadDim, vhArg = vHeadDim;
        void** args = stackalloc void*[] {
            &srcArg, &kArg, &vArg,
            &slArg, &nhArg, &nopeArg, &vhArg
        };
        uint blocks = (uint)(seqLen * numHeads);
        const uint block = 128;
        CudaDriverApi.cuLaunchKernel(_mlaSplitKvBF32Func,
                blocks, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// In-place RoPE on the rope sub-dim of Q (per head). Norm-pair convention.
    /// One CUDA block per (token, head).
    /// </summary>
    public void LaunchMlaRopeQpe(
        nint q, nint cosTab, nint sinTab,
        int seqLen, int numHeads, int qkNopeHeadDim, int qkRopeHeadDim,
        int positionOffset, nint stream)
    {
        if (_mlaRopeQpeF32Func == 0)
            throw new InvalidOperationException("MLA helper kernels not available.");

        nint qArg = q, cosArg = cosTab, sinArg = sinTab;
        int slArg = seqLen, nhArg = numHeads;
        int nopeArg = qkNopeHeadDim, ropeArg = qkRopeHeadDim;
        int poArg = positionOffset;
        void** args = stackalloc void*[] {
            &qArg, &cosArg, &sinArg,
            &slArg, &nhArg, &nopeArg, &ropeArg, &poArg
        };
        uint blocks = (uint)(seqLen * numHeads);
        const uint block = 128;
        CudaDriverApi.cuLaunchKernel(_mlaRopeQpeF32Func,
                blocks, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// In-place RoPE on the MQA-shared K_pe (one rope vector per token, no head dim).
    /// One CUDA block per token.
    /// </summary>
    public void LaunchMlaRopeKpe(
        nint kPe, nint cosTab, nint sinTab,
        int seqLen, int qkRopeHeadDim, int positionOffset, nint stream)
    {
        if (_mlaRopeKpeF32Func == 0)
            throw new InvalidOperationException("MLA helper kernels not available.");

        nint kArg = kPe, cosArg = cosTab, sinArg = sinTab;
        int slArg = seqLen, ropeArg = qkRopeHeadDim, poArg = positionOffset;
        void** args = stackalloc void*[] {
            &kArg, &cosArg, &sinArg,
            &slArg, &ropeArg, &poArg
        };
        uint blocks = (uint)seqLen;
        const uint block = 128;
        CudaDriverApi.cuLaunchKernel(_mlaRopeKpeF32Func,
                blocks, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// F32 RMSNorm with explicit (numRows, dim) layout. One CUDA block per row.
    /// Used by MLA's q_a_layernorm and kv_a_layernorm.
    /// </summary>
    public void LaunchMlaRmsNormF32(
        nint input, nint weight, nint output,
        int numRows, int dim, float epsilon, nint stream)
    {
        if (_mlaRmsNormF32Func == 0)
            throw new InvalidOperationException("MLA helper kernels not available.");

        nint inArg = input, wArg = weight, outArg = output;
        int dimArg = dim;
        float epsArg = epsilon;
        void** args = stackalloc void*[] {
            &inArg, &wArg, &outArg, &dimArg, &epsArg
        };
        const uint block = 128;
        CudaDriverApi.cuLaunchKernel(_mlaRmsNormF32Func,
                (uint)numRows, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    // ── MoE launchers ────────────────────────────────────────────────────

    /// <summary>
    /// Additive router-bias add (issue #246): <c>logits[t, e] += bias[e]</c> for every
    /// token. Must be called BEFORE <see cref="LaunchMoeSoftmaxTopk"/> — mirrors
    /// <c>MoeSwiGluMlp.Route</c>'s CPU ordering (bias added to raw logits, then
    /// softmax, then top-k). Used by identity-MoTE / Qwen3 aux-loss-free routing.
    /// </summary>
    public void LaunchMoeGateBiasAddF32(nint logits, nint bias, int seqLen, int numExperts, nint stream)
    {
        if (_moeGateBiasAddF32Func == 0)
            throw new InvalidOperationException("MoE gate-bias-add kernel not available.");

        nint logitsArg = logits, biasArg = bias;
        int slArg = seqLen, neArg = numExperts;
        void** args = stackalloc void*[] {&logitsArg, &biasArg, &slArg, &neArg};
        uint gridDim = (uint)((seqLen * numExperts + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_moeGateBiasAddF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// gpt-oss clamped SwiGLU activation (issue #348): <c>out = x/(1+exp(-alpha*x)) * (y+1)</c>
    /// where <c>x=min(gate,limit)</c>, <c>y=clamp(up,-limit,limit)</c>. Safe to call with
    /// <paramref name="output"/> aliasing <paramref name="gate"/>.
    /// </summary>
    public void LaunchSwiGLUOaiF32(nint gate, nint up, nint output, int n, int seqLen,
        float alpha, float limit, nint stream)
    {
        if (_swigluOaiF32Func == 0)
            throw new InvalidOperationException(
                "swiglu_oai_f32 kernel not available. Recompile native/kernels/moe_ffn.cu to PTX.");

        nint gateArg = gate, upArg = up, outArg = output;
        int nArg = n, slArg = seqLen;
        float alphaArg = alpha, limitArg = limit;
        void** args = stackalloc void*[] {&gateArg, &upArg, &outArg, &nArg, &slArg, &alphaArg, &limitArg};
        uint total = (uint)(n * seqLen);
        uint gridDim = (total + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(_swigluOaiF32Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Per-token softmax + top-k selection for MoE routing. Reads
    /// <c>logits[seqLen, numExperts]</c>; writes <c>topkIdx[seqLen, topK]</c>
    /// (int32) and <c>topkWeight[seqLen, topK]</c> (F32, raw softmax probabilities
    /// — caller invokes <see cref="LaunchMoeRenormTopk"/> separately when
    /// <c>norm_topk_prob</c> is true).
    /// </summary>
    public void LaunchMoeSoftmaxTopk(
        nint logits, nint topkIdx, nint topkWeight,
        int seqLen, int numExperts, int topK, nint stream)
    {
        if (_moeSoftmaxTopkF32Func == 0)
            throw new InvalidOperationException(
                "MoE kernels not available. Compile native/kernels/moe_ffn.cu to PTX.");
        if (topK > 64)
            throw new ArgumentOutOfRangeException(nameof(topK),
                "topK > 64 is not supported by the GPU MoE kernel (kernel-side fixed-size scratch).");

        nint logitsArg = logits, idxArg = topkIdx, wArg = topkWeight;
        int slArg = seqLen, neArg = numExperts, kArg = topK;
        void** args = stackalloc void*[] {
            &logitsArg, &idxArg, &wArg,
            &slArg, &neArg, &kArg
        };
        // Shared memory: numExperts floats (softmax probs) + 4 floats (warp scratch).
        uint sharedBytes = (uint)((numExperts + 4) * sizeof(float));
        const uint block = 128;
        CudaDriverApi.cuLaunchKernel(_moeSoftmaxTopkF32Func,
                (uint)seqLen, 1, 1, block, 1, 1,
                sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Per-token in-place renormalisation of top-k weights to sum to 1.0
    /// (Mixtral / Qwen3-MoE convention; skip for Qwen1.5-MoE).
    /// </summary>
    public void LaunchMoeRenormTopk(nint topkWeight, int seqLen, int topK, nint stream)
    {
        if (_moeRenormTopkF32Func == 0)
            throw new InvalidOperationException("MoE kernels not available.");
        nint wArg = topkWeight;
        int slArg = seqLen, kArg = topK;
        void** args = stackalloc void*[] { &wArg, &slArg, &kArg };
        const uint block = 32;
        CudaDriverApi.cuLaunchKernel(_moeRenormTopkF32Func,
                (uint)seqLen, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Zero a flat F32 device buffer of <paramref name="n"/> elements.</summary>
    public void LaunchMoeZeroF32(nint buf, int n, nint stream)
    {
        if (_moeZeroF32Func == 0)
            throw new InvalidOperationException("MoE kernels not available.");
        nint bArg = buf;
        int nArg = n;
        void** args = stackalloc void*[] { &bArg, &nArg };
        const uint block = 256;
        uint grid = (uint)Math.Min(MaxDequantGridSize, (n + (int)block - 1) / (int)block);
        if (grid == 0) grid = 1;
        CudaDriverApi.cuLaunchKernel(_moeZeroF32Func,
                grid, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Per-expert weighted accumulator. For each batch row b at output token
    /// <c>tokenIndices[b]</c>: <c>output[t,:] += topkWeight[t, slotIndex] * down[b,:]</c>.
    /// Single block per batch row.
    /// </summary>
    public void LaunchMoeAxpyScaledRowF32(
        nint output, nint down, nint topkWeight, nint tokenIndices,
        int batchSize, int hidden, int topK, int slotIndex, nint stream)
    {
        if (_moeAxpyScaledRowF32Func == 0)
            throw new InvalidOperationException("MoE kernels not available.");
        nint outArg = output, downArg = down, wArg = topkWeight, tiArg = tokenIndices;
        int bArg = batchSize, hArg = hidden, kArg = topK, sArg = slotIndex;
        void** args = stackalloc void*[] {
            &outArg, &downArg, &wArg, &tiArg,
            &bArg, &hArg, &kArg, &sArg
        };
        const uint block = 256;
        CudaDriverApi.cuLaunchKernel(_moeAxpyScaledRowF32Func,
                (uint)batchSize, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Unweighted accumulator: <c>output[t,:] += down[t,:]</c> for all tokens.
    /// Used by the shared-expert path (no per-token gating, e.g. DeepSeek).
    /// </summary>
    public void LaunchMoeAxpyUnweightedF32(
        nint output, nint down, int seqLen, int hidden, nint stream)
    {
        if (_moeAxpyUnweightedF32Func == 0)
            throw new InvalidOperationException("MoE kernels not available.");
        nint outArg = output, downArg = down;
        int slArg = seqLen, hArg = hidden;
        void** args = stackalloc void*[] { &outArg, &downArg, &slArg, &hArg };
        const uint block = 256;
        CudaDriverApi.cuLaunchKernel(_moeAxpyUnweightedF32Func,
                (uint)seqLen, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Per-token sigmoid-gated accumulator: <c>output[t,:] += scale[t] * down[t,:]</c>.
    /// Used by Qwen1.5-MoE shared_expert_gate path.
    /// </summary>
    public void LaunchMoeAxpyScaledPerTokenF32(
        nint output, nint down, nint scale, int seqLen, int hidden, nint stream)
    {
        if (_moeAxpyScaledPerTokenF32Func == 0)
            throw new InvalidOperationException("MoE kernels not available.");
        nint outArg = output, downArg = down, scArg = scale;
        int slArg = seqLen, hArg = hidden;
        void** args = stackalloc void*[] {
            &outArg, &downArg, &scArg, &slArg, &hArg
        };
        const uint block = 256;
        CudaDriverApi.cuLaunchKernel(_moeAxpyScaledPerTokenF32Func,
                (uint)seqLen, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Per-token sigmoid-gated dot product:
    /// <c>scaleOut[t] = sigmoid(Σ hidden[t,k] * g[k])</c>. One block per token.
    /// </summary>
    public void LaunchMoeSigmoidLogitF32(
        nint hidden, nint g, nint scaleOut, int seqLen, int hiddenSize, nint stream)
    {
        if (_moeSigmoidLogitF32Func == 0)
            throw new InvalidOperationException("MoE kernels not available.");
        nint hArg = hidden, gArg = g, sArg = scaleOut;
        int slArg = seqLen, hsArg = hiddenSize;
        void** args = stackalloc void*[] {
            &hArg, &gArg, &sArg, &slArg, &hsArg
        };
        const uint block = 128;
        // Shared memory: 4 floats for warp-reduce scratch.
        uint sharedBytes = 4 * sizeof(float);
        CudaDriverApi.cuLaunchKernel(_moeSigmoidLogitF32Func,
                (uint)seqLen, 1, 1, block, 1, 1,
                sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Gathers <paramref name="batchSize"/> rows from <c>src[seqLen, hidden]</c>
    /// into <c>dst[batchSize, hidden]</c> indexed by
    /// <c>tokenIndices[b]</c>. Used to assemble per-expert input batches.
    /// </summary>
    public void LaunchMoeGatherTokenRowsF32(
        nint src, nint dst, nint tokenIndices, int batchSize, int hidden, nint stream)
    {
        if (_moeGatherTokenRowsF32Func == 0)
            throw new InvalidOperationException("MoE kernels not available.");
        nint srcArg = src, dstArg = dst, tiArg = tokenIndices;
        int bArg = batchSize, hArg = hidden;
        void** args = stackalloc void*[] {
            &srcArg, &dstArg, &tiArg, &bArg, &hArg
        };
        const uint block = 256;
        CudaDriverApi.cuLaunchKernel(_moeGatherTokenRowsF32Func,
                (uint)batchSize, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    #region MLA FP16 launchers
    /// <summary>
    /// FP16 sibling of <see cref="LaunchAttentionMla"/>. Same online-softmax
    /// algorithm; FP16 inputs/outputs (Q, K_nope, K_pe, V, output) with FP32
    /// softmax accumulator. Shared memory layout matches the F32 kernel
    /// (all FP32 scratch).
    /// </summary>
    public void LaunchAttentionMlaF16(
        nint q, nint kNope, nint kPe, nint v, nint output,
        int seqQ, int seqKv,
        int numHeads, int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
        int positionOffset, float softmaxScale, nint stream)
    {
        if (_attentionMlaF16Func == 0)
            throw new InvalidOperationException(
                "MLA FP16 attention kernel not available. Rebuild PTX from native/kernels/attention_mla.cu.");

        nint qArg = q, kNopeArg = kNope, kPeArg = kPe, vArg = v, outArg = output;
        int sqArg = seqQ, skvArg = seqKv;
        int nhArg = numHeads;
        int nopeArg = qkNopeHeadDim, ropeArg = qkRopeHeadDim, vhArg = vHeadDim;
        int poArg = positionOffset;
        float scaleArg = softmaxScale;

        void** args = stackalloc void*[] {
            &qArg, &kNopeArg, &kPeArg, &vArg, &outArg,
            &sqArg, &skvArg,
            &nhArg, &nopeArg, &ropeArg, &vhArg,
            &poArg, &scaleArg
        };

        int numBlocks = seqQ * numHeads;
        // Shared layout (FP32): q_nope[qkNope] + q_pe[qkRope] + score_tile[128] + out_accum[vHead] + warp_scratch[32]
        const int TileKv = 128;
        uint sharedBytes = (uint)((qkNopeHeadDim + qkRopeHeadDim + TileKv + vHeadDim + 32) * sizeof(float));
        const uint MlaBlockSize = 128;

        CudaDriverApi.cuLaunchKernel(_attentionMlaF16Func,
                (uint)numBlocks, 1, 1, MlaBlockSize, 1, 1,
                sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// FP16 sibling of <see cref="LaunchMlaSplitKvB"/>. Per-head split of the
    /// kv_b expansion: FP16 in / FP16 out.
    /// </summary>
    public void LaunchMlaSplitKvBF16(
        nint kvBExpanded, nint kNopeDst, nint vDst,
        int seqLen, int numHeads, int qkNopeHeadDim, int vHeadDim, nint stream)
    {
        if (_mlaSplitKvBF16Func == 0)
            throw new InvalidOperationException(
                "MLA FP16 split helper not available. Rebuild PTX from native/kernels/mla_helpers.cu.");

        nint srcArg = kvBExpanded, kArg = kNopeDst, vArg = vDst;
        int slArg = seqLen, nhArg = numHeads, nopeArg = qkNopeHeadDim, vhArg = vHeadDim;
        void** args = stackalloc void*[] {
            &srcArg, &kArg, &vArg,
            &slArg, &nhArg, &nopeArg, &vhArg
        };
        uint blocks = (uint)(seqLen * numHeads);
        const uint block = 128;
        CudaDriverApi.cuLaunchKernel(_mlaSplitKvBF16Func,
                blocks, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// FP16 sibling of <see cref="LaunchMlaRopeQpe"/>. Cos/sin tables stay F32.
    /// </summary>
    public void LaunchMlaRopeQpeF16(
        nint q, nint cosTab, nint sinTab,
        int seqLen, int numHeads, int qkNopeHeadDim, int qkRopeHeadDim,
        int positionOffset, nint stream)
    {
        if (_mlaRopeQpeF16Func == 0)
            throw new InvalidOperationException("MLA FP16 RoPE-Q-pe helper not available.");

        nint qArg = q, cosArg = cosTab, sinArg = sinTab;
        int slArg = seqLen, nhArg = numHeads;
        int nopeArg = qkNopeHeadDim, ropeArg = qkRopeHeadDim;
        int poArg = positionOffset;
        void** args = stackalloc void*[] {
            &qArg, &cosArg, &sinArg,
            &slArg, &nhArg, &nopeArg, &ropeArg, &poArg
        };
        uint blocks = (uint)(seqLen * numHeads);
        const uint block = 128;
        CudaDriverApi.cuLaunchKernel(_mlaRopeQpeF16Func,
                blocks, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// FP16 sibling of <see cref="LaunchMlaRopeKpe"/>. Cos/sin tables stay F32.
    /// </summary>
    public void LaunchMlaRopeKpeF16(
        nint kPe, nint cosTab, nint sinTab,
        int seqLen, int qkRopeHeadDim, int positionOffset, nint stream)
    {
        if (_mlaRopeKpeF16Func == 0)
            throw new InvalidOperationException("MLA FP16 RoPE-K-pe helper not available.");

        nint kArg = kPe, cosArg = cosTab, sinArg = sinTab;
        int slArg = seqLen, ropeArg = qkRopeHeadDim, poArg = positionOffset;
        void** args = stackalloc void*[] {
            &kArg, &cosArg, &sinArg,
            &slArg, &ropeArg, &poArg
        };
        uint blocks = (uint)seqLen;
        const uint block = 128;
        CudaDriverApi.cuLaunchKernel(_mlaRopeKpeF16Func,
                blocks, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// FP16 sibling of <see cref="LaunchMlaRmsNormF32"/>. FP16 input, FP32 weight,
    /// FP16 output, FP32 reduction. Used by MLA's q_a_layernorm and
    /// kv_a_layernorm on the FP16 path.
    /// </summary>
    public void LaunchMlaRmsNormF16(
        nint inputF16, nint weightF32, nint outputF16,
        int numRows, int dim, float epsilon, nint stream)
    {
        if (_mlaRmsNormF16Func == 0)
            throw new InvalidOperationException("MLA FP16 RMSNorm helper not available.");

        nint inArg = inputF16, wArg = weightF32, outArg = outputF16;
        int dimArg = dim;
        float epsArg = epsilon;
        void** args = stackalloc void*[] {
            &inArg, &wArg, &outArg, &dimArg, &epsArg
        };
        const uint block = 128;
        CudaDriverApi.cuLaunchKernel(_mlaRmsNormF16Func,
                (uint)numRows, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }
    #endregion

    #region MLA Phase B launchers

    /// <summary>
    /// MLA Phase B absorbed attention — F32. One CUDA block per (query_token, head).
    /// Reads the compact latent KV cache (<paramref name="cKv"/>, <paramref name="kPe"/>)
    /// and writes the per-head latent V output (<paramref name="cVOut"/>) of shape
    /// <c>[seqQ, numHeads, kvLoraRank]</c>. Caller is responsible for the W_UK
    /// absorption that produces <paramref name="qAbsorbed"/> and the W_UV
    /// expansion that turns <paramref name="cVOut"/> into the post-attention
    /// per-head output (use <see cref="LaunchMlaQAbsorbUk"/> /
    /// <see cref="LaunchMlaVExpandUv"/> respectively, or cuBLAS GEMM).
    /// </summary>
    /// <param name="qAbsorbed">F32 [seqQ, numHeads, kvLoraRank].</param>
    /// <param name="qPe">F32 [seqQ, numHeads, qkRopeHeadDim] (RoPE-applied Q rope sub-dim).</param>
    /// <param name="cKv">F32 [seqKv, kvLoraRank] (shared latent cache).</param>
    /// <param name="kPe">F32 [seqKv, qkRopeHeadDim] (shared rope-K cache, RoPE-applied).</param>
    /// <param name="cVOut">F32 [seqQ, numHeads, kvLoraRank] (latent attention output).</param>
    /// <param name="seqQ">Number of query tokens this call produces output for.</param>
    /// <param name="seqKv">Total cached length the queries attend over (= cachedLength + seqQ in autoregression).</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="kvLoraRank">Latent KV rank (the compressed dim).</param>
    /// <param name="qkRopeHeadDim">Per-token rope-K dim (must be even).</param>
    /// <param name="positionOffset">Absolute position of query token 0 (causal mask base).</param>
    /// <param name="softmaxScale">Combined softmax scale: <c>(1 / sqrt(qk_head_dim)) * yarn_mscale²</c>.</param>
    /// <param name="stream">CUDA stream.</param>
    public void LaunchAttentionMlaLatent(
        nint qAbsorbed, nint qPe, nint cKv, nint kPe, nint cVOut,
        int seqQ, int seqKv,
        int numHeads, int kvLoraRank, int qkRopeHeadDim,
        int positionOffset, float softmaxScale, nint stream)
    {
        if (_attentionMlaLatentF32Func == 0)
            throw new InvalidOperationException(
                "MLA Phase B kernel not available. Compile native/kernels/attention_mla_latent.cu to PTX.");

        nint qaArg = qAbsorbed, qpeArg = qPe, ckvArg = cKv, kpeArg = kPe, outArg = cVOut;
        int sqArg = seqQ, skvArg = seqKv;
        int nhArg = numHeads, klArg = kvLoraRank, ropeArg = qkRopeHeadDim;
        int poArg = positionOffset;
        float scaleArg = softmaxScale;

        void** args = stackalloc void*[] {
            &qaArg, &qpeArg, &ckvArg, &kpeArg, &outArg,
            &sqArg, &skvArg,
            &nhArg, &klArg, &ropeArg,
            &poArg, &scaleArg
        };

        int numBlocks = seqQ * numHeads;
        // Shared memory layout: q_abs[kvLora] + q_pe[qkRope] + score_tile[128]
        //                       + out_accum[kvLora] + warp_scratch[32]
        const int TileKv = 128;
        uint sharedBytes = (uint)((kvLoraRank + qkRopeHeadDim + TileKv + kvLoraRank + 32) * sizeof(float));
        // Block size 128 (matches __launch_bounds__ in attention_mla_latent.cu).
        const uint MlaBlockSize = 128;

        CudaDriverApi.cuLaunchKernel(_attentionMlaLatentF32Func,
                (uint)numBlocks, 1, 1, MlaBlockSize, 1, 1,
                sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Q absorption: <c>Q_absorbed[h, t] = W_UK[h]^T @ Q_nope[h, t]</c> per
    /// (token, head). One block per (t, h); each block emits the kvLoraRank-wide
    /// absorbed Q vector. Reads <c>kv_b_proj</c> directly (W_UK lives at the
    /// per-head row offset <c>h * (qkNope + vHead)</c>; W_UV is offset by
    /// <c>+ qkNope</c> in the same packed layout). No separate W_UK upload.
    /// </summary>
    public void LaunchMlaQAbsorbUk(
        nint q, nint kvBProj, nint qAbsorbed,
        int seqQ, int numHeads,
        int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim, int kvLoraRank,
        nint stream)
    {
        if (_mlaQAbsorbUkF32Func == 0)
            throw new InvalidOperationException("MLA Phase B helpers not available.");

        nint qArg = q, wArg = kvBProj, outArg = qAbsorbed;
        int sqArg = seqQ, nhArg = numHeads;
        int nopeArg = qkNopeHeadDim, ropeArg = qkRopeHeadDim, vhArg = vHeadDim, klArg = kvLoraRank;
        void** args = stackalloc void*[] {
            &qArg, &wArg, &outArg,
            &sqArg, &nhArg, &nopeArg, &ropeArg, &vhArg, &klArg
        };
        uint blocks = (uint)(seqQ * numHeads);
        const uint block = 128;
        CudaDriverApi.cuLaunchKernel(_mlaQAbsorbUkF32Func,
                blocks, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// V expansion: <c>out[h, t] = W_UV[h] @ c_v_out[h, t]</c> per (token, head).
    /// One block per (t, h); each block emits the vHeadDim-wide expanded output.
    /// Reads <c>kv_b_proj</c> directly (W_UV lives at row offset
    /// <c>h * (qkNope + vHead) + qkNope</c>).
    /// </summary>
    public void LaunchMlaVExpandUv(
        nint cVOut, nint kvBProj, nint attnOut,
        int seqQ, int numHeads,
        int qkNopeHeadDim, int vHeadDim, int kvLoraRank,
        nint stream)
    {
        if (_mlaVExpandUvF32Func == 0)
            throw new InvalidOperationException("MLA Phase B helpers not available.");

        nint inArg = cVOut, wArg = kvBProj, outArg = attnOut;
        int sqArg = seqQ, nhArg = numHeads;
        int nopeArg = qkNopeHeadDim, vhArg = vHeadDim, klArg = kvLoraRank;
        void** args = stackalloc void*[] {
            &inArg, &wArg, &outArg,
            &sqArg, &nhArg, &nopeArg, &vhArg, &klArg
        };
        uint blocks = (uint)(seqQ * numHeads);
        const uint block = 128;
        CudaDriverApi.cuLaunchKernel(_mlaVExpandUvF32Func,
                blocks, 1, 1, block, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    #endregion

    /// <inheritdoc/>
    public void Dispose()
    {
        _rmsnormModule.Dispose();
        _ropeModule.Dispose();
        _swigluModule.Dispose();
        _addModule.Dispose();
        _softmaxModule.Dispose();
        _embeddingModule.Dispose();
        _attentionModule.Dispose();
        _kvCacheUpdateModule.Dispose();
        _biasAddModule.Dispose();
        _perHeadRmsNormModule.Dispose();
        _convertModule.Dispose();
        _dequantModule.Dispose();
        _dequantIQuantsModule?.Dispose();
        _iq2Module?.Dispose();
        _dequantIQ3Module?.Dispose();
        _dequantIQ1Module?.Dispose();
        _dequantBf16Mxfp4Module?.Dispose();
        _quantizedGemvModule.Dispose();
        _fusedAddRmsNormModule.Dispose();
        _rmsnormF32InModule.Dispose();
        _addF32Module.Dispose();
        _embeddingF32OutModule.Dispose();
        _ropeF32Module.Dispose();
        _attentionF32Module.Dispose();
        _swigluF32Module.Dispose();
        _biasAddF32Module.Dispose();
        _perHeadRmsNormF32Module.Dispose();
        _rmsnormF32Module.Dispose();
        _copyRmsNormF32Module?.Dispose();
        _quantizedGemvF32InModule.Dispose();
        _quantizedGemvMmqModule?.Dispose();
        _i2sGemvModule.Dispose();
        _dequantI2sModule.Dispose();
        _pq2_0GemvModule.Dispose();
        _dequantPQ2_0Module.Dispose();
        _pq2_0RepackModule.Dispose();
        _relu2Module.Dispose();
        _relu2F32Module.Dispose();
        _relu2GluRmsNormModule.Dispose();
        _fusedAddRmsNormF32ResModule.Dispose();
        _quantKvModule?.Dispose();
        _turboquantModule?.Dispose();
        _kvWriteModule?.Dispose();
        _fusedRopeKvWriteModule?.Dispose();
        _attentionMlaModule?.Dispose();
        _mlaHelpersModule?.Dispose();
        _moeFfnModule?.Dispose();
        _moeGroupedGemvModule?.Dispose();
        _attentionMlaLatentModule?.Dispose();
        // GDN/conv1d modules: _l2NormHeadsF32Module aliases _gdnScanF32Module
        // (same .ptx file), so dispose only the scan module — disposing both
        // would double-free the underlying CUmodule handle.
        _conv1dCausalF32Module?.Dispose();
        _gdnScanF32Module?.Dispose();
        _mamba2ScanF32Module?.Dispose();
        _groupRmsNormF32Module?.Dispose();
        _elementwiseF32Module?.Dispose();
        _gemma4F32Module?.Dispose();
    }
}
