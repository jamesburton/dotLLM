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
    private readonly CudaModule _biasAddModule;
    private readonly CudaModule _perHeadRmsNormModule;
    private readonly CudaModule _convertModule;
    private readonly CudaModule _dequantModule;
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
    private readonly CudaModule _quantizedGemvF32InModule;
    private readonly CudaModule? _quantizedGemvMmqModule;
    private readonly nint _quantizedGemvQ2_KMmqFunc;
    private readonly nint _quantizedGemvQ4_KMmqFunc;
    private readonly nint _quantizedGemvQ5_KMmqFunc;
    private readonly nint _quantizedGemvQ6_KMmqFunc;
    // MMVQ-large variants — 1 row per CUDA block, 128 threads (4 warps).
    // Tuned for k≥1024 (≥4 super-blocks/row); fall back to MMQ-4-rows for smaller k.
    private readonly nint _quantizedGemvQ2_KMmvqLargeFunc;
    private readonly nint _quantizedGemvQ4_KMmvqLargeFunc;
    private readonly nint _quantizedGemvQ5_KMmvqLargeFunc;
    private readonly nint _quantizedGemvQ6_KMmvqLargeFunc;
    // Pre-Q8_1 variants. Read INT8/dx/sx2 from device-resident scratch (populated
    // once per fused projection group via _quantizeXToQ8_1Func) instead of
    // re-quantizing the input inside every CUDA block. Eliminates the redundant
    // Stage 1 work that scales with output dim n (n× for MMVQ-large, n/4× for MMQ-4-rows).
    private readonly CudaModule? _quantizeXModule;
    private readonly nint _quantizeXToQ8_1Func;
    private readonly nint _quantizedGemvQ2_KMmqPreqFunc;
    private readonly nint _quantizedGemvQ4_KMmqPreqFunc;
    private readonly nint _quantizedGemvQ5_KMmqPreqFunc;
    private readonly nint _quantizedGemvQ6_KMmqPreqFunc;
    private readonly nint _quantizedGemvQ2_KMmvqLargePreqFunc;
    private readonly nint _quantizedGemvQ4_KMmvqLargePreqFunc;
    private readonly nint _quantizedGemvQ5_KMmvqLargePreqFunc;
    private readonly nint _quantizedGemvQ6_KMmvqLargePreqFunc;
    /// <summary>
    /// Device's maximum opt-in dynamic shared-memory bytes per block (queried once at
    /// kernel-load via CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN). The
    /// on-the-fly MMQ kernels are opted into this cap via cuFuncSetAttribute so
    /// arbitrarily large k (up to ~k=53000 on a 100 KB cap) launches succeed without
    /// recompiling. 0 means we couldn't query — fall back to the default 48 KB cap.
    /// </summary>
    private readonly int _maxDynamicSharedBytesOptIn;

    private readonly nint _rmsnormFunc;
    private readonly nint _rmsnormF32Func;
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
    private readonly nint _swigluF32Func;
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
    private readonly nint _dequantQ8_0Func;
    private readonly nint _dequantQ4_0Func;
    private readonly nint _dequantQ4_1Func;
    private readonly nint _dequantQ5_0Func;
    private readonly nint _dequantQ5_1Func;
    private readonly nint _dequantQ2_KFunc;
    private readonly nint _dequantQ3_KFunc;
    private readonly nint _dequantQ4_KFunc;
    private readonly nint _dequantQ5_KFunc;
    private readonly nint _dequantQ6_KFunc;
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
        _quantizedGemvF32InModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "quantized_gemv_f32in.ptx"));

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
            // MMVQ-large variants (k≥1024). TryGetFunction so a stale PTX without the
            // new kernels still loads — HasMmvqLarge* will report false and the dispatcher
            // will fall back to the MMQ-4-rows path.
            _quantizedGemvQ2_KMmvqLargeFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q2_k_mmvq_large");
            _quantizedGemvQ4_KMmvqLargeFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q4_k_mmvq_large");
            _quantizedGemvQ5_KMmvqLargeFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q5_k_mmvq_large");
            _quantizedGemvQ6_KMmvqLargeFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q6_k_mmvq_large");
            // Pre-quantized GEMV variants (consume scratch from quantize_x.ptx kernel).
            // TryGetFunction so a stale PTX without the new symbols still loads.
            _quantizedGemvQ2_KMmqPreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q2_k_mmq_preq");
            _quantizedGemvQ4_KMmqPreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q4_k_mmq_preq");
            _quantizedGemvQ5_KMmqPreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q5_k_mmq_preq");
            _quantizedGemvQ6_KMmqPreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q6_k_mmq_preq");
            _quantizedGemvQ2_KMmvqLargePreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q2_k_mmvq_large_preq");
            _quantizedGemvQ4_KMmvqLargePreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q4_k_mmvq_large_preq");
            _quantizedGemvQ5_KMmvqLargePreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q5_k_mmvq_large_preq");
            _quantizedGemvQ6_KMmvqLargePreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q6_k_mmvq_large_preq");

            // The on-the-fly MMQ kernels size their per-chunk Stage 1 scratch (s_xq/s_dx/s_sx[2])
            // dynamically from `k`. For k up to ~12 KiB-shmem-worth (Qwen3-8B intermediate=12288)
            // we fit under the 48 KB default cap, but Llama-70B-class intermediate=14336 lands at
            // ~15.7 KB and Llama-405B-class intermediate=53248 lands at ~58 KB — past 48 KB. Opt
            // each on-the-fly variant into the device's full optin cap (typically 100+ KB on
            // Ampere/Ada/Hopper) so any in-budget k launches without recompiling.
            int devForOptIn;
            if (CudaDriverApi.cuCtxGetDevice(out devForOptIn) == 0
                && CudaDriverApi.cuDeviceGetAttribute(out int optIn,
                    CudaDriverApi.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, devForOptIn) == 0
                && optIn > 0)
            {
                _maxDynamicSharedBytesOptIn = optIn;
                SetMaxDynamicSharedBytes(_quantizedGemvQ4_KMmqFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ2_KMmqFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ5_KMmqFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ6_KMmqFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ2_KMmvqLargeFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ4_KMmvqLargeFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ5_KMmvqLargeFunc, optIn);
                SetMaxDynamicSharedBytes(_quantizedGemvQ6_KMmvqLargeFunc, optIn);
            }
        }

        // Pre-Q8_1 input quantization kernel (optional — PTX may be missing on stale builds).
        string quantizeXPath = Path.Combine(ptxDir, "quantize_x.ptx");
        if (File.Exists(quantizeXPath))
        {
            _quantizeXModule = CudaModule.LoadFromFile(quantizeXPath);
            _quantizeXToQ8_1Func = _quantizeXModule.GetFunction("quantize_x_to_q8_1");
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
        _swigluF32Func = _swigluF32Module.GetFunction("swiglu_f32");
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
        _dequantQ6_KFunc = _dequantModule.GetFunction("dequant_q6_k_f16");

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
        }
    }

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

    /// <summary>Disable the Phase-B grouped-GEMV path. Forces the per-expert
    /// <see cref="LaunchQuantizedGemv"/> fallback in <see cref="CudaMoeFfn"/>.</summary>
    public static bool DisableMoeGroupedGemv { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MOE_GROUPED_GEMV") == "1";

    /// <summary>True when a grouped-GEMV kernel is available for the given quant type.</summary>
    public bool HasMoeGroupedGemv(QuantizationType qt) => qt switch
    {
        QuantizationType.Q2_K => HasMoeGroupedGemvQ2K,
        QuantizationType.Q4_K => HasMoeGroupedGemvQ4K,
        QuantizationType.Q5_K => HasMoeGroupedGemvQ5K,
        QuantizationType.Q6_K => HasMoeGroupedGemvQ6K,
        QuantizationType.Q8_0 => HasMoeGroupedGemvQ8_0,
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

    /// <summary>FP32 RoPE: in-place rotation on FP32 Q and K.</summary>
    public void LaunchRoPEF32(nint q, nint k, nint positions,
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
            or QuantizationType.Q2_K;

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
            or QuantizationType.Q8_0 => 32,
        QuantizationType.Q3_K or QuantizationType.Q4_K
            or QuantizationType.Q5_K or QuantizationType.Q6_K
            or QuantizationType.Q2_K => 256,
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
    public bool HasMmqQ2K => _quantizedGemvQ2_KMmqFunc != 0 && !DisableMmqQ2K;

    /// <summary>True when the MMQ-style Q4_K GEMV kernel is loaded (PTX present).</summary>
    public bool HasMmqQ4K => _quantizedGemvMmqModule != null && !DisableMmqQ4K;

    /// <summary>True when the MMQ-style Q5_K GEMV kernel is loaded (PTX present).</summary>
    public bool HasMmqQ5K => _quantizedGemvQ5_KMmqFunc != 0 && !DisableMmqQ5K;

    /// <summary>True when the MMQ-style Q6_K GEMV kernel is loaded (PTX present).</summary>
    public bool HasMmqQ6K => _quantizedGemvQ6_KMmqFunc != 0 && !DisableMmqQ6K;

    /// <summary>True when the MMVQ-large Q2_K GEMV kernel is loaded and not disabled.</summary>
    public bool HasMmvqLargeQ2K => _quantizedGemvQ2_KMmvqLargeFunc != 0 && !DisableMmvqLargeQ2K;

    /// <summary>True when the MMVQ-large Q4_K GEMV kernel (1 row × 128 threads) is loaded
    /// AND not disabled. Default ON — pre-Q8_1 input quantization (<see cref="HasPreQ8_1"/>)
    /// removes the redundant Stage 1 cost that previously made this kernel regress
    /// (<c>docs/perf/MLPUP_GEMV_GAP.md</c> §H4). Per-quant-type override:
    /// <c>DOTLLM_DISABLE_MMVQ_LARGE_Q4K=1</c>.</summary>
    public bool HasMmvqLargeQ4K => _quantizedGemvQ4_KMmvqLargeFunc != 0 && !DisableMmvqLargeQ4K;

    /// <summary>True when the MMVQ-large Q5_K GEMV kernel is loaded and not disabled.</summary>
    public bool HasMmvqLargeQ5K => _quantizedGemvQ5_KMmvqLargeFunc != 0 && !DisableMmvqLargeQ5K;

    /// <summary>True when the MMVQ-large Q6_K GEMV kernel is loaded and not disabled.</summary>
    public bool HasMmvqLargeQ6K => _quantizedGemvQ6_KMmvqLargeFunc != 0 && !DisableMmvqLargeQ6K;

    /// <summary>True when the pre-Q8_1 input-quantization kernel is loaded and not disabled.
    /// When this is on (default) and a scratch buffer is provided to the MMQ GEMV launcher,
    /// Stage 1 runs once via <see cref="LaunchQuantizeXToQ8_1"/> and the GEMV uses the
    /// <c>_preq</c> kernel variants — eliminating the per-block redundant input quant.
    /// Override: <c>DOTLLM_DISABLE_PREQ8_1=1</c>.</summary>
    public bool HasPreQ8_1 => _quantizeXToQ8_1Func != 0 && !DisablePreQ8_1;

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

    /// <summary>Disable pre-Q8_1 input quantization (falls back to on-the-fly Stage 1 in every GEMV).</summary>
    /// <remarks>Useful for A/B comparison or when the model's max k makes the pre-quant scratch
    /// buffer awkward to size. Default off — pre-Q8_1 is the recommended path.</remarks>
    public static bool DisablePreQ8_1 { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_PREQ8_1") == "1";

    /// <summary>True when this MMQ GEMV variant is available for the given quantization type.</summary>
    public bool HasMmq(QuantizationType qt) => qt switch
    {
        QuantizationType.Q2_K => HasMmqQ2K,
        QuantizationType.Q4_K => HasMmqQ4K,
        QuantizationType.Q5_K => HasMmqQ5K,
        QuantizationType.Q6_K => HasMmqQ6K,
        _ => false,
    };

    /// <summary>True when the MMVQ-large variant is available for the given quantization type.</summary>
    public bool HasMmvqLarge(QuantizationType qt) => qt switch
    {
        QuantizationType.Q2_K => HasMmvqLargeQ2K,
        QuantizationType.Q4_K => HasMmvqLargeQ4K,
        QuantizationType.Q5_K => HasMmvqLargeQ5K,
        QuantizationType.Q6_K => HasMmvqLargeQ6K,
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
    /// Supports Q2_K, Q4_K, Q5_K, Q6_K — gate the call with <see cref="HasMmq"/>. All four
    /// quant types have full MMQ + MMVQ-large coverage in both on-the-fly and pre-Q8_1 modes.
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

        // Prefer MMVQ-large for k ≥ threshold when the variant is loaded and not disabled.
        // The DOTLLM_DISABLE_MMVQ_LARGE_<QT> env vars (separate from DOTLLM_DISABLE_MMQ_<QT>)
        // force the MMQ-4-rows path for A/B comparison without bypassing dp4a entirely.
        if (k >= MmvqLargeKThreshold && HasMmvqLarge(qt))
        {
            nint largeFunc = usePreq
                ? qt switch
                {
                    QuantizationType.Q2_K => _quantizedGemvQ2_KMmvqLargePreqFunc,
                    QuantizationType.Q4_K => _quantizedGemvQ4_KMmvqLargePreqFunc,
                    QuantizationType.Q5_K => _quantizedGemvQ5_KMmvqLargePreqFunc,
                    QuantizationType.Q6_K => _quantizedGemvQ6_KMmvqLargePreqFunc,
                    _ => 0,
                }
                : qt switch
                {
                    QuantizationType.Q2_K => _quantizedGemvQ2_KMmvqLargeFunc,
                    QuantizationType.Q4_K => _quantizedGemvQ4_KMmvqLargeFunc,
                    QuantizationType.Q5_K => _quantizedGemvQ5_KMmvqLargeFunc,
                    QuantizationType.Q6_K => _quantizedGemvQ6_KMmvqLargeFunc,
                    _ => 0,
                };
            if (largeFunc != 0)
            {
                // Must mirror MMVQ_LARGE_THREADS in quantized_gemv_mmq.cu.
                const uint MmvqLargeThreads = 128;
                if (usePreq)
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
                _ => 0,
            }
            : qt switch
            {
                QuantizationType.Q2_K => _quantizedGemvQ2_KMmqFunc,
                QuantizationType.Q4_K => _quantizedGemvQ4_KMmqFunc,
                QuantizationType.Q5_K => _quantizedGemvQ5_KMmqFunc,
                QuantizationType.Q6_K => _quantizedGemvQ6_KMmqFunc,
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
    /// Pre-launch check that the requested dynamic shmem fits the device budget. Failing
    /// fast here gives a much clearer error than CUDA's generic CUDA_ERROR_INVALID_VALUE
    /// when sharedMemBytes exceeds the opt-in cap. Skipped if we couldn't query the cap.
    /// </summary>
    private void CheckDynamicSharedBudget(uint dynShmem, QuantizationType qt, int k)
    {
        if (_maxDynamicSharedBytesOptIn <= 0) return;
        if (dynShmem <= (uint)_maxDynamicSharedBytesOptIn) return;
        throw new InvalidOperationException(
            $"MMQ GEMV {qt} k={k} requires {dynShmem} bytes of dynamic shared memory, " +
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
                // Already FP16, just copy
                CudaDriverApi.cuMemcpyDtoD_v2(dst, src, (nuint)(totalElements * 2)).ThrowOnError();
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
        _biasAddModule.Dispose();
        _perHeadRmsNormModule.Dispose();
        _convertModule.Dispose();
        _dequantModule.Dispose();
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
        _quantizedGemvF32InModule.Dispose();
        _quantizedGemvMmqModule?.Dispose();
        _quantKvModule?.Dispose();
        _kvWriteModule?.Dispose();
        _fusedRopeKvWriteModule?.Dispose();
        _attentionMlaModule?.Dispose();
        _mlaHelpersModule?.Dispose();
        _moeFfnModule?.Dispose();
        _moeGroupedGemvModule?.Dispose();
        _attentionMlaLatentModule?.Dispose();
    }
}
