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
    private readonly nint _attentionFunc;
    private readonly nint _biasAddFunc;
    private readonly nint _perHeadRmsNormFunc;
    private readonly nint _convertF16ToF32Func;
    private readonly nint _convertF32ToF16Func;
    private readonly nint _quantizedGemvQ8_0Func;
    private readonly nint _quantizedGemvQ4_KFunc;
    private readonly nint _quantizedGemvQ5_0Func;
    private readonly nint _quantizedGemvQ5_KFunc;
    private readonly nint _quantizedGemvQ6_KFunc;
    private readonly nint _dequantQ8_0Func;
    private readonly nint _dequantQ4_0Func;
    private readonly nint _dequantQ5_0Func;
    private readonly nint _dequantQ4_KFunc;
    private readonly nint _dequantQ5_KFunc;
    private readonly nint _dequantQ6_KFunc;
    private readonly CudaModule? _quantKvModule;
    private readonly nint _quantKvQ8_0Func;
    private readonly nint _quantKvQ4_0Func;

    // ── MLA (Multi-head Latent Attention) Phase A naive forward kernel ──
    // F32 throughout to match the CPU MlaAttention.Execute oracle byte-for-byte
    // algorithmically. Optional — PTX may not be present on stale builds.
    private readonly CudaModule? _attentionMlaModule;
    private readonly nint _attentionMlaF32Func;

    // MLA forward-path helpers: per-head split of kv_b expansion, RoPE on
    // the rope sub-dim of Q (per head) and on the shared K_pe, and a
    // (numRows, dim) F32 RMSNorm used by q_a_layernorm / kv_a_layernorm.
    private readonly CudaModule? _mlaHelpersModule;
    private readonly nint _mlaSplitKvBF32Func;
    private readonly nint _mlaRopeQpeF32Func;
    private readonly nint _mlaRopeKpeF32Func;
    private readonly nint _mlaRmsNormF32Func;

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
        _attentionFunc = _attentionModule.GetFunction("attention_f16");
        _biasAddFunc = _biasAddModule.GetFunction("bias_add_f16");
        _perHeadRmsNormFunc = _perHeadRmsNormModule.GetFunction("per_head_rmsnorm_f16");
        _convertF16ToF32Func = _convertModule.GetFunction("convert_f16_to_f32");
        _convertF32ToF16Func = _convertModule.GetFunction("convert_f32_to_f16");
        _quantizedGemvQ8_0Func = _quantizedGemvModule.GetFunction("quantized_gemv_q8_0");
        _quantizedGemvQ4_KFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q4_k");
        _quantizedGemvQ5_0Func = _quantizedGemvModule.GetFunction("quantized_gemv_q5_0");
        _quantizedGemvQ5_KFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q5_k");
        _quantizedGemvQ6_KFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q6_k");
        _dequantQ8_0Func = _dequantModule.GetFunction("dequant_q8_0_f16");
        _dequantQ4_0Func = _dequantModule.GetFunction("dequant_q4_0_f16");
        _dequantQ5_0Func = _dequantModule.GetFunction("dequant_q5_0_f16");
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
        }

        // MLA Phase A naive forward kernel (F32). Optional — only required when
        // the model is DeepSeek-V2 / V3 with MLA attention.
        string attentionMlaPath = Path.Combine(ptxDir, "attention_mla.ptx");
        if (File.Exists(attentionMlaPath))
        {
            _attentionMlaModule = CudaModule.LoadFromFile(attentionMlaPath);
            _attentionMlaF32Func = _attentionMlaModule.GetFunction("attention_mla_f32");
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
    }

    /// <summary>True when the MLA Phase A attention kernel is available on this kernel module.</summary>
    public bool HasMlaAttentionKernel => _attentionMlaF32Func != 0;

    /// <summary>True when all MLA forward-path helper kernels (split, RoPE, RMSNorm) are available.</summary>
    public bool HasMlaHelpers =>
        _mlaSplitKvBF32Func != 0 && _mlaRopeQpeF32Func != 0
        && _mlaRopeKpeF32Func != 0 && _mlaRmsNormF32Func != 0;

    /// <summary>
    /// True when all MoE FFN helper kernels (softmax-topk, renorm, zero, axpy
    /// variants, sigmoid logit, token gather) are available. Required by
    /// <c>CudaMoeFfn.Forward</c>.
    /// </summary>
    public bool HasMoeKernels =>
        _moeSoftmaxTopkF32Func != 0 && _moeRenormTopkF32Func != 0
        && _moeZeroF32Func != 0 && _moeAxpyScaledRowF32Func != 0
        && _moeAxpyUnweightedF32Func != 0 && _moeAxpyScaledPerTokenF32Func != 0
        && _moeSigmoidLogitF32Func != 0 && _moeGatherTokenRowsF32Func != 0;

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
    public void LaunchFusedAddRmsNorm(nint residual, nint x, nint weight, nint output,
                                        int hiddenSize, float eps, int rows, nint stream)
    {
        nint resArg = residual, xArg = x, wArg = weight, outArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;

        void** args = stackalloc void*[] {&resArg, &xArg, &wArg, &outArg, &nArg, &epsArg};
        CudaDriverApi.cuLaunchKernel(_fusedAddRmsNormFunc,
                (uint)rows, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
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
            _ => throw new NotSupportedException($"Embedding type {embedDtype} not supported on GPU.")
        };

        void** args = stackalloc void*[] {&tableArg, &idsArg, &outArg, &slArg, &hsArg};

        CudaDriverApi.cuLaunchKernel(func,
                (uint)seqLen, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

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
            or QuantizationType.Q5_K or QuantizationType.Q6_K;

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
        _quantKvModule?.Dispose();
        _attentionMlaModule?.Dispose();
        _mlaHelpersModule?.Dispose();
        _moeFfnModule?.Dispose();
    }
}
