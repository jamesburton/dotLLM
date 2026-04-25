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
    private readonly nint _quantizedGemvQ4_KMmqFunc;
    private readonly nint _quantizedGemvQ5_KMmqFunc;
    private readonly nint _quantizedGemvQ6_KMmqFunc;
    // MMVQ-large variants — 1 row per CUDA block, 128 threads (4 warps).
    // Tuned for k≥1024 (≥4 super-blocks/row); fall back to MMQ-4-rows for smaller k.
    private readonly nint _quantizedGemvQ4_KMmvqLargeFunc;
    private readonly nint _quantizedGemvQ5_KMmvqLargeFunc;
    private readonly nint _quantizedGemvQ6_KMmvqLargeFunc;

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
            _quantizedGemvQ5_KMmqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q5_k_mmq");
            _quantizedGemvQ6_KMmqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q6_k_mmq");
            // MMVQ-large variants (k≥1024). TryGetFunction so a stale PTX without the
            // new kernels still loads — HasMmvqLarge* will report false and the dispatcher
            // will fall back to the MMQ-4-rows path.
            _quantizedGemvQ4_KMmvqLargeFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q4_k_mmvq_large");
            _quantizedGemvQ5_KMmvqLargeFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q5_k_mmvq_large");
            _quantizedGemvQ6_KMmvqLargeFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q6_k_mmvq_large");
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

    /// <summary>True when the MMQ-style Q4_K GEMV kernel is loaded (PTX present).</summary>
    public bool HasMmqQ4K => _quantizedGemvMmqModule != null && !DisableMmqQ4K;

    /// <summary>True when the MMQ-style Q5_K GEMV kernel is loaded (PTX present).</summary>
    public bool HasMmqQ5K => _quantizedGemvQ5_KMmqFunc != 0 && !DisableMmqQ5K;

    /// <summary>True when the MMQ-style Q6_K GEMV kernel is loaded (PTX present).</summary>
    public bool HasMmqQ6K => _quantizedGemvQ6_KMmqFunc != 0 && !DisableMmqQ6K;

    /// <summary>True when the MMVQ-large Q4_K GEMV kernel (1 row × 128 threads) is loaded
    /// AND the MMVQ-large dispatch is enabled (default: disabled — see remarks).</summary>
    /// <remarks>
    /// The MMVQ-large kernels are functionally correct (covered by
    /// CudaMmqKernelTests.MmvqLargeQ4K_MatchesLegacy_Qwen3_8B_Shapes) but do not currently
    /// achieve the perf target from <c>docs/perf/MLPUP_GEMV_GAP.md</c> on RTX 3060 with
    /// dotLLM's on-the-fly Stage 1 input quantization. The 1-row-per-block × 128-thread
    /// structure runs Stage 1 once per output row (24576x for Qwen3-8B's MlpUp) instead of
    /// once per 4-row tile (6144x). Without llama.cpp's pre-Q8_1-quantize-x kernel pattern,
    /// that 4× redundant Stage 1 work outweighs the dp4a-side win. The kernel is shipped
    /// behind an opt-in env var so the structural work stays in tree for follow-up
    /// (Stage 1 amortization via a separate quantize-x launch — see brief §H4).
    /// </remarks>
    public bool HasMmvqLargeQ4K => _quantizedGemvQ4_KMmvqLargeFunc != 0 && EnableMmvqLargeQ4K;

    /// <summary>True when the MMVQ-large Q5_K GEMV kernel is loaded and routing is enabled.</summary>
    public bool HasMmvqLargeQ5K => _quantizedGemvQ5_KMmvqLargeFunc != 0 && EnableMmvqLargeQ5K;

    /// <summary>True when the MMVQ-large Q6_K GEMV kernel is loaded and routing is enabled.</summary>
    public bool HasMmvqLargeQ6K => _quantizedGemvQ6_KMmvqLargeFunc != 0 && EnableMmvqLargeQ6K;

    /// <summary>Test/benchmark hook to force the legacy Q4_K GEMV kernel even when MMQ is loaded.</summary>
    public static bool DisableMmqQ4K { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMQ_Q4K") == "1";

    /// <summary>Test/benchmark hook to force the legacy Q5_K GEMV kernel even when MMQ is loaded.</summary>
    public static bool DisableMmqQ5K { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMQ_Q5K") == "1";

    /// <summary>Test/benchmark hook to force the legacy Q6_K GEMV kernel even when MMQ is loaded.</summary>
    public static bool DisableMmqQ6K { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMQ_Q6K") == "1";

    /// <summary>Opt-in routing to the MMVQ-large Q4_K kernel (default: off — see <see cref="HasMmvqLargeQ4K"/> remarks).</summary>
    /// <remarks>
    /// To force the MMQ-4-rows path even when the MMVQ-large kernel is enabled (A/B comparison),
    /// set <c>DOTLLM_DISABLE_MMVQ_LARGE_Q4K=1</c> — that takes precedence over the enable knob.
    /// </remarks>
    public static bool EnableMmvqLargeQ4K { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_ENABLE_MMVQ_LARGE_Q4K") == "1"
        && Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMVQ_LARGE_Q4K") != "1";

    /// <summary>Opt-in routing to the MMVQ-large Q5_K kernel (default: off).</summary>
    public static bool EnableMmvqLargeQ5K { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_ENABLE_MMVQ_LARGE_Q5K") == "1"
        && Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMVQ_LARGE_Q5K") != "1";

    /// <summary>Opt-in routing to the MMVQ-large Q6_K kernel (default: off).</summary>
    public static bool EnableMmvqLargeQ6K { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_ENABLE_MMVQ_LARGE_Q6K") == "1"
        && Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMVQ_LARGE_Q6K") != "1";

    /// <summary>True when this MMQ GEMV variant is available for the given quantization type.</summary>
    public bool HasMmq(QuantizationType qt) => qt switch
    {
        QuantizationType.Q4_K => HasMmqQ4K,
        QuantizationType.Q5_K => HasMmqQ5K,
        QuantizationType.Q6_K => HasMmqQ6K,
        _ => false,
    };

    /// <summary>True when the MMVQ-large variant is available for the given quantization type.</summary>
    public bool HasMmvqLarge(QuantizationType qt) => qt switch
    {
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
    /// Supports Q4_K, Q5_K, Q6_K — gate the call with <see cref="HasMmq"/>.
    /// </summary>
    public void LaunchQuantizedGemvMmq(nint quantWeight, QuantizationType qt,
                                         nint x, nint y, int n, int k, nint stream)
    {
        if (_quantizedGemvMmqModule == null)
            throw new InvalidOperationException(
                "MMQ GEMV kernel not available. Compile native/kernels/quantized_gemv_mmq.cu to PTX.");

        // Prefer MMVQ-large for k ≥ threshold when the variant is loaded and not disabled.
        // The DOTLLM_DISABLE_MMVQ_LARGE_<QT> env vars (separate from DOTLLM_DISABLE_MMQ_<QT>)
        // force the MMQ-4-rows path for A/B comparison without bypassing dp4a entirely.
        if (k >= MmvqLargeKThreshold && HasMmvqLarge(qt))
        {
            nint largeFunc = qt switch
            {
                QuantizationType.Q4_K => _quantizedGemvQ4_KMmvqLargeFunc,
                QuantizationType.Q5_K => _quantizedGemvQ5_KMmvqLargeFunc,
                QuantizationType.Q6_K => _quantizedGemvQ6_KMmvqLargeFunc,
                _ => 0,
            };
            if (largeFunc != 0)
            {
                nint wL = quantWeight, xL = x, yL = y;
                int nL = n, kL = k;
                void** argsL = stackalloc void*[] { &wL, &xL, &yL, &nL, &kL };
                // Must mirror MMVQ_LARGE_THREADS in quantized_gemv_mmq.cu.
                const uint MmvqLargeThreads = 128;
                CudaDriverApi.cuLaunchKernel(largeFunc,
                        (uint)n, 1, 1, MmvqLargeThreads, 1, 1,
                        0, stream, (nint)argsL, 0).ThrowOnError();
                return;
            }
        }

        nint func = qt switch
        {
            QuantizationType.Q4_K => _quantizedGemvQ4_KMmqFunc,
            QuantizationType.Q5_K => _quantizedGemvQ5_KMmqFunc,
            QuantizationType.Q6_K => _quantizedGemvQ6_KMmqFunc,
            _ => 0,
        };

        if (func == 0)
            throw new NotSupportedException($"MMQ GEMV not available for {qt}.");

        nint wArg = quantWeight, xArg = x, yArg = y;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] { &wArg, &xArg, &yArg, &nArg, &kArg };

        // Must mirror MMQ_ROWS_PER_BLOCK in quantized_gemv_mmq.cu.
        const int MmqRowsPerBlock = 4;
        uint gridDim = (uint)((n + MmqRowsPerBlock - 1) / MmqRowsPerBlock);

        CudaDriverApi.cuLaunchKernel(func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
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
    }
}
