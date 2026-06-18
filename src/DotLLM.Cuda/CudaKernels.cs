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
    private readonly CudaModule _i2sGemvModule;
    private readonly CudaModule _dequantI2sModule;
    private readonly CudaModule _relu2Module;
    private readonly CudaModule _relu2F32Module;
    private readonly CudaModule _relu2GluRmsNormModule;
    private readonly CudaModule _fusedAddRmsNormF32ResModule;

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
    private readonly nint _i2sGemvF16InFunc;
    private readonly nint _i2sGemv2F16InFunc;
    private readonly nint _i2sGemv3F16InFunc;
    private readonly nint _i2sGemvNormF16InFunc;
    private readonly nint _i2sGemvF32InFunc;
    private readonly nint _i2sGemvA8Func;
    private readonly nint _dequantI2sF16Func;
    private readonly nint _relu2Func;
    private readonly nint _relu2F32Func;
    private readonly nint _relu2GluRmsNormFunc;
    private readonly nint _fusedAddRmsNormF32ResFunc;
    private readonly nint _copyF16ToF32Func;
    private readonly nint _addF32ResF16Func;
    private readonly nint _rmsNormF32InF16WFunc;
    private readonly nint _dequantQ8_0Func;
    private readonly nint _dequantQ4_0Func;
    private readonly nint _dequantQ5_0Func;
    private readonly nint _dequantQ4_KFunc;
    private readonly nint _dequantQ5_KFunc;
    private readonly nint _dequantQ6_KFunc;
    private readonly CudaModule? _quantKvModule;
    private readonly nint _quantKvQ8_0Func;
    private readonly nint _quantKvQ4_0Func;


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
        _i2sGemvModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "i2_s_gemv.ptx"));
        _dequantI2sModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant_i2_s.ptx"));
        _relu2Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "relu2.ptx"));
        _relu2F32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "relu2_f32.ptx"));
        _relu2GluRmsNormModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "relu2_glu_rmsnorm.ptx"));
        _fusedAddRmsNormF32ResModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "fused_add_rmsnorm_f32res.ptx"));

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
        _i2sGemvF16InFunc = _i2sGemvModule.GetFunction("i2_s_gemv_f16in");
        _i2sGemv2F16InFunc = _i2sGemvModule.GetFunction("i2_s_gemv2_f16in");
        _i2sGemv3F16InFunc = _i2sGemvModule.GetFunction("i2_s_gemv3_f16in");
        _i2sGemvNormF16InFunc = _i2sGemvModule.GetFunction("i2_s_gemv_norm_f16in");
        _i2sGemvF32InFunc = _i2sGemvModule.GetFunction("i2_s_gemv_f32in");
        _i2sGemvA8Func = _i2sGemvModule.GetFunction("i2_s_gemv_a8");
        _dequantI2sF16Func = _dequantI2sModule.GetFunction("dequant_i2_s_f16");
        _relu2Func = _relu2Module.GetFunction("relu2_f16");
        _relu2F32Func = _relu2F32Module.GetFunction("relu2_f32");
        _relu2GluRmsNormFunc = _relu2GluRmsNormModule.GetFunction("relu2_glu_rmsnorm_f16");
        _fusedAddRmsNormF32ResFunc = _fusedAddRmsNormF32ResModule.GetFunction("fused_add_rmsnorm_f32res");
        _copyF16ToF32Func = _fusedAddRmsNormF32ResModule.GetFunction("copy_f16_to_f32");
        _addF32ResF16Func = _fusedAddRmsNormF32ResModule.GetFunction("add_f32res_f16");
        _rmsNormF32InF16WFunc = _fusedAddRmsNormF32ResModule.GetFunction("rmsnorm_f32in_f16w");
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
        // v2 warp-per-row: I2sRowsPerBlock output rows per 256-thread block → ceil(n / rows) blocks.
        CudaDriverApi.cuLaunchKernel(_i2sGemvF16InFunc,
                (uint)((n + I2sRowsPerBlock - 1) / I2sRowsPerBlock), 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
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
        CudaDriverApi.cuLaunchKernel(_i2sGemv2F16InFunc,
                grid, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
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
        CudaDriverApi.cuLaunchKernel(_i2sGemv3F16InFunc,
                grid, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
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
        CudaDriverApi.cuLaunchKernel(_i2sGemvNormF16InFunc,
                grid, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>I2_S ternary GEMV with FP32 activations/output. Exact-match twin for CPU-vs-GPU tests.</summary>
    public void LaunchI2_SGemvF32In(nint quantWeight, nint xF32, nint yF32, int n, int k, nint stream)
    {
        nint wArg = quantWeight, xArg = xF32, yArg = yF32;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] {&wArg, &xArg, &yArg, &nArg, &kArg};
        CudaDriverApi.cuLaunchKernel(_i2sGemvF32InFunc,
                (uint)((n + I2sRowsPerBlock - 1) / I2sRowsPerBlock), 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
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
        uint gridDim = (uint)Math.Min((totalBlocks + BlockSize - 1) / BlockSize, MaxDequantGridSize);
        if (gridDim == 0) gridDim = 1;

        CudaDriverApi.cuLaunchKernel(_dequantI2sF16Func,
                gridDim, 1, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
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
        _i2sGemvModule.Dispose();
        _dequantI2sModule.Dispose();
        _relu2Module.Dispose();
        _relu2F32Module.Dispose();
        _relu2GluRmsNormModule.Dispose();
        _fusedAddRmsNormF32ResModule.Dispose();
        _quantKvModule?.Dispose();
    }
}
