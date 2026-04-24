using DotLLM.Hip.Interop;

namespace DotLLM.Hip.Kernels;

/// <summary>
/// Wraps the <c>rmsnorm_f32</c> / <c>rmsnorm_f16</c> kernels from <c>rmsnorm.co</c>.
/// One block per row, warp (wavefront) shuffle reduction — identical math to the
/// CUDA reference kernel.
/// </summary>
public sealed unsafe class RmsNormKernel : IDisposable
{
    private const int BlockSize = 256;

    private readonly HipModule _module;
    private readonly nint _fp32Func;
    private readonly nint _fp16Func;

    /// <summary>
    /// Loads <c>rmsnorm.co</c> from the given path and resolves the kernel entry points.
    /// </summary>
    public RmsNormKernel(string coPath)
    {
        _module = HipModule.LoadFromFile(coPath);
        _fp32Func = _module.GetFunction("rmsnorm_f32");
        _fp16Func = _module.GetFunction("rmsnorm_f16");
    }

    /// <summary>
    /// Launches the FP32 RMS-norm kernel.
    /// <c>output[row, i] = input[row, i] * rsqrt(mean(input[row,:]^2) + eps) * weight[i]</c>.
    /// </summary>
    /// <param name="input">FP32 input device pointer of shape (rows, n).</param>
    /// <param name="weight">FP32 weight device pointer of shape (n).</param>
    /// <param name="output">FP32 output device pointer of shape (rows, n).</param>
    /// <param name="n">Hidden size.</param>
    /// <param name="eps">Normalization epsilon.</param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="stream">Stream handle (0 for default stream).</param>
    public void Launch(nint input, nint weight, nint output,
                       int n, float eps, uint rows, nint stream)
    {
        nint inputArg = input, weightArg = weight, outputArg = output;
        int nArg = n;
        float epsArg = eps;

        void** args = stackalloc void*[] { &inputArg, &weightArg, &outputArg, &nArg, &epsArg };
        HipDriverApi.hipModuleLaunchKernel(_fp32Func,
            rows, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Launches the FP16 RMS-norm kernel. Same shape conventions as <see cref="Launch"/>,
    /// but tensors are FP16.
    /// </summary>
    public void LaunchF16(nint input, nint weight, nint output,
                          int n, float eps, uint rows, nint stream)
    {
        nint inputArg = input, weightArg = weight, outputArg = output;
        int nArg = n;
        float epsArg = eps;

        void** args = stackalloc void*[] { &inputArg, &weightArg, &outputArg, &nArg, &epsArg };
        HipDriverApi.hipModuleLaunchKernel(_fp16Func,
            rows, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose() => _module.Dispose();
}
