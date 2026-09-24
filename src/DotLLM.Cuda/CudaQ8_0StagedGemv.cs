using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Optional shared-memory-staged Q8_0 GEMV (<c>native/kernels/q8_0_gemv_f32in_staged.cu</c>,
/// issue #482): a bit-identical, coalesced twin of <see cref="CudaKernels.LaunchQuantizedGemvF32In"/>.
/// </summary>
/// <remarks>
/// <para>
/// The original kernel's global loads are uncoalesced (each warp-wide activation load touches 32
/// cache lines), which makes it the dominant cost of a Q8_0-weighted MTP draft head. This twin keeps
/// the per-thread block ownership, the fma order and the reduction tree, so its output is identical
/// bit for bit — see the kernel's header comment for the argument.
/// </para>
/// <para>
/// Owned separately from <see cref="CudaKernels"/> so it stays optional: <see cref="TryLoad"/>
/// returns <see langword="null"/> when the PTX has not been generated yet (or does not JIT), and
/// callers fall back to the original kernel. <c>DOTLLM_CUDA_Q8_STAGED_GEMV=0</c> forces the fallback.
/// </para>
/// </remarks>
internal sealed class CudaQ8_0StagedGemv : IDisposable
{
    /// <summary>PTX file name, resolved in the same directory as the rest of the kernels.</summary>
    public const string PtxFileName = "q8_0_gemv_f32in_staged.ptx";

    /// <summary>The <c>extern "C"</c> kernel name.</summary>
    public const string KernelName = "q8_0_gemv_f32in_staged";

    /// <summary>Output rows per thread block (the kernel's <c>Q8S_ROWS</c>).</summary>
    public const int RowsPerBlock = 2;

    /// <summary>Threads per block: <see cref="RowsPerBlock"/> groups of 256 (the kernel's <c>Q8S_GROUP</c>).</summary>
    public const int ThreadsPerBlock = RowsPerBlock * 256;

    /// <summary><see langword="true"/> when <c>DOTLLM_CUDA_Q8_STAGED_GEMV=0</c> disables the staged kernel.</summary>
    public static bool DisabledByEnv { get; } =
        Environment.GetEnvironmentVariable("DOTLLM_CUDA_Q8_STAGED_GEMV") == "0";

    private CudaModule? _module;
    private readonly nint _func;

    private CudaQ8_0StagedGemv(CudaModule module, nint func)
    {
        _module = module;
        _func = func;
    }

    /// <summary>
    /// Loads the kernel from <paramref name="ptxDir"/>, or returns <see langword="null"/> when it is
    /// disabled, the PTX is absent, or the module/function does not load. The CUDA context must be current.
    /// </summary>
    public static CudaQ8_0StagedGemv? TryLoad(string ptxDir)
    {
        if (DisabledByEnv)
            return null;

        string path = Path.Combine(ptxDir, PtxFileName);
        if (!File.Exists(path))
            return null;

        CudaModule? module = null;
        try
        {
            module = CudaModule.LoadFromFile(path);
            nint func = module.TryGetFunction(KernelName);
            if (func == 0)
            {
                module.Dispose();
                return null;
            }
            return new CudaQ8_0StagedGemv(module, func);
        }
        catch (Exception ex) when (ex is CudaException or IOException)
        {
            module?.Dispose();
            return null;
        }
    }

    /// <summary>
    /// <c>y_f32[n] = W_q8_0[n,k] @ x_f32[k]</c> — same contract (and bits) as
    /// <see cref="CudaKernels.LaunchQuantizedGemvF32In"/>. <paramref name="k"/> must be a multiple of 32.
    /// </summary>
    public unsafe void Launch(nint quantWeight, nint xF32, nint yF32, int n, int k, nint stream)
    {
        ObjectDisposedException.ThrowIf(_module is null, this);
        nint wArg = quantWeight, xArg = xF32, yArg = yF32;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] { &wArg, &xArg, &yArg, &nArg, &kArg };
        CudaDriverApi.cuLaunchKernel(_func,
                (uint)((n + RowsPerBlock - 1) / RowsPerBlock), 1, 1, ThreadsPerBlock, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _module?.Dispose();
        _module = null;
    }
}
