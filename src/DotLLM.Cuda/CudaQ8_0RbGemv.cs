using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Optional register-blocked Q8_0 GEMV (<c>native/kernels/q8_0_gemv_f32in_rb.cu</c>, issue #486):
/// a bit-identical twin of <see cref="CudaKernels.LaunchQuantizedGemvF32In"/> and
/// <see cref="CudaQ8_0StagedGemv"/>, plus a multi-column form that reads the weights once for up
/// to <see cref="MaxColumns"/> input rows.
/// </summary>
/// <remarks>
/// <para>
/// Same per-thread block ownership, fma order and reduction tree as the original kernel — only the
/// data flow differs (four rows per thread block sharing each staged activation block, warp-private
/// staging with no block-wide barrier until the reduction). See the kernel's header comment for the
/// bit-identity argument. Each column of <see cref="LaunchMulti"/> equals <see cref="Launch"/> on
/// that column alone, bit for bit.
/// </para>
/// <para>
/// Optional like the staged kernel: <see cref="TryLoad"/> returns <see langword="null"/> when the PTX
/// is absent or does not JIT, and callers fall back. <c>DOTLLM_CUDA_Q8_RB_GEMV=0</c> forces the
/// fallback (to the staged kernel, then the original).
/// </para>
/// </remarks>
internal sealed class CudaQ8_0RbGemv : IDisposable
{
    /// <summary>PTX file name, resolved in the same directory as the rest of the kernels.</summary>
    public const string PtxFileName = "q8_0_gemv_f32in_rb.ptx";

    /// <summary>Single-column kernel name.</summary>
    public const string KernelName = "q8_0_gemv_f32in_rb";

    /// <summary>Multi-column kernel name.</summary>
    public const string MultiKernelName = "q8_0_gemv_f32in_rb_multi";

    /// <summary>Output rows per thread block (the kernel's <c>Q8R_ROWS</c>).</summary>
    public const int RowsPerBlock = 4;

    /// <summary>Most input columns one <see cref="LaunchMulti"/> kernel launch handles (<c>Q8R_MAX_COLS</c>).</summary>
    public const int MaxColumns = 8;

    /// <summary>Env switch: <c>DOTLLM_CUDA_Q8_RB_GEMV=0</c> disables this kernel.</summary>
    public const string DisableEnvVar = "DOTLLM_CUDA_Q8_RB_GEMV";

    /// <summary><see langword="true"/> when <c>DOTLLM_CUDA_Q8_RB_GEMV=0</c> disables the kernel.</summary>
    public static bool DisabledByEnv { get; } = Environment.GetEnvironmentVariable(DisableEnvVar) == "0";

    private CudaModule? _module;
    private readonly nint _func;
    private readonly nint _multiFunc;

    private CudaQ8_0RbGemv(CudaModule module, nint func, nint multiFunc)
    {
        _module = module;
        _func = func;
        _multiFunc = multiFunc;
    }

    /// <summary>Whether the multi-column entry point loaded (it ships in the same PTX).</summary>
    public bool HasMulti => _multiFunc != 0;

    /// <summary>
    /// Loads the kernel from <paramref name="ptxDir"/>, or returns <see langword="null"/> when it is
    /// disabled, the PTX is absent, or the module/function does not load. The CUDA context must be current.
    /// </summary>
    public static CudaQ8_0RbGemv? TryLoad(string ptxDir)
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
            return new CudaQ8_0RbGemv(module, func, module.TryGetFunction(MultiKernelName));
        }
        catch (Exception ex) when (ex is CudaException or IOException)
        {
            module?.Dispose();
            return null;
        }
    }

    /// <summary>
    /// Whether the kernel accepts these operands: <paramref name="k"/> a positive multiple of 32, the
    /// activation pointer 16-byte aligned and its column stride <paramref name="ldx"/> (floats) a
    /// multiple of 4. The weight pointer needs no alignment.
    /// </summary>
    public static bool Accepts(nint x, int k, int ldx) =>
        k > 0 && (k & 31) == 0 && (x & 15) == 0 && (ldx & 3) == 0;

    /// <summary>Threads per block: one warp per 32 Q8_0 blocks of a row, at most 8 warps.</summary>
    private static uint BlockDim(int k) => (uint)(32 * Math.Min(8, ((k >> 5) + 31) >> 5));

    /// <summary>
    /// <c>y_f32[n] = W_q8_0[n,k] @ x_f32[k]</c> — same contract and bits as
    /// <see cref="CudaKernels.LaunchQuantizedGemvF32In"/>. Requires <see cref="Accepts"/>(x, k, k).
    /// </summary>
    public unsafe void Launch(nint quantWeight, nint xF32, nint yF32, int n, int k, nint stream)
    {
        ObjectDisposedException.ThrowIf(_module is null, this);
        nint wArg = quantWeight, xArg = xF32, yArg = yF32;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] { &wArg, &xArg, &yArg, &nArg, &kArg };
        CudaDriverApi.cuLaunchKernel(_func,
                (uint)((n + RowsPerBlock - 1) / RowsPerBlock), 1, 1, BlockDim(k), 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// <c>y[c] = W_q8_0[n,k] @ x[c]</c> for <c>c</c> in <c>[0, ncols)</c>: column <c>c</c> of the input
    /// starts at <c>xF32 + c * ldx</c> floats and its output at <c>yF32 + c * ldy</c> floats. Column
    /// groups of <see cref="MaxColumns"/> are launched in turn. Each output column is bit-identical to
    /// <see cref="Launch"/> on that input column. Requires <see cref="HasMulti"/> and
    /// <see cref="Accepts"/>(x, k, ldx).
    /// </summary>
    public unsafe void LaunchMulti(nint quantWeight, nint xF32, int ldx, nint yF32, int ldy,
                                   int n, int k, int ncols, nint stream)
    {
        ObjectDisposedException.ThrowIf(_module is null, this);
        if (_multiFunc == 0)
            throw new InvalidOperationException($"{MultiKernelName} not present in {PtxFileName}.");
        uint grid = (uint)((n + RowsPerBlock - 1) / RowsPerBlock);
        uint block = BlockDim(k);
        nint wArg = quantWeight, xArg = 0, yArg = 0;
        int ldxArg = ldx, ldyArg = ldy, nArg = n, kArg = k, colsArg = 0;
        void** args = stackalloc void*[] { &wArg, &xArg, &ldxArg, &yArg, &ldyArg, &nArg, &kArg, &colsArg };
        for (int c0 = 0; c0 < ncols; c0 += MaxColumns)
        {
            xArg = xF32 + (nint)((long)c0 * ldx * sizeof(float));
            yArg = yF32 + (nint)((long)c0 * ldy * sizeof(float));
            colsArg = Math.Min(MaxColumns, ncols - c0);
            CudaDriverApi.cuLaunchKernel(_multiFunc, grid, 1, 1, block, 1, 1,
                    0, stream, (nint)args, 0).ThrowOnError();
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _module?.Dispose();
        _module = null;
    }
}
