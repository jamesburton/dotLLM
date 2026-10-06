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

    /// <summary>
    /// Env switch (issue #492): <c>DOTLLM_CUDA_Q8_RB_MULTI_SPECIALIZED=0</c> keeps
    /// <see cref="LaunchMulti"/> on the generic <c>_rb_multi</c> kernel, so the specialised
    /// entry points can be A/B'd against the exact code that shipped with #486.
    /// </summary>
    public const string DisableSpecializedEnvVar = "DOTLLM_CUDA_Q8_RB_MULTI_SPECIALIZED";

    /// <summary>Name of the <paramref name="cols"/>-column specialised entry point.</summary>
    public static string SpecializedKernelName(int cols) => $"{MultiKernelName}_c{cols}";

    /// <summary><see langword="true"/> when <c>DOTLLM_CUDA_Q8_RB_GEMV=0</c> disables the kernel.</summary>
    public static bool DisabledByEnv { get; } = Environment.GetEnvironmentVariable(DisableEnvVar) == "0";

    /// <summary><see langword="true"/> when <c>DOTLLM_CUDA_Q8_RB_MULTI_SPECIALIZED=0</c> forces the generic kernel.</summary>
    public static bool SpecializedDisabledByEnv { get; } =
        Environment.GetEnvironmentVariable(DisableSpecializedEnvVar) == "0";

    private CudaModule? _module;
    private readonly nint _func;
    private readonly nint _multiFunc;

    /// <summary>
    /// Specialised multi-column entry points indexed by column count (slot 0 unused), or 0 when the
    /// PTX predates issue #492. Resolved once at load; <see cref="LaunchMulti"/> only indexes it.
    /// </summary>
    private readonly nint[] _colFuncs;

    private CudaQ8_0RbGemv(CudaModule module, nint func, nint multiFunc, nint[] colFuncs)
    {
        _module = module;
        _func = func;
        _multiFunc = multiFunc;
        _colFuncs = colFuncs;
    }

    /// <summary>Whether the multi-column entry point loaded (it ships in the same PTX).</summary>
    public bool HasMulti => _multiFunc != 0;

    /// <summary>
    /// Whether the <paramref name="cols"/>-column specialised entry point (issue #492) is present
    /// <em>and</em> enabled — i.e. whether <see cref="LaunchMulti"/> will use it for that group size.
    /// Test hook: a parity/perf test can assert it is not silently measuring the generic kernel.
    /// </summary>
    public bool HasSpecialized(int cols) =>
        !SpecializedDisabledByEnv && (uint)cols < (uint)_colFuncs.Length && _colFuncs[cols] != 0;

    /// <summary>
    /// Diagnostic (issue #492): the driver's static attributes for the <paramref name="cols"/>-column
    /// specialised kernel — registers per thread, static shared bytes, local (spill) bytes and the
    /// resident blocks per SM at <paramref name="k"/>'s launch shape. Returns <see langword="false"/>
    /// when that entry point is absent. Lets a test print what <c>-Xptxas -v</c> would, without a
    /// <c>-cubin</c> build. Not used by any production path.
    /// </summary>
    public bool TryGetKernelInfo(int cols, int k, out int regs, out int sharedBytes, out int localBytes, out int blocksPerSm)
    {
        regs = sharedBytes = localBytes = blocksPerSm = 0;
        if ((uint)cols >= (uint)_colFuncs.Length || _colFuncs[cols] == 0)
            return false;
        nint f = _colFuncs[cols];
        CudaDriverApi.cuFuncGetAttribute(out regs, CudaDriverApi.CU_FUNC_ATTRIBUTE_NUM_REGS, f);
        CudaDriverApi.cuFuncGetAttribute(out sharedBytes, CudaDriverApi.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f);
        CudaDriverApi.cuFuncGetAttribute(out localBytes, CudaDriverApi.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f);
        CudaDriverApi.cuOccupancyMaxActiveBlocksPerMultiprocessor(out blocksPerSm, f, (int)BlockDim(k), 0);
        return true;
    }

    /// <summary>
    /// Same diagnostic for the generic <c>_rb_multi</c> kernel — the A/B baseline's side of the
    /// register/shared/occupancy comparison.
    /// </summary>
    public bool TryGetGenericKernelInfo(int k, out int regs, out int sharedBytes, out int localBytes, out int blocksPerSm)
    {
        regs = sharedBytes = localBytes = blocksPerSm = 0;
        if (_multiFunc == 0)
            return false;
        CudaDriverApi.cuFuncGetAttribute(out regs, CudaDriverApi.CU_FUNC_ATTRIBUTE_NUM_REGS, _multiFunc);
        CudaDriverApi.cuFuncGetAttribute(out sharedBytes, CudaDriverApi.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, _multiFunc);
        CudaDriverApi.cuFuncGetAttribute(out localBytes, CudaDriverApi.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, _multiFunc);
        CudaDriverApi.cuOccupancyMaxActiveBlocksPerMultiprocessor(out blocksPerSm, _multiFunc, (int)BlockDim(k), 0);
        return true;
    }

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
            // Issue #492: the NCOLS-specialised entries ship in the same module. Absent in an older
            // PTX -> the slot stays 0 and LaunchMulti falls back to the generic kernel.
            var colFuncs = new nint[MaxColumns + 1];
            for (int c = 1; c <= MaxColumns; c++)
            {
                nint cf = module.TryGetFunction(SpecializedKernelName(c));
                colFuncs[c] = cf;
                // Their 37,888 B of static shared memory only leaves room for the two blocks per SM
                // that __launch_bounds__(256, 2) budgets registers for at the maximum carveout; ask
                // for it explicitly rather than trusting the driver's default heuristic. Advisory:
                // a driver that rejects the attribute simply keeps its own choice.
                if (cf != 0)
                    CudaDriverApi.cuFuncSetAttribute(cf, CudaDriverApi.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100);
            }
            return new CudaQ8_0RbGemv(module, func, module.TryGetFunction(MultiKernelName), colFuncs);
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
    /// <para>
    /// Issue #492: a group whose size has a specialised entry point goes to it instead of the generic
    /// 8-column kernel — same bits, sized accumulators and two resident blocks per SM instead of one.
    /// </para>
    /// </summary>
    public void LaunchMulti(nint quantWeight, nint xF32, int ldx, nint yF32, int ldy,
                            int n, int k, int ncols, nint stream) =>
        LaunchMultiCore(quantWeight, xF32, ldx, yF32, ldy, n, k, ncols, stream, preferSpecialized: true);

    /// <summary>
    /// <see cref="LaunchMulti"/> pinned to the generic <c>_rb_multi</c> kernel — the A/B baseline the
    /// specialised entries (issue #492) must match bit for bit, and the shape
    /// <c>DOTLLM_CUDA_Q8_RB_MULTI_SPECIALIZED=0</c> selects. Test/benchmark hook.
    /// </summary>
    public void LaunchMultiGeneric(nint quantWeight, nint xF32, int ldx, nint yF32, int ldy,
                                   int n, int k, int ncols, nint stream) =>
        LaunchMultiCore(quantWeight, xF32, ldx, yF32, ldy, n, k, ncols, stream, preferSpecialized: false);

    private unsafe void LaunchMultiCore(nint quantWeight, nint xF32, int ldx, nint yF32, int ldy,
                                        int n, int k, int ncols, nint stream, bool preferSpecialized)
    {
        ObjectDisposedException.ThrowIf(_module is null, this);
        if (_multiFunc == 0)
            throw new InvalidOperationException($"{MultiKernelName} not present in {PtxFileName}.");
        uint grid = (uint)((n + RowsPerBlock - 1) / RowsPerBlock);
        uint block = BlockDim(k);
        nint wArg = quantWeight, xArg = 0, yArg = 0;
        int ldxArg = ldx, ldyArg = ldy, nArg = n, kArg = k, colsArg = 0;
        // The specialised entries (issue #492) take the same arguments minus the runtime column
        // count, so one stackalloc'd array serves both: the generic kernel reads 8 slots, a
        // specialised one reads the first 7.
        void** args = stackalloc void*[] { &wArg, &xArg, &ldxArg, &yArg, &ldyArg, &nArg, &kArg, &colsArg };
        for (int c0 = 0; c0 < ncols; c0 += MaxColumns)
        {
            xArg = xF32 + (nint)((long)c0 * ldx * sizeof(float));
            yArg = yF32 + (nint)((long)c0 * ldy * sizeof(float));
            colsArg = Math.Min(MaxColumns, ncols - c0);
            nint func = preferSpecialized && HasSpecialized(colsArg) ? _colFuncs[colsArg] : _multiFunc;
            CudaDriverApi.cuLaunchKernel(func, grid, 1, 1, block, 1, 1,
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
