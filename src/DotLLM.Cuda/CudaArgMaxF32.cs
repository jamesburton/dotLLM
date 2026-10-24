using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Optional device argmax over an FP32 vector (<c>native/kernels/argmax_f32.cu</c>, issue #486): the
/// MTP greedy draft reads one token id back instead of the whole logits row.
/// </summary>
/// <remarks>
/// <para>
/// Same result as <c>TensorPrimitives.IndexOfMax</c> on the host row: largest value, lowest index on
/// a tie, first NaN if any, <c>+0</c> above <c>-0</c>. The kernel's comparison is a strict total order,
/// so the answer does not depend on the reduction order.
/// </para>
/// <para>
/// Owns a 4-byte device result slot and a pinned 4-byte host slot, so a call allocates nothing.
/// Optional: <see cref="TryLoad"/> returns <see langword="null"/> when the PTX is absent or does not
/// JIT, and callers keep the full-logits path.
/// </para>
/// </remarks>
internal sealed class CudaArgMaxF32 : IDisposable
{
    /// <summary>PTX file name, resolved in the same directory as the rest of the kernels.</summary>
    public const string PtxFileName = "argmax_f32.ptx";

    /// <summary>The <c>extern "C"</c> kernel name.</summary>
    public const string KernelName = "argmax_f32";

    private const uint Threads = 1024;   // ARGMAX_THREADS

    private CudaModule? _module;
    private readonly nint _func;
    private nint _resultDevice;
    private nint _resultHostPinned;

    private CudaArgMaxF32(CudaModule module, nint func, nint resultDevice, nint resultHostPinned)
    {
        _module = module;
        _func = func;
        _resultDevice = resultDevice;
        _resultHostPinned = resultHostPinned;
    }

    /// <summary>
    /// Loads the kernel from <paramref name="ptxDir"/>, or returns <see langword="null"/> when the PTX
    /// is absent or the module/function does not load. The CUDA context must be current.
    /// </summary>
    public static CudaArgMaxF32? TryLoad(string ptxDir)
    {
        string path = Path.Combine(ptxDir, PtxFileName);
        if (!File.Exists(path))
            return null;

        CudaModule? module = null;
        nint resultDevice = 0, resultHost = 0;
        try
        {
            module = CudaModule.LoadFromFile(path);
            nint func = module.TryGetFunction(KernelName);
            if (func == 0)
            {
                module.Dispose();
                return null;
            }
            CudaDriverApi.cuMemAlloc_v2(out resultDevice, sizeof(int)).ThrowOnError();
            CudaDriverApi.cuMemHostAlloc(out resultHost, sizeof(int), 0).ThrowOnError();
            return new CudaArgMaxF32(module, func, resultDevice, resultHost);
        }
        catch (Exception ex) when (ex is CudaException or IOException)
        {
            if (resultDevice != 0) CudaDriverApi.cuMemFree_v2(resultDevice);
            if (resultHost != 0) CudaDriverApi.cuMemFreeHost(resultHost);
            module?.Dispose();
            return null;
        }
    }

    /// <summary>
    /// Enqueues the argmax of <c>xF32[0..n)</c> on <paramref name="stream"/> and its 4-byte copy into
    /// the pinned host slot. Call <see cref="Result"/> after synchronizing the stream.
    /// </summary>
    public unsafe void Launch(nint xF32, int n, nint stream)
    {
        ObjectDisposedException.ThrowIf(_module is null, this);
        if (n <= 0)
            throw new ArgumentOutOfRangeException(nameof(n));
        nint xArg = xF32, outArg = _resultDevice;
        int nArg = n;
        void** args = stackalloc void*[] { &xArg, &nArg, &outArg };
        CudaDriverApi.cuLaunchKernel(_func, 1, 1, 1, Threads, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoHAsync_v2(_resultHostPinned, _resultDevice, sizeof(int), stream).ThrowOnError();
    }

    /// <summary>The index written by the last <see cref="Launch"/>; valid once its stream is synchronized.</summary>
    public unsafe int Result
    {
        get
        {
            ObjectDisposedException.ThrowIf(_module is null, this);
            return *(int*)_resultHostPinned;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_resultDevice != 0) { CudaDriverApi.cuMemFree_v2(_resultDevice); _resultDevice = 0; }
        if (_resultHostPinned != 0) { CudaDriverApi.cuMemFreeHost(_resultHostPinned); _resultHostPinned = 0; }
        _module?.Dispose();
        _module = null;
    }
}
