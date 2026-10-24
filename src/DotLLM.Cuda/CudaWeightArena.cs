using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Bump sub-allocator for model-lifetime weight tensors (#1211). The driver rounds every <c>cuMemAlloc</c> of 2 MiB or more
/// up to a 2 MiB multiple and packs smaller ones into 2 MiB pages, so one allocation per tensor wasted ~150 MiB on Flint
/// (a 6.25 MiB packed QKV takes 8 MiB, a 5 MiB O projection 6 MiB). The arena maps physical memory into one reserved
/// address range as it grows (<see cref="CudaGrowableDeviceBuffer"/>), so the tensors sit back to back and at most one
/// granule is left over. Nothing in it is freed individually; the whole arena goes with the weights.
/// </summary>
/// <remarks>
/// The loader opens one with <see cref="Begin"/> on its thread; allocation helpers ask <see cref="Current"/> and fall back to
/// their own <c>cuMemAlloc</c> when it is absent, full or the driver has no virtual-memory API.
/// <c>DOTLLM_CUDA_WEIGHT_ARENA=0</c> turns it off (A/B).
/// </remarks>
internal sealed class CudaWeightArena : IDisposable
{
    private const long Alignment = 256;

    [ThreadStatic]
    private static CudaWeightArena? t_current;

    private readonly CudaGrowableDeviceBuffer _buffer;
    private long _used;

    private CudaWeightArena(CudaGrowableDeviceBuffer buffer) => _buffer = buffer;

    /// <summary>The arena the current thread's weight load allocates from, or null.</summary>
    internal static CudaWeightArena? Current => t_current;

    /// <summary>Bytes handed out (with alignment padding).</summary>
    internal long UsedBytes => _used;

    /// <summary>Physical bytes mapped behind the arena.</summary>
    internal long MappedBytes => _buffer.MappedBytes;

    /// <summary>
    /// Opens an arena for the current context's device and makes it <see cref="Current"/> on this thread, or returns null
    /// (and leaves <see cref="Current"/> null) when it is disabled or unsupported.
    /// </summary>
    /// <returns>The arena; the caller owns it and ends the scope with <see cref="End"/>.</returns>
    internal static CudaWeightArena? Begin()
    {
        if (Environment.GetEnvironmentVariable("DOTLLM_CUDA_WEIGHT_ARENA") == "0"
            || CudaDriverApi.cuCtxGetDevice(out int device) != 0
            || CudaDriverVirtualMemory.TryCreate(device) is not { } vm
            || CudaDriverApi.cuDeviceTotalMem_v2(out nuint total, device) != 0)
            return null;

        // Reserve address space for the whole card: the weights can never need more, so the pointer never moves.
        t_current = new CudaWeightArena(new CudaGrowableDeviceBuffer(vm, (long)total));
        return t_current;
    }

    /// <summary>Stops this thread allocating from the arena (the arena itself stays alive).</summary>
    internal static void End() => t_current = null;

    /// <summary>Allocates <paramref name="bytes"/>, 256-byte aligned.</summary>
    /// <param name="bytes">Size in bytes.</param>
    /// <returns>The device pointer, or 0 when the arena cannot grow (the caller then allocates on its own).</returns>
    internal nint TryAlloc(long bytes)
    {
        long offset = (_used + Alignment - 1) / Alignment * Alignment;
        if (offset + bytes > _buffer.ReservedBytes)
            return 0;
        try
        {
            _buffer.EnsureCapacity(offset + bytes);
        }
        catch (CudaException)
        {
            return 0;
        }
        _used = offset + bytes;
        return _buffer.Pointer + (nint)offset;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (ReferenceEquals(t_current, this))
            t_current = null;
        _buffer.Dispose();
    }
}
