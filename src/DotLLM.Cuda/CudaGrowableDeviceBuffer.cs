using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// The CUDA virtual-memory calls <see cref="CudaGrowableDeviceBuffer"/> makes, behind an interface so its bookkeeping (offsets,
/// rounding, rollback on failure, re-reservation) is testable without a GPU (issue #1080).
/// </summary>
internal interface ICudaVirtualMemory
{
    /// <summary>Allocation granularity in bytes: every reservation, physical allocation and mapping is a multiple of it.</summary>
    long Granularity { get; }

    /// <summary>Reserves <paramref name="bytes"/> of device address space, with no memory behind it.</summary>
    /// <param name="bytes">Size, a multiple of <see cref="Granularity"/>.</param>
    /// <returns>The base address of the range.</returns>
    nint Reserve(long bytes);

    /// <summary>Returns a range from <see cref="Reserve"/>; nothing may still be mapped in it.</summary>
    /// <param name="address">Base address.</param>
    /// <param name="bytes">The reserved size.</param>
    void FreeReservation(nint address, long bytes);

    /// <summary>Allocates <paramref name="bytes"/> of physical device memory.</summary>
    /// <param name="bytes">Size, a multiple of <see cref="Granularity"/>.</param>
    /// <returns>The allocation's handle.</returns>
    ulong CreatePhysical(long bytes);

    /// <summary>Releases a physical allocation; its memory is freed once it is no longer mapped anywhere.</summary>
    /// <param name="handle">Handle from <see cref="CreatePhysical"/>.</param>
    void ReleasePhysical(ulong handle);

    /// <summary>Maps all of a physical allocation at <paramref name="address"/> and grants the device read/write access.</summary>
    /// <param name="address">Address inside a reservation.</param>
    /// <param name="bytes">The allocation's size.</param>
    /// <param name="handle">Handle from <see cref="CreatePhysical"/>.</param>
    void Map(nint address, long bytes, ulong handle);

    /// <summary>Unmaps one mapping made by <see cref="Map"/>.</summary>
    /// <param name="address">The address it was mapped at.</param>
    /// <param name="bytes">Its size.</param>
    void Unmap(nint address, long bytes);
}

/// <summary>
/// A device buffer that grows in place (issue #1080). It reserves a virtual address range once and maps physical memory into it
/// as it grows, so the pointer stays the same, the contents stay where they are, and a growth costs only the added bytes.
/// A copying growth needs the old and the new buffer at once, which on a card serving near its fitted context window does not
/// fit, so every agent turn used to re-prefill its whole prompt.
/// </summary>
/// <remarks>
/// <para>Growth past the reservation reserves a larger range and maps the same physical allocations into it, so even then
/// nothing is copied. Only then does <see cref="Pointer"/> change; the caller must have no work in flight on the buffer.</para>
/// <para>A failed growth leaves the buffer as it was: whatever it allocated or mapped is undone before the exception propagates.</para>
/// <para>Not thread-safe. The owning CUDA context must be current for every call.</para>
/// </remarks>
internal sealed class CudaGrowableDeviceBuffer : IDisposable
{
    private readonly ICudaVirtualMemory _vm;
    private readonly List<(ulong Handle, long Bytes)> _chunks = [];
    private nint _base;
    private long _reserved;
    private long _mapped;

    /// <summary>Creates an empty buffer with <paramref name="reserveBytes"/> (rounded up) of address space reserved.</summary>
    /// <param name="vm">The virtual-memory calls.</param>
    /// <param name="reserveBytes">Address space to reserve up front: the largest size the buffer is expected to reach.</param>
    public CudaGrowableDeviceBuffer(ICudaVirtualMemory vm, long reserveBytes)
    {
        ArgumentNullException.ThrowIfNull(vm);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(reserveBytes);
        _vm = vm;
        _reserved = RoundUp(reserveBytes, vm.Granularity);
        _base = vm.Reserve(_reserved);
    }

    /// <summary>Device address of the buffer's first byte. Stable across growth within the reservation.</summary>
    public nint Pointer => _base;

    /// <summary>Bytes of physical memory mapped, a multiple of the granularity: what the buffer actually occupies.</summary>
    public long MappedBytes => _mapped;

    /// <summary>Bytes of address space reserved.</summary>
    public long ReservedBytes => _reserved;

    /// <summary>Physical allocations behind the buffer: one per growth.</summary>
    public int ChunkCount => _chunks.Count;

    /// <summary>Rounds <paramref name="bytes"/> up to a multiple of <paramref name="granularity"/>.</summary>
    /// <param name="bytes">Byte count, zero or more.</param>
    /// <param name="granularity">A positive granularity.</param>
    /// <returns>The rounded count.</returns>
    public static long RoundUp(long bytes, long granularity)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(granularity);
        return checked((bytes + granularity - 1) / granularity * granularity);
    }

    /// <summary>Physical bytes <see cref="EnsureCapacity"/> would add to hold <paramref name="bytes"/>: zero when it already does.</summary>
    /// <param name="bytes">Capacity wanted.</param>
    /// <returns>Bytes to be newly allocated.</returns>
    public long GrowthBytes(long bytes) => Math.Max(0, RoundUp(bytes, _vm.Granularity) - _mapped);

    /// <summary>Grows the buffer to hold at least <paramref name="bytes"/>, keeping its contents. Never shrinks.</summary>
    /// <param name="bytes">Capacity wanted.</param>
    /// <returns><see langword="true"/> when <see cref="Pointer"/> moved (the growth went past the reservation).</returns>
    public bool EnsureCapacity(long bytes)
    {
        ObjectDisposedException.ThrowIf(_base == 0, this);
        long target = RoundUp(bytes, _vm.Granularity);
        if (target <= _mapped) return false;

        bool moved = false;
        if (target > _reserved)
        {
            Rereserve(target);
            moved = true;
        }

        long delta = target - _mapped;
        ulong handle = _vm.CreatePhysical(delta);
        try
        {
            _vm.Map(_base + (nint)_mapped, delta, handle);
        }
        catch
        {
            _vm.ReleasePhysical(handle);
            throw;
        }
        _chunks.Add((handle, delta));
        _mapped = target;
        return moved;
    }

    /// <summary>Moves the mapped chunks into a new, larger reservation. On failure the old reservation stays in use.</summary>
    private void Rereserve(long bytes)
    {
        nint next = _vm.Reserve(bytes);
        int mapped = 0;
        long offset = 0;
        try
        {
            foreach (var (handle, size) in _chunks)
            {
                _vm.Map(next + (nint)offset, size, handle);
                mapped++;
                offset += size;
            }
        }
        catch
        {
            UnmapChunks(next, mapped);
            _vm.FreeReservation(next, bytes);
            throw;
        }

        UnmapChunks(_base, _chunks.Count);
        _vm.FreeReservation(_base, _reserved);
        _base = next;
        _reserved = bytes;
    }

    private void UnmapChunks(nint address, int count)
    {
        long offset = 0;
        for (int i = 0; i < count; i++)
        {
            _vm.Unmap(address + (nint)offset, _chunks[i].Bytes);
            offset += _chunks[i].Bytes;
        }
    }

    /// <summary>Unmaps and releases the physical memory, then the address range. The caller must have no work in flight on it.</summary>
    public void Dispose()
    {
        if (_base == 0) return;
        UnmapChunks(_base, _chunks.Count);
        foreach (var (handle, _) in _chunks)
            _vm.ReleasePhysical(handle);
        _chunks.Clear();
        _vm.FreeReservation(_base, _reserved);
        _base = 0;
        _mapped = 0;
        _reserved = 0;
    }
}

/// <summary><see cref="ICudaVirtualMemory"/> over the CUDA driver's <c>cuMem*</c> virtual-memory API, for one device.</summary>
internal sealed class CudaDriverVirtualMemory : ICudaVirtualMemory
{
    private readonly CudaDriverApi.CudaMemAllocationProp _prop;
    private readonly CudaDriverApi.CudaMemAccessDesc _access;

    private CudaDriverVirtualMemory(int deviceId, long granularity)
    {
        var location = new CudaDriverApi.CudaMemLocation { Type = CudaDriverApi.CU_MEM_LOCATION_TYPE_DEVICE, Id = deviceId };
        _prop = new CudaDriverApi.CudaMemAllocationProp { Type = CudaDriverApi.CU_MEM_ALLOCATION_TYPE_PINNED, Location = location };
        _access = new CudaDriverApi.CudaMemAccessDesc { Location = location, Flags = CudaDriverApi.CU_MEM_ACCESS_FLAGS_PROT_READWRITE };
        Granularity = granularity;
    }

    /// <inheritdoc/>
    public long Granularity { get; }

    /// <summary>
    /// The device's virtual-memory API, or <see langword="null"/> when the device or driver does not support it (the caller
    /// then keeps its copying growth).
    /// </summary>
    /// <param name="deviceId">CUDA device ordinal.</param>
    /// <returns>The API, or null.</returns>
    public static CudaDriverVirtualMemory? TryCreate(int deviceId)
    {
        try
        {
            if (CudaDriverApi.cuDeviceGet(out int device, deviceId) != 0
                || CudaDriverApi.cuDeviceGetAttribute(out int supported,
                    CudaDriverApi.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device) != 0
                || supported == 0)
                return null;
            var probe = new CudaDriverVirtualMemory(deviceId, 1);
            if (CudaDriverApi.cuMemGetAllocationGranularity(out nuint granularity, probe._prop,
                    CudaDriverApi.CU_MEM_ALLOC_GRANULARITY_MINIMUM) != 0 || granularity == 0)
                return null;
            return new CudaDriverVirtualMemory(deviceId, (long)granularity);
        }
        catch (EntryPointNotFoundException)
        {
            return null;   // a driver older than the VMM API
        }
    }

    /// <inheritdoc/>
    public nint Reserve(long bytes)
    {
        CudaDriverApi.cuMemAddressReserve(out nint address, checked((nuint)bytes), 0, 0, 0).ThrowOnError();
        return address;
    }

    /// <inheritdoc/>
    public void FreeReservation(nint address, long bytes)
        => CudaDriverApi.cuMemAddressFree(address, checked((nuint)bytes)).ThrowOnError();

    /// <inheritdoc/>
    public ulong CreatePhysical(long bytes)
    {
        CudaDriverApi.cuMemCreate(out ulong handle, checked((nuint)bytes), _prop, 0).ThrowOnError();
        return handle;
    }

    /// <inheritdoc/>
    public void ReleasePhysical(ulong handle) => CudaDriverApi.cuMemRelease(handle).ThrowOnError();

    /// <inheritdoc/>
    public void Map(nint address, long bytes, ulong handle)
    {
        CudaDriverApi.cuMemMap(address, checked((nuint)bytes), 0, handle, 0).ThrowOnError();
        int rc = CudaDriverApi.cuMemSetAccess(address, checked((nuint)bytes), _access, 1);
        if (rc != 0)
        {
            CudaDriverApi.cuMemUnmap(address, checked((nuint)bytes));
            rc.ThrowOnError();
        }
    }

    /// <inheritdoc/>
    public void Unmap(nint address, long bytes) => CudaDriverApi.cuMemUnmap(address, checked((nuint)bytes)).ThrowOnError();
}
