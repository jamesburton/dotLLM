using DotLLM.Hip.Interop;

namespace DotLLM.Hip.DirectStorage;

/// <summary>An independently owned device allocation populated by direct storage I/O.</summary>
public sealed class HipDirectWeightBuffer : IDisposable
{
    private readonly int _deviceId;
    private readonly nint _pointer;
    private readonly nint _allocation;
    private readonly nint _externalMemory;
    private readonly nint _resource;
    private readonly nint _heap;
    private int _disposed;

    internal HipDirectWeightBuffer(int deviceId, nint pointer, nint externalMemory, nint resource,
        nint heap, long byteLength, long allocatedBytes)
    {
        _deviceId = deviceId;
        _pointer = pointer;
        _allocation = pointer;
        _externalMemory = externalMemory;
        _resource = resource;
        _heap = heap;
        ByteLength = byteLength;
        AllocatedBytes = allocatedBytes;
    }

    internal HipDirectWeightBuffer(int deviceId, nint pointer, nint allocation, long byteLength, long allocatedBytes)
    {
        _deviceId = deviceId;
        _pointer = pointer;
        _allocation = allocation;
        ByteLength = byteLength;
        AllocatedBytes = allocatedBytes;
    }

    /// <summary>Address of the requested file bytes on the selected HIP device.</summary>
    public nint Pointer
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            return _pointer;
        }
    }

    /// <summary>Number of requested file bytes.</summary>
    public long ByteLength { get; }

    /// <summary>Device allocation size including platform alignment and file-range padding.</summary>
    public long AllocatedBytes { get; }

    internal nint Resource => _resource;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        int error = HipDriverApi.hipSetDevice(_deviceId);
        if (error == 0)
        {
            error = HipDriverApi.hipDeviceSynchronize();
            int freeError = HipDriverApi.hipFree(_allocation);
            int destroyError = _externalMemory != 0 ? HipExternalMemoryApi.DestroyExternalMemory(_externalMemory) : 0;
            if (error == 0) error = freeError != 0 ? freeError : destroyError;
        }
        ComVtbl.Release(_resource);
        ComVtbl.Release(_heap);
        error.ThrowOnError();
    }
}
