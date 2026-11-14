using DotLLM.Hip.Interop;

namespace DotLLM.Hip.DirectStorage;

/// <summary>An independently owned, device-local D3D12 allocation imported into HIP.</summary>
public sealed class HipDirectWeightBuffer : IDisposable
{
    private readonly int _deviceId;
    private readonly nint _pointer;
    private readonly nint _externalMemory;
    private readonly nint _resource;
    private readonly nint _heap;
    private int _disposed;

    internal HipDirectWeightBuffer(int deviceId, nint pointer, nint externalMemory, nint resource,
        nint heap, long byteLength, long allocatedBytes)
    {
        _deviceId = deviceId;
        _pointer = pointer;
        _externalMemory = externalMemory;
        _resource = resource;
        _heap = heap;
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

    /// <summary>Device allocation size including D3D12's 64 KiB heap alignment.</summary>
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
            int freeError = HipDriverApi.hipFree(_pointer);
            int destroyError = HipExternalMemoryApi.DestroyExternalMemory(_externalMemory);
            if (error == 0) error = freeError != 0 ? freeError : destroyError;
        }
        ComVtbl.Release(_resource);
        ComVtbl.Release(_heap);
        error.ThrowOnError();
    }
}
