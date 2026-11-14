using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>An owned allocation on one explicit HIP device.</summary>
internal sealed class HipDeviceBuffer : IDisposable
{
    private nint _pointer;
    internal nint Pointer => _pointer != 0 ? _pointer : throw new ObjectDisposedException(nameof(HipDeviceBuffer));
    internal long ByteLength { get; }
    private readonly int _deviceId;

    internal HipDeviceBuffer(HipContext context, long bytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(bytes);
        context.MakeCurrent();
        HipDriverApi.hipMemGetInfo(out nuint free, out _).ThrowOnError();
        if ((ulong)bytes > free)
            throw new OutOfMemoryException($"HIP allocation requires {bytes} bytes; device {context.DeviceId} has {free} bytes free.");
        HipDriverApi.hipMalloc(out _pointer, checked((nuint)bytes)).ThrowOnError();
        _deviceId = context.DeviceId;
        ByteLength = bytes;
    }

    public void Dispose()
    {
        nint pointer = Interlocked.Exchange(ref _pointer, 0);
        if (pointer == 0) return;
        HipDriverApi.hipSetDevice(_deviceId).ThrowOnError();
        HipDriverApi.hipFree(pointer).ThrowOnError();
    }
}
