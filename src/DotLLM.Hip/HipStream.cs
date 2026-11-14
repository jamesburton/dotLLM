using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// RAII wrapper around a HIP stream. Provides ordered execution of GPU operations.
/// </summary>
public sealed class HipStream : IDisposable
{
    private nint _stream;

    /// <summary>The native HIP stream handle. Pass to kernel launches and memcpy calls.</summary>
    public nint Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_stream == 0, this);
            return _stream;
        }
    }

    /// <summary>Device ordinal that owns the stream.</summary>
    public int DeviceId { get; }

    private HipStream(nint stream, int deviceId)
    {
        _stream = stream;
        DeviceId = deviceId;
    }

    /// <summary>Creates a new HIP stream on the current context's device.</summary>
    public static HipStream Create()
    {
        HipDriverApi.hipGetDevice(out int deviceId).ThrowOnError();
        HipDriverApi.hipStreamCreate(out nint stream).ThrowOnError();
        return new HipStream(stream, deviceId);
    }

    /// <summary>Blocks the host thread until all operations on this stream complete.</summary>
    public void Synchronize()
    {
        nint stream = Handle;
        HipDriverApi.hipSetDevice(DeviceId).ThrowOnError();
        HipDriverApi.hipStreamSynchronize(stream).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint stream = Interlocked.Exchange(ref _stream, 0);
        if (stream != 0)
        {
            HipDriverApi.hipSetDevice(DeviceId).ThrowOnError();
            HipDriverApi.hipStreamDestroy(stream);
        }
    }
}
