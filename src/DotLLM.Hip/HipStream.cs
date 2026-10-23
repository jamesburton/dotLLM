using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// RAII wrapper around a HIP stream. Provides ordered execution of GPU operations.
/// </summary>
public sealed class HipStream : IDisposable
{
    private nint _stream;

    /// <summary>The native HIP stream handle. Pass to kernel launches and memcpy calls.</summary>
    public nint Handle => _stream;

    private HipStream(nint stream)
    {
        _stream = stream;
    }

    /// <summary>Creates a new HIP stream on the current context's device.</summary>
    public static HipStream Create()
    {
        HipDriverApi.hipStreamCreate(out nint stream).ThrowOnError();
        return new HipStream(stream);
    }

    /// <summary>Blocks the host thread until all operations on this stream complete.</summary>
    public void Synchronize()
    {
        HipDriverApi.hipStreamSynchronize(_stream).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint stream = Interlocked.Exchange(ref _stream, 0);
        if (stream != 0)
            HipDriverApi.hipStreamDestroy(stream);
    }
}
