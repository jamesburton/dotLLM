using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// Wrapper around HIP's primary-context model. Binds the calling thread to the
/// primary context of a device via <c>hipSetDevice</c>. The runtime manages the
/// underlying context lifetime — no explicit destroy is required.
/// </summary>
/// <remarks>
/// We deliberately avoid the driver-style <c>hipCtxCreate</c> path: modern ROCm
/// ties <c>hipModuleLoadData</c> code-object initialization to the primary context,
/// and on Windows ROCm the driver-style contexts are partially stubbed. This
/// class is therefore a thin, disposable handle over the "current device"
/// concept — analogous to <c>CudaContext</c> but implemented against HIP's
/// runtime API.
/// </remarks>
public sealed class HipContext : IDisposable
{
    private int _disposed;

    /// <summary>Device ordinal this context is bound to.</summary>
    public int DeviceId { get; }

    private HipContext(int deviceId)
    {
        DeviceId = deviceId;
    }

    /// <summary>
    /// Initializes the HIP runtime and binds the calling thread to the primary
    /// context for the specified device.
    /// </summary>
    /// <param name="deviceId">Device ordinal (0-based).</param>
    public static HipContext Create(int deviceId)
    {
        HipLibraryResolver.Register();
        HipDriverApi.hipInit(0).ThrowOnError();
        HipDriverApi.hipSetDevice(deviceId).ThrowOnError();
        return new HipContext(deviceId);
    }

    /// <summary>Makes this device current on the calling thread.</summary>
    public void MakeCurrent()
    {
        HipDriverApi.hipSetDevice(DeviceId).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        // Primary context lifecycle is owned by the HIP runtime. Syncing here
        // ensures any pending kernels finish before the process tears down.
        try { HipDriverApi.hipDeviceSynchronize(); } catch { /* best-effort */ }
    }
}
