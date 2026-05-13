using DotLLM.Core.Models;

namespace DotLLM.Vulkan;

/// <summary>
/// Device-local mirror of <see cref="DotLLM.Models.Architectures.GdnStateCache"/>.
/// Stores per-GDN-layer recurrent state (the rolling conv1d Q/K/V history plus the
/// full <c>[NVHead, DState, DState]</c> associative-memory matrix) for one sequence.
/// </summary>
/// <remarks>
/// <para>
/// One conv-state buffer of <c>(DConv-1) * (2*NKHead + NVHead) * DState</c> F32
/// elements per GDN layer, plus one gdn-state buffer of <c>NVHead * DState * DState</c>
/// F32 elements per GDN layer. Both are zero-initialised at construction — GDN
/// begins each sequence with empty associative memory and no conv history.
/// </para>
/// <para>
/// The Vulkan API surface used elsewhere in this codebase does not expose
/// <c>vkCmdFillBuffer</c>, so we zero the buffers via a one-shot host→staging
/// upload at construction time (same pattern as <see cref="VulkanSsmStateCache"/>).
/// After that the buffers are mutated in place by the GDN forward kernels and
/// per-token <c>vkCmdCopyBuffer</c> writes.
/// </para>
/// </remarks>
internal sealed class VulkanGdnStateCache : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _numGdnLayers;
    private readonly int _convStateElements;
    private readonly int _gdnStateElements;

    // Indexed by GDN-layer ordinal (NOT absolute layer index); the model owns
    // the (absoluteLayerIndex -> ordinal) lookup.
    private readonly VulkanDevice.Buffer[] _convStateBuffers;
    private readonly VulkanDevice.Buffer[] _gdnStateBuffers;

    private bool _disposed;

    /// <summary>Number of GDN layers covered.</summary>
    public int NumGdnLayers => _numGdnLayers;

    /// <summary>Per-layer conv-state element count.</summary>
    public int ConvStateElements => _convStateElements;

    /// <summary>Per-layer gdn-state element count (<c>NVHead * DState * DState</c>).</summary>
    public int GdnStateElements => _gdnStateElements;

    /// <summary>Total bytes allocated across both buffer arrays.</summary>
    public long AllocatedBytes =>
        (long)_numGdnLayers * (_convStateElements + _gdnStateElements) * sizeof(float);

    public VulkanGdnStateCache(VulkanDevice device, GatedDeltaNetConfig gdn, int numGdnLayers)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (numGdnLayers < 0) throw new ArgumentOutOfRangeException(nameof(numGdnLayers));

        _device = device;
        _numGdnLayers = numGdnLayers;
        _convStateElements = gdn.ConvStateElements;
        _gdnStateElements = gdn.StateElements;

        _convStateBuffers = new VulkanDevice.Buffer[numGdnLayers];
        _gdnStateBuffers = new VulkanDevice.Buffer[numGdnLayers];

        if (numGdnLayers == 0) return;

        long convBytes = (long)_convStateElements * sizeof(float);
        long stateBytes = (long)_gdnStateElements * sizeof(float);
        long maxBytes = Math.Max(convBytes, stateBytes);
        if (maxBytes <= 0) return;

        byte[] zeros = new byte[maxBytes];
        using var staging = device.Allocate(maxBytes);
        device.Upload(zeros.AsSpan(0, (int)maxBytes), staging);

        for (int i = 0; i < numGdnLayers; i++)
        {
            _convStateBuffers[i] = device.AllocateDeviceLocal(convBytes);
            _gdnStateBuffers[i] = device.AllocateDeviceLocal(stateBytes);
            device.CopyBufferSynchronous(staging, _convStateBuffers[i], (ulong)convBytes);
            device.CopyBufferSynchronous(staging, _gdnStateBuffers[i], (ulong)stateBytes);
        }
    }

    /// <summary>Returns the conv-state buffer for GDN-layer ordinal <paramref name="ordinal"/>.</summary>
    public VulkanDevice.Buffer GetConvStateBuffer(int ordinal)
    {
        ThrowIfDisposed();
        if ((uint)ordinal >= (uint)_numGdnLayers)
            throw new ArgumentOutOfRangeException(nameof(ordinal));
        return _convStateBuffers[ordinal];
    }

    /// <summary>Returns the gdn-state buffer for GDN-layer ordinal <paramref name="ordinal"/>.</summary>
    public VulkanDevice.Buffer GetGdnStateBuffer(int ordinal)
    {
        ThrowIfDisposed();
        if ((uint)ordinal >= (uint)_numGdnLayers)
            throw new ArgumentOutOfRangeException(nameof(ordinal));
        return _gdnStateBuffers[ordinal];
    }

    /// <summary>Re-zeroes every layer's state. Use between independent sequences.</summary>
    public void Reset()
    {
        ThrowIfDisposed();
        if (_numGdnLayers == 0) return;

        long convBytes = (long)_convStateElements * sizeof(float);
        long stateBytes = (long)_gdnStateElements * sizeof(float);
        long maxBytes = Math.Max(convBytes, stateBytes);
        if (maxBytes <= 0) return;

        byte[] zeros = new byte[maxBytes];
        using var staging = _device.Allocate(maxBytes);
        _device.Upload(zeros.AsSpan(0, (int)maxBytes), staging);

        for (int i = 0; i < _numGdnLayers; i++)
        {
            _device.CopyBufferSynchronous(staging, _convStateBuffers[i], (ulong)convBytes);
            _device.CopyBufferSynchronous(staging, _gdnStateBuffers[i], (ulong)stateBytes);
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VulkanGdnStateCache));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numGdnLayers; i++)
        {
            _convStateBuffers[i]?.Dispose();
            _gdnStateBuffers[i]?.Dispose();
        }
    }
}
