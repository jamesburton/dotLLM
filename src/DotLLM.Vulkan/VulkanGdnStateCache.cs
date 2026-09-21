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
public sealed class VulkanGdnStateCache : IGdnState
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

    /// <summary>
    /// Allocates per-layer conv-state and gdn-state device buffers, zero-initialised
    /// via a single staging upload. Buffers are device-local F32.
    /// </summary>
    /// <param name="device">Vulkan device on which to allocate.</param>
    /// <param name="gdn">GDN hyperparameters shared by all GDN layers.</param>
    /// <param name="numGdnLayers">
    /// Count of GDN layers (blocks whose <c>HybridLayerKind</c> is
    /// <see cref="DotLLM.Core.Models.HybridLayerKind.GatedDeltaNet"/>).
    /// </param>
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
    internal VulkanDevice.Buffer GetConvStateBuffer(int ordinal)
    {
        ThrowIfDisposed();
        if ((uint)ordinal >= (uint)_numGdnLayers)
            throw new ArgumentOutOfRangeException(nameof(ordinal));
        return _convStateBuffers[ordinal];
    }

    /// <summary>Returns the gdn-state buffer for GDN-layer ordinal <paramref name="ordinal"/>.</summary>
    internal VulkanDevice.Buffer GetGdnStateBuffer(int ordinal)
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

    /// <summary>
    /// Private geometry-only constructor used by <see cref="Clone"/> — allocates the same
    /// per-layer buffer shapes without the zeroing upload, because every byte is about to be
    /// overwritten by a device-to-device copy.
    /// </summary>
    private VulkanGdnStateCache(VulkanDevice device, int numGdnLayers, int convStateElements, int gdnStateElements)
    {
        _device = device;
        _numGdnLayers = numGdnLayers;
        _convStateElements = convStateElements;
        _gdnStateElements = gdnStateElements;
        _convStateBuffers = new VulkanDevice.Buffer[numGdnLayers];
        _gdnStateBuffers = new VulkanDevice.Buffer[numGdnLayers];

        long convBytes = (long)convStateElements * sizeof(float);
        long stateBytes = (long)gdnStateElements * sizeof(float);
        for (int i = 0; i < numGdnLayers; i++)
        {
            _convStateBuffers[i] = device.AllocateDeviceLocal(convBytes);
            _gdnStateBuffers[i] = device.AllocateDeviceLocal(stateBytes);
        }
    }

    /// <summary>
    /// Device-to-device snapshot of every layer's recurrent state (issue #435). The returned cache
    /// is an independent allocation the caller owns and must dispose; it is the Vulkan counterpart
    /// of <c>GdnStateCache.Clone</c> / <c>CudaGdnStateCache.Clone</c>, and exists for the same
    /// reason (issue #287): a speculative verify batch advances the GDN recurrence for tokens that
    /// may then be rejected, and a pure sequential recurrence has no position addressing to undo.
    /// </summary>
    public VulkanGdnStateCache Clone()
    {
        ThrowIfDisposed();
        var copy = new VulkanGdnStateCache(_device, _numGdnLayers, _convStateElements, _gdnStateElements);
        CopyTo(copy);
        return copy;
    }

    /// <summary>
    /// Copies every layer's conv-state and gdn-state buffer into <paramref name="destination"/>
    /// (device-to-device, no host round-trip). Geometry must match exactly.
    /// </summary>
    public void CopyTo(VulkanGdnStateCache destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        ThrowIfDisposed();
        destination.ThrowIfDisposed();
        if (destination._numGdnLayers != _numGdnLayers
            || destination._convStateElements != _convStateElements
            || destination._gdnStateElements != _gdnStateElements)
        {
            throw new ArgumentException(
                $"VulkanGdnStateCache geometry mismatch: source " +
                $"({_numGdnLayers} layers, conv={_convStateElements}, state={_gdnStateElements}) vs destination " +
                $"({destination._numGdnLayers} layers, conv={destination._convStateElements}, state={destination._gdnStateElements}).",
                nameof(destination));
        }

        long convBytes = (long)_convStateElements * sizeof(float);
        long stateBytes = (long)_gdnStateElements * sizeof(float);
        for (int i = 0; i < _numGdnLayers; i++)
        {
            if (convBytes > 0)
                _device.CopyBufferSynchronous(_convStateBuffers[i], destination._convStateBuffers[i], (ulong)convBytes);
            if (stateBytes > 0)
                _device.CopyBufferSynchronous(_gdnStateBuffers[i], destination._gdnStateBuffers[i], (ulong)stateBytes);
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VulkanGdnStateCache));
    }

    /// <inheritdoc/>
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
