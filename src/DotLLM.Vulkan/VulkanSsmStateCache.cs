using DotLLM.Core.Models;

namespace DotLLM.Vulkan;

/// <summary>
/// Device-local mirror of <see cref="DotLLM.Models.Architectures.SsmStateCache"/>. Stores
/// the per-sequence Mamba2 recurrent state (conv1d sliding-window history + SSM hidden
/// state) for every SSM layer of a single sequence in zeroed device-local buffers.
/// </summary>
/// <remarks>
/// <para>
/// One <c>conv_state</c> buffer of <c>(d_conv-1) * conv_dim</c> F32 elements per SSM
/// layer, plus one <c>ssm_state</c> buffer of <c>n_head * head_dim * d_state</c> F32
/// elements per SSM layer. Both are allocated device-local and zero-initialised at
/// construction (Mamba2 begins each sequence with a zero state by convention; mirrors
/// <see cref="DotLLM.Models.Architectures.SsmStateCache"/>'s constructor behaviour).
/// </para>
/// <para>
/// The Vulkan API surface in this codebase does not expose <c>vkCmdFillBuffer</c>, so
/// we zero the buffers via a one-shot host-staging upload during construction. After
/// that the conv_state is updated by per-token <c>vkCmdCopyBuffer</c> writes from the
/// SSM forward pass, and ssm_state is updated in place by the
/// <see cref="DotLLM.Vulkan.Kernels.Mamba2SelectiveScanF32Kernel"/>.
/// </para>
/// </remarks>
internal sealed class VulkanSsmStateCache : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _numSsmLayers;
    private readonly int _convStateElements;
    private readonly int _ssmStateElements;

    // One device-local buffer per SSM layer for each of the two state tensors.
    // Indexed by SSM-layer ordinal (NOT by absolute layer index) — the model
    // owns the (absoluteLayerIndex -> ssmOrdinal) lookup.
    private readonly VulkanDevice.Buffer[] _convStateBuffers;
    private readonly VulkanDevice.Buffer[] _ssmStateBuffers;

    private bool _disposed;

    /// <summary>Number of SSM layers covered by this cache.</summary>
    public int NumSsmLayers => _numSsmLayers;

    /// <summary>Elements per layer in the conv state (<c>(d_conv-1) * conv_dim</c>).</summary>
    public int ConvStateElements => _convStateElements;

    /// <summary>Elements per layer in the SSM state (<c>n_head * head_dim * d_state</c>).</summary>
    public int SsmStateElements => _ssmStateElements;

    /// <summary>Total bytes allocated across all conv-state and ssm-state buffers.</summary>
    public long AllocatedBytes =>
        (long)_numSsmLayers * (_convStateElements + _ssmStateElements) * sizeof(float);

    public VulkanSsmStateCache(VulkanDevice device, MambaSsmConfig ssm, int numSsmLayers)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (numSsmLayers < 0) throw new ArgumentOutOfRangeException(nameof(numSsmLayers));

        _device = device;
        _numSsmLayers = numSsmLayers;
        _convStateElements = ssm.ConvStateElements;
        _ssmStateElements = ssm.SsmStateElements;

        _convStateBuffers = new VulkanDevice.Buffer[numSsmLayers];
        _ssmStateBuffers = new VulkanDevice.Buffer[numSsmLayers];

        if (numSsmLayers == 0) return;

        long convBytes = (long)_convStateElements * sizeof(float);
        long ssmBytes = (long)_ssmStateElements * sizeof(float);

        // Stage one zeroed buffer sized for the larger of the two — reused across
        // every device-local zero-init below. .NET zero-inits managed arrays for
        // free, so the host-side bytes are zero on first map.
        long maxBytes = Math.Max(convBytes, ssmBytes);
        if (maxBytes <= 0) return;

        byte[] zeros = new byte[maxBytes];
        using var staging = device.Allocate(maxBytes);
        // Force a pre-clear of the staging memory in case the driver hands us
        // dirty pages — Upload only writes the requested length.
        device.Upload(zeros.AsSpan(0, (int)maxBytes), staging);

        for (int i = 0; i < numSsmLayers; i++)
        {
            _convStateBuffers[i] = device.AllocateDeviceLocal(convBytes);
            _ssmStateBuffers[i] = device.AllocateDeviceLocal(ssmBytes);

            device.CopyBufferSynchronous(staging, _convStateBuffers[i], (ulong)convBytes);
            device.CopyBufferSynchronous(staging, _ssmStateBuffers[i], (ulong)ssmBytes);
        }
    }

    /// <summary>
    /// Returns the conv-state device buffer for SSM-layer ordinal <paramref name="ssmLayerOrdinal"/>.
    /// </summary>
    public VulkanDevice.Buffer GetConvStateBuffer(int ssmLayerOrdinal)
    {
        ThrowIfDisposed();
        if ((uint)ssmLayerOrdinal >= (uint)_numSsmLayers)
            throw new ArgumentOutOfRangeException(nameof(ssmLayerOrdinal));
        return _convStateBuffers[ssmLayerOrdinal];
    }

    /// <summary>
    /// Returns the SSM-state device buffer for SSM-layer ordinal <paramref name="ssmLayerOrdinal"/>.
    /// </summary>
    public VulkanDevice.Buffer GetSsmStateBuffer(int ssmLayerOrdinal)
    {
        ThrowIfDisposed();
        if ((uint)ssmLayerOrdinal >= (uint)_numSsmLayers)
            throw new ArgumentOutOfRangeException(nameof(ssmLayerOrdinal));
        return _ssmStateBuffers[ssmLayerOrdinal];
    }

    /// <summary>Re-zeroes every layer's state. Useful at the start of a fresh sequence.</summary>
    public void Reset()
    {
        ThrowIfDisposed();
        if (_numSsmLayers == 0) return;

        long convBytes = (long)_convStateElements * sizeof(float);
        long ssmBytes = (long)_ssmStateElements * sizeof(float);
        long maxBytes = Math.Max(convBytes, ssmBytes);
        if (maxBytes <= 0) return;

        byte[] zeros = new byte[maxBytes];
        using var staging = _device.Allocate(maxBytes);
        _device.Upload(zeros.AsSpan(0, (int)maxBytes), staging);

        for (int i = 0; i < _numSsmLayers; i++)
        {
            _device.CopyBufferSynchronous(staging, _convStateBuffers[i], (ulong)convBytes);
            _device.CopyBufferSynchronous(staging, _ssmStateBuffers[i], (ulong)ssmBytes);
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VulkanSsmStateCache));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numSsmLayers; i++)
        {
            _convStateBuffers[i]?.Dispose();
            _ssmStateBuffers[i]?.Dispose();
        }
    }
}
