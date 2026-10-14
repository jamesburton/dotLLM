using DotLLM.Core.Models;

namespace DotLLM.Vulkan;

/// <summary>
/// Device-resident per-sequence recurrent state of the LFM2 gated short-convolution layers: for each conv layer the last
/// <c>L_cache - 1</c> projected inputs (<c>B*X</c>) per channel, time-major (<c>[slot 0 = oldest .. L-2][channel]</c>), matching
/// <see cref="DotLLM.Models.Architectures.Lfm2ConvStateCache"/> and llama.cpp's <c>conv_state</c> (#1015).
/// </summary>
public sealed class VulkanLfm2ConvStateCache : ISsmState
{
    private readonly VulkanDevice _device;
    private readonly VulkanDevice.Buffer[] _buffers;
    private readonly long _bytesPerLayer;
    private bool _disposed;

    /// <summary>Creates a zero-initialised state.</summary>
    public VulkanLfm2ConvStateCache(VulkanDevice device, int numConvLayers, int stateSlots, int hiddenSize)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        NumSsmLayers = numConvLayers;
        StateSlots = stateSlots;
        HiddenSize = hiddenSize;
        _bytesPerLayer = (long)stateSlots * hiddenSize * sizeof(float);
        _buffers = new VulkanDevice.Buffer[numConvLayers];
        for (int i = 0; i < numConvLayers; i++)
            _buffers[i] = device.AllocateDeviceLocal(_bytesPerLayer);
        Reset();
    }

    /// <inheritdoc/>
    public int NumSsmLayers { get; }

    /// <summary>Rows per layer (<c>L_cache - 1</c>).</summary>
    public int StateSlots { get; }

    /// <summary>Channels per row.</summary>
    public int HiddenSize { get; }

    /// <summary>Total device bytes.</summary>
    public long AllocatedBytes => _bytesPerLayer * NumSsmLayers;

    internal VulkanDevice.Buffer GetBuffer(int convOrdinal)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _buffers[convOrdinal];
    }

    /// <inheritdoc/>
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (NumSsmLayers == 0 || _bytesPerLayer == 0) return;
        byte[] zeros = new byte[_bytesPerLayer];
        using var staging = _device.Allocate(_bytesPerLayer);
        _device.Upload(zeros.AsSpan(), staging);
        foreach (var b in _buffers)
            _device.CopyBufferSynchronous(staging, b, (ulong)_bytesPerLayer);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var b in _buffers) b.Dispose();
    }
}
