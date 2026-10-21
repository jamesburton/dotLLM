using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Per-sequence recurrent state of the LFM2 gated short-convolution layers: for each conv layer the last
/// <c>L_cache - 1</c> projected inputs (<c>B*X</c>) of every channel, stored time-major
/// (<c>[layer][slot 0 = oldest .. L_cache-2 = newest][channel]</c>), exactly the <c>conv_state</c> llama.cpp's
/// <c>build_shortconv_block</c> carries between ubatches.
/// </summary>
public sealed unsafe class Lfm2ConvStateCache : ISsmState
{
    private readonly int _numLayers;
    private readonly int _perLayerElements;
    private nint _data;
    private bool _disposed;

    /// <summary>Creates a zero-initialised state for <paramref name="numConvLayers"/> conv layers.</summary>
    /// <param name="numConvLayers">Number of short-conv layers in the model.</param>
    /// <param name="stateSlots">Rows kept per layer (<c>L_cache - 1</c>).</param>
    /// <param name="hiddenSize">Channels per row.</param>
    public Lfm2ConvStateCache(int numConvLayers, int stateSlots, int hiddenSize)
    {
        _numLayers = numConvLayers;
        StateSlots = stateSlots;
        HiddenSize = hiddenSize;
        _perLayerElements = stateSlots * hiddenSize;
        long bytes = (long)numConvLayers * _perLayerElements * sizeof(float);
        if (bytes > 0)
        {
            _data = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
            NativeMemory.Clear((void*)_data, (nuint)bytes);
        }
    }

    /// <inheritdoc/>
    public int NumSsmLayers => _numLayers;

    /// <summary>Rows kept per layer (<c>L_cache - 1</c>).</summary>
    public int StateSlots { get; }

    /// <summary>Channels per row.</summary>
    public int HiddenSize { get; }

    /// <summary>Total bytes owned.</summary>
    public long AllocatedBytes => (long)_numLayers * _perLayerElements * sizeof(float);

    /// <summary>Pointer to the <c>[StateSlots, HiddenSize]</c> state of conv-layer ordinal <paramref name="convOrdinal"/>.</summary>
    public float* Layer(int convOrdinal)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)convOrdinal >= (uint)_numLayers) throw new ArgumentOutOfRangeException(nameof(convOrdinal));
        return (float*)_data + (long)convOrdinal * _perLayerElements;
    }

    /// <inheritdoc/>
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_data != 0)
            NativeMemory.Clear((void*)_data, (nuint)AllocatedBytes);
    }

    /// <summary>Copies another cache's contents into this one (same geometry).</summary>
    public void CopyFrom(Lfm2ConvStateCache other)
    {
        ArgumentNullException.ThrowIfNull(other);
        ObjectDisposedException.ThrowIf(_disposed, this);
        ObjectDisposedException.ThrowIf(other._disposed, other);
        if (other._numLayers != _numLayers || other.StateSlots != StateSlots || other.HiddenSize != HiddenSize)
            throw new ArgumentException("State geometry mismatch.", nameof(other));
        if (_data != 0)
            Buffer.MemoryCopy((void*)other._data, (void*)_data, AllocatedBytes, AllocatedBytes);
    }

    internal void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    /// <summary>Creates an independent copy with the same geometry.</summary>
    public Lfm2ConvStateCache Clone()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var copy = new Lfm2ConvStateCache(_numLayers, StateSlots, HiddenSize);
        copy.CopyFrom(this);
        return copy;
    }
    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_data != 0)
        {
            NativeMemory.AlignedFree((void*)_data);
            _data = 0;
        }
    }
}
