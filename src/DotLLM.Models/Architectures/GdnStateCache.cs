using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Per-sequence recurrent state cache for the Gated DeltaNet (GDN) layers of a
/// Qwen3MoeHybrid model. One cache instance covers all GDN layers for a single sequence.
/// </summary>
/// <remarks>
/// <para>
/// For each GDN layer the cache stores two buffers:
/// </para>
/// <list type="bullet">
///   <item>
///     <description>
///       <c>conv_state</c> — the rolling K history for causal conv1d, shape
///       <c>[(DConv−1) × NKHead × DState]</c> row-major.
///     </description>
///   </item>
///   <item>
///     <description>
///       <c>gdn_state</c> — the full associative-memory matrix state, shape
///       <c>[NVHead × DState × DState]</c> row-major. This is substantially
///       larger than a Mamba-2 vector state: with NVHead=32, DState=64,
///       each layer holds 32 × 64 × 64 × 4 B = 512 KB.
///     </description>
///   </item>
/// </list>
/// <para>
/// Buffers are unmanaged, 64-byte aligned (AVX-512 friendly), zero-initialised
/// on creation. Forward passes obtain <see cref="Span{T}"/> slices via
/// <see cref="GetConvState"/> and <see cref="GetGdnState"/> and mutate them in place.
/// </para>
/// </remarks>
public sealed unsafe class GdnStateCache : IGdnState
{
    private readonly int _numGdnLayers;
    private readonly int _convStateElements;
    private readonly int _gdnStateElements;

    // Contiguous per-layer blocks. GDN layer ordinal i occupies:
    //   conv:  _convState[i*_convStateElements .. (i+1)*_convStateElements)
    //   state: _gdnState [i*_gdnStateElements  .. (i+1)*_gdnStateElements)
    private nint _convState;
    private nint _gdnState;

    private bool _disposed;

    /// <summary>Number of GDN layers covered by this cache.</summary>
    public int NumGdnLayers => _numGdnLayers;

    /// <summary>
    /// Elements per layer in the conv rolling-K buffer:
    /// <c>(DConv−1) × NKHead × DState</c>.
    /// </summary>
    public int ConvStateElements => _convStateElements;

    /// <summary>
    /// Elements per layer in the GDN matrix state:
    /// <c>NVHead × DState × DState</c>.
    /// </summary>
    public int GdnStateElements => _gdnStateElements;

    /// <summary>
    /// Creates a new GDN state cache for the given config and layer count.
    /// All buffers are zero-initialised (zero state = no prior history).
    /// </summary>
    /// <param name="gdn">GDN hyperparameters shared by all GDN layers.</param>
    /// <param name="numGdnLayers">
    /// Count of GDN layers (blocks whose <c>HybridLayerKind</c> is
    /// <see cref="DotLLM.Core.Models.HybridLayerKind.GatedDeltaNet"/>).
    /// </param>
    public GdnStateCache(GatedDeltaNetConfig gdn, int numGdnLayers)
    {
        if (numGdnLayers < 0) throw new ArgumentOutOfRangeException(nameof(numGdnLayers));

        _numGdnLayers = numGdnLayers;
        _convStateElements = gdn.ConvStateElements; // (DConv-1) * NKHead * DState
        _gdnStateElements = gdn.StateElements;      // NVHead * DState * DState

        if (numGdnLayers == 0)
        {
            _convState = 0;
            _gdnState = 0;
            return;
        }

        long convBytes = (long)_numGdnLayers * _convStateElements * sizeof(float);
        long stateBytes = (long)_numGdnLayers * _gdnStateElements * sizeof(float);

        _convState = (nint)NativeMemory.AlignedAlloc((nuint)convBytes, 64);
        _gdnState = (nint)NativeMemory.AlignedAlloc((nuint)stateBytes, 64);

        // GDN starts with zero state (empty associative memory, no K history).
        NativeMemory.Clear((void*)_convState, (nuint)convBytes);
        NativeMemory.Clear((void*)_gdnState, (nuint)stateBytes);
    }

    /// <summary>
    /// Returns the conv rolling-K state slice for GDN layer ordinal
    /// <paramref name="gdnLayerIndex"/>. Indexed by GDN-layer ordinal, not
    /// by absolute block index.
    /// </summary>
    public Span<float> GetConvState(int gdnLayerIndex)
    {
        ThrowIfDisposed();
        if ((uint)gdnLayerIndex >= (uint)_numGdnLayers)
            throw new ArgumentOutOfRangeException(nameof(gdnLayerIndex));
        return new Span<float>(
            (float*)_convState + (long)gdnLayerIndex * _convStateElements,
            _convStateElements);
    }

    /// <summary>
    /// Returns the GDN matrix state slice for GDN layer ordinal
    /// <paramref name="gdnLayerIndex"/>. Shape: <c>[NVHead, DState, DState]</c> row-major.
    /// Indexed by GDN-layer ordinal, not by absolute block index.
    /// </summary>
    public Span<float> GetGdnState(int gdnLayerIndex)
    {
        ThrowIfDisposed();
        if ((uint)gdnLayerIndex >= (uint)_numGdnLayers)
            throw new ArgumentOutOfRangeException(nameof(gdnLayerIndex));
        return new Span<float>(
            (float*)_gdnState + (long)gdnLayerIndex * _gdnStateElements,
            _gdnStateElements);
    }

    /// <summary>
    /// Zeroes every layer's state. Call between independent sequences.
    /// </summary>
    public void Reset()
    {
        ThrowIfDisposed();
        if (_numGdnLayers == 0) return;
        NativeMemory.Clear((void*)_convState, (nuint)((long)_numGdnLayers * _convStateElements * sizeof(float)));
        NativeMemory.Clear((void*)_gdnState, (nuint)((long)_numGdnLayers * _gdnStateElements * sizeof(float)));
    }

    /// <summary>Total bytes allocated across both state buffers.</summary>
    public long AllocatedBytes =>
        (long)_numGdnLayers * (_convStateElements + _gdnStateElements) * sizeof(float);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_convState != 0) { NativeMemory.AlignedFree((void*)_convState); _convState = 0; }
        if (_gdnState != 0) { NativeMemory.AlignedFree((void*)_gdnState); _gdnState = 0; }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GdnStateCache));
    }

    /// <summary>Finalizer — last-ditch free if the cache was not disposed.</summary>
    ~GdnStateCache()
    {
        if (_disposed) return;
        if (_convState != 0) NativeMemory.AlignedFree((void*)_convState);
        if (_gdnState != 0) NativeMemory.AlignedFree((void*)_gdnState);
    }
}
