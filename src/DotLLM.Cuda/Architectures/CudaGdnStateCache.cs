using DotLLM.Core.Models;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Device-side per-sequence recurrent state cache for the Gated DeltaNet (GDN) layers of a
/// Qwen3MoeHybrid model. Mirror of <see cref="DotLLM.Models.Architectures.GdnStateCache"/>
/// but allocating GPU memory via <c>cuMemAlloc_v2</c>.
/// </summary>
/// <remarks>
/// <para>
/// One cache instance covers all GDN layers for a single sequence. Per GDN layer the cache
/// stores two device buffers, both zero-initialised at construction:
/// </para>
/// <list type="bullet">
///   <item>
///     <description>
///       <c>conv_state</c> — rolling Q/K/V history for the causal conv1d, shape
///       <c>[(DConv−1) × convDim]</c> row-major, where
///       <c>convDim = (2·NKHead + NVHead)·DState</c>. Total elements per layer:
///       <see cref="GatedDeltaNetConfig.ConvStateElements"/>.
///     </description>
///   </item>
///   <item>
///     <description>
///       <c>gdn_state</c> — the full associative-memory matrix state, shape
///       <c>[NVHead × DState × DState]</c> row-major (row = key dim, col = value dim).
///       Total elements per layer: <see cref="GatedDeltaNetConfig.StateElements"/>.
///     </description>
///   </item>
/// </list>
/// <para>
/// Buffers are F32 on device. The kernels in <c>gated_delta_net_scan.cu</c> /
/// <c>conv1d_causal.cu</c> consume and mutate these pointers directly via PInvoke wrappers.
/// </para>
/// </remarks>
internal sealed unsafe class CudaGdnStateCache : IGdnState
{
    private readonly int _numGdnLayers;
    private readonly int _convStateElements;
    private readonly int _gdnStateElements;

    // Contiguous per-layer blocks. GDN layer ordinal i occupies:
    //   conv:  _convState + i * _convStateElements * sizeof(float)
    //   state: _gdnState  + i * _gdnStateElements  * sizeof(float)
    private nint _convState;
    private nint _gdnState;

    private bool _disposed;

    /// <summary>Number of GDN layers covered by this cache.</summary>
    public int NumGdnLayers => _numGdnLayers;

    /// <summary>Elements per layer in the conv rolling buffer.</summary>
    public int ConvStateElements => _convStateElements;

    /// <summary>Elements per layer in the GDN matrix state.</summary>
    public int GdnStateElements => _gdnStateElements;

    /// <summary>Total bytes allocated across both state buffers.</summary>
    public long AllocatedBytes =>
        (long)_numGdnLayers * (_convStateElements + _gdnStateElements) * sizeof(float);

    /// <summary>
    /// Creates a new GDN state cache for the given config and layer count. All buffers
    /// are zero-initialised (zero state = no prior history) using <c>cuMemsetD8_v2</c>.
    /// </summary>
    public CudaGdnStateCache(GatedDeltaNetConfig gdn, int numGdnLayers)
    {
        if (numGdnLayers < 0) throw new ArgumentOutOfRangeException(nameof(numGdnLayers));

        _numGdnLayers = numGdnLayers;
        _convStateElements = gdn.ConvStateElements; // (DConv-1) * convDim
        _gdnStateElements = gdn.StateElements;      // NVHead * DState * DState

        if (numGdnLayers == 0)
        {
            _convState = 0;
            _gdnState = 0;
            return;
        }

        long convBytes = (long)_numGdnLayers * _convStateElements * sizeof(float);
        long stateBytes = (long)_numGdnLayers * _gdnStateElements * sizeof(float);

        CudaDriverApi.cuMemAlloc_v2(out _convState, (nuint)convBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _gdnState, (nuint)stateBytes).ThrowOnError();

        // Zero-init.
        CudaDriverApi.cuMemsetD8_v2(_convState, 0, (nuint)convBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_gdnState, 0, (nuint)stateBytes).ThrowOnError();
    }

    /// <summary>
    /// Returns the device pointer to GDN layer <paramref name="gdnLayerIndex"/>'s
    /// conv rolling state buffer. Length: <see cref="ConvStateElements"/> floats.
    /// </summary>
    public nint GetConvStatePtr(int gdnLayerIndex)
    {
        ThrowIfDisposed();
        if ((uint)gdnLayerIndex >= (uint)_numGdnLayers)
            throw new ArgumentOutOfRangeException(nameof(gdnLayerIndex));
        return _convState + (nint)((long)gdnLayerIndex * _convStateElements * sizeof(float));
    }

    /// <summary>
    /// Returns the device pointer to GDN layer <paramref name="gdnLayerIndex"/>'s
    /// matrix-state buffer. Length: <see cref="GdnStateElements"/> floats
    /// (shape <c>[NVHead, DState, DState]</c> row-major).
    /// </summary>
    public nint GetGdnStatePtr(int gdnLayerIndex)
    {
        ThrowIfDisposed();
        if ((uint)gdnLayerIndex >= (uint)_numGdnLayers)
            throw new ArgumentOutOfRangeException(nameof(gdnLayerIndex));
        return _gdnState + (nint)((long)gdnLayerIndex * _gdnStateElements * sizeof(float));
    }

    /// <summary>
    /// Zeroes every layer's state. Call between independent sequences.
    /// </summary>
    public void Reset()
    {
        ThrowIfDisposed();
        if (_numGdnLayers == 0) return;
        long convBytes = (long)_numGdnLayers * _convStateElements * sizeof(float);
        long stateBytes = (long)_numGdnLayers * _gdnStateElements * sizeof(float);
        CudaDriverApi.cuMemsetD8_v2(_convState, 0, (nuint)convBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_gdnState, 0, (nuint)stateBytes).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_convState != 0) { CudaDriverApi.cuMemFree_v2(_convState); _convState = 0; }
        if (_gdnState != 0) { CudaDriverApi.cuMemFree_v2(_gdnState); _gdnState = 0; }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CudaGdnStateCache));
    }

    /// <summary>Finalizer — last-ditch free if not disposed.</summary>
    ~CudaGdnStateCache()
    {
        if (_disposed) return;
        if (_convState != 0) CudaDriverApi.cuMemFree_v2(_convState);
        if (_gdnState != 0) CudaDriverApi.cuMemFree_v2(_gdnState);
    }
}
