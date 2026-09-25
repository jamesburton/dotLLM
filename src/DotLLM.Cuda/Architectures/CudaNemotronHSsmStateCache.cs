using DotLLM.Core.Models;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Device-side per-sequence recurrent state cache for the Mamba2 SSM layers of a NemotronH
/// model. Mirror of <see cref="DotLLM.Models.Architectures.SsmStateCache"/> (CPU) and
/// <see cref="CudaGdnStateCache"/> (same pattern, different recurrence config), allocating GPU
/// memory via <c>cuMemAlloc_v2</c>.
/// </summary>
/// <remarks>
/// One cache instance covers all SSM layers for a single sequence. Per SSM layer the cache
/// stores two device buffers, both zero-initialised at construction:
/// <c>conv_state</c> (<c>(DConv-1) * ConvDim</c> F32 elements — rolling conv1d history) and
/// <c>ssm_state</c> (<c>DInner * DState</c> F32 elements — the Mamba2 recurrent state matrix,
/// shape <c>[n_head, head_dim, d_state]</c>). <c>mamba2_selective_scan_f32</c>
/// (native/kernels/mamba2_selective_scan.cu) and <c>conv1d_causal_f32</c> consume and mutate
/// these pointers directly.
/// </remarks>
internal sealed unsafe class CudaNemotronHSsmStateCache : ISsmState
{
    private readonly MambaSsmConfig _ssm;
    private readonly int _numSsmLayers;
    private readonly int _convStateElements;
    private readonly int _ssmStateElements;

    // Contiguous per-layer blocks. SSM layer ordinal i occupies:
    //   conv:  _convState + i * _convStateElements * sizeof(float)
    //   state: _ssmState  + i * _ssmStateElements  * sizeof(float)
    private nint _convState;
    private nint _ssmState;

    private bool _disposed;

    /// <inheritdoc/>
    public int NumSsmLayers => _numSsmLayers;

    /// <summary>Elements per layer in the conv rolling buffer.</summary>
    public int ConvStateElements => _convStateElements;

    /// <summary>Elements per layer in the SSM matrix state.</summary>
    public int SsmStateElements => _ssmStateElements;

    /// <summary>Total bytes allocated across both state buffers.</summary>
    public long AllocatedBytes =>
        (long)_numSsmLayers * (_convStateElements + _ssmStateElements) * sizeof(float);

    /// <summary>
    /// Device-to-device deep-copies this cache's current contents into a freshly-allocated
    /// <see cref="CudaNemotronHSsmStateCache"/> of the same shape — mirrors
    /// <see cref="CudaGdnStateCache.Clone"/> (used there for speculative-decoding state
    /// rollback; kept here for the same future use).
    /// </summary>
    public CudaNemotronHSsmStateCache Clone()
    {
        ThrowIfDisposed();
        var clone = new CudaNemotronHSsmStateCache(_ssm, _numSsmLayers);
        CopyTo(clone);
        return clone;
    }

    /// <summary>Device-to-device overwrites <paramref name="destination"/>'s buffers with this
    /// cache's current contents via <c>cuMemcpyDtoD_v2</c>. Both caches must share the same shape.</summary>
    public void CopyTo(CudaNemotronHSsmStateCache destination)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(destination);
        destination.ThrowIfDisposed();
        if (destination._numSsmLayers != _numSsmLayers
            || destination._convStateElements != _convStateElements
            || destination._ssmStateElements != _ssmStateElements)
        {
            throw new ArgumentException(
                "Destination CudaNemotronHSsmStateCache shape does not match this cache's shape.", nameof(destination));
        }

        if (_numSsmLayers == 0) return;

        long convBytes = (long)_numSsmLayers * _convStateElements * sizeof(float);
        long stateBytes = (long)_numSsmLayers * _ssmStateElements * sizeof(float);
        if (convBytes > 0)
            CudaDriverApi.cuMemcpyDtoD_v2(destination._convState, _convState, (nuint)convBytes).ThrowOnError();
        if (stateBytes > 0)
            CudaDriverApi.cuMemcpyDtoD_v2(destination._ssmState, _ssmState, (nuint)stateBytes).ThrowOnError();
    }

    /// <summary>Creates a new SSM state cache for the given config and layer count. All buffers
    /// are zero-initialised (zero state = no prior history) using <c>cuMemsetD8_v2</c>.</summary>
    public CudaNemotronHSsmStateCache(MambaSsmConfig ssm, int numSsmLayers)
    {
        if (numSsmLayers < 0) throw new ArgumentOutOfRangeException(nameof(numSsmLayers));

        _ssm = ssm;
        _numSsmLayers = numSsmLayers;
        _convStateElements = ssm.ConvStateElements; // (DConv-1) * ConvDim
        _ssmStateElements = ssm.SsmStateElements;   // DInner * DState

        if (numSsmLayers == 0)
        {
            _convState = 0;
            _ssmState = 0;
            return;
        }

        long convBytes = (long)_numSsmLayers * _convStateElements * sizeof(float);
        long stateBytes = (long)_numSsmLayers * _ssmStateElements * sizeof(float);

        CudaDriverApi.cuMemAlloc_v2(out _convState, (nuint)convBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _ssmState, (nuint)stateBytes).ThrowOnError();

        CudaDriverApi.cuMemsetD8_v2(_convState, 0, (nuint)convBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_ssmState, 0, (nuint)stateBytes).ThrowOnError();
    }

    /// <summary>Device pointer to SSM layer <paramref name="ssmLayerIndex"/>'s conv rolling
    /// state buffer. Length: <see cref="ConvStateElements"/> floats.</summary>
    public nint GetConvStatePtr(int ssmLayerIndex)
    {
        ThrowIfDisposed();
        if ((uint)ssmLayerIndex >= (uint)_numSsmLayers)
            throw new ArgumentOutOfRangeException(nameof(ssmLayerIndex));
        return _convState + (nint)((long)ssmLayerIndex * _convStateElements * sizeof(float));
    }

    /// <summary>Device pointer to SSM layer <paramref name="ssmLayerIndex"/>'s matrix-state
    /// buffer. Length: <see cref="SsmStateElements"/> floats (shape
    /// <c>[n_head, head_dim, d_state]</c> row-major).</summary>
    public nint GetSsmStatePtr(int ssmLayerIndex)
    {
        ThrowIfDisposed();
        if ((uint)ssmLayerIndex >= (uint)_numSsmLayers)
            throw new ArgumentOutOfRangeException(nameof(ssmLayerIndex));
        return _ssmState + (nint)((long)ssmLayerIndex * _ssmStateElements * sizeof(float));
    }

    /// <inheritdoc/>
    public void Reset()
    {
        ThrowIfDisposed();
        if (_numSsmLayers == 0) return;
        long convBytes = (long)_numSsmLayers * _convStateElements * sizeof(float);
        long stateBytes = (long)_numSsmLayers * _ssmStateElements * sizeof(float);
        CudaDriverApi.cuMemsetD8_v2(_convState, 0, (nuint)convBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_ssmState, 0, (nuint)stateBytes).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_convState != 0) { CudaDriverApi.cuMemFree_v2(_convState); _convState = 0; }
        if (_ssmState != 0) { CudaDriverApi.cuMemFree_v2(_ssmState); _ssmState = 0; }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CudaNemotronHSsmStateCache));
    }

    ~CudaNemotronHSsmStateCache()
    {
        if (_disposed) return;
        if (_convState != 0) CudaDriverApi.cuMemFree_v2(_convState);
        if (_ssmState != 0) CudaDriverApi.cuMemFree_v2(_ssmState);
    }
}
