using DotLLM.Core.Models;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Device-side per-sequence recurrent state cache for a Mamba-3 model's SSM layers.
/// Mirror of <see cref="DotLLM.Models.Architectures.Mamba3State"/> but allocating GPU
/// memory via <c>cuMemAlloc_v2</c> — same allocation/dispose/reset idiom as the
/// structurally-analogous <see cref="CudaGdnStateCache"/>, but with FOUR buffers per
/// layer (matching Mamba3State) instead of GDN's two, and no conv-state buffer
/// (Mamba-3 has no causal-conv1d step — see the plan's Global Constraints).
/// </summary>
/// <remarks>
/// <para>
/// Per Mamba-3 layer the cache stores, all zero-initialised at construction:
/// </para>
/// <list type="bullet">
///   <item><description><c>ssm_state</c> — <c>[nHead, headDim, dState]</c> canonical SSM hidden state.</description></item>
///   <item><description><c>cum_angle</c> — <c>[nHead, numRopeAngles]</c> running cumulative RoPE angle.</description></item>
///   <item><description><c>k_state</c> — SISO <c>[nHead, dState]</c>; MIMO <c>[mimoRank, nHead, dState]</c> — previous chunk's last-token post-RoPE K.</description></item>
///   <item><description><c>v_state</c> — <c>[nHead, headDim]</c> — previous chunk's last-token V.</description></item>
/// </list>
/// <para>
/// Buffers are F32 on device. <c>mamba3_ssd_scan_siso_f32</c> /
/// <c>mamba3_ssd_scan_mimo_f32</c> / <c>mamba3_data_rope_f32</c> /
/// <c>mamba3_chunk_boundary_f32</c> consume and mutate these pointers directly via
/// the <see cref="CudaKernels"/> launchers.
/// </para>
/// </remarks>
public sealed unsafe class CudaMamba3StateCache : IMambaState
{
    private readonly Mamba3Config _m3;
    private readonly int _numLayers;
    private readonly int _ssmStateElementsPerLayer;
    private readonly int _cumAngleElementsPerLayer;
    private readonly int _kStateElementsPerLayer;
    private readonly int _vStateElementsPerLayer;

    // Contiguous per-layer blocks, same pointer-arithmetic pattern as CudaGdnStateCache.
    private nint _ssmState;
    private nint _cumAngle;
    private nint _kState;
    private nint _vState;

    private bool _disposed;

    /// <inheritdoc/>
    public int NumLayers => _numLayers;

    /// <summary>Elements per layer in the SSM hidden state (<c>nHead * headDim * dState</c>).</summary>
    public int SsmStateElementsPerLayer => _ssmStateElementsPerLayer;

    /// <summary>Elements per layer in the cumulative RoPE angle buffer (<c>nHead * numRopeAngles</c>).</summary>
    public int CumAngleElementsPerLayer => _cumAngleElementsPerLayer;

    /// <summary>Elements per layer in the K state (rank-aware — see class doc).</summary>
    public int KStateElementsPerLayer => _kStateElementsPerLayer;

    /// <summary>Elements per layer in the V state (<c>nHead * headDim</c>).</summary>
    public int VStateElementsPerLayer => _vStateElementsPerLayer;

    /// <summary>Total bytes allocated across all four state buffers.</summary>
    public long AllocatedBytes =>
        (long)_numLayers * (_ssmStateElementsPerLayer + _cumAngleElementsPerLayer
                            + _kStateElementsPerLayer + _vStateElementsPerLayer) * sizeof(float);

    /// <summary>
    /// Creates a new Mamba-3 state cache for the given config and layer count. All
    /// buffers are zero-initialised (zero state = start of sequence) via
    /// <c>cuMemsetD8_v2</c>.
    /// </summary>
    /// <param name="m3">Mamba-3 hyperparameters (<see cref="ModelConfig.Mamba3Config"/>).</param>
    /// <param name="numLayers">Number of Mamba-3 layers covered by this cache.</param>
    public CudaMamba3StateCache(Mamba3Config m3, int numLayers)
    {
        ArgumentNullException.ThrowIfNull(m3);
        if (numLayers < 0) throw new ArgumentOutOfRangeException(nameof(numLayers));

        _m3 = m3;
        _numLayers = numLayers;
        // k_state carries a rank axis in MIMO — mirrors Mamba3State's kRank logic exactly.
        int kRank = m3.IsMimo ? m3.MimoRank : 1;
        _ssmStateElementsPerLayer = m3.NumHeads * m3.HeadDim * m3.StateSize;
        _cumAngleElementsPerLayer = m3.NumHeads * m3.NumRopeAngles;
        _kStateElementsPerLayer = kRank * m3.NumHeads * m3.StateSize;
        _vStateElementsPerLayer = m3.NumHeads * m3.HeadDim;

        if (numLayers == 0)
        {
            _ssmState = 0; _cumAngle = 0; _kState = 0; _vState = 0;
            return;
        }

        if (_ssmStateElementsPerLayer <= 0 || _cumAngleElementsPerLayer <= 0
            || _kStateElementsPerLayer <= 0 || _vStateElementsPerLayer <= 0)
            throw new ArgumentException(
                "CudaMamba3StateCache requires positive ssm/cum_angle/k_state/v_state element counts; check Mamba3Config dims.",
                nameof(m3));

        long ssmBytes = (long)_numLayers * _ssmStateElementsPerLayer * sizeof(float);
        long cumBytes = (long)_numLayers * _cumAngleElementsPerLayer * sizeof(float);
        long kBytes = (long)_numLayers * _kStateElementsPerLayer * sizeof(float);
        long vBytes = (long)_numLayers * _vStateElementsPerLayer * sizeof(float);

        CudaDriverApi.cuMemAlloc_v2(out _ssmState, (nuint)ssmBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _cumAngle, (nuint)cumBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _kState, (nuint)kBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _vState, (nuint)vBytes).ThrowOnError();

        CudaDriverApi.cuMemsetD8_v2(_ssmState, 0, (nuint)ssmBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_cumAngle, 0, (nuint)cumBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_kState, 0, (nuint)kBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_vState, 0, (nuint)vBytes).ThrowOnError();
    }

    /// <summary>Device pointer to layer <paramref name="layerIndex"/>'s ssm_state, length <see cref="SsmStateElementsPerLayer"/> floats.</summary>
    public nint GetSsmStatePtr(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers) throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _ssmState + (nint)((long)layerIndex * _ssmStateElementsPerLayer * sizeof(float));
    }

    /// <summary>Device pointer to layer <paramref name="layerIndex"/>'s cum_angle, length <see cref="CumAngleElementsPerLayer"/> floats.</summary>
    public nint GetCumAnglePtr(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers) throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _cumAngle + (nint)((long)layerIndex * _cumAngleElementsPerLayer * sizeof(float));
    }

    /// <summary>Device pointer to layer <paramref name="layerIndex"/>'s k_state, length <see cref="KStateElementsPerLayer"/> floats.</summary>
    public nint GetKStatePtr(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers) throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _kState + (nint)((long)layerIndex * _kStateElementsPerLayer * sizeof(float));
    }

    /// <summary>Device pointer to layer <paramref name="layerIndex"/>'s v_state, length <see cref="VStateElementsPerLayer"/> floats.</summary>
    public nint GetVStatePtr(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers) throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _vState + (nint)((long)layerIndex * _vStateElementsPerLayer * sizeof(float));
    }

    /// <summary>
    /// Device-to-device deep-copies this cache into a freshly-allocated
    /// <see cref="CudaMamba3StateCache"/> of the same shape — mirrors
    /// <see cref="CudaGdnStateCache.Clone"/>'s speculative-decoding checkpoint role
    /// (issue #287), extended to Mamba-3's four-buffer state.
    /// </summary>
    public CudaMamba3StateCache Clone()
    {
        ThrowIfDisposed();
        var clone = new CudaMamba3StateCache(_m3, _numLayers);
        CopyTo(clone);
        return clone;
    }

    /// <summary>Device-to-device overwrites <paramref name="destination"/>'s buffers with this cache's current contents.</summary>
    public void CopyTo(CudaMamba3StateCache destination)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(destination);
        destination.ThrowIfDisposed();
        if (destination._numLayers != _numLayers
            || destination._ssmStateElementsPerLayer != _ssmStateElementsPerLayer
            || destination._cumAngleElementsPerLayer != _cumAngleElementsPerLayer
            || destination._kStateElementsPerLayer != _kStateElementsPerLayer
            || destination._vStateElementsPerLayer != _vStateElementsPerLayer)
        {
            throw new ArgumentException("Destination CudaMamba3StateCache shape does not match this cache's shape.", nameof(destination));
        }

        if (_numLayers == 0) return;

        long ssmBytes = (long)_numLayers * _ssmStateElementsPerLayer * sizeof(float);
        long cumBytes = (long)_numLayers * _cumAngleElementsPerLayer * sizeof(float);
        long kBytes = (long)_numLayers * _kStateElementsPerLayer * sizeof(float);
        long vBytes = (long)_numLayers * _vStateElementsPerLayer * sizeof(float);

        CudaDriverApi.cuMemcpyDtoD_v2(destination._ssmState, _ssmState, (nuint)ssmBytes).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoD_v2(destination._cumAngle, _cumAngle, (nuint)cumBytes).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoD_v2(destination._kState, _kState, (nuint)kBytes).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoD_v2(destination._vState, _vState, (nuint)vBytes).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Reset()
    {
        ThrowIfDisposed();
        if (_numLayers == 0) return;
        long ssmBytes = (long)_numLayers * _ssmStateElementsPerLayer * sizeof(float);
        long cumBytes = (long)_numLayers * _cumAngleElementsPerLayer * sizeof(float);
        long kBytes = (long)_numLayers * _kStateElementsPerLayer * sizeof(float);
        long vBytes = (long)_numLayers * _vStateElementsPerLayer * sizeof(float);
        CudaDriverApi.cuMemsetD8_v2(_ssmState, 0, (nuint)ssmBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_cumAngle, 0, (nuint)cumBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_kState, 0, (nuint)kBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_vState, 0, (nuint)vBytes).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_ssmState != 0) { CudaDriverApi.cuMemFree_v2(_ssmState); _ssmState = 0; }
        if (_cumAngle != 0) { CudaDriverApi.cuMemFree_v2(_cumAngle); _cumAngle = 0; }
        if (_kState != 0) { CudaDriverApi.cuMemFree_v2(_kState); _kState = 0; }
        if (_vState != 0) { CudaDriverApi.cuMemFree_v2(_vState); _vState = 0; }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CudaMamba3StateCache));
    }

    /// <summary>Finalizer — last-ditch free if not disposed.</summary>
    ~CudaMamba3StateCache()
    {
        if (_disposed) return;
        if (_ssmState != 0) CudaDriverApi.cuMemFree_v2(_ssmState);
        if (_cumAngle != 0) CudaDriverApi.cuMemFree_v2(_cumAngle);
        if (_kState != 0) CudaDriverApi.cuMemFree_v2(_kState);
        if (_vState != 0) CudaDriverApi.cuMemFree_v2(_vState);
    }
}
