using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU-resident KV-cache storing FP16 key and value vectors per layer.
/// Layout: [maxSeqLen, numKvHeads * headDim] per layer, FP16.
/// </summary>
public sealed class CudaKvCache : IKvCache, IPerLayerKvCache
{
    private readonly nint[] _keys;
    private readonly nint[] _values;
    private readonly int _numLayers;
    private readonly KvGeometry _geom;     // per-layer KV row width (numKvHeads(l) * headDim(l))
    private readonly bool _uniform;        // _geom.IsUniform, hoisted for the hot path
    private readonly int _uniformStride;   // _geom.UniformStride when _uniform; else 0
    private readonly int _maxSeqLen;
    private int _currentLength;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <inheritdoc/>
    int IPerLayerKvCache.LayerCount => _numLayers;

    /// <inheritdoc/>
    public int KvStrideOf(int layerIndex) => _geom.KvStrideOf(layerIndex);

    /// <summary>Per-layer KV row width (FP16 elements); scalar shortcut hoisted when uniform.</summary>
    private int Stride(int layerIndex) => _uniform ? _uniformStride : _geom.KvStrideOf(layerIndex);

    /// <summary>
    /// Allocates GPU KV-cache buffers for all layers with a single uniform
    /// <c>numKvHeads * headDim</c> stride. Byte-identical to the per-layer constructor
    /// with <see cref="KvGeometry.Uniform"/>.
    /// </summary>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numKvHeads">Number of KV attention heads.</param>
    /// <param name="headDim">Dimension per head.</param>
    /// <param name="maxSeqLen">Maximum sequence length.</param>
    public CudaKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen)
        : this(KvGeometry.Uniform(numLayers, numKvHeads, headDim), maxSeqLen)
    {
    }

    /// <summary>
    /// Allocates GPU KV-cache buffers from a Core <see cref="KvGeometry"/> descriptor.
    /// Uniform for every dense/GQA/MoE model (byte-identical to the scalar constructor);
    /// per-layer for Gemma-4. NOTE: the Gemma-4 CUDA attention path is currently cacheless,
    /// so this per-layer support is the constructor/buffer-sizing surface only — it unblocks
    /// a future CUDA Gemma-4 decode path without shipping one here.
    /// </summary>
    public CudaKvCache(KvGeometry geometry, int maxSeqLen)
    {
        _numLayers = geometry.LayerCount;
        _geom = geometry;
        _uniform = geometry.IsUniform;
        _uniformStride = geometry.IsUniform ? geometry.UniformStride : 0;
        _maxSeqLen = maxSeqLen;
        _keys = new nint[_numLayers];
        _values = new nint[_numLayers];

        for (int i = 0; i < _numLayers; i++)
        {
            long bytesPerLayer = (long)maxSeqLen * geometry.KvStrideOf(i) * sizeof(ushort); // FP16
            CudaDriverApi.cuMemAlloc_v2(out _keys[i], (nuint)bytesPerLayer).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _values[i], (nuint)bytesPerLayer).ThrowOnError();
        }
    }

    /// <summary>
    /// Updates KV-cache from device pointers (used by <see cref="CudaTransformerModel"/>).
    /// </summary>
    /// <param name="keysDevice">Device pointer to new K data [seqLen, kvStride] FP16.</param>
    /// <param name="valuesDevice">Device pointer to new V data [seqLen, kvStride] FP16.</param>
    /// <param name="positions">Host-side positions for updating _currentLength.</param>
    /// <param name="seqLen">Number of new tokens.</param>
    /// <param name="layerIndex">Layer index.</param>
    /// <param name="stream">CUDA stream (currently unused — copies are synchronous).</param>
    internal void UpdateDevice(nint keysDevice, nint valuesDevice,
                                 ReadOnlySpan<int> positions, int seqLen,
                                 int layerIndex, nint stream)
    {
        long rowBytes = (long)Stride(layerIndex) * sizeof(ushort); // FP16 KV-cache

        // Detect contiguous positions for bulk copy (common case: prefill or sequential decode)
        bool contiguous = seqLen > 0;
        for (int i = 0; i < seqLen; i++)
        {
            if ((uint)positions[i] >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max KV-cache length {_maxSeqLen}.");
            if (i > 0 && positions[i] != positions[i - 1] + 1)
                contiguous = false;
        }

        if (contiguous && seqLen > 1)
        {
            // Bulk copy: single D2D transfer for all positions
            long bulkBytes = (long)seqLen * rowBytes;
            nint kDst = _keys[layerIndex] + (nint)(positions[0] * rowBytes);
            nint vDst = _values[layerIndex] + (nint)(positions[0] * rowBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, keysDevice, (nuint)bulkBytes, stream).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, valuesDevice, (nuint)bulkBytes, stream).ThrowOnError();
        }
        else
        {
            // Per-position copy (non-contiguous positions, e.g., prompt cache partial reuse)
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                nint kDst = _keys[layerIndex] + (nint)(pos * rowBytes);
                nint vDst = _values[layerIndex] + (nint)(pos * rowBytes);
                nint kSrc = keysDevice + (nint)(i * rowBytes);
                nint vSrc = valuesDevice + (nint)(i * rowBytes);
                CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kSrc, (nuint)rowBytes, stream).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vSrc, (nuint)rowBytes, stream).ThrowOnError();
            }
        }

        // Update length
        int maxPos = positions[seqLen - 1];
        for (int i = 0; i < seqLen; i++)
        {
            if (positions[i] > maxPos) maxPos = positions[i];
        }
        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    internal void UpdateDevicePositioned(nint keysDevice, nint valuesDevice,
                                           ReadOnlySpan<int> positions, int seqLen,
                                           int layerIndex, nint stream, CudaKernels kernels,
                                           nint positionsDevice)
    {
        if (seqLen != 1)
        {
            UpdateDevice(keysDevice, valuesDevice, positions, seqLen, layerIndex, stream);
            return;
        }

        int pos = positions[0];
        if ((uint)pos >= (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(positions),
                $"Position {pos} exceeds max KV-cache length {_maxSeqLen}.");

        kernels.LaunchKvCacheUpdatePos(
            keysDevice, valuesDevice, _keys[layerIndex], _values[layerIndex],
            positionsDevice, Stride(layerIndex), stream);

        int newLength = pos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    /// <summary>Returns device pointer to cached keys for the given layer.</summary>
    internal nint GetKeysPtr(int layerIndex) => _keys[layerIndex];

    /// <summary>Returns device pointer to cached values for the given layer.</summary>
    internal nint GetValuesPtr(int layerIndex) => _values[layerIndex];

    /// <summary>
    /// Single-token KV-cache write where the destination row index is read from a
    /// device-resident int. Used by the CUDA Graphs decode replay path: under graph
    /// capture, host-computed addresses (as in <see cref="UpdateDevice"/>) get baked
    /// into the graph at instantiate time and the next replay would clobber the
    /// same row. Writing through a device-side index keeps the graph topology
    /// invariant while still landing each token at the correct position.
    /// </summary>
    /// <param name="newKeyDevice">Device pointer to new K row (FP16, kvStride elements).</param>
    /// <param name="newValueDevice">Device pointer to new V row (FP16, kvStride elements).</param>
    /// <param name="layerIndex">Layer index.</param>
    /// <param name="posPtrDevice">Device pointer to a 4-byte int holding the absolute row index.</param>
    /// <param name="stream">CUDA stream.</param>
    /// <param name="kernels">Kernel dispatcher.</param>
    internal void UpdateDeviceSingleDevicePos(
        nint newKeyDevice, nint newValueDevice,
        int layerIndex, nint posPtrDevice, nint stream, CudaKernels kernels)
    {
        int stride = Stride(layerIndex);
        kernels.LaunchKvWriteOneF16(newKeyDevice, _keys[layerIndex], stride, posPtrDevice, stream);
        kernels.LaunchKvWriteOneF16(newValueDevice, _values[layerIndex], stride, posPtrDevice, stream);
    }

    /// <summary>
    /// Updates the host-side <see cref="CurrentLength"/> after a graph-launched
    /// decode step. The graph itself wrote to the cache at the device-side index;
    /// this just keeps the metadata in sync so subsequent prefill / non-graph
    /// callers see the right length.
    /// </summary>
    internal void AdvanceLengthForGraphDecode(int newLength)
    {
        if (newLength > _currentLength) _currentLength = newLength;
    }

    /// <summary>
    /// Decode-step fused RoPE + KV-cache write. Replaces a separate
    /// <see cref="CudaKernels.LaunchRoPE"/> followed by two
    /// <c>cuMemcpyDtoDAsync</c>s with a single launch.
    /// Q is rotated in place on <paramref name="qSrc"/>; rotated K is written to
    /// <c>K_cache[layer] + position * kvStride</c>; V is plain-copied to
    /// <c>V_cache[layer] + position * kvStride</c>. Updates host-side length.
    /// </summary>
    /// <param name="qSrc">Device pointer to Q (in-place rotation target).</param>
    /// <param name="kSrc">Device pointer to K projection scratch.</param>
    /// <param name="vSrc">Device pointer to V projection scratch.</param>
    /// <param name="positionsDevice">Device int[1] holding the RoPE position.</param>
    /// <param name="position">Host-side absolute position (used for cache row + length update).</param>
    /// <param name="layerIndex">Layer index.</param>
    /// <param name="numHeads">Number of Q heads.</param>
    /// <param name="numKvHeads">Number of KV heads.</param>
    /// <param name="headDim">Per-head dimension.</param>
    /// <param name="ropeDim">RoPE rotation width (≤ headDim).</param>
    /// <param name="ropeTheta">RoPE base.</param>
    /// <param name="ropeType">0 = standard pairs, 1 = NeoX split halves.</param>
    /// <param name="stream">CUDA stream.</param>
    /// <param name="kernels">Kernel dispatcher.</param>
    internal void FusedRopeAndUpdateDevice(
        nint qSrc, nint kSrc, nint vSrc,
        nint positionsDevice, int position,
        int layerIndex,
        int numHeads, int numKvHeads, int headDim,
        int ropeDim, float ropeTheta, int ropeType,
        nint stream, CudaKernels kernels)
    {
        if ((uint)position >= (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(position),
                $"Position {position} exceeds max KV-cache length {_maxSeqLen}.");

        kernels.LaunchFusedRopeKvWriteF16(
            qSrc, kSrc, vSrc,
            _keys[layerIndex], _values[layerIndex],
            positionsDevice, position,
            numHeads, numKvHeads, headDim,
            ropeDim, Stride(layerIndex), ropeTheta, ropeType,
            stream);

        int newLength = position + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    /// <summary>
    /// Returns true when the fused-RoPE+KV-write kernel is loaded on
    /// <paramref name="kernels"/>. Used to gate the fused decode path.
    /// </summary>
    internal static bool SupportsFusedRopeAndUpdate(CudaKernels kernels) =>
        kernels.HasFusedRopeKvWriteKernel;

    // ── IKvCache interface implementation ─────────────────────────────

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        throw new NotSupportedException("CudaKvCache.Update(ITensor) not supported. Use UpdateDevice().");
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        throw new NotSupportedException("CudaKvCache.Update(TensorRef) not supported. Use UpdateDevice().");
    }

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
    {
        throw new NotSupportedException("CudaKvCache.GetKeys(ITensor) not supported. Use GetKeysPtr().");
    }

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
    {
        throw new NotSupportedException("CudaKvCache.GetValues(ITensor) not supported. Use GetValuesPtr().");
    }

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex) =>
        new(_currentLength, Stride(layerIndex), DType.Float16, 0, _keys[layerIndex]);

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex) =>
        new(_currentLength, Stride(layerIndex), DType.Float16, 0, _values[layerIndex]);

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
        WasRolledBack = true;
    }

    /// <summary>
    /// Set on the first <see cref="Rollback"/> call against this cache.
    /// <see cref="CudaTransformerModel"/>'s graph-replay fast path consults
    /// this flag to guard against the silent-corruption case where a captured
    /// <c>cuGraphExec</c> would replay with a stale device-side write-pos.
    /// Speculative decoding is the only Rollback caller today and is currently
    /// restricted to greedy (non-graph) — this flag exists so a future
    /// speculative+graph combination fails fast at the graph-launch site.
    /// </summary>
    public bool WasRolledBack { get; private set; }

    /// <inheritdoc/>
    public void Dispose()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            if (_keys[i] != 0) { CudaDriverApi.cuMemFree_v2(_keys[i]); _keys[i] = 0; }
            if (_values[i] != 0) { CudaDriverApi.cuMemFree_v2(_values[i]); _values[i] = 0; }
        }
    }
}
