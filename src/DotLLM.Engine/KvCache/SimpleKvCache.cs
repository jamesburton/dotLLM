using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine.KvCache;

/// <summary>
/// Simple pre-allocated KV-cache. Stores per-layer K and V projections in contiguous
/// FP32 buffers of shape <c>[maxSeqLen, numKvHeads * headDim]</c>, 64-byte aligned.
/// A geometry with sliding windows (<see cref="KvGeometry.HasSlidingWindows"/>) keeps only the recent rows of those
/// layers (<see cref="IWindowedKvCache"/>, issue #1169).
/// </summary>
public sealed unsafe class SimpleKvCache : IKvCache, IPerLayerKvCache, IWindowedKvCache
{
    /// <inheritdoc/>
    public bool SupportsExactF32PrefixCopy => _rows is null;

    private readonly nint[] _keys;   // [numLayers] pointers to K buffers
    private readonly nint[] _values; // [numLayers] pointers to V buffers
    private readonly int _numLayers;
    private readonly KvGeometry _geom;   // per-layer KV row width (numKvHeads(l) * headDim(l))
    private readonly bool _uniform;      // _geom.IsUniform, hoisted for the hot path
    private readonly int _uniformStride; // _geom.UniformStride when _uniform; else 0
    private readonly SlidingWindowRows? _rows; // null unless the geometry has compact sliding-window layers
    private int _maxSeqLen;
    private int _currentLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>Total bytes allocated for KV-cache buffers.</summary>
    public long AllocatedBytes
    {
        get
        {
            if (_rows is not null)
                return _rows.AllocatedBytes(i => 2L * _geom.KvStrideOf(i) * sizeof(float));
            long rows = 0;
            for (int i = 0; i < _numLayers; i++)
                rows += _geom.KvStrideOf(i);
            return rows * 2 * _maxSeqLen * sizeof(float);
        }
    }

    /// <summary>
    /// Creates a new KV-cache with pre-allocated buffers for all layers, all using a
    /// single uniform per-layer stride (<c>numKvHeads * headDim</c>). Byte-identical
    /// to the per-layer constructor with <see cref="KvGeometry.Uniform"/>.
    /// </summary>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numKvHeads">Number of KV attention heads per layer.</param>
    /// <param name="headDim">Dimension per attention head.</param>
    /// <param name="maxSeqLen">Maximum number of positions this cache can hold.</param>
    /// <exception cref="ArgumentOutOfRangeException">A dimension is not positive.</exception>
    /// <exception cref="OverflowException">The row stride or a layer's byte count overflows.</exception>
    public SimpleKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen)
        : this(KvGeometry.Uniform(numLayers, numKvHeads, headDim), maxSeqLen)
    {
    }

    /// <summary>
    /// Creates a new KV-cache with pre-allocated, per-layer-strided buffers. For a
    /// uniform geometry this is byte-identical to the scalar constructor; for Gemma-4
    /// (distinct sliding vs global KV row widths) each layer's K/V buffer is sized to
    /// that layer's stride so the two layer classes are addressed correctly.
    /// </summary>
    /// <param name="geometry">Per-layer KV-cache geometry (see <see cref="KvGeometry.FromConfig(DotLLM.Core.Models.ModelConfig)"/>).</param>
    /// <param name="maxSeqLen">Maximum number of positions this cache can hold.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxSeqLen"/> is not positive.</exception>
    /// <exception cref="OverflowException">A layer's buffer byte count does not fit in 64 bits.</exception>
    public SimpleKvCache(KvGeometry geometry, int maxSeqLen)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);

        // Validate before recording a layer count, so a rejected construction leaves the finalizer
        // nothing to walk. Window rows never exceed maxSeqLen, so this bounds every allocation below.
        for (int i = 0; i < geometry.LayerCount; i++)
            _ = checked((long)maxSeqLen * geometry.KvStrideOf(i) * sizeof(float));

        _numLayers = geometry.LayerCount;
        _geom = geometry;
        _uniform = geometry.IsUniform;
        _uniformStride = geometry.IsUniform ? geometry.UniformStride : 0;
        _maxSeqLen = maxSeqLen;

        _keys = new nint[_numLayers];
        _values = new nint[_numLayers];
        if (geometry.HasSlidingWindows)
            _rows = new SlidingWindowRows(geometry, maxSeqLen);

        for (int i = 0; i < _numLayers; i++)
        {
            int rows = _rows?.CapacityOf(i) ?? maxSeqLen;
            nuint bufferBytes = (nuint)((long)rows * geometry.KvStrideOf(i) * sizeof(float));
            _keys[i] = (nint)NativeMemory.AlignedAlloc(bufferBytes, 64);
            _values[i] = (nint)NativeMemory.AlignedAlloc(bufferBytes, 64);
        }
    }

    /// <summary>Releases unmanaged buffers if <see cref="Dispose()"/> was not called.</summary>
    ~SimpleKvCache() => Dispose(disposing: false);

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        int seqLen = positions.Length;
        int maxPos = -1;

        float* kSrc = (float*)keys.DataPointer;
        float* vSrc = (float*)values.DataPointer;
        float* kDst = (float*)_keys[layerIndex];
        float* vDst = (float*)_values[layerIndex];

        int stride = _uniform ? _uniformStride : _geom.KvStrideOf(layerIndex);
        int rowBytes = stride * sizeof(float);

        int rowBase = 0;
        if (_rows is not null && _rows.IsWindowed(layerIndex))
        {
            PrepareWindow(positions, layerIndex, rowBytes);
            kDst = (float*)_keys[layerIndex];
            vDst = (float*)_values[layerIndex];
            rowBase = _rows.BaseOf(layerIndex);
        }

        for (int i = 0; i < seqLen; i++)
        {
            int pos = positions[i];
            if ((uint)pos >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {pos} exceeds max cache length {_maxSeqLen}.");

            if (pos > maxPos) maxPos = pos;
            long row = pos - rowBase;

            Buffer.MemoryCopy(
                kSrc + i * stride,
                kDst + row * stride,
                rowBytes, rowBytes);

            Buffer.MemoryCopy(
                vSrc + i * stride,
                vDst + row * stride,
                rowBytes, rowBytes);
        }

        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        int stride = _uniform ? _uniformStride : _geom.KvStrideOf(layerIndex);
        var kRef = new TensorRef(positions.Length, stride, keys.DType, keys.DeviceId, keys.DataPointer);
        var vRef = new TensorRef(positions.Length, stride, values.DType, values.DeviceId, values.DataPointer);
        Update(kRef, vRef, positions, layerIndex);
    }

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef GetKeysRef(int layerIndex)
        => new(LiveRows(layerIndex), _uniform ? _uniformStride : _geom.KvStrideOf(layerIndex), DType.Float32, -1, _keys[layerIndex]);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef GetValuesRef(int layerIndex)
        => new(LiveRows(layerIndex), _uniform ? _uniformStride : _geom.KvStrideOf(layerIndex), DType.Float32, -1, _values[layerIndex]);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int RowBaseOf(int layerIndex) => _rows?.BaseOf(layerIndex) ?? 0;

    /// <summary>Rows held for <paramref name="layerIndex"/>: positions <c>[RowBaseOf, CurrentLength)</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int LiveRows(int layerIndex) => _currentLength - RowBaseOf(layerIndex);

    /// <summary>Compacts or resizes a windowed layer so <paramref name="positions"/> fit (see <see cref="SlidingWindowRows"/>).</summary>
    private void PrepareWindow(ReadOnlySpan<int> positions, int layerIndex, int rowBytes)
    {
        int lo = int.MaxValue, hi = -1;
        foreach (int p in positions) { lo = Math.Min(lo, p); hi = Math.Max(hi, p); }
        if (hi >= _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(positions), $"Position {hi} exceeds max cache length {_maxSeqLen}.");
        _rows!.Prepare(layerIndex, lo, hi, _currentLength, ref _keys[layerIndex], ref _values[layerIndex], rowBytes, rowBytes);
    }

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ITensor GetKeys(int layerIndex)
    {
        var shape = new TensorShape(LiveRows(layerIndex), _uniform ? _uniformStride : _geom.KvStrideOf(layerIndex));
        return new TensorView(shape, DType.Float32, -1, _keys[layerIndex]);
    }

    /// <summary>
    /// Number of layers in this cache.
    /// </summary>
    public int NumLayers => _numLayers;

    /// <inheritdoc/>
    int IPerLayerKvCache.LayerCount => _numLayers;

    /// <summary>
    /// Per-row stride (<c>numKvHeads * headDim</c>, FP32 elements) of layer 0. For a
    /// per-layer (Gemma-4) geometry use <see cref="KvStrideOf(int)"/> instead.
    /// </summary>
    public int KvStride => _uniform ? _uniformStride : _geom.KvStrideOf(0);

    /// <summary>
    /// Per-row stride (FP32 elements) for <paramref name="layerIndex"/>. Equals
    /// <see cref="KvStride"/> for every uniform model; differs per layer for Gemma-4.
    /// </summary>
    public int KvStrideOf(int layerIndex) => _geom.KvStrideOf(layerIndex);

    /// <summary>
    /// Returns a <see cref="ReadOnlySpan{T}"/> over the contiguous FP32 keys
    /// buffer for the given layer, covering positions <c>[0, CurrentLength)</c>.
    /// Used by the hybrid CPU-prefill / iGPU-decode handoff to upload prefill
    /// state into a GPU KV cache.
    /// </summary>
    public ReadOnlySpan<float> KeysSpan(int layerIndex)
    {
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        ThrowIfWindowed(layerIndex);
        int floats = _currentLength * (_uniform ? _uniformStride : _geom.KvStrideOf(layerIndex));
        return new ReadOnlySpan<float>((void*)_keys[layerIndex], floats);
    }

    /// <summary>
    /// Returns a <see cref="ReadOnlySpan{T}"/> over the contiguous FP32 values
    /// buffer for the given layer, covering positions <c>[0, CurrentLength)</c>.
    /// </summary>
    public ReadOnlySpan<float> ValuesSpan(int layerIndex)
    {
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        ThrowIfWindowed(layerIndex);
        int floats = _currentLength * (_uniform ? _uniformStride : _geom.KvStrideOf(layerIndex));
        return new ReadOnlySpan<float>((void*)_values[layerIndex], floats);
    }

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ITensor GetValues(int layerIndex)
    {
        var shape = new TensorShape(LiveRows(layerIndex), _uniform ? _uniformStride : _geom.KvStrideOf(layerIndex));
        return new TensorView(shape, DType.Float32, -1, _values[layerIndex]);
    }

    /// <summary>
    /// Resets the visible length of the cache to the given position.
    /// Data beyond this position remains in memory but will be overwritten.
    /// Used by prompt caching to truncate to the matched prefix length.
    /// </summary>
    internal void SetCurrentLength(int length)
    {
        if ((uint)length > (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(length));
        Truncate(length);
    }

    /// <summary>
    /// True when the cache can be cut back to <paramref name="length"/>: always for a full cache; for compact
    /// sliding-window layers only while they still hold the rows a query at that position needs.
    /// </summary>
    internal bool CanTruncateTo(int length) => _rows is null || _rows.CanTruncateTo(length);

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        Truncate(length);
    }

    private void Truncate(int length)
    {
        if (_rows is not null)
        {
            if (!_rows.CanTruncateTo(length))
                throw new InvalidOperationException(
                    $"Cannot cut the KV-cache back to {length}: a sliding-window layer already dropped rows that length needs "
                    + $"(margin {SlidingWindowRows.Margin} positions).");
            if (length == 0) _rows.Reset();
        }
        _currentLength = length;
    }

    private void ThrowIfWindowed(int layerIndex)
    {
        if (_rows is not null && _rows.IsWindowed(layerIndex))
            throw new NotSupportedException(
                $"Layer {layerIndex} keeps only its sliding-window rows; it has no position-0-based span. Build the cache with "
                + "KvGeometry.FromConfig(config) (no compact windows) for full-history consumers.");
    }

    /// <inheritdoc/>
    /// <remarks>Reallocates every layer's buffers and copies the committed rows (issue #1025). Pointers from earlier <c>Get*Ref</c> calls are invalidated.</remarks>
    public bool TryGrowTo(int maxLength)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (maxLength <= _maxSeqLen) return true;
        for (int i = 0; i < _numLayers; i++)
        {
            // Window buffers do not depend on the context length.
            if (_rows is not null && _rows.IsWindowed(i)) continue;
            long stride = _geom.KvStrideOf(i);
            nuint newBytes = (nuint)(maxLength * stride * sizeof(float));
            nuint keepBytes = (nuint)(_currentLength * stride * sizeof(float));
            _keys[i] = Regrow(_keys[i], newBytes, keepBytes);
            _values[i] = Regrow(_values[i], newBytes, keepBytes);
        }
        _maxSeqLen = maxLength;
        _rows?.SetFullCapacity(maxLength);
        return true;

        static nint Regrow(nint old, nuint newBytes, nuint keepBytes)
        {
            void* p = NativeMemory.AlignedAlloc(newBytes, 64);
            Buffer.MemoryCopy((void*)old, p, newBytes, keepBytes);
            NativeMemory.AlignedFree((void*)old);
            return (nint)p;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;

        for (int i = 0; i < _numLayers; i++)
        {
            if (_keys[i] != 0)
            {
                NativeMemory.AlignedFree((void*)_keys[i]);
                _keys[i] = 0;
            }
            if (_values[i] != 0)
            {
                NativeMemory.AlignedFree((void*)_values[i]);
                _values[i] = 0;
            }
        }
    }
}
