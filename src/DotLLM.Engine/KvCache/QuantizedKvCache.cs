using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine.KvCache;

/// <summary>
/// KV-cache with quantized storage (Q8_0 or Q4_0) and an optional full-precision window
/// for recent tokens. Uses a dual-region design:
/// <list type="bullet">
/// <item><b>Quantized buffer</b>: older tokens stored in Q8_0/Q4_0 format (append-only).</item>
/// <item><b>Full-precision ring buffer</b>: recent W tokens in FP32 (quantize-on-evict).</item>
/// </list>
/// When <c>windowSize == 0</c>, all tokens are quantized immediately (no ring buffer).
/// In that mode a geometry with sliding windows (<see cref="KvGeometry.HasSlidingWindows"/>) keeps only the recent rows
/// of those layers (<see cref="IWindowedKvCache"/>, issue #1169).
/// </summary>
public sealed unsafe class QuantizedKvCache : IQuantizedKvCache, IPerLayerKvCache, IWindowedKvCache, IPrefixReusableKvCache, IKvPrefixSeedable
{
    private const int BlockSize = 32;
    private const int Q8_0BlockBytes = QuantFormat.Q8_0BlockBytes;
    private const int Q4_0BlockBytes = QuantFormat.Q4_0BlockBytes;

    private readonly nint[] _keysQuant;     // [numLayers] quantized K buffers
    private readonly nint[] _valuesQuant;   // [numLayers] quantized V buffers
    private readonly nint[]? _keysWindow;   // [numLayers] FP32 ring buffers (null when windowSize=0)
    private readonly nint[]? _valuesWindow; // [numLayers] FP32 ring buffers (null when windowSize=0)
    private readonly int _numLayers;
    private readonly KvGeometry _geom;       // per-layer KV row width (numKvHeads(l) * headDim(l))
    private readonly bool _uniform;          // _geom.IsUniform, hoisted for the hot path
    private readonly int _uniformStride;     // _geom.UniformStride when _uniform; else 0
    private int _maxSeqLen;                  // raised by TryGrowTo (#1843)
    private readonly int _windowSize;
    private readonly int[] _keyQuantRowBytes;   // [numLayers] quantized K row bytes
    private readonly int[] _valueQuantRowBytes; // [numLayers] quantized V row bytes
    private readonly int[] _layerQuantizedLength; // per-layer eviction tracking
    // Compact sliding-window rows; pure quantized mode only. shortcut: the FP32-window mode keeps full rows for sliding
    // layers, upgrade if that mode is ever used with long-context SWA models (rule 1 says not to grow FP32 windows).
    private readonly SlidingWindowRows? _rows;
    private int _currentLength;
    private int _quantizedLength;
    private bool _disposed;

    /// <summary>Per-layer KV row width (FP32 elements); scalar shortcut hoisted when uniform.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int Stride(int layerIndex) => _uniform ? _uniformStride : _geom.KvStrideOf(layerIndex);

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <inheritdoc/>
    public int QuantizedLength => _quantizedLength;

    /// <inheritdoc/>
    public int WindowLength => _windowSize > 0
        ? Math.Min(_currentLength, _windowSize)
        : 0;

    /// <inheritdoc/>
    public int WindowCapacity => _windowSize;

    /// <inheritdoc/>
    public KvCacheDType KeyDType { get; }

    /// <inheritdoc/>
    public KvCacheDType ValueDType { get; }

    /// <inheritdoc/>
    public int KeyQuantizedRowBytes => _keyQuantRowBytes[0];

    /// <inheritdoc/>
    public int ValueQuantizedRowBytes => _valueQuantRowBytes[0];

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int KeyQuantizedRowBytesOf(int layerIndex) => _keyQuantRowBytes[layerIndex];

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int ValueQuantizedRowBytesOf(int layerIndex) => _valueQuantRowBytes[layerIndex];

    /// <inheritdoc/>
    int IPerLayerKvCache.LayerCount => _numLayers;

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int KvStrideOf(int layerIndex) => _geom.KvStrideOf(layerIndex);

    /// <summary>Total bytes allocated for all KV-cache buffers (quantized + window).</summary>
    public long AllocatedBytes => QuantRowBytesAllocated() + _windowBytes;

    private readonly long _windowBytes;

    /// <summary>
    /// Creates a new quantized KV-cache with a single uniform per-layer stride
    /// (<c>numKvHeads * headDim</c>). Byte-identical to the per-layer constructor with
    /// <see cref="KvGeometry.Uniform"/>.
    /// </summary>
    public QuantizedKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen,
                             KvCacheDType keyDType, KvCacheDType valueDType, int windowSize)
        : this(KvGeometry.Uniform(numLayers, numKvHeads, headDim), maxSeqLen,
               keyDType, valueDType, windowSize)
    {
    }

    /// <summary>
    /// Creates a new quantized KV-cache with per-layer-strided buffers. For a uniform
    /// geometry this is byte-identical to the scalar constructor; for Gemma-4 (distinct
    /// sliding vs global KV row widths) each layer's quantized + window buffers are sized
    /// to that layer's stride so the two layer classes are addressed correctly.
    /// </summary>
    public QuantizedKvCache(KvGeometry geometry, int maxSeqLen,
                             KvCacheDType keyDType, KvCacheDType valueDType, int windowSize)
    {
        if (keyDType == KvCacheDType.F32 && valueDType == KvCacheDType.F32)
            throw new ArgumentException("At least one of keyDType/valueDType must be quantized. Use SimpleKvCache for full precision.");
        if (windowSize < 0)
            throw new ArgumentOutOfRangeException(nameof(windowSize), windowSize, "Window size must be >= 0.");

        int numLayers = geometry.LayerCount;
        _numLayers = numLayers;
        _geom = geometry;
        _uniform = geometry.IsUniform;
        _uniformStride = geometry.IsUniform ? geometry.UniformStride : 0;
        _maxSeqLen = maxSeqLen;
        _windowSize = windowSize;
        KeyDType = keyDType;
        ValueDType = valueDType;

        _keyQuantRowBytes = new int[numLayers];
        _valueQuantRowBytes = new int[numLayers];
        _layerQuantizedLength = new int[numLayers];

        _keysQuant = new nint[numLayers];
        _valuesQuant = new nint[numLayers];
        if (windowSize == 0 && geometry.HasSlidingWindows)
            _rows = new SlidingWindowRows(geometry, maxSeqLen);

        long totalBytes = 0;

        // Allocate quantized buffers (sized for maxSeqLen — worst case all tokens quantized),
        // each at its own per-layer stride. The %BlockSize quantization constraint is checked
        // per layer (uniform models hit the exact same check on the exact same value).
        for (int i = 0; i < numLayers; i++)
        {
            int stride = geometry.KvStrideOf(i);
            if (stride % BlockSize != 0)
                throw new ArgumentException(
                    $"kvStride ({stride}) for layer {i} must be a multiple of {BlockSize} for quantization.",
                    nameof(geometry));

            _keyQuantRowBytes[i] = ComputeQuantRowBytes(stride, keyDType);
            _valueQuantRowBytes[i] = ComputeQuantRowBytes(stride, valueDType);

            long rows = _rows?.CapacityOf(i) ?? maxSeqLen;
            nuint kQuantBytes = (nuint)(rows * _keyQuantRowBytes[i]);
            nuint vQuantBytes = (nuint)(rows * _valueQuantRowBytes[i]);
            _keysQuant[i] = (nint)NativeMemory.AlignedAlloc(kQuantBytes, 64);
            _valuesQuant[i] = (nint)NativeMemory.AlignedAlloc(vQuantBytes, 64);
            totalBytes += (long)(kQuantBytes + vQuantBytes);
        }

        // Allocate window buffers (FP32), each at its own per-layer stride.
        if (windowSize > 0)
        {
            _keysWindow = new nint[numLayers];
            _valuesWindow = new nint[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                nuint windowBytes = (nuint)((long)windowSize * geometry.KvStrideOf(i) * sizeof(float));
                _keysWindow[i] = (nint)NativeMemory.AlignedAlloc(windowBytes, 64);
                _valuesWindow[i] = (nint)NativeMemory.AlignedAlloc(windowBytes, 64);
                totalBytes += (long)(windowBytes * 2);
            }
        }

        _windowBytes = totalBytes - QuantRowBytesAllocated();
    }

    /// <summary>Bytes held by the quantized row buffers (window buffers excluded).</summary>
    private long QuantRowBytesAllocated()
    {
        if (_rows is not null)
            return _rows.AllocatedBytes(i => (long)_keyQuantRowBytes[i] + _valueQuantRowBytes[i]);
        long total = 0;
        for (int i = 0; i < _numLayers; i++)
            total += (long)_maxSeqLen * (_keyQuantRowBytes[i] + _valueQuantRowBytes[i]);
        return total;
    }

    /// <summary>Releases unmanaged buffers if <see cref="Dispose()"/> was not called.</summary>
    ~QuantizedKvCache() => Dispose(disposing: false);

    /// <inheritdoc/>
    [SkipLocalsInit]
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        int seqLen = positions.Length;
        float* kSrc = (float*)keys.DataPointer;
        float* vSrc = (float*)values.DataPointer;
        int stride = Stride(layerIndex);
        int keyRowBytes = _keyQuantRowBytes[layerIndex];
        int valueRowBytes = _valueQuantRowBytes[layerIndex];
        int fpRowBytes = stride * sizeof(float);

        // Compute new sequence length (idempotent across layer calls with same positions).
        int maxPos = positions[0];
        for (int i = 1; i < seqLen; i++)
            if (positions[i] > maxPos) maxPos = positions[i];
        int newLength = maxPos + 1;

        if (_windowSize > 0)
        {
            int prevQuantLen = _layerQuantizedLength[layerIndex];
            int newQuantLen = Math.Max(0, newLength - _windowSize);

            // Phase 1: evict ring slots that hold data from PREVIOUS calls (safe to read).
            // When seqLen > windowSize, some eviction positions fall in the current batch —
            // those ring slots have not been written yet (or hold stale data), so we must not
            // read them. Evict only up to the first position in the current batch.
            int minNewPos = positions[0];
            for (int i = 1; i < seqLen; i++)
                if (positions[i] < minNewPos) minNewPos = positions[i];
            int safeEvictEnd = Math.Min(newQuantLen, minNewPos);

            for (int evictPos = prevQuantLen; evictPos < safeEvictEnd; evictPos++)
            {
                int ringIdx = evictPos % _windowSize;
                QuantizeRow((float*)_keysWindow![layerIndex] + ringIdx * stride,
                            (byte*)_keysQuant[layerIndex] + (long)evictPos * keyRowBytes,
                            stride, KeyDType);
                QuantizeRow((float*)_valuesWindow![layerIndex] + ringIdx * stride,
                            (byte*)_valuesQuant[layerIndex] + (long)evictPos * valueRowBytes,
                            stride, ValueDType);
            }

            // Phase 2: positions from the current batch that will be evicted immediately
            // (pos < newQuantLen) — quantize directly from the source data, skip the ring.
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                if ((uint)pos >= (uint)_maxSeqLen)
                    throw new ArgumentOutOfRangeException(nameof(positions),
                        $"Position {pos} exceeds max cache length {_maxSeqLen}.");

                if (pos < newQuantLen)
                {
                    QuantizeRow(kSrc + i * stride,
                                (byte*)_keysQuant[layerIndex] + (long)pos * keyRowBytes,
                                stride, KeyDType);
                    QuantizeRow(vSrc + i * stride,
                                (byte*)_valuesQuant[layerIndex] + (long)pos * valueRowBytes,
                                stride, ValueDType);
                }
            }

            _layerQuantizedLength[layerIndex] = newQuantLen;

            // Phase 3: positions that stay in the window — write FP32 into the ring buffer.
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                if (pos >= newQuantLen)
                {
                    int ringIdx = pos % _windowSize;
                    Buffer.MemoryCopy(
                        kSrc + i * stride,
                        (float*)_keysWindow![layerIndex] + ringIdx * stride,
                        fpRowBytes, fpRowBytes);
                    Buffer.MemoryCopy(
                        vSrc + i * stride,
                        (float*)_valuesWindow![layerIndex] + ringIdx * stride,
                        fpRowBytes, fpRowBytes);
                }
            }

            _quantizedLength = newQuantLen;
        }
        else
        {
            // Pure quantized mode: quantize directly at each position (position-addressed; a compact sliding-window
            // layer addresses row pos - base).
            int rowBase = 0;
            if (_rows is not null && _rows.IsWindowed(layerIndex))
            {
                int minPos = positions[0];
                for (int i = 1; i < seqLen; i++)
                    if (positions[i] < minPos) minPos = positions[i];
                if (maxPos >= _maxSeqLen)
                    throw new ArgumentOutOfRangeException(nameof(positions),
                        $"Position {maxPos} exceeds max cache length {_maxSeqLen}.");
                _rows.Prepare(layerIndex, minPos, maxPos, _currentLength,
                    ref _keysQuant[layerIndex], ref _valuesQuant[layerIndex], keyRowBytes, valueRowBytes);
                rowBase = _rows.BaseOf(layerIndex);
            }

            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                if ((uint)pos >= (uint)_maxSeqLen)
                    throw new ArgumentOutOfRangeException(nameof(positions),
                        $"Position {pos} exceeds max cache length {_maxSeqLen}.");
                long row = pos - rowBase;

                QuantizeRow(kSrc + i * stride,
                            (byte*)_keysQuant[layerIndex] + row * keyRowBytes,
                            stride, KeyDType);
                QuantizeRow(vSrc + i * stride,
                            (byte*)_valuesQuant[layerIndex] + row * valueRowBytes,
                            stride, ValueDType);
            }

            _quantizedLength = newLength;
        }

        _currentLength = newLength;
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        int stride = Stride(layerIndex);
        var kRef = new TensorRef(positions.Length, stride, keys.DType, keys.DeviceId, keys.DataPointer);
        var vRef = new TensorRef(positions.Length, stride, values.DType, values.DeviceId, values.DataPointer);
        Update(kRef, vRef, positions, layerIndex);
    }

    /// <inheritdoc/>
    /// <remarks>For a compact sliding-window layer the quantized rows start at this position, not 0.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int RowBaseOf(int layerIndex) => _rows?.BaseOf(layerIndex) ?? 0;

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public nint GetQuantizedKeysPtr(int layerIndex) => _keysQuant[layerIndex];

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public nint GetQuantizedValuesPtr(int layerIndex) => _valuesQuant[layerIndex];

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public nint GetWindowKeysPtr(int layerIndex)
        => _keysWindow != null ? _keysWindow[layerIndex] : 0;

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public nint GetWindowValuesPtr(int layerIndex)
        => _valuesWindow != null ? _valuesWindow[layerIndex] : 0;

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef GetKeysRef(int layerIndex)
    {
        // Return window data if available, otherwise empty ref.
        if (_keysWindow != null && WindowLength > 0)
            return new TensorRef(WindowLength, Stride(layerIndex), DType.Float32, -1, _keysWindow[layerIndex]);
        return default;
    }

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef GetValuesRef(int layerIndex)
    {
        if (_valuesWindow != null && WindowLength > 0)
            return new TensorRef(WindowLength, Stride(layerIndex), DType.Float32, -1, _valuesWindow[layerIndex]);
        return default;
    }

    /// <summary>
    /// Whether a rollback to <paramref name="length"/> keeps every row below it (<see cref="IPrefixReusableKvCache"/>, issue #1843:
    /// the CPU side of CUDA's and Vulkan's quantized prefix reuse). With a full-precision window, only before the first row was
    /// evicted to quantized storage: past that the ring already holds later rows, so a cut would leave the window wrong (the CUDA
    /// <c>CudaQuantizedKvCache</c> rule). Compact sliding-window layers must still hold the rows the cut needs.
    /// </summary>
    public bool CanTruncateTo(int length)
        => (uint)length <= (uint)_currentLength
           && (_windowSize == 0 || _quantizedLength == 0)
           && (_rows?.CanTruncateTo(length) ?? true);

    /// <summary>
    /// Raises <see cref="MaxLength"/> to <paramref name="maxLength"/> keeping the committed rows (issue #1843, as
    /// <see cref="SimpleKvCache.TryGrowTo"/>): each full-history layer's quantized K and V buffers are reallocated and the held
    /// rows copied as stored (no dequant). The FP32 window ring and compact sliding-window layers do not depend on the length.
    /// Host memory, so it never refuses.
    /// </summary>
    public bool TryGrowTo(int maxLength)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (maxLength <= _maxSeqLen) return true;
        for (int i = 0; i < _numLayers; i++)
        {
            if (_rows is not null && _rows.IsWindowed(i)) continue;
            _keysQuant[i] = Regrow(_keysQuant[i], (long)maxLength * _keyQuantRowBytes[i], (long)_currentLength * _keyQuantRowBytes[i]);
            _valuesQuant[i] = Regrow(_valuesQuant[i], (long)maxLength * _valueQuantRowBytes[i], (long)_currentLength * _valueQuantRowBytes[i]);
        }
        _maxSeqLen = maxLength;
        _rows?.SetFullCapacity(maxLength);
        return true;

        static nint Regrow(nint old, long newBytes, long keepBytes)
        {
            void* p = NativeMemory.AlignedAlloc((nuint)newBytes, 64);
            Buffer.MemoryCopy((void*)old, p, newBytes, keepBytes);
            NativeMemory.AlignedFree((void*)old);
            return (nint)p;
        }
    }

    /// <summary>
    /// Copies rows <c>[0, rows)</c> of every layer from another <see cref="QuantizedKvCache"/> with the same geometry and K/V types
    /// (the #1780 protected-entry seed, issue #1843), as stored. Refused for a full-precision window or compact sliding-window
    /// layers (as on CUDA), another type or geometry, too few source rows, or too short a cache; this cache is then unchanged.
    /// </summary>
    public bool TrySeedPrefixFrom(IKvCache source, int rows)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (source is not QuantizedKvCache src || ReferenceEquals(src, this) || src._disposed) return false;
        if (_windowSize != 0 || src._windowSize != 0 || _rows is not null || src._rows is not null) return false;
        if (src.KeyDType != KeyDType || src.ValueDType != ValueDType || src._numLayers != _numLayers) return false;
        if (rows <= 0 || rows > src._currentLength || rows > _maxSeqLen) return false;
        for (int i = 0; i < _numLayers; i++)
            if (src._geom.KvStrideOf(i) != _geom.KvStrideOf(i)) return false;
        for (int i = 0; i < _numLayers; i++)
        {
            Buffer.MemoryCopy((void*)src._keysQuant[i], (void*)_keysQuant[i], (long)rows * _keyQuantRowBytes[i], (long)rows * _keyQuantRowBytes[i]);
            Buffer.MemoryCopy((void*)src._valuesQuant[i], (void*)_valuesQuant[i], (long)rows * _valueQuantRowBytes[i], (long)rows * _valueQuantRowBytes[i]);
        }
        _currentLength = rows;
        _quantizedLength = rows;
        return true;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// With a full-precision window, appending past the window evicts rows into quantized storage, after which a cut to the entry
    /// length is wrong (see <see cref="CanTruncateTo"/>); compact sliding-window layers are judged by
    /// <see cref="SlidingWindowRows.CanTruncateToAfterAppend"/>.
    /// </remarks>
    public bool CanRollbackAfterAppend(int minPos, int maxPos)
    {
        int newLength = Math.Max(_currentLength, maxPos + 1);
        if (newLength > _currentLength && _windowSize > 0 && Math.Max(_quantizedLength, newLength - _windowSize) > 0) return false;
        return _rows?.CanTruncateToAfterAppend(_currentLength, minPos, maxPos) ?? true;
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        if (_rows is not null)
        {
            if (!_rows.CanTruncateTo(length))
                throw new InvalidOperationException(
                    $"Cannot roll the KV-cache back to {length}: a sliding-window layer already dropped rows that length needs "
                    + $"(margin {SlidingWindowRows.Margin} positions).");
            if (length == 0) _rows.Reset();
        }
        _currentLength = length;
        if (_quantizedLength > length)
            _quantizedLength = length;
    }

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex) =>
        throw new NotSupportedException("Use IQuantizedKvCache methods for quantized cache access.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex) =>
        throw new NotSupportedException("Use IQuantizedKvCache methods for quantized cache access.");

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
            if (_keysQuant[i] != 0) { NativeMemory.AlignedFree((void*)_keysQuant[i]); _keysQuant[i] = 0; }
            if (_valuesQuant[i] != 0) { NativeMemory.AlignedFree((void*)_valuesQuant[i]); _valuesQuant[i] = 0; }

            if (_keysWindow != null && _keysWindow[i] != 0) { NativeMemory.AlignedFree((void*)_keysWindow[i]); _keysWindow[i] = 0; }
            if (_valuesWindow != null && _valuesWindow[i] != 0) { NativeMemory.AlignedFree((void*)_valuesWindow[i]); _valuesWindow[i] = 0; }
        }
    }

    // ──────────────────── Inline scalar quantization ────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ComputeQuantRowBytes(int kvStride, KvCacheDType dtype) => dtype switch
    {
        KvCacheDType.F32 => kvStride * sizeof(float), // should not happen, but safe
        KvCacheDType.Q8_0 => kvStride / BlockSize * Q8_0BlockBytes,
        KvCacheDType.Q4_0 => kvStride / BlockSize * Q4_0BlockBytes,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype))
    };

    /// <summary>
    /// Quantizes a single FP32 row to Q8_0 or Q4_0. Simple scalar implementation
    /// since this runs once per eviction (not on the hot attention path).
    /// </summary>
    [SkipLocalsInit]
    private static void QuantizeRow(float* src, byte* dest, int elementCount, KvCacheDType dtype)
    {
        if (dtype == KvCacheDType.F32) return; // no-op
        if (dtype == KvCacheDType.Q8_0)
            QuantizeRowQ8_0(src, dest, elementCount);
        else
            QuantizeRowQ4_0(src, dest, elementCount);
    }

    [SkipLocalsInit]
    private static void QuantizeRowQ8_0(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;
        for (int block = 0; block < blockCount; block++)
        {
            float* bs = src + block * BlockSize;
            byte* bd = dest + block * Q8_0BlockBytes;

            float maxAbs = 0;
            for (int i = 0; i < BlockSize; i++)
            {
                float abs = MathF.Abs(bs[i]);
                if (abs > maxAbs) maxAbs = abs;
            }

            float scale = maxAbs / 127.0f;
            Unsafe.WriteUnaligned(bd, (Half)scale);
            sbyte* qs = (sbyte*)(bd + 2);

            if (scale == 0)
            {
                for (int i = 0; i < BlockSize; i++) qs[i] = 0;
            }
            else
            {
                float invScale = 1.0f / scale;
                for (int i = 0; i < BlockSize; i++)
                    qs[i] = (sbyte)Math.Clamp((int)MathF.Round(bs[i] * invScale), -127, 127);
            }
        }
    }

    [SkipLocalsInit]
    private static void QuantizeRowQ4_0(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;
        for (int block = 0; block < blockCount; block++)
        {
            float* bs = src + block * BlockSize;
            byte* bd = dest + block * Q4_0BlockBytes;

            float maxAbs = 0;
            for (int i = 0; i < BlockSize; i++)
            {
                float abs = MathF.Abs(bs[i]);
                if (abs > maxAbs) maxAbs = abs;
            }

            float d = maxAbs / 7.0f;
            Unsafe.WriteUnaligned(bd, (Half)d);
            byte* qs = bd + 2;

            if (d == 0)
            {
                for (int j = 0; j < 16; j++) qs[j] = 0x88;
            }
            else
            {
                float invD = 1.0f / d;
                for (int j = 0; j < 16; j++)
                {
                    int lo = Math.Clamp((int)MathF.Round(bs[2 * j] * invD) + 8, 0, 15);
                    int hi = Math.Clamp((int)MathF.Round(bs[2 * j + 1] * invD) + 8, 0, 15);
                    qs[j] = (byte)((hi << 4) | lo);
                }
            }
        }
    }
}
