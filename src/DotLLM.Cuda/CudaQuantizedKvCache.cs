using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU-resident quantized KV-cache with dual-region storage:
/// quantized buffer (Q8_0/Q4_0 in device memory) + FP16 window for recent tokens.
/// <para>
/// Attention uses a temporary FP16 scratch buffer: dequant quantized region → scratch,
/// then combine with window data and call the regular attention kernel.
/// Memory savings come from permanent quantized storage between attention calls.
/// </para>
/// </summary>
public sealed class CudaQuantizedKvCache : IQuantizedKvCache
{
    private const int BlockSize = 32;
    private const int Q8_0BlockBytes = 34;
    private const int Q4_0BlockBytes = 18;

    private readonly nint[] _keysQuant;     // device ptrs: quantized K
    private readonly nint[] _valuesQuant;   // device ptrs: quantized V
    private readonly nint[]? _keysWindow;   // device ptrs: FP16 K window
    private readonly nint[]? _valuesWindow; // device ptrs: FP16 V window
    private readonly int _numLayers;
    private readonly int _kvStride;
    private readonly int _maxSeqLen;
    private readonly int _windowSize;
    private readonly int _keyQuantRowBytes;
    private readonly int _valueQuantRowBytes;
    private readonly int[] _layerQuantizedLength; // per-layer eviction tracking
    private int _currentLength;
    private int _quantizedLength;

    // Scratch buffers for dequantized K/V during attention (one pair, reused across layers)
    private nint _kScratch;  // device ptr: FP16 [maxSeqLen, kvStride]
    private nint _vScratch;  // device ptr: FP16 [maxSeqLen, kvStride]

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <inheritdoc/>
    public int QuantizedLength => _quantizedLength;

    /// <inheritdoc/>
    public int WindowLength => _windowSize > 0 ? Math.Min(_currentLength, _windowSize) : 0;

    /// <inheritdoc/>
    public int WindowCapacity => _windowSize;

    /// <inheritdoc/>
    public KvCacheDType KeyDType { get; }

    /// <inheritdoc/>
    public KvCacheDType ValueDType { get; }

    /// <inheritdoc/>
    public int KeyQuantizedRowBytes => _keyQuantRowBytes;

    /// <inheritdoc/>
    public int ValueQuantizedRowBytes => _valueQuantRowBytes;

    /// <summary>Total device memory allocated in bytes.</summary>
    public long AllocatedBytes { get; }

    /// <summary>
    /// Creates a GPU quantized KV-cache.
    /// </summary>
    public CudaQuantizedKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen,
                                 KvCacheConfig config)
    {
        _numLayers = numLayers;
        _kvStride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;
        _windowSize = config.MixedPrecisionWindowSize;
        KeyDType = config.KeyDType;
        ValueDType = config.ValueDType;

        if (_kvStride % BlockSize != 0)
            throw new ArgumentException(
                $"kvStride ({_kvStride}) must be a multiple of {BlockSize} for quantization.");
        System.Diagnostics.Debug.Assert(_kvStride % BlockSize == 0,
            $"kvStride ({_kvStride}) must be a multiple of {BlockSize}");

        _keyQuantRowBytes = ComputeQuantRowBytes(_kvStride, config.KeyDType);
        _valueQuantRowBytes = ComputeQuantRowBytes(_kvStride, config.ValueDType);
        _layerQuantizedLength = new int[numLayers];

        _keysQuant = new nint[numLayers];
        _valuesQuant = new nint[numLayers];

        long totalBytes = 0;

        // Allocate quantized buffers
        for (int i = 0; i < numLayers; i++)
        {
            nuint kBytes = (nuint)((long)maxSeqLen * _keyQuantRowBytes);
            nuint vBytes = (nuint)((long)maxSeqLen * _valueQuantRowBytes);
            CudaDriverApi.cuMemAlloc_v2(out _keysQuant[i], kBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _valuesQuant[i], vBytes).ThrowOnError();
            totalBytes += (long)(kBytes + vBytes);
        }

        // Allocate FP16 window buffers
        if (_windowSize > 0)
        {
            _keysWindow = new nint[numLayers];
            _valuesWindow = new nint[numLayers];
            nuint windowBytes = (nuint)((long)_windowSize * _kvStride * sizeof(ushort));
            for (int i = 0; i < numLayers; i++)
            {
                CudaDriverApi.cuMemAlloc_v2(out _keysWindow[i], windowBytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out _valuesWindow[i], windowBytes).ThrowOnError();
                totalBytes += (long)(windowBytes * 2);
            }
        }

        // Allocate scratch buffers for attention dequant (one pair, reused across layers)
        long scratchBytes = (long)maxSeqLen * _kvStride * sizeof(ushort);
        CudaDriverApi.cuMemAlloc_v2(out _kScratch, (nuint)scratchBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _vScratch, (nuint)scratchBytes).ThrowOnError();
        totalBytes += scratchBytes * 2;

        AllocatedBytes = totalBytes;
    }

    /// <summary>
    /// Updates KV-cache from device FP16 data. Handles quantize-on-evict for the window.
    /// Position-addressed and idempotent: safe to call once per layer with the same positions.
    /// </summary>
    internal void UpdateDevice(nint keysDevice, nint valuesDevice,
                                ReadOnlySpan<int> positions, int seqLen,
                                int layerIndex, nint stream, CudaKernels kernels)
    {
        long fp16RowBytes = (long)_kvStride * sizeof(ushort);

        // Compute new sequence length (idempotent across layer calls with same positions).
        int maxPos = positions[0];
        for (int i = 1; i < seqLen; i++)
            if (positions[i] > maxPos) maxPos = positions[i];
        int newLength = maxPos + 1;

        if (_windowSize > 0)
        {
            // Per-layer eviction: each layer independently tracks how far it has evicted.
            int prevQuantLen = _layerQuantizedLength[layerIndex];
            int newQuantLen = Math.Max(0, newLength - _windowSize);

            for (int evictPos = prevQuantLen; evictPos < newQuantLen; evictPos++)
            {
                int ringIdx = evictPos % _windowSize;

                nint evictedK = _keysWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint quantDstK = _keysQuant[layerIndex] + (nint)((long)evictPos * _keyQuantRowBytes);
                kernels.LaunchQuantKv(evictedK, quantDstK, _kvStride, KeyDType, stream);

                nint evictedV = _valuesWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint quantDstV = _valuesQuant[layerIndex] + (nint)((long)evictPos * _valueQuantRowBytes);
                kernels.LaunchQuantKv(evictedV, quantDstV, _kvStride, ValueDType, stream);
            }

            _layerQuantizedLength[layerIndex] = newQuantLen;

            // Write new FP16 data into window ring buffer (position-addressed, idempotent).
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                int ringIdx = pos % _windowSize;
                nint kDst = _keysWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint vDst = _valuesWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint kSrc = keysDevice + (nint)(i * fp16RowBytes);
                nint vSrc = valuesDevice + (nint)(i * fp16RowBytes);

                CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kSrc, (nuint)fp16RowBytes, stream).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vSrc, (nuint)fp16RowBytes, stream).ThrowOnError();
            }

            _quantizedLength = newQuantLen;
        }
        else
        {
            // Pure quantized: quantize directly at each position (position-addressed).
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                nint kSrc = keysDevice + (nint)(i * fp16RowBytes);
                nint vSrc = valuesDevice + (nint)(i * fp16RowBytes);
                nint kDst = _keysQuant[layerIndex] + (nint)((long)pos * _keyQuantRowBytes);
                nint vDst = _valuesQuant[layerIndex] + (nint)((long)pos * _valueQuantRowBytes);

                kernels.LaunchQuantKv(kSrc, kDst, _kvStride, KeyDType, stream);
                kernels.LaunchQuantKv(vSrc, vDst, _kvStride, ValueDType, stream);
            }

            _quantizedLength = newLength;
        }

        _currentLength = newLength;
    }

    /// <summary>
    /// Graph-capture variant of <see cref="UpdateDevice"/>. All host-observable state
    /// (eviction counters, ring offsets) is moved device-side via predicated kernels
    /// that read the absolute decode position from <paramref name="posPtrDevice"/>.
    /// <para>
    /// Per layer, three launches:
    /// <list type="number">
    /// <item><description><c>kv_write_one_f16_ring</c> for K and V — writes the new FP16 row
    /// into the per-layer ring at <c>pos % windowSize</c>.</description></item>
    /// <item><description><c>quant_f16_to_q{8,4}_0_dyn</c> for K and V — predicated; quantizes the
    /// row that just fell out of the window (<c>evict_pos = pos - windowSize</c>). No-op
    /// while the window is still filling.</description></item>
    /// </list>
    /// Only valid when <see cref="WindowCapacity"/> &gt; 0 (mixed-precision window mode).
    /// Pure-quantized mode (no FP16 window) is rare and stays on the eager path.
    /// </para>
    /// </summary>
    internal void UpdateDeviceForGraph(nint keysDevice, nint valuesDevice,
                                         int layerIndex, nint posPtrDevice,
                                         nint stream, CudaKernels kernels)
    {
        if (_windowSize <= 0)
            throw new InvalidOperationException(
                "UpdateDeviceForGraph requires a mixed-precision window (windowSize > 0).");

        // Step 1 (must run BEFORE the ring write): predicated quantize-on-evict.
        // Reads the FP16 row at ring slot `(pos - windowSize) % windowSize` — which is
        // about to be overwritten by the new write below — and lands it in the Q-cache
        // at row index `evict_pos = pos - windowSize`. No-op while pos < windowSize.
        // Critical ordering: if we wrote first then quantized, we'd quantize the NEW
        // row's bytes into the OLD row's Q-cache slot, corrupting attention reads.
        kernels.LaunchQuantKvDyn(
            _keysWindow![layerIndex], _keysQuant[layerIndex],
            _kvStride, _windowSize, KeyDType, posPtrDevice, stream);
        kernels.LaunchQuantKvDyn(
            _valuesWindow![layerIndex], _valuesQuant[layerIndex],
            _kvStride, _windowSize, ValueDType, posPtrDevice, stream);

        // Step 2: write the new FP16 row into the per-layer ring buffer at
        // slot `pos % windowSize`. After eviction has read the old contents.
        kernels.LaunchKvWriteOneF16Ring(
            keysDevice, _keysWindow[layerIndex], _kvStride, _windowSize, posPtrDevice, stream);
        kernels.LaunchKvWriteOneF16Ring(
            valuesDevice, _valuesWindow[layerIndex], _kvStride, _windowSize, posPtrDevice, stream);
    }

    /// <summary>
    /// Graph-capture variant of <see cref="PrepareAttentionScratch"/>. Two launches per
    /// K/V: <c>kv_dequant_q{8,4}_0_dyn</c> dequantizes the live quantized prefix into
    /// <see cref="_kScratch"/> / <see cref="_vScratch"/>; <c>kv_window_to_scratch_dyn</c>
    /// scatters the FP16 ring contents into the contiguous tail of the same scratch.
    /// Both kernels read the decode position from <paramref name="posPtrDevice"/> — the
    /// "how many quantized rows" / "where does the window start" arithmetic happens
    /// device-side, so the launch topology is identical across decode steps.
    /// </summary>
    internal (nint kPtr, nint vPtr) PrepareAttentionScratchForGraph(int layerIndex,
                                                                      nint posPtrDevice,
                                                                      nint stream,
                                                                      CudaKernels kernels)
    {
        if (_windowSize <= 0)
            throw new InvalidOperationException(
                "PrepareAttentionScratchForGraph requires a mixed-precision window.");

        // Phase 1: dequant the [0, quantizedLength) prefix (predicated; no-op until window fills).
        kernels.LaunchKvDequantDyn(
            _keysQuant[layerIndex], _kScratch,
            _kvStride, _windowSize, _maxSeqLen, KeyDType, posPtrDevice, stream);
        kernels.LaunchKvDequantDyn(
            _valuesQuant[layerIndex], _vScratch,
            _kvStride, _windowSize, _maxSeqLen, ValueDType, posPtrDevice, stream);

        // Phase 2: scatter the live FP16 window into the scratch tail.
        kernels.LaunchKvWindowToScratchDyn(
            _keysWindow![layerIndex], _kScratch, _kvStride, _windowSize, posPtrDevice, stream);
        kernels.LaunchKvWindowToScratchDyn(
            _valuesWindow![layerIndex], _vScratch, _kvStride, _windowSize, posPtrDevice, stream);

        return (_kScratch, _vScratch);
    }

    /// <summary>
    /// Updates host-side counters (<see cref="CurrentLength"/>, <see cref="QuantizedLength"/>,
    /// per-layer eviction state) after a CUDA-Graph decode step. The graph itself wrote
    /// the FP16 ring slot and (when applicable) quantized the evicted row device-side; this
    /// just keeps the metadata consistent so subsequent eager calls / sampler stop-checks
    /// see the right values.
    /// </summary>
    internal void AdvanceLengthForGraphDecode(int newLength)
    {
        if (newLength > _currentLength) _currentLength = newLength;

        if (_windowSize > 0)
        {
            int newQuantLen = Math.Max(0, newLength - _windowSize);
            if (newQuantLen > _quantizedLength) _quantizedLength = newQuantLen;
            // Each layer evicts in lockstep with the global write position; bump
            // each layer's counter so a fall-back to the eager path (e.g. after a
            // rollback) sees the correct prevQuantLen.
            for (int i = 0; i < _numLayers; i++)
            {
                if (newQuantLen > _layerQuantizedLength[i])
                    _layerQuantizedLength[i] = newQuantLen;
            }
        }
    }

    /// <summary>
    /// Prepares dequantized FP16 scratch buffers for attention. Returns device pointers
    /// to contiguous FP16 K/V covering the full sequence (quantized + window).
    /// </summary>
    internal (nint kPtr, nint vPtr) PrepareAttentionScratch(int layerIndex, nint stream, CudaKernels kernels)
    {
        long fp16RowBytes = (long)_kvStride * sizeof(ushort);

        // Phase 1: Dequant quantized region → scratch[0..quantizedLength)
        if (_quantizedLength > 0)
        {
            int totalElements = _quantizedLength * _kvStride;
            kernels.LaunchDequantToF16(
                _keysQuant[layerIndex],
                KeyDType == KvCacheDType.Q8_0 ? Core.Configuration.QuantizationType.Q8_0 : Core.Configuration.QuantizationType.Q4_0,
                _kScratch, totalElements, stream);

            kernels.LaunchDequantToF16(
                _valuesQuant[layerIndex],
                ValueDType == KvCacheDType.Q8_0 ? Core.Configuration.QuantizationType.Q8_0 : Core.Configuration.QuantizationType.Q4_0,
                _vScratch, totalElements, stream);
        }

        // Phase 2: Copy window region → scratch[quantizedLength..currentLength)
        // Window is a ring buffer. Use bulk copy when contiguous (no wraparound).
        int windowLen = WindowLength;
        if (windowLen > 0 && _keysWindow != null)
        {
            int ringStart = _quantizedLength % _windowSize;
            bool windowContiguous = (ringStart + windowLen <= _windowSize);

            if (windowContiguous)
            {
                // No wraparound — single bulk copy
                long bulkBytes = (long)windowLen * fp16RowBytes;
                nint kSrc = _keysWindow[layerIndex] + (nint)(ringStart * fp16RowBytes);
                nint vSrc = _valuesWindow![layerIndex] + (nint)(ringStart * fp16RowBytes);
                nint kDst = _kScratch + (nint)(_quantizedLength * fp16RowBytes);
                nint vDst = _vScratch + (nint)(_quantizedLength * fp16RowBytes);
                CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kSrc, (nuint)bulkBytes, stream).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vSrc, (nuint)bulkBytes, stream).ThrowOnError();
            }
            else
            {
                // Wraparound — two bulk copies (tail + head of ring buffer)
                int tailLen = _windowSize - ringStart;
                int headLen = windowLen - tailLen;

                // Tail: ringStart..windowSize
                long tailBytes = (long)tailLen * fp16RowBytes;
                nint kDst = _kScratch + (nint)(_quantizedLength * fp16RowBytes);
                nint vDst = _vScratch + (nint)(_quantizedLength * fp16RowBytes);
                nint kSrc = _keysWindow[layerIndex] + (nint)(ringStart * fp16RowBytes);
                nint vSrc = _valuesWindow![layerIndex] + (nint)(ringStart * fp16RowBytes);
                CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kSrc, (nuint)tailBytes, stream).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vSrc, (nuint)tailBytes, stream).ThrowOnError();

                // Head: 0..headLen
                if (headLen > 0)
                {
                    long headBytes = (long)headLen * fp16RowBytes;
                    nint kDst2 = kDst + (nint)tailBytes;
                    nint vDst2 = vDst + (nint)tailBytes;
                    CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst2, _keysWindow[layerIndex], (nuint)headBytes, stream).ThrowOnError();
                    CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst2, _valuesWindow![layerIndex], (nuint)headBytes, stream).ThrowOnError();
                }
            }
        }

        return (_kScratch, _vScratch);
    }

    // ── IQuantizedKvCache implementation ────────────────────────────

    /// <inheritdoc/>
    public nint GetQuantizedKeysPtr(int layerIndex) => _keysQuant[layerIndex];

    /// <inheritdoc/>
    public nint GetQuantizedValuesPtr(int layerIndex) => _valuesQuant[layerIndex];

    /// <inheritdoc/>
    public nint GetWindowKeysPtr(int layerIndex)
        => _keysWindow != null ? _keysWindow[layerIndex] : 0;

    /// <inheritdoc/>
    public nint GetWindowValuesPtr(int layerIndex)
        => _valuesWindow != null ? _valuesWindow[layerIndex] : 0;

    // ── IKvCache interface (throw for unsupported host-side operations) ──

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("Use UpdateDevice() for GPU quantized cache.");

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("Use UpdateDevice() for GPU quantized cache.");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException("Use PrepareAttentionScratch() for GPU quantized cache.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException("Use PrepareAttentionScratch() for GPU quantized cache.");

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
        => new(WindowLength, _kvStride, DType.Float16, 0,
               _keysWindow != null ? _keysWindow[layerIndex] : 0);

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => new(WindowLength, _kvStride, DType.Float16, 0,
               _valuesWindow != null ? _valuesWindow[layerIndex] : 0);

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
        if (_quantizedLength > length)
            _quantizedLength = length;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            if (_keysQuant[i] != 0) { CudaDriverApi.cuMemFree_v2(_keysQuant[i]); _keysQuant[i] = 0; }
            if (_valuesQuant[i] != 0) { CudaDriverApi.cuMemFree_v2(_valuesQuant[i]); _valuesQuant[i] = 0; }

            if (_keysWindow != null && _keysWindow[i] != 0) { CudaDriverApi.cuMemFree_v2(_keysWindow[i]); _keysWindow[i] = 0; }
            if (_valuesWindow != null && _valuesWindow[i] != 0) { CudaDriverApi.cuMemFree_v2(_valuesWindow[i]); _valuesWindow[i] = 0; }
        }

        if (_kScratch != 0) { CudaDriverApi.cuMemFree_v2(_kScratch); _kScratch = 0; }
        if (_vScratch != 0) { CudaDriverApi.cuMemFree_v2(_vScratch); _vScratch = 0; }
    }

    private static int ComputeQuantRowBytes(int kvStride, KvCacheDType dtype) => dtype switch
    {
        KvCacheDType.F32 => kvStride * sizeof(ushort), // FP16 on GPU
        KvCacheDType.Q8_0 => kvStride / BlockSize * Q8_0BlockBytes,
        KvCacheDType.Q4_0 => kvStride / BlockSize * Q4_0BlockBytes,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype))
    };
}
