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
/// <para>
/// Prefix reuse (issue #1594): the cache owns its buffers and keeps its rows after a request ends, so the live <c>PrefixCache</c>
/// retains it and a later request cuts it back to the common prompt prefix (<see cref="IPrefixReusableKvCache"/>), grows it
/// (<see cref="TryGrowTo"/>) or seeds a fresh cache from it (<see cref="TrySeedPrefixFrom"/>). Rows move in their stored
/// quantized format, never widened. With a mixed-precision window a cut is only safe before any row has been evicted
/// (<see cref="CanTruncateTo"/>).
/// </para>
/// </summary>
public sealed class CudaQuantizedKvCache : IQuantizedKvCache, IPerLayerKvCache, IPrefixReusableKvCache, IKvPrefixSeedable,
    IDeviceResidentKvCache
{
    private const int BlockSize = 32;
    private const int Q8_0BlockBytes = QuantFormat.Q8_0BlockBytes;
    private const int Q4_0BlockBytes = QuantFormat.Q4_0BlockBytes;

    private readonly nint[] _keysQuant;     // device ptrs: quantized K
    private readonly nint[] _valuesQuant;   // device ptrs: quantized V
    private readonly nint[]? _keysWindow;   // device ptrs: FP16 K window
    private readonly nint[]? _valuesWindow; // device ptrs: FP16 V window
    private readonly int _numLayers;
    private readonly KvGeometry _geom;       // per-layer KV row width (numKvHeads(l) * headDim(l))
    private readonly bool _uniform;          // _geom.IsUniform, hoisted for the hot path
    private readonly int _uniformStride;     // _geom.UniformStride when _uniform; else 0
    private readonly int _maxStride;         // max per-layer stride — sizes the shared scratch
    private int _maxSeqLen;                  // raised by TryGrowTo (issue #1594)
    private int _generation;                 // bumped whenever device buffers are replaced (TryGrowTo)
    private readonly CudaContext? _context;  // context the buffers live on; null = the ambient current context
    private readonly int _windowSize;
    private readonly int[] _keyQuantRowBytes;   // [numLayers] quantized K row bytes
    private readonly int[] _valueQuantRowBytes; // [numLayers] quantized V row bytes
    private readonly int[] _layerQuantizedLength; // per-layer eviction tracking
    private int _currentLength;
    private int _quantizedLength;

    // Scratch buffers for dequantized K/V during attention (one pair, reused across layers;
    // sized to the LARGEST per-layer stride so any layer's dequant fits).
    private nint _kScratch;  // device ptr: FP16 [maxSeqLen, maxStride]
    private nint _vScratch;  // device ptr: FP16 [maxSeqLen, maxStride]

    /// <summary>Per-layer KV row width (FP16 elements); scalar shortcut hoisted when uniform.</summary>
    private int Stride(int layerIndex) => _uniform ? _uniformStride : _geom.KvStrideOf(layerIndex);

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <inheritdoc/>
    public long DeviceBytes => BytesFor(_geom, _maxSeqLen, new KvCacheConfig(KeyDType, ValueDType, _windowSize));

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
    public int KeyQuantizedRowBytes => _keyQuantRowBytes[0];

    /// <inheritdoc/>
    public int ValueQuantizedRowBytes => _valueQuantRowBytes[0];

    /// <inheritdoc/>
    public int KeyQuantizedRowBytesOf(int layerIndex) => _keyQuantRowBytes[layerIndex];

    /// <inheritdoc/>
    public int ValueQuantizedRowBytesOf(int layerIndex) => _valueQuantRowBytes[layerIndex];

    /// <inheritdoc/>
    int IPerLayerKvCache.LayerCount => _numLayers;

    /// <inheritdoc/>
    public int KvStrideOf(int layerIndex) => _geom.KvStrideOf(layerIndex);

    /// <summary>Total device memory allocated in bytes.</summary>
    public long AllocatedBytes { get; private set; }

    /// <summary>
    /// Changes whenever device buffers are replaced (<see cref="TryGrowTo"/>). The captured quantized decode graph bakes the buffer
    /// pointers and <see cref="MaxLength"/> in, so <see cref="CudaTransformerModel"/> recaptures when this moves.
    /// </summary>
    internal int Generation => _generation;

    /// <summary>
    /// Creates a GPU quantized KV-cache with a single uniform <c>numKvHeads * headDim</c>
    /// stride. Byte-identical to the per-layer constructor with <see cref="KvGeometry.Uniform"/>.
    /// </summary>
    public CudaQuantizedKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen,
                                 KvCacheConfig config)
        : this(KvGeometry.Uniform(numLayers, numKvHeads, headDim), maxSeqLen, config)
    {
    }

    /// <summary>
    /// Creates a GPU quantized KV-cache from a Core <see cref="KvGeometry"/> descriptor.
    /// Uniform for every dense/GQA/MoE model (byte-identical to the scalar constructor);
    /// per-layer for Gemma-4. NOTE: Gemma-4 CUDA attention is currently cacheless, so this
    /// generalises the constructor + buffer/scratch sizing only (no CUDA Gemma-4 decode path).
    /// </summary>
    public CudaQuantizedKvCache(KvGeometry geometry, int maxSeqLen, KvCacheConfig config)
        : this(geometry, maxSeqLen, config, context: null)
    {
    }

    /// <summary>
    /// As <see cref="CudaQuantizedKvCache(KvGeometry, int, KvCacheConfig)"/>, with the CUDA context the buffers live on. Growth, seeding
    /// and disposal make it current first: a retained prefix-cache entry can be grown or evicted from a thread other than the one
    /// that created it, and the current context is per thread.
    /// </summary>
    public CudaQuantizedKvCache(KvGeometry geometry, int maxSeqLen, KvCacheConfig config, CudaContext? context)
    {
        _context = context;
        _context?.MakeCurrent();
        int numLayers = geometry.LayerCount;
        _numLayers = numLayers;
        _geom = geometry;
        _uniform = geometry.IsUniform;
        _uniformStride = geometry.IsUniform ? geometry.UniformStride : 0;
        _maxSeqLen = maxSeqLen;
        _windowSize = config.MixedPrecisionWindowSize;
        KeyDType = config.KeyDType;
        ValueDType = config.ValueDType;

        _keyQuantRowBytes = new int[numLayers];
        _valueQuantRowBytes = new int[numLayers];
        _layerQuantizedLength = new int[numLayers];

        _keysQuant = new nint[numLayers];
        _valuesQuant = new nint[numLayers];

        // #1848: the whole cache (quantized rows, FP16 window, and the FP16 full-history dequant scratch) is admitted against free
        // VRAM before anything is allocated; under WDDM an over-commit spills to shared system memory instead of failing.
        CudaAdmission.EnsureFits(
            $"KV cache ({config.KeyDType.ToString().ToLowerInvariant()}/{config.ValueDType.ToString().ToLowerInvariant()}) for {maxSeqLen} tokens",
            BytesFor(geometry, maxSeqLen, config));

        long totalBytes = 0;
        int maxStride = 0;

        // Allocate quantized buffers (per-layer stride). The %BlockSize quantization
        // constraint is validated per layer (uniform models hit the identical check).
        for (int i = 0; i < numLayers; i++)
        {
            int stride = geometry.KvStrideOf(i);
            if (stride % BlockSize != 0)
                throw new ArgumentException(
                    $"kvStride ({stride}) for layer {i} must be a multiple of {BlockSize} for quantization.");
            if (stride > maxStride) maxStride = stride;

            _keyQuantRowBytes[i] = ComputeQuantRowBytes(stride, config.KeyDType);
            _valueQuantRowBytes[i] = ComputeQuantRowBytes(stride, config.ValueDType);

            nuint kBytes = (nuint)((long)maxSeqLen * _keyQuantRowBytes[i]);
            nuint vBytes = (nuint)((long)maxSeqLen * _valueQuantRowBytes[i]);
            CudaDriverApi.cuMemAlloc_v2(out _keysQuant[i], kBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _valuesQuant[i], vBytes).ThrowOnError();
            totalBytes += (long)(kBytes + vBytes);
        }

        _maxStride = maxStride;

        // Allocate FP16 window buffers (per-layer stride)
        if (_windowSize > 0)
        {
            _keysWindow = new nint[numLayers];
            _valuesWindow = new nint[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                nuint windowBytes = (nuint)((long)_windowSize * geometry.KvStrideOf(i) * sizeof(ushort));
                CudaDriverApi.cuMemAlloc_v2(out _keysWindow[i], windowBytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out _valuesWindow[i], windowBytes).ThrowOnError();
                totalBytes += (long)(windowBytes * 2);
            }
        }

        // Allocate scratch buffers for attention dequant (one pair, reused across layers;
        // sized to the largest per-layer stride so any layer fits).
        long scratchBytes = (long)maxSeqLen * maxStride * sizeof(ushort);
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
        int stride = Stride(layerIndex);
        int keyRowBytes = _keyQuantRowBytes[layerIndex];
        int valueRowBytes = _valueQuantRowBytes[layerIndex];
        long fp16RowBytes = (long)stride * sizeof(ushort);

        // Compute new sequence length (idempotent across layer calls with same positions).
        int maxPos = positions[0];
        for (int i = 1; i < seqLen; i++)
            if (positions[i] > maxPos) maxPos = positions[i];
        int newLength = maxPos + 1;

        if (_windowSize > 0)
        {
            int prevQuantLen = _layerQuantizedLength[layerIndex];
            int newQuantLen = Math.Max(0, newLength - _windowSize);

            // Phase 1: evict ring slots from PREVIOUS calls (safe to read).
            int minNewPos = positions[0];
            for (int i = 1; i < seqLen; i++)
                if (positions[i] < minNewPos) minNewPos = positions[i];
            int safeEvictEnd = Math.Min(newQuantLen, minNewPos);

            for (int evictPos = prevQuantLen; evictPos < safeEvictEnd; evictPos++)
            {
                int ringIdx = evictPos % _windowSize;

                nint evictedK = _keysWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint quantDstK = _keysQuant[layerIndex] + (nint)((long)evictPos * keyRowBytes);
                kernels.LaunchQuantKv(evictedK, quantDstK, stride, KeyDType, stream);

                nint evictedV = _valuesWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint quantDstV = _valuesQuant[layerIndex] + (nint)((long)evictPos * valueRowBytes);
                kernels.LaunchQuantKv(evictedV, quantDstV, stride, ValueDType, stream);
            }

            // Phase 2: current-batch positions that will be evicted immediately —
            // quantize directly from source, skip the ring.
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                if (pos < newQuantLen)
                {
                    nint kSrc = keysDevice + (nint)(i * fp16RowBytes);
                    nint vSrc = valuesDevice + (nint)(i * fp16RowBytes);
                    nint kDst = _keysQuant[layerIndex] + (nint)((long)pos * _keyQuantRowBytes[layerIndex]);
                    nint vDst = _valuesQuant[layerIndex] + (nint)((long)pos * _valueQuantRowBytes[layerIndex]);

                    kernels.LaunchQuantKv(kSrc, kDst, stride, KeyDType, stream);
                    kernels.LaunchQuantKv(vSrc, vDst, stride, ValueDType, stream);
                }
            }

            _layerQuantizedLength[layerIndex] = newQuantLen;

            // Phase 3: positions that stay in the window — write FP16 into the ring.
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                if (pos >= newQuantLen)
                {
                    int ringIdx = pos % _windowSize;
                    nint kDst = _keysWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                    nint vDst = _valuesWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                    nint kSrc = keysDevice + (nint)(i * fp16RowBytes);
                    nint vSrc = valuesDevice + (nint)(i * fp16RowBytes);

                    CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kSrc, (nuint)fp16RowBytes, stream).ThrowOnError();
                    CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vSrc, (nuint)fp16RowBytes, stream).ThrowOnError();
                }
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
                nint kDst = _keysQuant[layerIndex] + (nint)((long)pos * keyRowBytes);
                nint vDst = _valuesQuant[layerIndex] + (nint)((long)pos * valueRowBytes);

                kernels.LaunchQuantKv(kSrc, kDst, stride, KeyDType, stream);
                kernels.LaunchQuantKv(vSrc, vDst, stride, ValueDType, stream);
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
        int stride = Stride(layerIndex);
        kernels.LaunchQuantKvDyn(
            _keysWindow![layerIndex], _keysQuant[layerIndex],
            stride, _windowSize, KeyDType, posPtrDevice, stream);
        kernels.LaunchQuantKvDyn(
            _valuesWindow![layerIndex], _valuesQuant[layerIndex],
            stride, _windowSize, ValueDType, posPtrDevice, stream);

        // Step 2: write the new FP16 row into the per-layer ring buffer at
        // slot `pos % windowSize`. After eviction has read the old contents.
        kernels.LaunchKvWriteOneF16Ring(
            keysDevice, _keysWindow[layerIndex], stride, _windowSize, posPtrDevice, stream);
        kernels.LaunchKvWriteOneF16Ring(
            valuesDevice, _valuesWindow[layerIndex], stride, _windowSize, posPtrDevice, stream);
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

        int stride = Stride(layerIndex);
        // Phase 1: dequant the [0, quantizedLength) prefix (predicated; no-op until window fills).
        kernels.LaunchKvDequantDyn(
            _keysQuant[layerIndex], _kScratch,
            stride, _windowSize, _maxSeqLen, KeyDType, posPtrDevice, stream);
        kernels.LaunchKvDequantDyn(
            _valuesQuant[layerIndex], _vScratch,
            stride, _windowSize, _maxSeqLen, ValueDType, posPtrDevice, stream);

        // Phase 2: scatter the live FP16 window into the scratch tail.
        kernels.LaunchKvWindowToScratchDyn(
            _keysWindow![layerIndex], _kScratch, stride, _windowSize, posPtrDevice, stream);
        kernels.LaunchKvWindowToScratchDyn(
            _valuesWindow![layerIndex], _vScratch, stride, _windowSize, posPtrDevice, stream);

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
        int stride = Stride(layerIndex);
        long fp16RowBytes = (long)stride * sizeof(ushort);

        // Phase 1: Dequant quantized region → scratch[0..quantizedLength)
        if (_quantizedLength > 0)
        {
            int totalElements = _quantizedLength * stride;
            // KV-format dequant, NOT the GGUF weight dequant: KV Q4_0 packs adjacent pairs (issue #956).
            kernels.LaunchKvDequantToF16(_keysQuant[layerIndex], KeyDType, _kScratch, totalElements, stream);
            kernels.LaunchKvDequantToF16(_valuesQuant[layerIndex], ValueDType, _vScratch, totalElements, stream);
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
        => new(WindowLength, Stride(layerIndex), DType.Float16, 0,
               _keysWindow != null ? _keysWindow[layerIndex] : 0);

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => new(WindowLength, Stride(layerIndex), DType.Float16, 0,
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
    /// <remarks>See <see cref="CanTruncate"/>.</remarks>
    public bool CanTruncateTo(int length) => CanTruncate(_windowSize, _currentLength, _quantizedLength, length);

    /// <summary>
    /// Whether a cache with an FP16 window of <paramref name="windowSize"/> rows (0 = pure quantized), holding
    /// <paramref name="currentLength"/> rows of which <paramref name="quantizedLength"/> were evicted into the quantized region, can
    /// be cut back to <paramref name="length"/> and then appended to without corrupting any row. Pure quantized: always, every row
    /// sits at its own position. Windowed: only while nothing has been evicted yet (every row is still in the ring at its own slot),
    /// or for a no-op cut. Once rows were evicted, a cut breaks the invariant both update paths rely on (quantized length =
    /// length - window): the eager <c>UpdateDevice</c> lowers the quantized length below the rows already evicted, so attention reads
    /// ring slots still holding the cut rows; the graph path evicts row pos - window from its ring slot again, overwriting a
    /// correctly quantized row with a cut one (#1594 review). Pure so it is testable without a GPU.
    /// </summary>
    internal static bool CanTruncate(int windowSize, int currentLength, int quantizedLength, int length)
        => (uint)length <= (uint)currentLength && (windowSize <= 0 || quantizedLength == 0 || length == currentLength);

    /// <summary>
    /// Raises <see cref="MaxLength"/> to <paramref name="maxLength"/> keeping the committed rows (issue #1594, the CUDA side of
    /// #1781). Layer by layer it allocates larger quantized K and V, copies the live quantized rows device to device in their
    /// stored format (no widening) and frees the old pair; the FP16 window is a ring sized independently of
    /// <see cref="MaxLength"/> and stays. The shared FP16 dequant scratch (<c>[maxSeqLen, maxStride]</c> for K and V, an existing
    /// #1107 expansion: 4 x maxStride bytes per position) is reallocated at the new size, not copied: attention rebuilds it on
    /// every call. The peak extra memory is admitted against free VRAM first (never spill); a refusal or allocation failure returns
    /// <see langword="false"/> and leaves the cache valid at its old <see cref="MaxLength"/>.
    /// </summary>
    /// <param name="maxLength">Positions the cache must hold.</param>
    /// <returns><see langword="true"/> when <see cref="MaxLength"/> is now at least <paramref name="maxLength"/>.</returns>
    public bool TryGrowTo(int maxLength)
    {
        if (maxLength <= _maxSeqLen) return true;
        _context?.MakeCurrent();

        long[] rowBytes = new long[_numLayers]; // planning path, once per growth
        for (int i = 0; i < _numLayers; i++) rowBytes[i] = (long)_keyQuantRowBytes[i] + _valueQuantRowBytes[i];
        long scratchBytes = (long)maxLength * _maxStride * sizeof(ushort);
        if (!CudaKvGrowth.Admit(CudaKvGrowth.PeakGrowthBytes(rowBytes, _maxSeqLen, maxLength, fixedExtraBytes: 2 * scratchBytes)))
            return false;

        // Scratch first: the new pair is allocated before the old is freed, which the admission counted in full.
        nint ks = 0, vs = 0;
        int rc = CudaDriverApi.cuMemAlloc_v2(out ks, (nuint)scratchBytes);
        if (rc == 0) rc = CudaDriverApi.cuMemAlloc_v2(out vs, (nuint)scratchBytes);
        if (rc != 0)
        {
            if (ks != 0) CudaDriverApi.cuMemFree_v2(ks);
            if (vs != 0) CudaDriverApi.cuMemFree_v2(vs);
            Console.Error.WriteLine($"[dotLLM.Cuda] quantized KV growth to {maxLength} positions failed (scratch, CUresult {rc}); keeping {_maxSeqLen}.");
            return false;
        }
        long oldScratch = (long)_maxSeqLen * _maxStride * sizeof(ushort);
        CudaDriverApi.cuMemFree_v2(_kScratch);
        CudaDriverApi.cuMemFree_v2(_vScratch);
        _kScratch = ks;
        _vScratch = vs;
        AllocatedBytes += 2 * (scratchBytes - oldScratch);
        _generation++;

        // Quantized rows are live below _quantizedLength (pure quantized: every row); window rows sit in the ring, untouched.
        int keepRows = _windowSize > 0 ? _quantizedLength : _currentLength;
        for (int layer = 0; layer < _numLayers; layer++)
        {
            nuint kBytes = (nuint)((long)maxLength * _keyQuantRowBytes[layer]);
            nuint vBytes = (nuint)((long)maxLength * _valueQuantRowBytes[layer]);
            nuint kKeep = (nuint)((long)keepRows * _keyQuantRowBytes[layer]);
            nuint vKeep = (nuint)((long)keepRows * _valueQuantRowBytes[layer]);
            nint nk = 0, nv = 0;
            rc = CudaDriverApi.cuMemAlloc_v2(out nk, kBytes);
            if (rc == 0) rc = CudaDriverApi.cuMemAlloc_v2(out nv, vBytes);
            if (rc == 0 && kKeep > 0) rc = CudaDriverApi.cuMemcpyDtoD_v2(nk, _keysQuant[layer], kKeep);
            if (rc == 0 && vKeep > 0) rc = CudaDriverApi.cuMemcpyDtoD_v2(nv, _valuesQuant[layer], vKeep);
            if (rc != 0)
            {
                if (nk != 0) CudaDriverApi.cuMemFree_v2(nk);
                if (nv != 0) CudaDriverApi.cuMemFree_v2(nv);
                // Layers below `layer` (and the scratch) already hold maxLength rows; the cache stays valid at the old length.
                Console.Error.WriteLine($"[dotLLM.Cuda] quantized KV growth to {maxLength} positions failed at layer {layer} (CUresult {rc}); keeping {_maxSeqLen}.");
                return false;
            }
            CudaDriverApi.cuMemFree_v2(_keysQuant[layer]);
            CudaDriverApi.cuMemFree_v2(_valuesQuant[layer]);
            _keysQuant[layer] = nk;
            _valuesQuant[layer] = nv;
            AllocatedBytes += (long)(kBytes + vBytes) - (long)_maxSeqLen * rowBytes[layer];
        }
        _maxSeqLen = maxLength;
        CudaKvGrowth.RecordGrowth();
        return true;
    }

    /// <summary>
    /// Copies rows <c>[0, rows)</c> of every layer from another pure-quantized <see cref="CudaQuantizedKvCache"/> of the same geometry
    /// and K/V types into this one, device to device in the stored quantized format (the #1780 protected-entry seed on CUDA, issue
    /// #1594), and sets <see cref="CurrentLength"/> to <paramref name="rows"/>. The source is not modified. False, with this cache
    /// unchanged, for a mixed-precision window on either side (its ring holds the newest rows, not a prefix), another type,
    /// geometry, format or context, too few source rows, or too little room here.
    /// </summary>
    /// <param name="source">The cache whose leading rows to copy.</param>
    /// <param name="rows">Rows to copy.</param>
    /// <returns>Whether the rows were copied.</returns>
    public bool TrySeedPrefixFrom(IKvCache source, int rows)
    {
        if (source is not CudaQuantizedKvCache src || ReferenceEquals(src, this)) return false;
        if (_windowSize > 0 || src._windowSize > 0) return false;
        if (rows <= 0 || rows > src._currentLength || rows > _maxSeqLen) return false;
        if (src._numLayers != _numLayers || src.KeyDType != KeyDType || src.ValueDType != ValueDType
            || !ReferenceEquals(src._context, _context)) return false;
        for (int i = 0; i < _numLayers; i++)
            if (src.Stride(i) != Stride(i)) return false;

        _context?.MakeCurrent();
        for (int i = 0; i < _numLayers; i++)
        {
            CudaDriverApi.cuMemcpyDtoD_v2(_keysQuant[i], src._keysQuant[i], (nuint)((long)rows * _keyQuantRowBytes[i])).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoD_v2(_valuesQuant[i], src._valuesQuant[i], (nuint)((long)rows * _valueQuantRowBytes[i])).ThrowOnError();
        }
        _currentLength = rows;
        _quantizedLength = rows;
        return true;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _context?.MakeCurrent();
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

    /// <summary>
    /// Device bytes a cache of <paramref name="maxSeqLen"/> positions allocates (#1848), mirroring the constructor: quantized K and V
    /// rows per layer, the FP16 window, and the FP16 K and V dequant scratch over the whole history at the widest stride.
    /// </summary>
    /// <param name="geometry">Per-layer KV row widths.</param>
    /// <param name="maxSeqLen">Positions.</param>
    /// <param name="config">KV types and window.</param>
    /// <returns>Bytes.</returns>
    internal static long BytesFor(KvGeometry geometry, int maxSeqLen, KvCacheConfig config)
    {
        long bytes = 0;
        int maxStride = 0;
        for (int i = 0; i < geometry.LayerCount; i++)
        {
            int stride = geometry.KvStrideOf(i);
            maxStride = Math.Max(maxStride, stride);
            bytes += (long)maxSeqLen * (ComputeQuantRowBytes(stride, config.KeyDType) + ComputeQuantRowBytes(stride, config.ValueDType));
            if (config.MixedPrecisionWindowSize > 0)
                bytes += 2L * config.MixedPrecisionWindowSize * stride * sizeof(ushort);
        }
        return bytes + 2L * maxSeqLen * maxStride * sizeof(ushort);
    }

    /// <summary>Device bytes one position costs in a cache of this configuration (#1848 context fit), window excluded.</summary>
    /// <param name="geometry">Per-layer KV row widths.</param>
    /// <param name="config">KV types.</param>
    /// <returns>Bytes per position.</returns>
    internal static long BytesPerPosition(KvGeometry geometry, KvCacheConfig config)
        => BytesFor(geometry, 1, config with { MixedPrecisionWindowSize = 0 });

    private static int ComputeQuantRowBytes(int kvStride, KvCacheDType dtype) => dtype switch
    {
        KvCacheDType.F32 => kvStride * sizeof(ushort), // FP16 on GPU
        KvCacheDType.Q8_0 => kvStride / BlockSize * Q8_0BlockBytes,
        KvCacheDType.Q4_0 => kvStride / BlockSize * Q4_0BlockBytes,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype))
    };
}
