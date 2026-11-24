using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU-resident quantized KV-cache with dual-region storage:
/// quantized buffer (Q8_0/Q4_0 in device memory) + FP16 window for recent tokens.
/// <para>
/// Attention reads the stored rows in place (issue #1850, CLAUDE.md rule 2): <c>attention_kvq</c> (decode, and the prefill
/// fallback) and <c>attention_flash_mma_prefill_kvq</c> (tensor-core prefill, head_dim 64 / 128) dequantize each Q8_0 / Q4_0
/// element in registers inside the tile load, and read the FP16 window ring directly. There is no FP16 history scratch, so a
/// position costs exactly its quantized K and V row bytes.
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
    private readonly int _maxStride;         // max per-layer stride — sizes the test-only reference scratch (ReferenceDequantScratch)
    private int _maxSeqLen;                  // raised by TryGrowTo (issue #1594)
    private int _generation;                 // bumped whenever device buffers are replaced (TryGrowTo)
    private readonly CudaContext? _context;  // context the buffers live on; null = the ambient current context
    private readonly int _windowSize;
    private readonly int[] _keyQuantRowBytes;   // [numLayers] quantized K row bytes
    private readonly int[] _valueQuantRowBytes; // [numLayers] quantized V row bytes
    private readonly int[] _layerQuantizedLength; // per-layer eviction tracking
    private int _currentLength;
    private int _quantizedLength;

    // Test-only oracle (issue #1850): the pre-#1850 full-history FP16 dequant scratch, allocated lazily and ONLY when this cache's
    // ReferenceDequantScratch is set, so parity tests can run the legacy path in the same process. Never allocated in production.
    private nint _refKScratch;  // device ptr: FP16 [maxSeqLen, maxStride]
    private nint _refVScratch;  // device ptr: FP16 [maxSeqLen, maxStride]

    /// <summary>
    /// Test-only oracle switch (issue #1850), per cache: when set on THIS cache, <see cref="CudaTransformerModel"/> runs its
    /// attention the pre-#1850 way, over a full-history FP16 scratch the cache dequantizes into (<see cref="PrepareReferenceScratch"/>),
    /// and skips the quantized decode graph. A reference/baseline comparison under CLAUDE.md's oracle exemption: set only by tests on
    /// a cache they own (no env var, no process-wide state, so no other cache can fall back to the scratch), lazily allocated,
    /// outside <see cref="BytesFor"/>.
    /// </summary>
    internal bool ReferenceDequantScratch { get; set; }

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
        : this(geometry, maxSeqLen, config, context, inPlaceAttentionLoaded: null)
    {
    }

    /// <summary>
    /// As <see cref="CudaQuantizedKvCache(KvGeometry, int, KvCacheConfig, CudaContext?)"/>. <paramref name="inPlaceAttentionLoaded"/>
    /// is whether the model's kernel set loaded the in-place attention kernels (<see cref="CudaKernels.HasAttentionKvq"/>): attention
    /// over this cache has no other path since #1850, so <see langword="false"/> refuses here, before anything is allocated, instead
    /// of at the first forward (or mid-way through a decode-graph capture). <see langword="null"/> skips the check (kernel tests).
    /// </summary>
    internal CudaQuantizedKvCache(KvGeometry geometry, int maxSeqLen, KvCacheConfig config, CudaContext? context, bool? inPlaceAttentionLoaded)
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

        // Both sides must be block formats: the quantize-on-write kernels (LaunchQuantKv) and the in-place attention kernels
        // (#1850) have no FP16 arm, so an f16/q8_0 mix used to throw at the first KV write, mid-forward. Refuse it up front.
        if (KeyDType is not (KvCacheDType.Q8_0 or KvCacheDType.Q4_0) || ValueDType is not (KvCacheDType.Q8_0 or KvCacheDType.Q4_0))
            throw new NotSupportedException(
                $"The dense CUDA quantized KV cache needs keys AND values in a block format, q8_0 or q4_0 (got " +
                $"{KeyDType.ToString().ToLowerInvariant()}/{ValueDType.ToString().ToLowerInvariant()}); mixing f16 with a block format is " +
                "not supported. Set --cache-type-k and --cache-type-v each to q8_0 or q4_0 (they may differ), or both to f16.");
        if (inPlaceAttentionLoaded == false)
            throw new InvalidOperationException(
                "The q8_0/q4_0 KV cache needs the in-place attention kernels (attention_kvq.ptx), which did not load on this device " +
                "(missing, stale or unloadable PTX next to DotLLM.Cuda). Rebuild or reinstall the native PTX, or use f16 KV " +
                "(--cache-type-k f16 --cache-type-v f16).");

        // #1848: the whole cache (quantized rows and the FP16 window) is admitted against free VRAM before anything is allocated;
        // under WDDM an over-commit spills to shared system memory instead of failing.
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

            // One launch per tensor per contiguous run of ring slots: consecutive positions are consecutive slots until the ring
            // wraps, and consecutive quantized rows, so a run [evictPos, evictPos + rows) is one source span and one destination span.
            for (int evictPos = prevQuantLen; evictPos < safeEvictEnd;)
            {
                int ringIdx = evictPos % _windowSize;
                int rows = Math.Min(safeEvictEnd - evictPos, _windowSize - ringIdx);

                nint evictedK = _keysWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint quantDstK = _keysQuant[layerIndex] + (nint)((long)evictPos * keyRowBytes);
                kernels.LaunchQuantKv(evictedK, quantDstK, rows * stride, KeyDType, stream);

                nint evictedV = _valuesWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint quantDstV = _valuesQuant[layerIndex] + (nint)((long)evictPos * valueRowBytes);
                kernels.LaunchQuantKv(evictedV, quantDstV, rows * stride, ValueDType, stream);
                evictPos += rows;
            }

            // Phase 2: current-batch positions that will be evicted immediately —
            // quantize directly from source, skip the ring (one launch per contiguous run, #1845).
            QuantizeRuns(keysDevice, valuesDevice, positions, seqLen, layerIndex, stream, kernels, belowPos: newQuantLen);

            _layerQuantizedLength[layerIndex] = newQuantLen;

            // Phase 3: positions that stay in the window — write FP16 into the ring, one copy per tensor per maximal run of
            // consecutive batch rows with consecutive positions, split where the ring wraps.
            for (int i = 0; i < seqLen;)
            {
                int pos = positions[i];
                if (pos < newQuantLen) { i++; continue; }
                int ringIdx = pos % _windowSize;
                int rows = 1;
                while (i + rows < seqLen && positions[i + rows] == pos + rows && ringIdx + rows < _windowSize) rows++;

                nint kDst = _keysWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint vDst = _valuesWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint kSrc = keysDevice + (nint)(i * fp16RowBytes);
                nint vSrc = valuesDevice + (nint)(i * fp16RowBytes);
                CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kSrc, (nuint)(rows * fp16RowBytes), stream).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vSrc, (nuint)(rows * fp16RowBytes), stream).ThrowOnError();
                i += rows;
            }

            _quantizedLength = newQuantLen;
        }
        else
        {
            // Pure quantized: quantize directly at each position (position-addressed). Rows whose positions are consecutive are
            // contiguous in both the source batch and the cache, so each such run is ONE launch per tensor (#1845: per-row
            // launches cost 2 x rows x layers launches per prefill, ~237K for a 2,966-token TwIL-8B prompt).
            QuantizeRuns(keysDevice, valuesDevice, positions, seqLen, layerIndex, stream, kernels, belowPos: int.MaxValue);

            _quantizedLength = newLength;
        }

        _currentLength = newLength;
    }

    /// <summary>
    /// Quantizes batch rows <c>i</c> with <c>positions[i] &lt; belowPos</c> into quantized row <c>positions[i]</c>, one
    /// <c>LaunchQuantKv</c> per tensor for every maximal run of consecutive batch rows with consecutive positions. Byte-identical
    /// to one launch per row: the quantize kernels treat each 32-element block independently.
    /// </summary>
    private void QuantizeRuns(nint keysDevice, nint valuesDevice, ReadOnlySpan<int> positions, int seqLen, int layerIndex,
                              nint stream, CudaKernels kernels, int belowPos)
    {
        int stride = Stride(layerIndex);
        long fp16RowBytes = (long)stride * sizeof(ushort);
        int keyRowBytes = _keyQuantRowBytes[layerIndex];
        int valueRowBytes = _valueQuantRowBytes[layerIndex];
        int i = 0;
        while (i < seqLen)
        {
            if (positions[i] >= belowPos) { i++; continue; }
            int start = i;
            while (i + 1 < seqLen && positions[i + 1] == positions[i] + 1 && positions[i + 1] < belowPos) i++;
            int rows = i - start + 1;
            int pos = positions[start];
            nint kSrc = keysDevice + (nint)(start * fp16RowBytes);
            nint vSrc = valuesDevice + (nint)(start * fp16RowBytes);
            nint kDst = _keysQuant[layerIndex] + (nint)((long)pos * keyRowBytes);
            nint vDst = _valuesQuant[layerIndex] + (nint)((long)pos * valueRowBytes);
            kernels.LaunchQuantKv(kSrc, kDst, rows * stride, KeyDType, stream);
            kernels.LaunchQuantKv(vSrc, vDst, rows * stride, ValueDType, stream);
            i++;
        }
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
    /// Test-only oracle (issue #1850, <see cref="ReferenceDequantScratch"/>): the pre-#1850 attention input, dequantizing the
    /// quantized region and copying the window into a full-history FP16 scratch (allocated here on first use). Returns device
    /// pointers to contiguous FP16 K/V covering the full sequence (quantized + window). Production attention reads in place.
    /// </summary>
    internal (nint kPtr, nint vPtr) PrepareReferenceScratch(int layerIndex, nint stream, CudaKernels kernels)
    {
        int stride = Stride(layerIndex);
        long fp16RowBytes = (long)stride * sizeof(ushort);
        if (_refKScratch == 0)
        {
            long bytes = (long)_maxSeqLen * _maxStride * sizeof(ushort);
            CudaDriverApi.cuMemAlloc_v2(out _refKScratch, (nuint)bytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _refVScratch, (nuint)bytes).ThrowOnError();
        }
        nint kScratch = _refKScratch, vScratch = _refVScratch;

        // Phase 1: Dequant quantized region → scratch[0..quantizedLength)
        if (_quantizedLength > 0)
        {
            int totalElements = _quantizedLength * stride;
            // KV-format dequant, NOT the GGUF weight dequant: KV Q4_0 packs adjacent pairs (issue #956).
            kernels.LaunchKvDequantToF16(_keysQuant[layerIndex], KeyDType, kScratch, totalElements, stream);
            kernels.LaunchKvDequantToF16(_valuesQuant[layerIndex], ValueDType, vScratch, totalElements, stream);
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
                nint kDst = kScratch + (nint)(_quantizedLength * fp16RowBytes);
                nint vDst = vScratch + (nint)(_quantizedLength * fp16RowBytes);
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
                nint kDst = kScratch + (nint)(_quantizedLength * fp16RowBytes);
                nint vDst = vScratch + (nint)(_quantizedLength * fp16RowBytes);
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

        return (kScratch, vScratch);
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
        => throw new NotSupportedException("GPU quantized cache rows are read in place by the attention kernels.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException("GPU quantized cache rows are read in place by the attention kernels.");

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
    public bool SupportsAbortRollback => _windowSize <= 0;

    /// <inheritdoc/>
    /// <remarks>Windowed: appending past the FP16 window evicts rows into the quantized region, after which a cut to the entry length is unsafe (see <see cref="CanTruncate"/>).</remarks>
    public bool CanRollbackAfterAppend(int minPos, int maxPos)
    {
        int newLength = Math.Max(_currentLength, maxPos + 1);
        int newQuant = _windowSize > 0 ? Math.Max(_quantizedLength, newLength - _windowSize) : newLength;
        return CanTruncate(_windowSize, newLength, newQuant, _currentLength);
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
    /// <see cref="MaxLength"/> and stays. Attention reads the rows in place (#1850), so there is no history scratch to resize.
    /// The peak extra memory is admitted against free VRAM first (never spill); a refusal or allocation failure returns
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
        if (!CudaKvGrowth.Admit(CudaKvGrowth.PeakGrowthBytes(rowBytes, _maxSeqLen, maxLength)))
            return false;

        FreeReferenceScratch();   // test-only oracle: re-allocated at the new length on next use
        _generation++;
        int rc;

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
                // Layers below `layer` already hold maxLength rows; the cache stays valid at the old length.
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

        FreeReferenceScratch();
    }

    private void FreeReferenceScratch()
    {
        if (_refKScratch != 0) { CudaDriverApi.cuMemFree_v2(_refKScratch); _refKScratch = 0; }
        if (_refVScratch != 0) { CudaDriverApi.cuMemFree_v2(_refVScratch); _refVScratch = 0; }
    }

    /// <summary>
    /// Device bytes a cache of <paramref name="maxSeqLen"/> positions allocates (#1848), mirroring the constructor: quantized K and V
    /// rows per layer and the FP16 window. Since #1850 there is no FP16 dequant scratch: attention reads the rows in place.
    /// </summary>
    /// <param name="geometry">Per-layer KV row widths.</param>
    /// <param name="maxSeqLen">Positions.</param>
    /// <param name="config">KV types and window.</param>
    /// <returns>Bytes.</returns>
    internal static long BytesFor(KvGeometry geometry, int maxSeqLen, KvCacheConfig config)
    {
        long bytes = 0;
        for (int i = 0; i < geometry.LayerCount; i++)
        {
            int stride = geometry.KvStrideOf(i);
            bytes += (long)maxSeqLen * (ComputeQuantRowBytes(stride, config.KeyDType) + ComputeQuantRowBytes(stride, config.ValueDType));
            if (config.MixedPrecisionWindowSize > 0)
                bytes += 2L * config.MixedPrecisionWindowSize * stride * sizeof(ushort);
        }
        return bytes;
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
