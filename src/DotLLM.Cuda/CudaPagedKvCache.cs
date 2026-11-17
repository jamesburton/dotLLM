using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU-resident paged KV-cache (issue #252) using block-based allocation from a shared
/// <see cref="CudaKvBlockPool"/>. Mirrors <see cref="DotLLM.Engine.KvCache.PagedKvCache"/>'s v1
/// design: block storage is scattered across the pool, and attention kernel compatibility is
/// maintained via a per-instance device-side staging (scratch) buffer that
/// <see cref="PrepareAttentionScratch"/> gathers blocks into — the existing CUDA attention
/// kernels (<c>attention_f16</c> / <c>attention_f16_gqa_split_kv</c> / G3 / G-flash) read the
/// gathered scratch exactly as they read a plain <see cref="CudaKvCache"/>'s contiguous buffer.
/// Since #1839 decode (one query token) skips that gather and reads the pool in place through
/// per-layer block-pointer arrays (<see cref="PrepareNativeBlockPtrs"/>, issue #200's kernel); the
/// gather remains for prefill, sink-bearing layers and stale PTX. See
/// <c>docs/perf/CUDA_PAGED_ATTENTION_DESIGN.md</c>.
/// </summary>
/// <remarks>
/// Like <see cref="CudaKvCache"/> and <see cref="CudaQuantizedKvCache"/>, this cache is driven
/// exclusively through device-pointer entry points (<see cref="UpdateDevice"/>,
/// <see cref="PrepareAttentionScratch"/>) — the <see cref="IKvCache"/> host-tensor surface
/// (<c>Update(ITensor,...)</c>, <c>GetKeys</c>/<c>GetValues</c>, and — unlike the sibling
/// caches — <c>GetKeysRef</c>/<c>GetValuesRef</c> too, since paged storage has no single
/// contiguous device pointer to hand back without a gather) all throw
/// <see cref="NotSupportedException"/> directing callers to the device-pointer methods.
/// </remarks>
public sealed class CudaPagedKvCache : IKvCache, IPerLayerKvCache
{
    private readonly CudaKvBlockPool _pool;
    private readonly CudaKvBlockTable _blockTable;
    private readonly int _numLayers;
    private readonly int _maxSeqLen;
    private readonly int _maxStride; // largest per-layer stride — sizes the shared scratch

    // Scratch buffers for the gathered contiguous view (one pair, reused across layers;
    // sized to the LARGEST per-layer stride so any layer's gather fits). FP16, matching
    // the pool's on-device element type and CudaKvCache's plain-cache layout.
    private nint _kScratch;
    private nint _vScratch;

    // Issue #200 / #1839: block-pointer arrays for the direct block-table-read attention kernel
    // (attention_f16_paged / LaunchAttentionPaged), the default decode path in place of the gather
    // above. ONE pinned host + ONE device array holds every layer's K and V pointers, laid out
    // [kv (0=K, 1=V)][layer][maxBlocksPerSeq] (see PtrSlot). It is rebuilt and uploaded with a
    // single stream-ordered 2D copy only when the block table's Version changes (a new block
    // every BlockSize tokens, CoW, seed, fork, free), not once per layer per step (#1839: the
    // old per-layer synchronous H2D pair cost ~17 ms/token on an RTX 3060 at 40 layers).
    private readonly int _maxBlocksPerSeq;
    private nint _blockPtrsHost;
    private nint _blockPtrsDevice;
    private nint _uploadEvent;      // recorded after each upload; waited on before the host array is rewritten
    private bool _uploadPending;
    private int _uploadedVersion = -1;
    private int _uploadedBlocks;
    private long _blockPtrUploads;

    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _blockTable.CurrentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <inheritdoc/>
    int IPerLayerKvCache.LayerCount => _numLayers;

    /// <inheritdoc/>
    public int KvStrideOf(int layerIndex) => _pool.KvStrideOf(layerIndex);

    /// <summary>
    /// Unmanaged device bytes reserved for this cache's contiguous K/V scratch buffers.
    /// Shared block-pool bytes are reported separately by <see cref="CudaKvBlockPool.AllocatedBytes"/>.
    /// </summary>
    public long AllocatedBytes => 2L * _maxSeqLen * _maxStride * sizeof(ushort);

    /// <summary>The block table mapping logical positions to physical blocks.</summary>
    internal CudaKvBlockTable BlockTable => _blockTable;

    /// <summary>The block pool backing this cache.</summary>
    internal CudaKvBlockPool Pool => _pool;

    /// <summary>
    /// Seeds the cache with shared prefix blocks whose refcounts have already been
    /// incremented. Bumps the visible length to <paramref name="tokenCount"/>.
    /// </summary>
    internal void SeedSharedPrefix(IReadOnlyList<int> blockIds, int tokenCount) =>
        _blockTable.SeedSharedBlocks(blockIds, tokenCount);

    /// <summary>
    /// Snapshots block IDs covering the first <paramref name="tokenCount"/> tokens
    /// (full blocks only). Caller appends to <paramref name="blockIds"/>.
    /// </summary>
    internal void SnapshotFullBlocks(int tokenCount, List<int> blockIds) =>
        _blockTable.SnapshotFullBlocks(tokenCount, blockIds);

    /// <summary>
    /// Creates a new GPU-resident paged KV-cache backed by the given block pool.
    /// </summary>
    /// <param name="pool">Shared block pool for allocation.</param>
    /// <param name="maxSeqLen">Maximum sequence length this cache can hold.</param>
    public CudaPagedKvCache(CudaKvBlockPool pool, int maxSeqLen)
    {
        ArgumentNullException.ThrowIfNull(pool);
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        _pool = pool;
        _numLayers = pool.NumLayers;
        _maxSeqLen = maxSeqLen;
        _blockTable = new CudaKvBlockTable(pool);

        int maxStride = 0;
        for (int i = 0; i < _numLayers; i++)
            maxStride = Math.Max(maxStride, pool.KvStrideOf(i));
        _maxStride = maxStride;

        // Issue #268: without this, a CudaPagedKvCache constructed after an idle-eviction ->
        // lazy-reload cycle (a fresh CudaContext, per docs/SERVER.md's Keep-Alive/Idle-Unload
        // feature) can allocate against whatever context happens to still be current on this
        // thread -- possibly the destroyed pre-reload context -- throwing "CUDA error 201:
        // invalid device context". Mirrors CudaKvBlockPool's own constructor, which already does
        // this via its own _context field; reuse the SAME pool's context here rather than adding
        // a redundant constructor parameter, since every CudaPagedKvCache is always backed by a
        // pool that already carries the right context.
        pool.Context?.MakeCurrent();

        long scratchBytes = (long)maxSeqLen * maxStride * sizeof(ushort);
        // #1848: the gather scratch is allocated per request on top of the pool, so it is admitted against free VRAM before
        // anything is allocated (WDDM would otherwise spill it into shared memory).
        CudaAdmission.EnsureFits($"paged KV gather scratch for {maxSeqLen} tokens", 2 * scratchBytes);
        CudaDriverApi.cuMemAlloc_v2(out _kScratch, (nuint)scratchBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _vScratch, (nuint)scratchBytes).ThrowOnError();

        // Issue #200: block-pointer arrays sized once for the worst case, lazily populated
        // by PrepareNativeBlockPtrs. Allocated unconditionally (tiny — pointer-sized entries,
        // not KV content) so the opt-in native path never needs a first-call allocation on
        // the decode hot path; unused entirely when DOTLLM_ATTN_PAGED_NATIVE is off.
        _maxBlocksPerSeq = (maxSeqLen + pool.BlockSize - 1) / pool.BlockSize;
        nuint ptrArrayBytes = (nuint)(2L * _numLayers * _maxBlocksPerSeq * IntPtr.Size);
        CudaDriverApi.cuMemHostAlloc(out _blockPtrsHost, ptrArrayBytes, 0).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _blockPtrsDevice, ptrArrayBytes).ThrowOnError();
        CudaDriverApi.cuEventCreate(out _uploadEvent, CudaDriverApi.CU_EVENT_DISABLE_TIMING).ThrowOnError();
    }

    /// <summary>
    /// Index of the (K or V, layer, logical block) entry in the combined block-pointer array:
    /// layout <c>[kv][layer][maxBlocksPerSeq]</c>, so each layer's K (or V) pointers are
    /// contiguous, as <c>attention_f16_paged</c> indexes them.
    /// </summary>
    /// <param name="kv">0 for K, 1 for V.</param>
    /// <param name="layer">Layer index.</param>
    /// <param name="block">Logical block index.</param>
    /// <param name="numLayers">Layers in the cache.</param>
    /// <param name="maxBlocks">Per-layer row length of the array.</param>
    internal static int PtrSlot(int kv, int layer, int block, int numLayers, int maxBlocks)
        => (kv * numLayers + layer) * maxBlocks + block;

    /// <summary>
    /// Whether the device block-pointer array must be rebuilt: the table changed since the last
    /// upload, or the table now has more blocks than were uploaded (issue #1839).
    /// </summary>
    internal static bool NeedsBlockPtrUpload(int tableVersion, int uploadedVersion, int blockCount, int uploadedBlocks)
        => tableVersion != uploadedVersion || blockCount > uploadedBlocks;

    /// <summary>Number of block-pointer uploads performed (tests: proves the per-step cost is gone).</summary>
    internal long BlockPtrUploads => _blockPtrUploads;

    /// <summary>
    /// Decode-time (single token) RoPE + K/V write straight into the paged pool (issue #1839), the
    /// paged twin of <see cref="CudaKvCache.FusedRopeAndUpdateDevice"/>: one
    /// <c>fused_rope_kv_write_f16</c> launch per layer instead of a RoPE launch plus two D2D copies.
    /// The kernel writes row <c>cache_pos</c> of the base it is given, so passing the target block's
    /// base and the offset inside the block lands the row in place. Block allocation and CoW run
    /// first, exactly as in <see cref="UpdateDevice"/>.
    /// </summary>
    internal void FusedRopeAndUpdateDevice(
        nint qSrc, nint kSrc, nint vSrc,
        nint positionsDevice, int position,
        int layerIndex,
        int numHeads, int numKvHeads, int headDim,
        int ropeDim, float ropeTheta, int ropeType,
        nint stream, CudaKernels kernels,
        nint ropeInvFreq = 0, float ropeMscale = 1.0f)
    {
        if ((uint)position >= (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(position),
                $"Position {position} exceeds max cache length {_maxSeqLen}.");

        _blockTable.EnsureCapacity(position + 1);
        _blockTable.EnsureWritable(position, stream);
        var (blockId, offset) = _blockTable.Resolve(position);

        kernels.LaunchFusedRopeKvWriteF16(
            qSrc, kSrc, vSrc,
            _pool.GetKeyPtr(blockId, layerIndex), _pool.GetValuePtr(blockId, layerIndex),
            positionsDevice, offset,
            numHeads, numKvHeads, headDim,
            ropeDim, _pool.KvStrideOf(layerIndex), ropeTheta, ropeType,
            stream, ropeInvFreq, ropeMscale);

        if (position + 1 > _blockTable.CurrentLength)
            _blockTable.Advance(position + 1);
    }

    /// <summary>
    /// Appends new K/V rows (device pointers) at the given positions for one layer, allocating
    /// pool blocks on demand and copy-on-writing shared blocks before overwriting them. Mirrors
    /// <see cref="CudaKvCache.UpdateDevice(nint, nint, ReadOnlySpan{int}, int, int, nint)"/>'s
    /// device-pointer contract, plus <see cref="DotLLM.Engine.KvCache.PagedKvCache"/>'s
    /// block-boundary batching (a run of consecutive positions inside the same block becomes a
    /// single D2D copy instead of one copy per token).
    /// </summary>
    /// <param name="keysDevice">Device pointer to new K data [seqLen, kvStride] FP16.</param>
    /// <param name="valuesDevice">Device pointer to new V data [seqLen, kvStride] FP16.</param>
    /// <param name="positions">Host-side positions for the new rows.</param>
    /// <param name="seqLen">Number of new tokens.</param>
    /// <param name="layerIndex">Layer index.</param>
    /// <param name="stream">CUDA stream.</param>
    internal void UpdateDevice(nint keysDevice, nint valuesDevice,
                                ReadOnlySpan<int> positions, int seqLen,
                                int layerIndex, nint stream)
    {
        if (seqLen == 0) return;

        int stride = _pool.KvStrideOf(layerIndex);
        long rowBytes = (long)stride * sizeof(ushort);
        int blockSize = _pool.BlockSize;

        int maxPos = _blockTable.CurrentLength - 1;
        for (int v = 0; v < seqLen; v++)
        {
            int pos = positions[v];
            if ((uint)pos >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {pos} exceeds max cache length {_maxSeqLen}.");
            if (pos > maxPos) maxPos = pos;
        }
        _blockTable.EnsureCapacity(maxPos + 1);

        // Batch contiguous tokens within the same block into a single D2D copy — during
        // prefill positions are sequential (0,1,2,...N) so most tokens share a block.
        int i = 0;
        while (i < seqLen)
        {
            int pos = positions[i];
            _blockTable.EnsureWritable(pos, stream);
            var (blockId, offset) = _blockTable.Resolve(pos);

            int runLen = 1;
            int remaining = blockSize - offset;
            while (runLen < remaining && i + runLen < seqLen &&
                   positions[i + runLen] == pos + runLen)
            {
                runLen++;
            }

            nint kDst = _pool.GetKeyPtr(blockId, layerIndex) + (nint)(offset * rowBytes);
            nint vDst = _pool.GetValuePtr(blockId, layerIndex) + (nint)(offset * rowBytes);
            long batchBytes = runLen * rowBytes;
            nint kSrc = keysDevice + (nint)(i * rowBytes);
            nint vSrc = valuesDevice + (nint)(i * rowBytes);

            CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kSrc, (nuint)batchBytes, stream).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vSrc, (nuint)batchBytes, stream).ThrowOnError();

            i += runLen;
        }

        int newLength = maxPos + 1;
        if (newLength > _blockTable.CurrentLength)
            _blockTable.Advance(newLength);
    }

    /// <summary>
    /// Gathers this layer's blocks (in logical order) into the contiguous FP16 scratch buffer via
    /// device-to-device async copies on <paramref name="stream"/>, and returns device pointers to
    /// the gathered K/V covering the full visible sequence. This is the GPU-side equivalent of
    /// <see cref="DotLLM.Engine.KvCache.PagedKvCache"/>'s private staging gather (same
    /// block-by-block gather shape, D2D copy instead of <c>Buffer.MemoryCopy</c>). The returned
    /// pointers feed the existing (unmodified) attention kernels exactly like
    /// <see cref="CudaKvCache.GetKeysPtr(int)"/>.
    /// </summary>
    internal (nint kPtr, nint vPtr) PrepareAttentionScratch(int layerIndex, nint stream)
    {
        int currentLength = _blockTable.CurrentLength;
        if (currentLength == 0) return (_kScratch, _vScratch);

        int stride = _pool.KvStrideOf(layerIndex);
        int blockSize = _pool.BlockSize;
        long rowBytes = (long)stride * sizeof(ushort);

        int fullBlocks = currentLength / blockSize;
        int tailTokens = currentLength % blockSize;

        long blockBytes = (long)blockSize * rowBytes;
        for (int b = 0; b < fullBlocks; b++)
        {
            var (blockId, _) = _blockTable.Resolve(b * blockSize);
            nint kSrc = _pool.GetKeyPtr(blockId, layerIndex);
            nint vSrc = _pool.GetValuePtr(blockId, layerIndex);
            nint kDst = _kScratch + (nint)((long)b * blockBytes);
            nint vDst = _vScratch + (nint)((long)b * blockBytes);

            CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kSrc, (nuint)blockBytes, stream).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vSrc, (nuint)blockBytes, stream).ThrowOnError();
        }

        if (tailTokens > 0)
        {
            var (blockId, _) = _blockTable.Resolve(fullBlocks * blockSize);
            nint kSrc = _pool.GetKeyPtr(blockId, layerIndex);
            nint vSrc = _pool.GetValuePtr(blockId, layerIndex);
            long tailBytes = tailTokens * rowBytes;
            nint kDst = _kScratch + (nint)((long)fullBlocks * blockBytes);
            nint vDst = _vScratch + (nint)((long)fullBlocks * blockBytes);

            CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kSrc, (nuint)tailBytes, stream).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vSrc, (nuint)tailBytes, stream).ThrowOnError();
        }

        return (_kScratch, _vScratch);
    }

    /// <summary>
    /// Issue #200 / #1839: returns this layer's K and V block-pointer arrays for the direct
    /// block-table-read attention kernel (<c>attention_f16_paged</c> /
    /// <c>CudaKernels.LaunchAttentionPaged</c>), which reads the pool in place instead of
    /// <see cref="PrepareAttentionScratch"/>'s per-block D2D gather of the whole history.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The arrays for ALL layers are rebuilt host-side and uploaded together, and only when the
    /// block table's <see cref="CudaKvBlockTable.Version"/> changed since the last upload. In decode
    /// that is layer 0 of the step that allocates a new block (every <c>BlockSize</c> tokens) or
    /// copies-on-write one; every other call is pointer arithmetic with no driver call.
    /// </para>
    /// <para>
    /// The upload is a <c>cuMemcpy2DAsync</c> on <paramref name="stream"/>, so it is ordered after
    /// any CoW copy and before the attention launch that reads it. The pinned host array is only
    /// rewritten after <c>cuEventSynchronize</c> on the previous upload's event, so an in-flight
    /// upload never reads a half-rewritten buffer. In decode that event completed a step ago.
    /// </para>
    /// </remarks>
    /// <returns>
    /// Device pointers to this layer's K and V block-pointer arrays, plus the block count covering
    /// the visible sequence (the kernel derives <c>logical_block = tkv / blockSize</c> itself).
    /// </returns>
    internal unsafe (nint kBlockPtrs, nint vBlockPtrs, int blockCount) PrepareNativeBlockPtrs(int layerIndex, nint stream)
    {
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));

        int currentLength = _blockTable.CurrentLength;
        int blockSize = _pool.BlockSize;
        int blockCount = (currentLength + blockSize - 1) / blockSize;
        nint kArr = _blockPtrsDevice + (nint)((long)PtrSlot(0, layerIndex, 0, _numLayers, _maxBlocksPerSeq) * sizeof(nint));
        nint vArr = _blockPtrsDevice + (nint)((long)PtrSlot(1, layerIndex, 0, _numLayers, _maxBlocksPerSeq) * sizeof(nint));
        if (blockCount == 0) return (kArr, vArr, 0);

        int tableBlocks = _blockTable.BlockCount;
        if (tableBlocks > _maxBlocksPerSeq)
            throw new InvalidOperationException(
                $"CudaPagedKvCache: sequence needs {tableBlocks} blocks but the block-pointer " +
                $"arrays were only sized for {_maxBlocksPerSeq} (maxSeqLen at construction).");

        if (NeedsBlockPtrUpload(_blockTable.Version, _uploadedVersion, tableBlocks, _uploadedBlocks))
            UploadBlockPtrs(tableBlocks, stream);

        return (kArr, vArr, blockCount);
    }

    private unsafe void UploadBlockPtrs(int tableBlocks, nint stream)
    {
        if (_uploadPending)
        {
            CudaDriverApi.cuEventSynchronize(_uploadEvent).ThrowOnError();
            _uploadPending = false;
        }

        var host = (nint*)_blockPtrsHost;
        for (int layer = 0; layer < _numLayers; layer++)
        {
            int kBase = PtrSlot(0, layer, 0, _numLayers, _maxBlocksPerSeq);
            int vBase = PtrSlot(1, layer, 0, _numLayers, _maxBlocksPerSeq);
            for (int b = 0; b < tableBlocks; b++)
            {
                int blockId = _blockTable.BlockIdAt(b);
                host[kBase + b] = _pool.GetKeyPtr(blockId, layer);
                host[vBase + b] = _pool.GetValuePtr(blockId, layer);
            }
        }

        if (tableBlocks > 0)
        {
            nuint pitch = (nuint)((long)_maxBlocksPerSeq * sizeof(nint));
            var copy = new CudaDriverApi.CudaMemcpy2D
            {
                SrcMemoryType = CudaDriverApi.CU_MEMORYTYPE_HOST,
                SrcHost = _blockPtrsHost,
                SrcPitch = pitch,
                DstMemoryType = CudaDriverApi.CU_MEMORYTYPE_DEVICE,
                DstDevice = _blockPtrsDevice,
                DstPitch = pitch,
                WidthInBytes = (nuint)((long)tableBlocks * sizeof(nint)),
                Height = (nuint)(2 * _numLayers),
            };
            CudaDriverApi.cuMemcpy2DAsync_v2(ref copy, stream).ThrowOnError();
            CudaDriverApi.cuEventRecord(_uploadEvent, stream).ThrowOnError();
            _uploadPending = true;
        }

        _uploadedVersion = _blockTable.Version;
        _uploadedBlocks = tableBlocks;
        _blockPtrUploads++;
    }

    /// <summary>
    /// Resets the visible length of the cache to the given position.
    /// Used by prompt caching to truncate to the matched prefix length.
    /// </summary>
    internal void SetCurrentLength(int length)
    {
        if ((uint)length > (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(length));
        _blockTable.SetCurrentLength(length);
    }

    // ── IKvCache interface: device-pointer-only cache (mirrors CudaKvCache / CudaQuantizedKvCache) ──

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex) =>
        throw new NotSupportedException("CudaPagedKvCache.Update(ITensor) not supported. Use UpdateDevice().");

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex) =>
        throw new NotSupportedException("CudaPagedKvCache.Update(TensorRef) not supported. Use UpdateDevice().");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex) =>
        throw new NotSupportedException("CudaPagedKvCache.GetKeys(ITensor) not supported. Use PrepareAttentionScratch().");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex) =>
        throw new NotSupportedException("CudaPagedKvCache.GetValues(ITensor) not supported. Use PrepareAttentionScratch().");

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex) =>
        throw new NotSupportedException(
            "CudaPagedKvCache.GetKeysRef() not supported: paged storage has no single contiguous " +
            "device pointer without a gather. Use PrepareAttentionScratch(layerIndex, stream).");

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex) =>
        throw new NotSupportedException(
            "CudaPagedKvCache.GetValuesRef() not supported: paged storage has no single contiguous " +
            "device pointer without a gather. Use PrepareAttentionScratch(layerIndex, stream).");

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_blockTable.CurrentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _blockTable.SetCurrentLength(length);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _blockTable.Free();

        // Issue #268: same MakeCurrent rationale as the constructor -- mirrors CudaKvBlockPool.Dispose.
        _pool.Context?.MakeCurrent();

        if (_kScratch != 0) { CudaDriverApi.cuMemFree_v2(_kScratch); _kScratch = 0; }
        if (_vScratch != 0) { CudaDriverApi.cuMemFree_v2(_vScratch); _vScratch = 0; }

        if (_uploadEvent != 0)
        {
            if (_uploadPending) CudaDriverApi.cuEventSynchronize(_uploadEvent);
            CudaDriverApi.cuEventDestroy_v2(_uploadEvent);
            _uploadEvent = 0;
        }
        if (_blockPtrsDevice != 0) { CudaDriverApi.cuMemFree_v2(_blockPtrsDevice); _blockPtrsDevice = 0; }
        if (_blockPtrsHost != 0) { CudaDriverApi.cuMemFreeHost(_blockPtrsHost); _blockPtrsHost = 0; }
    }
}
