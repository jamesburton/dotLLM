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
/// A direct block-table-read attention kernel (eliminating this gather) is issue #200's separate,
/// still-blocked scope — see <c>docs/perf/CUDA_PAGED_ATTENTION_DESIGN.md</c>.
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

    // Issue #200: block-pointer arrays for the direct block-table-read attention kernel
    // (attention_f16_paged / LaunchAttentionPaged), an alternative to the gather above.
    // Host side is pinned memory (cuMemHostAlloc) so the small per-decode-step H2D refresh
    // in PrepareNativeBlockPtrs is genuinely async (same reasoning as issue #251's pinned
    // D2H pool) rather than blocking on pageable memory. Sized once for the worst case
    // (ceil(maxSeqLen / blockSize) blocks); reused every call, never reallocated.
    private readonly int _maxBlocksPerSeq;
    private nint _kBlockPtrsHost;
    private nint _vBlockPtrsHost;
    private nint _kBlockPtrsDevice;
    private nint _vBlockPtrsDevice;

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
        CudaDriverApi.cuMemAlloc_v2(out _kScratch, (nuint)scratchBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _vScratch, (nuint)scratchBytes).ThrowOnError();

        // Issue #200: block-pointer arrays sized once for the worst case, lazily populated
        // by PrepareNativeBlockPtrs. Allocated unconditionally (tiny — pointer-sized entries,
        // not KV content) so the opt-in native path never needs a first-call allocation on
        // the decode hot path; unused entirely when DOTLLM_ATTN_PAGED_NATIVE is off.
        _maxBlocksPerSeq = (maxSeqLen + pool.BlockSize - 1) / pool.BlockSize;
        nuint ptrArrayBytes = (nuint)((long)_maxBlocksPerSeq * IntPtr.Size);
        CudaDriverApi.cuMemHostAlloc(out _kBlockPtrsHost, ptrArrayBytes, 0).ThrowOnError();
        CudaDriverApi.cuMemHostAlloc(out _vBlockPtrsHost, ptrArrayBytes, 0).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _kBlockPtrsDevice, ptrArrayBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _vBlockPtrsDevice, ptrArrayBytes).ThrowOnError();
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
    /// Issue #200: builds this layer's block-pointer arrays for the direct block-table-read
    /// attention kernel (<c>attention_f16_paged</c> / <c>CudaKernels.LaunchAttentionPaged</c>),
    /// eliminating <see cref="PrepareAttentionScratch"/>'s D2D gather of KV content. Resolves
    /// each logical block's device pointer for this layer host-side (cheap pointer
    /// arithmetic, no device round-trip — mirrors <see cref="CudaKvBlockPool.GetKeyPtr"/>'s
    /// own cost), then refreshes only the small pointer array on the device
    /// (<c>blockCount * sizeof(nint)</c> bytes — a handful of blocks in the common case, e.g.
    /// 17 blocks × 8 bytes × 2 arrays ≈ 272 bytes, vs. the full KV byte gather
    /// <see cref="PrepareAttentionScratch"/> performs every call).
    /// </summary>
    /// <remarks>
    /// Uses a <b>synchronous</b> <c>cuMemcpyHtoD_v2</c>, not the async entry point, even though
    /// the host staging buffer is pinned. The pinned buffer is reused across every layer of a
    /// decode step (never reallocated); an async H2D copy only enqueues the transfer, so the
    /// very next call (next layer) could overwrite the host buffer before the GPU actually
    /// reads it — a real host/device race, not a hypothetical one, since this method is called
    /// once per layer in a tight loop with no synchronization between calls. A synchronous copy
    /// costs a small fixed PCIe-latency stall (~1-5 µs, dominated by latency not the ~272-byte
    /// payload) but guarantees the buffer is safe to overwrite the moment this call returns.
    /// Double-buffering the host array (ping-pong between two pinned buffers) would let this
    /// go async again; not done here since this path is opt-in/unvalidated and correctness
    /// takes priority over an unmeasured latency win (see docs/perf/CUDA_PAGED_ATTENTION_DESIGN.md).
    /// </remarks>
    /// <returns>
    /// Device pointers to the K and V block-pointer arrays, plus the block count (the
    /// attention kernel derives <c>logical_block = tkv / blockSize</c> itself, so the caller
    /// does not need to pass the count separately — it's returned only for callers that want
    /// to reason about it, e.g. tests).
    /// </returns>
    internal unsafe (nint kBlockPtrs, nint vBlockPtrs, int blockCount) PrepareNativeBlockPtrs(int layerIndex, nint stream)
    {
        int currentLength = _blockTable.CurrentLength;
        int blockSize = _pool.BlockSize;
        int blockCount = (currentLength + blockSize - 1) / blockSize;
        if (blockCount == 0) return (_kBlockPtrsDevice, _vBlockPtrsDevice, 0);
        if (blockCount > _maxBlocksPerSeq)
            throw new InvalidOperationException(
                $"CudaPagedKvCache: sequence needs {blockCount} blocks but the block-pointer " +
                $"arrays were only sized for {_maxBlocksPerSeq} (maxSeqLen at construction).");

        var kHost = (nint*)_kBlockPtrsHost;
        var vHost = (nint*)_vBlockPtrsHost;
        for (int b = 0; b < blockCount; b++)
        {
            int blockId = _blockTable.BlockIdAt(b);
            kHost[b] = _pool.GetKeyPtr(blockId, layerIndex);
            vHost[b] = _pool.GetValuePtr(blockId, layerIndex);
        }

        nuint bytes = (nuint)((long)blockCount * sizeof(nint));
        CudaDriverApi.cuMemcpyHtoD_v2(_kBlockPtrsDevice, _kBlockPtrsHost, bytes).ThrowOnError();
        CudaDriverApi.cuMemcpyHtoD_v2(_vBlockPtrsDevice, _vBlockPtrsHost, bytes).ThrowOnError();

        return (_kBlockPtrsDevice, _vBlockPtrsDevice, blockCount);
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

        if (_kBlockPtrsDevice != 0) { CudaDriverApi.cuMemFree_v2(_kBlockPtrsDevice); _kBlockPtrsDevice = 0; }
        if (_vBlockPtrsDevice != 0) { CudaDriverApi.cuMemFree_v2(_vBlockPtrsDevice); _vBlockPtrsDevice = 0; }
        if (_kBlockPtrsHost != 0) { CudaDriverApi.cuMemFreeHost(_kBlockPtrsHost); _kBlockPtrsHost = 0; }
        if (_vBlockPtrsHost != 0) { CudaDriverApi.cuMemFreeHost(_vBlockPtrsHost); _vBlockPtrsHost = 0; }
    }
}
