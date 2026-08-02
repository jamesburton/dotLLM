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

        long scratchBytes = (long)maxSeqLen * maxStride * sizeof(ushort);
        CudaDriverApi.cuMemAlloc_v2(out _kScratch, (nuint)scratchBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _vScratch, (nuint)scratchBytes).ThrowOnError();
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

        if (_kScratch != 0) { CudaDriverApi.cuMemFree_v2(_kScratch); _kScratch = 0; }
        if (_vScratch != 0) { CudaDriverApi.cuMemFree_v2(_vScratch); _vScratch = 0; }
    }
}
