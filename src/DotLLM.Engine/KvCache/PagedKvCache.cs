using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine.KvCache;

/// <summary>
/// Paged KV-cache using block-based allocation from a shared <see cref="KvBlockPool"/>.
/// Memory is allocated in fixed-size blocks on demand, reducing waste for variable-length sequences.
/// <para>
/// Attention reads the blocks in place through <see cref="IPagedKvCache"/> (#1268). The contiguous
/// <see cref="GetKeysRef"/>/<see cref="GetValuesRef"/> views remain for callers without a paged kernel: they decode
/// the history into F32 staging buffers that are allocated lazily, on first use only.
/// </para>
/// </summary>
public sealed unsafe class PagedKvCache : IPagedKvCache, IHostStagedKvCache
{
    private readonly KvBlockPool _pool;
    private readonly KvBlockTable _blockTable;
    private readonly int _numLayers;
    private readonly int _kvStride;
    private readonly int _maxSeqLen;

    // Legacy contiguous views (GetKeysRef/GetValuesRef), allocated on first use; the paged attention kernel never needs them.
    private nint _keyStagingPtr;
    private nint _valueStagingPtr;
    private readonly long _stagingBytes;

    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _blockTable.CurrentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <inheritdoc/>
    public int BlockSize => _pool.BlockSize;

    /// <inheritdoc/>
    public int KvStride => _kvStride;

    /// <inheritdoc/>
    public ReadOnlySpan<int> BlockIds => _blockTable.BlockIds;

    /// <summary>Row storage type of the backing pool.</summary>
    public DType ElementType => _pool.ElementType;

    /// <summary>Bytes held by the lazily allocated legacy F32 staging buffers (0 unless a contiguous view was requested).</summary>
    public long StagingBytes => (_keyStagingPtr == 0 ? 0 : _stagingBytes) +
                                (_valueStagingPtr == 0 ? 0 : _stagingBytes);

    /// <summary>
    /// Unmanaged bytes this sequence holds: its pool blocks (K and V, all layers) plus any legacy staging.
    /// Blocks shared with other sequences through the prefix cache are counted by every holder.
    /// The whole pool's reservation is <see cref="KvBlockPool.AllocatedBytes"/>.
    /// </summary>
    public long AllocatedBytes => _blockTable.BlockCount * _pool.BlockBytesAllLayers + StagingBytes;

    /// <summary>The block table mapping logical positions to physical blocks.</summary>
    internal KvBlockTable BlockTable => _blockTable;

    /// <summary>The block pool backing this cache.</summary>
    internal KvBlockPool Pool => _pool;

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
    /// Creates a new paged KV-cache backed by the given block pool.
    /// </summary>
    /// <param name="pool">Shared block pool for allocation.</param>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="kvStride">KV stride (numKvHeads * headDim).</param>
    /// <param name="maxSeqLen">Maximum sequence length this cache can hold.</param>
    public PagedKvCache(KvBlockPool pool, int numLayers, int kvStride, int maxSeqLen)
    {
        ArgumentNullException.ThrowIfNull(pool);
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (kvStride <= 0) throw new ArgumentOutOfRangeException(nameof(kvStride));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (numLayers != pool.NumLayers)
            throw new ArgumentException("Pool layer count does not match cache layer count.", nameof(pool));
        if (kvStride != pool.KvStride)
            throw new ArgumentException("Pool KV stride does not match cache KV stride.", nameof(pool));

        _pool = pool;
        _numLayers = numLayers;
        _kvStride = kvStride;
        _maxSeqLen = maxSeqLen;
        _blockTable = new KvBlockTable(pool);
        _stagingBytes = checked((long)maxSeqLen * kvStride * sizeof(float));
    }

    /// <inheritdoc/>
    public TensorRef GetKeyBlocksRef(int layerIndex)
    {
        ValidateLayer(layerIndex);
        return new TensorRef(checked(_pool.TotalBlocks * BlockSize), _kvStride,
            _pool.ElementType, -1, (nint)_pool.GetKeyPtr(0, layerIndex));
    }

    /// <inheritdoc/>
    public TensorRef GetValueBlocksRef(int layerIndex)
    {
        ValidateLayer(layerIndex);
        return new TensorRef(checked(_pool.TotalBlocks * BlockSize), _kvStride,
            _pool.ElementType, -1, (nint)_pool.GetValuePtr(0, layerIndex));
    }

    /// <inheritdoc/>
    /// <remarks>Rows are F32 <c>[positions.Length, KvStride]</c> on the host; they are encoded into the pool's element type.</remarks>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        ValidateLayer(layerIndex);
        int seqLen = positions.Length;
        if (keys.DType != DType.Float32 || values.DType != DType.Float32 ||
            keys.DeviceId != -1 || values.DeviceId != -1 ||
            keys.Dim0 != seqLen || values.Dim0 != seqLen ||
            keys.Dim1 != _kvStride || values.Dim1 != _kvStride)
            throw new ArgumentException($"Expected CPU float32 [{seqLen}, {_kvStride}] K/V projections.");
        if (seqLen == 0) return;

        int maxPos = _blockTable.CurrentLength - 1;

        float* kSrc = (float*)keys.DataPointer;
        float* vSrc = (float*)values.DataPointer;

        int rowBytes = _pool.RowBytes;
        int blockSize = _pool.BlockSize;

        // Validate and find max position, ensure capacity upfront
        for (int v = 0; v < seqLen; v++)
        {
            int pos = positions[v];
            if ((uint)pos >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {pos} exceeds max cache length {_maxSeqLen}.");
            if (pos > maxPos) maxPos = pos;
        }
        _blockTable.EnsureCapacity(maxPos + 1);

        // Batch contiguous tokens within the same block into a single encode.
        // During prefill, positions are sequential (0,1,2,...N) so most tokens
        // share a block and can be written together.
        int i = 0;
        while (i < seqLen)
        {
            int pos = positions[i];
            _blockTable.EnsureWritable(pos);
            var (blockId, offset) = _blockTable.Resolve(pos);

            // Count how many consecutive tokens fit in this same block
            int runLen = 1;
            int remaining = blockSize - offset;
            while (runLen < remaining && i + runLen < seqLen &&
                   positions[i + runLen] == pos + runLen)
            {
                runLen++;
            }

            _pool.EncodeRows(kSrc + (long)i * _kvStride, _pool.GetKeyPtr(blockId, layerIndex) + (long)offset * rowBytes, runLen);
            _pool.EncodeRows(vSrc + (long)i * _kvStride, _pool.GetValuePtr(blockId, layerIndex) + (long)offset * rowBytes, runLen);

            i += runLen;
        }

        int newLength = maxPos + 1;
        if (newLength > _blockTable.CurrentLength)
            _blockTable.Advance(newLength);
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        var kRef = new TensorRef(positions.Length, _kvStride, keys.DType, keys.DeviceId, keys.DataPointer);
        var vRef = new TensorRef(positions.Length, _kvStride, values.DType, values.DeviceId, values.DataPointer);
        Update(kRef, vRef, positions, layerIndex);
    }

    /// <inheritdoc/>
    /// <remarks>Legacy contiguous view: decodes the history into lazily allocated F32 staging. Paged-aware kernels use <see cref="GetKeyBlocksRef"/>.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef GetKeysRef(int layerIndex)
    {
        GatherIntoStaging(ref _keyStagingPtr, layerIndex, isKey: true);
        return new TensorRef(_blockTable.CurrentLength, _kvStride, DType.Float32, -1, _keyStagingPtr);
    }

    /// <inheritdoc/>
    /// <remarks>Legacy contiguous view: decodes the history into lazily allocated F32 staging. Paged-aware kernels use <see cref="GetValueBlocksRef"/>.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef GetValuesRef(int layerIndex)
    {
        GatherIntoStaging(ref _valueStagingPtr, layerIndex, isKey: false);
        return new TensorRef(_blockTable.CurrentLength, _kvStride, DType.Float32, -1, _valueStagingPtr);
    }

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
    {
        GatherIntoStaging(ref _keyStagingPtr, layerIndex, isKey: true);
        var shape = new TensorShape(_blockTable.CurrentLength, _kvStride);
        return new TensorView(shape, DType.Float32, -1, _keyStagingPtr);
    }

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
    {
        GatherIntoStaging(ref _valueStagingPtr, layerIndex, isKey: false);
        var shape = new TensorShape(_blockTable.CurrentLength, _kvStride);
        return new TensorView(shape, DType.Float32, -1, _valueStagingPtr);
    }

    // ── IHostStagedKvCache: host-staging surface for cross-pool / cross-device KV handoff ──

    /// <inheritdoc/>
    public int StagedLayerElementCount(int layerIndex)
    {
        ValidateLayer(layerIndex);
        return _blockTable.CurrentLength * _kvStride;
    }

    /// <inheritdoc/>
    public void DownloadLayer(int layerIndex, Span<float> keys, Span<float> values)
    {
        ValidateLayer(layerIndex);
        int count = _blockTable.CurrentLength * _kvStride;
        if (count == 0) return;
        if (keys.Length < count || values.Length < count)
            throw new ArgumentException($"keys/values must contain at least {count} floats.");

        // Decode straight into the caller's buffers: no staging needed.
        fixed (float* kp = keys)
        fixed (float* vp = values)
        {
            DecodeLayer(layerIndex, isKey: true, kp);
            DecodeLayer(layerIndex, isKey: false, vp);
        }
    }

    /// <inheritdoc/>
    public void UploadLayer(int layerIndex, int length, ReadOnlySpan<float> keys, ReadOnlySpan<float> values)
    {
        ValidateLayer(layerIndex);
        if (length <= 0) return;
        long expected = (long)length * _kvStride;
        if (keys.Length < expected || values.Length < expected)
            throw new ArgumentException($"keys/values must contain at least length × kvStride = {expected} floats.");

        int[] positions = ArrayPool<int>.Shared.Rent(length);
        try
        {
            for (int p = 0; p < length; p++) positions[p] = p;
            fixed (float* kp = keys)
            fixed (float* vp = values)
            {
                var kRef = new TensorRef(length, _kvStride, DType.Float32, -1, (nint)kp);
                var vRef = new TensorRef(length, _kvStride, DType.Float32, -1, (nint)vp);
                Update(kRef, vRef, positions.AsSpan(0, length), layerIndex);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(positions);
        }
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

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_blockTable.CurrentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _blockTable.SetCurrentLength(length);
    }

    private void ValidateLayer(int layerIndex)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
    }

    /// <summary>
    /// Decodes the layer's history into a contiguous F32 staging buffer (allocated on first use) for legacy consumers.
    /// </summary>
    private void GatherIntoStaging(ref nint stagingPtr, int layerIndex, bool isKey)
    {
        ValidateLayer(layerIndex);
        if (stagingPtr == 0)
        {
            stagingPtr = (nint)NativeMemory.AlignedAlloc((nuint)_stagingBytes, 64);
            if (stagingPtr == 0) throw new OutOfMemoryException();
        }

        DecodeLayer(layerIndex, isKey, (float*)stagingPtr);
    }

    /// <summary>Decodes <see cref="CurrentLength"/> rows of one layer, block by block in logical order, into <paramref name="dst"/>.</summary>
    private void DecodeLayer(int layerIndex, bool isKey, float* dst)
    {
        int currentLength = _blockTable.CurrentLength;
        int blockSize = _pool.BlockSize;
        for (int start = 0; start < currentLength; start += blockSize)
        {
            var (blockId, _) = _blockTable.Resolve(start);
            byte* src = isKey ? _pool.GetKeyPtr(blockId, layerIndex) : _pool.GetValuePtr(blockId, layerIndex);
            _pool.DecodeRows(src, dst + (long)start * _kvStride, Math.Min(blockSize, currentLength - start));
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

        if (disposing)
        {
            // Only touch managed objects during explicit Dispose
            _blockTable.Free();
        }

        // Free unmanaged staging buffers regardless
        if (_keyStagingPtr != 0)
            NativeMemory.AlignedFree((void*)_keyStagingPtr);
        if (_valueStagingPtr != 0)
            NativeMemory.AlignedFree((void*)_valueStagingPtr);
        _keyStagingPtr = 0;
        _valueStagingPtr = 0;
    }

    /// <summary>Releases unmanaged staging buffers if <see cref="Dispose()"/> was not called.</summary>
    ~PagedKvCache() => Dispose(disposing: false);
}
