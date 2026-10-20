using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

public sealed unsafe partial class CudaQwen3HybridDenseTransformerModel
{
    private readonly Guid _prefixSnapshotOwner = Guid.NewGuid();

    /// <inheritdoc/>
    public string PrefixSnapshotIdentity => $"{_prefixSnapshotOwner:N}:{_kvKeyDType}:{_kvValueDType}:{_ropeDim}:{_ropeTheta:R}:{_mtpHead is not null}:{Config.MaxSequenceLength}";

    /// <inheritdoc/>
    public bool SupportsPrefixSnapshotStorage(RecurrentPrefixSnapshotStorage storage)
        => !_disposed && storage is RecurrentPrefixSnapshotStorage.Device or RecurrentPrefixSnapshotStorage.Host;

    /// <inheritdoc/>
    public bool HasPrefixSnapshotRequestHeadroom(IKvCache kvCache)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (kvCache is not CudaHybridKvCacheHandle) throw new ArgumentException("A CUDA hybrid KV handle is required.", nameof(kvCache));
        // Tiered requests overwrite the old active storage, so it can be freed before growing. Snapshot copies are separate.
        long growth = Math.Max(0, checked(ComputeKvBytes(kvCache.MaxLength) - KvAllocatedBytes));
        long free = FreeVramOverride ?? QueryFreeDeviceBytes();
        return free >= checked(growth + KvGrowthVramMarginBytes);
    }

    private readonly record struct PrefixSegment(nint Pointer, long Bytes);

    private PrefixSegment[] PrefixSegments(int length)
    {
        int stride = checked(Config.NumKvHeads * Config.HeadDim);
        var segments = new PrefixSegment[checked(_attentionLayerCount * 2 + 2)];
        for (int i = 0; i < _attentionLayerCount; i++)
        {
            segments[2 * i] = new PrefixSegment(IsQuantKv(_kvKeyDType) ? _kQuant![i] : _f16KCache![i],
                checked((long)length * QuantRowBytes(_kvKeyDType, stride)));
            segments[2 * i + 1] = new PrefixSegment(IsQuantKv(_kvValueDType) ? _vQuant![i] : _f16VCache![i],
                checked((long)length * QuantRowBytes(_kvValueDType, stride)));
        }
        int layers = _gdnCache.NumGdnLayers;
        segments[^2] = new PrefixSegment(layers == 0 ? 0 : _gdnCache.GetConvStatePtr(0), checked((long)layers * _gdnCache.ConvStateElements * sizeof(float)));
        segments[^1] = new PrefixSegment(layers == 0 ? 0 : _gdnCache.GetGdnStatePtr(0), checked((long)layers * _gdnCache.GdnStateElements * sizeof(float)));
        return segments;
    }

    private void ValidatePrefixCapture(IKvCache kv, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (kv is not CudaHybridKvCacheHandle || !ReferenceEquals(kv, _lastForwardKvCache))
            throw new ArgumentException("The prefix must belong to this model's current live KV handle.", nameof(kv));
        if (length <= 0 || length != kv.CurrentLength || length != _f16CacheCurrentLength)
            throw new ArgumentException("Capture must occur at the exact live recurrent/KV token boundary.", nameof(length));
    }

    /// <inheritdoc/>
    public long EstimatePrefixSnapshotBytes(IKvCache kvCache, int prefixLength)
    {
        ValidatePrefixCapture(kvCache, prefixLength);
        return PrefixSegments(prefixLength).Sum(s => s.Bytes);
    }

    /// <inheritdoc/>
    public IRecurrentPrefixSnapshot CapturePrefixSnapshot(IKvCache kvCache, int prefixLength, RecurrentPrefixSnapshotStorage storage)
    {
        ValidatePrefixCapture(kvCache, prefixLength);
        _context.MakeCurrent(); _stream.Synchronize();
        var segments = PrefixSegments(prefixLength);
        var snapshot = AllocatePrefixSnapshot(prefixLength, segments.Sum(s => s.Bytes), storage);
        try
        {
            using var copy = CudaStream.Create();
            long offset = 0;
            try
            {
                foreach (var segment in segments)
                {
                    if (segment.Bytes > 0)
                        CopyPrefixBytes(snapshot.Data + (nint)offset, storage, segment.Pointer, RecurrentPrefixSnapshotStorage.Device, segment.Bytes, copy.Handle);
                    offset += segment.Bytes;
                }
            }
            finally { copy.Synchronize(); }
            return snapshot;
        }
        catch { snapshot.Dispose(); throw; }
    }

    /// <inheritdoc/>
    public IRecurrentPrefixSnapshot TransferPrefixSnapshot(IRecurrentPrefixSnapshot source, RecurrentPrefixSnapshotStorage storage)
    {
        var original = RequirePrefixSnapshot(source);
        _context.MakeCurrent();
        var copy = AllocatePrefixSnapshot(original.PrefixLength, original.ByteCount, storage);
        try
        {
            using var stream = CudaStream.Create();
            try { CopyPrefixBytes(copy.Data, storage, original.Data, original.Storage, original.ByteCount, stream.Handle); }
            finally { stream.Synchronize(); }
            return copy;
        }
        catch { copy.Dispose(); throw; }
    }

    /// <inheritdoc/>
    public void RestorePrefixSnapshot(IRecurrentPrefixSnapshot snapshot, IKvCache kvCache)
    {
        var saved = RequirePrefixSnapshot(snapshot);
        if (kvCache is not CudaHybridKvCacheHandle handle || saved.PrefixLength > handle.MaxLength)
            throw new ArgumentException("Destination KV handle cannot hold the snapshot.", nameof(kvCache));
        _context.MakeCurrent(); _stream.Synchronize();
        // The old live rows are disposable: every input is about to be replaced. Avoid a redundant growth copy.
        _f16CacheCurrentLength = 0;
        EnsureF16KvCache(handle.MaxLength, Config.NumKvHeads, Config.HeadDim);
        var segments = PrefixSegments(saved.PrefixLength);
        if (segments.Sum(s => s.Bytes) != saved.ByteCount)
            throw new ArgumentException("Snapshot state geometry mismatch.", nameof(snapshot));
        using var copy = CudaStream.Create();
        long offset = 0;
        try
        {
            foreach (var segment in segments)
            {
                if (segment.Bytes > 0)
                    CopyPrefixBytes(segment.Pointer, RecurrentPrefixSnapshotStorage.Device, saved.Data + (nint)offset, saved.Storage, segment.Bytes, copy.Handle);
                offset += segment.Bytes;
            }
        }
        finally { copy.Synchronize(); }
        handle.Rollback(0); handle.Advance(saved.PrefixLength);
        _lastForwardKvCache = handle;
        _f16CacheCurrentLength = saved.PrefixLength;
        _rowSnapshotValidRows = 0;
        // Per-request MTP heads are deliberately not resumed: TextGenerator disables MTP on a cached-prefix hit.
        // At least one suffix token runs, regenerating all trunk hidden rows used by ordinary decode.
    }

    private CudaPrefixSnapshot RequirePrefixSnapshot(IRecurrentPrefixSnapshot snapshot)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (snapshot is not CudaPrefixSnapshot saved || !ReferenceEquals(saved.Owner, this) || saved.Identity != PrefixSnapshotIdentity)
            throw new ArgumentException("Snapshot belongs to different weights or forward settings.", nameof(snapshot));
        ObjectDisposedException.ThrowIf(saved.Data == 0, snapshot);
        return saved;
    }

    private CudaPrefixSnapshot AllocatePrefixSnapshot(int length, long bytes, RecurrentPrefixSnapshotStorage storage)
    {
        if (!SupportsPrefixSnapshotStorage(storage)) throw new NotSupportedException("Snapshot storage is unavailable.");
        nint pointer;
        if (storage == RecurrentPrefixSnapshotStorage.Host)
            CudaDriverApi.cuMemHostAlloc(out pointer, checked((nuint)bytes), 0).ThrowOnError();
        else
        {
            CudaDriverApi.cuMemGetInfo_v2(out nuint free, out _).ThrowOnError();
            if ((ulong)free < checked((ulong)bytes + 512UL * 1024 * 1024))
                throw new InvalidOperationException("Device snapshot would leave less than 512 MiB free.");
            CudaDriverApi.cuMemAlloc_v2(out pointer, checked((nuint)bytes)).ThrowOnError();
        }
        return new CudaPrefixSnapshot(this, pointer, bytes, length, PrefixSnapshotIdentity, storage);
    }

    private static void CopyPrefixBytes(nint destination, RecurrentPrefixSnapshotStorage destinationStorage,
        nint source, RecurrentPrefixSnapshotStorage sourceStorage, long bytes, nint stream)
    {
        nuint count = checked((nuint)bytes);
        if (destinationStorage == RecurrentPrefixSnapshotStorage.Host && sourceStorage == RecurrentPrefixSnapshotStorage.Host)
            Buffer.MemoryCopy((void*)source, (void*)destination, bytes, bytes);
        else if (destinationStorage == RecurrentPrefixSnapshotStorage.Host)
            CudaDriverApi.cuMemcpyDtoHAsync_v2(destination, source, count, stream).ThrowOnError();
        else if (sourceStorage == RecurrentPrefixSnapshotStorage.Host)
            CudaDriverApi.cuMemcpyHtoDAsync_v2(destination, source, count, stream).ThrowOnError();
        else CudaDriverApi.cuMemcpyDtoDAsync_v2(destination, source, count, stream).ThrowOnError();
    }

    private sealed partial class CudaPrefixSnapshot(CudaQwen3HybridDenseTransformerModel owner, nint data, long bytes,
        int length, string identity, RecurrentPrefixSnapshotStorage storage) : IRecurrentPrefixSnapshot
    {
        private nint _data = data;
        public CudaQwen3HybridDenseTransformerModel Owner => owner;
        public nint Data => _data;
        public long ByteCount => bytes;
        public int PrefixLength => length;
        public string Identity => identity;
        public RecurrentPrefixSnapshotStorage Storage => storage;
        public void Dispose()
        {
            nint pointer = Interlocked.Exchange(ref _data, 0);
            if (pointer == 0) return;
            owner._context.MakeCurrent();
            if (Storage == RecurrentPrefixSnapshotStorage.Host) CudaDriverApi.cuMemFreeHost(pointer).ThrowOnError();
            else CudaDriverApi.cuMemFree_v2(pointer).ThrowOnError();
        }
    }
}

