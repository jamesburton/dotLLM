using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;

namespace DotLLM.Engine.PromptCache;

/// <summary>Serialized, bounded retention of independent recurrent snapshots. Tokens always name an exact state boundary.</summary>
internal sealed class RecurrentPrefixSnapshotCache : IDisposable
{
    private readonly IRecurrentPrefixSnapshotProvider _provider;
    private readonly RecurrentPrefixCacheOptions _options;
    private readonly object _gate = new();
    private readonly List<Entry> _entries = [];
    private readonly RecurrentPrefixDiskStore? _disk;
    private string _identity;
    private long _clock, _hits, _vramHits, _hostHits, _diskHits, _misses, _evictions, _restoreFailures;
    private sealed class Entry(int[] tokens, IRecurrentPrefixSnapshot snapshot, long use)
    {
        public int[] Tokens = tokens;
        public IRecurrentPrefixSnapshot? Snapshot = snapshot;
        public RecurrentPrefixDiskStore.FileEntry? Disk;
        public long RawBytes => Snapshot?.ByteCount ?? Disk!.RawBytes;
        public long Use = use;
    }

    public RecurrentPrefixSnapshotCache(IRecurrentPrefixSnapshotProvider provider, RecurrentPrefixCacheOptions options)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(options.MaxEntries, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(options.HostBudgetBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(options.DiskBudgetBytes);
        if ((options.DiskBudgetBytes > 0) != !string.IsNullOrWhiteSpace(options.DiskDirectory))
            throw new ArgumentException("Disk directory and positive disk budget must be configured together.", nameof(options));
        _provider = provider; _options = options; _identity = provider.PrefixSnapshotIdentity;
        if (options.DiskBudgetBytes > 0 && provider is IPersistableRecurrentPrefixSnapshotProvider)
        {
            try { _disk = new RecurrentPrefixDiskStore(options.DiskDirectory!, options.DiskBudgetBytes); }
            catch (IOException) { /* Unavailable storage falls back to the existing memory tiers. */ }
            catch (UnauthorizedAccessException) { }
        }
    }

    public RecurrentPrefixCacheStatistics Statistics
    {
        get
        {
            lock (_gate) return new()
            {
                Enabled = true, Entries = _entries.Count, MaxEntries = _options.MaxEntries,
                HostBytes = HostBytes, HostBudgetBytes = _options.HostBudgetBytes,
                DiskBytes = _disk?.Bytes ?? 0, DiskBudgetBytes = _options.DiskBudgetBytes,
                DiskEntries = _entries.Count(e => e.Disk is not null), DiskHits = _diskHits,
                DiskTransport = _disk is null ? null : "file-stream",
                Hits = _hits, VramHits = _vramHits, HostHits = _hostHits, Misses = _misses,
                Evictions = _evictions, RestoreFailures = _restoreFailures,
            };
        }
    }

    private long HostBytes => _entries.Where(e => e.Snapshot?.Storage == RecurrentPrefixSnapshotStorage.Host).Sum(e => e.RawBytes);

    public int TryRestore(int[] tokens, IKvCache kv, out string? tier, out double milliseconds)
    {
        lock (_gate)
        {
            CheckIdentity(); tier = null; milliseconds = 0;
            var entry = _entries.Where(e => e.Tokens.Length < tokens.Length && tokens.AsSpan(0, e.Tokens.Length).SequenceEqual(e.Tokens))
                .OrderByDescending(e => e.Tokens.Length).FirstOrDefault();
            if (!PrepareRequest(kv, entry) || entry is null || !_entries.Contains(entry)) { _misses++; return 0; }
            long start = Stopwatch.GetTimestamp();
            try
            {
                bool diskHit = entry.Disk is not null;
                if (diskHit) LoadDiskEntry(entry, kv);
                _provider.RestorePrefixSnapshot(entry.Snapshot!, kv);
                milliseconds = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                entry.Use = ++_clock; _hits++;
                if (diskHit) { _diskHits++; tier = "disk"; }
                else if (entry.Snapshot!.Storage == RecurrentPrefixSnapshotStorage.Host) { _hostHits++; tier = "host"; }
                else { _vramHits++; tier = "vram"; }
                return entry.Tokens.Length;
            }
            catch (Exception)
            {
                // A partial copy must never be treated as a hit. The generator resets all live state before cold prefill.
                _restoreFailures++; _misses++; Remove(entry, eviction: false); return 0;
            }
        }
    }

    public bool TryCapture(int[] tokens, IKvCache kv, int length)
    {
        lock (_gate)
        {
            CheckIdentity();
            IRecurrentPrefixSnapshot? snapshot = null;
            try
            {
                long bytes = _provider.EstimatePrefixSnapshotBytes(kv, length);
                if (bytes <= 0) return false;
                if (!_provider.SupportsPrefixSnapshotStorage(RecurrentPrefixSnapshotStorage.Device)
                    && (!_provider.SupportsPrefixSnapshotStorage(RecurrentPrefixSnapshotStorage.Host) || bytes > _options.HostBudgetBytes)) return false;
                // Advancing a conversation replaces its own boundary, rather than evicting another conversation.
                var superseded = _entries.Where(e => e.Tokens.Length <= length
                    && tokens.AsSpan(0, e.Tokens.Length).SequenceEqual(e.Tokens)).MaxBy(e => e.Tokens.Length);
                if (superseded is not null) Remove(superseded, eviction: false);
                // Capacity is global across tiers. Reclaim the true LRU before temporary transport allocations.
                while (_entries.Count >= _options.MaxEntries) Remove(_entries.MinBy(e => e.Use)!, eviction: true);
                if (_provider.SupportsPrefixSnapshotStorage(RecurrentPrefixSnapshotStorage.Device))
                {
                    snapshot = TryDeviceCapture();
                    // Reclaim cold device copies into the bounded host tier before retrying admission.
                    foreach (var cold in _entries.Where(e => e.Snapshot?.Storage == RecurrentPrefixSnapshotStorage.Device).OrderBy(e => e.Use).ToArray())
                    {
                        if (snapshot is not null) break;
                        if (!MakeHostRoom(cold.RawBytes, cold))
                        {
                            if (!Spill(cold, null)) Remove(cold, eviction: true);
                            snapshot = TryDeviceCapture();
                            continue;
                        }
                        try
                        {
                            var host = _provider.TransferPrefixSnapshot(cold.Snapshot!, RecurrentPrefixSnapshotStorage.Host);
                            var previous = cold.Snapshot; cold.Snapshot = host; Release(previous);
                        }
                        catch (Exception) { continue; }
                        snapshot = TryDeviceCapture();
                    }
                }
                if (snapshot is null && _provider.SupportsPrefixSnapshotStorage(RecurrentPrefixSnapshotStorage.Host) && MakeHostRoom(bytes))
                    snapshot = _provider.CapturePrefixSnapshot(kv, length, RecurrentPrefixSnapshotStorage.Host);
                if (snapshot is null) return false;
                if (snapshot.ByteCount != bytes || snapshot.PrefixLength != length || snapshot.Identity != _identity)
                    throw new InvalidOperationException("Provider returned an incompatible prefix snapshot.");
                var prefix = tokens.AsSpan(0, length).ToArray();
                foreach (var duplicate in _entries.Where(e => e.Tokens.AsSpan().SequenceEqual(prefix)).ToArray()) Remove(duplicate, eviction: false);
                while (_entries.Count >= _options.MaxEntries) Remove(_entries.MinBy(e => e.Use)!, eviction: true);
                _entries.Add(new Entry(prefix, snapshot, ++_clock)); snapshot = null;
                return true;

                IRecurrentPrefixSnapshot? TryDeviceCapture()
                {
                    try { return _provider.CapturePrefixSnapshot(kv, length, RecurrentPrefixSnapshotStorage.Device); }
                    catch (Exception) { return null; }
                }
            }
            catch (Exception) { Release(snapshot); return false; }
        }
    }

    private bool PrepareRequest(IKvCache kv, Entry? protectedEntry)
    {
        try
        {
            if (_provider.HasPrefixSnapshotRequestHeadroom(kv)) return true;
            foreach (var cold in _entries.Where(e => e.Snapshot?.Storage == RecurrentPrefixSnapshotStorage.Device).OrderBy(e => e.Use).ToArray())
            {
                bool demoted = false;
                if (MakeHostRoom(cold.RawBytes, cold, protectedEntry))
                {
                    try
                    {
                        var host = _provider.TransferPrefixSnapshot(cold.Snapshot!, RecurrentPrefixSnapshotStorage.Host);
                        var previous = cold.Snapshot; cold.Snapshot = host; demoted = true; Release(previous);
                    }
                    catch (Exception) { /* A failed demotion must not pin device memory needed by the active request. */ }
                }
                if (!demoted && !Spill(cold, protectedEntry)) Remove(cold, eviction: true);
                if (_provider.HasPrefixSnapshotRequestHeadroom(kv)) return true;
            }
            return _provider.HasPrefixSnapshotRequestHeadroom(kv);
        }
        catch (Exception)
        {
            // A failed memory query cannot leave device snapshots blocking cold prefill.
            foreach (var entry in _entries.Where(e => e.Snapshot?.Storage == RecurrentPrefixSnapshotStorage.Device).ToArray())
                Remove(entry, eviction: true);
            return false;
        }
    }
    private bool MakeHostRoom(long bytes, Entry? except = null, Entry? protectedDisk = null)
    {
        if (bytes > _options.HostBudgetBytes || !_provider.SupportsPrefixSnapshotStorage(RecurrentPrefixSnapshotStorage.Host)) return false;
        while (HostBytes > _options.HostBudgetBytes - bytes)
        {
            var oldest = _entries.Where(e => !ReferenceEquals(e, except) && e.Snapshot?.Storage == RecurrentPrefixSnapshotStorage.Host).MinBy(e => e.Use);
            if (oldest is null) return false;
            if (Spill(oldest, protectedDisk ?? except)) continue;
            if (except is not null && oldest.Use > except.Use && except.Disk is null) return false;
            Remove(oldest, eviction: true);
        }
        return true;
    }

    private bool Spill(Entry entry, Entry? protect)
    {
        if (_disk is null || entry.Snapshot is not IPersistableRecurrentPrefixSnapshot snapshot) return false;
        try
        {
            long payloadBytes = _disk.Measure(snapshot);
            long fileBytes = checked(payloadBytes + RecurrentPrefixDiskStore.HeaderBytes);
            while (_disk.Bytes > _options.DiskBudgetBytes - fileBytes)
            {
                var oldest = _entries.Where(e => e.Disk is not null && !ReferenceEquals(e, protect)).MinBy(e => e.Use);
                if (oldest is null) return false;
                Remove(oldest, eviction: true);
            }
            var file = _disk.Write(snapshot, entry.Tokens, payloadBytes);
            entry.Disk = file; entry.Snapshot = null; Release(snapshot);
            return true;
        }
        catch (Exception) { return false; }
    }

    private void LoadDiskEntry(Entry entry, IKvCache kv)
    {
        var file = entry.Disk!;
        // Validate the entire file before evicting any valid host state or allocating pinned memory.
        using var source = _disk!.OpenVerified(file, entry.Tokens, _identity);
        IRecurrentPrefixSnapshot? restored = null;
        try
        {
            if (_provider.SupportsPrefixSnapshotStorage(RecurrentPrefixSnapshotStorage.Device))
            {
                TryDeviceImport();
                foreach (var cold in _entries.Where(e => e.Snapshot?.Storage == RecurrentPrefixSnapshotStorage.Device)
                    .OrderBy(e => e.Use).ToArray())
                {
                    if (restored is not null) break;
                    bool demoted = false;
                    if (MakeHostRoom(cold.RawBytes, cold, entry))
                    {
                        try
                        {
                            var host = _provider.TransferPrefixSnapshot(cold.Snapshot!, RecurrentPrefixSnapshotStorage.Host);
                            var previous = cold.Snapshot; cold.Snapshot = host; demoted = true; Release(previous);
                        }
                        catch (Exception) { }
                    }
                    if (!demoted && !Spill(cold, entry)) Remove(cold, eviction: true);
                    TryDeviceImport();
                }
            }
            if (restored is null)
            {
                if (!MakeHostRoom(file.RawBytes, entry)) throw new IOException("No host capacity for disk snapshot restore.");
                source.Position = RecurrentPrefixDiskStore.HeaderBytes;
                restored = ((IPersistableRecurrentPrefixSnapshotProvider)_provider).ReadPrefixSnapshot(
                    source, RecurrentPrefixSnapshotStorage.Host, file.RawBytes);
                ValidateImport(restored, RecurrentPrefixSnapshotStorage.Host);
            }
            entry.Snapshot = restored; restored = null;
            _disk.Delete(file); entry.Disk = null;

            void ValidateImport(IRecurrentPrefixSnapshot candidate, RecurrentPrefixSnapshotStorage storage)
            {
                if (source.Position != file.FileBytes || candidate.ByteCount != file.RawBytes
                    || candidate.PrefixLength != entry.Tokens.Length || candidate.Identity != _identity || candidate.Storage != storage)
                    throw new InvalidDataException("Provider imported an incompatible disk snapshot.");
            }
            void TryDeviceImport()
            {
                try
                {
                    source.Position = RecurrentPrefixDiskStore.HeaderBytes;
                    restored = ((IPersistableRecurrentPrefixSnapshotProvider)_provider).ReadPrefixSnapshot(
                        source, RecurrentPrefixSnapshotStorage.Device, file.RawBytes);
                    ValidateImport(restored, RecurrentPrefixSnapshotStorage.Device);
                    // Import itself consumes VRAM. Admission must still leave room for live KV growth.
                    if (!_provider.HasPrefixSnapshotRequestHeadroom(kv)) { Release(restored); restored = null; }
                }
                catch (Exception) { Release(restored); restored = null; }
            }
        }
        finally { Release(restored); }
    }

    private void CheckIdentity()
    {
        string identity = _provider.PrefixSnapshotIdentity;
        if (identity == _identity) return;
        ClearCore(); _identity = identity;
    }

    private void Remove(Entry entry, bool eviction)
    {
        _entries.Remove(entry); Release(entry.Snapshot);
        if (entry.Disk is { } file) _disk?.Delete(file);
        if (eviction) _evictions++;
    }
    private static void Release(IRecurrentPrefixSnapshot? snapshot)
    {
        try { snapshot?.Dispose(); }
        catch (Exception) { /* Cache cleanup must not mask a restore/copy failure or strand the other entries. */ }
    }
    private void ClearCore() { foreach (var e in _entries) Release(e.Snapshot); _entries.Clear(); _disk?.Clear(); }
    public void Clear() { lock (_gate) ClearCore(); }
    public void Dispose() { lock (_gate) { ClearCore(); _disk?.Dispose(); } }
}




