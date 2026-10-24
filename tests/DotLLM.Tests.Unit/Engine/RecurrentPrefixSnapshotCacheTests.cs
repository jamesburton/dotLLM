using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

public sealed class RecurrentPrefixSnapshotCacheTests
{
    private static int[] Tokens(int id, int n = 20) => Enumerable.Repeat(id, n).ToArray();

    [Fact]
    public void AlternatingPrefixes_UseIndependentSnapshotsAndLongestBoundary()
    {
        var provider = new Provider();
        using var cache = new RecurrentPrefixSnapshotCache(provider, new() { MaxEntries = 3, HostBudgetBytes = 300 });
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        provider.State = 11; Assert.True(cache.TryCapture(Tokens(1), kv, 18));
        provider.State = 22; Assert.True(cache.TryCapture(Tokens(2), kv, 19));
        provider.State = 33; Assert.True(cache.TryCapture(Tokens(1), kv, 19));
        Assert.Equal(19, cache.TryRestore(Tokens(1), kv, out var tier, out _)); Assert.Equal(33, provider.State); Assert.Equal("host", tier);
        Assert.Equal(19, cache.TryRestore(Tokens(2), kv, out _, out _)); Assert.Equal(22, provider.State);
        Assert.Equal(0, cache.TryRestore(Tokens(3), kv, out _, out _));
        Assert.Equal(2, cache.Statistics.HostHits);
    }

    [Fact]
    public void HostBudgetAndLru_EvictLeastRecentlyUsedBeforeAllocation()
    {
        var provider = new Provider();
        using var cache = new RecurrentPrefixSnapshotCache(provider, new() { MaxEntries = 20, HostBudgetBytes = 200 });
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        Assert.True(cache.TryCapture(Tokens(1), kv, 19)); Assert.True(cache.TryCapture(Tokens(2), kv, 19));
        Assert.Equal(19, cache.TryRestore(Tokens(1), kv, out _, out _));
        Assert.True(cache.TryCapture(Tokens(3), kv, 19));
        Assert.Equal(0, cache.TryRestore(Tokens(2), kv, out _, out _));
        Assert.Equal(19, cache.TryRestore(Tokens(1), kv, out _, out _));
        Assert.Equal(200, cache.Statistics.HostBytes); Assert.Equal(200, provider.PeakHostBytes);
        Assert.Equal(1, cache.Statistics.Evictions);
        cache.Clear(); Assert.Equal(0, provider.HostBytes);
    }

    [Fact]
    public void IdentityChangeAndRestoreFailure_BecomeMissesAndReleaseSnapshots()
    {
        var provider = new Provider();
        using var cache = new RecurrentPrefixSnapshotCache(provider, new() { MaxEntries = 2, HostBudgetBytes = 200 });
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        Assert.True(cache.TryCapture(Tokens(1), kv, 19));
        provider.PrefixSnapshotIdentity = "other-weights-or-kv-dtype";
        Assert.Equal(0, cache.TryRestore(Tokens(1), kv, out _, out _)); Assert.Equal(0, provider.HostBytes);
        Assert.True(cache.TryCapture(Tokens(1), kv, 19)); provider.FailRestore = true;
        Assert.Equal(0, cache.TryRestore(Tokens(1), kv, out _, out _)); Assert.Equal(1, cache.Statistics.RestoreFailures);
        Assert.Equal(0, provider.HostBytes);
    }

    [Fact]
    public void OversizedAndFailedCaptures_AreSafeMissesWithoutRetainedAllocations()
    {
        var provider = new Provider();
        using var cache = new RecurrentPrefixSnapshotCache(provider, new() { MaxEntries = 2, HostBudgetBytes = 99 });
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        Assert.False(cache.TryCapture(Tokens(1), kv, 19)); Assert.Equal(0, provider.PeakHostBytes);
    }

    [Fact]
    public void DeviceDemotion_RestoresBothTiersWithoutAliasing()
    {
        var provider = new Provider { DeviceLimit = 1 };
        using var cache = new RecurrentPrefixSnapshotCache(provider, new() { MaxEntries = 2, HostBudgetBytes = 100 });
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        provider.State = 11; Assert.True(cache.TryCapture(Tokens(1), kv, 19));
        provider.State = 22; Assert.True(cache.TryCapture(Tokens(2), kv, 19));
        Assert.Equal(19, cache.TryRestore(Tokens(1), kv, out var host, out _)); Assert.Equal("host", host); Assert.Equal(11, provider.State);
        Assert.Equal(19, cache.TryRestore(Tokens(2), kv, out var vram, out _)); Assert.Equal("vram", vram); Assert.Equal(22, provider.State);
        Assert.Equal(100, provider.HostBytes); Assert.Equal(100, provider.DeviceBytes);
    }

    [Fact]
    public void CapacityAndTierPressure_EvictGlobalLruWithoutDiscardingNewerHostEntry()
    {
        var provider = new Provider { DeviceLimit = 1 };
        using var cache = new RecurrentPrefixSnapshotCache(provider, new() { MaxEntries = 2, HostBudgetBytes = 100 });
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        Assert.True(cache.TryCapture(Tokens(1), kv, 19)); Assert.True(cache.TryCapture(Tokens(2), kv, 19));
        Assert.Equal(19, cache.TryRestore(Tokens(1), kv, out _, out _)); // A host is newer than B device
        Assert.True(cache.TryCapture(Tokens(3), kv, 19));
        Assert.Equal(19, cache.TryRestore(Tokens(1), kv, out _, out _));
        Assert.Equal(0, cache.TryRestore(Tokens(2), kv, out _, out _));
        Assert.Equal(1, cache.Statistics.Evictions);
    }
    [Fact]
    public void CaptureAndDemotionFailures_FallBackWithoutLosingIndependentState()
    {
        var provider = new Provider { DeviceLimit = 1 };
        using var cache = new RecurrentPrefixSnapshotCache(provider, new() { MaxEntries = 3, HostBudgetBytes = 200 });
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        provider.State = 11; Assert.True(cache.TryCapture(Tokens(1), kv, 19));
        provider.FailTransfer = true; provider.State = 22;
        Assert.True(cache.TryCapture(Tokens(2), kv, 19)); // failed demotion falls back to host capture
        Assert.Equal(19, cache.TryRestore(Tokens(1), kv, out _, out _)); Assert.Equal(11, provider.State);
        Assert.Equal(19, cache.TryRestore(Tokens(2), kv, out _, out _)); Assert.Equal(22, provider.State);
        provider.FailCapture = true;
        Assert.False(cache.TryCapture(Tokens(3), kv, 19));
        Assert.Equal(19, cache.TryRestore(Tokens(1), kv, out _, out _));
        Assert.Equal(100, provider.HostBytes); Assert.Equal(100, provider.DeviceBytes);
    }
    [Fact]
    public void ActiveRequestGrowth_DemotesSnapshotsBeforeRestoringOrColdPrefill()
    {
        var provider = new Provider { DeviceLimit = 2 };
        using var cache = new RecurrentPrefixSnapshotCache(provider, new() { MaxEntries = 3, HostBudgetBytes = 200 });
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        provider.State = 11; Assert.True(cache.TryCapture(Tokens(1), kv, 19));
        provider.State = 22; Assert.True(cache.TryCapture(Tokens(2), kv, 19));
        provider.RequestNeedsDeviceSpace = true;
        Assert.Equal(0, cache.TryRestore(Tokens(3), kv, out _, out _)); // even a miss needs room for active prefill
        Assert.Equal(0, provider.DeviceBytes); Assert.Equal(200, provider.HostBytes);
        Assert.Equal(19, cache.TryRestore(Tokens(1), kv, out var tier, out _));
        Assert.Equal("host", tier); Assert.Equal(11, provider.State);
    }
    [Fact]
    public void DisposeFailures_DoNotEscapeRestoreMissOrPreventClearingOtherEntries()
    {
        var provider = new Provider();
        using var cache = new RecurrentPrefixSnapshotCache(provider, new() { MaxEntries = 3, HostBudgetBytes = 300 });
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        Assert.True(cache.TryCapture(Tokens(1), kv, 19)); Assert.True(cache.TryCapture(Tokens(2), kv, 19));
        provider.FailRestore = true; provider.FailDispose = true;
        var restoreError = Record.Exception(() => Assert.Equal(0, cache.TryRestore(Tokens(1), kv, out _, out _)));
        var clearError = Record.Exception(cache.Clear);
        long retainedAfterClear = provider.HostBytes;
        provider.FailDispose = false; cache.Clear();
        Assert.Null(restoreError); Assert.Null(clearError); Assert.Equal(0, retainedAfterClear);
    }

    [Fact]
    public void FailedCaptureCleanup_DisposeFailureStillReturnsMiss()
    {
        var provider = new Provider { InvalidCapture = true, FailDispose = true };
        using var cache = new RecurrentPrefixSnapshotCache(provider, new() { MaxEntries = 2, HostBudgetBytes = 200 });
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        var error = Record.Exception(() => Assert.False(cache.TryCapture(Tokens(1), kv, 19)));
        provider.FailDispose = false;
        Assert.Null(error); Assert.Equal(0, provider.HostBytes);
    }
    private sealed class Provider : IRecurrentPrefixSnapshotProvider
    {
        public string PrefixSnapshotIdentity { get; set; } = "model-f32-rope-mtp";
        public int State; public bool FailRestore; public bool FailCapture; public bool FailTransfer; public bool FailDispose; public bool InvalidCapture;
        public long HostBytes; public long PeakHostBytes; public long DeviceBytes; public int DeviceLimit; public bool RequestNeedsDeviceSpace;
        public bool SupportsPrefixSnapshotStorage(RecurrentPrefixSnapshotStorage storage) => storage == RecurrentPrefixSnapshotStorage.Host || DeviceLimit > 0;
        public bool HasPrefixSnapshotRequestHeadroom(IKvCache kvCache) => !RequestNeedsDeviceSpace || DeviceBytes == 0;
        public long EstimatePrefixSnapshotBytes(IKvCache kvCache, int prefixLength) => 100;
        public IRecurrentPrefixSnapshot CapturePrefixSnapshot(IKvCache kvCache, int prefixLength, RecurrentPrefixSnapshotStorage storage)
        {
            if (FailCapture) throw new IOException("injected capture failure");
            return Allocate(PrefixSnapshotIdentity, InvalidCapture ? prefixLength + 1 : prefixLength, State, storage);
        }
        public IRecurrentPrefixSnapshot TransferPrefixSnapshot(IRecurrentPrefixSnapshot source, RecurrentPrefixSnapshotStorage storage)
        {
            if (FailTransfer) throw new IOException("injected transfer failure");
            return Allocate(source.Identity, source.PrefixLength, ((Snapshot)source).State, storage);
        }
        private IRecurrentPrefixSnapshot Allocate(string identity, int length, int state, RecurrentPrefixSnapshotStorage storage)
        {
            if (storage == RecurrentPrefixSnapshotStorage.Device)
            {
                if (DeviceBytes / 100 >= DeviceLimit) throw new InvalidOperationException("device admission refused");
                DeviceBytes += 100;
            }
            else { HostBytes += 100; PeakHostBytes = Math.Max(PeakHostBytes, HostBytes); }
            return new Snapshot(this, identity, length, state, storage);
        }
        public void RestorePrefixSnapshot(IRecurrentPrefixSnapshot snapshot, IKvCache kvCache)
        {
            if (FailRestore) throw new IOException("injected copy failure");
            State = ((Snapshot)snapshot).State;
        }
        private sealed class Snapshot(Provider owner, string identity, int length, int state, RecurrentPrefixSnapshotStorage storage) : IRecurrentPrefixSnapshot
        {
            private bool _disposed;
            public int State => state;
            public long ByteCount => 100;
            public string Identity => identity;
            public int PrefixLength => length;
            public RecurrentPrefixSnapshotStorage Storage => storage;
            public void Dispose() { if (!_disposed) { _disposed = true; if (storage == RecurrentPrefixSnapshotStorage.Host) owner.HostBytes -= 100; else owner.DeviceBytes -= 100; if (owner.FailDispose) throw new IOException("injected free failure"); } }
        }
    }
}




