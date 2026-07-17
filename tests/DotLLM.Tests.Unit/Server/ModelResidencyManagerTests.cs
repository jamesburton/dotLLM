using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Pure unit tests for <see cref="ModelResidencyManager"/> (#369): LRU eviction under
/// count/byte budgets and keep-alive expiry sweeping. No real model weights are needed — snapshots
/// carry only bookkeeping fields (<c>Model</c> etc. are left null, which <see cref="ResidentModelSnapshot.Dispose"/>
/// tolerates).
/// </summary>
public sealed class ModelResidencyManagerTests
{
    private static ResidentModelSnapshot Snapshot(string key, long bytes, DateTimeOffset lastUsed, double? keepAlive = null) => new()
    {
        Key = key,
        Options = new ServerOptions { Model = key, ModelId = key },
        EstimatedBytes = bytes,
        LastUsedUtc = lastUsed,
        KeepAliveSecondsOverride = keepAlive,
    };

    [Fact]
    public void EnforceBudget_DefaultMaxOne_EvictsExistingStashImmediately()
    {
        // Default MaxResidentModels = 1 must reproduce the original hot-swap-always-disposes
        // behavior: stashing anything while a new model is about to become active evicts it.
        var mgr = new ModelResidencyManager();
        mgr.Stash(Snapshot("model-a", bytes: 100, DateTimeOffset.UtcNow));

        var evicted = mgr.EnforceBudget(activeBytes: 200);

        Assert.Single(evicted);
        Assert.Equal("model-a", evicted[0].Key);
        Assert.Equal(0, mgr.StashedCount);
    }

    [Fact]
    public void EnforceBudget_MaxResidentModelsTwo_KeepsOneStashedAlongsideActive()
    {
        var mgr = new ModelResidencyManager { MaxResidentModels = 2 };
        mgr.Stash(Snapshot("model-a", bytes: 100, DateTimeOffset.UtcNow));

        var evicted = mgr.EnforceBudget(activeBytes: 200); // 1 stashed + 1 active = 2, fits exactly

        Assert.Empty(evicted);
        Assert.Equal(1, mgr.StashedCount);
        Assert.True(mgr.Contains("model-a"));
    }

    [Fact]
    public void EnforceBudget_CountBudget_EvictsLeastRecentlyUsedFirst()
    {
        var mgr = new ModelResidencyManager { MaxResidentModels = 2 };
        var now = DateTimeOffset.UtcNow;
        mgr.Stash(Snapshot("oldest", bytes: 10, now - TimeSpan.FromMinutes(10)));
        mgr.Stash(Snapshot("newer", bytes: 10, now - TimeSpan.FromMinutes(1)));

        // Adding a 3rd (the about-to-be-active model) pushes total to 3 > MaxResidentModels(2).
        var evicted = mgr.EnforceBudget(activeBytes: 10);

        Assert.Single(evicted);
        Assert.Equal("oldest", evicted[0].Key);
        Assert.True(mgr.Contains("newer"));
        Assert.False(mgr.Contains("oldest"));
    }

    [Fact]
    public void EnforceBudget_ByteBudget_EvictsUntilUnderBudgetRegardlessOfCount()
    {
        var mgr = new ModelResidencyManager { MaxResidentModels = 10, MemoryBudgetBytes = 250 };
        var now = DateTimeOffset.UtcNow;
        mgr.Stash(Snapshot("a", bytes: 100, now - TimeSpan.FromMinutes(5)));
        mgr.Stash(Snapshot("b", bytes: 100, now - TimeSpan.FromMinutes(3)));

        // Active model is 100 bytes -> total would be 300 > 250 budget. Evict LRU ("a") to get to 200.
        var evicted = mgr.EnforceBudget(activeBytes: 100);

        Assert.Single(evicted);
        Assert.Equal("a", evicted[0].Key);
        Assert.True(mgr.Contains("b"));
    }

    [Fact]
    public void EnforceBudget_UnlimitedByteBudget_OnlyCountBounds()
    {
        var mgr = new ModelResidencyManager { MaxResidentModels = 5, MemoryBudgetBytes = 0 };
        mgr.Stash(Snapshot("a", bytes: long.MaxValue / 4, DateTimeOffset.UtcNow));

        var evicted = mgr.EnforceBudget(activeBytes: long.MaxValue / 4);

        Assert.Empty(evicted); // 0 = unlimited byte budget, well under the count cap of 5
    }

    [Fact]
    public void SweepExpired_RemovesModelsPastKeepAlive_KeepsOthers()
    {
        var mgr = new ModelResidencyManager { DefaultKeepAliveSeconds = 60 };
        var now = DateTimeOffset.UtcNow;
        mgr.Stash(Snapshot("expired", bytes: 1, now - TimeSpan.FromSeconds(120))); // past default keep-alive
        mgr.Stash(Snapshot("fresh", bytes: 1, now - TimeSpan.FromSeconds(5)));     // well within it

        var expired = mgr.SweepExpired(now);

        Assert.Single(expired);
        Assert.Equal("expired", expired[0].Key);
        Assert.True(mgr.Contains("fresh"));
        Assert.False(mgr.Contains("expired"));
    }

    [Fact]
    public void SweepExpired_NegativeKeepAliveOverride_NeverExpires()
    {
        var mgr = new ModelResidencyManager { DefaultKeepAliveSeconds = 1 };
        var now = DateTimeOffset.UtcNow;
        mgr.Stash(Snapshot("pinned", bytes: 1, now - TimeSpan.FromHours(1), keepAlive: -1));

        var expired = mgr.SweepExpired(now);

        Assert.Empty(expired);
        Assert.True(mgr.Contains("pinned"));
    }

    [Fact]
    public void SweepExpired_ZeroKeepAlive_ExpiresImmediately()
    {
        var mgr = new ModelResidencyManager { DefaultKeepAliveSeconds = 300 };
        var now = DateTimeOffset.UtcNow;
        mgr.Stash(Snapshot("ephemeral", bytes: 1, now - TimeSpan.FromMilliseconds(1), keepAlive: 0));

        var expired = mgr.SweepExpired(now);

        Assert.Single(expired);
        Assert.Equal("ephemeral", expired[0].Key);
    }

    [Fact]
    public void TryTake_RemovesAndReturnsSnapshot_MissingKeyReturnsNull()
    {
        var mgr = new ModelResidencyManager { MaxResidentModels = 2 };
        mgr.Stash(Snapshot("model-a", bytes: 1, DateTimeOffset.UtcNow));

        var taken = mgr.TryTake("model-a");
        Assert.NotNull(taken);
        Assert.Equal("model-a", taken!.Key);
        Assert.False(mgr.Contains("model-a"));

        Assert.Null(mgr.TryTake("model-a"));
        Assert.Null(mgr.TryTake("does-not-exist"));
    }

    [Fact]
    public void Snapshot_ReportsExpiresInSeconds_AndNullForNeverExpiring()
    {
        var mgr = new ModelResidencyManager { DefaultKeepAliveSeconds = 100 };
        var now = DateTimeOffset.UtcNow;
        mgr.Stash(Snapshot("a", bytes: 1, now - TimeSpan.FromSeconds(40)));
        mgr.Stash(Snapshot("b", bytes: 1, now, keepAlive: -1));

        var infos = mgr.Snapshot(now).ToDictionary(i => i.Key);

        Assert.InRange(infos["a"].ExpiresInSeconds!.Value, 55, 61);
        Assert.Null(infos["b"].ExpiresInSeconds);
    }
}
