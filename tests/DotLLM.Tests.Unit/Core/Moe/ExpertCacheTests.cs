using DotLLM.Core.Moe;
using Xunit;

namespace DotLLM.Tests.Unit.Moe;

/// <summary>
/// Issue #1070: the device expert cache's bookkeeping (<see cref="ExpertCache"/>) on hand-built routing traces. CPU only.
/// </summary>
public sealed class ExpertCacheTests
{
    /// <summary>One layer of <paramref name="experts"/> experts in one pool of <paramref name="slots"/> slots.</summary>
    private static ExpertCache Single(int experts, int slots, ExpertCachePolicy policy, double protectedFraction = 0.8)
        => new([experts], [0], [slots], policy, protectedFraction);

    /// <summary>Looks up <paramref name="ids"/> in layer <paramref name="layer"/> and returns the outcomes and slots.</summary>
    private static (ExpertLookupResult[] Results, int[] Slots) Look(ExpertCache c, int layer, params int[] ids)
    {
        var slots = new int[ids.Length];
        var results = new ExpertLookupResult[ids.Length];
        c.Lookup(layer, ids, slots, results);
        return (results, slots);
    }

    private static int[] Resident(ExpertCache c, int layer, int experts)
        => [.. Enumerable.Range(0, experts).Where(e => c.SlotOf(layer, e) >= 0)];

    [Fact]
    public void ColdCache_FillsSlotsInIndexOrder_AllMisses()
    {
        foreach (var policy in Enum.GetValues<ExpertCachePolicy>())
        {
            var c = Single(5, 3, policy);
            var (r, s) = Look(c, 0, 0, 1, 2);
            Assert.All(r, x => Assert.Equal(ExpertLookupResult.Inserted, x));
            Assert.Equal([0, 1, 2], s);
            Assert.Equal(3, c.Misses);
            Assert.Equal(3, c.Insertions);
            Assert.Equal(0, c.Evictions);
            Assert.Equal(0, c.Hits);
        }
    }

    [Fact]
    public void Lru_EvictsLeastRecentlyUsed()
    {
        var c = Single(5, 3, ExpertCachePolicy.Lru);
        Look(c, 0, 0, 1, 2);
        Assert.Equal(ExpertLookupResult.Hit, Look(c, 0, 0).Results[0]);   // order (LRU first): 1, 2, 0
        Assert.Equal(1, Look(c, 0, 3).Slots[0]);                           // evicts 1 (slot 1)
        Assert.Equal(2, Look(c, 0, 4).Slots[0]);                           // evicts 2 (slot 2)
        Assert.Equal([0, 3, 4], Resident(c, 0, 5));
        Assert.Equal(2, c.Evictions);
        Assert.Equal(1, c.Hits);
        Assert.Equal(5, c.Misses);
    }

    [Fact]
    public void Slru_ReHitExpertsSurviveAScanThatLruWouldNot()
    {
        // Trace: fill {0,1,2}, re-hit 0 and 1, then two new experts. SLRU keeps the re-hit 0 and 1 in the protected segment and
        // recycles probation; LRU evicts 2 and then 0.
        int[][] trace = [[0, 1, 2], [0], [1], [3], [4]];
        var slru = Single(5, 3, ExpertCachePolicy.Slru);
        var lru = Single(5, 3, ExpertCachePolicy.Lru);
        foreach (var call in trace) { Look(slru, 0, call); Look(lru, 0, call); }
        Assert.Equal([0, 1, 4], Resident(slru, 0, 5));
        Assert.Equal([1, 3, 4], Resident(lru, 0, 5));
    }

    [Fact]
    public void Slru_PromotionOverCapacityDemotesTheProtectedLruEntry()
    {
        // 3 slots, protected capacity floor(3 * 0.8) = 2.
        var c = Single(6, 3, ExpertCachePolicy.Slru);
        Look(c, 0, 0, 1, 2);
        Look(c, 0, 0);          // protected: 0
        Look(c, 0, 1);          // protected: 0, 1        probation: 2
        Look(c, 0, 3);          // evicts 2 (probation)    probation: 3
        Look(c, 0, 3);          // promote 3 -> protected 0, 1, 3 over capacity -> demote 0 to probation
        Assert.Equal(ExpertLookupResult.Inserted, Look(c, 0, 4).Results[0]);  // victim: probation head = 0
        Assert.Equal(-1, c.SlotOf(0, 0));
        Assert.Equal([1, 3, 4], Resident(c, 0, 6));
    }

    [Fact]
    public void Slru_ZeroProtectedCapacity_BehavesAsLru()
    {
        int[][] trace = [[0, 1, 2], [0], [1], [3], [4], [2], [0]];
        var slru = Single(5, 3, ExpertCachePolicy.Slru, protectedFraction: 0);
        var lru = Single(5, 3, ExpertCachePolicy.Lru);
        foreach (var call in trace) { Look(slru, 0, call); Look(lru, 0, call); }
        Assert.Equal(Resident(lru, 0, 5), Resident(slru, 0, 5));
        Assert.Equal(lru.Hits, slru.Hits);
        Assert.Equal(lru.Evictions, slru.Evictions);
    }

    [Fact]
    public void Lfu_EvictsFewestLookups_TiesToLeastRecent()
    {
        var c = Single(6, 3, ExpertCachePolicy.Lfu);
        Look(c, 0, 0, 1, 2);       // freq 1, 1, 1
        Look(c, 0, 0);             // 0: 2
        Look(c, 0, 0);             // 0: 3
        Look(c, 0, 1);             // 1: 2
        Assert.Equal(2, Look(c, 0, 3).Slots[0]);   // evicts 2 (freq 1)
        Assert.Equal(2, Look(c, 0, 4).Slots[0]);   // evicts 3 (freq 1, the newest arrival but the only freq-1 entry)
        Assert.Equal([0, 1, 4], Resident(c, 0, 6));

        // Tie on frequency: 1 (freq 2, last used at call 4) and 4 (freq 2 after this hit, call 7): 1 is less recent.
        Look(c, 0, 4);
        Assert.Equal(1, c.SlotOf(0, 1));
        Assert.Equal(1, Look(c, 0, 5).Slots[0]);
        Assert.Equal(-1, c.SlotOf(0, 1));
        Assert.Equal([0, 4, 5], Resident(c, 0, 6));
    }

    [Fact]
    public void Lookup_PinsEveryIdOfTheCall_AndBypassesWhenEverySlotIsPinned()
    {
        foreach (var policy in Enum.GetValues<ExpertCachePolicy>())
        {
            var c = Single(8, 2, policy);
            Look(c, 0, 0);                                        // slot 0
            var (r, s) = Look(c, 0, 0, 1, 2);
            Assert.Equal([ExpertLookupResult.Hit, ExpertLookupResult.Inserted, ExpertLookupResult.Bypass], r);
            Assert.Equal([0, 1, -1], s);
            Assert.Equal(0, c.SlotOf(0, 0));                      // the hit was not evicted for the misses
            Assert.Equal(-1, c.SlotOf(0, 2));                     // a bypass does not change the cache
            Assert.Equal(1, c.Bypasses);
            Assert.Equal(0, c.Evictions);

            // Two misses in one call never share a slot, even though both residents are evictable.
            (r, s) = Look(c, 0, 3, 4);
            Assert.Equal([ExpertLookupResult.Inserted, ExpertLookupResult.Inserted], r);
            Assert.NotEqual(s[0], s[1]);
            Assert.Equal(2, c.Evictions);

            (r, s) = Look(c, 0, 5, 6, 7);
            Assert.Equal([ExpertLookupResult.Inserted, ExpertLookupResult.Inserted, ExpertLookupResult.Bypass], r);
            Assert.Equal(-1, s[2]);
        }
    }

    [Fact]
    public void Counters_HitsPlusMissesEqualLookups_InsertionsPlusBypassesEqualMisses()
    {
        var rng = new Random(1070);
        foreach (var policy in Enum.GetValues<ExpertCachePolicy>())
        {
            var c = new ExpertCache([16, 0, 12, 16], [0, 0, 1, 0], [5, 3], policy);
            long lookups = 0;
            for (int call = 0; call < 500; call++)
            {
                int layer = new[] { 0, 2, 3 }[rng.Next(3)];
                int n = layer == 2 ? 12 : 16;
                int[] ids = [.. Enumerable.Range(0, n).OrderBy(_ => rng.Next()).Take(rng.Next(1, 8)).Order()];
                Look(c, layer, ids);
                lookups += ids.Length;
            }
            Assert.Equal(lookups, c.Hits + c.Misses);
            Assert.Equal(c.Misses, c.Insertions + c.Bypasses);
            Assert.True(c.Evictions <= c.Insertions);
            Assert.True(c.Hits > 0 && c.Evictions > 0 && c.Bypasses > 0);

            c.ResetCounters();
            Assert.Equal(0, c.Hits + c.Misses + c.Evictions + c.Insertions + c.Bypasses);
        }
    }

    [Fact]
    public void LayoutPools_AreSharedAcrossLayersAndIsolatedFromEachOther()
    {
        // Layers 0 and 2 share pool 0 (2 slots); layer 1 has its own pool 1 (1 slot). Same expert id in two layers = two keys.
        var c = new ExpertCache([4, 4, 4], [0, 1, 0], [2, 1], ExpertCachePolicy.Lru);
        Assert.Equal(0, c.PoolOf(0));
        Assert.Equal(1, c.PoolOf(1));
        Assert.Equal(4, c.ExpertsIn(2));
        Assert.Equal(3, c.TotalSlots);

        Look(c, 0, 0, 1);                                   // pool 0: (0,0) (0,1)
        Look(c, 1, 0);                                      // pool 1: (1,0)
        Assert.Equal(ExpertLookupResult.Inserted, Look(c, 2, 0).Results[0]);   // pool 0 evicts (0,0), the LRU entry
        Assert.Equal(-1, c.SlotOf(0, 0));
        Assert.Equal((2, 0), c.ExpertAt(0, c.SlotOf(2, 0)));
        Assert.Equal(0, c.SlotOf(1, 0));                    // pool 1 untouched
        Assert.Equal((1, 0), c.ExpertAt(1, 0));
    }

    [Fact]
    public void ZeroSlotPool_BypassesEverything()
    {
        var c = Single(4, 0, ExpertCachePolicy.Slru);
        var (r, s) = Look(c, 0, 0, 3);
        Assert.Equal([ExpertLookupResult.Bypass, ExpertLookupResult.Bypass], r);
        Assert.Equal([-1, -1], s);
        Assert.Equal(2, c.Misses);
        Assert.Equal(0, c.Insertions);
    }

    [Fact]
    public void Clear_ForgetsResidentsButKeepsCounters()
    {
        var c = Single(4, 2, ExpertCachePolicy.Slru);
        Look(c, 0, 0, 1);
        Look(c, 0, 0);
        c.Clear();
        Assert.Empty(Resident(c, 0, 4));
        Assert.Equal((-1, -1), c.ExpertAt(0, 0));
        Assert.Equal(1, c.Hits);
        Assert.Equal([0, 1], Look(c, 0, 2, 3).Slots);
    }

    [Fact]
    public void Lookup_RejectsDuplicatesAndOutOfRangeIds()
    {
        var c = Single(4, 2, ExpertCachePolicy.Lru);
        Assert.Throws<ArgumentException>(() => Look(c, 0, 1, 1));
        Assert.Throws<ArgumentException>(() => Look(c, 0, 4));
        var noExperts = new ExpertCache([0, 4], [0, 0], [2], ExpertCachePolicy.Lru);
        Assert.Throws<ArgumentException>(() => Look(noExperts, 0, 0));
    }
}
