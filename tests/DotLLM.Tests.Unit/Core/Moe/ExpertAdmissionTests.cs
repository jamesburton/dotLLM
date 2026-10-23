using DotLLM.Core.Moe;
using Xunit;

namespace DotLLM.Tests.Unit.Moe;

/// <summary>
/// Issue #1071: non-admitting lookups (<see cref="ExpertLookupResult.Cold"/>), <see cref="ExpertCache.Admit"/> /
/// <see cref="ExpertCache.Promote"/>, and promote-only admission with a decayed 64-token score
/// (<see cref="WindowedPromotionAdmission"/>). CPU only.
/// </summary>
public sealed class ExpertAdmissionTests
{
    private const long Bytes = 1000;

    private static ExpertCache Single(int experts, int slots, ExpertCachePolicy policy = ExpertCachePolicy.Slru)
        => new([experts], [0], [slots], policy);

    private static ExpertLookupResult[] LookCold(ExpertCache c, int layer, params int[] ids)
    {
        var slots = new int[ids.Length];
        var results = new ExpertLookupResult[ids.Length];
        c.Lookup(layer, ids, slots, results, admitMisses: false);
        return results;
    }

    /// <summary>One decode call of one layer: lookup without admission, observe, then offer every cold expert.</summary>
    private static List<int> Call(ExpertCache c, WindowedPromotionAdmission a, int layer, params int[] ids)
    {
        var results = LookCold(c, layer, ids);
        a.Observe(layer, ids, Enumerable.Repeat(1, ids.Length).ToArray());
        var promoted = new List<int>();
        for (int i = 0; i < ids.Length; i++)
            if (results[i] == ExpertLookupResult.Cold && a.TryPromote(layer, ids[i], Bytes, out _) >= 0)
                promoted.Add(ids[i]);
        return promoted;
    }

    [Fact]
    public void NonAdmittingLookup_LeavesMissesCold_HitsArePinnedAndCounted()
    {
        var c = Single(6, 2);
        c.Lookup(0, [0, 1], new int[2], new ExpertLookupResult[2]);           // admitting: 0, 1 resident
        var r = LookCold(c, 0, 0, 2, 3);
        Assert.Equal([ExpertLookupResult.Hit, ExpertLookupResult.Cold, ExpertLookupResult.Cold], r);
        Assert.Equal(-1, c.SlotOf(0, 2));
        Assert.Equal(1, c.Hits);
        Assert.Equal(4, c.Misses);
        Assert.Equal(2, c.ColdMisses);
        Assert.Equal(2, c.Insertions);
        Assert.True(c.IsPinned(0, c.SlotOf(0, 0)));
        Assert.False(c.IsPinned(0, c.SlotOf(0, 1)));
    }

    [Fact]
    public void Admit_PlacesAColdMissExactlyAsAnAdmittingLookupWould()
    {
        foreach (var policy in Enum.GetValues<ExpertCachePolicy>())
        {
            int[][] trace = [[0, 1, 2], [0, 3], [4, 1], [2, 5, 0], [3, 4]];
            var admitting = Single(6, 3, policy);
            var cold = Single(6, 3, policy);
            foreach (var ids in trace)
            {
                var s1 = new int[ids.Length];
                admitting.Lookup(0, ids, s1, new ExpertLookupResult[ids.Length]);
                var s2 = new int[ids.Length];
                var r2 = new ExpertLookupResult[ids.Length];
                cold.Lookup(0, ids, s2, r2, admitMisses: false);
                for (int i = 0; i < ids.Length; i++)
                    if (r2[i] == ExpertLookupResult.Cold) s2[i] = cold.Admit(0, ids[i], out _);
                Assert.Equal(s1, s2);
            }
            Assert.Equal(admitting.Evictions, cold.Evictions);
            Assert.Equal(admitting.Insertions, cold.Insertions);
            Assert.Equal(0, cold.ColdMisses);
        }
    }

    [Fact]
    public void Admit_And_Promote_RejectExpertsThatAreNotColdMissesOfTheCurrentCall()
    {
        var c = Single(6, 2);
        LookCold(c, 0, 0, 1);
        Assert.Equal(0, c.Admit(0, 0, out _));                                 // 0 was cold: admitted into slot 0
        Assert.Throws<InvalidOperationException>(() => c.Admit(0, 0, out _));  // now resident
        Assert.Throws<InvalidOperationException>(() => c.Admit(0, 4, out _));  // not part of the call
        Assert.Throws<InvalidOperationException>(() => c.Promote(0, 1, 0, out _));   // slot 0 is pinned by this call
        c.Promote(0, 1, 1, out bool evicted);
        Assert.False(evicted);
        Assert.Equal(1, c.SlotOf(0, 1));
        Assert.Equal(1, c.Promotions);
        Assert.Equal(1, c.ColdMisses);   // a promoted miss was still computed elsewhere this call
    }

    [Fact]
    public void Promotion_TakesFreeSlotsFirst_ThenOnlyBeatsTheWeakestResident()
    {
        var c = Single(8, 2);
        var a = new WindowedPromotionAdmission(c);
        a.BeginForward(1, long.MaxValue);
        Assert.Equal([0, 1], Call(c, a, 0, 0, 1));    // free slots: both promoted

        // Expert 2 has score 1, the residents 1 each: not strictly greater, so it stays cold.
        a.BeginForward(1, long.MaxValue);
        Assert.Equal([], Call(c, a, 0, 2));
        a.BeginForward(1, long.MaxValue);
        Assert.Equal([2], Call(c, a, 0, 2, 0));       // score 2 beats expert 1's 1 (0 is pinned: hit in this call)
        Assert.Equal(-1, c.SlotOf(0, 1));
        Assert.True(c.SlotOf(0, 2) >= 0 && c.SlotOf(0, 0) >= 0);
        Assert.Equal(3, c.Promotions);
        Assert.Equal(1, c.Evictions);
    }

    [Fact]
    public void Scores_DecayByHalfPerSixtyFourTokenWindow()
    {
        var c = Single(4, 1);
        var a = new WindowedPromotionAdmission(c, windowTokens: 64, decay: 0.5);
        a.BeginForward(1, 0);
        LookCold(c, 0, 1);
        a.Observe(0, [1], [8]);
        Assert.Equal(8, a.Score(0, 1));
        a.BeginForward(63, 0);                          // token 64: one window elapsed
        Assert.Equal(4, a.Score(0, 1));
        a.BeginForward(128, 0);                         // token 192: three windows
        Assert.Equal(1, a.Score(0, 1));
        Assert.Equal(0, a.Score(0, 2));
    }

    [Fact]
    public void Promotion_RespectsTheCopyBudget_AndRefillsPerForward()
    {
        var c = Single(8, 4);
        var a = new WindowedPromotionAdmission(c);
        a.BeginForward(1, copyBudgetBytes: 2 * Bytes);
        Assert.Equal([0, 1], Call(c, a, 0, 0, 1, 2));   // budget for two copies only
        Assert.Equal(0, a.RemainingBudgetBytes);
        a.BeginForward(1, copyBudgetBytes: 2 * Bytes);
        Assert.Equal([2, 3], Call(c, a, 0, 2, 3));
    }

    [Fact]
    public void PromoteOnly_CopiesFarFewerExpertsThanLru_OnATemporallyLocalTrace()
    {
        // A synthetic decode trace with a slowly drifting hot set (the #1063 section 4 shape): promote-only copies a fraction of
        // what LRU copies, for a smaller drop in hit rate.
        var rng = new Random(4);
        const int Experts = 64, Slots = 12, TopK = 4, Tokens = 4000;
        var lru = Single(Experts, Slots, ExpertCachePolicy.Lru);
        var promo = Single(Experts, Slots);
        var a = new WindowedPromotionAdmission(promo);
        long promoHits = 0, lruHits = 0;
        for (int t = 0; t < Tokens; t++)
        {
            int hotBase = (t / 500) * 3;
            var ids = new HashSet<int>();
            while (ids.Count < TopK)
                ids.Add(rng.NextDouble() < 0.8 ? (hotBase + rng.Next(10)) % Experts : rng.Next(Experts));
            int[] call = [.. ids.Order()];

            var r = new ExpertLookupResult[TopK];
            lru.Lookup(0, call, new int[TopK], r);
            lruHits += r.Count(x => x == ExpertLookupResult.Hit);

            a.BeginForward(1, long.MaxValue);
            var before = promo.Hits;
            Call(promo, a, 0, call);
            promoHits += promo.Hits - before;
        }
        long lruCopies = lru.Insertions, promoCopies = promo.Promotions;
        double lruRate = (double)lruHits / (Tokens * TopK), promoRate = (double)promoHits / (Tokens * TopK);
        Assert.True(promoCopies * 3 < lruCopies, $"promote-only copied {promoCopies}, LRU {lruCopies}");
        Assert.True(promoRate > lruRate - 0.15, $"promote-only hit rate {promoRate:P1} vs LRU {lruRate:P1}");
    }
}
