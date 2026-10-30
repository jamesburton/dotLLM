using DotLLM.Vulkan.Tiering;
using DotLLM.Tests.Unit.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan.Tiering;

/// <summary>
/// The per-layer slot allocator of expert tiering (#939): pure bookkeeping, no GPU. Each test pins a behaviour a plausible wrong implementation breaks
/// (evict MRU instead of LRU, evict an expert the same call needs, hand back a slot that does not hold the expert, count a duplicate as a miss).
/// </summary>
public sealed class ExpertSlotCacheTests
{
    private static (int[] Slots, int[] MissE, int[] MissS, int Misses) Look(ExpertSlotCache c, params int[] ids)
    {
        var slots = new int[ids.Length]; var me = new int[ids.Length]; var ms = new int[ids.Length];
        int n = c.Lookup(ids, slots, me, ms);
        return (slots, me[..n], ms[..n], n);
    }

    [Fact]
    public void ColdCache_MissesEverythingIntoDistinctFreeSlots()
    {
        var c = new ExpertSlotCache(numExperts: 16, slots: 4);
        var (slots, me, ms, n) = Look(c, 9, 3, 12);
        Assert.Equal(3, n);
        Assert.Equal([9, 3, 12], me);
        Assert.Equal(3, ms.Distinct().Count());
        Assert.Equal(ms, slots);                       // the slot reported per id is the slot the miss must be loaded into
        Assert.Equal(0, c.Hits); Assert.Equal(3, c.Misses); Assert.Equal(0, c.Evictions);
        for (int i = 0; i < me.Length; i++) { Assert.Equal(ms[i], c.SlotOf(me[i])); Assert.Equal(me[i], c.ExpertOf(ms[i])); }
    }

    [Fact]
    public void ResidentExperts_AreHits_AndReturnTheSlotTheyWereLoadedInto()
    {
        var c = new ExpertSlotCache(16, 4);
        var first = Look(c, 5, 6);
        var again = Look(c, 6, 5);
        Assert.Equal(0, again.Misses);
        Assert.Equal(first.Slots[1], again.Slots[0]);  // expert 6 is where it was loaded
        Assert.Equal(first.Slots[0], again.Slots[1]);
        Assert.Equal(2, c.Hits);
    }

    [Fact]
    public void Eviction_TakesTheLeastRecentlyUsed_NotTheMostRecent()
    {
        var c = new ExpertSlotCache(16, 3);
        Look(c, 1);            // oldest
        Look(c, 2);
        Look(c, 3);            // newest, cache full
        Look(c, 1);            // touch 1: 2 is now the least recently used
        var (_, me, ms, _) = Look(c, 4);
        Assert.Equal([4], me);
        Assert.Equal(-1, c.SlotOf(2));                 // 2 was evicted
        Assert.True(c.SlotOf(1) >= 0 && c.SlotOf(3) >= 0);
        Assert.Equal(c.SlotOf(4), ms[0]);
        Assert.Equal(1, c.Evictions);
    }

    [Fact]
    public void ExpertsTheSameCallNeeds_AreNeverEvictedByItsOwnMisses()
    {
        // 3 slots, resident {1,2,3} with 1 the LRU. A call needing {1, 4, 5}: 1 is a hit and must survive; the misses evict 2 and 3, not 1.
        var c = new ExpertSlotCache(16, 3);
        Look(c, 1); Look(c, 2); Look(c, 3);
        var (slots, me, ms, n) = Look(c, 1, 4, 5);
        Assert.Equal(2, n);
        Assert.True(c.SlotOf(1) >= 0, "a hit pinned by the call was evicted by the same call's misses");
        Assert.Equal(-1, c.SlotOf(2)); Assert.Equal(-1, c.SlotOf(3));
        Assert.Equal(3, slots.Distinct().Count());     // every id has its own slot at once
        Assert.Equal(c.SlotOf(4), slots[1]); Assert.Equal(c.SlotOf(5), slots[2]);
        Assert.Equal([4, 5], me);
    }

    [Fact]
    public void DuplicateIds_ResolveToOneSlot_AndCountOnce()
    {
        var c = new ExpertSlotCache(16, 4);
        var (slots, me, _, n) = Look(c, 7, 7);
        Assert.Equal(1, n); Assert.Equal([7], me);
        Assert.Equal(slots[0], slots[1]);
        Assert.Equal(1, c.Misses); Assert.Equal(0, c.Hits);
    }

    [Fact]
    public void MoreDistinctIdsThanSlots_Throws()
    {
        var c = new ExpertSlotCache(16, 2);
        Assert.Throws<InvalidOperationException>(() => Look(c, 1, 2, 3));
    }

    [Fact]
    public void OutOfRangeId_Throws()
    {
        var c = new ExpertSlotCache(16, 4);
        Assert.Throws<ArgumentOutOfRangeException>(() => Look(c, 16));
        Assert.Throws<ArgumentOutOfRangeException>(() => Look(c, -1));
    }

    [Fact]
    public void SlotsAreAPermutation_AfterALongRandomWorkload()
    {
        // Invariant check under churn: expert<->slot maps stay mutually consistent, and the slot a lookup returns holds that expert.
        var rng = new Random(11);
        var c = new ExpertSlotCache(64, 12);
        for (int step = 0; step < 5000; step++)
        {
            var ids = Enumerable.Range(0, 10).Select(_ => rng.Next(0, 64)).Distinct().ToArray();
            var (slots, _, _, _) = Look(c, ids);
            for (int i = 0; i < ids.Length; i++) Assert.Equal(ids[i], c.ExpertOf(slots[i]));
            Assert.Equal(ids.Length, slots.Distinct().Count());
        }
        int resident = 0;
        for (int e = 0; e < 64; e++) if (c.SlotOf(e) >= 0) { resident++; Assert.Equal(e, c.ExpertOf(c.SlotOf(e))); }
        Assert.Equal(c.Resident, resident);
        Assert.Equal(12, resident);
        Assert.Equal(c.Hits + c.Misses, c.Hits + c.Misses);
    }

    [Fact]
    public void HotWorkingSet_ThatFits_StopsMissing()
    {
        var c = new ExpertSlotCache(512, 64);
        var hot = Enumerable.Range(100, 40).ToArray();
        var rng = new Random(3);
        for (int i = 0; i < 200; i++) Look(c, hot.OrderBy(_ => rng.Next()).Take(10).ToArray());
        long missesBefore = c.Misses;
        for (int i = 0; i < 200; i++) Look(c, hot.OrderBy(_ => rng.Next()).Take(10).ToArray());
        Assert.Equal(missesBefore, c.Misses);          // the whole hot set is resident: no further misses
        Assert.True(c.Hits > 0);
    }

    [Fact]
    public void Clear_ForgetsEverythingButKeepsCounters_AndResetCountersKeepsResidents()
    {
        var c = new ExpertSlotCache(16, 4);
        Look(c, 1, 2);
        c.ResetCounters();
        Assert.Equal(0, c.Misses); Assert.True(c.SlotOf(1) >= 0);
        Look(c, 1, 3);
        Assert.Equal(1, c.Hits); Assert.Equal(1, c.Misses);
        c.Clear();
        Assert.Equal(-1, c.SlotOf(1)); Assert.Equal(0, c.Resident);
        Assert.Equal(1, c.Hits);
    }
}

/// <summary>Residency arithmetic of a tiered load (#939): the routed banks count only their slots, everything else is unchanged. No GPU.</summary>
public sealed class ExpertTierResidencyPlanTests
{
    [Fact]
    public void TieredEstimate_CountsOnlySlotBytes_OfThePackedRoutedBanks()
    {
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-tierplan-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string path = Path.Combine(dir, "q4e.gguf");
            File.WriteAllBytes(path, VulkanQwen4ExpParityTests.Build(11));
            using var gguf = DotLLM.Models.Gguf.GgufFile.Open(path);
            var cfg = DotLLM.Models.Gguf.GgufModelConfigExtractor.Extract(gguf.Metadata);
            var (full, _) = DotLLM.Vulkan.Qwen4ExpResidencyPlan.EstimateWeights(gguf.TensorsByName, cfg, integerDot: true, expertSlots: 0);
            var (tiered, _) = DotLLM.Vulkan.Qwen4ExpResidencyPlan.EstimateWeights(gguf.TensorsByName, cfg, integerDot: true, expertSlots: 32);
            long routed = 0;
            foreach (var (name, d) in gguf.TensorsByName)
                if (name.Contains("_exps.weight", StringComparison.Ordinal))
                    routed += DotLLM.Core.Configuration.QuantizationType.F32 == d.QuantizationType ? 0
                        : DotLLM.Cpu.Kernels.Dequantize.RowByteSize(d.Shape[0], d.QuantizationType) * d.Shape[1] * d.Shape[2];
            Assert.True(routed > 0, "fixture has no packed routed banks");
            long expected = routed - routed / cfg.Moe!.NumExperts * (cfg.Moe.NumExperts - 32);
            Assert.Equal(full - routed + expected, tiered);
            Assert.True(tiered < full / 2);
        }
        finally { try { Directory.Delete(dir, true); } catch (IOException) { } }
    }
}
