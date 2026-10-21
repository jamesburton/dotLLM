using DotLLM.Core.Moe;
using Xunit;

namespace DotLLM.Tests.Unit.Moe;

/// <summary>Issue #1070: the expert-offload budget arithmetic, slot split, spill rule and options parsing. CPU only.</summary>
public sealed class MoeOffloadPlannerTests
{
    private const long MiB = 1024L * 1024, GiB = 1024 * MiB;

    [Fact]
    public void CacheBudget_IsFreeMinusReserveMinusEverythingElse_NeverNegative()
    {
        var other = new MoeOffloadPlanner.Reservations(AlwaysResidentWeights: 3 * GiB, ForwardState: 1 * GiB, KvCache: 512 * MiB, Scratch: 256 * MiB);
        Assert.Equal(4 * GiB + 768 * MiB, other.Total);
        Assert.Equal(10 * GiB - 512 * MiB - other.Total, MoeOffloadPlanner.CacheBudgetBytes(10 * GiB, 512 * MiB, other));
        Assert.Equal(0, MoeOffloadPlanner.CacheBudgetBytes(4 * GiB, 512 * MiB, other));
    }

    [Theory]
    [InlineData(null, 6144L, false)]    // auto: the whole budget
    [InlineData(2048L, 2048L, false)]   // below the budget: honoured
    [InlineData(9000L, 6144L, true)]    // above: clamped, with a warning
    public void ResolveCacheBytes_ClampsRequestsToTheBudget(long? requestMib, long expectMib, bool expectClamped)
    {
        long bytes = MoeOffloadPlanner.ResolveCacheBytes(6144 * MiB, requestMib, maxUsefulBytes: 100 * GiB, out bool clamped);
        Assert.Equal(expectMib * MiB, bytes);
        Assert.Equal(expectClamped, clamped);
    }

    [Fact]
    public void ResolveCacheBytes_NeverExceedsWhatCachesEveryExpert()
    {
        Assert.Equal(300 * MiB, MoeOffloadPlanner.ResolveCacheBytes(6 * GiB, null, 300 * MiB, out bool clamped));
        Assert.False(clamped);
        Assert.Equal(0, MoeOffloadPlanner.ResolveCacheBytes(0, 100, 300 * MiB, out clamped));
        Assert.True(clamped);
    }

    [Fact]
    public void SplitSlots_IsProportionalToEachLayoutsTotalBytes()
    {
        // Every layout caches the same fraction of its experts (here 50 %).
        Assert.Equal([5, 5], MoeOffloadPlanner.SplitSlots([100, 200], [10, 10], 1500));
        // Bolt-like: three layouts with 720 / 240 / 304 experts at about 25 % of the total bytes.
        long[] slotBytes = [2_621_440, 1_966_080, 2_949_120];
        int[] experts = [720, 240, 304];
        long total = slotBytes[0] * experts[0] + slotBytes[1] * experts[1] + slotBytes[2] * experts[2];
        int[] s = MoeOffloadPlanner.SplitSlots(slotBytes, experts, total / 4);
        for (int i = 0; i < 3; i++)
            Assert.InRange((double)s[i] / experts[i], 0.24, 0.26);
        Assert.True(s[0] * slotBytes[0] + s[1] * slotBytes[1] + s[2] * slotBytes[2] <= total / 4);
    }

    [Fact]
    public void SplitSlots_GivesRoundingRemainderToTheLowestCachedFraction_WithinBudget()
    {
        // f = 1650 / 3000 = 0.55 -> floor(5.5) = 5 each (1500 bytes); 150 left: one more 100-byte slot fits, a 200-byte one does not.
        Assert.Equal([6, 5], MoeOffloadPlanner.SplitSlots([100, 200], [10, 10], 1650));
        // 250 left after the floor: a 100-byte slot (tie at 0.5, lowest index), then the 200-byte layout no longer fits, so
        // the 100-byte layout takes another.
        Assert.Equal([7, 5], MoeOffloadPlanner.SplitSlots([100, 200], [10, 10], 1750));
        Assert.Equal([6, 6], MoeOffloadPlanner.SplitSlots([100, 200], [10, 10], 1800));
    }

    [Fact]
    public void SplitSlots_CapsAtTheExpertCount_AndHandlesEmptyInputs()
    {
        Assert.Equal([10, 10], MoeOffloadPlanner.SplitSlots([100, 200], [10, 10], long.MaxValue / 4));
        Assert.Equal([0, 0], MoeOffloadPlanner.SplitSlots([100, 200], [10, 10], 0));
        Assert.Equal([0, 2], MoeOffloadPlanner.SplitSlots([100, 200], [0, 10], 450));
        Assert.Throws<ArgumentException>(() => MoeOffloadPlanner.SplitSlots([100], [1, 2], 10));
        Assert.Throws<ArgumentException>(() => MoeOffloadPlanner.SplitSlots([0], [1], 10));
    }

    [Fact]
    public void ExpertLayout_AlignsTheDownMatrixAndTheSlot()
    {
        var l = new ExpertLayout(UpBytes: 1000, DownBytes: 1000);
        Assert.Equal(1024, l.DownOffset);
        Assert.Equal(2048, l.SlotBytes);
        Assert.Equal(new ExpertLayout(1000, 1000), l);   // equal byte sizes share a pool
    }

    [Theory]
    [InlineData(50.0, 300.0, true)]    // 6x slower: spilled
    [InlineData(70.0, 300.0, false)]   // 4.3x: within the 5x tolerance
    [InlineData(300.0, 0.0, false)]    // no reference measurement: never flags
    public void LooksSpilled_FlagsAFiveTimesBandwidthDrop(double pool, double reference, bool expect)
        => Assert.Equal(expect, MoeOffloadPlanner.LooksSpilled(pool, reference));

    [Fact]
    public void FitsPinBudget_UsesTheFractionOfPhysicalRam()
    {
        Assert.True(MoeOffloadPlanner.FitsPinBudget(16 * GiB, 36 * GiB, 0.45));    // NVIDIA Bolt file on T5500
        Assert.False(MoeOffloadPlanner.FitsPinBudget(40 * GiB, 32 * GiB, 0.45));   // Q4_K_M on the dev box
        Assert.False(MoeOffloadPlanner.FitsPinBudget(1, 0, 0.45));
    }

    [Fact]
    public void FitsPinCommit_LeavesTheReserveFree_AndAllowsAnUnknownCommit()
    {
        // #1083: AD-Q4_K_M's 40.17 GiB store passed the RAM fraction on the 96 GB dev box (43 GiB) but other processes had ~57 GB
        // committed of a 111.5 GB limit; pinning it exhausted the commit mid-run.
        long store = (long)(40.17 * GiB);
        Assert.False(MoeOffloadPlanner.FitsPinCommit(store, (long)(42.5 * GiB)));   // the dev box at load: fits RAM, not commit
        Assert.True(MoeOffloadPlanner.FitsPinCommit(16 * GiB, (long)(42.5 * GiB)));  // NVIDIA Bolt's 16 GiB store on the same box
        Assert.True(MoeOffloadPlanner.FitsPinCommit(store, store + MoeOffloadPlanner.PinCommitReserveBytes));   // exactly the reserve left
        Assert.False(MoeOffloadPlanner.FitsPinCommit(store, store + MoeOffloadPlanner.PinCommitReserveBytes - 1));
        Assert.True(MoeOffloadPlanner.FitsPinCommit(store, 0));   // unknown commit (non-Windows): the RAM fraction decides alone
    }

    [Fact]
    public void Options_FromCli_ExplicitValuesWin_UnsetValuesFallBackToEnvironment()
    {
        var env = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [MoeOffloadOptions.ModeEnvVar] = "on",
            [MoeOffloadOptions.CacheMibEnvVar] = "4096",
            [MoeOffloadOptions.PolicyEnvVar] = "lfu",
        };
        string? Get(string k) => env.TryGetValue(k, out var v) ? v : null;

        var fromEnv = MoeOffloadOptions.FromCli(null, null, null, getEnv: Get);
        Assert.Equal(MoeOffloadMode.On, fromEnv.Mode);
        Assert.Equal(4096, fromEnv.CacheMib);
        Assert.Equal(ExpertCachePolicy.Lfu, fromEnv.Policy);

        var explicitDefaults = MoeOffloadOptions.FromCli("auto", "auto", "slru", getEnv: Get);
        Assert.Equal(MoeOffloadMode.Auto, explicitDefaults.Mode);
        Assert.Null(explicitDefaults.CacheMib);
        Assert.Equal(ExpertCachePolicy.Slru, explicitDefaults.Policy);

        var lib = MoeOffloadOptions.WithEnvironmentFallback(null, Get);
        Assert.Equal(MoeOffloadMode.On, lib.Mode);
        Assert.Equal(ExpertCachePolicy.Lfu, lib.Policy);
        Assert.Equal(MoeOffloadMode.Off, MoeOffloadOptions.WithEnvironmentFallback(new() { Mode = MoeOffloadMode.Off }, Get).Mode);
    }

    [Theory]
    [InlineData("sometimes", null, null)]
    [InlineData(null, "0", null)]
    [InlineData(null, "-5", null)]
    [InlineData(null, null, "arc")]
    public void Options_Validate_RejectsUnknownValues(string? mode, string? mib, string? policy)
        => Assert.NotNull(MoeOffloadOptions.Validate(mode, mib, policy));

    [Fact]
    public void Options_Validate_AcceptsEveryDocumentedValue()
    {
        foreach (var m in new[] { "auto", "OFF", "on" })
            foreach (var p in new[] { "slru", "LRU", "lfu" })
                Assert.Null(MoeOffloadOptions.Validate(m, "512", p));
        Assert.Null(MoeOffloadOptions.Validate(null, "auto", null));
    }

    [Fact]
    public void TelemetrySummary_ReportsPerTokenRates()
    {
        var s = new MoeTelemetrySnapshot(Hits: 812, Misses: 188, Evictions: 150, BytesH2D: 713_000_000, CopyWaitMicros: 98_000,
            Tokens: 10, CacheSlots: 812, CacheBytes: (long)(6.1 * GiB), Policy: ExpertCachePolicy.Slru);
        Assert.Equal("MoE offload: cache 6.1 GiB (812 slots, SLRU), hit 81.2%, H2D 71.3 MB/token, copy wait 9.8 ms/token, evictions 150",
            s.FormatSummary());
        Assert.Null((s with { Policy = null }).FormatSummary());
        var d = s.Since(s with { Hits = 800, Tokens = 4 });
        Assert.Equal(12, d.Hits);
        Assert.Equal(6, d.Tokens);
        Assert.Equal(812, d.CacheSlots);
    }
}
