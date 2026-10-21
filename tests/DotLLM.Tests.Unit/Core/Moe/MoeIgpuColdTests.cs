using DotLLM.Core.Moe;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Moe;

/// <summary>
/// Issue #1083: the iGPU side of the MoE cold-expert decision, CPU only (no GPU is touched).
/// <list type="bullet">
///   <item>The three-way assignment (<see cref="MoeColdAssignment"/>): device + copy, CPU, iGPU; the iGPU's fixed submit cost; the
///   two-way #1071 behaviour exactly when there is no iGPU.</item>
///   <item>Calibration: the key, the record's iGPU timings (JSON round trip, fit, coverage).</item>
///   <item>The <c>igpu</c> / <c>cpu+igpu</c> cold modes, <c>--moe-igpu-device</c>, and the iGPU telemetry summary.</item>
///   <item>The integrated-GPU device choice (<see cref="VulkanIgpuDeviceSelector"/>, pure).</item>
/// </list>
/// </summary>
public sealed class MoeIgpuColdTests : IDisposable
{
    private static readonly ExpertLayout Bolt2048 = new(UpBytes: 1_310_720, DownBytes: 1_310_720);
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"dotllm-moe-igpu-{Guid.NewGuid():N}");

    public MoeIgpuColdTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
    }

    // ───────────────────────── assignment ─────────────────────────

    [Fact]
    public void Assign_WithoutAnIgpu_IsExactlyTheTwoWayAssignment_EvenWithFiniteIgpuCosts()
    {
        var rng = new Random(1083);
        for (int trial = 0; trial < 300; trial++)
        {
            int n = rng.Next(1, 24);
            var costs = new ColdExpertCost[n];
            for (int i = 0; i < n; i++)
                costs[i] = new ColdExpertCost(rng.Next(100, 3000), rng.Next(100, 2000), rng.Next(5, 200), rng.Next(50, 3000));
            double resident = rng.Next(0, 4000), overhead = rng.Next(0, 300);
            int workers = rng.Next(1, 16);

            bool[] onCpu = new bool[n];
            int twoWay = MoeColdAssignment.Assign(costs, resident, workers, overhead, onCpu);
            var place = new ColdPlace[n];
            var (cpu, igpu) = MoeColdAssignment.Assign(costs, resident, workers, overhead, double.PositiveInfinity, place);

            Assert.Equal(0, igpu);
            Assert.Equal(twoWay, cpu);
            for (int i = 0; i < n; i++) Assert.Equal(onCpu[i], place[i] == ColdPlace.Cpu);
            Assert.Equal(MoeColdAssignment.Makespan(costs, onCpu, resident, workers, overhead),
                MoeColdAssignment.Makespan(costs, place, resident, workers, overhead, double.PositiveInfinity), 9);
        }
    }

    [Fact]
    public void Fiddler_ThreeSides_CheapestWins_TiesToTheDeviceThenTheCpu_IgnoresADisabledIgpu()
    {
        ColdExpertCost[] costs =
        [
            new(CpuUs: 300, CopyUs: 800, GpuUs: 100, IgpuUs: 200),   // iGPU cheapest
            new(CpuUs: 300, CopyUs: 800, GpuUs: 100, IgpuUs: 400),   // CPU cheapest
            new(CpuUs: 900, CopyUs: 800, GpuUs: 100, IgpuUs: 900),   // three-way tie at 900 -> device
            new(CpuUs: 500, CopyUs: 800, GpuUs: 100, IgpuUs: 500),   // CPU / iGPU tie -> CPU
            new(CpuUs: 5000, CopyUs: double.PositiveInfinity, GpuUs: 100, IgpuUs: double.PositiveInfinity),   // only the CPU is finite
        ];
        var place = new ColdPlace[costs.Length];
        Assert.Equal(4, MoeColdAssignment.Fiddler(costs, igpuSubmitUs: 100, place));
        Assert.Equal([ColdPlace.Igpu, ColdPlace.Cpu, ColdPlace.Device, ColdPlace.Cpu, ColdPlace.Cpu], place);

        MoeColdAssignment.Fiddler(costs, igpuSubmitUs: double.PositiveInfinity, place);
        Assert.DoesNotContain(ColdPlace.Igpu, place);
    }

    [Fact]
    public void Assign_SplitsTheColdSetAcrossCpuAndIgpu_WhenCopyingIsExpensive()
    {
        // Four experts: copy + device 5000 each, CPU 1000 (one worker), iGPU 1000, submit 100, cold-leg round trip 50.
        // All on the CPU: 50 + 4000. Two and two: 50 + max(2000, 100 + 2000) = 2150, the optimum.
        var costs = Enumerable.Repeat(new ColdExpertCost(CpuUs: 1000, CopyUs: 4900, GpuUs: 100, IgpuUs: 1000), 4).ToArray();
        var place = new ColdPlace[4];
        var (cpu, igpu) = MoeColdAssignment.Assign(costs, residentGpuUs: 0, cpuWorkers: 1, coldLegOverheadUs: 50, igpuSubmitUs: 100, place);
        Assert.Equal((2, 2), (cpu, igpu));
        Assert.Equal(2150, MoeColdAssignment.Makespan(costs, place, 0, 1, 50, 100), 9);
    }

    [Fact]
    public void IgpuSubmitCost_KeepsAMarginalExpertOffTheIgpu_AndAnEmptyIgpuSetPaysNothing()
    {
        // The iGPU computes it 50 us faster than the CPU, but its submission costs 200 us: CPU.
        var one = new[] { new ColdExpertCost(CpuUs: 1000, CopyUs: 4900, GpuUs: 100, IgpuUs: 950) };
        var place = new ColdPlace[1];
        Assert.Equal((1, 0), MoeColdAssignment.Assign(one, 0, 4, 50, igpuSubmitUs: 200, place));
        Assert.Equal(ColdPlace.Cpu, place[0]);
        // Free submission: the iGPU.
        Assert.Equal((0, 1), MoeColdAssignment.Assign(one, 0, 4, 50, igpuSubmitUs: 0, place));

        // No expert on the iGPU: a huge submit cost changes nothing.
        ColdPlace[] cpuOnly = [ColdPlace.Cpu];
        Assert.Equal(1050, MoeColdAssignment.Makespan(one, cpuOnly, 0, 4, 50, igpuSubmitUs: 1e9), 9);
        Assert.Equal(5000, MoeColdAssignment.Makespan(one, [ColdPlace.Device], 0, 4, 50, igpuSubmitUs: 1e9), 9);
    }

    [Fact]
    public void Assign_CpuAndIgpuRunConcurrently_TheColdSpanIsTheSlowerLegPlusOneRoundTrip()
    {
        ColdExpertCost[] costs = [new(400, 1e6, 0, 900), new(400, 1e6, 0, 900), new(700, 1e6, 0, 300)];
        ColdPlace[] place = [ColdPlace.Cpu, ColdPlace.Cpu, ColdPlace.Igpu];
        // CPU: max(800 / 2, 400) = 400; iGPU: 100 + 300 = 400; one round trip of 30 on top.
        Assert.Equal(430, MoeColdAssignment.Makespan(costs, place, residentGpuUs: 10, cpuWorkers: 2, coldLegOverheadUs: 30, igpuSubmitUs: 100), 9);
    }

    [Fact]
    public void Assign_NeverWorseThanAnyOneSidedOrFiddlerAssignment_AndNeverOnAnInfiniteSide()
    {
        var rng = new Random(10831);
        for (int trial = 0; trial < 300; trial++)
        {
            int n = rng.Next(1, 24);
            var costs = new ColdExpertCost[n];
            for (int i = 0; i < n; i++)
            {
                double igpu = rng.Next(6) == 0 ? double.PositiveInfinity : rng.Next(50, 3000);   // a layer the iGPU refused
                double copy = rng.Next(8) == 0 ? double.PositiveInfinity : rng.Next(100, 2000);   // cpu+igpu: copying ruled out
                costs[i] = new ColdExpertCost(rng.Next(100, 3000), copy, rng.Next(5, 200), igpu);
            }
            double resident = rng.Next(0, 4000), overhead = rng.Next(0, 300), submit = rng.Next(0, 500);
            int workers = rng.Next(1, 16);
            var place = new ColdPlace[n];
            MoeColdAssignment.Assign(costs, resident, workers, overhead, submit, place);
            double got = MoeColdAssignment.Makespan(costs, place, resident, workers, overhead, submit);
            for (int i = 0; i < n; i++)
                Assert.False(double.IsPositiveInfinity(costs[i].Us(place[i])) && !AllInfinite(costs[i]), $"trial {trial}: expert {i} on an infinite side");

            var fiddler = new ColdPlace[n];
            MoeColdAssignment.Fiddler(costs, submit, fiddler);
            Assert.True(got <= MoeColdAssignment.Makespan(costs, fiddler, resident, workers, overhead, submit) + 1e-6, $"trial {trial}: Fiddler");
            foreach (var side in new[] { ColdPlace.Device, ColdPlace.Cpu, ColdPlace.Igpu })
            {
                var one = new ColdPlace[n];
                for (int i = 0; i < n; i++) one[i] = double.IsPositiveInfinity(costs[i].Us(side)) ? fiddler[i] : side;
                double m = MoeColdAssignment.Makespan(costs, one, resident, workers, overhead, submit);
                Assert.True(got <= m + 1e-6, $"trial {trial}: {got} vs all-{side} {m}");
            }
        }

        static bool AllInfinite(ColdExpertCost c)
            => double.IsPositiveInfinity(c.DeviceUs) && double.IsPositiveInfinity(c.CpuUs) && double.IsPositiveInfinity(c.IgpuUs);
    }

    [Fact]
    public void CostModel_IgpuCostOnlyWithASubmitCostAndALayoutTiming()
    {
        var withIgpu = new MoeCostModel(3.2e9, 130,
            [new MoeLayoutCost(Bolt2048, new LinearCost(500, 300), MoeCostModel.DefaultGpuCost(2_621_440), new LinearCost(200, 40))],
            IgpuSubmitUs: 120);
        Assert.True(withIgpu.HasIgpu);
        Assert.Equal(240, withIgpu.ColdCost(Bolt2048, 1).IgpuUs, 9);
        Assert.Equal(520, withIgpu.ColdCost(Bolt2048, 8).IgpuUs, 9);
        Assert.True(double.IsPositiveInfinity(withIgpu.ColdCost(new ExpertLayout(1, 1), 1).IgpuUs));   // not calibrated

        // No submit cost (the default): no iGPU, whatever the layouts say.
        var noIgpu = withIgpu with { IgpuSubmitUs = double.PositiveInfinity };
        Assert.False(noIgpu.HasIgpu);
        Assert.True(double.IsPositiveInfinity(noIgpu.ColdCost(Bolt2048, 1).IgpuUs));
        Assert.False(new MoeCostModel(1, 1, []).HasIgpu);
        Assert.True(double.IsPositiveInfinity(new ColdExpertCost(1, 2, 3).IgpuUs));
    }

    // ───────────────────────── calibration ─────────────────────────

    [Fact]
    public void CalibrationKey_TheIgpuIsPartOfIt_AndNoIgpuKeepsTheCpuOnlyKey()
    {
        string model = Path.Combine(_dir, "m.gguf");
        File.WriteAllBytes(model, new byte[10]);
        var cpuOnly = MoeCalibrationKey.ForFile("RTX 3060", "0000:03:00.0", model, 15);
        var noIgpu = MoeCalibrationKey.ForFile("RTX 3060", "0000:03:00.0", model, 15, igpu: null);
        var arc = MoeCalibrationKey.ForFile("RTX 3060", "0000:03:00.0", model, 15, igpu: "Intel(R) Arc(TM) Graphics (vk type 1)");
        Assert.Equal(cpuOnly.ToString(), noIgpu.ToString());
        Assert.Equal(cpuOnly.ToString(), (cpuOnly with { Igpu = "" }).ToString());
        Assert.NotEqual(cpuOnly.ToString(), arc.ToString());
        Assert.NotEqual(arc.ToString(), (arc with { Igpu = "AMD Radeon 8060S (vk type 1)" }).ToString());
        Assert.EndsWith("|igpu:Intel(R) Arc(TM) Graphics (vk type 1)", arc.ToString(), StringComparison.Ordinal);
    }

    private static MoeCalibrationRecord IgpuRecord() => new(3.1e9, 60,
        [
            new MoeLayoutTiming(1_000_000, 1_200_000, CpuUs1: 800, CpuUs8: 2200, IgpuUs1: 400, IgpuUs8: 750),
            new MoeLayoutTiming(400_000, 410_000, CpuUs1: 300, CpuUs8: 650),   // a layout the iGPU refused
        ],
        new DateTime(2026, 10, 21, 9, 0, 0, DateTimeKind.Utc), IgpuSubmitUs: 150);

    [Fact]
    public void CalibrationRecord_IgpuTimings_RoundTripThroughTheStore_AndAbsentOnesStayNaN()
    {
        string path = Path.Combine(_dir, "cal.json");
        var key = new MoeCalibrationKey("g", "p", "c", "m.gguf", 1, 2, 3, "Arc");
        Assert.True(MoeCalibrationStore.TrySave(path, key, IgpuRecord()));
        var back = MoeCalibrationStore.TryLoad(path, key)!;
        Assert.Equal(150, back.IgpuSubmitUs);
        Assert.Equal(IgpuRecord().Layouts, back.Layouts);
        Assert.True(back.Layouts[0].HasIgpu);
        Assert.False(back.Layouts[1].HasIgpu);
        Assert.True(double.IsNaN(back.Layouts[1].IgpuUs1));
        Assert.Null(MoeCalibrationStore.TryLoad(path, key with { Igpu = "" }));

        // A record without an iGPU stores no iGPU fields at all and reads back NaN.
        var cpuKey = key with { Igpu = "" };
        Assert.True(MoeCalibrationStore.TrySave(path, cpuKey, IgpuRecord() with { IgpuSubmitUs = double.NaN }));
        string json = File.ReadAllText(path);
        Assert.DoesNotContain("NaN", json, StringComparison.Ordinal);
        Assert.True(double.IsNaN(MoeCalibrationStore.TryLoad(path, cpuKey)!.IgpuSubmitUs));
        Assert.NotNull(MoeCalibrationStore.TryLoad(path, key));   // the iGPU entry is kept
    }

    [Fact]
    public void CalibrationRecord_FitsTheIgpuWithoutItsSubmitCost_AndReportsCoverage()
    {
        var r = IgpuRecord();
        Assert.True(r.HasIgpu);
        var m = r.ToCostModel();
        Assert.Equal(150, m.IgpuSubmitUs);
        var big = m.For(new ExpertLayout(1_000_000, 1_200_000))!;
        // One-expert submissions of 400 / 750 us minus the empty submission's 150: 250 at n = 1, 600 at n = 8.
        Assert.Equal(250, big.Igpu!.Value.Us(1), 9);
        Assert.Equal(600, big.Igpu!.Value.Us(8), 9);
        Assert.Null(m.For(new ExpertLayout(400_000, 410_000))!.Igpu);
        Assert.True(double.IsPositiveInfinity(m.ColdCost(new ExpertLayout(400_000, 410_000), 1).IgpuUs));

        var small = new ExpertLayout(400_000, 410_000);
        var large = new ExpertLayout(1_000_000, 1_200_000);
        Assert.True(r.Covers([large, small]));
        Assert.True(r.Covers([large, small], igpuLayouts: [large]));
        Assert.False(r.Covers([large, small], igpuLayouts: [large, small]));   // the small layout has no iGPU timing
        Assert.False((r with { IgpuSubmitUs = double.NaN }).Covers([large], igpuLayouts: [large]));
        Assert.True((r with { IgpuSubmitUs = double.NaN }).Covers([large]));

        var none = r with { IgpuSubmitUs = double.NaN };
        Assert.False(none.HasIgpu);
        Assert.False(none.ToCostModel().HasIgpu);
        Assert.Contains("iGPU submit+fence 150 us", r.Describe(), StringComparison.Ordinal);
        Assert.DoesNotContain("iGPU", none.Describe(), StringComparison.Ordinal);
    }

    // ───────────────────────── options and telemetry ─────────────────────────

    [Fact]
    public void Options_IgpuColdModes_AndTheIgpuDevice_FromCliAndEnvironment()
    {
        var env = new Dictionary<string, string>(StringComparer.Ordinal);
        string? Get(string k) => env.TryGetValue(k, out var v) ? v : null;

        Assert.Equal(MoeColdMode.Igpu, MoeOffloadOptions.ParseCold("igpu"));
        Assert.Equal(MoeColdMode.CpuIgpu, MoeOffloadOptions.ParseCold("CPU+iGPU"));
        Assert.Equal(MoeColdMode.CpuIgpu, MoeOffloadOptions.ParseCold("igpu+cpu"));
        Assert.Equal("cpu+igpu", MoeOffloadOptions.FormatCold(MoeColdMode.CpuIgpu));
        Assert.Equal("igpu", MoeOffloadOptions.FormatCold(MoeColdMode.Igpu));
        foreach (var mode in Enum.GetValues<MoeColdMode>())
            Assert.Equal(mode, MoeOffloadOptions.ParseCold(MoeOffloadOptions.FormatCold(mode)));

        var d = MoeOffloadOptions.FromCli(new MoeCliValues(), getEnv: Get);
        Assert.Null(d.IgpuDevice);
        Assert.True(d.MayUseIgpuLeg);   // auto
        Assert.False(d.RequiresIgpuLeg);
        Assert.False((d with { Cold = MoeColdMode.Cpu }).MayUseIgpuLeg);
        Assert.False((d with { Cold = MoeColdMode.Copy }).MayUseIgpuLeg);
        Assert.True((d with { Cold = MoeColdMode.Igpu }).RequiresIgpuLeg);
        Assert.True((d with { Cold = MoeColdMode.CpuIgpu }).RequiresIgpuLeg);
        Assert.True((d with { Cold = MoeColdMode.Igpu }).MayUseCpuLeg);   // the CPU stays the fallback for refused layers

        var cli = MoeOffloadOptions.FromCli(new MoeCliValues(Cold: "cpu+igpu", IgpuDevice: " Arc "), getEnv: Get);
        Assert.Equal(MoeColdMode.CpuIgpu, cli.Cold);
        Assert.Equal("Arc", cli.IgpuDevice);

        env[MoeOffloadOptions.ColdEnvVar] = "igpu";
        env[MoeOffloadOptions.IgpuDeviceEnvVar] = "1";
        var e = MoeOffloadOptions.FromCli(new MoeCliValues(), getEnv: Get);
        Assert.Equal(MoeColdMode.Igpu, e.Cold);
        Assert.Equal("1", e.IgpuDevice);
        Assert.Equal("Arc", MoeOffloadOptions.FromCli(new MoeCliValues(IgpuDevice: "Arc"), getEnv: Get).IgpuDevice);   // CLI wins
        var lib = MoeOffloadOptions.WithEnvironmentFallback(null, Get);
        Assert.Equal(MoeColdMode.Igpu, lib.Cold);
        Assert.Equal("1", lib.IgpuDevice);

        Assert.NotNull(MoeOffloadOptions.Validate(new MoeCliValues(Cold: "gpu"), _ => null));
        Assert.Null(MoeOffloadOptions.Validate(new MoeCliValues(Cold: "cpu+igpu", IgpuDevice: "0"), _ => null));
    }

    [Fact]
    public void TelemetrySummary_AppendsTheIgpuLeg_WhenItRan()
    {
        var s = new MoeTelemetrySnapshot(Hits: 50, Misses: 50, Evictions: 0, BytesH2D: 0, CopyWaitMicros: 0, Tokens: 10,
            CacheSlots: 10, CacheBytes: 1L << 30, Policy: ExpertCachePolicy.Slru, Routed: 100, ColdExpertsCpu: 20,
            CpuComputeMicros: 50_000, CpuWaitMicros: 32_000, Promotions: 3, ColdExpertsIgpu: 30, IgpuComputeMicros: 41_000,
            IgpuWaitMicros: 6_000);
        Assert.EndsWith("; cold: CPU 20% of routed, CPU compute 5.0 ms/token, CPU wait 3.2 ms/token, promotions 3; " +
            "iGPU 30% of routed, iGPU compute 4.1 ms/token, iGPU wait 0.6 ms/token", s.FormatSummary(), StringComparison.Ordinal);
        Assert.Contains("; cold: CPU 0% of routed", (s with { ColdExpertsCpu = 0, Promotions = 0 }).FormatSummary(), StringComparison.Ordinal);
        Assert.DoesNotContain("iGPU", (s with { ColdExpertsIgpu = 0 }).FormatSummary(), StringComparison.Ordinal);
        var d = s.Since(s with { ColdExpertsIgpu = 10, IgpuComputeMicros = 1_000, IgpuWaitMicros = 500 });
        Assert.Equal((20, 40_000, 5_500), (d.ColdExpertsIgpu, d.IgpuComputeMicros, d.IgpuWaitMicros));
        Assert.Equal(0.3, s.ColdIgpuFraction, 9);
    }

    // ───────────────────────── device choice ─────────────────────────

    private static readonly VulkanPhysicalDeviceInfo Rtx = new(0, "NVIDIA GeForce RTX 3060", DeviceType: 2, VendorId: 0x10DE);
    private static readonly VulkanPhysicalDeviceInfo Arc = new(1, "Intel(R) Arc(TM) Graphics", DeviceType: 1, VendorId: 0x8086);
    private static readonly VulkanPhysicalDeviceInfo Radeon = new(2, "AMD Radeon RX 7900 XT", DeviceType: 2, VendorId: 0x1002);

    [Fact]
    public void DeviceSelector_PicksTheIntegratedGpu_NeverNvidia_AndHonoursAnOverride()
    {
        Assert.Equal(Arc, VulkanIgpuDeviceSelector.Select([Rtx, Arc], null, out string why));
        Assert.Contains("integrated", why, StringComparison.Ordinal);

        // No integrated GPU: no iGPU leg (the discrete cards are not candidates without an override).
        Assert.Null(VulkanIgpuDeviceSelector.Select([Rtx, Radeon], "", out why));
        Assert.Contains("no integrated Vulkan GPU", why, StringComparison.Ordinal);
        Assert.Null(VulkanIgpuDeviceSelector.Select([], null, out why));

        // An integrated NVIDIA device (Tegra-like) is still NVIDIA: never.
        var tegra = new VulkanPhysicalDeviceInfo(0, "NVIDIA Tegra", DeviceType: 1, VendorId: 0x10DE);
        Assert.Null(VulkanIgpuDeviceSelector.Select([tegra], null, out _));

        // Overrides: index, name substring (case-insensitive); never NVIDIA; a discrete pick is allowed with a warning.
        Assert.Equal(Arc, VulkanIgpuDeviceSelector.Select([Rtx, Arc, Radeon], "1", out _));
        Assert.Equal(Arc, VulkanIgpuDeviceSelector.Select([Rtx, Arc, Radeon], "arc", out _));
        Assert.Null(VulkanIgpuDeviceSelector.Select([Rtx, Arc], "0", out why));
        Assert.Contains("NVIDIA", why, StringComparison.Ordinal);
        Assert.Null(VulkanIgpuDeviceSelector.Select([Rtx, Arc], "rtx", out _));
        Assert.Equal(Radeon, VulkanIgpuDeviceSelector.Select([Rtx, Arc, Radeon], "radeon", out why));
        Assert.Contains("NOT an integrated GPU", why, StringComparison.Ordinal);
        Assert.Null(VulkanIgpuDeviceSelector.Select([Rtx, Arc], "7", out why));
        Assert.Contains("matches no Vulkan device", why, StringComparison.Ordinal);
    }
}
