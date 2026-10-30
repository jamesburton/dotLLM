using DotLLM.Core.Moe;
using Xunit;

namespace DotLLM.Tests.Unit.Moe;

/// <summary>
/// Issue #1071: the cost model and the per-layer CPU / device assignment of uncached experts (<see cref="MoeCostModel"/>,
/// <see cref="MoeColdAssignment"/>). Pure functions, CPU only.
/// </summary>
public sealed class MoeColdAssignmentTests
{
    private static readonly ExpertLayout Bolt2048 = new(UpBytes: 1_310_720, DownBytes: 1_310_720);   // 2.62 MB per expert

    [Fact]
    public void LinearCost_FitsTheTwoMeasurements_AndClampsNoise()
    {
        var c = LinearCost.FitFrom1And8(us1: 800, us8: 2200);
        Assert.Equal(200, c.B, 9);
        Assert.Equal(600, c.A, 9);
        Assert.Equal(800, c.Us(1), 9);
        Assert.Equal(2200, c.Us(8), 9);

        // n = 8 measured faster than n = 1 (timer noise): no negative slope.
        var noisy = LinearCost.FitFrom1And8(us1: 900, us8: 850);
        Assert.Equal(0, noisy.B);
        Assert.Equal(900, noisy.Us(100), 9);
    }

    [Fact]
    public void CostModel_CopyIsBytesOverTheLink_UncalibratedLayoutNeverGoesToTheCpu()
    {
        var model = new MoeCostModel(H2DBytesPerSecond: 3.2e9, SyncRoundTripUs: 130,
            [new MoeLayoutCost(Bolt2048, new LinearCost(500, 300), MoeCostModel.DefaultGpuCost(2_621_440))]);
        Assert.Equal(2_621_440 / 3.2e9 * 1e6, model.CopyUs(2_621_440), 6);

        var cost = model.ColdCost(Bolt2048, tokens: 1);
        Assert.Equal(800, cost.CpuUs, 9);
        Assert.Equal(model.CopyUs(2_621_440), cost.CopyUs, 9);
        Assert.True(cost.GpuUs is > 0 and < 20, $"gpu {cost.GpuUs}");   // 2.6 MB at 300 GB/s ≈ 8.7 us

        var unknown = model.ColdCost(new ExpertLayout(1, 1), tokens: 1);
        Assert.True(double.IsPositiveInfinity(unknown.CpuUs));
        Assert.True(double.IsPositiveInfinity(new MoeCostModel(0, 0, []).CopyUs(1)));
    }

    [Fact]
    public void Fiddler_CheaperSideWins_WhenBalanceDoesNotMatter()
    {
        // One expert each: CPU 300 vs device 900 -> CPU; CPU 2000 vs device 900 -> device.
        bool[] onCpu = new bool[2];
        int n = MoeColdAssignment.Assign([new(300, 800, 100), new(2000, 800, 100)], residentGpuUs: 0, cpuWorkers: 4,
            cpuLegOverheadUs: 0, onCpu);
        Assert.Equal(1, n);
        Assert.True(onCpu[0]);
        Assert.False(onCpu[1]);
    }

    [Fact]
    public void FiddlerRule_CpuWhenCheaperThanCopyPlusGpu()
    {
        bool[] onCpu = new bool[4];
        // CPU 300 < 900; CPU 2000 > 900; equal costs stay on the device; an infinite device cost goes to the CPU.
        int n = MoeColdAssignment.Fiddler([new(300, 800, 100), new(2000, 800, 100), new(900, 800, 100),
            new(5000, double.PositiveInfinity, 100)], onCpu);
        Assert.Equal(2, n);
        Assert.Equal([true, false, false, true], onCpu);
    }

    [Fact]
    public void GreedyBalance_MovesExpertsOffAnOverloadedCpu()
    {
        // Six experts that Fiddler sends to the CPU (500 < 600 each). One CPU worker: CPU span 3000 vs device 0. The greedy pass
        // moves experts to the device until max(CPU, device) stops falling: 3 / 3 gives max(1500, 1800) = 1800, 4 / 2 gives
        // max(2000, 1200) = 2000, so it stops at 3 on the CPU.
        var costs = Enumerable.Repeat(new ColdExpertCost(500, 550, 50), 6).ToArray();
        bool[] onCpu = new bool[6];
        int n = MoeColdAssignment.Assign(costs, residentGpuUs: 0, cpuWorkers: 1, cpuLegOverheadUs: 0, onCpu);
        Assert.Equal(3, n);
        Assert.Equal(1800, MoeColdAssignment.Makespan(costs, onCpu, 0, 1, 0), 9);

        // With six workers the CPU runs them all in parallel (span 500) and nothing moves.
        n = MoeColdAssignment.Assign(costs, residentGpuUs: 0, cpuWorkers: 6, cpuLegOverheadUs: 0, onCpu);
        Assert.Equal(6, n);
    }

    [Fact]
    public void GreedyBalance_MovesExpertsOntoAnIdleCpu_WhenTheDeviceIsTheBottleneck()
    {
        // Fiddler keeps all four on the device (CPU 1000 > device 900), but the device also has 3000 us of cached experts:
        // moving work to the idle CPU workers shortens the layer.
        var costs = Enumerable.Repeat(new ColdExpertCost(1000, 800, 100), 4).ToArray();
        bool[] onCpu = new bool[4];
        int n = MoeColdAssignment.Assign(costs, residentGpuUs: 3000, cpuWorkers: 4, cpuLegOverheadUs: 100, onCpu);
        Assert.Equal(4, n);
        Assert.Equal(3000, MoeColdAssignment.Makespan(costs, onCpu, 3000, 4, 100), 9);
    }

    [Fact]
    public void CpuLegOverhead_KeepsASingleMarginalExpertOnTheDevice()
    {
        // CPU 850 beats device 900 by 50 us, but the CPU leg's extra round trip (130 us) makes it lose: device span 900 vs
        // CPU span 980.
        bool[] onCpu = new bool[1];
        Assert.Equal(0, MoeColdAssignment.Assign([new(850, 800, 100)], residentGpuUs: 0, cpuWorkers: 4, cpuLegOverheadUs: 130, onCpu));
        Assert.False(onCpu[0]);
    }

    [Fact]
    public void Assign_IsDeterministic_EmptyIsEmpty_InfiniteCostsNeverMove()
    {
        Assert.Equal(0, MoeColdAssignment.Assign([], 0, 4, 0, []));

        // An uncalibrated layout (CPU cost infinite) stays on the device whatever the balance says.
        bool[] onCpu = new bool[2];
        int n = MoeColdAssignment.Assign([new(double.PositiveInfinity, 800, 100), new(10, 800, 100)], residentGpuUs: 50_000,
            cpuWorkers: 1, cpuLegOverheadUs: 0, onCpu);
        Assert.Equal(1, n);
        Assert.False(onCpu[0]);
        Assert.True(onCpu[1]);

        var costs = new ColdExpertCost[] { new(400, 300, 50), new(400, 300, 50), new(400, 300, 50), new(700, 100, 50) };
        bool[] a = new bool[4], b = new bool[4];
        MoeColdAssignment.Assign(costs, 200, 2, 50, a);
        MoeColdAssignment.Assign(costs, 200, 2, 50, b);
        Assert.Equal(a, b);
    }

    [Fact]
    public void Assign_NeverWorseThanAllOnOneSide()
    {
        var rng = new Random(1071);
        for (int trial = 0; trial < 200; trial++)
        {
            int n = rng.Next(1, 24);
            var costs = new ColdExpertCost[n];
            for (int i = 0; i < n; i++)
                costs[i] = new ColdExpertCost(rng.Next(100, 3000), rng.Next(100, 2000), rng.Next(5, 200));
            double resident = rng.Next(0, 4000);
            int workers = rng.Next(1, 16);
            double overhead = rng.Next(0, 300);
            bool[] onCpu = new bool[n];
            MoeColdAssignment.Assign(costs, resident, workers, overhead, onCpu);
            double got = MoeColdAssignment.Makespan(costs, onCpu, resident, workers, overhead);
            double allDevice = MoeColdAssignment.Makespan(costs, new bool[n], resident, workers, overhead);
            double allCpu = MoeColdAssignment.Makespan(costs, Enumerable.Repeat(true, n).ToArray(), resident, workers, overhead);
            Assert.True(got <= allDevice + 1e-6 && got <= allCpu + 1e-6, $"trial {trial}: {got} vs device {allDevice} / cpu {allCpu}");
        }
    }

    [Fact]
    public void PromotionBudget_IsAQuarterOfTheLastTokensLinkTime_WithAFloor()
    {
        var model = new MoeCostModel(H2DBytesPerSecond: 4e9, SyncRoundTripUs: 100, []);
        // 50 ms per token at 4 GB/s, a quarter of it: 50 MB per token.
        Assert.Equal(50_000_000, model.PromotionBudgetBytes(tokens: 1, lastTokenUs: 50_000));
        Assert.Equal(100_000_000, model.PromotionBudgetBytes(tokens: 2, lastTokenUs: 50_000));
        Assert.Equal(3_000_000, model.PromotionBudgetBytes(tokens: 1, lastTokenUs: 0, minimumBytes: 3_000_000));
        Assert.Equal(7, new MoeCostModel(0, 0, []).PromotionBudgetBytes(1, 50_000, minimumBytes: 7));
    }
}
