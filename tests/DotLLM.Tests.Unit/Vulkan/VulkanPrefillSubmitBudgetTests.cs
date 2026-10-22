using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1100: the prefill submission policy bounds estimated GPU time per submission instead of using a fixed "layers by row
/// count" table. CPU-only: the policy is pure arithmetic plus a least-squares calibration.
/// </summary>
/// <remarks>
/// The Bonsai 2 27B case is the regression: 64 layers (every 4th full attention), hidden 5120, FFN 17408, 24 query / 4 KV heads of
/// 256. On Strix (~8.6 TFLOP/s dense, ~1 TFLOP/s hd256 attention) the old table put 8 layers (two attention) of a 1024-row tile in
/// one submission, which at ~85k positions is over 4 s of GPU work, and the device was lost.
/// </remarks>
public sealed class VulkanPrefillSubmitBudgetTests
{
    private const int BonsaiLayers = 64;
    private const int Hidden = 5120;
    private const int Ffn = 17408;
    private const int Heads = 24;
    private const int KvHeads = 4;
    private const int HeadDim = 256;

    // The model's ParamsPerLayer formula for the Bonsai shape.
    private const double BonsaiParamsPerLayer = (3.0 * Hidden * Ffn) + ((double)Hidden * ((3.0 * Heads) + (2.0 * KvHeads)) * HeadDim);

    private const double StrixDense = 1 / 8600.0;   // seconds per GFLOP
    private const double StrixAttention = 1 / 1000.0;

    [Fact]
    public void LayersInSubmission_PacksUpToTheBudget_AndAlwaysTakesOne()
    {
        var budget = new VulkanPrefillSubmitBudget(1.0);
        bool[] kinds = Pattern(16);

        // Prior rates: a dense layer of 90 GFLOP is 0.09 s; an attention layer adds 100 GFLOP x 4e-3 = 0.4 s.
        // Cumulative 0.09, 0.18, 0.27, 0.76, 0.85, 0.94, then 1.03 > 1.0: six layers.
        Assert.Equal(6, budget.LayersInSubmission(kinds, 0, 90, 100));

        // Without attention cost, eleven 0.09 s layers fit (0.99 s); a twelfth would not.
        Assert.Equal(11, budget.LayersInSubmission(kinds, 0, 90, 0));

        // The tail never runs past the last layer.
        Assert.Equal(2, budget.LayersInSubmission(kinds, 14, 1, 0));

        // A single layer over budget still goes alone, never zero.
        Assert.Equal(1, budget.LayersInSubmission(kinds, 3, 10_000, 10_000));
    }

    [Fact]
    public void DeeperKv_GivesFewerLayersPerSubmission()
    {
        var budget = Calibrated(StrixDense, StrixAttention);
        bool[] kinds = Pattern(BonsaiLayers);
        double dense = VulkanPrefillSubmitBudget.DenseGflops(BonsaiParamsPerLayer, 1024);
        int previous = int.MaxValue;
        foreach (int depth in new[] { 1024, 8192, 32768, 65536 })
        {
            double attention = VulkanPrefillSubmitBudget.AttentionGflops(1024, depth, Heads, HeadDim);
            int layers = budget.LayersInSubmission(kinds, 0, dense, attention);
            Assert.True(layers <= previous, $"depth {depth}: {layers} layers after {previous}");
            Assert.True(budget.EstimateSeconds(dense * layers, attention * CountAttention(kinds, 0, layers)) <= budget.BudgetSeconds
                        || layers == 1);
            previous = layers;
        }

        Assert.True(previous < 8, "at 64k the old fixed 8 layers per submission must no longer fit");
    }

    [Fact]
    public void AttentionGflops_IsCausal()
    {
        // One row at KV length 1000 attends to 1000 positions; 1000 rows ending at 1000 attend to 500.5 on average.
        Assert.Equal(4.0 * 1000 * 8 * 64 / 1e9, VulkanPrefillSubmitBudget.AttentionGflops(1, 1000, 8, 64), 12);
        Assert.Equal(4.0 * 1000 * 500.5 * 8 * 64 / 1e9, VulkanPrefillSubmitBudget.AttentionGflops(1000, 1000, 8, 64), 9);
    }

    [Fact]
    public void RowsPerForward_SplitsOnlyWhenOneAttentionLayerExceedsTheBudget()
    {
        var budget = Calibrated(StrixDense, StrixAttention);

        // Shallow: a whole 1024-row tile's layer is ~0.1 s.
        Assert.Equal(1024, budget.RowsPerForward(1024, 4096, BonsaiParamsPerLayer, Heads, HeadDim));

        // Where Strix lost the device (~85k): one attention layer of 1024 rows is ~2.2 s, so the tile is split.
        int rows = budget.RowsPerForward(1024, 86_000, BonsaiParamsPerLayer, Heads, HeadDim);
        Assert.InRange(rows, VulkanPrefillSubmitBudget.MinRowsPerForward, 1023);
        Assert.Equal(0, rows % VulkanPrefillSubmitBudget.MinRowsPerForward);
        Assert.True(LayerSeconds(budget, rows, 86_000) <= budget.BudgetSeconds);
        Assert.True(LayerSeconds(budget, rows + VulkanPrefillSubmitBudget.MinRowsPerForward, 86_000) > budget.BudgetSeconds,
            "the split must be the largest slice that fits, not an arbitrary small one");

        // Deeper still: smaller slices, never below the minimum.
        int deeper = budget.RowsPerForward(1024, 131_072, BonsaiParamsPerLayer, Heads, HeadDim);
        Assert.True(deeper <= rows);
        Assert.Equal(VulkanPrefillSubmitBudget.MinRowsPerForward,
            budget.RowsPerForward(1024, 10_000_000, BonsaiParamsPerLayer, Heads, HeadDim));
    }

    [Fact]
    public void BonsaiAt128k_EverySubmissionFitsTheBudget()
    {
        var budget = Calibrated(StrixDense, StrixAttention);
        bool[] kinds = Pattern(BonsaiLayers);
        int rows = budget.RowsPerForward(1024, 131_072, BonsaiParamsPerLayer, Heads, HeadDim);
        double dense = VulkanPrefillSubmitBudget.DenseGflops(BonsaiParamsPerLayer, rows);
        double attention = VulkanPrefillSubmitBudget.AttentionGflops(rows, 131_072, Heads, HeadDim);
        for (int first = 0; first < BonsaiLayers;)
        {
            int n = budget.LayersInSubmission(kinds, first, dense, attention);
            double seconds = (dense * n * StrixDense) + (attention * CountAttention(kinds, first, n) * StrixAttention);
            Assert.True(seconds <= budget.BudgetSeconds * 1.0001, $"layers {first}..{first + n - 1}: {seconds:F3} s");
            first += n;
        }
    }

    /// <summary>
    /// The Strix outlier: a 58-row tail at 32,826 positions ran as one 1.65 s submission because its FLOPs alone (~0.95 s at the
    /// measured rates) fit the budget. Costed at <see cref="VulkanPrefillSubmitBudget.MinEffectiveRows"/>, it is split.
    /// </summary>
    [Fact]
    public void ShortTail_IsCostedAtTheEffectiveRows_AndSplit()
    {
        var budget = Calibrated(StrixDense, StrixAttention);
        bool[] kinds = Pattern(BonsaiLayers);
        double flopDense = VulkanPrefillSubmitBudget.DenseGflops(BonsaiParamsPerLayer, 58);
        double flopAttention = VulkanPrefillSubmitBudget.AttentionGflops(58, 32_826, Heads, HeadDim);
        // By FLOPs alone the whole 64-layer stack is about one budget, so it went out in one or two submissions.
        Assert.InRange(budget.EstimateSeconds(flopDense * BonsaiLayers, flopAttention * (BonsaiLayers / 4)), 0.8, 1.25);

        int rows = VulkanPrefillSubmitBudget.EffectiveRows(58);
        Assert.Equal(VulkanPrefillSubmitBudget.MinEffectiveRows, rows);
        Assert.Equal(1024, VulkanPrefillSubmitBudget.EffectiveRows(1024));
        int layers = budget.LayersInSubmission(kinds, 0,
            VulkanPrefillSubmitBudget.DenseGflops(BonsaiParamsPerLayer, rows),
            VulkanPrefillSubmitBudget.AttentionGflops(rows, 32_826, Heads, HeadDim));
        Assert.True(layers < BonsaiLayers / 2, $"{layers} layers in the first submission");
    }

    [Fact]
    public void Observe_RecoversBothRates()
    {
        var budget = new VulkanPrefillSubmitBudget();
        double trueDense = 2e-4, trueAttention = 1.5e-3;
        var rng = new Random(1100);
        for (int i = 0; i < 40; i++)
        {
            double d = 500 + (rng.NextDouble() * 5000);
            double a = rng.NextDouble() * 3000;
            budget.Observe(d, a, (d * trueDense) + (a * trueAttention));
        }

        // The prior pseudo-observations bias the fit slightly; within 5% once the data dominates.
        Assert.Equal(trueDense, budget.DenseSecondsPerGflop, trueDense * 0.05);
        Assert.Equal(trueAttention, budget.AttentionSecondsPerGflop, trueAttention * 0.05);
        Assert.Equal(40, budget.Observations);
    }

    [Fact]
    public void Observe_WithoutAttentionWork_KeepsTheConservativeAttentionPrior()
    {
        var budget = new VulkanPrefillSubmitBudget();
        for (int i = 0; i < 20; i++)
            budget.Observe(5000 + (i * 100), 5, (5000 + (i * 100)) * 1e-4);

        // Dense is learnt; attention stays near the slow prior instead of collapsing to "free".
        Assert.Equal(1e-4, budget.DenseSecondsPerGflop, 1e-5);
        Assert.True(budget.AttentionSecondsPerGflop > 0.5 * VulkanPrefillSubmitBudget.PriorAttentionSecondsPerGflop,
            $"attention rate {budget.AttentionSecondsPerGflop} drifted from the prior without evidence");
    }

    [Fact]
    public void Observe_IgnoresGarbage()
    {
        var budget = new VulkanPrefillSubmitBudget();
        budget.Observe(100, 100, 0);
        budget.Observe(100, 100, double.NaN);
        budget.Observe(0, 0, 1);
        budget.Observe(-1, 10, 1);
        Assert.Equal(0, budget.Observations);
        Assert.Equal(VulkanPrefillSubmitBudget.PriorDenseSecondsPerGflop, budget.DenseSecondsPerGflop);
        Assert.Equal(VulkanPrefillSubmitBudget.PriorAttentionSecondsPerGflop, budget.AttentionSecondsPerGflop);
    }

    [Fact]
    public void Constructor_RejectsANonPositiveBudget()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new VulkanPrefillSubmitBudget(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new VulkanPrefillSubmitBudget(double.NaN));
    }

    /// <summary>A budget fitted to exact rates from many varied submissions.</summary>
    private static VulkanPrefillSubmitBudget Calibrated(double dense, double attention)
    {
        // Work drawn afresh for every observation: with forgetting, only the last ~10 submissions count, and each kind of work
        // must vary within them for its rate to be determined.
        var budget = new VulkanPrefillSubmitBudget();
        var rng = new Random(42);
        for (int i = 0; i < 60; i++)
        {
            double d = 1000 + (rng.NextDouble() * 4000);
            double a = 500 + (rng.NextDouble() * 2500);
            budget.Observe(d, a, (d * dense) + (a * attention));
        }

        Assert.Equal(dense, budget.DenseSecondsPerGflop, dense * 0.05);
        Assert.Equal(attention, budget.AttentionSecondsPerGflop, attention * 0.05);
        return budget;
    }

    private static double LayerSeconds(VulkanPrefillSubmitBudget budget, int rows, int kvEnd)
        => budget.EstimateSeconds(VulkanPrefillSubmitBudget.DenseGflops(BonsaiParamsPerLayer, rows),
                                  VulkanPrefillSubmitBudget.AttentionGflops(rows, kvEnd, Heads, HeadDim));

    /// <summary>Qwen3.5-style hybrid: every 4th layer is full attention.</summary>
    private static bool[] Pattern(int layers)
    {
        var kinds = new bool[layers];
        for (int l = 0; l < layers; l++) kinds[l] = l % 4 == 3;
        return kinds;
    }

    private static int CountAttention(bool[] kinds, int first, int n)
    {
        int c = 0;
        for (int l = first; l < first + n; l++) if (kinds[l]) c++;
        return c;
    }
}
