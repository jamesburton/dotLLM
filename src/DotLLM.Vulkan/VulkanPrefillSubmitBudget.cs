using System.Globalization;

namespace DotLLM.Vulkan;

/// <summary>
/// Bounds the GPU time of each prefill submission (issue #1100). A long prefill records several layers per command buffer to avoid
/// host round trips, but one submission must stay well inside the GPU watchdog: Windows' TDR (2 s by default, and the display-driving
/// Meteor Lake Arc enforces it) or a driver's own limit. A fixed "layers per submission by row count" table, calibrated on a 4B model,
/// does not bound time: a 27B layer is ~10x the work, and an attention layer's work grows with the KV depth, so a 1024-row tile at
/// ~85k positions lost the device on Strix.
/// </summary>
/// <remarks>
/// <para>
/// <b>Cost model.</b> A layer's work is split into <em>dense</em> GFLOPs (projections and FFN: <c>2 x params x rows</c>) and
/// <em>attention</em> GFLOPs (<c>4 x rows x mean KV length x heads x headDim</c>, causal). The two run on different kernels with very
/// different efficiency (Strix: ~8.6 TFLOP/s dense vs ~1 TFLOP/s for hd256 attention), so each has its own rate in seconds per GFLOP.
/// </para>
/// <para>
/// <b>Calibration.</b> Every prefill submission's wall time is fed back (<see cref="Observe"/>) and the two rates are refitted by
/// least squares with exponential forgetting. Two fixed pseudo-observations at deliberately slow prior rates keep the fit defined
/// when one kind of work has not been seen yet (an attention rate learnt only from shallow tiles is barely constrained), so an
/// unknown rate errs towards smaller submissions, not a device loss.
/// </para>
/// <para>
/// <b>Policy.</b> <see cref="LayersInSubmission"/> packs consecutive layers until the estimate would exceed the budget (at least one).
/// <see cref="RowsPerForward"/> splits a forward's rows when a single attention layer alone would exceed it.
/// </para>
/// </remarks>
internal sealed class VulkanPrefillSubmitBudget
{
    /// <summary>Default per-submission budget, seconds: half the Windows TDR default, leaving room for a 2x misestimate.</summary>
    internal const double DefaultBudgetSeconds = 1.0;

    /// <summary>Prior dense rate, seconds per GFLOP (1 TFLOP/s: slower than any GPU this backend targets, so conservative).</summary>
    internal const double PriorDenseSecondsPerGflop = 1e-3;

    /// <summary>Prior attention rate, seconds per GFLOP (0.25 TFLOP/s).</summary>
    internal const double PriorAttentionSecondsPerGflop = 4e-3;

    /// <summary>
    /// Fewest rows a forward is costed at. A tile shorter than this does not fill the GPU (too few query blocks per head, too few
    /// GEMM row tiles), so its time does not fall in proportion to its FLOPs: on Strix a 58-row tail at 32k positions took 1.65 s,
    /// against ~0.95 s estimated from its FLOPs at the rates the full tiles had just measured.
    /// </summary>
    internal const int MinEffectiveRows = 128;

    /// <summary>Smallest row slice <see cref="RowsPerForward"/> produces.</summary>
    internal const int MinRowsPerForward = 64;

    /// <summary>
    /// Scale of the prior pseudo-observations, GFLOP: small next to one prefill submission's work (hundreds to thousands of GFLOP),
    /// so observed work outweighs the prior within a submission or two, while a kind of work seen only in trace amounts stays near
    /// its prior. Kept small also because dense and attention work are correlated in real submissions: a heavier (slow) attention
    /// prior leaks into the dense fit and makes it optimistic (300 gave dense 8 % fast in the calibration test, 100 about 1 %).
    /// </summary>
    private const double PriorGflops = 100;

    /// <summary>Weight kept per new observation (exponential forgetting): the fit tracks the last ~10 submissions.</summary>
    private const double Forget = 0.9;

    private double _sDD, _sDA, _sAA, _sDT, _sAT;

    /// <summary>Creates a budget.</summary>
    /// <param name="budgetSeconds">Target GPU time per submission.</param>
    public VulkanPrefillSubmitBudget(double budgetSeconds = DefaultBudgetSeconds)
    {
        if (!(budgetSeconds > 0)) throw new ArgumentOutOfRangeException(nameof(budgetSeconds));
        BudgetSeconds = budgetSeconds;
        DenseSecondsPerGflop = PriorDenseSecondsPerGflop;
        AttentionSecondsPerGflop = PriorAttentionSecondsPerGflop;
    }

    /// <summary>
    /// The budget from <c>DOTLLM_VK_SUBMIT_BUDGET_MS</c> when it parses to a positive number, else <see cref="DefaultBudgetSeconds"/>.
    /// </summary>
    /// <returns>A new budget.</returns>
    public static VulkanPrefillSubmitBudget FromEnvironment()
        => double.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VK_SUBMIT_BUDGET_MS"), NumberStyles.Float,
               CultureInfo.InvariantCulture, out double ms) && ms > 0
            ? new VulkanPrefillSubmitBudget(ms / 1000)
            : new VulkanPrefillSubmitBudget();

    /// <summary>Target GPU time per submission, seconds.</summary>
    public double BudgetSeconds { get; }

    /// <summary>Current dense rate estimate, seconds per GFLOP.</summary>
    public double DenseSecondsPerGflop { get; private set; }

    /// <summary>Current attention rate estimate, seconds per GFLOP.</summary>
    public double AttentionSecondsPerGflop { get; private set; }

    /// <summary>Submissions observed so far.</summary>
    public int Observations { get; private set; }

    /// <summary>Longest observed submission, seconds (diagnostics).</summary>
    public double LongestObservedSeconds { get; private set; }

    /// <summary>Rows a forward of <paramref name="rows"/> rows is costed at: at least <see cref="MinEffectiveRows"/>.</summary>
    /// <param name="rows">Rows in the forward.</param>
    /// <returns>Rows to cost.</returns>
    public static int EffectiveRows(int rows) => Math.Max(rows, MinEffectiveRows);

    /// <summary>Dense GFLOPs of one layer over <paramref name="rows"/> rows.</summary>
    /// <param name="paramsPerLayer">Weight elements of one layer (projections + FFN).</param>
    /// <param name="rows">Rows (tokens) in the forward.</param>
    /// <returns>GFLOPs.</returns>
    public static double DenseGflops(double paramsPerLayer, int rows) => 2.0 * paramsPerLayer * rows / 1e9;

    /// <summary>
    /// Attention GFLOPs (QK^T and PV) of one causal attention layer over <paramref name="rows"/> query rows ending at KV length
    /// <paramref name="kvEnd"/>: row <c>i</c> attends to <c>kvEnd - rows + i + 1</c> positions.
    /// </summary>
    /// <param name="rows">Query rows.</param>
    /// <param name="kvEnd">KV length after the forward (one past its last position).</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="headDim">Head dimension.</param>
    /// <returns>GFLOPs.</returns>
    public static double AttentionGflops(int rows, int kvEnd, int numHeads, int headDim)
    {
        double meanKv = Math.Max(1.0, kvEnd - ((rows - 1) / 2.0));
        return 4.0 * rows * meanKv * numHeads * headDim / 1e9;
    }

    /// <summary>Estimated seconds for <paramref name="denseGflops"/> dense and <paramref name="attentionGflops"/> attention work.</summary>
    /// <param name="denseGflops">Dense work.</param>
    /// <param name="attentionGflops">Attention work.</param>
    /// <returns>Seconds.</returns>
    public double EstimateSeconds(double denseGflops, double attentionGflops)
        => (denseGflops * DenseSecondsPerGflop) + (attentionGflops * AttentionSecondsPerGflop);

    /// <summary>
    /// How many consecutive layers starting at <paramref name="firstLayer"/> fit one submission: layers are added while the estimate
    /// stays within the budget, and at least one is always taken.
    /// </summary>
    /// <param name="isAttention">Per layer: true for a full-attention layer.</param>
    /// <param name="firstLayer">First layer of the submission.</param>
    /// <param name="denseGflopsPerLayer">Dense work of one layer for this forward.</param>
    /// <param name="attentionGflopsPerLayer">Attention work of one attention layer for this forward.</param>
    /// <returns>Layers in the submission (1 .. remaining layers).</returns>
    public int LayersInSubmission(ReadOnlySpan<bool> isAttention, int firstLayer, double denseGflopsPerLayer, double attentionGflopsPerLayer)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(firstLayer);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(firstLayer, isAttention.Length);
        double seconds = 0;
        int n = 0;
        for (int l = firstLayer; l < isAttention.Length; l++)
        {
            double layer = EstimateSeconds(denseGflopsPerLayer, isAttention[l] ? attentionGflopsPerLayer : 0);
            if (n > 0 && seconds + layer > BudgetSeconds) break;
            seconds += layer;
            n++;
        }
        return n;
    }

    /// <summary>
    /// Rows per forward so that one attention layer (plus its dense work) fits the budget: <paramref name="rows"/> when it already
    /// does, else the largest multiple of <see cref="MinRowsPerForward"/> that does (at least <see cref="MinRowsPerForward"/>).
    /// Estimated at the forward's final KV length, the deepest any slice reaches.
    /// </summary>
    /// <param name="rows">Rows in the forward.</param>
    /// <param name="kvEnd">KV length after the forward.</param>
    /// <param name="paramsPerLayer">Weight elements of one layer.</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="headDim">Head dimension.</param>
    /// <returns>Rows per forward (1 .. <paramref name="rows"/>).</returns>
    public int RowsPerForward(int rows, int kvEnd, double paramsPerLayer, int numHeads, int headDim)
    {
        if (rows <= MinRowsPerForward || LayerSeconds(rows) <= BudgetSeconds) return rows;
        int best = MinRowsPerForward;
        for (int r = 2 * MinRowsPerForward; r < rows; r += MinRowsPerForward)
        {
            if (LayerSeconds(r) > BudgetSeconds) break;
            best = r;
        }
        return best;

        double LayerSeconds(int r)
            => EstimateSeconds(DenseGflops(paramsPerLayer, EffectiveRows(r)), AttentionGflops(EffectiveRows(r), kvEnd, numHeads, headDim));
    }

    /// <summary>
    /// Feeds back one submission's measured wall time and refits both rates (least squares with forgetting, anchored by the priors).
    /// Non-positive or non-finite input is ignored.
    /// </summary>
    /// <param name="denseGflops">Dense work the submission held.</param>
    /// <param name="attentionGflops">Attention work the submission held.</param>
    /// <param name="seconds">Measured wall time of the submission.</param>
    public void Observe(double denseGflops, double attentionGflops, double seconds)
    {
        if (!(seconds > 0) || !double.IsFinite(seconds) || !(denseGflops >= 0) || !(attentionGflops >= 0)
            || denseGflops + attentionGflops <= 0)
            return;

        _sDD = (_sDD * Forget) + (denseGflops * denseGflops);
        _sDA = (_sDA * Forget) + (denseGflops * attentionGflops);
        _sAA = (_sAA * Forget) + (attentionGflops * attentionGflops);
        _sDT = (_sDT * Forget) + (denseGflops * seconds);
        _sAT = (_sAT * Forget) + (attentionGflops * seconds);
        Observations++;
        if (seconds > LongestObservedSeconds) LongestObservedSeconds = seconds;

        // The priors as two fixed pseudo-observations: (PriorGflops, 0) and (0, PriorGflops) at the prior rates.
        double p2 = PriorGflops * PriorGflops;
        double dd = _sDD + p2, aa = _sAA + p2, da = _sDA;
        double dt = _sDT + (p2 * PriorDenseSecondsPerGflop), at = _sAT + (p2 * PriorAttentionSecondsPerGflop);
        double det = (dd * aa) - (da * da);
        if (!(det > 0)) return;
        double dense = ((dt * aa) - (at * da)) / det;
        double attention = ((at * dd) - (dt * da)) / det;

        // A rate can never be free: floor each at 1/64 of its prior (64 TFLOP/s dense, 16 attention).
        DenseSecondsPerGflop = Math.Max(dense, PriorDenseSecondsPerGflop / 64);
        AttentionSecondsPerGflop = Math.Max(attention, PriorAttentionSecondsPerGflop / 64);
    }
}
