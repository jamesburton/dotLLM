namespace DotLLM.Core.Moe;

/// <summary>A linear time model <c>us(n) = A + B · n</c> for one expert over <c>n</c> tokens.</summary>
/// <param name="A">Fixed microseconds per call (weight read / launch).</param>
/// <param name="B">Microseconds per token.</param>
public readonly record struct LinearCost(double A, double B)
{
    /// <summary>Microseconds for <paramref name="n"/> tokens (never negative).</summary>
    /// <param name="n">Token count.</param>
    /// <returns>The estimate.</returns>
    public double Us(int n) => Math.Max(0, A + B * n);

    /// <summary>
    /// Fits <c>a + b·n</c> through two measurements at <c>n = 1</c> and <c>n = 8</c> (#1071 calibration). A slope below zero
    /// (timer noise: the n = 8 run measured faster) is clamped to 0, and the intercept to at least 0.
    /// </summary>
    /// <param name="us1">Median microseconds at n = 1.</param>
    /// <param name="us8">Median microseconds at n = 8.</param>
    /// <returns>The fitted model.</returns>
    public static LinearCost FitFrom1And8(double us1, double us8)
    {
        double b = Math.Max(0, (us8 - us1) / 7.0);
        return new LinearCost(Math.Max(0, us1 - b), b);
    }
}

/// <summary>
/// Per-layout costs of the cold-expert decision: CPU time and device time per expert as a function of its token count.
/// </summary>
/// <param name="Layout">The expert layout (up and down bytes) these costs belong to.</param>
/// <param name="Cpu">CPU microseconds (single-threaded, one expert).</param>
/// <param name="Gpu">Device microseconds once the expert is resident.</param>
public sealed record MoeLayoutCost(ExpertLayout Layout, LinearCost Cpu, LinearCost Gpu);

/// <summary>
/// The cost model of the per-layer cold-expert assignment (issue #1071, design #1063 sections 6 and 8c), built from a load-time
/// calibration: pinned host-to-device bandwidth, the device-host-device round trip the CPU leg adds, and per-layout CPU and
/// device time per expert. Pure arithmetic; the backend measures the inputs.
/// </summary>
/// <param name="H2DBytesPerSecond">Measured pinned host-to-device bandwidth.</param>
/// <param name="SyncRoundTripUs">Median D2H + host-sync round trip in microseconds (one extra per layer that uses the CPU).</param>
/// <param name="Layouts">Costs per expert layout.</param>
public sealed record MoeCostModel(double H2DBytesPerSecond, double SyncRoundTripUs, IReadOnlyList<MoeLayoutCost> Layouts)
{
    /// <summary>Effective device read bandwidth assumed for the expert GEMVs when they are not measured (RTX 3060 class).</summary>
    public const double DefaultDeviceBytesPerSecond = 300e9;

    /// <summary>Tokens the expert-grouped kernel serves per weight read (one warp row dotted against up to 8 tokens).</summary>
    public const int GpuTokensPerWeightRead = 8;

    /// <summary>Default share of the last token's time the link may spend on promotion copies (design 8c: about 25 %).</summary>
    public const double DefaultPromotionLinkShare = 0.25;

    /// <summary>
    /// Default device time of one expert: its bytes at <paramref name="deviceBytesPerSecond"/> per weight read, one read per
    /// <see cref="GpuTokensPerWeightRead"/> tokens, expressed as <c>a + b·n</c> with the read amortised linearly.
    /// </summary>
    /// <param name="expertBytes">Up plus down bytes of the expert.</param>
    /// <param name="deviceBytesPerSecond">Effective device read bandwidth.</param>
    /// <returns>The device cost model.</returns>
    public static LinearCost DefaultGpuCost(long expertBytes, double deviceBytesPerSecond = DefaultDeviceBytesPerSecond)
    {
        double read = expertBytes / deviceBytesPerSecond * 1e6;
        return new LinearCost(read, read / GpuTokensPerWeightRead);
    }

    /// <summary>Microseconds to copy <paramref name="bytes"/> host to device over the calibrated link.</summary>
    /// <param name="bytes">Bytes.</param>
    /// <returns>The estimate (infinite when the bandwidth is unknown).</returns>
    public double CopyUs(long bytes) => H2DBytesPerSecond > 0 ? bytes / H2DBytesPerSecond * 1e6 : double.PositiveInfinity;

    /// <summary>The costs of <paramref name="layout"/>, or null when it was not calibrated.</summary>
    /// <param name="layout">Expert layout.</param>
    /// <returns>The costs, or null.</returns>
    public MoeLayoutCost? For(ExpertLayout layout)
    {
        foreach (var c in Layouts)
            if (c.Layout == layout) return c;
        return null;
    }

    /// <summary>
    /// The three costs of one uncached expert of <paramref name="layout"/> with <paramref name="tokens"/> tokens. A layout that
    /// was not calibrated costs infinity on the CPU (so it is copied, the #1070 behaviour).
    /// </summary>
    /// <param name="layout">Expert layout.</param>
    /// <param name="tokens">Tokens routed to the expert in this call.</param>
    /// <returns>The costs.</returns>
    public ColdExpertCost ColdCost(ExpertLayout layout, int tokens)
    {
        long bytes = layout.UpBytes + layout.DownBytes;
        var c = For(layout);
        double gpu = (c?.Gpu ?? DefaultGpuCost(bytes)).Us(tokens);
        return new ColdExpertCost(c is null ? double.PositiveInfinity : c.Cpu.Us(tokens), CopyUs(bytes), gpu);
    }

    /// <summary>
    /// Device time of a resident (cached) expert of <paramref name="layout"/> with <paramref name="tokens"/> tokens.
    /// </summary>
    /// <param name="layout">Expert layout.</param>
    /// <param name="tokens">Tokens routed to the expert.</param>
    /// <returns>Microseconds.</returns>
    public double ResidentGpuUs(ExpertLayout layout, int tokens)
        => (For(layout)?.Gpu ?? DefaultGpuCost(layout.UpBytes + layout.DownBytes)).Us(tokens);

    /// <summary>
    /// Bytes promote-only admission may copy for the next <paramref name="tokens"/> tokens: what the link moves in
    /// <paramref name="linkShare"/> of the last token's time (design 8c), per token.
    /// </summary>
    /// <param name="tokens">Tokens of the next forward.</param>
    /// <param name="lastTokenUs">Wall time of one token of the previous forward, in microseconds (0 when unknown).</param>
    /// <param name="linkShare">Share of that time the copies may use.</param>
    /// <param name="minimumBytes">Floor (e.g. one expert), so admission can make progress before the first measurement.</param>
    /// <returns>The byte budget.</returns>
    public long PromotionBudgetBytes(int tokens, double lastTokenUs, double linkShare = DefaultPromotionLinkShare, long minimumBytes = 0)
    {
        if (H2DBytesPerSecond <= 0 || tokens <= 0) return minimumBytes;
        double perToken = H2DBytesPerSecond * Math.Max(lastTokenUs, 0) * 1e-6 * linkShare;
        return Math.Max(minimumBytes, (long)Math.Min(perToken * tokens, long.MaxValue / 2.0));
    }
}

/// <summary>The costs of one uncached expert in one layer call, in microseconds.</summary>
/// <param name="CpuUs">Computing it on the CPU (single-threaded).</param>
/// <param name="CopyUs">Copying its weights to the device.</param>
/// <param name="GpuUs">Computing it on the device once copied.</param>
public readonly record struct ColdExpertCost(double CpuUs, double CopyUs, double GpuUs)
{
    /// <summary>The device-side cost: copy plus compute.</summary>
    public double DeviceUs => CopyUs + GpuUs;
}

/// <summary>
/// The per-layer CPU / device assignment of uncached experts (issue #1071, design #1063 section 8c). Pure.
/// <list type="number">
///   <item><b>Fiddler rule</b> (arXiv 2402.07033): an expert goes to the CPU when computing it there is cheaper than copying it
///   and computing it on the device.</item>
///   <item><b>DALI-style greedy balance</b> (arXiv 2602.03495): in ascending order of <c>|cpu − (copy + gpu)|</c>, each expert
///   is moved to the other side when that lowers the layer's makespan <c>max(CPU span, device span)</c>.</item>
/// </list>
/// The CPU span is the expert-parallel makespan bound over the workers, <c>max(Σcpu / workers, max cpu)</c>, plus the extra round
/// trip the CPU leg costs; the device span is the resident experts' time plus <c>Σ(copy + gpu)</c> of the copied ones.
/// </summary>
public static class MoeColdAssignment
{
    /// <summary>Greedy passes at most (each pass visits every expert once).</summary>
    public const int MaxPasses = 4;

    /// <summary>Assigns each expert of <paramref name="costs"/> to the CPU or the device.</summary>
    /// <param name="costs">Costs of the layer call's uncached experts.</param>
    /// <param name="residentGpuUs">Device time already committed to the cached experts of the call.</param>
    /// <param name="cpuWorkers">CPU leg workers (one expert each at a time).</param>
    /// <param name="cpuLegOverheadUs">Fixed cost of using the CPU leg at all (the extra round trip); 0 when no expert is on it.</param>
    /// <param name="onCpu">Receives true for each expert placed on the CPU.</param>
    /// <returns>Experts placed on the CPU.</returns>
    /// <exception cref="ArgumentException"><paramref name="onCpu"/> is shorter than <paramref name="costs"/>.</exception>
    public static int Assign(ReadOnlySpan<ColdExpertCost> costs, double residentGpuUs, int cpuWorkers, double cpuLegOverheadUs,
        Span<bool> onCpu)
    {
        if (onCpu.Length < costs.Length) throw new ArgumentException("onCpu is shorter than costs.", nameof(onCpu));
        int n = costs.Length;
        if (n == 0) return 0;
        cpuWorkers = Math.Max(1, cpuWorkers);

        // 1. Fiddler. The greedy pass only accepts improving single moves, so it starts from the best of the Fiddler assignment
        //    and the two one-sided ones (every expert that can run there on the device, or on the CPU).
        for (int i = 0; i < n; i++) onCpu[i] = costs[i].CpuUs < costs[i].DeviceUs;
        double start = Makespan(costs, onCpu, residentGpuUs, cpuWorkers, cpuLegOverheadUs);
        Span<bool> trial = n <= 256 ? stackalloc bool[n] : new bool[n];
        for (int side = 0; side < 2; side++)
        {
            for (int i = 0; i < n; i++)
                trial[i] = side == 0 ? double.IsPositiveInfinity(costs[i].DeviceUs) : !double.IsPositiveInfinity(costs[i].CpuUs);
            double m = Makespan(costs, trial, residentGpuUs, cpuWorkers, cpuLegOverheadUs);
            if (m < start - 1e-9)
            {
                start = m;
                trial.CopyTo(onCpu);
            }
        }

        // 2. Greedy balance, cheapest moves first (ties to the lower index, so the result is deterministic).
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        var keys = new double[n];
        for (int i = 0; i < n; i++)
        {
            double d = Math.Abs(costs[i].CpuUs - costs[i].DeviceUs);
            keys[i] = double.IsNaN(d) ? double.PositiveInfinity : d;
        }
        Array.Sort(order, (a, b) => keys[a] != keys[b] ? keys[a].CompareTo(keys[b]) : a.CompareTo(b));

        double best = start;
        for (int pass = 0; pass < MaxPasses; pass++)
        {
            bool improved = false;
            foreach (int i in order)
            {
                // Never move an expert onto a side where it cannot run at all (infinite cost).
                if (onCpu[i] ? double.IsPositiveInfinity(costs[i].DeviceUs) : double.IsPositiveInfinity(costs[i].CpuUs)) continue;
                onCpu[i] = !onCpu[i];
                double m = Makespan(costs, onCpu, residentGpuUs, cpuWorkers, cpuLegOverheadUs);
                if (m < best - 1e-9) { best = m; improved = true; }
                else onCpu[i] = !onCpu[i];
            }
            if (!improved) break;
        }

        int cpu = 0;
        for (int i = 0; i < n; i++) if (onCpu[i]) cpu++;
        return cpu;
    }

    /// <summary>The layer makespan of an assignment (see the type remarks).</summary>
    /// <param name="costs">Costs of the uncached experts.</param>
    /// <param name="onCpu">The assignment.</param>
    /// <param name="residentGpuUs">Device time of the cached experts.</param>
    /// <param name="cpuWorkers">CPU leg workers.</param>
    /// <param name="cpuLegOverheadUs">Fixed cost of a non-empty CPU leg.</param>
    /// <returns>Microseconds.</returns>
    public static double Makespan(ReadOnlySpan<ColdExpertCost> costs, ReadOnlySpan<bool> onCpu, double residentGpuUs,
        int cpuWorkers, double cpuLegOverheadUs)
    {
        double cpuSum = 0, cpuMax = 0, device = residentGpuUs;
        bool anyCpu = false;
        for (int i = 0; i < costs.Length; i++)
        {
            if (onCpu[i])
            {
                anyCpu = true;
                cpuSum += costs[i].CpuUs;
                cpuMax = Math.Max(cpuMax, costs[i].CpuUs);
            }
            else
            {
                device += costs[i].DeviceUs;
            }
        }
        double cpuSpan = anyCpu ? cpuLegOverheadUs + Math.Max(cpuSum / Math.Max(1, cpuWorkers), cpuMax) : 0;
        return Math.Max(cpuSpan, device);
    }
}
