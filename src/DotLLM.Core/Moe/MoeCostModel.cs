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
/// Per-layout costs of the cold-expert decision: CPU time, device time and (#1083) integrated-GPU time per expert as a function of
/// its token count.
/// </summary>
/// <param name="Layout">The expert layout (up and down bytes) these costs belong to.</param>
/// <param name="Cpu">CPU microseconds (single-threaded, one expert).</param>
/// <param name="Gpu">Device microseconds once the expert is resident.</param>
/// <param name="Igpu">
/// Integrated-GPU microseconds of one expert inside an iGPU submission, without the submission's fixed round trip
/// (<see cref="MoeCostModel.IgpuSubmitUs"/>); null when the layout was not measured on an iGPU (#1083).
/// </param>
public sealed record MoeLayoutCost(ExpertLayout Layout, LinearCost Cpu, LinearCost Gpu, LinearCost? Igpu = null);

/// <summary>
/// The cost model of the per-layer cold-expert assignment (issue #1071, design #1063 sections 6 and 8c; the iGPU leg: #1083, 8c′),
/// built from a load-time calibration: pinned host-to-device bandwidth, the device-host-device round trip a cold leg adds,
/// per-layout CPU, device and iGPU time per expert, and the iGPU's submit-plus-fence round trip. Pure arithmetic; the backend
/// measures the inputs.
/// </summary>
/// <param name="H2DBytesPerSecond">Measured pinned host-to-device bandwidth.</param>
/// <param name="SyncRoundTripUs">Median D2H + host-sync round trip in microseconds (one extra per layer that uses a cold leg).</param>
/// <param name="Layouts">Costs per expert layout.</param>
/// <param name="IgpuSubmitUs">
/// Fixed microseconds of one iGPU submission (record, submit, fence) whatever it computes; positive infinity (the default) when
/// there is no iGPU leg, which keeps every expert off it.
/// </param>
public sealed record MoeCostModel(double H2DBytesPerSecond, double SyncRoundTripUs, IReadOnlyList<MoeLayoutCost> Layouts,
    double IgpuSubmitUs = double.PositiveInfinity)
{
    /// <summary>Effective device read bandwidth assumed for the expert GEMVs when they are not measured (RTX 3060 class).</summary>
    public const double DefaultDeviceBytesPerSecond = 300e9;

    /// <summary>Tokens the expert-grouped kernel serves per weight read (one warp row dotted against up to 8 tokens).</summary>
    public const int GpuTokensPerWeightRead = 8;

    /// <summary>Default share of the last token's time the link may spend on promotion copies (design 8c: about 25 %).</summary>
    public const double DefaultPromotionLinkShare = 0.25;

    /// <summary>True when the model has an iGPU leg (a finite <see cref="IgpuSubmitUs"/>).</summary>
    public bool HasIgpu => !double.IsPositiveInfinity(IgpuSubmitUs) && !double.IsNaN(IgpuSubmitUs);

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
    /// The costs of one uncached expert of <paramref name="layout"/> with <paramref name="tokens"/> tokens. A layout that was not
    /// calibrated costs infinity on the CPU (so it is copied, the #1070 behaviour) and on the iGPU; so does any layout when the
    /// model has no iGPU leg.
    /// </summary>
    /// <param name="layout">Expert layout.</param>
    /// <param name="tokens">Tokens routed to the expert in this call.</param>
    /// <returns>The costs.</returns>
    public ColdExpertCost ColdCost(ExpertLayout layout, int tokens)
    {
        long bytes = layout.UpBytes + layout.DownBytes;
        var c = For(layout);
        double gpu = (c?.Gpu ?? DefaultGpuCost(bytes)).Us(tokens);
        double igpu = HasIgpu && c?.Igpu is { } ig ? ig.Us(tokens) : double.PositiveInfinity;
        return new ColdExpertCost(c is null ? double.PositiveInfinity : c.Cpu.Us(tokens), CopyUs(bytes), gpu, igpu);
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

/// <summary>Where one uncached expert of a layer call runs (#1071, #1083).</summary>
public enum ColdPlace : byte
{
    /// <summary>On the device (CUDA): copied into the cache or streamed, then computed there.</summary>
    Device,

    /// <summary>On the CPU, from the memory-mapped bank (#1071).</summary>
    Cpu,

    /// <summary>On the integrated GPU (Vulkan), from the zero-copy bank import (#1083).</summary>
    Igpu,
}

/// <summary>The costs of one uncached expert in one layer call, in microseconds.</summary>
/// <param name="CpuUs">Computing it on the CPU (single-threaded).</param>
/// <param name="CopyUs">Copying its weights to the device.</param>
/// <param name="GpuUs">Computing it on the device once copied.</param>
/// <param name="IgpuUs">Computing it on the integrated GPU inside a submission (#1083); infinity when there is no iGPU leg.</param>
public readonly record struct ColdExpertCost(double CpuUs, double CopyUs, double GpuUs, double IgpuUs = double.PositiveInfinity)
{
    /// <summary>The device-side cost: copy plus compute.</summary>
    public double DeviceUs => CopyUs + GpuUs;

    /// <summary>The cost of running the expert at <paramref name="place"/>.</summary>
    /// <param name="place">The side.</param>
    /// <returns>Microseconds (infinity when it cannot run there).</returns>
    public double Us(ColdPlace place) => place switch
    {
        ColdPlace.Device => DeviceUs,
        ColdPlace.Cpu => CpuUs,
        _ => IgpuUs,
    };
}

/// <summary>
/// The per-layer assignment of uncached experts to the device, the CPU and (#1083) the integrated GPU (issue #1071, design #1063
/// sections 8c and 8c′). Pure.
/// <list type="number">
///   <item><b>Fiddler rule</b> (arXiv 2402.07033), generalised to three sides: an expert goes to its individually cheapest side
///   (device on a tie, then the CPU).</item>
///   <item><b>DALI-style greedy balance</b> (arXiv 2602.03495): in ascending order of the gap between an expert's two cheapest sides,
///   each expert is moved to the other side that lowers the layer's makespan most, until no single move helps.</item>
/// </list>
/// The makespan is <c>max(device span, cold span)</c>. The device span is the resident experts' time plus <c>Σ(copy + gpu)</c> of
/// the copied ones. The cold span is zero when no expert is cold, else the cold legs' extra round trip plus
/// <c>max(CPU span, iGPU span)</c>, because the CPU and iGPU legs run concurrently: the CPU span is the expert-parallel bound
/// <c>max(Σcpu / workers, max cpu)</c>; the iGPU span is the submission's fixed cost plus <c>Σigpu</c> (one queue), and zero when no
/// expert is on the iGPU, so an empty iGPU set never pays the submit.
/// </summary>
public static class MoeColdAssignment
{
    /// <summary>Greedy passes at most (each pass visits every expert once).</summary>
    public const int MaxPasses = 4;

    private static readonly ColdPlace[] Sides = [ColdPlace.Device, ColdPlace.Cpu, ColdPlace.Igpu];

    /// <summary>Assigns each expert of <paramref name="costs"/> to the CPU or the device (the #1071 two-way form; no iGPU).</summary>
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
        Span<ColdPlace> place = costs.Length <= 256 ? stackalloc ColdPlace[costs.Length] : new ColdPlace[costs.Length];
        var (cpu, _) = Assign(costs, residentGpuUs, cpuWorkers, cpuLegOverheadUs, double.PositiveInfinity, place);
        for (int i = 0; i < costs.Length; i++) onCpu[i] = place[i] == ColdPlace.Cpu;
        return cpu;
    }

    /// <summary>
    /// Assigns each expert of <paramref name="costs"/> to the device, the CPU or the integrated GPU (#1083), minimising
    /// <c>max(device + copy, cold-leg overhead + max(CPU, iGPU))</c> for the layer call (see the type remarks).
    /// </summary>
    /// <param name="costs">Costs of the layer call's uncached experts.</param>
    /// <param name="residentGpuUs">Device time already committed to the cached experts of the call.</param>
    /// <param name="cpuWorkers">CPU leg workers (one expert each at a time).</param>
    /// <param name="coldLegOverheadUs">Fixed cost of any cold leg (the extra device-host-device round trip); paid once per call.</param>
    /// <param name="igpuSubmitUs">Fixed cost of a non-empty iGPU submission; infinity disables the iGPU side.</param>
    /// <param name="place">Receives each expert's side.</param>
    /// <returns>Experts placed on the CPU and on the iGPU.</returns>
    /// <exception cref="ArgumentException"><paramref name="place"/> is shorter than <paramref name="costs"/>.</exception>
    public static (int Cpu, int Igpu) Assign(ReadOnlySpan<ColdExpertCost> costs, double residentGpuUs, int cpuWorkers,
        double coldLegOverheadUs, double igpuSubmitUs, Span<ColdPlace> place)
    {
        if (place.Length < costs.Length) throw new ArgumentException("place is shorter than costs.", nameof(place));
        int n = costs.Length;
        if (n == 0) return (0, 0);
        cpuWorkers = Math.Max(1, cpuWorkers);
        bool igpu = IgpuEnabled(igpuSubmitUs);

        // 1. Fiddler. The greedy pass only accepts improving single moves, so it starts from the best of the Fiddler assignment and
        //    the one-sided ones (every expert that can run there on that side, the rest where Fiddler puts them).
        Fiddler(costs, igpuSubmitUs, place);
        double start = Makespan(costs, place, residentGpuUs, cpuWorkers, coldLegOverheadUs, igpuSubmitUs);
        Span<ColdPlace> trial = n <= 256 ? stackalloc ColdPlace[n] : new ColdPlace[n];
        Span<ColdPlace> fiddler = n <= 256 ? stackalloc ColdPlace[n] : new ColdPlace[n];
        place[..n].CopyTo(fiddler);
        foreach (var side in Sides)
        {
            if (side == ColdPlace.Igpu && !igpu) continue;
            for (int i = 0; i < n; i++)
                trial[i] = double.IsPositiveInfinity(costs[i].Us(side)) ? fiddler[i] : side;
            double m = Makespan(costs, trial, residentGpuUs, cpuWorkers, coldLegOverheadUs, igpuSubmitUs);
            if (m < start - 1e-9)
            {
                start = m;
                trial.CopyTo(place);
            }
        }

        // 2. Greedy balance, cheapest moves first (ties to the lower index, so the result is deterministic).
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        var keys = new double[n];
        for (int i = 0; i < n; i++) keys[i] = TwoCheapestGap(costs[i], igpu);
        Array.Sort(order, (a, b) => keys[a] != keys[b] ? keys[a].CompareTo(keys[b]) : a.CompareTo(b));

        double best = start;
        for (int pass = 0; pass < MaxPasses; pass++)
        {
            bool improved = false;
            foreach (int i in order)
            {
                var from = place[i];
                var bestSide = from;
                foreach (var to in Sides)
                {
                    // Never move an expert onto a side where it cannot run at all (infinite cost), or onto a disabled iGPU.
                    if (to == from || (to == ColdPlace.Igpu && !igpu) || double.IsPositiveInfinity(costs[i].Us(to))) continue;
                    place[i] = to;
                    double m = Makespan(costs, place, residentGpuUs, cpuWorkers, coldLegOverheadUs, igpuSubmitUs);
                    if (m < best - 1e-9)
                    {
                        best = m;
                        bestSide = to;
                    }
                }
                place[i] = bestSide;
                if (bestSide != from) improved = true;
            }
            if (!improved) break;
        }

        int cpu = 0, ig = 0;
        for (int i = 0; i < n; i++)
        {
            if (place[i] == ColdPlace.Cpu) cpu++;
            else if (place[i] == ColdPlace.Igpu) ig++;
        }
        return (cpu, ig);
    }

    /// <summary>
    /// The Fiddler rule alone (arXiv 2402.07033): an expert goes to the CPU exactly when computing it there is cheaper than copying
    /// it and computing it on the device. <see cref="Assign(ReadOnlySpan{ColdExpertCost}, double, int, double, Span{bool})"/> starts its
    /// balance from this (or a better one-sided) assignment.
    /// </summary>
    /// <param name="costs">Costs of the uncached experts.</param>
    /// <param name="onCpu">Receives true for each expert the rule places on the CPU.</param>
    /// <returns>Experts placed on the CPU.</returns>
    public static int Fiddler(ReadOnlySpan<ColdExpertCost> costs, Span<bool> onCpu)
    {
        int cpu = 0;
        for (int i = 0; i < costs.Length; i++)
            if (onCpu[i] = costs[i].CpuUs < costs[i].DeviceUs) cpu++;
        return cpu;
    }

    /// <summary>
    /// The Fiddler rule over three sides (#1083): each expert goes to its individually cheapest side, the device on a tie, then the
    /// CPU, then the iGPU. With the iGPU disabled (<paramref name="igpuSubmitUs"/> infinite) this is exactly
    /// <see cref="Fiddler(ReadOnlySpan{ColdExpertCost}, Span{bool})"/>.
    /// </summary>
    /// <param name="costs">Costs of the uncached experts.</param>
    /// <param name="igpuSubmitUs">Fixed cost of an iGPU submission; infinity disables the iGPU side.</param>
    /// <param name="place">Receives each expert's side.</param>
    /// <returns>Experts placed off the device.</returns>
    public static int Fiddler(ReadOnlySpan<ColdExpertCost> costs, double igpuSubmitUs, Span<ColdPlace> place)
    {
        bool igpu = IgpuEnabled(igpuSubmitUs);
        int cold = 0;
        for (int i = 0; i < costs.Length; i++)
        {
            var c = costs[i];
            var side = ColdPlace.Device;
            double best = c.DeviceUs;
            if (c.CpuUs < best) { side = ColdPlace.Cpu; best = c.CpuUs; }
            if (igpu && c.IgpuUs < best) side = ColdPlace.Igpu;
            place[i] = side;
            if (side != ColdPlace.Device) cold++;
        }
        return cold;
    }

    /// <summary>The layer makespan of a two-way assignment (see the type remarks; no iGPU).</summary>
    /// <param name="costs">Costs of the uncached experts.</param>
    /// <param name="onCpu">The assignment.</param>
    /// <param name="residentGpuUs">Device time of the cached experts.</param>
    /// <param name="cpuWorkers">CPU leg workers.</param>
    /// <param name="cpuLegOverheadUs">Fixed cost of a non-empty CPU leg.</param>
    /// <returns>Microseconds.</returns>
    public static double Makespan(ReadOnlySpan<ColdExpertCost> costs, ReadOnlySpan<bool> onCpu, double residentGpuUs,
        int cpuWorkers, double cpuLegOverheadUs)
    {
        Span<ColdPlace> place = costs.Length <= 256 ? stackalloc ColdPlace[costs.Length] : new ColdPlace[costs.Length];
        for (int i = 0; i < costs.Length; i++) place[i] = onCpu[i] ? ColdPlace.Cpu : ColdPlace.Device;
        return Makespan(costs, place, residentGpuUs, cpuWorkers, cpuLegOverheadUs, double.PositiveInfinity);
    }

    /// <summary>The layer makespan of a three-way assignment (see the type remarks).</summary>
    /// <param name="costs">Costs of the uncached experts.</param>
    /// <param name="place">The assignment.</param>
    /// <param name="residentGpuUs">Device time of the cached experts.</param>
    /// <param name="cpuWorkers">CPU leg workers.</param>
    /// <param name="coldLegOverheadUs">Fixed cost of any non-empty cold leg (paid once).</param>
    /// <param name="igpuSubmitUs">Fixed cost of a non-empty iGPU submission (infinity: an iGPU expert makes the span infinite).</param>
    /// <returns>Microseconds.</returns>
    public static double Makespan(ReadOnlySpan<ColdExpertCost> costs, ReadOnlySpan<ColdPlace> place, double residentGpuUs,
        int cpuWorkers, double coldLegOverheadUs, double igpuSubmitUs)
    {
        double cpuSum = 0, cpuMax = 0, igpuSum = 0, device = residentGpuUs;
        bool anyCpu = false, anyIgpu = false;
        for (int i = 0; i < costs.Length; i++)
        {
            switch (place[i])
            {
                case ColdPlace.Cpu:
                    anyCpu = true;
                    cpuSum += costs[i].CpuUs;
                    cpuMax = Math.Max(cpuMax, costs[i].CpuUs);
                    break;
                case ColdPlace.Igpu:
                    anyIgpu = true;
                    igpuSum += costs[i].IgpuUs;
                    break;
                default:
                    device += costs[i].DeviceUs;
                    break;
            }
        }
        double cpuSpan = anyCpu ? Math.Max(cpuSum / Math.Max(1, cpuWorkers), cpuMax) : 0;
        double igpuSpan = anyIgpu ? igpuSubmitUs + igpuSum : 0;
        double coldSpan = anyCpu || anyIgpu ? coldLegOverheadUs + Math.Max(cpuSpan, igpuSpan) : 0;
        return Math.Max(coldSpan, device);
    }

    private static bool IgpuEnabled(double igpuSubmitUs) => !double.IsPositiveInfinity(igpuSubmitUs) && !double.IsNaN(igpuSubmitUs);

    /// <summary>
    /// The gap between an expert's two cheapest available sides (the greedy order key; with the iGPU disabled it is
    /// <c>|cpu − device|</c>, the #1071 key). NaN sorts last.
    /// </summary>
    private static double TwoCheapestGap(in ColdExpertCost c, bool igpu)
    {
        double a = c.DeviceUs, b = c.CpuUs;
        double lo = Math.Min(a, b), hi = Math.Max(a, b);
        if (igpu)
        {
            double g = c.IgpuUs;
            if (g < lo) { hi = lo; lo = g; }
            else if (g < hi) hi = g;
        }
        double d = Math.Abs(hi - lo);
        return double.IsNaN(d) ? double.PositiveInfinity : d;
    }
}
