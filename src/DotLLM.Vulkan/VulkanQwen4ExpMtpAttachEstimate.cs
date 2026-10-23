namespace DotLLM.Vulkan;

/// <summary>
/// Pure arithmetic of the qwen4exp MTP-head attach guard (#820): will the head fit next to the resident trunk? Kept free of device calls so the
/// accounting is unit-testable. Every term is memory the attach (or the first requests after it) will newly allocate; what the process already
/// holds, including the per-forward scratch allocated at load, is in <see cref="LiveBytes"/> and must not be added again.
/// </summary>
/// <param name="LiveBytes">What this process holds on the device now (resident trunk, scratch sized at load, any KV already grown).</param>
/// <param name="HeadBytes">The head's device weights as they will be uploaded (after the attach-time requantisation) plus the verify snapshots.</param>
/// <param name="ContextBytes">K/V + indexer caches the sequence state can still grow to (lazy growth) plus the head's own K/V.</param>
/// <param name="ScratchBytes">The head's absorb-chunk scratch.</param>
/// <param name="OtherProcessBytes">GPU memory other processes hold on this adapter.</param>
/// <param name="CapacityBytes">The usable resident capacity (the point where the next submit fails).</param>
/// <param name="MarginBytes">Allowance for what is not itemised (all-rows logits read-back, descriptor pools, driver bookkeeping).</param>
internal readonly record struct Qwen4ExpMtpAttachEstimate(
    long LiveBytes, long HeadBytes, long ContextBytes, long ScratchBytes, long OtherProcessBytes, long CapacityBytes,
    long MarginBytes = Qwen4ExpMtpAttachEstimate.DefaultMarginBytes)
{
    /// <summary>256 MiB: measured growth of everything not itemised after attach (1K prefill + decode + MTP rounds) is under 0.1 GiB.</summary>
    public const long DefaultMarginBytes = 256L << 20;

    /// <summary>Total the device must hold with the head attached and the context grown to capacity.</summary>
    public long NeededBytes => LiveBytes + HeadBytes + ContextBytes + ScratchBytes + OtherProcessBytes + MarginBytes;

    /// <summary>True when <see cref="NeededBytes"/> is within <see cref="CapacityBytes"/>.</summary>
    public bool Fits => NeededBytes <= CapacityBytes;

    /// <summary>Bytes over the capacity (0 when it fits).</summary>
    public long ShortfallBytes => Math.Max(0, NeededBytes - CapacityBytes);

    /// <summary>Slack the canary adds beyond every itemised byte (and beyond <see cref="MarginBytes"/>): 256 MiB.</summary>
    public const long ProbeSlackBytes = 256L << 20;

    /// <summary>What the attach newly allocates, with the margin: the head, snapshots, context growth and scratch.</summary>
    public long NewBytes => HeadBytes + ContextBytes + ScratchBytes + MarginBytes;

    /// <summary>What the canary allocates and touches: <see cref="NewBytes"/> plus <see cref="ProbeSlackBytes"/>.</summary>
    public long ProbeBytes => NewBytes + ProbeSlackBytes;

    /// <summary>The attach guard's verdict.</summary>
    /// <param name="Fits">Attach the head.</param>
    /// <param name="Reason">Numbers behind the verdict.</param>
    public readonly record struct Verdict(bool Fits, string Reason);

    /// <summary>
    /// Combines the arithmetic with a canary probe. The probe measures the real headroom of the moment, so when it ran it decides: a clean pass attaches
    /// even where the arithmetic (which has to assume the OS limit) would decline, and a failure refuses even where the arithmetic fits (the limit has
    /// drifted down, or something else took the memory). Without a conclusive probe the arithmetic decides.
    /// </summary>
    public Verdict Decide(HeadroomProbeResult? probe)
    {
        static string G(long b) => $"{b / (double)(1L << 30):F2} GiB";
        if (probe is { Outcome: HeadroomOutcome.Fits })
            return new(true, $"a probe allocated and touched {G(probe.Value.ReachedBytes)} on a second device (head {G(HeadBytes)} + context {G(ContextBytes)} + scratch {G(ScratchBytes)} + margin {G(MarginBytes + ProbeSlackBytes)}); arithmetic: {Describe()}");
        if (probe is { Outcome: HeadroomOutcome.DoesNotFit })
            return new(false, $"the driver ran out of memory after {G(probe.Value.ReachedBytes)} of the {G(probe.Value.RequestedBytes)} the attach needs (head {G(HeadBytes)} incl. verify snapshots + context growth {G(ContextBytes)} + scratch {G(ScratchBytes)} + margin) with {G(LiveBytes)} already resident and {G(OtherProcessBytes)} held by other processes ({probe.Value.Detail})");
        return new(Fits, Describe() + (probe is null ? "" : "; " + probe.Value.Describe()));
    }

    /// <summary>The itemised sum, for the refusal text and the attach trace.</summary>
    public string Describe()
    {
        static string G(long b) => $"{b / (double)(1L << 30):F2} GiB";
        return $"resident {G(LiveBytes)} + head and verify snapshots {G(HeadBytes)} + context growth to capacity {G(ContextBytes)} + head scratch {G(ScratchBytes)} + " +
               $"margin {G(MarginBytes)} + held by other processes {G(OtherProcessBytes)} = {G(NeededBytes)} against a {G(CapacityBytes)} resident budget " +
               (Fits ? $"(headroom {G(CapacityBytes - NeededBytes)})" : $"(short by {G(ShortfallBytes)})");
    }
}
