namespace DotLLM.Core.Moe;

/// <summary>Device bytes of one routed expert: its up and down matrices. Experts with equal sizes share a slot pool.</summary>
/// <param name="UpBytes">Bytes of the up (in) matrix.</param>
/// <param name="DownBytes">Bytes of the down (out) matrix.</param>
public readonly record struct ExpertLayout(long UpBytes, long DownBytes)
{
    /// <summary>Alignment of every slot (and of the down matrix inside it), in bytes.</summary>
    public const long SlotAlignment = 256;

    /// <summary>Offset of the down matrix inside a slot (the up bytes rounded up to <see cref="SlotAlignment"/>).</summary>
    public long DownOffset => AlignUp(UpBytes);

    /// <summary>Bytes one slot of this layout occupies (both matrices, aligned).</summary>
    public long SlotBytes => AlignUp(DownOffset + DownBytes);

    /// <summary>Rounds <paramref name="bytes"/> up to <see cref="SlotAlignment"/>.</summary>
    /// <param name="bytes">Byte count.</param>
    /// <returns>The aligned count.</returns>
    public static long AlignUp(long bytes) => (bytes + SlotAlignment - 1) / SlotAlignment * SlotAlignment;
}

/// <summary>
/// The device-memory arithmetic of expert offload (issue #1070, #1063 addendum "hard rule"): the cache budget is measured free
/// memory minus a reserve and minus everything else the model will ever allocate, the cache is allocated once from that budget,
/// and its slots are split across expert layouts in proportion to each layout's total expert bytes. Pure functions, unit-tested
/// on the CPU.
/// </summary>
public static class MoeOffloadPlanner
{
    /// <summary>One MiB.</summary>
    public const long MiB = 1024L * 1024;

    /// <summary>Default device reserve (#1063 addendum: 512 MiB on a display-attached GPU).</summary>
    public const long DefaultReserveBytes = 512 * MiB;

    /// <summary>Everything the device must hold besides the expert cache, by category.</summary>
    /// <param name="AlwaysResidentWeights">Weights that never leave the device (SSM, attention, router, latent, shared expert, head).</param>
    /// <param name="ForwardState">Forward activations at the planned token count plus the recurrent (SSM) state.</param>
    /// <param name="KvCache">KV cache at the maximum context.</param>
    /// <param name="Scratch">Worst-case scratch: MoE block scratch, GEMM F16 scratch, staging arena.</param>
    public readonly record struct Reservations(long AlwaysResidentWeights, long ForwardState, long KvCache, long Scratch)
    {
        /// <summary>Sum of every category.</summary>
        public long Total => AlwaysResidentWeights + ForwardState + KvCache + Scratch;
    }

    /// <summary>
    /// The expert-cache budget: <c>free − reserve − everything else</c>, never negative.
    /// </summary>
    /// <param name="freeBytes">Device free memory measured before any upload.</param>
    /// <param name="reserveBytes">Memory that must stay free after every allocation.</param>
    /// <param name="other">Everything else the model allocates.</param>
    /// <returns>Bytes available to the expert cache.</returns>
    public static long CacheBudgetBytes(long freeBytes, long reserveBytes, Reservations other)
        => Math.Max(0, freeBytes - reserveBytes - other.Total);

    /// <summary>
    /// The cache size to allocate: the request (MiB) clamped to <paramref name="budgetBytes"/>, or the whole budget when no size was
    /// requested. Also never more than <paramref name="maxUsefulBytes"/> (every expert cached).
    /// </summary>
    /// <param name="budgetBytes">From <see cref="CacheBudgetBytes"/>.</param>
    /// <param name="requestedMib">Requested size in MiB, or null for automatic.</param>
    /// <param name="maxUsefulBytes">Bytes that cache every expert (more is never used).</param>
    /// <param name="clamped">True when an explicit request exceeded the budget and was reduced.</param>
    /// <returns>Bytes to allocate for cache slots.</returns>
    public static long ResolveCacheBytes(long budgetBytes, long? requestedMib, long maxUsefulBytes, out bool clamped)
    {
        long want = requestedMib is { } mib ? mib * MiB : budgetBytes;
        clamped = requestedMib is not null && want > budgetBytes;
        return Math.Max(0, Math.Min(Math.Min(want, budgetBytes), maxUsefulBytes));
    }

    /// <summary>
    /// Splits <paramref name="cacheBytes"/> into slots per layout, proportional to each layout's total expert bytes (so every layout
    /// caches the same fraction of its experts), capped at the layout's expert count. Bytes left by rounding go one slot at a time
    /// to the layout with the lowest cached fraction that still fits.
    /// </summary>
    /// <param name="slotBytes">Bytes of one slot per layout (<see cref="ExpertLayout.SlotBytes"/>).</param>
    /// <param name="expertCount">Experts with each layout, over all layers.</param>
    /// <param name="cacheBytes">Bytes to split.</param>
    /// <returns>Slots per layout; their total bytes never exceed <paramref name="cacheBytes"/>.</returns>
    /// <exception cref="ArgumentException">The spans differ in length or hold a non-positive slot size.</exception>
    public static int[] SplitSlots(ReadOnlySpan<long> slotBytes, ReadOnlySpan<int> expertCount, long cacheBytes)
    {
        if (slotBytes.Length != expertCount.Length) throw new ArgumentException("one slot size per layout is required.", nameof(slotBytes));
        int n = slotBytes.Length;
        var slots = new int[n];
        long total = 0;
        for (int i = 0; i < n; i++)
        {
            if (slotBytes[i] <= 0) throw new ArgumentException("slot sizes must be positive.", nameof(slotBytes));
            ArgumentOutOfRangeException.ThrowIfNegative(expertCount[i], nameof(expertCount));
            total += slotBytes[i] * expertCount[i];
        }
        if (total == 0 || cacheBytes <= 0) return slots;

        // Same fraction f of every layout's experts: slots_i = floor(f * count_i), f = cache / total.
        double f = Math.Min(1.0, (double)cacheBytes / total);
        long used = 0;
        for (int i = 0; i < n; i++)
        {
            slots[i] = (int)Math.Min(expertCount[i], Math.Floor(f * expertCount[i]));
            used += slots[i] * slotBytes[i];
        }
        while (used > cacheBytes)   // floating-point guard; normally never taken
        {
            int i = Array.FindIndex(slots, s => s > 0);
            slots[i]--;
            used -= slotBytes[i];
        }

        // Rounding remainder: one slot at a time to the layout with the lowest cached fraction whose slot still fits.
        while (true)
        {
            int best = -1;
            double bestFrac = double.MaxValue;
            for (int i = 0; i < n; i++)
            {
                if (slots[i] >= expertCount[i] || used + slotBytes[i] > cacheBytes) continue;
                double frac = (double)slots[i] / expertCount[i];
                if (frac < bestFrac) { bestFrac = frac; best = i; }
            }
            if (best < 0) return slots;
            slots[best]++;
            used += slotBytes[best];
        }
    }

    /// <summary>
    /// Whether a sampled read over a candidate allocation is a sign of WDDM spill: its bandwidth is below
    /// <c>1 / <paramref name="maxSlowdown"/></c> of a reference read over a small resident buffer.
    /// </summary>
    /// <param name="poolBytesPerSecond">Measured bandwidth over the pool.</param>
    /// <param name="referenceBytesPerSecond">Measured bandwidth over the reference buffer.</param>
    /// <param name="maxSlowdown">Tolerated slowdown (the addendum's 5×).</param>
    /// <returns>True when the pool looks spilled.</returns>
    public static bool LooksSpilled(double poolBytesPerSecond, double referenceBytesPerSecond, double maxSlowdown = 5.0)
        => referenceBytesPerSecond > 0 && poolBytesPerSecond * maxSlowdown < referenceBytesPerSecond;

    /// <summary>Whether the offloaded banks may be held as one pinned host copy (else they stream through a staging ring).</summary>
    /// <param name="offloadedBytes">Total bytes of the offloaded expert banks.</param>
    /// <param name="physicalRamBytes">Physical RAM of the host.</param>
    /// <param name="pinBudgetFraction">Allowed share of RAM (<see cref="MoeOffloadOptions.PinBudgetFraction"/>).</param>
    /// <returns>True to pin the whole store.</returns>
    public static bool FitsPinBudget(long offloadedBytes, long physicalRamBytes, double pinBudgetFraction)
        => physicalRamBytes > 0 && offloadedBytes <= physicalRamBytes * pinBudgetFraction;
}
