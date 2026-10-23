namespace DotLLM.Vulkan.Tiering;

/// <summary>
/// Per-layer slot allocator for tiered routed experts (#939): <c>slots</c> device slots serve <c>numExperts</c> experts, evicting the least recently used
/// expert on a miss. Pure bookkeeping (no GPU, no I/O) so the policy is unit-testable on its own.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Lookup"/> resolves one routing decision (the ids a token selected in this layer). Every id the call needs is <i>pinned</i> for the duration of the
/// call: a miss never evicts an expert the same call is about to use, so the slots it hands back are all valid at once. That needs <c>slots &gt;= distinct ids</c>
/// per call, which the constructor cannot know - <see cref="Lookup"/> throws if a call needs more distinct experts than there are slots.
/// </para>
/// <para>Policy per the stage-1 traces (#938): LRU matches LFU-decay within 1 point, so the simpler LRU is used; the slot count is uniform across layers.</para>
/// </remarks>
internal sealed class ExpertSlotCache
{
    private readonly int[] _slotOf;      // expert -> slot, -1 when absent
    private readonly int[] _expertOf;    // slot -> expert, -1 when free
    private readonly int[] _prev, _next; // intrusive LRU list over slots; head = least recently used
    private readonly long[] _pinStamp;   // slot -> id of the Lookup call that pinned it
    private int _head = -1, _tail = -1;
    private int _freeCount;
    private long _call;

    /// <summary>Experts the layer has.</summary>
    public int NumExperts => _slotOf.Length;

    /// <summary>Slots available.</summary>
    public int Slots => _expertOf.Length;

    /// <summary>Lookups that found their expert resident.</summary>
    public long Hits { get; private set; }

    /// <summary>Lookups that had to load their expert.</summary>
    public long Misses { get; private set; }

    /// <summary>Misses that displaced a resident expert (the rest took a free slot).</summary>
    public long Evictions { get; private set; }

    /// <summary>Creates an empty cache.</summary>
    /// <param name="numExperts">Experts in the layer.</param>
    /// <param name="slots">Slot count (1..<paramref name="numExperts"/>).</param>
    public ExpertSlotCache(int numExperts, int slots)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(numExperts, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(slots, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(slots, numExperts);
        _slotOf = new int[numExperts]; Array.Fill(_slotOf, -1);
        _expertOf = new int[slots]; Array.Fill(_expertOf, -1);
        _prev = new int[slots]; _next = new int[slots];
        _pinStamp = new long[slots];
        _freeCount = slots;
        // Free slots form the initial LRU list in index order so a cold cache fills slot 0, 1, 2, ... (deterministic for tests).
        for (int s = 0; s < slots; s++) PushTail(s);
    }

    /// <summary>Slot currently holding <paramref name="expert"/>, or -1.</summary>
    public int SlotOf(int expert) => _slotOf[expert];

    /// <summary>Expert currently held by <paramref name="slot"/>, or -1 when the slot is free.</summary>
    public int ExpertOf(int slot) => _expertOf[slot];

    /// <summary>Number of experts currently resident.</summary>
    public int Resident => Slots - _freeCount;

    /// <summary>
    /// Resolves <paramref name="ids"/> to slots. Resident experts are touched (most recently used); each absent expert is assigned a free slot or the
    /// least recently used slot not pinned by this call and reported as a miss for the caller to fill.
    /// </summary>
    /// <param name="ids">Expert ids this routing decision needs (duplicates allowed, they resolve to one slot).</param>
    /// <param name="slotIds">Receives the slot of each id (same length as <paramref name="ids"/> or longer).</param>
    /// <param name="missExperts">Receives the experts that must be loaded (capacity at least the number of distinct ids).</param>
    /// <param name="missSlots">Receives the destination slot of each miss, parallel to <paramref name="missExperts"/>.</param>
    /// <returns>Number of misses written.</returns>
    public int Lookup(ReadOnlySpan<int> ids, Span<int> slotIds, Span<int> missExperts, Span<int> missSlots)
    {
        if (slotIds.Length < ids.Length) throw new ArgumentException("slotIds is shorter than ids.", nameof(slotIds));
        long call = ++_call;
        // Pass 1: touch and pin every resident expert so no eviction in pass 2 can take a slot this call is about to use.
        foreach (int e in ids)
        {
            if ((uint)e >= (uint)NumExperts) throw new ArgumentOutOfRangeException(nameof(ids), $"expert id {e} outside [0, {NumExperts}).");
            int s = _slotOf[e];
            if (s < 0) continue;
            _pinStamp[s] = call;
            Unlink(s); PushTail(s);
        }
        // Pass 2: assign slots to the absent experts.
        int misses = 0;
        for (int i = 0; i < ids.Length; i++)
        {
            int e = ids[i];
            int s = _slotOf[e];
            if (s >= 0)
            {
                slotIds[i] = s;
                // A pass-2 slot (assigned a miss earlier in this call) is also a repeat of a duplicate id: not a hit and not a new miss.
                if (_pinStamp[s] == call && IsPendingMiss(missSlots, misses, s)) continue;
                Hits++;
                continue;
            }
            int victim = _head;
            while (victim >= 0 && _pinStamp[victim] == call) victim = _next[victim];
            if (victim < 0) throw new InvalidOperationException($"a routing decision needs more than {Slots} distinct experts but the layer has only {Slots} slots.");
            if (_expertOf[victim] >= 0) { _slotOf[_expertOf[victim]] = -1; Evictions++; } else _freeCount--;
            _expertOf[victim] = e; _slotOf[e] = victim;
            _pinStamp[victim] = call;
            Unlink(victim); PushTail(victim);
            missExperts[misses] = e; missSlots[misses] = victim; misses++;
            slotIds[i] = victim;
            Misses++;
        }
        return misses;
    }

    private static bool IsPendingMiss(ReadOnlySpan<int> missSlots, int count, int slot)
    {
        for (int i = 0; i < count; i++) if (missSlots[i] == slot) return true;
        return false;
    }

    /// <summary>Forgets every resident expert (the slot contents are no longer valid). Counters are kept.</summary>
    public void Clear()
    {
        Array.Fill(_slotOf, -1); Array.Fill(_expertOf, -1);
        _head = _tail = -1; _freeCount = Slots;
        for (int s = 0; s < Slots; s++) PushTail(s);
    }

    /// <summary>Zeroes the hit / miss / eviction counters.</summary>
    public void ResetCounters() { Hits = Misses = Evictions = 0; }

    private void Unlink(int s)
    {
        int p = _prev[s], n = _next[s];
        if (p >= 0) _next[p] = n; else _head = n;
        if (n >= 0) _prev[n] = p; else _tail = p;
    }

    private void PushTail(int s)
    {
        _prev[s] = _tail; _next[s] = -1;
        if (_tail >= 0) _next[_tail] = s; else _head = s;
        _tail = s;
    }
}
