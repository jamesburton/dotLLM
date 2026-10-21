namespace DotLLM.Core.Moe;

/// <summary>Outcome of one expert in <see cref="ExpertCache.Lookup"/>.</summary>
public enum ExpertLookupResult : byte
{
    /// <summary>Resident: its slot already holds the expert's weights.</summary>
    Hit,

    /// <summary>Miss admitted to the cache: the caller must copy the expert into the returned slot before using it.</summary>
    Inserted,

    /// <summary>
    /// Miss not admitted: every slot of the pool is pinned by this call. The caller streams the expert through scratch memory
    /// and the cache does not change (prefill of a block that touches more experts than the pool holds).
    /// </summary>
    Bypass,

    /// <summary>
    /// Miss left uncached by a non-admitting lookup (#1071): the expert runs where its weights already live (the CPU), and enters
    /// the cache only if <see cref="ExpertCache.Admit"/> or <see cref="ExpertCache.Promote"/> later places it.
    /// </summary>
    Cold,
}

/// <summary>
/// Bookkeeping of a device expert cache (issue #1070, design #1063 section 8b): fixed slot pools, each shared by every layer
/// whose experts have that pool's layout (llama.cpp #30112), keyed by (layer, expert), with an online replacement policy
/// (<see cref="ExpertCachePolicy"/>). Pure C# (no device memory, no I/O): the backend owns the slot memory and copies weights
/// in when <see cref="Lookup"/> reports <see cref="ExpertLookupResult.Inserted"/>.
/// </summary>
/// <remarks>
/// <para>
/// One <see cref="Lookup"/> resolves every expert one layer call needs. Each of them is <i>pinned</i> for the rest of the call: a
/// miss never evicts an expert the same call is about to use, so every slot handed back stays valid until the next call.
/// When a pool has no unpinned slot left, further misses of the call are <see cref="ExpertLookupResult.Bypass"/>.
/// </para>
/// <para>Counters are per expert per call: a prefill block that routes 30 tokens to one expert counts one hit or one miss.</para>
/// <para>
/// Prior art: <c>DotLLM.Vulkan.Tiering.ExpertSlotCache</c> (#939, per-layer LRU). TODO(#1070 follow-up): move the Vulkan
/// qwen4exp tiering onto this type once its tests can run on the Vulkan box.
/// </para>
/// </remarks>
public sealed class ExpertCache
{
    private readonly int[] _layerOffset;   // layer -> first global key
    private readonly int[] _layerExperts;  // layer -> expert count
    private readonly int[] _poolOfLayer;   // layer -> pool
    private readonly int[] _keyLayer;      // global key -> layer
    private readonly SlotPool[] _pools;
    private readonly long[] _keyStamp;     // global key -> last call that named it (duplicate detection)
    private long _call;
    private int _callLayer = -1;           // layer of the current call (Admit / Promote act on it)

    /// <summary>Creates an empty cache.</summary>
    /// <param name="expertsPerLayer">Experts of each layer (0 for a layer without experts).</param>
    /// <param name="poolOfLayer">Pool index of each layer (ignored for a layer without experts).</param>
    /// <param name="slotsPerPool">Slots of each pool (0 is allowed: every miss of that pool bypasses).</param>
    /// <param name="policy">Replacement policy, the same for every pool.</param>
    /// <param name="protectedFraction">SLRU only: share of each pool's slots the protected segment may hold.</param>
    /// <exception cref="ArgumentException">The arrays disagree in length, or a pool index is out of range.</exception>
    public ExpertCache(IReadOnlyList<int> expertsPerLayer, IReadOnlyList<int> poolOfLayer, IReadOnlyList<int> slotsPerPool,
        ExpertCachePolicy policy, double protectedFraction = 0.8)
    {
        ArgumentNullException.ThrowIfNull(expertsPerLayer);
        ArgumentNullException.ThrowIfNull(poolOfLayer);
        ArgumentNullException.ThrowIfNull(slotsPerPool);
        if (poolOfLayer.Count != expertsPerLayer.Count)
            throw new ArgumentException("poolOfLayer must have one entry per layer.", nameof(poolOfLayer));
        if (protectedFraction is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(protectedFraction), "must be in [0, 1].");

        int layers = expertsPerLayer.Count;
        _layerOffset = new int[layers];
        _layerExperts = new int[layers];
        _poolOfLayer = new int[layers];
        int keys = 0;
        for (int l = 0; l < layers; l++)
        {
            int e = expertsPerLayer[l];
            ArgumentOutOfRangeException.ThrowIfNegative(e, nameof(expertsPerLayer));
            _layerOffset[l] = keys;
            _layerExperts[l] = e;
            _poolOfLayer[l] = -1;
            if (e > 0)
            {
                int p = poolOfLayer[l];
                if ((uint)p >= (uint)slotsPerPool.Count)
                    throw new ArgumentException($"layer {l} names pool {p}, outside [0, {slotsPerPool.Count}).", nameof(poolOfLayer));
                _poolOfLayer[l] = p;
            }
            keys = checked(keys + e);
        }

        _keyLayer = new int[keys];
        for (int l = 0; l < layers; l++)
            Array.Fill(_keyLayer, l, _layerOffset[l], _layerExperts[l]);
        _keyStamp = new long[keys];

        _pools = new SlotPool[slotsPerPool.Count];
        for (int p = 0; p < _pools.Length; p++)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(slotsPerPool[p], nameof(slotsPerPool));
            _pools[p] = new SlotPool(slotsPerPool[p], keys, policy, protectedFraction);
        }
        Policy = policy;
    }

    /// <summary>The replacement policy.</summary>
    public ExpertCachePolicy Policy { get; }

    /// <summary>Number of layers (including layers without experts).</summary>
    public int Layers => _layerOffset.Length;

    /// <summary>Number of slot pools.</summary>
    public int Pools => _pools.Length;

    /// <summary>Slots of all pools together.</summary>
    public int TotalSlots
    {
        get
        {
            int n = 0;
            foreach (var p in _pools) n += p.Slots;
            return n;
        }
    }

    /// <summary>Lookups (expert, call) that found the expert resident.</summary>
    public long Hits { get; private set; }

    /// <summary>Lookups that did not (<see cref="Insertions"/> + <see cref="Bypasses"/>).</summary>
    public long Misses { get; private set; }

    /// <summary>Misses that took a slot from a resident expert (the rest took a free slot or bypassed).</summary>
    public long Evictions { get; private set; }

    /// <summary>Misses admitted to a slot.</summary>
    public long Insertions { get; private set; }

    /// <summary>Misses that found every slot of their pool pinned by the same call.</summary>
    public long Bypasses { get; private set; }

    /// <summary>Misses a non-admitting lookup left uncached (computed elsewhere, #1071) and that were not admitted afterwards.</summary>
    public long ColdMisses { get; private set; }

    /// <summary>Cold misses copied into a chosen slot by promote-only admission (<see cref="Promote"/>).</summary>
    public long Promotions { get; private set; }

    /// <summary>Number of distinct (layer, expert) keys.</summary>
    public int KeyCount => _keyLayer.Length;

    /// <summary>Dense key of (<paramref name="layer"/>, <paramref name="expert"/>), in <c>[0, <see cref="KeyCount"/>)</c>.</summary>
    /// <param name="layer">Layer index.</param>
    /// <param name="expert">Expert index within the layer.</param>
    /// <returns>The key.</returns>
    public int KeyOf(int layer, int expert) => Key(layer, expert);

    /// <summary>True when <paramref name="slot"/> of <paramref name="pool"/> is pinned by the current call (it must not be replaced).</summary>
    /// <param name="pool">Pool index.</param>
    /// <param name="slot">Slot index.</param>
    /// <returns>True when pinned.</returns>
    public bool IsPinned(int pool, int slot) => _pools[pool].IsPinned(slot, _call);

    /// <summary>Pool that serves <paramref name="layer"/>, or -1 when the layer has no experts.</summary>
    /// <param name="layer">Layer index.</param>
    /// <returns>The pool index.</returns>
    public int PoolOf(int layer) => _poolOfLayer[layer];

    /// <summary>Experts of <paramref name="layer"/> (0 for a layer without experts).</summary>
    /// <param name="layer">Layer index.</param>
    /// <returns>The expert count.</returns>
    public int ExpertsIn(int layer) => _layerExperts[layer];

    /// <summary>Slots of pool <paramref name="pool"/>.</summary>
    /// <param name="pool">Pool index.</param>
    /// <returns>The slot count.</returns>
    public int SlotsIn(int pool) => _pools[pool].Slots;

    /// <summary>Slot holding <paramref name="expert"/> of <paramref name="layer"/>, or -1 when it is not resident.</summary>
    /// <param name="layer">Layer index.</param>
    /// <param name="expert">Expert index within the layer.</param>
    /// <returns>The slot, or -1.</returns>
    public int SlotOf(int layer, int expert)
    {
        int p = _poolOfLayer[layer];
        return p < 0 ? -1 : _pools[p].SlotOfKey(Key(layer, expert));
    }

    /// <summary>The (layer, expert) held by <paramref name="slot"/> of <paramref name="pool"/>, or (-1, -1) when the slot is free.</summary>
    /// <param name="pool">Pool index.</param>
    /// <param name="slot">Slot index.</param>
    /// <returns>The resident expert.</returns>
    public (int Layer, int Expert) ExpertAt(int pool, int slot)
    {
        int key = _pools[pool].KeyOfSlot(slot);
        if (key < 0) return (-1, -1);
        int layer = _keyLayer[key];
        return (layer, key - _layerOffset[layer]);
    }

    /// <summary>
    /// Resolves the experts one call of <paramref name="layer"/> needs. Resident experts are touched first (in the given order)
    /// and pinned; then each absent expert takes a free slot or the policy's victim among the unpinned slots, or bypasses when
    /// every slot is pinned.
    /// </summary>
    /// <param name="layer">Layer index (must have experts).</param>
    /// <param name="experts">Distinct expert ids of the call.</param>
    /// <param name="slots">Receives each expert's slot (-1 for a bypass); at least as long as <paramref name="experts"/>.</param>
    /// <param name="results">Receives each expert's outcome; at least as long as <paramref name="experts"/>.</param>
    /// <returns>Number of misses (insertions, bypasses and cold misses).</returns>
    /// <exception cref="ArgumentException">An id is out of range or repeated, or an output span is too short.</exception>
    /// <param name="admitMisses">True (copy-on-miss, #1070) = place every miss now; false (#1071) = leave the misses
    /// <see cref="ExpertLookupResult.Cold"/>: the caller computes them where their weights live, and may then place some of them
    /// with <see cref="Admit"/> or <see cref="Promote"/> before the next call.</param>
    public int Lookup(int layer, ReadOnlySpan<int> experts, Span<int> slots, Span<ExpertLookupResult> results, bool admitMisses = true)
    {
        if (slots.Length < experts.Length) throw new ArgumentException("slots is shorter than experts.", nameof(slots));
        if (results.Length < experts.Length) throw new ArgumentException("results is shorter than experts.", nameof(results));
        int p = _poolOfLayer[layer];
        if (p < 0) throw new ArgumentException($"layer {layer} has no experts.", nameof(layer));
        var pool = _pools[p];
        long call = ++_call;
        _callLayer = layer;
        int n = _layerExperts[layer];

        // Pass 1: validate, then touch and pin every resident expert, so no eviction in pass 2 can take a slot this call uses.
        for (int i = 0; i < experts.Length; i++)
        {
            int e = experts[i];
            if ((uint)e >= (uint)n) throw new ArgumentException($"expert id {e} outside [0, {n}) for layer {layer}.", nameof(experts));
            int key = Key(layer, e);
            if (_keyStamp[key] == call) throw new ArgumentException($"expert id {e} appears twice in one lookup.", nameof(experts));
            _keyStamp[key] = call;
            int s = pool.SlotOfKey(key);
            slots[i] = s;
            if (s < 0) continue;
            pool.Touch(s, call);
            results[i] = ExpertLookupResult.Hit;
            Hits++;
        }

        // Pass 2: place the absent experts.
        int misses = 0;
        for (int i = 0; i < experts.Length; i++)
        {
            if (slots[i] >= 0) continue;
            misses++;
            Misses++;
            if (!admitMisses)
            {
                results[i] = ExpertLookupResult.Cold;
                ColdMisses++;
                continue;
            }
            int s = pool.Admit(Key(layer, experts[i]), call, out bool evicted);
            slots[i] = s;
            if (s < 0)
            {
                results[i] = ExpertLookupResult.Bypass;
                Bypasses++;
                continue;
            }
            results[i] = ExpertLookupResult.Inserted;
            Insertions++;
            if (evicted) Evictions++;
        }
        return misses;
    }

    /// <summary>
    /// Places a cold miss of the current call (the last non-admitting lookup) by the replacement policy, exactly as an admitting
    /// lookup would have: a free slot or the policy's unpinned victim, pinned for the rest of the call; -1 (a bypass) when every
    /// slot is pinned. The caller copies the expert in.
    /// </summary>
    /// <param name="layer">Layer of the current call.</param>
    /// <param name="expert">An expert the current call left cold.</param>
    /// <param name="evicted">True when the slot held another expert.</param>
    /// <returns>The slot, or -1.</returns>
    /// <exception cref="InvalidOperationException">The expert is not a cold miss of the current call.</exception>
    public int Admit(int layer, int expert, out bool evicted)
    {
        var pool = ColdMissPool(layer, expert, out int key);
        ColdMisses--;
        int s = pool.Admit(key, _call, out evicted);
        if (s < 0)
        {
            Bypasses++;
            return -1;
        }
        Insertions++;
        if (evicted) Evictions++;
        return s;
    }

    /// <summary>
    /// Promote-only admission (#1071): copies a cold miss of the current call into <paramref name="slot"/>, chosen by the caller
    /// (the weakest resident by its own score), evicting whatever the slot holds. The slot is pinned for the rest of the call. The
    /// miss stays counted as cold (it was computed elsewhere this call); <see cref="Promotions"/> counts the copy.
    /// </summary>
    /// <param name="layer">Layer of the current call.</param>
    /// <param name="expert">An expert the current call left cold.</param>
    /// <param name="slot">Target slot of the layer's pool; must not be pinned by the current call.</param>
    /// <param name="evicted">True when the slot held another expert.</param>
    /// <exception cref="InvalidOperationException">The expert is not a cold miss of the call, or the slot is pinned.</exception>
    public void Promote(int layer, int expert, int slot, out bool evicted)
    {
        var pool = ColdMissPool(layer, expert, out int key);
        if (pool.IsPinned(slot, _call))
            throw new InvalidOperationException($"slot {slot} is pinned by the current call and cannot take a promotion.");
        pool.PlaceAt(key, slot, _call, out evicted);
        Promotions++;
        if (evicted) Evictions++;
    }

    private SlotPool ColdMissPool(int layer, int expert, out int key)
    {
        if (layer != _callLayer)
            throw new InvalidOperationException($"layer {layer} is not the current call's layer ({_callLayer}).");
        key = Key(layer, expert);
        var pool = _pools[_poolOfLayer[layer]];
        if (_keyStamp[key] != _call || pool.SlotOfKey(key) >= 0)
            throw new InvalidOperationException($"expert {expert} of layer {layer} is not a cold miss of the current call.");
        return pool;
    }

    /// <summary>Forgets every resident expert (slot contents are no longer valid). Counters are kept.</summary>
    public void Clear()
    {
        foreach (var p in _pools) p.Clear();
    }

    /// <summary>Zeroes every counter.</summary>
    public void ResetCounters() => Hits = Misses = Evictions = Insertions = Bypasses = ColdMisses = Promotions = 0;

    private int Key(int layer, int expert) => _layerOffset[layer] + expert;

    /// <summary>One pool: slot table plus the policy's ordering (intrusive lists over slots).</summary>
    private sealed class SlotPool
    {
        private const int Probation = 0, Protected = 1;

        private readonly ExpertCachePolicy _policy;
        private readonly int[] _slotOfKey;   // global key -> slot, -1 when absent (only keys of this pool's layers are ever set)
        private readonly int[] _keyOfSlot;   // slot -> key, -1 when free
        private readonly int[] _prev, _next; // list links per slot
        private readonly byte[] _segment;    // SLRU segment of each slot (LRU/LFU: always Probation)
        private readonly long[] _pinStamp;   // last call that pinned the slot
        private readonly long[] _freq;       // LFU: lookups since insertion (-1 = free)
        private readonly long[] _lastUse;    // LFU tie-break: last touching call
        private readonly int[] _head = [-1, -1], _tail = [-1, -1], _count = [0, 0];
        private readonly int _protectedCapacity;

        public SlotPool(int slots, int keys, ExpertCachePolicy policy, double protectedFraction)
        {
            _policy = policy;
            _slotOfKey = new int[keys];
            Array.Fill(_slotOfKey, -1);
            _keyOfSlot = new int[slots];
            _prev = new int[slots];
            _next = new int[slots];
            _segment = new byte[slots];
            _pinStamp = new long[slots];
            _freq = new long[slots];
            _lastUse = new long[slots];
            _protectedCapacity = policy == ExpertCachePolicy.Slru ? (int)Math.Floor(slots * protectedFraction) : 0;
            Clear();
        }

        public int Slots => _keyOfSlot.Length;

        public int SlotOfKey(int key) => _slotOfKey[key];

        public int KeyOfSlot(int slot) => _keyOfSlot[slot];

        public bool IsPinned(int slot, long call) => _pinStamp[slot] == call;

        public void Clear()
        {
            Array.Fill(_slotOfKey, -1);
            Array.Fill(_keyOfSlot, -1);
            Array.Fill(_freq, -1);
            Array.Clear(_lastUse);
            Array.Clear(_pinStamp);
            _head[0] = _head[1] = _tail[0] = _tail[1] = -1;
            _count[0] = _count[1] = 0;
            // Free slots start in probation in index order, so a cold cache fills slot 0, 1, 2, ... (deterministic for tests).
            for (int s = 0; s < Slots; s++) PushTail(Probation, s);
        }

        /// <summary>Hit on <paramref name="s"/>: pin it and update the policy order.</summary>
        public void Touch(int s, long call)
        {
            _pinStamp[s] = call;
            _lastUse[s] = call;
            _freq[s]++;
            switch (_policy)
            {
                case ExpertCachePolicy.Lru:
                    Unlink(s);
                    PushTail(Probation, s);
                    break;
                case ExpertCachePolicy.Slru:
                    Unlink(s);
                    PushTail(Protected, s);
                    // Over capacity: the protected segment's least recent entry drops back to probation as its most recent.
                    while (_count[Protected] > _protectedCapacity)
                    {
                        int d = _head[Protected];
                        Unlink(d);
                        PushTail(Probation, d);
                    }
                    break;
                case ExpertCachePolicy.Lfu:
                    break; // frequency and recency are per-slot fields; no list order
            }
        }

        /// <summary>Places <paramref name="key"/> in a free slot or the policy's unpinned victim; -1 when every slot is pinned.</summary>
        public int Admit(int key, long call, out bool evicted)
        {
            evicted = false;
            int victim = _policy == ExpertCachePolicy.Lfu ? LfuVictim(call) : ListVictim(call);
            if (victim < 0) return -1;

            int old = _keyOfSlot[victim];
            if (old >= 0)
            {
                _slotOfKey[old] = -1;
                evicted = true;
            }
            _keyOfSlot[victim] = key;
            _slotOfKey[key] = victim;
            _pinStamp[victim] = call;
            _lastUse[victim] = call;
            _freq[victim] = 1;
            if (_policy != ExpertCachePolicy.Lfu)
            {
                // New arrivals enter at the most recent end of probation (LRU has only that one list).
                Unlink(victim);
                PushTail(Probation, victim);
            }
            return victim;
        }

        /// <summary>Places <paramref name="key"/> in <paramref name="slot"/> (chosen by the caller), as a new arrival.</summary>
        public void PlaceAt(int key, int slot, long call, out bool evicted)
        {
            int old = _keyOfSlot[slot];
            evicted = old >= 0;
            if (evicted) _slotOfKey[old] = -1;
            _keyOfSlot[slot] = key;
            _slotOfKey[key] = slot;
            _pinStamp[slot] = call;
            _lastUse[slot] = call;
            _freq[slot] = 1;
            if (_policy != ExpertCachePolicy.Lfu)
            {
                Unlink(slot);
                PushTail(Probation, slot);
            }
        }

        /// <summary>LRU / SLRU victim: the least recent unpinned slot of probation, then of protected. Free slots sort first.</summary>
        private int ListVictim(long call)
        {
            for (int seg = Probation; seg <= Protected; seg++)
                for (int s = _head[seg]; s >= 0; s = _next[s])
                    if (_pinStamp[s] != call) return s;
            return -1;
        }

        /// <summary>LFU victim: a free slot (lowest index), else the unpinned slot with the fewest lookups, ties to the least recent.</summary>
        private int LfuVictim(long call)
        {
            int best = -1;
            for (int s = 0; s < Slots; s++)
            {
                if (_pinStamp[s] == call) continue;
                if (best < 0 || _freq[s] < _freq[best] || (_freq[s] == _freq[best] && _lastUse[s] < _lastUse[best]))
                    best = s;
            }
            return best;
        }

        private void Unlink(int s)
        {
            int seg = _segment[s];
            int p = _prev[s], n = _next[s];
            if (p >= 0) _next[p] = n; else _head[seg] = n;
            if (n >= 0) _prev[n] = p; else _tail[seg] = p;
            _count[seg]--;
        }

        private void PushTail(int seg, int s)
        {
            _segment[s] = (byte)seg;
            _prev[s] = _tail[seg];
            _next[s] = -1;
            if (_tail[seg] >= 0) _next[_tail[seg]] = s; else _head[seg] = s;
            _tail[seg] = s;
            _count[seg]++;
        }
    }
}
