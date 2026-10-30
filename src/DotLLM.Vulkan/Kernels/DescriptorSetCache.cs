using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Small lookup-by-buffer-handles cache of populated descriptor sets. One
/// instance per kernel — trades a little host-side memory for eliminating
/// <c>vkAllocateDescriptorSets</c> + <c>vkUpdateDescriptorSets</c> on the
/// hot forward path whenever a kernel is called with the same buffer set
/// as a previous call.
/// </summary>
/// <remarks>
/// <para>
/// The cache keys on the descriptor set's buffer handles — not on any
/// push-constant values, because the Vulkan spec lets the same set be
/// rebound under different push constants, and the kernel re-issues
/// <c>vkCmdPushConstants</c> per call anyway. Within one forward pass
/// every kernel is called with many distinct buffer tuples (one per
/// layer × one per projection), but across forwards the tuples repeat —
/// weights are fixed, activation scratch is fixed, only the token sequence
/// changes. So the cache warms up in the first forward and stays warm for
/// the life of the model.
/// </para>
/// <para>
/// Structure is a linear probe over a fixed-capacity array of slots. With
/// <c>Capacity = 256</c> we comfortably cover SmolLM-135M's 211 matmul
/// descriptor variants per forward and still fit larger models; the
/// linear scan cost is trivial compared to a single
/// <c>vkAllocateDescriptorSets</c> round-trip. On overflow the cache
/// resets its own backing pool and drops every entry — this is a slow
/// path that only hits when a caller runs more than <c>Capacity</c>
/// distinct buffer tuples per kernel.
/// </para>
/// <para>
/// Raw handles are only a safe key while the buffer lives: drivers recycle <c>VkBuffer</c>
/// handle values, so a buffer destroyed and re-created (a per-request KV cache is the common
/// case) can come back under a handle that still has an entry. Entries naming a destroyed
/// buffer are therefore evicted via the device's destroy log before every lookup — see
/// <see cref="VulkanDevice.BufferDestroyEpoch"/> and issue #467.
/// </para>
/// </remarks>
internal sealed class DescriptorSetCache
{
    /// <summary>
    /// Fixed cache slot count. MUST equal the pool's <see cref="KernelSupport.DefaultMaxSetsPerPool"/>
    /// so the cache can hold every set the pool can allocate. Previously 256 while the pool held
    /// 1024 — a model with &gt;256 distinct buffer tuples for one kernel (e.g. gemma4-26B: 30 layers ×
    /// many projections) overflowed the cache MID-COMMAND-BUFFER, triggering <see cref="Reset"/>
    /// (vkResetDescriptorPool) which freed descriptor sets still referenced by already-recorded
    /// dispatches — silent cross-layer corruption that a pipeline barrier could not fix (only a full
    /// submit between layers, which drains the queue). Keeping the two equal makes overflow unreachable
    /// until a model genuinely needs more sets than the pool holds, at which point GetOrCreate fails loudly.
    /// </summary>
    internal const int Capacity = (int)KernelSupport.DefaultMaxSetsPerPool;

    /// <summary>Hard upper bound on buffers per descriptor set — matches the widest kernel (mamba3 canonical SSD MIMO scan, 13 bindings: state, v, qRoped, kRoped, qkPreDotSum, scale, gamma, adt, d, z, mimoZ, mimoO, y).</summary>
    private const int MaxBuffersPerSet = 13;

    private readonly VulkanDevice _device;
    private readonly nint _pool;
    private readonly nint _setLayout;
    private readonly int _buffersPerSet;
    private readonly uint _writesMask;

    // Parallel arrays indexed by slot. _keys[i] holds MaxBuffersPerSet nints;
    // unused trailing slots are zero. _sets[i] is the descriptor set handle
    // populated for that key; 0 means empty.
    private readonly nint[] _keys;
    private readonly nint[] _sets;
    private int _count;

    // Descriptor sets whose entries were evicted because one of their buffers was destroyed
    // (issue #467). Still allocated from _pool, so they are rewritten on the next miss rather
    // than allocating afresh — without reuse, per-request KV caches would exhaust the pool.
    private readonly nint[] _freeSets;
    private int _freeCount;

    // The device's BufferDestroyEpoch this cache has evicted up to; see EvictDestroyed.
    private long _seenDestroyEpoch;
    private HashSet<nint>? _destroyedScratch;

    /// <summary>
    /// Builds the cache against <paramref name="pipeline"/>'s descriptor-set
    /// layout. The pipeline also supplies its SPIR-V-reflected storage-buffer
    /// writes mask (<see cref="ComputePipeline.StorageWritesMask"/>) — this is
    /// the single choke point every kernel dispatch passes through, so
    /// <see cref="GetOrCreate"/> declares the dispatch's read/write buffer
    /// set to the device's active <see cref="VulkanHazardTracker"/> (issue
    /// #144) right before the caller records the dispatch.
    /// </summary>
    public DescriptorSetCache(VulkanDevice device, nint pool, ComputePipeline pipeline, int buffersPerSet)
    {
        if (buffersPerSet <= 0 || buffersPerSet > MaxBuffersPerSet)
            throw new ArgumentOutOfRangeException(nameof(buffersPerSet));
        _device = device;
        _pool = pool;
        _setLayout = pipeline.DescriptorSetLayout;
        _writesMask = pipeline.StorageWritesMask;
        _buffersPerSet = buffersPerSet;
        _keys = new nint[Capacity * MaxBuffersPerSet];
        _sets = new nint[Capacity];
        _freeSets = new nint[Capacity];
        _seenDestroyEpoch = device.BufferDestroyEpoch;
    }

    /// <summary>Live entries. Exposed for tests.</summary>
    internal int Count => _count;

    /// <summary>
    /// Returns a populated descriptor set for <paramref name="buffers"/> —
    /// allocates + writes one on first call, reuses it thereafter. The
    /// caller owns the cache lifetime; <see cref="Reset"/> drops every
    /// entry (e.g. on pool exhaustion).
    /// </summary>
    public nint GetOrCreate(ReadOnlySpan<nint> buffers)
    {
        if (buffers.Length != _buffersPerSet)
            throw new ArgumentException(
                $"Expected {_buffersPerSet} buffers, got {buffers.Length}.", nameof(buffers));

        // Hazard-scoped barriers (issue #144): every kernel calls GetOrCreate
        // with the exact buffer list of the dispatch it is about to record,
        // so this is where the dispatch's access set is declared. Emits a
        // batched barrier into the current command buffer only on a real
        // RAW/WAR/WAW conflict. No-op (null) outside a tracked forward.
        _device.ActiveHazards?.OnDispatch(buffers, _writesMask);

        // A destroyed buffer's handle value may already belong to a new buffer, so an entry keyed
        // on it would bind the freed allocation (issue #467). One volatile read when nothing moved.
        if (_device.BufferDestroyEpoch != _seenDestroyEpoch)
            EvictDestroyed();

        // Linear scan — 256 entries × up-to-4 pointer comparisons is
        // ~a microsecond, well below vkAllocateDescriptorSets latency.
        for (int i = 0; i < _count; i++)
        {
            if (Matches(i, buffers))
                return _sets[i];
        }

        // Miss — write + insert, reusing an evicted set when there is one. Overflow is NOT recoverable mid-pass:
        // resetting the pool here would free sets still referenced by dispatches
        // already recorded in the open command buffer (silent corruption). Since
        // Capacity == the pool's maxSets, reaching here means the model needs more
        // concurrent descriptor sets than the pool holds — fail loudly so the pool
        // size can be raised, rather than corrupting the forward.
        if (_count >= Capacity)
        {
            throw new InvalidOperationException(
                $"DescriptorSetCache overflow: more than Capacity={Capacity} distinct buffer tuples " +
                $"(buffersPerSet={_buffersPerSet}) for one kernel in a single forward. Increase " +
                $"{nameof(KernelSupport)}.{nameof(KernelSupport.DefaultMaxSetsPerPool)} (and this Capacity, kept equal to it).");
        }

        nint set = _freeCount > 0
            ? _freeSets[--_freeCount]
            : KernelSupport.AllocateDescriptorSet(_device, _pool, _setLayout);
        KernelSupport.WriteBufferBindings(_device, set, buffers);
        _device.RecordBuffersBound(buffers);

        int slot = _count;
        int baseIdx = slot * MaxBuffersPerSet;
        for (int j = 0; j < _buffersPerSet; j++)
            _keys[baseIdx + j] = buffers[j];
        _sets[slot] = set;
        _count++;
        return set;
    }

    /// <summary>
    /// Forgets every cached entry and resets the underlying descriptor
    /// pool. Call when the caller has externally invalidated the sets
    /// (e.g. the kernel's scratch buffers were re-allocated).
    /// </summary>
    public void Reset()
    {
        // #369: shared READ lock — see VulkanDevice.s_lifecycleLock's doc
        // comment. vkResetDescriptorPool bulk-frees every descriptor set in
        // the pool, the same "device frees an object" hazard class as
        // AllocateDescriptorSet's vkAllocateDescriptorSets.
        VulkanDevice.s_lifecycleLock.EnterReadLock();
        try
        {
            VulkanApi.vkResetDescriptorPool(_device.Handle, _pool, 0)
                .ThrowOnError("vkResetDescriptorPool DescriptorSetCache");
        }
        finally
        {
            VulkanDevice.s_lifecycleLock.ExitReadLock();
        }
        Array.Clear(_keys);
        Array.Clear(_sets);
        _count = 0;
        _freeCount = 0;
        _seenDestroyEpoch = _device.BufferDestroyEpoch;
    }

    /// <summary>
    /// Drops every entry that references a buffer destroyed since the last call, moving its set to
    /// the free list for reuse.
    /// </summary>
    /// <remarks>
    /// Rewriting an evicted set is safe even mid-recording: its entry names a destroyed buffer, so
    /// no pending or recording command buffer can legally still use it. The one exception is the
    /// overrun path — more than <see cref="VulkanDevice.DestroyLogCapacity"/> destroys since this
    /// cache last ran — which evicts everything, including sets a recording command buffer might
    /// hold. That needs thousands of buffer destroys between two dispatches of one kernel inside a
    /// single recording, which no forward pass does; it is accepted rather than guarded.
    /// </remarks>
    private void EvictDestroyed()
    {
        var destroyed = _destroyedScratch ??= new HashSet<nint>();
        destroyed.Clear();
        bool complete = _device.TryCollectDestroyedSince(_seenDestroyEpoch, destroyed, out long epoch);
        _seenDestroyEpoch = epoch;

        int i = 0;
        while (i < _count)
        {
            if (complete && !ReferencesAny(i, destroyed))
            {
                i++;
                continue;
            }

            // Evict slot i: park its set, then move the last live entry into the hole.
            _freeSets[_freeCount++] = _sets[i];
            int last = --_count;
            if (i != last)
            {
                Array.Copy(_keys, last * MaxBuffersPerSet, _keys, i * MaxBuffersPerSet, MaxBuffersPerSet);
                _sets[i] = _sets[last];
            }
            Array.Clear(_keys, last * MaxBuffersPerSet, MaxBuffersPerSet);
            _sets[last] = 0;
        }
    }

    private bool ReferencesAny(int slot, HashSet<nint> handles)
    {
        int baseIdx = slot * MaxBuffersPerSet;
        for (int j = 0; j < _buffersPerSet; j++)
        {
            if (handles.Contains(_keys[baseIdx + j]))
                return true;
        }
        return false;
    }

    private bool Matches(int slot, ReadOnlySpan<nint> buffers)
    {
        int baseIdx = slot * MaxBuffersPerSet;
        for (int j = 0; j < _buffersPerSet; j++)
        {
            if (_keys[baseIdx + j] != buffers[j])
                return false;
        }
        return true;
    }
}
