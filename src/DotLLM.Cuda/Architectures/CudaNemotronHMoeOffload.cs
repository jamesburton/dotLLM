// src/DotLLM.Cuda/Architectures/CudaNemotronHMoeOffload.cs
using DotLLM.Core.Moe;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;

namespace DotLLM.Cuda.Architectures;

/// <summary>What the CUDA <c>nemotron_h_moe</c> loader decided about expert offload (issue #1070).</summary>
/// <param name="CacheBytes">Device bytes of the expert-cache slots.</param>
/// <param name="CacheSlots">Slots of all pools.</param>
/// <param name="Pools">Slot pools (one per expert layout).</param>
/// <param name="Policy">Replacement policy.</param>
/// <param name="PinnedStore">True when the offloaded banks are held as one pinned host copy; false = pinned staging ring over the mmap.</param>
/// <param name="StoreBytes">Bytes of the offloaded expert banks.</param>
/// <param name="BudgetBytes">Measured cache budget (free − reserve − KV at max context − staging arena).</param>
/// <param name="RequestClamped">True when an explicit <c>--moe-cache-mib</c> exceeded the budget and was clamped.</param>
/// <param name="ReserveBytes">Device memory kept free.</param>
/// <param name="FreeBytesAfterLoad">Device free memory measured after the cache was allocated.</param>
/// <param name="ArenaWaveExperts">Experts per staging wave (the arena holds two waves).</param>
/// <param name="ArenaBytes">Device bytes of the staging arena.</param>
/// <param name="SpillRetries">Times the spill probe shrank the cache.</param>
public sealed record CudaMoeOffloadInfo(long CacheBytes, int CacheSlots, int Pools, ExpertCachePolicy Policy, bool PinnedStore,
    long StoreBytes, long BudgetBytes, bool RequestClamped, long ReserveBytes, long FreeBytesAfterLoad, int ArenaWaveExperts,
    long ArenaBytes, int SpillRetries);

/// <summary>
/// Routed-expert offload for <c>nemotron_h_moe</c> on CUDA (issue #1070; design #1063 sections 8b and 8d, and the addendum's
/// VRAM-budget hard rule):
/// <list type="bullet">
///   <item>The routed expert banks stay on the host: one pinned copy when they fit the pin budget, else the memory-mapped file
///   read through a pinned staging ring. Everything else stays on the GPU.</item>
///   <item>A device expert cache of fixed slot pools (one per expert layout, shared by every layer with that layout) is
///   allocated once, from the measured free memory, and never grows. <see cref="ExpertCache"/> keeps the bookkeeping.</item>
///   <item>Copy-on-miss: after host routing, every miss of the layer call is copied on a dedicated copy stream; the compute
///   stream waits on an event before the expert kernels.</item>
///   <item>Experts that cannot be admitted (every slot pinned by the same call: a prefill block touching more experts than the
///   pool holds) stream through a two-wave staging arena: wave i+1 copies while wave i computes.</item>
/// </list>
/// The expert kernels, their per-group arithmetic and the slot-ordered weighted sum are the resident path's own, so
/// residency does not change the result: only the per-group weight pointers and the order of the groups in the schedule
/// differ (each group writes its own slot rows).
/// </summary>
public sealed unsafe partial class CudaNemotronHTransformerModel
{
    /// <summary>Test hook: experts per staging wave (default: the largest top-k of any MoE layer). 1 forces one wave per expert.</summary>
    internal static int? MoeArenaWaveExpertsOverride { get; set; }

    /// <summary>
    /// Test hook: receives every offloaded layer call's lookup (layer index, distinct experts in ascending order) before the
    /// cache resolves it, so a test can replay the routing trace through an offline <see cref="ExpertCache"/>.
    /// </summary>
    internal static Action<int, int[]>? MoeLookupTraceHook { get; set; }

    /// <summary>
    /// Test hook: bytes of a dummy host-to-device copy queued on the copy stream ahead of every batch of expert copies, so
    /// the expert copies land well after the compute stream reaches the expert kernels. With small synthetic experts the
    /// copies otherwise finish first, which would hide a missing compute-stream wait on the copy event.
    /// </summary>
    internal static long MoeCopyStreamDelayBytesForTests { get; set; }

    private MoeOffloadState? _offload;

    /// <summary>The offload decision, or null when every expert is resident.</summary>
    public CudaMoeOffloadInfo? MoeOffload => _offload?.Info;

    /// <summary>The live expert cache (tests read its counters), or null when every expert is resident.</summary>
    internal ExpertCache? MoeExpertCache => _offload?.Cache;

    /// <summary>
    /// Decides residency for a <c>nemotron_h_moe</c> model before any upload: <see cref="MoeOffloadMode.On"/> always offloads,
    /// <see cref="MoeOffloadMode.Off"/> throws when the model does not fit, <see cref="MoeOffloadMode.Auto"/> offloads exactly
    /// when it does not fit.
    /// </summary>
    private static bool DecideMoeOffload(DeviceBudget estimate, MoeOffloadOptions options)
    {
        if (options.Mode == MoeOffloadMode.On) return true;
        CudaDriverApi.cuMemGetInfo_v2(out nuint free, out _).ThrowOnError();
        bool fits = estimate.Total <= (long)free - ReserveBytes(options);
        if (fits) return false;
        if (options.Mode == MoeOffloadMode.Off) ThrowIfMoeModelDoesNotFit(estimate, ReserveBytes(options));
        return true;
    }

    private static long ReserveBytes(MoeOffloadOptions o)
        => o.ReserveMib is { } mib ? mib * MoeOffloadPlanner.MiB : MoeVramReserveBytes;

    private void DisposeMoeOffload()
    {
        _offload?.Dispose();
        _offload = null;
    }

    // ───────────────────────── per-call: plan, copies, launches ─────────────────────────

    /// <summary>
    /// Resolves the used experts of one offloaded layer call through the cache, orders the groups (cached and newly admitted
    /// first, then the streamed ones by wave) in <paramref name="groupExperts"/>, and writes each group's weight pointers.
    /// </summary>
    /// <returns>Number of leading groups that need no staging wave.</returns>
    private int PlanOffloadedGroups(DeviceMoe moe, int[] groupExperts, int groups, long[] groupUp, long[] groupDown)
        => _offload!.Plan(moe, groupExperts, groups, groupUp, groupDown);

    /// <summary>
    /// Runs the expert-grouped up (+relu²) and down GEMVs for groups <c>[g0, g0 + count)</c> of the uploaded schedule.
    /// The kernel reads <c>groupStart[g]</c> as absolute offsets into the slot list, so a group subset is just an offset of
    /// the three per-group arrays.
    /// </summary>
    private void LaunchExpertGroups(DeviceMoe moe, in MoeSchedule sched, int g0, int count, nint expertIn, int expertDim, nint streamH)
    {
        if (count <= 0) return;
        nint upPtrs = _moeSchedDevice + (nint)(sched.UpPtrsOffset + (long)g0 * sizeof(long));
        nint downPtrs = _moeSchedDevice + (nint)(sched.DownPtrsOffset + (long)g0 * sizeof(long));
        nint groupStart = _moeSchedDevice + (nint)(sched.GroupStartOffset + (long)g0 * sizeof(int));
        nint slotIds = _moeSchedDevice + (nint)sched.SlotIdsOffset;
        nint slotToken = _moeSchedDevice + (nint)sched.SlotTokenOffset;

        _kernels.LaunchMoeExpertGroupedGemvF32(moe.UpQt, upPtrs, groupStart, slotIds, slotToken,
            expertIn, _moeMid, moe.MoeIntermediateSize, expertDim, count, relu2: true, streamH);
        MoePathCounters.CountLaunch(moe.UpQt, count);
        _kernels.LaunchMoeExpertGroupedGemvF32(moe.DownQt, downPtrs, groupStart, slotIds, xRow: 0,
            _moeMid, _moeSlotOut, expertDim, moe.MoeIntermediateSize, count, relu2: false, streamH);
        MoePathCounters.CountLaunch(moe.DownQt, count);
    }

    /// <summary>
    /// The offloaded expert phase of one layer call: enqueue the copies of the admitted misses and of the first two staging
    /// waves on the copy stream, then run the ready groups and each wave once its copy event fires, refilling the arena half
    /// a finished wave frees.
    /// </summary>
    private void RunOffloadedExperts(DeviceMoe moe, in MoeSchedule sched, nint expertIn, int expertDim, nint streamH)
    {
        var off = _offload!;
        nint readyEvent = off.EnqueueReadyCopies(moe, _moeEvent);
        int waves = off.Waves;
        nint[] waveEvents = off.WaveEvents;
        for (int w = 0; w < Math.Min(2, waves); w++) waveEvents[w] = off.EnqueueWaveCopies(moe, w);

        if (readyEvent != 0) off.WaitForCopies(streamH, readyEvent);
        LaunchExpertGroups(moe, sched, 0, off.ReadyGroups, expertIn, expertDim, streamH);

        int perWave = off.WaveExperts;
        for (int w = 0; w < waves; w++)
        {
            off.WaitForCopies(streamH, waveEvents[w]);
            int first = w * perWave;
            int count = Math.Min(perWave, off.BypassCount - first);
            LaunchExpertGroups(moe, sched, off.ReadyGroups + first, count, expertIn, expertDim, streamH);
            if (w + 2 < waves)
            {
                // Wave w+2 reuses wave w's arena half: its copy may start only after wave w's kernels.
                nint kernelsDone = off.RentEvent();
                CudaDriverApi.cuEventRecord(kernelsDone, streamH).ThrowOnError();
                CudaDriverApi.cuStreamWaitEvent(off.CopyStream, kernelsDone, 0).ThrowOnError();
                waveEvents[w + 2] = off.EnqueueWaveCopies(moe, w + 2);
            }
        }
    }

    // ───────────────────────── load ─────────────────────────

    /// <summary>
    /// Sets up offload after every other device allocation exists (weights, forward state and scratch pre-sized to their worst
    /// case): the host store, then the cache pools and the staging arena from the measured budget, then the spill probe.
    /// </summary>
    private void InitializeMoeOffload(NemotronHLayerWeights[] cpuLayers, MoeOffloadOptions options)
    {
        _offload = MoeOffloadState.Create(this, cpuLayers, options);
    }

    /// <summary>Device bytes of the KV cache at <paramref name="tokens"/> tokens (<see cref="CudaNemotronHKvCache"/> is FP32).</summary>
    internal static long KvCacheBytes(int attentionLayers, int numKvHeads, int headDim, int tokens)
        => 2L * attentionLayers * tokens * numKvHeads * headDim * sizeof(float);

    /// <summary>All offload state of one model: the cache, its device memory, the copy stream and the host store.</summary>
    private sealed class MoeOffloadState : IDisposable
    {
        private const long StagingChunkBytes = 32 * MoeOffloadPlanner.MiB;
        private const int StagingChunks = 4;
        private const int MaxSpillRetries = 3;

        private readonly ExpertLayout[] _layouts;
        private readonly int[] _poolOfLayer;
        private readonly nint[] _poolBase;
        private readonly nint[] _upHost, _downHost;     // per layer: host base of the up / down bank (pinned copy or mmap)
        private readonly List<nint> _pinned = [];
        private StagingRing? _ring;
        private nint _arena;
        private long _arenaSlotBytes;
        private nint _copyStream;

        private readonly List<nint> _events = [];
        private int _eventCursor;
        private readonly List<(nint Start, nint End)> _timingPending = [];
        private readonly Stack<(nint Start, nint End)> _timingFree = new();

        // Per-call plan (valid between Plan and the last wave of RunOffloadedExperts).
        private int[] _lookupSlots = [];
        private ExpertLookupResult[] _lookupResults = [];
        private int[] _order = [];
        private int[] _readyInsertedExperts = [];
        private int[] _readyInsertedSlots = [];
        private int _readyInsertedCount;
        private int[] _bypassExperts = [];

        public ExpertCache Cache { get; private set; } = null!;
        public CudaMoeOffloadInfo Info { get; private set; } = null!;
        public nint CopyStream => _copyStream;
        public int WaveExperts { get; }
        public int ReadyGroups { get; private set; }
        public int BypassCount { get; private set; }
        public int Waves => (BypassCount + WaveExperts - 1) / WaveExperts;
        public nint[] WaveEvents { get; private set; } = [];

        private MoeOffloadState(int layers, int waveExperts, ExpertLayout[] layouts, int[] poolOfLayer)
        {
            _layouts = layouts;
            _poolOfLayer = poolOfLayer;
            _poolBase = new nint[layouts.Length];
            _upHost = new nint[layers];
            _downHost = new nint[layers];
            WaveExperts = waveExperts;
        }

        public static MoeOffloadState Create(CudaNemotronHTransformerModel model, NemotronHLayerWeights[] cpuLayers,
            MoeOffloadOptions options)
        {
            var log = options.Log ?? (msg => Console.Error.WriteLine($"[dotllm] {msg}"));
            int layers = cpuLayers.Length;

            // Layouts: experts with equal (up, down) byte sizes share a pool across layers (llama.cpp #30112).
            var layoutIndex = new Dictionary<ExpertLayout, int>();
            var layoutList = new List<ExpertLayout>();
            var poolOfLayer = new int[layers];
            var expertsPerLayer = new int[layers];
            var expertsPerLayout = new List<int>();
            long storeBytes = 0;
            int maxTopK = 1;
            for (int l = 0; l < layers; l++)
            {
                poolOfLayer[l] = -1;
                if (cpuLayers[l].Moe is not { } m) continue;
                var layout = new ExpertLayout(m.UpPerExpertBytes, m.DownPerExpertBytes);
                if (!layoutIndex.TryGetValue(layout, out int p))
                {
                    p = layoutList.Count;
                    layoutIndex.Add(layout, p);
                    layoutList.Add(layout);
                    expertsPerLayout.Add(0);
                }
                poolOfLayer[l] = p;
                expertsPerLayer[l] = m.NumExperts;
                expertsPerLayout[p] += m.NumExperts;
                storeBytes += (m.UpPerExpertBytes + m.DownPerExpertBytes) * m.NumExperts;
                maxTopK = Math.Max(maxTopK, m.NumExpertsPerTok);
            }
            var layouts = layoutList.ToArray();
            int waveExperts = Math.Max(1, MoeArenaWaveExpertsOverride ?? maxTopK);

            // Host store: one pinned copy when it fits the pin budget, else a pinned staging ring over the mmap.
            long ram = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            bool pin = MoeOffloadPlanner.FitsPinBudget(storeBytes, ram, options.PinBudgetFraction);
            MoeOffloadState? state = null;
            try
            {
                state = new MoeOffloadState(layers, waveExperts, layouts, poolOfLayer);
                if (pin) pin = state.TryPinStore(cpuLayers, log);
                if (!pin)
                {
                    state._ring = new StagingRing(StagingChunks, StagingChunkBytes);
                    state.UseMmapStore(cpuLayers);
                }

                CudaDriverApi.cuStreamCreate(out state._copyStream, CudaDriverApi.CU_STREAM_NON_BLOCKING).ThrowOnError();
                state.AllocateCache(model, options, layouts, expertsPerLayout.ToArray(), expertsPerLayer, poolOfLayer,
                    storeBytes, pin, log);
                return state;
            }
            catch
            {
                state?.Dispose();
                throw;
            }
        }

        /// <summary>Copies every offloaded bank into page-locked host memory; false (nothing kept) when pinning fails.</summary>
        private bool TryPinStore(NemotronHLayerWeights[] cpuLayers, Action<string> log)
        {
            try
            {
                for (int l = 0; l < cpuLayers.Length; l++)
                {
                    if (cpuLayers[l].Moe is not { } m) continue;
                    long up = m.UpPerExpertBytes * m.NumExperts, down = m.DownPerExpertBytes * m.NumExperts;
                    _upHost[l] = PinCopy(m.UpExpsWeight, up);
                    _downHost[l] = PinCopy(m.DownExpsWeight, down);
                }
                return true;
            }
            catch (CudaException ex)
            {
                log($"MoE offload: pinning the expert store failed ({ex.Message}); streaming experts from the memory-mapped file instead.");
                foreach (nint p in _pinned) CudaDriverApi.cuMemFreeHost(p);
                _pinned.Clear();
                return false;
            }
        }

        private nint PinCopy(nint src, long bytes)
        {
            CudaDriverApi.cuMemHostAlloc(out nint p, (nuint)bytes, 0).ThrowOnError();
            _pinned.Add(p);
            Buffer.MemoryCopy((void*)src, (void*)p, bytes, bytes);
            return p;
        }

        private void UseMmapStore(NemotronHLayerWeights[] cpuLayers)
        {
            for (int l = 0; l < cpuLayers.Length; l++)
            {
                if (cpuLayers[l].Moe is not { } m) continue;
                _upHost[l] = m.UpExpsWeight;
                _downHost[l] = m.DownExpsWeight;
            }
        }

        /// <summary>
        /// Allocates the staging arena and the cache pools once, from the measured budget, then probes for WDDM spill. With an
        /// automatic size a failed probe (or a free-memory check below the reserve) shrinks the cache by 10 % and retries, up to
        /// <see cref="MaxSpillRetries"/> times; an explicit size fails instead.
        /// </summary>
        private void AllocateCache(CudaNemotronHTransformerModel model, MoeOffloadOptions options, ExpertLayout[] layouts,
            int[] expertsPerLayout, int[] expertsPerLayer, int[] poolOfLayer, long storeBytes, bool pinned, Action<string> log)
        {
            long reserve = ReserveBytes(options);
            var cfg = model.Config;
            int kvTokens = Math.Clamp(options.ContextTokens ?? cfg.MaxSequenceLength, 1, cfg.MaxSequenceLength);
            long kv = KvCacheBytes(model._attentionLayerCount, cfg.NumKvHeads, cfg.HeadDim, kvTokens);

            _arenaSlotBytes = layouts.Max(l => l.SlotBytes);
            long arenaBytes = 2L * WaveExperts * _arenaSlotBytes;
            CudaDriverApi.cuMemAlloc_v2(out _arena, (nuint)arenaBytes).ThrowOnError();

            // Free memory is measured now, after every other allocation (weights, forward state, scratch at its worst case
            // and the arena), so only the KV cache the caller allocates later remains to be subtracted.
            CudaDriverApi.cuMemGetInfo_v2(out nuint freeNow, out _).ThrowOnError();
            long budget = MoeOffloadPlanner.CacheBudgetBytes((long)freeNow, reserve,
                new MoeOffloadPlanner.Reservations(AlwaysResidentWeights: 0, ForwardState: 0, KvCache: kv, Scratch: 0));
            long[] slotBytes = [.. layouts.Select(l => l.SlotBytes)];
            long maxUseful = 0;
            for (int p = 0; p < layouts.Length; p++) maxUseful += slotBytes[p] * expertsPerLayout[p];
            long cacheBytes = MoeOffloadPlanner.ResolveCacheBytes(budget, options.CacheMib, maxUseful, out bool clamped);
            if (clamped)
                log($"MoE offload: --moe-cache-mib {options.CacheMib} exceeds the measured VRAM budget of {budget / MoeOffloadPlanner.MiB} MiB " +
                    $"(free {(long)freeNow / MoeOffloadPlanner.MiB} MiB - reserve {reserve / MoeOffloadPlanner.MiB} MiB - KV at " +
                    $"{kvTokens} tokens {kv / MoeOffloadPlanner.MiB} MiB); using {cacheBytes / MoeOffloadPlanner.MiB} MiB.");

            int retries = 0;
            while (true)
            {
                int[] slots = MoeOffloadPlanner.SplitSlots(slotBytes, expertsPerLayout, cacheBytes);
                AllocatePools(layouts, slots);
                CudaDriverApi.cuMemGetInfo_v2(out nuint freeAfter, out _).ThrowOnError();
                string? problem = (long)freeAfter < reserve + kv
                    ? $"free VRAM after allocating the cache is {(long)freeAfter / MoeOffloadPlanner.MiB} MiB, below the reserve " +
                      $"{reserve / MoeOffloadPlanner.MiB} MiB + KV {kv / MoeOffloadPlanner.MiB} MiB"
                    : ProbeForSpill(slots, layouts);
                if (problem is null)
                {
                    long allocated = 0;
                    for (int p = 0; p < layouts.Length; p++) allocated += (long)slots[p] * slotBytes[p];
                    Cache = new ExpertCache(expertsPerLayer, poolOfLayer, slots, options.Policy);
                    Info = new CudaMoeOffloadInfo(allocated, slots.Sum(), layouts.Length, options.Policy, pinned, storeBytes,
                        budget, clamped, reserve, (long)freeAfter, WaveExperts, arenaBytes, retries);
                    MoeTelemetry.SetCache(Info.CacheSlots, allocated, options.Policy);
                    log($"MoE offload: {storeBytes / (double)(1L << 30):F2} GiB of experts in a {(pinned ? "pinned host store" : "memory-mapped store with a pinned staging ring")}; " +
                        $"VRAM cache {allocated / (double)(1L << 30):F2} GiB ({Info.CacheSlots} slots in {layouts.Length} pool(s), " +
                        $"{options.Policy.ToString().ToUpperInvariant()}), staging arena {arenaBytes / MoeOffloadPlanner.MiB} MiB, " +
                        $"{(long)freeAfter / MoeOffloadPlanner.MiB} MiB free.");
                    return;
                }

                FreePools();
                if (options.CacheMib is not null || retries >= MaxSpillRetries || cacheBytes == 0)
                    throw new InvalidOperationException(
                        $"MoE expert offload cannot place a {cacheBytes / MoeOffloadPlanner.MiB} MiB VRAM expert cache: {problem}. " +
                        "Windows WDDM spills over-committed VRAM into shared system memory instead of failing (issue #1063 addendum). " +
                        "Pass a smaller --moe-cache-mib, free VRAM held by other processes, or raise the reserve.");
                retries++;
                cacheBytes = cacheBytes * 9 / 10;
                log($"MoE offload: {problem}; shrinking the VRAM expert cache by 10 % to {cacheBytes / MoeOffloadPlanner.MiB} MiB " +
                    $"(retry {retries} of {MaxSpillRetries}).");
            }
        }

        private void AllocatePools(ExpertLayout[] layouts, int[] slots)
        {
            for (int p = 0; p < layouts.Length; p++)
            {
                long bytes = (long)slots[p] * layouts[p].SlotBytes;
                if (bytes == 0) continue;
                CudaDriverApi.cuMemAlloc_v2(out _poolBase[p], (nuint)bytes).ThrowOnError();
            }
        }

        private void FreePools()
        {
            for (int p = 0; p < _poolBase.Length; p++) FreeIfNonZero(ref _poolBase[p]);
        }

        /// <summary>
        /// Spill probe (#1063 addendum): touches every page of the pools and the arena (so WDDM must make them resident), then
        /// times sampled device-to-device reads over them against the same volume read from a small fresh buffer. A 5×
        /// bandwidth drop means part of the pool lives in shared system memory. Returns a description, or null when it passes.
        /// </summary>
        private string? ProbeForSpill(int[] slots, ExpertLayout[] layouts)
        {
            const long Chunk = 4 * MoeOffloadPlanner.MiB;
            const int Samples = 16;
            var regions = new List<(nint Ptr, long Bytes)>();
            for (int p = 0; p < layouts.Length; p++)
                if (_poolBase[p] != 0) regions.Add((_poolBase[p], (long)slots[p] * layouts[p].SlotBytes));
            regions.Add((_arena, 2L * WaveExperts * _arenaSlotBytes));
            foreach (var (ptr, bytes) in regions) CudaDriverApi.cuMemsetD8_v2(ptr, 0, (nuint)bytes).ThrowOnError();
            // The memsets run on the legacy NULL stream, which the non-blocking copy stream does not wait for: finish them first
            // so they cannot overlap (and slow) the timed reads.
            CudaDriverApi.cuStreamSynchronize(0).ThrowOnError();

            CudaDriverApi.cuMemAlloc_v2(out nint probe, (nuint)(2 * Chunk)).ThrowOnError();
            nint start = 0, end = 0;
            try
            {
                CudaDriverApi.cuMemsetD8_v2(probe, 0, (nuint)(2 * Chunk)).ThrowOnError();
                CudaDriverApi.cuStreamSynchronize(0).ThrowOnError();
                CudaDriverApi.cuEventCreate(out start, CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();
                CudaDriverApi.cuEventCreate(out end, CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();
                nint dst = probe + (nint)Chunk;

                // Samples spread evenly over every region (always including each region's last chunk).
                var samples = new List<(nint Ptr, long Bytes)>();
                foreach (var (ptr, bytes) in regions)
                {
                    int n = (int)Math.Clamp(bytes / Chunk, 1, Samples);
                    for (int i = 0; i < n; i++)
                    {
                        long len = Math.Min(Chunk, bytes);
                        long off = n == 1 ? 0 : (bytes - len) * i / (n - 1);
                        samples.Add((ptr + (nint)off, len));
                    }
                }
                // The reference issues the same copies (count and sizes) from the small buffer, so per-copy overhead cancels out.
                long total = samples.Sum(s => s.Bytes);
                double poolSeconds = Time(() => { foreach (var (ptr, len) in samples) CopyDtoD(dst, ptr, len); });
                double refSeconds = Time(() => { foreach (var (_, len) in samples) CopyDtoD(dst, probe, len); });
                double poolBw = total / Math.Max(poolSeconds, 1e-9), refBw = total / Math.Max(refSeconds, 1e-9);

                // Below half a millisecond the read is too small to tell spill from timer noise (a spilled multi-MiB sample alone
                // takes milliseconds over PCIe).
                const double MinJudgedSeconds = 0.5e-3;
                return poolSeconds >= MinJudgedSeconds && MoeOffloadPlanner.LooksSpilled(poolBw, refBw)
                    ? $"the spill probe read the cache at {poolBw / 1e9:F1} GB/s against {refBw / 1e9:F1} GB/s for a small resident buffer"
                    : null;
            }
            finally
            {
                if (start != 0) CudaDriverApi.cuEventDestroy_v2(start);
                if (end != 0) CudaDriverApi.cuEventDestroy_v2(end);
                CudaDriverApi.cuMemFree_v2(probe);
            }

            void CopyDtoD(nint dst, nint src, long bytes)
                => CudaDriverApi.cuMemcpyDtoDAsync_v2(dst, src, (nuint)bytes, _copyStream).ThrowOnError();

            double Time(Action work)
            {
                work(); // warm-up pass (first-touch paging, clocks)
                CudaDriverApi.cuEventRecord(start, _copyStream).ThrowOnError();
                work();
                CudaDriverApi.cuEventRecord(end, _copyStream).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float ms, start, end).ThrowOnError();
                return ms / 1000.0;
            }
        }

        // ── per call ──

        /// <summary>See <see cref="PlanOffloadedGroups"/>.</summary>
        public int Plan(DeviceMoe moe, int[] groupExperts, int groups, long[] groupUp, long[] groupDown)
        {
            _eventCursor = 0;
            if (_lookupSlots.Length < groups)
            {
                _lookupSlots = new int[groups];
                _lookupResults = new ExpertLookupResult[groups];
                _order = new int[groups];
                _readyInsertedExperts = new int[groups];
                _readyInsertedSlots = new int[groups];
                _bypassExperts = new int[groups];
            }

            int layer = moe.LayerIndex;
            MoeLookupTraceHook?.Invoke(layer, groupExperts.AsSpan(0, groups).ToArray());
            long evictionsBefore = Cache.Evictions;
            Cache.Lookup(layer, groupExperts.AsSpan(0, groups), _lookupSlots, _lookupResults);

            int pool = _poolOfLayer[layer];
            var layout = _layouts[pool];
            int ready = 0, hits = 0;
            _readyInsertedCount = 0;
            BypassCount = 0;
            for (int g = 0; g < groups; g++)
            {
                int e = groupExperts[g];
                switch (_lookupResults[g])
                {
                    case ExpertLookupResult.Hit:
                        hits++;
                        _order[ready++] = e;
                        break;
                    case ExpertLookupResult.Inserted:
                        _order[ready++] = e;
                        _readyInsertedExperts[_readyInsertedCount] = e;
                        _readyInsertedSlots[_readyInsertedCount++] = _lookupSlots[g];
                        break;
                    default:
                        _bypassExperts[BypassCount++] = e;
                        break;
                }
            }

            // Group order: ready groups (ascending expert), then streamed groups (ascending expert, wave by wave).
            int gi = 0;
            for (int i = 0; i < ready; i++)
            {
                int e = _order[i];
                int slot = Cache.SlotOf(layer, e);
                nint slotBase = _poolBase[pool] + (nint)(slot * layout.SlotBytes);
                groupExperts[gi] = e;
                groupUp[gi] = slotBase;
                groupDown[gi] = slotBase + (nint)layout.DownOffset;
                gi++;
            }
            for (int j = 0; j < BypassCount; j++)
            {
                nint b = ArenaSlot(j);
                groupExperts[gi] = _bypassExperts[j];
                groupUp[gi] = b;
                groupDown[gi] = b + (nint)layout.DownOffset;
                gi++;
            }
            ReadyGroups = ready;
            if (WaveEvents.Length < Waves) WaveEvents = new nint[Math.Max(Waves, 2)];

            int misses = groups - hits;
            MoeTelemetry.RecordLookups(hits, misses, Cache.Evictions - evictionsBefore,
                (long)misses * (moe.UpPerExpertBytes + moe.DownPerExpertBytes));
            return ready;
        }

        private nint ArenaSlot(int bypassIndex)
        {
            int wave = bypassIndex / WaveExperts, idx = bypassIndex % WaveExperts;
            return _arena + (nint)(((wave % 2) * (long)WaveExperts + idx) * _arenaSlotBytes);
        }

        /// <summary>
        /// Enqueues the copies of this call's admitted misses into their cache slots. The copy stream first waits on
        /// <paramref name="routerEvent"/> (recorded on the compute stream after every earlier layer's expert kernels), so a slot is
        /// never overwritten while an earlier kernel still reads it. Returns the event to wait on, or 0 when nothing was copied.
        /// </summary>
        public nint EnqueueReadyCopies(DeviceMoe moe, nint routerEvent)
        {
            CudaDriverApi.cuStreamWaitEvent(_copyStream, routerEvent, 0).ThrowOnError();
            if (_readyInsertedCount == 0) return 0;
            DelayCopyStreamForTests();
            int pool = _poolOfLayer[moe.LayerIndex];
            var layout = _layouts[pool];
            for (int i = 0; i < _readyInsertedCount; i++)
            {
                nint slotBase = _poolBase[pool] + (nint)(_readyInsertedSlots[i] * layout.SlotBytes);
                CopyExpert(moe, _readyInsertedExperts[i], slotBase, slotBase + (nint)layout.DownOffset);
            }
            return RecordCopyEvent();
        }

        /// <summary>Enqueues wave <paramref name="wave"/>'s streamed experts into its arena half and returns its event.</summary>
        public nint EnqueueWaveCopies(DeviceMoe moe, int wave)
        {
            var layout = _layouts[_poolOfLayer[moe.LayerIndex]];
            int first = wave * WaveExperts, last = Math.Min(first + WaveExperts, BypassCount);
            if (last > first) DelayCopyStreamForTests();
            for (int j = first; j < last; j++)
            {
                nint b = ArenaSlot(j);
                CopyExpert(moe, _bypassExperts[j], b, b + (nint)layout.DownOffset);
            }
            return RecordCopyEvent();
        }

        private nint _delayHost, _delayDevice;
        private long _delayBytes;

        /// <summary>Queues <see cref="MoeCopyStreamDelayBytesForTests"/> bytes of dummy copy on the copy stream (tests only).</summary>
        private void DelayCopyStreamForTests()
        {
            long bytes = MoeCopyStreamDelayBytesForTests;
            if (bytes <= 0) return;
            if (bytes != _delayBytes)
            {
                FreeDelayBuffers();
                CudaDriverApi.cuMemHostAlloc(out _delayHost, (nuint)bytes, 0).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out _delayDevice, (nuint)bytes).ThrowOnError();
                _delayBytes = bytes;
            }
            CudaDriverApi.cuMemcpyHtoDAsync_v2(_delayDevice, _delayHost, (nuint)bytes, _copyStream).ThrowOnError();
        }

        private void FreeDelayBuffers()
        {
            if (_delayHost != 0) { CudaDriverApi.cuMemFreeHost(_delayHost); _delayHost = 0; }
            FreeIfNonZero(ref _delayDevice);
            _delayBytes = 0;
        }

        private void CopyExpert(DeviceMoe moe, int expert, nint dstUp, nint dstDown)
        {
            int layer = moe.LayerIndex;
            nint srcUp = _upHost[layer] + (nint)(expert * moe.UpPerExpertBytes);
            nint srcDown = _downHost[layer] + (nint)(expert * moe.DownPerExpertBytes);
            if (_ring is null)
            {
                CudaDriverApi.cuMemcpyHtoDAsync_v2(dstUp, srcUp, (nuint)moe.UpPerExpertBytes, _copyStream).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoDAsync_v2(dstDown, srcDown, (nuint)moe.DownPerExpertBytes, _copyStream).ThrowOnError();
            }
            else
            {
                _ring.Copy(dstUp, srcUp, moe.UpPerExpertBytes, _copyStream);
                _ring.Copy(dstDown, srcDown, moe.DownPerExpertBytes, _copyStream);
            }
        }

        private nint RecordCopyEvent()
        {
            _ring?.Close(_copyStream);
            nint evt = RentEvent();
            CudaDriverApi.cuEventRecord(evt, _copyStream).ThrowOnError();
            return evt;
        }

        /// <summary>An event from the per-call pool (recycled at the next <see cref="Plan"/>; a re-record never affects an earlier wait).</summary>
        public nint RentEvent()
        {
            if (_eventCursor == _events.Count)
            {
                CudaDriverApi.cuEventCreate(out nint e, CudaDriverApi.CU_EVENT_DISABLE_TIMING).ThrowOnError();
                _events.Add(e);
            }
            return _events[_eventCursor++];
        }

        /// <summary>Makes the compute stream wait for <paramref name="copyEvent"/>, bracketed by timing events for <c>copy_wait_us</c>.</summary>
        public void WaitForCopies(nint computeStream, nint copyEvent)
        {
            var pair = _timingFree.Count > 0 ? _timingFree.Pop() : NewTimingPair();
            CudaDriverApi.cuEventRecord(pair.Start, computeStream).ThrowOnError();
            CudaDriverApi.cuStreamWaitEvent(computeStream, copyEvent, 0).ThrowOnError();
            CudaDriverApi.cuEventRecord(pair.End, computeStream).ThrowOnError();
            _timingPending.Add(pair);
        }

        private static (nint Start, nint End) NewTimingPair()
        {
            CudaDriverApi.cuEventCreate(out nint a, CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();
            CudaDriverApi.cuEventCreate(out nint b, CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();
            return (a, b);
        }

        /// <summary>
        /// Adds the completed copy waits to <c>copy_wait_us</c>. Call only when the compute stream has passed every pending
        /// pair (after the router event synchronize, or after the forward's stream synchronize).
        /// </summary>
        public void HarvestCopyWait()
        {
            if (_timingPending.Count == 0) return;
            double micros = 0;
            foreach (var pair in _timingPending)
            {
                if (CudaDriverApi.cuEventElapsedTime(out float ms, pair.Start, pair.End) == 0) micros += ms * 1000.0;
                _timingFree.Push(pair);
            }
            _timingPending.Clear();
            MoeTelemetry.RecordCopyWait((long)micros);
        }

        public void Dispose()
        {
            if (_copyStream != 0)
            {
                CudaDriverApi.cuStreamSynchronize(_copyStream);
                CudaDriverApi.cuStreamDestroy_v2(_copyStream);
                _copyStream = 0;
            }
            FreePools();
            FreeIfNonZero(ref _arena);
            FreeDelayBuffers();
            foreach (nint e in _events) CudaDriverApi.cuEventDestroy_v2(e);
            _events.Clear();
            foreach (var (a, b) in _timingPending.Concat(_timingFree))
            {
                CudaDriverApi.cuEventDestroy_v2(a);
                CudaDriverApi.cuEventDestroy_v2(b);
            }
            _timingPending.Clear();
            _timingFree.Clear();
            foreach (nint p in _pinned) CudaDriverApi.cuMemFreeHost(p);
            _pinned.Clear();
            _ring?.Dispose();
            MoeTelemetry.ClearCache();
        }
    }

    /// <summary>
    /// Pinned staging ring for an expert store that is not pinned (design 8b): bytes are copied from the memory-mapped file into
    /// a pinned chunk, then sent with an async H2D on the copy stream. A chunk is reused only after the event recorded when it
    /// was last closed has fired, so the host never overwrites bytes an in-flight copy still reads.
    /// </summary>
    private sealed class StagingRing : IDisposable
    {
        private readonly nint[] _chunks;
        private readonly nint[] _events;
        private readonly bool[] _armed;
        private readonly long _chunkBytes;
        private int _current;
        private long _offset;

        public StagingRing(int chunks, long chunkBytes)
        {
            _chunks = new nint[chunks];
            _events = new nint[chunks];
            _armed = new bool[chunks];
            _chunkBytes = chunkBytes;
            try
            {
                for (int i = 0; i < chunks; i++)
                {
                    CudaDriverApi.cuMemHostAlloc(out _chunks[i], (nuint)chunkBytes, 0).ThrowOnError();
                    CudaDriverApi.cuEventCreate(out _events[i], CudaDriverApi.CU_EVENT_DISABLE_TIMING).ThrowOnError();
                }
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        /// <summary>Stages <paramref name="bytes"/> from <paramref name="src"/> and enqueues their H2D copy to <paramref name="dst"/>.</summary>
        public void Copy(nint dst, nint src, long bytes, nint stream)
        {
            while (bytes > 0)
            {
                if (_offset == _chunkBytes) Advance(stream);
                long piece = Math.Min(bytes, _chunkBytes - _offset);
                nint stage = _chunks[_current] + (nint)_offset;
                Buffer.MemoryCopy((void*)src, (void*)stage, piece, piece);
                CudaDriverApi.cuMemcpyHtoDAsync_v2(dst, stage, (nuint)piece, stream).ThrowOnError();
                _offset += piece;
                dst += (nint)piece;
                src += (nint)piece;
                bytes -= piece;
            }
        }

        /// <summary>Records the current chunk's event (after its last H2D) and moves to the next chunk; the next copy starts fresh.</summary>
        public void Close(nint stream)
        {
            if (_offset > 0) Advance(stream);
        }

        private void Advance(nint stream)
        {
            CudaDriverApi.cuEventRecord(_events[_current], stream).ThrowOnError();
            _armed[_current] = true;
            _current = (_current + 1) % _chunks.Length;
            _offset = 0;
            if (_armed[_current]) CudaDriverApi.cuEventSynchronize(_events[_current]).ThrowOnError();
        }

        public void Dispose()
        {
            for (int i = 0; i < _chunks.Length; i++)
            {
                if (_events[i] != 0) { CudaDriverApi.cuEventSynchronize(_events[i]); CudaDriverApi.cuEventDestroy_v2(_events[i]); _events[i] = 0; }
                if (_chunks[i] != 0) { CudaDriverApi.cuMemFreeHost(_chunks[i]); _chunks[i] = 0; }
            }
        }
    }
}
