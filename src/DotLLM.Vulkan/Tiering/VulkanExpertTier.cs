using System.Diagnostics;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan.Tiering;

/// <summary>
/// Expert tiering for a Vulkan MoE model (#939, stage 2 of #825): the routed experts of every layer live on the SSD and a small per-layer pool of device
/// <i>slots</i> caches the ones decode touches. Default miss policy is <b>stall</b>: after the router's top-k the forward ends its command buffer, the host reads
/// the selected ids back, loads the missing experts straight from the GGUF file (unbuffered overlapped reads into a mapped host-visible ring, no page-cache
/// growth), copies them into their slots, rewrites the ids as slot numbers and resumes. The kernels run the exact same arithmetic on the exact same bytes as the
/// resident path (they only see slot numbers instead of expert numbers), so greedy output is identical to a fully resident run.
/// </summary>
/// <remarks>
/// Prototype scope: single-row (decode-shaped) forwards only, qwen4exp; prefill runs token by token through the same path (correct, slow). The id -> slot remap is
/// done by the host (it already holds the ids at the stall); a device remap dispatch only pays off once the miss path goes asynchronous.
/// </remarks>
internal sealed unsafe class VulkanExpertTier : IDisposable
{
    /// <summary>Per-layer tiering state: the slot cache and where each expert's three pieces live on disk.</summary>
    internal sealed class Layer
    {
        /// <summary>Slot allocator / LRU for this layer.</summary>
        public required ExpertSlotCache Cache { get; init; }

        /// <summary>Slot banks in piece order: [0] gate (W1), [1] up (W3), [2] down (W2). Allocated by the uploader.</summary>
        public required VulkanDevice.Buffer[] SlotBanks { get; init; }

        /// <summary>Bytes of ONE expert in each piece (gate, up, down).</summary>
        public required long[] PieceBytes { get; init; }

        /// <summary>Index into the shard path list for each piece.</summary>
        public required int[] File { get; init; }

        /// <summary>Absolute file offset of expert 0 of each piece (expert e is at this + e * PieceBytes).</summary>
        public required long[] FileOffset { get; init; }

        /// <summary>Bytes read from the SSD for this layer (aligned over-read included).</summary>
        public long BytesRead { get; internal set; }
    }

    private readonly VulkanDevice _device;
    private readonly UnbufferedFileReader _reader;
    private readonly Layer[] _layers;
    private readonly int _topK;
    private readonly VulkanDevice.Buffer _ring;
    private readonly byte* _ringBase;       // 4096-aligned start of the usable ring inside the mapped buffer
    private readonly long _ringBaseOffset;  // byte offset of _ringBase inside _ring
    private readonly long _ringStride;      // bytes reserved per missing expert
    private readonly VulkanDevice.Buffer _idsReadback, _idsUpload;
    private readonly int* _idsReadbackPtr, _idsUploadPtr;
    private bool _disposed;

    /// <summary>Slots per layer.</summary>
    public int Slots { get; }

    /// <summary>Experts per layer.</summary>
    public int NumExperts { get; }

    /// <summary>The forward's "submit the open command buffer and wait" step, set by the model before it records a forward.</summary>
    public Action? EndSubmit { get; set; }

    /// <summary>The forward's "open a fresh command buffer" step; returns the new command buffer.</summary>
    public Func<nint>? Begin { get; set; }

    /// <summary>Total lookups served from a resident slot.</summary>
    public long Hits { get; private set; }

    /// <summary>Total lookups that had to load an expert from the SSD.</summary>
    public long Misses { get; private set; }

    /// <summary>Total misses that displaced a resident expert.</summary>
    public long Evictions { get; private set; }

    /// <summary>Layer stalls that loaded at least one expert.</summary>
    public long MissStalls { get; private set; }

    /// <summary>Layer visits (one per MoE layer per token).</summary>
    public long LayerVisits { get; private set; }

    /// <summary>Bytes read from the SSD, aligned over-read included.</summary>
    public long BytesRead => _reader.BytesRead;

    /// <summary>Wall ticks spent ending the command buffer and waiting for the GPU to reach the router (useful work, not stall).</summary>
    public long GpuWaitTicks { get; private set; }

    /// <summary>Wall ticks spent in the SSD reads of the miss path.</summary>
    public long ReadTicks { get; private set; }

    /// <summary>Wall ticks spent recording the post-miss copies and ids and re-opening the command buffer.</summary>
    public long ResumeTicks { get; private set; }

    /// <summary>True when the reads bypass the OS page cache.</summary>
    public bool Unbuffered => _reader.Unbuffered;

    /// <summary>Layer state for diagnostics and tests.</summary>
    public IReadOnlyList<Layer> Layers => _layers;

    /// <summary>
    /// Creates the tier.
    /// </summary>
    /// <param name="device">Device whose memory holds the ring and id buffers.</param>
    /// <param name="paths">GGUF shard files, indexed by <see cref="Layer.File"/>.</param>
    /// <param name="layers">One entry per MoE layer.</param>
    /// <param name="numExperts">Routed experts per layer.</param>
    /// <param name="topK">Experts selected per token.</param>
    /// <param name="slots">Slots per layer.</param>
    /// <param name="unbuffered">Bypass the page cache (default; false for tests or non-Windows).</param>
    public VulkanExpertTier(VulkanDevice device, IReadOnlyList<string> paths, Layer[] layers, int numExperts, int topK, int slots, bool unbuffered = true)
    {
        _device = device; _layers = layers; _topK = topK; Slots = slots; NumExperts = numExperts;
        if (slots < topK) throw new ArgumentOutOfRangeException(nameof(slots), $"{slots} slots per layer cannot hold the {topK} experts one token selects.");
        _reader = new UnbufferedFileReader(paths, unbuffered);
        long stride = 0;
        const long A = UnbufferedFileReader.Align;
        foreach (var l in layers)
        {
            long s = 0;
            // Worst case an aligned piece span is the piece rounded up to a sector plus one more sector (the expert offset shifts the start).
            for (int p = 0; p < 3; p++) s += ((l.PieceBytes[p] + A - 1) & ~(A - 1)) + A;
            stride = Math.Max(stride, s);
        }
        _ringStride = stride;
        long ringBytes = _ringStride * topK + UnbufferedFileReader.Align;
        _ring = device.Allocate(ringBytes);
        nint mapped = device.MapMemoryWithRetry(_ring.Memory, 0, (ulong)ringBytes, "expert tier ring");
        long pad = (UnbufferedFileReader.Align - (mapped & (UnbufferedFileReader.Align - 1))) & (UnbufferedFileReader.Align - 1);
        _ringBase = (byte*)mapped + pad; _ringBaseOffset = pad;
        _idsReadback = device.AllocateHostReadback(4096);
        _idsReadbackPtr = (int*)device.MapMemoryWithRetry(_idsReadback.Memory, 0, 4096, "expert tier ids");
        _idsUpload = device.Allocate(4096);
        _idsUploadPtr = (int*)device.MapMemoryWithRetry(_idsUpload.Memory, 0, 4096, "expert tier ids upload");
    }

    /// <summary>Aligned file range that covers <c>[offset, offset + bytes)</c>: its start, its length and where the wanted bytes begin inside it.</summary>
    internal static (long Start, int Length, int Delta) PieceSpan(long offset, long bytes)
    {
        const long A = UnbufferedFileReader.Align;
        long start = offset & ~(A - 1);
        long end = (offset + bytes + A - 1) & ~(A - 1);
        return (start, checked((int)(end - start)), (int)(offset - start));
    }

    /// <summary>
    /// The host step of a tiered MoE layer, called by the recorder right after the router's top-k. Ends the open command buffer, makes the experts
    /// <paramref name="topkIndices"/> names resident, then opens a fresh command buffer that already contains the slot copies and the rewritten ids. Returns it.
    /// </summary>
    /// <param name="layer">MoE layer index.</param>
    /// <param name="cmd">The open command buffer, recorded up to and including the top-k.</param>
    /// <param name="topkIndices">Device buffer holding this token's expert ids; overwritten in place with slot numbers.</param>
    public nint AfterTopk(int layer, nint cmd, VulkanDevice.Buffer topkIndices)
    {
        var end = EndSubmit ?? throw new InvalidOperationException("The tier has no command-buffer host (EndSubmit/Begin unset).");
        var begin = Begin!;
        var L = _layers[layer];
        int topK = _topK;
        long t0 = Stopwatch.GetTimestamp();
        // Hand the ids to the host: the top-k results must be visible to the transfer, the copy to the host.
        KernelSupport.ComputeTransferFullBarrier(cmd);
        VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, topkIndices, _idsReadback, 0, 0, (ulong)(topK * sizeof(int)));
        KernelSupport.TransferToHostBarrier(cmd);
        end();
        long t1 = Stopwatch.GetTimestamp();

        Span<int> ids = stackalloc int[topK];
        for (int i = 0; i < topK; i++) ids[i] = _idsReadbackPtr[i];
        Span<int> slotIds = stackalloc int[topK];
        Span<int> missE = stackalloc int[topK];
        Span<int> missS = stackalloc int[topK];
        long hits0 = L.Cache.Hits, miss0 = L.Cache.Misses, ev0 = L.Cache.Evictions;
        int nMiss = L.Cache.Lookup(ids, slotIds, missE, missS);
        Hits += L.Cache.Hits - hits0; Misses += L.Cache.Misses - miss0; Evictions += L.Cache.Evictions - ev0;
        LayerVisits++;

        // Load every missing expert (three pieces each) in one batch so the drive sees the full queue.
        Span<UnbufferedFileReader.Request> reqs = stackalloc UnbufferedFileReader.Request[3 * topK];
        Span<int> delta = stackalloc int[3 * topK];
        int nReq = 0;
        for (int j = 0; j < nMiss; j++)
        {
            byte* dst = _ringBase + (long)j * _ringStride;
            for (int p = 0; p < 3; p++)
            {
                var (start, length, d) = PieceSpan(L.FileOffset[p] + (long)missE[j] * L.PieceBytes[p], L.PieceBytes[p]);
                reqs[nReq] = new UnbufferedFileReader.Request(L.File[p], start, length, dst, d + (int)L.PieceBytes[p]);
                delta[3 * j + p] = d;
                dst += length;
                L.BytesRead += length;
            }
            nReq += 3;
        }
        if (nReq > 0) { _reader.Read(reqs[..nReq]); MissStalls++; }
        long t2 = Stopwatch.GetTimestamp();

        cmd = begin();
        for (int j = 0; j < nMiss; j++)
        {
            long src = _ringBaseOffset + (long)j * _ringStride;
            for (int p = 0; p < 3; p++)
            {
                var (_, length, _) = PieceSpan(L.FileOffset[p] + (long)missE[j] * L.PieceBytes[p], L.PieceBytes[p]);
                VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, _ring, L.SlotBanks[p], (ulong)(src + delta[3 * j + p]),
                    (ulong)((long)missS[j] * L.PieceBytes[p]), (ulong)L.PieceBytes[p]);
                src += length;
            }
        }
        for (int i = 0; i < topK; i++) _idsUploadPtr[i] = slotIds[i];
        VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, _idsUpload, topkIndices, 0, 0, (ulong)(topK * sizeof(int)));
        KernelSupport.ComputeTransferFullBarrier(cmd);   // slot bytes + slot ids -> the expert kernels
        long t3 = Stopwatch.GetTimestamp();
        GpuWaitTicks += t1 - t0; ReadTicks += t2 - t1; ResumeTicks += t3 - t2;
        return cmd;
    }

    /// <summary>Clears every layer's cache (slot contents become unreachable) and the counters.</summary>
    public void Reset()
    {
        foreach (var l in _layers) { l.Cache.Clear(); l.Cache.ResetCounters(); l.BytesRead = 0; }
        Hits = Misses = Evictions = MissStalls = LayerVisits = 0;
        GpuWaitTicks = ReadTicks = ResumeTicks = 0;
    }

    /// <summary>Zeroes the counters but keeps the resident experts (warm-up -> measured window).</summary>
    public void ResetCounters()
    {
        foreach (var l in _layers) { l.Cache.ResetCounters(); l.BytesRead = 0; }
        Hits = Misses = Evictions = MissStalls = LayerVisits = 0;
        GpuWaitTicks = ReadTicks = ResumeTicks = 0;
    }

    /// <summary>One-line summary for logs.</summary>
    public string Describe()
    {
        double total = Hits + Misses;
        double f = 1000.0 / Stopwatch.Frequency;
        return $"slots {Slots}/{NumExperts} per layer: {Hits} hits, {Misses} misses ({(total > 0 ? 100.0 * Hits / total : 0):F1}% hit), {Evictions} evictions, " +
               $"{MissStalls}/{LayerVisits} layer visits stalled, {BytesRead / 1048576.0:F0} MiB read ({(Unbuffered ? "unbuffered" : "buffered")}); " +
               $"gpu-to-router {GpuWaitTicks * f:F0} ms, ssd {ReadTicks * f:F0} ms, resume {ResumeTicks * f:F0} ms";
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _reader.Dispose();
        _ring.Dispose(); _idsReadback.Dispose(); _idsUpload.Dispose();
    }
}
