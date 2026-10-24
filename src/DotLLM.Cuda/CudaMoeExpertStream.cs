using System.Diagnostics;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;

namespace DotLLM.Cuda;

/// <summary>
/// GPU leg for the CPU-resident expert layers of the generic CUDA MoE host (#1202): on a multi-token call, part of a
/// layer's routed experts are copied from the mmap'd GGUF into a small VRAM ring and run there, while the CPU leg
/// computes the rest. This is llama.cpp's "offload the op for large batches", plus a CPU share sized so that both legs
/// finish together.
/// </summary>
/// <remarks>
/// <para>
/// <b>No second copy of the weights.</b> The expert banks' mmap pages are page-locked in place
/// (<c>cuMemHostRegister</c>, read-only) and DMA'd straight to the ring: there is no pinned or staged host copy, and the
/// copies are truly asynchronous, so the link, the GPU and the CPU work at the same time. The experts stay Q4_1 (or
/// whatever the file has) on the device; the ring holds <see cref="Slots"/> waves of <see cref="WaveExperts"/> experts,
/// admitted against free VRAM once and never grown.
/// </para>
/// <para>
/// <b>Transfer cost (CLAUDE.md rule 3).</b> The source is the OS page cache, so a cold file is read from disk by the
/// page-in, then crosses the link once per call: <c>experts x (gate + up + down)</c> bytes, 1.875 MiB per Flint Q4_1
/// expert, about 0.83 ms at the 2.25 GB/s of the RTX 3060 eGPU link (Thunderbolt, PCIe 3.0 x4).
/// </para>
/// <para>
/// <b>Ordering.</b> Copies run on a non-blocking copy stream; wave <c>w</c> uses slot <c>w % Slots</c>. Its copy waits
/// for the compute of wave <c>w - Slots</c> (recorded before the copy is enqueued: waves are computed in copy order), and
/// its compute waits for its copy.
/// </para>
/// </remarks>
internal sealed unsafe class CudaMoeExpertStream : IDisposable
{
    /// <summary>Experts per ring slot (the default; tests shrink it to exercise slot reuse on small models).</summary>
    internal const int DefaultWaveExperts = 8;

    /// <summary>Test hook: experts per ring slot for streams created while set.</summary>
    internal static int? TestWaveExperts { get; set; }

    /// <summary>Max |x| of the CPU part's F16 upload before it is scaled down by a power of two (a test hook lowers it).</summary>
    internal static float CpuPartF16Limit { get; set; } = 16384f;

    /// <summary>Test hook: forces the split point (given the expert count) instead of the cost model.</summary>
    internal static Func<int, int>? TestSplit { get; set; }

    /// <summary>Ring slots.</summary>
    internal const int Slots = 8;

    // Free VRAM that must remain after the ring is allocated (WDDM spills silently instead of failing).
    internal const long RingMarginBytes = 256L << 20;

    private readonly int _numExperts;
    private readonly int _wave;
    private readonly long _gateBytes, _upBytes, _downBytes;
    private readonly List<nint> _registered = [];
    private readonly nint[] _gatePtr, _upPtr, _downPtr;
    private readonly nint[] _copyDone = new nint[Slots], _computeDone = new nint[Slots];
    private readonly Action<string> _log;
    private nint _copyStream;
    private nint _ring;
    private bool _ringFailed;
    private long _nextWave;

    // Waves already copied for the next CPU layer: (layer, slot, first expert, end expert), in wave order.
    private readonly Queue<(MoeLayerWeights Layer, int Slot, int Begin, int End)> _prefetched = new();

    // Cost model (EMA): CPU leg ms per routed assignment, link bytes per ms.
    private double _cpuMsPerAssignment = 0.15;
    private double _linkBytesPerMs = 2.2e6;

    // F16 staging of the CPU part's routed sum: pinned host + device, [rows, hidden].
    private nint _cpuOutHost, _cpuOutDevice;
    private long _cpuOutElems;

    private CudaMoeExpertStream(int numExperts, long gateBytes, long upBytes, long downBytes, Action<string> log)
    {
        _numExperts = numExperts;
        _wave = TestWaveExperts ?? DefaultWaveExperts;
        _gateBytes = gateBytes;
        _upBytes = upBytes;
        _downBytes = downBytes;
        _gatePtr = new nint[numExperts];
        _upPtr = new nint[numExperts];
        _downPtr = new nint[numExperts];
        _log = log;
    }

    /// <summary>VRAM of the staging ring for experts of <paramref name="expertBytes"/> bytes (gate + up + down).</summary>
    internal static long RingBytes(long expertBytes) => (long)Slots * DefaultWaveExperts * expertBytes;

    /// <summary>Experts per ring slot (one wave).</summary>
    internal int WaveExperts => _wave;

    /// <summary>Bytes of one expert (gate + up + down).</summary>
    internal long ExpertBytes => _gateBytes + _upBytes + _downBytes;

    /// <summary>Gate weights by expert id for the bound wave.</summary>
    internal ReadOnlySpan<nint> Gate => _gatePtr;

    /// <summary>Up weights by expert id for the bound wave.</summary>
    internal ReadOnlySpan<nint> Up => _upPtr;

    /// <summary>Down weights by expert id for the bound wave.</summary>
    internal ReadOnlySpan<nint> Down => _downPtr;

    /// <summary>Experts streamed to the GPU so far.</summary>
    internal long ExpertsStreamed { get; private set; }

    /// <summary>
    /// Creates the stream for the CPU-resident expert layers, or returns null (with a log line) when they cannot be
    /// streamed: mixed shapes, or the driver refuses to page-lock the mapped file.
    /// </summary>
    /// <param name="layers">The CPU bundles of every CPU-expert layer.</param>
    /// <param name="log">Load log.</param>
    internal static CudaMoeExpertStream? TryCreate(IReadOnlyList<MoeLayerWeights> layers, Action<string> log)
    {
        if (layers.Count == 0) return null;
        var first = layers[0];
        int hidden = first.HiddenSize, inter = first.IntermediateSize, experts = first.NumExperts;
        if (first.GateExpsRaw == 0 || first.UpExpsRaw == 0 || first.DownExpsRaw == 0)
        {
            log("MoE expert streaming off: the CPU expert banks are not raw mmap views; prefill keeps them on the CPU.");
            return null;
        }
        long gate = Dequantize.RowByteSize((long)inter * hidden, first.GateExpsRawQt);
        long up = Dequantize.RowByteSize((long)inter * hidden, first.UpExpsRawQt);
        long down = Dequantize.RowByteSize((long)hidden * inter, first.DownExpsRawQt);
        long lo = long.MaxValue, hi = 0;
        foreach (var l in layers)
        {
            if (l.HiddenSize != hidden || l.IntermediateSize != inter || l.NumExperts != experts
                || l.GateExpsRawQt != first.GateExpsRawQt || l.UpExpsRawQt != first.UpExpsRawQt
                || l.DownExpsRawQt != first.DownExpsRawQt || l.GateExpsRaw == 0 || l.UpExpsRaw == 0 || l.DownExpsRaw == 0)
            {
                log("MoE expert streaming off: the CPU expert layers differ in shape or type; prefill keeps them on the CPU.");
                return null;
            }
            foreach (var (b, n) in new[] { (l.GateExpsRaw, gate), (l.UpExpsRaw, up), (l.DownExpsRaw, down) })
            {
                lo = Math.Min(lo, b);
                hi = Math.Max(hi, b + n * experts);
            }
        }

        var s = new CudaMoeExpertStream(experts, gate, up, down, log);
        try
        {
            // One read-only registration over the span of every CPU layer's expert banks (they share pages with the
            // neighbouring tensors, which two registrations could not overlap). Page-locking is not a copy.
            const long Page = 4096;
            long start = lo & ~(Page - 1), end = (hi + Page - 1) & ~(Page - 1);
            int rc = CudaDriverApi.cuMemHostRegister_v2((nint)start, (nuint)(end - start), CudaDriverApi.CU_MEMHOSTREGISTER_READ_ONLY);
            if (rc != 0)
            {
                log($"MoE expert streaming off: page-locking the mapped experts failed (CUDA error {rc}); prefill keeps them on the CPU.");
                s.Dispose();
                return null;
            }
            s._registered.Add((nint)start);
            CudaDriverApi.cuStreamCreate(out s._copyStream, CudaDriverApi.CU_STREAM_NON_BLOCKING).ThrowOnError();
            for (int i = 0; i < Slots; i++)
            {
                CudaDriverApi.cuEventCreate(out s._copyDone[i], CudaDriverApi.CU_EVENT_DISABLE_TIMING).ThrowOnError();
                CudaDriverApi.cuEventCreate(out s._computeDone[i], CudaDriverApi.CU_EVENT_DISABLE_TIMING).ThrowOnError();
            }
            log($"MoE expert streaming on: {(end - start) >> 20} MiB of mapped experts page-locked in place for GPU prefill.");
            return s;
        }
        catch
        {
            s.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Allocates the ring on first use, if free VRAM allows it with <see cref="RingMarginBytes"/> to spare, and measures the
    /// link once. False (logged once) when it does not fit: the caller then keeps the layer on the CPU.
    /// </summary>
    internal bool EnsureRing()
    {
        if (_ring != 0) return true;
        if (_ringFailed) return false;
        long ringBytes = (long)Slots * _wave * ExpertBytes;
        CudaDriverApi.cuMemGetInfo_v2(out nuint free, out _).ThrowOnError();
        if ((long)free < ringBytes + RingMarginBytes
            || CudaDriverApi.cuMemAlloc_v2(out _ring, (nuint)ringBytes) != 0)
        {
            _ring = 0;
            _ringFailed = true;
            _log($"MoE expert streaming off: the {ringBytes >> 20} MiB staging ring does not fit ({(long)free >> 20} MiB free).");
            return false;
        }

        // Link rate from a ring-sized copy of the first registered pages. The first DMA of a page pays for locking it
        // (registration is lazy), so the rate comes from a second pass over the same pages.
        nint src = _registered.Count > 0 ? _registered[0] : 0;
        for (int pass = 0; pass < 2; pass++)
        {
            var sw = Stopwatch.StartNew();
            if (CudaDriverApi.cuMemcpyHtoDAsync_v2(_ring, src, (nuint)ringBytes, _copyStream) != 0
                || CudaDriverApi.cuStreamSynchronize(_copyStream) != 0)
                break;
            _linkBytesPerMs = ringBytes / Math.Max(sw.Elapsed.TotalMilliseconds, 1e-3);
        }
        return true;
    }

    /// <summary>Experts of <paramref name="layer"/> whose copy is already in flight (prefetched), counted from expert 0.</summary>
    internal int PrefetchedEnd(MoeLayerWeights layer)
    {
        while (_prefetched.Count > 0 && !ReferenceEquals(_prefetched.Peek().Layer, layer))
            _prefetched.Dequeue();   // a stale prefetch (the forward did not reach that layer): its slot is simply reused
        int end = 0;
        foreach (var p in _prefetched) end = p.End;
        return end;
    }

    /// <summary>
    /// Chooses the split point: experts <c>[0, split)</c> on the GPU, <c>[split, E)</c> on the CPU, minimising the slower
    /// leg under the cost model (link time of the GPU experts with rows vs CPU time of the CPU assignments).
    /// </summary>
    /// <param name="offsets">Per-expert row offsets (assignment counts as a prefix sum over all experts).</param>
    /// <param name="minSplit">Experts already prefetched: they stay on the GPU.</param>
    internal int PlanSplit(ReadOnlySpan<int> offsets, int minSplit)
    {
        if (TestSplit is { } forced)
            return Math.Clamp(forced(_numExperts), minSplit, _numExperts);
        return PlanSplit(offsets, _numExperts, minSplit, ExpertBytes / _linkBytesPerMs, _cpuMsPerAssignment);
    }

    /// <summary>
    /// The split that minimises <c>max(GPU, CPU)</c>: the GPU leg costs <paramref name="expertMs"/> per expert with rows in
    /// <c>[0, split)</c> (its link time), the CPU leg <paramref name="cpuMsPerAssignment"/> per assignment in <c>[split, E)</c>.
    /// Ties go to the smaller split (less link traffic).
    /// </summary>
    /// <param name="offsets">Assignment prefix sums per expert, <c>[numExperts + 1]</c>.</param>
    /// <param name="numExperts">Experts.</param>
    /// <param name="minSplit">Lowest allowed split (experts already prefetched).</param>
    /// <param name="expertMs">Link time of one expert.</param>
    /// <param name="cpuMsPerAssignment">CPU time of one routed assignment.</param>
    /// <returns>The split point.</returns>
    internal static int PlanSplit(ReadOnlySpan<int> offsets, int numExperts, int minSplit, double expertMs, double cpuMsPerAssignment)
    {
        int total = offsets[numExperts];
        int best = minSplit;
        double bestCost = double.MaxValue;
        int active = 0;
        for (int s = 0; s <= numExperts; s++)
        {
            if (s > 0 && offsets[s] > offsets[s - 1]) active++;
            if (s < minSplit) continue;
            double cost = Math.Max(active * expertMs, (total - offsets[s]) * cpuMsPerAssignment);
            if (cost < bestCost)
            {
                bestCost = cost;
                best = s;
            }
        }
        return best;
    }

    /// <summary>Records a CPU leg run for the cost model.</summary>
    internal void ObserveCpu(double ms, int assignments)
    {
        if (assignments <= 0) return;
        _cpuMsPerAssignment = 0.7 * _cpuMsPerAssignment + 0.3 * (ms / assignments);
    }

    /// <summary>
    /// Makes experts <c>[begin, end)</c> of <paramref name="layer"/> resident in a slot (a prefetched wave, or a new copy
    /// of the experts that have rows), makes <paramref name="computeStream"/> wait for it, and points
    /// <see cref="Gate"/> / <see cref="Up"/> / <see cref="Down"/> at it. Pair with <see cref="EndWave"/>.
    /// </summary>
    internal int BeginWave(MoeLayerWeights layer, int begin, int end, ReadOnlySpan<int> offsets, nint computeStream)
    {
        int slot;
        if (_prefetched.Count > 0 && ReferenceEquals(_prefetched.Peek().Layer, layer) && _prefetched.Peek().Begin == begin)
        {
            var p = _prefetched.Dequeue();
            slot = p.Slot;
        }
        else
        {
            slot = EnqueueCopy(layer, begin, end, offsets);
        }

        CudaDriverApi.cuStreamWaitEvent(computeStream, _copyDone[slot], 0).ThrowOnError();
        nint b = SlotBase(slot);
        for (int e = begin; e < end; e++)
        {
            long j = e - begin;
            _gatePtr[e] = b + (nint)(j * _gateBytes);
            _upPtr[e] = b + (nint)(_wave * _gateBytes + j * _upBytes);
            _downPtr[e] = b + (nint)(_wave * (_gateBytes + _upBytes) + j * _downBytes);
        }
        return slot;
    }

    /// <summary>Marks the slot's compute as enqueued, so a later copy may reuse it.</summary>
    internal void EndWave(int slot, nint computeStream)
        => CudaDriverApi.cuEventRecord(_computeDone[slot], computeStream).ThrowOnError();

    /// <summary>
    /// Copies the first experts of the next CPU layer ahead of its routing (all of them, used or not: a large batch
    /// touches nearly all), so the link keeps working through that layer's attention.
    /// </summary>
    /// <param name="layer">The next CPU-expert layer.</param>
    /// <param name="experts">Experts to prefetch, from expert 0; rounded down to whole waves, at most the ring.</param>
    internal void Prefetch(MoeLayerWeights layer, int experts)
    {
        PrefetchedEnd(layer);   // drops a stale queue
        if (_prefetched.Count > 0) return;
        int waves = Math.Min(Slots, experts / _wave);
        for (int w = 0; w < waves; w++)
        {
            int begin = w * _wave, end = Math.Min(begin + _wave, _numExperts);
            _prefetched.Enqueue((layer, EnqueueCopy(layer, begin, end, default), begin, end));
        }
    }

    /// <summary>
    /// Uploads the CPU part's routed sum (F32 host rows) as F16 on <paramref name="computeStream"/> and adds it to
    /// <paramref name="outputF32"/>. Rows are divided by a power of two first when their max |x| exceeds 16384 (Flint's MoE
    /// output reaches 3.3e4, #1191), and the device multiplies it back: power-of-two scaling is exact, so the F16 copy
    /// cannot overflow and rounds as an unscaled one would (subnormals aside).
    /// </summary>
    internal void AddCpuPart(ReadOnlySpan<float> rows, nint outputF32, CudaKernels kernels, nint computeStream)
    {
        long n = rows.Length;
        if (n > _cpuOutElems)
        {
            // Last upload may still be in flight from the old buffers.
            CudaDriverApi.cuStreamSynchronize(computeStream).ThrowOnError();
            if (_cpuOutHost != 0) CudaDriverApi.cuMemFreeHost(_cpuOutHost);
            if (_cpuOutDevice != 0) CudaDriverApi.cuMemFree_v2(_cpuOutDevice);
            _cpuOutHost = _cpuOutDevice = 0;
            long cap = (long)System.Numerics.BitOperations.RoundUpToPowerOf2((ulong)n);
            CudaDriverApi.cuMemHostAlloc(out _cpuOutHost, (nuint)(cap * sizeof(ushort)), 0).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _cpuOutDevice, (nuint)(cap * sizeof(ushort))).ThrowOnError();
            _cpuOutElems = cap;
        }

        // The previous layer's upload from these pinned rows finished at this layer's routing sync.
        var half = new Span<Half>((void*)_cpuOutHost, (int)n);
        float amax = System.Numerics.Tensors.TensorPrimitives.MaxMagnitude(rows);
        float factor = amax > CpuPartF16Limit ? MathF.Pow(2f, MathF.Ceiling(MathF.Log2(amax / CpuPartF16Limit))) : 1f;
        if (factor == 1f)
        {
            System.Numerics.Tensors.TensorPrimitives.ConvertToHalf(rows, half);
        }
        else
        {
            float[] tmp = System.Buffers.ArrayPool<float>.Shared.Rent((int)n);
            try
            {
                System.Numerics.Tensors.TensorPrimitives.Multiply(rows, 1f / factor, tmp.AsSpan(0, (int)n));
                System.Numerics.Tensors.TensorPrimitives.ConvertToHalf(tmp.AsSpan(0, (int)n), half);
            }
            finally
            {
                System.Buffers.ArrayPool<float>.Shared.Return(tmp);
            }
        }
        CudaDriverApi.cuMemcpyHtoDAsync_v2(_cpuOutDevice, _cpuOutHost, (nuint)(n * sizeof(ushort)), computeStream).ThrowOnError();
        kernels.LaunchMoeAddF16F32(outputF32, _cpuOutDevice, (int)n, factor, computeStream);
    }

    private nint SlotBase(int slot) => _ring + (nint)((long)slot * _wave * ExpertBytes);

    // Enqueues the copy of experts [begin, end) into the next slot; with offsets, only the experts that have rows.
    private int EnqueueCopy(MoeLayerWeights layer, int begin, int end, ReadOnlySpan<int> offsets)
    {
        long wave = _nextWave++;
        int slot = (int)(wave % Slots);
        if (wave >= Slots)
            CudaDriverApi.cuStreamWaitEvent(_copyStream, _computeDone[slot], 0).ThrowOnError();
        nint b = SlotBase(slot);
        CopyRuns(layer.GateExpsRaw, _gateBytes, b, begin, end, offsets);
        CopyRuns(layer.UpExpsRaw, _upBytes, b + (nint)(_wave * _gateBytes), begin, end, offsets);
        CopyRuns(layer.DownExpsRaw, _downBytes, b + (nint)(_wave * (_gateBytes + _upBytes)), begin, end, offsets);
        CudaDriverApi.cuEventRecord(_copyDone[slot], _copyStream).ThrowOnError();
        return slot;
    }

    // One copy per run of consecutive experts that have rows (all experts when offsets is empty).
    private void CopyRuns(nint bankBase, long bytes, nint dst, int begin, int end, ReadOnlySpan<int> offsets)
    {
        int e = begin;
        while (e < end)
        {
            if (!offsets.IsEmpty && offsets[e + 1] == offsets[e]) { e++; continue; }
            int runEnd = e + 1;
            while (runEnd < end && (offsets.IsEmpty || offsets[runEnd + 1] > offsets[runEnd])) runEnd++;
            CudaDriverApi.cuMemcpyHtoDAsync_v2(dst + (nint)((e - begin) * bytes), bankBase + (nint)(e * bytes),
                (nuint)((runEnd - e) * bytes), _copyStream).ThrowOnError();
            ExpertsStreamed += runEnd - e;
            e = runEnd;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_copyStream != 0)
        {
            CudaDriverApi.cuStreamSynchronize(_copyStream);
            CudaDriverApi.cuStreamDestroy_v2(_copyStream);
            _copyStream = 0;
        }
        foreach (nint p in _registered) CudaDriverApi.cuMemHostUnregister(p);
        _registered.Clear();
        for (int i = 0; i < Slots; i++)
        {
            if (_copyDone[i] != 0) CudaDriverApi.cuEventDestroy_v2(_copyDone[i]);
            if (_computeDone[i] != 0) CudaDriverApi.cuEventDestroy_v2(_computeDone[i]);
            _copyDone[i] = _computeDone[i] = 0;
        }
        if (_ring != 0) CudaDriverApi.cuMemFree_v2(_ring);
        if (_cpuOutDevice != 0) CudaDriverApi.cuMemFree_v2(_cpuOutDevice);
        if (_cpuOutHost != 0) CudaDriverApi.cuMemFreeHost(_cpuOutHost);
        _ring = _cpuOutDevice = _cpuOutHost = 0;
    }
}
