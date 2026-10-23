// src/DotLLM.Cuda/Architectures/CudaNemotronHMoeColdCpu.cs
using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Moe;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Cold experts on the CPU at the latent cut (issue #1071; design #1063 section 8c). For an offloaded layer that may use the CPU:
/// <list type="number">
///   <item><c>latent_down</c> is enqueued before the router read-back event, and the router logits and the expert input
///   (<c>x_latent</c>, or the normed <c>h</c> on a layer without a latent) are both copied into pinned memory under that event.</item>
///   <item>After host routing the layer's used experts are split into a device set (cached, or copied in) and a CPU set
///   (<see cref="MoeOffloadState.Plan"/>); the device launches its groups only.</item>
///   <item>Concurrently, a worker pool computes the CPU set expert-parallel (one expert per worker at a time, each
///   single-threaded, with the CPU model's own kernels), writing each slot's output row into pinned memory.</item>
///   <item>When the CPU set is done its rows are copied to the device next to the device-computed slot rows, and the existing
///   gather kernel assembles the slot-ordered buffer the unchanged weighted sum reads. The summation order is therefore the
///   resident order; only the CPU experts' GEMV rounding differs from the device's.</item>
/// </list>
/// A layer call whose CPU set is empty skips all of this after the read-back (no extra sync, no gather).
/// </summary>
public sealed unsafe partial class CudaNemotronHTransformerModel
{
    /// <summary>
    /// Test hook: receives every MoE layer call's selection (layer index, the <c>[n · topK]</c> routed expert ids in slot order),
    /// resident or offloaded, so a test can prove two runs selected identically (#1071).
    /// </summary>
    internal static Action<int, int[]>? MoeRoutingTraceHook { get; set; }

    // Cold-leg scratch, allocated at load (with the rest of the MoE scratch) when an offloaded layer may use the CPU.
    private bool _moeColdScratch;
    private nint _moeSlotMerged;      // device [slots, maxExpertDim]: the gathered slot rows the weighted sum reads
    private nint _moeColdIdxDevice;   // device int[slots]: gather index (slot -> row of the extended _moeSlotOut)
    private nint _moeColdPinned;      // pinned host: input rows | CPU output rows | gather index
    private long _moeColdOutOffset, _moeColdIdxOffset;

    /// <summary>Pinned host copy of the layer call's expert input rows <c>[n, expertDim]</c> (valid after the router event).</summary>
    private float* ColdInputRows => (float*)_moeColdPinned;

    /// <summary>
    /// Sizes the cold-leg scratch for blocks of <paramref name="tokens"/> tokens: the pinned staging and the gather buffers.
    /// <see cref="AllocateMoeScratchCore"/> doubles <c>_moeSlotOut</c> when this is on, so the CPU rows can sit right after the
    /// device-computed ones (one source buffer for the gather).
    /// </summary>
    private void AllocateMoeColdScratch(int tokens)
    {
        var s = _moeSpec;
        long slots = (long)tokens * s.TopK;
        long inFloats = (long)tokens * s.ExpertDim, outFloats = slots * s.ExpertDim;
        CudaDriverApi.cuMemAlloc_v2(out _moeSlotMerged, (nuint)(Math.Max(outFloats, 1) * sizeof(float))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _moeColdIdxDevice, (nuint)(Math.Max(slots, 1) * sizeof(int))).ThrowOnError();
        _moeColdOutOffset = (inFloats * sizeof(float) + 255) & ~255L;
        _moeColdIdxOffset = (_moeColdOutOffset + outFloats * sizeof(float) + 255) & ~255L;
        CudaDriverApi.cuMemHostAlloc(out _moeColdPinned, (nuint)(_moeColdIdxOffset + Math.Max(slots, 1) * sizeof(int)), 0).ThrowOnError();
        _moeColdScratch = true;
    }

    private void FreeMoeColdScratch()
    {
        FreeIfNonZero(ref _moeSlotMerged);
        FreeIfNonZero(ref _moeColdIdxDevice);
        if (_moeColdPinned != 0) { CudaDriverApi.cuMemFreeHost(_moeColdPinned); _moeColdPinned = 0; }
        _moeColdScratch = false;
    }

    /// <summary>
    /// The CPU half of an offloaded layer call whose plan put <c>off.CpuGroups</c> experts on the CPU (the last groups of the
    /// schedule): runs them on the worker pool while the device works through its own groups, sends their slot rows to the
    /// device after the device-computed rows, and gathers both into slot order. Returns the slot buffer for the weighted sum.
    /// </summary>
    private nint RunCpuColdLeg(DeviceMoe moe, in MoeSchedule sched, int n, int expertDim, nint streamH)
    {
        var off = _offload!;
        int topK = moe.NumExpertsPerTok;
        int slots = n * topK;
        int firstCpu = sched.Groups - off.CpuGroups;
        byte* stage = (byte*)_moePinned + _moePinnedScheduleOffset;
        int* groupStart = (int*)(stage + sched.GroupStartOffset);
        int* slotIds = (int*)(stage + sched.SlotIdsOffset);
        int cpuStart = groupStart[firstCpu];
        int cpuRows = slots - cpuStart;
        float* outRows = (float*)((byte*)_moeColdPinned + _moeColdOutOffset);

        // The device's groups are queued; mark where they end so the stall of the stream on the CPU rows can be timed.
        var waitTiming = off.BeginCpuWait(streamH);

        long t0 = Stopwatch.GetTimestamp();
        off.CpuLeg!.Run(moe.Host, expertDim, topK, _moeGroupExperts, firstCpu, sched.Groups, groupStart, slotIds,
            ColdInputRows, outRows);
        MoeTelemetry.RecordCpuCompute((long)Stopwatch.GetElapsedTime(t0).TotalMicroseconds);

        int* idx = (int*)((byte*)_moeColdPinned + _moeColdIdxOffset);
        BuildColdGatherIndex(slotIds, cpuStart, slots, idx);

        // On the compute stream, after the device's groups: the CPU rows behind the device rows, the index, then the gather.
        // (The compute stream, not the copy stream: these few KB must not queue behind weight copies, and the weighted sum that
        // reads them is on the compute stream anyway.)
        nint rowsDst = _moeSlotOut + (nint)((long)slots * expertDim * sizeof(float));
        CudaDriverApi.cuMemcpyHtoDAsync_v2(rowsDst, (nint)outRows, (nuint)((long)cpuRows * expertDim * sizeof(float)), streamH).ThrowOnError();
        CudaDriverApi.cuMemcpyHtoDAsync_v2(_moeColdIdxDevice, (nint)idx, (nuint)((long)slots * sizeof(int)), streamH).ThrowOnError();
        off.EndCpuWait(waitTiming, streamH);
        _kernels.LaunchMoeGatherTokenRowsF32(_moeSlotOut, _moeSlotMerged, _moeColdIdxDevice, slots, expertDim, streamH);
        return _moeSlotMerged;
    }

    /// <summary>
    /// The gather index of a layer call with a CPU leg: slot s reads row <c>idx[s]</c> of the extended slot buffer, whose first
    /// <paramref name="slots"/> rows are the device-computed slot rows (indexed by slot) and whose next rows are the CPU rows, in
    /// schedule order (row <c>slots + (p − cpuStart)</c> holds the slot at schedule position p ≥ <paramref name="cpuStart"/>).
    /// </summary>
    /// <param name="slotIds">Slot ids in schedule order (the CPU groups last, from position <paramref name="cpuStart"/>).</param>
    /// <param name="cpuStart">First schedule position of the CPU groups.</param>
    /// <param name="slots">Slots of the call (<c>n · topK</c>).</param>
    /// <param name="idx">Receives the index, <paramref name="slots"/> entries.</param>
    internal static void BuildColdGatherIndex(int* slotIds, int cpuStart, int slots, int* idx)
    {
        for (int s = 0; s < slots; s++) idx[s] = s;
        for (int p = cpuStart; p < slots; p++) idx[slotIds[p]] = slots + (p - cpuStart);
    }

    /// <summary>
    /// The CPU cold-leg workers. A layer call's CPU experts are handed out one at a time: each worker claims the next expert
    /// when it finishes one, so a slow expert does not hold up the others. Workers come from <see cref="Parallel.For(int, int, ParallelOptions, Action{int})"/>
    /// (the .NET pool's long-lived threads, the calling thread included) capped at the configured thread count.
    /// </summary>
    /// <remarks>
    /// Measured on the dev box (busy, owner's server running) for an empty 9-expert dispatch: <c>Parallel.For</c> 6–9 µs per
    /// layer at 4–8 threads; a dedicated spin-wait <c>ComputeThreadPool</c> 1.5 µs at 4 threads but 0.8–150 ms at 4–15 threads
    /// once its spinning workers lose their cores to other processes. The CPU leg therefore uses <c>Parallel.For</c>.
    /// </remarks>
    internal sealed class CpuColdLeg : IDisposable
    {
        private readonly int _threads;
        private readonly ParallelOptions _parallel;

        // Per-run state (read by the workers).
        private NemotronHMoeWeights _moe = null!;
        private int _expertDim, _topK, _first, _end;
        private int[] _experts = [];
        private int* _groupStart, _slotIds;
        private float* _inRows, _outRows;
        private int _next;
        private Exception? _error;

        /// <summary>Creates the workers.</summary>
        /// <param name="threads">Threads including the caller (1 = run on the caller only).</param>
        public CpuColdLeg(int threads)
        {
            _threads = Math.Max(1, threads);
            _parallel = new ParallelOptions { MaxDegreeOfParallelism = _threads };
        }

        /// <summary>Threads that compute experts (including the caller).</summary>
        public int Threads => _threads;

        /// <summary>
        /// Computes groups <c>[first, end)</c> of the schedule on the CPU. Group g is expert <c>experts[g]</c> over slot positions
        /// <c>[groupStart[g], groupStart[g + 1])</c> of <paramref name="slotIds"/>; the output row of position p goes to
        /// <c>outRows + (p − groupStart[first]) · expertDim</c>. Token of slot s = s / topK, its input row
        /// <c>inRows + token · expertDim</c>.
        /// </summary>
        public void Run(NemotronHMoeWeights moe, int expertDim, int topK, int[] experts, int first, int end, int* groupStart,
            int* slotIds, float* inRows, float* outRows)
        {
            _moe = moe;
            _expertDim = expertDim;
            _topK = topK;
            _experts = experts;
            _first = first;
            _end = end;
            _groupStart = groupStart;
            _slotIds = slotIds;
            _inRows = inRows;
            _outRows = outRows;
            _next = first;
            _error = null;

            int workers = Math.Min(_threads, end - first);
            if (workers <= 1)
                Work();
            else
                Parallel.For(0, workers, _parallel, _ => Work());

            if (_error is { } e)
                throw new InvalidOperationException("The CPU cold-expert leg failed (#1071).", e);
        }

        private void Work()
        {
            float[]? rented = null;
            try
            {
                int ed = _expertDim, inter = _moe.MoeIntermediateSize;
                int cpuStart = _groupStart[_first];
                int g;
                while ((g = Interlocked.Increment(ref _next) - 1) < _end)
                {
                    int p0 = _groupStart[g], cnt = _groupStart[g + 1] - p0;
                    int need = checked(cnt * (ed + inter));
                    if (rented is null || rented.Length < need)
                    {
                        if (rented is not null) System.Buffers.ArrayPool<float>.Shared.Return(rented);
                        rented = System.Buffers.ArrayPool<float>.Shared.Rent(need);
                    }
                    fixed (float* scratch = rented)
                    {
                        float* gIn = scratch, gMid = scratch + (long)cnt * ed;
                        for (int j = 0; j < cnt; j++)
                        {
                            int token = _slotIds[p0 + j] / _topK;
                            new ReadOnlySpan<float>(_inRows + (long)token * ed, ed).CopyTo(new Span<float>(gIn + (long)j * ed, ed));
                        }
                        NemotronHMoeExpertCompute.Compute(_moe, _experts[g], ed, gIn, cnt, gMid,
                            _outRows + (long)(p0 - cpuStart) * ed, pool: null);
                    }
                }
            }
            catch (Exception ex)
            {
                Interlocked.CompareExchange(ref _error, ex, null);
                Volatile.Write(ref _next, _end);   // stop handing out work
            }
            finally
            {
                if (rented is not null) System.Buffers.ArrayPool<float>.Shared.Return(rented);
            }
        }

        public void Dispose()
        {
            // Nothing long-lived: the workers belong to the .NET thread pool.
        }
    }

    // ───────────────────────── calibration (#1071, design 6) ─────────────────────────

    /// <summary>
    /// The cost model of the offloaded layers: <see cref="MoeOffloadOptions.CostModelOverride"/>, else the cached calibration of this
    /// machine and file, else a fresh measurement (pinned H2D bandwidth, the D2H + sync round trip, CPU time of one expert per
    /// layout at n = 1 and 8), written back to the cache. Null when it cannot be measured (the caller then copies every miss).
    /// </summary>
    private MoeCostModel? CalibrateMoeCold(NemotronHLayerWeights[] cpuLayers, MoeOffloadOptions options, int cpuThreads,
        nint copyStream, Action<string> log)
    {
        if (options.CostModelOverride is { } fixedModel)
        {
            log("MoE cold experts: using the caller's cost model (calibration skipped).");
            return fixedModel;
        }

        // One representative expert per layout: the first layer that has it.
        var samples = new List<(ExpertLayout Layout, NemotronHMoeWeights Moe)>();
        foreach (var l in cpuLayers)
            if (l.Moe is { } m && samples.All(s => s.Layout != new ExpertLayout(m.UpPerExpertBytes, m.DownPerExpertBytes)))
                samples.Add((new ExpertLayout(m.UpPerExpertBytes, m.DownPerExpertBytes), m));
        if (samples.Count == 0) return null;

        string gpu = CudaDevice.GetDevice(_deviceId).Name;
        byte[] bus = new byte[32];
        string pci = CudaDriverApi.cuDeviceGetPCIBusId(bus, bus.Length, _deviceId) == 0
            ? System.Text.Encoding.ASCII.GetString(bus).TrimEnd('\0') : "unknown";
        var key = MoeCalibrationKey.ForFile(gpu, pci, options.ModelPath, cpuThreads);
        string path = options.CalibrationPath ?? MoeCalibrationStore.DefaultPath;

        if (!options.Calibrate && MoeCalibrationStore.TryLoad(path, key) is { } cached && cached.Covers(samples.Select(s => s.Layout)))
        {
            log($"MoE cold experts: cached calibration ({cached.MeasuredUtc:u}): {cached.Describe()}");
            return cached.ToCostModel();
        }

        try
        {
            long t0 = Stopwatch.GetTimestamp();
            double h2d = MeasureH2D(copyStream);
            double rtt = MeasureRoundTrip();
            var timings = new List<MoeLayoutTiming>();
            int hidden = Config.HiddenSize;
            foreach (var (layout, moe) in samples)
            {
                var (us1, us8) = MeasureCpuExpert(moe, NemotronHMoeExpertCompute.ExpertDim(moe, hidden));
                timings.Add(new MoeLayoutTiming(layout.UpBytes, layout.DownBytes, us1, us8));
            }
            var record = new MoeCalibrationRecord(h2d, rtt, timings, DateTime.UtcNow);
            bool saved = MoeCalibrationStore.TrySave(path, key, record);
            log($"MoE cold experts: calibrated in {Stopwatch.GetElapsedTime(t0).TotalMilliseconds:F0} ms: {record.Describe()}" +
                (saved ? $" (cached in {path})" : " (cache not writable)"));
            return record.ToCostModel();
        }
        catch (Exception ex) when (ex is CudaException or InvalidOperationException or OutOfMemoryException)
        {
            log($"MoE cold experts: calibration failed ({ex.Message}); copying every miss instead (--moe-cold copy).");
            return null;
        }
    }

    /// <summary>Pinned host-to-device bandwidth: a 16 MiB copy, four times after a warm-up, on the copy stream.</summary>
    private static double MeasureH2D(nint stream)
    {
        const long Bytes = 16L * 1024 * 1024;
        nint host = 0, dev = 0, start = 0, end = 0;
        try
        {
            CudaDriverApi.cuMemHostAlloc(out host, (nuint)Bytes, 0).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dev, (nuint)Bytes).ThrowOnError();
            CudaDriverApi.cuEventCreate(out start, CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();
            CudaDriverApi.cuEventCreate(out end, CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();
            new Span<byte>((void*)host, (int)Bytes).Fill(1);
            CudaDriverApi.cuMemcpyHtoDAsync_v2(dev, host, (nuint)Bytes, stream).ThrowOnError();   // warm-up
            CudaDriverApi.cuEventRecord(start, stream).ThrowOnError();
            for (int i = 0; i < 4; i++)
                CudaDriverApi.cuMemcpyHtoDAsync_v2(dev, host, (nuint)Bytes, stream).ThrowOnError();
            CudaDriverApi.cuEventRecord(end, stream).ThrowOnError();
            CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
            CudaDriverApi.cuEventElapsedTime(out float ms, start, end).ThrowOnError();
            return 4.0 * Bytes / Math.Max(ms / 1000.0, 1e-9);
        }
        finally
        {
            if (start != 0) CudaDriverApi.cuEventDestroy_v2(start);
            if (end != 0) CudaDriverApi.cuEventDestroy_v2(end);
            if (dev != 0) CudaDriverApi.cuMemFree_v2(dev);
            if (host != 0) CudaDriverApi.cuMemFreeHost(host);
        }
    }

    /// <summary>
    /// The D2H + host-sync round trip the CPU leg adds per layer: a 4 KB read-back on the compute stream, an event, and its host
    /// synchronize; median of 50.
    /// </summary>
    private double MeasureRoundTrip()
    {
        const int Reps = 50;
        // 4 KB (one decode token's x_latent), or less when a tiny test model's router scratch is smaller.
        nuint bytes = (nuint)Math.Min(4096L, (long)_moeScratchTokens * Math.Max(_moeSpec.NumExperts, 1) * sizeof(float));
        nint streamH = _stream.Handle;
        var samples = new double[Reps];
        for (int i = -3; i < Reps; i++)
        {
            long t = Stopwatch.GetTimestamp();
            CudaDriverApi.cuMemcpyDtoHAsync_v2(_moePinned, _moeRouterLogits, bytes, streamH).ThrowOnError();
            CudaDriverApi.cuEventRecord(_moeEvent, streamH).ThrowOnError();
            CudaDriverApi.cuEventSynchronize(_moeEvent).ThrowOnError();
            if (i >= 0) samples[i] = Stopwatch.GetElapsedTime(t).TotalMicroseconds;
        }
        Array.Sort(samples);
        return samples[Reps / 2];
    }

    /// <summary>CPU microseconds (single-threaded, median of 10) of expert 0 of <paramref name="moe"/> at n = 1 and n = 8.</summary>
    private static (double Us1, double Us8) MeasureCpuExpert(NemotronHMoeWeights moe, int expertDim)
    {
        const int Reps = 10, N = 8;
        int inter = moe.MoeIntermediateSize;
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)((long)N * expertDim * sizeof(float)), 64);
        float* mid = (float*)NativeMemory.AlignedAlloc((nuint)((long)N * inter * sizeof(float)), 64);
        float* y = (float*)NativeMemory.AlignedAlloc((nuint)((long)N * expertDim * sizeof(float)), 64);
        try
        {
            var rng = new Random(1071);
            for (long i = 0; i < (long)N * expertDim; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            NemotronHMoeExpertCompute.Touch(moe, 0);
            return (Median(1), Median(N));

            double Median(int n)
            {
                NemotronHMoeExpertCompute.Compute(moe, 0, expertDim, x, n, mid, y);   // warm-up
                var t = new double[Reps];
                for (int r = 0; r < Reps; r++)
                {
                    long s = Stopwatch.GetTimestamp();
                    NemotronHMoeExpertCompute.Compute(moe, 0, expertDim, x, n, mid, y);
                    t[r] = Stopwatch.GetElapsedTime(s).TotalMicroseconds;
                }
                Array.Sort(t);
                return t[Reps / 2];
            }
        }
        finally
        {
            NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(mid);
            NativeMemory.AlignedFree(y);
        }
    }
}
