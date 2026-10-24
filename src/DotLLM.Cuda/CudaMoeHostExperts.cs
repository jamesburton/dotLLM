using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// CPU leg of the generic CUDA MoE host (<c>--cpu-moe</c> / <c>--n-cpu-moe</c>, #1187): computes one layer call's routed
/// experts on the CPU, reading the expert banks straight from the mmap'd GGUF (no pinned or host second copy of the
/// weights, #1156), with the CPU model's own kernels (<see cref="MoeSwiGluMlp.ExecuteRoutedFromAssignments"/>).
/// </summary>
/// <remarks>
/// Only activations cross the link: the F32 expert input and the top-k weights down (<c>seqLen × (hidden + topK)</c>
/// floats), the routed sum up (<c>seqLen × hidden</c>). The device keeps the router, the shared expert and everything else.
/// On multi-token calls <see cref="Stream"/> (when present) moves part of the experts to the GPU (#1202).
/// </remarks>
internal sealed unsafe class CudaMoeHostExperts : IDisposable
{
    private readonly ComputeThreadPool? _pool;
    private readonly bool _packedLegacyExperts;
    private float* _host;          // [hidden rows | routed rows | top-k weights], F32
    private long _hostFloats;
    private readonly List<GCHandle> _hostJobs = [];
    private Exception? _hostJobError;

    /// <summary>Bumped whenever the pinned activation buffer moves: a captured graph that copies through it is then stale.</summary>
    public int Generation { get; private set; }

    /// <summary>Creates the CPU leg.</summary>
    /// <param name="threads">Total CPU threads of the leg, including the forward's own thread (1 = no pool).</param>
    /// <param name="packedLegacyExperts">Use the packed x Q8_1 dot for legacy quant banks, as the CPU model does for Flint.</param>
    public CudaMoeHostExperts(int threads, bool packedLegacyExperts)
    {
        _pool = threads >= 2 ? new ComputeThreadPool(threads) : null;
        _packedLegacyExperts = packedLegacyExperts;
    }

    /// <summary>Routed experts computed on the CPU so far.</summary>
    public long ExpertsComputed { get; private set; }

    /// <summary>Host wall time of the CPU legs (download, compute, upload), in microseconds.</summary>
    public long ComputeMicroseconds { get; private set; }

    /// <summary>GPU leg for multi-token calls (#1202), or null when the experts cannot be streamed.</summary>
    public CudaMoeExpertStream? Stream { get; set; }

    /// <summary>
    /// Runs the routed experts of one layer call on the CPU (inputs from <see cref="EnqueueDownload"/>, already complete) and
    /// queues the upload of their weighted sum to <paramref name="routedOutF32"/> on <paramref name="stream"/>.
    /// </summary>
    /// <param name="moe">The layer's CPU bundle (raw quant views into the mmap).</param>
    /// <param name="topkIdx">Host top-k expert ids <c>[seqLen, topK]</c>.</param>
    /// <param name="seqLen">Tokens in this call.</param>
    /// <param name="routedOutF32">Device F32 output <c>[seqLen, hidden]</c>; fully overwritten.</param>
    /// <param name="stream">Stream the upload is queued on.</param>
    public void Run(DotLLM.Models.Architectures.MoeLayerWeights moe, ReadOnlySpan<int> topkIdx, int seqLen, nint routedOutF32,
        nint stream)
    {
        var rows = Compute(moe, topkIdx, seqLen, firstExpert: 0, out _);
        fixed (float* p = rows)
            CudaDriverApi.cuMemcpyHtoDAsync_v2(routedOutF32, (nint)p, (nuint)((long)rows.Length * sizeof(float)), stream).ThrowOnError();
    }

    /// <summary>
    /// Graph-capturable <see cref="Run"/> (#1210): queues the CPU computation as a host function on <paramref name="stream"/>
    /// (it reads the top-k ids and inputs that <see cref="EnqueueDownload"/> and the caller's id copy queued before it) and
    /// the upload of its result after it, so the host never waits. The job stays registered until <see cref="ReleaseHostJobs"/>,
    /// because a captured graph keeps calling it; an exception in it is kept for <see cref="ThrowHostJobError"/>.
    /// </summary>
    /// <param name="moe">The layer's CPU bundle.</param>
    /// <param name="hostTopkIdx">Pinned host top-k ids <c>[seqLen, topK]</c>, filled by a copy queued before this call.</param>
    /// <param name="seqLen">Tokens in the call.</param>
    /// <param name="routedOutF32">Device F32 output <c>[seqLen, hidden]</c>; fully overwritten.</param>
    /// <param name="stream">The stream.</param>
    public void EnqueueRun(DotLLM.Models.Architectures.MoeLayerWeights moe, nint hostTopkIdx, int seqLen, nint routedOutF32,
        nint stream)
    {
        var handle = GCHandle.Alloc(new HostJob(this, moe, hostTopkIdx, seqLen));
        _hostJobs.Add(handle);
        CudaDriverApi.cuLaunchHostFunc(stream, (nint)(delegate* unmanaged<nint, void>)&RunHostJob, GCHandle.ToIntPtr(handle))
            .ThrowOnError();
        long rowFloats = (long)seqLen * moe.HiddenSize;
        CudaDriverApi.cuMemcpyHtoDAsync_v2(routedOutF32, (nint)(_host + rowFloats), (nuint)(rowFloats * sizeof(float)), stream)
            .ThrowOnError();
    }

    /// <summary>Frees the jobs <see cref="EnqueueRun"/> registered. No graph that calls them may be launched afterwards.</summary>
    public void ReleaseHostJobs()
    {
        foreach (var h in _hostJobs) h.Free();
        _hostJobs.Clear();
    }

    /// <summary>Rethrows (once) an exception a host job raised on the driver thread.</summary>
    public void ThrowHostJobError()
    {
        var e = Interlocked.Exchange(ref _hostJobError, null);
        if (e is not null)
            throw new InvalidOperationException("The CPU leg of a captured MoE decode failed.", e);
    }

    [UnmanagedCallersOnly]
    private static void RunHostJob(nint userData)
    {
        var job = (HostJob)GCHandle.FromIntPtr(userData).Target!;
        try
        {
            int total = job.SeqLen * job.Moe.NumExpertsPerTok;
            job.Owner.Compute(job.Moe, new ReadOnlySpan<int>((void*)job.HostTopkIdx, total), job.SeqLen, firstExpert: 0, out _);
        }
        catch (Exception e)
        {
            // Nothing may unwind into the driver; the caller rethrows after the stream synchronises.
            Interlocked.CompareExchange(ref job.Owner._hostJobError, e, null);
        }
    }

    private sealed record HostJob(CudaMoeHostExperts Owner, DotLLM.Models.Architectures.MoeLayerWeights Moe, nint HostTopkIdx,
        int SeqLen);

    /// <summary>
    /// Queues the copies of the call's expert input and top-k weights into pinned host memory on <paramref name="stream"/>;
    /// the caller synchronises once for these and the top-k ids. The previous call's upload from the same memory is ordered
    /// before them on the stream.
    /// </summary>
    public void EnqueueDownload(DotLLM.Models.Architectures.MoeLayerWeights moe, nint hiddenF32, nint topkWeightF32, int seqLen,
        nint stream)
    {
        long rowFloats = (long)seqLen * moe.HiddenSize;
        int total = seqLen * moe.NumExpertsPerTok;
        EnsureHost(2 * rowFloats + total, stream);
        CudaDriverApi.cuMemcpyDtoHAsync_v2((nint)_host, hiddenF32, (nuint)(rowFloats * sizeof(float)), stream).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoHAsync_v2((nint)(_host + 2 * rowFloats), topkWeightF32, (nuint)((long)total * sizeof(float)),
            stream).ThrowOnError();
    }

    /// <summary>
    /// Computes the weighted routed sum of the experts with id <c>&gt;= <paramref name="firstExpert"/></c> on the CPU from
    /// the inputs <see cref="EnqueueDownload"/> fetched. Assignments to lower experts contribute nothing (they run on the GPU).
    /// </summary>
    /// <returns>The host F32 rows <c>[seqLen, hidden]</c>, valid until the next call.</returns>
    public ReadOnlySpan<float> Compute(DotLLM.Models.Architectures.MoeLayerWeights moe, ReadOnlySpan<int> topkIdx, int seqLen,
        int firstExpert, out int assignments)
    {
        long t0 = Stopwatch.GetTimestamp();
        int hidden = moe.HiddenSize, k = moe.NumExpertsPerTok, numExperts = moe.NumExperts, inter = moe.IntermediateSize;
        int total = seqLen * k;
        long rowFloats = (long)seqLen * hidden;
        float* hIn = _host, hOut = _host + rowFloats, hW = _host + 2 * rowFloats;

        // Bucket the assignments per expert (ascending id), the layout ExecuteRoutedFromAssignments expects. The routine
        // skips zero weights in its final sum, so the GPU's assignments are zeroed rather than removed.
        Span<int> cursors = stackalloc int[numExperts + 1];
        Span<int> unique = stackalloc int[numExperts];
        int[] bucketBuf = System.Buffers.ArrayPool<int>.Shared.Rent(2 * total);
        try
        {
            Span<int> bucketTokens = bucketBuf.AsSpan(0, total), bucketSlots = bucketBuf.AsSpan(total, total);
            cursors.Clear();
            assignments = 0;
            for (int i = 0; i < total; i++)
            {
                int e = topkIdx[i];
                if (e >= firstExpert)
                {
                    cursors[e + 1]++;
                    assignments++;
                }
                else
                {
                    hW[i] = 0f;
                }
            }
            int uniqueCount = 0;
            for (int e = 0; e < numExperts; e++)
            {
                if (cursors[e + 1] > 0) unique[uniqueCount++] = e;
                cursors[e + 1] += cursors[e];
            }
            Span<int> fill = stackalloc int[numExperts];
            cursors[..numExperts].CopyTo(fill);
            for (int i = 0; i < total; i++)
            {
                int e = topkIdx[i];
                if (e < firstExpert) continue;
                int pos = fill[e]++;
                bucketTokens[pos] = i / k;
                bucketSlots[pos] = i % k;
            }

            MoeSwiGluMlp.ExecuteRoutedFromAssignments(
                hidden: new ReadOnlySpan<float>(hIn, (int)rowFloats),
                gateExpsRawBase: moe.GateExpsRaw, gateExpsQt: moe.GateExpsRawQt,
                gateExpsRowBytes: Dequantize.RowByteSize((long)inter * hidden, moe.GateExpsRawQt), gateExpsF32Ptrs: default,
                upExpsRawBase: moe.UpExpsRaw, upExpsQt: moe.UpExpsRawQt,
                upExpsRowBytes: Dequantize.RowByteSize((long)inter * hidden, moe.UpExpsRawQt), upExpsF32Ptrs: default,
                downExpsRawBase: moe.DownExpsRaw, downExpsQt: moe.DownExpsRawQt,
                downExpsRowBytes: Dequantize.RowByteSize((long)hidden * inter, moe.DownExpsRawQt), downExpsF32Ptrs: default,
                assignExpert: topkIdx[..total], assignWeight: new ReadOnlySpan<float>(hW, total),
                bucketCursors: cursors, bucketTokens: bucketTokens, bucketSlots: bucketSlots,
                uniqueExperts: unique, uniqueExpertCount: uniqueCount,
                output: new Span<float>(hOut, (int)rowFloats),
                numExperts: numExperts, numExpertsPerTok: k, hiddenSize: hidden, intermediateSize: inter, seqLen: seqLen,
                sharedGateProj: default, sharedUpProj: default, sharedDownProj: default,
                sharedIntermediateSize: 0, sharedExpertGate: default,
                loraAdapter: null, loraLayer: -1,
                threadPool: _pool,
                packedLegacyExperts: _packedLegacyExperts);
            ExpertsComputed += uniqueCount;
        }
        finally
        {
            System.Buffers.ArrayPool<int>.Shared.Return(bucketBuf);
        }

        ComputeMicroseconds += (long)Stopwatch.GetElapsedTime(t0).TotalMicroseconds;
        return new ReadOnlySpan<float>(hOut, (int)rowFloats);
    }

    // Pinned, so the copies in and out are asynchronous. Activations only ([hidden rows | routed rows | top-k weights]).
    private void EnsureHost(long floats, nint stream)
    {
        if (_hostFloats >= floats) return;
        if (_host != null)
        {
            CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();   // the last upload may still read the old buffer
            CudaDriverApi.cuMemFreeHost((nint)_host);
        }
        long cap = Math.Max(floats, 2 * _hostFloats);
        CudaDriverApi.cuMemHostAlloc(out nint p, (nuint)(cap * sizeof(float)), 0).ThrowOnError();
        _host = (float*)p;
        _hostFloats = cap;
        Generation++;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        ReleaseHostJobs();
        Stream?.Dispose();
        Stream = null;
        _pool?.Dispose();
        if (_host != null) CudaDriverApi.cuMemFreeHost((nint)_host);
        _host = null;
        _hostFloats = 0;
    }
}
