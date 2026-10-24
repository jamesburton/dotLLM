using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
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
/// </remarks>
internal sealed unsafe class CudaMoeHostExperts : IDisposable
{
    private readonly ComputeThreadPool? _pool;
    private readonly bool _packedLegacyExperts;
    private float* _host;          // [hidden rows | routed rows | top-k weights], F32
    private long _hostFloats;

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

    /// <summary>
    /// Runs the routed experts of one layer call on the CPU and writes their weighted sum to <paramref name="routedOutF32"/>.
    /// The caller has synchronised the stream after routing, so the device inputs are final.
    /// </summary>
    /// <param name="moe">The layer's CPU bundle (raw quant views into the mmap).</param>
    /// <param name="hiddenF32">Device F32 expert input <c>[seqLen, hidden]</c>.</param>
    /// <param name="topkWeightF32">Device F32 top-k weights <c>[seqLen, topK]</c>.</param>
    /// <param name="topkIdx">Host top-k expert ids <c>[seqLen, topK]</c>.</param>
    /// <param name="seqLen">Tokens in this call.</param>
    /// <param name="routedOutF32">Device F32 output <c>[seqLen, hidden]</c>; fully overwritten.</param>
    public void Run(DotLLM.Models.Architectures.MoeLayerWeights moe, nint hiddenF32, nint topkWeightF32,
        ReadOnlySpan<int> topkIdx, int seqLen, nint routedOutF32)
    {
        long t0 = Stopwatch.GetTimestamp();
        int hidden = moe.HiddenSize, k = moe.NumExpertsPerTok, numExperts = moe.NumExperts, inter = moe.IntermediateSize;
        int total = seqLen * k;
        long rowFloats = (long)seqLen * hidden;
        EnsureHost(2 * rowFloats + total);
        float* hIn = _host, hOut = _host + rowFloats, hW = _host + 2 * rowFloats;
        CudaDriverApi.cuMemcpyDtoH_v2((nint)hIn, hiddenF32, (nuint)(rowFloats * sizeof(float))).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoH_v2((nint)hW, topkWeightF32, (nuint)((long)total * sizeof(float))).ThrowOnError();

        // Bucket the assignments per expert (ascending id), the layout ExecuteRoutedFromAssignments expects.
        Span<int> cursors = stackalloc int[numExperts + 1];
        Span<int> unique = stackalloc int[numExperts];
        int[] bucketBuf = System.Buffers.ArrayPool<int>.Shared.Rent(2 * total);
        try
        {
            Span<int> bucketTokens = bucketBuf.AsSpan(0, total), bucketSlots = bucketBuf.AsSpan(total, total);
            cursors.Clear();
            foreach (int e in topkIdx[..total]) cursors[e + 1]++;
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
                int pos = fill[topkIdx[i]]++;
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

        CudaDriverApi.cuMemcpyHtoD_v2(routedOutF32, (nint)hOut, (nuint)(rowFloats * sizeof(float))).ThrowOnError();
        ComputeMicroseconds += (long)Stopwatch.GetElapsedTime(t0).TotalMicroseconds;
    }

    private void EnsureHost(long floats)
    {
        if (_hostFloats >= floats) return;
        if (_host != null) NativeMemory.AlignedFree(_host);
        long cap = Math.Max(floats, 2 * _hostFloats);
        _host = (float*)NativeMemory.AlignedAlloc((nuint)(cap * sizeof(float)), 64);
        _hostFloats = cap;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pool?.Dispose();
        if (_host != null) NativeMemory.AlignedFree(_host);
        _host = null;
        _hostFloats = 0;
    }
}
