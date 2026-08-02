using System.Diagnostics;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Opt-in timing harness for issue #200 — compares per-decode-step wall time of the existing
/// gather-based dispatch (<see cref="CudaPagedKvCache.PrepareAttentionScratch"/> +
/// <see cref="CudaKernels.LaunchAttention"/>) against the new direct block-table-read kernel
/// (<see cref="CudaPagedKvCache.PrepareNativeBlockPtrs"/> + <see cref="CudaKernels.LaunchAttentionPaged"/>),
/// for the same growing single-sequence KV history.
/// </summary>
/// <remarks>
/// Correctness is covered by <see cref="CudaAttentionF16PagedTests"/>; this harness only reports
/// timing and does NOT gate on an absolute speedup threshold (GPU wall-clock timing is noisy
/// across machines/driver versions/thermal state, and this kernel is unvalidated/opt-in) — it
/// prints both wall times and the ratio so a human can judge whether eliminating the KV-byte
/// gather is winning as expected. Set <c>DOTLLM_CUDA_PERF=1</c> to run.
///
/// Deliberately a kernel-level microbenchmark, not a real-model end-to-end harness (unlike
/// <c>CudaBatchedDecodePerfHarness</c> for issue #251) — this isolates the one thing #200 changes
/// (the KV-read dispatch inside a single attention call) from everything else a full decode step
/// does, and does not require a downloaded GGUF model to run.
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaAttentionF16PagedPerfHarness
{
    private readonly ITestOutputHelper _output;
    public CudaAttentionF16PagedPerfHarness(ITestOutputHelper output) => _output = output;

    private const int BlockSize = 16; // matches CudaKvBlockPool's CPU-mirroring default

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    private static unsafe nint AllocAndFillDeviceFp16Random(Random rng, int count)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)(count * sizeof(ushort))).ThrowOnError();
        var host = new ushort[count];
        for (int i = 0; i < count; i++)
            host[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 2.0 - 1.0));
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(devPtr, (nint)p, (nuint)(count * sizeof(ushort))).ThrowOnError();
        return devPtr;
    }

    /// <summary>
    /// Real Bonsai-27B decode shape (numHeads=24, numKvHeads=4, headDim=256 — see
    /// <c>docs/CUDA.md</c>'s Future Work / this issue's motivating ncu investigation), grown to
    /// seqKv~=258 (the depth the original profiling used) across <c>BlockSize=16</c>-token pages
    /// (~17 blocks) so the gather path's D2D copy volume and the paged path's block-pointer-array
    /// H2D volume are both representative of a real decode step.
    /// </summary>
    [SkippableFact]
    public unsafe void MeasureGatherVsPagedNativeAttentionThroughput()
    {
        Skip.IfNot(
            string.Equals(Environment.GetEnvironmentVariable("DOTLLM_CUDA_PERF"), "1", StringComparison.Ordinal),
            "DOTLLM_CUDA_PERF=1 not set.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        const int numHeads = 24, numKvHeads = 4, headDim = 256;
        const int seqKv = 258;
        const int totalBlocks = 64;
        const int maxSeqLen = 512;
        const int iterations = 200;
        const int warmup = 20;

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasAttentionF16Paged, "attention_f16_paged not present in PTX (stale build)");

        int kvStride = numKvHeads * headDim;
        int qStride = numHeads * headDim;

        using var pool = new CudaKvBlockPool(numLayers: 1, numKvHeads, headDim, BlockSize, totalBlocks, ctx);
        using var cache = new CudaPagedKvCache(pool, maxSeqLen);

        var rng = new Random(0xFACADE);
        // Fill the cache in block-sized bursts (mirrors real prefill batching, not a token-at-a-time
        // loop, to keep harness setup fast — UpdateDevice's own correctness is covered elsewhere).
        int pos = 0;
        while (pos < seqKv)
        {
            int run = Math.Min(BlockSize, seqKv - pos);
            nint kRun = AllocAndFillDeviceFp16Random(rng, run * kvStride);
            nint vRun = AllocAndFillDeviceFp16Random(rng, run * kvStride);
            var positions = new int[run];
            for (int i = 0; i < run; i++) positions[i] = pos + i;
            cache.UpdateDevice(kRun, vRun, positions, run, layerIndex: 0, stream.Handle);
            CudaDriverApi.cuMemFree_v2(kRun);
            CudaDriverApi.cuMemFree_v2(vRun);
            pos += run;
        }
        stream.Synchronize();
        Assert.Equal(seqKv, cache.CurrentLength);

        nint dQ = AllocAndFillDeviceFp16Random(rng, qStride);
        CudaDriverApi.cuMemAlloc_v2(out nint dOut, (nuint)(qStride * sizeof(ushort))).ThrowOnError();

        try
        {
            nint s = stream.Handle;
            int positionOffset = seqKv - 1;

            void RunGather()
            {
                var (kPtr, vPtr) = cache.PrepareAttentionScratch(0, s);
                kernels.LaunchAttention(dQ, kPtr, vPtr, dOut,
                    seqQ: 1, seqKv, numHeads, numKvHeads, headDim, positionOffset, slidingWindow: 0, s);
            }

            void RunPagedNative()
            {
                var (kBlockPtrs, vBlockPtrs, _) = cache.PrepareNativeBlockPtrs(0, s);
                kernels.LaunchAttentionPaged(dQ, kBlockPtrs, vBlockPtrs, dOut,
                    seqQ: 1, seqKv, BlockSize, numHeads, numKvHeads, headDim, positionOffset, slidingWindow: 0, s);
            }

            for (int i = 0; i < warmup; i++) { RunGather(); RunPagedNative(); }
            stream.Synchronize();

            var swGather = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++) RunGather();
            stream.Synchronize();
            swGather.Stop();

            var swPaged = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++) RunPagedNative();
            stream.Synchronize();
            swPaged.Stop();

            double gatherUsPerCall = swGather.Elapsed.TotalMicroseconds / iterations;
            double pagedUsPerCall = swPaged.Elapsed.TotalMicroseconds / iterations;

            _output.WriteLine(
                $"shape: numHeads={numHeads} numKvHeads={numKvHeads} headDim={headDim} seqKv={seqKv} blockSize={BlockSize} " +
                $"(~{(seqKv + BlockSize - 1) / BlockSize} blocks)");
            _output.WriteLine($"gather-based (PrepareAttentionScratch + LaunchAttention): {gatherUsPerCall:F2} us/call");
            _output.WriteLine($"paged-native (PrepareNativeBlockPtrs + LaunchAttentionPaged): {pagedUsPerCall:F2} us/call");
            _output.WriteLine($"ratio (gather/paged): {gatherUsPerCall / pagedUsPerCall:F3}x " +
                "(>1 means paged-native is faster; honest report, no asserted threshold)");
        }
        finally
        {
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dOut != 0) CudaDriverApi.cuMemFree_v2(dOut);
        }
    }
}
