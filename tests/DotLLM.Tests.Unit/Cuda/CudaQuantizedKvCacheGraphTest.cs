using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Direct equivalence checks for the graph-friendly variants on
/// <see cref="CudaQuantizedKvCache"/>. Compares the per-layer Q-cache + FP16 ring
/// byte-for-byte between the eager <c>UpdateDevice</c> path and <c>UpdateDeviceForGraph</c>, and
/// the in-place attention output (#1850: eager <c>attention_kvq</c> with the host quantized length vs
/// graph <c>attention_kvq_dyn</c> deriving it from the device position) bit for bit. Uses synthetic FP16 data so we can isolate the eviction state machine
/// from any model-loading concerns.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaQuantizedKvCacheGraphTest : IDisposable
{
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;
    private readonly ITestOutputHelper _out;

    public CudaQuantizedKvCacheGraphTest(ITestOutputHelper output)
    {
        _out = output;
        if (!CudaDevice.IsAvailable()) return;
        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        if (ptxDir != null)
            _kernels = new CudaKernels(ptxDir);
    }

    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }

    [SkippableFact]
    public void UpdateDeviceForGraph_UnderCudaGraph_MatchesEager()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");

        const int numLayers = 1;
        const int numKvHeads = 3;
        const int headDim = 64;
        const int kvStride = numKvHeads * headDim;
        const int maxSeqLen = 32;
        const int windowSize = 8;
        var cfg = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize);

        using var cacheEager = new CudaQuantizedKvCache(numLayers, numKvHeads, headDim, maxSeqLen, cfg);
        using var cacheGraph = new CudaQuantizedKvCache(numLayers, numKvHeads, headDim, maxSeqLen, cfg);

        nuint rowBytes = (nuint)(kvStride * sizeof(ushort));
        CudaDriverApi.cuMemAlloc_v2(out nint kRowDev, rowBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint vRowDev, rowBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint posDev, sizeof(int)).ThrowOnError();

        nint capturedGraph = 0;
        nint execGraph = 0;

        try
        {
            ushort[] hostRow = new ushort[kvStride];
            int[] positions = new[] { 0 };

            for (int pos = 0; pos < 16; pos++)
            {
                for (int i = 0; i < kvStride; i++)
                    hostRow[i] = FloatToHalf((pos * 0.01f) + (i * 0.001f));
                fixed (ushort* p = hostRow)
                    CudaDriverApi.cuMemcpyHtoD_v2(kRowDev, (nint)p, rowBytes).ThrowOnError();
                for (int i = 0; i < kvStride; i++)
                    hostRow[i] = FloatToHalf(1.0f + (pos * 0.02f) + (i * 0.002f));
                fixed (ushort* p = hostRow)
                    CudaDriverApi.cuMemcpyHtoD_v2(vRowDev, (nint)p, rowBytes).ThrowOnError();

                positions[0] = pos;

                cacheEager.UpdateDevice(kRowDev, vRowDev, positions, 1, 0, _stream!.Handle, _kernels!);

                // Capture once on the first iteration; replay thereafter.
                CudaDriverApi.cuMemcpyHtoD_v2(posDev, (nint)(&pos), sizeof(int)).ThrowOnError();
                if (execGraph == 0)
                {
                    CudaDriverApi.cuStreamBeginCapture_v2(_stream.Handle, CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).ThrowOnError();
                    cacheGraph.UpdateDeviceForGraph(kRowDev, vRowDev, 0, posDev, _stream.Handle, _kernels!);
                    CudaDriverApi.cuStreamEndCapture(_stream.Handle, out capturedGraph).ThrowOnError();
                    CudaDriverApi.cuGraphInstantiateWithFlags(out execGraph, capturedGraph, 0).ThrowOnError();
                }
                CudaDriverApi.cuGraphLaunch(execGraph, _stream.Handle).ThrowOnError();
                cacheGraph.AdvanceLengthForGraphDecode(pos + 1);
                _stream.Synchronize();

                CompareWindow(cacheEager, cacheGraph, kvStride, pos);
                CompareQuant(cacheEager, cacheGraph, kvStride, pos);
            }
        }
        finally
        {
            if (execGraph != 0) CudaDriverApi.cuGraphExecDestroy(execGraph);
            if (capturedGraph != 0) CudaDriverApi.cuGraphDestroy(capturedGraph);
            CudaDriverApi.cuMemFree_v2(kRowDev);
            CudaDriverApi.cuMemFree_v2(vRowDev);
            CudaDriverApi.cuMemFree_v2(posDev);
        }
    }

    [SkippableFact]
    public void UpdateDeviceForGraph_MatchesEager_AcrossEvictionBoundary()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");

        const int numLayers = 1;
        const int numKvHeads = 3;
        const int headDim = 64;
        const int kvStride = numKvHeads * headDim;     // 192
        const int maxSeqLen = 32;
        const int windowSize = 8;
        var cfg = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize);

        using var cacheEager = new CudaQuantizedKvCache(numLayers, numKvHeads, headDim, maxSeqLen, cfg);
        using var cacheGraph = new CudaQuantizedKvCache(numLayers, numKvHeads, headDim, maxSeqLen, cfg);

        // Allocate device scratch for K/V row inputs (FP16) and the device-resident pos.
        nuint rowBytes = (nuint)(kvStride * sizeof(ushort));
        CudaDriverApi.cuMemAlloc_v2(out nint kRowDev, rowBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint vRowDev, rowBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint posDev, sizeof(int)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint seqKvDev, sizeof(int)).ThrowOnError();
        const int numHeads = numKvHeads * 2;
        nuint qBytes = (nuint)(numHeads * headDim * sizeof(ushort));
        CudaDriverApi.cuMemAlloc_v2(out nint qDev, qBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint outEager, qBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint outGraph, qBytes).ThrowOnError();
        {
            var hq = new ushort[numHeads * headDim];
            for (int i = 0; i < hq.Length; i++) hq[i] = FloatToHalf(MathF.Sin(i * 0.37f));
            fixed (ushort* p = hq) CudaDriverApi.cuMemcpyHtoD_v2(qDev, (nint)p, qBytes).ThrowOnError();
        }

        try
        {
            // Drive both caches through positions 0..15 (crosses eviction at pos=8).
            // Each position writes a unique deterministic FP16 pattern.
            ushort[] hostRow = new ushort[kvStride];
            int[] positions = new[] { 0 };

            for (int pos = 0; pos < 16; pos++)
            {
                // Synthetic K row: index-dependent to surface any wrong slot writes.
                for (int i = 0; i < kvStride; i++)
                    hostRow[i] = FloatToHalf((pos * 0.01f) + (i * 0.001f));

                fixed (ushort* p = hostRow)
                    CudaDriverApi.cuMemcpyHtoD_v2(kRowDev, (nint)p, rowBytes).ThrowOnError();

                // Synthetic V row: differently keyed.
                for (int i = 0; i < kvStride; i++)
                    hostRow[i] = FloatToHalf(1.0f + (pos * 0.02f) + (i * 0.002f));
                fixed (ushort* p = hostRow)
                    CudaDriverApi.cuMemcpyHtoD_v2(vRowDev, (nint)p, rowBytes).ThrowOnError();

                positions[0] = pos;

                // Eager update.
                cacheEager.UpdateDevice(kRowDev, vRowDev, positions, 1, 0, _stream!.Handle, _kernels!);

                // Graph update: post pos to device and call the new method directly.
                CudaDriverApi.cuMemcpyHtoD_v2(posDev, (nint)(&pos), sizeof(int)).ThrowOnError();
                cacheGraph.UpdateDeviceForGraph(kRowDev, vRowDev, 0, posDev, _stream.Handle, _kernels!);
                cacheGraph.AdvanceLengthForGraphDecode(pos + 1);

                _stream.Synchronize();

                // Compare both caches: Q-cache, FP16 ring, scratch.
                CompareWindow(cacheEager, cacheGraph, kvStride, pos);
                CompareQuant(cacheEager, cacheGraph, kvStride, pos);

                // Attention over each cache, read in place: eager (host quantized length) vs graph (derived on the device).
                int currentLen = pos + 1;
                CudaDriverApi.cuMemcpyHtoD_v2(seqKvDev, (nint)(&currentLen), sizeof(int)).ThrowOnError();
                _kernels!.LaunchAttentionKvq(qDev, cacheEager.GetQuantizedKeysPtr(0), cacheEager.GetQuantizedValuesPtr(0),
                    cacheEager.GetWindowKeysPtr(0), cacheEager.GetWindowValuesPtr(0), outEager,
                    1, currentLen, numHeads, numKvHeads, headDim, pos, 0, _stream.Handle, 0,
                    KvCacheDType.Q8_0, KvCacheDType.Q8_0, cacheEager.QuantizedLength, windowSize);
                _kernels.LaunchAttentionKvqDyn(qDev, cacheGraph.GetQuantizedKeysPtr(0), cacheGraph.GetQuantizedValuesPtr(0),
                    cacheGraph.GetWindowKeysPtr(0), cacheGraph.GetWindowValuesPtr(0), outGraph,
                    1, seqKvDev, numHeads, numKvHeads, headDim, posDev, 0, _stream.Handle, 0,
                    KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize);
                _stream.Synchronize();
                CompareScratch(outEager, outGraph, numHeads * headDim, 1, $"attention pos={pos}");
            }
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(kRowDev);
            CudaDriverApi.cuMemFree_v2(vRowDev);
            CudaDriverApi.cuMemFree_v2(posDev);
            CudaDriverApi.cuMemFree_v2(seqKvDev);
            CudaDriverApi.cuMemFree_v2(qDev);
            CudaDriverApi.cuMemFree_v2(outEager);
            CudaDriverApi.cuMemFree_v2(outGraph);
        }
    }

    /// <summary>
    /// #1850 review: the windowed eager update batches the Phase-1 ring eviction and the Phase-3 ring write into one launch / copy per
    /// contiguous run, split where the ring wraps. Feeding the same rows one per call (every run is one row, so the batching is
    /// inert) and in uneven chunks (5/7/13/... rows: multi-row evictions and ring writes that wrap at slot 8) must leave the
    /// quantized rows and both FP16 rings byte-identical. Mixed Q8_0 K / Q4_0 V covers both quantize kernels.
    /// </summary>
    [SkippableFact]
    public void UpdateDevice_BatchedRingRuns_MatchOneRowPerCall()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");

        const int numKvHeads = 3, headDim = 64, kvStride = numKvHeads * headDim, maxSeqLen = 64, windowSize = 8, total = 45;
        int[] chunks = [5, 7, 13, 1, 3, 9, 7];
        Assert.Equal(total, chunks.Sum());
        var cfg = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q4_0, windowSize);
        using var perRow = new CudaQuantizedKvCache(1, numKvHeads, headDim, maxSeqLen, cfg);
        using var chunked = new CudaQuantizedKvCache(1, numKvHeads, headDim, maxSeqLen, cfg);

        long rowBytes = (long)kvStride * sizeof(ushort);
        var hk = new ushort[total * kvStride];
        var hv = new ushort[total * kvStride];
        for (int i = 0; i < hk.Length; i++)
        {
            hk[i] = FloatToHalf(MathF.Sin(i * 0.013f) * (1 + (i / kvStride) % 5));
            hv[i] = FloatToHalf(MathF.Cos(i * 0.029f) * (1 + (i / 32) % 7));
        }
        CudaDriverApi.cuMemAlloc_v2(out nint kAll, (nuint)(total * rowBytes)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint vAll, (nuint)(total * rowBytes)).ThrowOnError();
        try
        {
            fixed (ushort* p = hk) CudaDriverApi.cuMemcpyHtoD_v2(kAll, (nint)p, (nuint)(total * rowBytes)).ThrowOnError();
            fixed (ushort* p = hv) CudaDriverApi.cuMemcpyHtoD_v2(vAll, (nint)p, (nuint)(total * rowBytes)).ThrowOnError();

            for (int pos = 0; pos < total; pos++)
                perRow.UpdateDevice(kAll + (nint)(pos * rowBytes), vAll + (nint)(pos * rowBytes), [pos], 1, 0, _stream!.Handle, _kernels!);
            int start = 0;
            foreach (int n in chunks)
            {
                int[] positions = Enumerable.Range(start, n).ToArray();
                chunked.UpdateDevice(kAll + (nint)(start * rowBytes), vAll + (nint)(start * rowBytes), positions, n, 0, _stream!.Handle, _kernels!);
                start += n;
            }
            _stream!.Synchronize();

            Assert.Equal(total - windowSize, perRow.QuantizedLength);
            Assert.Equal(perRow.QuantizedLength, chunked.QuantizedLength);
            int quantLen = perRow.QuantizedLength;
            AssertSameBytes(perRow.GetQuantizedKeysPtr(0), chunked.GetQuantizedKeysPtr(0), (long)quantLen * perRow.KeyQuantizedRowBytes, "quantized K");
            AssertSameBytes(perRow.GetQuantizedValuesPtr(0), chunked.GetQuantizedValuesPtr(0), (long)quantLen * perRow.ValueQuantizedRowBytes, "quantized V");
            AssertSameBytes(perRow.GetWindowKeysPtr(0), chunked.GetWindowKeysPtr(0), windowSize * rowBytes, "K ring");
            AssertSameBytes(perRow.GetWindowValuesPtr(0), chunked.GetWindowValuesPtr(0), windowSize * rowBytes, "V ring");
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(kAll);
            CudaDriverApi.cuMemFree_v2(vAll);
        }
    }

    private static void AssertSameBytes(nint a, nint b, long bytes, string what)
    {
        var ha = new byte[bytes];
        var hb = new byte[bytes];
        fixed (byte* pa = ha) CudaDriverApi.cuMemcpyDtoH_v2((nint)pa, a, (nuint)bytes).ThrowOnError();
        fixed (byte* pb = hb) CudaDriverApi.cuMemcpyDtoH_v2((nint)pb, b, (nuint)bytes).ThrowOnError();
        int first = -1;
        for (int i = 0; i < ha.Length; i++) if (ha[i] != hb[i]) { first = i; break; }
        Assert.True(first < 0, $"{what} differs at byte {first} of {bytes}");
    }

    private void CompareWindow(CudaQuantizedKvCache eager, CudaQuantizedKvCache graph, int kvStride, int pos)
    {
        // Ring buffer compare for layer 0.
        nint eagerK = eager.GetWindowKeysPtr(0);
        nint graphK = graph.GetWindowKeysPtr(0);
        int windowSize = eager.WindowCapacity;
        long bytes = (long)windowSize * kvStride * sizeof(ushort);
        var a = new byte[bytes];
        var b = new byte[bytes];
        fixed (byte* pa = a) CudaDriverApi.cuMemcpyDtoH_v2((nint)pa, eagerK, (nuint)bytes).ThrowOnError();
        fixed (byte* pb = b) CudaDriverApi.cuMemcpyDtoH_v2((nint)pb, graphK, (nuint)bytes).ThrowOnError();
        Assert.True(a.AsSpan().SequenceEqual(b), $"Window K mismatch at pos={pos}");
    }

    private void CompareQuant(CudaQuantizedKvCache eager, CudaQuantizedKvCache graph, int kvStride, int pos)
    {
        nint eagerQ = eager.GetQuantizedKeysPtr(0);
        nint graphQ = graph.GetQuantizedKeysPtr(0);
        int quantLen = Math.Max(0, pos + 1 - eager.WindowCapacity);
        if (quantLen == 0) return;

        long rowBytes = eager.KeyQuantizedRowBytes;
        long bytes = (long)quantLen * rowBytes;
        var a = new byte[bytes];
        var b = new byte[bytes];
        fixed (byte* pa = a) CudaDriverApi.cuMemcpyDtoH_v2((nint)pa, eagerQ, (nuint)bytes).ThrowOnError();
        fixed (byte* pb = b) CudaDriverApi.cuMemcpyDtoH_v2((nint)pb, graphQ, (nuint)bytes).ThrowOnError();
        if (!a.AsSpan().SequenceEqual(b))
        {
            // Find first mismatch row.
            for (int row = 0; row < quantLen; row++)
            {
                int rowOff = (int)(row * rowBytes);
                if (!a.AsSpan(rowOff, (int)rowBytes).SequenceEqual(b.AsSpan(rowOff, (int)rowBytes)))
                {
                    _out.WriteLine($"Q-cache K row {row} differs at pos={pos}.");
                    break;
                }
            }
            Assert.Fail($"Quantized K mismatch at pos={pos}");
        }
    }

    private void CompareScratch(nint eager, nint graph, int kvStride, int currentLen, string tag)
    {
        long bytes = (long)currentLen * kvStride * sizeof(ushort);
        var a = new byte[bytes];
        var b = new byte[bytes];
        fixed (byte* pa = a) CudaDriverApi.cuMemcpyDtoH_v2((nint)pa, eager, (nuint)bytes).ThrowOnError();
        fixed (byte* pb = b) CudaDriverApi.cuMemcpyDtoH_v2((nint)pb, graph, (nuint)bytes).ThrowOnError();
        if (!a.AsSpan().SequenceEqual(b))
        {
            for (int row = 0; row < currentLen; row++)
            {
                int rowOff = row * kvStride * sizeof(ushort);
                int rowBytes = kvStride * sizeof(ushort);
                if (!a.AsSpan(rowOff, rowBytes).SequenceEqual(b.AsSpan(rowOff, rowBytes)))
                {
                    _out.WriteLine($"{tag}: row {row} differs.");
                    break;
                }
            }
            Assert.Fail($"{tag} mismatch");
        }
    }

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

    // Simple FP32 → IEEE 754 binary16. Round-to-nearest-even, no NaN/Inf handling
    // required for our deterministic test data.
    private static ushort FloatToHalf(float f)
    {
        uint x = *(uint*)&f;
        uint sign = (x >> 31) & 0x1;
        int exp = (int)((x >> 23) & 0xFF) - 127 + 15;
        uint mant = x & 0x7FFFFF;
        if (exp <= 0)
        {
            // Subnormal/zero: just zero out for our test patterns (small enough).
            if (exp < -10) return (ushort)(sign << 15);
            mant |= 0x800000;
            int shift = 14 - exp;
            uint half = mant >> shift;
            // Round-to-nearest with ties-to-even
            uint round = (mant >> (shift - 1)) & 1;
            half += round;
            return (ushort)((sign << 15) | half);
        }
        if (exp >= 31) return (ushort)((sign << 15) | (31 << 10));
        // Normal
        uint mantBits = mant >> 13;
        uint roundBit = (mant >> 12) & 1;
        uint sticky = mant & 0xFFF;
        if (roundBit != 0 && (sticky != 0 || (mantBits & 1) != 0))
        {
            mantBits++;
            if (mantBits == 0x400)  // mantissa overflow
            {
                mantBits = 0;
                exp++;
                if (exp >= 31) return (ushort)((sign << 15) | (31 << 10));
            }
        }
        return (ushort)((sign << 15) | ((uint)exp << 10) | mantBits);
    }
}
