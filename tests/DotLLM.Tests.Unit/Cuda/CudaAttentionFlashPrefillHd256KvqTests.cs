using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1094: the head_dim-256 flash PREFILL kernel reading Q8_0 / Q4_0 KV blocks directly
/// (<c>attention_flash_mma_prefill_hd256_kvq_*</c>) versus the path it replaces -- dequantize the
/// whole K/V history into an F16 scratch (<see cref="CudaKernels.LaunchKvDequantToF16"/>) and run
/// the F16 kernel over it.
///
/// <para>
/// Both kernels share one body (<c>attention_flash_mma_prefill_hd256.cuh</c>) and the quantized
/// tile load produces each half with the same expression the eager dequant kernels use, so the
/// MMA inputs are identical and the outputs must match <b>bit for bit</b>. No tolerance.
/// </para>
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaAttentionFlashPrefillHd256KvqTests : IDisposable
{
    private const int NumHeads = 24;
    private const int NumKvHeads = 4;
    private const int HeadDim = 256;

    private readonly ITestOutputHelper _out;
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;

    /// <summary>Creates the CUDA context, stream and kernel set when a device is available.</summary>
    /// <param name="output">xUnit output sink.</param>
    public CudaAttentionFlashPrefillHd256KvqTests(ITestOutputHelper output)
    {
        _out = output;
        if (!CudaDevice.IsAvailable()) return;
        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        if (ptxDir != null) _kernels = new CudaKernels(ptxDir);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
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

    private static ushort[] RandomHalf(Random rng, long n, float magnitude)
    {
        var h = new ushort[n];
        for (long i = 0; i < n; i++)
            h[i] = BitConverter.HalfToUInt16Bits((Half)(float)((rng.NextDouble() * 2 - 1) * magnitude));
        return h;
    }

    private static int BlockBytes(KvCacheDType t) => t == KvCacheDType.Q8_0 ? 34 : 18;

    /// <summary>
    /// Random KV-format blocks: an F16 scale in [0.002, 0.032) then random payload bytes. Any byte
    /// pattern is a valid Q8_0 / Q4_0 (adjacent-pair nibble) payload, so this covers the full code
    /// range of both formats.
    /// </summary>
    private static byte[] RandomQuantBlocks(Random rng, long elems, KvCacheDType t, float scaleBoost)
    {
        long blocks = elems / 32;
        int bb = BlockBytes(t);
        var buf = new byte[blocks * bb];
        for (long b = 0; b < blocks; b++)
        {
            ushort sc = BitConverter.HalfToUInt16Bits((Half)((0.002f + (float)rng.NextDouble() * 0.03f) * scaleBoost));
            buf[b * bb] = (byte)(sc & 0xFF);
            buf[b * bb + 1] = (byte)(sc >> 8);
            for (int i = 2; i < bb; i++) buf[b * bb + i] = (byte)rng.Next(256);
        }
        return buf;
    }

    private static nint Upload<T>(T[] host) where T : unmanaged
    {
        nuint bytes = (nuint)((long)host.Length * sizeof(T));
        CudaDriverApi.cuMemAlloc_v2(out nint d, bytes).ThrowOnError();
        fixed (T* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, bytes).ThrowOnError();
        return d;
    }

    private static nint Alloc(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
        return d;
    }

    private static float[] Download(nint d, long n)
    {
        var h = new float[n];
        fixed (float* p = h)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, d, (nuint)(n * 4)).ThrowOnError();
        return h;
    }

    private static readonly (KvCacheDType K, KvCacheDType V)[] KvqCombos =
    {
        (KvCacheDType.Q8_0, KvCacheDType.Q8_0), (KvCacheDType.Q4_0, KvCacheDType.Q4_0),
        (KvCacheDType.Q8_0, KvCacheDType.Q4_0), (KvCacheDType.Q4_0, KvCacheDType.Q8_0),
    };

    /// <summary>All four K/V dtype pairs, crossed with the (seqQ, positionOffset) cases.</summary>
    public static IEnumerable<object[]> Cases()
    {
        (int SeqQ, int Off)[] shapes = { (1, 0), (16, 0), (17, 5), (512, 0), (513, 3600), (1000, 3600) };
        foreach (var (k, v) in KvqCombos)
            foreach (var (sq, off) in shapes)
                yield return new object[] { sq, off, k, v };
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void KvqPrefill_BitIdenticalToScratchDequant(int seqQ, int posOffset, KvCacheDType kDtype, KvCacheDType vDtype)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");
        Skip.IfNot(_kernels!.HasAttentionFlashPrefillHd256, "flash prefill hd256 kernel not loaded (pre-Ampere or stale PTX)");

        // Same PTX arch and module family as the F16 kernel: when that one loads, this one must too.
        Assert.True(_kernels.HasAttentionFlashPrefillHd256Kvq(kDtype, vDtype),
            $"flash prefill hd256 kvq kernel for {kDtype}/{vDtype} is not loaded");

        int seqKv = posOffset + seqQ;
        long qElems = (long)seqQ * NumHeads * HeadDim;
        long kvElems = (long)seqKv * NumKvHeads * HeadDim;
        var rng = new Random(1094 + seqQ * 31 + posOffset + (int)kDtype * 7 + (int)vDtype);

        // K scales boosted so q.k/16 has O(1) spread: softmax is non-trivial, so a wrong key,
        // wrong head or wrong block offset changes the output visibly.
        nint dQ = Upload(RandomHalf(rng, qElems, 1.7f));
        nint dKq = Upload(RandomQuantBlocks(rng, kvElems, kDtype, kDtype == KvCacheDType.Q8_0 ? 0.6f : 6f));
        nint dVq = Upload(RandomQuantBlocks(rng, kvElems, vDtype, 1f));
        nint dKs = Alloc(kvElems * 2), dVs = Alloc(kvElems * 2);
        nint dOutRef = Alloc(qElems * 4), dOutKvq = Alloc(qElems * 4);
        try
        {
            // Poison the output so an unwritten row cannot masquerade as a match.
            var poison = new float[qElems];
            Array.Fill(poison, float.NaN);
            fixed (float* p = poison)
            {
                CudaDriverApi.cuMemcpyHtoD_v2(dOutKvq, (nint)p, (nuint)(qElems * 4)).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoD_v2(dOutRef, (nint)p, (nuint)(qElems * 4)).ThrowOnError();
            }

            // Reference: today's path -- full-history dequant into F16 scratch, then the F16 kernel.
            _kernels.LaunchKvDequantToF16(dKq, kDtype, dKs, (int)kvElems, _stream!.Handle);
            _kernels.LaunchKvDequantToF16(dVq, vDtype, dVs, (int)kvElems, _stream.Handle);
            _kernels.LaunchAttentionFlashPrefillHd256(dQ, dKs, dVs, dOutRef,
                seqQ, seqKv, NumHeads, NumKvHeads, posOffset, _stream.Handle);
            _kernels.LaunchAttentionFlashPrefillHd256Kvq(dQ, dKq, dVq, kDtype, vDtype, dOutKvq,
                seqQ, seqKv, NumHeads, NumKvHeads, posOffset, _stream.Handle);
            _stream.Synchronize();

            var want = Download(dOutRef, qElems);
            var got = Download(dOutKvq, qElems);
            int diff = 0, nan = 0;
            double maxRef = 0;
            for (long i = 0; i < qElems; i++)
            {
                if (float.IsNaN(want[i])) nan++;
                maxRef = Math.Max(maxRef, Math.Abs(want[i]));
                if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(want[i])) diff++;
            }
            _out.WriteLine($"seqQ={seqQ} off={posOffset} {kDtype}/{vDtype}: bit-diff elements = {diff} / {qElems} (maxRef={maxRef:F3})");
            Assert.Equal(0, nan);          // the reference wrote every row, so the comparison is meaningful
            Assert.True(maxRef > 0, "degenerate reference (all zeros)");
            Assert.Equal(0, diff);
        }
        finally
        {
            foreach (var p in new[] { dQ, dKq, dVq, dKs, dVs, dOutRef, dOutKvq })
                CudaDriverApi.cuMemFree_v2(p);
        }
    }

    /// <summary>
    /// Microbenchmark, opt-in (<c>DOTLLM_FLASH_BENCH=1</c>): fused kvq prefill vs dequant-into-scratch
    /// + F16 prefill at seqQ = 512 over a deep quantized history. Printed, not asserted.
    /// </summary>
    /// <param name="seqKv">KV history length (including the 512 query rows).</param>
    [SkippableTheory]
    [Trait("Category", "Benchmark")]
    [InlineData(4096)]
    [InlineData(65536)]
    public void Bench_KvqPrefill_vs_ScratchDequant(int seqKv)
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_FLASH_BENCH"), "1", StringComparison.Ordinal),
            "set DOTLLM_FLASH_BENCH=1");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");
        Skip.IfNot(_kernels!.HasAttentionFlashPrefillHd256, "kernel not loaded");

        const int seqQ = 512;
        int posOffset = seqKv - seqQ;
        long qElems = (long)seqQ * NumHeads * HeadDim;
        long kvElems = (long)seqKv * NumKvHeads * HeadDim;
        var rng = new Random(seqKv);
        nint dQ = Upload(RandomHalf(rng, qElems, 1.7f));
        nint dOut = Alloc(qElems * 4);
        nint dKs = Alloc(kvElems * 2), dVs = Alloc(kvElems * 2);
        try
        {
            foreach (var (kd, vd) in KvqCombos)
            {
                Skip.IfNot(_kernels.HasAttentionFlashPrefillHd256Kvq(kd, vd), "kvq kernel not loaded");
                nint dKq = Upload(RandomQuantBlocks(rng, kvElems, kd, 1f));
                nint dVq = Upload(RandomQuantBlocks(rng, kvElems, vd, 1f));
                try
                {
                    double Time(Action launch, int iters)
                    {
                        launch(); _stream!.Synchronize();
                        double best = double.MaxValue;
                        for (int rep = 0; rep < 3; rep++)
                        {
                            var sw = Stopwatch.StartNew();
                            for (int i = 0; i < iters; i++) launch();
                            _stream.Synchronize();
                            best = Math.Min(best, sw.Elapsed.TotalMilliseconds / iters);
                        }
                        return best;
                    }

                    int iters = seqKv >= 16384 ? 3 : 10;
                    double scratch = Time(() =>
                    {
                        _kernels.LaunchKvDequantToF16(dKq, kd, dKs, (int)kvElems, _stream!.Handle);
                        _kernels.LaunchKvDequantToF16(dVq, vd, dVs, (int)kvElems, _stream.Handle);
                        _kernels.LaunchAttentionFlashPrefillHd256(dQ, dKs, dVs, dOut,
                            seqQ, seqKv, NumHeads, NumKvHeads, posOffset, _stream.Handle);
                    }, iters);
                    double f16Only = Time(() => _kernels.LaunchAttentionFlashPrefillHd256(dQ, dKs, dVs, dOut,
                        seqQ, seqKv, NumHeads, NumKvHeads, posOffset, _stream!.Handle), iters);
                    double fused = Time(() => _kernels.LaunchAttentionFlashPrefillHd256Kvq(dQ, dKq, dVq, kd, vd, dOut,
                        seqQ, seqKv, NumHeads, NumKvHeads, posOffset, _stream!.Handle), iters);
                    _out.WriteLine($"BENCH seqQ={seqQ} seqKv={seqKv} {kd}/{vd}: scratch-dequant+f16={scratch:F2} ms " +
                                   $"(f16 kernel alone {f16Only:F2} ms) fused-kvq={fused:F2} ms speedup={scratch / fused:F2}x");
                }
                finally
                {
                    CudaDriverApi.cuMemFree_v2(dKq);
                    CudaDriverApi.cuMemFree_v2(dVq);
                }
            }
        }
        finally
        {
            foreach (var p in new[] { dQ, dOut, dKs, dVs })
                CudaDriverApi.cuMemFree_v2(p);
        }
    }
}
