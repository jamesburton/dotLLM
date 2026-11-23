using System;
using System.Collections.Generic;
using System.IO;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Engine.KvCache.Codecs;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Diagnostic microbench (#1924), opt-in with <c>DOTLLM_TQ_PREFILL_BENCH=1</c>: the tensor-core in-place TurboQuant prefill kernel
/// against its q4_0 / q8_0 / F16 siblings (same body, different tile loader) at Qwen3-4B geometry (32 / 8 heads, head dim 128),
/// one 1024-row chunk at several history depths. Timed with CUDA events over back-to-back launches; outputs are not checked.
/// Codes are random (a constant code would make every table lookup a shared-memory broadcast and hide bank conflicts).
/// <c>DOTLLM_TQ_PREFILL_BENCH_OFFSETS</c> (comma list) overrides the offsets; <c>DOTLLM_TQ_PREFILL_BENCH_REPS</c> the rep count.
/// Framework (RTX 3060) numbers are relative only: quote the ratios.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaTurboQuantPrefillKernelBench
{
    private readonly ITestOutputHelper _out;

    public CudaTurboQuantPrefillKernelBench(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void TqPrefill_Vs_Q4_Q8_F16()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_TQ_PREFILL_BENCH") == "1", "set DOTLLM_TQ_PREFILL_BENCH=1");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string ptx = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "ptx"));
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptx);
        nint s = stream.Handle;

        const int numHeads = 32, numKvHeads = 8, hd = 128, seqQ = 1024;
        int[] offsets = ParseInts(Environment.GetEnvironmentVariable("DOTLLM_TQ_PREFILL_BENCH_OFFSETS")) ?? [0, 6144, 15360, 31744];
        int reps = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_TQ_PREFILL_BENCH_REPS"), out int rr) ? rr : 10;
        int maxKv = 0; foreach (int o in offsets) maxKv = Math.Max(maxKv, o + seqQ);
        long kvElems = (long)maxKv * numKvHeads * hd;

        var allocs = new List<nint>();
        nint Alloc(long bytes, byte fill)
        {
            CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(d, fill, (nuint)bytes).ThrowOnError();
            allocs.Add(d);
            return d;
        }
        nint UpRandom(long bytes, int seed)
        {
            nint d = Alloc(bytes, 0);
            var host = new byte[bytes];
            new Random(seed).NextBytes(host);
            fixed (byte* p = host) CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)bytes).ThrowOnError();
            return d;
        }

        CudaDriverApi.cuEventCreate(out nint e0, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out nint e1, 0).ThrowOnError();
        double TimeUs(Action launch)
        {
            for (int i = 0; i < 2; i++) launch();
            CudaDriverApi.cuEventRecord(e0, s).ThrowOnError();
            for (int i = 0; i < reps; i++) launch();
            CudaDriverApi.cuEventRecord(e1, s).ThrowOnError();
            CudaDriverApi.cuEventSynchronize(e1).ThrowOnError();
            CudaDriverApi.cuEventElapsedTime(out float ms, e0, e1).ThrowOnError();
            return ms * 1000.0 / reps;
        }

        try
        {
            nint q = Alloc((long)seqQ * numHeads * hd * 2, 0x11);   // finite halves
            nint o = Alloc((long)seqQ * numHeads * hd * 2, 0);
            nint kF16 = Alloc(kvElems * 2, 0x11), vF16 = Alloc(kvElems * 2, 0x11);
            nint kQ8 = Alloc(kvElems / 32 * 34, 0x11), vQ8 = Alloc(kvElems / 32 * 34, 0x11);
            nint kQ4 = Alloc(kvElems / 32 * 18, 0x11), vQ4 = Alloc(kvElems / 32 * 18, 0x11);
            // F16 norms in [0.5, 2) (random per head vector, so the deferred-norm multiply sees real data).
            nint UpNorms(int seed)
            {
                var h = new Half[maxKv * numKvHeads + 2];
                var r = new Random(seed);
                for (int i = 0; i < h.Length; i++) h[i] = (Half)(0.5f + 1.5f * (float)r.NextDouble());
                nint d = Alloc(h.Length * 2L, 0);
                fixed (Half* p = h) CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)(h.Length * 2)).ThrowOnError();
                return d;
            }
            nint nK = UpNorms(5), nV = UpNorms(6);

            var tq = new Dictionary<int, (nint kc, nint vc, nint cent, int cu)>();
            foreach (int bits in new[] { 2, 3, 4, 8 })
            {
                var layout = new TurboQuantLayout(hd, numKvHeads, bits, TqNormFormat.F16);
                int cu = layout.CodeUintsPerVec;
                long codeBytes = (long)maxKv * numKvHeads * cu * 4 + 4;
                var codec = new TurboQuantCodec(hd, bits, 0x7A3C_91E5_2B44UL);
                float[] c = codec.Centroids.ToArray();
                nint dc = Alloc(c.Length * 4, 0);
                fixed (float* p = c) CudaDriverApi.cuMemcpyHtoD_v2(dc, (nint)p, (nuint)(c.Length * 4)).ThrowOnError();
                tq[bits] = (UpRandom(codeBytes, bits * 17), UpRandom(codeBytes, bits * 31), dc, cu);
            }

            _out.WriteLine($"Qwen3-4B geometry {numHeads}/{numKvHeads} hd{hd}, seqQ {seqQ}, reps {reps}; us per launch (ratio vs q4q4)");
            foreach (int off in offsets)
            {
                int seqKv = off + seqQ;
                var rows = new List<(string name, double us)>();
                rows.Add(("f16f16", TimeUs(() => kernels.LaunchAttentionFlashPrefillKvq(q, kF16, vF16, null, null, o, seqQ, seqKv,
                    numHeads, numKvHeads, hd, off, s))));
                rows.Add(("q8q8", TimeUs(() => kernels.LaunchAttentionFlashPrefillKvq(q, kQ8, vQ8, KvCacheDType.Q8_0, KvCacheDType.Q8_0, o,
                    seqQ, seqKv, numHeads, numKvHeads, hd, off, s))));
                double q4 = TimeUs(() => kernels.LaunchAttentionFlashPrefillKvq(q, kQ4, vQ4, KvCacheDType.Q4_0, KvCacheDType.Q4_0, o,
                    seqQ, seqKv, numHeads, numKvHeads, hd, off, s));
                rows.Add(("q4q4", q4));
                foreach (var (bits, t) in tq)
                {
                    foreach (var (variant, label) in TqVariants(kernels, hd, bits))
                    {
                        CudaKernels.TqPrefillVariantOverride = variant;
                        try
                        {
                            rows.Add(($"tq{bits} {label}", TimeUs(() => kernels.LaunchAttentionFlashPrefillKvTq(q, t.kc, t.vc, nK, nV, t.cent, o,
                                seqQ, seqKv, numHeads, numKvHeads, hd, off, bits, t.cu, TqNormFormat.F16, s))));
                        }
                        finally { CudaKernels.TqPrefillVariantOverride = null; }
                    }
                }
                foreach (var (name, us) in rows)
                    _out.WriteLine($"BENCH off={off,6} seqKv={seqKv,6} {name,-16} {us,10:F1} us  x{us / q4:F3}");
            }
        }
        finally
        {
            CudaDriverApi.cuEventDestroy_v2(e0);
            CudaDriverApi.cuEventDestroy_v2(e1);
            foreach (nint p in allocs) CudaDriverApi.cuMemFree_v2(p);
        }
    }

    /// <summary>The prefill variants to time for one bit width: always the legacy generic entry, plus the specialised one when loaded.</summary>
    private static IEnumerable<(CudaKernels.TqPrefillVariant, string)> TqVariants(CudaKernels k, int hd, int bits)
    {
        yield return (CudaKernels.TqPrefillVariant.Legacy, "legacy");
        if (k.HasAttentionFlashPrefillKvTqSpecialized(hd, bits)) yield return (CudaKernels.TqPrefillVariant.Specialized, "spec");
        if (k.HasAttentionFlashPrefillKvTqStaged(hd, bits)) yield return (CudaKernels.TqPrefillVariant.Staged, "cpa");
    }

    private static int[]? ParseInts(string? s)
    {
        if (string.IsNullOrWhiteSpace(s)) return null;
        var parts = s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var r = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++) r[i] = int.Parse(parts[i]);
        return r;
    }
}
