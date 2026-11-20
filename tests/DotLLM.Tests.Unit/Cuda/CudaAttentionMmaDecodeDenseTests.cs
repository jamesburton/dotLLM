using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1922 (E1): kernel-level oracle and microbenchmark for the dense split-KV tensor-core decode kernel
/// (<c>attention_mma_decode_v2_dense.cu</c>), head dim 64 / 128, F16 output, normalised split partials (F16 default, F32
/// control). References: an F64 CPU attention over the same F16 inputs, and (quantized KV) the dense F16 entry over the
/// dequantized scratch, which must be bit-identical.
/// <para>
/// Fixture strength: every KV head and query head has distinct random data, numKvHeads &gt; 1 and group &gt; 1 (except the
/// MHA case), and K rows are scaled by a ramp over position so the per-split score maxima differ by more than 1 (the combine's
/// reweight is observable). seqKv values include ones that are not multiples of 16 or of the split.
/// </para>
/// <para>Microbench: <c>DOTLLM_ATTN_MICROBENCH=1</c> (optional <c>DOTLLM_ATTN_MICROBENCH_SEQKV</c>, <c>_ITERS</c>, <c>_OUT</c>).</para>
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaAttentionMmaDecodeDenseTests
{
    private readonly ITestOutputHelper _out;
    public CudaAttentionMmaDecodeDenseTests(ITestOutputHelper output) => _out = output;

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

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    private sealed class Rig : IDisposable
    {
        public CudaContext Ctx = null!;
        public CudaStream Stream = null!;
        public CudaKernels Kernels = null!;
        public void Dispose() { Stream.Dispose(); Kernels.Dispose(); Ctx.Dispose(); }
    }

    private static Rig? TrySetUp(out string? skip)
    {
        skip = null;
        if (!IsCudaDriverPresent()) { skip = "No CUDA GPU available"; return null; }
        string? ptxDir = FindPtxDir();
        if (ptxDir == null) { skip = "PTX files not found"; return null; }
        var rig = new Rig { Ctx = CudaContext.Create(0), Stream = CudaStream.Create() };
        rig.Kernels = new CudaKernels(ptxDir);
        return rig;
    }

    private static readonly KvCacheDType F16 = KvCacheDType.F32;   // FP16 on the GPU

    internal static ushort[] RandomHalf(Random rng, long n, float magnitude, float offset = 0f)
    {
        var h = new ushort[n];
        for (long i = 0; i < n; i++)
            h[i] = BitConverter.HalfToUInt16Bits((Half)(offset + (float)(rng.NextDouble() * 2.0 - 1.0) * magnitude));
        return h;
    }

    /// <summary>Random K with a per-position ramp (0.25 .. 1.75) so per-split score maxima differ.</summary>
    private static ushort[] RampedK(Random rng, int seqKv, int kvRow)
    {
        var h = new ushort[(long)seqKv * kvRow];
        for (int j = 0; j < seqKv; j++)
        {
            float s = 0.25f + 1.5f * j / Math.Max(1, seqKv - 1);
            for (int e = 0; e < kvRow; e++)
                h[(long)j * kvRow + e] = BitConverter.HalfToUInt16Bits((Half)((float)(rng.NextDouble() * 2.0 - 1.0) * s));
        }
        return h;
    }

    private static nint Upload<T>(T[] host) where T : unmanaged
    {
        long bytes = (long)host.Length * sizeof(T);
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
        fixed (T* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)bytes).ThrowOnError();
        return d;
    }

    private static nint Alloc(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)Math.Max(bytes, 16)).ThrowOnError();
        return d;
    }

    private static ushort[] DownloadHalf(nint src, int elems)
    {
        var host = new ushort[elems];
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, src, (nuint)(elems * 2)).ThrowOnError();
        return host;
    }

    private static float H(ushort b) => (float)BitConverter.UInt16BitsToHalf(b);

    internal static double[] CpuReference(ushort[] q, ushort[] k, ushort[] v, int seqKv, int numHeads, int numKvHeads, int hd)
    {
        var o = new double[numHeads * hd];
        var sc = new double[seqKv];
        int kvRow = numKvHeads * hd;
        int group = numHeads / numKvHeads;
        double scale = 1.0 / Math.Sqrt(hd);
        for (int h = 0; h < numHeads; h++)
        {
            int hk = h / group;
            double mx = double.NegativeInfinity;
            for (int j = 0; j < seqKv; j++)
            {
                double a = 0;
                for (int d = 0; d < hd; d++)
                    a += (double)H(q[h * hd + d]) * H(k[(long)j * kvRow + hk * hd + d]);
                a *= scale;
                sc[j] = a;
                if (a > mx) mx = a;
            }
            double sum = 0;
            for (int j = 0; j < seqKv; j++) { sc[j] = Math.Exp(sc[j] - mx); sum += sc[j]; }
            for (int j = 0; j < seqKv; j++)
            {
                double w = sc[j] / sum;
                for (int d = 0; d < hd; d++)
                    o[h * hd + d] += w * H(v[(long)j * kvRow + hk * hd + d]);
            }
        }
        return o;
    }

    /// <summary>One dense v2 launch; returns the F16 output bits. kvSplit 0 = the production heuristic.</summary>
    private static ushort[] RunDense(Rig rig, nint dQ, nint dK, nint dV, KvCacheDType kd, KvCacheDType vd,
        int seqKv, int numHeads, int numKvHeads, int hd, int kvSplit, out int splitUsed)
    {
        int maxSafe = rig.Kernels.MaxSafeAttentionMmaDecodeDense(numKvHeads, hd, kd, vd);
        Assert.True(maxSafe >= 1, $"maxSafe {maxSafe}");
        int split = kvSplit > 0 ? kvSplit : CudaKernels.ComputeAttentionMmaV2KvSplit(seqKv, maxSafe);
        Assert.True(split <= maxSafe, $"forced split {split} > maxSafe {maxSafe}");
        splitUsed = split;
        int pb = CudaKernels.AttentionMmaDecodeDensePartialBytes;
        nint dOut = Alloc(numHeads * hd * 2), dPm = Alloc(numHeads * split * 4), dPs = Alloc(numHeads * split * 4),
             dPo = Alloc((long)numHeads * split * hd * pb);
        try
        {
            rig.Kernels.LaunchAttentionMmaDecodeDense(dQ, dK, dV, kd, vd, dOut, seqKv, numHeads, numKvHeads, hd, split,
                dPm, dPs, dPo, rig.Stream.Handle);
            rig.Stream.Synchronize();
            return DownloadHalf(dOut, numHeads * hd);
        }
        finally { foreach (var p in new[] { dOut, dPm, dPs, dPo }) CudaDriverApi.cuMemFree_v2(p); }
    }

    public static IEnumerable<object[]> OracleCases()
    {
        var shapes = new (int nh, int nkv, int hd)[] { (32, 8, 128), (24, 8, 128), (32, 8, 64), (14, 2, 64), (8, 8, 128) };
        int[] seqs = { 1, 15, 16, 17, 127, 128, 129, 383, 385, 4097 };
        foreach (var (nh, nkv, hd) in shapes)
            foreach (int s in seqs)
                yield return new object[] { nh, nkv, hd, s, 0 };
        // Forced splits: uneven tails, more splits than 16-key tiles, and the long single-split case.
        yield return new object[] { 32, 8, 128, 32768, 0 };
        yield return new object[] { 32, 8, 128, 385, 3 };
        yield return new object[] { 32, 8, 128, 2051, 3 };
        yield return new object[] { 32, 8, 64, 100, 3 };
        yield return new object[] { 14, 2, 64, 4097, 7 };
    }

    [SkippableTheory]
    [MemberData(nameof(OracleCases))]
    public void Dense_F16_MatchesF64Reference(int numHeads, int numKvHeads, int hd, int seqKv, int forcedSplit)
    {
        using var rig = TrySetUp(out string? skip);
        Skip.If(rig == null, skip);
        bool prev = CudaKernels.DenseDecodeV2F32Partials;
        try
        {
            int kvRow = numKvHeads * hd;
            var rng = new Random(0x1922 ^ (seqKv * 31 + hd * 7 + numHeads));
            var q = RandomHalf(rng, numHeads * hd, 2.5f);
            var k = RampedK(rng, seqKv, kvRow);
            var v = RandomHalf(rng, (long)seqKv * kvRow, 1.0f);
            var r = CpuReference(q, k, v, seqKv, numHeads, numKvHeads, hd);
            nint dQ = Upload(q), dK = Upload(k), dV = Upload(v);
            try
            {
                foreach (bool p32 in new[] { false, true })
                {
                    CudaKernels.DenseDecodeV2F32Partials = p32;
                    Skip.IfNot(rig!.Kernels.HasAttentionMmaDecodeDense(hd, F16, F16), "dense v2 kernel not loaded (stale PTX)");
                    var o = RunDense(rig, dQ, dK, dV, F16, F16, seqKv, numHeads, numKvHeads, hd, forcedSplit, out int split);
                    double maxErr = 0, maxRel = 0;
                    for (int i = 0; i < r.Length; i++)
                    {
                        double e = Math.Abs(H(o[i]) - r[i]);
                        maxErr = Math.Max(maxErr, e);
                        maxRel = Math.Max(maxRel, e / (Math.Abs(r[i]) + 1e-3));
                        // Two F16 roundings at most (output; partial) plus the F16 P operand: 2e-3 relative + 1e-3 absolute.
                        Assert.True(e <= 1e-3 + 2e-3 * Math.Abs(r[i]),
                            $"p32={p32} split={split} elem {i}: {H(o[i])} vs {r[i]} ({numHeads}/{numKvHeads}/{hd} seqKv={seqKv})");
                    }
                    _out.WriteLine($"{numHeads}/{numKvHeads}/hd{hd} seqKv={seqKv,6} split={split} p32={p32}: max|err|={maxErr:E2} maxRel={maxRel:E2}");
                }
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(dQ); CudaDriverApi.cuMemFree_v2(dK); CudaDriverApi.cuMemFree_v2(dV);
            }
        }
        finally { CudaKernels.DenseDecodeV2F32Partials = prev; }
    }

    /// <summary>
    /// Flat scores (Q = 0, every weight equal) and |V| around 20 over one long split: an UNNORMALISED F16 partial would hold
    /// about 20 x 32768 = 655,360 and overflow (65,504); the normalised partial stays near 20. Guards the F16 partial design.
    /// </summary>
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(3)]
    public void Dense_F16Partial_DoesNotOverflow_OnFlatScoresAndLargeV(int forcedSplit)
    {
        using var rig = TrySetUp(out string? skip);
        Skip.If(rig == null, skip);
        bool prev = CudaKernels.DenseDecodeV2F32Partials;
        try
        {
            CudaKernels.DenseDecodeV2F32Partials = false;
            Skip.IfNot(rig!.Kernels.HasAttentionMmaDecodeDense(128, F16, F16), "dense v2 kernel not loaded");
            const int nh = 32, nkv = 8, hd = 128, seqKv = 32768;
            var rng = new Random(0x0F16);
            var q = new ushort[nh * hd];   // all +0.0
            var k = RandomHalf(rng, (long)seqKv * nkv * hd, 1.0f);
            var v = RandomHalf(rng, (long)seqKv * nkv * hd, 1.0f, offset: 20f);
            var r = CpuReference(q, k, v, seqKv, nh, nkv, hd);
            nint dQ = Upload(q), dK = Upload(k), dV = Upload(v);
            try
            {
                var o = RunDense(rig, dQ, dK, dV, F16, F16, seqKv, nh, nkv, hd, forcedSplit, out _);
                for (int i = 0; i < r.Length; i++)
                {
                    float g = H(o[i]);
                    Assert.True(float.IsFinite(g) && Math.Abs(g - r[i]) <= 0.05, $"elem {i}: {g} vs {r[i]}");
                }
            }
            finally { CudaDriverApi.cuMemFree_v2(dQ); CudaDriverApi.cuMemFree_v2(dK); CudaDriverApi.cuMemFree_v2(dV); }
        }
        finally { CudaKernels.DenseDecodeV2F32Partials = prev; }
    }

    private static byte[] RandomQuantBlocks(Random rng, long elems, KvCacheDType t)
    {
        int bb = t == KvCacheDType.Q8_0 ? 34 : 18;
        long blocks = elems / 32;
        var buf = new byte[blocks * bb];
        for (long b = 0; b < blocks; b++)
        {
            ushort sc = BitConverter.HalfToUInt16Bits((Half)(0.002f + (float)rng.NextDouble() * 0.06f));
            buf[b * bb] = (byte)(sc & 0xFF);
            buf[b * bb + 1] = (byte)(sc >> 8);
            for (int i = 2; i < bb; i++) buf[b * bb + i] = (byte)rng.Next(256);
        }
        return buf;
    }

    private static readonly (KvCacheDType K, KvCacheDType V)[] KvqCombos =
    {
        (KvCacheDType.Q8_0, KvCacheDType.Q8_0), (KvCacheDType.Q4_0, KvCacheDType.Q4_0),
        (KvCacheDType.Q8_0, KvCacheDType.Q4_0), (KvCacheDType.Q4_0, KvCacheDType.Q8_0),
    };

    /// <summary>Fused Q8_0 / Q4_0 loaders == dequantize into an F16 scratch + the dense F16 entry, bit for bit (both partial arms).</summary>
    [SkippableTheory]
    [InlineData(32, 8, 128, 17)]
    [InlineData(32, 8, 128, 4097)]
    [InlineData(14, 2, 64, 385)]
    [InlineData(32, 8, 64, 2051)]
    public void Dense_Quantized_BitIdenticalToDequantScratch(int numHeads, int numKvHeads, int hd, int seqKv)
    {
        using var rig = TrySetUp(out string? skip);
        Skip.If(rig == null, skip);
        bool prev = CudaKernels.DenseDecodeV2F32Partials;
        try
        {
            long kvElems = (long)seqKv * numKvHeads * hd;
            foreach (bool p32 in new[] { false, true })
                foreach (var (kd, vd) in KvqCombos)
                {
                    CudaKernels.DenseDecodeV2F32Partials = p32;
                    Skip.IfNot(rig!.Kernels.HasAttentionMmaDecodeDense(hd, kd, vd), "dense v2 kvq kernel not loaded");
                    var rng = new Random(0x1922 + seqKv + (int)kd * 7 + (int)vd + hd);
                    nint dQ = Upload(RandomHalf(rng, numHeads * hd, 2.5f));
                    nint dKq = Upload(RandomQuantBlocks(rng, kvElems, kd)), dVq = Upload(RandomQuantBlocks(rng, kvElems, vd));
                    nint dKs = Alloc(kvElems * 2), dVs = Alloc(kvElems * 2);
                    try
                    {
                        rig.Kernels.LaunchKvDequantToF16(dKq, kd, dKs, (int)kvElems, rig.Stream.Handle);
                        rig.Kernels.LaunchKvDequantToF16(dVq, vd, dVs, (int)kvElems, rig.Stream.Handle);
                        rig.Stream.Synchronize();
                        // The entries differ in register count, so their co-residency ceilings (and so the default split) can
                        // differ: pin both to the same split, the one both can launch.
                        int split = CudaKernels.ComputeAttentionMmaV2KvSplit(seqKv, Math.Min(
                            rig.Kernels.MaxSafeAttentionMmaDecodeDense(numKvHeads, hd, F16, F16),
                            rig.Kernels.MaxSafeAttentionMmaDecodeDense(numKvHeads, hd, kd, vd)));
                        var a = RunDense(rig, dQ, dKs, dVs, F16, F16, seqKv, numHeads, numKvHeads, hd, split, out int s1);
                        var b = RunDense(rig, dQ, dKq, dVq, kd, vd, seqKv, numHeads, numKvHeads, hd, split, out int s2);
                        int diff = 0, nonZero = 0;
                        for (int i = 0; i < a.Length; i++) { if (a[i] != b[i]) diff++; if ((a[i] & 0x7FFF) != 0) nonZero++; }
                        _out.WriteLine($"{numHeads}/{numKvHeads}/hd{hd} seqKv={seqKv} {kd}/{vd} p32={p32} split={s1}: bit-diff {diff}, non-zero {nonZero}/{a.Length}");
                        Assert.True(nonZero > a.Length / 2, "fixture produced a mostly-zero output");
                        Assert.Equal(0, diff);
                    }
                    finally { foreach (var p in new[] { dQ, dKq, dVq, dKs, dVs }) CudaDriverApi.cuMemFree_v2(p); }
                }
        }
        finally { CudaKernels.DenseDecodeV2F32Partials = prev; }
    }

    // ── TurboQuant in place (E3): the dense v2 TQ entry vs the scalar attention_kvtq on the same codes, norms and centroids. ──

    private sealed record TqFixture(uint[] KCodes, uint[] VCodes, ushort[] KNorms, ushort[] VNorms, float[] Centroids, int Cu);

    /// <summary>
    /// Random codes (every centroid used), per-(position, head) F16 norms that differ across keys and heads, and a sorted
    /// non-uniform centroid table scaled by 1/sqrt(hd) like the codec's.
    /// </summary>
    private static TqFixture MakeTq(Random rng, int seqKv, int nkv, int hd, int bits)
    {
        int cu = (hd * bits + 31) / 32;
        long vecs = (long)seqKv * nkv;
        var kc = new uint[vecs * cu]; var vc = new uint[vecs * cu];
        for (long i = 0; i < kc.Length; i++) { kc[i] = (uint)rng.NextInt64(0, 1L << 32); vc[i] = (uint)rng.NextInt64(0, 1L << 32); }
        var kn = new ushort[vecs]; var vn = new ushort[vecs];
        for (long i = 0; i < vecs; i++)
        {
            kn[i] = BitConverter.HalfToUInt16Bits((Half)(0.5f + 2.5f * (float)rng.NextDouble()));
            vn[i] = BitConverter.HalfToUInt16Bits((Half)(0.5f + 2.5f * (float)rng.NextDouble()));
        }
        int levels = 1 << bits;
        var c = new float[levels];
        for (int i = 0; i < levels; i++) c[i] = (float)((-1.0 + 2.0 * (i + 0.5) / levels + 0.15 * Math.Sin(i * 1.7)) * 2.5 / Math.Sqrt(hd));
        Array.Sort(c);
        return new TqFixture(kc, vc, kn, vn, c, cu);
    }

    private static ushort[] RunTqDense(Rig rig, nint dQ, nint kc, nint vc, nint kn, nint vn, nint cent, int seqKv, int nh, int nkv,
        int hd, int bits, int cu, int forcedSplit)
    {
        int maxSafe = rig.Kernels.MaxSafeAttentionMmaDecodeDenseTq(nkv, hd);
        Assert.True(maxSafe >= 1);
        int split = forcedSplit > 0 ? forcedSplit : CudaKernels.ComputeAttentionMmaV2KvSplit(seqKv, maxSafe);
        nint dOut = Alloc(nh * hd * 2), dPm = Alloc(nh * split * 4), dPs = Alloc(nh * split * 4),
             dPo = Alloc((long)nh * split * hd * CudaKernels.AttentionMmaDecodeDensePartialBytes);
        try
        {
            rig.Kernels.LaunchAttentionMmaDecodeDenseTq(dQ, kc, vc, kn, vn, cent, dOut, seqKv, nh, nkv, hd, split, dPm, dPs, dPo,
                bits, cu, DotLLM.Core.Attention.TqNormFormat.F16, rig.Stream.Handle);
            rig.Stream.Synchronize();
            return DownloadHalf(dOut, nh * hd);
        }
        finally { foreach (var p in new[] { dOut, dPm, dPs, dPo }) CudaDriverApi.cuMemFree_v2(p); }
    }

    private static ushort[] RunTqScalar(Rig rig, nint dQ, nint kc, nint vc, nint kn, nint vn, nint cent, int seqKv, int nh, int nkv,
        int hd, int bits, int cu)
    {
        nint dOut = Alloc(nh * hd * 2);
        try
        {
            rig.Kernels.LaunchAttentionKvTq(dQ, kc, vc, kn, vn, cent, dOut, 1, seqKv, nh, nkv, hd, seqKv - 1, 0, rig.Stream.Handle, 0,
                bits, cu, DotLLM.Core.Attention.TqNormFormat.F16);
            rig.Stream.Synchronize();
            return DownloadHalf(dOut, nh * hd);
        }
        finally { CudaDriverApi.cuMemFree_v2(dOut); }
    }

    public static IEnumerable<object[]> TqCases()
    {
        foreach (int bits in new[] { 1, 2, 3, 4, 5, 6, 7, 8 })
            yield return new object[] { 32, 8, 128, bits, 1037, 0 };
        yield return new object[] { 14, 2, 64, 3, 385, 3 };
        yield return new object[] { 14, 2, 64, 4, 4097, 0 };
        yield return new object[] { 16, 4, 256, 4, 300, 0 };
        yield return new object[] { 16, 4, 256, 3, 129, 2 };
        yield return new object[] { 32, 8, 128, 4, 8191, 0 };
    }

    /// <summary>
    /// E3 oracle: dense v2 TQ == scalar attention_kvtq within the F16 element-rounding budget, for tq1..tq8 (tq3/5/6/7 straddle
    /// words; tq7 needs the third word), hd 64/128/256, uneven splits. Substitution: the same codes with K and V swapped must move
    /// the output far outside that budget, so the fixture can see a K/V mix-up.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(TqCases))]
    public void DenseTq_MatchesScalarKvTq(int nh, int nkv, int hd, int bits, int seqKv, int forcedSplit)
    {
        using var rig = TrySetUp(out string? skip);
        Skip.If(rig == null, skip);
        Skip.IfNot(rig!.Kernels.HasAttentionMmaDecodeDenseTq(hd) && rig.Kernels.HasAttentionKvTq, "TQ kernels not loaded");
        var rng = new Random(0x7E3 + bits * 101 + hd + seqKv);
        var f = MakeTq(rng, seqKv, nkv, hd, bits);
        nint dQ = Upload(RandomHalf(rng, nh * hd, 2.0f));
        nint kc = Upload(f.KCodes), vc = Upload(f.VCodes), kn = Upload(f.KNorms), vn = Upload(f.VNorms), ce = Upload(f.Centroids);
        try
        {
            var r = RunTqScalar(rig, dQ, kc, vc, kn, vn, ce, seqKv, nh, nkv, hd, bits, f.Cu);
            var o = RunTqDense(rig, dQ, kc, vc, kn, vn, ce, seqKv, nh, nkv, hd, bits, f.Cu, forcedSplit);
            var swapped = RunTqDense(rig, dQ, vc, kc, vn, kn, ce, seqKv, nh, nkv, hd, bits, f.Cu, forcedSplit);
            double maxErr = 0, maxRef = 0, swapDiff = 0;
            for (int i = 0; i < r.Length; i++)
            {
                double e = Math.Abs(H(o[i]) - H(r[i]));
                maxErr = Math.Max(maxErr, e); maxRef = Math.Max(maxRef, Math.Abs(H(r[i])));
                swapDiff = Math.Max(swapDiff, Math.Abs(H(swapped[i]) - H(r[i])));
                // Measured worst 3.05e-5 (framework, first run): the scalar kernel and v2 differ only by the F16 rounding of each
                // decoded element and of P. Budget 1e-4 + 1e-3 |r| is ~3x that.
                Assert.True(e <= 1e-4 + 1e-3 * Math.Abs(H(r[i])), $"tq{bits} hd{hd} elem {i}: {H(o[i])} vs scalar {H(r[i])}");
            }
            _out.WriteLine($"tq{bits} {nh}/{nkv}/hd{hd} seqKv={seqKv}: max|v2-scalar|={maxErr:E2} max|ref|={maxRef:E2} swapped-KV diff={swapDiff:E2}");
            Assert.True(swapDiff > 20 * (1e-4 + 1e-3 * maxRef), $"substitution: swapping K and V moved the output only {swapDiff:E2}");
        }
        finally { foreach (var p in new[] { dQ, kc, vc, kn, vn, ce }) CudaDriverApi.cuMemFree_v2(p); }
    }

    /// <summary>The C# wrapper must refuse a kvSplit above the co-residency ceiling before launching (a hard CUDA error otherwise).</summary>
    [SkippableFact]
    public void Dense_RejectsSplitAboveCoResidencyCeiling()
    {
        using var rig = TrySetUp(out string? skip);
        Skip.If(rig == null, skip);
        Skip.IfNot(rig!.Kernels.HasAttentionMmaDecodeDense(128, F16, F16), "dense v2 kernel not loaded");
        int maxSafe = rig.Kernels.MaxSafeAttentionMmaDecodeDense(8, 128, F16, F16);
        Assert.True(maxSafe >= 1);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            rig.Kernels.LaunchAttentionMmaDecodeDense(0, 0, 0, F16, F16, 0, 1024, 32, 8, 128, maxSafe + 1, 0, 0, 0, rig.Stream.Handle));
    }

    // ── Microbench: scalar attention_f16 / attention_kvq vs dense v2 (both partial arms), shape 32/8/128. ──
    [SkippableFact]
    public void Microbench_Scalar_vs_DenseV2()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_ATTN_MICROBENCH") == "1", "set DOTLLM_ATTN_MICROBENCH=1");
        using var rig = TrySetUp(out string? skip);
        Skip.If(rig == null, skip);
        const int nh = 32, nkv = 8, hd = 128;
        var seqList = (Environment.GetEnvironmentVariable("DOTLLM_ATTN_MICROBENCH_SEQKV") ?? "512,4096,8192,32768")
            .Split(',').Select(int.Parse).ToArray();
        int iters = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_ATTN_MICROBENCH_ITERS"), out int it) ? it : 24;
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("| seqKv | KV | kernel | kvSplit | ms/launch | eff GB/s |");
        sb.AppendLine("|---:|---|---|---:|---:|---:|");
        bool prev = CudaKernels.DenseDecodeV2F32Partials;
        try
        {
            foreach (int seqKv in seqList)
            {
                long kvElems = (long)seqKv * nkv * hd;
                var rng = new Random(seqKv);
                nint dQ = Upload(RandomHalf(rng, nh * hd, 2.5f));
                nint dOut = Alloc(nh * hd * 2);
                nint dPm = Alloc(nh * 64 * 4), dPs = Alloc(nh * 64 * 4), dPo = Alloc((long)nh * 64 * hd * 4);
                try
                {
                    foreach (var kd in new[] { F16, KvCacheDType.Q8_0, KvCacheDType.Q4_0 })
                    {
                        nint dK, dV; long bytes;
                        if (kd == F16) { dK = Upload(RandomHalf(rng, kvElems, 1f)); dV = Upload(RandomHalf(rng, kvElems, 1f)); bytes = kvElems * 4; }
                        else
                        {
                            dK = Upload(RandomQuantBlocks(rng, kvElems, kd)); dV = Upload(RandomQuantBlocks(rng, kvElems, kd));
                            bytes = kvElems / 32 * (kd == KvCacheDType.Q8_0 ? 34 : 18) * 2;
                        }
                        try
                        {
                            foreach (string arm in new[] { "scalar", "v2-f16p", "v2-f32p" })
                            {
                                int sp = 0;
                                Action launch;
                                if (arm == "scalar")
                                {
                                    launch = kd == F16
                                        ? () => rig!.Kernels.LaunchAttention(dQ, dK, dV, dOut, 1, seqKv, nh, nkv, hd, seqKv - 1, 0, rig.Stream.Handle)
                                        : () => rig!.Kernels.LaunchAttentionKvq(dQ, dK, dV, 0, 0, dOut, 1, seqKv, nh, nkv, hd, seqKv - 1, 0,
                                            rig.Stream.Handle, 0, kd, kd, seqKv, 0);
                                }
                                else
                                {
                                    CudaKernels.DenseDecodeV2F32Partials = arm == "v2-f32p";
                                    if (!rig!.Kernels.HasAttentionMmaDecodeDense(hd, kd, kd)) continue;
                                    sp = CudaKernels.ComputeAttentionMmaV2KvSplit(seqKv, rig.Kernels.MaxSafeAttentionMmaDecodeDense(nkv, hd, kd, kd));
                                    int spc = sp;
                                    launch = () => rig.Kernels.LaunchAttentionMmaDecodeDense(dQ, dK, dV, kd, kd, dOut, seqKv, nh, nkv, hd, spc,
                                        dPm, dPs, dPo, rig.Stream.Handle);
                                }
                                int n = arm == "scalar" && seqKv >= 16384 ? Math.Max(4, iters / 4) : iters;
                                double best = double.MaxValue;
                                for (int rep = 0; rep < 3; rep++)
                                {
                                    launch(); rig!.Stream.Synchronize();
                                    var sw = Stopwatch.StartNew();
                                    for (int i = 0; i < n; i++) launch();
                                    rig.Stream.Synchronize();
                                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds / n);
                                }
                                sb.AppendLine($"| {seqKv} | {(kd == F16 ? "f16" : kd.ToString())} | {arm} | {sp} | {best:F3} | {bytes / (best * 1e-3) / 1e9:F0} |");
                            }
                        }
                        finally { CudaDriverApi.cuMemFree_v2(dK); CudaDriverApi.cuMemFree_v2(dV); }
                    }
                }
                finally { foreach (var p in new[] { dQ, dOut, dPm, dPs, dPo }) CudaDriverApi.cuMemFree_v2(p); }
            }
        }
        finally { CudaKernels.DenseDecodeV2F32Partials = prev; }
        _out.WriteLine(sb.ToString());
        string? outFile = Environment.GetEnvironmentVariable("DOTLLM_ATTN_MICROBENCH_OUT");
        if (!string.IsNullOrEmpty(outFile)) File.AppendAllText(outFile, sb.ToString());
    }
}
