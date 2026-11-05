using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// #1616 part 2 probe: does a multiplication-free packed-bitmask kernel beat dotLLM's production PQ2_0 W2A8 path
/// (unpack-to-int8 + sign-trick dot) on real Bonsai-2 shapes? Parity tests run in CI; the timing harness is opt-in
/// (<c>Category=Benchmark</c> and <c>DOTLLM_TERNARY_PROBE=1</c>) and must hold the <c>cpu-bench</c> lock.
/// Results: docs/TERNARY_ADAPTER.md section "Zero-multiplication kernels for ternary model weights".
/// </summary>
public sealed unsafe class TernaryZeroMulProbe
{
    private readonly ITestOutputHelper _out;
    public TernaryZeroMulProbe(ITestOutputHelper output) => _out = output;

    // ------------------------------------------------------------------ parity (CI) ----

    private static byte[] RandomPq2Rows(Random rng, int rows, int k)
    {
        int groups = k / 128;
        var b = new byte[(long)rows * groups * 34];
        for (long g = 0; g < (long)rows * groups; g++)
        {
            ushort half = BitConverter.HalfToUInt16Bits((Half)(0.01f + (float)rng.NextDouble() * 0.2f));
            b[g * 34] = (byte)half; b[g * 34 + 1] = (byte)(half >> 8);
            for (int i = 0; i < 32; i++)
            {
                int packed = 0;
                for (int f = 0; f < 4; f++)
                {
                    int code = rng.Next(3); // 0,1,2 only (real PQ2_0 never stores 3)
                    packed |= code << (2 * f);
                }
                b[g * 34 + 2 + i] = (byte)packed;
            }
        }
        return b;
    }

    /// <summary>Exact double-precision oracle over the SAME quantized activations each kernel consumes.</summary>
    private static double[] OracleQ8(byte[] w, sbyte[] q, float[] s, int m, int k, int t)
    {
        int groups = k / 128;
        var o = new double[m];
        for (int r = 0; r < m; r++)
        {
            double acc = 0;
            for (int g = 0; g < groups; g++)
            {
                long gb = ((long)r * groups + g) * 34;
                double gs = (double)BitConverter.UInt16BitsToHalf((ushort)(w[gb] | (w[gb + 1] << 8)));
                for (int e = 0; e < 128; e++)
                {
                    int code = (w[gb + 2 + e / 4] >> (2 * (e & 3))) & 3;
                    int blk = g * 4 + e / 32;
                    acc += gs * (code - 1) * q[(long)t * k + g * 128 + e] * s[(long)t * (k / 32) + blk];
                }
            }
            o[r] = acc;
        }
        return o;
    }

    private static double[] OraclePlanar(byte[] w, sbyte[] q, float[] s, int m, int k, int t)
    {
        int groups = k / 128;
        var o = new double[m];
        for (int r = 0; r < m; r++)
        {
            double acc = 0;
            for (int g = 0; g < groups; g++)
            {
                long gb = ((long)r * groups + g) * 34;
                double gs = (double)BitConverter.UInt16BitsToHalf((ushort)(w[gb] | (w[gb + 1] << 8)));
                for (int e = 0; e < 128; e++)
                {
                    int code = (w[gb + 2 + e / 4] >> (2 * (e & 3))) & 3;
                    int plane = e & 3, b = e / 4;
                    acc += gs * (code - 1) * q[(long)t * k + g * 128 + plane * 32 + b] * s[((long)t * groups + g) * 4 + plane];
                }
            }
            o[r] = acc;
        }
        return o;
    }

    [SkippableTheory]
    [InlineData(128, 8, 1)]
    [InlineData(512, 7, 3)]
    [InlineData(1280, 33, 4)]
    public void K2_And_K3_MatchTheirOracles(int k, int m, int n)
    {
        // The probe kernels are AVX2-only experiments (not production code); the production planar kernel has its own all-ISA test below.
        Skip.IfNot(Avx2.IsSupported, "probe kernels need AVX2");
        var rng = new Random(1616 + k);
        byte[] w = RandomPq2Rows(rng, m, k);
        var x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * (i % 97 == 0 ? 8f : 1f);

        var xq = new sbyte[n * k]; var xs = new float[n * k / 32];
        var pq = new sbyte[n * k]; var ps = new float[n * (k / 128) * 4]; var pc = new float[n * (k / 128)];
        var bits = new byte[w.Length];
        var cK2 = new float[n * m]; var cK2w = new float[n * m]; var cK3 = new float[n * m];
        fixed (byte* wp = w) fixed (byte* bp = bits) fixed (float* xp = x) fixed (sbyte* xqp = xq) fixed (float* xsp = xs)
        fixed (sbyte* pqp = pq) fixed (float* psp = ps) fixed (float* pcp = pc)
        fixed (float* c2 = cK2) fixed (float* c2w = cK2w) fixed (float* c3 = cK3)
        {
            TernaryMaskedProbeKernels.RepackToBitmask34(wp, bp, m, k, out long threes);
            Assert.Equal(0, threes);
            TernaryMaskedProbeKernels.PrepQ8Contiguous(xp, n, k, xqp, xsp);
            TernaryMaskedProbeKernels.PrepPlanar(xp, n, k, pqp, psp, pcp);
            foreach (bool vnni in VnniModes())
            {
                Array.Clear(cK2); Array.Clear(cK3);
                TernaryMaskedProbeKernels.K2Rows256(bp, xqp, xsp, c2, m, 0, m, k, n, vnni);
                TernaryMaskedProbeKernels.K3Rows(wp, pqp, psp, pcp, c3, m, 0, m, k, n, vnni);
                for (int t = 0; t < n; t++)
                {
                    double[] o2 = OracleQ8(w, xq, xs, m, k, t);
                    double[] o3 = OraclePlanar(w, pq, ps, m, k, t);
                    for (int r = 0; r < m; r++)
                    {
                        Assert.InRange(Math.Abs(o2[r] - cK2[t * m + r]), 0, 2e-4 * (1 + Math.Abs(o2[r])));
                        Assert.InRange(Math.Abs(o3[r] - cK3[t * m + r]), 0, 2e-4 * (1 + Math.Abs(o3[r])));
                    }
                }
            }
            if (Avx512Vbmi.IsSupported && Avx512BW.IsSupported)
            {
                TernaryMaskedProbeKernels.K2Rows512(bp, xqp, xsp, c2w, m, 0, m, k, n);
                TernaryMaskedProbeKernels.K2Rows256(bp, xqp, xsp, c2, m, 0, m, k, n, false);
                for (int i = 0; i < cK2.Length; i++)
                    Assert.InRange(Math.Abs(cK2[i] - cK2w[i]), 0, 1e-4 * (1 + Math.Abs(cK2[i])));
            }
        }
    }

    [Theory]
    [InlineData(128, 8)]
    [InlineData(1280, 40)]
    [InlineData(4096, 70)]
    public void ProductionPlanarGemv_MatchesF64Truth_AsWellAsProduction_AndIsADifferentGrid(int k, int m)
    {
        var rng = new Random(99 + k);
        byte[] w = RandomPq2Rows(rng, m, k);
        var x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * (i % 53 == 0 ? 5f : 1f);
        var truth = new double[m];
        var cProd = new float[m]; var cPl = new float[m];
        fixed (byte* wp = w) fixed (float* xp = x) fixed (float* a = cProd) fixed (float* b = cPl)
        {
            var tmp = new float[k];
            long rb = (long)(k / 128) * 34;
            for (int r = 0; r < m; r++)
            {
                Dequantize.ToFloat32((nint)(wp + r * rb), k, QuantizationType.PQ2_0, tmp);
                double s = 0; for (int i = 0; i < k; i++) s += (double)tmp[i] * x[i];
                truth[r] = s;
            }
            // The Q8_0 W2A8 tier (GemvPQ2_0 itself dispatches to planar by default since #1620); without SSSE3 the dispatcher's float tier.
            if (global::System.Runtime.Intrinsics.X86.Ssse3.IsSupported) MatMul.GemmPQ2_0W2A8ForTest(wp, xp, a, m, k, 1);
            else MatMul.GemvPQ2_0(wp, xp, a, m, k, null);
            MatMul.GemvPQ2_0Planar(wp, xp, b, m, k, null);
        }
        // Portable fallback (what runs when AVX2 is disabled): same activations/grid, scalar accumulation.
        var cSc = new float[m];
        var pq = new sbyte[k]; var ps = new float[k / 32]; var pc = new float[k / 128];
        fixed (byte* wp = w) fixed (float* xp = x) fixed (float* o = cSc) fixed (sbyte* pqp = pq) fixed (float* psp = ps) fixed (float* pcp = pc)
        {
            MatMul.PreparePlanarActivations(xp, k, pqp, psp, pcp);
            MatMul.PQ2PlanarRowsScalar(wp, pqp, psp, pcp, o, 0, m, k);
        }
        double rms = Math.Sqrt(truth.Sum(v => v * v) / m);
        double Err(float[] c) => Math.Sqrt(Enumerable.Range(0, m).Sum(i => Math.Pow(c[i] - truth[i], 2)) / m) / rms;
        Assert.True(Err(cSc) < 0.02, $"scalar planar rel-RMS error {Err(cSc)}");
        for (int r = 0; r < m; r++) Assert.InRange(Math.Abs(cSc[r] - cPl[r]), 0, 2e-4 * (1 + Math.Abs(cSc[r])));   // AVX2 vs scalar: float-rounding only
        Assert.True(Err(cPl) < 0.02, $"planar rel-RMS error {Err(cPl)}");
        Assert.True(Err(cPl) < 3 * Err(cProd) + 1e-4, $"planar {Err(cPl)} vs production {Err(cProd)}");
        // Perturbation proof the planar path really ran: a different quantization grid cannot be bit-identical to Q8_0.
        Assert.NotEqual(cProd, cPl);
    }

    // ------------------------------------------------------------------ claim reproduction (opt-in) ----

    private static double[] TimeUs(Action a, int inner, int reps)
    {
        for (int i = 0; i < 2000; i++) a();
        var t = new double[reps];
        for (int r = 0; r < reps; r++)
        {
            long s = Stopwatch.GetTimestamp();
            for (int i = 0; i < inner; i++) a();
            t[r] = (Stopwatch.GetTimestamp() - s) * 1e6 / Stopwatch.Frequency / inner;
        }
        Array.Sort(t);
        return t;
    }

    /// <summary>
    /// The reference's headline measurement ("Experiment 1": one int8 down-projection K x 4096, 'token' = one such Forward call)
    /// plus the full adapter (quantize + down + ReLU/requantize + up + residual add) on the dotLLM kernel, same shapes.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void Claim_DownProjection_And_FullAdapter_D4096()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_TERNARY_PROBE") == "1", "set DOTLLM_TERNARY_PROBE=1 (hold the cpu-bench lock)");
        const int D = 4096;
        int reps = 15;
        var rng = new Random(42);
        var x8 = (sbyte*)NativeMemory.AlignedAlloc((nuint)D, 64);
        for (int i = 0; i < D; i++) x8[i] = (sbyte)rng.Next(-128, 127);
        foreach (int k in new[] { 32, 64, 128 })
        {
            int wpr = DotLLM.Cpu.Kernels.TernaryAdapterKernel.WordsPerRow(D);
            var masks = (ulong*)NativeMemory.AlignedAlloc((nuint)((long)k * wpr * 8), 64);
            NativeMemory.Clear(masks, (nuint)((long)k * wpr * 8));
            for (int r = 0; r < k; r++)
                for (int c = 0; c < D; c++)
                {
                    int v = rng.Next(-1, 2);
                    if (v != 0) masks[(long)r * wpr + (c >> 6) * 2 + (v < 0 ? 1 : 0)] |= 1UL << (c & 63);
                }
            float* sp = (float*)NativeMemory.AlignedAlloc((nuint)(k * 4), 64);
            float* op = (float*)NativeMemory.AlignedAlloc((nuint)(k * 4), 64);
            for (int i = 0; i < k; i++) sp[i] = 1f;
            foreach (var isa in new[] { DotLLM.Cpu.Kernels.TernaryKernelIsa.Auto, DotLLM.Cpu.Kernels.TernaryKernelIsa.Scalar })
            {
                double[] td = TimeUs(() => DotLLM.Cpu.Kernels.TernaryAdapterKernel.DownRows(x8, D, masks, k, sp, op, isa), 2000, reps);
                _out.WriteLine($"dotLLM down-projection K={k} D={D} isa={isa}: min {td[0]:F2} / median {td[reps / 2]:F2} / max {td[^1]:F2} us  -> {1e6 / td[reps / 2]:N0} /s (median)");
            }
            NativeMemory.AlignedFree(sp); NativeMemory.AlignedFree(op);
            NativeMemory.AlignedFree(masks);
        }

        // Full adapter K=32, D=4096 through the hook (float residual row in, row updated in place).
        var down = new DotLLM.Models.Adapters.Ternary.PackedTernaryTensor(32, D);
        var up = new DotLLM.Models.Adapters.Ternary.PackedTernaryTensor(D, 32);
        for (int r = 0; r < 32; r++) for (int c = 0; c < D; c++) down[r, c] = (sbyte)rng.Next(-1, 2);
        for (int r = 0; r < D; r++) for (int c = 0; c < 32; c++) up[r, c] = (sbyte)rng.Next(-1, 2);
        var pkg = new DotLLM.Models.Adapters.Ternary.NsAdapterPackage(new DotLLM.Models.Adapters.Ternary.NsAdapterManifest { TargetLayer = 3, Alpha = 1f }, down, up);
        using var hook = new DotLLM.Models.Adapters.Ternary.TernaryResidualHook(D);
        hook.Add(pkg);
        var src = new float[D]; for (int i = 0; i < D; i++) src[i] = (float)(rng.NextDouble() * 2 - 1);
        var row = new float[D];
        double[] t = TimeUs(() => { src.CopyTo(row, 0); hook.ModifyResidual(row, 3, 1); }, 500, reps);
        _out.WriteLine($"dotLLM FULL adapter K=32 D={D} (float row in: absmax-int8 + down + relu/requant + up + residual add): min {t[0]:F2} / median {t[reps / 2]:F2} / max {t[^1]:F2} us -> {1e6 / t[reps / 2]:N0} /s");
        NativeMemory.AlignedFree(x8);
    }

    [Theory]
    [InlineData(128, 5)]
    [InlineData(1280, 9)]
    public void ReferencePortedKernels_ComputeTheIntegerTernaryDot_OfThePq2Weights(int k, int m)
    {
        var rng = new Random(31 + k);
        byte[] w = RandomPq2Rows(rng, m, k);
        var x = new sbyte[k]; for (int i = 0; i < k; i++) x[i] = (sbyte)rng.Next(-128, 127);
        var packed = new byte[m * k / 4]; var unp = new sbyte[m * k];
        var o1 = new float[m]; var o2 = new float[m];
        fixed (byte* wp = w) fixed (byte* pp = packed) fixed (sbyte* up = unp) fixed (sbyte* xp = x) fixed (float* a = o1) fixed (float* b = o2)
        {
            TernaryMaskedProbeKernels.RepackToReferencePacked(wp, pp, m, k);
            TernaryMaskedProbeKernels.UnpackReference(pp, up, m, k);
            TernaryMaskedProbeKernels.RefSimdPackedRows(pp, xp, a, 0, m, k);
            TernaryMaskedProbeKernels.RefUnpackedRows(up, xp, b, 0, m, k);
        }
        for (int r = 0; r < m; r++)
        {
            long exp = 0;
            for (int e = 0; e < k; e++)
            {
                int g = e / 128, i = e % 128;
                int code = (w[((long)r * (k / 128) + g) * 34 + 2 + i / 4] >> (2 * (i & 3))) & 3;
                exp += (code - 1) * x[e];
            }
            Assert.Equal((float)exp, o1[r]);
            Assert.Equal((float)exp, o2[r]);
        }
    }

    private static IEnumerable<bool> VnniModes()
    {
        yield return false;
        if (AvxVnni.IsSupported) yield return true;
    }

    // ------------------------------------------------------------------ timing harness (opt-in) ----

    private struct Ctx
    {
        public int Kind;           // 1 K2-256, 2 K2-512, 3 K3, 9 stream
        public byte* W; public sbyte* Xq; public float* Xs; public float* Xc; public float* C;
        public int M, K, N; public bool Vnni; public long StreamBytes; public byte* Stream;
    }

    private static void Worker(nint p, int idx, int cnt)
    {
        ref Ctx c = ref *(Ctx*)p;
        if (c.Kind == 9)
        {
            long per = c.StreamBytes / cnt;
            TernaryMaskedProbeKernels.StreamBytes(c.Stream + idx * per, per);
            return;
        }
        int chunk = ((c.M + cnt - 1) / cnt + 3) & ~3;
        int start = idx * chunk;
        if (start >= c.M) return;
        int count = Math.Min(chunk, c.M - start);
        switch (c.Kind)
        {
            case 1: TernaryMaskedProbeKernels.K2Rows256(c.W, c.Xq, c.Xs, c.C, c.M, start, count, c.K, c.N, c.Vnni); break;
            case 2: TernaryMaskedProbeKernels.K2Rows512(c.W, c.Xq, c.Xs, c.C, c.M, start, count, c.K, c.N); break;
            case 3: TernaryMaskedProbeKernels.K3Rows(c.W, c.Xq, c.Xs, c.Xc, c.C, c.M, start, count, c.K, c.N, c.Vnni); break;
            case 4: for (int t = 0; t < c.N; t++) TernaryMaskedProbeKernels.RefSimdPackedRows(c.W, c.Xq + (long)t * c.K, c.C + (long)t * c.M, start, count, c.K); break;
            case 5: for (int t = 0; t < c.N; t++) TernaryMaskedProbeKernels.RefUnpackedRows((sbyte*)c.W, c.Xq + (long)t * c.K, c.C + (long)t * c.M, start, count, c.K); break;
        }
    }

    private sealed class Weights : IDisposable
    {
        public byte*[] Pq2 = []; public byte*[] Bits = []; public byte*[] RefPacked = []; public byte*[] RefUnp = []; public long BytesEach; public int M, K;
        public void Dispose() { foreach (var p in Bits) NativeMemory.AlignedFree(p); foreach (var p in Pq2) NativeMemory.AlignedFree(p); foreach (var p in RefPacked) NativeMemory.AlignedFree(p); foreach (var p in RefUnp) NativeMemory.AlignedFree(p); }
    }

    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void Probe_RealBonsai2Shapes()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_TERNARY_PROBE") == "1", "set DOTLLM_TERNARY_PROBE=1 (hold the cpu-bench lock)");
        var loc = KnownTestFixtures.TernaryBonsai27B_Q2_0;
        string? path = loc.Path;
        Skip.If(path is null, "Bonsai-2 PQ2_0 GGUF not found");
        using var gguf = GgufFile.Open(path!);

        int threads = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_TERNARY_PROBE_THREADS"), out int th) ? th : Environment.ProcessorCount / 2;
        int reps = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_TERNARY_PROBE_REPS"), out int rp) ? rp : 8;
        using var pool = new ComputeThreadPool(threads);
        _out.WriteLine($"ISA: Avx2={Avx2.IsSupported} AvxVnni={AvxVnni.IsSupported} Avx512BW={Avx512BW.IsSupported} Vbmi={Avx512Vbmi.IsSupported}; pool threads={pool.ThreadCount}; reps={reps}");

        var groupsByShape = gguf.Tensors.Where(t => t.QuantizationType == QuantizationType.PQ2_0 && t.Shape.Rank == 2)
            .GroupBy(t => (m: t.Shape[1], k: t.Shape[0])).OrderByDescending(g => g.Count() * (long)g.Key.m * g.Key.k).ToList();
        _out.WriteLine("PQ2_0 2-D shapes (m=out, k=in) x count: " + string.Join(", ", groupsByShape.Select(g => $"{g.Key.m}x{g.Key.k}x{g.Count()}")));

        foreach (var grp in groupsByShape.Take(int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_TERNARY_PROBE_SHAPES"), out int ns) ? ns : 4))
        {
            // Shape convention check: GGUF stores [ne0=k, ne1=m]; the dotLLM descriptor reports (m, k) per the PQ2_0 loaders.
            int m = grp.Key.m, k = grp.Key.k;
            if (k % 128 != 0) { _out.WriteLine($"skip {m}x{k}: k % 128 != 0"); continue; }
            var tensors = grp.Take(12).ToList();
            RunShape(gguf, pool, tensors, m, k, reps);
        }
    }

    /// <summary>
    /// Accuracy of each int8-activation kernel against an F64 oracle (dequantized real weights x UNQUANTIZED activations),
    /// on the first rows of real tensors. The production-vs-K3 figure above compares two different int8 grids; this
    /// measures each against the truth.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void Parity_RealWeights_VsF64Truth()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_TERNARY_PROBE") == "1", "set DOTLLM_TERNARY_PROBE=1");
        string? path = KnownTestFixtures.TernaryBonsai27B_Q2_0.Path;
        Skip.If(path is null, "Bonsai-2 PQ2_0 GGUF not found");
        using var gguf = GgufFile.Open(path!);
        foreach (var t in gguf.Tensors.Where(t => t.QuantizationType == QuantizationType.PQ2_0 && t.Shape.Rank == 2 && t.Shape[0] % 128 == 0)
                     .GroupBy(t => (t.Shape[0], t.Shape[1])).Select(g => g.First()).Take(5))
        {
            int m = 64, k = t.Shape[0];   // GGUF order: Shape[0] = ne0 = input dim k (contiguous), Shape[1] = rows m
            if (t.Shape[1] < m) continue;
            long rowBytes = (long)(k / 128) * 34;
            byte* wsrc = (byte*)(gguf.DataBasePointer + (nint)t.DataOffset);
            var rng = new Random(5);
            foreach (string dist in new[] { "uniform", "gauss+outliers" })
            {
                var x = new float[k];
                for (int i = 0; i < k; i++)
                {
                    double u = rng.NextDouble() * 2 - 1;
                    x[i] = dist == "uniform" ? (float)u
                        : (float)(Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()) * (i % 509 == 0 ? 30 : 1));
                }
                var truth = new double[m];
                var tmp = new float[k];
                for (int r = 0; r < m; r++)
                {
                    Dequantize.ToFloat32((nint)(wsrc + r * rowBytes), k, QuantizationType.PQ2_0, tmp);
                    double a = 0; for (int i = 0; i < k; i++) a += (double)tmp[i] * x[i];
                    truth[r] = a;
                }
                var cProd = new float[m]; var cK2 = new float[m]; var cK3 = new float[m];
                var xq = new sbyte[k]; var xs = new float[k / 32]; var pq = new sbyte[k]; var ps = new float[k / 32]; var pc = new float[k / 128];
                var bits = new byte[m * rowBytes];
                fixed (float* xp = x) fixed (float* cp = cProd) fixed (float* c2 = cK2) fixed (float* c3 = cK3)
                fixed (sbyte* xqp = xq) fixed (float* xsp = xs) fixed (sbyte* pqp = pq) fixed (float* psp = ps) fixed (float* pcp = pc) fixed (byte* bp = bits)
                {
                    MatMul.GemmPQ2_0(wsrc, xp, cp, m, k, 1, null);
                    TernaryMaskedProbeKernels.RepackToBitmask34(wsrc, bp, m, k, out _);
                    TernaryMaskedProbeKernels.PrepQ8Contiguous(xp, 1, k, xqp, xsp);
                    TernaryMaskedProbeKernels.PrepPlanar(xp, 1, k, pqp, psp, pcp);
                    TernaryMaskedProbeKernels.K2Rows256(bp, xqp, xsp, c2, m, 0, m, k, 1, AvxVnni.IsSupported);
                    TernaryMaskedProbeKernels.K3Rows(wsrc, pqp, psp, pcp, c3, m, 0, m, k, 1, AvxVnni.IsSupported);
                }
                double rms = Math.Sqrt(truth.Sum(v => v * v) / m);
                double E(float[] c) => Math.Sqrt(Enumerable.Range(0, m).Sum(i => Math.Pow(c[i] - truth[i], 2)) / m) / rms;
                _out.WriteLine($"{t.Name} k={k} {dist}: rel-RMS error vs F64 truth: prod={E(cProd):E2} K2={E(cK2):E2} K3(planar)={E(cK3):E2}");
            }
        }
    }

    private void RunShape(GgufFile gguf, ComputeThreadPool pool, List<GgufTensorDescriptor> tensors, int m, int k, int reps)
    {
        long rowBytes = (long)(k / 128) * 34;
        long bytesEach = rowBytes * m;
        int rot = tensors.Count;
        var w = new Weights { BytesEach = bytesEach, M = m, K = k, Pq2 = new byte*[rot], Bits = new byte*[rot], RefPacked = new byte*[rot], RefUnp = new byte*[Math.Min(rot, 3)] };
        try
        {
            long threesTotal = 0;
            for (int i = 0; i < rot; i++)
            {
                // Own aligned copies (the mmap'd GGUF pages are the same bytes; copying lets both kernels see identical cache placement).
                w.Pq2[i] = (byte*)NativeMemory.AlignedAlloc((nuint)bytesEach, 64);
                Buffer.MemoryCopy((byte*)(gguf.DataBasePointer + (nint)tensors[i].DataOffset), w.Pq2[i], bytesEach, bytesEach);
                w.Bits[i] = (byte*)NativeMemory.AlignedAlloc((nuint)bytesEach, 64);
                w.RefPacked[i] = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * k / 4), 64);
                TernaryMaskedProbeKernels.RepackToReferencePacked(w.Pq2[i], w.RefPacked[i], m, k);
                if (i < 3)
                {
                    w.RefUnp[i] = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * k), 64);
                    TernaryMaskedProbeKernels.UnpackReference(w.RefPacked[i], (sbyte*)w.RefUnp[i], m, k);
                }
                TernaryMaskedProbeKernels.RepackToBitmask34(w.Pq2[i], w.Bits[i], m, k, out long threes);
                threesTotal += threes;
            }
            _out.WriteLine($"\n=== shape m={m} k={k}: {bytesEach / 1048576.0:F1} MiB/matrix x {rot} rotation = {bytesEach * rot / 1048576.0:F0} MiB; code-3 count in real data: {threesTotal} ===");
            foreach (int n in new[] { 1, 4, 16 })
                foreach (int th in new[] { 1, pool.ThreadCount })
                    RunConfig(pool, w, m, k, n, th, reps);
        }
        finally { w.Dispose(); }
    }

    private void RunConfig(ComputeThreadPool pool, Weights w, int m, int k, int n, int threads, int reps)
    {
        var rng = new Random(7);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * k * 4), 64);
        for (long i = 0; i < (long)n * k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * (i % 211 == 0 ? 6f : 1f);
        sbyte* xq = (sbyte*)NativeMemory.AlignedAlloc((nuint)((long)n * k), 64);
        float* xs = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * k / 32 * 4), 64);
        sbyte* pq = (sbyte*)NativeMemory.AlignedAlloc((nuint)((long)n * k), 64);
        float* ps = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * k / 32 * 4), 64);
        float* pc = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * k / 128 * 4), 64);
        float* cBase = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * 4), 64);
        float* c2 = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * 4), 64);
        float* c5 = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * 4), 64);
        float* c3 = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * 4), 64);
        float* cRef = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * 4), 64);
        try
        {
            TernaryMaskedProbeKernels.PrepQ8Contiguous(x, n, k, xq, xs);
            TernaryMaskedProbeKernels.PrepPlanar(x, n, k, pq, ps, pc);
            ComputeThreadPool? pl = threads == 1 ? null : pool;
            bool vnni = AvxVnni.IsSupported;
            bool has512 = Avx512Vbmi.IsSupported && Avx512BW.IsSupported;

            // kernels: id -> action over rotation slot r
            var names = new List<string> { "prod GemmPQ2_0 (unpack+sign dot)", "K2 mask add/sub 256", "K3 planar-offset 256" };
            if (has512) names.Insert(2, "K2 mask add/sub 512 (vpermb)");
            Action<int> Prod = r => MatMul.GemmPQ2_0(w.Pq2[r], x, cBase, m, k, n, pl);
            Action<int> K2 = r => RunPool(pool, pl, new Ctx { Kind = 1, W = w.Bits[r], Xq = xq, Xs = xs, C = c2, M = m, K = k, N = n, Vnni = vnni });
            Action<int> K2w = r => RunPool(pool, pl, new Ctx { Kind = 2, W = w.Bits[r], Xq = xq, Xs = xs, C = c5, M = m, K = k, N = n });
            Action<int> K3 = r => RunPool(pool, pl, new Ctx { Kind = 3, W = w.Pq2[r], Xq = pq, Xs = ps, Xc = pc, C = c3, M = m, K = k, N = n, Vnni = vnni });
            var kernels = new List<Action<int>> { Prod, K2 };
            if (has512) kernels.Add(K2w);
            kernels.Add(K3);
            names.Add("REF SimdTernaryKernel (LUT unpack, packed)");
            kernels.Add(r => RunPool(pool, pl, new Ctx { Kind = 4, W = w.RefPacked[r], Xq = xq, C = cRef, M = m, K = k, N = n }));
            names.Add("REF UnpackedTernaryMatrix (int8 weights)");
            kernels.Add(r => RunPool(pool, pl, new Ctx { Kind = 5, W = w.RefUnp[r % w.RefUnp.Length], Xq = xq, C = cRef, M = m, K = k, N = n }));

            // parity vs production on real weights (matrix 0): same int8 grid for K2 (bit-near), different grid for K3.
            Prod(0); K2(0); K3(0); if (has512) K2w(0);
            double maxRelK2 = 0, maxRelK3 = 0, maxRel512 = 0, rms = 0;
            for (long i = 0; i < (long)n * m; i++) rms += (double)cBase[i] * cBase[i];
            rms = Math.Sqrt(rms / ((long)n * m));
            for (long i = 0; i < (long)n * m; i++)
            {
                maxRelK2 = Math.Max(maxRelK2, Math.Abs(c2[i] - cBase[i]) / rms);
                maxRelK3 = Math.Max(maxRelK3, Math.Abs(c3[i] - cBase[i]) / rms);
                if (has512) maxRel512 = Math.Max(maxRel512, Math.Abs(c5[i] - cBase[i]) / rms);
            }

            int rot = w.Pq2.Length;
            int callsPerRep = Math.Max(rot * 2, 8);
            var samples = names.Select(_ => new List<double>()).ToList();
            for (int rep = 0; rep < reps; rep++)
            {
                var order = Enumerable.Range(0, kernels.Count).ToList();
                if (rep % 2 == 1) order.Reverse();   // same-session paired A/B, both orders
                foreach (int ki in order)
                {
                    kernels[ki](0); // warm (JIT + first-touch)
                    var sw = Stopwatch.StartNew();
                    for (int call = 0; call < callsPerRep; call++) kernels[ki](call % rot);
                    sw.Stop();
                    samples[ki].Add(sw.Elapsed.TotalMilliseconds / callsPerRep);
                }
            }
            double bytes = w.BytesEach;
            _out.WriteLine($"-- n={n} threads={(threads == 1 ? 1 : pool.ThreadCount)}: per-call ms [min / median / max]  GB/s@median (weights only)  | parity vs prod (max|d|/rms): K2={maxRelK2:E1} K3={maxRelK3:E1}{(has512 ? $" K2-512={maxRel512:E1}" : "")}");
            for (int i = 0; i < names.Count; i++)
            {
                var s = samples[i].OrderBy(v => v).ToList();
                double med = s[s.Count / 2];
                _out.WriteLine($"   {names[i],-34} {s[0],8:F3} / {med,8:F3} / {s[^1],8:F3} ms   {bytes / (med / 1000) / 1e9,7:F1} GB/s   vs prod {samples[0].OrderBy(v => v).ElementAt(samples[0].Count / 2) / med,5:F2}x");
            }

            if (n == 1)
            {
                // Roofline: pure streaming read of the same rotation (same threads), no decode.
                var rs = new List<double>();
                for (int rep = 0; rep < reps; rep++)
                {
                    var sw = Stopwatch.StartNew();
                    for (int call = 0; call < callsPerRep; call++)
                    {
                        var ctx = new Ctx { Kind = 9, Stream = w.Pq2[call % rot], StreamBytes = (long)w.BytesEach };
                        RunPool(pool, pl, ctx);
                    }
                    sw.Stop();
                    rs.Add(sw.Elapsed.TotalMilliseconds / callsPerRep);
                }
                rs.Sort();
                _out.WriteLine($"   {"stream-read roofline (same threads)",-34} {rs[0],8:F3} / {rs[rs.Count / 2],8:F3} / {rs[^1],8:F3} ms   {bytes / (rs[rs.Count / 2] / 1000) / 1e9,7:F1} GB/s");
            }

            // activation prep cost (shared by all matmuls of the same input in a real layer)
            var sw2 = Stopwatch.StartNew();
            for (int i = 0; i < 50; i++) TernaryMaskedProbeKernels.PrepPlanar(x, n, k, pq, ps, pc);
            double prepPlanar = sw2.Elapsed.TotalMilliseconds / 50;
            sw2.Restart();
            for (int i = 0; i < 50; i++) TernaryMaskedProbeKernels.PrepQ8Contiguous(x, n, k, xq, xs);
            _out.WriteLine($"   activation prep per call: Q8-contiguous {sw2.Elapsed.TotalMilliseconds / 50:F4} ms, planar {prepPlanar:F4} ms (not in the figures above)");
        }
        finally
        {
            foreach (void* p in new void*[] { x, xq, xs, pq, ps, pc, cBase, c2, c5, c3, cRef }) NativeMemory.AlignedFree(p);
        }
    }

    private static void RunPool(ComputeThreadPool pool, ComputeThreadPool? pl, Ctx ctx)
    {
        if (pl is null) Worker((nint)(&ctx), 0, 1);
        else pool.Dispatch((nint)(&ctx), &Worker);
    }
}
