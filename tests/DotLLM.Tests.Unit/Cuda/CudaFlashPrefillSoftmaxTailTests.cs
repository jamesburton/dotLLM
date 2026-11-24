using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1953: the kvq flash prefill kernel's P operand at long context. After the online-softmax rescale P = exp(s - max) lies in
/// (0, 1], and the P.V tensor-core MMA takes it as F16, whose subnormal step is 2^-24 (6e-8): a key 17+ nats below the row max
/// rounds to 0 or to one subnormal step, while the F32 denominator l keeps its exact value. At 32K keys behind an attention sink,
/// thousands of such keys carry a mass that the numerator loses (or over-counts) and l does not, a bias on every row.
/// <para>The fixture is that regime and nothing else: one sink key (V = 0) 15.6-18.4 nats above every other key (V ~ 1, one sign),
/// so the output IS the tail mass, about 1.8e-3 at 32K keys. The random fixtures of <c>FlashPrefillF16Entry_AgreesWithAttentionF16</c>
/// give a near-uniform softmax where no P is small, which is why they could not see this. Oracle: scalar <c>attention_f16</c>
/// (F32 P).</para>
/// </summary>
public sealed unsafe partial class CudaAttentionKvqTests
{
    public static IEnumerable<object[]> SinkTailCases() =>
    [
        // headDim, numHeads, numKvHeads, seqQ, seqKv
        [128, 8, 2, 16, 32771],     // 32K keys, one q-tile at the end of the history
        [128, 8, 2, 33, 4099],      // 4K keys, partial q-tiles
        [64, 8, 2, 16, 32771],
    ];

    /// <summary>
    /// The F16 entry keeps the tail mass of a sink-dominated long row: relative error against <c>attention_f16</c> at most 1% on every
    /// element (the dev kernel before #1953 is off by tens of percent here).
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(SinkTailCases))]
    public void FlashPrefillF16Entry_SinkDominatedLongRow_KeepsTheTailMass(int headDim, int numHeads, int numKvHeads, int seqQ, int seqKv)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");
        Skip.IfNot(_kernels!.HasAttentionFlashPrefillKvq(headDim, null, null), "flash prefill F16 entry not loaded (needs Ampere+)");
        var (worst, at, got, want) = SinkTailError(_kernels, headDim, numHeads, numKvHeads, seqQ, seqKv);
        _out.WriteLine($"hd{headDim} {numHeads}/{numKvHeads} seqQ {seqQ} seqKv {seqKv}: worst relative error {worst:E3} at {at} " +
                       $"({got:E4} vs attention_f16 {want:E4})");
        Assert.True(worst <= 0.01, $"flash prefill loses the softmax tail: {got:E4} vs attention_f16 {want:E4} at {at} " +
                                   $"(relative error {worst:E3})");
    }

    /// <summary>Runs the sink/tail fixture through <paramref name="kernels"/>' F16 flash entry and the scalar oracle.</summary>
    private (double Worst, int At, double Got, double Want) SinkTailError(CudaKernels kernels, int headDim, int numHeads, int numKvHeads,
        int seqQ, int seqKv)
    {
        int offset = seqKv - seqQ;
        int stride = numKvHeads * headDim;
        var rng = new Random(1953 + headDim + seqKv);
        var k = new ushort[(long)seqKv * stride];
        var v = new ushort[(long)seqKv * stride];
        for (int j = 0; j < seqKv; j++)
            for (int h = 0; h < numKvHeads; h++)
            {
                long b = (long)j * stride + h * headDim;
                // Score of key j = q[0] * k[0] / sqrt(hd) (q's other components are 0): sink 20, tail -2..-6.
                k[b] = FloatToHalf(j == 0 ? 20f : -2f - 4f * (float)rng.NextDouble());
                for (int d = 1; d < headDim; d++) k[b + d] = FloatToHalf((float)(rng.NextDouble() * 0.2 - 0.1));
                for (int d = 0; d < headDim; d++) v[b + d] = FloatToHalf(j == 0 ? 0f : 0.75f + 0.5f * (float)rng.NextDouble());
            }
        var q = new ushort[seqQ * numHeads * headDim];
        // q[0] scaled so that 8 * 20 / sqrt(128) = 14.1 and the tail sits 15.6-18.4 nats below at hd128 (hd64: same gaps x sqrt 2,
        // q[0] scaled down to match).
        float q0 = 8f * MathF.Sqrt(headDim / 128f);
        for (int r = 0; r < seqQ; r++)
            for (int h = 0; h < numHeads; h++)
                q[(r * numHeads + h) * headDim] = FloatToHalf(q0 * (1f + 0.02f * h));
        nint dq = Upload(q), dk = Upload(k), dv = Upload(v);
        nint oRef = Alloc(q.Length * 2L), oNew = Alloc(q.Length * 2L);
        try
        {
            _kernels!.LaunchAttention(dq, dk, dv, oRef, seqQ, seqKv, numHeads, numKvHeads, headDim, offset, 0, _stream!.Handle);
            kernels.LaunchAttentionFlashPrefillKvq(dq, dk, dv, null, null, oNew, seqQ, seqKv, numHeads, numKvHeads, headDim, offset,
                _stream.Handle);
            _stream.Synchronize();
            ushort[] want = ReadHalf(oRef, q.Length), got = ReadHalf(oNew, q.Length);
            double worst = 0;
            int at = 0;
            for (int i = 0; i < want.Length; i++)
            {
                double a = HalfToFloat(got[i]), b = HalfToFloat(want[i]);
                double err = Math.Abs(a - b) / Math.Max(Math.Abs(b), 1e-30);
                if (!(err <= worst)) { worst = err; at = i; }
            }
            return (worst, at, HalfToFloat(got[at]), HalfToFloat(want[at]));
        }
        finally
        {
            foreach (nint p in new[] { dq, dk, dv, oRef, oNew }) CudaDriverApi.cuMemFree_v2(p);
        }
    }

    private static ushort[] ReadHalf(nint dev, int n)
    {
        var h = new ushort[n];
        fixed (ushort* p = h) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dev, (nuint)(n * 2)).ThrowOnError();
        return h;
    }

    /// <summary>
    /// Opt-in (#1953): several builds of <c>attention_flash_mma_prefill_kvq.ptx</c> in one process,
    /// <c>DOTLLM_1953_VARIANTS</c> = <c>name=path;...</c>. Per build: the sink/tail error, the regular F16-entry oracle cases
    /// (normalised error vs <c>attention_f16</c>), and with <c>DOTLLM_1953_BENCH=1</c> ms per launch (1,024 rows at 4,096 and
    /// 31,744 cached keys, 32/8/128, f16 / q8_0 / q4_0; interleaved rounds, median).
    /// </summary>
    [SkippableFact]
    public void FlashPrefill1953_VariantSweep()
    {
        string? spec = Environment.GetEnvironmentVariable("DOTLLM_1953_VARIANTS");
        Skip.If(string.IsNullOrWhiteSpace(spec), "DOTLLM_1953_VARIANTS not set");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");
        var builds = new List<(string Name, CudaKernels K, string Dir)>();
        try
        {
            foreach (string item in spec!.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                int eq = item.IndexOf('=');
                string name = item[..eq], path = item[(eq + 1)..];
                Assert.True(File.Exists(path), $"variant {name}: {path} not found");
                string dir = Path.Combine(Path.GetTempPath(), $"dotllm-1953-{Environment.ProcessId}-{name}");
                Directory.CreateDirectory(dir);
                foreach (string f in Directory.GetFiles(_kernels!.PtxDirectory, "*.ptx")) File.Copy(f, Path.Combine(dir, Path.GetFileName(f)), true);
                File.Copy(path, Path.Combine(dir, "attention_flash_mma_prefill_kvq.ptx"), true);
                builds.Add((name, new CudaKernels(dir), dir));
            }
            foreach (var b in builds)
            {
                foreach (object[] c in SinkTailCases())
                {
                    var (w, at, got, want) = SinkTailError(b.K, (int)c[0], (int)c[1], (int)c[2], (int)c[3], (int)c[4]);
                    _out.WriteLine($"TAIL {b.Name,-10} hd{c[0]} seqKv {c[4]}: worst rel {w:E3} ({got:E4} vs {want:E4} at {at})");
                }
                foreach (object[] c in F16FlashCases())
                {
                    int hd = (int)c[0], nh = (int)c[1], nkv = (int)c[2], seqQ = (int)c[3], off = (int)c[4], seqKv = off + seqQ;
                    using var f = new Fixture(this, KvCacheDType.Q8_0, KvCacheDType.Q8_0, hd, nh, nkv, seqQ, seqKv, seqKv, window: 0, sinks: false);
                    _kernels!.LaunchAttention(f.Q, f.KScratch, f.VScratch, f.OutRef, seqQ, seqKv, nh, nkv, hd, off, 0, _stream!.Handle);
                    b.K.LaunchAttentionFlashPrefillKvq(f.Q, f.KScratch, f.VScratch, null, null, f.OutNew, seqQ, seqKv, nh, nkv, hd, off, _stream.Handle);
                    _stream.Synchronize();
                    ushort[] r = f.Read(f.OutRef), n = f.Read(f.OutNew);
                    double worst = 0;
                    for (int i = 0; i < r.Length; i++)
                    {
                        double a = HalfToFloat(n[i]), s = HalfToFloat(r[i]);
                        worst = Math.Max(worst, Math.Abs(a - s) / (3e-3 + 3e-3 * Math.Abs(s)));
                    }
                    _out.WriteLine($"ORACLE {b.Name,-10} hd{hd} {nh}/{nkv} q{seqQ} off{off}: normalised error {worst:F3}");
                }
            }
            if (Environment.GetEnvironmentVariable("DOTLLM_1953_BENCH") != "1") return;
            int rounds = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_1953_ROUNDS"), out int rr) ? rr : 5;
            const int seqQb = 1024, reps = 5;
            foreach (int depth in new[] { 4096, 31744 })
                foreach (KvCacheDType? fmt in new KvCacheDType?[] { null, KvCacheDType.Q8_0, KvCacheDType.Q4_0 })
                {
                    int seqKv = depth + seqQb;
                    using var f = new Fixture(this, fmt ?? KvCacheDType.Q8_0, fmt ?? KvCacheDType.Q8_0, 128, 32, 8, seqQb, seqKv, seqKv,
                        window: 0, sinks: false);
                    nint k = fmt is null ? f.KScratch : f.KQuant, v = fmt is null ? f.VScratch : f.VQuant;
                    var ms = builds.ToDictionary(b => b.Name, _ => new List<double>());
                    for (int r = 0; r < rounds; r++)
                        foreach (var b in builds)
                        {
                            void Launch() => b.K.LaunchAttentionFlashPrefillKvq(f.Q, k, v, fmt, fmt, f.OutNew, seqQb, seqKv, 32, 8, 128, depth, _stream!.Handle);
                            Launch(); Launch();
                            _stream!.Synchronize();
                            var sw = System.Diagnostics.Stopwatch.StartNew();
                            for (int i = 0; i < reps; i++) Launch();
                            _stream.Synchronize();
                            ms[b.Name].Add(sw.Elapsed.TotalMilliseconds / reps);
                        }
                    double baseMed = MedianOf(ms[builds[0].Name]);
                    foreach (var b in builds)
                    {
                        var l = ms[b.Name];
                        _out.WriteLine($"BENCH d{depth} {fmt?.ToString() ?? "F16",-5} {b.Name,-10} median {MedianOf(l):F3} ms " +
                                       $"[{l.Min():F3}-{l.Max():F3}] time vs {builds[0].Name} {MedianOf(l) / baseMed:F4}");
                    }
                }
        }
        finally
        {
            foreach (var b in builds)
            {
                b.K.Dispose();
                try { Directory.Delete(b.Dir, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
            }
        }
    }

    private static double MedianOf(List<double> a)
    {
        var s = a.OrderBy(x => x).ToArray();
        return s.Length % 2 == 1 ? s[s.Length / 2] : 0.5 * (s[s.Length / 2 - 1] + s[s.Length / 2]);
    }
}
