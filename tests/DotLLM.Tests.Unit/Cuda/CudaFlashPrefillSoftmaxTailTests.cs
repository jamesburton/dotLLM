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
/// <para>The fixture is that regime and nothing else: one sink key (V = 0) 15.6-18.4 nats above every other key (V 48..80, one
/// sign), so the output IS the tail mass (x ~64, in F16's normal range). Every CUDA tensor-core attention kernel packs P the same
/// way, so each gets the case: kvq prefill, hd256 prefill, dense decode, hd256 mma decode v1 / v2 and G-flash. Before the fix
/// each returns 0 here (the whole tail flushed). The random fixtures of <c>FlashPrefillF16Entry_AgreesWithAttentionF16</c>
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
        AssertTail($"kvq F16 flash prefill hd{headDim} {numHeads}/{numKvHeads} seqQ {seqQ} seqKv {seqKv}",
            SinkTailError(_kernels, headDim, numHeads, numKvHeads, seqQ, seqKv));
    }

    /// <summary>Host Q/K/V of the sink/tail fixture (F16 bits). Score of key j = q0 * k_j[0] / sqrt(hd) (q's other components are 0).</summary>
    private static (ushort[] Q, ushort[] K, ushort[] V) SinkTailData(int headDim, int numHeads, int numKvHeads, int seqQ, int seqKv)
    {
        int stride = numKvHeads * headDim;
        var rng = new Random(1953 + headDim + seqKv);
        var k = new ushort[(long)seqKv * stride];
        var v = new ushort[(long)seqKv * stride];
        for (int j = 0; j < seqKv; j++)
            for (int h = 0; h < numKvHeads; h++)
            {
                long b = (long)j * stride + h * headDim;
                // Sink k[0] = 20, tail -2..-6. V: sink 0, tail 48..80 (one sign): the output is the tail mass x ~64.
                k[b] = FloatToHalf(j == 0 ? 20f : -2f - 4f * (float)rng.NextDouble());
                for (int d = 1; d < headDim; d++) k[b + d] = FloatToHalf((float)(rng.NextDouble() * 0.2 - 0.1));
                for (int d = 0; d < headDim; d++) v[b + d] = FloatToHalf(j == 0 ? 0f : 48f + 32f * (float)rng.NextDouble());
            }
        var q = new ushort[seqQ * numHeads * headDim];
        // 8 * 20 / sqrt(128) = 14.1, the tail 15.6-18.4 nats below the sink; q0 scales with sqrt(hd) so every head_dim sees the same gaps.
        float q0 = 8f * MathF.Sqrt(headDim / 128f);
        for (int r = 0; r < seqQ; r++)
            for (int h = 0; h < numHeads; h++)
                q[(r * numHeads + h) * headDim] = FloatToHalf(q0 * (1f + 0.02f * h));
        return (q, k, v);
    }

    private float[] ReadHalfAsFloat(nint dev, int n) => ReadHalf(dev, n).Select(HalfToFloat).ToArray();

    private static float[] ReadFloat(nint dev, int n)
    {
        var h = new float[n];
        fixed (float* p = h) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dev, (nuint)(n * 4L)).ThrowOnError();
        return h;
    }

    /// <summary>
    /// The sink/tail fixture through <paramref name="kernel"/> (given device Q, K, V; returns its output as floats) and through the
    /// scalar <c>attention_f16</c> oracle (F32 P): worst relative error over rows [minRow, seqQ).
    /// </summary>
    private (double Worst, int At, double Got, double Want) SinkTailRun(int headDim, int numHeads, int numKvHeads, int seqQ,
        int seqKv, int minRow, Func<nint, nint, nint, float[]> kernel)
    {
        var (q, k, v) = SinkTailData(headDim, numHeads, numKvHeads, seqQ, seqKv);
        nint dq = Upload(q), dk = Upload(k), dv = Upload(v), oRef = Alloc(q.Length * 2L);
        try
        {
            _kernels!.LaunchAttention(dq, dk, dv, oRef, seqQ, seqKv, numHeads, numKvHeads, headDim, seqKv - seqQ, 0, _stream!.Handle);
            _stream.Synchronize();
            float[] want = ReadHalfAsFloat(oRef, q.Length), got = kernel(dq, dk, dv);
            double worst = 0;
            int at = 0;
            for (int i = minRow * numHeads * headDim; i < want.Length; i++)
            {
                double err = Math.Abs(got[i] - want[i]) / Math.Max(Math.Abs(want[i]), 1e-30);
                if (!(err <= worst)) { worst = err; at = i; }
            }
            return (worst, at, got[at], want[at]);
        }
        finally
        {
            foreach (nint p in new[] { dq, dk, dv, oRef }) CudaDriverApi.cuMemFree_v2(p);
        }
    }

    /// <summary>The sink/tail fixture through <paramref name="kernels"/>' kvq F16 flash prefill entry.</summary>
    private (double Worst, int At, double Got, double Want) SinkTailError(CudaKernels kernels, int headDim, int numHeads, int numKvHeads,
        int seqQ, int seqKv) =>
        SinkTailRun(headDim, numHeads, numKvHeads, seqQ, seqKv, 0, (dq, dk, dv) =>
        {
            int n = seqQ * numHeads * headDim;
            nint o = Alloc(n * 2L);
            try
            {
                kernels.LaunchAttentionFlashPrefillKvq(dq, dk, dv, null, null, o, seqQ, seqKv, numHeads, numKvHeads, headDim,
                    seqKv - seqQ, _stream!.Handle);
                _stream.Synchronize();
                return ReadHalfAsFloat(o, n);
            }
            finally { CudaDriverApi.cuMemFree_v2(o); }
        });

    private void AssertTail(string what, (double Worst, int At, double Got, double Want) r)
    {
        _out.WriteLine($"{what}: worst relative error {r.Worst:E3} at {r.At} ({r.Got:E4} vs attention_f16 {r.Want:E4})");
        Assert.True(r.Worst <= 0.01, $"{what} loses the softmax tail: {r.Got:E4} vs attention_f16 {r.Want:E4} at {r.At} " +
                                     $"(relative error {r.Worst:E3})");
    }

    /// <summary>#1953, same P packing: the hd256 flash prefill (F32 output; hd256 models' F16 prefill).</summary>
    [SkippableFact]
    public void FlashPrefillHd256_SinkDominatedLongRow_KeepsTheTailMass()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null || !_kernels.HasAttentionFlashPrefillHd256, "hd256 flash prefill not loaded");
        const int hd = 256, nh = 8, nkv = 2, seqQ = 16, seqKv = 32771;
        AssertTail("hd256 flash prefill 32K", SinkTailRun(hd, nh, nkv, seqQ, seqKv, 0, (dq, dk, dv) =>
        {
            int n = seqQ * nh * hd;
            nint o = Alloc(n * 4L);
            try
            {
                _kernels.LaunchAttentionFlashPrefillHd256(dq, dk, dv, o, seqQ, seqKv, nh, nkv, seqKv - seqQ, _stream!.Handle);
                _stream.Synchronize();
                return ReadFloat(o, n);
            }
            finally { CudaDriverApi.cuMemFree_v2(o); }
        }));
    }

    /// <summary>
    /// #1953, same P packing: the dense split-KV decode (#1922, F16 output). Split 1 runs the whole 32K row in one block (the bug's
    /// regime); split 0 = the production heuristic.
    /// </summary>
    [SkippableTheory]
    [InlineData(128, 1)]
    [InlineData(128, 0)]
    [InlineData(64, 1)]
    public void MmaDecodeDense_SinkDominatedLongRow_KeepsTheTailMass(int hd, int forcedSplit)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null || !_kernels.HasAttentionMmaDecodeDense(hd, KvCacheDType.F32, KvCacheDType.F32), "dense decode v2 not loaded");
        const int nh = 8, nkv = 2, seqKv = 32771;
        int maxSafe = _kernels.MaxSafeAttentionMmaDecodeDense(nkv, hd, KvCacheDType.F32, KvCacheDType.F32);
        Skip.If(maxSafe < 1, "dense decode v2 cannot launch here");
        int split = forcedSplit > 0 ? forcedSplit : CudaKernels.ComputeAttentionMmaV2KvSplit(seqKv, maxSafe);
        AssertTail($"dense decode hd{hd} split {split} 32K", SinkTailRun(hd, nh, nkv, 1, seqKv, 0, (dq, dk, dv) =>
        {
            int n = nh * hd;
            nint o = Alloc(n * 2L), pm = Alloc(nh * split * 4L), ps = Alloc(nh * split * 4L),
                 po = Alloc((long)nh * split * hd * CudaKernels.AttentionMmaDecodeDensePartialBytes);
            try
            {
                _kernels.LaunchAttentionMmaDecodeDense(dq, dk, dv, KvCacheDType.F32, KvCacheDType.F32, o, seqKv, nh, nkv, hd, split,
                    pm, ps, po, _stream!.Handle);
                _stream.Synchronize();
                return ReadHalfAsFloat(o, n);
            }
            finally { foreach (nint p in new[] { o, pm, ps, po }) CudaDriverApi.cuMemFree_v2(p); }
        }));
    }

    /// <summary>#1953, same P packing: the hd256 cooperative mma decode, v2 (#988, the default) and v1, at split 1.</summary>
    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void MmaDecodeGqaSplit_SinkDominatedLongRow_KeepsTheTailMass(bool v2)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null || !_kernels.HasAttentionMmaDecodeGqaSplit, "mma decode gqa split not loaded");
        const int hd = 256, nh = 8, nkv = 2, seqKv = 32771;
        bool saved = CudaKernels.MmaDecodeV2Enabled;
        try
        {
            CudaKernels.MmaDecodeV2Enabled = v2;
            Skip.If(_kernels.MaxSafeAttentionMmaDecodeGqaSplit(nkv, hd, nh / nkv) < 1, "mma decode cannot launch here");
            AssertTail($"mma decode {(v2 ? "v2" : "v1")} hd256 32K", SinkTailRun(hd, nh, nkv, 1, seqKv, 0, (dq, dk, dv) =>
            {
                int n = nh * hd;
                nint o = Alloc(n * 4L), pm = Alloc(nh * 4L), ps = Alloc(nh * 4L), po = Alloc((long)nh * hd * 4);
                try
                {
                    _kernels.LaunchAttentionMmaDecodeGqaSplit(dq, dk, dv, o, seqKv, nh, nkv, 1, pm, ps, po, _stream!.Handle);
                    _stream.Synchronize();
                    return ReadFloat(o, n);
                }
                finally { foreach (nint p in new[] { o, pm, ps, po }) CudaDriverApi.cuMemFree_v2(p); }
            }));
        }
        finally { CudaKernels.MmaDecodeV2Enabled = saved; }
    }

    /// <summary>
    /// #1953, the same flush in the G3 fallback (cuBLAS QK^T, causal softmax, cuBLAS P.V): its FP16 probability plane now holds
    /// P * 2^15 and the P.V GEMM's alpha is 2^-15. Offset path (32 query rows over a 32K history, the KL gate's G3 arm).
    /// <c>DOTLLM_1953_G3_MUTANT=1</c> pairs a PTX built with <c>-DG3_P_SCALE=1.0f</c> with alpha 1 (the pre-fix numerics).
    /// </summary>
    [SkippableTheory]
    [InlineData(128)]
    [InlineData(64)]
    public void G3Offset_SinkDominatedLongRow_KeepsTheTailMass(int hd)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null || !_kernels.HasAttentionSoftmaxCausalOffsetF32In, "G3 offset softmax not loaded");
        const int nh = 8, nkv = 2, seqQ = CudaG3Attention.OffsetMinQueryRows, seqKv = 32771;
        using var cublas = CudaCublasHandle.Create();
        cublas.SetStream(_stream!);
        using var g3 = new CudaG3Attention(_kernels);
        bool saved = CudaG3Attention.Enabled;
        float savedAlpha = CudaG3Attention.ProbsScaleInverse;
        try
        {
            CudaG3Attention.Enabled = true;
            if (Environment.GetEnvironmentVariable("DOTLLM_1953_G3_MUTANT") == "1") CudaG3Attention.ProbsScaleInverse = 1f;
            Skip.IfNot(g3.CanUseOffset(seqQ, seqKv, seqKv - seqQ, 0, nh, nkv), "G3 offset path not usable for this shape");
            AssertTail($"G3 offset hd{hd} 32K", SinkTailRun(hd, nh, nkv, seqQ, seqKv, 0, (dq, dk, dv) =>
            {
                int n = seqQ * nh * hd;
                nint o = Alloc(n * 2L);
                try
                {
                    g3.RunOffset(cublas.Handle, dq, dk, dv, o, seqQ, seqKv - seqQ, nh, nkv, hd, _stream!.Handle);
                    _stream.Synchronize();
                    return ReadHalfAsFloat(o, n);
                }
                finally { CudaDriverApi.cuMemFree_v2(o); }
            }));
        }
        finally
        {
            CudaG3Attention.Enabled = saved;
            CudaG3Attention.ProbsScaleInverse = savedAlpha;
        }
    }

    /// <summary>#1953, same P packing: G-flash (attention_flash_mma, hd64 cold prefill, opt-in): rows 4096+ of an 8,192-row prefill.</summary>
    [SkippableFact]
    public void FlashMmaHd64_SinkDominatedLongRow_KeepsTheTailMass()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null || !_kernels.HasAttentionFlashMma, "attention_flash_mma not loaded");
        const int hd = 64, nh = 8, nkv = 2, seq = 8192;
        AssertTail("G-flash hd64 rows 4096-8191", SinkTailRun(hd, nh, nkv, seq, seq, 4096, (dq, dk, dv) =>
        {
            int n = seq * nh * hd;
            nint o = Alloc(n * 2L);
            try
            {
                _kernels.LaunchAttentionFlashMma(dq, dk, dv, o, seq, nh, nkv, hd, 1f / MathF.Sqrt(hd), _stream!.Handle);
                _stream.Synchronize();
                return ReadHalfAsFloat(o, n);
            }
            finally { CudaDriverApi.cuMemFree_v2(o); }
        }));
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
