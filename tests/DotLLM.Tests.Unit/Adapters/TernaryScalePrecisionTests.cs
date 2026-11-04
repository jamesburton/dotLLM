using System.Text;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Adapters.Ternary;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.ChunkReuse;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Adapters;

/// <summary>
/// Are F16 / bf16 per-row adapter scales sufficient (#1620, #1107)? Compares the adapter output delta computed with F32 scales against
/// scales rounded RNE to F16 / bf16 (accumulation stays F32 in registers; the weights are exact ternary), over the reference's own
/// sample adapters, synthetic adapters with realistic absmean-scale distributions (including range extremes), and end to end on a
/// tiny model. The numbers are printed; the assertions encode the verdict recorded in docs/TERNARY_ADAPTER.md.
/// </summary>
public sealed unsafe class TernaryScalePrecisionTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly string _dir = Directory.CreateTempSubdirectory("nsscale").FullName;
    public TernaryScalePrecisionTests(ITestOutputHelper o) => _out = o;
    public void Dispose() { try { Directory.Delete(_dir, true); } catch { } }

    private static string Fx(string name) => Path.Combine(AppContext.BaseDirectory, "Adapters", "Fixtures", name + ".nsadapter");

    // ---------------------------------------------------------------------------------------------- rounding rules ----

    /// <summary>F32 -> F16 round-to-nearest-even (IEEE binary16, overflow to +-Inf), back to F32 (exact).</summary>
    internal static float RoundF16(float x) => (float)(Half)x;

    /// <summary>F32 -> bfloat16 round-to-nearest-even (keeps the F32 exponent, 8 significant bits), back to F32 (exact).</summary>
    internal static float RoundBf16(float x)
    {
        uint b = (uint)BitConverter.SingleToInt32Bits(x);
        if ((b & 0x7FFFFFFFu) > 0x7F800000u) return x;               // NaN
        b += 0x7FFFu + ((b >> 16) & 1u);
        return BitConverter.Int32BitsToSingle((int)(b & 0xFFFF0000u));
    }

    private static NsAdapterPackage WithScales(NsAdapterPackage p, Func<float, float> round)
    {
        var down = new PackedTernaryTensor(p.Down.Rows, p.Down.Columns, (byte[])p.Down.Data.Clone(), p.Down.Scales.Select(round).ToArray());
        var up = new PackedTernaryTensor(p.Up.Rows, p.Up.Columns, (byte[])p.Up.Data.Clone(), p.Up.Scales.Select(round).ToArray());
        var m = new NsAdapterManifest
        {
            TargetModel = p.Manifest.TargetModel, TargetLayer = p.Manifest.TargetLayer, InputDimension = p.Manifest.InputDimension, Rank = p.Manifest.Rank,
            Alpha = p.Manifest.Alpha, ActivationThreshold = p.Manifest.ActivationThreshold, Domain = p.Manifest.Domain,
        };
        return new NsAdapterPackage(m, down, up);
    }

    // ---------------------------------------------------------------------------------------------- managed circuit (port of the reference) ----

    private sealed record Out(float[] Delta, sbyte[] Q, bool Open);

    private static Out Circuit(NsAdapterPackage p, float[] h, bool floatQ = false)
    {
        int d = p.Manifest.InputDimension, k = p.Manifest.Rank;
        float maxH = 0f; foreach (float v in h) maxH = Math.Max(maxH, Math.Abs(v));
        float qs = maxH > 1e-6f ? 127f / maxH : 0f;
        var x = new sbyte[d];
        for (int i = 0; i < d; i++) x[i] = (sbyte)Math.Clamp((int)Math.Round(h[i] * qs), -128, 127);
        var inter = new float[k];
        for (int j = 0; j < k; j++)
        {
            long s = 0;
            for (int c = 0; c < d; c++) { sbyte w = p.Down[j, c]; if (w == 1) s += x[c]; else if (w == -1) s -= x[c]; }
            inter[j] = Math.Max(0f, (float)s * p.Down.Scales[j]);
        }
        float max = inter.Max();
        var delta = new float[d]; var q = new sbyte[k];
        if (!(max > Math.Max(p.Manifest.ActivationThreshold, 1e-6f))) return new Out(delta, q, false);
        float quant = 127f / max;
        for (int j = 0; j < k; j++) q[j] = (sbyte)Math.Clamp((int)Math.Round(inter[j] * quant), -128, 127);
        float dq = (max / 127f) * p.Manifest.Alpha;
        for (int r = 0; r < d; r++)
        {
            long s = 0; double sf = 0;
            for (int j = 0; j < k; j++) { sbyte w = p.Up[r, j]; int wi = w; s += wi * q[j]; sf += wi * (double)(inter[j] * quant); }
            delta[r] = floatQ ? (float)(sf * p.Up.Scales[r] * dq) : (float)s * p.Up.Scales[r] * dq;
        }
        return new Out(delta, q, true);
    }

    // ---------------------------------------------------------------------------------------------- generators ----

    private sealed class Rng
    {
        private ulong _s;
        public Rng(ulong seed) => _s = seed;
        public ulong Next() { _s ^= _s << 13; _s ^= _s >> 7; _s ^= _s << 17; return _s; }
        public double U() => ((Next() >> 11) + 0.5) / 9007199254740992.0;
        public double Gauss() => Math.Sqrt(-2 * Math.Log(U())) * Math.Cos(2 * Math.PI * U());
    }

    /// <summary>Synthetic adapter: ternary weights with ~1/3 zeros (absmean quantisation of Gaussian weights), per-row scale = median * exp(sigma * N(0,1)).</summary>
    private static NsAdapterPackage Synth(int rank, int d, ulong seed, double median, double sigma, float alpha = 1f)
    {
        var g = new Rng(seed);
        sbyte W() { double u = g.U(); return u < 0.33 ? (sbyte)0 : u < 0.665 ? (sbyte)1 : (sbyte)-1; }
        var down = new PackedTernaryTensor(rank, d); var up = new PackedTernaryTensor(d, rank);
        for (int r = 0; r < rank; r++) for (int c = 0; c < d; c++) down[r, c] = W();
        for (int r = 0; r < d; r++) for (int c = 0; c < rank; c++) up[r, c] = W();
        for (int r = 0; r < rank; r++) down.Scales[r] = (float)(median * Math.Exp(sigma * g.Gauss()));
        for (int r = 0; r < d; r++) up.Scales[r] = (float)(median * Math.Exp(sigma * g.Gauss()));
        var m = new NsAdapterManifest { TargetModel = "synthetic", TargetLayer = 1, InputDimension = d, Rank = rank, Alpha = alpha, Domain = "synthetic" };
        return new NsAdapterPackage(m, down, up);
    }

    private static float[] Residual(Rng g, int d)
    {
        var h = new float[d];
        for (int i = 0; i < d; i++) h[i] = (float)g.Gauss() * (i % 97 == 0 ? 12f : 1f) + 0.3f;   // outlier channels + a mean, like real residuals
        return h;
    }

    // ---------------------------------------------------------------------------------------------- metrics ----

    private sealed record Metrics(double MaxRel, double MeanRel, double RelL2, double QFlip, double OutInt8Flip, int GateFlips, int Rows, int Zeroed);

    private static Metrics Compare(NsAdapterPackage p, Func<float, float> round, int rows, ulong seed)
    {
        var pr = WithScales(p, round);
        var g = new Rng(seed);
        int d = p.Manifest.InputDimension;
        double maxRel = 0, sumRel = 0; long nRel = 0, qDiff = 0, qTot = 0, oDiff = 0, oTot = 0; double num = 0, den = 0; int gate = 0, zeroed = 0;
        for (int row = 0; row < rows; row++)
        {
            float[] h = Residual(g, d);
            Out a = Circuit(p, h), b = Circuit(pr, h);
            if (a.Open != b.Open) { gate++; continue; }
            if (!a.Open) continue;
            for (int j = 0; j < a.Q.Length; j++) { qTot++; if (a.Q[j] != b.Q[j]) qDiff++; }
            float amax = a.Delta.Max(Math.Abs);
            float qs = amax > 0 ? 127f / amax : 0f;
            for (int i = 0; i < d; i++)
            {
                double x = a.Delta[i], y = b.Delta[i];
                num += (y - x) * (y - x); den += x * x;
                if (x != 0) { double rel = Math.Abs(y - x) / Math.Abs(x); maxRel = Math.Max(maxRel, rel); sumRel += rel; nRel++; }
                if (x != 0 && y == 0) zeroed++;
                oTot++;
                if ((int)Math.Round(x * qs) != (int)Math.Round(y * qs)) oDiff++;
            }
        }
        return new Metrics(maxRel, nRel > 0 ? sumRel / nRel : 0, den > 0 ? Math.Sqrt(num / den) : 0, qTot > 0 ? (double)qDiff / qTot : 0, oTot > 0 ? (double)oDiff / oTot : 0, gate, rows, zeroed);
    }

    /// <summary>The adapter's own int8 requantisation noise: relL2 between the shipped circuit and the same circuit with an unquantised (float) bottleneck.</summary>
    private static double Int8Floor(NsAdapterPackage p, int rows, ulong seed)
    {
        var g = new Rng(seed); double num = 0, den = 0;
        for (int row = 0; row < rows; row++)
        {
            float[] h = Residual(g, p.Manifest.InputDimension);
            Out a = Circuit(p, h), b = Circuit(p, h, floatQ: true);
            for (int i = 0; i < a.Delta.Length; i++) { double x = b.Delta[i], y = a.Delta[i]; num += (y - x) * (y - x); den += x * x; }
        }
        return den > 0 ? Math.Sqrt(num / den) : 0;
    }

    private string Report(string name, NsAdapterPackage p, int rows = 60)
    {
        var sb = new StringBuilder();
        float[] all = p.Down.Scales.Concat(p.Up.Scales).Where(s => s != 0).Select(Math.Abs).ToArray();
        sb.AppendLine($"{name}: rank={p.Manifest.Rank} D={p.Manifest.InputDimension} |scale| min={all.Min():G4} max={all.Max():G4}");
        sb.AppendLine($"   int8-requantisation noise floor (relL2 of the shipped circuit vs an unquantised bottleneck): {Int8Floor(p, rows, 42):E2}");
        foreach (var (fmt, round) in new (string, Func<float, float>)[] { ("F16 ", RoundF16), ("bf16", RoundBf16) })
        {
            Metrics m = Compare(p, round, rows, 42);
            sb.AppendLine($"   {fmt}: maxRel={m.MaxRel:E2} meanRel={m.MeanRel:E2} relL2={m.RelL2:E2} q_j flips={m.QFlip:P3} outInt8 flips={m.OutInt8Flip:P3} gateFlips={m.GateFlips}/{m.Rows} zeroed={m.Zeroed}");
        }
        return sb.ToString();
    }

    // ---------------------------------------------------------------------------------------------- tests ----

    [Fact]
    public void RoundingRules_AreRoundToNearestEven_WithTheDocumentedRanges()
    {
        // F16: 11 significant bits, overflow above 65504 (rounds to inf at >= 65520), subnormals below 6.1035156e-5 down to 5.96e-8.
        Assert.Equal(1.0f, RoundF16(1.0f + 1f / 4096f));            // exactly half an ulp above 1.0 -> ties to even (1.0)
        Assert.Equal(1.0f + 1f / 1024f, RoundF16(1.0f + 3f / 4096f)); // 1.5 ulp -> ties to even (up)
        Assert.Equal(65504f, RoundF16(65504f));
        Assert.True(float.IsPositiveInfinity(RoundF16(65520f)));
        Assert.Equal(0f, RoundF16(2.9e-8f));                          // below half the smallest subnormal -> 0
        Assert.Equal(5.9604645e-8f, RoundF16(3.1e-8f));
        // bf16: 8 significant bits, F32 exponent range.
        Assert.Equal(1.0f, RoundBf16(1.0f + 1f / 256f));              // half ulp (ulp = 2^-7) -> ties to even
        Assert.Equal(1.0f + 1f / 64f, RoundBf16(1.0f + 3f / 256f));
        Assert.Equal(3.0e38f, RoundBf16(3.0e38f), 3.0e38f * 0.004f);
        Assert.Equal(1e-30f, RoundBf16(1e-30f), 1e-30f * 0.004f);
    }

    [Fact]
    public void ReferenceSampleAdapters_UseUnitScales_SoAnyFormatIsExact()
    {
        foreach (string n in new[] { "bracket-parity", "csharp14-syntax-attractor", "modulo97-counter" })
        {
            var p = NsAdapterPackage.Load(Fx(n));
            Assert.All(p.Down.Scales.Concat(p.Up.Scales), s => Assert.Equal(1.0f, s));
            _out.WriteLine(Report(n, p, 40));
            Assert.Equal(0.0, Compare(p, RoundF16, 20, 1).RelL2);
            Assert.Equal(0.0, Compare(p, RoundBf16, 20, 1).RelL2);
        }
    }

    [Fact]
    public void Synthetic_RealisticScaleDistributions_Report_And_F16VerdictHolds()
    {
        // Median scale of an absmean-quantised row = mean|w| of the float weights it came from: ~0.02 (init std), up to ~1 (large trained), down to ~1e-3.
        (string name, double median, double sigma)[] cases =
        [
            ("absmean 0.02 (init scale)", 0.02, 0.3),
            ("absmean 0.002", 0.002, 0.5),
            ("absmean 1.0 (unit-like)", 1.0, 0.3),
            ("absmean 40", 40.0, 0.3),
        ];
        foreach (var (name, median, sigma) in cases)
            foreach (int rank in new[] { 8, 32, 128 })
            {
                var p = Synth(rank, 4096, (ulong)(rank * 7919 + (int)(median * 1000) + 1), median, sigma);
                _out.WriteLine(Report($"{name} r={rank}", p, 30));
                Metrics f16 = Compare(p, RoundF16, 30, 9), bf = Compare(p, RoundBf16, 30, 9);
                // F16 rel error per element is bounded by 2^-11 in each of up to two scale factors (the down scale only perturbs q_j by <=1 level).
                Assert.True(f16.RelL2 < 4e-3, $"{name} r={rank}: F16 relL2 {f16.RelL2:E2}");
                Assert.Equal(0, f16.GateFlips);
                // F16 scale noise stays at or below the adapter's own int8 requantisation noise (typically 0.2-0.8x); bf16 reaches ~1x or more.
                double floor = Int8Floor(p, 30, 9);
                Assert.True(f16.RelL2 <= floor, $"{name} r={rank}: F16 relL2 {f16.RelL2:E2} vs int8 floor {floor:E2}");
                // bf16 is ~8x coarser than F16 (2^-8 vs 2^-11): the same bound but looser, reported for the record.
                Assert.True(bf.RelL2 < 3e-2, $"{name} r={rank}: bf16 relL2 {bf.RelL2:E2}");
            }
    }

    [Fact]
    public void Synthetic_RangeExtremes_F16_Overflows_AndUnderflows_Bf16_DoesNot()
    {
        // Scales of 1e-6 (below F16 normal range; subnormal spacing 6e-8 => ~6% relative error) and 1e5 (above F16 max 65504 => inf).
        var tiny = Synth(16, 4096, 3, 1e-6, 0.3);
        var huge = Synth(16, 4096, 4, 1e5, 0.3);
        _out.WriteLine(Report("scales ~1e-6", tiny, 20));
        _out.WriteLine(Report("scales ~1e5", huge, 20));
        Assert.True(Compare(tiny, RoundF16, 20, 5).RelL2 > 1e-2, "F16 must visibly degrade subnormal scales");
        Assert.True(Compare(tiny, RoundBf16, 20, 5).RelL2 < 3e-2);
        Assert.True(huge.Up.Scales.Select(RoundF16).Any(float.IsInfinity), "F16 overflows scales above 65504");
        Assert.True(Compare(huge, RoundBf16, 20, 5).RelL2 < 3e-2);
    }


    // ---------------------------------------------------------------------------------------------- the F16 in-memory format ----

    private static float[] Apply(NsAdapterPackage[] pkgs, float[] h, TernaryScaleFormat fmt, DotLLM.Cpu.Kernels.TernaryKernelIsa isa, out TernaryScaleFormat[] used)
    {
        using var hook = new TernaryResidualHook(pkgs[0].Manifest.InputDimension) { Isa = isa, ScaleFormat = fmt };
        foreach (var p in pkgs) hook.Add(p);
        used = Enumerable.Range(0, hook.CircuitCount).Select(hook.ScaleFormatAt).ToArray();
        var row = (float[])h.Clone();
        hook.ModifyResidual(row, pkgs[0].Manifest.TargetLayer, 1);
        return row;
    }

    private static IEnumerable<DotLLM.Cpu.Kernels.TernaryKernelIsa> Isas()
    {
        yield return DotLLM.Cpu.Kernels.TernaryKernelIsa.Scalar;
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported) yield return DotLLM.Cpu.Kernels.TernaryKernelIsa.Avx2;
    }

    private static void AssertBits(float[] a, float[] b, string ctx)
    {
        for (int i = 0; i < a.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(b[i]), $"{ctx}: element {i}: {a[i]:R} != {b[i]:R}");
    }

    [Theory]
    [InlineData(8, 203)]    // D not a multiple of 8: exercises the zero-padded tail of the F16 up-scale
    [InlineData(32, 512)]
    [InlineData(5, 64)]
    public void F16Storage_IsBitIdentical_ToF32Hook_OnScalesPreRoundedRNE(int rank, int d)
    {
        var p = Synth(rank, d, (ulong)(rank * 31 + d), 0.02, 0.4, alpha: 2f);
        var rounded = WithScales(p, RoundF16);
        var g = new Rng(7);
        foreach (var isa in Isas())
            for (int row = 0; row < 8; row++)
            {
                float[] h = Residual(g, d);
                float[] f16 = Apply([p], h, TernaryScaleFormat.F16, isa, out var used);
                Assert.Equal(TernaryScaleFormat.F16, used[0]);
                AssertBits(Apply([rounded], h, TernaryScaleFormat.F32, isa, out _), f16, $"{isa} row {row}: F16 storage vs F32 storage of RNE-rounded scales");
            }
        // Perturbation proof that the F16 path really changes the numbers (these scales are not F16-exact).
        float[] hh = Residual(new Rng(8), d);
        Assert.NotEqual(Apply([p], hh, TernaryScaleFormat.F32, Isas().Last(), out _), Apply([p], hh, TernaryScaleFormat.F16, Isas().Last(), out _));
    }

    [Fact]
    public void F16Storage_FallsBackToF32_ForCircuitsOutsideTheF16NormalRange_AndNeverLosesThem()
    {
        var tiny = Synth(8, 128, 11, 1e-6, 0.3);     // subnormal scales
        var huge = Synth(8, 128, 12, 1e5, 0.3);      // overflow
        var ok = Synth(8, 128, 13, 0.05, 0.3);
        float[] h = Residual(new Rng(3), 128);
        foreach (var isa in Isas())
        {
            float[] mixed = Apply([ok, tiny, huge], h, TernaryScaleFormat.F16, isa, out var used);
            Assert.Equal([TernaryScaleFormat.F16, TernaryScaleFormat.F32, TernaryScaleFormat.F32], used);
            // The two fallback circuits are exact F32; the F16 one equals the RNE-rounded package. Compare against an all-F32 hook using the rounded 'ok'.
            AssertBits(Apply([WithScales(ok, RoundF16), tiny, huge], h, TernaryScaleFormat.F32, isa, out _), mixed, $"{isa} mixed formats");
        }
        Assert.True(float.IsFinite(Apply([huge], h, TernaryScaleFormat.F16, Isas().Last(), out _).Sum()));
    }

    [Fact]
    public void F16Storage_ZeroScaleRows_AreKept_AndDefaultIsF32()
    {
        var p = Synth(4, 64, 21, 0.1, 0.2);
        p.Up.Scales[5] = 0f; p.Down.Scales[1] = 0f;
        float[] h = Residual(new Rng(4), 64);
        foreach (var isa in Isas())
        {
            AssertBits(Apply([WithScales(p, RoundF16)], h, TernaryScaleFormat.F32, isa, out _), Apply([p], h, TernaryScaleFormat.F16, isa, out var u), $"{isa} zero scales");
            Assert.Equal(TernaryScaleFormat.F16, u[0]);
            Apply([p], h, TernaryScaleFormat.F32, isa, out var d);
            Assert.Equal(TernaryScaleFormat.F32, d[0]);
        }
        using var hook = new TernaryResidualHook(64);
        Assert.Equal(TernaryScaleFormat.F32, hook.ScaleFormat);   // default: bit-identical to the reference
    }

    // ---------------------------------------------------------------------------------------------- end to end on a tiny model ----

    private sealed class Tiny : IDisposable
    {
        private readonly GgufFile _gguf;
        public ModelConfig Config { get; }
        public TransformerModel Model { get; }
        public Tiny(string dir)
        {
            string path = Path.Combine(dir, "tiny.gguf");
            File.WriteAllBytes(path, SyntheticDenseGguf.Build("llama"));
            _gguf = GgufFile.Open(path);
            Config = GgufModelConfigExtractor.Extract(_gguf.Metadata);
            Model = TransformerModel.LoadFromGguf(_gguf, Config, ThreadingConfig.SingleThreaded);
        }
        public float[] Logits(int[] ids)
        {
            using ITensor t = Model.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1);
            return new ReadOnlySpan<float>((float*)t.DataPointer, (int)t.Shape.ElementCount).ToArray();
        }
        public void Dispose() { Model.Dispose(); _gguf.Dispose(); }
    }

    private static double KlTop(float[] p, float[] q, int vocab, out int top1Agree, out int rows)
    {
        rows = p.Length / vocab; top1Agree = 0; double kl = 0;
        for (int r = 0; r < rows; r++)
        {
            double Lse(float[] a) { double m = double.MinValue; for (int i = 0; i < vocab; i++) m = Math.Max(m, a[r * vocab + i]); double s = 0; for (int i = 0; i < vocab; i++) s += Math.Exp(a[r * vocab + i] - m); return m + Math.Log(s); }
            double lp = Lse(p), lq = Lse(q), k = 0; int ap = 0, aq = 0;
            for (int i = 0; i < vocab; i++)
            {
                double a = p[r * vocab + i] - lp, b = q[r * vocab + i] - lq;
                k += Math.Exp(a) * (a - b);
                if (p[r * vocab + i] > p[r * vocab + ap]) ap = i;
                if (q[r * vocab + i] > q[r * vocab + aq]) aq = i;
            }
            kl += k; if (ap == aq) top1Agree++;
        }
        return kl / rows;
    }

    [Fact]
    public void EndToEnd_TinyModel_F16AndBf16ScalesLeaveLogitsWithinTheInt8Floor()
    {
        using var tiny = new Tiny(_dir);
        int d = tiny.Config.HiddenSize, vocab = tiny.Config.VocabSize;
        int[][] seqs = [[3, 7, 11, 2, 40, 5], [9, 1, 30, 17, 4, 22, 8, 12], [5, 5, 6, 7, 8, 9, 10, 11, 12, 13]];
        float[][] baseline = seqs.Select(tiny.Logits).ToArray();
        foreach (double median in new[] { 0.02, 1.0 })
        {
            var pkg = Synth(8, d, (ulong)(median * 100 + 5), median, 0.3, alpha: (float)(3.0 / Math.Max(median * 40, 1.0)) * (float)(median < 0.5 ? 20 : 1));
            double klEffect = 0, klF16 = 0, klBf = 0; int tot = 0, aggF16 = 0, aggBf = 0, aggEff = 0;
            foreach (var (name, round) in new (string, Func<float, float>)[] { ("f32", x => x), ("f16", RoundF16), ("bf16", RoundBf16) })
            {
                using var hook = new TernaryResidualHook(d);
                hook.Add(WithScales(pkg, round));
                tiny.Model.ResidualModifier = hook;
                float[][] outs = seqs.Select(tiny.Logits).ToArray();
                tiny.Model.ResidualModifier = null;
                if (name == "f32") { _f32 = outs; continue; }
                for (int s = 0; s < seqs.Length; s++)
                {
                    double k = KlTop(_f32![s], outs[s], vocab, out int agree, out int rows);
                    if (name == "f16") { klF16 += k * rows; aggF16 += agree; } else { klBf += k * rows; aggBf += agree; }
                    if (name == "f16") { tot += rows; klEffect += KlTop(baseline[s], _f32![s], vocab, out int ae, out _) * rows; aggEff += ae; }
                }
            }
            _out.WriteLine($"tiny model D={d} scale median {median}: adapter effect vs no adapter: KL={klEffect / tot:E2} top1 agree {aggEff}/{tot};  " +
                $"F16 vs F32 scales: KL={klF16 / tot:E2} top1 {aggF16}/{tot};  bf16 vs F32 scales: KL={klBf / tot:E2} top1 {aggBf}/{tot}");
            Assert.True(klF16 / tot <= klEffect / tot * 1e-2 + 1e-12, "F16-scale KL must be <1% of the adapter's own effect");
        }
    }

    private float[][]? _f32;
}
