using DotLLM.Core.Configuration;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Adapters.Ternary;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.ChunkReuse;
using System.Runtime.Intrinsics;
using Xunit;

namespace DotLLM.Tests.Unit.Adapters;

/// <summary>Ternary residual adapter (#1616): kernel exactness, .nsadapter round-trip, composition, engine wiring.</summary>
public sealed unsafe class TernaryAdapterTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("nsad").FullName;

    public void Dispose() { try { Directory.Delete(_dir, true); } catch { } }

    // ------------------------------------------------------------------ independent reference ----

    /// <summary>
    /// Independent implementation of the canonical order from managed sbyte matrices (never touches the packed
    /// layout or the kernel): 16-lane P/M accumulators, pairwise fold, per-rank alpha, relu, per-lane acc.
    /// </summary>
    private static float[] Reference(float[] x, int d, int rank, sbyte[] down, sbyte[] upT, float[] rankAlpha, bool relu)
    {
        int chunks = (d + 15) / 16;
        var h = new float[rank];
        for (int j = 0; j < rank; j++)
        {
            var p = new float[16];
            var m = new float[16];
            for (int c = 0; c < chunks; c++)
                for (int l = 0; l < 16; l++)
                {
                    int col = c * 16 + l;
                    float v = col < d ? x[col] : 0f;
                    sbyte w = col < d ? down[j * d + col] : (sbyte)0;
                    p[l] += w == 1 ? v : 0f;
                    m[l] += w == -1 ? v : 0f;
                }
            float s = Fold(p) - Fold(m);
            float a = relu ? (s > 0f ? s : 0f) : s;
            h[j] = rankAlpha[j] * a;
        }
        var y = new float[d];
        for (int col = 0; col < d; col++)
        {
            float acc = 0f;
            for (int j = 0; j < rank; j++)
            {
                sbyte w = upT[j * d + col];
                acc += w == 1 ? h[j] : 0f;
                acc -= w == -1 ? h[j] : 0f;
            }
            y[col] = x[col] + acc;
        }
        return y;
    }

    private static float Fold(float[] a)
    {
        var t = (float[])a.Clone();
        for (int i = 0; i < 8; i++) t[i] += t[i + 8];
        for (int i = 0; i < 4; i++) t[i] += t[i + 4];
        for (int i = 0; i < 2; i++) t[i] += t[i + 2];
        return t[0] + t[1];
    }

    private static sbyte[] RandomTernary(Random rng, int n, double zeroFrac = 0.4)
    {
        var a = new sbyte[n];
        for (int i = 0; i < n; i++)
            a[i] = rng.NextDouble() < zeroFrac ? (sbyte)0 : (sbyte)(rng.Next(2) == 0 ? -1 : 1);
        return a;
    }

    private static float[] RandomX(Random rng, int n)
    {
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 4 - 2) * (i % 7 == 0 ? 40f : 1f);
        return x;
    }

    private static IEnumerable<TernaryKernelIsa> Isas()
    {
        yield return TernaryKernelIsa.Scalar;
        if (Vector256.IsHardwareAccelerated) yield return TernaryKernelIsa.Vector256;
        if (Vector512.IsHardwareAccelerated) yield return TernaryKernelIsa.Vector512;
    }

    private static float[] RunKernel(float[] x, int d, int rank, sbyte[] down, sbyte[] upT, float[] rankAlpha, bool relu, TernaryKernelIsa isa)
    {
        int wpr = TernaryAdapterKernel.WordsPerRow(d);
        ulong[] pd = NsAdapterFile.Pack(down, rank, d);
        ulong[] pu = NsAdapterFile.Pack(upT, rank, d);
        var row = (float[])x.Clone();
        var h = new float[rank + 1];
        fixed (float* r = row) fixed (ulong* wd = pd) fixed (ulong* wu = pu) fixed (float* ra = rankAlpha) fixed (float* hh = h)
            TernaryAdapterKernel.ApplyRow(r, d, rank, wd, wu, ra, relu, hh, isa);
        Assert.Equal(rank * wpr, pd.Length);
        return row;
    }

    private static void AssertBitEqual(float[] expected, float[] actual, string ctx)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                Assert.Fail($"{ctx}: element {i}: expected {expected[i]:R} got {actual[i]:R}");
    }

    // ------------------------------------------------------------------ (f1) kernel exactness ----

    [Fact]
    public void Kernel_BitExact_VsIndependentReference_AllIsas_ManyShapes()
    {
        var rng = new Random(1616);
        int[] ds = [1, 7, 15, 16, 17, 31, 33, 63, 64, 65, 100, 127, 128, 129, 200, 256, 300, 513];
        int[] ranks = [1, 2, 3, 8, 17, 64];
        float[] alphas = [1f, 0.37f, -2f];
        int cases = 0;
        foreach (int d in ds)
            foreach (int rank in ranks)
                foreach (float alpha in alphas)
                {
                    bool relu = (cases & 1) == 0;
                    var down = RandomTernary(rng, rank * d);
                    var up = RandomTernary(rng, rank * d);
                    var x = RandomX(rng, d);
                    var ra = Enumerable.Repeat(alpha, rank).ToArray();
                    var expected = Reference(x, d, rank, down, up, ra, relu);
                    foreach (var isa in Isas())
                        AssertBitEqual(expected, RunKernel(x, d, rank, down, up, ra, relu, isa), $"d={d} r={rank} a={alpha} relu={relu} isa={isa}");
                    cases++;
                }
        Assert.True(cases > 300);
    }

    [Fact]
    public void Kernel_PerRankAlpha_ZeroRows_AndIdentitySigma_BitExact()
    {
        var rng = new Random(77);
        int d = 150, rank = 12;
        var down = RandomTernary(rng, rank * d);
        var up = RandomTernary(rng, rank * d);
        Array.Clear(down, 3 * d, d);          // zero down row
        Array.Clear(up, 5 * d, 2 * d);        // zero up rows
        var ra = Enumerable.Range(0, rank).Select(i => (i - 4) * 0.25f).ToArray();
        var x = RandomX(rng, d);
        foreach (bool relu in new[] { true, false })
        {
            var expected = Reference(x, d, rank, down, up, ra, relu);
            foreach (var isa in Isas())
                AssertBitEqual(expected, RunKernel(x, d, rank, down, up, ra, relu, isa), $"relu={relu} isa={isa}");
        }
    }

    [Fact]
    public void Kernel_AllZeroWeights_LeavesRowBitIdentical()
    {
        int d = 70, rank = 4;
        var x = RandomX(new Random(3), d);
        var z = new sbyte[rank * d];
        foreach (var isa in Isas())
            AssertBitEqual(x, RunKernel(x, d, rank, z, z, Enumerable.Repeat(3f, rank).ToArray(), true, isa), $"isa={isa}");
    }

    [Fact]
    public void Kernel_CloseToDoubleOracle()
    {
        var rng = new Random(9);
        int d = 257, rank = 16;
        var down = RandomTernary(rng, rank * d);
        var up = RandomTernary(rng, rank * d);
        var x = RandomX(rng, d);
        var ra = Enumerable.Repeat(0.5f, rank).ToArray();
        var got = RunKernel(x, d, rank, down, up, ra, true, TernaryKernelIsa.Auto);
        for (int col = 0; col < d; col++)
        {
            double acc = 0;
            for (int j = 0; j < rank; j++)
            {
                double s = 0;
                for (int c = 0; c < d; c++) s += down[j * d + c] * (double)x[c];
                double hj = 0.5 * Math.Max(s, 0);
                acc += up[j * d + col] * hj;
            }
            Assert.InRange(Math.Abs(x[col] + acc - got[col]), 0, 1e-3 * (1 + Math.Abs(x[col] + acc)));
        }
    }

    [Fact]
    public void Pack_Layout_PlusMinusBitsPerColumn()
    {
        // Column 65 = +1 => block 1 plus bit 1; column 3 = -1 => block 0 minus bit 3.
        var v = new sbyte[130];
        v[65] = 1; v[3] = -1;
        ulong[] w = NsAdapterFile.Pack(v, 1, 130);
        Assert.Equal(6, w.Length);
        Assert.Equal(0UL, w[0]);
        Assert.Equal(1UL << 3, w[1]);
        Assert.Equal(1UL << 1, w[2]);
        Assert.Equal(0UL, w[3]);
    }

    // ------------------------------------------------------------------ (f2) format ----

    private static List<TernaryLayerSpec> RandomSpecs(Random rng, int d, params (int layer, int rank, float alpha)[] ls)
        => ls.Select(l => new TernaryLayerSpec(l.layer, l.rank, l.alpha, RandomTernary(rng, l.rank * d), RandomTernary(rng, l.rank * d))).ToList();

    [Fact]
    public void NsAdapter_RoundTrip_ProducesSameOutputAsInMemoryReference()
    {
        var rng = new Random(5);
        int d = 100;
        var specs = RandomSpecs(rng, d, (0, 4, 0.5f), (2, 9, -1.25f));
        string path = Path.Combine(_dir, "a.nsadapter");
        NsAdapterFile.Write(path, "rt", d, true, specs);
        using var ad = NsAdapterFile.Open(path);
        Assert.Equal("rt", ad.Name);
        Assert.Equal(d, ad.DModel);
        Assert.True(ad.Relu);
        Assert.Equal(2, ad.LayerCount);
        Assert.Equal(13, ad.TotalRank);
        Assert.True(ad.AppliesTo(0));
        Assert.False(ad.AppliesTo(1));
        Assert.True(ad.AppliesTo(2));
        Assert.False(ad.AppliesTo(3));
        foreach (var s in specs)
        {
            var x = RandomX(rng, d * 3);
            var rows = (float[])x.Clone();
            ad.ModifyResidual(rows, s.Layer, 3);
            for (int t = 0; t < 3; t++)
            {
                var exp = Reference(x[(t * d)..((t + 1) * d)], d, s.Rank, s.Down, s.UpT, Enumerable.Repeat(s.Alpha, s.Rank).ToArray(), true);
                AssertBitEqual(exp, rows[(t * d)..((t + 1) * d)], $"layer {s.Layer} token {t}");
            }
        }
        // Untouched layer leaves the buffer alone.
        var y = RandomX(rng, d);
        var y2 = (float[])y.Clone();
        ad.ModifyResidual(y2, 1, 1);
        AssertBitEqual(y, y2, "unadapted layer");
        ad.ValidateFor(d, 3);
        Assert.Throws<InvalidOperationException>(() => ad.ValidateFor(d + 1, 3));
        Assert.Throws<InvalidOperationException>(() => ad.ValidateFor(d, 2));
    }

    [Fact]
    public void NsAdapter_RejectsCorruption()
    {
        var rng = new Random(6);
        int d = 70;
        string path = Path.Combine(_dir, "c.nsadapter");
        NsAdapterFile.Write(path, "c", d, false, RandomSpecs(rng, d, (1, 3, 1f)));
        byte[] good = File.ReadAllBytes(path);

        string Mutate(string name, Action<byte[]> edit)
        {
            var b = (byte[])good.Clone();
            edit(b);
            string p = Path.Combine(_dir, name);
            File.WriteAllBytes(p, b);
            return p;
        }

        Assert.Throws<InvalidDataException>(() => NsAdapterFile.Open(Mutate("magic", b => b[0] = (byte)'X')));
        Assert.Throws<InvalidDataException>(() => NsAdapterFile.Open(Mutate("ver", b => b[4] = 9)));
        // Flip a payload byte: checksum catches it...
        ulong payOff = BitConverter.ToUInt64(good, 16);
        string flipped = Mutate("flip", b => b[(int)payOff + 1] ^= 0x01);
        Assert.Throws<InvalidDataException>(() => NsAdapterFile.Open(flipped));
        // ...and with the checksum off, structural validation still rejects an overlapping +1/-1 weight.
        string both = Mutate("both", b => { b[(int)payOff] = 0x01; b[(int)payOff + 8] = 0x01; });
        Assert.Throws<InvalidDataException>(() => NsAdapterFile.Open(both, verifyChecksum: false));
        // Mask bit beyond d_model (d=70: tail has 6 valid bits in block 1).
        string beyond = Mutate("beyond", b => b[(int)payOff + 16 + 1] = 0xFF);
        Assert.Throws<InvalidDataException>(() => NsAdapterFile.Open(beyond, verifyChecksum: false));
        Assert.Throws<InvalidDataException>(() => NsAdapterFile.Open(Mutate("trunc", b => b[8] = 0xFF)));
        using (NsAdapterFile.Open(path)) { }
    }

    // ------------------------------------------------------------------ (f3) composition ----

    [Fact]
    public void Composer_RankConcat_MatchesConcatenatedReference_BitExact_AndSumOfDeltas_Close()
    {
        var rng = new Random(8);
        int d = 129;
        var sa = RandomSpecs(rng, d, (0, 5, 0.5f), (1, 3, 2f));
        var sb = RandomSpecs(rng, d, (0, 7, -0.75f), (2, 4, 1f));
        string pa = Path.Combine(_dir, "ca.nsadapter"), pb = Path.Combine(_dir, "cb.nsadapter");
        NsAdapterFile.Write(pa, "A", d, true, sa);
        NsAdapterFile.Write(pb, "B", d, true, sb);
        using var a = NsAdapterFile.Open(pa);
        using var b = NsAdapterFile.Open(pb);
        using var comp = TernaryAdapterComposer.Compose([a, b], [1f, 0.5f]);
        Assert.Equal("stack[A+B]", comp.Name);
        Assert.Equal(5 + 7, comp.RankAt(0));
        Assert.Equal(3, comp.RankAt(1));
        Assert.Equal(4, comp.RankAt(2));

        // Many input rows: with ReLU a single row can leave any one rank row at exactly zero contribution, which let an
        // offset mutant (row += r - 1) survive; across 24 rows every rank row contributes somewhere.
        for (int rep = 0; rep < 24; rep++)
        {
        var x = RandomX(rng, d);
        // Layer 0: both adapters -> concatenated reference with per-row alpha (B scaled by weight 0.5).
        var cd = sa[0].Down.Concat(sb[0].Down).ToArray();
        var cu = sa[0].UpT.Concat(sb[0].UpT).ToArray();
        var cra = Enumerable.Repeat(0.5f, 5).Concat(Enumerable.Repeat(-0.75f * 0.5f, 7)).ToArray();
        var expected = Reference(x, d, 12, cd, cu, cra, true);
        var got = (float[])x.Clone();
        comp.ModifyResidual(got, 0, 1);
        AssertBitEqual(expected, got, "concat reference");

        // Sum of the individual deltas (parallel experts), within float tolerance: rounding order differs only in the up-sum.
        var ga = (float[])x.Clone(); a.ModifyResidual(ga, 0, 1);
        var gb = (float[])x.Clone(); b.ModifyResidual(gb, 0, 1);
        // B was weighted 0.5 inside the composite, so rebuild its delta with the same alpha scaling.
        var rbHalf = Reference(x, d, 7, sb[0].Down, sb[0].UpT, Enumerable.Repeat(-0.75f * 0.5f, 7).ToArray(), true);
        for (int i = 0; i < d; i++)
        {
            float sum = x[i] + (ga[i] - x[i]) + (rbHalf[i] - x[i]);
            Assert.InRange(Math.Abs(sum - got[i]), 0, 1e-3 * (1 + Math.Abs(sum)));
        }

        // Layers covered by only one adapter reduce to that adapter (B weight 0.5 on layer 2).
        var x1 = RandomX(rng, d);
        var g1 = (float[])x1.Clone(); comp.ModifyResidual(g1, 1, 1);
        var e1 = (float[])x1.Clone(); a.ModifyResidual(e1, 1, 1);
        AssertBitEqual(e1, g1, "layer 1 == A alone");
        var g2 = (float[])x1.Clone(); comp.ModifyResidual(g2, 2, 1);
        var e2 = Reference(x1, d, 4, sb[1].Down, sb[1].UpT, Enumerable.Repeat(0.5f, 4).ToArray(), true);
        AssertBitEqual(e2, g2, "layer 2 == B*0.5 alone");
        }

        Assert.Throws<NotSupportedException>(() => TernaryAdapterComposer.Compose([a, b], null, maxRank: 11));
    }

    // ------------------------------------------------------------------ (f4) engine wiring ----

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

    private static readonly int[] Ids = [3, 7, 11, 2, 40, 5];

    private TernaryResidualAdapter MakeAdapter(Tiny t, int layer, float alpha, int seed = 21)
    {
        var rng = new Random(seed);
        int d = t.Config.HiddenSize;
        var specs = RandomSpecs(rng, d, (layer, 8, alpha));
        string path = Path.Combine(_dir, $"m{layer}_{alpha}_{seed}.nsadapter");
        NsAdapterFile.Write(path, "e2e", d, true, specs);
        return NsAdapterFile.Open(path);
    }

    [Fact]
    public void Engine_NoAdapter_Unchanged_AdapterChangesOutput_DetachRestores()
    {
        using var tiny = new Tiny(_dir);
        Assert.Null(tiny.Model.ResidualModifier);
        float[] baseline = tiny.Logits(Ids);
        AssertBitEqual(baseline, tiny.Logits(Ids), "determinism");

        using var ad = MakeAdapter(tiny, 1, 3f);
        ad.ValidateFor(tiny.Config.HiddenSize, tiny.Config.NumLayers);
        tiny.Model.ResidualModifier = ad;
        float[] steered = tiny.Logits(Ids);
        double maxDiff = 0;
        for (int i = 0; i < steered.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(steered[i] - baseline[i]));
        Assert.True(maxDiff > 1e-3, $"adapter did not change logits (max diff {maxDiff})");

        tiny.Model.ResidualModifier = null;
        AssertBitEqual(baseline, tiny.Logits(Ids), "after detach");
    }

    [Fact]
    public void Engine_ZeroAlpha_IsBitIdenticalToBaseline()
    {
        using var tiny = new Tiny(_dir);
        float[] baseline = tiny.Logits(Ids);
        using var ad = MakeAdapter(tiny, 0, 0f);
        tiny.Model.ResidualModifier = ad;
        AssertBitEqual(baseline, tiny.Logits(Ids), "alpha=0");
    }

    private sealed class Recorder : IResidualModifier
    {
        public readonly List<(int layer, int tokens)> Calls = new();
        public readonly List<float[]> Snapshots = new();
        public bool AppliesTo(int layerIndex) => true;
        public void ModifyResidual(Span<float> residual, int layerIndex, int tokenCount)
        {
            Calls.Add((layerIndex, tokenCount));
            Snapshots.Add(residual.ToArray());
        }
    }

    [Fact]
    public void Engine_ModifierRunsAfterEachLayer_InOrder_WithAllTokenRows()
    {
        using var tiny = new Tiny(_dir);
        var rec = new Recorder();
        tiny.Model.ResidualModifier = rec;
        _ = tiny.Logits(Ids);
        Assert.Equal(Enumerable.Range(0, tiny.Config.NumLayers).Select(l => (l, Ids.Length)).ToArray(), rec.Calls.ToArray());
        // Snapshots are the layer outputs: consecutive layers must differ.
        Assert.NotEqual(rec.Snapshots[0], rec.Snapshots[1]);
        Assert.Equal(Ids.Length * tiny.Config.HiddenSize, rec.Snapshots[0].Length);
    }

    [Fact]
    public void Engine_AdapterAppliedAtItsLayer_NotAnother()
    {
        using var tiny = new Tiny(_dir);
        // Two adapters with identical weights but different target layer must produce different outputs...
        using var a0 = MakeAdapter(tiny, 0, 3f, seed: 33);
        using var a1 = MakeAdapter(tiny, 1, 3f, seed: 33);
        tiny.Model.ResidualModifier = a0;
        float[] l0 = tiny.Logits(Ids);
        tiny.Model.ResidualModifier = a1;
        float[] l1 = tiny.Logits(Ids);
        Assert.NotEqual(l0, l1);

        // ...and applying the adapter at layer 0 must equal a Recorder-style manual injection: run a modifier that
        // applies the adapter's layer-0 circuit to a layer-0 output only (checks the residual handed over is the
        // layer-0 output, and that the next layer consumes the modified stream).
        var manual = new OnlyAt(a0, 0);
        tiny.Model.ResidualModifier = manual;
        AssertBitEqual(l0, tiny.Logits(Ids), "manual layer-0 injection == adapter at layer 0");
        Assert.Equal(1, manual.Fired);
    }

    private sealed class OnlyAt(TernaryResidualAdapter inner, int layer) : IResidualModifier
    {
        public int Fired;
        public bool AppliesTo(int layerIndex) => layerIndex == layer;
        public void ModifyResidual(Span<float> residual, int layerIndex, int tokenCount)
        {
            Fired++;
            inner.ModifyResidual(residual, layerIndex, tokenCount);
        }
    }
}
