using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// Qwen4-Exp PLE n-gram branch (#816): exact int64 hash, row gather over quantised tables, gate, dilated conv with state,
/// EOS reset and chunk invariance, against the HF <c>Qwen4ExpTextPLELayer</c> fixtures from <c>Reference/gen_ple.py</c>.
/// </summary>
public sealed unsafe class Qwen4ExpPleTests
{
    private static readonly Qwen4ExpReferenceFixture Fx = Qwen4ExpReferenceFixture.Load("ple_branch.json");
    private static readonly Qwen4ExpReferenceFixture Real = Qwen4ExpReferenceFixture.Load("ple_hash_real.json");
    private readonly int _s = Fx.Int("hc_count"), _h = Fx.Int("hidden_size"), _t = Fx.Int("seq_len"), _eos = Fx.Int("eos");
    private readonly int _ngram = Fx.Int("ngram"), _hpn = Fx.Int("heads_per_ngram"), _rowDim = Fx.Int("row_dim"), _k = Fx.Int("conv_kernel");
    private readonly float _eps = (float)Fx.Meta.GetProperty("eps").GetDouble();

    private static void AssertClose(float[] expected, float[] actual, string what, float tol = 2e-5f)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0; int at = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float d = MathF.Abs(expected[i] - actual[i]) / (1f + MathF.Abs(expected[i]));
            if (d > worst) { worst = d; at = i; }
        }
        Assert.True(worst <= tol, $"{what}: worst rel diff {worst:E3} at {at} (expected {expected[at]}, got {actual[at]})");
    }

    private static int[] ToInts(long[] v) => v.Select(x => checked((int)x)).ToArray();

    private static long[] Rows(int[] ids, int[] history, long[] mult, long[] offs, long[] vocab, int ngram, int hpn, int eos)
    {
        var rows = new long[ids.Length * (ngram - 1) * hpn];
        Qwen4ExpPle.BuildRowIndices(ids, history, ngram, hpn, eos, mult, offs, vocab, rows);
        return rows;
    }

    private int[] FreshHistory() => Enumerable.Repeat(_eos, _ngram - 1).ToArray();

    // ── hash ────────────────────────────────────────────────────────────────

    [Fact]
    public void Hash_MatchesHf_WithEosResets()
    {
        var ids = ToInts(Fx.I64("ids"));
        Assert.Contains(_eos, ids);
        var got = Rows(ids, FreshHistory(), Fx.I64("multipliers"), Fx.I64("offsets"), Fx.I64("vocab_sizes"), _ngram, _hpn, _eos);
        Assert.Equal(Fx.I64("rows"), got);
    }

    [Fact]
    public void Hash_RealModelConstants_AreExactInt64()
    {
        int eos = Real.Int("eos");
        var ids = ToInts(Real.I64("ids"));
        var got = Rows(ids, Enumerable.Repeat(eos, 2).ToArray(), Real.I64("multipliers"), Real.I64("offsets"),
                       Real.I64("vocab_sizes"), 3, 8, eos);
        Assert.Equal(Real.I64("rows"), got);
        // The same rows computed through double would be wrong: prove the constants really need the int64 path.
        Assert.True(Real.I64("multipliers")[0] > (1L << 44));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(7)]
    public void Hash_Chunked_EqualsSingleShot(int chunk)
    {
        var ids = ToInts(Fx.I64("ids"));
        var mult = Fx.I64("multipliers"); var offs = Fx.I64("offsets"); var voc = Fx.I64("vocab_sizes");
        int nh = (_ngram - 1) * _hpn;
        var whole = Rows(ids, FreshHistory(), mult, offs, voc, _ngram, _hpn, _eos);
        var hist = FreshHistory();
        var parts = new List<long>();
        for (int i = 0; i < ids.Length; i += chunk)
        {
            var c = ids.AsSpan(i, Math.Min(chunk, ids.Length - i)).ToArray();
            parts.AddRange(Rows(c, hist, mult, offs, voc, _ngram, _hpn, _eos));
            Qwen4ExpPle.AdvanceHistory(hist, c);
        }
        Assert.Equal(whole, parts.ToArray());
        Assert.Equal(ids.Length * nh, parts.Count);
    }

    [Fact]
    public void Hash_EosResetsTheWindow_ButNotTheEosTokenItself()
    {
        var mult = Fx.I64("multipliers"); var offs = Fx.I64("offsets"); var voc = Fx.I64("vocab_sizes");
        int nh = (_ngram - 1) * _hpn;
        int a = 11, b = 12, c = 13;
        // a b EOS c : position 3 (c) must hash exactly like c at the start of a fresh sequence.
        var seq = Rows([a, b, _eos, c], FreshHistory(), mult, offs, voc, _ngram, _hpn, _eos);
        var fresh = Rows([c], FreshHistory(), mult, offs, voc, _ngram, _hpn, _eos);
        Assert.Equal(fresh, seq.AsSpan(3 * nh, nh).ToArray());
        // ... but the EOS token itself still sees its real predecessors (HF: the EOS does not cut its own context).
        var eosAlone = Rows([_eos], FreshHistory(), mult, offs, voc, _ngram, _hpn, _eos);
        Assert.NotEqual(eosAlone, seq.AsSpan(2 * nh, nh).ToArray());
        // EOS one back: c's 2-gram sees EOS (== fresh), while a token two back is also cut.
        var afterTwo = Rows([_eos, c, a], FreshHistory(), mult, offs, voc, _ngram, _hpn, _eos);
        var freshCA = Rows([c, a], FreshHistory(), mult, offs, voc, _ngram, _hpn, _eos);
        Assert.Equal(freshCA, afterTwo.AsSpan(nh, 2 * nh).ToArray());
    }

    [Fact]
    public void Hash_RowsStayInsideTheirHeadRange()
    {
        var offs = Fx.I64("offsets"); var voc = Fx.I64("vocab_sizes");
        var rows = Fx.I64("rows");
        int nh = (_ngram - 1) * _hpn;
        for (int i = 0; i < rows.Length; i++)
        {
            int h = i % nh;
            Assert.InRange(rows[i], offs[h], offs[h] + voc[h] - 1);
        }
        Assert.Equal(2L, Qwen4ExpPle.FloorMod(-7, 3));
    }

    // ── gather ──────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(QuantizationType.F32)]
    [InlineData(QuantizationType.BF16)]
    [InlineData(QuantizationType.F16)]
    [InlineData(QuantizationType.Q8_0)]
    [InlineData(QuantizationType.IQ4_NL)]
    public void GatherRows_EqualsFullTableDequant_ForEveryFormat(QuantizationType qt)
    {
        int rowDim = 160;   // the real row width: 5 blocks of 32
        int rows = 37;
        long rowBytes = Dequantize.RowByteSize(rowDim, qt);
        var raw = new byte[rows * rowBytes];
        var rng = new Random(1234);
        rng.NextBytes(raw);
        if (qt is QuantizationType.Q8_0 or QuantizationType.IQ4_NL or QuantizationType.F16)
        {
            // keep fp16 scales finite and modest: clear the exponent MSBs of every scale
            int block = qt == QuantizationType.Q8_0 ? 34 : qt == QuantizationType.IQ4_NL ? 18 : 2;
            for (int o = 0; o + 1 < raw.Length; o += block) raw[o + 1] = (byte)(raw[o + 1] & 0x3B);
        }
        if (qt == QuantizationType.BF16)
            for (int o = 1; o < raw.Length; o += 2) raw[o] = (byte)(raw[o] & 0x3F);   // finite exponents

        var h = GCHandle.Alloc(raw, GCHandleType.Pinned);
        try
        {
            nint table = h.AddrOfPinnedObject();
            var full = new float[rows * rowDim];
            Dequantize.ToFloat32(table, (long)rows * rowDim, qt, full);

            long[] pick = [36, 0, 5, 5, 17, 36];   // repeats and the last row
            var got = new float[pick.Length * rowDim];
            Qwen4ExpPle.GatherRows(table, qt, rows, rowDim, pick, got);
            for (int i = 0; i < pick.Length; i++)
                Assert.Equal(full.AsSpan((int)pick[i] * rowDim, rowDim).ToArray(), got.AsSpan(i * rowDim, rowDim).ToArray());

            Assert.Throws<ArgumentOutOfRangeException>(() => Qwen4ExpPle.GatherRows(table, qt, rows, rowDim, [rows], got));
            Assert.Throws<ArgumentOutOfRangeException>(() => Qwen4ExpPle.GatherRows(table, qt, rows, rowDim, [-1], got));
        }
        finally { h.Free(); }
    }

    // ── gate / conv ─────────────────────────────────────────────────────────

    [Fact]
    public void Gate_MatchesHf_AndScalar_AndZeroGivesHalf()
    {
        var res = Fx.F32("residual");
        var emb = Fx.F32("emb");
        var key = new float[_t * _s * _h];
        Gemv(Fx.F32("key_proj"), emb, key, _s * _h, emb.Length / _t, _t);
        Qwen4ExpGatedResidual.GroupRmsNorm(key, Fx.F32("norm_key"), _s, _h, _eps, key, _t);
        var query = new float[res.Length];
        Qwen4ExpGatedResidual.GroupRmsNorm(res, Fx.F32("norm_query"), _s, _h, _eps, query, _t);

        var gate = new float[_t * _s];
        Qwen4ExpPle.ComputeGate(key, query, _s, _h, gate, _t);
        AssertClose(Fx.F32("gate"), gate, "gate vs HF");
        var scalar = new float[_t * _s];
        Qwen4ExpPle.ComputeGateScalar(key, query, _s, _h, scalar, _t);
        AssertClose(scalar, gate, "gate vs scalar", 1e-5f);

        // Gates differ per stream (the per-stream dot, not a shared one) and sit away from 0.5 (non-degenerate).
        Assert.True(Math.Abs(gate[0] - gate[1]) > 1e-4f);

        // torch.sign(0) == 0 -> sigmoid(0) = 0.5 exactly.
        var z = new float[_h];
        var g0 = new float[1];
        Qwen4ExpPle.ComputeGate(z, z, 1, _h, g0, 1);
        Assert.Equal(0.5f, g0[0]);
    }

    [Fact]
    public void DilatedConv_MatchesHf_ScalarAndChunked()
    {
        int c = _s * _h, hist = (_k - 1) * _ngram;
        var x = Fx.F32("normed");
        var w = Qwen4ExpPle.TransposeConvWeight(Fx.F32("conv1d"), c, _k);

        var state = new float[hist * c];
        var y = new float[_t * c];
        Qwen4ExpPle.DilatedConvSilu(x, state, w, _k, _ngram, c, y, _t);
        AssertClose(Fx.F32("conv_out"), y, "conv vs HF");

        var scalar = new float[_t * c];
        Qwen4ExpPle.DilatedConvSiluScalar(x, new float[hist * c], w, _k, _ngram, c, scalar, _t);
        AssertClose(scalar, y, "conv vs scalar", 1e-5f);

        // The state now holds the last 9 input rows.
        Assert.Equal(x.AsSpan((_t - hist) * c, hist * c).ToArray(), state);

        // Chunked (including chunks shorter than the 9-row history) == single shot, bitwise.
        foreach (int chunk in new[] { 1, 2, 4, 10 })
        {
            var st = new float[hist * c];
            var yc = new float[_t * c];
            for (int i = 0; i < _t; i += chunk)
            {
                int n = Math.Min(chunk, _t - i);
                Qwen4ExpPle.DilatedConvSilu(x.AsSpan(i * c, n * c), st, w, _k, _ngram, c, yc.AsSpan(i * c, n * c), n);
            }
            Assert.Equal(y, yc);
        }

        // Taps genuinely dilated: a conv with dilation 1 must differ.
        var d1 = new float[_t * c];
        Qwen4ExpPle.DilatedConvSilu(x, new float[3 * c], w, _k, 1, c, d1, _t);
        Assert.NotEqual(y, d1);
    }

    // ── whole branch ────────────────────────────────────────────────────────

    private static void Gemv(float[] weight, float[] x, float[] y, int m, int k, int n)
    {
        fixed (float* w = weight) fixed (float* xp = x) fixed (float* yp = y)
            MatMul.GemmF32(w, xp, yp, m, k, n);
    }

    private static Qwen4ExpProjection F32Proj(float[] weight, int outDim, int inDim) =>
        (input, output, tokens) =>
        {
            fixed (float* w = weight) fixed (float* i = input) fixed (float* o = output)
                MatMul.GemmF32(w, i, o, outDim, inDim, tokens);
        };

    private (Qwen4ExpPleBranch Branch, GCHandle Pin) BuildBranch()
    {
        var table = Fx.F32("table");
        var pin = GCHandle.Alloc(table, GCHandleType.Pinned);
        int embDim = (_ngram - 1) * _hpn * _rowDim;
        var branch = new Qwen4ExpPleBranch(
            pin.AddrOfPinnedObject(), QuantizationType.F32, table.Length / _rowDim, _rowDim,
            _ngram, _hpn, _eos, _k, Fx.I64("multipliers"), Fx.I64("offsets"), Fx.I64("vocab_sizes"),
            _s, _h, _eps,
            Fx.F32("norm_key"), Fx.F32("norm_query"), Fx.F32("norm_conv"),
            Qwen4ExpPle.TransposeConvWeight(Fx.F32("conv1d"), _s * _h, _k),
            F32Proj(Fx.F32("key_proj"), _s * _h, embDim), F32Proj(Fx.F32("value_proj"), _h, embDim));
        return (branch, pin);
    }

    [Fact]
    public void Branch_SingleShot_MatchesHf()
    {
        var (branch, pin) = BuildBranch();
        try
        {
            var ids = ToInts(Fx.I64("ids"));
            var res = Fx.F32("residual");
            var expected = new float[res.Length];
            var hf = Fx.F32("output");
            for (int i = 0; i < res.Length; i++) expected[i] = res[i] + hf[i];

            var r = (float[])res.Clone();
            branch.Apply(ids, branch.CreateState(), r);
            AssertClose(expected, r, "residual + PLE vs HF");
        }
        finally { pin.Free(); }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(8)]
    public void Branch_Chunked_EqualsSingleShot(int chunk)
    {
        var (branch, pin) = BuildBranch();
        try
        {
            var ids = ToInts(Fx.I64("ids"));
            var res = Fx.F32("residual");
            int row = _s * _h;

            var whole = (float[])res.Clone();
            branch.Apply(ids, branch.CreateState(), whole);

            var chunked = (float[])res.Clone();
            var st = branch.CreateState();
            for (int i = 0; i < ids.Length; i += chunk)
            {
                int n = Math.Min(chunk, ids.Length - i);
                branch.Apply(ids.AsSpan(i, n), st, chunked.AsSpan(i * row, n * row));
            }
            AssertClose(whole, chunked, "chunked vs single-shot", 1e-6f);
        }
        finally { pin.Free(); }
    }

    [Fact]
    public void Branch_StateReset_RestartsTheSequence()
    {
        var (branch, pin) = BuildBranch();
        try
        {
            var ids = ToInts(Fx.I64("ids"));
            var res = Fx.F32("residual");
            var st = branch.CreateState();
            var first = (float[])res.Clone();
            branch.Apply(ids, st, first);
            st.Reset();
            var second = (float[])res.Clone();
            branch.Apply(ids, st, second);
            Assert.Equal(first, second);

            // Without a reset the carried history changes the result (state is actually used).
            var third = (float[])res.Clone();
            branch.Apply(ids, st, third);
            Assert.NotEqual(first, third);
        }
        finally { pin.Free(); }
    }
}
