using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// NVFP4 second-level scales in the grouped MoE kernel (#1578, #1579): routed banks carry a per-EXPERT output scale
/// (<c>ffn_{gate,up,down}_exps.scale</c> [E]) and shared projections a per-tensor scalar, applied as
/// <c>res = mul_mat(w, x) * scale</c> (llama.cpp <c>build_lora_mm_id</c> / <c>build_lora_mm</c>).
/// </summary>
/// <remarks>
/// The oracle is written independently of the kernel: dequantised banks (<see cref="Dequantize.ToFloat32"/>) dotted against the
/// UNQUANTISED f32 activation, the scale multiplied in where llama.cpp multiplies it (after each matmul, before SwiGLU for gate/up),
/// SiLU in double. The kernel quantises activations to Q8_0 so the two agree to ~1%, while every scale mutant is off by
/// whole multiples (the scales span 0.4..5 and differ per expert, so "expert 0's scale for all", "no scale" and "scale of
/// the wrong projection" each move the output by tens of percent - asserted below, which is the substitution proof that the
/// fixture can tell right from wrong).
/// </remarks>
public sealed unsafe class Nvfp4MoeScaleTests
{
    private const int NumExperts = 10;
    private const int TopK = 4;
    private const int Hidden = 256;
    private const int Inter = 128;

    private static readonly float[] GateScale = [0.5f, 1.7f, 0.4f, 3.1f, 2.2f, 0.9f, 4.6f, 1.2f, 0.7f, 2.9f];
    private static readonly float[] UpScale = [2.0f, 0.6f, 1.3f, 0.45f, 3.3f, 1.1f, 0.8f, 4.0f, 1.9f, 0.5f];
    private static readonly float[] DownScale = [1.4f, 0.5f, 2.6f, 1.0f, 0.43f, 3.7f, 1.5f, 0.65f, 5.0f, 2.1f];

    [Theory]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(5, false)]
    [InlineData(5, true)]
    public void RoutedExpertScales_MatchIndependentOracle(int seqLen, bool withPool)
    {
        var f = new Fixture(seqLen, 777 + seqLen);
        try
        {
            using var pool = withPool ? new ComputeThreadPool(4) : null;
            pool?.SetDispatchMode(DispatchMode.SpinWait);
            float[] actual = f.Run(pool, GateScale, UpScale, DownScale, 1f, 1f, 1f);
            float[] oracle = f.Oracle(GateScale, UpScale, DownScale, 1f, 1f, 1f);
            Assert.True(RelErr(actual, oracle) < 0.03, $"rel err {RelErr(actual, oracle)}");

            // Substitution proof: each plausible wrong implementation lands far from the oracle, i.e. this fixture discriminates.
            float[] noScale = f.Oracle(null, null, null, 1f, 1f, 1f);
            float[] expert0ForAll = f.Oracle(Fill(GateScale[0]), Fill(UpScale[0]), Fill(DownScale[0]), 1f, 1f, 1f);
            float[] downOnly = f.Oracle(null, null, DownScale, 1f, 1f, 1f);
            float[] swapped = f.Oracle(UpScale, GateScale, DownScale, 1f, 1f, 1f);
            float[] downMissing = f.Oracle(GateScale, UpScale, null, 1f, 1f, 1f);
            foreach (var (name, wrong) in new[] { ("no scale", noScale), ("expert 0 for all", expert0ForAll), ("down only", downOnly),
                                                  ("gate/up swapped", swapped), ("down missing", downMissing) })
                Assert.True(RelErr(wrong, oracle) > 0.08, $"fixture cannot tell '{name}' from correct (rel diff {RelErr(wrong, oracle)})");
            // ... and the kernel result is nowhere near any of them.
            foreach (var wrong in new[] { noScale, expert0ForAll, downOnly, swapped, downMissing })
                Assert.True(RelErr(actual, wrong) > 0.08);
        }
        finally { f.Dispose(); }
    }

    [Fact]
    public void SharedExpertScales_AreAppliedPerProjection()
    {
        var f = new Fixture(3, 4242, withShared: true);
        try
        {
            float[] actual = f.Run(null, GateScale, UpScale, DownScale, 0.37f, 2.9f, 1.6f);
            float[] oracle = f.Oracle(GateScale, UpScale, DownScale, 0.37f, 2.9f, 1.6f);
            Assert.True(RelErr(actual, oracle) < 0.03, $"rel err {RelErr(actual, oracle)}");
            foreach (var wrong in new[]
            {
                f.Oracle(GateScale, UpScale, DownScale, 1f, 1f, 1f),
                f.Oracle(GateScale, UpScale, DownScale, 0.37f, 2.9f, 1f),
                f.Oracle(GateScale, UpScale, DownScale, 2.9f, 0.37f, 1.6f),
                f.Oracle(GateScale, UpScale, DownScale, 0.37f, 2.9f, 1.6f * 1.6f),
            })
                Assert.True(RelErr(actual, wrong) > 0.05, "shared scale mutant not distinguishable");
        }
        finally { f.Dispose(); }
    }

    [Fact]
    public void DecodeFastPath_WithScales_IsBitIdenticalToGroupedPathAndRuns()
    {
        var f = new Fixture(1, 99, withShared: true);
        try
        {
            using var pool = new ComputeThreadPool(5);
            pool.SetDispatchMode(DispatchMode.SpinWait);
            float[] grouped = f.Run(null, GateScale, UpScale, DownScale, 0.37f, 2.9f, 1.6f);
            long before = MoeSwiGluMlp.DecodeN1Calls;
            long fusedBefore = MoeSwiGluMlp.DecodeN1SharedFusedCalls;
            float[] fast = f.Run(pool, GateScale, UpScale, DownScale, 0.37f, 2.9f, 1.6f);
            Assert.True(MoeSwiGluMlp.DecodeN1Calls > before, "NVFP4 n=1 decode fast path did not run");
            Assert.True(MoeSwiGluMlp.DecodeN1SharedFusedCalls > fusedBefore, "fused shared expert did not run");
            for (int i = 0; i < grouped.Length; i++)
                Assert.Equal(BitConverter.SingleToInt32Bits(grouped[i]), BitConverter.SingleToInt32Bits(fast[i]));
        }
        finally { f.Dispose(); }
    }

    [Fact]
    public void PrefillBucketing_IsBatchInvariantForScaledExperts()
    {
        // A token's result must not depend on which other tokens share its expert bucket.
        var f = new Fixture(6, 31337);
        try
        {
            float[] all = f.Run(null, GateScale, UpScale, DownScale, 1f, 1f, 1f);
            for (int t = 0; t < 6; t++)
            {
                var single = f.Slice(t);
                try
                {
                    float[] one = single.Run(null, GateScale, UpScale, DownScale, 1f, 1f, 1f);
                    for (int i = 0; i < Hidden; i++)
                        Assert.True(MathF.Abs(one[i] - all[t * Hidden + i]) <= 1e-4f * (1f + MathF.Abs(one[i])));
                }
                finally { single.DisposeActivationsOnly(); }
            }
        }
        finally { f.Dispose(); }
    }

    // ──────────────────── loader gate ────────────────────

    [Fact]
    public void ScalesAppliedByCpuTransformer_CoversProjectionsButNotEmbeddingsOrForeignMoe()
    {
        var t = new Dictionary<string, GgufTensorDescriptor>(StringComparer.Ordinal);
        void W(string name, QuantizationType qt = QuantizationType.NVFP4, bool scale = true)
        {
            t[name] = new GgufTensorDescriptor(name, new TensorShape(64, 4), qt, 0);
            if (scale) t[name[..^".weight".Length] + ".scale"] = new GgufTensorDescriptor(name[..^".weight".Length] + ".scale", new TensorShape(1), QuantizationType.F32, 0);
        }
        W("blk.0.attn_q.weight"); W("blk.0.attn_output.weight"); W("blk.1.ffn_down.weight"); W("output.weight");
        W("blk.2.ffn_gate_exps.weight"); W("blk.2.ffn_up_shexp.weight");
        W("token_embd.weight");                         // NVFP4 embedding with a scale: nobody applies it
        W("blk.3.attn_k.weight", QuantizationType.Q4_1); // not NVFP4: its sibling scale is not an NVFP4 scale

        var grouped = Nvfp4Support.ScalesAppliedByCpuTransformer(t, groupedMoe: true);
        Assert.Equal(new[] { "blk.0.attn_output.scale", "blk.0.attn_q.scale", "blk.1.ffn_down.scale", "blk.2.ffn_gate_exps.scale",
                             "blk.2.ffn_up_shexp.scale", "output.scale" }, grouped.OrderBy(x => x, StringComparer.Ordinal).ToArray());
        var plain = Nvfp4Support.ScalesAppliedByCpuTransformer(t, groupedMoe: false);
        Assert.DoesNotContain("blk.2.ffn_gate_exps.scale", plain);
        Assert.DoesNotContain("blk.2.ffn_up_shexp.scale", plain);

        // The embedding scale stays unapplied, so the loader still refuses the file.
        var ex = Assert.Throws<NotSupportedException>(() => Nvfp4Support.RejectUnappliedScales(t, grouped, "test"));
        Assert.Contains("token_embd.scale", ex.Message);
        // A backend that applies none refuses everything.
        Assert.Throws<NotSupportedException>(() => Nvfp4Support.RejectUnappliedScales(t, null, "test"));
    }

    // ──────────────────── helpers ────────────────────

    private static float[] Fill(float v) => Enumerable.Repeat(v, NumExperts).ToArray();

    private static double RelErr(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    private sealed class Fixture : IDisposable
    {
        private readonly int _seqLen;
        private readonly bool _withShared;
        private readonly float[] _x, _router;
        private readonly byte* _gate, _up, _down, _sGate, _sUp, _sDown;
        private readonly long _gateExp, _downExp;
        private readonly bool _ownsBanks;
        private readonly Fixture? _parent;

        public Fixture(int seqLen, int seed, bool withShared = false)
        {
            _seqLen = seqLen; _withShared = withShared; _ownsBanks = true;
            var rng = new Random(seed);
            _gateExp = Dequantize.RowByteSize((long)Inter * Hidden, QuantizationType.NVFP4);
            _downExp = Dequantize.RowByteSize((long)Hidden * Inter, QuantizationType.NVFP4);
            _gate = Bank(rng, _gateExp * NumExperts); _up = Bank(rng, _gateExp * NumExperts); _down = Bank(rng, _downExp * NumExperts);
            if (withShared) { _sGate = Bank(rng, _gateExp); _sUp = Bank(rng, _gateExp); _sDown = Bank(rng, _downExp); }
            _x = new float[seqLen * Hidden];
            for (int i = 0; i < _x.Length; i++) _x[i] = rng.NextSingle() * 2f - 1f;
            _router = new float[NumExperts * Hidden];
            for (int i = 0; i < _router.Length; i++) _router[i] = rng.NextSingle() * 2f - 1f;
        }

        private Fixture(Fixture parent, int token)
        {
            _parent = parent; _seqLen = 1; _withShared = parent._withShared; _ownsBanks = false;
            _gateExp = parent._gateExp; _downExp = parent._downExp;
            _gate = parent._gate; _up = parent._up; _down = parent._down; _sGate = parent._sGate; _sUp = parent._sUp; _sDown = parent._sDown;
            _x = parent._x.AsSpan(token * Hidden, Hidden).ToArray(); _router = parent._router;
        }

        public Fixture Slice(int token) => new(this, token);
        public void DisposeActivationsOnly() { }

        public void Dispose()
        {
            if (!_ownsBanks) return;
            NativeMemory.AlignedFree(_gate); NativeMemory.AlignedFree(_up); NativeMemory.AlignedFree(_down);
            if (_withShared) { NativeMemory.AlignedFree(_sGate); NativeMemory.AlignedFree(_sUp); NativeMemory.AlignedFree(_sDown); }
        }

        /// <summary>Random codes; the four sub-block scales per block are valid UE4M3 bytes in a small positive range.</summary>
        private static byte* Bank(Random rng, long bytes)
        {
            byte* p = (byte*)NativeMemory.AlignedAlloc((nuint)bytes, 64);
            for (long i = 0; i < bytes; i++) p[i] = (byte)rng.Next(256);
            for (long off = 0; off < bytes; off += QuantFormat.Nvfp4BlockBytes)
                for (int s = 0; s < 4; s++) p[off + s] = (byte)rng.Next(0x18, 0x20);   // small sub-block scales so gate pre-activations are O(1): SiLU stays non-linear and gate/up scales are distinguishable
            return p;
        }

        public float[] Run(ComputeThreadPool? pool, float[]? g, float[]? u, float[]? d, float sg, float su, float sd)
        {
            int[] assignExpert = new int[TopK * _seqLen];
            float[] assignWeight = new float[TopK * _seqLen];
            int[] cursors = new int[NumExperts + 1];
            int[] tokens = new int[TopK * _seqLen];
            int[] slots = new int[TopK * _seqLen];
            int[] unique = new int[NumExperts];
            int uniqueCount = MoeSwiGluMlp.Route(_x, _router, assignExpert, assignWeight, cursors, tokens, slots, unique,
                NumExperts, TopK, Hidden, _seqLen, normTopKProb: true);
            float[] output = new float[Hidden * _seqLen];
            nint[] sgp = _withShared ? [(nint)_sGate] : [], sup = _withShared ? [(nint)_sUp] : [], sdp = _withShared ? [(nint)_sDown] : [];
            MoeSwiGluMlp.ExecuteRoutedFromAssignments(
                _x,
                (nint)_gate, QuantizationType.NVFP4, _gateExp, ReadOnlySpan<nint>.Empty,
                (nint)_up, QuantizationType.NVFP4, _gateExp, ReadOnlySpan<nint>.Empty,
                (nint)_down, QuantizationType.NVFP4, _downExp, ReadOnlySpan<nint>.Empty,
                assignExpert, assignWeight, cursors, tokens, slots, unique, uniqueCount,
                output,
                NumExperts, TopK, Hidden, Inter, _seqLen,
                sharedGateProj: _withShared ? [0] : [], sharedUpProj: _withShared ? [0] : [], sharedDownProj: _withShared ? [0] : [],
                sharedIntermediateSize: _withShared ? Inter : 0, sharedExpertGate: ReadOnlySpan<float>.Empty,
                loraAdapter: null, loraLayer: 0, threadPool: pool,
                sharedGateRaw: sgp, sharedGateRawQt: QuantizationType.NVFP4,
                sharedUpRaw: sup, sharedUpRawQt: QuantizationType.NVFP4,
                sharedDownRaw: sdp, sharedDownRawQt: QuantizationType.NVFP4,
                gateExpsScale: g, upExpsScale: u, downExpsScale: d,
                sharedGateScale: sg, sharedUpScale: su, sharedDownScale: sd);
            return output;
        }

        private float[] Dq(byte* p, int elements)
        {
            float[] r = new float[elements];
            Dequantize.ToFloat32((nint)p, elements, QuantizationType.NVFP4, r);
            return r;
        }

        private static double Dot(float[] w, int rowOffset, int k, float[] x, int xOffset)
        {
            double s = 0;
            for (int i = 0; i < k; i++) s += (double)w[rowOffset + i] * x[xOffset + i];
            return s;
        }

        private static double Silu(double v) => v / (1 + Math.Exp(-v));

        /// <summary>Independent f32-activation oracle with the scales applied where llama.cpp applies them.</summary>
        public float[] Oracle(float[]? g, float[]? u, float[]? d, float sg, float su, float sd)
        {
            int[] assignExpert = new int[TopK * _seqLen];
            float[] assignWeight = new float[TopK * _seqLen];
            int[] cursors = new int[NumExperts + 1];
            int[] tokens = new int[TopK * _seqLen];
            int[] slots = new int[TopK * _seqLen];
            int[] unique = new int[NumExperts];
            MoeSwiGluMlp.Route(_x, _router, assignExpert, assignWeight, cursors, tokens, slots, unique,
                NumExperts, TopK, Hidden, _seqLen, normTopKProb: true);

            float[][] wg = new float[NumExperts][], wu = new float[NumExperts][], wd = new float[NumExperts][];
            for (int e = 0; e < NumExperts; e++)
            {
                wg[e] = Dq(_gate + e * _gateExp, Inter * Hidden);
                wu[e] = Dq(_up + e * _gateExp, Inter * Hidden);
                wd[e] = Dq(_down + e * _downExp, Hidden * Inter);
            }
            float[]? sG = null, sU = null, sD = null;
            if (_withShared) { sG = Dq(_sGate, Inter * Hidden); sU = Dq(_sUp, Inter * Hidden); sD = Dq(_sDown, Hidden * Inter); }

            float[] output = new float[Hidden * _seqLen];
            for (int t = 0; t < _seqLen; t++)
            {
                double[] acc = new double[Hidden];
                for (int slot = 0; slot < TopK; slot++)
                {
                    int e = assignExpert[t * TopK + slot];
                    double w = assignWeight[t * TopK + slot];
                    double[] act = new double[Inter];
                    for (int j = 0; j < Inter; j++)
                    {
                        double gv = Dot(wg[e], j * Hidden, Hidden, _x, t * Hidden) * (g?[e] ?? 1f);
                        double uv = Dot(wu[e], j * Hidden, Hidden, _x, t * Hidden) * (u?[e] ?? 1f);
                        act[j] = Silu(gv) * uv;
                    }
                    for (int r = 0; r < Hidden; r++)
                    {
                        double s = 0;
                        for (int j = 0; j < Inter; j++) s += wd[e][r * Inter + j] * act[j];
                        acc[r] += w * s * (d?[e] ?? 1f);
                    }
                }
                if (_withShared)
                {
                    double[] act = new double[Inter];
                    for (int j = 0; j < Inter; j++)
                    {
                        double gv = Dot(sG!, j * Hidden, Hidden, _x, t * Hidden) * sg;
                        double uv = Dot(sU!, j * Hidden, Hidden, _x, t * Hidden) * su;
                        act[j] = Silu(gv) * uv;
                    }
                    for (int r = 0; r < Hidden; r++)
                    {
                        double s = 0;
                        for (int j = 0; j < Inter; j++) s += sD![r * Inter + j] * act[j];
                        acc[r] += s * sd;
                    }
                }
                for (int r = 0; r < Hidden; r++) output[t * Hidden + r] = (float)acc[r];
            }
            return output;
        }
    }
}
