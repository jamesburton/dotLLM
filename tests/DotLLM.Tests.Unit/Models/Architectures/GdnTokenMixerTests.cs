using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// #843: the shared Gated-DeltaNet token mixer. Qwen3.5 (SiLU output gate) and Qwen4-Exp (sigmoid output gate) differ only in that
/// activation; these tests drive the mixer directly and discriminate the two gates.
/// </summary>
public sealed unsafe class GdnTokenMixerTests : IDisposable
{
    private const int Hidden = 16, NV = 4, NK = 2, DS = 8, DC = 4, T = 5;
    private static readonly int ConvDim = (2 * NK + NV) * DS, VDim = NV * DS, KDim = NK * DS;
    private static readonly GatedDeltaNetConfig Gdn = new(FullAttnInterval: 4, NVHead: NV, NKHead: NK, DState: DS, DInner: VDim, DConv: DC);

    private readonly List<nint> _allocs = [];

    public void Dispose() { foreach (var p in _allocs) NativeMemory.Free((void*)p); }

    private nint Matrix(Random rng, int rows, int cols, float scale)
    {
        var p = (float*)NativeMemory.Alloc((nuint)(rows * cols * sizeof(float)));
        for (int i = 0; i < rows * cols; i++) p[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        _allocs.Add((nint)p);
        return (nint)p;
    }

    private GdnTokenMixingWeights Weights(int seed)
    {
        var rng = new Random(seed);
        float[] V(int n, float lo, float hi) => Enumerable.Range(0, n).Select(_ => lo + (float)rng.NextDouble() * (hi - lo)).ToArray();
        return new GdnTokenMixingWeights
        {
            QkvWeight = Matrix(rng, ConvDim, Hidden, 0.4f), QkvQuantType = QuantizationType.F32, QkvInputDim = Hidden, QkvOutputDim = ConvDim,
            GateWeight = Matrix(rng, VDim, Hidden, 1.5f), GateQuantType = QuantizationType.F32, GateInputDim = Hidden, GateOutputDim = VDim,
            AlphaWeight = Matrix(rng, NV, Hidden, 0.4f), AlphaQuantType = QuantizationType.F32, AlphaInputDim = Hidden, AlphaOutputDim = NV,
            BetaWeight = Matrix(rng, NV, Hidden, 0.4f), BetaQuantType = QuantizationType.F32, BetaInputDim = Hidden, BetaOutputDim = NV,
            OutWeight = Matrix(rng, Hidden, VDim, 0.4f), OutQuantType = QuantizationType.F32, OutInputDim = VDim, OutOutputDim = Hidden,
            A = V(NV, -1.0f, -0.1f), DtBias = V(NV, -0.2f, 0.2f), SsmNormWeight = V(DS, 0.5f, 1.5f),
            Conv1dWeight = V(DC * ConvDim, -0.5f, 0.5f), Conv1dBias = new float[ConvDim],
        };
    }

    private sealed class Capture
    {
        public float[]? Z, CoreIntoOut;
    }

    /// <summary>Naive F32 GEMM that also records the gate projection output (z) and the input of ssm_out (the gated, normed core).</summary>
    private static GdnGemm Gemm(GdnTokenMixingWeights w, Capture cap) => (proj, weight, qt, input, output, outDim, inDim, n) =>
    {
        Assert.Equal(QuantizationType.F32, qt);
        var wp = (float*)weight;
        for (int t = 0; t < n; t++)
            for (int o = 0; o < outDim; o++)
            {
                float acc = 0;
                for (int i = 0; i < inDim; i++) acc += input[t * inDim + i] * wp[o * inDim + i];
                output[t * outDim + o] = acc;
            }
        if (weight == w.GateWeight) cap.Z = output.Slice(0, n * outDim).ToArray();
        if (weight == w.OutWeight) cap.CoreIntoOut = input.Slice(0, n * inDim).ToArray();
    };

    private (float[] Y, Capture Cap) Run(GdnTokenMixingWeights w, GdnOutputGate gate, float[] x)
    {
        var cap = new Capture();
        using var cache = new GdnStateCache(Gdn, 1);
        var y = new float[T * Hidden];
        var scratch = new GdnMixerScratch(new float[T * ConvDim], new float[T * VDim], new float[T * NV], new float[T * NV],
                                          new float[(DC - 1 + T) * ConvDim], new float[T * KDim], new float[T * KDim], new float[T * VDim], new float[T * VDim]);
        GdnTokenMixer.Forward(w, Gdn, absoluteLayer: 0, ordinal: 0, T, Hidden, 1e-6f, gate, x, y, Gemm(w, cap), scratch, cache);
        return (y, cap);
    }

    [Fact]
    public void SigmoidAndSiLuGates_DifferExactlyByTheActivation()
    {
        var w = Weights(7);
        var rng = new Random(11);
        var x = Enumerable.Range(0, T * Hidden).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();

        var (ySilu, silu) = Run(w, GdnOutputGate.SiLu, x);
        var (ySig, sig) = Run(w, GdnOutputGate.Sigmoid, x);

        Assert.NotEqual(ySilu, ySig);                                   // the gate matters end to end
        Assert.Equal(silu.Z, sig.Z);                                    // identical z: the only difference is the activation
        // Same normed core in both runs, so core_sigmoid / core_silu == sigmoid(z) / (z * sigmoid(z)) == 1 / z elementwise.
        int checkedElems = 0;
        for (int i = 0; i < silu.Z!.Length; i++)
        {
            float z = silu.Z[i];
            if (MathF.Abs(z) < 0.25f || MathF.Abs(silu.CoreIntoOut![i]) < 1e-6f) continue;
            float ratio = sig.CoreIntoOut![i] / silu.CoreIntoOut[i];
            Assert.True(MathF.Abs(ratio - 1f / z) <= 2e-4f * MathF.Abs(1f / z), $"elem {i}: z={z} ratio={ratio} expected {1f / z}");
            checkedElems++;
        }
        Assert.True(checkedElems > T * VDim / 2, $"only {checkedElems} elements were informative");
        // Control: running the SAME gate twice would give ratio 1 everywhere, which these tolerances reject for most elements.
        int farFromOne = 0;
        for (int i = 0; i < silu.Z.Length; i++)
            if (MathF.Abs(silu.Z[i]) >= 0.25f && MathF.Abs(1f / silu.Z[i] - 1f) > 0.1f) farFromOne++;
        Assert.True(farFromOne > checkedElems / 2, "the check could not tell the two gates apart");
    }

    [Fact]
    public void Sigmoid_And_SiLu_AreEachDeterministic_AndAdvanceTheStateIdentically()
    {
        var w = Weights(9);
        var rng = new Random(13);
        var x = Enumerable.Range(0, T * Hidden).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        // The recurrent state does not depend on the output gate: both gates leave the same GDN state behind.
        float[] StateAfter(GdnOutputGate g)
        {
            var cap = new Capture();
            using var cache = new GdnStateCache(Gdn, 1);
            var scratch = new GdnMixerScratch(new float[T * ConvDim], new float[T * VDim], new float[T * NV], new float[T * NV],
                                              new float[(DC - 1 + T) * ConvDim], new float[T * KDim], new float[T * KDim], new float[T * VDim], new float[T * VDim]);
            GdnTokenMixer.Forward(w, Gdn, 0, 0, T, Hidden, 1e-6f, g, x, new float[T * Hidden], Gemm(w, cap), scratch, cache);
            return cache.GetGdnState(0).ToArray().Concat(cache.GetConvState(0).ToArray()).ToArray();
        }
        Assert.Equal(StateAfter(GdnOutputGate.SiLu), StateAfter(GdnOutputGate.Sigmoid));
        Assert.Equal(StateAfter(GdnOutputGate.Sigmoid), StateAfter(GdnOutputGate.Sigmoid));
        Assert.Contains(StateAfter(GdnOutputGate.SiLu), f => f != 0f);   // control: the state really moved
    }
}
