using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using Architecture = DotLLM.Core.Configuration.Architecture;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #375 slice 2 — the nemotron_h_moe routed-MoE FFN forward, checked against an
/// in-test reference transcribed from llama.cpp's <c>build_moe_ffn</c> as invoked by
/// <c>src/models/nemotron-h.cpp</c>'s <c>build_ffn_layer</c>: router on the FULL hidden
/// input → sigmoid → + per-expert bias for SELECTION ONLY → top-k → gating weights from
/// the UNBIASED probabilities → renorm by their sum → × scale (2.5) → weighted sum of
/// UNGATED relu² experts → + shared relu² expert. The fixture's selection bias is
/// crafted so biased and unbiased top-k CHOOSE DIFFERENT EXPERTS, and biased vs
/// unbiased WEIGHTS differ — so an implementation that conflates the two (either
/// direction) fails, as does one that skips renorm or the 2.5 scale.
/// </summary>
public sealed class NemotronHMoeForwardTests
{
    private const int Vocab = 16;
    private const int Hidden = 8;
    private const int NExpert = 4;
    private const int TopK = 2;
    private const int MoeInter = 8;
    private const int SharedInter = 6;
    private const float Eps = 1e-5f;
    private const float WScale = 2.5f;

    [Fact]
    public unsafe void MoeLayer_MatchesLlamaCppReferenceSemantics()
    {
        var rng = new Random(375);
        float[] embed = Rand(rng, Vocab * Hidden, 0.4f);
        float[] norm = RandPos(rng, Hidden);
        float[] router = Rand(rng, NExpert * Hidden, 0.6f);   // [nExpert, hidden] rows
        // Bias crafted to flip selection: strongly promote expert 0, demote expert 1.
        float[] selBias = [2.0f, -2.0f, 0.05f, -0.05f];
        float[] upBank = Rand(rng, NExpert * MoeInter * Hidden, 0.35f);
        float[] downBank = Rand(rng, NExpert * Hidden * MoeInter, 0.35f);
        float[] upShexp = Rand(rng, SharedInter * Hidden, 0.35f);
        float[] downShexp = Rand(rng, Hidden * SharedInter, 0.35f);
        float[] outNorm = RandPos(rng, Hidden);
        float[] wOut = Rand(rng, Vocab * Hidden, 0.4f);

        int[] tokens = [5, 9];
        int[] positions = [0, 1];

        // ── reference ──
        float[][] expected = new float[tokens.Length][];
        bool selectionDifferedFromUnbiased = false;
        for (int t = 0; t < tokens.Length; t++)
        {
            float[] x = embed.AsSpan(tokens[t] * Hidden, Hidden).ToArray();
            float[] h = RmsNorm(x, norm);

            // router → sigmoid
            float[] probs = new float[NExpert];
            for (int e = 0; e < NExpert; e++)
            {
                double s = 0;
                for (int i = 0; i < Hidden; i++) s += (double)router[e * Hidden + i] * h[i];
                probs[e] = 1f / (1f + MathF.Exp(-(float)s));
            }

            int[] top = TopKIndices(probs.Zip(selBias, (p, b) => p + b).ToArray(), TopK);
            int[] topUnbiased = TopKIndices(probs, TopK);
            if (!top.SequenceEqual(topUnbiased)) selectionDifferedFromUnbiased = true;

            // weights from UNBIASED probs, renorm, scale
            float[] w = top.Select(e => probs[e]).ToArray();
            float sum = w.Sum();
            for (int k = 0; k < TopK; k++) w[k] = w[k] / MathF.Max(sum, 6.103515625e-5f) * WScale;

            float[] acc = new float[Hidden];
            for (int k = 0; k < TopK; k++)
            {
                int e = top[k];
                float[] mid = new float[MoeInter];
                for (int m = 0; m < MoeInter; m++)
                {
                    double s = 0;
                    for (int i = 0; i < Hidden; i++)
                        s += (double)upBank[(e * MoeInter + m) * Hidden + i] * h[i];
                    float r = MathF.Max(0f, (float)s);
                    mid[m] = r * r;
                }
                for (int o = 0; o < Hidden; o++)
                {
                    double s = 0;
                    for (int m = 0; m < MoeInter; m++)
                        s += (double)downBank[(e * Hidden + o) * MoeInter + m] * mid[m];
                    acc[o] += w[k] * (float)s;
                }
            }

            // shared expert, unweighted
            float[] shMid = new float[SharedInter];
            for (int m = 0; m < SharedInter; m++)
            {
                double s = 0;
                for (int i = 0; i < Hidden; i++) s += (double)upShexp[m * Hidden + i] * h[i];
                float r = MathF.Max(0f, (float)s);
                shMid[m] = r * r;
            }
            for (int o = 0; o < Hidden; o++)
            {
                double s = 0;
                for (int m = 0; m < SharedInter; m++) s += (double)downShexp[o * SharedInter + m] * shMid[m];
                acc[o] += (float)s;
            }

            float[] resid = new float[Hidden];
            for (int i = 0; i < Hidden; i++) resid[i] = x[i] + acc[i];
            expected[t] = MatVec(wOut, RmsNorm(resid, outNorm), Vocab, Hidden);
        }
        // The fixture must actually exercise the bias-flips-selection case, or the
        // selection-vs-weights distinction is untested.
        Assert.True(selectionDifferedFromUnbiased,
            "fixture failed to produce a biased-selection flip — adjust selBias");

        // ── the model ──
        var config = new ModelConfig
        {
            Architecture = Architecture.NemotronHMoe,
            VocabSize = Vocab,
            HiddenSize = Hidden,
            IntermediateSize = MoeInter,
            NumLayers = 1,
            NumAttentionHeads = 1,
            NumKvHeads = 1,
            HeadDim = Hidden,
            MaxSequenceLength = 8,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.None,
            ActivationFunction = ActivationFunction.ReluSquared,
            NormType = NormType.RMSNorm,
            NormEpsilon = Eps,
            TiedEmbeddings = false,
            HybridLayout = new HybridLayerLayout
            {
                LayerKind = [HybridLayerKind.Ffn],
                HeadCountKv = [0],
                FeedForwardLength = [MoeInter],
            },
            SsmConfig = new MambaSsmConfig(DConv: 4, DInner: 8, DState: 4, NGroup: 1, NHead: 1),
            ChatTemplate = null,
        };

        var pins = new List<GCHandle>();
        nint Pin(float[] a)
        {
            var hnd = GCHandle.Alloc(a, GCHandleType.Pinned);
            pins.Add(hnd);
            return hnd.AddrOfPinnedObject();
        }
        try
        {
            var layers = new NemotronHLayerWeights[]
            {
                new()
                {
                    AttnNormWeight = norm,
                    Moe = new NemotronHMoeWeights
                    {
                        GateInpWeight = Pin(router), GateInpQuantType = QuantizationType.F32,
                        SelectionBias = selBias,
                        UpExpsWeight = Pin(upBank), UpExpsQuantType = QuantizationType.F32,
                        UpPerExpertBytes = (long)MoeInter * Hidden * sizeof(float),
                        DownExpsWeight = Pin(downBank), DownExpsQuantType = QuantizationType.F32,
                        DownPerExpertBytes = (long)Hidden * MoeInter * sizeof(float),
                        UpShexpWeight = Pin(upShexp), UpShexpQuantType = QuantizationType.F32,
                        DownShexpWeight = Pin(downShexp), DownShexpQuantType = QuantizationType.F32,
                        NumExperts = NExpert,
                        NumExpertsPerTok = TopK,
                        MoeIntermediateSize = MoeInter,
                        SharedIntermediateSize = SharedInter,
                        NormalizeWeights = true,
                        WeightsScale = WScale,
                    },
                },
            };

            using var model = NemotronHTransformerModel.BuildFromPrebuiltWeights(
                config, layers, outNorm,
                Pin(embed), QuantizationType.F32,
                Pin(wOut), QuantizationType.F32, Vocab, Hidden);

            using var logits = model.Forward(tokens, positions, deviceId: 0);
            float* lp = (float*)logits.DataPointer;

            for (int t = 0; t < tokens.Length; t++)
                for (int i = 0; i < Vocab; i++)
                {
                    float actual = lp[t * Vocab + i];
                    Assert.True(Math.Abs(expected[t][i] - actual) <= 2e-4f,
                        $"pos {t} logit {i}: expected {expected[t][i]:G9}, got {actual:G9} — " +
                        "nemotron_h_moe MoE semantics must match llama.cpp build_moe_ffn (issue #375)");
                }
        }
        finally
        {
            foreach (var hnd in pins) hnd.Free();
        }
    }

    private static int[] TopKIndices(float[] scores, int k)
    {
        var s = (float[])scores.Clone();
        int[] idx = new int[k];
        for (int j = 0; j < k; j++)
        {
            int best = 0;
            for (int e = 1; e < s.Length; e++) if (s[e] > s[best]) best = e;
            idx[j] = best;
            s[best] = float.NegativeInfinity;
        }
        return idx;
    }

    private static float[] Rand(Random rng, int n, float scale)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return a;
    }

    private static float[] RandPos(Random rng, int n)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = 1.0f + (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        return a;
    }

    private static float[] RmsNorm(float[] x, float[] w)
    {
        double ss = 0;
        for (int i = 0; i < x.Length; i++) ss += (double)x[i] * x[i];
        float inv = (float)(1.0 / Math.Sqrt(ss / x.Length + Eps));
        float[] y = new float[x.Length];
        for (int i = 0; i < x.Length; i++) y[i] = x[i] * inv * w[i];
        return y;
    }

    private static float[] MatVec(float[] m, float[] x, int rows, int cols)
    {
        float[] y = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            double acc = 0;
            for (int c = 0; c < cols; c++) acc += (double)m[r * cols + c] * x[c];
            y[r] = (float)acc;
        }
        return y;
    }
}
