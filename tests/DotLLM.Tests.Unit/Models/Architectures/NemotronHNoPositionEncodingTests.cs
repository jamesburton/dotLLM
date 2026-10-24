using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.Architectures;
using Architecture = DotLLM.Core.Configuration.Architecture;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #372 — Nemotron-H attention applies NO position encoding. Both
/// references rotate nothing (llama.cpp <c>src/models/nemotron-h.cpp</c> goes
/// straight from <c>build_qkv</c> to <c>build_attn</c>; HF transformers'
/// <c>NemotronHAttention.forward</c> never calls <c>apply_rotary_pos_emb</c>) —
/// position information comes entirely from the Mamba2 layers. dotLLM used to
/// rotate Q/K here, and cross-backend parity certified the shared error (the
/// #311 pattern), so this test compares against an <b>in-test plain-attention
/// reference</b>, not against another backend.
/// </summary>
public sealed class NemotronHNoPositionEncodingTests
{
    private const int Vocab = 16;
    private const int Hidden = 8;
    private const int HeadDim = 8;
    private const float Eps = 1e-5f;

    /// <summary>
    /// Four-token prefill through a single attention layer must match a plain
    /// (rotation-free) causal-attention reference computed in the test.
    /// DISCRIMINATING (proven by re-adding the rotation to the model): the old
    /// RoPE path rotates Q/K at positions >= 1 (~1-3 rad on the lowest pair at
    /// theta=10000), which moves the cross-position attention scores and shifts
    /// the logits ~5e-3 — 50x the 1e-4 tolerance. Position 0 is
    /// rotation-invariant (angle 0), so the assertion covers every position,
    /// not just the last. The reference uses the production Softmax.ExecuteFast
    /// so softmax numerics cancel and only a rotation can move the result.
    /// </summary>
    [Fact]
    public unsafe void AttentionPrefill_MatchesPlainAttentionReference_NoRotation()
    {
        var rng = new Random(372);
        float[] embed = RandArr(rng, Vocab * Hidden);
        float[] attnNorm = RandArr(rng, Hidden, offset: 1.0f, scale: 0.1f);
        // Larger Q/K weights widen the RoPE-vs-no-RoPE gap (the negative control's
        // margin): bigger scores make the softmax more sensitive to score rotation.
        float[] wq = RandArr(rng, HeadDim * Hidden, scale: 0.8f);
        float[] wk = RandArr(rng, HeadDim * Hidden, scale: 0.8f);
        float[] wv = RandArr(rng, HeadDim * Hidden);
        float[] wo = RandArr(rng, Hidden * HeadDim);
        float[] outNorm = RandArr(rng, Hidden, offset: 1.0f, scale: 0.1f);
        float[] wOut = RandArr(rng, Vocab * Hidden);

        int[] tokens = [3, 11, 7, 14];
        int[] positions = [0, 1, 2, 3];

        // ── in-test reference: RMSNorm → QKV → causal softmax attention (NO
        // rotation anywhere) → o_proj → residual → final RMSNorm → lm_head ──
        float[][] x = new float[tokens.Length][];
        float[][] q = new float[tokens.Length][];
        float[][] k = new float[tokens.Length][];
        float[][] v = new float[tokens.Length][];
        for (int t = 0; t < tokens.Length; t++)
        {
            x[t] = embed.AsSpan(tokens[t] * Hidden, Hidden).ToArray();
            float[] h = RmsNorm(x[t], attnNorm);
            q[t] = MatVec(wq, h, HeadDim, Hidden);
            k[t] = MatVec(wk, h, HeadDim, Hidden);
            v[t] = MatVec(wv, h, HeadDim, Hidden);
        }
        float[][] expected = new float[tokens.Length][];
        for (int t = 0; t < tokens.Length; t++)
        {
            // Causal softmax over positions 0..t — via the SAME fast-softmax the
            // production attention kernel uses (approximate exp), so the only
            // difference the assertion can see is a position rotation, not
            // softmax numerics.
            float[] scores = new float[t + 1];
            for (int j = 0; j <= t; j++)
            {
                double s = 0;
                for (int d = 0; d < HeadDim; d++) s += (double)q[t][d] * k[j][d];
                scores[j] = (float)(s / Math.Sqrt(HeadDim));
            }
            DotLLM.Cpu.Kernels.Softmax.ExecuteFast(scores, scores);
            float[] ctx = new float[HeadDim];
            for (int j = 0; j <= t; j++)
                for (int d = 0; d < HeadDim; d++)
                    ctx[d] += scores[j] * v[j][d];

            float[] o = MatVec(wo, ctx, Hidden, HeadDim);
            float[] resid = new float[Hidden];
            for (int d = 0; d < Hidden; d++) resid[d] = x[t][d] + o[d];
            expected[t] = MatVec(wOut, RmsNorm(resid, outNorm), Vocab, Hidden);
        }

        // ── the model ──
        var config = new ModelConfig
        {
            Architecture = Architecture.NemotronH,
            VocabSize = Vocab,
            HiddenSize = Hidden,
            IntermediateSize = Hidden,
            NumLayers = 1,
            NumAttentionHeads = 1,
            NumKvHeads = 1,
            HeadDim = HeadDim,
            MaxSequenceLength = 8,
            AttentionType = AttentionType.GQA,
            // Deliberately present and non-trivial: the model must IGNORE it.
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: HeadDim, Type: RoPEType.Norm),
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = Eps,
            TiedEmbeddings = false,
            HybridLayout = new HybridLayerLayout
            {
                LayerKind = [HybridLayerKind.Attention],
                HeadCountKv = [1],
                FeedForwardLength = [0],
            },
            SsmConfig = new MambaSsmConfig(DConv: 4, DInner: 8, DState: 4, NGroup: 1, NHead: 1),
            ChatTemplate = null,
        };

        nint embedPtr = Pin(embed);
        nint wqPtr = Pin(wq); nint wkPtr = Pin(wk); nint wvPtr = Pin(wv); nint woPtr = Pin(wo);
        nint wOutPtr = Pin(wOut);
        try
        {
            var layers = new NemotronHLayerWeights[]
            {
                new()
                {
                    AttnNormWeight = attnNorm,
                    Attention = new NemotronHAttentionWeights
                    {
                        QWeight = wqPtr, QQuantType = QuantizationType.F32, QInputDim = Hidden, QOutputDim = HeadDim,
                        KWeight = wkPtr, KQuantType = QuantizationType.F32, KInputDim = Hidden, KOutputDim = HeadDim,
                        VWeight = wvPtr, VQuantType = QuantizationType.F32, VInputDim = Hidden, VOutputDim = HeadDim,
                        OWeight = woPtr, OQuantType = QuantizationType.F32, OInputDim = HeadDim, OOutputDim = Hidden,
                        NumKvHeads = 1,
                    },
                },
            };

            using var model = NemotronHTransformerModel.BuildFromPrebuiltWeights(
                config, layers, outNorm,
                embedPtr, QuantizationType.F32,
                wOutPtr, QuantizationType.F32, Vocab, Hidden);

            using var logits = model.Forward(tokens, positions, deviceId: 0);
            float* lp = (float*)logits.DataPointer;

            for (int t = 0; t < tokens.Length; t++)
                for (int i = 0; i < Vocab; i++)
                {
                    float actual = lp[t * Vocab + i];
                    Assert.True(Math.Abs(expected[t][i] - actual) <= 1e-4f,
                        $"pos {t} logit {i}: expected {expected[t][i]:G9}, got {actual:G9} — " +
                        "Nemotron-H attention must apply NO position encoding (issue #372)");
                }
        }
        finally
        {
            foreach (var h in _pins) h.Free();
        }
    }

    private readonly List<GCHandle> _pins = new();
    private nint Pin(float[] a)
    {
        var h = GCHandle.Alloc(a, GCHandleType.Pinned);
        _pins.Add(h);
        return h.AddrOfPinnedObject();
    }

    private static float[] RandArr(Random rng, int n, float offset = 0f, float scale = 0.2f)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = offset + (float)(rng.NextDouble() * 2 - 1) * scale;
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

    /// <summary>Row-major [rows, cols] × vector.</summary>
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
