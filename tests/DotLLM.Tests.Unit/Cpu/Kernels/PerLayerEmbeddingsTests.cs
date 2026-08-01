using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Numeric coverage for the Gemma-4 Per-Layer Embeddings (PLE) CPU kernel. Both
/// stages (<see cref="PerLayerEmbeddings.ComputeInputs"/> and
/// <see cref="PerLayerEmbeddings.InjectLayer"/>) are checked against an
/// independent scalar reference that reproduces the HF <c>modeling_gemma4.py</c>
/// math directly.
/// </summary>
public sealed class PerLayerEmbeddingsTests
{
    private const float Tol = 2e-4f;

    private static float GeluTanh(float x)
    {
        float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + MathF.Tanh(inner));
    }

    private static void RmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> w, float eps, Span<float> o)
    {
        float ss = 0;
        for (int i = 0; i < x.Length; i++) ss += x[i] * x[i];
        float scale = 1.0f / MathF.Sqrt(ss / x.Length + eps);
        for (int i = 0; i < x.Length; i++) o[i] = x[i] * scale * w[i];
    }

    // Row-major [M,K] · input[K] → out[M]  (HF Linear: y = x·W^T).
    private static void Gemv(float[] w, ReadOnlySpan<float> x, Span<float> o, int m, int k)
    {
        for (int r = 0; r < m; r++)
        {
            float acc = 0;
            for (int c = 0; c < k; c++) acc += w[r * k + c] * x[c];
            o[r] = acc;
        }
    }

    [Fact]
    public unsafe void ComputeInputs_MatchesReference()
    {
        const int seq = 2, hidden = 4, layers = 2, ple = 2, eps = 0;
        int lp = layers * ple;

        float[] identity = [0.3f, -0.1f, 0.2f, 0.5f, -0.4f, 0.6f, 0.1f, -0.2f]; // [seq, lp]
        float[] embeds = [0.5f, -0.2f, 0.1f, 0.4f, 0.3f, 0.7f, -0.5f, 0.2f];    // [seq, hidden]
        float[] projW = new float[lp * hidden];
        for (int i = 0; i < projW.Length; i++) projW[i] = MathF.Cos(0.7f * i + 1f) * 0.3f;
        float[] projNorm = [1.1f, 0.9f];

        // Reference.
        float projScale = 1.0f / MathF.Sqrt(hidden);
        float combine = 1.0f / MathF.Sqrt(2.0f);
        float[] expected = new float[seq * lp];
        Span<float> proj = stackalloc float[lp];
        Span<float> normed = stackalloc float[ple];
        for (int t = 0; t < seq; t++)
        {
            Gemv(projW, embeds.AsSpan(t * hidden, hidden), proj, lp, hidden);
            for (int l = 0; l < layers; l++)
            {
                Span<float> block = proj.Slice(l * ple, ple);
                for (int i = 0; i < ple; i++) block[i] *= projScale;
                RmsNorm(block, projNorm, eps, normed);
                for (int i = 0; i < ple; i++)
                {
                    int off = t * lp + l * ple + i;
                    expected[off] = (normed[i] + identity[off]) * combine;
                }
            }
        }

        float[] output = new float[seq * lp];
        float[] scratch = new float[seq * lp];
        fixed (float* pId = identity, pEmb = embeds, pProj = projW, pScr = scratch, pOut = output)
        {
            PerLayerEmbeddings.ComputeInputs(pId, pEmb, pProj, projNorm, pScr, pOut,
                seq, hidden, layers, ple, eps);
        }

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], output[i], Tol);
    }

    [Fact]
    public unsafe void InjectLayer_MatchesReference()
    {
        const int seq = 2, hidden = 4, layers = 2, ple = 2, eps = 0;
        const int layerIdx = 1;
        int lp = layers * ple;

        float[] hiddenState = [0.5f, -0.2f, 0.1f, 0.4f, 0.3f, 0.7f, -0.5f, 0.2f]; // [seq, hidden]
        float[] perLayerInputs = new float[seq * lp];
        for (int i = 0; i < perLayerInputs.Length; i++) perLayerInputs[i] = MathF.Sin(0.5f * i + 0.3f) * 0.4f;
        float[] gateW = new float[ple * hidden];
        for (int i = 0; i < gateW.Length; i++) gateW[i] = MathF.Cos(0.4f * i + 0.2f) * 0.5f;
        float[] projW = new float[hidden * ple];
        for (int i = 0; i < projW.Length; i++) projW[i] = MathF.Sin(0.9f * i + 1.1f) * 0.5f;
        float[] postNorm = [1.05f, 0.95f, 1.10f, 0.90f];

        // Reference: r + post_norm(proj(gelu_tanh(gate(h)) * ple[:,layerIdx,:])).
        float[] expected = (float[])hiddenState.Clone();
        Span<float> g = stackalloc float[ple];
        Span<float> p = stackalloc float[hidden];
        Span<float> pn = stackalloc float[hidden];
        for (int t = 0; t < seq; t++)
        {
            Gemv(gateW, hiddenState.AsSpan(t * hidden, hidden), g, ple, hidden);
            for (int i = 0; i < ple; i++)
                g[i] = GeluTanh(g[i]) * perLayerInputs[t * lp + layerIdx * ple + i];
            Gemv(projW, g, p, hidden, ple);
            RmsNorm(p, postNorm, eps, pn);
            for (int i = 0; i < hidden; i++)
                expected[t * hidden + i] += pn[i];
        }

        float[] gateScratch = new float[seq * ple];
        float[] projScratch = new float[seq * hidden];
        fixed (float* pH = hiddenState, pPli = perLayerInputs, pGate = gateW, pProj = projW,
               pGs = gateScratch, pPs = projScratch)
        {
            PerLayerEmbeddings.InjectLayer(pH, pPli, layerIdx, layers, pGate, pProj, postNorm,
                pGs, pPs, seq, hidden, ple, eps);
        }

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], hiddenState[i], Tol);
    }
}
