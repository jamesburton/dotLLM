using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Tests for bidirectional (non-causal) attention — the B0 foundation for diffusion / encoder language
/// models (issue: diffusion-LM mechanism spike). Discriminates bidirectional from causal: a query at
/// position 0 must be able to attend to a high-scoring FUTURE key (which causal masks away).
/// </summary>
public sealed class AttentionBidirectionalTests
{
    [Fact]
    public void Bidirectional_QueryAttendsToFutureKey_CausalDoesNot()
    {
        // seq=2, 1 head, headDim=2, scale=1. Construct so query 0 strongly prefers key 1 (the future):
        //   score(q0,k0)=0, score(q0,k1)=10  → softmax over {0,10} ≈ one-hot on key 1.
        // Bidirectional: out0 ≈ v1. Causal: query 0 sees only key 0, so out0 == v0.
        int seq = 2, numHeads = 1, numKvHeads = 1, headDim = 2;
        float scale = 1f;

        float[] q = { 1f, 0f,   // q0
                      1f, 0f }; // q1
        float[] k = { 0f, 0f,   // k0 → dot(q0,k0)=0
                      10f, 0f }; // k1 → dot(q0,k1)=10
        float[] v = { 5f, 5f,   // v0
                      9f, 9f };  // v1

        float[] outBi = new float[seq * numHeads * headDim];
        float[] outCausal = new float[seq * numHeads * headDim];

        Attention.ExecuteBidirectional(q, k, v, outBi, seq, numHeads, numKvHeads, headDim, scale);
        Attention.Execute(q, k, v, outCausal, seq, seq, numHeads, numKvHeads, headDim,
                          positionOffset: 0, scale: scale);

        // Bidirectional out0 ≈ v1 = [9,9] (score gap of 10 makes softmax effectively one-hot, robust to
        // the kernel's fast-exp softmax).
        Assert.Equal(9f, outBi[0], 0.1f);
        Assert.Equal(9f, outBi[1], 0.1f);

        // Causal out0 == v0 = [5,5] exactly (query 0 attends only to key 0; single-element softmax = 1).
        Assert.Equal(5f, outCausal[0], 1e-4f);
        Assert.Equal(5f, outCausal[1], 1e-4f);

        // Discriminating: the two modes must differ at position 0 — proves the causal mask was actually
        // dropped (a vacuous test where both paths masked identically would fail here).
        Assert.True(MathF.Abs(outBi[0] - outCausal[0]) > 1f,
            $"bidirectional and causal outputs did not diverge (bi={outBi[0]}, causal={outCausal[0]})");
    }

    [Fact]
    public void Bidirectional_MatchesNaiveFullSoftmaxReference()
    {
        // Random-ish small case vs a naive full (unmasked) softmax reference. Uses a large score spread so
        // the kernel's fast-exp softmax and an exact reference agree to a loose tolerance.
        int seq = 4, numHeads = 2, numKvHeads = 1, headDim = 3;
        float scale = 1f / MathF.Sqrt(headDim);
        var rng = new Random(42);

        float[] q = new float[seq * numHeads * headDim];
        float[] k = new float[seq * numKvHeads * headDim];
        float[] v = new float[seq * numKvHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] outBi = new float[seq * numHeads * headDim];
        Attention.ExecuteBidirectional(q, k, v, outBi, seq, numHeads, numKvHeads, headDim, scale);

        // Naive reference: full (no-mask) softmax @ V per head, GQA broadcast (numKvHeads=1).
        int qStride = numHeads * headDim, kvStride = numKvHeads * headDim;
        float[] expected = new float[seq * numHeads * headDim];
        for (int h = 0; h < numHeads; h++)
        {
            for (int i = 0; i < seq; i++)
            {
                Span<float> scores = stackalloc float[seq];
                float max = float.NegativeInfinity;
                for (int j = 0; j < seq; j++)
                {
                    float dot = 0;
                    for (int d = 0; d < headDim; d++)
                        dot += q[i * qStride + h * headDim + d] * k[j * kvStride + d];
                    scores[j] = dot * scale;
                    if (scores[j] > max) max = scores[j];
                }
                float sum = 0;
                for (int j = 0; j < seq; j++) { scores[j] = MathF.Exp(scores[j] - max); sum += scores[j]; }
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0;
                    for (int j = 0; j < seq; j++) acc += scores[j] / sum * v[j * kvStride + d];
                    expected[i * qStride + h * headDim + d] = acc;
                }
            }
        }

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], outBi[i], 0.02f);
    }
}
