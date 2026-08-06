using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// gpt-oss kernel-level tests: attention sinks (softmax-denominator virtual
/// key), the clamped swiglu_oai activation, and the quantized-expert MoE
/// kernel's routing (top-k on raw logits + softmax after top-k) — all against
/// independently computed references matching llama.cpp semantics.
/// </summary>
public sealed unsafe class GptOssKernelTests
{
    // ──────────────────── swiglu_oai ────────────────────

    [Theory]
    [InlineData(0.5f, 0.25f)]
    [InlineData(-2.0f, 1.0f)]
    [InlineData(10.0f, 3.0f)]    // gate above limit → clamped to 7
    [InlineData(3.0f, 12.0f)]    // up above limit → clamped to 7
    [InlineData(3.0f, -12.0f)]   // up below -limit → clamped to -7
    public void SwiGluOai_MatchesLlamaCppFormula(float gate, float up)
    {
        float[] g = [gate];
        float[] u = [up];
        float[] outv = new float[1];
        MoeQuantSwiGluMlp.SwiGluOai(g, u, outv);

        const float alpha = 1.702f, limit = 7.0f;
        float x = MathF.Min(gate, limit);
        float y = Math.Clamp(up, -limit, limit);
        float expected = x / (1f + MathF.Exp(alpha * -x)) * (y + 1f);

        Assert.Equal(expected, outv[0], 6);
    }

    // ──────────────────── Attention sinks ────────────────────

    /// <summary>
    /// Independent naive reference: per-head softmax whose denominator
    /// includes exp(sink - m); output = Σ p_j V_j (sink contributes nothing).
    /// </summary>
    private static void SinkAttentionReference(
        float[] q, float[] k, float[] v, float[] sinks, float[] output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset, int? window)
    {
        int groupSize = numHeads / numKvHeads;
        int qStride = numHeads * headDim;
        int kvStride = numKvHeads * headDim;
        float scale = 1f / MathF.Sqrt(headDim);

        for (int h = 0; h < numHeads; h++)
        {
            int kvH = h / groupSize;
            for (int i = 0; i < seqQ; i++)
            {
                var scores = new double[seqKv];
                for (int j = 0; j < seqKv; j++)
                {
                    double dot = 0;
                    for (int d = 0; d < headDim; d++)
                        dot += q[i * qStride + h * headDim + d] * k[j * kvStride + kvH * headDim + d];
                    scores[j] = dot * scale;
                    if (j > positionOffset + i) scores[j] = double.NegativeInfinity;
                    if (window is int w && j < positionOffset + i - w + 1) scores[j] = double.NegativeInfinity;
                }

                double m = sinks[h];
                foreach (double s in scores) m = Math.Max(m, s);
                double sum = Math.Exp(sinks[h] - m);
                var p = new double[seqKv];
                for (int j = 0; j < seqKv; j++) { p[j] = Math.Exp(scores[j] - m); sum += p[j]; }

                for (int d = 0; d < headDim; d++)
                {
                    double acc = 0;
                    for (int j = 0; j < seqKv; j++)
                        acc += p[j] / sum * v[j * kvStride + kvH * headDim + d];
                    output[i * qStride + h * headDim + d] = (float)acc;
                }
            }
        }
    }

    private static (float[] q, float[] k, float[] v) RandomQkv(
        Random rng, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        float[] q = new float[seqQ * numHeads * headDim];
        float[] k = new float[seqKv * numKvHeads * headDim];
        float[] v = new float[seqKv * numKvHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return (q, k, v);
    }

    [Fact]
    public void Attention_WithSinks_NaivePath_MatchesReference()
    {
        const int seqQ = 3, seqKv = 3, numHeads = 4, numKvHeads = 2, headDim = 8;
        var rng = new Random(11);
        var (q, k, v) = RandomQkv(rng, seqQ, seqKv, numHeads, numKvHeads, headDim);
        float[] sinks = [0.5f, -1.0f, 2.0f, 0.0f];

        float[] expected = new float[seqQ * numHeads * headDim];
        SinkAttentionReference(q, k, v, sinks, expected, seqQ, seqKv, numHeads, numKvHeads, headDim, 0, null);

        float[] actual = new float[expected.Length];
        Attention.Execute(q, k, v, actual, seqQ, seqKv, numHeads, numKvHeads, headDim,
                          positionOffset: 0, slidingWindowSize: null, sinks: sinks);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 4);
    }

    [Fact]
    public void Attention_WithSinks_TiledPath_MatchesReference()
    {
        // scoreSize must exceed the 8 KB stackalloc threshold to hit the
        // tiled online-softmax path: seqQ * seqKv > 2048 floats. Use a LARGE
        // sink (exp(10) ≈ 22k vs sum ≈ seqKv) so the sink roughly halves the
        // output — far above the FastExp approximation noise (~1e-3) that the
        // tiled path carries with or without sinks.
        const int seqQ = 1, seqKv = 3000, numHeads = 2, numKvHeads = 1, headDim = 8;
        var rng = new Random(23);
        var (q, k, v) = RandomQkv(rng, seqQ, seqKv, numHeads, numKvHeads, headDim);
        float[] sinks = [10.0f, 9.0f];

        float[] expected = new float[seqQ * numHeads * headDim];
        SinkAttentionReference(q, k, v, sinks, expected, seqQ, seqKv, numHeads, numKvHeads, headDim,
                               positionOffset: seqKv - 1, window: null);

        // Baseline without sinks — the sink must shrink outputs substantially,
        // proving the test discriminates.
        float[] noSink = new float[expected.Length];
        Attention.Execute(q, k, v, noSink, seqQ, seqKv, numHeads, numKvHeads, headDim,
                          positionOffset: seqKv - 1);

        float[] actual = new float[expected.Length];
        Attention.Execute(q, k, v, actual, seqQ, seqKv, numHeads, numKvHeads, headDim,
                          positionOffset: seqKv - 1, slidingWindowSize: null, sinks: sinks);

        float maxSinkEffect = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(expected[i] - actual[i]) < 5e-3f,
                $"idx {i}: expected {expected[i]}, got {actual[i]}");
            maxSinkEffect = MathF.Max(maxSinkEffect, MathF.Abs(noSink[i] - actual[i]));
        }
        Assert.True(maxSinkEffect > 0.01f, "sink had no measurable effect — test not discriminating");
    }

    [Fact]
    public void Attention_WithSinksAndSlidingWindow_MatchesReference()
    {
        const int seqQ = 4, seqKv = 4, numHeads = 2, numKvHeads = 2, headDim = 4, window = 2;
        var rng = new Random(37);
        var (q, k, v) = RandomQkv(rng, seqQ, seqKv, numHeads, numKvHeads, headDim);
        float[] sinks = [0.7f, -0.3f];

        float[] expected = new float[seqQ * numHeads * headDim];
        SinkAttentionReference(q, k, v, sinks, expected, seqQ, seqKv, numHeads, numKvHeads, headDim, 0, window);

        float[] actual = new float[expected.Length];
        Attention.Execute(q, k, v, actual, seqQ, seqKv, numHeads, numKvHeads, headDim,
                          positionOffset: 0, slidingWindowSize: window, sinks: sinks);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 4);
    }

    [Fact]
    public void Attention_SinkNegativeInfinityEquivalent_LargeNegativeSinkIsNoOp()
    {
        // A hugely negative sink adds ~0 to the denominator → same as no sink.
        const int seqQ = 2, seqKv = 2, numHeads = 1, numKvHeads = 1, headDim = 4;
        var rng = new Random(5);
        var (q, k, v) = RandomQkv(rng, seqQ, seqKv, numHeads, numKvHeads, headDim);

        // Baseline: exact-softmax scalar reference (the sink path uses exact
        // TensorPrimitives softmax; the sink-free fast path uses approximate
        // FastExp, which differs by ~1%).
        float[] baseline = new float[seqQ * numHeads * headDim];
        Attention.ExecuteScalar(q, k, v, baseline, seqQ, seqKv, numHeads, numKvHeads, headDim, 0);

        float[] withSink = new float[baseline.Length];
        Attention.Execute(q, k, v, withSink, seqQ, seqKv, numHeads, numKvHeads, headDim,
                          positionOffset: 0, slidingWindowSize: null, sinks: new[] { -1e30f });

        for (int i = 0; i < baseline.Length; i++)
            Assert.Equal(baseline[i], withSink[i], 4);
    }

    // ──────────────────── Quantized-expert MoE routing ────────────────────

    [Fact]
    public void MoeQuant_SoftmaxAfterTopK_F32Experts_MatchesManualReference()
    {
        // Tiny config: E=4, k=2, H=4, I=4. F32 experts so the expert GEMVs are
        // exact and only the routing/activation semantics are under test.
        const int E = 4, topK = 2, H = 4, I = 4;
        var rng = new Random(99);

        float[] router = new float[E * H];
        float[] routerBias = new float[E];
        float[] gateW = new float[E * I * H];
        float[] upW = new float[E * I * H];
        float[] downW = new float[E * H * I];
        float[] gateB = new float[E * I];
        float[] upB = new float[E * I];
        float[] downB = new float[E * H];
        foreach (var arr in new[] { router, routerBias, gateW, upW, downW, gateB, upB, downB })
            for (int i = 0; i < arr.Length; i++) arr[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] x = new float[H];
        for (int i = 0; i < H; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        // ── Manual reference (double precision) ──
        var logits = new double[E];
        for (int e = 0; e < E; e++)
        {
            double dot = routerBias[e];
            for (int j = 0; j < H; j++) dot += router[e * H + j] * x[j];
            logits[e] = dot;
        }
        // top-k on raw logits (stable: lower index wins ties)
        var order = Enumerable.Range(0, E).OrderByDescending(e => logits[e]).Take(topK).ToArray();
        double m = order.Max(e => logits[e]);
        double sum = order.Sum(e => Math.Exp(logits[e] - m));
        var expected = new double[H];
        foreach (int e in order)
        {
            double w = Math.Exp(logits[e] - m) / sum;
            var act = new double[I];
            for (int i = 0; i < I; i++)
            {
                double g = gateB[e * I + i], u = upB[e * I + i];
                for (int j = 0; j < H; j++)
                {
                    g += gateW[(e * I + i) * H + j] * x[j];
                    u += upW[(e * I + i) * H + j] * x[j];
                }
                double xg = Math.Min(g, 7.0);
                double yu = Math.Clamp(u, -7.0, 7.0);
                act[i] = xg / (1.0 + Math.Exp(1.702 * -xg)) * (yu + 1.0);
            }
            for (int d = 0; d < H; d++)
            {
                double o = downB[e * H + d];
                for (int i = 0; i < I; i++) o += downW[(e * H + d) * I + i] * act[i];
                expected[d] += w * o;
            }
        }

        // ── Kernel ──
        float[] output = new float[H];
        fixed (float* xp = x)
        fixed (float* op = output)
        fixed (float* gw = gateW)
        fixed (float* uw = upW)
        fixed (float* dw = downW)
        {
            MoeQuantSwiGluMlp.Execute(
                hidden: xp, output: op, seqLen: 1,
                routerWeight: router, routerBias: routerBias,
                gateExpsBase: (nint)gw, gateQt: QuantizationType.F32,
                upExpsBase: (nint)uw, upQt: QuantizationType.F32,
                downExpsBase: (nint)dw, downQt: QuantizationType.F32,
                gateBias: gateB, upBias: upB, downBias: downB,
                numExperts: E, numExpertsPerTok: topK,
                hiddenSize: H, intermediateSize: I,
                softmaxAfterTopK: true, useSwiGluOai: true,
                pool: null);
        }

        for (int d = 0; d < H; d++)
            Assert.True(Math.Abs(expected[d] - output[d]) < 1e-4,
                $"dim {d}: expected {expected[d]}, got {output[d]}");
    }

    [Fact]
    public void MoeQuant_Mxfp4Experts_MatchesDequantReference()
    {
        // Exact oracle (issue #275): the MXFP4 GEMVs compute
        // dot(dequantized MXFP4 row, f32 activations) directly — no Q8_0
        // round-trip of the activations (that step was removed by #275: it
        // added noise dotLLM's CUDA/Vulkan MXFP4 path does not have, since
        // both dequantize to F16 and run a full-precision GEMM/GEMV with no
        // activation quantization at all). Rebuild that pipeline in double
        // precision — only float summation-order noise remains.
        const int E = 4, topK = 2, H = 64, I = 64;
        var rng = new Random(314);

        float[] router = new float[E * H];
        for (int i = 0; i < router.Length; i++) router[i] = (float)(rng.NextDouble() * 2 - 1);

        static byte[] RandomMxfp4(int elements, Random r)
        {
            int blocks = elements / 32;
            byte[] data = new byte[blocks * 17];
            for (int b = 0; b < blocks; b++)
            {
                data[b * 17] = (byte)r.Next(122, 133);
                for (int j = 0; j < 16; j++) data[b * 17 + 1 + j] = (byte)r.Next(256);
            }
            return data;
        }

        byte[] gateQ = RandomMxfp4(E * I * H, rng);
        byte[] upQ = RandomMxfp4(E * I * H, rng);
        byte[] downQ = RandomMxfp4(E * H * I, rng);

        float[] gateF = new float[E * I * H];
        float[] upF = new float[E * I * H];
        float[] downF = new float[E * H * I];
        fixed (byte* p = gateQ) Dequantize.ToFloat32((nint)p, gateF.Length, QuantizationType.MXFP4, gateF);
        fixed (byte* p = upQ) Dequantize.ToFloat32((nint)p, upF.Length, QuantizationType.MXFP4, upF);
        fixed (byte* p = downQ) Dequantize.ToFloat32((nint)p, downF.Length, QuantizationType.MXFP4, downF);

        float[] x = new float[H];
        for (int i = 0; i < H; i++) x[i] = (float)(rng.NextDouble() * 0.5 - 0.25);

        // ── Kernel under test (MXFP4 experts) ──
        float[] outQ = new float[H];
        fixed (float* xp = x)
        fixed (float* oq = outQ)
        fixed (byte* gq = gateQ)
        fixed (byte* uq = upQ)
        fixed (byte* dq = downQ)
        {
            MoeQuantSwiGluMlp.Execute(xp, oq, 1, router, ReadOnlySpan<float>.Empty,
                (nint)gq, QuantizationType.MXFP4, (nint)uq, QuantizationType.MXFP4,
                (nint)dq, QuantizationType.MXFP4,
                ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty,
                E, topK, H, I, softmaxAfterTopK: true, useSwiGluOai: true, pool: null);
        }

        // ── Manual reference ──
        var logits = new double[E];
        for (int e = 0; e < E; e++)
        {
            double dot = 0;
            for (int j = 0; j < H; j++) dot += router[e * H + j] * x[j];
            logits[e] = dot;
        }
        var order = Enumerable.Range(0, E).OrderByDescending(e => logits[e]).Take(topK).ToArray();
        double m = order.Max(e => logits[e]);
        double sum = order.Sum(e => Math.Exp(logits[e] - m));

        var expected = new double[H];
        var magnitude = new double[H];
        foreach (int e in order)
        {
            double w = Math.Exp(logits[e] - m) / sum;
            var act = new float[I];
            for (int i = 0; i < I; i++)
            {
                double g = 0, u = 0;
                for (int j = 0; j < H; j++)
                {
                    g += (double)gateF[(e * I + i) * H + j] * x[j];
                    u += (double)upF[(e * I + i) * H + j] * x[j];
                }
                double xg = Math.Min(g, 7.0);
                double yu = Math.Clamp(u, -7.0, 7.0);
                act[i] = (float)(xg / (1.0 + Math.Exp(1.702 * -xg)) * (yu + 1.0));
            }
            for (int d = 0; d < H; d++)
            {
                double o = 0, mag = 0;
                for (int i = 0; i < I; i++)
                {
                    double term = (double)downF[(e * H + d) * I + i] * act[i];
                    o += term;
                    mag += Math.Abs(term);
                }
                expected[d] += w * o;
                magnitude[d] += w * mag;
            }
        }

        for (int d = 0; d < H; d++)
        {
            // Ordinary float summation-order tolerance only — no quantization noise remains.
            double tol = magnitude[d] * 1e-5 + 1e-4;
            Assert.True(Math.Abs(expected[d] - outQ[d]) <= tol,
                $"dim {d}: expected {expected[d]}, got {outQ[d]} (tol {tol})");
        }
    }
}
