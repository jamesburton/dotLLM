using System;
using DotLLM.Engine.KvCache.Codecs;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>
/// Validates the TurboQuant MSE codec (rotation + per-coordinate Lloyd–Max scalar quant +
/// per-vector norm). Quality is checked against the paper's per-bit distortion: for a unit
/// vector the relative reconstruction error <c>‖x−x̃‖²/‖x‖²</c> approaches the standard-normal
/// Lloyd–Max distortion <c>ε_b</c> (≈0.363, 0.118, 0.0345, 0.0095 for b=1..4). A
/// <b>discriminating</b> rotation test proves the seeded rotation is actually inverted on
/// decode (decoding with the wrong seed collapses reconstruction), so a no-op / mis-seeded
/// rotation cannot pass.
/// </summary>
public sealed class TurboQuantCodecTests
{
    private readonly ITestOutputHelper _output;
    public TurboQuantCodecTests(ITestOutputHelper output) => _output = output;

    // Standard-normal Lloyd–Max normalized MSE per bit-width (b → ε_b).
    private static double ExpectedEps(int bits) => bits switch
    {
        1 => 0.3634,
        2 => 0.1175,
        3 => 0.03454,
        4 => 0.009497,
        _ => throw new ArgumentOutOfRangeException(nameof(bits)),
    };

    [Theory]
    [InlineData(128, 2)]
    [InlineData(128, 3)]
    [InlineData(128, 4)]
    [InlineData(256, 4)]
    public void Reconstruction_RelativeMse_TracksPaperBound(int headDim, int bits)
    {
        var codec = new TurboQuantCodec(headDim, bits, seed: 0xABCDEF01);
        var rng = new Random(1234);

        int vectors = 400;
        double sumRel = 0;
        double sumCos = 0;
        Span<byte> codes = stackalloc byte[codec.CodeBytesPerVector];
        var recon = new float[headDim];

        for (int v = 0; v < vectors; v++)
        {
            float[] x = RandomGaussian(rng, headDim);
            float norm = codec.Encode(x, codes);
            codec.Decode(codes, norm, recon);

            double err = 0, sq = 0, dot = 0, rn = 0;
            for (int i = 0; i < headDim; i++)
            {
                double d = x[i] - recon[i];
                err += d * d;
                sq += (double)x[i] * x[i];
                dot += (double)x[i] * recon[i];
                rn += (double)recon[i] * recon[i];
            }
            sumRel += err / sq;
            sumCos += dot / (Math.Sqrt(sq) * Math.Sqrt(rn) + 1e-12);
        }

        double relMse = sumRel / vectors;
        double cos = sumCos / vectors;
        double eps = ExpectedEps(bits);
        _output.WriteLine($"d={headDim} b={bits}: relMse={relMse:F4} (ε_b={eps:F4}, ratio={relMse / eps:F2}x), cos={cos:F4}");

        // Lloyd–Max is optimal for the asymptotic Gaussian; with the finite-d RHT rotation
        // the empirical distortion sits close to ε_b. Bound generously both ways: a much
        // larger value means the rotation/codebook is wrong; a much smaller one (≈0) would
        // mean the test is degenerate (e.g. reconstruction == input by accident).
        Assert.InRange(relMse, 0.5 * eps, 1.6 * eps);
        Assert.True(cos > 1.0 - 2.0 * eps, $"cosine {cos:F4} too low for b={bits}");
    }

    [Fact]
    public void Reconstruction_ImprovesMonotonically_WithBits()
    {
        const int d = 128;
        var rng = new Random(7);
        float[][] xs = new float[200][];
        for (int i = 0; i < xs.Length; i++) xs[i] = RandomGaussian(rng, d);

        double prev = double.PositiveInfinity;
        for (int bits = 1; bits <= 4; bits++)
        {
            var codec = new TurboQuantCodec(d, bits, seed: 99);
            double rel = MeanRelMse(codec, xs);
            _output.WriteLine($"b={bits}: relMse={rel:F4}");
            Assert.True(rel < prev, $"b={bits} relMse {rel:F4} not better than previous {prev:F4}");
            prev = rel;
        }
    }

    [Fact]
    public void Rotation_IsInverted_WrongSeedCollapsesReconstruction()
    {
        // DISCRIMINATING: encode with one seed, decode with a codec built from a DIFFERENT
        // seed (same d/bits). If the rotation is genuinely seeded + inverted, the wrong
        // inverse rotation scrambles the direction and reconstruction error explodes. A
        // codec that skipped/ignored the rotation would reconstruct equally well either way.
        const int d = 128, bits = 4;
        var enc = new TurboQuantCodec(d, bits, seed: 11111);
        var right = new TurboQuantCodec(d, bits, seed: 11111);
        var wrong = new TurboQuantCodec(d, bits, seed: 22222);
        var rng = new Random(3);

        Span<byte> codes = stackalloc byte[enc.CodeBytesPerVector];
        var rOk = new float[d];
        var rBad = new float[d];

        double okSum = 0, badSum = 0;
        int n = 100;
        for (int v = 0; v < n; v++)
        {
            float[] x = RandomGaussian(rng, d);
            float norm = enc.Encode(x, codes);
            right.Decode(codes, norm, rOk);
            wrong.Decode(codes, norm, rBad);
            okSum += RelMse(x, rOk);
            badSum += RelMse(x, rBad);
        }
        double ok = okSum / n, bad = badSum / n;
        _output.WriteLine($"same-seed relMse={ok:F4}, wrong-seed relMse={bad:F4} ({bad / ok:F1}x worse)");

        Assert.True(ok < 0.02, $"same-seed reconstruction should be tight, got {ok:F4}");
        Assert.True(bad > 10 * ok, $"wrong-seed reconstruction should collapse, got {bad:F4} vs {ok:F4}");
    }

    [Fact]
    public void SameSeed_ProducesIdenticalCodes_Deterministic()
    {
        const int d = 256, bits = 3;
        var a = new TurboQuantCodec(d, bits, seed: 555);
        var b = new TurboQuantCodec(d, bits, seed: 555);
        var rng = new Random(42);
        float[] x = RandomGaussian(rng, d);

        var ca = new byte[a.CodeBytesPerVector];
        var cb = new byte[b.CodeBytesPerVector];
        float na = a.Encode(x, ca);
        float nb = b.Encode(x, cb);

        Assert.Equal(na, nb);
        Assert.Equal(ca, cb);
    }

    [Fact]
    public void Centroids_MatchPaperValues_ForB1AndB2()
    {
        const int d = 256;
        float invSqrtD = 1.0f / MathF.Sqrt(d);

        var b1 = new TurboQuantCodec(d, 1, seed: 1);
        // Paper b=1: ±√(2/(πd)).
        float expected1 = MathF.Sqrt(2.0f / (MathF.PI * d));
        Assert.Equal(-expected1, b1.Centroids[0], 3);
        Assert.Equal(expected1, b1.Centroids[1], 3);

        var b2 = new TurboQuantCodec(d, 2, seed: 1);
        // Paper b=2: ±0.4528/√d (inner), ±1.5104/√d (outer).
        Assert.Equal(-1.5104f * invSqrtD, b2.Centroids[0], 3);
        Assert.Equal(-0.4528f * invSqrtD, b2.Centroids[1], 3);
        Assert.Equal(0.4528f * invSqrtD, b2.Centroids[2], 3);
        Assert.Equal(1.5104f * invSqrtD, b2.Centroids[3], 3);
    }

    [Fact]
    public void ZeroVector_RoundTripsToZero()
    {
        const int d = 128, bits = 4;
        var codec = new TurboQuantCodec(d, bits, seed: 7);
        var codes = new byte[codec.CodeBytesPerVector];
        var recon = new float[d];

        float norm = codec.Encode(new float[d], codes);
        codec.Decode(codes, norm, recon);

        Assert.Equal(0f, norm);
        foreach (float r in recon) Assert.Equal(0f, r);
    }

    [Fact]
    public void NonPowerOfTwoHeadDim_Throws()
        => Assert.Throws<ArgumentException>(() => new TurboQuantCodec(96, 4, seed: 1));

    // ── QJL residual stage (Algorithm 2) ──────────────────────────────────

    [Fact]
    public void Qjl_RequiresAtLeastTwoBits()
        => Assert.Throws<ArgumentOutOfRangeException>(() => new TurboQuantCodec(128, 1, seed: 1, useQjl: true));

    [Fact]
    public void Qjl_CodeLayout_PacksMseCodesPlusSignsPlusNorm()
    {
        // d=128, 4-bit budget with QJL ⇒ MSE at 3 bits: ceil(128*3/8)=48 code bytes +
        // ceil(128/8)=16 sign bytes + 4 (γ) = 68. Pure-MSE 4-bit is ceil(128*4/8)=64.
        var qjl = new TurboQuantCodec(128, 4, seed: 1, useQjl: true);
        var mse = new TurboQuantCodec(128, 4, seed: 1, useQjl: false);
        Assert.True(qjl.UseQjl);
        Assert.False(mse.UseQjl);
        Assert.Equal(8, qjl.LevelCount);   // MSE stage runs at bits-1 = 3 bits
        Assert.Equal(16, mse.LevelCount);
        Assert.Equal(68, qjl.CodeBytesPerVector);
        Assert.Equal(64, mse.CodeBytesPerVector);
    }

    [Fact]
    public void Qjl_RemovesSelfScoreContractionBias()
    {
        // DISCRIMINATING: the canonical high-attention case is a query aligned with its key
        // (y = x), where the score is ‖x‖². The MSE stage is a contraction, so ⟨x, x̃_mse⟩
        // underestimates ‖x‖² by ≈‖r‖² (the orthogonality principle: the quant error r is
        // ~orthogonal to the reconstruction, so ⟨x,r⟩≈‖r‖²>0). QJL folds in an unbiased estimate
        // of that residual, driving the mean self-score error to ≈0. We compare the QJL codec
        // against the *pure-MSE codec at QJL's internal bit-width* (same seed ⇒ identical MSE
        // reconstruction), isolating the QJL correction. A no-op / mis-scaled QJL cannot pass:
        // it would leave the contraction bias intact.
        const int d = 128, budget = 4;
        var mse3 = new TurboQuantCodec(d, budget - 1, seed: 4242, useQjl: false); // QJL's MSE stage
        var qjl4 = new TurboQuantCodec(d, budget, seed: 4242, useQjl: true);
        var rng = new Random(2025);

        int n = 3000;
        double biasMse = 0, biasQjl = 0;
        var cMse = new byte[mse3.CodeBytesPerVector];
        var cQjl = new byte[qjl4.CodeBytesPerVector];
        var rMse = new float[d];
        var rQjl = new float[d];

        for (int v = 0; v < n; v++)
        {
            float[] x = RandomGaussian(rng, d);
            float nMse = mse3.Encode(x, cMse); mse3.Decode(cMse, nMse, rMse);
            float nQjl = qjl4.Encode(x, cQjl); qjl4.Decode(cQjl, nQjl, rQjl);

            double trueScore = 0, sMse = 0, sQjl = 0;
            for (int i = 0; i < d; i++)
            {
                trueScore += (double)x[i] * x[i];
                sMse += (double)x[i] * rMse[i];
                sQjl += (double)x[i] * rQjl[i];
            }
            biasMse += (sMse - trueScore) / trueScore;
            biasQjl += (sQjl - trueScore) / trueScore;
        }
        biasMse /= n;
        biasQjl /= n;
        _output.WriteLine($"self-score mean rel bias: MSE(3b)={biasMse:F5}, QJL(4b)={biasQjl:F5}");

        // The pure-MSE stage contracts: clearly negative bias (≈ -ε_3 ≈ -0.034).
        Assert.True(biasMse < -0.01, $"expected MSE contraction bias, got {biasMse:F5}");
        // QJL removes (nearly) all of it: the residual mean bias is a small fraction of MSE's.
        Assert.True(Math.Abs(biasQjl) < 0.25 * Math.Abs(biasMse),
            $"QJL did not debias: |{biasQjl:F5}| vs |{biasMse:F5}|");
    }

    [Fact]
    public void Qjl_DebiasesCrossScores_AgainstFullPrecisionQuery()
    {
        // Beyond the aligned case: for a query y that is a noisy variant of the key x (a realistic
        // high-attention pair), the MSE score is still biased low; QJL's mean error is ≈0.
        const int d = 256, budget = 4;
        var mse3 = new TurboQuantCodec(d, budget - 1, seed: 9, useQjl: false);
        var qjl4 = new TurboQuantCodec(d, budget, seed: 9, useQjl: true);
        var rng = new Random(77);

        int n = 3000;
        double biasMse = 0, biasQjl = 0;
        var cMse = new byte[mse3.CodeBytesPerVector];
        var cQjl = new byte[qjl4.CodeBytesPerVector];
        var rMse = new float[d];
        var rQjl = new float[d];

        for (int v = 0; v < n; v++)
        {
            float[] x = RandomGaussian(rng, d);
            float[] noise = RandomGaussian(rng, d);
            var y = new float[d];
            for (int i = 0; i < d; i++) y[i] = x[i] + 0.3f * noise[i]; // correlated query

            float nMse = mse3.Encode(x, cMse); mse3.Decode(cMse, nMse, rMse);
            float nQjl = qjl4.Encode(x, cQjl); qjl4.Decode(cQjl, nQjl, rQjl);

            double trueScore = 0, sMse = 0, sQjl = 0;
            for (int i = 0; i < d; i++)
            {
                trueScore += (double)y[i] * x[i];
                sMse += (double)y[i] * rMse[i];
                sQjl += (double)y[i] * rQjl[i];
            }
            biasMse += (sMse - trueScore) / trueScore;
            biasQjl += (sQjl - trueScore) / trueScore;
        }
        biasMse /= n;
        biasQjl /= n;
        _output.WriteLine($"cross-score mean rel bias: MSE(3b)={biasMse:F5}, QJL(4b)={biasQjl:F5}");

        Assert.True(biasMse < -0.01, $"expected MSE contraction bias, got {biasMse:F5}");
        Assert.True(Math.Abs(biasQjl) < 0.3 * Math.Abs(biasMse),
            $"QJL did not debias cross-scores: |{biasQjl:F5}| vs |{biasMse:F5}|");
    }

    [Fact]
    public void Qjl_RaisesL2Error_WhileLoweringScoreBias_Tradeoff()
    {
        // The QJL residual is a JL-noisy estimate of r: it fixes inner-product bias but RAISES the
        // per-vector ℓ2 reconstruction error vs pure MSE at the same total budget. Documenting the
        // trade prevents anyone "optimising" QJL by chasing relMse (which would defeat the point).
        const int d = 128, budget = 4;
        var mse = new TurboQuantCodec(d, budget, seed: 31, useQjl: false);
        var qjl = new TurboQuantCodec(d, budget, seed: 31, useQjl: true);
        var rng = new Random(8);
        float[][] xs = new float[400][];
        for (int i = 0; i < xs.Length; i++) xs[i] = RandomGaussian(rng, d);

        double relMse = MeanRelMse(mse, xs);
        double relQjl = MeanRelMse(qjl, xs);
        _output.WriteLine($"relMse: pure-MSE(4b)={relMse:F4}, QJL(4b)={relQjl:F4}");
        Assert.True(relQjl > relMse, $"expected QJL ℓ2 error {relQjl:F4} > MSE {relMse:F4}");
    }

    [Fact]
    public void Qjl_SameSeed_ProducesIdenticalCodes_Deterministic()
    {
        const int d = 256, bits = 4;
        var a = new TurboQuantCodec(d, bits, seed: 1234, useQjl: true);
        var b = new TurboQuantCodec(d, bits, seed: 1234, useQjl: true);
        var rng = new Random(99);
        float[] x = RandomGaussian(rng, d);

        var ca = new byte[a.CodeBytesPerVector];
        var cb = new byte[b.CodeBytesPerVector];
        float na = a.Encode(x, ca);
        float nb = b.Encode(x, cb);

        Assert.Equal(na, nb);
        Assert.Equal(ca, cb); // includes MSE codes, QJL sign bits, and the packed residual norm γ

        // ... and a different seed yields different sign bits (the sketch is genuinely seeded).
        var c = new TurboQuantCodec(d, bits, seed: 4321, useQjl: true);
        var cc = new byte[c.CodeBytesPerVector];
        c.Encode(x, cc);
        Assert.NotEqual(ca, cc);
    }

    [Fact]
    public void Qjl_ZeroVector_RoundTripsToZero()
    {
        const int d = 128, bits = 4;
        var codec = new TurboQuantCodec(d, bits, seed: 7, useQjl: true);
        var codes = new byte[codec.CodeBytesPerVector];
        var recon = new float[d];

        float norm = codec.Encode(new float[d], codes);
        codec.Decode(codes, norm, recon);

        Assert.Equal(0f, norm);
        foreach (float r in recon) Assert.Equal(0f, r); // γ=0 ⇒ QJL adds nothing
    }

    // ── helpers ──────────────────────────────────────────────────────────

    private static double MeanRelMse(TurboQuantCodec codec, float[][] xs)
    {
        var codes = new byte[codec.CodeBytesPerVector];
        var recon = new float[codec.HeadDim];
        double sum = 0;
        foreach (float[] x in xs)
        {
            float norm = codec.Encode(x, codes);
            codec.Decode(codes, norm, recon);
            sum += RelMse(x, recon);
        }
        return sum / xs.Length;
    }

    private static double RelMse(ReadOnlySpan<float> x, ReadOnlySpan<float> recon)
    {
        double err = 0, sq = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double d = x[i] - recon[i];
            err += d * d;
            sq += (double)x[i] * x[i];
        }
        return err / sq;
    }

    private static float[] RandomGaussian(Random rng, int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++)
        {
            // Box–Muller.
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            v[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return v;
    }
}
