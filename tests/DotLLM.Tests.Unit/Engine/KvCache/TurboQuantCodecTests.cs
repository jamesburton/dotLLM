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
