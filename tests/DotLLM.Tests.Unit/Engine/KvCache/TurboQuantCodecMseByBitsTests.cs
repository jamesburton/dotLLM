using System;
using DotLLM.Engine.KvCache.Codecs;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>Round-trip relative MSE by bit width at head dims 64 and 128 (#1892).</summary>
public sealed class TurboQuantCodecMseByBitsTests
{
    private readonly ITestOutputHelper _output;
    public TurboQuantCodecMseByBitsTests(ITestOutputHelper output) => _output = output;

    internal static double RelMse(int d, int bits, Func<Random, float[]> gen, int vectors = 400, ulong seed = 0xABCDEF01)
    {
        var codec = new TurboQuantCodec(d, bits, seed);
        var rng = new Random(1234);
        var codes = new byte[codec.CodeBytesPerVector];
        var recon = new float[d];
        double sum = 0;
        for (int v = 0; v < vectors; v++)
        {
            float[] x = gen(rng);
            float norm = codec.Encode(x, codes);
            codec.Decode(codes, norm, recon);
            double err = 0, sq = 0;
            for (int i = 0; i < d; i++) { double e = x[i] - recon[i]; err += e * e; sq += (double)x[i] * x[i]; }
            sum += err / sq;
        }
        return sum / vectors;
    }

    internal static float[] Gaussian(Random rng, int d)
    {
        var x = new float[d];
        for (int i = 0; i < d; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            x[i] = (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return x;
    }

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    public void Gaussian_RelativeMse_FallsMonotonicallyWithBits(int d)
    {
        double prev = double.MaxValue;
        foreach (int bits in new[] { 2, 3, 4, 8 })
        {
            double m = RelMse(d, bits, r => Gaussian(r, d));
            _output.WriteLine($"gaussian d={d} b={bits}: relMse={m:E4}");
            Assert.True(m < prev, $"d={d}: bits={bits} relMse {m} not below previous {prev}");
            prev = m;
        }
    }

    // Standard-normal Lloyd-Max normalised MSE per bit width (b -> eps_b). b=8 is the measured plateau (4.9e-5 at d=64 and 128).
    private static double Eps(int bits) => bits switch { 2 => 0.1175, 3 => 0.03454, 4 => 0.009497, 8 => 4.9e-5, _ => throw new ArgumentOutOfRangeException(nameof(bits)) };

    [Theory]
    [InlineData(64, 2)] [InlineData(64, 3)] [InlineData(64, 4)] [InlineData(64, 8)]
    [InlineData(128, 2)] [InlineData(128, 3)] [InlineData(128, 4)] [InlineData(128, 8)]
    public void Gaussian_RelativeMse_TracksPaperValue(int d, int bits)
    {
        double m = RelMse(d, bits, r => Gaussian(r, d));
        _output.WriteLine($"gaussian d={d} b={bits}: relMse={m:E4} eps={Eps(bits):E4} ratio={m / Eps(bits):F2}");
        Assert.InRange(m, 0.9 * Eps(bits), 1.1 * Eps(bits));
    }

    /// <summary>
    /// K-like vectors: one huge shared bias direction (Qwen2.5 K is ~96% a constant bias, absmax 130 vs median 5.6) plus noise.
    /// The rotation flattens the outlier energy, so the codec must stay near the paper value and monotone in bits (#1892).
    /// </summary>
    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    public void BiasDominated_RelativeMse_FallsMonotonicallyAndStaysBounded(int d)
    {
        var bias = Gaussian(new Random(99), d);
        for (int i = 0; i < d; i++) bias[i] *= i % 7 == 0 ? 60f : 4f;
        double prev = double.MaxValue;
        foreach (int bits in new[] { 2, 3, 4, 8 })
        {
            double m = RelMse(d, bits, r => { var x = Gaussian(r, d); for (int i = 0; i < d; i++) x[i] = bias[i] + 3f * x[i]; return x; });
            _output.WriteLine($"bias-dominated d={d} b={bits}: relMse={m:E4}");
            Assert.True(m < prev, $"d={d} bits={bits}: {m} not below {prev}");
            Assert.True(m < 2.0 * Eps(bits), $"d={d} bits={bits}: {m} more than 2x the paper value {Eps(bits)}");
            prev = m;
        }
    }
}
