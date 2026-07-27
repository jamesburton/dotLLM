using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Round-trip correctness gate for <see cref="Quantize"/>: quantize F32 → bytes →
/// <see cref="Dequantize"/> back to F32 and assert the reconstruction is within the
/// format's expected quantization error. This proves the quantizers emit blocks that the
/// existing dequantizers read back BIT-COMPATIBLY (the contract for synthesizing GGUF
/// fixtures). All inputs are deterministic (a seeded xorshift PRNG — no Math.Random / time).
/// </summary>
public sealed unsafe class QuantizeRoundTripTests
{
    private readonly ITestOutputHelper _out;

    public QuantizeRoundTripTests(ITestOutputHelper output) => _out = output;

    /// <summary>Deterministic xorshift32 → float in a symmetric range. No external entropy.</summary>
    private sealed class Xorshift
    {
        private uint _s;
        public Xorshift(uint seed) => _s = seed == 0 ? 0x9E3779B9u : seed;
        public uint NextUInt() { uint x = _s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; _s = x; return x; }
        /// <summary>Uniform float in [-scale, scale).</summary>
        public float NextSigned(float scale)
        {
            // 24-bit mantissa fraction in [0,1)
            float u = (NextUInt() >> 8) * (1.0f / 16777216.0f);
            return (u * 2f - 1f) * scale;
        }
    }

    private static float[] RandomVec(int n, uint seed, float scale)
    {
        var rng = new Xorshift(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = rng.NextSigned(scale);
        return v;
    }

    /// <summary>Quantize then dequantize through the real on-disk codecs; returns reconstruction.</summary>
    private static float[] RoundTrip(float[] input, QuantizationType qt)
    {
        byte[] q = Quantize.FromFloat32(input, input.Length, qt);
        // Sanity: emitted size must equal the dequant's row stride (bit-layout contract).
        Assert.Equal(Dequantize.RowByteSize(input.Length, qt), q.Length);

        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)q.Length, 64);
        try
        {
            q.AsSpan().CopyTo(new Span<byte>((void*)ptr, q.Length));
            var dest = new float[input.Length];
            Dequantize.ToFloat32(ptr, input.Length, qt, dest);
            return dest;
        }
        finally { NativeMemory.AlignedFree((void*)ptr); }
    }

    private static (float maxAbs, float maxRel) Errors(float[] a, float[] b)
    {
        float maxAbs = 0f, maxRel = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float e = MathF.Abs(a[i] - b[i]);
            if (e > maxAbs) maxAbs = e;
            float denom = MathF.Max(MathF.Abs(a[i]), 1e-4f);
            float rel = e / denom;
            if (rel > maxRel) maxRel = rel;
        }
        return (maxAbs, maxRel);
    }

    // ──────────────────── Q8_0 ────────────────────

    [Theory]
    [InlineData(32u, 1.0f)]
    [InlineData(2u, 5.0f)]
    [InlineData(7u, 0.01f)]
    public void Q8_0_RoundTrip_WithinMaxOver127(uint seed, float scale)
    {
        const int n = 32 * 8;
        float[] input = RandomVec(n, seed, scale);
        float[] rt = RoundTrip(input, QuantizationType.Q8_0);

        // Q8_0 abs error per block <= block-max / 127 (8-bit linear, symmetric).
        // Validate per block against that block's own max magnitude.
        for (int blk = 0; blk < n / 32; blk++)
        {
            float amax = 0f;
            for (int i = 0; i < 32; i++) amax = MathF.Max(amax, MathF.Abs(input[blk * 32 + i]));
            float tol = amax / 127f + 1e-6f;
            for (int i = 0; i < 32; i++)
                Assert.True(MathF.Abs(input[blk * 32 + i] - rt[blk * 32 + i]) <= tol,
                    $"Q8_0 block {blk} elem {i}: |{input[blk * 32 + i]} - {rt[blk * 32 + i]}| > {tol}");
        }
        var (maxAbs, _) = Errors(input, rt);
        _out.WriteLine($"Q8_0 seed={seed} scale={scale}: maxAbs={maxAbs:E4}");
    }

    [Fact]
    public void Q8_0_AllZeros_RoundTripsToZero()
    {
        float[] input = new float[64];
        float[] rt = RoundTrip(input, QuantizationType.Q8_0);
        Assert.All(rt, v => Assert.Equal(0f, v));
    }

    // ──────────────────── Q5_0 ────────────────────

    [Theory]
    [InlineData(11u, 1.0f)]
    [InlineData(13u, 8.0f)]
    public void Q5_0_RoundTrip_WithinExpectedError(uint seed, float scale)
    {
        const int n = 32 * 8;
        float[] input = RandomVec(n, seed, scale);
        float[] rt = RoundTrip(input, QuantizationType.Q5_0);
        var (maxAbs, _) = Errors(input, rt);
        // 5-bit signed (32 levels over [-max,max]): step ~ max/16. Bound per block max/16.
        float gmax = 0f; foreach (var x in input) gmax = MathF.Max(gmax, MathF.Abs(x));
        float tol = gmax / 16f + 1e-5f;
        Assert.True(maxAbs <= tol, $"Q5_0 maxAbs {maxAbs} > {tol}");
        _out.WriteLine($"Q5_0 seed={seed} scale={scale}: maxAbs={maxAbs:E4} tol={tol:E4}");
    }

    [Fact]
    public void Q5_0_AllZeros_RoundTripsToZero()
    {
        float[] input = new float[64];
        float[] rt = RoundTrip(input, QuantizationType.Q5_0);
        Assert.All(rt, v => Assert.Equal(0f, v));
    }

    // ──────────────────── Q5_1 ────────────────────

    [Theory]
    [InlineData(21u, 1.0f)]
    [InlineData(23u, 4.0f)]
    public void Q5_1_RoundTrip_WithinExpectedError(uint seed, float scale)
    {
        const int n = 32 * 8;
        float[] input = RandomVec(n, seed, scale);
        float[] rt = RoundTrip(input, QuantizationType.Q5_1);
        var (maxAbs, _) = Errors(input, rt);
        // 5-bit unsigned over [min,max] per block: step = (max-min)/31. Bound by global span/31.
        float gmin = float.MaxValue, gmax = float.MinValue;
        foreach (var x in input) { gmin = MathF.Min(gmin, x); gmax = MathF.Max(gmax, x); }
        float tol = (gmax - gmin) / 31f + 1e-5f;
        Assert.True(maxAbs <= tol, $"Q5_1 maxAbs {maxAbs} > {tol}");
        _out.WriteLine($"Q5_1 seed={seed} scale={scale}: maxAbs={maxAbs:E4} tol={tol:E4}");
    }

    [Fact]
    public void Q5_1_AllZeros_RoundTripsToZero()
    {
        float[] input = new float[64];
        float[] rt = RoundTrip(input, QuantizationType.Q5_1);
        Assert.All(rt, v => Assert.Equal(0f, v));
    }

    // ──────────────────── Q4_K ────────────────────

    [Theory]
    [InlineData(101u, 1.0f)]
    [InlineData(103u, 6.0f)]
    [InlineData(107u, 0.05f)]
    public void Q4_K_RoundTrip_WithinKQuantTolerance(uint seed, float scale)
    {
        const int n = 256 * 3; // 3 super-blocks
        float[] input = RandomVec(n, seed, scale);
        float[] rt = RoundTrip(input, QuantizationType.Q4_K);
        var (maxAbs, maxRel) = Errors(input, rt);

        // Q4_K: 4-bit nibble over a per-sub-block affine range. Worst-case abs error is
        // ~ (range/15)/2 per sub-block plus the 6-bit scale/min quantization. Bound by a
        // generous fraction of the global magnitude span (K-quant is lossy but stable).
        float gmin = float.MaxValue, gmax = float.MinValue;
        foreach (var x in input) { gmin = MathF.Min(gmin, x); gmax = MathF.Max(gmax, x); }
        float span = gmax - gmin;
        float tol = span / 15f + 1e-4f; // half-step would be /30; allow full step for scale-quant slack
        Assert.True(maxAbs <= tol, $"Q4_K maxAbs {maxAbs} > {tol} (span {span})");
        _out.WriteLine($"Q4_K seed={seed} scale={scale}: maxAbs={maxAbs:E4} maxRel={maxRel:E3} tol={tol:E4}");
    }

    [Fact]
    public void Q4_K_AllZeros_RoundTripsToZero()
    {
        float[] input = new float[256];
        float[] rt = RoundTrip(input, QuantizationType.Q4_K);
        Assert.All(rt, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Q4_K_ConstantBlock_RoundTripsExactlyish()
    {
        // A constant positive block: min clamps to 0, all nibbles equal -> near-exact.
        float[] input = new float[256];
        Array.Fill(input, 3.5f);
        float[] rt = RoundTrip(input, QuantizationType.Q4_K);
        var (maxAbs, _) = Errors(input, rt);
        Assert.True(maxAbs <= 3.5f / 15f + 1e-3f, $"Q4_K constant maxAbs {maxAbs}");
        _out.WriteLine($"Q4_K constant: maxAbs={maxAbs:E4}");
    }

    // ──────────────────── F32 / F16 passthrough ────────────────────

    [Fact]
    public void F32_RoundTrip_IsExact()
    {
        float[] input = RandomVec(40, 99u, 10f);
        float[] rt = RoundTrip(input, QuantizationType.F32);
        for (int i = 0; i < input.Length; i++) Assert.Equal(input[i], rt[i]);
    }
}
