using System;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Verifies BitNet I2_S quantization (f32 → ternary I2_S) round-trips through the production
/// <see cref="Dequantize.DequantizeI2_S"/> kernel, matching transformers' absmean WeightQuant
/// semantics (scale = mean(|w|); q = clamp(round(w/scale), -1, 1); dequant = q·scale) and the
/// exact interleaved 128-element block bit-packing (byte gp holds elements {gp,+32,+64,+96}).
/// </summary>
public sealed unsafe class BitNetQuantizeI2STests
{
    [Fact]
    public void QuantizeToI2S_RoundTripsThroughDequantize_MatchesAbsmeanTernary()
    {
        const int count = 256; // 2 I2_S blocks
        var rng = new Random(unchecked((int)0xB17BE7));
        var w = new float[count];
        for (int i = 0; i < count; i++) w[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        // Reference: transformers BitNet WeightQuant (per-tensor absmean).
        double mean = 0;
        for (int i = 0; i < count; i++) mean += Math.Abs(w[i]);
        mean /= count;
        float scale = (float)Math.Max(mean, 1e-5);
        var expected = new float[count];
        for (int i = 0; i < count; i++)
            expected[i] = Math.Clamp((float)Math.Round(w[i] / scale), -1f, 1f) * scale;

        var packed = new byte[count / 4 + sizeof(float)];
        BitNetQuantize.QuantizeToI2S(w, count, packed);

        var got = new float[count];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, count, QuantizationType.I2_S, got);

        for (int i = 0; i < count; i++)
            Assert.Equal(expected[i], got[i], 4);
    }

    [Fact]
    public void QuantizeToI2S_PacksElementsAtCorrectInterleavedPositions()
    {
        // One 128-element block. The kernel decodes byte 0 → elements {0,+32,+64,+96}
        // at bit offsets {6,4,2,0}; this test pins that interleaving.
        const int count = 128;
        var w = new float[count];
        w[0] = 2.0f;   // → +1 ternary
        w[32] = -2.0f; // → -1 ternary
        w[96] = 2.0f;  // → +1 ternary
        float scale = 6f / 128f; // mean(|w|) = (2+2+2)/128

        var packed = new byte[count / 4 + sizeof(float)];
        BitNetQuantize.QuantizeToI2S(w, count, packed);

        var got = new float[count];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, count, QuantizationType.I2_S, got);

        Assert.Equal(scale, got[0], 5);
        Assert.Equal(-scale, got[32], 5);
        Assert.Equal(scale, got[96], 5);
        Assert.Equal(0f, got[1], 5);
        Assert.Equal(0f, got[64], 5);
    }
}
