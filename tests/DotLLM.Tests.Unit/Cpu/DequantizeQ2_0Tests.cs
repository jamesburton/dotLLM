using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu;

/// <summary>Upstream ggml Q2_0 (#823): 64-element blocks, fp16 d + 16 code bytes, value = (code - 1) * d.</summary>
public sealed unsafe class DequantizeQ2_0Tests
{
    [Fact]
    public void Dequantize_MatchesReferenceFormula_AcrossBlocks()
    {
        const int blocks = 3;
        var rng = new Random(42);
        byte[] data = new byte[blocks * 18];
        rng.NextBytes(data);
        float[] d = [0.0625f, 0.5f, 0.0123f];
        for (int b = 0; b < blocks; b++)
        {
            ushort bits = BitConverter.HalfToUInt16Bits((Half)d[b]);
            data[b * 18] = (byte)bits;
            data[b * 18 + 1] = (byte)(bits >> 8);
        }
        var got = new float[blocks * 64];
        fixed (byte* p = data) Dequantize.ToFloat32((nint)p, got.Length, QuantizationType.Q2_0, got);

        for (int b = 0; b < blocks; b++)
            for (int j = 0; j < 64; j++)
            {
                int code = (data[b * 18 + 2 + j / 4] >> (2 * (j % 4))) & 3;
                float expect = (code - 1) * (float)(Half)d[b];
                Assert.Equal(expect, got[b * 64 + j]);
            }
        Assert.Equal(blocks * 18, Dequantize.RowByteSize(blocks * 64, QuantizationType.Q2_0));
        Assert.Equal(blocks * 18, QuantizationType.Q2_0.ComputeByteCount(blocks * 64));
    }

    [Fact]
    public void CodeThree_IsPlusTwoScales()
    {
        byte[] data = new byte[18];
        ushort bits = BitConverter.HalfToUInt16Bits((Half)1.0f);
        data[0] = (byte)bits; data[1] = (byte)(bits >> 8);
        data[2] = 0b11_10_01_00;   // elements 0..3 = codes 0,1,2,3
        var got = new float[64];
        fixed (byte* p = data) Dequantize.ToFloat32((nint)p, 64, QuantizationType.Q2_0, got);
        Assert.Equal([-1f, 0f, 1f, 2f], got[..4]);
    }
}
