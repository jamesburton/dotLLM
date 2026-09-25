using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Bit-perfect oracle parity for the CPU IQ1_S dequant. The hand-coded reference
/// inlined here mirrors ggml-quants.c <c>dequantize_row_iq1_s</c>; the production
/// kernel must match it bit-for-bit.
/// </summary>
public sealed unsafe class DequantizeIQ1Tests
{
    private const int IQ1_S_BlockBytes = 50;
    private const int KQuantGroupSize = 256;
    private const float IQ1S_DELTA = 0.125f;

    [Fact]
    public void IQ1_S_CpuMatchesScalarReference()
    {
        var rng = new Random(unchecked((int)0xCAFE_F00D));
        const int SuperBlocks = 4;
        const int Elem = SuperBlocks * 256;
        byte[] packed = new byte[SuperBlocks * IQ1_S_BlockBytes];
        rng.NextBytes(packed);
        fixed (byte* pBase = packed)
        {
            for (int sb = 0; sb < SuperBlocks; sb++)
                *(Half*)(pBase + sb * IQ1_S_BlockBytes) = (Half)0.01f;
        }

        float[] production = new float[Elem];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, Elem, QuantizationType.IQ1_S, production);

        float[] reference = ReferenceDequantIQ1_S(packed, SuperBlocks);
        AssertExactlyEqual("IQ1_S", reference, production);
    }

    /// <summary>
    /// Hand-constructed IQ1_S block. With qs[0..3] = 0 and qh[0] = 0:
    ///   grid index = 0 -> grid[0] = 0xffffffffffffffff = 8 x int8(-1)
    ///   dl = d * (2*0 + 1) = d
    ///   delta = +IQ1S_DELTA = +0.125
    /// Per-element: y = d * (-1 + 0.125) = d * -0.875.
    /// With d = 1.0: y[0..7] = -0.875.
    /// </summary>
    [Fact]
    public void IQ1_S_HandConstructedBlock_Element0Matches()
    {
        byte[] packed = new byte[IQ1_S_BlockBytes];
        fixed (byte* p = packed)
        {
            *(Half*)(p + 0) = (Half)1.0f;
            // qs[0..31] all zero, qh[0..7] all zero  (already from new byte[])
        }
        float[] dequant = new float[256];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, 256, QuantizationType.IQ1_S, dequant);

        float expected = (float)((double)(Half)1.0f * (-1.0 + IQ1S_DELTA));
        for (int j = 0; j < 8; j++)
        {
            Assert.Equal(expected, dequant[j]);
        }
    }

    private static float[] ReferenceDequantIQ1_S(byte[] src, int superBlocks)
    {
        float[] dst = new float[superBlocks * 256];
        ReadOnlySpan<ulong> grid = Dequantize.Iq1SGrid;
        fixed (byte* pBase = src)
        {
            int outIdx = 0;
            for (int sb = 0; sb < superBlocks; sb++)
            {
                byte* blockBase = pBase + sb * IQ1_S_BlockBytes;
                float d = (float)*(Half*)blockBase;
                byte* qs = blockBase + 2;
                ushort* qh = (ushort*)(blockBase + 2 + 32);
                for (int ib = 0; ib < 8; ib++)
                {
                    ushort qhVal = qh[ib];
                    float dl = d * (2 * ((qhVal >> 12) & 7) + 1);
                    float delta = (qhVal & 0x8000) != 0 ? -IQ1S_DELTA : IQ1S_DELTA;
                    int qsBase = ib * 4;
                    for (int l = 0; l < 4; l++)
                    {
                        int idx = qs[qsBase + l] | (((qhVal >> (3 * l)) & 7) << 8);
                        ulong gridEntry = grid[idx];
                        int outOff = outIdx + ib * 32 + l * 8;
                        for (int j = 0; j < 8; j++)
                        {
                            sbyte g = (sbyte)((gridEntry >> (8 * j)) & 0xff);
                            dst[outOff + j] = dl * (g + delta);
                        }
                    }
                }
                outIdx += 256;
            }
        }
        return dst;
    }

    private static void AssertExactlyEqual(string label, float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        int mismatches = 0;
        for (int i = 0; i < expected.Length && mismatches < 8; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
            {
                mismatches++;
                Assert.Fail($"{label} mismatch at index {i}: expected={expected[i]:G9} (0x{BitConverter.SingleToInt32Bits(expected[i]):X8}) actual={actual[i]:G9} (0x{BitConverter.SingleToInt32Bits(actual[i]):X8})");
            }
        }
    }
}
