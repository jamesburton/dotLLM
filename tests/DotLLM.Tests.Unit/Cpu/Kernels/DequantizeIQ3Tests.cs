using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Bit-perfect oracle parity for the CPU IQ3_XXS / IQ3_S dequant. The hand-coded
/// references inlined here mirror ggml-quants.c <c>dequantize_row_iq3_xxs</c> /
/// <c>dequantize_row_iq3_s</c>; the production kernels must match them bit-for-bit.
/// </summary>
public sealed unsafe class DequantizeIQ3Tests
{
    private const int IQ3_XXS_BlockBytes = 98;
    private const int IQ3_S_BlockBytes = 110;
    private const int KQuantGroupSize = 256;

    // ─────────────────────── IQ3_XXS ───────────────────────

    [Fact]
    public void IQ3_XXS_CpuMatchesScalarReference()
    {
        var rng = new Random(unchecked((int)0xC0FFEE_99));
        const int SuperBlocks = 4;
        const int Elem = SuperBlocks * KQuantGroupSize;
        byte[] packed = new byte[SuperBlocks * IQ3_XXS_BlockBytes];
        rng.NextBytes(packed);
        // Place a sane Half scale per block — random Half bits may produce NaN.
        fixed (byte* pBase = packed)
        {
            for (int sb = 0; sb < SuperBlocks; sb++)
                *(Half*)(pBase + sb * IQ3_XXS_BlockBytes) = (Half)0.01f;
        }

        float[] production = new float[Elem];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, Elem, QuantizationType.IQ3_XXS, production);

        float[] reference = ReferenceDequantIQ3_XXS(packed, SuperBlocks);
        AssertExactlyEqual("IQ3_XXS", reference, production);
    }

    /// <summary>
    /// Hand-constructed IQ3_XXS block:
    ///   qs[ib32*8 + 2*l + (0|1)] = 0 -> grid[0] = {0x04, 0x04, 0x04, 0x04};
    ///   aux32 = 0 -> all signs positive, sub-scale s4=0, so
    ///   db = d * (0.5 + 0) * 0.5 = d * 0.25;
    /// With d = 1.0, every element should equal 4 * 0.25 = 1.0.
    /// </summary>
    [Fact]
    public void IQ3_XXS_HandConstructedBlock_AllOnes()
    {
        byte[] packed = new byte[IQ3_XXS_BlockBytes];
        fixed (byte* p = packed)
        {
            *(Half*)(p + 0) = (Half)1.0f;
            // qs[0..63] and scales_and_signs[0..31] all zero from new byte[].
        }
        float[] dequant = new float[KQuantGroupSize];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, KQuantGroupSize, QuantizationType.IQ3_XXS, dequant);

        // d in Half precision -> 1.0 -> exact
        float expected = (float)(Half)1.0f * 0.25f * 4.0f;  // db * grid_byte = 0.25 * 4
        for (int i = 0; i < KQuantGroupSize; i++)
            Assert.Equal(expected, dequant[i]);
    }

    // ─────────────────────── IQ3_S ───────────────────────

    [Fact]
    public void IQ3_S_CpuMatchesScalarReference()
    {
        var rng = new Random(unchecked((int)0xDEAD_4321));
        const int SuperBlocks = 4;
        const int Elem = SuperBlocks * KQuantGroupSize;
        byte[] packed = new byte[SuperBlocks * IQ3_S_BlockBytes];
        rng.NextBytes(packed);
        fixed (byte* pBase = packed)
        {
            for (int sb = 0; sb < SuperBlocks; sb++)
                *(Half*)(pBase + sb * IQ3_S_BlockBytes) = (Half)0.01f;
        }

        float[] production = new float[Elem];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, Elem, QuantizationType.IQ3_S, production);

        float[] reference = ReferenceDequantIQ3_S(packed, SuperBlocks);
        AssertExactlyEqual("IQ3_S", reference, production);
    }

    /// <summary>
    /// Hand-constructed IQ3_S block:
    ///   qs[*] = 0, qh[*] = 0 -> grid[0] = {0x01, 0x01, 0x01, 0x01};
    ///   signs[*] = 0 -> all signs positive;
    ///   scales[*] = 0 -> sub-scale = 0 -> db = d * (1 + 2*0) = d.
    /// With d = 1.0, every element should equal 1.0 * 1 = 1.0.
    /// </summary>
    [Fact]
    public void IQ3_S_HandConstructedBlock_AllOnes()
    {
        byte[] packed = new byte[IQ3_S_BlockBytes];
        fixed (byte* p = packed)
        {
            *(Half*)(p + 0) = (Half)1.0f;
            // all other bytes zero.
        }
        float[] dequant = new float[KQuantGroupSize];
        fixed (byte* p = packed)
            Dequantize.ToFloat32((nint)p, KQuantGroupSize, QuantizationType.IQ3_S, dequant);

        float expected = (float)(Half)1.0f * 1.0f;
        for (int i = 0; i < KQuantGroupSize; i++)
            Assert.Equal(expected, dequant[i]);
    }

    // ───────────────────── References ─────────────────────

    /// <summary>
    /// Mirrors ggml-quants.c <c>dequantize_row_iq3_xxs</c> exactly. Uses the
    /// CPU oracle's grid + ksigns tables so any divergence is in the
    /// production kernel arithmetic, not in our codebook copy.
    /// </summary>
    private static float[] ReferenceDequantIQ3_XXS(byte[] src, int superBlocks)
    {
        float[] dst = new float[superBlocks * KQuantGroupSize];
        ReadOnlySpan<byte> grid = Dequantize.Iq3XxsGrid;
        ReadOnlySpan<byte> ksigns = Dequantize.KsignsIq2Xs;

        fixed (byte* pBase = src)
        {
            int outIdx = 0;
            for (int sb = 0; sb < superBlocks; sb++)
            {
                byte* blockBase = pBase + sb * IQ3_XXS_BlockBytes;
                float d = (float)*(Half*)blockBase;
                byte* qs = blockBase + 2;
                byte* scalesAndSigns = qs + KQuantGroupSize / 4;
                for (int ib32 = 0; ib32 < KQuantGroupSize / 32; ib32++)
                {
                    uint aux32 = *(uint*)(scalesAndSigns + 4 * ib32);
                    float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
                    int qsBase = ib32 * 8;
                    for (int l = 0; l < 4; l++)
                    {
                        int g1 = qs[qsBase + 2 * l + 0];
                        int g2 = qs[qsBase + 2 * l + 1];
                        byte signs = ksigns[(int)((aux32 >> (7 * l)) & 0x7f)];
                        int outOff = outIdx + ib32 * 32 + l * 8;
                        for (int j = 0; j < 4; j++)
                        {
                            float s0 = (signs & (1 << (j + 0))) != 0 ? -1f : 1f;
                            float s1 = (signs & (1 << (j + 4))) != 0 ? -1f : 1f;
                            dst[outOff + j + 0] = db * grid[g1 * 4 + j] * s0;
                            dst[outOff + j + 4] = db * grid[g2 * 4 + j] * s1;
                        }
                    }
                }
                outIdx += KQuantGroupSize;
            }
        }
        return dst;
    }

    /// <summary>
    /// Mirrors ggml-quants.c <c>dequantize_row_iq3_s</c> exactly. Sub-blocks
    /// come in pairs sharing a scales[ib32/2] byte and a qh byte.
    /// </summary>
    private static float[] ReferenceDequantIQ3_S(byte[] src, int superBlocks)
    {
        float[] dst = new float[superBlocks * KQuantGroupSize];
        ReadOnlySpan<byte> grid = Dequantize.Iq3SGrid;

        fixed (byte* pBase = src)
        {
            int outIdx = 0;
            for (int sb = 0; sb < superBlocks; sb++)
            {
                byte* blockBase = pBase + sb * IQ3_S_BlockBytes;
                float d = (float)*(Half*)blockBase;
                byte* qs = blockBase + 2;
                byte* qh = qs + KQuantGroupSize / 4;
                byte* signs = qh + KQuantGroupSize / 32;
                byte* scales = signs + KQuantGroupSize / 8;

                int qsOff = 0, signsOff = 0, qhOff = 0;
                for (int ib32 = 0; ib32 < KQuantGroupSize / 32; ib32 += 2)
                {
                    byte sByte = scales[ib32 / 2];
                    float db1 = d * (1 + 2 * (sByte & 0xf));
                    float db2 = d * (1 + 2 * (sByte >> 4));

                    byte qh0 = qh[qhOff + 0];
                    for (int l = 0; l < 4; l++)
                    {
                        int g1 = qs[qsOff + 2 * l + 0] | (((qh0 << (8 - 2 * l)) & 0x100));
                        int g2 = qs[qsOff + 2 * l + 1] | (((qh0 << (7 - 2 * l)) & 0x100));
                        byte mask = signs[signsOff + l];
                        int outOff = outIdx + ib32 * 32 + l * 8;
                        for (int j = 0; j < 4; j++)
                        {
                            float s0 = (mask & (1 << (j + 0))) != 0 ? -1f : 1f;
                            float s1 = (mask & (1 << (j + 4))) != 0 ? -1f : 1f;
                            dst[outOff + j + 0] = db1 * grid[g1 * 4 + j] * s0;
                            dst[outOff + j + 4] = db1 * grid[g2 * 4 + j] * s1;
                        }
                    }
                    qsOff += 8;
                    signsOff += 4;

                    byte qh1 = qh[qhOff + 1];
                    for (int l = 0; l < 4; l++)
                    {
                        int g1 = qs[qsOff + 2 * l + 0] | (((qh1 << (8 - 2 * l)) & 0x100));
                        int g2 = qs[qsOff + 2 * l + 1] | (((qh1 << (7 - 2 * l)) & 0x100));
                        byte mask = signs[signsOff + l];
                        int outOff = outIdx + (ib32 + 1) * 32 + l * 8;
                        for (int j = 0; j < 4; j++)
                        {
                            float s0 = (mask & (1 << (j + 0))) != 0 ? -1f : 1f;
                            float s1 = (mask & (1 << (j + 4))) != 0 ? -1f : 1f;
                            dst[outOff + j + 0] = db2 * grid[g1 * 4 + j] * s0;
                            dst[outOff + j + 4] = db2 * grid[g2 * 4 + j] * s1;
                        }
                    }
                    qsOff += 8;
                    signsOff += 4;
                    qhOff += 2;
                }
                outIdx += KQuantGroupSize;
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
