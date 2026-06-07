using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// IQ-family dequantization kernels. IQ formats use table lookups rather than
/// linear signed integer decode.
/// </summary>
public static unsafe partial class Dequantize
{
    /// <summary>IQ4_NL block size in bytes: 2(d) + 16(qs) = 18.</summary>
    internal const int IQ4_NL_BlockBytes = 18;

    /// <summary>IQ4_XS block size in bytes: 2(d) + 2(scales_h) + 4(scales_l) + 128(qs) = 136.</summary>
    internal const int IQ4_XS_BlockBytes = 136;

    /// <summary>Number of elements per IQ4_NL block.</summary>
    internal const int IQ4_NL_GroupSize = 32;

    /// <summary>
    /// Non-linear signed lookup shared by IQ4_NL and IQ4_XS. Values match
    /// ggml's <c>kvalues_iq4nl</c> table.
    /// </summary>
    internal static ReadOnlySpan<sbyte> KValuesIq4Nl =>
    [
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    ];

    /// <summary>
    /// Dequantizes IQ4_NL. Layout per 32-value block:
    /// <c>d(Half@0), qs[16]@2</c>. Low nibbles produce values 0..15,
    /// high nibbles produce values 16..31.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeIQ4_NL(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % IQ4_NL_GroupSize != 0)
            throw new ArgumentException(
                $"IQ4_NL element count must be a multiple of {IQ4_NL_GroupSize}, got {elementCount}",
                nameof(elementCount));

        long blockCount = elementCount / IQ4_NL_GroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;
        ReadOnlySpan<sbyte> lookup = KValuesIq4Nl;

        for (long b = 0; b < blockCount; b++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(blockBase);
            byte* qs = blockBase + 2;

            for (int j = 0; j < IQ4_NL_GroupSize / 2; j++)
            {
                byte q = qs[j];
                dest[outIdx + j] = d * lookup[q & 0x0F];
                dest[outIdx + j + IQ4_NL_GroupSize / 2] = d * lookup[q >> 4];
            }

            outIdx += IQ4_NL_GroupSize;
            blockBase += IQ4_NL_BlockBytes;
        }
    }

    /// <summary>
    /// Dequantizes IQ4_XS. Layout per 256-value super-block:
    /// <c>d(Half@0), scales_h(uint16@2), scales_l[4]@4, qs[128]@8</c>.
    /// Each 32-value sub-block uses a 6-bit scale encoded as 4 low bits in
    /// <c>scales_l</c> plus 2 high bits in <c>scales_h</c>, biased by -32.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeIQ4_XS(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"IQ4_XS element count must be a multiple of {KQuantGroupSize}, got {elementCount}",
                nameof(elementCount));

        long blockCount = elementCount / KQuantGroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;
        ReadOnlySpan<sbyte> lookup = KValuesIq4Nl;

        for (long b = 0; b < blockCount; b++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(blockBase);
            ushort scalesH = Unsafe.ReadUnaligned<ushort>(blockBase + 2);
            byte* scalesL = blockBase + 4;
            byte* qs = blockBase + 8;

            for (int ib = 0; ib < KQuantGroupSize / IQ4_NL_GroupSize; ib++)
            {
                int low = (scalesL[ib / 2] >> (4 * (ib % 2))) & 0x0F;
                int high = (scalesH >> (2 * ib)) & 0x03;
                int ls = low | (high << 4);
                float dl = d * (ls - 32);

                int subOut = outIdx + ib * IQ4_NL_GroupSize;
                byte* subQs = qs + ib * 16;
                for (int j = 0; j < IQ4_NL_GroupSize / 2; j++)
                {
                    byte q = subQs[j];
                    dest[subOut + j] = dl * lookup[q & 0x0F];
                    dest[subOut + j + IQ4_NL_GroupSize / 2] = dl * lookup[q >> 4];
                }
            }

            outIdx += KQuantGroupSize;
            blockBase += IQ4_XS_BlockBytes;
        }
    }
}
