using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// IQ3-family dequantization kernels. The 3-bit IQ formats (IQ3_XXS / IQ3_S)
/// encode 4 elements at a time by looking up a small per-codeword grid (256 /
/// 512 entries × 4 unsigned bytes) and flipping per-element signs from the
/// shared 128-entry ksigns table (the same one used by IQ2_XXS / IQ2_XS).
/// All layouts share <c>QK_K = 256</c> elements per super-block. The on-disk
/// byte layouts mirror <c>block_iq3_xxs</c> and <c>block_iq3_s</c> in
/// <c>ggml-common.h</c>:
/// <code>
///   IQ3_XXS ( 98 B): half d; uint8 qs[QK_K/4];                                       // 3.0625 bpw
///                   // last (3*QK_K/8 - QK_K/4) = QK_K/8 = 32 bytes of qs[] hold the
///                   // per-32-element 4-bit scale + 4*7-bit sign indices packed into
///                   // a uint32 (scales_and_signs = qs + QK_K/4).
///   IQ3_S   (110 B): half d; uint8 qs[QK_K/4]; uint8 qh[QK_K/32];                    // 3.4375 bpw
///                   //         uint8 signs[QK_K/8]; uint8 scales[QK_K/64];
/// </code>
/// Per-pair-of-4 dequant:
///   <c>val[j] = db * grid[gridIdx*4 + j] * (signs &amp; (1&lt;&lt;j) ? -1 : +1)</c>.
/// IQ3 grid entries are stored as <c>uint32</c> in ggml; we expose them as a
/// flat <c>byte[]</c> of 4-byte tuples so the Vulkan codebook SSBO (same
/// pattern as <see cref="Iq2XxsGrid"/>) can read individual bytes.
/// </summary>
public static unsafe partial class Dequantize
{
    /// <summary>IQ3_XXS block size in bytes: 2(d) + 3*(QK_K/8) = 2 + 96 = 98 / 256 elements (3.0625 bpw).</summary>
    internal const int IQ3_XXS_BlockBytes = 98;

    /// <summary>IQ3_S block size in bytes:
    /// 2(d) + QK_K/4 (qs=64) + QK_K/32 (qh=8) + QK_K/8 (signs=32) + QK_K/64 (scales=4)
    /// = 2 + 64 + 8 + 32 + 4 = 110 / 256 elements (3.4375 bpw).</summary>
    internal const int IQ3_S_BlockBytes = 110;

    /// <summary>
    /// <c>iq3xxs_grid</c> from ggml-common.h — 256 entries × 4 unsigned bytes.
    /// Each entry packs 4 unsigned int8 grid points (little-endian within the
    /// 32-bit word). Byte values are in the set <c>{0x04, 0x0c, 0x14, 0x1c,
    /// 0x24, 0x2c, 0x34, 0x3e}</c>.
    /// </summary>
    internal static ReadOnlySpan<byte> Iq3XxsGrid => Iq3XxsGridData;

    private static readonly byte[] Iq3XxsGridData = BuildIq3GridBytes(Iq3XxsGridWords);

    /// <summary>
    /// <c>iq3s_grid</c> from ggml-common.h — 512 entries × 4 unsigned bytes.
    /// Each entry packs 4 unsigned int8 grid points (little-endian within the
    /// 32-bit word). Byte values are in the set <c>{0x01, 0x03, 0x05, 0x07,
    /// 0x09, 0x0b, 0x0d, 0x0f}</c>.
    /// </summary>
    internal static ReadOnlySpan<byte> Iq3SGrid => Iq3SGridData;

    private static readonly byte[] Iq3SGridData = BuildIq3GridBytes(Iq3SGridWords);

    private static byte[] BuildIq3GridBytes(ReadOnlySpan<uint> words)
    {
        var bytes = new byte[words.Length * 4];
        for (int i = 0; i < words.Length; i++)
        {
            uint w = words[i];
            bytes[i * 4 + 0] = (byte)(w & 0xff);
            bytes[i * 4 + 1] = (byte)((w >> 8) & 0xff);
            bytes[i * 4 + 2] = (byte)((w >> 16) & 0xff);
            bytes[i * 4 + 3] = (byte)((w >> 24) & 0xff);
        }
        return bytes;
    }

    private static ReadOnlySpan<uint> Iq3XxsGridWords =>
    [
        0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
        0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
        0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
        0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
        0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
        0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
        0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
        0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
        0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
        0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
        0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
        0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
        0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
        0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
        0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
        0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
        0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
        0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
        0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
        0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
        0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
        0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
        0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
        0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
        0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
        0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
        0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
        0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
        0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
        0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
        0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
        0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
    ];

    private static ReadOnlySpan<uint> Iq3SGridWords =>
    [
        0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
        0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
        0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
        0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
        0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
        0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
        0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
        0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
        0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
        0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
        0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
        0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
        0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
        0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
        0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
        0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
        0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
        0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
        0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
        0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
        0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
        0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
        0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
        0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
        0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
        0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
        0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
        0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
        0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
        0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
        0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
        0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
        0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
        0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
        0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
        0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
        0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
        0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
        0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
        0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
        0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
        0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
        0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
        0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
        0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
        0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
        0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
        0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
        0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
        0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
        0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
        0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
        0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
        0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
        0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
        0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
        0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
        0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
        0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
        0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
        0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
        0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
        0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
        0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
    ];

    /// <summary>
    /// Dequantizes IQ3_XXS. Block layout (98 bytes / 256 elements):
    /// <c>d(Half@0), qs[64]@2, scales_and_signs[32]@66 (8 uint32 LE)</c>.
    /// Per 32-element sub-block <c>ib32 in [0,8)</c>:
    /// <c>aux32 = scales_and_signs[ib32]</c>;
    /// <c>db = d * (0.5 + (aux32 &gt;&gt; 28)) * 0.5</c>.
    /// For each of 4 pair-of-4 groups <c>l</c>: two 8-bit grid indices from
    /// <c>qs[ib32*8 + 2*l]</c> / <c>qs[ib32*8 + 2*l + 1]</c> select 4-byte grid
    /// rows; the per-pair sign byte is <c>ksigns_iq2xs[(aux32 &gt;&gt; 7*l) &amp; 0x7f]</c>;
    /// elements 0..3 use the first grid, 4..7 use the second; sign bits 0..3 then 4..7
    /// of the sign byte flip each element.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeIQ3_XXS(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"IQ3_XXS element count must be a multiple of {KQuantGroupSize}, got {elementCount}",
                nameof(elementCount));

        long blockCount = elementCount / KQuantGroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;
        ReadOnlySpan<byte> grid = Iq3XxsGrid;
        ReadOnlySpan<byte> ksigns = KsignsIq2Xs;

        for (long b = 0; b < blockCount; b++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(blockBase);
            byte* qs = blockBase + 2;
            byte* scalesAndSigns = qs + KQuantGroupSize / 4;     // offset 66

            for (int ib32 = 0; ib32 < KQuantGroupSize / 32; ib32++)
            {
                uint aux32 = Unsafe.ReadUnaligned<uint>(scalesAndSigns + 4 * ib32);
                float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
                int qsBase = ib32 * 8;
                for (int l = 0; l < 4; l++)
                {
                    int g1 = qs[qsBase + 2 * l + 0];
                    int g2 = qs[qsBase + 2 * l + 1];
                    int signsIdx = (int)((aux32 >> (7 * l)) & 0x7f);
                    byte signs = ksigns[signsIdx];
                    int outOff = outIdx + ib32 * 32 + l * 8;
                    int gOff1 = g1 * 4;
                    int gOff2 = g2 * 4;
                    for (int j = 0; j < 4; j++)
                    {
                        float sign1 = (signs & (1 << (j + 0))) != 0 ? -1f : 1f;
                        float sign2 = (signs & (1 << (j + 4))) != 0 ? -1f : 1f;
                        dest[outOff + j + 0] = db * grid[gOff1 + j] * sign1;
                        dest[outOff + j + 4] = db * grid[gOff2 + j] * sign2;
                    }
                }
            }

            outIdx += KQuantGroupSize;
            blockBase += IQ3_XXS_BlockBytes;
        }
    }

    /// <summary>
    /// Dequantizes IQ3_S. Block layout (110 bytes / 256 elements):
    /// <c>d(Half@0), qs[64]@2, qh[8]@66, signs[32]@74, scales[4]@106</c>.
    /// 8 sub-blocks of 32 elements; sub-blocks come in pairs sharing a
    /// <c>scales[ib32/2]</c> byte (low nibble = first sub, high = second) and a
    /// <c>qh[ib32]</c> byte (1 high-bit per grid index, 4 bits per sub-block).
    /// Per pair-of-4 <c>l</c> in <c>[0,4)</c>:
    /// <c>gridIdx = qs[2*l] | ((qh &gt;&gt; (l+0)) &amp; 1) &lt;&lt; 8</c> (and similar for
    /// the second grid). The 32-byte <c>signs[]</c> table stores a full 8-bit
    /// sign mask per pair (matches IQ2_S; no ksigns indirection).
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeIQ3_S(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"IQ3_S element count must be a multiple of {KQuantGroupSize}, got {elementCount}",
                nameof(elementCount));

        long blockCount = elementCount / KQuantGroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;
        ReadOnlySpan<byte> grid = Iq3SGrid;

        for (long b = 0; b < blockCount; b++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(blockBase);
            byte* qs = blockBase + 2;                             // 64 bytes
            byte* qh = qs + KQuantGroupSize / 4;                  // offset 66, 8 bytes
            byte* signs = qh + KQuantGroupSize / 32;              // offset 74, 32 bytes
            byte* scales = signs + KQuantGroupSize / 8;           // offset 106, 4 bytes

            // ggml processes sub-blocks two at a time so its inner loop layout
            // is unrolled below. The qs / signs / qh pointers advance per inner
            // iteration in ggml; we match that by carrying running offsets.
            int qsOff = 0;
            int signsOff = 0;
            int qhOff = 0;

            for (int ib32 = 0; ib32 < KQuantGroupSize / 32; ib32 += 2)
            {
                byte scaleByte = scales[ib32 / 2];
                float db1 = d * (1 + 2 * (scaleByte & 0xf));
                float db2 = d * (1 + 2 * (scaleByte >> 4));

                // ── First sub-block of the pair (ib32). ──
                byte qh0 = qh[qhOff + 0];
                for (int l = 0; l < 4; l++)
                {
                    // ggml: (qh[0] << (8 - 2*l)) & 256 picks bit (2*l) of qh0 -> 0x100 if set.
                    int g1 = qs[qsOff + 2 * l + 0] | (((qh0 << (8 - 2 * l)) & 0x100));
                    int g2 = qs[qsOff + 2 * l + 1] | (((qh0 << (7 - 2 * l)) & 0x100));
                    byte signMask = signs[signsOff + l];
                    int outOff = outIdx + ib32 * 32 + l * 8;
                    int gOff1 = g1 * 4;
                    int gOff2 = g2 * 4;
                    for (int j = 0; j < 4; j++)
                    {
                        float sign1 = (signMask & (1 << (j + 0))) != 0 ? -1f : 1f;
                        float sign2 = (signMask & (1 << (j + 4))) != 0 ? -1f : 1f;
                        dest[outOff + j + 0] = db1 * grid[gOff1 + j] * sign1;
                        dest[outOff + j + 4] = db1 * grid[gOff2 + j] * sign2;
                    }
                }
                qsOff += 8;
                signsOff += 4;

                // ── Second sub-block (ib32 + 1). ──
                byte qh1 = qh[qhOff + 1];
                for (int l = 0; l < 4; l++)
                {
                    int g1 = qs[qsOff + 2 * l + 0] | (((qh1 << (8 - 2 * l)) & 0x100));
                    int g2 = qs[qsOff + 2 * l + 1] | (((qh1 << (7 - 2 * l)) & 0x100));
                    byte signMask = signs[signsOff + l];
                    int outOff = outIdx + (ib32 + 1) * 32 + l * 8;
                    int gOff1 = g1 * 4;
                    int gOff2 = g2 * 4;
                    for (int j = 0; j < 4; j++)
                    {
                        float sign1 = (signMask & (1 << (j + 0))) != 0 ? -1f : 1f;
                        float sign2 = (signMask & (1 << (j + 4))) != 0 ? -1f : 1f;
                        dest[outOff + j + 0] = db2 * grid[gOff1 + j] * sign1;
                        dest[outOff + j + 4] = db2 * grid[gOff2 + j] * sign2;
                    }
                }
                qsOff += 8;
                signsOff += 4;
                qhOff += 2;
            }

            outIdx += KQuantGroupSize;
            blockBase += IQ3_S_BlockBytes;
        }
    }
}
