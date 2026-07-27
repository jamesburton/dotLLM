using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// K-quant quantization kernels (Q4_K) — the inverse of <see cref="Dequantize.DequantizeQ4_KScalar"/>.
/// Mirrors llama.cpp's <c>quantize_row_q4_K_reference</c> (ggml-quants.c): per 256-element
/// super-block, 8 sub-blocks of 32 each get an unsigned scale + min, the super-block stores a
/// single fp16 <c>d</c> (scaling the 6-bit scales) and fp16 <c>dmin</c> (scaling the 6-bit mins),
/// then 8 6-bit scales + 8 6-bit mins packed into 12 bytes and 256 4-bit quants into 128 bytes.
/// Emits bytes that <see cref="Dequantize.DequantizeQ4_K"/> reads back bit-for-bit.
/// </summary>
public static unsafe partial class Quantize
{
    // ──────────────────── Q4_K ────────────────────
    // Layout (144 bytes / 256 elems): d(Half@0), dmin(Half@2), scales[12]@4, qs[128]@16.
    // Decode (per sub-block j, element i): val = d*sc[j]*nibble - dmin*mn[j].
    // qs packing matches the dequant: sub-block pairs (j even = low nibble of 32 bytes,
    // j odd = high nibble of the SAME 32 bytes), with elements emitted in sub-block order.
    private static void QuantizeQ4_K(ReadOnlySpan<float> src, long elementCount, Span<byte> dest)
    {
        long blocks = elementCount / KQuantGroupSize;
        fixed (byte* dpBase = dest)
        {
            byte* dp = dpBase;
            int idx = 0;

            // Per super-block working buffers (8 sub-blocks).
            float* subScale = stackalloc float[8];
            float* subMin = stackalloc float[8];
            byte* scQuant = stackalloc byte[8]; // 6-bit unsigned scale ids
            byte* mnQuant = stackalloc byte[8]; // 6-bit unsigned min ids
            byte* nibbles = stackalloc byte[256];

            for (long b = 0; b < blocks; b++)
            {
                // 1) Per sub-block: best (scale, min) so that x ≈ scale*q + min, q∈[0,15].
                float maxScale = 0f, maxMin = 0f;
                for (int j = 0; j < 8; j++)
                {
                    MakeQkx2(src, idx + j * 32, out float sc, out float mn);
                    subScale[j] = sc;
                    subMin[j] = mn;     // mn is the *negative offset* magnitude (>= 0): x ≈ sc*q - mn
                    if (sc > maxScale) maxScale = sc;
                    if (mn > maxMin) maxMin = mn;
                }

                // 2) Super-block d / dmin quantize the 8 scales / mins to 6-bit unsigned.
                float invScale = maxScale > 0f ? 63f / maxScale : 0f;
                float invMin = maxMin > 0f ? 63f / maxMin : 0f;
                for (int j = 0; j < 8; j++)
                {
                    int ls = (int)MathF.Round(subScale[j] * invScale);
                    int lm = (int)MathF.Round(subMin[j] * invMin);
                    if (ls > 63) ls = 63; if (ls < 0) ls = 0;
                    if (lm > 63) lm = 63; if (lm < 0) lm = 0;
                    scQuant[j] = (byte)ls;
                    mnQuant[j] = (byte)lm;
                }
                float d = maxScale > 0f ? maxScale / 63f : 0f;
                float dmin = maxMin > 0f ? maxMin / 63f : 0f;

                // 3) Re-quantize each element to its 4-bit nibble using the QUANTIZED
                //    scale/min (so encode error matches decode exactly).
                for (int j = 0; j < 8; j++)
                {
                    float dsc = d * scQuant[j];
                    float dmn = dmin * mnQuant[j];
                    float invd = dsc > 0f ? 1f / dsc : 0f;
                    for (int i = 0; i < 32; i++)
                    {
                        // decode: val = dsc*nib - dmn  =>  nib = (val + dmn)/dsc
                        int nib = (int)MathF.Round((src[idx + j * 32 + i] + dmn) * invd);
                        if (nib > 15) nib = 15; if (nib < 0) nib = 0;
                        nibbles[j * 32 + i] = (byte)nib;
                    }
                }

                // 4) Emit: d, dmin, packed scales, packed qs.
                Unsafe.WriteUnaligned(dp, (Half)d);
                Unsafe.WriteUnaligned(dp + 2, (Half)dmin);
                PackQ4Q5Scales(scQuant, mnQuant, dp + 4);

                // qs packing: for each pair (sb even, sb+1 odd) sharing 32 bytes,
                // even sub-block → low nibble, odd sub-block → high nibble. The
                // dequant reads pairIdx = j/2, nibbleHalf = j%2, qsByte = pairIdx*32 + i.
                byte* qs = dp + 16;
                for (int pair = 0; pair < 4; pair++)
                {
                    int lowSb = pair * 2;      // even sub-block
                    int highSb = pair * 2 + 1; // odd sub-block
                    for (int i = 0; i < 32; i++)
                    {
                        int lo = nibbles[lowSb * 32 + i] & 0xF;
                        int hi = nibbles[highSb * 32 + i] & 0xF;
                        qs[pair * 32 + i] = (byte)(lo | (hi << 4));
                    }
                }

                idx += KQuantGroupSize;
                dp += Q4_K_BlockBytes;
            }
        }
    }

    /// <summary>
    /// Finds a non-negative scale and min for a 32-element block so that
    /// <c>x ≈ scale*q - min</c> with <c>q ∈ [0,15]</c>. Returns <paramref name="min"/> as the
    /// magnitude of the negative offset (>= 0), matching the Q4_K decode
    /// <c>val = d*sc*nib - dmin*mn</c>. Simplified affine fit (llama.cpp uses an iterative
    /// <c>make_qkx2_quants</c>; this plain min/max fit is deterministic and within K-quant
    /// tolerance for the fixture's purposes).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MakeQkx2(ReadOnlySpan<float> src, int offset, out float scale, out float min)
    {
        float vmin = float.MaxValue, vmax = float.MinValue;
        for (int i = 0; i < 32; i++)
        {
            float x = src[offset + i];
            if (x < vmin) vmin = x;
            if (x > vmax) vmax = x;
        }
        // Q4_K mins are unsigned (dmin >= 0, mn >= 0) and SUBTRACTED, so the affine
        // floor must be <= 0 (decode can only reach values >= -dmin*mn). Clamp vmin to 0.
        if (vmin > 0f) vmin = 0f;
        scale = (vmax - vmin) / 15f;
        min = -vmin; // >= 0
    }

    /// <summary>
    /// Inverse of <see cref="Dequantize.UnpackQ4Q5Scales"/>: packs 8 6-bit scales + 8 6-bit
    /// mins into 12 bytes, matching llama.cpp <c>get_scale_min_k4</c>'s read pattern.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void PackQ4Q5Scales(byte* scales8, byte* mins8, byte* out12)
    {
        for (int j = 0; j < 12; j++) out12[j] = 0;

        // Sub-blocks 0-3: scale low 6 bits in bytes 0-3, min low 6 bits in bytes 4-7.
        for (int j = 0; j < 4; j++)
        {
            out12[j] = (byte)(scales8[j] & 63);
            out12[j + 4] = (byte)(mins8[j] & 63);
        }
        // Sub-blocks 4-7: low nibble of scale & min in bytes 8-11; the high 2 bits of
        // each go into the top 2 bits of bytes 0-7 (scales→0-3, mins→4-7).
        for (int j = 4; j < 8; j++)
        {
            int sc = scales8[j] & 63;
            int mn = mins8[j] & 63;
            out12[j + 4] = (byte)((sc & 0xF) | ((mn & 0xF) << 4));
            out12[j - 4] |= (byte)(((sc >> 4) & 0x3) << 6);
            out12[j] |= (byte)(((mn >> 4) & 0x3) << 6);
        }
    }
}
