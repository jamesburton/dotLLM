using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Quantization kernels: convert float32 rows into the on-disk block layouts that
/// <see cref="Dequantize"/> reads back. These are the inverse of the dequantizers and
/// are used to synthesize GGUF fixtures (see
/// <c>DotLLM.Models.Gguf.SyntheticGemma4Gguf</c>) — every block they emit is
/// <b>bit-compatible</b> with <see cref="Dequantize.ToFloat32"/>, verified by round-trip
/// unit tests. Correctness (exact layout + reasonable quant error) is the contract; these
/// are not on the inference hot path, so the scalar reference implementations are used.
/// </summary>
public static unsafe partial class Quantize
{
    /// <summary>Group size shared by Q8_0/Q5_0/Q5_1 (32 elements per block).</summary>
    private const int GroupSize32 = 32;

    /// <summary>Q8_0 block size in bytes: 2 (Half d) + 32 (sbyte qs).</summary>
    private const int Q8_0BlockBytes = QuantFormat.Q8_0BlockBytes;

    /// <summary>Q5_0 block size in bytes: 2 (Half d) + 4 (qh) + 16 (qs).</summary>
    private const int Q5_0BlockBytes = QuantFormat.Q5_0BlockBytes;

    /// <summary>Q5_1 block size in bytes: 2 (Half d) + 2 (Half m) + 4 (qh) + 16 (qs).</summary>
    private const int Q5_1BlockBytes = QuantFormat.Q5_1BlockBytes;

    /// <summary>Q4_K super-block size in bytes: 2(d)+2(dmin)+12(scales)+128(qs).</summary>
    private const int Q4_K_BlockBytes = QuantFormat.Q4_KBlockBytes;

    /// <summary>Q4_K super-block element count.</summary>
    private const int KQuantGroupSize = QuantFormat.KQuantGroupSize;

    /// <summary>
    /// Returns the number of bytes one quantized row of <paramref name="elementCount"/>
    /// elements occupies in <paramref name="quantType"/> — identical to
    /// <see cref="Dequantize.RowByteSize(long, QuantizationType)"/>, re-exposed here so
    /// callers can size destination buffers without referencing the dequant side.
    /// </summary>
    public static long RowByteSize(long elementCount, QuantizationType quantType)
        => Dequantize.RowByteSize(elementCount, quantType);

    /// <summary>
    /// Quantizes <paramref name="src"/> (length <paramref name="elementCount"/>) into
    /// <paramref name="dest"/> using <paramref name="quantType"/>. <paramref name="dest"/>
    /// must be at least <see cref="RowByteSize"/> bytes. The element count must be a
    /// multiple of the format's block size (32 for Q8_0/Q5_0/Q5_1, 256 for Q4_K).
    /// </summary>
    public static void FromFloat32(ReadOnlySpan<float> src, long elementCount,
        QuantizationType quantType, Span<byte> dest)
    {
        long need = RowByteSize(elementCount, quantType);
        if (dest.Length < need)
            throw new ArgumentException($"Destination too small: {dest.Length} < {need}", nameof(dest));
        if (src.Length < elementCount)
            throw new ArgumentException($"Source too small: {src.Length} < {elementCount}", nameof(src));

        switch (quantType)
        {
            case QuantizationType.F32:
                QuantizeF32(src, elementCount, dest);
                break;
            case QuantizationType.F16:
                QuantizeF16(src, elementCount, dest);
                break;
            case QuantizationType.Q8_0:
                QuantizeQ8_0(src, elementCount, dest);
                break;
            case QuantizationType.Q5_0:
                QuantizeQ5_0(src, elementCount, dest);
                break;
            case QuantizationType.Q5_1:
                QuantizeQ5_1(src, elementCount, dest);
                break;
            case QuantizationType.Q4_K:
                QuantizeQ4_K(src, elementCount, dest);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(quantType), quantType,
                    $"Quantize does not support {quantType}.");
        }
    }

    /// <summary>Allocates a byte[] of the exact quantized row size and quantizes into it.</summary>
    public static byte[] FromFloat32(ReadOnlySpan<float> src, long elementCount, QuantizationType quantType)
    {
        var buf = new byte[RowByteSize(elementCount, quantType)];
        FromFloat32(src, elementCount, quantType, buf);
        return buf;
    }

    private static void QuantizeF32(ReadOnlySpan<float> src, long elementCount, Span<byte> dest)
    {
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(src[..(int)elementCount]);
        bytes.CopyTo(dest);
    }

    private static void QuantizeF16(ReadOnlySpan<float> src, long elementCount, Span<byte> dest)
    {
        fixed (byte* dp = dest)
        {
            var half = (Half*)dp;
            for (int i = 0; i < elementCount; i++)
                half[i] = (Half)src[i];
        }
    }

    // ──────────────────── Q8_0 ────────────────────
    // Block (34 bytes / 32 elems): d(Half@0) + qs[32](sbyte)@2.
    // Mirrors ggml quantize_row_q8_0: d = max(|x|)/127, qs = round(x / d).
    private static void QuantizeQ8_0(ReadOnlySpan<float> src, long elementCount, Span<byte> dest)
    {
        long blocks = elementCount / GroupSize32;
        fixed (byte* dpBase = dest)
        {
            byte* dp = dpBase;
            int idx = 0;
            for (long b = 0; b < blocks; b++)
            {
                float amax = 0f;
                for (int i = 0; i < GroupSize32; i++)
                {
                    float a = MathF.Abs(src[idx + i]);
                    if (a > amax) amax = a;
                }
                float d = amax / 127f;
                float id = d != 0f ? 1f / d : 0f;

                Unsafe.WriteUnaligned(dp, (Half)d);
                sbyte* qs = (sbyte*)(dp + 2);
                for (int i = 0; i < GroupSize32; i++)
                {
                    int q = (int)MathF.Round(src[idx + i] * id);
                    if (q > 127) q = 127;
                    if (q < -127) q = -127; // ggml clamps symmetrically (sbyte min -128 unused)
                    qs[i] = (sbyte)q;
                }

                idx += GroupSize32;
                dp += Q8_0BlockBytes;
            }
        }
    }

    // ──────────────────── Q5_0 ────────────────────
    // Block (22 bytes / 32 elems): d(Half@0) + qh[4](uint32)@2 + qs[16]@6.
    // Dequant: value = d * ((qh_bit<<4 | nibble) - 16). 5-bit signed-around-16.
    // Low nibbles → elements 0..15, high nibbles → 16..31 (matches dequant ordering).
    private static void QuantizeQ5_0(ReadOnlySpan<float> src, long elementCount, Span<byte> dest)
    {
        long blocks = elementCount / GroupSize32;
        fixed (byte* dpBase = dest)
        {
            byte* dp = dpBase;
            int idx = 0;
            for (long b = 0; b < blocks; b++)
            {
                // ggml quantize_row_q5_0: scale by the most extreme value (signed),
                // d = max|x| / -16  -> range [-16, 15].
                float amax = 0f, max = 0f;
                for (int i = 0; i < GroupSize32; i++)
                {
                    float x = src[idx + i];
                    float a = MathF.Abs(x);
                    if (a > amax) { amax = a; max = x; }
                }
                float d = max / -16f;
                float id = d != 0f ? 1f / d : 0f;

                Unsafe.WriteUnaligned(dp, (Half)d);
                byte* qs = dp + 6;
                uint qh = 0;
                for (int j = 0; j < 16; j++)
                {
                    float x0 = src[idx + j];
                    float x1 = src[idx + j + 16];
                    int q0 = (int)(x0 * id + 16.5f);
                    int q1 = (int)(x1 * id + 16.5f);
                    if (q0 > 31) q0 = 31; if (q0 < 0) q0 = 0;
                    if (q1 > 31) q1 = 31; if (q1 < 0) q1 = 0;
                    qs[j] = (byte)((q0 & 0xF) | ((q1 & 0xF) << 4));
                    qh |= (uint)((q0 >> 4) & 1) << j;
                    qh |= (uint)((q1 >> 4) & 1) << (j + 16);
                }
                Unsafe.WriteUnaligned(dp + 2, qh);

                idx += GroupSize32;
                dp += Q5_0BlockBytes;
            }
        }
    }

    // ──────────────────── Q5_1 ────────────────────
    // Block (24 bytes / 32 elems): d(Half@0) + m(Half@2) + qh[4](uint32)@4 + qs[16]@8.
    // Dequant: value = d * (qh_bit<<4 | nibble) + m. 5-bit unsigned + min.
    private static void QuantizeQ5_1(ReadOnlySpan<float> src, long elementCount, Span<byte> dest)
    {
        long blocks = elementCount / GroupSize32;
        fixed (byte* dpBase = dest)
        {
            byte* dp = dpBase;
            int idx = 0;
            for (long b = 0; b < blocks; b++)
            {
                float min = float.MaxValue, max = float.MinValue;
                for (int i = 0; i < GroupSize32; i++)
                {
                    float x = src[idx + i];
                    if (x < min) min = x;
                    if (x > max) max = x;
                }
                float d = (max - min) / 31f; // 5-bit range [0,31]
                float id = d != 0f ? 1f / d : 0f;

                Unsafe.WriteUnaligned(dp, (Half)d);
                Unsafe.WriteUnaligned(dp + 2, (Half)min);
                byte* qs = dp + 8;
                uint qh = 0;
                for (int j = 0; j < 16; j++)
                {
                    int q0 = (int)((src[idx + j] - min) * id + 0.5f);
                    int q1 = (int)((src[idx + j + 16] - min) * id + 0.5f);
                    if (q0 > 31) q0 = 31; if (q0 < 0) q0 = 0;
                    if (q1 > 31) q1 = 31; if (q1 < 0) q1 = 0;
                    qs[j] = (byte)((q0 & 0xF) | ((q1 & 0xF) << 4));
                    qh |= (uint)((q0 >> 4) & 1) << j;
                    qh |= (uint)((q1 >> 4) & 1) << (j + 16);
                }
                Unsafe.WriteUnaligned(dp + 4, qh);

                idx += GroupSize32;
                dp += Q5_1BlockBytes;
            }
        }
    }
}
