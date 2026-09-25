using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Dequantization kernels that convert quantized tensor data to float32.
/// Supports FP16, Q8_0, and F32 (passthrough). Used at model-load time to convert
/// memory-mapped GGUF tensor data into compute-ready float buffers.
/// </summary>
public static unsafe partial class Dequantize
{
    /// <summary>Q4_0 block size in bytes: 2 (Half scale) + 16 (packed nibble bytes).</summary>
    private const int Q4_0BlockBytes = QuantFormat.Q4_0BlockBytes;

    /// <summary>Q8_0 block size in bytes: 2 (Half scale) + 32 (sbyte quantized values).</summary>
    private const int Q8_0BlockBytes = QuantFormat.Q8_0BlockBytes;

    /// <summary>Number of elements per Q8_0 block.</summary>
    private const int Q8_0GroupSize = QuantFormat.LegacyGroupSize;

    /// <summary>Q5_0 block size in bytes: 2 (Half d) + 4 (qh) + 16 (qs) = 22.</summary>
    private const int Q5_0BlockBytes = QuantFormat.Q5_0BlockBytes;

    /// <summary>Number of elements per Q5_0 block.</summary>
    private const int Q5_0GroupSize = QuantFormat.LegacyGroupSize;

    /// <summary>Q4_1 block size in bytes: 2 (Half d) + 2 (Half m) + 16 (qs) = 20.</summary>
    private const int Q4_1BlockBytes = QuantFormat.Q4_1BlockBytes;

    /// <summary>Q5_1 block size in bytes: 2 (Half d) + 2 (Half m) + 4 (qh) + 16 (qs) = 24.</summary>
    private const int Q5_1BlockBytes = QuantFormat.Q5_1BlockBytes;

    /// <summary>Number of elements per I2_S block (x86 packing). 128 codes → 32 bytes.</summary>
    internal const int I2SBlockSize = 128;

    /// <summary>Number of elements per PQ2_0 group. Same 128-code packing as I2_S.</summary>
    internal const int PQ2_0GroupSize = QuantFormat.TernaryGroupSize;

    /// <summary>
    /// PQ2_0 group size in bytes: 2 (Half scale) + 32 (packed 2-bit codes, 4/byte) = 34.
    /// Unlike I2_S, the scale is PER GROUP (not per tensor) and comes BEFORE the codes
    /// (verified empirically against real Bonsai GGUF tensor bytes — see
    /// <see cref="QuantizationType.PQ2_0"/>'s doc comment for how this was confirmed).
    /// </summary>
    internal const int PQ2_0GroupBytes = QuantFormat.PQ2_0BlockBytes;

    /// <summary>MXFP4 block size in bytes: 1 (E8M0 scale) + 16 (packed nibble bytes) = 17.</summary>
    internal const int Mxfp4BlockBytes = QuantFormat.Mxfp4BlockBytes;

    /// <summary>Number of elements per MXFP4 block.</summary>
    internal const int Mxfp4GroupSize = QuantFormat.LegacyGroupSize;

    /// <summary>
    /// MXFP4 E2M1 value table, doubled (matches llama.cpp <c>kvalues_mxfp4</c>).
    /// Index = 4-bit code; values are 2× the nominal e2m1 value, compensated by
    /// halving the E8M0 block scale (<see cref="E8M0ToFloatHalf"/>).
    /// </summary>
    internal static ReadOnlySpan<sbyte> Mxfp4Values => [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    /// <summary>
    /// Returns the byte size of one row of <paramref name="elementCount"/> elements in the given quantization format.
    /// Useful for computing row strides when iterating weight matrices.
    /// </summary>
    public static long RowByteSize(long elementCount, QuantizationType quantType) => quantType switch
    {
        QuantizationType.F32 => elementCount * 4,
        QuantizationType.F16 => elementCount * 2,
        QuantizationType.BF16 => elementCount * 2,
        QuantizationType.Q4_0 => elementCount / Q8_0GroupSize * Q4_0BlockBytes,
        QuantizationType.Q4_1 => elementCount / Q8_0GroupSize * Q4_1BlockBytes,
        QuantizationType.Q8_0 => elementCount / Q8_0GroupSize * Q8_0BlockBytes,
        QuantizationType.Q5_0 => elementCount / Q5_0GroupSize * Q5_0BlockBytes,
        QuantizationType.Q5_1 => elementCount / Q5_0GroupSize * Q5_1BlockBytes,
        QuantizationType.IQ4_NL => elementCount / IQ4_NL_GroupSize * IQ4_NL_BlockBytes,
        QuantizationType.Q2_K => elementCount / KQuantGroupSize * Q2_K_BlockBytes,
        QuantizationType.Q3_K => elementCount / KQuantGroupSize * Q3_K_BlockBytes,
        QuantizationType.Q4_K => elementCount / KQuantGroupSize * Q4_K_BlockBytes,
        QuantizationType.Q5_K => elementCount / KQuantGroupSize * Q5_K_BlockBytes,
        QuantizationType.Q6_K => elementCount / KQuantGroupSize * Q6_K_BlockBytes,
        QuantizationType.IQ4_XS => elementCount / KQuantGroupSize * IQ4_XS_BlockBytes,
        QuantizationType.IQ2_XXS => elementCount / KQuantGroupSize * IQ2_XXS_BlockBytes,
        QuantizationType.IQ2_XS => elementCount / KQuantGroupSize * IQ2_XS_BlockBytes,
        QuantizationType.IQ2_S => elementCount / KQuantGroupSize * IQ2_S_BlockBytes,
        QuantizationType.IQ1_S => elementCount / KQuantGroupSize * IQ1_S_BlockBytes,
        QuantizationType.IQ3_XXS => elementCount / KQuantGroupSize * IQ3_XXS_BlockBytes,
        QuantizationType.IQ3_S => elementCount / KQuantGroupSize * IQ3_S_BlockBytes,
        // I2_S: packed 2-bit row stride only (k/4). The per-tensor scale lives at the tensor tail.
        QuantizationType.I2_S => elementCount / 4,
        QuantizationType.MXFP4 => elementCount / Mxfp4GroupSize * Mxfp4BlockBytes,
        // PQ2_0: scale is per-group (interleaved), so row stride includes it — contrast I2_S.
        QuantizationType.PQ2_0 => elementCount / PQ2_0GroupSize * PQ2_0GroupBytes,
        _ => throw new ArgumentOutOfRangeException(nameof(quantType), quantType,
            $"Unknown quantization type: {quantType}")
    };

    /// <summary>
    /// Converts quantized tensor data at <paramref name="src"/> to float32 in <paramref name="dest"/>.
    /// </summary>
    /// <param name="src">Pointer to the source tensor data (memory-mapped or allocated).</param>
    /// <param name="elementCount">Number of logical elements to dequantize.</param>
    /// <param name="quantType">Storage format of the source data.</param>
    /// <param name="dest">Destination span for float32 output. Must have length &gt;= <paramref name="elementCount"/>.</param>
    /// <exception cref="ArgumentOutOfRangeException">Unsupported quantization type.</exception>
    /// <exception cref="ArgumentException"><paramref name="dest"/> is too small.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ToFloat32(nint src, long elementCount, QuantizationType quantType, Span<float> dest)
    {
        if (dest.Length < elementCount)
            throw new ArgumentException($"Destination span too small: {dest.Length} < {elementCount}", nameof(dest));

        switch (quantType)
        {
            case QuantizationType.F32:
                DequantizeF32(src, elementCount, dest);
                break;
            case QuantizationType.F16:
                DequantizeFp16(src, elementCount, dest);
                break;
            case QuantizationType.BF16:
                DequantizeBf16(src, elementCount, dest);
                break;
            case QuantizationType.Q8_0:
                DequantizeQ8_0(src, elementCount, dest);
                break;
            case QuantizationType.Q4_0:
                DequantizeQ4_0Scalar(src, elementCount, dest);
                break;
            case QuantizationType.Q5_0:
                DequantizeQ5_0(src, elementCount, dest);
                break;
            case QuantizationType.Q4_1:
                DequantizeQ4_1Scalar(src, elementCount, dest);
                break;
            case QuantizationType.Q5_1:
                DequantizeQ5_1Scalar(src, elementCount, dest);
                break;
            case QuantizationType.IQ4_NL:
                DequantizeIQ4_NL(src, elementCount, dest);
                break;
            case QuantizationType.Q2_K:
                DequantizeQ2_K(src, elementCount, dest);
                break;
            case QuantizationType.Q3_K:
                DequantizeQ3_K(src, elementCount, dest);
                break;
            case QuantizationType.Q4_K:
                DequantizeQ4_K(src, elementCount, dest);
                break;
            case QuantizationType.Q5_K:
                DequantizeQ5_K(src, elementCount, dest);
                break;
            case QuantizationType.Q6_K:
                DequantizeQ6_K(src, elementCount, dest);
                break;
            case QuantizationType.IQ4_XS:
                DequantizeIQ4_XS(src, elementCount, dest);
                break;
            case QuantizationType.IQ2_XXS:
                DequantizeIQ2_XXS(src, elementCount, dest);
                break;
            case QuantizationType.IQ2_XS:
                DequantizeIQ2_XS(src, elementCount, dest);
                break;
            case QuantizationType.IQ2_S:
                DequantizeIQ2_S(src, elementCount, dest);
                break;
            case QuantizationType.IQ1_S:
                DequantizeIQ1_S(src, elementCount, dest);
                break;
            case QuantizationType.IQ3_XXS:
                DequantizeIQ3_XXS(src, elementCount, dest);
                break;
            case QuantizationType.IQ3_S:
                DequantizeIQ3_S(src, elementCount, dest);
                break;
            case QuantizationType.I2_S:
                DequantizeI2_S(src, elementCount, dest);
                break;
            case QuantizationType.MXFP4:
                DequantizeMxfp4Scalar(src, elementCount, dest);
                break;
            case QuantizationType.PQ2_0:
                DequantizePQ2_0(src, elementCount, dest);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(quantType), quantType,
                    $"Unsupported quantization type: {quantType}");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DequantizeF32(nint src, long elementCount, Span<float> dest)
    {
        new ReadOnlySpan<float>((void*)src, (int)elementCount).CopyTo(dest);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DequantizeFp16(nint src, long elementCount, Span<float> dest)
    {
        TensorPrimitives.ConvertToSingle(
            new ReadOnlySpan<Half>((void*)src, (int)elementCount),
            dest);
    }

    /// <summary>
    /// BF16 -> F32 expansion via shift-left-16 + reinterpret-as-F32. BF16 is
    /// the top 16 bits of the F32 binary representation, so the cast is bit-exact
    /// (no rounding on the read side; truncation already happened at quantisation
    /// time). Mirrors what the Vulkan BF16 matmul shaders do via
    /// <c>uintBitsToFloat(bf16_bits &lt;&lt; 16)</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DequantizeBf16(nint src, long elementCount, Span<float> dest)
    {
        ushort* p = (ushort*)src;
        for (long i = 0; i < elementCount; i++)
        {
            uint u = ((uint)p[i]) << 16;
            dest[(int)i] = BitConverter.Int32BitsToSingle((int)u);
        }
    }

    [SkipLocalsInit]
    private static void DequantizeQ8_0(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % Q8_0GroupSize != 0)
            throw new ArgumentException(
                $"Q8_0 element count must be a multiple of {Q8_0GroupSize}, got {elementCount}",
                nameof(elementCount));

        if (Avx2.IsSupported)
        {
            DequantizeQ8_0Avx2(src, elementCount, dest);
        }
        else
        {
            DequantizeQ8_0Scalar(src, elementCount, dest);
        }
    }

    /// <summary>
    /// Scalar Q8_0 dequantization. Always available as fallback and correctness reference.
    /// Each block: 2-byte Half scale + 32 sbyte quantized values → 32 floats.
    /// Formula: output[i] = (float)scale * qs[i]
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ8_0Scalar(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / Q8_0GroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;

        for (long b = 0; b < blockCount; b++)
        {
            float scale = (float)Unsafe.ReadUnaligned<Half>(blockBase);
            sbyte* qs = (sbyte*)(blockBase + 2);

            for (int i = 0; i < Q8_0GroupSize; i++)
            {
                dest[outIdx++] = scale * qs[i];
            }

            blockBase += Q8_0BlockBytes;
        }
    }

    // ──────────────────── MXFP4 ────────────────────

    /// <summary>
    /// Converts an E8M0 exponent byte to <c>2^(e-127) / 2</c> as float —
    /// i.e. the halved power-of-two block scale used with the doubled
    /// <see cref="Mxfp4Values"/> table. Mirrors llama.cpp's
    /// <c>ggml_e8m0_to_fp32_half</c>: <c>e &lt; 2</c> maps to the denormals
    /// 2^-128 / 2^-127; otherwise the bit pattern is <c>(e-1) &lt;&lt; 23</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float E8M0ToFloatHalf(byte e)
    {
        uint bits = e < 2
            ? 0x00200000u << e          // 2^-128, 2^-127 (denormal patterns)
            : (uint)(e - 1) << 23;      // 0.5 * 2^(e-127) = 2^(e-128)
        return BitConverter.UInt32BitsToSingle(bits);
    }

    /// <summary>
    /// Scalar MXFP4 dequantization — port of llama.cpp's
    /// <c>dequantize_row_mxfp4</c>. Block layout (17 bytes, 32 elements):
    /// <c>e (E8M0 scale byte @0), qs[16] @1</c>. Low nibbles → elements 0..15,
    /// high nibbles → elements 16..31. Formula:
    /// <c>value = kvalues[nibble] * e8m0_to_fp32_half(e)</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeMxfp4Scalar(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % Mxfp4GroupSize != 0)
            throw new ArgumentException(
                $"MXFP4 element count must be a multiple of {Mxfp4GroupSize}, got {elementCount}",
                nameof(elementCount));

        long blockCount = elementCount / Mxfp4GroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;
        ReadOnlySpan<sbyte> kvalues = Mxfp4Values;

        for (long b = 0; b < blockCount; b++)
        {
            float d = E8M0ToFloatHalf(blockBase[0]);
            byte* qs = blockBase + 1;

            for (int j = 0; j < 16; j++)
            {
                dest[outIdx + j] = kvalues[qs[j] & 0x0F] * d;
                dest[outIdx + j + 16] = kvalues[qs[j] >> 4] * d;
            }

            outIdx += Mxfp4GroupSize;
            blockBase += Mxfp4BlockBytes;
        }
    }

    // ──────────────────── Q4_0 ────────────────────
    /// <summary>
    /// Q4_0 scalar dequant. Block layout (18 bytes, 32 elements):
    /// <c>d(Half@0), qs[16]@2</c>. Formula: <c>value = d * (nibble - 8)</c>.
    /// </summary>
    /// <remarks>
    /// Nibble ordering follows llama.cpp <c>dequantize_row_q4_0</c>: within byte <c>j</c> the low
    /// nibble is element <c>j</c> and the high nibble is element <c>j + 16</c> — the two halves of
    /// the block are interleaved by nibble, not by adjacent pairs.
    /// </remarks>
    [SkipLocalsInit]
    internal static void DequantizeQ4_0Scalar(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % Q8_0GroupSize != 0)
            throw new ArgumentException(
                $"Q4_0 element count must be a multiple of {Q8_0GroupSize}, got {elementCount}",
                nameof(elementCount));
        long blockCount = elementCount / Q8_0GroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;
        for (long b = 0; b < blockCount; b++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(blockBase);
            byte* qs = blockBase + 2;
            for (int j = 0; j < 16; j++)
            {
                int lo = qs[j] & 0xF;
                int hi = (qs[j] >> 4) & 0xF;
                dest[outIdx + j]      = d * (lo - 8);
                dest[outIdx + j + 16] = d * (hi - 8);
            }
            outIdx += Q8_0GroupSize;
            blockBase += Q4_0BlockBytes;
        }
    }

    // ──────────────────── Q4_1 ────────────────────
    /// <summary>
    /// Q4_1 scalar dequant. Block layout (20 bytes, 32 elements):
    /// <c>d(Half@0), m(Half@2), qs[16]@4</c>. Formula: <c>value = d * nibble + m</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ4_1Scalar(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % Q8_0GroupSize != 0)
            throw new ArgumentException(
                $"Q4_1 element count must be a multiple of {Q8_0GroupSize}, got {elementCount}",
                nameof(elementCount));
        long blockCount = elementCount / Q8_0GroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;
        for (long b = 0; b < blockCount; b++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(blockBase);
            float m = (float)Unsafe.ReadUnaligned<Half>(blockBase + 2);
            byte* qs = blockBase + 4;
            for (int j = 0; j < 16; j++)
            {
                int lo = qs[j] & 0xF;
                int hi = (qs[j] >> 4) & 0xF;
                dest[outIdx + j]      = d * lo + m;
                dest[outIdx + j + 16] = d * hi + m;
            }
            outIdx += Q8_0GroupSize;
            blockBase += Q4_1BlockBytes;
        }
    }

    // ──────────────────── Q5_1 ────────────────────
    /// <summary>
    /// Q5_1 scalar dequant. Block layout (24 bytes, 32 elements):
    /// <c>d(Half@0), m(Half@2), qh[4]@4, qs[16]@8</c>.
    /// Formula: <c>value = d * ((qh_bit &lt;&lt; 4) | nibble) + m</c> (5-bit unsigned + min).
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ5_1Scalar(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % Q5_0GroupSize != 0)
            throw new ArgumentException(
                $"Q5_1 element count must be a multiple of {Q5_0GroupSize}, got {elementCount}",
                nameof(elementCount));
        long blockCount = elementCount / Q5_0GroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;
        for (long b = 0; b < blockCount; b++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(blockBase);
            float m = (float)Unsafe.ReadUnaligned<Half>(blockBase + 2);
            uint qh = Unsafe.ReadUnaligned<uint>(blockBase + 4);
            byte* qs = blockBase + 8;
            for (int j = 0; j < 16; j++)
            {
                int lo = qs[j] & 0xF;
                int hi = (qs[j] >> 4) & 0xF;
                int bit5Lo = (int)((qh >> j) & 1);
                int bit5Hi = (int)((qh >> (j + 16)) & 1);
                dest[outIdx + j]      = d * (lo | (bit5Lo << 4)) + m;
                dest[outIdx + j + 16] = d * (hi | (bit5Hi << 4)) + m;
            }
            outIdx += Q5_0GroupSize;
            blockBase += Q5_1BlockBytes;
        }
    }

    // ──────────────────── Q5_0 ────────────────────

    [SkipLocalsInit]
    private static void DequantizeQ5_0(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % Q5_0GroupSize != 0)
            throw new ArgumentException(
                $"Q5_0 element count must be a multiple of {Q5_0GroupSize}, got {elementCount}",
                nameof(elementCount));

        if (Avx2.IsSupported)
        {
            DequantizeQ5_0Avx2(src, elementCount, dest);
        }
        else
        {
            DequantizeQ5_0Scalar(src, elementCount, dest);
        }
    }

    /// <summary>
    /// Scalar Q5_0 dequantization. Block layout (22 bytes, 32 elements):
    /// <c>d(Half@0), qh[4]@2, qs[16]@6</c>.
    /// Formula: <c>value = d * (((qh_bit &lt;&lt; 4) | lo_nibble) - 16)</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ5_0Scalar(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / Q5_0GroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;

        for (long b = 0; b < blockCount; b++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(blockBase);
            uint qh = Unsafe.ReadUnaligned<uint>(blockBase + 2);
            byte* qs = blockBase + 6;

            for (int j = 0; j < 16; j++)
            {
                byte qsByte = qs[j];
                int lo = qsByte & 0xF;
                int hi = (qsByte >> 4) & 0xF;

                int bit5Lo = (int)((qh >> j) & 1);
                int bit5Hi = (int)((qh >> (j + 16)) & 1);

                // Low nibbles → elements 0..15, high nibbles → elements 16..31
                // (matches ggml's dequantize_row_q5_0 output ordering)
                dest[outIdx + j] = d * ((lo | (bit5Lo << 4)) - 16);
                dest[outIdx + j + 16] = d * ((hi | (bit5Hi << 4)) - 16);
            }

            outIdx += Q5_0GroupSize;
            blockBase += Q5_0BlockBytes;
        }
    }

    // ──────────────────── I2_S (BitNet ternary) ────────────────────

    /// <summary>
    /// Scalar I2_S dequantization (BitNet b1.58 ternary). Block layout: 128 codes → 32 bytes,
    /// 4 codes per byte. Within a block, byte at <c>group_pos</c> (0..31) holds the codes for
    /// elements {group_pos, +32, +64, +96} at bit offsets {6,4,2,0}. Codes map 0→-1, 1→0, 2→+1.
    /// A single per-tensor float32 scale is stored immediately after the packed data (offset n/4).
    /// Formula: <c>value = (code - 1) * scale</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeI2_S(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % I2SBlockSize != 0)
            throw new ArgumentException(
                $"I2_S element count must be a multiple of {I2SBlockSize}, got {elementCount}",
                nameof(elementCount));

        byte* data = (byte*)src;
        float scale = Unsafe.ReadUnaligned<float>(data + elementCount / 4);
        long blockCount = elementCount / I2SBlockSize;

        for (long b = 0; b < blockCount; b++)
        {
            byte* blockBase = data + b * 32;
            int outBase = (int)(b * I2SBlockSize);
            for (int gp = 0; gp < 32; gp++)
            {
                byte packed = blockBase[gp];
                dest[outBase + gp] = (((packed >> 6) & 0x3) - 1) * scale;
                dest[outBase + gp + 32] = (((packed >> 4) & 0x3) - 1) * scale;
                dest[outBase + gp + 64] = (((packed >> 2) & 0x3) - 1) * scale;
                dest[outBase + gp + 96] = ((packed & 0x3) - 1) * scale;
            }
        }
    }

    // ──────────────────── PQ2_0 (PrismML Bonsai ternary) ────────────────────

    /// <summary>
    /// Scalar PQ2_0 dequantization (PrismML Bonsai ternary). Group layout: 128 elements per
    /// 34-byte group, laid out as <c>scale(Half, 2 bytes) + codes[32](uint8, 4 codes/byte)</c>
    /// — note the scale comes BEFORE the codes (opposite of where a reader familiar with
    /// <see cref="DequantizeI2_S"/>'s tensor-TAIL scale might expect it — this is a genuinely
    /// different, per-GROUP scale, empirically confirmed, not to be assumed from I2_S's shape).
    /// Within a group, byte at index <c>b</c> (0..31) holds the codes for the 4 CONSECUTIVE
    /// elements {4b, 4b+1, 4b+2, 4b+3} at ASCENDING bit offsets {0,2,4,6} (element <c>4b+k</c> at
    /// bit offset <c>2k</c>) — verified byte-for-byte against PrismML's own reference
    /// <c>dequantize_row_q2_0</c> in their <c>PrismML-Eng/llama.cpp</c> fork (<c>ggml-quants.c</c>,
    /// <c>byte_index = j/4; bit_offset = (j%4)*2</c>). This is NOT the same convention as
    /// <see cref="DequantizeI2_S"/>'s strided {gp,+32,+64,+96}/descending-bits scheme — an earlier
    /// version of this function wrongly assumed PQ2_0 shared I2_S's exact bit-interleave (issue
    /// #269 follow-up investigation, 2026-08-05), which silently scrambled every weight's position
    /// within its 128-element group while leaving per-tensor statistics looking numerically
    /// unremarkable (same value set, wrong positions) — the root cause of Bonsai-27B's garbled
    /// generation and nightmarish (610,988) real-corpus perplexity. Codes map 0→-1, 1→0, 2→+1,
    /// 3→+2. Formula: <c>value = (code - 1) * group_scale</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizePQ2_0(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % PQ2_0GroupSize != 0)
            throw new ArgumentException(
                $"PQ2_0 element count must be a multiple of {PQ2_0GroupSize}, got {elementCount}",
                nameof(elementCount));

        byte* data = (byte*)src;
        long groupCount = elementCount / PQ2_0GroupSize;

        for (long g = 0; g < groupCount; g++)
        {
            byte* groupBase = data + g * PQ2_0GroupBytes;
            float scale = (float)Unsafe.ReadUnaligned<Half>(groupBase);
            byte* codes = groupBase + 2;
            int outBase = (int)(g * PQ2_0GroupSize);

            for (int b = 0; b < 32; b++)
            {
                byte packed = codes[b];
                int outIdx = outBase + 4 * b;
                dest[outIdx] = ((packed & 0x3) - 1) * scale;
                dest[outIdx + 1] = (((packed >> 2) & 0x3) - 1) * scale;
                dest[outIdx + 2] = (((packed >> 4) & 0x3) - 1) * scale;
                dest[outIdx + 3] = (((packed >> 6) & 0x3) - 1) * scale;
            }
        }
    }

    // ──────────────────── Q5_0 AVX2 ────────────────────

    /// <summary>
    /// AVX2-accelerated Q5_0 dequantization. Processes one 32-element block per iteration:
    /// unpacks low/high nibbles into a 256-bit vector, ORs in the 5th bit from <c>qh</c> via
    /// <see cref="MatMul.ExtractQ5HighBits"/>, subtracts 16 to recover the signed value, then
    /// widens sbyte→short→int→float and multiplies by the broadcast scale.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ5_0Avx2(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / Q5_0GroupSize;
        byte* blockBase = (byte*)src;

        Vector128<byte> nibbleMask = Vector128.Create((byte)0x0F);
        Vector256<sbyte> sixteen = Vector256.Create((sbyte)16);

        fixed (float* destPtr = dest)
        {
            float* outPtr = destPtr;

            for (long b = 0; b < blockCount; b++)
            {
                // Broadcast the Half scale to all 8 lanes.
                float scale = (float)Unsafe.ReadUnaligned<Half>(blockBase);
                Vector256<float> vScale = Vector256.Create(scale);

                // Load the 4-byte high-bit field and the 16-byte packed-nibble payload.
                uint qh = Unsafe.ReadUnaligned<uint>(blockBase + 2);
                Vector128<byte> qsRaw = Unsafe.ReadUnaligned<Vector128<byte>>(blockBase + 6);

                // Unpack nibbles: low 4 bits → elements 0..15, high 4 bits → elements 16..31.
                Vector128<byte> lo128 = Sse2.And(qsRaw, nibbleMask);
                Vector128<byte> hi128 = Sse2.And(
                    Sse2.ShiftRightLogical(qsRaw.AsUInt16(), 4).AsByte(),
                    nibbleMask);

                // Combine halves, OR in the 5th bit (0x10 per set bit), subtract 16 to center.
                Vector256<byte> q5vals = Avx2.Or(
                    Vector256.Create(lo128, hi128),
                    MatMul.ExtractQ5HighBits(qh));
                Vector256<sbyte> centered = Avx2.Subtract(q5vals.AsSByte(), sixteen);

                // Widen sbyte → short → int and convert to float × scale.
                Vector256<short> shortsLo = Avx2.ConvertToVector256Int16(centered.GetLower());
                Vector256<short> shortsHi = Avx2.ConvertToVector256Int16(centered.GetUpper());

                Vector256<int> ints0 = Avx2.ConvertToVector256Int32(shortsLo.GetLower());
                Vector256<int> ints1 = Avx2.ConvertToVector256Int32(shortsLo.GetUpper());
                Vector256<int> ints2 = Avx2.ConvertToVector256Int32(shortsHi.GetLower());
                Vector256<int> ints3 = Avx2.ConvertToVector256Int32(shortsHi.GetUpper());

                Vector256<float> f0 = Avx.Multiply(Avx.ConvertToVector256Single(ints0), vScale);
                Vector256<float> f1 = Avx.Multiply(Avx.ConvertToVector256Single(ints1), vScale);
                Vector256<float> f2 = Avx.Multiply(Avx.ConvertToVector256Single(ints2), vScale);
                Vector256<float> f3 = Avx.Multiply(Avx.ConvertToVector256Single(ints3), vScale);

                Avx.Store(outPtr, f0);
                Avx.Store(outPtr + 8, f1);
                Avx.Store(outPtr + 16, f2);
                Avx.Store(outPtr + 24, f3);

                outPtr += Q5_0GroupSize;
                blockBase += Q5_0BlockBytes;
            }
        }
    }

    // ──────────────────── Q8_0 AVX2 ────────────────────

    /// <summary>
    /// AVX2-accelerated Q8_0 dequantization. Processes one 32-element block per iteration
    /// using SIMD widen (sbyte → short → int → float) and broadcast multiply.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ8_0Avx2(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / Q8_0GroupSize;
        byte* blockBase = (byte*)src;

        fixed (float* destPtr = dest)
        {
            float* outPtr = destPtr;

            for (long b = 0; b < blockCount; b++)
            {
                // Read the Half scale and broadcast to all 8 lanes.
                float scale = (float)Unsafe.ReadUnaligned<Half>(blockBase);
                Vector256<float> vScale = Vector256.Create(scale);

                // Load 32 sbytes (quantized values).
                Vector256<sbyte> bytes = Unsafe.ReadUnaligned<Vector256<sbyte>>(blockBase + 2);

                // Widen sbyte → short: lower 16 and upper 16.
                Vector128<sbyte> bytesLo = bytes.GetLower();
                Vector128<sbyte> bytesHi = bytes.GetUpper();

                Vector256<short> shortsLo = Avx2.ConvertToVector256Int16(bytesLo);
                Vector256<short> shortsHi = Avx2.ConvertToVector256Int16(bytesHi);

                // Widen short → int (4 groups of 8).
                Vector256<int> ints0 = Avx2.ConvertToVector256Int32(shortsLo.GetLower());
                Vector256<int> ints1 = Avx2.ConvertToVector256Int32(shortsLo.GetUpper());
                Vector256<int> ints2 = Avx2.ConvertToVector256Int32(shortsHi.GetLower());
                Vector256<int> ints3 = Avx2.ConvertToVector256Int32(shortsHi.GetUpper());

                // Convert int → float and multiply by scale.
                Vector256<float> f0 = Avx.Multiply(Avx.ConvertToVector256Single(ints0), vScale);
                Vector256<float> f1 = Avx.Multiply(Avx.ConvertToVector256Single(ints1), vScale);
                Vector256<float> f2 = Avx.Multiply(Avx.ConvertToVector256Single(ints2), vScale);
                Vector256<float> f3 = Avx.Multiply(Avx.ConvertToVector256Single(ints3), vScale);

                // Store 4×8 = 32 floats.
                Avx.Store(outPtr, f0);
                Avx.Store(outPtr + 8, f1);
                Avx.Store(outPtr + 16, f2);
                Avx.Store(outPtr + 24, f3);

                outPtr += Q8_0GroupSize;
                blockBase += Q8_0BlockBytes;
            }
        }
    }
}
