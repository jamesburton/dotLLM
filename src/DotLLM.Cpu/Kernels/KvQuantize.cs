using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Quantization and dequantization kernels for KV-cache compression.
/// Supports Q8_0 (34 bytes per 32 elements) and Q4_0 (18 bytes per 32 elements).
/// </summary>
public static unsafe partial class KvQuantize
{
    /// <summary>Q8_0 block: 2 bytes (Half scale) + 32 bytes (int8 values) = 34 bytes.</summary>
    public const int Q8_0BlockBytes = 34;

    /// <summary>Q4_0 block: 2 bytes (Half scale) + 16 bytes (packed nibbles) = 18 bytes.</summary>
    public const int Q4_0BlockBytes = 18;

    /// <summary>Elements per quantization block (both Q8_0 and Q4_0).</summary>
    public const int BlockSize = 32;

    /// <summary>
    /// Returns the byte size of one quantized row of <paramref name="elementCount"/> elements.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int QuantizedRowBytes(int elementCount, KvCacheDType dtype) => dtype switch
    {
        KvCacheDType.Q8_0 => elementCount / BlockSize * Q8_0BlockBytes,
        KvCacheDType.Q4_0 => elementCount / BlockSize * Q4_0BlockBytes,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "Not a quantized type")
    };

    // ──────────────────── Q8_0 ────────────────────

    /// <summary>
    /// Quantizes FP32 data to Q8_0. Delegates to <see cref="MatMul.QuantizeF32ToQ8_0"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void F32ToQ8_0(float* src, byte* dest, int elementCount)
        => MatMul.QuantizeF32ToQ8_0(src, dest, elementCount);

    /// <summary>
    /// Dequantizes Q8_0 data to FP32.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Q8_0ToF32(byte* src, float* dest, int elementCount)
    {
        if (elementCount % BlockSize != 0)
            throw new ArgumentException(
                $"elementCount must be a multiple of {BlockSize}, got {elementCount}",
                nameof(elementCount));

        if (Avx512F.IsSupported)
            Q8_0ToF32Avx512(src, dest, elementCount);
        else if (Avx2.IsSupported)
            Q8_0ToF32Avx2(src, dest, elementCount);
        else
            Q8_0ToF32Scalar(src, dest, elementCount);
    }

    /// <summary>
    /// AVX-512 Q8_0 → FP32 dequantization. Widens 16 int8 values per instruction (VPMOVSXBD), halving
    /// the inner-loop trip count vs the AVX2 path. Numerically identical to <see cref="Q8_0ToF32Scalar"/>
    /// (<c>d * qs[i]</c>) — pure widen + scale, no rounding or lane-crossing.
    /// </summary>
    [SkipLocalsInit]
    internal static void Q8_0ToF32Avx512(byte* src, float* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;

        for (int block = 0; block < blockCount; block++)
        {
            byte* blockSrc = src + block * Q8_0BlockBytes;
            float* blockDst = dest + block * BlockSize;

            float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
            Vector512<float> vScale = Vector512.Create(d);
            sbyte* qs = (sbyte*)(blockSrc + 2);

            // 2 iterations of 16 for the 32-element block.
            for (int i = 0; i < BlockSize; i += 16)
            {
                Vector128<sbyte> bytes = Vector128.LoadUnsafe(ref Unsafe.AsRef<sbyte>(qs + i));
                Vector512<int> ints = Avx512F.ConvertToVector512Int32(bytes);       // VPMOVSXBD
                Vector512<float> floats = Avx512F.ConvertToVector512Single(ints);
                (floats * vScale).StoreUnsafe(ref Unsafe.AsRef<float>(blockDst + i));
            }
        }
    }

    [SkipLocalsInit]
    internal static void Q8_0ToF32Scalar(byte* src, float* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;

        for (int block = 0; block < blockCount; block++)
        {
            byte* blockSrc = src + block * Q8_0BlockBytes;
            float* blockDst = dest + block * BlockSize;

            float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
            sbyte* qs = (sbyte*)(blockSrc + 2);

            for (int i = 0; i < BlockSize; i++)
                blockDst[i] = d * qs[i];
        }
    }

    [SkipLocalsInit]
    internal static void Q8_0ToF32Avx2(byte* src, float* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;

        for (int block = 0; block < blockCount; block++)
        {
            byte* blockSrc = src + block * Q8_0BlockBytes;
            float* blockDst = dest + block * BlockSize;

            float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
            Vector256<float> vScale = Vector256.Create(d);
            sbyte* qs = (sbyte*)(blockSrc + 2);

            // Process 8 elements at a time (4 iterations for 32 elements)
            for (int i = 0; i < BlockSize; i += 8)
            {
                // Load 8 sbytes, sign-extend to int32, convert to float
                Vector128<byte> bytes = Vector128.CreateScalar(
                    Unsafe.ReadUnaligned<long>(qs + i)).AsByte();
                Vector256<int> ints = Avx2.ConvertToVector256Int32(bytes.AsSByte());
                Vector256<float> floats = Avx.ConvertToVector256Single(ints);
                Vector256<float> result = Avx.Multiply(floats, vScale);
                Avx.Store(blockDst + i, result);
            }
        }
    }

    // ──────────────────── Q4_0 ────────────────────

    /// <summary>
    /// Quantizes FP32 data to Q4_0 format.
    /// Block layout: <c>Half d (2 bytes) + uint8_t qs[16] (16 bytes)</c> = 18 bytes per 32 elements.
    /// Each byte packs two 4-bit values: low nibble = even element, high nibble = odd element.
    /// Dequant convention: <c>val = d * (nibble - 8)</c>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void F32ToQ4_0(float* src, byte* dest, int elementCount)
    {
        if (elementCount % BlockSize != 0)
            throw new ArgumentException(
                $"elementCount must be a multiple of {BlockSize}, got {elementCount}",
                nameof(elementCount));

        if (Avx512F.IsSupported)
            F32ToQ4_0Avx512(src, dest, elementCount);
        else if (Avx2.IsSupported)
            F32ToQ4_0Avx2(src, dest, elementCount);
        else
            F32ToQ4_0Scalar(src, dest, elementCount);
    }

    /// <summary>
    /// AVX-512 FP32 → Q4_0 quantization. Widens the max-abs scan and the round-to-int step to 16-wide;
    /// nibble packing stays scalar (inter-element dependency), exactly as the AVX2 path. Byte-identical
    /// to <see cref="F32ToQ4_0Scalar"/>: same round-to-nearest-even (VRNDSCALEPS, control 0) + clamp[0,15]
    /// the AVX2 path uses (<c>RoundToNearestInteger</c> then <c>Min</c>/<c>Max</c>).
    /// </summary>
    [SkipLocalsInit]
    internal static void F32ToQ4_0Avx512(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;
        Vector512<float> vEight = Vector512.Create(8.0f);
        Vector512<float> vZero = Vector512<float>.Zero;
        Vector512<float> vFifteen = Vector512.Create(15.0f);
        int* rounded = stackalloc int[BlockSize];

        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * BlockSize;
            byte* blockDst = dest + block * Q4_0BlockBytes;

            // Max-abs scan: 2 loads of 16 floats.
            Vector512<float> a0 = Vector512.Abs(Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(blockSrc)));
            Vector512<float> a1 = Vector512.Abs(Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(blockSrc + 16)));
            Vector512<float> maxAll = Vector512.Max(a0, a1);
            float maxAbs = HorizontalMaxAvx2(Avx.Max(maxAll.GetLower(), maxAll.GetUpper()));

            float d = maxAbs / 7.0f;
            Unsafe.WriteUnaligned(blockDst, (Half)d);

            byte* qs = blockDst + 2;
            if (d == 0)
            {
                for (int j = 0; j < 16; j++)
                    qs[j] = 0x88;
            }
            else
            {
                Vector512<float> vInvD = Vector512.Create(1.0f / d);

                // Round (to nearest even) + offset +8, clamp to [0,15], convert to int. 2 × 16-wide.
                for (int i = 0; i < BlockSize; i += 16)
                {
                    Vector512<float> vals = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(blockSrc + i));
                    Vector512<float> shifted = Avx512F.RoundScale(vals * vInvD, 0) + vEight;
                    Vector512<float> clamped = Vector512.Min(Vector512.Max(shifted, vZero), vFifteen);
                    Avx512F.ConvertToVector512Int32(clamped).StoreUnsafe(ref Unsafe.AsRef<int>(rounded + i));
                }

                // Pack nibbles: lo = even element, hi = odd element.
                for (int j = 0; j < 16; j++)
                    qs[j] = (byte)((rounded[2 * j + 1] << 4) | rounded[2 * j]);
            }
        }
    }

    /// <summary>
    /// Scalar reference implementation of FP32 → Q4_0 quantization.
    /// </summary>
    [SkipLocalsInit]
    internal static void F32ToQ4_0Scalar(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;

        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * BlockSize;
            byte* blockDst = dest + block * Q4_0BlockBytes;

            // Find max absolute value.
            float maxAbs = 0;
            for (int i = 0; i < BlockSize; i++)
            {
                float abs = MathF.Abs(blockSrc[i]);
                if (abs > maxAbs) maxAbs = abs;
            }

            // Scale: maps max absolute value to ±7 (4-bit signed range centered at 8).
            float d = maxAbs / 7.0f;
            Unsafe.WriteUnaligned(blockDst, (Half)d);

            byte* qs = blockDst + 2;
            if (d == 0)
            {
                // All values are zero — pack 8|8 nibbles.
                for (int j = 0; j < 16; j++)
                    qs[j] = 0x88; // (8 << 4) | 8
            }
            else
            {
                float invD = 1.0f / d;
                for (int j = 0; j < 16; j++)
                {
                    int lo = Math.Clamp((int)MathF.Round(blockSrc[2 * j] * invD) + 8, 0, 15);
                    int hi = Math.Clamp((int)MathF.Round(blockSrc[2 * j + 1] * invD) + 8, 0, 15);
                    qs[j] = (byte)((hi << 4) | lo);
                }
            }
        }
    }

    /// <summary>
    /// AVX2 SIMD implementation of FP32 → Q4_0 quantization.
    /// Vectorizes the max-abs scan and rounding. Nibble packing is scalar.
    /// </summary>
    [SkipLocalsInit]
    internal static void F32ToQ4_0Avx2(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;
        Vector256<float> vEight = Vector256.Create(8.0f);
        Vector256<float> vZero = Vector256<float>.Zero;
        Vector256<float> vFifteen = Vector256.Create(15.0f);
        int* rounded = stackalloc int[BlockSize];

        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * BlockSize;
            byte* blockDst = dest + block * Q4_0BlockBytes;

            // Max-abs scan: 4 loads of 8 floats.
            Vector256<float> v0 = Vector256.Abs(Avx.LoadVector256(blockSrc));
            Vector256<float> v1 = Vector256.Abs(Avx.LoadVector256(blockSrc + 8));
            Vector256<float> v2 = Vector256.Abs(Avx.LoadVector256(blockSrc + 16));
            Vector256<float> v3 = Vector256.Abs(Avx.LoadVector256(blockSrc + 24));

            Vector256<float> max01 = Avx.Max(v0, v1);
            Vector256<float> max23 = Avx.Max(v2, v3);
            Vector256<float> maxAll = Avx.Max(max01, max23);

            // Horizontal max via shuffles.
            float maxAbs = HorizontalMaxAvx2(maxAll);

            float d = maxAbs / 7.0f;
            Unsafe.WriteUnaligned(blockDst, (Half)d);

            byte* qs = blockDst + 2;
            if (d == 0)
            {
                for (int j = 0; j < 16; j++)
                    qs[j] = 0x88;
            }
            else
            {
                Vector256<float> vInvD = Vector256.Create(1.0f / d);

                // Round and clamp to [0, 15] using SIMD, then extract scalars for nibble packing.
                // Process 8 floats at a time.
                for (int i = 0; i < BlockSize; i += 8)
                {
                    Vector256<float> vals = Avx.LoadVector256(blockSrc + i);
                    Vector256<float> scaled = Avx.Multiply(vals, vInvD);
                    Vector256<float> shifted = Avx.Add(scaled, vEight); // +8 offset
                    Vector256<float> clamped = Avx.Min(Avx.Max(Avx.RoundToNearestInteger(shifted), vZero), vFifteen);
                    Avx2.Store(rounded + i, Avx.ConvertToVector256Int32(clamped));
                }

                // Pack nibbles: lo = even element, hi = odd element.
                for (int j = 0; j < 16; j++)
                    qs[j] = (byte)((rounded[2 * j + 1] << 4) | rounded[2 * j]);
            }
        }
    }

    /// <summary>
    /// Dequantizes Q4_0 data to FP32.
    /// Block layout: <c>Half d (2 bytes) + uint8_t qs[16] (16 bytes)</c>.
    /// Convention: <c>val = d * (nibble - 8)</c>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Q4_0ToF32(byte* src, float* dest, int elementCount)
    {
        if (elementCount % BlockSize != 0)
            throw new ArgumentException(
                $"elementCount must be a multiple of {BlockSize}, got {elementCount}",
                nameof(elementCount));

        if (Avx512F.IsSupported)
            Q4_0ToF32Avx512(src, dest, elementCount);
        else if (Avx2.IsSupported)
            Q4_0ToF32Avx2(src, dest, elementCount);
        else
            Q4_0ToF32Scalar(src, dest, elementCount);
    }

    /// <summary>
    /// AVX-512 Q4_0 → FP32 dequantization. Unpacks the 16 packed nibble bytes into the same interleaved
    /// order as the AVX2/scalar paths (<c>dest[2j]=d*(lo-8)</c>, <c>dest[2j+1]=d*(hi-8)</c>), then widens
    /// each 16-byte interleaved half with a single VPMOVSXBD instead of two 256-bit converts. Numerically
    /// identical to <see cref="Q4_0ToF32Scalar"/>.
    /// </summary>
    [SkipLocalsInit]
    internal static void Q4_0ToF32Avx512(byte* src, float* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;
        Vector512<int> vEight = Vector512.Create(8);

        for (int block = 0; block < blockCount; block++)
        {
            byte* blockSrc = src + block * Q4_0BlockBytes;
            float* blockDst = dest + block * BlockSize;

            float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
            Vector512<float> vScale = Vector512.Create(d);
            byte* qs = blockSrc + 2;

            // Extract nibbles (same as AVX2): lo = packed & 0x0F, hi = (packed >> 4) & 0x0F.
            Vector128<byte> packed = Vector128.LoadUnsafe(ref Unsafe.AsRef<byte>(qs));
            Vector128<byte> loMask = Vector128.Create((byte)0x0F);
            Vector128<byte> loNibbles = packed & loMask;
            Vector128<byte> hiNibbles = Vector128.ShiftRightLogical(packed.AsUInt16(), 4).AsByte() & loMask;

            // Interleave to output order: [lo0,hi0,...,lo7,hi7] | [lo8,hi8,...,lo15,hi15]
            Vector128<byte> interLo = Sse2.UnpackLow(loNibbles, hiNibbles);
            Vector128<byte> interHi = Sse2.UnpackHigh(loNibbles, hiNibbles);

            // Widen 16 sbytes → 16 int32 (VPMOVSXBD), subtract 8, scale, store.
            Vector512<int> i0 = Avx512F.ConvertToVector512Int32(interLo.AsSByte());
            (Avx512F.ConvertToVector512Single(i0 - vEight) * vScale)
                .StoreUnsafe(ref Unsafe.AsRef<float>(blockDst));

            Vector512<int> i1 = Avx512F.ConvertToVector512Int32(interHi.AsSByte());
            (Avx512F.ConvertToVector512Single(i1 - vEight) * vScale)
                .StoreUnsafe(ref Unsafe.AsRef<float>(blockDst + 16));
        }
    }

    /// <summary>
    /// Scalar reference implementation of Q4_0 → FP32 dequantization.
    /// </summary>
    [SkipLocalsInit]
    internal static void Q4_0ToF32Scalar(byte* src, float* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;

        for (int block = 0; block < blockCount; block++)
        {
            byte* blockSrc = src + block * Q4_0BlockBytes;
            float* blockDst = dest + block * BlockSize;

            float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
            byte* qs = blockSrc + 2;

            for (int j = 0; j < 16; j++)
            {
                byte packed = qs[j];
                int lo = (packed & 0x0F) - 8;
                int hi = (packed >> 4) - 8;
                blockDst[2 * j] = d * lo;
                blockDst[2 * j + 1] = d * hi;
            }
        }
    }

    /// <summary>
    /// AVX2 SIMD implementation of Q4_0 → FP32 dequantization.
    /// Unpacks 16 packed nibble bytes into 32 floats per block.
    /// Output order matches scalar: dest[2j] = lo nibble, dest[2j+1] = hi nibble.
    /// </summary>
    [SkipLocalsInit]
    internal static void Q4_0ToF32Avx2(byte* src, float* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;
        Vector256<int> vEight = Vector256.Create(8);

        for (int block = 0; block < blockCount; block++)
        {
            byte* blockSrc = src + block * Q4_0BlockBytes;
            float* blockDst = dest + block * BlockSize;

            float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
            Vector256<float> vScale = Vector256.Create(d);
            byte* qs = blockSrc + 2;

            // Load 16 packed bytes and extract nibbles.
            Vector128<byte> packed = Vector128.LoadUnsafe(ref Unsafe.AsRef<byte>(qs));
            Vector128<byte> loMask = Vector128.Create((byte)0x0F);
            Vector128<byte> loNibbles = packed & loMask;
            Vector128<byte> hiNibbles = Vector128.ShiftRightLogical(packed.AsUInt16(), 4).AsByte() & loMask;

            // Interleave lo and hi to get correct output order:
            // [lo0,hi0,lo1,hi1,...,lo7,hi7] in lower 128, [lo8,hi8,...,lo15,hi15] in upper 128
            Vector128<byte> interLo = Sse2.UnpackLow(loNibbles, hiNibbles);  // lo0,hi0,lo1,hi1,...,lo7,hi7
            Vector128<byte> interHi = Sse2.UnpackHigh(loNibbles, hiNibbles); // lo8,hi8,...,lo15,hi15

            // Process first 16 elements (from interLo): 2 groups of 8
            // Group 1: bytes 0..7 → 8 floats
            Vector256<int> i0 = Avx2.ConvertToVector256Int32(interLo.AsSByte());
            Vector256<float> f0 = Avx.Multiply(Avx.ConvertToVector256Single(Avx2.Subtract(i0, vEight)), vScale);
            Avx.Store(blockDst, f0);

            // Group 2: bytes 8..15 → 8 floats
            Vector128<byte> interLoHigh = Vector128.Shuffle(interLo, Vector128.Create(
                (byte)8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0));
            Vector256<int> i1 = Avx2.ConvertToVector256Int32(interLoHigh.AsSByte());
            Vector256<float> f1 = Avx.Multiply(Avx.ConvertToVector256Single(Avx2.Subtract(i1, vEight)), vScale);
            Avx.Store(blockDst + 8, f1);

            // Process second 16 elements (from interHi)
            Vector256<int> i2 = Avx2.ConvertToVector256Int32(interHi.AsSByte());
            Vector256<float> f2 = Avx.Multiply(Avx.ConvertToVector256Single(Avx2.Subtract(i2, vEight)), vScale);
            Avx.Store(blockDst + 16, f2);

            Vector128<byte> interHiHigh = Vector128.Shuffle(interHi, Vector128.Create(
                (byte)8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0));
            Vector256<int> i3 = Avx2.ConvertToVector256Int32(interHiHigh.AsSByte());
            Vector256<float> f3 = Avx.Multiply(Avx.ConvertToVector256Single(Avx2.Subtract(i3, vEight)), vScale);
            Avx.Store(blockDst + 24, f3);
        }
    }

    /// <summary>
    /// Horizontal max of a 256-bit float vector via reduction.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMaxAvx2(Vector256<float> v)
    {
        // Compare upper and lower 128-bit halves.
        Vector128<float> hi = v.GetUpper();
        Vector128<float> lo = v.GetLower();
        Vector128<float> max = Sse.Max(hi, lo);
        // Reduce 4-element 128-bit vector.
        max = Sse.Max(max, Sse.MoveHighToLow(max, max));
        max = Sse.Max(max, Sse.Shuffle(max, max, 0x01));
        return max.ToScalar();
    }
}
