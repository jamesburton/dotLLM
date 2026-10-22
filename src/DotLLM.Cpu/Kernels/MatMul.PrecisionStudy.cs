using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Switches used by the CPU precision study (#1109). Every switch defaults OFF, so production
/// dispatch is unchanged unless a study run opts in.
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>
    /// Reference arm for the precision study (#1109): route every quantized-weight projection whose
    /// production kernel quantizes the activation (Q8_0 / Q8_1 / Q8_K) through the dequantize-row and
    /// F32-dot path instead, so the activation stays F32 and only the weight format remains. KL between
    /// this arm and the production arm on the same model isolates the error the integer activation adds.
    /// </summary>
    /// <remarks>
    /// Reference and A/B use only. It is slow (one dequantized row at a time) and it is the "F32 stored
    /// on a hot path" that the precision rules forbid in production. <c>DOTLLM_CPU_REF_F32ACT=1</c>
    /// enables it at startup; tests and the study harness flip the property directly.
    /// </remarks>
    public static bool ReferenceF32Activations { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_CPU_REF_F32ACT") == "1";

    /// <summary>
    /// Prototype (#1109): use 256-bit <c>vpdpbusd</c> (<see cref="AvxVnni"/>) for the Q4_0 / Q4_1 /
    /// Q5_1 / IQ4_NL × Q8_1 block dot instead of the AVX2 <c>vpmaddubsw</c> + <c>vpmaddwd</c> pair.
    /// The integer sum is exact either way (no int16 saturation is possible: 15·127·2 &lt; 32767),
    /// so results are bit-identical. Default OFF; <c>DOTLLM_CPU_LEGACY_VNNI=1</c> enables it.
    /// </summary>
    public static bool LegacyVnniEnabled { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_CPU_LEGACY_VNNI") == "1";

    /// <summary>
    /// True when <paramref name="qt"/> is a weight format whose production CPU kernel quantizes the
    /// activation to an integer block format, and which <see cref="Dequantize.ToFloat32"/> can decode
    /// for the <see cref="ReferenceF32Activations"/> arm.
    /// </summary>
    /// <param name="qt">Weight quantization type.</param>
    /// <returns><see langword="true"/> when the reference arm should reroute this projection.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool UsesReferenceF32Path(QuantizationType qt) =>
        ReferenceF32Activations && qt is QuantizationType.Q8_0 or QuantizationType.Q5_0
            or QuantizationType.Q4_0 or QuantizationType.Q4_1 or QuantizationType.Q5_1
            or QuantizationType.IQ4_NL or QuantizationType.Q2_K or QuantizationType.Q3_K
            or QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K
            or QuantizationType.IQ4_XS;

    /// <summary>
    /// Float-activation tier for Q8_0 weights (#1109 study arm): <c>Σ d_w·q_w[i]·x[i]</c> with the
    /// int8 weights widened to F32 in registers and an F32 FMA against the unquantized activation.
    /// This is the best case for "keep activations F32": no stored dequantized copy, no activation
    /// quantization. Four rows share each activation load.
    /// </summary>
    /// <param name="w">First of four Q8_0 weight rows.</param>
    /// <param name="rowStride">Byte stride between consecutive rows.</param>
    /// <param name="x">F32 activation of <paramref name="blockCount"/>·32 elements.</param>
    /// <param name="blockCount">Number of 32-element blocks per row.</param>
    /// <param name="results">Receives four dot products.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void VecDotQ8_0F32Avx2_4Rows(byte* w, nint rowStride, float* x, int blockCount, float* results)
    {
        Vector256<float> a0 = Vector256<float>.Zero, a1 = Vector256<float>.Zero;
        Vector256<float> a2 = Vector256<float>.Zero, a3 = Vector256<float>.Zero;
        for (int b = 0; b < blockCount; b++)
        {
            float* xb = x + b * 32;
            Vector256<float> x0 = Avx.LoadVector256(xb), x1 = Avx.LoadVector256(xb + 8);
            Vector256<float> x2 = Avx.LoadVector256(xb + 16), x3 = Avx.LoadVector256(xb + 24);
            a0 = Q8_0BlockF32(w + b * Q8_0BlockBytes, x0, x1, x2, x3, a0);
            a1 = Q8_0BlockF32(w + rowStride + b * Q8_0BlockBytes, x0, x1, x2, x3, a1);
            a2 = Q8_0BlockF32(w + 2 * rowStride + b * Q8_0BlockBytes, x0, x1, x2, x3, a2);
            a3 = Q8_0BlockF32(w + 3 * rowStride + b * Q8_0BlockBytes, x0, x1, x2, x3, a3);
        }

        results[0] = HorizontalSumAvx2Float(a0);
        results[1] = HorizontalSumAvx2Float(a1);
        results[2] = HorizontalSumAvx2Float(a2);
        results[3] = HorizontalSumAvx2Float(a3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Q8_0BlockF32(byte* wb, Vector256<float> x0, Vector256<float> x1,
        Vector256<float> x2, Vector256<float> x3, Vector256<float> acc)
    {
        sbyte* q = (sbyte*)(wb + 2);

        // Sum the block in F32, then apply the block scale once (the same order llama.cpp's
        // dequantize-then-dot reference uses, up to FMA contraction).
        Vector256<float> s = Avx.Multiply(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(q)), x0);
        s = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(q + 8)), x1, s);
        s = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(q + 16)), x2, s);
        s = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(q + 24)), x3, s);
        return Fma.MultiplyAdd(Vector256.Create((float)Unsafe.ReadUnaligned<Half>(wb)), s, acc);
    }

    /// <summary>
    /// Float-activation tier for Q4_0 weights (#1109 study arm): nibbles widened to F32 in registers
    /// and dotted against the unquantized activation, four rows per activation load.
    /// </summary>
    /// <param name="w">First of four Q4_0 weight rows.</param>
    /// <param name="rowStride">Byte stride between consecutive rows.</param>
    /// <param name="x">F32 activation of <paramref name="blockCount"/>·32 elements.</param>
    /// <param name="blockCount">Number of 32-element blocks per row.</param>
    /// <param name="results">Receives four dot products.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void VecDotQ4_0F32Avx2_4Rows(byte* w, nint rowStride, float* x, int blockCount, float* results)
    {
        Vector256<float> a0 = Vector256<float>.Zero, a1 = Vector256<float>.Zero;
        Vector256<float> a2 = Vector256<float>.Zero, a3 = Vector256<float>.Zero;
        for (int b = 0; b < blockCount; b++)
        {
            float* xb = x + b * 32;
            Vector256<float> x0 = Avx.LoadVector256(xb), x1 = Avx.LoadVector256(xb + 8);
            Vector256<float> x2 = Avx.LoadVector256(xb + 16), x3 = Avx.LoadVector256(xb + 24);
            int off = b * Q4_0BlockBytes;
            a0 = Q4_0BlockF32(w + off, x0, x1, x2, x3, a0);
            a1 = Q4_0BlockF32(w + rowStride + off, x0, x1, x2, x3, a1);
            a2 = Q4_0BlockF32(w + 2 * rowStride + off, x0, x1, x2, x3, a2);
            a3 = Q4_0BlockF32(w + 3 * rowStride + off, x0, x1, x2, x3, a3);
        }

        results[0] = HorizontalSumAvx2Float(a0);
        results[1] = HorizontalSumAvx2Float(a1);
        results[2] = HorizontalSumAvx2Float(a2);
        results[3] = HorizontalSumAvx2Float(a3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Q4_0BlockF32(byte* wb, Vector256<float> x0, Vector256<float> x1,
        Vector256<float> x2, Vector256<float> x3, Vector256<float> acc)
    {
        // Q4_0 layout: 16 bytes; low nibbles are elements 0..15, high nibbles 16..31; value = nibble - 8.
        Vector128<byte> raw = Sse2.LoadVector128(wb + 2);
        Vector128<byte> mask = Vector128.Create((byte)0x0F);
        Vector128<sbyte> lo = Sse2.Subtract(Sse2.And(raw, mask), Vector128.Create((byte)8)).AsSByte();
        Vector128<sbyte> hi = Sse2.Subtract(Sse2.And(Sse2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask), Vector128.Create((byte)8)).AsSByte();
        Vector256<float> s = Avx.Multiply(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(lo)), x0);
        s = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(lo, 8))), x1, s);
        s = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(hi)), x2, s);
        s = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(hi, 8))), x3, s);
        return Fma.MultiplyAdd(Vector256.Create((float)Unsafe.ReadUnaligned<Half>(wb)), s, acc);
    }

    /// <summary>
    /// Scalar reference for the float-activation tiers: <c>Σ_b d_b · Σ_i q[b,i]·x[b,i]</c> in F32
    /// with the same per-block grouping as the vector kernels.
    /// </summary>
    /// <param name="qt"><see cref="QuantizationType.Q8_0"/> or <see cref="QuantizationType.Q4_0"/>.</param>
    /// <param name="w">Weight row.</param>
    /// <param name="x">F32 activation.</param>
    /// <param name="blockCount">Number of 32-element blocks.</param>
    /// <returns>The dot product.</returns>
    internal static float VecDotQuantF32Scalar(QuantizationType qt, byte* w, float* x, int blockCount)
    {
        int blockBytes = qt == QuantizationType.Q8_0 ? Q8_0BlockBytes : Q4_0BlockBytes;
        float sum = 0f;
        for (int b = 0; b < blockCount; b++)
        {
            byte* wb = w + b * blockBytes;
            float d = (float)Unsafe.ReadUnaligned<Half>(wb);
            float s = 0f;
            for (int i = 0; i < 32; i++)
            {
                int q = qt == QuantizationType.Q8_0
                    ? ((sbyte*)(wb + 2))[i]
                    : (i < 16 ? (wb[2 + i] & 0x0F) : (wb[2 + i - 16] >> 4)) - 8;
                s += q * x[b * 32 + i];
            }

            sum += d * s;
        }

        return sum;
    }

    /// <summary>Best available single-row Q8_0 × Q8_0 dot: VNNI, then AVX2 / SSSE3 / scalar.</summary>
    /// <param name="w">Q8_0 row.</param>
    /// <param name="xQ8">Q8_0-quantized activation.</param>
    /// <param name="blockCount">Number of 32-element blocks.</param>
    /// <returns>The dot product.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float VecDotQ8_0BestRow(byte* w, byte* xQ8, int blockCount)
        => AvxVnni.IsSupported ? VecDotQ8_0Vnni(w, xQ8, blockCount) : VecDotQ8_0Row(w, xQ8, blockCount);

    /// <summary>
    /// BF16 weights × F32 activation, four rows per activation load (#1109 study arm; the repo has no
    /// BF16 GEMV kernel). BF16→F32 is a 16-bit left shift, so the widen is exact and costs one
    /// <c>vpmovzxwd</c> + <c>vpslld</c> per 8 lanes. <paramref name="k"/> must be a multiple of 8.
    /// </summary>
    /// <param name="w">First of four BF16 weight rows (raw <see cref="ushort"/> bits).</param>
    /// <param name="rowStride">Byte stride between consecutive rows.</param>
    /// <param name="x">F32 activation of <paramref name="k"/> elements.</param>
    /// <param name="k">Row length.</param>
    /// <param name="results">Receives four dot products.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void VecDotBf16F32Avx2_4Rows(byte* w, nint rowStride, float* x, int k, float* results)
    {
        Vector256<float> a0 = Vector256<float>.Zero, a1 = Vector256<float>.Zero;
        Vector256<float> a2 = Vector256<float>.Zero, a3 = Vector256<float>.Zero;
        ushort* r0 = (ushort*)w, r1 = (ushort*)(w + rowStride);
        ushort* r2 = (ushort*)(w + 2 * rowStride), r3 = (ushort*)(w + 3 * rowStride);
        for (int i = 0; i < k; i += 8)
        {
            Vector256<float> xv = Avx.LoadVector256(x + i);
            a0 = Fma.MultiplyAdd(Bf16x8(r0 + i), xv, a0);
            a1 = Fma.MultiplyAdd(Bf16x8(r1 + i), xv, a1);
            a2 = Fma.MultiplyAdd(Bf16x8(r2 + i), xv, a2);
            a3 = Fma.MultiplyAdd(Bf16x8(r3 + i), xv, a3);
        }

        results[0] = HorizontalSumAvx2Float(a0);
        results[1] = HorizontalSumAvx2Float(a1);
        results[2] = HorizontalSumAvx2Float(a2);
        results[3] = HorizontalSumAvx2Float(a3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Bf16x8(ushort* p)
        => Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(p), 16).AsSingle();

    /// <summary>Scalar reference for <see cref="VecDotBf16F32Avx2_4Rows"/> (one row).</summary>
    /// <param name="w">BF16 row (raw bits).</param>
    /// <param name="x">F32 activation.</param>
    /// <param name="k">Row length.</param>
    /// <returns>The dot product.</returns>
    internal static float VecDotBf16F32Scalar(ushort* w, float* x, int k)
    {
        float sum = 0f;
        for (int i = 0; i < k; i++)
            sum += BitConverter.Int32BitsToSingle(w[i] << 16) * x[i];
        return sum;
    }
}
