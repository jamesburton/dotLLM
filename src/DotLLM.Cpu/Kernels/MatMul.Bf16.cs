using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// BF16 weight GEMV/GEMM by shift-widening (#1118). BF16 is the top half of an F32, so <c>bf16 &lt;&lt; 16</c> is the exact F32 value:
/// every tier widens 4/8/16 weights in registers (<c>vpmovzxwd</c> + <c>vpslld</c>) and multiplies into F32 accumulators. No F32
/// copy of the weight matrix is ever materialised (rule 2 of #1107): the GEMM's only staging is the same bounded four-row
/// scratch the F16 tile uses (<c>4·k</c> floats per worker), never stored beyond the row group.
/// </summary>
/// <remarks>
/// <para><b>Arithmetic contract.</b> Each output is one accumulator chain over <c>k</c>: on the AVX2/FMA tier an 8-lane FMA chain,
/// <see cref="HSum256"/>, then a scalar tail; on the opt-in AVX-512 tier a 16-lane FMA chain folded to 8 then the same tree; on the
/// 128-bit tier a 4-lane multiply-add chain (no FMA) and the <c>(a0+a1)+(a2+a3)</c> tree. The GEMV, the GEMM and every thread split
/// run the identical chain, so a row's output is bit-identical regardless of batch size or thread count, and
/// <see cref="DotBf16LaneReferenceScalar"/> reproduces each tier's exact result with scalar code (the test oracle).</para>
///
/// <para><b>Tiers.</b> Default: AVX2+FMA (8-wide). <c>DOTLLM_CPU_BF16_512=1</c> opts into the 16-wide AVX-512F variant for A/B on
/// hosts that have it (Zen 5 runs 512-bit ops on 256-bit datapaths, so no win is assumed). Below AVX2: portable <c>Vector128</c>.
/// <c>DOTLLM_CPU_BF16_KERNEL=0</c> routes BF16 back to the dequantize-rows path (A/B baseline). The preview <c>Avx512Bf16</c>
/// (<c>vdpbf16ps</c>) intrinsic is not used: it is not in a shipping .NET (dev-dotnet11 track).</para>
/// </remarks>
public static unsafe partial class MatMul
{
    private static readonly bool s_bf16EnvOff = Environment.GetEnvironmentVariable("DOTLLM_CPU_BF16_KERNEL") == "0";
    private static readonly bool s_bf16Env512 = Environment.GetEnvironmentVariable("DOTLLM_CPU_BF16_512") == "1";

    /// <summary>Runtime switch for the BF16 kernels (default on; <c>DOTLLM_CPU_BF16_KERNEL=0</c> turns them off). Settable for in-process A/B.</summary>
    public static bool Bf16KernelEnabled { get; set; } = !s_bf16EnvOff;

    /// <summary>Opt-in 16-wide AVX-512F variant (<c>DOTLLM_CPU_BF16_512=1</c>). Settable for in-process A/B.</summary>
    public static bool Bf16Avx512Enabled { get; set; } = s_bf16Env512;

    /// <summary>16-wide AVX-512F BF16 kernels are in force.</summary>
    internal static bool Bf16Avx512Tier
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => CpuFeatures.Avx512F && Bf16Avx512Enabled;
    }

    /// <summary>8-wide AVX2+FMA BF16 kernels are available (the default tier).</summary>
    internal static bool Bf16FmaTier
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => CpuFeatures.Avx2Fma;
    }

    /// <summary>4-wide portable <c>Vector128</c> BF16 kernels are available.</summary>
    internal static bool Bf16V128Tier
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => CpuFeatures.Vector128Accelerated;
    }

    /// <summary>The BF16 value as a Single (exact).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float Bf16ToSingle(ushort bits) => BitConverter.Int32BitsToSingle(bits << 16);

    // ──────────────────── References ────────────────────

    /// <summary>Plain sequential scalar dot: the mathematical reference (not bit-identical to the SIMD tiers).</summary>
    internal static float DotBf16Scalar(ushort* w, float* x, int k)
    {
        float sum = 0;
        for (int i = 0; i < k; i++) sum += Bf16ToSingle(w[i]) * x[i];
        return sum;
    }

    /// <summary>
    /// Scalar reproduction of a SIMD tier's exact arithmetic: <paramref name="lanes"/> accumulator chains (FMA when
    /// <paramref name="fma"/>), halves folded down to four lanes, the <c>(a0+a1)+(a2+a3)</c> tree, then the scalar tail.
    /// <c>lanes = 8, fma = true</c> is the AVX2/FMA tier, <c>16, true</c> the AVX-512 tier, <c>4, false</c> the Vector128 tier.
    /// </summary>
    internal static float DotBf16LaneReferenceScalar(ushort* w, float* x, int k, int lanes, bool fma)
    {
        float* acc = stackalloc float[lanes];
        for (int j = 0; j < lanes; j++) acc[j] = 0;
        int i = 0;
        for (; i + lanes - 1 < k; i += lanes)
            for (int j = 0; j < lanes; j++)
                acc[j] = fma ? MathF.FusedMultiplyAdd(Bf16ToSingle(w[i + j]), x[i + j], acc[j]) : acc[j] + Bf16ToSingle(w[i + j]) * x[i + j];
        for (int half = lanes / 2; half >= 4; half /= 2)
            for (int j = 0; j < half; j++) acc[j] += acc[j + half];
        float sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for (; i < k; i++) sum += Bf16ToSingle(w[i]) * x[i];
        return sum;
    }

    // ──────────────────── Widening ────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> WidenBf16x8Avx2(ushort* p)
        => Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(Unsafe.ReadUnaligned<Vector128<ushort>>(p)), 16).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> WidenBf16x16Avx512(ushort* p)
        => Avx512F.ShiftLeftLogical(Avx512F.ConvertToVector512Int32(Unsafe.ReadUnaligned<Vector256<ushort>>(p)), 16).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> WidenBf16x4V128(ushort* p)
        => Vector128.ShiftLeft(Vector128.WidenLower(Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<ulong>(p)).AsUInt16()), 16).AsSingle();

    /// <summary>Widens a BF16 row into F32 (exact), 8 at a time.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WidenBf16RowAvx2(ushort* src, float* dst, int k)
    {
        int i = 0;
        for (; i + 7 < k; i += 8) Avx.Store(dst + i, WidenBf16x8Avx2(src + i));
        for (; i < k; i++) dst[i] = Bf16ToSingle(src[i]);
    }

    // ──────────────────── AVX2 + FMA tier (8-wide) ────────────────────

    /// <summary>One BF16 row dot: the exact arithmetic of <see cref="DotF32Fma"/> on the widened row.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float DotBf16Fma(ushort* w, float* x, int k)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        int i = 0;
        for (; i + 7 < k; i += 8)
            acc = Fma.MultiplyAdd(WidenBf16x8Avx2(w + i), Avx.LoadVector256(x + i), acc);
        float sum = HSum256(acc);
        for (; i < k; i++) sum += Bf16ToSingle(w[i]) * x[i];
        return sum;
    }

    /// <summary>Four BF16 rows (<paramref name="k"/> elements apart) against one activation; per row the arithmetic of <see cref="DotBf16Fma"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Bf16Rows4Fma(ushort* w, float* x, float* y, int k)
    {
        ushort* w0 = w, w1 = w + k, w2 = w + 2L * k, w3 = w + 3L * k;
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default;
        int i = 0;
        for (; i + 7 < k; i += 8)
        {
            Vector256<float> xv = Avx.LoadVector256(x + i);
            a0 = Fma.MultiplyAdd(WidenBf16x8Avx2(w0 + i), xv, a0);
            a1 = Fma.MultiplyAdd(WidenBf16x8Avx2(w1 + i), xv, a1);
            a2 = Fma.MultiplyAdd(WidenBf16x8Avx2(w2 + i), xv, a2);
            a3 = Fma.MultiplyAdd(WidenBf16x8Avx2(w3 + i), xv, a3);
        }
        float s0 = HSum256(a0), s1 = HSum256(a1), s2 = HSum256(a2), s3 = HSum256(a3);
        for (; i < k; i++)
        {
            float xv = x[i];
            s0 += Bf16ToSingle(w0[i]) * xv; s1 += Bf16ToSingle(w1[i]) * xv;
            s2 += Bf16ToSingle(w2[i]) * xv; s3 += Bf16ToSingle(w3[i]) * xv;
        }
        y[0] = s0; y[1] = s1; y[2] = s2; y[3] = s3;
    }

    /// <summary>4 rows x 2 tokens widened in registers (no scratch); per output the arithmetic of <see cref="DotBf16Fma"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile4x2Bf16Fma(ushort* w, float* x0, float* x1, float* c0, float* c1, int k)
    {
        ushort* w0 = w, w1 = w + k, w2 = w + 2L * k, w3 = w + 3L * k;
        Vector256<float> a00 = default, a01 = default, a10 = default, a11 = default;
        Vector256<float> a20 = default, a21 = default, a30 = default, a31 = default;
        int i = 0;
        for (; i + 7 < k; i += 8)
        {
            Vector256<float> xa = Avx.LoadVector256(x0 + i);
            Vector256<float> xb = Avx.LoadVector256(x1 + i);
            Vector256<float> v = WidenBf16x8Avx2(w0 + i);
            a00 = Fma.MultiplyAdd(v, xa, a00); a01 = Fma.MultiplyAdd(v, xb, a01);
            v = WidenBf16x8Avx2(w1 + i);
            a10 = Fma.MultiplyAdd(v, xa, a10); a11 = Fma.MultiplyAdd(v, xb, a11);
            v = WidenBf16x8Avx2(w2 + i);
            a20 = Fma.MultiplyAdd(v, xa, a20); a21 = Fma.MultiplyAdd(v, xb, a21);
            v = WidenBf16x8Avx2(w3 + i);
            a30 = Fma.MultiplyAdd(v, xa, a30); a31 = Fma.MultiplyAdd(v, xb, a31);
        }
        float s00 = HSum256(a00), s01 = HSum256(a01), s10 = HSum256(a10), s11 = HSum256(a11);
        float s20 = HSum256(a20), s21 = HSum256(a21), s30 = HSum256(a30), s31 = HSum256(a31);
        for (; i < k; i++)
        {
            float xa = x0[i], xb = x1[i];
            float f0 = Bf16ToSingle(w0[i]), f1 = Bf16ToSingle(w1[i]), f2 = Bf16ToSingle(w2[i]), f3 = Bf16ToSingle(w3[i]);
            s00 += f0 * xa; s01 += f0 * xb; s10 += f1 * xa; s11 += f1 * xb;
            s20 += f2 * xa; s21 += f2 * xb; s30 += f3 * xa; s31 += f3 * xb;
        }
        c0[0] = s00; c0[1] = s10; c0[2] = s20; c0[3] = s30;
        c1[0] = s01; c1[1] = s11; c1[2] = s21; c1[3] = s31;
    }

    /// <summary>
    /// A/B switch between the two bit-identical AVX2/FMA GEMM forms: the register tile (default) and the F16-style four-row F32
    /// scratch (<c>DOTLLM_CPU_BF16_SCRATCH=1</c>). Settable for in-process A/B.
    /// </summary>
    public static bool Bf16FmaScratch { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_CPU_BF16_SCRATCH") == "1";

    /// <summary>BF16 GEMM over <paramref name="tileRows"/> rows on the AVX2/FMA tier, widened in registers (no scratch).</summary>
    [SkipLocalsInit]
    internal static void GemmBf16RowsFmaTile(ushort* tileW, int tileRows, float* b, float* c, int cStride, int k, int n)
    {
        for (int r0 = 0; r0 < tileRows; r0 += 4)
        {
            int rr = Math.Min(4, tileRows - r0);
            ushort* w = tileW + (long)r0 * k;
            int t = 0;
            if (rr == 4)
            {
                for (; t + 1 < n; t += 2)
                    Tile4x2Bf16Fma(w, b + (long)t * k, b + (long)(t + 1) * k,
                        c + (long)t * cStride + r0, c + (long)(t + 1) * cStride + r0, k);
            }
            for (; t < n; t++)
            {
                float* x = b + (long)t * k;
                float* o = c + (long)t * cStride + r0;
                for (int j = 0; j < rr; j++) o[j] = DotBf16Fma(w + (long)j * k, x, k);
            }
        }
    }

    /// <summary>
    /// BF16 GEMM over <paramref name="tileRows"/> rows on the AVX2/FMA tier: four rows widened into <paramref name="scratch"/>
    /// (<c>4·k</c> floats) once, then the 4x2 FMA tile of the F16 path over all tokens. Bit-identical to <see cref="DotBf16Fma"/> per output.
    /// </summary>
    [SkipLocalsInit]
    internal static void GemmBf16RowsFma(ushort* tileW, int tileRows, float* b, float* c, int cStride, int k, int n, float* scratch)
    {
        for (int r0 = 0; r0 < tileRows; r0 += 4)
        {
            int rr = Math.Min(4, tileRows - r0);
            for (int j = 0; j < rr; j++)
                WidenBf16RowAvx2(tileW + (long)(r0 + j) * k, scratch + (long)j * k, k);

            int t = 0;
            if (rr == 4)
            {
                for (; t + 1 < n; t += 2)
                    Tile4x2F32Fma(scratch, b + (long)t * k, b + (long)(t + 1) * k,
                        c + (long)t * cStride + r0, c + (long)(t + 1) * cStride + r0, k);
            }
            for (; t < n; t++)
            {
                float* x = b + (long)t * k;
                float* o = c + (long)t * cStride + r0;
                for (int j = 0; j < rr; j++)
                    o[j] = DotF32Fma(scratch + (long)j * k, x, k);
            }
        }
    }

    // ──────────────────── AVX-512F tier (16-wide, opt-in) ────────────────────

    /// <summary>One BF16 row dot, 16-wide FMA chain; <see cref="HorizontalSumAvx512Float"/> folds to the 8-lane tree.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float DotBf16Avx512(ushort* w, float* x, int k)
    {
        Vector512<float> acc = Vector512<float>.Zero;
        int i = 0;
        for (; i + 15 < k; i += 16)
            acc = Avx512F.FusedMultiplyAdd(WidenBf16x16Avx512(w + i), Avx512F.LoadVector512(x + i), acc);
        float sum = HorizontalSumAvx512Float(acc);
        for (; i < k; i++) sum += Bf16ToSingle(w[i]) * x[i];
        return sum;
    }

    /// <summary>Four BF16 rows against one activation, 16-wide; per row the arithmetic of <see cref="DotBf16Avx512"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Bf16Rows4Avx512(ushort* w, float* x, float* y, int k)
    {
        ushort* w0 = w, w1 = w + k, w2 = w + 2L * k, w3 = w + 3L * k;
        Vector512<float> a0 = default, a1 = default, a2 = default, a3 = default;
        int i = 0;
        for (; i + 15 < k; i += 16)
        {
            Vector512<float> xv = Avx512F.LoadVector512(x + i);
            a0 = Avx512F.FusedMultiplyAdd(WidenBf16x16Avx512(w0 + i), xv, a0);
            a1 = Avx512F.FusedMultiplyAdd(WidenBf16x16Avx512(w1 + i), xv, a1);
            a2 = Avx512F.FusedMultiplyAdd(WidenBf16x16Avx512(w2 + i), xv, a2);
            a3 = Avx512F.FusedMultiplyAdd(WidenBf16x16Avx512(w3 + i), xv, a3);
        }
        float s0 = HorizontalSumAvx512Float(a0), s1 = HorizontalSumAvx512Float(a1);
        float s2 = HorizontalSumAvx512Float(a2), s3 = HorizontalSumAvx512Float(a3);
        for (; i < k; i++)
        {
            float xv = x[i];
            s0 += Bf16ToSingle(w0[i]) * xv; s1 += Bf16ToSingle(w1[i]) * xv;
            s2 += Bf16ToSingle(w2[i]) * xv; s3 += Bf16ToSingle(w3[i]) * xv;
        }
        y[0] = s0; y[1] = s1; y[2] = s2; y[3] = s3;
    }

    /// <summary>4 rows x 2 tokens, widened in registers (no scratch); per output the arithmetic of <see cref="DotBf16Avx512"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile4x2Bf16Avx512(ushort* w, float* x0, float* x1, float* c0, float* c1, int k)
    {
        ushort* w0 = w, w1 = w + k, w2 = w + 2L * k, w3 = w + 3L * k;
        Vector512<float> a00 = default, a01 = default, a10 = default, a11 = default;
        Vector512<float> a20 = default, a21 = default, a30 = default, a31 = default;
        int i = 0;
        for (; i + 15 < k; i += 16)
        {
            Vector512<float> xa = Avx512F.LoadVector512(x0 + i);
            Vector512<float> xb = Avx512F.LoadVector512(x1 + i);
            Vector512<float> v = WidenBf16x16Avx512(w0 + i);
            a00 = Avx512F.FusedMultiplyAdd(v, xa, a00); a01 = Avx512F.FusedMultiplyAdd(v, xb, a01);
            v = WidenBf16x16Avx512(w1 + i);
            a10 = Avx512F.FusedMultiplyAdd(v, xa, a10); a11 = Avx512F.FusedMultiplyAdd(v, xb, a11);
            v = WidenBf16x16Avx512(w2 + i);
            a20 = Avx512F.FusedMultiplyAdd(v, xa, a20); a21 = Avx512F.FusedMultiplyAdd(v, xb, a21);
            v = WidenBf16x16Avx512(w3 + i);
            a30 = Avx512F.FusedMultiplyAdd(v, xa, a30); a31 = Avx512F.FusedMultiplyAdd(v, xb, a31);
        }
        float s00 = HorizontalSumAvx512Float(a00), s01 = HorizontalSumAvx512Float(a01);
        float s10 = HorizontalSumAvx512Float(a10), s11 = HorizontalSumAvx512Float(a11);
        float s20 = HorizontalSumAvx512Float(a20), s21 = HorizontalSumAvx512Float(a21);
        float s30 = HorizontalSumAvx512Float(a30), s31 = HorizontalSumAvx512Float(a31);
        for (; i < k; i++)
        {
            float xa = x0[i], xb = x1[i];
            float f0 = Bf16ToSingle(w0[i]), f1 = Bf16ToSingle(w1[i]), f2 = Bf16ToSingle(w2[i]), f3 = Bf16ToSingle(w3[i]);
            s00 += f0 * xa; s01 += f0 * xb; s10 += f1 * xa; s11 += f1 * xb;
            s20 += f2 * xa; s21 += f2 * xb; s30 += f3 * xa; s31 += f3 * xb;
        }
        c0[0] = s00; c0[1] = s10; c0[2] = s20; c0[3] = s30;
        c1[0] = s01; c1[1] = s11; c1[2] = s21; c1[3] = s31;
    }

    /// <summary>BF16 GEMM over <paramref name="tileRows"/> rows on the AVX-512F tier (register tile, no scratch).</summary>
    [SkipLocalsInit]
    internal static void GemmBf16RowsAvx512(ushort* tileW, int tileRows, float* b, float* c, int cStride, int k, int n)
    {
        for (int r0 = 0; r0 < tileRows; r0 += 4)
        {
            int rr = Math.Min(4, tileRows - r0);
            ushort* w = tileW + (long)r0 * k;
            int t = 0;
            if (rr == 4)
            {
                for (; t + 1 < n; t += 2)
                    Tile4x2Bf16Avx512(w, b + (long)t * k, b + (long)(t + 1) * k,
                        c + (long)t * cStride + r0, c + (long)(t + 1) * cStride + r0, k);
            }
            for (; t < n; t++)
            {
                float* x = b + (long)t * k;
                float* o = c + (long)t * cStride + r0;
                for (int j = 0; j < rr; j++) o[j] = DotBf16Avx512(w + (long)j * k, x, k);
            }
        }
    }

    // ──────────────────── Vector128 tier (4-wide, no FMA) ────────────────────

    /// <summary>One BF16 row dot, 4-wide multiply-add chain and the <c>(a0+a1)+(a2+a3)</c> tree.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float DotBf16V128Simd(ushort* w, float* x, int k)
    {
        Vector128<float> acc = Vector128<float>.Zero;
        int i = 0;
        for (; i + 3 < k; i += 4)
            acc += WidenBf16x4V128(w + i) * Vector128.Load(x + i);
        float sum = (acc.GetElement(0) + acc.GetElement(1)) + (acc.GetElement(2) + acc.GetElement(3));
        for (; i < k; i++) sum += Bf16ToSingle(w[i]) * x[i];
        return sum;
    }

    /// <summary>Four BF16 rows against one activation, 4-wide; per row the arithmetic of <see cref="DotBf16V128Simd"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Bf16Rows4V128Simd(ushort* w, float* x, float* y, int k)
    {
        ushort* w0 = w, w1 = w + k, w2 = w + 2L * k, w3 = w + 3L * k;
        Vector128<float> a0 = default, a1 = default, a2 = default, a3 = default;
        int i = 0;
        for (; i + 3 < k; i += 4)
        {
            Vector128<float> xv = Vector128.Load(x + i);
            a0 += WidenBf16x4V128(w0 + i) * xv;
            a1 += WidenBf16x4V128(w1 + i) * xv;
            a2 += WidenBf16x4V128(w2 + i) * xv;
            a3 += WidenBf16x4V128(w3 + i) * xv;
        }
        float s0 = (a0.GetElement(0) + a0.GetElement(1)) + (a0.GetElement(2) + a0.GetElement(3));
        float s1 = (a1.GetElement(0) + a1.GetElement(1)) + (a1.GetElement(2) + a1.GetElement(3));
        float s2 = (a2.GetElement(0) + a2.GetElement(1)) + (a2.GetElement(2) + a2.GetElement(3));
        float s3 = (a3.GetElement(0) + a3.GetElement(1)) + (a3.GetElement(2) + a3.GetElement(3));
        for (; i < k; i++)
        {
            float xv = x[i];
            s0 += Bf16ToSingle(w0[i]) * xv; s1 += Bf16ToSingle(w1[i]) * xv;
            s2 += Bf16ToSingle(w2[i]) * xv; s3 += Bf16ToSingle(w3[i]) * xv;
        }
        y[0] = s0; y[1] = s1; y[2] = s2; y[3] = s3;
    }

    // ──────────────────── Tier dispatch ────────────────────

    /// <summary>GEMV over <paramref name="m"/> rows on the tier in force (four rows at a time, then singles).</summary>
    [SkipLocalsInit]
    private static void Bf16Rows(ushort* w, float* x, float* y, int m, int k)
    {
        int row = 0;
        if (Bf16Avx512Tier)
        {
            for (; row + 3 < m; row += 4) Bf16Rows4Avx512(w + (long)row * k, x, y + row, k);
            for (; row < m; row++) y[row] = DotBf16Avx512(w + (long)row * k, x, k);
        }
        else if (Bf16FmaTier)
        {
            for (; row + 3 < m; row += 4) Bf16Rows4Fma(w + (long)row * k, x, y + row, k);
            for (; row < m; row++) y[row] = DotBf16Fma(w + (long)row * k, x, k);
        }
        else if (Bf16V128Tier)
        {
            for (; row + 3 < m; row += 4) Bf16Rows4V128Simd(w + (long)row * k, x, y + row, k);
            for (; row < m; row++) y[row] = DotBf16V128Simd(w + (long)row * k, x, k);
        }
        else
        {
            for (; row < m; row++) y[row] = DotBf16Scalar(w + (long)row * k, x, k);
        }
    }

    /// <summary>GEMM over <paramref name="tileRows"/> rows on the tier in force; <paramref name="scratch"/> (4·k floats) is used by the FMA tier only.</summary>
    [SkipLocalsInit]
    private static void GemmBf16Rows(ushort* tileW, int tileRows, float* b, float* c, int cStride, int k, int n, float* scratch)
    {
        if (Bf16Avx512Tier)
        {
            GemmBf16RowsAvx512(tileW, tileRows, b, c, cStride, k, n);
            return;
        }
        if (Bf16FmaTier)
        {
            if (Bf16FmaScratch) GemmBf16RowsFma(tileW, tileRows, b, c, cStride, k, n, scratch);
            else GemmBf16RowsFmaTile(tileW, tileRows, b, c, cStride, k, n);
            return;
        }
        for (int t = 0; t < n; t++)
            Bf16Rows(tileW, b + (long)t * k, c + (long)t * cStride, tileRows, k);
    }

    /// <summary>Scratch floats per worker the BF16 GEMM needs for a given k.</summary>
    internal static int GemmBf16ScratchFloats(int k) => 4 * k;

    // ──────────────────── Public entry points ────────────────────

    /// <summary>BF16 GEMV: <c>y[m] = W[m,k] · x[k]</c>, weights read as BF16 in place.</summary>
    /// <param name="weights">BF16 weights, row-major [m, k].</param>
    /// <param name="x">Input vector [k].</param>
    /// <param name="y">Output [m].</param>
    /// <param name="m">Rows.</param>
    /// <param name="k">Columns.</param>
    [SkipLocalsInit]
    public static void GemvBf16(nint weights, float* x, float* y, int m, int k)
    {
        if (!Bf16KernelEnabled)
        {
            GemvDequantRows((byte*)weights, QuantizationType.BF16, x, y, m, k);
            return;
        }
        Bf16Rows((ushort*)weights, x, y, m, k);
    }

    /// <summary>BF16 GEMV with optional row parallelism.</summary>
    /// <param name="weights">BF16 weights, row-major [m, k].</param>
    /// <param name="x">Input vector [k].</param>
    /// <param name="y">Output [m].</param>
    /// <param name="m">Rows.</param>
    /// <param name="k">Columns.</param>
    /// <param name="pool">Thread pool, or null.</param>
    [SkipLocalsInit]
    public static void GemvBf16(nint weights, float* x, float* y, int m, int k, ComputeThreadPool? pool)
    {
        if (!Bf16KernelEnabled)
        {
            GemvDequantRows((byte*)weights, QuantizationType.BF16, x, y, m, k, pool);
            return;
        }
        if (pool is null || m < ParallelMinRows)
        {
            Bf16Rows((ushort*)weights, x, y, m, k);
            return;
        }
        var ctx = new GemvBf16Ctx { Weights = (ushort*)weights, X = x, Y = y, M = m, K = k };
        pool.Dispatch((nint)(&ctx), &GemvBf16Worker);
    }

    /// <summary>BF16 GEMM: <c>C[n,m] = B[n,k] x W[m,k]^T</c>, weights read as BF16 in place (serial).</summary>
    /// <param name="weights">BF16 weights, row-major [m, k].</param>
    /// <param name="b">Inputs [n, k].</param>
    /// <param name="c">Outputs [n, m].</param>
    /// <param name="m">Weight rows.</param>
    /// <param name="k">Columns.</param>
    /// <param name="n">Tokens.</param>
    [SkipLocalsInit]
    public static void GemmBf16(nint weights, float* b, float* c, int m, int k, int n)
    {
        if (!Bf16KernelEnabled)
        {
            GemmDequantRows((byte*)weights, QuantizationType.BF16, b, c, m, k, n, null);
            return;
        }
        if (n == 1)
        {
            Bf16Rows((ushort*)weights, b, c, m, k);
            return;
        }
        int tileM = ComputeTileM(k * sizeof(ushort));
        float[] rented = ArrayPool<float>.Shared.Rent(GemmBf16ScratchFloats(k));
        try
        {
            fixed (float* scratch = rented)
            {
                for (int mStart = 0; mStart < m; mStart += tileM)
                {
                    int tileRows = Math.Min(tileM, m - mStart);
                    GemmBf16Rows((ushort*)weights + (long)mStart * k, tileRows, b, c + mStart, m, k, n, scratch);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary>BF16 GEMM with optional tile parallelism (per-worker four-row scratch).</summary>
    /// <param name="weights">BF16 weights, row-major [m, k].</param>
    /// <param name="b">Inputs [n, k].</param>
    /// <param name="c">Outputs [n, m].</param>
    /// <param name="m">Weight rows.</param>
    /// <param name="k">Columns.</param>
    /// <param name="n">Tokens.</param>
    /// <param name="pool">Thread pool, or null.</param>
    [SkipLocalsInit]
    public static void GemmBf16(nint weights, float* b, float* c, int m, int k, int n, ComputeThreadPool? pool)
    {
        if (!Bf16KernelEnabled)
        {
            GemmDequantRows((byte*)weights, QuantizationType.BF16, b, c, m, k, n, pool);
            return;
        }
        if (pool is null)
        {
            GemmBf16(weights, b, c, m, k, n);
            return;
        }
        if (n == 1)
        {
            GemvBf16(weights, b, c, m, k, pool);
            return;
        }

        // Shrink the L2-sized tile until every worker has one (multiple of 4 rows, floor 8), as the F16 path does (#1404).
        int tileM = ComputeTileM(k * sizeof(ushort));
        int perThread = ((m + pool.ThreadCount - 1) / pool.ThreadCount + 3) & ~3;
        tileM = Math.Min(tileM, Math.Max(8, perThread));
        int totalTiles = (m + tileM - 1) / tileM;
        if (totalTiles < 2)
        {
            GemmBf16(weights, b, c, m, k, n);
            return;
        }

        int threadCount = pool.ThreadCount;
        nint* scratchPtrs = stackalloc nint[threadCount];
        int scratchBytes = GemmBf16ScratchFloats(k) * sizeof(float);
        for (int i = 0; i < threadCount; i++)
            scratchPtrs[i] = pool.GetWorkerScratch(i, scratchBytes);

        var ctx = new GemmBf16Ctx
        {
            Weights = (ushort*)weights, B = b, C = c, M = m, K = k, N = n, TileM = tileM, ScratchPtrs = scratchPtrs,
        };
        pool.Dispatch((nint)(&ctx), &GemmBf16Worker);
    }

    private struct GemvBf16Ctx
    {
        public ushort* Weights;
        public float* X;
        public float* Y;
        public int M;
        public int K;
    }

    private struct GemmBf16Ctx
    {
        public ushort* Weights;
        public float* B;
        public float* C;
        public int M;
        public int K;
        public int N;
        public int TileM;
        public nint* ScratchPtrs;
    }

    private static void GemvBf16Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvBf16Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        Bf16Rows(ctx.Weights + (long)start * ctx.K, ctx.X, ctx.Y + start, count, ctx.K);
    }

    private static void GemmBf16Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmBf16Ctx>((void*)ctxPtr);
        int totalTiles = (ctx.M + ctx.TileM - 1) / ctx.TileM;
        ComputeThreadPool.PartitionRange(totalTiles, threadIdx, threadCount, out int startTile, out int endTile);
        float* scratch = (float*)ctx.ScratchPtrs[threadIdx];
        for (int tile = startTile; tile < endTile; tile++)
        {
            int mStart = tile * ctx.TileM;
            int tileRows = Math.Min(ctx.TileM, ctx.M - mStart);
            GemmBf16Rows(ctx.Weights + (long)mStart * ctx.K, tileRows, ctx.B, ctx.C + mStart, ctx.M, ctx.K, ctx.N, scratch);
        }
    }
}
