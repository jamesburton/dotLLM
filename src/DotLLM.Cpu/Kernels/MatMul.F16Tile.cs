using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// AVX2/FMA F16 GEMM tile kernel (#1404). The previous AVX2-tier loop converted every F16 weight
/// row once <em>per token</em> (token-outer, row-inner); this one converts a group of 4 weight rows
/// into a bounded per-thread F32 scratch once, then runs a 4-row x 2-token FMA register tile over
/// it, so conversion cost is amortised over all n tokens and each activation load feeds 4 rows.
/// Weights stay F16 in place; the scratch is 4 rows (not the whole matrix) and is never stored
/// beyond the group.
///
/// <para>Every output element is computed by the same single-accumulator-per-output arithmetic
/// (8-lane FMA over k, fixed horizontal sum, scalar tail) regardless of which tile / token pairing
/// it landed in, so results are independent of thread partitioning and bit-identical between the
/// single-threaded and pooled entry points.</para>
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>Scratch floats needed per worker by <see cref="GemmF16RowsFma"/> for a given k.</summary>
    internal static int GemmF16FmaScratchFloats(int k) => 4 * k;

    private static readonly bool s_f16FmaOff = Environment.GetEnvironmentVariable("DOTLLM_F16_NO_FMA") == "1"; // A/B switch

    /// <summary>True when the converted-tile FMA kernel is usable (AVX + FMA).</summary>
    internal static bool F16FmaTier
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Avx.IsSupported && Fma.IsSupported && !s_f16FmaOff;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HSum256(Vector256<float> v)
    {
        Vector128<float> s = Sse.Add(v.GetLower(), v.GetUpper());
        s = Sse3.HorizontalAdd(s, s);
        s = Sse3.HorizontalAdd(s, s);
        return s.ToScalar();
    }

    /// <summary>Single-output dot with the exact arithmetic of the 4x2 tile (see class remarks).</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float DotF32Fma(float* w, float* x, int k)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        int i = 0;
        for (; i + 7 < k; i += 8)
            acc = Fma.MultiplyAdd(Avx.LoadVector256(w + i), Avx.LoadVector256(x + i), acc);
        float sum = HSum256(acc);
        for (; i < k; i++) sum += w[i] * x[i];
        return sum;
    }

    /// <summary>
    /// 4 converted weight rows (<paramref name="w"/>, row stride <paramref name="k"/>) x 2 tokens
    /// (<paramref name="x0"/>, <paramref name="x1"/>). Outputs go to <c>c0[row]</c>, <c>c1[row]</c>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile4x2F32Fma(float* w, float* x0, float* x1, float* c0, float* c1, int k)
    {
        float* w0 = w, w1 = w + k, w2 = w + 2L * k, w3 = w + 3L * k;
        Vector256<float> a00 = default, a01 = default, a10 = default, a11 = default;
        Vector256<float> a20 = default, a21 = default, a30 = default, a31 = default;
        int i = 0;
        for (; i + 7 < k; i += 8)
        {
            Vector256<float> xa = Avx.LoadVector256(x0 + i);
            Vector256<float> xb = Avx.LoadVector256(x1 + i);
            Vector256<float> v = Avx.LoadVector256(w0 + i);
            a00 = Fma.MultiplyAdd(v, xa, a00); a01 = Fma.MultiplyAdd(v, xb, a01);
            v = Avx.LoadVector256(w1 + i);
            a10 = Fma.MultiplyAdd(v, xa, a10); a11 = Fma.MultiplyAdd(v, xb, a11);
            v = Avx.LoadVector256(w2 + i);
            a20 = Fma.MultiplyAdd(v, xa, a20); a21 = Fma.MultiplyAdd(v, xb, a21);
            v = Avx.LoadVector256(w3 + i);
            a30 = Fma.MultiplyAdd(v, xa, a30); a31 = Fma.MultiplyAdd(v, xb, a31);
        }
        float s00 = HSum256(a00), s01 = HSum256(a01), s10 = HSum256(a10), s11 = HSum256(a11);
        float s20 = HSum256(a20), s21 = HSum256(a21), s30 = HSum256(a30), s31 = HSum256(a31);
        for (; i < k; i++)
        {
            float xa = x0[i], xb = x1[i];
            s00 += w0[i] * xa; s01 += w0[i] * xb;
            s10 += w1[i] * xa; s11 += w1[i] * xb;
            s20 += w2[i] * xa; s21 += w2[i] * xb;
            s30 += w3[i] * xa; s31 += w3[i] * xb;
        }
        c0[0] = s00; c0[1] = s10; c0[2] = s20; c0[3] = s30;
        c1[0] = s01; c1[1] = s11; c1[2] = s21; c1[3] = s31;
    }

    /// <summary>
    /// F16 GEMM over <paramref name="tileRows"/> weight rows on the AVX2/FMA tier. Writes
    /// <c>c[t·cStride + row]</c>. <paramref name="scratch"/> holds <see cref="GemmF16FmaScratchFloats"/> floats.
    /// </summary>
    [SkipLocalsInit]
    internal static void GemmF16RowsFma(Half* tileW, int tileRows, float* b, float* c, int cStride,
        int k, int n, float* scratch)
    {
        for (int r0 = 0; r0 < tileRows; r0 += 4)
        {
            int rr = Math.Min(4, tileRows - r0);
            for (int j = 0; j < rr; j++)
                TensorPrimitives.ConvertToSingle(
                    new ReadOnlySpan<Half>(tileW + (long)(r0 + j) * k, k), new Span<float>(scratch + (long)j * k, k));

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
}
