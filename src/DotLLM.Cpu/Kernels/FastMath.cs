using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Exponential helpers for the attention softmax: a fused shift+exp+store+sum, plus an
/// approximate <c>exp</c> built on IEEE-754 bit manipulation.
/// <para>
/// <b>The approximation is off by default (#501).</b> This class was written on the premise that
/// "errors in exp get normalized away when dividing by the sum". That is false: normalizing fixes
/// the <em>mean</em> of the weights, but the ~1-2% error is <em>relative and per-element</em>, so
/// it survives normalization as a reweighting of the mixture each head computes. It is free on a
/// full-precision model and expensive on a heavily quantized one — Q3_K Llama-3.2-1B pays
/// +1.71% perplexity for it, Q8_0 pays nothing. See <see cref="UseFastExp"/> for the measurements.
/// </para>
/// <para>
/// The core trick (Schraudolph 1999): <c>exp(x) ≈ reinterpret_as_float((int)(x * C0 + C1))</c> where
/// <c>C0 = 2^23 / ln(2)</c> maps x into the IEEE-754 exponent field and <c>C1</c> is a bias constant.
/// This yields ~1-2% max relative error in ~3 SIMD ops vs ~12 for a polynomial approximation.
/// </para>
/// </summary>
public static class FastMath
{
    /// <summary>Scale factor: 2^23 / ln(2). Maps x/ln(2) into the IEEE-754 exponent bit field.</summary>
    private const float C0 = 12102203.0f;

    /// <summary>
    /// Bias constant: (127 - 0.0579) * 2^23. Minimax-optimized for softmax use (minimizes
    /// relative error over [-88, 0] range typical of attention scores after max-subtraction).
    /// </summary>
    private const float C1 = 1064866805.0f;

    /// <summary>Lower clamp: exp(-87.3) ≈ 1.3e-38 (near float min normal). Prevents negative int bits.</summary>
    private const float MinClamp = -87.3f;

    /// <summary>Upper clamp: exp(88.7) ≈ 3.0e+38 (near float max). Prevents overflow.</summary>
    private const float MaxClamp = 88.7f;

    /// <summary>
    /// Opt-in switch (<c>DOTLLM_FAST_EXP=1</c>) that restores the Schraudolph bit-trick
    /// <c>exp</c> on every path in this class. <b>Off by default</b> since #501.
    /// </summary>
    /// <remarks>
    /// <para>The premise "errors in exp get normalized away when dividing by the sum" is only
    /// true for the <em>mean</em> of the attention weights; the ~1-2% <em>relative</em> error is
    /// per-element and does not cancel, so it perturbs which values the head actually mixes.</para>
    /// <para>Measured on <c>Llama-3.2-1B-pure</c>, wikitext-2, ctx 512, 40 chunks, CPU, on
    /// llama.cpp's exact token stream — paired per-chunk, accurate minus approximate:</para>
    /// <list type="bullet">
    ///   <item><description><b>Q8_0</b>: +0.00042 +/- 0.00059 nats (t = +0.7) — no effect.
    ///   This is why the approximation looked free: it is, on a full-precision model.</description></item>
    ///   <item><description><b>Q3_K</b>: <b>-0.01724 +/- 0.00185 nats</b> (t = -9.3), i.e. -1.71%
    ///   perplexity. Against llama.cpp the gap goes from +2.03% (t = +5.7) to +0.28% (t = +0.8,
    ///   not significant).</description></item>
    /// </list>
    /// <para>A 3-bit model's attention scores sit closer together, so a 1-2% reweighting changes
    /// the mixture materially. The cost scales with how damaged the model is — exactly the
    /// regime aggressive quantization exists to serve — so the approximation is kept only as an
    /// opt-in benchmarking lever.</para>
    /// </remarks>
    internal static readonly bool UseFastExp =
        Environment.GetEnvironmentVariable("DOTLLM_FAST_EXP") == "1";

    /// <summary>
    /// Scalar fast approximate exp. ~1-2% max relative error.
    /// Clamped to [-87.3, 88.7] for general use.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastExp(float x)
    {
        x = Math.Clamp(x, MinClamp, MaxClamp);
        if (!UseFastExp) return MathF.Exp(x);
        int bits = (int)(x * C0 + C1);
        return Unsafe.BitCast<int, float>(bits);
    }

    /// <summary>
    /// Fused shift + fast exp + store + sum in a single pass.
    /// Computes: <c>output[i] = fast_exp(input[i] + offset)</c>, returns <c>sum(output)</c>.
    /// Replaces separate <c>TensorPrimitives.Add + Exp + Sum</c> with one pass over the data.
    /// </summary>
    /// <param name="input">Input span (e.g., attention scores for one tile).</param>
    /// <param name="output">Output span. May alias <paramref name="input"/> for in-place operation.</param>
    /// <param name="offset">Additive offset (typically <c>-max</c> for numerical stability).</param>
    /// <returns>Sum of all exponentiated values.</returns>
    /// <remarks>
    /// The vectorized paths only clamp the lower bound (<c>-87.3f</c>). This is safe because the intended
    /// use is attention softmax where <paramref name="offset"/> is <c>-max(input)</c>, guaranteeing
    /// <c>input[i] + offset ≤ 0</c>. Callers passing a positive offset must ensure
    /// <c>input[i] + offset ≤ 88.7f</c> to avoid integer overflow in the bit trick.
    /// </remarks>
    [SkipLocalsInit]
    public static float ExpSumAndStore(ReadOnlySpan<float> input, Span<float> output, float offset)
    {
        if (!UseFastExp)
            return ExpSumAndStoreAccurate(input, output, offset);

        int length = input.Length;
        ref float src = ref MemoryMarshal.GetReference(input);
        ref float dst = ref MemoryMarshal.GetReference(output);

        if (Avx512F.IsSupported)
            return ExpSumAndStoreAvx512(ref src, ref dst, length, offset);

        if (Avx2.IsSupported)
            return ExpSumAndStoreAvx2(ref src, ref dst, length, offset);

        return ExpSumAndStoreScalar(ref src, ref dst, length, offset);
    }

    /// <summary>
    /// Accurate <c>exp(input + offset)</c> + store + sum, via <see cref="TensorPrimitives"/>.
    /// </summary>
    /// <remarks>
    /// Three vectorized passes over one attention tile rather than the bit trick's one, which is
    /// affordable because a tile is sized to stay in L1 — the passes hit cache, not memory. The
    /// lower clamp is kept so a <c>-infinity</c> running max (an empty online-softmax
    /// accumulator) still produces <c>~0</c> rather than a NaN downstream.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ExpSumAndStoreAccurate(ReadOnlySpan<float> input, Span<float> output, float offset)
    {
        Span<float> dst = output[..input.Length];
        TensorPrimitives.Add(input, offset, dst);
        TensorPrimitives.Max(dst, MinClamp, dst);
        TensorPrimitives.Exp(dst, dst);
        return TensorPrimitives.Sum(dst);
    }

    /// <summary>
    /// Fast approximate softmax using IEEE-754 bit-manipulation exp.
    /// For attention scores where full precision is unnecessary.
    /// Standard <see cref="Softmax.Execute"/> should be used for sampling softmax.
    /// </summary>
    /// <param name="input">Input span (logits/scores).</param>
    /// <param name="result">Destination span. May alias <paramref name="input"/> for in-place operation.</param>
    [SkipLocalsInit]
    public static void Softmax(ReadOnlySpan<float> input, Span<float> result)
    {
        float max = TensorPrimitives.Max(input);
        float sum = ExpSumAndStore(input, result, -max);
        TensorPrimitives.Multiply(result, 1.0f / sum, result);
    }

    // ──────────────────── AVX-512 path ────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ExpSumAndStoreAvx512(ref float src, ref float dst, int length, float offset)
    {
        var offsetVec = Vector512.Create(offset);
        var c0Vec = Vector512.Create(C0);
        var c1Vec = Vector512.Create(C1);
        var minVec = Vector512.Create(MinClamp);
        var sumVec = Vector512<float>.Zero;

        nuint i = 0;
        nuint vecLen = (nuint)(length & ~15); // 16 floats per iteration

        for (; i < vecLen; i += 16)
        {
            var x = Vector512.LoadUnsafe(ref src, i);
            var shifted = Avx512F.Add(x, offsetVec);
            shifted = Avx512F.Max(shifted, minVec);
            var y = Avx512F.FusedMultiplyAdd(shifted, c0Vec, c1Vec);
            var bits = Avx512F.ConvertToVector512Int32WithTruncation(y);
            var exp = bits.AsSingle();
            Vector512.StoreUnsafe(exp, ref dst, i);
            sumVec = Avx512F.Add(sumVec, exp);
        }

        float sum = Vector512.Sum(sumVec);

        // Scalar tail
        for (; i < (nuint)length; i++)
        {
            float val = Unsafe.Add(ref src, i) + offset;
            val = MathF.Max(val, MinClamp);
            int bits = (int)(val * C0 + C1);
            float exp = Unsafe.BitCast<int, float>(bits);
            Unsafe.Add(ref dst, i) = exp;
            sum += exp;
        }

        return sum;
    }

    // ──────────────────── AVX2 path ────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ExpSumAndStoreAvx2(ref float src, ref float dst, int length, float offset)
    {
        var offsetVec = Vector256.Create(offset);
        var c0Vec = Vector256.Create(C0);
        var c1Vec = Vector256.Create(C1);
        var minVec = Vector256.Create(MinClamp);
        var sumVec = Vector256<float>.Zero;

        nuint i = 0;
        nuint vecLen = (nuint)(length & ~7); // 8 floats per iteration

        for (; i < vecLen; i += 8)
        {
            var x = Vector256.LoadUnsafe(ref src, i);
            var shifted = Avx.Add(x, offsetVec);
            shifted = Avx.Max(shifted, minVec);
            Vector256<float> y;
            if (Fma.IsSupported)
                y = Fma.MultiplyAdd(shifted, c0Vec, c1Vec);
            else
                y = Avx.Add(Avx.Multiply(shifted, c0Vec), c1Vec);
            var bits = Avx.ConvertToVector256Int32WithTruncation(y);
            var exp = bits.AsSingle();
            Vector256.StoreUnsafe(exp, ref dst, i);
            sumVec = Avx.Add(sumVec, exp);
        }

        float sum = Vector256.Sum(sumVec);

        // Scalar tail
        for (; i < (nuint)length; i++)
        {
            float val = Unsafe.Add(ref src, i) + offset;
            val = MathF.Max(val, MinClamp);
            int bits = (int)(val * C0 + C1);
            float exp = Unsafe.BitCast<int, float>(bits);
            Unsafe.Add(ref dst, i) = exp;
            sum += exp;
        }

        return sum;
    }

    // ──────────────────── Scalar fallback ────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ExpSumAndStoreScalar(ref float src, ref float dst, int length, float offset)
    {
        float sum = 0f;

        for (int i = 0; i < length; i++)
        {
            float val = Unsafe.Add(ref src, (nuint)i) + offset;
            float exp = FastExp(val);
            Unsafe.Add(ref dst, (nuint)i) = exp;
            sum += exp;
        }

        return sum;
    }
}
