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
/// <b>Audit #527: "+1.71%" is a pure quant-ladder number with a null Q8_0 control and no
/// shipping-grade row — read it as "the cost is unmeasured on shipping quants", not as a size.</b>
/// <b>Measured since (#531, 2026-09-24): on Q4_K_M the cost is -0.031% (t = -0.47, 95% CI
/// [-0.163%, +0.101%], n = 64) — not resolvable at that sample size and bounded below ~0.16%.
/// The +1.71% on Q3_K reproduces exactly, but its F32-decoded control moves by the same
/// +1.73%, so the penalty tracks how <em>degraded</em> the weights are and not the quantized
/// path.</b> "Expensive on a heavily quantized one" above is therefore the wrong axis: it is
/// expensive on a heavily <em>damaged</em> one.
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
    /// <para><b>Audit #527: the -1.71% is not a shipping-grade quality figure, and the
    /// "+2.03% -> +0.28% against llama.cpp" line above is not a valid engine claim.</b> Both arms
    /// of the A/B are dotLLM on the same tokens, so the paired significance (t = -9.3) is real;
    /// what is not established is the size on anything anyone ships. <c>Llama-3.2-1B-pure</c> Q3_K
    /// is a pure quant-ladder fixture, and a degraded model amplifies a fixed difference by one to
    /// two orders of magnitude (the same engine delta measures +0.029% / +0.359% / +2.319% on
    /// Q8_0- / Q3_K- / Q2_K-derived weights). The Q8_0 control here is null; no shipping-grade
    /// quant (Q4_K_M / Q5_K_M / Q6_K) was measured. The engine-vs-engine numbers additionally
    /// predate #516 and carry the BOS caveat: a <c>--tokens-file</c> stream holds BOS at index 0
    /// only, so unless <c>--bos</c> was also passed every chunk but the first was scored with no
    /// attention sink - a softmax-regime confound in a softmax-precision experiment. See
    /// <c>docs/PERPLEXITY.md</c>, "How to measure quality against llama.cpp, then". The
    /// default-OFF decision is unaffected: it rests on the absence of a throughput benefit.</para>
    /// <para><b>#531 (2026-09-24) settled both halves of that paragraph, and the BOS half was
    /// wrong.</b> The 2026-09-23 run <em>did</em> pass <c>--bos</c> over llama.cpp's own ids; its
    /// stream was aligned. What it was not is the Q3_K kernel this engine runs today. Re-run on
    /// <c>dev</c> with plain <c>--corpus</c> (BOS-aligned since #516), 40-window prefix of a
    /// 64-window sweep: the <c>Q3_K-decoded-F32</c> fixture returns <b>22.2692 / 21.8886</b> and
    /// window 0 = <b>11.763354</b> — the two rows above and #501's quoted window 0, to every
    /// printed digit — while the packed <c>Q3_K</c> fixture returns <b>22.5090 / 22.1019</b>.
    /// So the "+2.03% -> +0.28%" line compares dotLLM running <em>F32-decoded weights</em>
    /// against llama.cpp running real packed Q3_K. The like-for-like packed figure is
    /// <b>+1.03%</b> (64 chunks) / <b>+1.325%</b> (564) and <c>docs/PERPLEXITY.md</c> is the
    /// aligned source for it; +0.28% belongs beside that document's decoded-F32 rows (+0.36% /
    /// +0.681%), not beside its packed ones.</para>
    /// <para><b>The shipping-grade measurement #527 asked for</b> (#531; Llama-3.2-1B Q4_K_M
    /// requantized from the healthy Q8_0-derived F32 model, wikitext-2 LF, ctx 512, 64 windows,
    /// plain <c>--corpus</c>, paired per window, 63 df, fast minus accurate):</para>
    /// <list type="bullet">
    ///   <item><description><b>Q4_K_M</b>: -0.00031 +/- 0.00066 nats (t = -0.47), -0.031% PPL,
    ///   95% CI [-0.163%, +0.101%] — <b>not resolvable at n = 64</b>, bounded below ~0.16%.</description></item>
    ///   <item><description><b>Q4_K_M decoded to F32</b> (control): +0.00012 +/- 0.00037 nats
    ///   (t = +0.32) — the control agrees with the quantized row, so nothing is hiding in the
    ///   K-quant path.</description></item>
    ///   <item><description><b>Q3_K</b>: +0.01693 +/- 0.00178 nats (t = +9.50), +1.708% — the
    ///   -1.71% above, reproduced under the aligned protocol.</description></item>
    ///   <item><description><b>Q3_K decoded to F32</b> (control): +0.01719 +/- 0.00136 nats
    ///   (t = +12.61), +1.734% — <b>the control moves as much as the quantized row.</b> There is
    ///   no quantized matmul in it at all, so the penalty is not a property of Q3_K; it is a
    ///   property of weights degraded to PPL ~23.</description></item>
    /// </list>
    /// <para>Full working: <c>.docs/measurements/2026-09-24-531-fastexp-shipping-grade.md</c>.</para>
    /// <para>A 3-bit model's attention scores sit closer together, so a 1-2% reweighting changes
    /// the mixture materially. The cost scales with how damaged the model is — exactly the
    /// regime aggressive quantization exists to serve — so the approximation is kept only as an
    /// opt-in benchmarking lever. (#531 confirms the "how damaged" part and removes the
    /// "how quantized" part: a shipping Q4_K_M is damaged too little to pay anything measurable.)</para>
    /// <para><b>This lever is CPU-only.</b> #501 also removed the mirrored <c>fast_exp_neg</c>
    /// from the CUDA attention kernels, and those ship as precompiled PTX with no equivalent
    /// switch. Setting <c>DOTLLM_FAST_EXP=1</c> therefore makes the CPU and CUDA backends diverge
    /// by roughly the approximation's own error (~1%, ~5e-3 abs on attention output) — useful as
    /// a deliberate discriminator, but do not run cross-backend parity with it set.</para>
    /// </remarks>
    internal static readonly bool UseFastExp =
        Environment.GetEnvironmentVariable("DOTLLM_FAST_EXP") == "1";

    /// <summary>
    /// Scalar exp, clamped to [-87.3, 88.7]. Precise (<see cref="MathF.Exp"/>) by default since
    /// #501; the Schraudolph bit trick (~1-2% max relative error) only runs under
    /// <see cref="UseFastExp"/>.
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
    /// Fused shift + exp + store + sum in a single pass.
    /// Computes: <c>output[i] = exp(input[i] + offset)</c>, returns <c>sum(output)</c>.
    /// Replaces separate <c>TensorPrimitives.Add + Exp + Sum</c> with one pass over the data.
    /// The exp is precise since #501; <c>DOTLLM_FAST_EXP=1</c> selects the Schraudolph
    /// bit-trick variants below (see <see cref="UseFastExp"/>).
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
