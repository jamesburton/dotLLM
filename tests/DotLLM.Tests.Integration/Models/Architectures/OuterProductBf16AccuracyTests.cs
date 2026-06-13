using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.Evaluation;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Same-process A/B accuracy test for the Q8_0 <b>BF16</b> outer-product prefill kernel against the
/// exact-integer outer-product kernel. Runs the same model and prompt twice — once with
/// <see cref="TransformerModel.UseBf16OuterProductQ8Prefill"/> off (exact int) and once on (bf16) —
/// then reports the full-logits divergence and the perplexity of each path on a fixed passage.
/// </summary>
/// <remarks>
/// <para>
/// <b>This is an approximation comparison, not a parity check.</b> The bf16 kernel rounds each scaled
/// Q8_0 value to an 8-bit mantissa, so its logits differ from the integer path by genuine bf16
/// rounding error — not merely FP summation order. The assertion is therefore a <i>relative</i>
/// tolerance bound, deliberately looser than the integer-parity test's 1e-3.
/// </para>
/// <para>
/// <b>Hardware gate.</b> The bf16 kernel is <c>#if NET11_0_OR_GREATER</c> and requires
/// <see cref="Avx512Bf16.IsSupported"/> (Zen4/Zen5, Strix). On any other target — including the AVX2
/// CI/dev box — the bf16 flag is a silent no-op fallback to the integer kernel, so the comparison
/// would be vacuous (maxRelDiff ≈ 0, identical perplexity). These tests therefore <b>skip</b> unless
/// AVX512-BF16 is present; they only run on the target hardware where the bf16 answer is meaningful.
/// </para>
/// <para>
/// <b>Discriminating guard.</b> When the test does run, it asserts the bf16 tile counter strictly
/// increased — proof the bf16 microkernel actually executed and the comparison is not silently the
/// integer path against itself (which would falsely report "bf16 is perfectly accurate").
/// </para>
/// <para>
/// <b>Coverage caveat.</b> Only the AVX-512 4×6 tile has a bf16 variant, so only the
/// <c>(N / 6) * 6</c> token positions per group carry bf16 rounding; the <c>N % 6</c> 3-token
/// remainder and single-token tails are computed with the exact integer kernels. The reported
/// perplexity delta therefore reflects bf16 for the 4×6-tiled positions, not literally every token.
/// A prompt must tokenize to at least 6 tokens or the 4×6 tile never engages and the tile-counter
/// guard fails — keep that in mind when adding model variants (e.g. a Llama-3.2-1B class).
/// </para>
/// </remarks>
[Collection("SmallModel")]
public class OuterProductBf16AccuracyTests
{
    private readonly SmallModelFixture _fixture;

    public OuterProductBf16AccuracyTests(SmallModelFixture fixture)
    {
        _fixture = fixture;
    }

    [SkippableTheory]
    [InlineData("The capital of France is Paris and the weather today")]
    public void Bf16OuterProduct_WithinToleranceOfInteger_AndReportsPerplexity(string prompt)
        => OuterProductBf16Accuracy.AssertWithinToleranceOfInteger(_fixture.FilePath, prompt);
}

/// <summary>
/// Shared driver for the BF16-vs-integer Q8_0 outer-product A/B accuracy measurement, reusable across
/// models (SmolLM-135M now; Llama-3.2-1B can be added by pointing a second test class at its fixture).
/// </summary>
internal static class OuterProductBf16Accuracy
{
    // A short fixed passage for the perplexity A/B. Tokenizes to a multi-token corpus so prefill runs
    // with n > 1 (the only case the outer-product path handles), and is long enough to score several
    // targets through PerplexityEvaluator.
    private const string PerplexityPassage =
        "The quick brown fox jumps over the lazy dog. " +
        "Paris is the capital of France and a major European city. " +
        "Machine learning models predict the next token from prior context.";

    /// <summary>
    /// True only when the bf16 outer-product kernel can actually execute: net11 build AND AVX512-BF16
    /// hardware. <c>Avx512Bf16</c> does not exist in net10's reference assemblies, so the capability
    /// probe must itself be <c>#if</c>-guarded — on net10 this is always false and the test skips.
    /// </summary>
    private static bool Bf16KernelAvailable =>
#if NET11_0_OR_GREATER
        System.Runtime.Intrinsics.X86.Avx512Bf16.IsSupported;
#else
        false;
#endif

    /// <summary>
    /// Loads <paramref name="ggufPath"/>, runs <paramref name="prompt"/> through the integer and bf16
    /// Q8_0 outer-product prefill paths in one process, asserts the bf16 logits stay within a relative
    /// tolerance of the integer baseline, and reports both perplexity numbers plus the delta. Skips
    /// entirely when AVX512-BF16 is unavailable (the only configuration where bf16 actually executes).
    /// </summary>
    /// <param name="ggufPath">Path to a Q8_0 GGUF model file.</param>
    /// <param name="prompt">A prompt that tokenizes to more than one token (so prefill runs with n &gt; 1).</param>
    public static void AssertWithinToleranceOfInteger(string ggufPath, string prompt)
    {
        // Gate on the actual hardware/runtime capability. On net10 or non-AVX512-BF16 hardware the
        // bf16 flag is a no-op, so running the comparison would prove nothing — skip loudly instead.
        Skip.IfNot(Bf16KernelAvailable,
            "AVX512-BF16 not available (or not built for net11) — the bf16 outer-product kernel falls " +
            "back to the integer path, making this A/B comparison vacuous. Runs only on Zen4/Zen5/Strix.");

        var gguf = GgufFile.Open(ggufPath);
        using var _ = gguf;
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        // Parallel threading exercises the parallel outer-product worker (the production path).
        using var model = TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.Auto);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int[] tokenIds = tokenizer.Encode(prompt);
        Assert.True(tokenIds.Length > 1,
            $"Need a multi-token prompt to exercise prefill (n>1); got {tokenIds.Length} tokens.");

        int[] positions = new int[tokenIds.Length];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = i;

        int vocabSize = model.Config.VocabSize;

        // --- Baseline: exact-integer outer-product path ---
        model.UseOuterProductQ8Prefill = true;
        model.UseBf16OuterProductQ8Prefill = false;
        float[] integerLogits;
        int logitCount;
        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1))
        {
            logitCount = (int)logits.ElementCount;
            Assert.Equal(tokenIds.Length * vocabSize, logitCount);
            integerLogits = new float[logitCount];
            unsafe
            {
                new ReadOnlySpan<float>((void*)logits.DataPointer, logitCount).CopyTo(integerLogits);
            }
        }

        // --- BF16 outer-product path ---
        long bf16TilesBefore = MatMul.OuterProductQ8_0Avx512Bf16TileCount;
        model.UseBf16OuterProductQ8Prefill = true;
        float[] bf16Logits = new float[logitCount];
        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1))
        {
            Assert.Equal(logitCount, (int)logits.ElementCount);
            unsafe
            {
                new ReadOnlySpan<float>((void*)logits.DataPointer, logitCount).CopyTo(bf16Logits);
            }
        }
        long bf16TilesAfter = MatMul.OuterProductQ8_0Avx512Bf16TileCount;

        // Proof the bf16 microkernel actually executed: a silent fallback to the integer kernel would
        // make maxRelDiff ≈ 0 and the test pass while reporting "bf16 perfectly accurate" — exactly the
        // vacuous result this guard exists to prevent.
        Assert.True(bf16TilesAfter - bf16TilesBefore > 0,
            $"BF16 4×6 tile never executed during the flag-on run (delta={bf16TilesAfter - bf16TilesBefore}); " +
            "the bf16 path silently fell back to the integer kernel, so the accuracy comparison is vacuous.");

        // --- Compare full logits ---
        // Tolerance: assert on the VECTOR (tile-max) scale, not per-element relative. bf16 carries an
        // 8-bit mantissa (~2^-8 ≈ 4e-3 relative per rounded value); errors partly cancel over long-K
        // dot products. A per-element relative bound is the wrong metric here: logits are differences
        // of large-magnitude sums, so a small-magnitude logit (a fraction of the vector scale) can show
        // a large *relative* error for a perfectly correct kernel — that produces spurious failures.
        // The robust metric is the worst absolute deviation normalized by the largest baseline logit
        // magnitude: maxAbsDiff / maxBaselineAbsLogit ≤ 5e-2. That fails loudly on a genuinely broken
        // kernel (mis-packed lanes / wrong scale fold blow far past 5e-2) without punishing correct bf16
        // rounding on near-zero logits. Per-element maxRelDiff is still computed and REPORTED as a
        // diagnostic, just not used as the gate.
        const float scaleRelTol = 5e-2f;
        const float absFloor = 1e-2f;

        float maxAbsDiff = 0f;
        float maxRelDiff = 0f;
        float maxBaselineAbs = 0f;
        int worstIdx = -1;
        double sumSqDiff = 0;
        for (int i = 0; i < logitCount; i++)
        {
            float a = integerLogits[i];
            float b = bf16Logits[i];
            Assert.True(float.IsFinite(a) && float.IsFinite(b),
                $"Non-finite logit at index {i}: integer={a}, bf16={b}");

            float absDiff = MathF.Abs(a - b);
            float denom = MathF.Max(MathF.Abs(a), MathF.Abs(b));
            float relDiff = denom > absFloor ? absDiff / denom : absDiff;

            if (relDiff > maxRelDiff)
            {
                maxRelDiff = relDiff;
                worstIdx = i;
            }
            maxAbsDiff = MathF.Max(maxAbsDiff, absDiff);
            maxBaselineAbs = MathF.Max(maxBaselineAbs, MathF.Abs(a));
            sumSqDiff += (double)absDiff * absDiff;
        }
        double rmsDiff = Math.Sqrt(sumSqDiff / logitCount);
        // Normalize the worst absolute deviation by the vector (tile-max) scale.
        float scaleNormDiff = maxBaselineAbs > absFloor ? maxAbsDiff / maxBaselineAbs : maxAbsDiff;

        // --- Perplexity A/B/C on a fixed passage ---
        // Three paths on the SAME passage so the bf16 cost is anchored against the TRUE baseline:
        //   inner   = flag off entirely (inner-product / cache-tiled reduction) — the production default
        //   integer = exact-integer outer-product (R4 maddubs microkernel)
        //   bf16    = bf16 outer-product (VDPBF16PS microkernel)
        // The headline "bf16 accuracy cost" is bf16-vs-inner; integer-vs-inner proves the outer-product
        // restructuring alone (no bf16) introduces no quality regression — without it, a bf16 delta
        // measured only against integer-outer would hide any cost the outer-product path already carries.
        int[] passageTokens = tokenizer.Encode(PerplexityPassage);
        model.UseOuterProductQ8Prefill = false;
        var innerPpl = PerplexityEvaluator.Evaluate(model, passageTokens);
        model.UseOuterProductQ8Prefill = true;
        model.UseBf16OuterProductQ8Prefill = false;
        var intPpl = PerplexityEvaluator.Evaluate(model, passageTokens);
        model.UseBf16OuterProductQ8Prefill = true;
        var bf16Ppl = PerplexityEvaluator.Evaluate(model, passageTokens);
        // bf16 cost anchored against the true inner-product baseline (the production default path).
        double pplDelta = bf16Ppl.Perplexity - innerPpl.Perplexity;
        double pplRelDelta = innerPpl.Perplexity != 0 ? pplDelta / innerPpl.Perplexity : pplDelta;
        // Outer-product restructuring (no bf16) vs inner-product — should be ~0 (no quality regression).
        double intInnerDelta = intPpl.Perplexity - innerPpl.Perplexity;
        double intInnerRelDelta = innerPpl.Perplexity != 0 ? intInnerDelta / innerPpl.Perplexity : intInnerDelta;

        string worst = worstIdx >= 0
            ? $"at index {worstIdx} (integer={integerLogits[worstIdx]}, bf16={bf16Logits[worstIdx]})"
            : "(vectors bit-identical)";

        // Surface every number — the key deliverable is the runnable A/B accuracy answer.
        Console.WriteLine(
            $"[OuterProductBf16Accuracy] model={System.IO.Path.GetFileName(ggufPath)} " +
            $"tokens={tokenIds.Length} logits={logitCount} bf16Tiles={bf16TilesAfter - bf16TilesBefore}");
        Console.WriteLine(
            $"  logits: maxAbsDiff={maxAbsDiff:E3} scaleNormDiff={scaleNormDiff:E3} (vec-scale gate) " +
            $"maxRelDiff={maxRelDiff:E3} (per-elem, diagnostic) rmsDiff={rmsDiff:E3} " +
            $"maxBaselineAbs={maxBaselineAbs:E3} {worst}");
        Console.WriteLine(
            $"  perplexity: inner={innerPpl.Perplexity:F6} integer={intPpl.Perplexity:F6} bf16={bf16Ppl.Perplexity:F6} " +
            $"(scored={innerPpl.ScoredTokenCount} tokens)");
        Console.WriteLine(
            $"  integer-vs-inner: delta={intInnerDelta:+0.000000;-0.000000} relDelta={intInnerRelDelta:+0.######%;-0.######%} (outer-product restructuring cost)");
        Console.WriteLine(
            $"  bf16-vs-inner:    delta={pplDelta:+0.000000;-0.000000} relDelta={pplRelDelta:+0.######%;-0.######%} (headline bf16 accuracy cost)");

        // Gate on the vector (tile-max) scale, per the task's "≤5e-2 relative on the tile-max scale".
        Assert.True(scaleNormDiff <= scaleRelTol,
            $"BF16 outer-product logits diverged from the integer baseline beyond tolerance. " +
            $"scaleNormDiff={scaleNormDiff:E3} (tol={scaleRelTol:E3}, = maxAbsDiff/maxBaselineAbs) " +
            $"maxAbsDiff={maxAbsDiff:E3} maxBaselineAbs={maxBaselineAbs:E3} maxRelDiff={maxRelDiff:E3} rmsDiff={rmsDiff:E3} {worst}.");

        // No-quality-regression gate for the outer-product restructuring itself (no bf16): the integer
        // outer-product and the inner-product baseline both match exact scalar Q8_0 truth, so their
        // perplexities differ only by accumulated FP rounding — bounded well under 1% on this passage.
        // A genuine restructuring bug (mis-packed lanes, wrong reduction) blows far past this.
        const double outerRestructuringPplTol = 0.01;   // 1% relative perplexity
        Assert.True(Math.Abs(intInnerRelDelta) <= outerRestructuringPplTol,
            $"Integer outer-product perplexity regressed from the inner-product baseline beyond FP-rounding " +
            $"tolerance: inner={innerPpl.Perplexity:F6} integer={intPpl.Perplexity:F6} " +
            $"relDelta={intInnerRelDelta:+0.######%;-0.######%} (tol={outerRestructuringPplTol:P0}). " +
            "The outer-product restructuring changed model quality — this is a kernel/integration bug, not bf16 cost.");
    }

    /// <summary>
    /// Inner / integer / bf16 perplexity over a longer corpus (not just the short fixed passage), so the
    /// bf16-vs-inner accuracy cost averages out sample noise. Compares the three prefill reductions on the
    /// identical token stream — only the relative deltas are meaningful (absolute perplexity is irrelevant
    /// to an A/B/C on the same tokens). Skips when AVX512-BF16 is unavailable.
    /// </summary>
    /// <param name="ggufPath">Path to a Q8_0 GGUF model file.</param>
    /// <param name="corpus">A multi-paragraph passage; scored over the model's context window.</param>
    public static void AssertPerplexityOverCorpus(string ggufPath, string corpus)
    {
        Skip.IfNot(Bf16KernelAvailable,
            "AVX512-BF16 not available (or not built for net11) — bf16 falls back to the integer path, " +
            "making this A/B/C vacuous. Runs only on Zen4/Zen5/Strix.");

        var gguf = GgufFile.Open(ggufPath);
        using var _ = gguf;
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.Auto);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int[] tokens = tokenizer.Encode(corpus);
        Assert.True(tokens.Length >= 64,
            $"Corpus too short for a robust A/B/C (got {tokens.Length} tokens); supply a longer passage.");

        model.UseOuterProductQ8Prefill = false;
        var innerPpl = PerplexityEvaluator.Evaluate(model, tokens);
        model.UseOuterProductQ8Prefill = true;
        model.UseBf16OuterProductQ8Prefill = false;
        var intPpl = PerplexityEvaluator.Evaluate(model, tokens);
        model.UseBf16OuterProductQ8Prefill = true;
        var bf16Ppl = PerplexityEvaluator.Evaluate(model, tokens);

        double bf16Rel = innerPpl.Perplexity != 0 ? (bf16Ppl.Perplexity - innerPpl.Perplexity) / innerPpl.Perplexity : 0;
        double intRel = innerPpl.Perplexity != 0 ? (intPpl.Perplexity - innerPpl.Perplexity) / innerPpl.Perplexity : 0;

        Console.WriteLine(
            $"[Bf16PerplexityCorpus] model={System.IO.Path.GetFileName(ggufPath)} corpusTokens={tokens.Length} " +
            $"scored={innerPpl.ScoredTokenCount}");
        Console.WriteLine(
            $"  perplexity: inner={innerPpl.Perplexity:F6} integer={intPpl.Perplexity:F6} bf16={bf16Ppl.Perplexity:F6}");
        Console.WriteLine(
            $"  integer-vs-inner: relDelta={intRel:+0.######%;-0.######%} (outer-product restructuring cost)");
        Console.WriteLine(
            $"  bf16-vs-inner:    relDelta={bf16Rel:+0.######%;-0.######%} (headline bf16 accuracy cost, longer corpus)");

        // Restructuring must stay quality-neutral; bf16 is an approximation so it gets a looser bound but
        // still must not blow up (a real bf16 bug — wrong scale fold / mis-packed lanes — degrades ppl
        // by many percent, not a fraction of one).
        Assert.True(Math.Abs(intRel) <= 0.01,
            $"Integer outer-product perplexity regressed from inner over the corpus: relDelta={intRel:+0.######%;-0.######%} (tol=1%).");
        Assert.True(Math.Abs(bf16Rel) <= 0.05,
            $"bf16 perplexity diverged from inner beyond 5% over the corpus: relDelta={bf16Rel:+0.######%;-0.######%} — likely a bf16 kernel bug, not rounding.");
    }
}
