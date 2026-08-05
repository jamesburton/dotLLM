using DotLLM.Core.Configuration;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Quantization;

/// <summary>
/// The runner must be deterministic on one backend before it can be trusted to compare two (#256).
/// </summary>
// Category=Fixtures: these cases need the local ~/.dotllm/quant-ladder/ fixtures but run entirely
// on QuantGateBackend.Cpu, so they must not be tagged GPU — that category means "requires an
// NVIDIA GPU" (README.md / CONTRIBUTING.md), and a contributor filtering "Category!=GPU" on a
// CPU box would silently lose this gate.
[Trait("Category", "Fixtures")]
[Collection("QuantLadder")]
public sealed class QuantGateBackendRunnerTests
{
    private const int CorpusTokens = 512;

    // The gate's own set, not a restatement: these tests exist to say the runner is trustworthy on
    // the inputs the gate actually uses, and a private copy would drift silently.
    private static readonly string[] DecodePrompts = CrossBackendQuantGateTests.DecodePrompts;
    private static int DecodeSteps => DecodePrompts.Length;

    private readonly QuantLadderFixture _ladder;

    /// <summary>Receives the shared ladder index from the <c>QuantLadder</c> collection.</summary>
    /// <param name="ladder">Ladder index shared across the gate's test classes.</param>
    public QuantGateBackendRunnerTests(QuantLadderFixture ladder) => _ladder = ladder;

    /// <summary>
    /// Two identical CPU runs must agree exactly. If the harness is not repeatable against itself,
    /// any cross-backend difference it reports is uninterpretable.
    /// </summary>
    [SkippableFact]
    public void Run_IsDeterministicOnOneBackend()
    {
        QuantLadderEntry? entry = RequireQ8Fixture();

        QuantGateRun a = QuantGateBackendRunner.Run(
            entry!, QuantGateBackend.Cpu, QuantGateCorpus.Path, CorpusTokens, DecodePrompts);
        QuantGateRun b = QuantGateBackendRunner.Run(
            entry!, QuantGateBackend.Cpu, QuantGateCorpus.Path, CorpusTokens, DecodePrompts);

        Assert.Equal(a.Perplexity.MeanNegativeLogLikelihood, b.Perplexity.MeanNegativeLogLikelihood, 12);
        Assert.Equal(a.DecodeTokens, b.DecodeTokens);
        Assert.Equal(a.KvDecodeTokens, b.KvDecodeTokens);

        // Tokens alone do not test the kernels. Argmax is invariant to any perturbation smaller than
        // the top-2 logit gap, so a nondeterministic GEMV — the exact kernel the cached leg exists
        // to cover — would emit identical tokens and pass. Compare the vectors, on both legs.
        AssertBitIdentical("uncached", a.DecodeLogits, b.DecodeLogits);
        AssertBitIdentical("cached", a.KvDecodeLogits, b.KvDecodeLogits);
    }

    /// <summary>Asserts two runs' logit rows are bit-identical.</summary>
    /// <param name="leg">Leg name for failure text.</param>
    /// <param name="left">First run's rows.</param>
    /// <param name="right">Second run's rows.</param>
    private static void AssertBitIdentical(string leg, float[][] left, float[][] right)
    {
        Assert.Equal(left.Length, right.Length);
        for (int step = 0; step < left.Length; step++)
        {
            float[] l = left[step];
            float[] r = right[step];
            Assert.Equal(l.Length, r.Length);
            for (int v = 0; v < l.Length; v++)
            {
                Assert.True(
                    l[v].Equals(r[v]),
                    $"{leg} step {step}, logit {v}: {l[v]:R} != {r[v]:R} across two identical runs.");
            }
        }
    }

    /// <summary>All three legs must actually produce data; an empty leg would assert nothing.</summary>
    [SkippableFact]
    public void Run_ProducesAllThreeLegs()
    {
        QuantLadderEntry? entry = RequireQ8Fixture();

        QuantGateRun run = QuantGateBackendRunner.Run(
            entry!, QuantGateBackend.Cpu, QuantGateCorpus.Path, CorpusTokens, DecodePrompts);

        Assert.True(run.Perplexity.ScoredTokens > 0);
        Assert.True(double.IsFinite(run.Perplexity.MeanNegativeLogLikelihood));

        AssertLegIsPopulated("uncached", run.DecodeTokens, run.DecodeLogits);
        AssertLegIsPopulated("cached", run.KvDecodeTokens, run.KvDecodeLogits);

        // The prompt set was chosen so this holds on every fixture; surfaced here rather than
        // assumed, because a vacuous leg passes a cross-backend token comparison for free.
        Assert.True(
            run.DecodeIsInformative,
            $"uncached leg produced only [{string.Join(", ", run.DecodeTokens)}] — re-run the prompt search.");
    }

    /// <summary>
    /// The cached leg must differ from the uncached one, proving it is a distinct measurement.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The cached leg exists to reach the <c>seqLen == 1</c> GEMV path. If it silently degraded
    /// into repeating the uncached forward — a cache that was never populated, or a step that
    /// re-submitted the whole prompt — it would return the same logits and every other assertion in
    /// this class would still pass, while the gate reported GEMV coverage it did not have.
    /// </para>
    /// <para>
    /// The two legs score genuinely different things: the uncached leg reads the logits <i>at</i>
    /// the prompt's last position, the cached leg reads them one position later, after the prompt's
    /// own argmax has been appended. So their rows must differ, and identical rows mean the cached
    /// step did not happen.
    /// </para>
    /// </remarks>
    [SkippableFact]
    public void Run_CachedLegIsNotARepeatOfTheUncachedOne()
    {
        QuantLadderEntry? entry = RequireQ8Fixture();

        QuantGateRun run = QuantGateBackendRunner.Run(
            entry!, QuantGateBackend.Cpu, QuantGateCorpus.Path, CorpusTokens, DecodePrompts);

        for (int step = 0; step < run.DecodeLogits.Length; step++)
        {
            float[] uncached = run.DecodeLogits[step];
            float[] cached = run.KvDecodeLogits[step];
            Assert.Equal(uncached.Length, cached.Length);
            Assert.Contains(
                Enumerable.Range(0, uncached.Length),
                i => !uncached[i].Equals(cached[i]));
        }
    }

    /// <summary>Asserts one leg returned the expected number of populated, finite logit rows.</summary>
    /// <param name="leg">Leg name for failure text.</param>
    /// <param name="tokens">Emitted ids.</param>
    /// <param name="logits">Logit rows behind them.</param>
    private static void AssertLegIsPopulated(string leg, int[] tokens, float[][] logits)
    {
        Assert.Equal(DecodeSteps, tokens.Length);
        Assert.Equal(DecodeSteps, logits.Length);

        // A zero-filled or otherwise degenerate row satisfies every length assertion, so the content
        // is checked too: real logits are finite and are not all the same value.
        for (int step = 0; step < logits.Length; step++)
        {
            float[] row = logits[step];
            Assert.True(row.Length > 0, $"{leg} step {step}: empty row.");
            Assert.All(row, v => Assert.True(float.IsFinite(v), $"{leg} step {step} contains {v}."));
            Assert.True(
                row.Distinct().Count() > 1,
                $"{leg} step {step}: all {row.Length} logits are {row[0]:R} — the row was never populated.");
        }
    }

    /// <summary>
    /// Proves the extracted logit row tracks the <i>final</i> position rather than a fixed row.
    /// </summary>
    /// <remarks>
    /// This is the case that discriminates <c>LastRowOf</c>'s offset arithmetic. The CPU backend
    /// returns <c>[seqLen, vocab]</c>, so reading row 0 (offset <c>0</c> instead of
    /// <c>ElementCount - vocab</c>) would return the logits after the shared first token for both
    /// prompts below — identical vectors, and every other test in this class still passing.
    /// </remarks>
    [SkippableFact]
    public void Run_ReadsTheFinalPositionsLogits()
    {
        QuantLadderEntry? entry = RequireQ8Fixture();

        // Two prompts sharing a first token but ending at different positions.
        QuantGateRun longer = QuantGateBackendRunner.Run(
            entry!, QuantGateBackend.Cpu, QuantGateCorpus.Path,
            CorpusTokens, ["The capital of France is"]);
        QuantGateRun shorter = QuantGateBackendRunner.Run(
            entry!, QuantGateBackend.Cpu, QuantGateCorpus.Path,
            CorpusTokens, ["The capital of"]);

        float[] a = longer.DecodeLogits[0];
        float[] b = shorter.DecodeLogits[0];
        Assert.Equal(a.Length, b.Length);
        Assert.Contains(Enumerable.Range(0, a.Length), i => !a[i].Equals(b[i]));
    }

    /// <summary>CPU is always available; absent GPU backends must report so rather than throw.</summary>
    [Fact]
    public void IsAvailable_CpuIsAlwaysTrue()
        => Assert.True(QuantGateBackendRunner.IsAvailable(QuantGateBackend.Cpu));

    /// <summary>
    /// Resolves the Q8_0 fixture, skipping when either it or the shared corpus is absent. Both are
    /// git-ignored local artifacts, so a missing one is a "not provisioned here", not a failure.
    /// </summary>
    /// <returns>The Q8_0 ladder entry; never null past the skips.</returns>
    private QuantLadderEntry? RequireQ8Fixture()
    {
        QuantLadderEntry? entry = _ladder.Available.FirstOrDefault(e => e.Type == QuantizationType.Q8_0);
        Skip.If(entry is null, $"Q8_0 fixture not present under {_ladder.RootDirectory}");
        Skip.IfNot(File.Exists(QuantGateCorpus.Path), $"corpus not present at {QuantGateCorpus.Path}");
        return entry;
    }
}
