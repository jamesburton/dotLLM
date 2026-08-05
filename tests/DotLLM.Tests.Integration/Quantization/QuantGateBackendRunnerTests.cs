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
    private const string DecodePrompt = "The capital of France is";
    private const int CorpusTokens = 512;
    private const int DecodeSteps = 4;

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
            entry!, QuantGateBackend.Cpu, QuantGateCorpus.Path,
            CorpusTokens, DecodePrompt, DecodeSteps);
        QuantGateRun b = QuantGateBackendRunner.Run(
            entry!, QuantGateBackend.Cpu, QuantGateCorpus.Path,
            CorpusTokens, DecodePrompt, DecodeSteps);

        Assert.Equal(a.Perplexity.MeanNegativeLogLikelihood, b.Perplexity.MeanNegativeLogLikelihood, 12);
        Assert.Equal(a.DecodeTokens, b.DecodeTokens);

        // Tokens alone do not test the decode kernel. Argmax is invariant to any perturbation
        // smaller than the top-2 logit gap, so a nondeterministic decode GEMV — the exact kernel
        // this leg exists to cover — would emit identical tokens and pass. Compare the vectors.
        Assert.Equal(a.DecodeLogits.Length, b.DecodeLogits.Length);
        for (int step = 0; step < a.DecodeLogits.Length; step++)
        {
            float[] left = a.DecodeLogits[step];
            float[] right = b.DecodeLogits[step];
            Assert.Equal(left.Length, right.Length);
            for (int v = 0; v < left.Length; v++)
            {
                Assert.True(
                    left[v].Equals(right[v]),
                    $"decode step {step}, logit {v}: {left[v]:R} != {right[v]:R} across two identical runs.");
            }
        }
    }

    /// <summary>Both legs must actually produce data; an empty decode leg would assert nothing.</summary>
    [SkippableFact]
    public void Run_ProducesBothLegs()
    {
        QuantLadderEntry? entry = RequireQ8Fixture();

        QuantGateRun run = QuantGateBackendRunner.Run(
            entry!, QuantGateBackend.Cpu, QuantGateCorpus.Path,
            CorpusTokens, DecodePrompt, DecodeSteps);

        Assert.True(run.Perplexity.ScoredTokens > 0);
        Assert.True(double.IsFinite(run.Perplexity.MeanNegativeLogLikelihood));
        Assert.Equal(DecodeSteps, run.DecodeTokens.Length);
        Assert.Equal(DecodeSteps, run.DecodeLogits.Length);
        Assert.All(run.DecodeLogits, l => Assert.True(l.Length > 0));

        // A zero-filled or otherwise degenerate row satisfies every length assertion above, so the
        // content is checked too: real logits are finite and are not all the same value.
        for (int step = 0; step < run.DecodeLogits.Length; step++)
        {
            float[] row = run.DecodeLogits[step];
            Assert.All(row, v => Assert.True(float.IsFinite(v), $"decode step {step} contains {v}."));
            Assert.True(
                row.Distinct().Count() > 1,
                $"decode step {step}: all {row.Length} logits are {row[0]:R} — the row was never populated.");
        }

        // Greedy decode on a small model can emit EOS or lock into a repeat, which would make a
        // later cross-backend token comparison vacuously equal. Surfaced, not assumed.
        Assert.True(
            run.DecodeIsInformative,
            $"decode leg emitted only [{string.Join(", ", run.DecodeTokens)}] — pick a different prompt.");
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
            CorpusTokens, DecodePrompt, DecodeSteps);
        QuantGateRun shorter = QuantGateBackendRunner.Run(
            entry!, QuantGateBackend.Cpu, QuantGateCorpus.Path,
            CorpusTokens, "The capital of", DecodeSteps);

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
