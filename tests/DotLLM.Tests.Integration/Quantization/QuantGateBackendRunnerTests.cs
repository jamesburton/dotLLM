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
