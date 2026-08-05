using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Quantization;

/// <summary>
/// Chooses and then guards the gate's decode prompt (#256).
/// </summary>
/// <remarks>
/// <para>
/// <c>CrossBackendQuantGateTests</c> asserts that the CPU reference emits more than one distinct
/// token across its decode steps. When it does not, the top-1 comparison is vacuous: every step
/// carries the same id, the two backends compare equal for free, and a green cell says nothing
/// about the kernels. The first full sweep hit this on four cells — Q2_K and Q3_K on both
/// backends — where the <c>--pure</c> 1B fixtures locked into a repeat of token 13551.
/// </para>
/// <para>
/// <b>Ruling: fix the prompt, do not soften the assertion.</b> The alternative — recording
/// non-informative and passing anyway — leaves the gate's decode leg resting on the cosine arm
/// alone for exactly the low-bit types where a decode defect is most likely to hide.
/// </para>
/// <para>
/// This file holds both halves of that: <see cref="Probe_ReportsDecodeTokensPerCandidatePrompt"/>
/// is the search, run by hand when a prompt needs choosing, and
/// <see cref="ChosenPrompt_IsInformativeOnEveryFixture"/> is the standing guard that fails if the
/// chosen prompt ever stops discriminating on a fixture.
/// </para>
/// </remarks>
[Trait("Category", "Fixtures")]
[Collection("QuantLadder")]
public sealed class QuantGateDecodePromptTests
{
    /// <summary>
    /// Candidate prompts for the search, ordered from the one in use towards progressively
    /// stronger nudges away from a repeat loop.
    /// </summary>
    /// <remarks>
    /// A near-destroyed 2-bit model has lost most of its next-token structure, so the candidates
    /// that survive tend to be the ones whose continuation is forced by shallow surface features —
    /// an open list, an unclosed bracket, a counting sequence — rather than by semantics.
    /// </remarks>
    public static readonly string[] Candidates =
    [
        "The capital of France is",
        "1, 2, 3, 4,",
        "The quick brown fox jumps over the lazy",
        "def add(a, b):\n    return",
        "Monday, Tuesday, Wednesday,",
        "A B C D E F",
        "Once upon a time, in a land far away, there lived a",
        "{\"name\": \"Alice\", \"age\":",
        "The three primary colours are red, green and",
        "one two three four five six seven",
    ];

    private readonly QuantLadderFixture _ladder;
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the probe against the shared ladder index.</summary>
    /// <param name="ladder">Ladder index shared across the <c>QuantLadder</c> collection.</param>
    /// <param name="output">xunit sink for the measured token sequences.</param>
    public QuantGateDecodePromptTests(QuantLadderFixture ladder, ITestOutputHelper output)
    {
        _ladder = ladder;
        _output = output;
    }

    /// <summary>
    /// Decodes every candidate prompt against every available fixture on CPU and reports which
    /// candidates are informative everywhere.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The search, not a gate: it always passes, because its output is the measurement. Run it,
    /// read the <c>WINNERS</c> line, and set <see cref="CrossBackendQuantGateTests.DecodePrompt"/>
    /// from it. <see cref="ChosenPrompt_IsInformativeOnEveryFixture"/> is what keeps the answer
    /// honest afterwards.
    /// </para>
    /// <para>
    /// Costs one CPU load per fixture — minutes each on the 1B fixtures, cold — which is why it is
    /// separated from the standing guard rather than folded into it.
    /// </para>
    /// </remarks>
    [SkippableFact]
    public void Probe_ReportsDecodeTokensPerCandidatePrompt()
    {
        Skip.If(_ladder.Available.Count == 0, $"no ladder fixtures under {_ladder.RootDirectory}");

        // One id per (fixture, candidate). The search is then over SUBSETS of candidates: the gate
        // needs a set whose ids are all distinct on every fixture, which no single prompt can
        // deliver on its own.
        var byFixture = new Dictionary<string, IReadOnlyList<int>>(StringComparer.Ordinal);

        foreach (QuantLadderEntry entry in _ladder.Available)
        {
            IReadOnlyList<int> ids = QuantGateBackendRunner.ProbeDecodePrompts(entry, Candidates);
            byFixture[entry.Type.ToString()] = ids;

            for (int i = 0; i < Candidates.Length; i++)
            {
                _output.WriteLine(
                    $"  PROMPT\t{entry.Type}\t{i}\t{ids[i]}\t" +
                    Candidates[i].Replace("\n", "\\n", StringComparison.Ordinal));
            }
        }

        int want = CrossBackendQuantGateTests.DecodeSteps;
        var winners = new List<int[]>();
        foreach (int[] subset in Subsets(Candidates.Length, want))
        {
            int ok = byFixture.Values.Count(ids => subset.Select(i => ids[i]).Distinct().Count() == want);
            if (ok == byFixture.Count)
                winners.Add(subset);
        }

        foreach (int[] subset in winners.Take(10))
        {
            _output.WriteLine(
                $"  WINNER\t[{string.Join(",", subset)}]\t" +
                string.Join(" | ", subset.Select(i => Candidates[i].Replace("\n", "\\n", StringComparison.Ordinal))));
        }

        _output.WriteLine(winners.Count > 0
            ? $"  WINNERS\t{winners.Count} subset(s) of size {want} give {want} distinct ids on all {byFixture.Count} fixtures"
            : $"  WINNERS\tNONE — no {want}-subset discriminates on all {byFixture.Count} fixtures. " +
              "Add candidates, or reduce DecodeSteps.");
    }

    /// <summary>
    /// The prompt set the gate actually uses must produce distinct ids on every fixture present.
    /// </summary>
    /// <remarks>
    /// The standing guard. It fails on the fixture, naming it, rather than letting the failure
    /// surface later as a red GPU cell in the gate — where the message would blame the backend
    /// pairing for a property of the prompts and the CPU reference alone.
    /// </remarks>
    [SkippableFact]
    public void ChosenPrompts_AreInformativeOnEveryFixture()
    {
        Skip.If(_ladder.Available.Count == 0, $"no ladder fixtures under {_ladder.RootDirectory}");

        string[] chosen = CrossBackendQuantGateTests.DecodePrompts;
        var degenerate = new List<string>();

        foreach (QuantLadderEntry entry in _ladder.Available)
        {
            IReadOnlyList<int> ids = QuantGateBackendRunner.ProbeDecodePrompts(entry, chosen);
            int distinct = ids.Distinct().Count();
            _output.WriteLine($"  CHOSEN\t{entry.Type}\t{distinct}/{chosen.Length}\t[{string.Join(",", ids)}]");

            // The gate's own bar is >1 distinct, but the chosen set was selected for ALL distinct;
            // asserting the stronger property here means a fixture that degrades from 4 to 2 is
            // caught while it is still a warning rather than after it has reached the gate's floor.
            if (distinct < chosen.Length)
                degenerate.Add($"{entry.Type} produced [{string.Join(",", ids)}] ({distinct} distinct)");
        }

        Assert.True(degenerate.Count == 0,
            $"The gate's decode prompt set is degenerate on {degenerate.Count} fixture(s): " +
            $"{string.Join("; ", degenerate)}. The top-1 arm of both decode legs weakens on those cells. " +
            "Re-run Probe_ReportsDecodeTokensPerCandidatePrompt and take a set from its WINNER lines.");
    }

    /// <summary>Enumerates every <paramref name="k"/>-sized index subset of <paramref name="n"/>.</summary>
    /// <param name="n">Population size.</param>
    /// <param name="k">Subset size.</param>
    /// <returns>Each subset, ascending.</returns>
    private static IEnumerable<int[]> Subsets(int n, int k)
    {
        if (k > n) yield break;

        var idx = new int[k];
        for (int i = 0; i < k; i++) idx[i] = i;

        while (true)
        {
            yield return (int[])idx.Clone();

            int j = k - 1;
            while (j >= 0 && idx[j] == n - k + j) j--;
            if (j < 0) yield break;

            idx[j]++;
            for (int m = j + 1; m < k; m++) idx[m] = idx[m - 1] + 1;
        }
    }
}
