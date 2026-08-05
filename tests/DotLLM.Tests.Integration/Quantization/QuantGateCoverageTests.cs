using DotLLM.Core.Configuration;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Quantization;

/// <summary>
/// Ladder-level coverage checks that need the fixtures but no GPU (#256).
/// </summary>
/// <remarks>
/// Split from <see cref="QuantGateMatrixCoverageTests"/> on purpose. These two assertions hold on
/// any machine with the ladder, so tying them to <c>Category=GPU</c> would mean the index could
/// silently rot on every box that has the fixtures and no device — which is most of them.
/// </remarks>
[Trait("Category", "Fixtures")]
[Collection("QuantLadder")]
public sealed class QuantGateCoverageTests
{
    private readonly QuantLadderFixture _ladder;
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the coverage checks against the shared ladder index.</summary>
    /// <param name="ladder">Ladder index shared across the <c>QuantLadder</c> collection.</param>
    /// <param name="output">xunit sink for the coverage summary.</param>
    public QuantGateCoverageTests(QuantLadderFixture ladder, ITestOutputHelper output)
    {
        _ladder = ladder;
        _output = output;
    }

    /// <summary>Every ladder fixture the machine holds is indexed exactly once.</summary>
    [Fact]
    public void EveryAvailableFixture_IsIndexedExactlyOnce()
    {
        var types = _ladder.Available.Select(e => e.Type).ToList();
        Assert.Equal(types.Count, types.Distinct().Count());

        _output.WriteLine($"ladder root: {_ladder.RootDirectory}");
        _output.WriteLine($"available: {types.Count}/{QuantLadderFixture.Expected.Count}");
        if (_ladder.Missing.Count > 0)
            _output.WriteLine($"missing: {string.Join(", ", _ladder.Missing)}");
    }

    /// <summary>
    /// The gate's theory data enumerates the full matrix — every expected type against every GPU
    /// backend, once each.
    /// </summary>
    /// <remarks>
    /// Guards the enumeration itself rather than the run. <c>Cases()</c> is built from
    /// <see cref="QuantLadderFixture.Expected"/> rather than from what is present on the machine,
    /// so that a missing fixture becomes an explicit skipped cell instead of vanishing. That is
    /// only true while <c>Cases()</c> stays complete, and nothing else checks it.
    /// </remarks>
    [Fact]
    public void GateCases_CoverEveryTypeOnEveryGpuBackend()
    {
        var expected = new HashSet<(QuantizationType, QuantGateBackend)>();
        foreach (var (type, _, _) in QuantLadderFixture.Expected)
        {
            expected.Add((type, QuantGateBackend.Cuda));
            expected.Add((type, QuantGateBackend.Vulkan));
        }

        var actual = CrossBackendQuantGateTests.Cases()
            .Select(row => ((QuantizationType)row[0]!, (QuantGateBackend)row[1]!))
            .ToList();

        _output.WriteLine($"cases: {actual.Count}, expected: {expected.Count}");

        // Counted before the set comparison: a duplicated pair and a missing pair cancel out in a
        // set difference, and the gate would then run 42 cells covering 41 combinations.
        Assert.Equal(actual.Count, actual.Distinct().Count());
        Assert.Equal(expected.Count, actual.Count);
        Assert.Empty(expected.Except(actual));
    }
}

/// <summary>
/// Reports the full {type × backend} matrix the gate actually ran and fails on an undeclared gap
/// (#256).
/// </summary>
/// <remarks>
/// <para>
/// This is the direct answer to the issue's core complaint: <b>a test that skips looks identical
/// to one that passes</b>. Every cell must be classified — ran, failed, or skipped for a stated
/// reason — and an unclassified or absent cell is itself a failure.
/// </para>
/// <para>
/// <b>On ordering.</b> This class reads static state accumulated by
/// <see cref="CrossBackendQuantGateTests"/>, so it is only meaningful once the gate has run. Both
/// classes sit in the <c>QuantLadder</c> collection, and xunit runs the classes of one collection
/// sequentially — so at the moment this test executes, <c>Results</c> is either empty (the gate
/// has not run yet) or complete (it has). It is never partially filled, which is what makes the
/// "if any cell was recorded, all of them must have been" rule below sound rather than racy.
/// The empty case skips, and Task 8's sweeper is what guarantees the gate is invoked first in the
/// run that matters.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("QuantLadder")]
public sealed class QuantGateMatrixCoverageTests
{
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the matrix report.</summary>
    /// <param name="output">xunit sink for the matrix.</param>
    public QuantGateMatrixCoverageTests(ITestOutputHelper output) => _output = output;

    /// <summary>Prints the matrix and fails on any absent cell or any skip without a reason.</summary>
    [SkippableFact]
    public void CoverageMatrix_HasNoUndeclaredGaps()
    {
        var results = CrossBackendQuantGateTests.Results;
        Skip.If(results.Count == 0,
            "gate has not run in this session — run CrossBackendQuantGateTests first");

        var seen = new HashSet<(QuantizationType, QuantGateBackend)>();
        var undeclared = new List<string>();
        var duplicated = new List<string>();

        foreach (var cell in results.OrderBy(c => c.Type.ToString(), StringComparer.Ordinal)
                                    .ThenBy(c => c.Backend.ToString(), StringComparer.Ordinal))
        {
            _output.WriteLine($"{cell.Type,-10} {cell.Backend,-8} {cell.Outcome,-8} {cell.Detail}");

            if (!seen.Add((cell.Type, cell.Backend)))
                duplicated.Add($"{cell.Type}/{cell.Backend}");

            // The reason is the whole point. A skip that does not say why is indistinguishable from
            // a pass in a summary line, which is the failure mode this gate exists to end.
            if (string.Equals(cell.Outcome, "skipped", StringComparison.Ordinal)
                && string.IsNullOrWhiteSpace(cell.Detail))
            {
                undeclared.Add($"{cell.Type}/{cell.Backend}");
            }
        }

        // Absent cells, not just unexplained ones. A cell that never reached Record at all — because
        // the theory row was dropped, or the test died before recording — leaves no trace in the
        // matrix, and an empty row reads as "nothing to report" rather than "never measured".
        var missing = new List<string>();
        foreach (var (type, _, _) in QuantLadderFixture.Expected)
        {
            foreach (var backend in new[] { QuantGateBackend.Cuda, QuantGateBackend.Vulkan })
            {
                if (!seen.Contains((type, backend)))
                    missing.Add($"{type}/{backend}");
            }
        }

        _output.WriteLine($"cells recorded: {seen.Count}/{QuantLadderFixture.Expected.Count * 2}");

        Assert.True(duplicated.Count == 0,
            $"cells recorded more than once: {string.Join(", ", duplicated)}. " +
            "A duplicated cell means the matrix covers fewer combinations than its row count suggests.");

        Assert.True(undeclared.Count == 0,
            $"cells skipped without a stated reason: {string.Join(", ", undeclared)}. " +
            "An absent test must never read as a passing one.");

        Assert.True(missing.Count == 0,
            $"cells never recorded at all: {string.Join(", ", missing)}. " +
            "The gate ran, so every expected cell should have reached Record — as a pass, a failure, " +
            "or a declared skip. A cell with no row is not coverage.");
    }
}
