using DotLLM.Core.Configuration;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Quantization;

/// <summary>
/// The fixture index must describe the ladder truthfully: every expected type accounted for as
/// either available or missing, and never both.
/// </summary>
[Collection("QuantLadder")]
public sealed class QuantLadderFixtureTests
{
    private readonly QuantLadderFixture _ladder;

    public QuantLadderFixtureTests(QuantLadderFixture ladder) => _ladder = ladder;

    /// <summary>
    /// Availability is partitioned, not overlapping. A type appearing in both lists, or in
    /// neither, means the index is lying about coverage — which is the defect #256 is about.
    /// </summary>
    [Fact]
    public void AvailableAndMissing_PartitionTheExpectedSet()
    {
        var expected = QuantLadderFixture.Expected.Select(e => e.Type).ToHashSet();
        var available = _ladder.Available.Select(e => e.Type).ToHashSet();
        var missing = _ladder.Missing.ToHashSet();

        Assert.Empty(available.Intersect(missing));
        Assert.Equal(expected, available.Union(missing).ToHashSet());
    }

    /// <summary>Every advertised path must exist and be non-empty — a truncated quantize leaves a stub.</summary>
    [Fact]
    public void Available_PathsExistAndAreNonTrivial()
    {
        foreach (var entry in _ladder.Available)
        {
            Assert.True(File.Exists(entry.FilePath), $"{entry.Type}: {entry.FilePath}");
            Assert.True(new FileInfo(entry.FilePath).Length > 1_000_000,
                $"{entry.Type} is {new FileInfo(entry.FilePath).Length} bytes — a truncated quantize leaves a stub");
        }
    }

    /// <summary>
    /// Context length is a property of the fixture's base model, not a free choice: the 135M
    /// ladder scores at 512 and the ≥1B ladder at 128. Both backends in a comparison must use the
    /// same value, so it lives in the index rather than at each call site.
    /// </summary>
    [Fact]
    public void Expected_ContextLengths_MatchTheirBaseModel()
    {
        foreach (var (type, relativePath, ctx) in QuantLadderFixture.Expected)
        {
            int wanted = relativePath.Contains("SmolLM2-135M", StringComparison.Ordinal) ? 512 : 128;
            Assert.True(ctx == wanted, $"{type}: expected ctx {wanted}, got {ctx}");
        }
    }

    /// <summary>No duplicate types — a copy-paste in the table would silently halve coverage.</summary>
    [Fact]
    public void Expected_HasNoDuplicateTypes()
    {
        var types = QuantLadderFixture.Expected.Select(e => e.Type).ToList();
        Assert.Equal(types.Count, types.Distinct().Count());
    }
}
