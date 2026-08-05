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
            Assert.True(new FileInfo(entry.FilePath).Length > QuantLadderFixture.MinFixtureBytes,
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

    /// <summary>
    /// Against an empty ladder root, classification must actually reach <c>Missing</c> for every
    /// expected type — not just leave <c>Available</c> vacuously empty. This is the failure mode
    /// #256 exists to catch: an empty directory previously still made every test in this class
    /// pass. The fixture is constructed directly here (not via the shared collection fixture) and
    /// the environment mutation is scoped to this test with save/restore in <c>finally</c>, since
    /// xunit can run other collections concurrently and must not observe this override.
    /// </summary>
    [Fact]
    public void EmptyLadderRoot_ClassifiesEveryExpectedTypeAsMissing()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), $"quant-ladder-empty-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);
        string? previous = Environment.GetEnvironmentVariable(QuantLadderFixture.DirEnvVar);

        try
        {
            Environment.SetEnvironmentVariable(QuantLadderFixture.DirEnvVar, tempRoot);
            var fixture = new QuantLadderFixture();

            Assert.Empty(fixture.Available);
            Assert.Equal(
                QuantLadderFixture.Expected.Select(e => e.Type).ToHashSet(),
                fixture.Missing.ToHashSet());
        }
        finally
        {
            Environment.SetEnvironmentVariable(QuantLadderFixture.DirEnvVar, previous);
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    /// <summary>
    /// A stub file — present at the expected relative path but far short of
    /// <see cref="QuantLadderFixture.MinFixtureBytes"/> — must classify as <c>Missing</c>, not
    /// <c>Available</c>. This proves the size check is load-bearing: a crashed
    /// <c>llama-quantize</c> leaves exactly this kind of truncated file behind, and existence
    /// alone would wrongly call it usable. No real fixture is touched — the stub is a few KB of
    /// zeros written under a fresh temp directory.
    /// </summary>
    [Fact]
    public void TruncatedStubFile_ClassifiesAsMissing_NotAvailable()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), $"quant-ladder-stub-{Guid.NewGuid():N}");
        string? previous = Environment.GetEnvironmentVariable(QuantLadderFixture.DirEnvVar);

        try
        {
            var (stubType, relativePath, _) = QuantLadderFixture.Expected[0];
            string stubFullPath = Path.Combine(tempRoot, relativePath);
            Directory.CreateDirectory(Path.GetDirectoryName(stubFullPath)!);
            File.WriteAllBytes(stubFullPath, new byte[4096]);

            Environment.SetEnvironmentVariable(QuantLadderFixture.DirEnvVar, tempRoot);
            var fixture = new QuantLadderFixture();

            Assert.Contains(stubType, fixture.Missing);
            Assert.DoesNotContain(fixture.Available, e => e.Type == stubType);
        }
        finally
        {
            Environment.SetEnvironmentVariable(QuantLadderFixture.DirEnvVar, previous);
            Directory.Delete(tempRoot, recursive: true);
        }
    }
}
