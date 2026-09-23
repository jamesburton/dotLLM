using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #502: a test class that assigns a process-wide <c>*Override</c> static must run inside
/// <see cref="CudaCollection"/>, which is the only thing in this assembly that serializes it.
/// </summary>
/// <remarks>
/// <para>The statics in question (<c>MatMul.I2SUseW2A8Override</c>,
/// <c>CudaSmallSGemvDispatch.{Mmq,MmqTile,MaxColumns,Dp4a,SingleColumn}Override</c>, …) are read
/// <b>per call</b> on the production path, deliberately, so that a test can A/B two kernels in one
/// session. The cost of that design is that the pin is only valid while nothing else is running:
/// two classes pinning the same static in their constructors and clearing it in <c>Dispose</c> will
/// un-pin each other, and the result is not a crash but a quietly different numeric path.</para>
/// <para><b>Why a source scan rather than reflection.</b> The defect is "this class writes to a
/// static", which leaves no trace in the type's metadata — reflection can see the
/// <c>[Collection]</c> attribute but not the assignments that make it necessary. Scanning the
/// sources is the only oracle available in-process. <see cref="CallerFilePathAttribute"/> locates
/// them, so the test is silent (not falsely green) when run from a package where they are absent.</para>
/// <para><b>Mutants:</b> deleting <c>[Collection(CudaCollection.Name)]</c> from
/// <c>CudaMoeFfnBitNetI2STests</c>, <c>CudaMoeFfnBitNetI2SBatchedGemmTests</c> or
/// <c>CudaSmallSGemvDispatchMmqTests</c> fails <see cref="OverrideMutatingClasses_AreSerialized"/>;
/// so does adding a new class that sets one without joining the collection.</para>
/// </remarks>
public sealed partial class CudaTestCollectionContractTests
{
    /// <summary>Assignment to a static whose name ends in <c>Override</c>, ignoring <c>==</c>/<c>=&gt;</c>.</summary>
    [GeneratedRegex(@"\b[A-Za-z0-9_]*Override\s*=(?![=>])", RegexOptions.None, matchTimeoutMilliseconds: 1000)]
    private static partial Regex OverrideAssignment();

    private const string CollectionAttribute = "[Collection(CudaCollection.Name)]";

    [SkippableFact]
    public void OverrideMutatingClasses_AreSerialized()
    {
        string dir = SourceDirectory();
        Skip.If(!Directory.Exists(dir), $"CUDA test sources not found at {dir}");

        var offenders = new List<string>();
        foreach (string file in Directory.GetFiles(dir, "*.cs"))
        {
            // This file names the pattern it searches for; it mutates nothing.
            if (Path.GetFileName(file) == ThisFileName) continue;

            string[] lines = File.ReadAllLines(file);
            var hits = lines
                .Select(static (text, i) => (Text: text.Trim(), Line: i + 1))
                .Where(static l => !l.Text.StartsWith("//", StringComparison.Ordinal)
                                   && !l.Text.StartsWith('*')
                                   && OverrideAssignment().IsMatch(l.Text))
                .ToList();
            if (hits.Count == 0) continue;

            if (!lines.Any(static l => l.Contains(CollectionAttribute, StringComparison.Ordinal)))
                offenders.Add($"{Path.GetFileName(file)} (first at line {hits[0].Line}: {hits[0].Text})");
        }

        Assert.True(offenders.Count == 0,
            "These CUDA test files assign a process-wide *Override static but do not carry "
            + $"{CollectionAttribute}, so they run in parallel with every other class and their pin "
            + "can be cleared mid-test by a sibling's Dispose (#502):\n  "
            + string.Join("\n  ", offenders));
    }

    /// <summary>
    /// The scan is only meaningful if it actually sees the assignments — an over-tightened regex or a
    /// wrong directory would make <see cref="OverrideMutatingClasses_AreSerialized"/> vacuously green.
    /// </summary>
    [SkippableFact]
    public void TheScan_FindsTheKnownOverrideMutators()
    {
        string dir = SourceDirectory();
        Skip.If(!Directory.Exists(dir), $"CUDA test sources not found at {dir}");

        string[] known =
        [
            "CudaMoeFfnBitNetI2STests.cs",
            "CudaMoeFfnBitNetI2SBatchedGemmTests.cs",
            "CudaSmallSGemvDispatchMmqTests.cs",
        ];
        foreach (string name in known)
        {
            string path = Path.Combine(dir, name);
            Assert.True(File.Exists(path), $"{name} has moved — update this guard's known-mutator list.");
            Assert.True(File.ReadLines(path).Any(l => OverrideAssignment().IsMatch(l)),
                $"the *Override assignment scan no longer matches anything in {name}; if that class "
                + "genuinely stopped pinning a static, drop it from this list, otherwise the regex is broken.");
        }
    }

    private const string ThisFileName = "CudaTestCollectionContractTests.cs";

    private static string SourceDirectory([CallerFilePath] string thisFile = "")
        => Path.GetDirectoryName(thisFile)!;
}
