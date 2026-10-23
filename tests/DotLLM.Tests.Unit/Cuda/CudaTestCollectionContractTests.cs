using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #502: a test class that assigns a process-wide <c>*Override</c> static must run inside
/// <see cref="CudaCollection"/>, which is the only thing in this assembly that serializes it.
/// Issue #503 widens that to every GPU-traited class under <c>Cuda/</c>: they all share one
/// device and one CUDA context, so any two running concurrently can corrupt each other.
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
/// <para><b>Why the #503 rule is reflection and the #502 rule is not.</b> "This class declares
/// <c>[Trait("Category", "GPU")]</c>" <i>is</i> in the metadata, and so is the
/// <c>[Collection]</c>/<c>[CollectionDefinition]</c> pair, so
/// <see cref="EveryGpuTestClass_IsInANonParallelCollection"/> can ask the loaded assembly
/// directly — which is per <i>class</i>, not per file, and therefore correct for the several
/// files here that declare more than one. The #502 rule cannot do that, because a static
/// assignment leaves no metadata trace at all. The two live together because they share a
/// subject, not a technique. This mirrors <c>GpuCollectionGuardTests</c> in
/// <c>DotLLM.Tests.Integration</c>.</para>
/// <para><b>Any</b> <c>DisableParallelization = true</c> collection satisfies the #503 rule, not
/// just <see cref="CudaCollection"/>: xUnit v2 runs non-parallel collections one at a time and
/// never alongside the parallel batch, so the nine kernel-parity classes in the separate
/// <c>CudaKernels</c> collection (see <c>TestCollections.cs</c>) are already mutually exclusive
/// with it. Requiring one specific name would force a pointless churn on them.</para>
/// <para><b>Mutants:</b> deleting <c>[Collection(CudaCollection.Name)]</c> from
/// <c>CudaMoeFfnBitNetI2STests</c>, <c>CudaMoeFfnBitNetI2SBatchedGemmTests</c> or
/// <c>CudaSmallSGemvDispatchMmqTests</c> fails <see cref="OverrideMutatingClasses_AreSerialized"/>;
/// so does adding a new class that sets one without joining the collection. Deleting it from any
/// GPU-traited class (verified against <c>CudaMlaForwardTests</c>) fails
/// <see cref="EveryGpuTestClass_IsInANonParallelCollection"/>, naming that class.</para>
/// <para><b>Not covered by either rule:</b> <c>CudaQuantExpansionGateTests</c> toggles the
/// process-wide <c>CudaKernels.AllowQuantExpansion</c> but is CPU-only and its static is not
/// named <c>*Override</c>, so it carries the attribute by hand. A second such global would need
/// the #502 regex widened.</para>
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

    /// <summary>
    /// Issue #503: every test class under the <c>DotLLM.Tests.Unit.Cuda</c> namespace that declares
    /// <c>[Trait("Category", "GPU")]</c> must sit in a collection with
    /// <c>DisableParallelization = true</c>. Only 43 of ~113 did, which is what let #502 (a
    /// sibling's <c>Dispose</c> un-pinning a static mid-test) and #484 (a device-wide
    /// <c>cuMemGetInfo</c> assertion racing a sibling's allocations) both present as real defects.
    /// Needs no GPU.
    /// </summary>
    [Fact]
    public void EveryGpuTestClass_IsInANonParallelCollection()
    {
        IReadOnlyDictionary<string, bool> definitions = NonParallelByCollectionName();
        Assert.True(definitions.TryGetValue(CudaCollection.Name, out bool cudaSerial) && cudaSerial,
            $"The '{CudaCollection.Name}' collection must exist with DisableParallelization = true.");

        var offenders = new List<string>();
        foreach (Type t in GpuTestClasses().OrderBy(static t => t.FullName, StringComparer.Ordinal))
        {
            string? collection = CollectionName(t);
            if (collection is null)
                offenders.Add($"{t.FullName}: no [Collection] — runs in parallel with every other CUDA class");
            else if (!definitions.TryGetValue(collection, out bool serial))
                offenders.Add($"{t.FullName}: collection '{collection}' has no [CollectionDefinition] in this "
                    + "assembly, so xUnit runs it in parallel");
            else if (!serial)
                offenders.Add($"{t.FullName}: collection '{collection}' does not set DisableParallelization = true");
        }

        Assert.True(offenders.Count == 0,
            "These CUDA test classes declare [Trait(\"Category\", \"GPU\")] but are not serialized, so they "
            + $"share one device and one CUDA context with whatever else is running (#503). Add "
            + $"{CollectionAttribute}:\n  " + string.Join("\n  ", offenders));
    }

    /// <summary>
    /// The reflection rule is only meaningful if it actually classifies the GPU classes — a namespace
    /// typo or a trait-matching slip would make <see cref="EveryGpuTestClass_IsInANonParallelCollection"/>
    /// vacuously green over an empty set. Names the classes the issue itself cites, rather than a count
    /// floor, so the test says which assumption broke.
    /// </summary>
    [Fact]
    public void TheGpuClassScan_FindsTheKnownGpuClasses()
    {
        var found = GpuTestClasses().Select(static t => t.Name).ToHashSet(StringComparer.Ordinal);

        string[] known =
        [
            "CudaMlaForwardTests",
            "CudaBatchedDecodeTests",
            "CudaGraphCaptureEquivalenceTest",
            "CudaNemotronHTransformerModelForwardTests",
            "CudaMamba3StateCacheTests",
        ];
        foreach (string name in known)
        {
            Assert.True(found.Contains(name),
                $"{name} is no longer seen as a GPU test class. Either it was renamed/removed (update this "
                + "list), or the GPU-class detection in this guard is broken and the #503 rule is vacuous. "
                + $"It currently sees {found.Count} GPU classes.");
        }

        // A CPU-only class in the same directory must NOT be dragged in: the rule keys on the GPU
        // trait, not on the namespace, so these keep their parallelism.
        Assert.DoesNotContain("PtxIsaVersionTests", found);
        Assert.DoesNotContain("PQ2_0Dp4aLayoutEmulationTests", found);
    }

    /// <summary>Test classes under <c>DotLLM.Tests.Unit.Cuda</c> carrying the GPU trait on the class or any method.</summary>
    private static List<Type> GpuTestClasses() =>
        typeof(CudaCollection).Assembly.GetTypes()
            .Where(static t => t.IsClass && !t.IsAbstract
                && (t.Namespace ?? string.Empty).StartsWith("DotLLM.Tests.Unit.Cuda", StringComparison.Ordinal)
                && IsTestClass(t) && HasGpuTrait(t))
            .ToList();

    private static bool IsTestClass(Type t) =>
        t.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance
                     | BindingFlags.Static | BindingFlags.DeclaredOnly)
            .Any(static m => m.GetCustomAttributes<FactAttribute>(inherit: true).Any());

    private static bool HasGpuTrait(Type t) =>
        IsGpuTrait(t.GetCustomAttributesData())
        || t.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance
                        | BindingFlags.Static | BindingFlags.DeclaredOnly)
            .Any(static m => IsGpuTrait(m.GetCustomAttributesData()));

    private static bool IsGpuTrait(IEnumerable<CustomAttributeData> attributes) =>
        attributes.Any(static a => a.AttributeType == typeof(TraitAttribute)
            && a.ConstructorArguments.Count == 2
            && (string?)a.ConstructorArguments[0].Value == "Category"
            && (string?)a.ConstructorArguments[1].Value == "GPU");

    /// <summary>Every <c>[CollectionDefinition]</c> in this assembly, mapped to its DisableParallelization flag.</summary>
    private static Dictionary<string, bool> NonParallelByCollectionName()
    {
        var definitions = new Dictionary<string, bool>(StringComparer.Ordinal);
        foreach (Type t in typeof(CudaCollection).Assembly.GetTypes())
        {
            var def = t.GetCustomAttribute<CollectionDefinitionAttribute>();
            if (def is null) continue;
            string name = t.GetCustomAttributesData()
                .Where(static a => a.AttributeType == typeof(CollectionDefinitionAttribute)
                                   && a.ConstructorArguments.Count == 1)
                .Select(static a => a.ConstructorArguments[0].Value as string)
                .FirstOrDefault() ?? t.FullName!;
            definitions[name] = def.DisableParallelization;
        }
        return definitions;
    }

    private static string? CollectionName(Type t) =>
        t.GetCustomAttributesData()
            .Where(static a => a.AttributeType == typeof(CollectionAttribute) && a.ConstructorArguments.Count == 1)
            .Select(static a => a.ConstructorArguments[0].Value as string)
            .FirstOrDefault();

    private const string ThisFileName = "CudaTestCollectionContractTests.cs";

    private static string SourceDirectory([CallerFilePath] string thisFile = "")
        => Path.GetDirectoryName(thisFile)!;
}
