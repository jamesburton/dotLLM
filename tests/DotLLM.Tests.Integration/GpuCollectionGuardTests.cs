using System.Reflection;
using System.Text.RegularExpressions;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration;

/// <summary>
/// Guards #483: every integration test class that touches a GPU must sit in a
/// non-parallel collection (<see cref="GpuCollection"/> or another
/// <c>DisableParallelization = true</c> definition), so GPU classes can never contend for one
/// device. A new GPU test class added without <c>[Collection(GpuCollection.Name)]</c> fails here
/// on any machine — no GPU needed — instead of hanging a CUDA run on a 12 GB card.
/// </summary>
/// <remarks>
/// A class counts as a GPU class when any of these hold:
/// <list type="bullet">
/// <item>its namespace has a <c>Cuda</c> or <c>Vulkan</c> segment;</item>
/// <item>it, or one of its methods, carries <c>[Trait("Category", "GPU")]</c>;</item>
/// <item>its source file references <c>DotLLM.Cuda</c> or <c>DotLLM.Vulkan</c> (a <c>using</c>
/// or a qualified name). This leg reads the source tree and is skipped, with a note, when the
/// tests run from a location where the sources cannot be found.</item>
/// </list>
/// A CPU-only class wrongly caught by the heuristic only costs parallelism, never correctness.
/// </remarks>
public sealed class GpuCollectionGuardTests
{
    private static readonly Regex GpuReference = new(
        @"^\s*using\s+DotLLM\.(Cuda|Vulkan)\b|\bDotLLM\.(Cuda|Vulkan)\.[A-Z]",
        RegexOptions.Multiline | RegexOptions.ExplicitCapture | RegexOptions.Compiled,
        TimeSpan.FromSeconds(5));

    private static readonly Regex ClassDeclaration = new(
        @"\bclass\s+(?<name>\w+)", RegexOptions.ExplicitCapture | RegexOptions.Compiled,
        TimeSpan.FromSeconds(5));

    private readonly ITestOutputHelper _output;

    public GpuCollectionGuardTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void EveryGpuTestClass_IsInANonParallelCollection()
    {
        Assembly assembly = typeof(GpuCollection).Assembly;
        Type[] types = assembly.GetTypes();

        // name -> DisableParallelization, from every [CollectionDefinition] in this assembly.
        var definitions = new Dictionary<string, bool>(StringComparer.Ordinal);
        foreach (Type t in types)
        {
            var def = t.GetCustomAttribute<CollectionDefinitionAttribute>();
            if (def is null) continue;
            string name = CollectionDefinitionName(t);
            definitions[name] = def.DisableParallelization;
        }
        Assert.True(definitions.TryGetValue(GpuCollection.Name, out bool gpuSerial) && gpuSerial,
            $"The '{GpuCollection.Name}' collection must exist with DisableParallelization = true.");

        List<Type> testClasses = types.Where(IsTestClass).ToList();
        var gpuClasses = new HashSet<Type>();

        foreach (Type t in testClasses)
        {
            if (HasGpuNamespace(t) || HasGpuTrait(t))
                gpuClasses.Add(t);
        }

        string? projectDir = FindProjectDirectory();
        if (projectDir is null)
        {
            _output.WriteLine(
                "Source tree not found next to the test binaries; checked namespace and "
                + "[Trait(\"Category\", \"GPU\")] only, not DotLLM.Cuda/DotLLM.Vulkan source references.");
        }
        else
        {
            ILookup<string, Type> byName = testClasses.ToLookup(t => t.Name, StringComparer.Ordinal);
            foreach (string file in EnumerateSources(projectDir))
            {
                string text = File.ReadAllText(file);
                if (!GpuReference.IsMatch(text)) continue;
                foreach (Match m in ClassDeclaration.Matches(text))
                    foreach (Type t in byName[m.Groups["name"].Value])
                        gpuClasses.Add(t);
            }
        }

        var offenders = new List<string>();
        foreach (Type t in gpuClasses.OrderBy(t => t.FullName, StringComparer.Ordinal))
        {
            string? collection = CollectionName(t);
            if (collection is null)
            {
                offenders.Add($"{t.FullName}: no [Collection] (runs in parallel with other GPU classes)");
            }
            else if (!definitions.TryGetValue(collection, out bool serial))
            {
                offenders.Add($"{t.FullName}: collection '{collection}' has no [CollectionDefinition], "
                    + "so it runs in parallel");
            }
            else if (!serial)
            {
                offenders.Add($"{t.FullName}: collection '{collection}' does not set DisableParallelization = true");
            }
        }

        _output.WriteLine($"{gpuClasses.Count} GPU test classes checked of {testClasses.Count} test classes.");
        Assert.True(offenders.Count == 0,
            "GPU test classes must run serialized (#483). Add [Collection(GpuCollection.Name)], or — for a "
            + "class that needs a shared fixture — a fixture collection with DisableParallelization = true "
            + "(see SmallModelGpuCollection):\n  " + string.Join("\n  ", offenders));
    }

    private static bool IsTestClass(Type t) =>
        t.IsClass && !t.IsAbstract
        && t.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly)
            .Any(m => m.GetCustomAttributes<FactAttribute>(inherit: true).Any());

    private static bool HasGpuNamespace(Type t) =>
        (t.Namespace ?? string.Empty).Split('.').Any(s => s is "Cuda" or "Vulkan");

    private static bool HasGpuTrait(Type t) =>
        IsGpuTrait(t.GetCustomAttributesData())
        || t.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance
                        | BindingFlags.Static | BindingFlags.DeclaredOnly)
            .Any(m => IsGpuTrait(m.GetCustomAttributesData()));

    private static bool IsGpuTrait(IEnumerable<CustomAttributeData> attributes) =>
        attributes.Any(a => a.AttributeType == typeof(TraitAttribute)
            && a.ConstructorArguments.Count == 2
            && (string?)a.ConstructorArguments[0].Value == "Category"
            && (string?)a.ConstructorArguments[1].Value == "GPU");

    private static string? CollectionName(Type t) =>
        t.GetCustomAttributesData()
            .Where(a => a.AttributeType == typeof(CollectionAttribute) && a.ConstructorArguments.Count == 1)
            .Select(a => a.ConstructorArguments[0].Value as string)
            .FirstOrDefault();

    private static string CollectionDefinitionName(Type t) =>
        t.GetCustomAttributesData()
            .Where(a => a.AttributeType == typeof(CollectionDefinitionAttribute) && a.ConstructorArguments.Count == 1)
            .Select(a => a.ConstructorArguments[0].Value as string)
            .FirstOrDefault() ?? t.FullName!;

    private static string? FindProjectDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "DotLLM.Tests.Integration.csproj")))
                return dir.FullName;
        }
        return null;
    }

    private static IEnumerable<string> EnumerateSources(string projectDir)
    {
        string bin = Path.Combine(projectDir, "bin") + Path.DirectorySeparatorChar;
        string obj = Path.Combine(projectDir, "obj") + Path.DirectorySeparatorChar;
        return Directory.EnumerateFiles(projectDir, "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.StartsWith(bin, StringComparison.OrdinalIgnoreCase)
                        && !f.StartsWith(obj, StringComparison.OrdinalIgnoreCase));
    }
}
