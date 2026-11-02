using System.Reflection;
using System.Xml.Linq;
using Xunit;

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary><see cref="ExactPolicy"/> and the package's licence boundary.</summary>
public sealed class PolicyAndPackageTests
{
    public static TheoryData<ChunkRequestContext> Contexts()
    {
        var permissive = new ChunkCacheCapabilities
        {
            Versions = [1], Policies = ["exact", "shadow", "balanced", "fast", "auto"],
            Selectors = ["chunk_starts", "query_relevance"],
        };
        return new TheoryData<ChunkRequestContext>
        {
            new("m", 100, 0, null, false, false, false),
            new("m", 200_000, 150_000, permissive, true, true, true),
            new("m", 50_000, 49_000, permissive, false, false, true),
        };
    }

    /// <summary>
    /// Nothing lossy without an explicit opt-in (spec 5.5), and <see cref="ExactPolicy"/> is not that opt-in: even with
    /// the user's consent and a server offering every policy it returns the wire defaults.
    /// </summary>
    [Theory]
    [MemberData(nameof(Contexts))]
    public void ExactPolicy_AlwaysExact(ChunkRequestContext context)
    {
        ChunkRequestOptions o = ExactPolicy.Instance.Decide(context);

        Assert.Equal("exact", o.Policy);
        Assert.Null(o.Selector);
        Assert.Null(o.Budget);
        Assert.Equal("prefill", o.OnMiss);
        Assert.False(o.Strict);
        Assert.Equal("summary", o.Report);
        Assert.Equal(new ChunkRequestOptions(), o);
    }

    /// <summary>
    /// The package is MIT only while it links nothing GPL: no referenced assembly is a <c>DotLLM.*</c> engine assembly,
    /// and the project file has no project reference at all.
    /// </summary>
    [Fact]
    public void Package_HasNoDotLlmEngineReference()
    {
        Assembly client = typeof(ChunkCacheHttpClient).Assembly;
        Assert.Equal("DotLLM.ChunkCache.Client", client.GetName().Name);

        string[] dotllm = client.GetReferencedAssemblies().Select(a => a.Name!)
            .Where(n => n.StartsWith("DotLLM.", StringComparison.OrdinalIgnoreCase)).ToArray();
        Assert.True(dotllm.Length == 0, "client references: " + string.Join(", ", dotllm));

        XDocument csproj = XDocument.Load(Path.Combine(AppContext.BaseDirectory, "ClientProject", "DotLLM.ChunkCache.Client.csproj"));
        Assert.Empty(csproj.Descendants("ProjectReference"));
        Assert.Equal("MIT", csproj.Descendants("PackageLicenseExpression").Single().Value);
        Assert.Equal("DotLLM.ChunkCache.Client", csproj.Descendants("PackageId").Single().Value);

        string[] packages = csproj.Descendants("PackageReference").Select(p => (string)p.Attribute("Include")!).ToArray();
        Assert.All(packages, p => Assert.Contains(p, new[] { "Microsoft.Extensions.AI.Abstractions" }));
    }
}
