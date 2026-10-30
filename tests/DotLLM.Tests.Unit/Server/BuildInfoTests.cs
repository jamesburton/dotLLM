using System.Reflection;
using DotLLM.Core;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>#774: <c>--version</c> / <c>/props</c> report the assembly informational version, not a literal.</summary>
public class BuildInfoTests
{
    [Fact]
    public void Version_IsTheAssemblyInformationalVersion()
    {
        string? expected = typeof(BuildInfo).Assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;

        Assert.False(string.IsNullOrWhiteSpace(BuildInfo.Version));
        if (!string.IsNullOrWhiteSpace(expected))
            Assert.Equal(expected, BuildInfo.Version);
    }
}
