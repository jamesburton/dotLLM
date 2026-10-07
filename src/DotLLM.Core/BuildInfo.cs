using System.Reflection;

namespace DotLLM.Core;

/// <summary>
/// The version of the running build, as stamped by MinVer/SourceLink (e.g. <c>0.3.0-dev.2493+8df9b204…</c>).
/// </summary>
public static class BuildInfo
{
    /// <summary>
    /// The assembly informational version (<c>0.3.0-dev.N+sha</c>), falling back to the assembly version
    /// when no informational version is embedded. Lets a harness identify exactly what it is talking to.
    /// </summary>
    public static string Version { get; } = Resolve();

    private static string Resolve()
    {
        var asm = typeof(BuildInfo).Assembly;
        string? info = asm.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        if (!string.IsNullOrWhiteSpace(info))
            return info;
        return asm.GetName().Version?.ToString() ?? "unknown";
    }
}
