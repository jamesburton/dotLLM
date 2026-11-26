using System.Reflection;
using System.Security.Cryptography;

namespace DotLLM.Server.Tuning;

/// <summary>
/// The build's performance identity (#1902). It is not the commit: a dev rebuild that touches no kernel keeps the same identity. Assembly
/// MVIDs are content-derived under deterministic builds, so they move only when an assembly's code does; thresholds and CPU kernels live
/// in managed code, and the shaders that load are the copies in the output directory.
/// </summary>
public static class PerfIdentities
{
    private static readonly string[] s_kernelAssemblies = ["DotLLM.Cpu", "DotLLM.Cuda", "DotLLM.Vulkan", "DotLLM.Engine", "DotLLM.Models"];
    private static PerfIdentity? s_current;

    /// <summary>The identity of this process's build. The default-root result is computed once per process.</summary>
    /// <param name="kernelRoot">Directory to scan for shader and native files; null is the application output directory.</param>
    public static PerfIdentity Current(string? kernelRoot = null)
    {
        if (kernelRoot is null && s_current is { } cached) return cached;
        var id = new PerfIdentity
        {
            ReleaseVersion = ReleaseVersion(),
            KernelFilesHash = KernelFilesHash(kernelRoot ?? AppContext.BaseDirectory),
            KernelAssembliesHash = KernelAssembliesHash(),
            RuntimeVersion = Environment.Version.ToString(),
        };
        if (kernelRoot is null) s_current = id;
        return id;
    }

    private static string ReleaseVersion()
    {
        string? v = typeof(PerfIdentities).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
                    ?? typeof(PerfIdentities).Assembly.GetName().Version?.ToString();
        if (v is null) return "unknown";
        int plus = v.IndexOf('+');
        return plus >= 0 ? v[..plus] : v;
    }

    /// <summary>SHA-256 over the sorted (relative path, SHA-256) of <c>*.spv</c>, <c>*.ptx</c> and native libraries under <paramref name="root"/>.</summary>
    internal static string KernelFilesHash(string root)
    {
        var lines = new List<string>();
        if (Directory.Exists(root))
        {
            foreach (string f in Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories))
            {
                string rel = Path.GetRelativePath(root, f).Replace('\\', '/');
                string ext = Path.GetExtension(f).ToLowerInvariant();
                bool kernel = ext is ".spv" or ".ptx"
                              || (ext is ".dll" or ".so" or ".dylib"
                                  && (Path.GetFileName(f).Contains("dotllm", StringComparison.OrdinalIgnoreCase)
                                      || rel.Contains("/native/", StringComparison.Ordinal)));
                if (!kernel) continue;
                using var fs = new FileStream(f, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
                lines.Add(rel + "\t" + Convert.ToHexString(SHA256.HashData(fs)).ToLowerInvariant());
            }
        }
        lines.Sort(StringComparer.Ordinal);
        return ModelHash.Sha256Hex(string.Join('\n', lines));
    }

    private static string KernelAssembliesHash()
    {
        var lines = new List<string>();
        foreach (string name in s_kernelAssemblies)
        {
            try
            {
                var asm = AppDomain.CurrentDomain.GetAssemblies().FirstOrDefault(a => a.GetName().Name == name)
                          ?? Assembly.Load(new AssemblyName(name));
                lines.Add(name + "\t" + asm.ManifestModule.ModuleVersionId);
            }
            catch (Exception ex) when (ex is FileNotFoundException or FileLoadException or BadImageFormatException)
            {
                // not part of this build
            }
        }
        lines.Sort(StringComparer.Ordinal);
        return ModelHash.Sha256Hex(string.Join('\n', lines));
    }
}
