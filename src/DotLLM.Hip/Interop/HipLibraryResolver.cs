using System.Reflection;
using System.Runtime.InteropServices;

namespace DotLLM.Hip.Interop;

/// <summary>
/// Resolves "amdhip64" and "hipblas" library names to platform-specific paths.
/// Linux: libamdhip64.so, libhipblas.so. Windows: amdhip64.dll, hipblas.dll.
/// </summary>
internal static class HipLibraryResolver
{
    private static readonly bool Registered;

    /// <summary>
    /// Issue #504: see <c>CudaLibraryResolver</c>'s note — the flag-based guard claimed the flag
    /// before installing the resolver, so a concurrent caller could return early and then P/Invoke
    /// <c>amdhip64</c>/<c>hipblas</c> with no mapping. The type-initialization lock closes that window.
    /// </summary>
    static HipLibraryResolver()
    {
        NativeLibrary.SetDllImportResolver(
            typeof(HipLibraryResolver).Assembly,
            ResolveHipLibrary);
        Registered = true;
    }

    /// <summary>
    /// Ensures the resolver is installed; guaranteed complete on return. The body reads
    /// <see cref="Registered"/> so the type-initialization trigger cannot be elided.
    /// </summary>
    internal static void Register()
    {
        if (!Registered)
            throw new InvalidOperationException("HIP library resolver registration did not complete.");
    }

    private static nint ResolveHipLibrary(
        string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == "amdhip64")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                foreach (var candidate in WindowsRuntimeCandidates(
                             Environment.GetEnvironmentVariable,
                             File.Exists,
                             Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)))
                {
                    if (NativeLibrary.TryLoad(candidate, out nint handle))
                        return handle;
                }
            }
            else
            {
                // Modern ROCm uses libamdhip64.so.<major>; try a few flavors.
                foreach (var name in new[] { "libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5" })
                {
                    if (NativeLibrary.TryLoad(name, out nint h))
                        return h;
                }
            }
        }

        if (libraryName == "hipblas")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("hipblas.dll", out nint handle))
                    return handle;
            }
            else
            {
                foreach (var name in new[] { "libhipblas.so", "libhipblas.so.2", "libhipblas.so.1", "libhipblas.so.0" })
                {
                    if (NativeLibrary.TryLoad(name, out nint h))
                        return h;
                }
            }
        }

        return 0; // fall through to default resolution
    }

    /// <summary>
    /// Windows HIP runtimes to try, newest-first by intent (#1336). A user-space ROCm install (the pip/TheRock SDK:
    /// <c>DOTLLM_HIP_PATH</c>, <c>HIP_PATH</c>, <c>ROCM_PATH</c> or <c>~/.dotllm/rocm</c>) is newer than whatever the
    /// Radeon driver drops in System32, so its full paths come first. Then the versioned driver DLLs by name, highest
    /// major first. The bare <c>amdhip64.dll</c> is last: on current drivers it is a legacy HIP 5.7 shim that rejects
    /// gfx1151 code objects with <c>hipErrorInvalidKernelFile</c> (218), but it is the only name older installs have.
    /// </summary>
    /// <param name="getEnv">Environment lookup.</param>
    /// <param name="fileExists">File existence check.</param>
    /// <param name="userProfile">The user's profile directory.</param>
    /// <returns>Library paths or names, in load-attempt order.</returns>
    internal static IEnumerable<string> WindowsRuntimeCandidates(
        Func<string, string?> getEnv, Func<string, bool> fileExists, string userProfile)
    {
        string[] versioned = ["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"];
        string?[] roots =
        [
            getEnv("DOTLLM_HIP_PATH"), getEnv("HIP_PATH"), getEnv("ROCM_PATH"),
            string.IsNullOrEmpty(userProfile) ? null : Path.Combine(userProfile, ".dotllm", "rocm", "sdk", "_rocm_sdk_core"),
        ];
        foreach (var root in roots)
        {
            if (string.IsNullOrWhiteSpace(root))
                continue;
            foreach (var dir in new[] { Path.Combine(root, "bin"), root })
            {
                foreach (var name in versioned)
                {
                    string full = Path.Combine(dir, name);
                    if (fileExists(full))
                        yield return full;
                }
            }
        }
        // System32 / PATH, by name.
        foreach (var name in versioned)
            yield return name;
    }
}
