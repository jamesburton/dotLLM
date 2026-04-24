using System.Reflection;
using System.Runtime.InteropServices;

namespace DotLLM.Hip.Interop;

/// <summary>
/// Resolves "amdhip64" and "hipblas" library names to platform-specific paths.
/// Linux: libamdhip64.so, libhipblas.so. Windows: amdhip64.dll, hipblas.dll.
/// </summary>
internal static class HipLibraryResolver
{
    private static int _registered;

    /// <summary>
    /// Registers the resolver. Safe to call multiple times (idempotent).
    /// </summary>
    internal static void Register()
    {
        if (Interlocked.Exchange(ref _registered, 1) != 0) return;

        NativeLibrary.SetDllImportResolver(
            typeof(HipLibraryResolver).Assembly,
            ResolveHipLibrary);
    }

    private static nint ResolveHipLibrary(
        string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == "amdhip64")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("amdhip64.dll", out nint handle))
                    return handle;
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
}
