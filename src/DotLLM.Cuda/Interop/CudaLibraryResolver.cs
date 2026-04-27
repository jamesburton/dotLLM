using System.Reflection;
using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

/// <summary>
/// Resolves "cuda" and "cublas" library names to platform-specific paths.
/// Linux: libcuda.so.1, libcublas.so. Windows: nvcuda.dll, cublas64_*.dll.
///
/// CUDA 13 relocated runtime DLLs from <c>bin\</c> to <c>bin\x64\</c> on Windows;
/// the installer no longer adds that subdir to PATH, so the bare-name load fails
/// unless a co-located copy is present. This resolver falls back to probing
/// <c>%CUDA_PATH%</c> (and versioned <c>CUDA_PATH_V13_1</c>, <c>CUDA_PATH_V12_*</c>, ...)
/// under both the new <c>bin\x64\</c> and the legacy <c>bin\</c> layouts.
/// </summary>
internal static class CudaLibraryResolver
{
    private static int _registered;

    private static readonly string[] CublasVersions = ["13", "12", "11"];

    /// <summary>
    /// Registers the resolver. Safe to call multiple times (idempotent).
    /// </summary>
    internal static void Register()
    {
        if (Interlocked.Exchange(ref _registered, 1) != 0) return;

        NativeLibrary.SetDllImportResolver(
            typeof(CudaLibraryResolver).Assembly,
            ResolveCudaLibrary);
    }

    private static nint ResolveCudaLibrary(
        string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == "cuda")
        {
            // nvcuda.dll ships with the driver in C:\Windows\System32; libcuda.so.1 is on the ld cache.
            string osLib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? "nvcuda.dll"
                : "libcuda.so.1";

            if (NativeLibrary.TryLoad(osLib, out nint handle))
                return handle;
        }

        if (libraryName == "cublas")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                foreach (var ver in CublasVersions)
                {
                    string fileName = $"cublas64_{ver}.dll";
                    if (NativeLibrary.TryLoad(fileName, out nint h))
                        return h;
                    if (TryLoadFromCudaInstall(fileName, out h))
                        return h;
                }
            }
            else
            {
                if (NativeLibrary.TryLoad("libcublas.so", out nint h))
                    return h;
                foreach (var ver in CublasVersions)
                {
                    if (NativeLibrary.TryLoad($"libcublas.so.{ver}", out nint h2))
                        return h2;
                }
            }
        }

        return 0;
    }

    /// <summary>
    /// Probes <c>%CUDA_PATH%</c> and versioned <c>CUDA_PATH_V*</c> variables for the DLL,
    /// checking both the CUDA-13 <c>bin\x64\</c> layout and the legacy <c>bin\</c> layout.
    /// </summary>
    private static bool TryLoadFromCudaInstall(string fileName, out nint handle)
    {
        foreach (string? root in EnumerateCudaRoots())
        {
            if (string.IsNullOrEmpty(root)) continue;

            string x64 = Path.Combine(root, "bin", "x64", fileName);
            if (File.Exists(x64) && NativeLibrary.TryLoad(x64, out handle))
                return true;

            string flat = Path.Combine(root, "bin", fileName);
            if (File.Exists(flat) && NativeLibrary.TryLoad(flat, out handle))
                return true;
        }

        handle = 0;
        return false;
    }

    private static IEnumerable<string?> EnumerateCudaRoots()
    {
        yield return Environment.GetEnvironmentVariable("CUDA_PATH");
        yield return Environment.GetEnvironmentVariable("CUDA_HOME");

        // Versioned vars like CUDA_PATH_V13_1, CUDA_PATH_V12_6, CUDA_PATH_V11_8.
        // Enumerating the process block lets us find whichever release the user has.
        var vars = Environment.GetEnvironmentVariables();
        foreach (System.Collections.DictionaryEntry e in vars)
        {
            if (e.Key is string key && key.StartsWith("CUDA_PATH_V", StringComparison.OrdinalIgnoreCase))
                yield return e.Value as string;
        }

        // Fallback: probe standard install dirs when env vars are absent or
        // stale (e.g. CUDA_PATH still points at an older v11.8 alongside a
        // newer toolkit). Newest version first by lexical sort over the
        // "v<major>.<minor>" subdir names ("v13.1" > "v12.6" > "v11.8").
        if (OperatingSystem.IsWindows())
        {
            string baseDir = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA";
            if (Directory.Exists(baseDir))
            {
                foreach (var dir in Directory.EnumerateDirectories(baseDir)
                    .OrderByDescending(d => d, StringComparer.OrdinalIgnoreCase))
                {
                    yield return dir;
                }
            }
        }
        else
        {
            yield return "/usr/local/cuda";
            const string linuxBase = "/usr/local";
            if (Directory.Exists(linuxBase))
            {
                foreach (var dir in Directory.EnumerateDirectories(linuxBase, "cuda-*")
                    .OrderByDescending(d => d, StringComparer.Ordinal))
                {
                    yield return dir;
                }
            }
        }
    }
}
