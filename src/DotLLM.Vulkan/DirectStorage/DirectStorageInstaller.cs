using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace DotLLM.Vulkan.DirectStorage;

/// <summary>
/// Locates, and with the user's consent installs, Microsoft's <c>dstorage.dll</c> / <c>dstoragecore.dll</c> (#1304).
/// dotLLM does not redistribute these binaries (the DirectStorage SDK licence makes redistribution conditional on terms
/// and an indemnity), so the user fetches the package from NuGet themselves, after accepting Microsoft's licence text.
/// The package is pinned by SHA-256, extracted into <c>~/.dotllm/native/directstorage/{version}/{arch}</c>, and
/// <see cref="DirectWeightStore"/> searches that directory.
/// </summary>
public static class DirectStorageInstaller
{
    /// <summary>The pinned <c>Microsoft.Direct3D.DirectStorage</c> package version.</summary>
    public const string PackageVersion = "1.3.0";

    private const string PackageUrl =
        "https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.directstorage/" + PackageVersion
        + "/microsoft.direct3d.directstorage." + PackageVersion + ".nupkg";

    private const string PackageSha256 = "8ab6c1082537565388b79050d94fdd7f5f437cd0f1d909e7b19f786042b97177";

    private static readonly string[] s_files = ["dstorage.dll", "dstoragecore.dll"];

    /// <summary>Where the user-level install lives (<c>DOTLLM_DIRECTSTORAGE_DIR</c> overrides it); null when the platform has no DirectStorage build.</summary>
    public static string? InstallDirectory
    {
        get
        {
            string? arch = PackageArchFolder();
            if (arch is null) return null;
            string? overrideDir = Environment.GetEnvironmentVariable("DOTLLM_DIRECTSTORAGE_DIR");
            if (!string.IsNullOrEmpty(overrideDir)) return overrideDir;
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "native", "directstorage", PackageVersion, arch);
        }
    }

    /// <summary>True on Windows x64/ARM64, the platforms Microsoft ships DirectStorage for that dotLLM targets.</summary>
    public static bool IsSupportedPlatform => OperatingSystem.IsWindows() && PackageArchFolder() is not null;

    /// <summary>
    /// Full path of a usable <c>dstorage.dll</c> (with <c>dstoragecore.dll</c> beside it), or null. Searched in this order only: the application
    /// directory, then <see cref="InstallDirectory"/> (<c>DOTLLM_DIRECTSTORAGE_DIR</c> or <c>~/.dotllm/native/directstorage</c>). Never the current
    /// directory or <c>PATH</c>: the DLL is loaded by full path (<see cref="TryLoad"/>), so a planted copy elsewhere cannot be picked up.
    /// </summary>
    public static string? FindDll() => FindDll(
        string.Equals(Environment.GetEnvironmentVariable("DOTLLM_DIRECTSTORAGE_SKIP_APPDIR"), "1", StringComparison.Ordinal) ? null : AppContext.BaseDirectory,
        InstallDirectory);   // SKIP_APPDIR=1: diagnostic/test knob, search only the install directory (reproduces the post-fetch layout)

    /// <summary><see cref="FindDll()"/> with explicit directories (testable).</summary>
    internal static string? FindDll(string? appDir, string? installDir)
    {
        if (!IsSupportedPlatform) return null;
        foreach (string? dir in new[] { appDir, installDir })
        {
            if (string.IsNullOrEmpty(dir)) continue;
            string candidate = Path.GetFullPath(Path.Combine(dir, "dstorage.dll"));
            if (File.Exists(candidate) && File.Exists(Path.Combine(dir, "dstoragecore.dll"))) return candidate;
        }
        return null;
    }

    /// <summary>True when <see cref="FindDll()"/> finds the DLL pair (no default-search-path probing).</summary>
    public static bool IsAvailable() => FindDll() is not null;

    private const uint LoadLibrarySearchDllLoadDir = 0x00000100;
    private const uint LoadLibrarySearchSystem32 = 0x00000800;

    /// <summary>
    /// Loads <c>dstorage.dll</c> from <paramref name="fullPath"/> with <c>LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32</c>: its
    /// own folder (so <c>dstoragecore.dll</c> resolves beside it) and System32, and nothing else (not the current directory, not <c>PATH</c>).
    /// </summary>
    internal static bool TryLoad(string fullPath, out nint module)
    {
        module = 0;
        if (!OperatingSystem.IsWindows() || !Path.IsPathFullyQualified(fullPath)) return false;
        module = D3dNative.LoadLibraryEx(fullPath, 0, LoadLibrarySearchDllLoadDir | LoadLibrarySearchSystem32);
        return module != 0;
    }

    /// <summary>Marker recording that the user declined the offer, so it is made once.</summary>
    public static string DeclinedMarkerPath =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "native", "directstorage", "declined");

    /// <summary>
    /// Downloads the pinned package, verifies it, and hands the licence text to <paramref name="accept"/>.
    /// Only when that returns true are the two DLLs extracted (alongside <c>LICENSE.txt</c>); otherwise nothing is installed.
    /// </summary>
    /// <param name="accept">Shows the licence and returns whether the user agreed.</param>
    /// <param name="ct">Cancellation.</param>
    /// <returns>The directory containing <c>dstorage.dll</c>, or null when the user declined.</returns>
    /// <exception cref="IOException">The download or package verification failed.</exception>
    public static async Task<string?> InstallAsync(Func<string, bool> accept, CancellationToken ct = default)
    {
        string? target = InstallDirectory;
        if (target is null || !OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("DirectStorage is available on Windows x64 and ARM64 only.");

        byte[] package;
        using (var http = new HttpClient { Timeout = TimeSpan.FromMinutes(2) })
        {
            http.DefaultRequestHeaders.UserAgent.ParseAdd("dotLLM-directstorage-installer");
            try { package = await http.GetByteArrayAsync(PackageUrl, ct).ConfigureAwait(false); }
            catch (HttpRequestException e) { throw new IOException($"Could not download {PackageUrl}: {e.Message}", e); }
        }

        string hash = Convert.ToHexString(SHA256.HashData(package)).ToLowerInvariant();
        if (!string.Equals(hash, PackageSha256, StringComparison.Ordinal))
            throw new IOException($"Downloaded package hash {hash} does not match the pinned {PackageSha256}; nothing was installed.");

        using var zip = new ZipArchive(new MemoryStream(package, writable: false), ZipArchiveMode.Read);
        ZipArchiveEntry license = zip.GetEntry("LICENSE.txt")
            ?? throw new IOException("The package has no LICENSE.txt; nothing was installed.");
        string licenseText;
        using (var reader = new StreamReader(license.Open()))
            licenseText = await reader.ReadToEndAsync(ct).ConfigureAwait(false);

        if (!accept(licenseText)) return null;

        string arch = PackageArchFolder()!;
        Directory.CreateDirectory(target);
        foreach (string name in s_files)
        {
            ZipArchiveEntry entry = zip.GetEntry($"native/bin/{arch}/{name}")
                ?? throw new IOException($"The package has no native/bin/{arch}/{name}; nothing was installed.");
            string tmp = Path.Combine(target, name + ".tmp");
            entry.ExtractToFile(tmp, overwrite: true);
            File.Move(tmp, Path.Combine(target, name), overwrite: true);
        }
        File.WriteAllText(Path.Combine(target, "LICENSE.txt"), licenseText);
        // #1994: record the acceptance itself (when, which package, which licence text), not just a copy of the licence.
        File.WriteAllText(Path.Combine(target, "ACCEPTED.txt"),
            $"accepted_utc={DateTime.UtcNow:O}\npackage=Microsoft.Direct3D.DirectStorage {PackageVersion}\npackage_sha256={PackageSha256}\n" +
            $"license_sha256={Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(licenseText))).ToLowerInvariant()}\n");
        try { File.Delete(DeclinedMarkerPath); } catch (IOException) { }
        return target;
    }

    private static string? PackageArchFolder() => RuntimeInformation.ProcessArchitecture switch
    {
        Architecture.X64 => "x64",
        Architecture.Arm64 => "ARM64",
        _ => null,
    };
}
