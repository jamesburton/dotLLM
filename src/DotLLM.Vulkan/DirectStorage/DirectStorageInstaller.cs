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
    /// Path to a usable <c>dstorage.dll</c> (next to the assembly, or in <see cref="InstallDirectory"/>), or null.
    /// A copy that is only on the system search path is not reported here; <see cref="IsAvailable"/> covers that.
    /// </summary>
    public static string? FindDll()
    {
        if (!IsSupportedPlatform) return null;
        string? asmDir = Path.GetDirectoryName(typeof(DirectStorageInstaller).Assembly.Location);
        foreach (string? dir in new[] { asmDir, InstallDirectory })
        {
            if (string.IsNullOrEmpty(dir)) continue;
            string candidate = Path.Combine(dir, "dstorage.dll");
            if (File.Exists(candidate) && File.Exists(Path.Combine(dir, "dstoragecore.dll"))) return candidate;
        }
        return null;
    }

    /// <summary>True when <c>dstorage.dll</c> can be found anywhere the loader will look.</summary>
    public static bool IsAvailable()
    {
        if (!IsSupportedPlatform) return false;
        if (FindDll() is not null) return true;
        if (!NativeLibrary.TryLoad("dstorage.dll", out nint module)) return false;
        NativeLibrary.Free(module);
        return true;
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
