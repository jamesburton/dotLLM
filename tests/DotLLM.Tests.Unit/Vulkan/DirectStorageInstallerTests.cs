using DotLLM.Vulkan.DirectStorage;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>#1304: the consent-gated DirectStorage download. Needs network; skips when offline.</summary>
[Collection("DirectStorageEnv")]
public sealed class DirectStorageInstallerTests
{
    [SkippableFact]
    public async Task Decline_InstallsNothing_AcceptExtractsVerifiedDlls()
    {
        Skip.IfNot(DirectStorageInstaller.IsSupportedPlatform, "Windows x64/ARM64 only.");
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-ds-" + Guid.NewGuid().ToString("N"));
        string? old = Environment.GetEnvironmentVariable("DOTLLM_DIRECTSTORAGE_DIR");
        Environment.SetEnvironmentVariable("DOTLLM_DIRECTSTORAGE_DIR", dir);
        try
        {
            string? shown = null;
            string? declined;
            try { declined = await DirectStorageInstaller.InstallAsync(t => { shown = t; return false; }); }
            catch (IOException e) when (e.InnerException is HttpRequestException) { throw new SkipException("offline: " + e.Message); }
            Assert.Null(declined);
            Assert.Contains("MICROSOFT SOFTWARE LICENSE TERMS", shown);
            Assert.False(Directory.Exists(dir), "declining must leave nothing behind");

            string? installed = await DirectStorageInstaller.InstallAsync(_ => true);
            Assert.Equal(dir, installed);
            Assert.True(File.Exists(Path.Combine(dir, "dstorage.dll")));
            Assert.True(File.Exists(Path.Combine(dir, "dstoragecore.dll")));
            Assert.True(File.Exists(Path.Combine(dir, "LICENSE.txt")));
            Assert.NotNull(DirectStorageInstaller.FindDll());
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_DIRECTSTORAGE_DIR", old);
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }
}
