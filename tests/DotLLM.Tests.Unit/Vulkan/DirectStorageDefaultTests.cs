using System.Security.Cryptography;
using DotLLM.Vulkan;
using DotLLM.Vulkan.DirectStorage;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>#1994: DirectStorage as the default zero-copy route on Windows UMA, and the NuGet fetch that backs it.</summary>
[Collection("VulkanKernels")]
public class DirectStorageDefaultTests(ITestOutputHelper output)
{
    /// <summary>The full decision table (no GPU): an explicit 0/1 wins; unset needs Windows AND an integrated device AND the DLL.</summary>
    [Theory]
    [InlineData("0", true, true, true, false)]    // explicit off beats everything
    [InlineData("1", false, false, false, true)]  // explicit on beats everything (DLL permitting is checked later by the store)
    [InlineData(null, true, true, true, true)]    // the new default
    [InlineData(null, true, false, true, false)]  // discrete GPU: unchanged (staging)
    [InlineData(null, true, true, false, false)]  // DLL missing: staging (+ warning)
    [InlineData(null, false, true, true, false)]  // not Windows
    [InlineData("", true, true, true, true)]      // empty = unset
    public void Decide_TriState_Table(string? env, bool windows, bool integrated, bool dll, bool expected)
        => Assert.Equal(expected, DirectWeightStore.Decide(env, windows, integrated, dll));

    /// <summary><c>DOTLLM_VULKAN_DIRECT_STORAGE</c> is tri-state: 0 never, 1 always, unset = integrated device + DLL present only.</summary>
    [SkippableFact]
    public void Enabled_IsTriState_AndDefaultsToIntegratedWithDll()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        Skip.IfNot(OperatingSystem.IsWindows(), "Windows only");
        string? orig = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DIRECT_STORAGE");
        try
        {
            using var device = VulkanDevice.Create();
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DIRECT_STORAGE", "0");
            Assert.False(DirectWeightStore.EnabledFor(device));
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DIRECT_STORAGE", "1");
            Assert.True(DirectWeightStore.EnabledFor(device));
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DIRECT_STORAGE", null);
            bool integrated = device.PhysicalDeviceTypeValue == DotLLM.Vulkan.Interop.VkPhysicalDeviceType.IntegratedGpu;
            Assert.Equal(integrated && DirectStorageInstaller.IsAvailable(), DirectWeightStore.EnabledFor(device));
            output.WriteLine($"integrated={integrated} dll={DirectStorageInstaller.IsAvailable()} default={DirectWeightStore.EnabledFor(device)}");
        }
        finally { Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DIRECT_STORAGE", orig); }
    }

    /// <summary>
    /// End-to-end fetch into a temp directory: download from NuGet, verify the pinned SHA-256, show the licence, extract the two DLLs.
    /// Needs the network, so it only runs with <c>DOTLLM_TEST_NETWORK=1</c>. A refused licence must install nothing.
    /// </summary>
    [SkippableFact]
    public async Task Installer_FetchesVerifiesAndInstallsIntoDirectory_OnlyAfterAcceptance()
    {
        Skip.IfNot(DirectStorageInstaller.IsSupportedPlatform, "Windows x64/ARM64 only");
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_TEST_NETWORK") == "1", "set DOTLLM_TEST_NETWORK=1 to download from NuGet");
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-dsinstall-" + Guid.NewGuid().ToString("N")[..8]);
        string? orig = Environment.GetEnvironmentVariable("DOTLLM_DIRECTSTORAGE_DIR");
        try
        {
            Environment.SetEnvironmentVariable("DOTLLM_DIRECTSTORAGE_DIR", dir);

            // Declined licence: the package is downloaded and checked, but nothing is written.
            string? shown = null;
            Assert.Null(await DirectStorageInstaller.InstallAsync(t => { shown = t; return false; }));
            Assert.False(Directory.Exists(dir) && Directory.EnumerateFiles(dir).Any(), "a declined licence must install nothing");
            Assert.Contains("DirectStorage", shown, StringComparison.OrdinalIgnoreCase);

            // Accepted: both DLLs and the licence land in the dotLLM-owned directory.
            string? installed = await DirectStorageInstaller.InstallAsync(_ => true);
            Assert.Equal(dir, installed);
            foreach (string f in new[] { "dstorage.dll", "dstoragecore.dll", "LICENSE.txt" })
            {
                Assert.True(File.Exists(Path.Combine(dir, f)), f);
                output.WriteLine($"{f}: {new FileInfo(Path.Combine(dir, f)).Length} bytes sha256={Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Path.Combine(dir, f))))[..16]}");
            }
            Assert.NotNull(DirectStorageInstaller.FindDll());
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_DIRECTSTORAGE_DIR", orig);
            try { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }
}
