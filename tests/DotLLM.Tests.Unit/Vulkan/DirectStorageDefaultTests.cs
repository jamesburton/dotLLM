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
    /// <summary>A dstorage.dll pair in the current directory alone is never found or loaded: only the app dir and the install dir are searched, by full path.</summary>
    [SkippableFact]
    public void Dll_InCurrentDirectory_IsNotUsed()
    {
        Skip.IfNot(DirectStorageInstaller.IsSupportedPlatform, "Windows x64/ARM64 only");
        string root = Path.Combine(Path.GetTempPath(), "dotllm-dscwd-" + Guid.NewGuid().ToString("N")[..8]);
        string cwd = Path.Combine(root, "cwd"), app = Path.Combine(root, "app"), inst = Path.Combine(root, "inst");
        Directory.CreateDirectory(cwd); Directory.CreateDirectory(app); Directory.CreateDirectory(inst);
        string old = Directory.GetCurrentDirectory();
        try
        {
            File.WriteAllText(Path.Combine(cwd, "dstorage.dll"), "planted");
            File.WriteAllText(Path.Combine(cwd, "dstoragecore.dll"), "planted");
            Directory.SetCurrentDirectory(cwd);
            Assert.Null(DirectStorageInstaller.FindDll(app, inst));
            // A bare name (what the default search path would resolve against the CWD) is refused outright, and a planted full path is not a valid image.
            Assert.False(DirectStorageInstaller.TryLoad("dstorage.dll", out _));
            Assert.False(DirectStorageInstaller.TryLoad("kernel32.dll", out _));   // any bare name: only full paths are accepted
            Assert.False(DirectStorageInstaller.TryLoad(Path.Combine(cwd, "dstorage.dll"), out _));
            // The directories that ARE searched do work, app dir first.
            File.WriteAllText(Path.Combine(inst, "dstorage.dll"), "x"); File.WriteAllText(Path.Combine(inst, "dstoragecore.dll"), "x");
            Assert.Equal(Path.Combine(inst, "dstorage.dll"), DirectStorageInstaller.FindDll(app, inst));
            File.WriteAllText(Path.Combine(app, "dstorage.dll"), "x"); File.WriteAllText(Path.Combine(app, "dstoragecore.dll"), "x");
            Assert.Equal(Path.Combine(app, "dstorage.dll"), DirectStorageInstaller.FindDll(app, inst));
        }
        finally
        {
            Directory.SetCurrentDirectory(old);
            try { Directory.Delete(root, recursive: true); } catch (IOException) { }
        }
    }

    /// <summary>The startup path prints no DirectStorage hint (it was false for CPU/CUDA/dGPU runs); the Vulkan integrated load owns that message.</summary>
    [Fact]
    public void Startup_NonInteractive_PrintsNoHint_EvenWhenDllMissing()
    {
        string? np = Environment.GetEnvironmentVariable("DOTLLM_NO_PROMPT");
        var oldErr = Console.Error; var oldOut = Console.Out;
        try
        {
            Environment.SetEnvironmentVariable("DOTLLM_NO_PROMPT", "1");
            var err = new StringWriter(); var o = new StringWriter();
            Console.SetError(err); Console.SetOut(o);
            foreach (string cmd in new[] { "run", "chat", "bench", "serve" })
                DotLLM.Cli.Helpers.DirectStoragePrompt.OfferIfMissing([cmd, "model.gguf", "-d", "cpu"], dllMissing: true);
            Assert.Equal(string.Empty, err.ToString());
            Assert.DoesNotContain("DirectStorage", o.ToString());
        }
        finally { Console.SetError(oldErr); Console.SetOut(oldOut); Environment.SetEnvironmentVariable("DOTLLM_NO_PROMPT", np); }
    }

    /// <summary>The model-level copied-bytes ledger survives the per-loader <c>Reset()</c> (so the one warning per model load has the right total).</summary>
    [Fact]
    public void ModelLedger_SurvivesLoaderReset_AndSumsTensorsAndBanks()
    {
        VulkanWeightImportPolicy.BeginModelLoad();
        VulkanWeightImportPolicy.NoteStaged(0, 100L << 20, "not_source_bytes");
        VulkanWeightImportPolicy.Reset();                       // the next weights loader starts
        VulkanWeightImportPolicy.NoteStaged(0, 50L << 20, "not_source_bytes");
        VulkanWeightImportPolicy.NoteBankStaged(10L << 20);
        Assert.Equal(160L << 20, VulkanWeightImportPolicy.ModelCopiedBytes);
        VulkanWeightImportPolicy.BeginModelLoad();
        Assert.Equal(0, VulkanWeightImportPolicy.ModelCopiedBytes);
    }

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
            foreach (string f in new[] { "dstorage.dll", "dstoragecore.dll", "LICENSE.txt", "ACCEPTED.txt" })
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
