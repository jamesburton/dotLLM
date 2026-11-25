using System.Security.Cryptography;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;
using Xunit.Abstractions;
using DotLLM.Tests.Integration.Fixtures;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1991 write guard. The opt-in UMA import (<c>DOTLLM_VULKAN_HOST_IMPORT_WRITABLE=1</c>; off by default) maps the GGUF
/// through a shared <b>writable</b> view (the driver refuses read-only pages; copy-on-write is accepted but privatises every
/// page, so it costs what staging costs). Writable means a
/// stray device write would reach the checkpoint on disk, so this asserts the opposite on a real load: SHA-256 and
/// last-write time of the file are unchanged after the import-engaged load plus prefill and decode.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanWeightImportWriteGuardTests(ITestOutputHelper output)
{
    private static string? FindSmolLm()
    {
        string? o = Environment.GetEnvironmentVariable("DOTLLM_SMOLLM_135M_Q8_0_GGUF");
        if (!string.IsNullOrEmpty(o)) return File.Exists(o) ? o : null;
        return TestFixtureResolver.ResolveFile([], "QuantFactory", "SmolLM-135M-GGUF", ["SmolLM-135M.Q8_0.gguf"], null).Path
            ?? TestFixtureResolver.ResolveFile([], "bartowski", "SmolLM2-135M-Instruct-GGUF", ["SmolLM2-135M-Instruct-Q8_0.gguf"], null).Path;
    }

    private static string Hash(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        return Convert.ToHexString(SHA256.HashData(fs));
    }

    [SkippableFact]
    public void ImportedLoadAndDecode_NeverWritesTheModelFile()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? path = FindSmolLm();
        Skip.If(path is null, "SmolLM-135M Q8_0 not found (set DOTLLM_SMOLLM_135M_Q8_0_GGUF).");

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasExternalMemoryHost && device.PhysicalDeviceTypeValue == VkPhysicalDeviceType.IntegratedGpu,
            "Needs an integrated device with VK_EXT_external_memory_host.");

        string? origDisable = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_HOST_IMPORT");
        string? origWritable = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_HOST_IMPORT_WRITABLE");
        try
        {
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_HOST_IMPORT", null);
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_HOST_IMPORT_WRITABLE", "1");   // the opt-in (#1991: default stays off, see docs/GPU.md)

            string before = Hash(path!);
            DateTime mtime = File.GetLastWriteTimeUtc(path!);

            using (var gguf = GgufFile.Open(path!))
            {
                var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
                using var model = VulkanTransformerModel.LoadFromGguf(device, gguf, config, spvDir);
                output.WriteLine(VulkanWeightImportPolicy.Summary());
                // Without this the hash proves nothing: the guard must run on the imported (writable-mapped) path.
                Assert.True(VulkanWeightImportPolicy.ImportedTensorCount > 0, "opt-in UMA load imported nothing: " + VulkanWeightImportPolicy.Summary());

                int[] ids = [1, 2, 3, 4, 5, 6, 7, 8];
                using (ITensor p = model.Forward(ids, [0, 1, 2, 3, 4, 5, 6, 7], deviceId: -1)) { }
                using (ITensor d = model.Forward([9], [8], deviceId: -1)) { }
            }

            Assert.Equal(before, Hash(path!));
            Assert.Equal(mtime, File.GetLastWriteTimeUtc(path!));
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_HOST_IMPORT", origDisable);
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_HOST_IMPORT_WRITABLE", origWritable);
        }
    }

    [SkippableFact]
    public void WritableAlias_EnvIsTriState()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        string? orig = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_HOST_IMPORT_WRITABLE");
        try
        {
            using var device = VulkanDevice.Create();
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_HOST_IMPORT_WRITABLE", "0");
            Assert.False(VulkanWeightImportPolicy.WritableAliasEnabled(device));
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_HOST_IMPORT_WRITABLE", "1");
            Assert.True(VulkanWeightImportPolicy.WritableAliasEnabled(device));
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_HOST_IMPORT_WRITABLE", null);
            Assert.False(
                VulkanWeightImportPolicy.WritableAliasEnabled(device));   // unset = off
        }
        finally { Environment.SetEnvironmentVariable("DOTLLM_VULKAN_HOST_IMPORT_WRITABLE", orig); }
    }
}
