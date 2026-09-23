using System.IO.MemoryMappedFiles;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The guard that was missing: does the zero-copy host import engage on a <b>real,
/// mmap-backed GGUF load</b>?
/// </summary>
/// <remarks>
/// <para>
/// Every other import test allocates its source with
/// <c>NativeMemory.AlignedAlloc</c> — private, read-write, page-aligned memory that the
/// driver is happy to import. Production sources are <c>GgufFile</c>'s
/// <see cref="System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read"/> view. Those
/// tests were green while the import aliased <b>zero bytes of every real model</b>,
/// measured on gfx1151: <c>0 tensors / 0.0 MiB aliased; 211 tensors / 136.3 MiB staged
/// (last fallback: import_rejected)</c> for SmolLM-135M Q8_0 on the long-standing
/// <c>VulkanWeights</c> path, and the same for Bonsai 2 27B on the paths #508 added.
/// </para>
/// <para>
/// So this test asserts the OUTCOME, not the mechanism: after loading a real checkpoint
/// on a device that advertises the extension and is integrated (the two conditions
/// under which the import is supposed to fire), at least one tensor must have been
/// aliased. It fails — loudly, with the driver's own refusal stage and
/// <c>VkResult</c> — rather than skipping, because a silent zero is exactly the failure
/// mode that shipped.
/// </para>
/// <para>
/// <b>If this test is red</b>, read
/// <see cref="VulkanHostImportMmapAccessModeTests"/>: it imports the same bytes through
/// read-write, read-only and copy-on-write mappings and names which one the driver
/// refuses.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanRealLoadHostImportTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _output = output;

    private static string? FindSmolLm135M_Q8_0()
    {
        string? overridePath = Environment.GetEnvironmentVariable("DOTLLM_SMOLLM_135M_Q8_0_GGUF");
        if (!string.IsNullOrEmpty(overridePath))
            return File.Exists(overridePath) ? overridePath : null;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "test-cache", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf"),
            Path.Combine(home, ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf"),
            Path.Combine(home, ".dotllm", "test-cache", "bartowski", "SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q8_0.gguf"),
        ];
        return candidates.FirstOrDefault(File.Exists);
    }

    /// <summary>
    /// Load a real Q8_0 checkpoint through the production Vulkan weight path and assert the
    /// import outcome that the current <see cref="GgufFile.MappingAccess"/> mode entails:
    /// copy-on-write MUST alias at least one tensor, read-only MUST alias none and say why.
    /// Neither branch can be silently green while the import is broken.
    /// </summary>
    [SkippableFact]
    public void RealGgufLoad_AliasesAtLeastOneTensor()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? path = FindSmolLm135M_Q8_0();
        Skip.If(path is null,
            "SmolLM-135M Q8_0 not found (set DOTLLM_SMOLLM_135M_Q8_0_GGUF).");

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasExternalMemoryHost,
            "Driver does not expose VK_EXT_external_memory_host — the import cannot fire here.");
        Skip.IfNot(
            device.PhysicalDeviceTypeValue is VkPhysicalDeviceType.IntegratedGpu
                or VkPhysicalDeviceType.Cpu,
            $"Physical device type {device.PhysicalDeviceTypeValue} is not integrated — " +
            "the import is refused by design (issue #507).");

        string? originalDisable = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_HOST_IMPORT");
        try
        {
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_HOST_IMPORT", null);

            using var gguf = GgufFile.Open(path!);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = VulkanTransformerModel.LoadFromGguf(device, gguf, config, spvDir);

            string ledger = VulkanWeightImportPolicy.Summary();
            _output.WriteLine($"GgufFile.MappingAccess = {GgufFile.MappingAccess}");
            _output.WriteLine(ledger);

            if (GgufFile.MappingAccess == MemoryMappedFileAccess.CopyOnWrite)
            {
                // The mode in which the import is SUPPOSED to work. This is the guard: if the
                // import ever stops engaging on a real load again, this goes red.
                Assert.True(
                    VulkanWeightImportPolicy.ImportedTensorCount > 0,
                    "The zero-copy host import aliased NOTHING on a real mmap-backed GGUF load, on " +
                    "an integrated device that advertises VK_EXT_external_memory_host, with the " +
                    $"GGUF mapped copy-on-write. Ledger: {ledger}. " +
                    "Run VulkanHostImportMmapAccessModeTests for the per-mapping-mode verdict.");

                // Something imported, so #438's whole-file gguf.Dispose() must now be refused —
                // the composition rule evaluated on a real load, not on hand-built ledger entries.
                Assert.False(VulkanWeightImportPolicy.MayReleaseWholeHostMapping,
                    "Tensors were imported, so the source mapping is live device memory and the " +
                    "whole-mapping release must be blocked.");
            }
            else
            {
                // The DEFAULT mode, and a deliberate characterization rather than an accident:
                // amdvlk refuses vkAllocateMemory with VK_ERROR_INVALID_EXTERNAL_HANDLE on
                // read-only pages, so the import cannot fire and every tensor stages. Pinning
                // it here means that if a driver or mapping change ever makes the read-only
                // import work, we are told rather than left guessing.
                Assert.Equal(0, VulkanWeightImportPolicy.ImportedTensorCount);
                Assert.Contains("import_rejected", VulkanWeightImportPolicy.LastFallbackReason, StringComparison.Ordinal);
                Assert.True(VulkanWeightImportPolicy.MayReleaseWholeHostMapping,
                    "Nothing imported, so #438 may release the whole mapping.");
                _output.WriteLine(
                    "Read-only mapping: the import is known-dead here by driver refusal. " +
                    "Set DOTLLM_GGUF_MAP_COW=1 to exercise the working path.");
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_HOST_IMPORT", originalDisable);
        }
    }
}
