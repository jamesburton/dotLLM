using DotLLM.Tests.Unit.Models;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>DFlash on the Vulkan <c>qwen35</c> trunk (issue #1175): GPU verify, CPU drafter, token-identical to greedy.</summary>
[Collection("VulkanKernels")]
public sealed class VulkanDFlashTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"dotllm-vk-dflash-{Guid.NewGuid():N}");
    private readonly ITestOutputHelper _out;
    private readonly List<VulkanDevice> _devices = [];

    public VulkanDFlashTests(ITestOutputHelper output)
    {
        _out = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        foreach (var d in _devices) d.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void SpeculativeDecode_WithDFlash_IsTokenIdenticalToGreedy(bool snapshots)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        DFlashDrafterTests.AssertTokenIdentical(_dir, (gguf, config) =>
        {
            var device = VulkanDevice.Create();
            _devices.Add(device);
            var m = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
            return (m, m.CreateKvCache(config.MaxSequenceLength));
        }, snapshots, _out);
    }
}
