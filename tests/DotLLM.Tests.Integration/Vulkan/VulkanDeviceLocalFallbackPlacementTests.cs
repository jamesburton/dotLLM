using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Server.Endpoints;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #810: on the 512 MB BIOS split the strict DEVICE_LOCAL heap (heap 1, 68 GiB) can be exhausted
/// while the GTT heap (heap 0) still has room. These tests prove, against the real driver, WHERE the
/// fallback bytes land - not merely that an allocation succeeded. The control arm (no limit) must show
/// zero fallback bytes, the forced arm must move the overflow to a different heap and account for it.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class VulkanDeviceLocalFallbackPlacementTests(ITestOutputHelper output)
{
    private const long MiB = 1024 * 1024;

    private static void SkipIfVulkanNotServable() =>
        Skip.If(!DeviceEndpoint.Describe().Backends!.Any(b => b.Name == "vulkan" && b.Servable), "Vulkan is not servable on this machine.");

    private static int HeapHolding(VulkanDevice d, long minBytes, int exceptHeap = -1)
    {
        for (int h = 0; h < 16; h++)
            if (h != exceptHeap && d.LiveBytesOnHeap(h) >= minBytes) return h;
        return -1;
    }

    [SkippableFact]
    public void Control_NoLimit_DeviceLocalAllocationsStayOnPreferredHeap_WithZeroFallbackBytes()
    {
        SkipIfVulkanNotServable();
        using var dev = VulkanDevice.Create();
        dev.SetDeviceLocalLimitBytes(-1);
        using var a = dev.AllocateDeviceLocal(32 * MiB);
        using var b = dev.AllocateDeviceLocal(32 * MiB);
        output.WriteLine(dev.MemorySnapshot());
        int heap = HeapHolding(dev, 64 * MiB);
        Assert.True(heap >= 0, "both buffers should sit on one heap");
        for (int h = 0; h < 16; h++) Assert.Equal(0, dev.FallbackBytesOnHeap(h));
    }

    /// <summary>
    /// #1295: <c>DOTLLM_VRAM_BUDGET_MB</c> makes a new device behave as a card of that size: weights past the budget take the
    /// device-local fallback (offload) path, and greedy decoding is unchanged from the unconstrained load.
    /// </summary>
    [SkippableFact]
    public void SimulatedVramBudget_SmallModelTakesFallbackPath_GreedyIdenticalToUnconstrained()
    {
        SkipIfVulkanNotServable();
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_SMOLLM2_135M_Q8_0_GGUF", "bartowski", "SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q8_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("SmolLM2-135M Q8_0 GGUF"));

        var (unconstrained, _) = GreedyDecode(fixture.Path!, budgetMib: null);
        var (budgeted, fallbacks) = GreedyDecode(fixture.Path!, budgetMib: 64);   // the Q8_0 weights alone are ~138 MiB
        output.WriteLine($"unconstrained: {string.Join(' ', unconstrained)}");
        output.WriteLine($"64 MiB budget: {string.Join(' ', budgeted)} ({fallbacks} fallback allocations)");
        Assert.True(fallbacks > 0, "a 64 MiB budget must push allocations onto the fallback path");
        Assert.Equal(unconstrained, budgeted);
    }

    /// <summary>Prefill of a fixed prompt then greedy decode on a fresh device; returns the tokens and the fallback count.</summary>
    private static (int[] Tokens, long Fallbacks) GreedyDecode(string path, int? budgetMib)
    {
        const int steps = 8;
        Environment.SetEnvironmentVariable(DeviceMemoryBudget.SimulatedBudgetEnvVar, budgetMib?.ToString());
        try
        {
            using var gguf = GgufFile.Open(path);
            ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var dev = VulkanDevice.Create();   // the budget becomes this device's device-local limit at creation
            string spvDir = SpvDir();
            var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(dev, gguf, cfg, spvDir);
            using (model)
            {
                using var kv = kvFactory(64);
                int[] prompt = [1, 504, 3458, 282, 4649, 314];
                var tokens = new int[steps];
                int next = ArgmaxLastRow(model.Forward(prompt, Enumerable.Range(0, prompt.Length).ToArray(), deviceId: -1, kv), cfg.VocabSize);
                for (int s = 0; s < steps; s++)
                {
                    tokens[s] = next;
                    next = ArgmaxLastRow(model.Forward([next], [prompt.Length + s], deviceId: -1, kv), cfg.VocabSize);
                }
                return (tokens, dev.DeviceLocalFallbackCount);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(DeviceMemoryBudget.SimulatedBudgetEnvVar, null);
        }
    }

    private static string SpvDir()
    {
        for (string? probe = AppContext.BaseDirectory; probe is not null; probe = Path.GetDirectoryName(probe))
        {
            string candidate = Path.Combine(probe, "native", "vulkan", "spv");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("native/vulkan/spv not found above the test binaries.");
    }

    private static unsafe int ArgmaxLastRow(ITensor logits, int vocab)
    {
        using (logits)
        {
            int rows = logits.Shape.Rank == 2 ? logits.Shape[0] : 1;
            var last = new ReadOnlySpan<float>((void*)logits.DataPointer, rows * vocab).Slice((rows - 1) * vocab, vocab);
            int best = 0;
            for (int i = 1; i < last.Length; i++)
                if (last[i] > last[best]) best = i;
            return best;
        }
    }

    [SkippableFact]
    public void ForcedExhaustion_OverflowLandsOnADifferentHeap_AndIsAccountedAsFallback()
    {
        SkipIfVulkanNotServable();
        using var dev = VulkanDevice.Create();
        dev.SetDeviceLocalLimitBytes(48 * MiB);
        using var first = dev.AllocateDeviceLocal(32 * MiB);              // fits under the limit: preferred heap
        int preferred = HeapHolding(dev, 32 * MiB);
        Assert.True(preferred >= 0);
        for (int h = 0; h < 16; h++) Assert.Equal(0, dev.FallbackBytesOnHeap(h));

        using var second = dev.AllocateDeviceLocal(32 * MiB);             // 64 > 48: must fall back
        using var third = dev.AllocateDeviceLocal(32 * MiB);
        output.WriteLine(dev.MemorySnapshot());

        int fb = HeapHolding(dev, 64 * MiB, exceptHeap: preferred);
        Assert.True(fb >= 0, "overflow must land on a heap other than the exhausted one");
        Assert.NotEqual(preferred, fb);
        Assert.True(dev.FallbackBytesOnHeap(fb) >= 64 * MiB);
        Assert.Equal(0, dev.FallbackBytesOnHeap(preferred));
        Assert.True(dev.LiveBytesOnHeap(preferred) is >= 32 * MiB and < 64 * MiB, "preferred heap keeps only the first buffer");
    }
}
