using DotLLM.Server.Endpoints;
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
