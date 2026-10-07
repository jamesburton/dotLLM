using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

public class VulkanMemoryPlacementTests
{
    private const VkMemoryPropertyFlags DL = VkMemoryPropertyFlags.DeviceLocal;
    private const VkMemoryPropertyFlags HV = VkMemoryPropertyFlags.HostVisible;
    private const VkMemoryPropertyFlags HC = VkMemoryPropertyFlags.HostCoherent;
    private const VkMemoryPropertyFlags CA = VkMemoryPropertyFlags.HostCached;

    // Strix Halo, 512 MB BIOS split (measured 2026-10-07): heap0 = 34,890 MiB GTT, heap1 = 69,780 MiB DEVICE_LOCAL.
    private static readonly ulong[] Heaps = [34890UL << 20, 69780UL << 20];
    private static readonly VkMemTypeInfo[] Types =
    [
        new(DL, 1),              // 0 strict device-local (what weights ask for)
        new(HV | HC, 0),         // 1 GTT, uncached
        new(DL | HV | HC, 1),    // 2 combined, same heap as 0
        new(HV | HC | CA, 0),    // 3 GTT, cached
    ];

    [Fact]
    public void ExhaustedDeviceLocalHeap_FallsToHeap0Uncached_BeforeSameHeapTypes()
    {
        var order = VulkanMemoryPlacement.RankDeviceLocalFallback(Types, Heaps, 0xF, failedType: 0);
        // heap-0 uncached first, then heap-0 cached, same-heap combined type only last.
        Assert.Equal(new uint[] { 1, 3, 2 }, order);
    }

    [Fact]
    public void FailedTypeAndTypesOutsideTypeBits_AreNeverReturned()
    {
        var order = VulkanMemoryPlacement.RankDeviceLocalFallback(Types, Heaps, 0b0101, failedType: 0);
        Assert.DoesNotContain(0u, order);
        Assert.DoesNotContain(1u, order);
        Assert.DoesNotContain(3u, order);
        Assert.Equal(new uint[] { 2 }, order);
    }

    [Fact]
    public void OtherHeapCombinedType_RanksAheadOfPlainHostVisible_AndLargerHeapFirst()
    {
        // Discrete-style table: VRAM heap0, system heap1, small BAR heap2.
        ulong[] heaps = [8UL << 30, 32UL << 30, 256UL << 20];
        VkMemTypeInfo[] t = [new(DL, 0), new(HV | HC, 1), new(DL | HV | HC, 2), new(HV | HC, 2)];
        var order = VulkanMemoryPlacement.RankDeviceLocalFallback(t, heaps, 0xF, 0);
        Assert.Equal(new uint[] { 2, 1, 3 }, order);
    }

    [Fact]
    public void NoEligibleType_ReturnsEmpty()
    {
        VkMemTypeInfo[] t = [new(DL, 0)];
        Assert.Empty(VulkanMemoryPlacement.RankDeviceLocalFallback(t, [1UL << 30], 1, 0));
    }
}
