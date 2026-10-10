using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

public class VulkanMemoryCapacityTests
{
    private const long GiB = 1L << 30;
    private const VkMemoryHeapFlags DL = VkMemoryHeapFlags.DeviceLocal;
    private const VkMemoryHeapFlags NONE = (VkMemoryHeapFlags)0u;

    // Strix Halo, 512 MB BIOS split (measured 2026-10-07): heap0 GTT 34,890 MiB, heap1 DEVICE_LOCAL 69,780 MiB.
    private static readonly ulong[] NewSplit = [34890UL << 20, 69780UL << 20];
    private static readonly VkMemoryHeapFlags[] NewSplitFlags = [NONE, DL];

    [Fact]
    public void IntegratedGpu_CountsTheGttHeap_SoA71GiBModelFitsTheNewSplit()
    {
        long cap = VulkanMemoryCapacity.ResidentCapacityBytes(NewSplit, NewSplitFlags, VkPhysicalDeviceType.IntegratedGpu);
        Assert.Equal((34890L + 69780L) << 20, cap);
        // 122B Q4_K_M: 71.3 GiB payload * 1.15 headroom = 82 GiB. Device-local heap alone (68 GiB) rejected it.
        Assert.True(71.3 * GiB * 1.15 < cap);
        Assert.False(71.3 * GiB * 1.15 < 69780L << 20);
    }

    [Fact]
    public void DiscreteGpu_NeverCountsTheSystemMemoryHeap()
    {
        // 12 GiB VRAM + 32 GiB system heap: only VRAM is resident capacity.
        ulong[] sizes = [12UL << 30, 32UL << 30];
        VkMemoryHeapFlags[] flags = [DL, NONE];
        Assert.Equal(12 * GiB, VulkanMemoryCapacity.ResidentCapacityBytes(sizes, flags, VkPhysicalDeviceType.DiscreteGpu));
    }

    [Fact]
    public void OldCarveOutSplit_StillDeviceLocalDominated()
    {
        ulong[] sizes = [16UL << 30, 98304UL << 20];
        Assert.Equal(112 * GiB, VulkanMemoryCapacity.ResidentCapacityBytes(sizes, NewSplitFlags, VkPhysicalDeviceType.IntegratedGpu));
    }

    [Fact]
    public void NoHeaps_IsZero() =>
        Assert.Equal(0, VulkanMemoryCapacity.ResidentCapacityBytes([], [], VkPhysicalDeviceType.IntegratedGpu));
}

/// <summary>#880: the OS cap on integrated GPUs that the advertised heaps do not show.</summary>
public class VulkanUsableCapacityTests
{
    private const long GiB = 1L << 30;

    [Fact]
    public void Integrated_IsCappedAtTheUsableFractionOfRam()
    {
        // Strix Halo: 104.7 GiB of heaps, 127.1 GiB RAM -> ~80 GiB really usable (measured OOM at submit beyond ~80.5 GiB).
        long usable = VulkanMemoryCapacity.UsableCapacityBytes(105 * GiB, 127 * GiB, DotLLM.Vulkan.Interop.VkPhysicalDeviceType.IntegratedGpu);
        Assert.InRange(usable, 79 * GiB, 82 * GiB);
    }

    [Fact]
    public void Discrete_IsNotCapped_AndSmallHeapsWin()
    {
        Assert.Equal(12 * GiB, VulkanMemoryCapacity.UsableCapacityBytes(12 * GiB, 64 * GiB, DotLLM.Vulkan.Interop.VkPhysicalDeviceType.DiscreteGpu));
        Assert.Equal(8 * GiB, VulkanMemoryCapacity.UsableCapacityBytes(8 * GiB, 127 * GiB, DotLLM.Vulkan.Interop.VkPhysicalDeviceType.IntegratedGpu));
    }
}
