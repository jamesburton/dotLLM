using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1174: <see cref="VulkanWeightArena"/> packs small and medium tensors into shared 64 MiB slabs. Every sub-buffer must address its own
/// bytes (distinct offsets, nothing overlaps), host transfers must stage rather than map shared memory, and a slab's memory must outlive
/// the arena until its last tensor is disposed.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanWeightArenaTests
{
    [SkippableFact]
    public void SubBuffers_HoldDistinctData_AndFreeTheirSlabsWithTheLastTensor()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        long baseAllocs = device.LiveAllocationCount;

        // 1 KiB .. 40 MiB: several land in one slab, the 40 MiB pair cannot share one, and a 64 MiB tensor goes direct.
        long[] sizes = [1024, 4 << 20, 40 << 20, 40 << 20, 5_677_056, 256, 64L << 20];
        var buffers = new List<VulkanDevice.Buffer>();
        using (var arena = new VulkanWeightArena(device))
        {
            foreach (long s in sizes) buffers.Add(arena.AllocateDeviceLocal(s));
            Assert.Equal(2, arena.SlabCount);
        }

        Assert.False(buffers[^1].IsSubAllocated);
        for (int i = 0; i < sizes.Length - 1; i++)
        {
            Assert.True(buffers[i].IsSubAllocated);
            Assert.False(buffers[i].IsHostVisible);
        }

        // Distinct sub-buffers of one slab must not alias.
        var bySlab = buffers.Take(sizes.Length - 1).GroupBy(b => b.Memory);
        foreach (var g in bySlab)
        {
            var spans = g.OrderBy(b => b.MemoryOffset).ToList();
            for (int i = 1; i < spans.Count; i++)
                Assert.True(spans[i - 1].MemoryOffset + spans[i - 1].Size <= spans[i].MemoryOffset, "sub-buffers overlap");
        }

        // Write a different pattern into every buffer, then read all back: an offset bug shows as another buffer's pattern.
        for (int i = 0; i < buffers.Count; i++)
        {
            var data = new float[Math.Min(buffers[i].Size, 1 << 20) / sizeof(float)];
            for (int j = 0; j < data.Length; j++) data[j] = i * 1000 + j % 997;
            device.Upload(data, buffers[i]);
        }
        for (int i = 0; i < buffers.Count; i++)
        {
            var back = new float[Math.Min(buffers[i].Size, 1 << 20) / sizeof(float)];
            device.Download(buffers[i], back);
            for (int j = 0; j < back.Length; j++)
                Assert.True(back[j] == i * 1000 + j % 997, $"buffer {i} element {j}: {back[j]}");
        }

        // The arena is gone, but the slabs stay until their last tensor is disposed.
        Assert.Equal(baseAllocs + 3, device.LiveAllocationCount);
        foreach (var b in buffers) b.Dispose();
        Assert.Equal(baseAllocs, device.LiveAllocationCount);
    }
}
