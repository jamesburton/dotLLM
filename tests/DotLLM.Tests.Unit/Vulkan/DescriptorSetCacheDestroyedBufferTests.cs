using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #467: <see cref="DescriptorSetCache"/> keys on raw <c>VkBuffer</c> handles, which drivers
/// recycle. An entry naming a destroyed buffer must be evicted before the handle can come back,
/// or the recycled handle hits a descriptor set still addressing the freed allocation.
/// </summary>
/// <remarks>
/// These assert on the cache's bookkeeping rather than on kernel output: whether a stale set
/// produces wrong numbers depends on where the driver places the recycled allocation, so an
/// output-level check at this scale passes by luck on gfx1151 (the #464 forced-alias probe did).
/// The output-level guard is <see cref="VulkanResidentModelKvChurnTests"/>.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class DescriptorSetCacheDestroyedBufferTests
{
    private const int N = 256;

    [SkippableFact]
    public void DestroyedBuffer_EvictsOnlyTheEntriesThatReferenceIt()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        using var kernel = AddKernel.Create(device, spvDir);
        using var a = device.Allocate(N * sizeof(float));
        using var b = device.Allocate(N * sizeof(float));
        using var kept = device.Allocate(N * sizeof(float));
        var doomed = device.Allocate(N * sizeof(float));

        kernel.Launch(a, b, kept, N);
        kernel.Launch(a, b, doomed, N);
        Assert.Equal(2, kernel.DescriptorCache.Count);

        doomed.Dispose();
        kernel.Launch(a, b, kept, N);

        Assert.Equal(1, kernel.DescriptorCache.Count);
    }

    [SkippableFact]
    public void DestroyingANeverBoundBuffer_DoesNotMoveTheEpoch()
    {
        // Staging buffers (Download on a discrete GPU allocates one per call) are never bound in a
        // descriptor set; logging them would make every cache rescan once per decoded token.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        using var kernel = AddKernel.Create(device, spvDir);
        using var a = device.Allocate(N * sizeof(float));
        using var b = device.Allocate(N * sizeof(float));
        using var c = device.Allocate(N * sizeof(float));
        kernel.Launch(a, b, c, N);

        long before = device.BufferDestroyEpoch;
        device.Allocate(N * sizeof(float)).Dispose();
        Assert.Equal(before, device.BufferDestroyEpoch);

        c.Dispose();
        Assert.Equal(before + 1, device.BufferDestroyEpoch);
    }

    [SkippableFact]
    public void PerRequestBufferChurn_ReusesEvictedSets_InsteadOfExhaustingThePool()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        using var kernel = AddKernel.Create(device, spvDir);
        using var a = device.Allocate(N * sizeof(float));
        using var b = device.Allocate(N * sizeof(float));

        // More churn cycles than the pool holds sets: without reuse of evicted sets this would
        // hit the cache's overflow guard (or vkAllocateDescriptorSets would fail).
        for (int i = 0; i < DescriptorSetCache.Capacity + 16; i++)
        {
            using var c = device.Allocate(N * sizeof(float));
            kernel.Launch(a, b, c, N);
        }

        Assert.Equal(1, kernel.DescriptorCache.Count);
    }
}
