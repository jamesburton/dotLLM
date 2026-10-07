using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Pure policy for "how many bytes of weights can this device hold resident" (issue #812). Kept free of
/// Vulkan calls so it is unit-testable without a GPU.
/// </summary>
internal static class VulkanMemoryCapacity
{
    /// <summary>
    /// Resident-weight capacity: the largest DEVICE_LOCAL heap, plus - on an integrated (UMA) GPU only - every
    /// non-device-local heap too.
    /// </summary>
    /// <remarks>
    /// On the 96 GB BIOS carve-out the device-local heap was the whole story. With a 512 MB carve-out the
    /// driver advertises a 68 GiB DEVICE_LOCAL heap and a 34 GiB GTT heap, both backed by the same system RAM,
    /// and allocations above 68 GiB succeed on the GTT heap (~100 GiB total measured). Gating residency on the
    /// device-local heap alone made a 71 GiB model look like it did not fit, and the model then silently took
    /// the per-layer transient F32 upload path (0.01 tok/s). On a discrete GPU the system-memory heap is host
    /// RAM reached over PCIe, which is not "resident", so it is never counted there.
    /// </remarks>
    internal static long ResidentCapacityBytes(
        ReadOnlySpan<ulong> heapSizes, ReadOnlySpan<VkMemoryHeapFlags> heapFlags, int deviceType)
    {
        long deviceLocalMax = 0, other = 0;
        for (int i = 0; i < heapSizes.Length; i++)
        {
            if ((heapFlags[i] & VkMemoryHeapFlags.DeviceLocal) != 0)
                deviceLocalMax = Math.Max(deviceLocalMax, (long)heapSizes[i]);
            else
                other += (long)heapSizes[i];
        }
        return deviceType is VkPhysicalDeviceType.IntegratedGpu ? deviceLocalMax + other : deviceLocalMax;
    }
}
