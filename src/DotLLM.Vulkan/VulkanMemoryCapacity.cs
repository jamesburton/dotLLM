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
    /// <summary>
    /// Empirical fraction of physical RAM an integrated GPU can really keep resident (#880). Measured on Strix Halo / Windows, 127.1 GiB RAM,
    /// 2026-10-10: the heaps advertise 104.7 GiB (budgets 99.4 GiB) and allocation of that much succeeds, but a process whose GPU memory
    /// reaches ~80.5 GiB (~0.63-0.65 x RAM, varies with page-cache pressure) fails the next <c>vkQueueSubmit</c> with out-of-device-memory, and every later allocation then fails
    /// (sticky). A 79.2 GiB trunk + 1.1 GiB scratch loaded; adding a 113 MiB sequence state did not fit. Override: <c>DOTLLM_VK_UMA_USABLE_FRACTION</c>.
    /// </summary>
    internal static double UmaUsableFraction { get; } =
        double.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VK_UMA_USABLE_FRACTION"), System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out double f) && f > 0.1 && f <= 1.0 ? f : 0.64;

    /// <summary>
    /// What can really be resident: <paramref name="residentCapacity"/> on a discrete GPU; on an integrated GPU additionally capped at
    /// <see cref="UmaUsableFraction"/> of <paramref name="physicalRamBytes"/> (the OS limit the heap sizes do not show).
    /// </summary>
    internal static long UsableCapacityBytes(long residentCapacity, long physicalRamBytes, int deviceType)
        => deviceType is VkPhysicalDeviceType.IntegratedGpu && physicalRamBytes > 0
            ? Math.Min(residentCapacity, (long)(physicalRamBytes * UmaUsableFraction))
            : residentCapacity;

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
