using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>One Vulkan memory type: property flags and owning heap.</summary>
internal readonly record struct VkMemTypeInfo(VkMemoryPropertyFlags Flags, uint Heap);

/// <summary>
/// Pure placement policy for the device-local OOM fallback (issue #810). Kept free of Vulkan
/// calls so the ordering is unit-testable without a GPU.
/// </summary>
internal static class VulkanMemoryPlacement
{
    /// <summary>
    /// Candidate memory types to retry a device-local allocation that failed on
    /// <paramref name="failedType"/>, best first.
    /// </summary>
    /// <remarks>
    /// Order: (1) types on OTHER heaps, because a heap that rejected the allocation will
    /// reject every other type on it for the same reason (Strix Halo, 512 MB BIOS split:
    /// heap 1 DEVICE_LOCAL caps at 68 GiB while heap 0 GTT accepts ~100 GiB); within them
    /// DEVICE_LOCAL|HOST_VISIBLE before HOST_VISIBLE|HOST_COHERENT, larger heaps first,
    /// non-HOST_CACHED before cached (weights are GPU-read, never CPU-read). (2) Same-heap
    /// types last, as a safety net for a type-specific failure.
    /// </remarks>
    internal static List<uint> RankDeviceLocalFallback(
        ReadOnlySpan<VkMemTypeInfo> types, ReadOnlySpan<ulong> heapSizes, uint typeBits, uint failedType)
    {
        uint failedHeap = failedType < types.Length ? types[(int)failedType].Heap : uint.MaxValue;
        var ordered = new List<uint>(8);
        Span<VkMemoryPropertyFlags> rungs =
        [
            VkMemoryPropertyFlags.DeviceLocal | VkMemoryPropertyFlags.HostVisible,
            VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent,
        ];
        for (int pass = 0; pass < 2; pass++)
        {
            foreach (var required in rungs)
            {
                var list = new List<(uint Index, ulong HeapSize, bool Cached)>(4);
                for (int i = 0; i < types.Length; i++)
                {
                    if ((typeBits & (1u << i)) == 0 || i == failedType) continue;
                    var t = types[i];
                    if ((t.Flags & required) != required) continue;
                    if ((t.Heap != failedHeap) != (pass == 0)) continue;
                    ulong size = t.Heap < heapSizes.Length ? heapSizes[(int)t.Heap] : 0;
                    list.Add(((uint)i, size, (t.Flags & VkMemoryPropertyFlags.HostCached) != 0));
                }
                list.Sort(static (a, b) =>
                {
                    int c = b.HeapSize.CompareTo(a.HeapSize);
                    if (c != 0) return c;
                    c = a.Cached.CompareTo(b.Cached);
                    return c != 0 ? c : a.Index.CompareTo(b.Index);
                });
                foreach (var e in list)
                    if (!ordered.Contains(e.Index)) ordered.Add(e.Index);
            }
        }
        return ordered;
    }
}
