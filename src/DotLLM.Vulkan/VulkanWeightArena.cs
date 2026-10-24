using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Packs a model's small and medium weight tensors into shared 64 MiB device allocations
/// ("slabs"), each tensor keeping its own <c>VkBuffer</c> bound at an offset (#1174).
/// </summary>
/// <remarks>
/// <para>
/// One <c>vkAllocateMemory</c> per tensor is not free. On gfx1151 / amdvlk (Strix Halo) the driver
/// rounds every allocation between about 4 and 64 MiB up to a 2 MiB page: Nemotron 3.5 Bolt's 349
/// such tensors (4,044 MiB) cost 293 MiB of extra process GPU memory, measured with the exact
/// allocation sizes and nothing else. llama.cpp places tensors inside a few large buffers and pays
/// none of it. A slab is a multiple of 2 MiB, so the rounding disappears, and the allocation count
/// drops by an order of magnitude (<c>maxMemoryAllocationCount</c> is finite).
/// </para>
/// <para>
/// Kernels see no difference: a sub-buffer is an ordinary <c>VkBuffer</c>, bound with offset 0 /
/// <c>VK_WHOLE_SIZE</c> in its descriptor. It reports <see cref="VulkanDevice.Buffer.IsHostVisible"/>
/// false, so host transfers stage instead of mapping memory another tensor shares.
/// </para>
/// <para>
/// Tensors of <see cref="DirectThresholdBytes"/> or more keep a dedicated allocation (the driver
/// does not round those, and packing them would only waste slab tails). Placement is first-fit
/// over every slab opened so far. Each slab frees its memory when the arena and every tensor in
/// it have been disposed. <c>DOTLLM_VULKAN_WEIGHT_ARENA=0</c> restores one allocation per tensor.
/// Single-threaded: one loader owns an arena for the duration of its upload.
/// </para>
/// </remarks>
internal sealed class VulkanWeightArena : IDisposable
{
    /// <summary>Slab size: 64 MiB, a multiple of the driver's 2 MiB rounding.</summary>
    internal const long SlabBytes = 64L << 20;

    /// <summary>Tensors at least this large get their own allocation.</summary>
    internal const long DirectThresholdBytes = SlabBytes;

    private static readonly bool s_disabled =
        Environment.GetEnvironmentVariable("DOTLLM_VULKAN_WEIGHT_ARENA") == "0";

    private readonly VulkanDevice _device;
    private readonly List<(VulkanDevice.Buffer Slab, long Used)> _slabs = [];
    private bool _disposed;

    /// <summary>Creates an arena on <paramref name="device"/>.</summary>
    /// <param name="device">Device whose memory the slabs come from.</param>
    public VulkanWeightArena(VulkanDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
    }

    /// <summary>Number of slabs opened so far (test hook).</summary>
    internal int SlabCount => _slabs.Count;

    /// <summary>
    /// A device-local buffer of <paramref name="bytes"/> bytes: carved from a slab when small enough,
    /// otherwise a dedicated <see cref="VulkanDevice.AllocateDeviceLocal"/> allocation.
    /// </summary>
    /// <param name="bytes">Buffer size in bytes.</param>
    /// <returns>The buffer; the caller disposes it as usual.</returns>
    public VulkanDevice.Buffer AllocateDeviceLocal(long bytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (s_disabled || bytes >= DirectThresholdBytes) return _device.AllocateDeviceLocal(bytes);

        var bci = new VkBufferCreateInfo
        {
            sType = VkStructureType.BufferCreateInfo,
            size = (ulong)bytes,
            usage = VulkanDevice.BufferUsage,
            sharingMode = VkSharingMode.Exclusive,
        };
        VulkanApi.vkCreateBuffer(_device.Handle, bci, 0, out nint buffer).ThrowOnError("vkCreateBuffer (arena)");
        try
        {
            VulkanApi.vkGetBufferMemoryRequirements(_device.Handle, buffer, out var req);
            long align = Math.Max(1L, (long)req.alignment);
            long size = (long)req.size;
            for (int i = 0; i <= _slabs.Count; i++)
            {
                bool fresh = i == _slabs.Count;
                if (fresh)
                {
                    var created = _device.AllocateDeviceLocal(SlabBytes);
                    created.AddSlabRef(); // the arena's own reference, released in Dispose
                    _slabs.Add((created, 0));
                }

                var (slab, used) = _slabs[i];
                bool typeOk = slab.TypeIndex < 32 && (req.memoryTypeBits & (1u << (int)slab.TypeIndex)) != 0;
                long offset = (used + align - 1) / align * align;
                if (typeOk && offset + size <= slab.Size)
                {
                    VulkanApi.vkBindBufferMemory(_device.Handle, buffer, slab.Memory, (ulong)offset).ThrowOnError("vkBindBufferMemory (arena)");
                    _slabs[i] = (slab, offset + size);
                    return new VulkanDevice.Buffer(_device, buffer, slab, offset, bytes);
                }

                if (fresh)
                {
                    // Even an empty slab cannot hold it (incompatible memory type): drop the slab, allocate directly.
                    _slabs.RemoveAt(i);
                    slab.ReleaseSlabRef();
                    break;
                }
            }
        }
        catch
        {
            VulkanApi.vkDestroyBuffer(_device.Handle, buffer, 0);
            throw;
        }

        // No compatible slab type: fall back to a dedicated allocation.
        VulkanApi.vkDestroyBuffer(_device.Handle, buffer, 0);
        return _device.AllocateDeviceLocal(bytes);
    }

    /// <summary>Releases the arena's slab references; slabs whose tensors are all disposed free their memory now.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var (slab, _) in _slabs) slab.ReleaseSlabRef();
        _slabs.Clear();
    }
}
