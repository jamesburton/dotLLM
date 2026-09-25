namespace DotLLM.Core.Tensors;

/// <summary>
/// Declares the physical/virtual memory placement intent for a buffer the
/// backend is asked to allocate. Backends are free to fall back to a stricter
/// (less zero-copy) domain when the requested domain is not supported on the
/// target driver — callers must treat the returned buffer's actual domain as
/// authoritative.
/// </summary>
/// <remarks>
/// <para>
/// On a unified-memory APU (e.g. AMD Strix Halo, Apple Silicon, Intel
/// integrated) all three domains physically resolve to the same DDR. The
/// distinction matters for the driver: <see cref="DeviceLocal"/> permits a
/// tiled/swizzled layout that reads faster from a compute shader,
/// <see cref="HostVisibleZeroCopy"/> requires a linear layout because a CPU
/// <c>Span&lt;byte&gt;</c> aliases the same pages, and <see cref="Staging"/>
/// is the host-coherent scratch buffer between the two.
/// </para>
/// <para>
/// The <see cref="HostVisibleZeroCopy"/> domain is the recommendation H3 from
/// the GAIA/lemonade-server research note (see
/// <c>.planning/notes/gaia-lemonade-research.md</c> §6 H3): use
/// <c>VK_EXT_external_memory_host</c> to back a <c>VkBuffer</c> directly with
/// the mmap'd GGUF page-aligned address, eliminating the host→device staging
/// copy. llama.cpp does not do this on Vulkan today; this is dotLLM
/// differentiation on unified-memory iGPUs.
/// </para>
/// </remarks>
public enum MemoryDomain
{
    /// <summary>
    /// Driver picks the optimal device-local heap; the buffer is not
    /// host-mappable. Used for KV-cache and (on discrete GPUs) weights.
    /// On a UMA part the driver typically picks a tiled layout in shared
    /// DRAM that reads faster from a compute shader than a linear host
    /// alias would.
    /// </summary>
    DeviceLocal = 0,

    /// <summary>
    /// Buffer backed directly by a host-mmap'd memory range (e.g. the
    /// mmap'd GGUF tensor data section) via the
    /// <c>VK_EXT_external_memory_host</c> extension. The same physical
    /// pages back both a CPU <c>Span&lt;byte&gt;</c> and the
    /// <c>VkBuffer</c>; no staging copy is required at load time. Only
    /// useful on UMA drivers that expose the extension (AMD Strix Halo
    /// via amdvlk/radv, Intel iGPU, Apple Silicon via MoltenVK). The
    /// caller-supplied host pointer must be aligned to
    /// <c>minImportedHostPointerAlignment</c> (typically 4096 on x86-64);
    /// callers can pre-align by rounding the imported pointer down and
    /// offsetting subsequent buffer views into the import.
    /// </summary>
    HostVisibleZeroCopy = 1,

    /// <summary>
    /// Host-visible, host-coherent device memory used as a staging
    /// scratchpad for <c>vkCmdCopyBuffer</c> uploads into a
    /// <see cref="DeviceLocal"/> destination. Allocated once per upload
    /// session and reused across all weight rows.
    /// </summary>
    Staging = 2,
}
