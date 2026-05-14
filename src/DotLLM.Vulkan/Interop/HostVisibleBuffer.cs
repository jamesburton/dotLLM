using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;

namespace DotLLM.Vulkan.Interop;

/// <summary>
/// A <c>VkBuffer</c> + <c>VkDeviceMemory</c> pair whose backing pages are
/// imported directly from a host pointer (e.g. an <c>mmap</c>'d GGUF tensor
/// data section) via <c>VK_EXT_external_memory_host</c>. On a unified-memory
/// APU this is genuinely zero-copy — the GPU compute shader reads the same
/// DRAM pages a CPU <c>Span&lt;byte&gt;</c> would, with no host→device
/// staging step at load time.
/// </summary>
/// <remarks>
/// <para>
/// <b>Layout contract.</b> The imported allocation aliases an entire host
/// page range, not just the tensor's logical bytes. The <see cref="Handle"/>
/// is bound to this allocation with a <see cref="BindOffset"/> equal to the
/// offset of the caller's logical-start pointer from the page-aligned import
/// base. Compute shaders binding this buffer see the logical bytes starting
/// at descriptor offset 0 — the bind offset is hidden inside the allocation
/// and irrelevant to kernels.
/// </para>
/// <para>
/// <b>Lifetime.</b> The mmap'd source pointer must remain valid (and the
/// underlying <c>MemoryMappedFile</c> undisposed) for as long as this
/// instance lives. Disposing this instance destroys the buffer and frees the
/// device memory but does not touch the host mapping — that's the caller's
/// (<c>GgufFile</c>'s) responsibility.
/// </para>
/// <para>
/// <b>Failure semantics.</b> Construction throws nothing for a recoverable
/// failure (alignment mismatch, driver rejection, extension absent) — use
/// <see cref="TryCreate"/> which returns <c>null</c>. Callers must always
/// have a staging-copy fallback path; the zero-copy import is opportunistic.
/// </para>
/// </remarks>
public sealed class HostVisibleBuffer : IDisposable
{
    private readonly VulkanDevice _device;
    private nint _buffer;
    private nint _memory;
    private bool _disposed;

    /// <summary>Underlying <c>VkBuffer</c> handle. Compute shaders bind this directly.</summary>
    public nint Handle => _buffer;

    /// <summary>Underlying <c>VkDeviceMemory</c> handle (imported, not driver-owned).</summary>
    public nint Memory => _memory;

    /// <summary>Logical buffer size in bytes (the caller-requested range).</summary>
    public long Size { get; }

    /// <summary>
    /// Byte offset from the page-aligned import base to the caller's logical
    /// start. Hidden inside the buffer-memory binding — <see cref="Handle"/>
    /// reads logical byte 0 from the shader's perspective. Surfaced here for
    /// diagnostics and tests only.
    /// </summary>
    public long BindOffset { get; }

    /// <summary>The host pointer the allocation was imported from (page-aligned).</summary>
    public nint ImportedHostPointer { get; }

    /// <summary>Total bytes of the page-aligned import range backing this buffer.</summary>
    public long ImportedSize { get; }

    /// <summary>
    /// Memory-placement intent reported back to the caller. Always
    /// <see cref="MemoryDomain.HostVisibleZeroCopy"/> for instances of this type.
    /// </summary>
    public MemoryDomain Domain => MemoryDomain.HostVisibleZeroCopy;

    private HostVisibleBuffer(
        VulkanDevice device, nint buffer, nint memory, long size, long bindOffset,
        nint importedHostPointer, long importedSize)
    {
        _device = device;
        _buffer = buffer;
        _memory = memory;
        Size = size;
        BindOffset = bindOffset;
        ImportedHostPointer = importedHostPointer;
        ImportedSize = importedSize;
    }

    /// <summary>
    /// Attempts to wrap <paramref name="hostPointer"/> + <paramref name="size"/>
    /// in a Vulkan buffer backed by imported host memory. Returns <c>null</c>
    /// when the device does not support <c>VK_EXT_external_memory_host</c>,
    /// when the resulting page-aligned import range cannot be satisfied, or
    /// when the driver rejects the import — callers should then fall back to
    /// the staging-copy path.
    /// </summary>
    /// <param name="device">Vulkan device.</param>
    /// <param name="hostPointer">
    /// Host pointer to the logical buffer's first byte. Does <i>not</i> need
    /// to be page-aligned: the implementation rounds it down to the nearest
    /// <c>minImportedHostPointerAlignment</c> boundary and pre-binds the
    /// buffer at the corresponding offset, so the returned
    /// <see cref="Handle"/> reads bytes starting at <paramref name="hostPointer"/>
    /// from the shader's perspective.
    /// </param>
    /// <param name="size">Logical buffer size in bytes.</param>
    /// <returns>The imported buffer, or <c>null</c> when import is not possible.</returns>
    public static unsafe HostVisibleBuffer? TryCreate(
        VulkanDevice device, nint hostPointer, long size)
    {
        if (device is null) throw new ArgumentNullException(nameof(device));
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size));
        if (hostPointer == 0) throw new ArgumentException("Host pointer must be non-null.", nameof(hostPointer));

        if (!device.HasExternalMemoryHost) return null;

        ulong alignment = device.MinImportedHostPointerAlignment;
        if (alignment == 0) return null;

        // Round the imported pointer DOWN to the alignment boundary. The
        // buffer is bound at `bindOffset` so kernels still see logical byte 0
        // at the start of the descriptor.
        ulong rawAddr = unchecked((ulong)hostPointer);
        ulong alignedAddr = rawAddr & ~(alignment - 1);
        long bindOffset = unchecked((long)(rawAddr - alignedAddr));

        // The import size must also be a multiple of the alignment, and must
        // cover the entire logical range starting at the page-aligned base.
        long requestedEnd = bindOffset + size;
        ulong alignedSize = ((ulong)requestedEnd + (alignment - 1)) & ~(alignment - 1);

        // Sanity: integer overflow / pathological inputs.
        if (alignedSize > (ulong)long.MaxValue) return null;

        // 1) Resolve vkGetMemoryHostPointerPropertiesEXT lazily.
        nint fn = VulkanApi.vkGetDeviceProcAddr(device.Handle, "vkGetMemoryHostPointerPropertiesEXT");
        if (fn == 0) return null;
        var getHostPtrProps = Marshal.GetDelegateForFunctionPointer<
            VulkanDevice.VkGetMemoryHostPointerPropertiesEXT>(fn);

        // 2) Query which memory types can host this pointer.
        VkMemoryHostPointerPropertiesExt hostPtrProps = default;
        hostPtrProps.sType = VkStructureType.MemoryHostPointerPropertiesExt;
        nint alignedHostPtr = unchecked((nint)alignedAddr);
        int r = getHostPtrProps(device.Handle,
            VkExternalMemoryHandleTypeFlags.HostAllocationBitExt,
            alignedHostPtr, ref hostPtrProps);
        if (r < 0 || hostPtrProps.memoryTypeBits == 0) return null;

        // 3) Create the buffer with the external-memory create-info chained
        //    in pNext so the driver knows it will be bound to imported memory.
        VkExternalMemoryBufferCreateInfo extBci = default;
        extBci.sType = VkStructureType.ExternalMemoryBufferCreateInfo;
        extBci.handleTypes = VkExternalMemoryHandleTypeFlags.HostAllocationBitExt;

        var bci = new VkBufferCreateInfo
        {
            sType = VkStructureType.BufferCreateInfo,
            pNext = (nint)(&extBci),
            size = (ulong)alignedSize,
            // Imported host memory cannot be the destination of vkCmdCopyBuffer
            // (write-back to a CPU-coherent mmap'd file is asking for trouble);
            // omit TransferDst. Read-only storage-buffer reads in compute shaders
            // is the supported usage.
            usage = VkBufferUsageFlags.StorageBuffer | VkBufferUsageFlags.TransferSrc,
            sharingMode = VkSharingMode.Exclusive,
        };

        int br = VulkanApi.vkCreateBuffer(device.Handle, bci, 0, out nint buffer);
        if (br < 0) return null;

        bool ownBuffer = true;
        nint memory = 0;
        try
        {
            // 4) Query buffer memory requirements and intersect with the
            //    pointer-importable type bits to pick a compatible memory type.
            VulkanApi.vkGetBufferMemoryRequirements(device.Handle, buffer, out var req);

            uint typeBits = req.memoryTypeBits & hostPtrProps.memoryTypeBits;
            if (typeBits == 0) return null;

            if (!device.TryFindHostImportMemoryType(typeBits, out uint typeIndex))
                return null;

            // 5) Allocate device memory backed by the imported host pointer.
            VkImportMemoryHostPointerInfoExt importInfo = default;
            importInfo.sType = VkStructureType.ImportMemoryHostPointerInfoExt;
            importInfo.handleType = VkExternalMemoryHandleTypeFlags.HostAllocationBitExt;
            importInfo.pHostPointer = alignedHostPtr;

            // Spec requires `allocationSize` to be ≥ the buffer's requirements
            // AND a multiple of `minImportedHostPointerAlignment`. We already
            // page-aligned alignedSize; pick max(alignedSize, req.size) and
            // round up if necessary.
            ulong allocSize = (ulong)alignedSize;
            if (req.size > allocSize)
            {
                allocSize = (req.size + alignment - 1) & ~(alignment - 1);
            }

            var mai = new VkMemoryAllocateInfo
            {
                sType = VkStructureType.MemoryAllocateInfo,
                pNext = (nint)(&importInfo),
                allocationSize = allocSize,
                memoryTypeIndex = typeIndex,
            };
            int ar = VulkanApi.vkAllocateMemory(device.Handle, mai, 0, out memory);
            if (ar < 0)
            {
                memory = 0;
                return null;
            }

            // 6) Bind the buffer to the imported memory at offset=bindOffset.
            //    Kernels see byte 0 of the descriptor = byte 0 of the caller's
            //    logical range.
            int bindRes = VulkanApi.vkBindBufferMemory(
                device.Handle, buffer, memory, (ulong)bindOffset);
            if (bindRes < 0) return null;

            var result = new HostVisibleBuffer(
                device, buffer, memory,
                size: size, bindOffset: bindOffset,
                importedHostPointer: alignedHostPtr,
                importedSize: (long)alignedSize);

            // Transfer ownership: prevent the finally-block from destroying.
            ownBuffer = false;
            memory = 0;
            return result;
        }
        finally
        {
            if (memory != 0)
                VulkanApi.vkFreeMemory(device.Handle, memory, 0);
            if (ownBuffer)
                VulkanApi.vkDestroyBuffer(device.Handle, buffer, 0);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_buffer != 0)
        {
            VulkanApi.vkDestroyBuffer(_device.Handle, _buffer, 0);
            _buffer = 0;
        }
        // The imported VkDeviceMemory must be freed BUT it does NOT release
        // the underlying host mmap — that's the caller's lifecycle. The
        // mmap'd pages live in the GgufFile owning MemoryMappedFile.
        if (_memory != 0)
        {
            VulkanApi.vkFreeMemory(_device.Handle, _memory, 0);
            _memory = 0;
        }
    }
}
