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

    /// <summary>
    /// Diagnostic: VkResult code reported by the most recent failed
    /// <see cref="TryCreate"/> call. Zero when the last call succeeded.
    /// </summary>
    public static int LastImportFailureCode { get; private set; }

    /// <summary>
    /// Diagnostic: name of the Vulkan call that returned the most recent
    /// failure. Empty string when the last call succeeded. Possible values:
    /// "vkGetMemoryHostPointerPropertiesEXT", "vkCreateBuffer",
    /// "vkAllocateMemory", "vkBindBufferMemory", "memory_type_intersection".
    /// </summary>
    public static string LastImportFailureStage { get; private set; } = string.Empty;

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
    /// <remarks>
    /// <para>
    /// <b>Handle type fallback.</b> The implementation first tries
    /// <c>VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT</c> (process-
    /// allocated memory, e.g. <see cref="System.Runtime.InteropServices.NativeMemory.AlignedAlloc"/>),
    /// then falls back to <c>HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT</c> (memory
    /// mapped from a non-Vulkan source, e.g. <c>MemoryMappedFile</c>). amdvlk
    /// on Strix Halo (gfx1151) accepts heap-allocated pages via
    /// HOST_ALLOCATION but rejects read-only mmap'd file views — the foreign-
    /// memory handle type is the correct path for GGUF mmaps.
    /// </para>
    /// </remarks>
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

        nint alignedHostPtr = unchecked((nint)alignedAddr);

        // Candidate handle types, in order of preference for typical workloads.
        // HOST_ALLOCATION covers heap-allocated process memory (NativeMemory.AlignedAlloc,
        // malloc) and works on virtually every driver. HOST_MAPPED_FOREIGN_MEMORY is the
        // correct bit for memory NOT allocated by the process — read-only mmap'd file
        // views via MemoryMappedFile in particular — and is what amdvlk on gfx1151 requires
        // for GGUF imports. We don't pick one upfront because vkGetMemoryHostPointerPropertiesEXT
        // returning success doesn't guarantee vkAllocateMemory will accept the pointer with
        // that handle type — driver bugs occur. Instead we collect all candidates that
        // pass the query and try each in sequence inside the buffer/memory construction.
        ReadOnlySpan<uint> handleTypeCandidates = stackalloc uint[]
        {
            VkExternalMemoryHandleTypeFlags.HostMappedForeignMemoryBitExt,
            VkExternalMemoryHandleTypeFlags.HostAllocationBitExt,
        };

        Span<uint> usable = stackalloc uint[2];
        Span<uint> usableTypeBits = stackalloc uint[2];
        int usableCount = 0;
        foreach (uint candidate in handleTypeCandidates)
        {
            VkMemoryHostPointerPropertiesExt q = default;
            q.sType = VkStructureType.MemoryHostPointerPropertiesExt;
            int rq = getHostPtrProps(device.Handle, candidate, alignedHostPtr, ref q);
            if (rq >= 0 && q.memoryTypeBits != 0)
            {
                usable[usableCount] = candidate;
                usableTypeBits[usableCount] = q.memoryTypeBits;
                usableCount++;
            }
        }

        if (usableCount == 0)
        {
            LastImportFailureCode = 0;
            LastImportFailureStage = "vkGetMemoryHostPointerPropertiesEXT";
            return null;
        }

        // Try each viable handle type in order — vkGetMemoryHostPointerPropertiesEXT
        // returning success doesn't guarantee vkAllocateMemory will accept the
        // import (driver bugs / handle-type semantics mismatches occur). amdvlk
        // on gfx1151 in particular returns VK_ERROR_INVALID_EXTERNAL_HANDLE for
        // HOST_ALLOCATION on read-only MemoryMappedFile views but accepts
        // HOST_MAPPED_FOREIGN_MEMORY for the same pointer.
        for (int attempt = 0; attempt < usableCount; attempt++)
        {
            uint handleType = usable[attempt];
            uint typeBitsForHandle = usableTypeBits[attempt];

            // 3) Create the buffer with the external-memory create-info chained
            //    in pNext so the driver knows it will be bound to imported memory.
            VkExternalMemoryBufferCreateInfo extBci = default;
            extBci.sType = VkStructureType.ExternalMemoryBufferCreateInfo;
            extBci.handleTypes = handleType;

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
            if (br < 0)
            {
                LastImportFailureCode = br;
                LastImportFailureStage = "vkCreateBuffer";
                continue; // try next handle type
            }

            bool ownBuffer = true;
            nint memory = 0;
            try
            {
                // 4) Query buffer memory requirements and intersect with the
                //    pointer-importable type bits to pick a compatible memory type.
                VulkanApi.vkGetBufferMemoryRequirements(device.Handle, buffer, out var req);

                uint typeBits = req.memoryTypeBits & typeBitsForHandle;
                if (typeBits == 0)
                {
                    LastImportFailureCode = 0;
                    LastImportFailureStage = "memory_type_intersection";
                    continue;
                }

                if (!device.TryFindHostImportMemoryType(typeBits, out uint typeIndex))
                {
                    LastImportFailureCode = 0;
                    LastImportFailureStage = "memory_type_intersection";
                    continue;
                }

                // 5) Allocate device memory backed by the imported host pointer.
                VkImportMemoryHostPointerInfoExt importInfo = default;
                importInfo.sType = VkStructureType.ImportMemoryHostPointerInfoExt;
                importInfo.handleType = handleType;
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
                    LastImportFailureCode = ar;
                    LastImportFailureStage = "vkAllocateMemory";
                    memory = 0;
                    continue;
                }

                // 6) Bind the buffer to the imported memory at offset=bindOffset.
                //    Kernels see byte 0 of the descriptor = byte 0 of the caller's
                //    logical range.
                int bindRes = VulkanApi.vkBindBufferMemory(
                    device.Handle, buffer, memory, (ulong)bindOffset);
                if (bindRes < 0)
                {
                    LastImportFailureCode = bindRes;
                    LastImportFailureStage = "vkBindBufferMemory";
                    continue;
                }

                var result = new HostVisibleBuffer(
                    device, buffer, memory,
                    size: size, bindOffset: bindOffset,
                    importedHostPointer: alignedHostPtr,
                    importedSize: (long)alignedSize);

                // Transfer ownership: prevent the finally-block from destroying.
                ownBuffer = false;
                memory = 0;
                LastImportFailureCode = 0;
                LastImportFailureStage = string.Empty;
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

        // All candidate handle types rejected — LastImportFailureCode/Stage
        // reflect the most recent attempt.
        return null;
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
