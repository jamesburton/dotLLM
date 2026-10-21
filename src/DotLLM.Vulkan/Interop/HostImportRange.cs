namespace DotLLM.Vulkan.Interop;

/// <summary>
/// The host range a <c>VK_EXT_external_memory_host</c> import covers for a logical range <c>[address, address + size)</c>
/// (the arithmetic of <see cref="HostVisibleBuffer.TryCreate"/>, pure so it is testable without a device, #1083).
/// The Vulkan spec requires <c>VkImportMemoryHostPointerInfoEXT.pHostPointer</c> and
/// <c>VkMemoryAllocateInfo.allocationSize</c> to be multiples of
/// <c>VkPhysicalDeviceExternalMemoryHostPropertiesEXT.minImportedHostPointerAlignment</c> (a power of two), so the import starts at
/// the address rounded <b>down</b> to it, ends at the logical end rounded <b>up</b> to it, and the buffer is bound at the logical
/// start's offset from the import base.
/// </summary>
/// <param name="Base">The import's host pointer: the logical start rounded down to the alignment.</param>
/// <param name="BindOffset">Bytes from <see cref="Base"/> to the logical start (less than the alignment).</param>
/// <param name="Size">The import's size: <see cref="BindOffset"/> + the logical size, rounded up to the alignment.</param>
internal readonly record struct HostImportRange(ulong Base, long BindOffset, ulong Size)
{
    /// <summary>
    /// Computes the import range of <paramref name="size"/> bytes at <paramref name="address"/>; false when
    /// <paramref name="alignment"/> is not a power of two (the masks would be wrong; no conforming driver reports one) or the
    /// rounded range overflows.
    /// </summary>
    /// <param name="address">The logical start.</param>
    /// <param name="size">The logical size (positive).</param>
    /// <param name="alignment">The device's <c>minImportedHostPointerAlignment</c>.</param>
    /// <param name="range">Receives the range.</param>
    /// <returns>True when the range is valid.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="size"/> is not positive.</exception>
    public static bool TryCompute(ulong address, long size, ulong alignment, out HostImportRange range)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(size);
        range = default;
        if (!IsPowerOfTwo(alignment)) return false;
        ulong baseAddress = address & ~(alignment - 1);
        ulong bindOffset = address - baseAddress;
        ulong end = bindOffset + (ulong)size;
        if (end < bindOffset || end > (ulong)long.MaxValue - alignment) return false;   // overflow: pathological input
        if (baseAddress > ulong.MaxValue - RoundUp(end, alignment)) return false;       // the range wraps the address space
        range = new HostImportRange(baseAddress, (long)bindOffset, RoundUp(end, alignment));
        return true;
    }

    /// <summary>
    /// The <c>allocationSize</c> of the import: the range's size, or the buffer's required size when that is larger, rounded up to
    /// the alignment (the spec's multiple-of-alignment rule).
    /// </summary>
    /// <param name="requiredSize"><c>VkMemoryRequirements.size</c> of the buffer bound to the import.</param>
    /// <param name="alignment">The device's <c>minImportedHostPointerAlignment</c>.</param>
    /// <returns>Bytes.</returns>
    public ulong AllocationSize(ulong requiredSize, ulong alignment) =>
        requiredSize > Size ? RoundUp(requiredSize, alignment) : Size;

    /// <summary><paramref name="bytes"/> rounded up to a multiple of the power of two <paramref name="alignment"/>.</summary>
    /// <param name="bytes">Bytes.</param>
    /// <param name="alignment">A power of two.</param>
    /// <returns>The rounded size.</returns>
    public static ulong RoundUp(ulong bytes, ulong alignment) => (bytes + (alignment - 1)) & ~(alignment - 1);

    /// <summary>True when <paramref name="value"/> is a non-zero power of two.</summary>
    /// <param name="value">The value.</param>
    /// <returns>True for a power of two.</returns>
    public static bool IsPowerOfTwo(ulong value) => value != 0 && (value & (value - 1)) == 0;
}
