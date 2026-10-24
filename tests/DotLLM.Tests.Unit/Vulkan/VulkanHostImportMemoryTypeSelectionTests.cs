using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #507 — the <c>VK_EXT_external_memory_host</c> zero-copy weight import
/// must only engage on a device whose memory is physically the host's, i.e. an
/// integrated GPU (or a software/CPU device). On a discrete GPU an accepted
/// import leaves every weight in system RAM, read across PCIe for the model's
/// lifetime.
/// </summary>
/// <remarks>
/// <para>
/// These drive <see cref="VulkanDevice.TrySelectHostImportMemoryType"/> — the
/// pure decision the shipping <c>TryFindHostImportMemoryType</c> wrapper calls
/// after reading <c>vkGetPhysicalDeviceProperties</c> and
/// <c>vkGetPhysicalDeviceMemoryProperties</c> — with synthetic memory-type
/// tables. No device is created, so the discrete-GPU cases are covered on a
/// UMA-only host (this hardware), which is exactly the blind spot that let the
/// ungated import ship.
/// </para>
/// <para>
/// <b><see cref="StrixHalo_MeasuredTable_ImportsOnIntegratedGpu"/> is the
/// discriminating case for the gate's shape.</b> The issue proposed requiring
/// the chosen memory type to be <c>DEVICE_LOCAL</c> as well as
/// <c>HOST_VISIBLE</c>; that table is the one measured on gfx1151/amdvlk, where
/// every importable type is a non-device-local GTT type, so the
/// <c>DEVICE_LOCAL</c> rule would have refused the import on the very platform
/// the feature exists for.
/// </para>
/// </remarks>
public class VulkanHostImportMemoryTypeSelectionTests
{
    private const VkMemoryPropertyFlags DeviceLocal = VkMemoryPropertyFlags.DeviceLocal;
    private const VkMemoryPropertyFlags HostVisible = VkMemoryPropertyFlags.HostVisible;
    private const VkMemoryPropertyFlags HostCoherent = VkMemoryPropertyFlags.HostCoherent;
    private const VkMemoryPropertyFlags HostCached = VkMemoryPropertyFlags.HostCached;

    /// <summary>(a) UMA: the import engages, and picks the exact type the mask allows —
    /// not merely "something", which an off-by-one would also satisfy.</summary>
    [Fact]
    public void IntegratedGpu_SelectsExactMaskedHostVisibleType()
    {
        ReadOnlySpan<VkMemoryPropertyFlags> types =
        [
            /* 0 */ DeviceLocal,
            /* 1 */ DeviceLocal | HostVisible | HostCoherent,
            /* 2 */ HostVisible | HostCoherent | HostCached,
        ];

        // Mask excludes types 0 and 1, so the only legal answer is 2.
        Assert.True(VulkanDevice.TrySelectHostImportMemoryType(
            1u << 2, types, VkPhysicalDeviceType.IntegratedGpu, out uint idx));
        Assert.Equal(2u, idx);
    }

    /// <summary>(a') UMA, several eligible types in the mask: lowest index wins.</summary>
    [Fact]
    public void IntegratedGpu_MultipleEligibleTypes_SelectsLowestIndex()
    {
        ReadOnlySpan<VkMemoryPropertyFlags> types =
        [
            /* 0 */ DeviceLocal,
            /* 1 */ DeviceLocal | HostVisible | HostCoherent,
            /* 2 */ HostVisible | HostCoherent | HostCached,
        ];

        Assert.True(VulkanDevice.TrySelectHostImportMemoryType(
            0b110u, types, VkPhysicalDeviceType.IntegratedGpu, out uint idx));
        Assert.Equal(1u, idx);
    }

    /// <summary>(b) Discrete GPU: the host-visible types are system RAM behind PCIe,
    /// so the import must be refused and the caller must stage. The mask also offers a
    /// DEVICE_LOCAL-only type, which the removed "any type" fallback would have
    /// wrongly accepted.</summary>
    [Fact]
    public void DiscreteGpu_HostVisibleTypesAreSystemRam_RefusesImport()
    {
        ReadOnlySpan<VkMemoryPropertyFlags> types =
        [
            /* 0 */ DeviceLocal,                             // VRAM, not mappable
            /* 1 */ HostVisible | HostCoherent,              // system RAM (WC)
            /* 2 */ HostVisible | HostCoherent | HostCached, // system RAM (cached)
        ];

        Assert.False(VulkanDevice.TrySelectHostImportMemoryType(
            0b111u, types, VkPhysicalDeviceType.DiscreteGpu, out uint idx));
        Assert.Equal(0u, idx);
    }

    /// <summary>(c) Discrete GPU that also exposes a small ReBAR-style
    /// DEVICE_LOCAL|HOST_VISIBLE window: still refused. The import would land in
    /// system RAM regardless of what else the device happens to expose — and on a
    /// dGPU the ReBAR type is not in the imported pointer's type bits anyway.</summary>
    [Fact]
    public void DiscreteGpu_WithRebarWindow_StillRefusesImport()
    {
        ReadOnlySpan<VkMemoryPropertyFlags> types =
        [
            /* 0 */ DeviceLocal,
            /* 1 */ HostVisible | HostCoherent,
            /* 2 */ HostVisible | HostCoherent | HostCached,
            /* 3 */ DeviceLocal | HostVisible | HostCoherent, // 256 MiB ReBAR window
        ];

        // Mask as the driver would report it for an imported host pointer: the
        // host-memory types only, bit 3 clear.
        Assert.False(VulkanDevice.TrySelectHostImportMemoryType(
            0b0110u, types, VkPhysicalDeviceType.DiscreteGpu, out _));

        // Even with the ReBAR type in the mask the device type still refuses —
        // the gate is locality, not the presence of a mappable device-local type.
        Assert.False(VulkanDevice.TrySelectHostImportMemoryType(
            0b1110u, types, VkPhysicalDeviceType.DiscreteGpu, out _));

        // Control: the identical table on an integrated part imports.
        Assert.True(VulkanDevice.TrySelectHostImportMemoryType(
            0b0110u, types, VkPhysicalDeviceType.IntegratedGpu, out uint idx));
        Assert.Equal(1u, idx);
    }

    /// <summary>
    /// (d) The memory-type table and import mask measured on this project's own
    /// UMA target — AMD Strix Halo / gfx1151, amdvlk on Windows, 2026-09-23. Heap 0
    /// is GTT (host-visible, NOT flagged device-local), heap 1 is the device-local
    /// VRAM carve-out; <c>vkGetMemoryHostPointerPropertiesEXT</c> reports
    /// <c>0x2222</c> for HOST_MAPPED_FOREIGN_MEMORY and <c>0xAAAA</c> for
    /// HOST_ALLOCATION — every importable type is on heap 0.
    /// </summary>
    /// <remarks>
    /// This is the case that refutes the proposed DEVICE_LOCAL|HOST_VISIBLE rule:
    /// types 2/10 carry both flags but are absent from both masks, so that rule
    /// would disable zero-copy on the platform the import was built for.
    /// </remarks>
    [Theory]
    [InlineData(0x2222u, 1u)] // HOST_MAPPED_FOREIGN_MEMORY (mmap'd GGUF views)
    [InlineData(0xAAAAu, 1u)] // HOST_ALLOCATION (NativeMemory.AlignedAlloc pages)
    public void StrixHalo_MeasuredTable_ImportsOnIntegratedGpu(uint typeBits, uint expectedIndex)
    {
        ReadOnlySpan<VkMemoryPropertyFlags> types = StrixHaloMemoryTypes;

        Assert.True(VulkanDevice.TrySelectHostImportMemoryType(
            typeBits, types, VkPhysicalDeviceType.IntegratedGpu, out uint idx));
        Assert.Equal(expectedIndex, idx);
    }

    /// <summary>The same measured table on a discrete part is refused.</summary>
    [Fact]
    public void StrixHalo_MeasuredTable_RefusedWhenDeviceTypeIsDiscrete()
    {
        ReadOnlySpan<VkMemoryPropertyFlags> types = StrixHaloMemoryTypes;

        Assert.False(VulkanDevice.TrySelectHostImportMemoryType(
            0x2222u, types, VkPhysicalDeviceType.DiscreteGpu, out _));
    }

    /// <summary>An empty candidate mask is refused rather than defaulting to type 0.</summary>
    [Fact]
    public void EmptyTypeBits_RefusesImport()
    {
        ReadOnlySpan<VkMemoryPropertyFlags> types = [DeviceLocal | HostVisible];
        Assert.False(VulkanDevice.TrySelectHostImportMemoryType(
            0u, types, VkPhysicalDeviceType.IntegratedGpu, out _));
    }

    /// <summary>A mask that only selects non-host-visible types is refused — there is
    /// no "any type" fallback.</summary>
    [Fact]
    public void IntegratedGpu_MaskSelectsOnlyNonHostVisibleTypes_RefusesImport()
    {
        ReadOnlySpan<VkMemoryPropertyFlags> types =
        [
            /* 0 */ DeviceLocal,
            /* 1 */ DeviceLocal,
            /* 2 */ HostVisible | HostCoherent,
        ];

        Assert.False(VulkanDevice.TrySelectHostImportMemoryType(
            0b011u, types, VkPhysicalDeviceType.IntegratedGpu, out _));
    }

    /// <summary>gfx1151 / amdvlk memory types, verbatim (16 types over 2 heaps; the
    /// upper 8 mirror the lower 8). Flags 193/198/199/206 include vendor bits above
    /// the four the interop enum names; only HOST_VISIBLE (0x2) and DEVICE_LOCAL
    /// (0x1) matter here.</summary>
    private static ReadOnlySpan<VkMemoryPropertyFlags> StrixHaloMemoryTypes =>
    [
        /*  0 heap1 */ DeviceLocal,
        /*  1 heap0 */ HostVisible | HostCoherent,
        /*  2 heap1 */ DeviceLocal | HostVisible | HostCoherent,
        /*  3 heap0 */ HostVisible | HostCoherent | HostCached,
        /*  4 heap1 */ (VkMemoryPropertyFlags)193,
        /*  5 heap0 */ (VkMemoryPropertyFlags)198,
        /*  6 heap1 */ (VkMemoryPropertyFlags)199,
        /*  7 heap0 */ (VkMemoryPropertyFlags)206,
        /*  8 heap1 */ DeviceLocal,
        /*  9 heap0 */ HostVisible | HostCoherent,
        /* 10 heap1 */ DeviceLocal | HostVisible | HostCoherent,
        /* 11 heap0 */ HostVisible | HostCoherent | HostCached,
        /* 12 heap1 */ (VkMemoryPropertyFlags)193,
        /* 13 heap0 */ (VkMemoryPropertyFlags)198,
        /* 14 heap1 */ (VkMemoryPropertyFlags)199,
        /* 15 heap0 */ (VkMemoryPropertyFlags)206,
    ];
}
