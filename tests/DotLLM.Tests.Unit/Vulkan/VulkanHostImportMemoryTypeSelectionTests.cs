using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #507 — the <c>VK_EXT_external_memory_host</c> zero-copy weight import
/// must only engage when the chosen memory type is <c>DEVICE_LOCAL</c> as well
/// as <c>HOST_VISIBLE</c>.
/// </summary>
/// <remarks>
/// <para>
/// These drive <see cref="VulkanDevice.TrySelectHostImportMemoryType"/> — the
/// pure decision the shipping <c>TryFindHostImportMemoryType</c> wrapper calls
/// after reading <c>vkGetPhysicalDeviceMemoryProperties</c> — with synthetic
/// memory-type tables. No device is created, so the discrete-GPU cases are
/// covered on a UMA-only host (this hardware), which is exactly the blind spot
/// that let the ungated import ship.
/// </para>
/// </remarks>
public class VulkanHostImportMemoryTypeSelectionTests
{
    private const VkMemoryPropertyFlags DeviceLocal = VkMemoryPropertyFlags.DeviceLocal;
    private const VkMemoryPropertyFlags HostVisible = VkMemoryPropertyFlags.HostVisible;
    private const VkMemoryPropertyFlags HostCoherent = VkMemoryPropertyFlags.HostCoherent;
    private const VkMemoryPropertyFlags HostCached = VkMemoryPropertyFlags.HostCached;

    /// <summary>(a) UMA: the importable types are device-local as well as host-visible,
    /// so the import engages — and picks the exact type the mask allows, not merely
    /// "something".</summary>
    [Fact]
    public void Uma_AllTypesDeviceLocalAndHostVisible_SelectsFirstMaskedType()
    {
        ReadOnlySpan<VkMemoryPropertyFlags> types =
        [
            /* 0 */ DeviceLocal,
            /* 1 */ DeviceLocal | HostVisible | HostCoherent,
            /* 2 */ DeviceLocal | HostVisible | HostCoherent | HostCached,
        ];

        // Mask excludes type 0 (not host-visible) and type 1, so the only legal
        // answer is 2 — an off-by-one or a "first host-visible wins" scan fails.
        const uint typeBits = 1u << 2;

        Assert.True(VulkanDevice.TrySelectHostImportMemoryType(typeBits, types, out uint idx));
        Assert.Equal(2u, idx);
    }

    /// <summary>(a') UMA, both importable types legal: lowest index wins.</summary>
    [Fact]
    public void Uma_MultipleEligibleTypes_SelectsLowestIndex()
    {
        ReadOnlySpan<VkMemoryPropertyFlags> types =
        [
            /* 0 */ DeviceLocal,
            /* 1 */ DeviceLocal | HostVisible | HostCoherent,
            /* 2 */ DeviceLocal | HostVisible | HostCoherent | HostCached,
        ];

        Assert.True(VulkanDevice.TrySelectHostImportMemoryType(0b110u, types, out uint idx));
        Assert.Equal(1u, idx);
    }

    /// <summary>(b) Discrete GPU: the host-visible types are system RAM and are NOT
    /// device-local, so the import must be refused and the caller must stage. The
    /// mask also offers a DEVICE_LOCAL-only type, which the removed "any type"
    /// fallback would have wrongly accepted.</summary>
    [Fact]
    public void DiscreteGpu_HostVisibleTypesAreNotDeviceLocal_RefusesImport()
    {
        ReadOnlySpan<VkMemoryPropertyFlags> types =
        [
            /* 0 */ DeviceLocal,                          // VRAM, not mappable
            /* 1 */ HostVisible | HostCoherent,           // system RAM (WC)
            /* 2 */ HostVisible | HostCoherent | HostCached, // system RAM (cached)
        ];

        // Everything the driver could plausibly report for an imported host pointer,
        // plus the device-local-only VRAM type, to exercise the old fallback arm.
        const uint typeBits = 0b111u;

        Assert.False(VulkanDevice.TrySelectHostImportMemoryType(typeBits, types, out uint idx));
        Assert.Equal(0u, idx);
    }

    /// <summary>(c) Discrete GPU with a small ReBAR-style DEVICE_LOCAL|HOST_VISIBLE
    /// window that is NOT in the imported pointer's type bits: still refused. The
    /// existence of an eligible type on the device is not sufficient — it has to be
    /// one the import itself can use.</summary>
    [Fact]
    public void DiscreteGpu_RebarTypeOutsideImportMask_RefusesImport()
    {
        ReadOnlySpan<VkMemoryPropertyFlags> types =
        [
            /* 0 */ DeviceLocal,
            /* 1 */ HostVisible | HostCoherent,
            /* 2 */ HostVisible | HostCoherent | HostCached,
            /* 3 */ DeviceLocal | HostVisible | HostCoherent, // 256 MiB ReBAR window
        ];

        // The pointer-import mask covers only the host-memory types; bit 3 is clear.
        const uint typeBits = 0b0110u;

        Assert.False(VulkanDevice.TrySelectHostImportMemoryType(typeBits, types, out _));

        // Control: the same table DOES import once the ReBAR type is in the mask,
        // proving the refusal above is the mask, not a blanket "never on dGPU".
        Assert.True(VulkanDevice.TrySelectHostImportMemoryType(0b1110u, types, out uint idx));
        Assert.Equal(3u, idx);
    }

    /// <summary>An empty candidate mask is refused rather than defaulting to type 0.</summary>
    [Fact]
    public void EmptyTypeBits_RefusesImport()
    {
        ReadOnlySpan<VkMemoryPropertyFlags> types = [DeviceLocal | HostVisible];
        Assert.False(VulkanDevice.TrySelectHostImportMemoryType(0u, types, out _));
    }
}
