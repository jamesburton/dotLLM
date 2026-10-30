using Xunit;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Shared skip gate for tests that need a host-pointer import to succeed. The #507 UMA gate
/// refuses the import on a discrete GPU by design (it would leave weights in system RAM behind
/// PCIe), so those tests have nothing to assert there and must skip rather than fail.
/// </summary>
internal static class VulkanHostImportGate
{
    /// <summary>True on an integrated GPU or CPU/software device.</summary>
    public static bool PermitsImport(VulkanDevice device)
        => device.PhysicalDeviceTypeValue is VkPhysicalDeviceType.IntegratedGpu
            or VkPhysicalDeviceType.Cpu;

    /// <summary>Skips the calling test when the device's type makes the import refusal expected.</summary>
    public static void SkipUnlessImportPermitted(VulkanDevice device)
        => Skip.IfNot(PermitsImport(device),
            "Host-pointer import is refused on discrete GPUs by design (#507 not_integrated_gpu).");
}
