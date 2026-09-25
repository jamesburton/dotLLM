using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Centralises the targeted wave32 decision for the decode MMVQ pipelines
/// (issue #54 / #330). On AMD RDNA3.5 (gfx1151) the Vulkan driver defaults
/// compute to wave64; llama.cpp forces wave32 PER-KERNEL for K-quant decode
/// via <c>VkPipelineShaderStageRequiredSubgroupSizeCreateInfo</c> on just those
/// pipelines — never globally. This helper returns the required subgroup size
/// to chain into the MMVQ pipeline create info, or <c>0</c> (driver default /
/// unset) when the device does not support pinning a compute subgroup size or
/// when the env opt-out is set.
/// </summary>
/// <remarks>
/// Applied ONLY to <see cref="MatMulQ4KMmvqKernel"/> and
/// <see cref="MatMulQ8_0MmvqKernel"/>. Numerically a no-op — wave width only
/// changes scheduling, not results — so parity tests stay green whether the
/// pin is active or falls back to the default path.
/// </remarks>
internal static class Wave32SubgroupControl
{
    /// <summary>The wave width pinned on the decode MMVQ pipelines.</summary>
    internal const uint Wave32 = 32;

    /// <summary>
    /// Env-var opt-out for the targeted wave32 MMVQ pin (issue #54). Set
    /// <c>DOTLLM_VULKAN_DISABLE_WAVE32=1</c> to force the default (unset)
    /// subgroup-size path on the decode MMVQ pipelines — e.g. to A/B benchmark
    /// wave32 vs the driver default, or to work around a driver bug. Mirrors the
    /// <c>DOTLLM_VULKAN_DISABLE_MMVQ</c> convention. When unset, wave32 is pinned
    /// whenever the device advertises required-subgroup-size support for the
    /// compute stage.
    /// </summary>
    internal const string DisableWave32EnvVar = "DOTLLM_VULKAN_DISABLE_WAVE32";

    internal static bool IsWave32Disabled() =>
        System.Environment.GetEnvironmentVariable(DisableWave32EnvVar) == "1";

    /// <summary>
    /// Returns <see cref="Wave32"/> (32) when the decode MMVQ pipelines should be
    /// pinned to wave32 on <paramref name="device"/>, else <c>0</c> (leave the
    /// driver's per-pipeline default in place). Pins only when the device
    /// supports a required compute subgroup size of 32 AND the env opt-out is
    /// not set.
    /// </summary>
    internal static uint RequiredSubgroupSizeFor(VulkanDevice device)
    {
        if (IsWave32Disabled())
            return 0;
        return device.SupportsRequiredSubgroupSize(Wave32, VkShaderStageFlags.Compute)
            ? Wave32
            : 0u;
    }
}
