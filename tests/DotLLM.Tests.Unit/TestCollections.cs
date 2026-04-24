using Xunit;

namespace DotLLM.Tests.Unit;

/// <summary>
/// Marker for Vulkan kernel tests that may toggle process-global environment
/// variables (e.g. <c>DOTLLM_VULKAN_FORCE_SHARED_REDUCE</c>) to exercise both
/// the subgroup-arithmetic and shared-memory reduction paths. Apply
/// <c>[Collection("VulkanKernels")]</c> to every Vulkan test class to force
/// sequential execution and prevent env-var races between classes.
/// </summary>
[CollectionDefinition("VulkanKernels", DisableParallelization = true)]
public class VulkanKernelsCollection
{
}
