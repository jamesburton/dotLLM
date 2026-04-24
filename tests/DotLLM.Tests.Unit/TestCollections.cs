using Xunit;

namespace DotLLM.Tests.Unit;

/// <summary>
/// Marker for xUnit collections that must not run in parallel with each
/// other. Apply <c>[Collection("SequentialFileIO")]</c> to any test class
/// whose tests contend on shared file handles (e.g., the 442 KB Granite
/// <c>merges.txt</c>) when the suite is executed with xUnit's default
/// collection parallelism.
/// </summary>
[CollectionDefinition("SequentialFileIO", DisableParallelization = true)]
public class SequentialFileIOCollection
{
}

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
