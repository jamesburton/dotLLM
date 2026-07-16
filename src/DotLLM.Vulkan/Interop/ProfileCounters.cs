namespace DotLLM.Vulkan.Interop;

/// <summary>
/// Global census counters for the env-gated decode overhead profiler
/// (issue #143, <c>DOTLLM_VULKAN_DECODE_PROFILE=1</c>). Incremented
/// unconditionally on the recording path — a non-atomic increment of a
/// static long is sub-nanosecond next to the vkCmd* call it annotates.
/// Read as before/after snapshots by the profiler; never reset.
/// </summary>
internal static class ProfileCounters
{
    /// <summary>Total <c>vkCmdDispatch</c> calls recorded.</summary>
    internal static long Dispatches;

    /// <summary>Total <c>vkCmdCopyBuffer</c> calls recorded.</summary>
    internal static long Copies;

    /// <summary>Total <c>vkCmdPipelineBarrier</c> calls recorded (via KernelSupport).</summary>
    internal static long Barriers;

    /// <summary>Total <c>vkUpdateDescriptorSets</c> calls (via KernelSupport.WriteBufferBindings).</summary>
    internal static long DescriptorWrites;

    /// <summary>Total descriptor-set allocations (via KernelSupport.AllocateDescriptorSet).</summary>
    internal static long DescriptorAllocs;
}
