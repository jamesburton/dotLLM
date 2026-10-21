namespace DotLLM.Vulkan;

/// <summary>
/// The routed-expert scratch buffers the nemotron_h_moe expert kernels record into (#1083): implemented by the model's forward state
/// (<see cref="VulkanNemotronHForwardState"/>) and by the iGPU expert service's own scratch
/// (<see cref="VulkanNemotronHMoeExpertService"/>), so both drive the indexed MMVQ / F32 arms and the expert-grouped prefill through
/// the same recording code. Every buffer is sized for the caller's row count; <c>null</c> means the caller has no MoE scratch.
/// </summary>
internal interface INemotronHMoeRoutedScratch
{
    /// <summary>Per routed row, its expert id (int32, <c>[rows]</c>).</summary>
    VulkanDevice.Buffer? MoeTopkIndices { get; }

    /// <summary>Per routed row, its gating weight (<c>[rows]</c>; all ones for the iGPU service, which returns unweighted rows).</summary>
    VulkanDevice.Buffer? MoeTopkWeights { get; }

    /// <summary>F32 expert input per routed row, or the packed (expert-ordered) rows of the coopmat grouped path.</summary>
    VulkanDevice.Buffer? MoeExpandedInput { get; }

    /// <summary>Q8_1 int8 part of the routed-row input (MMVQ and grouped MMQ).</summary>
    VulkanDevice.Buffer? MoeExpandedInputXq { get; }

    /// <summary>Q8_1 scales of the routed-row input.</summary>
    VulkanDevice.Buffer? MoeExpandedInputXds { get; }

    /// <summary>Up projection output (relu² applied in place), <c>[rows, inter]</c>.</summary>
    VulkanDevice.Buffer? MoeUpInter { get; }

    /// <summary>Q8_1 int8 part of <see cref="MoeUpInter"/>.</summary>
    VulkanDevice.Buffer? MoeUpInterXq { get; }

    /// <summary>Q8_1 scales of <see cref="MoeUpInter"/>.</summary>
    VulkanDevice.Buffer? MoeUpInterXds { get; }

    /// <summary>Down projection output per routed (or packed) row, <c>[rows, expertDim]</c>.</summary>
    VulkanDevice.Buffer? MoeDownRows { get; }

    /// <summary>Rows per expert (grouped path).</summary>
    VulkanDevice.Buffer? MoeGroupCounts { get; }

    /// <summary>Per-expert row prefix offsets, then the indirect tile list (grouped path).</summary>
    VulkanDevice.Buffer? MoeGroupOffsets { get; }

    /// <summary>Per-expert scatter cursors (grouped path).</summary>
    VulkanDevice.Buffer? MoeGroupCounters { get; }

    /// <summary>Packed row → routed row (grouped path).</summary>
    VulkanDevice.Buffer? MoeGroupPerm { get; }

    /// <summary>Routed row → packed row (grouped path).</summary>
    VulkanDevice.Buffer? MoeGroupInvPerm { get; }

    /// <summary>Two indirect dispatch triples, up and down (grouped path).</summary>
    VulkanDevice.Buffer? MoeGroupDispatchArgs { get; }

    /// <summary>Q8_1 int8 part of the per-token expert input (grouped MMQ quantises tokens once).</summary>
    VulkanDevice.Buffer? MoeTokenXq { get; }

    /// <summary>Q8_1 scales of the per-token expert input.</summary>
    VulkanDevice.Buffer? MoeTokenXds { get; }
}
