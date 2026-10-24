namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// An expert-grouped MoE projection that reads F32 packed activations (rows sorted by expert) against a packed expert bank and writes F32
/// rows: the shared dispatch contract of <see cref="MoeGroupedMatmulKQuantCoopmatKernel"/> and <see cref="MoeGroupedMatmulIqCoopmatKernel"/>
/// (#1057), so a host can pick either by bank type and drive both through the same grouped-prefill glue.
/// </summary>
internal interface IMoeGroupedF32Kernel
{
    /// <summary>Token rows per workgroup; the indirect tile list must be built with this row tile.</summary>
    int RowTile { get; }

    /// <summary>Weight-row tiles per workgroup column for an output width of <paramref name="m"/>.</summary>
    /// <param name="m">Per-expert output rows.</param>
    /// <returns>The x extent of the grid.</returns>
    int MTiles(int m);

    /// <summary>Records the grouped matmul on the dense (legacy) grid.</summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="bank">Packed expert bank, <c>[numExperts, m, k]</c>.</param>
    /// <param name="packedInput">F32 activations in expert order, <c>[rows, k]</c>.</param>
    /// <param name="offsets">Per-expert row prefix offsets, <c>[numExperts + 1]</c>.</param>
    /// <param name="output">F32 output rows in expert order, <c>[rows, m]</c>.</param>
    /// <param name="m">Per-expert output rows.</param>
    /// <param name="k">Per-expert input width.</param>
    /// <param name="rows">Routed rows in total.</param>
    /// <param name="numExperts">Experts in the bank.</param>
    /// <param name="maxRowsPerExpert">Bound on the row-tile grid; 0 means <paramref name="rows"/>.</param>
    void Record(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer packedInput, VulkanDevice.Buffer offsets,
        VulkanDevice.Buffer output, int m, int k, int rows, int numExperts, int maxRowsPerExpert = 0);

    /// <summary>Records the grouped matmul over the indirect (expert, row-tile) list written by <see cref="MoeBuildTileListKernel"/>.</summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="bank">Packed expert bank, <c>[numExperts, m, k]</c>.</param>
    /// <param name="packedInput">F32 activations in expert order, <c>[rows, k]</c>.</param>
    /// <param name="offsetsAndTiles">Per-expert row offsets followed by the tile list.</param>
    /// <param name="output">F32 output rows in expert order, <c>[rows, m]</c>.</param>
    /// <param name="dispatchArgs">Indirect dispatch triples.</param>
    /// <param name="dispatchArgsOffset">Byte offset of this projection's triple.</param>
    /// <param name="m">Per-expert output rows.</param>
    /// <param name="k">Per-expert input width.</param>
    /// <param name="rows">Routed rows in total.</param>
    /// <param name="numExperts">Experts in the bank.</param>
    void RecordIndirect(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer packedInput, VulkanDevice.Buffer offsetsAndTiles,
        VulkanDevice.Buffer output, VulkanDevice.Buffer dispatchArgs, int dispatchArgsOffset, int m, int k, int rows, int numExperts);
}
