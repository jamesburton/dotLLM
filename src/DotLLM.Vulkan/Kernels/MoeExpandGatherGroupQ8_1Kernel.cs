using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Fused "broadcast + expand-group-by-expert" for Q8_1-quantized activations (#986): gathers pre-quantized Q8_1 rows
/// into expert-contiguous order, replacing the F32 gather + separate quantize pair. Reads the compact [seqLen, K] Q8_1
/// representation and writes the expanded [expandedRows, K] Q8_1 output using topK-based index mapping.
/// </summary>
/// <remarks>
/// <para>
/// The input must be pre-quantized by <see cref="QuantizeQ8_1RowsKernel"/> on the compact [seqLen] token rows BEFORE
/// the expert sort. This kernel then gathers both the packed int8 data and the (d, s) scales per 32-block into
/// expert-sorted order, eliminating the need for a post-gather quantize pass.
/// </para>
/// <para>
/// Produces the same permutation/invPermutation outputs as <see cref="MoeExpandGatherGroupF32Kernel"/> for the
/// weighted scatter to consume.
/// </para>
/// </remarks>
public sealed class MoeExpandGatherGroupQ8_1Kernel : IDisposable
{
    private const int WorkgroupSize = 128;
    private const int BuffersPerSet = 9;
    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeExpandGatherGroupQ8_1Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: BuffersPerSet);
    }

    /// <summary>Whether the SPIR-V is present.</summary>
    public static bool IsSupportedOn(string spvDir) => File.Exists(Path.Combine(spvDir, "moe_expand_gather_group_q8_1.spv"));

    /// <summary>Creates the kernel; throws if the SPIR-V is missing.</summary>
    public static MoeExpandGatherGroupQ8_1Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_expand_gather_group_q8_1.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}.");
        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[BuffersPerSet];
            for (int i = 0; i < BuffersPerSet; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)BuffersPerSet);
        return new MoeExpandGatherGroupQ8_1Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Records the Q8_1 gather into <paramref name="cmdBuf"/>: reads compact [seqLen] Q8_1 rows, writes
    /// expert-sorted [expandedRows] Q8_1 rows via topK broadcast + expert grouping.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="xqIn">Compact packed int8, <c>[seqLen, K/4]</c> uints.</param>
    /// <param name="xdsIn">Compact (d, s) scales, <c>[seqLen, K/32]</c> vec2.</param>
    /// <param name="indices">Expert index per routed row, <c>[expandedRows]</c>.</param>
    /// <param name="offsets">Expert prefix sums, <c>[numExperts + 1]</c>.</param>
    /// <param name="counters">Per-expert atomic counters, <c>[numExperts]</c> (zeroed).</param>
    /// <param name="xqOut">Expert-sorted packed int8, <c>[expandedRows, K/4]</c> uints.</param>
    /// <param name="xdsOut">Expert-sorted (d, s) scales, <c>[expandedRows, K/32]</c> vec2.</param>
    /// <param name="permutation">Packed→original mapping, <c>[expandedRows]</c>.</param>
    /// <param name="invPermutation">Original→packed mapping, <c>[expandedRows]</c>.</param>
    /// <param name="rows">Expanded row count (seqLen × topK).</param>
    /// <param name="k">Hidden dimension (multiple of 32).</param>
    /// <param name="numExperts">Number of experts.</param>
    /// <param name="topK">Top-K routing factor.</param>
    public unsafe void Record(nint cmdBuf,
        VulkanDevice.Buffer xqIn, VulkanDevice.Buffer xdsIn,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer offsets, VulkanDevice.Buffer counters,
        VulkanDevice.Buffer xqOut, VulkanDevice.Buffer xdsOut,
        VulkanDevice.Buffer permutation, VulkanDevice.Buffer invPermutation,
        int rows, int k, int numExperts, int topK)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (k <= 0 || (k % 32) != 0) throw new ArgumentException($"k must be a positive multiple of 32, got {k}", nameof(k));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));

        int seqLen = rows / topK;
        if (xqIn.Size < QuantizeQ8_1RowsKernel.PackedBytes(seqLen, k)) throw new ArgumentException("xqIn too small.", nameof(xqIn));
        if (xdsIn.Size < QuantizeQ8_1RowsKernel.ScaleBytes(seqLen, k)) throw new ArgumentException("xdsIn too small.", nameof(xdsIn));
        if (xqOut.Size < QuantizeQ8_1RowsKernel.PackedBytes(rows, k)) throw new ArgumentException("xqOut too small.", nameof(xqOut));
        if (xdsOut.Size < QuantizeQ8_1RowsKernel.ScaleBytes(rows, k)) throw new ArgumentException("xdsOut too small.", nameof(xdsOut));
        if (indices.Size < (long)rows * sizeof(int)) throw new ArgumentException("indices too small.", nameof(indices));
        if (offsets.Size < (long)(numExperts + 1) * sizeof(uint)) throw new ArgumentException("offsets too small.", nameof(offsets));
        if (counters.Size < (long)numExperts * sizeof(uint)) throw new ArgumentException("counters too small.", nameof(counters));
        if (permutation.Size < (long)rows * sizeof(uint)) throw new ArgumentException("permutation too small.", nameof(permutation));
        if (invPermutation.Size < (long)rows * sizeof(uint)) throw new ArgumentException("invPermutation too small.", nameof(invPermutation));

        Span<nint> buffers = stackalloc nint[BuffersPerSet]
        {
            xqIn.Handle, xdsIn.Handle, indices.Handle, offsets.Handle, counters.Handle,
            xqOut.Handle, xdsOut.Handle, permutation.Handle, invPermutation.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4] { (uint)rows, (uint)k, (uint)numExperts, (uint)topK };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)rows, 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptorPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
