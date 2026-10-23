using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Fused "broadcast + expand-group-by-expert": copies each token's activation row straight into expert-grouped order (16-byte
/// accesses) and records both permutations. Replaces <c>MoeBroadcastF32Kernel</c> + <c>MoeExpandGroupByExpertF32Kernel</c> on the
/// grouped MoE prefill path: two passes over a 32 MB matrix become one.
/// </summary>
public sealed class MoeExpandGatherGroupF32Kernel : IDisposable
{
    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeExpandGatherGroupF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 7);
    }

    /// <summary>True when the SPIR-V is present.</summary>
    public static bool IsSupportedOn(string spvDir) => File.Exists(Path.Combine(spvDir, "moe_expand_gather_group_f32.spv"));

    /// <summary>Creates the kernel.</summary>
    public static MoeExpandGatherGroupF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_expand_gather_group_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");
        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[7];
            for (int i = 0; i < bindings.Length; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 7);
        return new MoeExpandGatherGroupF32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous one-shot launch (tests).</summary>
    public void Launch(VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer offsets, VulkanDevice.Buffer counters,
        VulkanDevice.Buffer packed, VulkanDevice.Buffer permutation, VulkanDevice.Buffer invPermutation,
        int rows, int hidden, int numExperts, int topK)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, x, indices, offsets, counters, packed, permutation, invPermutation, rows, hidden, numExperts, topK);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the fused copy. <paramref name="x"/> is <c>[rows / topK, hidden]</c> (the per-token activations); <paramref name="rows"/> is
    /// the routed-row count (<c>seqLen * topK</c>); <paramref name="hidden"/> must be a multiple of 4.
    /// </summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer offsets,
        VulkanDevice.Buffer counters, VulkanDevice.Buffer packed, VulkanDevice.Buffer permutation, VulkanDevice.Buffer invPermutation,
        int rows, int hidden, int numExperts, int topK)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (hidden <= 0 || (hidden & 3) != 0) throw new ArgumentException("hidden must be a positive multiple of 4.", nameof(hidden));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (topK <= 0 || rows % topK != 0) throw new ArgumentException("rows must be a multiple of topK.", nameof(topK));

        if (x.Size < (long)(rows / topK) * hidden * sizeof(float)) throw new ArgumentException("x buffer too small.", nameof(x));
        if (packed.Size < (long)rows * hidden * sizeof(float)) throw new ArgumentException("packed buffer too small.", nameof(packed));
        if (indices.Size < (long)rows * sizeof(int)) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (permutation.Size < (long)rows * sizeof(uint)) throw new ArgumentException("permutation buffer too small.", nameof(permutation));
        if (invPermutation.Size < (long)rows * sizeof(uint)) throw new ArgumentException("invPermutation buffer too small.", nameof(invPermutation));
        if (offsets.Size < (long)(numExperts + 1) * sizeof(uint)) throw new ArgumentException("offsets buffer too small.", nameof(offsets));
        if (counters.Size < (long)numExperts * sizeof(uint)) throw new ArgumentException("counters buffer too small.", nameof(counters));

        Span<nint> buffers = stackalloc nint[7]
        {
            x.Handle, indices.Handle, offsets.Handle, counters.Handle, packed.Handle, permutation.Handle, invPermutation.Handle,
        };
        nint set = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[4] { (uint)rows, (uint)hidden, (uint)numExperts, (uint)topK };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
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

/// <summary>
/// Weighted combine over the expert-GROUPED down-projection output (<c>moe_weighted_scatter_grouped_f32</c>): the ungroup pass fused
/// into <c>MoeWeightedScatterF32Kernel</c>. Bit-identical to running the two kernels back to back.
/// </summary>
public sealed class MoeWeightedScatterGroupedF32Kernel : IDisposable
{
    private const int PushConstantBytes = 3 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeWeightedScatterGroupedF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>True when the SPIR-V is present.</summary>
    public static bool IsSupportedOn(string spvDir) => File.Exists(Path.Combine(spvDir, "moe_weighted_scatter_grouped_f32.spv"));

    /// <summary>Creates the kernel.</summary>
    public static MoeWeightedScatterGroupedF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_weighted_scatter_grouped_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");
        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            for (int i = 0; i < bindings.Length; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MoeWeightedScatterGroupedF32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous one-shot launch (tests).</summary>
    public void Launch(VulkanDevice.Buffer grouped, VulkanDevice.Buffer invPermutation, VulkanDevice.Buffer weights,
        VulkanDevice.Buffer output, int seqLen, int topK, int hiddenSize)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, grouped, invPermutation, weights, output, seqLen, topK, hiddenSize);
        ctx.SubmitAndWait();
    }

    /// <summary>Records <c>out[t] = sum_slot w[t, slot] * grouped[invPermutation[t * topK + slot]]</c>.</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer grouped, VulkanDevice.Buffer invPermutation, VulkanDevice.Buffer weights,
        VulkanDevice.Buffer output, int seqLen, int topK, int hiddenSize)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        long rows = (long)seqLen * topK;
        if (grouped.Size < rows * hiddenSize * sizeof(float)) throw new ArgumentException("grouped buffer too small.", nameof(grouped));
        if (invPermutation.Size < rows * sizeof(uint)) throw new ArgumentException("invPermutation buffer too small.", nameof(invPermutation));
        if (weights.Size < rows * sizeof(float)) throw new ArgumentException("weights buffer too small.", nameof(weights));
        if (output.Size < (long)seqLen * hiddenSize * sizeof(float)) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[4] { grouped.Handle, invPermutation.Handle, weights.Handle, output.Handle };
        nint set = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[3] { (uint)seqLen, (uint)topK, (uint)hiddenSize };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((hiddenSize + 15) / 16), (uint)((seqLen + 15) / 16), 1);
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
