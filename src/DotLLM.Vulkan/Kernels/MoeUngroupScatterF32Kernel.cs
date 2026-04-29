using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>Scatters grouped MoE rows back to their original routed-row order.</summary>
public sealed class MoeUngroupScatterF32Kernel : IDisposable
{
    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    private const int PushConstantBytes = 2 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeUngroupScatterF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>moe_ungroup_scatter_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeUngroupScatterF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_ungroup_scatter_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new MoeUngroupScatterF32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch; used by unit tests.</summary>
    public void Launch(VulkanDevice.Buffer packed, VulkanDevice.Buffer permutation, VulkanDevice.Buffer output,
        int rows, int hidden)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, packed, permutation, output, rows, hidden);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the ungroup/scatter dispatch into <paramref name="cmdBuf"/>.</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer packed, VulkanDevice.Buffer permutation,
        VulkanDevice.Buffer output, int rows, int hidden)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));

        long matrixBytes = (long)rows * hidden * sizeof(float);
        if (packed.Size < matrixBytes) throw new ArgumentException("packed buffer too small.", nameof(packed));
        if (output.Size < matrixBytes) throw new ArgumentException("output buffer too small.", nameof(output));
        if (permutation.Size < (long)rows * sizeof(uint)) throw new ArgumentException("permutation buffer too small.", nameof(permutation));

        Span<nint> buffers = stackalloc nint[3] { packed.Handle, permutation.Handle, output.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[2] { (uint)rows, (uint)hidden };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupX = (uint)((hidden + WorkgroupX - 1) / WorkgroupX);
        uint groupY = (uint)((rows + WorkgroupY - 1) / WorkgroupY);
        VulkanApi.vkCmdDispatch(cmdBuf, groupX, groupY, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
