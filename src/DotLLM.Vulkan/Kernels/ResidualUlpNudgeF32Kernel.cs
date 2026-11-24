using System.Buffers.Binary;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// TEST-ONLY (#1919): moves a contiguous range of F32 elements exactly one ULP away from zero, in place on a device buffer. It backs the
/// GenerationSession sensitivity control's Vulkan 1-ULP residual perturbation (see <c>IVulkanResidualModifier</c>); no production
/// forward records it, so the unset-modifier path costs nothing.
/// </summary>
internal sealed class ResidualUlpNudgeF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = 2 * sizeof(uint);   // offset, count

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private ResidualUlpNudgeF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 1);
    }

    /// <summary>Loads <c>residual_ulp_nudge_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static ResidualUlpNudgeF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "residual_ulp_nudge_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1).");
        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[1];
            bindings[0] = new VkDescriptorBinding(0);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch { module.Dispose(); throw; }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 1);
        return new ResidualUlpNudgeF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set.</summary>
    public void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the nudge of <paramref name="count"/> F32 elements of <paramref name="x"/> starting at element <paramref name="offset"/>.</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer x, int offset, int count)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (offset < 0 || (long)(offset + count) * sizeof(float) > x.Size) throw new ArgumentOutOfRangeException(nameof(offset));

        Span<nint> buffers = stackalloc nint[1] { x.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);
        Span<byte> pc = stackalloc byte[PushConstantBytes];
        BinaryPrimitives.WriteUInt32LittleEndian(pc, (uint)offset);
        BinaryPrimitives.WriteUInt32LittleEndian(pc[4..], (uint)count);
        fixed (byte* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((count + WorkgroupSize - 1) / WorkgroupSize), 1, 1);
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
