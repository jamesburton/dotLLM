using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// In-place SiLU activation: <c>x[i] = x[i] * sigmoid(x[i])</c>.
/// Mirrors <c>DotLLM.Cpu.Kernels.SiLu.Execute</c>.
/// </summary>
/// <remarks>
/// NemotronH's SSM block applies SiLU to the post-conv1d activation buffer
/// — the existing fused <see cref="SwiGluF32Kernel"/> can't be reused there
/// because it computes <c>silu(gate) * up</c> across two distinct buffers.
/// Pointwise, no reduction, one thread per element.
/// </remarks>
public sealed class SiluInplaceF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = sizeof(uint); // n

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private SiluInplaceF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 1);
    }

    /// <summary>Loads <c>silu_inplace_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static SiluInplaceF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "silu_inplace_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[1];
            bindings[0] = new VkDescriptorBinding(0);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindings,
                pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 1);
        return new SiluInplaceF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(VulkanDevice.Buffer x, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, x, n);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the in-place SiLU dispatch. <paramref name="x"/> is treated as
    /// a flat F32 array of length <paramref name="n"/>.
    /// </summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer x, int n)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));

        long bytes = (long)n * sizeof(float);
        if (x.Size < bytes) throw new ArgumentException("x buffer too small.", nameof(x));

        Span<nint> buffers = stackalloc nint[1] { x.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        uint pushN = (uint)n;
        VulkanApi.vkCmdPushConstants(
            cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
            0, sizeof(uint), (nint)(&pushN));

        uint groups = (uint)((n + WorkgroupSize - 1) / WorkgroupSize);
        VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);
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
