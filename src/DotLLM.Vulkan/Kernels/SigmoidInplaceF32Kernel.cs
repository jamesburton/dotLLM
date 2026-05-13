using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// In-place sigmoid activation: <c>x[i] = 1 / (1 + exp(-x[i]))</c>.
/// Mirrors <see cref="System.Numerics.Tensors.TensorPrimitives.Sigmoid(System.ReadOnlySpan{float}, System.Span{float})"/>.
/// </summary>
/// <remarks>
/// Used by the Qwen3MoeHybrid GDN token-mixing path to fold the write-gate
/// <c>beta = sigmoid(beta_proj)</c> onto the device, eliminating the
/// D2H/host-compute/H2D roundtrip that <c>ComputeDecayAndBetaOnHost</c>
/// previously required. Pointwise — one thread per element.
/// </remarks>
public sealed class SigmoidInplaceF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = sizeof(uint); // n

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private SigmoidInplaceF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 1);
    }

    /// <summary>Loads <c>sigmoid_inplace_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static SigmoidInplaceF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "sigmoid_inplace_f32.spv");
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
        return new SigmoidInplaceF32Kernel(device, module, pipeline, pool);
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
    /// Records the in-place sigmoid dispatch. <paramref name="x"/> is treated
    /// as a flat F32 array of length <paramref name="n"/>.
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
