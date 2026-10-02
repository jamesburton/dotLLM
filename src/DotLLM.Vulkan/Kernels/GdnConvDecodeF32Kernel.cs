using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Single-token fused GDN causal conv: conv1d + bias + SiLU + conv-state shift in one dispatch (replaces 2 copies, 3 barrier pairs and
/// 3 dispatches per GDN layer on the decode path). See <c>gdn_conv_decode_f32.comp</c>. Requires <c>dConv &gt;= 2</c>.
/// </summary>
public sealed class GdnConvDecodeF32Kernel : IDisposable
{
    private const int Workgroup = 256;
    // dConv, channels (u32)
    private const int PushConstantBytes = 2 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private GdnConvDecodeF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>Loads <c>gdn_conv_decode_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static GdnConvDecodeF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "gdn_conv_decode_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            bindings[3] = new VkDescriptorBinding(3);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new GdnConvDecodeF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(VulkanDevice.Buffer state, VulkanDevice.Buffer weight, VulkanDevice.Buffer bias, VulkanDevice.Buffer x, int dConv, int channels)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, state, weight, bias, x, dConv, channels);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the fused decode conv. <paramref name="state"/> is the <c>[dConv-1, channels]</c> conv state (shifted in place),
    /// <paramref name="x"/> the <c>[channels]</c> qkv row (overwritten with silu(conv)).
    /// </summary>
    public unsafe void Record(
        nint cmdBuf, VulkanDevice.Buffer state, VulkanDevice.Buffer weight, VulkanDevice.Buffer bias, VulkanDevice.Buffer x,
        int dConv, int channels)
    {
        if (dConv < 2) throw new ArgumentOutOfRangeException(nameof(dConv));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (state.Size < (long)(dConv - 1) * channels * sizeof(float)) throw new ArgumentException("state buffer too small.", nameof(state));
        if (weight.Size < (long)dConv * channels * sizeof(float)) throw new ArgumentException("weight buffer too small.", nameof(weight));
        if (bias.Size < (long)channels * sizeof(float)) throw new ArgumentException("bias buffer too small.", nameof(bias));
        if (x.Size < (long)channels * sizeof(float)) throw new ArgumentException("x buffer too small.", nameof(x));

        Span<nint> buffers = stackalloc nint[4] { state.Handle, weight.Handle, bias.Handle, x.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[2] { (uint)dConv, (uint)channels };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        }
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((channels + Workgroup - 1) / Workgroup), 1, 1);
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
