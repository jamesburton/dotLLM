using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ4_XS → FP32 dequantization. Reads a tightly-packed IQ4_XS blob and
/// produces a contiguous FP32 buffer.
/// </summary>
/// <remarks>
/// Layout matches <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeIQ4_XS</c> and
/// llama.cpp's <c>block_iq4_xs</c>. One workgroup per 256-element super-block,
/// 256 threads per workgroup, one thread per output element.
/// </remarks>
public sealed class Iq4XsDequantF32Kernel : IDisposable
{
    /// <summary>IQ4_XS super-block: 2 + 2 + 4 + 128 = 136 bytes.</summary>
    public const int IQ4_XSBlockBytes = 136;

    /// <summary>Elements per IQ4_XS super-block.</summary>
    public const int IQ4_XSGroupSize = 256;

    private const int PushConstantBytes = 2 * sizeof(uint); // totalSuperblocks, srcUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private Iq4XsDequantF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 2);
    }

    /// <summary>Loads <c>iq4_xs_dequant_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static Iq4XsDequantF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "iq4_xs_dequant_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[2];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 2);
        return new Iq4XsDequantF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the dequant synchronously.</summary>
    public void Launch(VulkanDevice.Buffer src, VulkanDevice.Buffer dst, int totalSuperblocks)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, src, dst, totalSuperblocks);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the dequant into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, int totalSuperblocks)
    {
        if (totalSuperblocks <= 0) throw new ArgumentOutOfRangeException(nameof(totalSuperblocks));

        long srcMin = (long)totalSuperblocks * IQ4_XSBlockBytes;
        long dstMin = (long)totalSuperblocks * IQ4_XSGroupSize * sizeof(float);
        if (src.Size < srcMin)
            throw new ArgumentException($"Source buffer too small: need >= {srcMin} bytes.", nameof(src));
        if (dst.Size < dstMin)
            throw new ArgumentException($"Destination buffer too small: need >= {dstMin} bytes.", nameof(dst));

        int srcUints = (int)((srcMin + 3) / 4);

        Span<nint> buffers = stackalloc nint[2] { src.Handle, dst.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[2] { (uint)totalSuperblocks, (uint)srcUints };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)totalSuperblocks, 1, 1);
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
