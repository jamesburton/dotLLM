using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q4_K → FP32 dequantization. Reads a tightly-packed Q4_K blob and produces a
/// contiguous FP32 buffer.
/// </summary>
/// <remarks>
/// Layout matches <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ4_K</c> and
/// llama.cpp's <c>block_q4_K</c>: 144 bytes per 256 elements (fp16 <c>d</c> +
/// <c>dmin</c>, 12 packed 6-bit scale/min bytes, 128 bytes of 4-bit quants).
/// One workgroup per super-block, 256 threads, one element per thread. Output
/// is bit-identical to the CPU scalar oracle (<c>precise</c>-qualified math) —
/// used by the issue-#147 GPU-side token-embed dequant at model load.
/// Large tensors (the token-embed table is millions of super-blocks) are
/// dispatched in bounded chunks via the <c>firstBlock</c> push constant so no
/// single <c>vkCmdDispatch</c> exceeds the portable group-count limit.
/// </remarks>
public sealed class Q4KDequantF32Kernel : IDisposable
{
    /// <summary>Q4_K super-block: 2 + 2 + 12 + 128 = 144 bytes.</summary>
    public const int Q4_KBlockBytes = 144;

    /// <summary>Elements per Q4_K super-block.</summary>
    public const int Q4_KGroupSize = 256;

    /// <summary>Workgroups per dispatch chunk — comfortably under the 65535 portable x-limit.</summary>
    private const int MaxBlocksPerDispatch = 32768;

    private const int PushConstantBytes = 3 * sizeof(uint); // totalBlocks, srcUints, firstBlock

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private Q4KDequantF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 2);
    }

    /// <summary>Loads <c>q4_k_dequant_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static Q4KDequantF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "q4_k_dequant_f32.spv");
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
        return new Q4KDequantF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the dequant synchronously (all chunks in one submit).</summary>
    public void Launch(VulkanDevice.Buffer src, VulkanDevice.Buffer dst, long totalBlocks)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, src, dst, totalBlocks);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the dequant into <paramref name="cmdBuf"/> without submitting.
    /// Chunks the dispatch so no single <c>vkCmdDispatch</c> exceeds
    /// <see cref="MaxBlocksPerDispatch"/> workgroups (chunks write disjoint output —
    /// no inter-chunk barrier needed).</summary>
    public unsafe void Record(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, long totalBlocks)
    {
        if (totalBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(totalBlocks));
        if (totalBlocks > uint.MaxValue) throw new ArgumentOutOfRangeException(nameof(totalBlocks));

        long srcMin = totalBlocks * Q4_KBlockBytes;
        long dstMin = totalBlocks * Q4_KGroupSize * sizeof(float);
        if (src.Size < srcMin)
            throw new ArgumentException($"Source buffer too small: need >= {srcMin} bytes.", nameof(src));
        if (dst.Size < dstMin)
            throw new ArgumentException($"Destination buffer too small: need >= {dstMin} bytes.", nameof(dst));

        uint srcUints = (uint)((srcMin + 3) / 4);

        Span<nint> buffers = stackalloc nint[2] { src.Handle, dst.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[3] { (uint)totalBlocks, srcUints, 0 };
        for (long first = 0; first < totalBlocks; first += MaxBlocksPerDispatch)
        {
            uint count = (uint)Math.Min(MaxBlocksPerDispatch, totalBlocks - first);
            pc[2] = (uint)first;
            fixed (uint* pcPtr = pc)
            {
                VulkanApi.vkCmdPushConstants(
                    cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                    0, PushConstantBytes, (nint)pcPtr);
            }
            VulkanApi.vkCmdDispatch(cmdBuf, count, 1, 1);
        }
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
