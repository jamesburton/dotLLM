using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q8_0 decode-path GEMV: <c>y[M] = W_q8[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Weight layout mirrors the CPU kernel <c>DotLLM.Cpu.Kernels.MatMul.GemvQ8_0</c>
/// and the CUDA kernel <c>quantized_gemv_q8_0</c>: each 32 contiguous columns of
/// a row form one Q8_0 block of 34 bytes — 2 bytes fp16 scale followed by
/// 32 signed int8 quantized values.
/// </para>
/// <para>
/// The activation vector <c>x</c> is FP32 (not pre-quantized) — this kernel is
/// the N=1 decode-path; prefill / batched paths that can amortize the
/// quantization of <c>x</c> are future work (matches how <c>GemmQ8_0</c>
/// delegates to <c>GemvQ8_0</c> when N==1 on the CPU side).
/// </para>
/// <para>
/// Dispatch: one workgroup per output row, 128 threads per workgroup,
/// shared-memory reduction. No subgroup / cooperative-matrix intrinsics —
/// broadest driver portability.
/// </para>
/// </remarks>
public sealed class MatMulQ8_0Kernel : IDisposable
{
    /// <summary>Q8_0 block: 2 bytes fp16 scale + 32 signed int8 values.</summary>
    public const int Q8_0BlockBytes = 34;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = 32;

    private const int WorkgroupSize = 128;
    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private bool _disposed;

    private MatMulQ8_0Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
    }

    /// <summary>Loads <c>matmul_q8_0.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulQ8_0Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_q8_0.spv");
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

        nint pool = CreateDescriptorPool(device);
        return new MatMulQ8_0Kernel(device, module, pipeline, pool);
    }

    private static unsafe nint CreateDescriptorPool(VulkanDevice device)
    {
        var poolSize = new VkDescriptorPoolSize
        {
            type = VkDescriptorType.StorageBuffer,
            descriptorCount = 3,
        };
        VkDescriptorPoolCreateInfo ci = default;
        ci.sType = VkStructureType.DescriptorPoolCreateInfo;
        ci.maxSets = 1;
        ci.poolSizeCount = 1;
        ci.pPoolSizes = (nint)(&poolSize);
        VulkanApi.vkCreateDescriptorPool(device.Handle, ci, 0, out nint pool)
            .ThrowOnError("vkCreateDescriptorPool");
        return pool;
    }

    /// <summary>
    /// Dispatches the GEMV: <c>y[M] = W[M,K] @ x[K]</c> with FP16-scaled int8 weights.
    /// Synchronous — returns after <c>vkQueueWaitIdle</c>.
    /// </summary>
    /// <param name="weightsQ8">
    /// Raw Q8_0 blob of <c>M * (K/32) * 34</c> bytes, rows contiguous.
    /// </param>
    /// <param name="x">FP32 activation buffer of length <paramref name="k"/>.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 32).</param>
    public unsafe void Launch(
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q8_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q8_0BlockBytes;
        // The row stride in bytes may not be a multiple of 4 (e.g. K=32, rowBytes=34).
        // We require the overall buffer to be large enough; the shader computes
        // `rowUints = ceil(rowBytes/4)` on the CPU side and reads individual bytes
        // out of the uint[] via shifts. Buffer total rounds up to a uint count.
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        // The shader reads up to 1 uint past the block start when an fp16 scale
        // straddles a uint boundary; ensure the buffer has the slack. In
        // practice weightsMin is already rounded up to uints by the allocator,
        // but assert the lower bound for clarity.
        if (weightsQ8.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ8.Size}.",
                nameof(weightsQ8));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        // 1. Allocate descriptor set.
        nint setLayout = _pipeline.DescriptorSetLayout;
        var dsai = new VkDescriptorSetAllocateInfo
        {
            sType = VkStructureType.DescriptorSetAllocateInfo,
            descriptorPool = _descriptorPool,
            descriptorSetCount = 1,
            pSetLayouts = (nint)(&setLayout),
        };
        VulkanApi.vkAllocateDescriptorSets(_device.Handle, dsai, out nint descriptorSet)
            .ThrowOnError("vkAllocateDescriptorSets");

        // 2. Bind buffers.
        Span<VkDescriptorBufferInfo> bufferInfos = stackalloc VkDescriptorBufferInfo[3];
        bufferInfos[0] = new VkDescriptorBufferInfo { buffer = weightsQ8.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[1] = new VkDescriptorBufferInfo { buffer = x.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[2] = new VkDescriptorBufferInfo { buffer = y.Handle, offset = 0, range = ulong.MaxValue };

        Span<VkWriteDescriptorSet> writes = stackalloc VkWriteDescriptorSet[3];
        fixed (VkDescriptorBufferInfo* bufPtr = bufferInfos)
        {
            for (int i = 0; i < 3; i++)
            {
                writes[i] = new VkWriteDescriptorSet
                {
                    sType = VkStructureType.WriteDescriptorSet,
                    dstSet = descriptorSet,
                    dstBinding = (uint)i,
                    descriptorCount = 1,
                    descriptorType = VkDescriptorType.StorageBuffer,
                    pBufferInfo = (nint)(bufPtr + i),
                };
            }
            fixed (VkWriteDescriptorSet* writesPtr = writes)
            {
                VulkanApi.vkUpdateDescriptorSets(_device.Handle, 3, (nint)writesPtr, 0, 0);
            }
        }

        // 3. Record and submit.
        var cbai = new VkCommandBufferAllocateInfo
        {
            sType = VkStructureType.CommandBufferAllocateInfo,
            commandPool = _device.CommandPool,
            level = VkCommandBufferLevel.Primary,
            commandBufferCount = 1,
        };
        VulkanApi.vkAllocateCommandBuffers(_device.Handle, cbai, out nint cmdBuf)
            .ThrowOnError("vkAllocateCommandBuffers");

        try
        {
            var begin = new VkCommandBufferBeginInfo
            {
                sType = VkStructureType.CommandBufferBeginInfo,
                flags = VkCommandBufferUsageFlags.OneTimeSubmit,
            };
            VulkanApi.vkBeginCommandBuffer(cmdBuf, begin).ThrowOnError("vkBeginCommandBuffer");

            VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
            VulkanApi.vkCmdBindDescriptorSets(
                cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
                0, 1, descriptorSet, 0, 0);

            Span<uint> pc = stackalloc uint[4]
            {
                (uint)m,
                (uint)k,
                (uint)blocksPerRow,
                (uint)rowUints,
            };
            fixed (uint* pcPtr = pc)
            {
                VulkanApi.vkCmdPushConstants(
                    cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                    0, PushConstantBytes, (nint)pcPtr);
            }

            // One workgroup per output row.
            VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, 1, 1);

            VulkanApi.vkEndCommandBuffer(cmdBuf).ThrowOnError("vkEndCommandBuffer");

            var submit = new VkSubmitInfo
            {
                sType = VkStructureType.SubmitInfo,
                commandBufferCount = 1,
                pCommandBuffers = (nint)(&cmdBuf),
            };
            VulkanApi.vkQueueSubmit(_device.Queue, 1, submit, 0).ThrowOnError("vkQueueSubmit");
            VulkanApi.vkQueueWaitIdle(_device.Queue).ThrowOnError("vkQueueWaitIdle");
        }
        finally
        {
            VulkanApi.vkFreeCommandBuffers(_device.Handle, _device.CommandPool, 1, cmdBuf);
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
