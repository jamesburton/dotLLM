using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE indexed expert matmul over a packed Q6_K expert bank:
/// <c>y[n, m] = dequant_q6k(bank[indices[n], m, :]) dot x[n, :]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Per-row Q6_K dequantisation in the inner loop — the bank stays in its
/// 210-byte-per-super-block GGUF layout on device. Sibling of
/// <see cref="MoeIndexedMatmulF32Kernel"/>; the dequant matches
/// <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ6_KScalar</c>
/// byte-for-byte and the indexed expert lookup comes straight from
/// <see cref="MoeIndexedMatmulF32Kernel"/>.
/// </para>
/// <para>
/// Lifts the resident-MoE memory cap on Strix Halo for Qwen3.6-A3B-Q6_K_XL
/// (Phase 10 follow-up): a fully F32-dequantised resident MoE for that
/// checkpoint would consume roughly 120 GB; the Q6_K-resident layout fits
/// in ~25 GB and the on-device dequant per matmul keeps the activation
/// path identical to the streaming F32 path.
/// </para>
/// <para>
/// Adapted to the inline-pool pattern: each <see cref="Launch"/> allocates a
/// fresh descriptor set, records a one-time-submit command buffer, waits on
/// the queue, then resets the descriptor pool. Matches the existing
/// <see cref="MoeIndexedMatmulF32Kernel"/> Launch convention.
/// </para>
/// </remarks>
public sealed class MoeIndexedMatmulQ6_KF32Kernel : IDisposable
{
    /// <summary>Q6_K super-block: 128(ql) + 64(qh) + 16(scales) + 2(d) = 210 bytes.</summary>
    public const int Q6_KBlockBytes = 210;

    /// <summary>Elements per Q6_K super-block.</summary>
    public const int Q6_KGroupSize = 256;

    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    // M, K, N, numExperts, blocksPerRow
    private const int PushConstantBytes = 5 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private bool _disposed;

    private MoeIndexedMatmulQ6_KF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
    }

    /// <summary>Loads <c>moe_indexed_matmul_q6_k_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeIndexedMatmulQ6_KF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_indexed_matmul_q6_k_f32.spv");
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

        nint pool = CreateDescriptorPool(device);
        return new MoeIndexedMatmulQ6_KF32Kernel(device, module, pipeline, pool);
    }

    private static unsafe nint CreateDescriptorPool(VulkanDevice device)
    {
        var poolSize = new VkDescriptorPoolSize
        {
            type = VkDescriptorType.StorageBuffer,
            descriptorCount = 4,
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
    /// Synchronous dispatch of the indexed Q6_K matmul. Returns after
    /// <c>vkQueueWaitIdle</c>.
    /// </summary>
    /// <param name="bankQ6K">
    /// Raw Q6_K bank of <c>numExperts * M * (K/256) * 210</c> bytes, expert
    /// matrices contiguous, rows contiguous within each expert.
    /// </param>
    /// <param name="x">F32 input rows [<paramref name="n"/> * K] row-major.</param>
    /// <param name="indices">int32 per-row expert index [<paramref name="n"/>].</param>
    /// <param name="y">F32 output rows [<paramref name="n"/> * M] row-major.</param>
    /// <param name="m">Per-expert weight row count (output dim).</param>
    /// <param name="k">Per-expert weight column count (must be a multiple of 256).</param>
    /// <param name="n">Number of output rows (typically <c>seqLen * topK</c>).</param>
    /// <param name="numExperts">Bank's first axis size — used for bounds-checking the index lookup.</param>
    public unsafe void Launch(
        VulkanDevice.Buffer bankQ6K, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if ((k % Q6_KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q6_KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q6_KGroupSize;
        long rowBytes = (long)blocksPerRow * Q6_KBlockBytes;
        long bankBytes = (long)numExperts * m * rowBytes;
        long xBytes = (long)n * k * sizeof(float);
        long idxBytes = (long)n * sizeof(int);
        long yBytes = (long)n * m * sizeof(float);
        if (bankQ6K.Size < bankBytes) throw new ArgumentException("bankQ6K buffer too small.", nameof(bankQ6K));
        if (x.Size < xBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (indices.Size < idxBytes) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (y.Size < yBytes) throw new ArgumentException("y buffer too small.", nameof(y));

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
        Span<VkDescriptorBufferInfo> bufferInfos = stackalloc VkDescriptorBufferInfo[4];
        bufferInfos[0] = new VkDescriptorBufferInfo { buffer = bankQ6K.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[1] = new VkDescriptorBufferInfo { buffer = x.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[2] = new VkDescriptorBufferInfo { buffer = indices.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[3] = new VkDescriptorBufferInfo { buffer = y.Handle, offset = 0, range = ulong.MaxValue };

        Span<VkWriteDescriptorSet> writes = stackalloc VkWriteDescriptorSet[4];
        fixed (VkDescriptorBufferInfo* bufPtr = bufferInfos)
        {
            for (int i = 0; i < 4; i++)
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
                VulkanApi.vkUpdateDescriptorSets(_device.Handle, 4, (nint)writesPtr, 0, 0);
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

            Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)m);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)k);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)n);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)numExperts);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[16..], (uint)blocksPerRow);
            fixed (byte* pcPtr = pcBytes)
            {
                VulkanApi.vkCmdPushConstants(
                    cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                    0, PushConstantBytes, (nint)pcPtr);
            }

            uint groupX = (uint)((m + WorkgroupX - 1) / WorkgroupX);
            uint groupY = (uint)((n + WorkgroupY - 1) / WorkgroupY);
            VulkanApi.vkCmdDispatch(cmdBuf, groupX, groupY, 1);

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
            // Pool-reset after the queue wait frees the descriptor set for the next Launch();
            // without this the single-set pool exhausts on the second invocation.
            VulkanApi.vkResetDescriptorPool(_device.Handle, _descriptorPool, 0);
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
