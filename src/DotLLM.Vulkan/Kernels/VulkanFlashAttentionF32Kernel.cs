using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Flash-Attention-v2 style FP32 attention kernel for the GQA prefill path.
/// Each workgroup processes one (query-head, query-tile of <see cref="QueryTileRows"/>
/// rows) pair, amortising K/V reads across the resident Q-tile.
/// </summary>
/// <remarks>
/// <para>
/// Parity target: <c>DotLLM.Cpu.Kernels.Attention.ExecuteScalar</c> (the CPU
/// scalar reference). Same scale, same masking semantics, same softmax shape.
/// Numerical drift comes from reordered reductions; tolerance is in line with
/// the per-token <see cref="AttentionF32Kernel"/> (abs 1e-4 / rel 1e-3 on
/// Llama-shape configs).
/// </para>
/// <para>
/// The dispatch geometry expects <c>seqQ &gt; 1</c>: the FA path reduces K/V
/// traffic by <c>BR</c> when many Q-rows share a KV tile. For <c>seqQ == 1</c>
/// (decode) the legacy <see cref="AttentionF32Kernel"/> already does one KV
/// read per workgroup — FA would only add overhead. Callers MUST route decode
/// to the legacy kernel.
/// </para>
/// <para>
/// Soft-cap support: when the model carries a non-zero attention soft-cap
/// (Gemma 2, Qwen3 thinking variants), raw scores are passed through
/// <c>softCap * tanh(score / softCap)</c> before softmax. Pass <c>0.0f</c> to
/// disable.
/// </para>
/// <para>
/// Mirrors the parent <see cref="AttentionF32Kernel"/> launch pattern:
/// self-contained synchronous <see cref="Launch"/> (allocate descriptor set,
/// record one-shot command buffer, submit, <c>vkQueueWaitIdle</c>, reset the
/// single-set descriptor pool). When the scaffold gains a command-buffer
/// batching path, both attention kernels gain a <c>Record</c> companion in
/// the same wave.
/// </para>
/// </remarks>
public sealed class VulkanFlashAttentionF32Kernel : IDisposable
{
    /// <summary>Compile-time upper bound on head_dim baked into the shader.</summary>
    public const int MaxHeadDim = 128;

    /// <summary>
    /// Number of query rows processed by a single workgroup. The KV stream
    /// is read once per workgroup, so this is the amortisation factor for
    /// KV memory traffic on the prefill path.
    /// </summary>
    public const int QueryTileRows = 16;

    /// <summary>KV tile (columns) per workgroup iteration.</summary>
    public const int KvTileCols = 64;

    // 8 uints + 1 float + 3 uint padding (= 12 * 4 = 48 bytes).
    private const int PushConstantBytes = 12 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private bool _disposed;

    private VulkanFlashAttentionF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
    }

    /// <summary>
    /// Loads <c>attention_flash_f32.spv</c> from the given directory and
    /// creates the compute pipeline. Throws if the SPV is missing — callers
    /// that want a graceful fallback should wrap in try/catch and route to
    /// <see cref="AttentionF32Kernel"/>, or use <see cref="TryCreate"/>.
    /// </summary>
    public static VulkanFlashAttentionF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "attention_flash_f32.spv");
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
        return new VulkanFlashAttentionF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>
    /// <c>TryCreate</c> companion — returns <c>null</c> when the SPV is
    /// missing or pipeline creation fails, instead of throwing. Used by the
    /// model loader so older builds without the FA SPV silently fall back to
    /// the per-token shader.
    /// </summary>
    public static VulkanFlashAttentionF32Kernel? TryCreate(VulkanDevice device, string spvDir)
    {
        try
        {
            return Create(device, spvDir);
        }
        catch (FileNotFoundException)
        {
            return null;
        }
        catch (VulkanException)
        {
            return null;
        }
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
    /// Dispatches the Flash-Attention kernel: same softmax(QK^T/sqrt(d) + mask) V
    /// math as <see cref="AttentionF32Kernel.Launch"/>, but each workgroup
    /// streams the KV sequence once across <see cref="QueryTileRows"/> Q-rows
    /// of one (head, q-tile) pair instead of one (token, head) pair.
    /// Synchronous — returns after <c>vkQueueWaitIdle</c>.
    /// </summary>
    /// <param name="q">FP32 Q tensor, layout <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="k">FP32 K tensor, layout <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="v">FP32 V tensor, layout <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="output">FP32 output, layout <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="seqQ">Query length (must be &gt;= 1; FA only amortises when &gt; 1).</param>
    /// <param name="seqKv">Key/value length (total context).</param>
    /// <param name="numHeads">Query-head count.</param>
    /// <param name="numKvHeads">KV-head count (must divide <paramref name="numHeads"/>).</param>
    /// <param name="headDim">Per-head dimension; must be &lt;= <see cref="MaxHeadDim"/>.</param>
    /// <param name="positionOffset">Offset added to q positions for causal masking (decode: cached-tokens count).</param>
    /// <param name="slidingWindow">Sliding-window size in tokens; <c>0</c> disables.</param>
    /// <param name="useAlibi">When <c>true</c> the shader applies ALiBi position biases.</param>
    /// <param name="softCap">Non-zero applies <c>softCap * tanh(s/softCap)</c> per Gemma-2 / Qwen3.</param>
    public unsafe void Launch(
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset = 0, int slidingWindow = 0, bool useAlibi = false, float softCap = 0.0f)
    {
        if (seqQ <= 0) throw new ArgumentOutOfRangeException(nameof(seqQ));
        if (seqKv <= 0) throw new ArgumentOutOfRangeException(nameof(seqKv));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKvHeads ({numKvHeads})", nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (headDim > MaxHeadDim)
            throw new ArgumentException(
                $"headDim ({headDim}) exceeds shader MAX_HEAD_DIM ({MaxHeadDim}). " +
                $"Either rebuild attention_flash_f32.comp with a larger bound or route to {nameof(AttentionF32Kernel)}.",
                nameof(headDim));
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));
        if (slidingWindow < 0) throw new ArgumentOutOfRangeException(nameof(slidingWindow));
        if (softCap < 0.0f) throw new ArgumentOutOfRangeException(nameof(softCap),
            "softCap must be non-negative (use 0 to disable).");

        long qBytes = (long)seqQ * numHeads * headDim * sizeof(float);
        long kvBytes = (long)seqKv * numKvHeads * headDim * sizeof(float);
        long outBytes = qBytes;
        if (q.Size < qBytes) throw new ArgumentException("Q buffer too small.", nameof(q));
        if (k.Size < kvBytes) throw new ArgumentException("K buffer too small.", nameof(k));
        if (v.Size < kvBytes) throw new ArgumentException("V buffer too small.", nameof(v));
        if (output.Size < outBytes) throw new ArgumentException("Output buffer too small.", nameof(output));

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
        bufferInfos[0] = new VkDescriptorBufferInfo { buffer = q.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[1] = new VkDescriptorBufferInfo { buffer = k.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[2] = new VkDescriptorBufferInfo { buffer = v.Handle, offset = 0, range = ulong.MaxValue };
        bufferInfos[3] = new VkDescriptorBufferInfo { buffer = output.Handle, offset = 0, range = ulong.MaxValue };

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

            // Push-constant layout (matches the shader's PushConstants block):
            //   [0] seqQ, [1] seqKv, [2] numHeads, [3] numKvHeads,
            //   [4] headDim, [5] positionOffset, [6] slidingWindow, [7] useAlibi,
            //   [8] softCap (float, reinterpreted), [9..11] padding for std140 alignment.
            Span<uint> pc = stackalloc uint[12];
            pc[0] = (uint)seqQ;
            pc[1] = (uint)seqKv;
            pc[2] = (uint)numHeads;
            pc[3] = (uint)numKvHeads;
            pc[4] = (uint)headDim;
            pc[5] = (uint)positionOffset;
            pc[6] = (uint)slidingWindow;
            pc[7] = useAlibi ? 1u : 0u;
            pc[8] = BitConverter.SingleToUInt32Bits(softCap);
            pc[9] = 0u;
            pc[10] = 0u;
            pc[11] = 0u;
            fixed (uint* pcPtr = pc)
            {
                VulkanApi.vkCmdPushConstants(
                    cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                    0, PushConstantBytes, (nint)pcPtr);
            }

            // One workgroup per (q-tile, head) pair.
            uint qTiles = ((uint)seqQ + (uint)QueryTileRows - 1u) / (uint)QueryTileRows;
            uint groups = qTiles * (uint)numHeads;
            VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);

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
