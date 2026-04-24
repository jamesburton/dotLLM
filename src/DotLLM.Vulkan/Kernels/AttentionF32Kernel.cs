using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// FP32 scaled-dot-product attention with causal masking, GQA head broadcast,
/// and flash-attention-style online softmax. One workgroup per
/// (query-token, query-head) pair; shared-memory tiled softmax over the KV
/// sequence mirrors <c>attention_f32.cu</c>.
/// </summary>
/// <remarks>
/// <para>
/// Parity target: the CUDA kernel <c>attention_f32</c>. Both do a running
/// max / sum_exp update per KV tile, rescale the output accumulator by
/// <c>exp(oldMax - newMax)</c>, and finally divide by the running sum. No
/// subgroup intrinsics (<c>subgroupMax</c> / <c>subgroupAdd</c>) — the
/// workgroup reduces through shared memory, same rationale as the
/// wave-1 kernels (broadest driver portability).
/// </para>
/// <para>
/// Tile size <c>TILE_KV = 256</c> matches CUDA. <c>MAX_HEAD_DIM = 256</c> in
/// the shader bounds the shared-memory footprint — well above any current
/// Llama/Mistral/Phi/DeepSeek/SmolLM head dim (64 or 128).
/// </para>
/// </remarks>
public sealed class AttentionF32Kernel : IDisposable
{
    /// <summary>Fixed compile-time upper bound on head_dim in the shader.</summary>
    public const int MaxHeadDim = 256;

    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = 7 * sizeof(uint); // seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, slidingWindow

    private readonly VulkanDevice _device;
    private readonly VulkanModule _sharedModule;
    private readonly ComputePipeline _sharedPipeline;
    private readonly VulkanModule? _subgroupModule;
    private readonly ComputePipeline? _subgroupPipeline;
    private readonly nint _descriptorPool;
    private readonly bool _useSubgroup;
    private bool _disposed;

    /// <summary>
    /// True when this kernel dispatches the <c>attention_f32_sg.spv</c>
    /// subgroup-arithmetic variant; false when it uses the shared-memory
    /// reference. Exposed for tests and telemetry.
    /// </summary>
    public bool UsesSubgroupReduce => _useSubgroup;

    private AttentionF32Kernel(
        VulkanDevice device,
        VulkanModule sharedModule, ComputePipeline sharedPipeline,
        VulkanModule? subgroupModule, ComputePipeline? subgroupPipeline,
        nint pool, bool useSubgroup)
    {
        _device = device;
        _sharedModule = sharedModule;
        _sharedPipeline = sharedPipeline;
        _subgroupModule = subgroupModule;
        _subgroupPipeline = subgroupPipeline;
        _descriptorPool = pool;
        _useSubgroup = useSubgroup;
    }

    /// <summary>
    /// Loads <c>attention_f32.spv</c> (always) and <c>attention_f32_sg.spv</c>
    /// (when the device advertises subgroup arithmetic) from the given
    /// directory and creates the pipelines.
    /// </summary>
    /// <remarks>
    /// Selects the subgroup variant at runtime when <see cref="VulkanDevice.HasSubgroupArithmetic"/>
    /// is <c>true</c> and <c>DOTLLM_VULKAN_FORCE_SHARED_REDUCE</c> is not set
    /// to <c>1</c>. Subgroup SPV is optional; silently falls back if missing.
    /// </remarks>
    public static AttentionF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string sharedPath = Path.Combine(spvDir, "attention_f32.spv");
        if (!File.Exists(sharedPath))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {sharedPath}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule sharedModule = VulkanModule.LoadFromFile(device, sharedPath);
        ComputePipeline sharedPipeline;
        try
        {
            sharedPipeline = CreatePipeline(sharedModule);
        }
        catch
        {
            sharedModule.Dispose();
            throw;
        }

        VulkanModule? subgroupModule = null;
        ComputePipeline? subgroupPipeline = null;
        bool useSubgroup = device.HasSubgroupArithmetic && !RmsNormF32Kernel.IsForceSharedReduce();
        if (useSubgroup)
        {
            string subgroupPath = Path.Combine(spvDir, "attention_f32_sg.spv");
            if (File.Exists(subgroupPath))
            {
                try
                {
                    subgroupModule = VulkanModule.LoadFromFile(device, subgroupPath);
                    subgroupPipeline = CreatePipeline(subgroupModule);
                }
                catch
                {
                    subgroupModule?.Dispose();
                    subgroupModule = null;
                    subgroupPipeline = null;
                    useSubgroup = false;
                }
            }
            else
            {
                useSubgroup = false;
            }
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new AttentionF32Kernel(
            device, sharedModule, sharedPipeline,
            subgroupModule, subgroupPipeline, pool, useSubgroup);
    }

    private static ComputePipeline CreatePipeline(VulkanModule module)
    {
        Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
        bindings[0] = new VkDescriptorBinding(0);
        bindings[1] = new VkDescriptorBinding(1);
        bindings[2] = new VkDescriptorBinding(2);
        bindings[3] = new VkDescriptorBinding(3);
        return module.CreateComputePipeline(
            entryPoint: "main",
            bindings: bindings,
            pushConstantBytes: PushConstantBytes);
    }

    /// <summary>Resets this kernel's descriptor pool; call at the start of each forward pass.</summary>
    internal void ResetDescriptors() => KernelSupport.ResetPool(_device, _descriptorPool);

    /// <summary>
    /// Dispatches attention: <c>output = softmax((Q K^T)/sqrt(headDim) + mask) V</c>
    /// for every (query token, query head) pair. Synchronous — returns after
    /// <c>vkQueueWaitIdle</c>.
    /// </summary>
    /// <param name="q">FP32 Q tensor, layout <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="k">FP32 K tensor, layout <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="v">FP32 V tensor, layout <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="output">FP32 output, layout <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="seqQ">Query length.</param>
    /// <param name="seqKv">Key/value length (total context).</param>
    /// <param name="numHeads">Query-head count.</param>
    /// <param name="numKvHeads">KV-head count (must divide <paramref name="numHeads"/>).</param>
    /// <param name="headDim">Per-head dimension; must be &lt;= <see cref="MaxHeadDim"/>.</param>
    /// <param name="positionOffset">Offset added to q positions for causal masking (decode: cached-tokens count).</param>
    /// <param name="slidingWindow">Sliding-window size in tokens; <c>0</c> disables.</param>
    public void Launch(
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset = 0, int slidingWindow = 0)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, slidingWindow);
        ctx.SubmitAndWait();
        ResetDescriptors();
    }

    /// <summary>Records attention into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset = 0, int slidingWindow = 0)
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
                $"headDim ({headDim}) exceeds shader MAX_HEAD_DIM ({MaxHeadDim}). Rebuild attention_f32.comp with a larger bound.",
                nameof(headDim));
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));
        if (slidingWindow < 0) throw new ArgumentOutOfRangeException(nameof(slidingWindow));

        long qBytes   = (long)seqQ  * numHeads   * headDim * sizeof(float);
        long kvBytes  = (long)seqKv * numKvHeads * headDim * sizeof(float);
        long outBytes = qBytes;
        if (q.Size      < qBytes)   throw new ArgumentException("Q buffer too small.",      nameof(q));
        if (k.Size      < kvBytes)  throw new ArgumentException("K buffer too small.",      nameof(k));
        if (v.Size      < kvBytes)  throw new ArgumentException("V buffer too small.",      nameof(v));
        if (output.Size < outBytes) throw new ArgumentException("Output buffer too small.", nameof(output));

        ComputePipeline pipeline = (_useSubgroup && _subgroupPipeline != null) ? _subgroupPipeline : _sharedPipeline;

        nint descriptorSet = KernelSupport.AllocateDescriptorSet(_device, _descriptorPool, pipeline.DescriptorSetLayout);
        Span<nint> buffers = stackalloc nint[4] { q.Handle, k.Handle, v.Handle, output.Handle };
        KernelSupport.WriteBufferBindings(_device, descriptorSet, buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[7]
        {
            (uint)seqQ,
            (uint)seqKv,
            (uint)numHeads,
            (uint)numKvHeads,
            (uint)headDim,
            (uint)positionOffset,
            (uint)slidingWindow,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // One workgroup per (tq, hq) pair.
        uint groups = (uint)seqQ * (uint)numHeads;
        VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_descriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _subgroupPipeline?.Dispose();
        _subgroupModule?.Dispose();
        _sharedPipeline.Dispose();
        _sharedModule.Dispose();
    }
}
