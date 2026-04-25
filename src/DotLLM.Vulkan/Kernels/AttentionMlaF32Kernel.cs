using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// FP32 Multi-head Latent Attention (DeepSeek-V2/V3) — post-projection
/// attention loop. Mirrors <c>DotLLM.Cpu.Kernels.MlaAttention.Execute</c>'s
/// per-head SDPA: causal mask, online softmax, weighted V sum.
/// </summary>
/// <remarks>
/// <para>
/// MLA differs from regular MHA/GQA in three ways the kernel handles:
/// <list type="bullet">
///   <item>Q has two contiguous sub-dims per head: <c>q_nope</c> (no
///     positional encoding) and <c>q_pe</c> (RoPE-rotated). Total Q head
///     dim is <c>qk_nope_head_dim + qk_rope_head_dim</c>.</item>
///   <item>K_pe is MQA-style shared across all heads — one rope-K per
///     token instead of per-head. Stored in its own buffer.</item>
///   <item>V uses its own head dim (<c>v_head_dim</c>) which may differ
///     from <c>qk_head_dim</c>. Output is per-head <c>v_head_dim</c>.</item>
/// </list>
/// </para>
/// <para>
/// First-pass implementation uses the shared-memory online-softmax variant
/// (no subgroup / coopmat tiling). Adding those variants is a follow-up
/// once the integration path is wired and end-to-end DeepSeek-V2-Lite
/// argmax parity is locked.
/// </para>
/// </remarks>
public sealed class AttentionMlaF32Kernel : IDisposable
{
    /// <summary>Compile-time upper bound on (qk_nope + qk_rope) per head — must mirror <c>MAX_QK_HEAD_DIM</c> in the shader.</summary>
    public const int MaxQkHeadDim = 256;

    /// <summary>Compile-time upper bound on v_head_dim — must mirror <c>MAX_V_HEAD_DIM</c> in the shader.</summary>
    public const int MaxVHeadDim = 256;

    private const int WorkgroupSize = 256;
    // seqQ, seqKv, numHeads, qkNopeHeadDim, qkRopeHeadDim, vHeadDim, positionOffset (u32) + scale (f32)
    private const int PushConstantBytes = 7 * sizeof(uint) + sizeof(float);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private AttentionMlaF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 5);
    }

    /// <summary>Loads <c>attention_mla_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static AttentionMlaF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "attention_mla_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[5];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            bindings[3] = new VkDescriptorBinding(3);
            bindings[4] = new VkDescriptorBinding(4);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        return new AttentionMlaF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Synchronous launch — submit and wait. Wraps <see cref="Record"/>; used
    /// by unit tests. Forward-pass code records into a persistent command
    /// buffer via <see cref="Record"/> directly.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer q, VulkanDevice.Buffer kNope, VulkanDevice.Buffer v,
        VulkanDevice.Buffer kPe, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads,
        int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
        int positionOffset, float scale)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, q, kNope, v, kPe, output,
               seqQ, seqKv, numHeads, qkNopeHeadDim, qkRopeHeadDim, vHeadDim,
               positionOffset, scale);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the MLA attention dispatch into <paramref name="cmdBuf"/>.
    /// All buffers are FP32 row-major.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="q">Q [<paramref name="seqQ"/>, <paramref name="numHeads"/> * (qkNope + qkRope)].</param>
    /// <param name="kNope">K_nope [<paramref name="seqKv"/>, <paramref name="numHeads"/> * <paramref name="qkNopeHeadDim"/>].</param>
    /// <param name="v">V [<paramref name="seqKv"/>, <paramref name="numHeads"/> * <paramref name="vHeadDim"/>].</param>
    /// <param name="kPe">K_pe [<paramref name="seqKv"/>, <paramref name="qkRopeHeadDim"/>] — shared across heads.</param>
    /// <param name="output">Output [<paramref name="seqQ"/>, <paramref name="numHeads"/> * <paramref name="vHeadDim"/>].</param>
    /// <param name="seqQ">Number of query tokens.</param>
    /// <param name="seqKv">Number of cached KV positions (≥ <paramref name="seqQ"/> on prefill, ≥ 1 on decode).</param>
    /// <param name="numHeads">Number of attention heads (MLA has no GQA; numHeads = numKvHeads).</param>
    /// <param name="qkNopeHeadDim">Non-rope Q·K sub-dim per head.</param>
    /// <param name="qkRopeHeadDim">Rope Q·K sub-dim per head (must be even, RoPE-applied to Q already).</param>
    /// <param name="vHeadDim">V head dim — output uses this width per head.</param>
    /// <param name="positionOffset">Causal position offset; query <c>tq</c> sits at absolute position <c>positionOffset + tq</c>.</param>
    /// <param name="scale">Pre-multiplied softmax scale: <c>(YaRN mscale²) / sqrt(qkNope + qkRope)</c>.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer q, VulkanDevice.Buffer kNope, VulkanDevice.Buffer v,
        VulkanDevice.Buffer kPe, VulkanDevice.Buffer output,
        int seqQ, int seqKv, int numHeads,
        int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
        int positionOffset, float scale)
    {
        if (seqQ <= 0) throw new ArgumentOutOfRangeException(nameof(seqQ));
        if (seqKv <= 0) throw new ArgumentOutOfRangeException(nameof(seqKv));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (qkNopeHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(qkNopeHeadDim));
        if (qkRopeHeadDim < 0) throw new ArgumentOutOfRangeException(nameof(qkRopeHeadDim));
        if (vHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(vHeadDim));
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));
        int qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;
        if (qkHeadDim > MaxQkHeadDim)
            throw new ArgumentException(
                $"qkNopeHeadDim + qkRopeHeadDim must be ≤ {MaxQkHeadDim}, got {qkHeadDim}.", nameof(qkNopeHeadDim));
        if (vHeadDim > MaxVHeadDim)
            throw new ArgumentException($"vHeadDim must be ≤ {MaxVHeadDim}, got {vHeadDim}.", nameof(vHeadDim));

        long qBytes = (long)seqQ * numHeads * qkHeadDim * sizeof(float);
        long kNopeBytes = (long)seqKv * numHeads * qkNopeHeadDim * sizeof(float);
        long vBytes = (long)seqKv * numHeads * vHeadDim * sizeof(float);
        long kPeBytes = (long)seqKv * qkRopeHeadDim * sizeof(float);
        long outBytes = (long)seqQ * numHeads * vHeadDim * sizeof(float);
        if (q.Size < qBytes) throw new ArgumentException("Q buffer too small.", nameof(q));
        if (kNope.Size < kNopeBytes) throw new ArgumentException("K_nope buffer too small.", nameof(kNope));
        if (v.Size < vBytes) throw new ArgumentException("V buffer too small.", nameof(v));
        if (kPe.Size < kPeBytes) throw new ArgumentException("K_pe buffer too small.", nameof(kPe));
        if (output.Size < outBytes) throw new ArgumentException("Output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[5]
        {
            q.Handle, kNope.Handle, v.Handle, kPe.Handle, output.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)seqQ);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)seqKv);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)numHeads);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)qkNopeHeadDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[16..], (uint)qkRopeHeadDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[20..], (uint)vHeadDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[24..], (uint)positionOffset);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pcBytes[28..], scale);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // One workgroup per (tq, hq) pair.
        uint groupCount = (uint)seqQ * (uint)numHeads;
        VulkanApi.vkCmdDispatch(cmdBuf, groupCount, 1, 1);
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
