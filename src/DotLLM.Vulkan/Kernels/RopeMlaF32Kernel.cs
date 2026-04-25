using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MLA RoPE (DeepSeek-V2/V3) — rotates the q_pe tail of each Q head and the
/// MQA-shared K_pe in place. Mirrors the post-projection RoPE step in
/// <c>DotLLM.Cpu.Kernels.MlaAttention.Execute</c>.
/// </summary>
/// <remarks>
/// <para>
/// Layout differences from <see cref="RopeF32Kernel"/>:
/// <list type="bullet">
///   <item>Q is <c>[seqLen, numHeads * (qkNope + qkRope)]</c> — only the
///     <c>qkRope</c> tail of each head is rotated; the <c>qkNope</c>
///     prefix is untouched.</item>
///   <item>K_pe is <c>[seqLen, qkRope]</c> — a single rope-K per token,
///     shared across heads (MQA convention). Stored in its own buffer.</item>
/// </list>
/// </para>
/// <para>
/// Pair convention is fixed Norm/interleaved (DeepSeek-V2/V3 use this; the
/// kernel does not expose a NeoX variant).
/// </para>
/// </remarks>
public sealed class RopeMlaF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    // seqLen, numHeads, qkNopeHeadDim, qkRopeHeadDim (u32) + theta (f32)
    private const int PushConstantBytes = 4 * sizeof(uint) + sizeof(float);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private RopeMlaF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>rope_mla_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static RopeMlaF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "rope_mla_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new RopeMlaF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Applies RoPE to the q_pe tail of <paramref name="q"/> and to
    /// <paramref name="kPe"/> in place. Synchronous wrapper around
    /// <see cref="Record"/>.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer q, VulkanDevice.Buffer kPe, VulkanDevice.Buffer positions,
        int seqLen, int numHeads, int qkNopeHeadDim, int qkRopeHeadDim, float theta)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, q, kPe, positions, seqLen, numHeads, qkNopeHeadDim, qkRopeHeadDim, theta);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the MLA RoPE dispatch into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="q">Q buffer [seqLen, numHeads * (qkNope + qkRope)] (FP32).</param>
    /// <param name="kPe">K_pe buffer [seqLen, qkRope] (FP32).</param>
    /// <param name="positions">Token positions [seqLen] (int32).</param>
    /// <param name="seqLen">Number of tokens being rotated.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="qkNopeHeadDim">Non-rope Q·K sub-dim per head (untouched).</param>
    /// <param name="qkRopeHeadDim">Rope Q·K sub-dim per head (must be even, &gt;0).</param>
    /// <param name="theta">RoPE base frequency.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer q, VulkanDevice.Buffer kPe, VulkanDevice.Buffer positions,
        int seqLen, int numHeads, int qkNopeHeadDim, int qkRopeHeadDim, float theta)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (qkNopeHeadDim < 0) throw new ArgumentOutOfRangeException(nameof(qkNopeHeadDim));
        if (qkRopeHeadDim <= 0 || (qkRopeHeadDim & 1) != 0)
            throw new ArgumentOutOfRangeException(
                nameof(qkRopeHeadDim), "qkRopeHeadDim must be a positive even number.");

        int qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;
        long qBytes = (long)seqLen * numHeads * qkHeadDim * sizeof(float);
        long kPeBytes = (long)seqLen * qkRopeHeadDim * sizeof(float);
        long posBytes = (long)seqLen * sizeof(int);
        if (q.Size < qBytes) throw new ArgumentException("Q buffer too small.", nameof(q));
        if (kPe.Size < kPeBytes) throw new ArgumentException("K_pe buffer too small.", nameof(kPe));
        if (positions.Size < posBytes) throw new ArgumentException("Positions buffer too small.", nameof(positions));

        Span<nint> buffers = stackalloc nint[3] { q.Handle, kPe.Handle, positions.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)numHeads);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)qkNopeHeadDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)qkRopeHeadDim);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pcBytes[16..], theta);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // Total work = max(Q pairs, K_pe pairs).
        int halfRope = qkRopeHeadDim / 2;
        int totalQPairs = seqLen * numHeads * halfRope;
        int totalKPePairs = seqLen * halfRope;
        int total = Math.Max(totalQPairs, totalKPePairs);
        uint groupCount = (uint)((total + WorkgroupSize - 1) / WorkgroupSize);
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
