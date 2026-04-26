using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE broadcast — expands per-token <c>NormOutput</c> into the per-(token,
/// slot) expanded input that feeds the indexed expert matmul:
/// <code>
///     out[n, h] = in[n / topK, h]
/// </code>
/// where the output is <c>[seqLen * topK, hiddenSize]</c> and the input is
/// <c>[seqLen, hiddenSize]</c>. Replaces the
/// <c>seqLen × topK</c> loop of <c>vkCmdCopyBuffer</c> regions in the MoE
/// forward path with a single compute dispatch — one transfer→compute
/// barrier pair drops out and the per-region command-recording overhead
/// disappears at prefill.
/// </summary>
/// <remarks>
/// Math is bit-exact wrt the copy loop it replaces — no FP arithmetic, just
/// a strided F32 load/store. Tolerance for the parity tests is therefore
/// zero (bit equality).
/// </remarks>
public sealed class MoeBroadcastF32Kernel : IDisposable
{
    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    // seqLen, topK, hidden (all u32)
    private const int PushConstantBytes = 3 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeBroadcastF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 2);
    }

    /// <summary>Loads <c>moe_broadcast_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeBroadcastF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_broadcast_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
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
        return new MoeBroadcastF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int seqLen, int topK, int hidden)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, input, output, seqLen, topK, hidden);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the broadcast dispatch into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="input">Per-token input [<paramref name="seqLen"/> × <paramref name="hidden"/>] F32 row-major.</param>
    /// <param name="output">Per-(token, slot) expanded output [<paramref name="seqLen"/> × <paramref name="topK"/> × <paramref name="hidden"/>] F32 row-major; fully overwritten.</param>
    /// <param name="seqLen">Number of source tokens.</param>
    /// <param name="topK">Per-token replication factor (1 ≤ topK).</param>
    /// <param name="hidden">Per-token feature dim.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int seqLen, int topK, int hidden)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));

        long inBytes = (long)seqLen * hidden * sizeof(float);
        long outBytes = (long)seqLen * topK * hidden * sizeof(float);
        if (input.Size < inBytes) throw new ArgumentException("input buffer too small.", nameof(input));
        if (output.Size < outBytes) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[2] { input.Handle, output.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)topK);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)hidden);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        long expandedRows = (long)seqLen * topK;
        uint groupX = (uint)((hidden + WorkgroupX - 1) / WorkgroupX);
        uint groupY = (uint)((expandedRows + WorkgroupY - 1) / WorkgroupY);
        VulkanApi.vkCmdDispatch(cmdBuf, groupX, groupY, 1);
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
