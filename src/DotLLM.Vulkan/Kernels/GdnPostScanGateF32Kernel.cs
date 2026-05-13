using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Per-head RMSNorm × silu(z) gating fused into one dispatch — step 6 of the
/// Qwen3MoeHybrid GDN token-mixing forward pass. Mirrors the CPU sequence in
/// <c>ForwardGdnBody</c> (RMSNorm in place with <c>ssm_norm_weight</c>, then
/// element-wise multiply by <c>silu(z)</c>).
/// </summary>
/// <remarks>
/// <para>
/// The matching shader <c>gdn_post_scan_gate_f32.comp</c> dispatches one
/// workgroup per <c>(vh, t)</c> pair with <c>local_size_x = 128</c>, computing
/// the RMSNorm reduction in shared memory and folding the silu(z) gate into a
/// single read-modify-write pass over the <c>dState</c> elements of that head.
/// RMSNorm here uses the standard transformer convention <c>sqrt(mean + eps)</c>
/// — eps INSIDE the sqrt — unlike the L2 norm in
/// <see cref="GdnL2NormalizeHeadsF32Kernel"/>.
/// </para>
/// </remarks>
public sealed class GdnPostScanGateF32Kernel : IDisposable
{
    // seq_len, n_v_head, d_state (u32), eps (f32)
    private const int PushConstantBytes = 3 * sizeof(uint) + sizeof(float);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private GdnPostScanGateF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>gdn_post_scan_gate_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static GdnPostScanGateF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "gdn_post_scan_gate_f32.spv");
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
        return new GdnPostScanGateF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer gdnOut, VulkanDevice.Buffer z, VulkanDevice.Buffer ssmNormWeight,
        int seqLen, int nVHead, int dState, float eps)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, gdnOut, z, ssmNormWeight, seqLen, nVHead, dState, eps);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the fused per-head RMSNorm × silu(z) gate dispatch.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="gdnOut">F32 <c>[seqLen, nVHead, dState]</c> — RW (normalised and gated in place).</param>
    /// <param name="z">F32 <c>[seqLen, nVHead, dState]</c> — the gate projection (read-only).</param>
    /// <param name="ssmNormWeight">F32 <c>[dState]</c> — RMSNorm gain shared across all heads.</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="nVHead">Number of GDN value heads.</param>
    /// <param name="dState">Per-head state width.</param>
    /// <param name="eps">RMSNorm stabilising constant (eps INSIDE the sqrt).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer gdnOut, VulkanDevice.Buffer z, VulkanDevice.Buffer ssmNormWeight,
        int seqLen, int nVHead, int dState, float eps)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nVHead <= 0) throw new ArgumentOutOfRangeException(nameof(nVHead));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));

        long bytes = (long)seqLen * nVHead * dState * sizeof(float);
        if (gdnOut.Size < bytes) throw new ArgumentException("gdnOut buffer too small.", nameof(gdnOut));
        if (z.Size < bytes) throw new ArgumentException("z buffer too small.", nameof(z));
        long wBytes = (long)dState * sizeof(float);
        if (ssmNormWeight.Size < wBytes) throw new ArgumentException("ssmNormWeight buffer too small.", nameof(ssmNormWeight));

        Span<nint> buffers = stackalloc nint[3] { gdnOut.Handle, z.Handle, ssmNormWeight.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)nVHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)dState);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pcBytes[12..], eps);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)nVHead, (uint)seqLen, 1);
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
