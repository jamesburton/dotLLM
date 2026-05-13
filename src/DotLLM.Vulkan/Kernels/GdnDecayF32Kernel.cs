using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Fused per-(token, head) decay computation for Qwen3MoeHybrid GDN token
/// mixing: <c>g[t, vh] = exp(softplus(alpha[t, vh] + dt_bias[vh]) * A[vh])</c>,
/// computed in place on the alpha buffer.
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the CUDA <c>gdn_decay_f32</c> kernel in
/// <c>native/kernels/gated_delta_net_scan.cu</c> and replaces the
/// D2H/host-compute/H2D roundtrip in
/// <c>VulkanQwen3MoeHybridTransformerModel.ComputeDecayAndBetaOnHost</c>.
/// </para>
/// <para>
/// Softplus is computed as <c>log(1 + exp(x))</c> with NO numerical guard,
/// matching the CPU reference exactly — large inputs saturate to <c>+inf</c>,
/// which propagates through to <c>exp(sp * A)</c> as <c>0</c> when <c>A</c> is
/// negative (the empirically-observed sign). Bit-parity tolerance vs the CPU
/// reference is ≤4 ULP, the standard transcendental drift on Vulkan.
/// </para>
/// </remarks>
public sealed class GdnDecayF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    // seq_len, n_v_head (u32)
    private const int PushConstantBytes = 2 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private GdnDecayF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>gdn_decay_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static GdnDecayF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "gdn_decay_f32.spv");
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
        return new GdnDecayF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer alphaBuf, VulkanDevice.Buffer dtBias, VulkanDevice.Buffer a,
        int seqLen, int nVHead)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, alphaBuf, dtBias, a, seqLen, nVHead);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the fused softplus + exp dispatch into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="alphaBuf">F32 <c>[seqLen, nVHead]</c> — in-place, holds the decay <c>g</c> on return.</param>
    /// <param name="dtBias">F32 <c>[nVHead]</c> — additive bias.</param>
    /// <param name="a">F32 <c>[nVHead]</c> — the multiplier inside the final <c>exp</c>.</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="nVHead">Number of value heads.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer alphaBuf, VulkanDevice.Buffer dtBias, VulkanDevice.Buffer a,
        int seqLen, int nVHead)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nVHead <= 0) throw new ArgumentOutOfRangeException(nameof(nVHead));

        long alphaBytes = (long)seqLen * nVHead * sizeof(float);
        long perHeadBytes = (long)nVHead * sizeof(float);
        if (alphaBuf.Size < alphaBytes) throw new ArgumentException("alphaBuf buffer too small.", nameof(alphaBuf));
        if (dtBias.Size < perHeadBytes) throw new ArgumentException("dtBias buffer too small.", nameof(dtBias));
        if (a.Size < perHeadBytes) throw new ArgumentException("a buffer too small.", nameof(a));

        Span<nint> buffers = stackalloc nint[3] { alphaBuf.Handle, dtBias.Handle, a.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)nVHead);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        long total = (long)seqLen * nVHead;
        uint groups = (uint)((total + WorkgroupSize - 1) / WorkgroupSize);
        VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);
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
