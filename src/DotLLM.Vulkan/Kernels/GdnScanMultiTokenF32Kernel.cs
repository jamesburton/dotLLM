using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Multi-token Gated DeltaNet (GDN) scan for Qwen3MoeHybrid — binds the full
/// whole-sequence q/k/v/g/beta/output buffers and walks the <c>seqLen</c>
/// dimension INSIDE the shader, mutating the per-sequence state matrix in
/// place between tokens.
/// </summary>
/// <remarks>
/// <para>
/// Replaces the host-driven <c>for t in 0..seqLen { CopyTokenRow; GdnScanStep }</c>
/// loop in <see cref="VulkanQwen3MoeHybridTransformerModel.RecordGdnLayer"/>:
/// the per-token shader required six D2D copies per token before each
/// dispatch, growing kernel launches as O(seqLen). The multi-token kernel
/// collapses the entire scan into a single dispatch per layer.
/// </para>
/// <para>
/// Bit-parity vs the per-token <see cref="GdnScanStepF32Kernel"/> and the CPU
/// reference is preserved: the t-loop runs in order with a barrier at the end
/// of each token, and each phase's reduction order is identical.
/// </para>
/// </remarks>
public sealed class GdnScanMultiTokenF32Kernel : IDisposable
{
    // n_v_head, n_k_head, d_state, v_heads_per_k_head, seq_len (all u32)
    private const int PushConstantBytes = 5 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private GdnScanMultiTokenF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 7);
    }

    /// <summary>Loads <c>gdn_scan_multi_token_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static GdnScanMultiTokenF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "gdn_scan_multi_token_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[7];
            for (int i = 0; i < 7; i++) bindings[i] = new VkDescriptorBinding((uint)i);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 7);
        return new GdnScanMultiTokenF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer state, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v,
        VulkanDevice.Buffer g, VulkanDevice.Buffer beta, VulkanDevice.Buffer output,
        int seqLen, int nVHead, int nKHead, int dState)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, state, q, k, v, g, beta, output, seqLen, nVHead, nKHead, dState);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the multi-token GDN scan dispatch into <paramref name="cmdBuf"/>.
    /// All buffers are full-sequence views (<c>q/k</c> shape <c>[seqLen, nKHead, dState]</c>,
    /// <c>v/output</c> shape <c>[seqLen, nVHead, dState]</c>, <c>g/beta</c> shape
    /// <c>[seqLen, nVHead]</c>); the per-token slicing happens inside the shader.
    /// </summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer state, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v,
        VulkanDevice.Buffer g, VulkanDevice.Buffer beta, VulkanDevice.Buffer output,
        int seqLen, int nVHead, int nKHead, int dState)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nVHead <= 0) throw new ArgumentOutOfRangeException(nameof(nVHead));
        if (nKHead <= 0) throw new ArgumentOutOfRangeException(nameof(nKHead));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (nVHead % nKHead != 0)
            throw new ArgumentException(
                $"nVHead ({nVHead}) must be a multiple of nKHead ({nKHead}).");

        int vHeadsPerKHead = nVHead / nKHead;

        Span<nint> buffers = stackalloc nint[7]
        {
            state.Handle, q.Handle, k.Handle, v.Handle,
            g.Handle, beta.Handle, output.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)nVHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)nKHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)dState);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)vHeadsPerKHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[16..], (uint)seqLen);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)nVHead, 1, 1);
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
