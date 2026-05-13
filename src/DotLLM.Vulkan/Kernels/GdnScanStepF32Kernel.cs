using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Single-token Gated DeltaNet (GDN) scan step for Qwen3MoeHybrid. Mirrors
/// <see cref="DotLLM.Cpu.Kernels.GatedDeltaNetScan.Execute"/> for one token —
/// the host dispatches this kernel <c>seqLen</c> times per layer, advancing
/// the per-token slice pointers on q/k/v/g/beta/output between dispatches.
/// </summary>
/// <remarks>
/// <para>
/// The matching shader <c>gdn_scan_step_f32.comp</c> dispatches one workgroup
/// per value head with <c>local_size_x = 128</c> (one thread per state column),
/// reading and writing the per-sequence state matrix in place. The state
/// buffer carries the recurrence across dispatches; q/k/v/g/beta/output point
/// at the current token's row (the caller does the slice arithmetic).
/// </para>
/// </remarks>
public sealed class GdnScanStepF32Kernel : IDisposable
{
    // n_v_head, n_k_head, d_state, v_heads_per_k_head (all u32)
    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private GdnScanStepF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 7);
    }

    /// <summary>Loads <c>gdn_scan_step_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static GdnScanStepF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "gdn_scan_step_f32.spv");
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
        return new GdnScanStepF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer state, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v,
        VulkanDevice.Buffer g, VulkanDevice.Buffer beta, VulkanDevice.Buffer output,
        int nVHead, int nKHead, int dState)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, state, q, k, v, g, beta, output, nVHead, nKHead, dState);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the GDN scan-step dispatch.
    /// </summary>
    /// <remarks>
    /// q/k/v/g/beta/output point at the per-token slice of their respective
    /// tensors — the caller binds the same buffer across the seqLen-long
    /// dispatch loop and uses an offset-bound view (or passes the same buffer
    /// when seqLen == 1 and the slice is at offset 0). Because Vulkan
    /// descriptor sets bind whole buffers, multi-token slicing is implemented
    /// in this codebase by allocating distinct scratch buffers per token-row
    /// or by issuing a vkCmdCopyBuffer staging — see the model body.
    /// </remarks>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer state, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v,
        VulkanDevice.Buffer g, VulkanDevice.Buffer beta, VulkanDevice.Buffer output,
        int nVHead, int nKHead, int dState)
    {
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
