using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// SSM split-xBC — extracts the <c>[x | B | C]</c> row-major projection
/// output produced by the SSM <c>conv1d_causal + silu</c> stage into three
/// separately bound destination buffers, in a single fused 2D dispatch.
/// Replaces the per-token loop of THREE <c>vkCmdCopyBuffer</c> regions
/// per token (one each for x, B, C) in
/// <c>VulkanNemotronHTransformerModel.RecordSsmLayer</c> step 7 — dispatch
/// count for this step drops from <c>3 · seqLen</c> to 1 per SSM layer
/// and the transfer↔compute stage transitions around the loop disappear.
/// </summary>
/// <remarks>
/// The math is a strided F32 load/store with no FP arithmetic, so the
/// output is bit-exact wrt the per-token-copy loop it replaces — parity
/// tolerance is therefore zero (exact equality).
/// </remarks>
public sealed class SsmSplitXbcF32Kernel : IDisposable
{
    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    // seqLen, dInner, bcDim, convDim (all u32)
    private const int PushConstantBytes = 4 * sizeof(uint);
    private const int BufferCount = 4; // xbc, x, b, c

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private SsmSplitXbcF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: BufferCount);
    }

    /// <summary>Loads <c>ssm_split_xbc_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static SsmSplitXbcF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "ssm_split_xbc_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[BufferCount];
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: BufferCount);
        return new SsmSplitXbcF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer xbc,
        VulkanDevice.Buffer x, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        int seqLen, int dInner, int bcDim)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, xbc, x, b, c, seqLen, dInner, bcDim);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the split dispatch into <paramref name="cmdBuf"/>. Each token's
    /// row of <paramref name="xbc"/> is laid out as <c>[x | B | C]</c> with
    /// widths <c>dInner</c>, <c>bcDim</c>, <c>bcDim</c> respectively
    /// (<c>convDim = dInner + 2*bcDim</c>); the kernel writes the three
    /// per-token slices into <paramref name="x"/>, <paramref name="b"/>,
    /// <paramref name="c"/> in one dispatch.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="xbc">Source [<paramref name="seqLen"/> × convDim] F32 row-major (convDim = <paramref name="dInner"/> + 2 · <paramref name="bcDim"/>).</param>
    /// <param name="x">Destination x [<paramref name="seqLen"/> × <paramref name="dInner"/>] F32 row-major; fully overwritten.</param>
    /// <param name="b">Destination B [<paramref name="seqLen"/> × <paramref name="bcDim"/>] F32 row-major; fully overwritten.</param>
    /// <param name="c">Destination C [<paramref name="seqLen"/> × <paramref name="bcDim"/>] F32 row-major; fully overwritten.</param>
    /// <param name="seqLen">Number of source tokens (rows).</param>
    /// <param name="dInner">Width of the x slice.</param>
    /// <param name="bcDim">Width of each of the B and C slices.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer xbc,
        VulkanDevice.Buffer x, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        int seqLen, int dInner, int bcDim)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (dInner <= 0) throw new ArgumentOutOfRangeException(nameof(dInner));
        if (bcDim <= 0) throw new ArgumentOutOfRangeException(nameof(bcDim));

        long convDim = (long)dInner + 2L * bcDim;
        long xbcBytes = (long)seqLen * convDim * sizeof(float);
        long xBytes = (long)seqLen * dInner * sizeof(float);
        long bcBytes = (long)seqLen * bcDim * sizeof(float);
        if (xbc.Size < xbcBytes) throw new ArgumentException("xbc buffer too small.", nameof(xbc));
        if (x.Size < xBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (b.Size < bcBytes) throw new ArgumentException("b buffer too small.", nameof(b));
        if (c.Size < bcBytes) throw new ArgumentException("c buffer too small.", nameof(c));

        Span<nint> buffers = stackalloc nint[BufferCount] { xbc.Handle, x.Handle, b.Handle, c.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)dInner);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)bcDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)convDim);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // 2D grid + branch axis (z = 3): x/B/C all share the dispatch. Width
        // axis covers max(dInner, bcDim); per-thread early-return handles the
        // ragged dInner ≠ bcDim case.
        int widest = Math.Max(dInner, bcDim);
        uint groupX = (uint)((widest + WorkgroupX - 1) / WorkgroupX);
        uint groupY = (uint)((seqLen + WorkgroupY - 1) / WorkgroupY);
        VulkanApi.vkCmdDispatch(cmdBuf, groupX, groupY, 3);
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
