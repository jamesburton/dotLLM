using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Mamba-2 / Mamba-3 post-scan per-head scalar skip add (the "D" term):
/// <code>
///     y[t, h*headDim + i] += x[t, h*headDim + i] * D[h]
/// </code>
/// <c>D</c> is a per-head scalar broadcast across <c>headDim</c>. This
/// is step #9 of the SSM body — after the selective scan produces
/// <c>y</c>, the learned skip-connection scaled by the per-head
/// <c>D</c> vector is added in.
/// </summary>
/// <remarks>
/// <para>
/// Memory layout mirrors the CPU reference in
/// <see cref="DotLLM.Models.Architectures.NemotronHTransformerModel"/>
/// (<c>ForwardSsmBody</c>):
/// </para>
/// <list type="bullet">
///   <item><description>
///     <b>y</b> shape <c>[seqLen, nHead*headDim]</c>, row-major F32,
///     read+write (in-place).
///   </description></item>
///   <item><description>
///     <b>x</b> shape <c>[seqLen, nHead*headDim]</c>, row-major F32, read.
///   </description></item>
///   <item><description>
///     <b>d</b> shape <c>[nHead]</c>, F32, read.
///   </description></item>
/// </list>
/// <para>
/// Dispatch is 2D, one thread per <c>(t, h*headDim+i)</c> output cell,
/// with a (16, 16) workgroup. Mirrors the size + style of
/// <see cref="BiasAddF32Kernel"/> — also a per-row broadcast op, just
/// with an extra "head bucket" indirection on the broadcast index.
/// </para>
/// </remarks>
public sealed class SsmDSkipF32Kernel : IDisposable
{
    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    // seqLen, nHead, headDim (all u32)
    private const int PushConstantBytes = 3 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private SsmDSkipF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>ssm_d_skip_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static SsmDSkipF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "ssm_d_skip_f32.spv");
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
        return new SsmDSkipF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer y, VulkanDevice.Buffer x, VulkanDevice.Buffer d,
        int seqLen, int nHead, int headDim)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, y, x, d, seqLen, nHead, headDim);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the in-place per-head skip add.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="y">
    /// F32 in-place output buffer, shape <c>[seqLen, nHead*headDim]</c> row-major.
    /// </param>
    /// <param name="x">
    /// F32 input buffer (the same activations that were fed into the scan),
    /// shape <c>[seqLen, nHead*headDim]</c> row-major.
    /// </param>
    /// <param name="d">F32 per-head scalar buffer, length <paramref name="nHead"/>.</param>
    /// <param name="seqLen">Number of time steps.</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="headDim">Per-head channel width; <c>dInner = nHead * headDim</c>.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer y, VulkanDevice.Buffer x, VulkanDevice.Buffer d,
        int seqLen, int nHead, int headDim)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));

        long dInner = (long)nHead * headDim;
        long ioBytes = (long)seqLen * dInner * sizeof(float);
        long dBytes = (long)nHead * sizeof(float);
        if (y.Size < ioBytes) throw new ArgumentException("y buffer too small.", nameof(y));
        if (x.Size < ioBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (d.Size < dBytes) throw new ArgumentException("d buffer too small.", nameof(d));

        Span<nint> buffers = stackalloc nint[3] { y.Handle, x.Handle, d.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)nHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)headDim);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        long cols = dInner;
        uint groupX = (uint)((cols + WorkgroupX - 1) / WorkgroupX);
        uint groupY = (uint)((seqLen + WorkgroupY - 1) / WorkgroupY);
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
