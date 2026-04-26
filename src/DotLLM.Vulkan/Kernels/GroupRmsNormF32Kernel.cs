using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Group RMS Normalization for Mamba-2 / NemotronH SSM <c>ssm_norm</c>.
/// Splits each token's <c>[dInner]</c> row into <c>nGroup</c> contiguous
/// groups of <c>groupDim</c> (where <c>dInner = nGroup * groupDim</c>),
/// applies RMSNorm independently per group, and scales by the matching
/// per-group slice of the norm weight:
/// <code>
///   for g in 0..nGroup:
///       base    = t * dInner + g * groupDim
///       sumSq   = sum_{i=0..groupDim} data[base+i]^2
///       rinv    = 1 / sqrt(sumSq / groupDim + eps)
///       data[base+i] = data[base+i] * rinv * weight[g*groupDim + i]
/// </code>
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the CPU reference loop at step 11 (<c>9. Group RMSNorm</c>) of
/// <c>NemotronHTransformerModel.ForwardSsmBody</c>. The data buffer is
/// updated in place; the weight buffer is read-only.
/// </para>
/// <para>
/// Dispatch is one workgroup per <c>(token, group)</c> pair —
/// <c>groupCount = (nGroup, seqLen, 1)</c>. Each workgroup runs a
/// shared-memory tree reduction to compute its own sum-of-squares (the same
/// pattern as <see cref="RmsNormF32Kernel"/>), then every thread normalises
/// its share of the <c>groupDim</c> elements. The per-group weight offset
/// (<c>g * groupDim</c>) is applied inside the shader.
/// </para>
/// </remarks>
public sealed class GroupRmsNormF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    // seqLen, nGroup, groupDim, eps
    private const int PushConstantBytes = 3 * sizeof(uint) + sizeof(float);
    private const int BufferCount = 2;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private GroupRmsNormF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: BufferCount);
    }

    /// <summary>Loads <c>group_rmsnorm_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static GroupRmsNormF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "group_rmsnorm_f32.spv");
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
        return new GroupRmsNormF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer data, VulkanDevice.Buffer weight,
        int seqLen, int nGroup, int groupDim, float eps)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, data, weight, seqLen, nGroup, groupDim, eps);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the group-RMSNorm dispatch into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="data">
    /// FP32 read-write buffer of shape <c>[seqLen, nGroup * groupDim]</c>
    /// row-major. Updated in place.
    /// </param>
    /// <param name="weight">
    /// FP32 read-only per-group norm weight of length
    /// <c>nGroup * groupDim</c>. Group <c>g</c>'s slice is
    /// <c>[g * groupDim, (g + 1) * groupDim)</c>.
    /// </param>
    /// <param name="seqLen">Number of tokens (rows). 0 is a no-op.</param>
    /// <param name="nGroup">Number of groups per row.</param>
    /// <param name="groupDim">Group width — number of features normalised together.</param>
    /// <param name="eps">Epsilon under the square root (typical: 1e-5 or 1e-6).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer data, VulkanDevice.Buffer weight,
        int seqLen, int nGroup, int groupDim, float eps)
    {
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nGroup <= 0) throw new ArgumentOutOfRangeException(nameof(nGroup));
        if (groupDim <= 0) throw new ArgumentOutOfRangeException(nameof(groupDim));
        if (seqLen == 0) return; // no-op

        long dInner = (long)nGroup * groupDim;
        long dataBytes = (long)seqLen * dInner * sizeof(float);
        long weightBytes = dInner * sizeof(float);
        if (data.Size < dataBytes) throw new ArgumentException("data buffer too small.", nameof(data));
        if (weight.Size < weightBytes) throw new ArgumentException("weight buffer too small.", nameof(weight));

        Span<nint> buffers = stackalloc nint[BufferCount] { data.Handle, weight.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes,        (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..],   (uint)nGroup);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..],   (uint)groupDim);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pcBytes[12..],  eps);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // One workgroup per (group, token) — both axes are independent.
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)nGroup, (uint)seqLen, 1);
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
