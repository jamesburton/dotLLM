using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Packs FP32 K/V rows into an IEEE-754 binary16 KV store (two elements per uint) —
/// the write path of an F16 <see cref="VulkanNemotronHKvCache"/> (issue #437).
/// </summary>
/// <remarks>
/// <para>The F32 cache appends with <c>vkCmdCopyBuffer</c>, which cannot convert, so an
/// F16 cache substitutes this dispatch. The consequence the caller must not get wrong is
/// the barrier: the append is now a SHADER write, so the barrier before attention reads
/// the cache is <c>COMPUTE -> COMPUTE</c>, not <c>TRANSFER -> COMPUTE</c>. A transfer-stage
/// source does not cover a shader write and Vulkan will not diagnose it —
/// <see cref="VulkanNemotronHKvCache.RecordPostUpdateBarrier"/> owns that choice so the
/// three consuming models cannot pick the wrong one.</para>
///
/// <para>Everything is expressed in <i>pairs</i> (one destination uint = two F32 source
/// elements), so the caller's element offsets must be even. The cache guarantees that by
/// requiring an even <c>kvStride</c>.</para>
/// </remarks>
public sealed class KvPackF16Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = 3 * sizeof(uint); // nPairs, srcPairOffset, dstPairOffset

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private KvPackF16Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 2);
    }

    /// <summary>Loads <c>kv_pack_f16.spv</c> from <paramref name="spvDir"/>.</summary>
    public static KvPackF16Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "kv_pack_f16.spv");
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
        return new KvPackF16Kernel(device, module, pipeline, pool);
    }

    /// <summary>Loads the kernel, or returns <c>null</c> when the blob is absent.</summary>
    public static KvPackF16Kernel? TryCreate(VulkanDevice device, string spvDir)
    {
        try { return Create(device, spvDir); }
        catch (FileNotFoundException) { return null; }
    }

    /// <summary>Drops every cached descriptor set; call after buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Records the pack of <paramref name="elementCount"/> FP32 elements starting at
    /// <paramref name="srcElementOffset"/> in <paramref name="src"/> into
    /// <paramref name="dst"/> starting at F16 element <paramref name="dstElementOffset"/>.
    /// All three must be even.
    /// </summary>
    public unsafe void Record(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst,
        long srcElementOffset, long dstElementOffset, long elementCount)
    {
        if (elementCount <= 0) throw new ArgumentOutOfRangeException(nameof(elementCount));
        if ((elementCount & 1) != 0) throw new ArgumentException("elementCount must be even (2 F16 per uint).", nameof(elementCount));
        if ((srcElementOffset & 1) != 0) throw new ArgumentException("srcElementOffset must be even.", nameof(srcElementOffset));
        if ((dstElementOffset & 1) != 0) throw new ArgumentException("dstElementOffset must be even.", nameof(dstElementOffset));

        long nPairs = elementCount / 2;
        if (src.Size < (srcElementOffset + elementCount) * sizeof(float))
            throw new ArgumentException("source buffer too small.", nameof(src));
        if (dst.Size < (dstElementOffset + elementCount) * sizeof(ushort))
            throw new ArgumentException("destination buffer too small.", nameof(dst));

        Span<nint> buffers = stackalloc nint[2] { src.Handle, dst.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)nPairs);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)(srcElementOffset / 2));
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)(dstElementOffset / 2));
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupCount = (uint)((nPairs + WorkgroupSize - 1) / WorkgroupSize);
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
