using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q8_0 token-embedding gather + dequantize (issue #352): gathers one
/// embedding row per input token from a device-resident RAW Q8_0 table,
/// dequantizing on the fly into the F32 hidden-state buffer.
/// </summary>
/// <remarks>
/// <para>
/// This is the device-resident replacement for the <c>vkCmdCopyBuffer</c> row
/// gather used when the embedding table was widened to F32 at upload. Keeping
/// the table in its Q8_0 byte layout (34 bytes per 32 weights vs 128) saves
/// ~3.76x its device bytes — ~772 MB on Llama-3.2-1B-Instruct-Q8_0.
/// </para>
/// <para>
/// Layout matches <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeQ8_0Scalar</c> and
/// llama.cpp's <c>block_q8_0</c>: fp16 scale + 32 int8 per 34-byte block, row
/// stride <c>(hidden/32) * 34</c>. Output is bit-identical to the CPU dequant of
/// the same rows (<c>precise</c>-qualified single multiply, same op order), so
/// the forward pass sees exactly the bytes the F32-upload path produced.
/// </para>
/// </remarks>
public sealed class Q8_0EmbedGatherF32Kernel : IDisposable
{
    /// <summary>Q8_0 block: 2-byte fp16 scale + 32 int8 = 34 bytes.</summary>
    public const int Q8_0BlockBytes = QuantFormat.Q8_0BlockBytes;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = QuantFormat.LegacyGroupSize;

    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = 4 * sizeof(uint); // nTokens, hidden, blocksPerRow, vocabSize

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private Q8_0EmbedGatherF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Loads <c>q8_0_embed_gather_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static Q8_0EmbedGatherF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "q8_0_embed_gather_f32.spv");
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
        return new Q8_0EmbedGatherF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>
    /// <see cref="Create"/> that returns <c>null</c> when the SPIR-V blob is
    /// absent, so callers can fall back to the F32 gather path.
    /// </summary>
    public static Q8_0EmbedGatherF32Kernel? TryCreate(VulkanDevice device, string? spvDir)
    {
        if (spvDir is null) return null;
        if (!File.Exists(Path.Combine(spvDir, "q8_0_embed_gather_f32.spv"))) return null;
        return Create(device, spvDir);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer table, VulkanDevice.Buffer ids, VulkanDevice.Buffer dst,
        int nTokens, int hidden, int vocabSize)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, table, ids, dst, nTokens, hidden, vocabSize);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the gather+dequant. <paramref name="ids"/> holds
    /// <paramref name="nTokens"/> int32 token ids; <paramref name="dst"/>
    /// receives <c>nTokens * hidden</c> floats.
    /// </summary>
    public unsafe void Record(
        nint cmdBuf, VulkanDevice.Buffer table, VulkanDevice.Buffer ids, VulkanDevice.Buffer dst,
        int nTokens, int hidden, int vocabSize)
    {
        if (nTokens <= 0) throw new ArgumentOutOfRangeException(nameof(nTokens));
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));
        if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (hidden % Q8_0GroupSize != 0)
            throw new ArgumentException(
                $"Q8_0 embedding rows need hidden % {Q8_0GroupSize} == 0, got {hidden}.", nameof(hidden));

        int blocksPerRow = hidden / Q8_0GroupSize;
        long tableMin = (long)vocabSize * blocksPerRow * Q8_0BlockBytes;
        if (table.Size < tableMin)
            throw new ArgumentException($"Embedding table too small: need >= {tableMin} bytes.", nameof(table));
        if (ids.Size < (long)nTokens * sizeof(int))
            throw new ArgumentException("Token-id buffer too small.", nameof(ids));
        if (dst.Size < (long)nTokens * hidden * sizeof(float))
            throw new ArgumentException("Destination buffer too small.", nameof(dst));

        Span<nint> buffers = stackalloc nint[3] { table.Handle, ids.Handle, dst.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4]
        {
            (uint)nTokens, (uint)hidden, (uint)blocksPerRow, (uint)vocabSize,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupsX = (uint)((hidden + WorkgroupSize - 1) / WorkgroupSize);
        VulkanApi.vkCmdDispatch(cmdBuf, groupsX, (uint)nTokens, 1);
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
