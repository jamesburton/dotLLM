using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// <see cref="GumbelArgMaxKernel"/> with per-row parameters (#1434): every row of one launch has its own seed, generation index and
/// <c>1 / temperature</c>, because the rows of a continuous-batch decode step belong to different sequences. Per row the result is
/// bit-for-bit <c>GumbelNoise.Sample(logits_row * invTemp_row, seed_row, tokenIndex_row)</c> (see <c>gumbel_argmax_rows_f32.comp</c>).
/// </summary>
public sealed class GumbelArgMaxRowsKernel : IDisposable
{
    private const int PushConstantBytes = 2 * sizeof(uint);

    /// <summary>Bytes of one row's parameters: <c>(seedLo, seedHi, tokenIndex, invTemp bits)</c> as four uints.</summary>
    public const int RowParamBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private GumbelArgMaxRowsKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>The shader file name.</summary>
    public const string SpvFileName = "gumbel_argmax_rows_f32.spv";

    /// <summary>True when the shader is present in <paramref name="spvDir"/>.</summary>
    public static bool IsAvailable(string? spvDir) => spvDir is not null && File.Exists(Path.Combine(spvDir, SpvFileName));

    /// <summary>Loads the shader from <paramref name="spvDir"/>.</summary>
    public static GumbelArgMaxRowsKernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, SpvFileName);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Build the Vulkan project (glslc) first.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            pipeline = module.CreateComputePipeline(entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new GumbelArgMaxRowsKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when any bound buffer has been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Packs one row's parameters into <paramref name="dst"/> (four uints).</summary>
    public static void PackRow(Span<uint> dst, ulong seed, int tokenIndex, float invTemperature)
    {
        dst[0] = (uint)seed;
        dst[1] = (uint)(seed >> 32);
        dst[2] = (uint)tokenIndex;
        dst[3] = BitConverter.SingleToUInt32Bits(invTemperature);
    }

    /// <summary>Synchronous launch of <see cref="Record"/>; used by unit tests. <paramref name="rowParams"/> is already uploaded.</summary>
    public void Launch(VulkanDevice.Buffer logits, VulkanDevice.Buffer ids, VulkanDevice.Buffer rowParams, int rows, int vocab, int rowStride)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        KernelSupport.HostToComputeBarrier(ctx.CommandBuffer);
        Record(ctx.CommandBuffer, logits, ids, rowParams, rows, vocab, rowStride);
        KernelSupport.ComputeToHostBarrier(ctx.CommandBuffer);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the draw of <paramref name="rows"/> rows into <paramref name="ids"/> (one uint per row).</summary>
    /// <param name="cmdBuf">Command buffer being recorded.</param>
    /// <param name="logits">F32 logits, row r at element <c>r * rowStride</c>.</param>
    /// <param name="ids">Receives <paramref name="rows"/> token ids.</param>
    /// <param name="rowParams"><paramref name="rows"/> x <see cref="RowParamBytes"/> bytes written by <see cref="PackRow"/>.</param>
    /// <param name="rows">Rows to draw (one workgroup each).</param>
    /// <param name="vocab">Vocabulary size (entries per row).</param>
    /// <param name="rowStride">Elements between rows.</param>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer logits, VulkanDevice.Buffer ids, VulkanDevice.Buffer rowParams,
        int rows, int vocab, int rowStride)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (vocab <= 0) throw new ArgumentOutOfRangeException(nameof(vocab));
        if (rowStride < vocab) throw new ArgumentOutOfRangeException(nameof(rowStride));
        if (logits.Size < ((long)(rows - 1) * rowStride + vocab) * sizeof(float))
            throw new ArgumentException("Logits buffer too small.", nameof(logits));
        if (ids.Size < (long)rows * sizeof(uint))
            throw new ArgumentException("Id buffer too small.", nameof(ids));
        if (rowParams.Size < (long)rows * RowParamBytes)
            throw new ArgumentException("Row-parameter buffer too small.", nameof(rowParams));

        Span<nint> buffers = stackalloc nint[3] { logits.Handle, ids.Handle, rowParams.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[2] { (uint)vocab, (uint)rowStride };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)rows, 1, 1);
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
