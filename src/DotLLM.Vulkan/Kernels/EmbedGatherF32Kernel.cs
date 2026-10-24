using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Token-embedding gather + dequantize from a table kept in its packed GGUF layout (#352, #1174): one row per input token is
/// dequantized into the F32 hidden state, so the vocab x hidden table never has to be widened to F32 on the device.
/// </summary>
/// <remarks>
/// <para>
/// One class for every format that has a gather shader: Q8_0 (<c>q8_0_embed_gather_f32</c>), PQ2_0 (<c>pq2_0_embed_gather_f32</c>),
/// Q4_K and Q6_K (<c>q4_k/q6_k_embed_gather_f32</c>). All four share bindings (table, ids, dst), push constants
/// (nTokens, hidden, blocksPerRow, vocabSize) and the dispatch (<c>ceil(hidden / 256)</c> x nTokens workgroups of 256), and each
/// shader reproduces the CPU dequant bit for bit.
/// </para>
/// <para>
/// A dense matmul kernel for a format is NOT a gather: <see cref="IsSupported"/> is the only test of whether a table may stay packed.
/// </para>
/// </remarks>
public sealed class EmbedGatherF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = 4 * sizeof(uint); // nTokens, hidden, blocksPerRow, vocabSize

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly int _groupSize;
    private bool _disposed;

    private EmbedGatherF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool,
        QuantizationType format, int groupSize)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
        Format = format;
        _groupSize = groupSize;
    }

    /// <summary>The table format this kernel gathers from.</summary>
    public QuantizationType Format { get; }

    /// <summary>The shader for <paramref name="format"/> and its elements per block, or null when the format has no gather.</summary>
    private static (string Spv, int GroupSize)? ShaderFor(QuantizationType format) => format switch
    {
        QuantizationType.Q8_0 => ("q8_0_embed_gather_f32.spv", 32),
        QuantizationType.PQ2_0 => ("pq2_0_embed_gather_f32.spv", 128),
        QuantizationType.Q4_K => ("q4_k_embed_gather_f32.spv", 256),
        QuantizationType.Q6_K => ("q6_k_embed_gather_f32.spv", 256),
        _ => null,
    };

    /// <summary>
    /// True when a <paramref name="format"/> table with rows of <paramref name="hidden"/> elements can be gathered packed:
    /// the format has a gather shader, it is present in <paramref name="spvDir"/>, and rows are whole blocks.
    /// <c>DOTLLM_VULKAN_DISABLE_EMBED_RESIDENT=1</c> answers false (the widened-table A/B arm).
    /// </summary>
    /// <param name="format">Source quantization of the embedding table.</param>
    /// <param name="hidden">Row length (hidden size).</param>
    /// <param name="spvDir">SPIR-V directory, or null.</param>
    /// <returns>Whether <see cref="Create"/> will succeed for this table.</returns>
    public static bool IsSupported(QuantizationType format, int hidden, string? spvDir)
        => ShaderFor(format) is { } s
           && hidden % s.GroupSize == 0
           && spvDir is not null
           && Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_EMBED_RESIDENT") != "1"
           && File.Exists(Path.Combine(spvDir, s.Spv));

    /// <summary>Loads the gather shader for <paramref name="format"/>.</summary>
    /// <param name="device">Device to create the pipeline on.</param>
    /// <param name="spvDir">SPIR-V directory.</param>
    /// <param name="format">Table format; must have a gather shader.</param>
    /// <returns>The kernel.</returns>
    /// <exception cref="NotSupportedException">The format has no gather shader.</exception>
    /// <exception cref="FileNotFoundException">The shader is missing from <paramref name="spvDir"/>.</exception>
    public static EmbedGatherF32Kernel Create(VulkanDevice device, string spvDir, QuantizationType format)
    {
        var (spv, groupSize) = ShaderFor(format)
            ?? throw new NotSupportedException($"No Vulkan embedding gather for {format}.");
        string path = Path.Combine(spvDir, spv);
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
            pipeline = module.CreateComputePipeline(entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new EmbedGatherF32Kernel(device, module, pipeline, pool, format, groupSize);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch of <see cref="Record"/>; used by unit tests.</summary>
    /// <param name="table">Packed embedding table.</param>
    /// <param name="ids">Token ids (int32).</param>
    /// <param name="dst">Receives <c>nTokens * hidden</c> floats.</param>
    /// <param name="nTokens">Number of tokens.</param>
    /// <param name="hidden">Row length.</param>
    /// <param name="vocabSize">Table row count.</param>
    public void Launch(VulkanDevice.Buffer table, VulkanDevice.Buffer ids, VulkanDevice.Buffer dst, int nTokens, int hidden, int vocabSize)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, table, ids, dst, nTokens, hidden, vocabSize);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the gather + dequant of <paramref name="nTokens"/> rows into <paramref name="dst"/>.</summary>
    /// <param name="cmdBuf">Command buffer being recorded.</param>
    /// <param name="table">Packed embedding table.</param>
    /// <param name="ids">Token ids (int32).</param>
    /// <param name="dst">Receives <c>nTokens * hidden</c> floats.</param>
    /// <param name="nTokens">Number of tokens.</param>
    /// <param name="hidden">Row length.</param>
    /// <param name="vocabSize">Table row count.</param>
    public unsafe void Record(
        nint cmdBuf, VulkanDevice.Buffer table, VulkanDevice.Buffer ids, VulkanDevice.Buffer dst,
        int nTokens, int hidden, int vocabSize)
    {
        if (nTokens <= 0) throw new ArgumentOutOfRangeException(nameof(nTokens));
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));
        if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (hidden % _groupSize != 0)
            throw new ArgumentException($"{Format} embedding rows need hidden % {_groupSize} == 0, got {hidden}.", nameof(hidden));

        int blocksPerRow = hidden / _groupSize;
        long tableMin = Dequantize.RowByteSize(hidden, Format) * vocabSize;
        if (table.Size < tableMin)
            throw new ArgumentException($"Embedding table too small: need >= {tableMin} bytes.", nameof(table));
        if (ids.Size < (long)nTokens * sizeof(int))
            throw new ArgumentException("Token-id buffer too small.", nameof(ids));
        if (dst.Size < (long)nTokens * hidden * sizeof(float))
            throw new ArgumentException("Destination buffer too small.", nameof(dst));

        Span<nint> buffers = stackalloc nint[3] { table.Handle, ids.Handle, dst.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4] { (uint)nTokens, (uint)hidden, (uint)blocksPerRow, (uint)vocabSize };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);

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
