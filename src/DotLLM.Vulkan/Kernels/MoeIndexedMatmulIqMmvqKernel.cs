using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>Packed IQ expert-bank formats served by <see cref="MoeIndexedMatmulIqMmvqKernel"/> (issue #823).</summary>
public enum MoeIqQuant
{
    /// <summary>IQ3_S: 110 B / 256 elements, needs the shared iq3s grid.</summary>
    IQ3_S,
    /// <summary>IQ4_XS: 136 B / 256 elements.</summary>
    IQ4_XS,
    /// <summary>IQ4_NL: 18 B / 32 elements.</summary>
    IQ4_NL,
}

/// <summary>
/// MoE INDEXED IQ-family MMVQ decode GEMV (issue #823): <c>y[n, m] = Σ_k dequant(bank[indices[n], m, k]) · x[n, k]</c> via integer dp4a
/// against the row-major Q8_1-quantized expanded MoE activations (<see cref="QuantizeQ8_1RowsKernel"/>). One class serves every IQ
/// format; the per-format shader is the dense <c>matmul_iq*_mmvq.comp</c> lane layout with the expert base offset and the 2-D
/// <c>(m, n)</c> grid of the Q5_1 indexed kernel, so an expert row is read exactly as the dense kernel reads a weight row.
/// </summary>
/// <remarks>
/// NOT bit-exact vs a float-in reference (the activation is int8-quantized); validated against the CPU dequant of the same bytes with a
/// Q8_1-quantized x. Created only when the device advertises integer-dot-product support; without it a bank of these types is widened
/// to F32 at upload (see <c>VulkanQwen3MoeMoeUpload.BankStaysPacked</c>). Dispatch: <c>(M, N)</c> workgroups of one wave32 subgroup.
/// </remarks>
public sealed class MoeIndexedMatmulIqMmvqKernel : IDisposable
{
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, numExperts, blocksPerRow

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly VulkanDevice.Buffer? _grid;
    private readonly int _buffersPerSet;
    private bool _disposed;

    /// <summary>The bank format this kernel reads.</summary>
    public MoeIqQuant Quant { get; }

    /// <summary>Bytes per block (super-block for IQ3_S / IQ4_XS).</summary>
    public int BlockBytes { get; }

    /// <summary>Elements per block: 256 (IQ3_S, IQ4_XS) or 32 (IQ4_NL). The bank's K must be a multiple of this.</summary>
    public int GroupSize { get; }

    private MoeIndexedMatmulIqMmvqKernel(MoeIqQuant quant, VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool,
        int buffersPerSet, VulkanDevice.Buffer? grid)
    {
        Quant = quant;
        (BlockBytes, GroupSize) = Describe(quant);
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _buffersPerSet = buffersPerSet;
        _grid = grid;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: buffersPerSet);
    }

    /// <summary>Block bytes and group size for <paramref name="quant"/>.</summary>
    public static (int BlockBytes, int GroupSize) Describe(MoeIqQuant quant) => quant switch
    {
        MoeIqQuant.IQ3_S => (QuantFormat.IQ3_SBlockBytes, QuantFormat.KQuantGroupSize),
        MoeIqQuant.IQ4_XS => (QuantFormat.IQ4_XSBlockBytes, QuantFormat.KQuantGroupSize),
        MoeIqQuant.IQ4_NL => (QuantFormat.IQ4_NLBlockBytes, QuantFormat.LegacyGroupSize),
        _ => throw new ArgumentOutOfRangeException(nameof(quant)),
    };

    /// <summary>The MMVQ shader file (without extension) for <paramref name="quant"/>.</summary>
    public static string ShaderName(MoeIqQuant quant) => quant switch
    {
        MoeIqQuant.IQ3_S => "moe_indexed_matmul_iq3_s_mmvq",
        MoeIqQuant.IQ4_XS => "moe_indexed_matmul_iq4_xs_mmvq",
        MoeIqQuant.IQ4_NL => "moe_indexed_matmul_iq4_nl_mmvq",
        _ => throw new ArgumentOutOfRangeException(nameof(quant)),
    };

    /// <summary>Maps the model-level quant type to the kernel format; <c>null</c> when this kernel family does not serve it.</summary>
    public static MoeIqQuant? FromQuantizationType(QuantizationType qt) => qt switch
    {
        QuantizationType.IQ3_S => MoeIqQuant.IQ3_S,
        QuantizationType.IQ4_XS => MoeIqQuant.IQ4_XS,
        QuantizationType.IQ4_NL => MoeIqQuant.IQ4_NL,
        _ => null,
    };

    /// <summary>
    /// Loads the shader for <paramref name="quant"/> from <paramref name="spvDir"/> and builds the pipeline. Returns <c>null</c> when the
    /// SPV is missing or the device lacks integer-dot-product support. <paramref name="iq3Codebooks"/> supplies the iq3s grid (required
    /// for <see cref="MoeIqQuant.IQ3_S"/>, ignored otherwise; the caller keeps ownership).
    /// </summary>
    internal static MoeIndexedMatmulIqMmvqKernel? TryCreate(VulkanDevice device, string spvDir, MoeIqQuant quant, Iq3Codebooks? iq3Codebooks = null)
    {
        if (!device.HasIntegerDotProduct)
            return null;
        string path = Path.Combine(spvDir, ShaderName(quant) + ".spv");
        if (!File.Exists(path))
            return null;
        VulkanDevice.Buffer? grid = null;
        if (quant == MoeIqQuant.IQ3_S)
        {
            grid = (iq3Codebooks ?? throw new ArgumentNullException(nameof(iq3Codebooks), "IQ3_S needs the shared iq3s grid.")).Iq3SGrid;
        }
        int buffersPerSet = quant == MoeIqQuant.IQ3_S ? 6 : 5;

        uint requiredSubgroupSize = Wave32SubgroupControl.RequiredSubgroupSizeFor(device);
        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[buffersPerSet];
            for (int i = 0; i < buffersPerSet; i++)
                bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindings,
                pushConstantBytes: PushConstantBytes,
                requiredSubgroupSize: requiredSubgroupSize);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)buffersPerSet);
        return new MoeIndexedMatmulIqMmvqKernel(quant, device, module, pipeline, pool, buffersPerSet, grid);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the indexed IQ MMVQ expert matmul into <paramref name="cmdBuf"/>.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="bank">Raw bank of <c>numExperts * M * (K/GroupSize) * BlockBytes</c> bytes.</param>
    /// <param name="xq">Packed-int8 quantized activations, row-major, <c>N*K/4</c> uints.</param>
    /// <param name="xds">Per-block (scale, sum) of the quantized activations, <c>N*K/32</c> vec2.</param>
    /// <param name="indices">int32 per-row expert index [<paramref name="n"/>].</param>
    /// <param name="y">F32 output rows [<paramref name="n"/> * M] row-major.</param>
    /// <param name="m">Per-expert weight row count (output dim).</param>
    /// <param name="k">Per-expert weight column count (multiple of <see cref="GroupSize"/>).</param>
    /// <param name="n">Number of output rows (typically <c>seqLen * topK</c>).</param>
    /// <param name="numExperts">Bank's first axis size.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer bank, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if ((k % GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {GroupSize} for {Quant}, got {k}", nameof(k));

        int blocksPerRow = k / GroupSize;
        long bankBytes = (long)numExperts * m * blocksPerRow * BlockBytes;
        if (bankBytes > uint.MaxValue)
            throw new ArgumentException($"bank is {bankBytes} bytes; the shader addresses it with 32-bit byte offsets.", nameof(bank));
        if (bank.Size < bankBytes) throw new ArgumentException("bank buffer too small.", nameof(bank));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(n, k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(n, k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (indices.Size < (long)n * sizeof(int)) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (y.Size < (long)n * m * sizeof(float)) throw new ArgumentException("y buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[_buffersPerSet];
        buffers[0] = bank.Handle; buffers[1] = xq.Handle; buffers[2] = xds.Handle; buffers[3] = indices.Handle; buffers[4] = y.Handle;
        if (_grid is not null) buffers[5] = _grid.Handle;
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5] { (uint)m, (uint)k, (uint)n, (uint)numExperts, (uint)blocksPerRow };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, (uint)n, 1);
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
        // The iq3s grid is owned by Iq3Codebooks.
    }
}
