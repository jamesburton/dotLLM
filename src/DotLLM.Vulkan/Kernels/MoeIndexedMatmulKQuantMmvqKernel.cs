using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE INDEXED K-quant (Q4_K / Q5_K / Q6_K) MMVQ decode GEMV:
/// <c>y[n, m] = sum_k dequant(bank[indices[n], m, k]) * x[n, k]</c> via integer dp4a against the row-major Q8_1-quantized expanded
/// MoE activations (<see cref="QuantizeQ8_1RowsKernel"/>). The indexed variant of the dense MMVQ decode kernels: the same COALESCED
/// lane = K-position layout and single subgroupAdd reduction, with the weight row looked up through the per-row expert index of a
/// packed <c>[numExperts, M, K]</c> bank. Dispatch: <c>(M, N)</c> workgroups of one wave32 subgroup each.
/// </summary>
/// <remarks>
/// Replaces the one-thread-per-cell indexed kernels on the DECODE path (small expanded-row count), whose adjacent threads read rows a
/// whole row apart (uncoalesced). NOT bit-exact vs the F32-in kernels (the activation is int8-quantized).
/// </remarks>
public sealed class MoeIndexedMatmulKQuantMmvqKernel : IDisposable
{
    /// <summary>Elements per K-quant super-block.</summary>
    public const int GroupSize = QuantFormat.KQuantGroupSize;

    private readonly int _blockBytes;

    private const int BuffersPerSet = 5;
    private const int PushConstantBytes = 6 * sizeof(uint); // M, K, N, numExperts, blocksPerRow, xDiv

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;
    private readonly int _rowsPerGroup;

    /// <summary>Output rows each workgroup produces (1 = one cell per workgroup; &gt; 1 = the multi-row variant, #876; Q4_K only).</summary>
    public int RowsPerGroup => _rowsPerGroup;

    private MoeIndexedMatmulKQuantMmvqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, int blockBytes, int rowsPerGroup)
    {
        _rowsPerGroup = rowsPerGroup;
        _blockBytes = blockBytes;
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: BuffersPerSet);
    }

    /// <summary>
    /// Loads <c>moe_indexed_matmul_{q4_k_xdiv,q5_k,q6_k}_mmvq.spv</c> from <paramref name="spvDir"/> and builds the pipeline. Returns
    /// <c>null</c> when the SPV is missing or the device lacks integer-dot-product support (callers keep their scalar/MMQ fallback).
    /// </summary>
    public static MoeIndexedMatmulKQuantMmvqKernel? TryCreate(VulkanDevice device, string spvDir, MoeGroupedKQuant quant, int rowsPerGroup = 1)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        (string name, int blockBytes) = quant switch
        {
            MoeGroupedKQuant.Q4_K => (rowsPerGroup > 1 ? "moe_indexed_matmul_q4_k_mmvq_xdiv_mr.spv" : "moe_indexed_matmul_q4_k_mmvq_xdiv.spv", QuantFormat.Q4_KBlockBytes),
            MoeGroupedKQuant.Q5_K => ("moe_indexed_matmul_q5_k_mmvq.spv", 176),
            _ => ("moe_indexed_matmul_q6_k_mmvq.spv", 210),
        };
        string path = Path.Combine(spvDir, name);
        if (!File.Exists(path))
            return null;

        uint requiredSubgroupSize = Wave32SubgroupControl.RequiredSubgroupSizeFor(device);

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[BuffersPerSet];
            for (int i = 0; i < BuffersPerSet; i++)
                bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindings,
                pushConstantBytes: PushConstantBytes,
                requiredSubgroupSize: requiredSubgroupSize,
                specConstants: rowsPerGroup > 1 ? new[] { (uint)rowsPerGroup } : default);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)BuffersPerSet);
        return new MoeIndexedMatmulKQuantMmvqKernel(device, module, pipeline, pool, blockBytes, rowsPerGroup);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the indexed MMVQ expert matmul into <paramref name="cmdBuf"/>.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="bank">Raw packed bank of <c>numExperts * M * (K/256) * blockBytes</c> bytes.</param>
    /// <param name="xq">Packed-int8 quantized activations, row-major (<see cref="QuantizeQ8_1RowsKernel"/>), <c>N*K/4</c> uints.</param>
    /// <param name="xds">Per-block (scale, sum) of the quantized activations, <c>N*K/32</c> vec2.</param>
    /// <param name="indices">int32 per-row expert index [<paramref name="n"/>].</param>
    /// <param name="y">F32 output rows [<paramref name="n"/> * M] row-major.</param>
    /// <param name="m">Per-expert weight row count (output dim).</param>
    /// <param name="k">Per-expert weight column count (must be a multiple of 256).</param>
    /// <param name="n">Number of output rows (typically <c>seqLen * topK</c>).</param>
    /// <param name="xDiv">Activation row for output row <c>r</c> is <c>r / xDiv</c>: 1 = one quantized row per output row; <c>topK</c> = a
    /// single decode row broadcast to its topK expert slots (skips the expand and the topK-fold quantize).</param>
    /// <param name="numExperts">Bank's first axis size — bounds-checks the index lookup.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer bank, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts, int xDiv = 1)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (xDiv <= 0) throw new ArgumentOutOfRangeException(nameof(xDiv));
        int xRows = (n + xDiv - 1) / xDiv;
        if ((k % GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / GroupSize;
        long rowBytes = (long)blocksPerRow * _blockBytes;
        long bankBytes = (long)numExperts * m * rowBytes;
        if (bank.Size < bankBytes) throw new ArgumentException("bank buffer too small.", nameof(bank));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(xRows, k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(xRows, k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (indices.Size < (long)n * sizeof(int)) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (y.Size < (long)n * m * sizeof(float)) throw new ArgumentException("y buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[BuffersPerSet]
        {
            bank.Handle, xq.Handle, xds.Handle, indices.Handle, y.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[6]
        {
            (uint)m, (uint)k, (uint)n, (uint)numExperts, (uint)blocksPerRow, (uint)xDiv,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // One wave32 workgroup per (m, n) output cell (or per NR consecutive rows of one n in the multi-row variant).
        if (_rowsPerGroup > 1 && (m % _rowsPerGroup) != 0)
            throw new ArgumentException($"m ({m}) must be a multiple of the rows-per-group ({_rowsPerGroup}).", nameof(m));
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)(m / _rowsPerGroup), (uint)n, 1);
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
