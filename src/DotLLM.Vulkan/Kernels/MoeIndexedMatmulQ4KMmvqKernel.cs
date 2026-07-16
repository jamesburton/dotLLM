using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE INDEXED Q4_K MMVQ decode GEMV (issue #137):
/// <c>y[n, m] = Σ_k dequant_q4k(bank[indices[n], m, k]) · x[n, k]</c> via integer
/// dp4a against the row-major Q8_1-quantized expanded MoE activations
/// (<see cref="QuantizeQ8_1RowsKernel"/>). The gathered/indexed variant of
/// <see cref="MatMulQ4KMmvqKernel"/> — same COALESCED lane=K-position layout,
/// per-sub-block <c>d·scale·(d_x·dot) − dmin·min·s</c> math and single
/// subgroupAdd reduction; the weight row is looked up through the per-row
/// expert index of a packed <c>[numExperts, M, K]</c> Q4_K bank.
/// </summary>
/// <remarks>
/// <para>
/// Replaces the one-thread-per-cell scalar
/// <see cref="MoeIndexedMatmulQ4_KF32Kernel"/> on the DECODE path (small
/// expanded-row count): the scalar kernel streams each expert row with
/// per-thread byte reads (uncoalesced), which capped the Gemma-4 26B expert
/// GEMVs far below the coalesced bandwidth the dense mmvq kernels reach.
/// </para>
/// <para>
/// NOT bit-exact vs the F32-in indexed kernel (the activation is
/// int8-quantized); validated argmax + tolerance vs the CPU MoE oracle.
/// </para>
/// <para>
/// Dispatch: 2D grid — <c>(M, N)</c> workgroups of one wave32 subgroup each.
/// </para>
/// </remarks>
public sealed class MoeIndexedMatmulQ4KMmvqKernel : IDisposable
{
    /// <summary>Q4_K super-block: 144 bytes for 256 elements.</summary>
    public const int Q4KBlockBytes = 144;

    /// <summary>Elements per Q4_K super-block.</summary>
    public const int Q4KGroupSize = 256;

    private const int BuffersPerSet = 5;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, numExperts, blocksPerRow

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeIndexedMatmulQ4KMmvqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: BuffersPerSet);
    }

    /// <summary>
    /// Loads <c>moe_indexed_matmul_q4_k_mmvq.spv</c> from <paramref name="spvDir"/> and
    /// builds the pipeline. Returns <c>null</c> when the SPV is missing OR the device
    /// does not advertise integer-dot-product support — the router then falls back to
    /// the scalar <see cref="MoeIndexedMatmulQ4_KF32Kernel"/>.
    /// </summary>
    public static MoeIndexedMatmulQ4KMmvqKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "moe_indexed_matmul_q4_k_mmvq.spv");
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
                requiredSubgroupSize: requiredSubgroupSize);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)BuffersPerSet);
        return new MoeIndexedMatmulQ4KMmvqKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the indexed Q4_K MMVQ expert matmul into <paramref name="cmdBuf"/>.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="bankQ4K">Raw Q4_K bank of <c>numExperts * M * (K/256) * 144</c> bytes.</param>
    /// <param name="xq">Packed-int8 quantized activations, row-major (<see cref="QuantizeQ8_1RowsKernel"/>), <c>N*K/4</c> uints.</param>
    /// <param name="xds">Per-block (scale, sum) of the quantized activations, <c>N*K/32</c> vec2.</param>
    /// <param name="indices">int32 per-row expert index [<paramref name="n"/>].</param>
    /// <param name="y">F32 output rows [<paramref name="n"/> * M] row-major.</param>
    /// <param name="m">Per-expert weight row count (output dim).</param>
    /// <param name="k">Per-expert weight column count (must be a multiple of 256).</param>
    /// <param name="n">Number of output rows (typically <c>seqLen * topK</c>).</param>
    /// <param name="numExperts">Bank's first axis size — bounds-checks the index lookup.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer bankQ4K, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if ((k % Q4KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q4KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q4KGroupSize;
        long rowBytes = (long)blocksPerRow * Q4KBlockBytes;
        long bankBytes = (long)numExperts * m * rowBytes;
        if (bankQ4K.Size < bankBytes) throw new ArgumentException("bankQ4K buffer too small.", nameof(bankQ4K));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(n, k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(n, k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (indices.Size < (long)n * sizeof(int)) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (y.Size < (long)n * m * sizeof(float)) throw new ArgumentException("y buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[BuffersPerSet]
        {
            bankQ4K.Handle, xq.Handle, xds.Handle, indices.Handle, y.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5]
        {
            (uint)m, (uint)k, (uint)n, (uint)numExperts, (uint)blocksPerRow,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // One wave32 workgroup per (m, n) output cell.
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
    }
}
