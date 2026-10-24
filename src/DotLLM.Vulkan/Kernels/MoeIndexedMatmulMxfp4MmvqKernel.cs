using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE INDEXED MXFP4 MMVQ decode GEMV (gpt-oss): <c>y[n, m] = sum_k dequant_mxfp4(bank[indices[n], m, k]) * x[n / xDiv, k] (+ bias)</c>
/// via integer dp4a against the row-major Q8_1-quantized activations (<see cref="QuantizeQ8_1RowsKernel"/>). One wave32 subgroup per
/// output cell, lane = weight block, so a row is read as contiguous 17-byte blocks; replaces the one-thread-per-cell
/// <see cref="MoeIndexedMatmulMxfp4F32Kernel"/> on the decode path. <c>xDiv</c> broadcasts one quantized activation row to its topK expert
/// slots; the optional per-expert bias is folded into the epilogue. Dispatch: <c>(M, N)</c> workgroups.
/// </summary>
/// <remarks>NOT bit-exact vs the F32-in kernel (the activation is int8-quantized).</remarks>
public sealed class MoeIndexedMatmulMxfp4MmvqKernel : IDisposable
{
    /// <summary>MXFP4 block: 1 byte E8M0 scale + 16 bytes of packed nibbles (17 bytes, 32 elements).</summary>
    public const int BlockBytes = QuantFormat.Mxfp4BlockBytes;
    /// <summary>Elements per MXFP4 block.</summary>
    public const int GroupSize = QuantFormat.LegacyGroupSize;

    private const int BuffersPerSet = 6;
    private const int PushConstantBytes = 7 * sizeof(uint); // M, K, N, numExperts, blocksPerRow, xDiv, hasBias

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeIndexedMatmulMxfp4MmvqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: BuffersPerSet);
    }

    /// <summary>
    /// Loads <c>moe_indexed_matmul_mxfp4_mmvq.spv</c> from <paramref name="spvDir"/> and builds the pipeline. Returns <c>null</c> when the
    /// SPV is missing or the device lacks integer-dot-product support (callers keep the scalar kernel).
    /// </summary>
    public static MoeIndexedMatmulMxfp4MmvqKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;
        string path = Path.Combine(spvDir, "moe_indexed_matmul_mxfp4_mmvq.spv");
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
        return new MoeIndexedMatmulMxfp4MmvqKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the indexed MXFP4 MMVQ expert matmul into <paramref name="cmdBuf"/>.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="bank">Raw packed bank of <c>numExperts * M * (K/32) * 17</c> bytes.</param>
    /// <param name="xq">Packed-int8 quantized activations, row-major, <c>(N/xDiv)*K/4</c> uints.</param>
    /// <param name="xds">Per-block (scale, sum) of the quantized activations, <c>(N/xDiv)*K/32</c> vec2.</param>
    /// <param name="indices">int32 per-row expert index [<paramref name="n"/>].</param>
    /// <param name="y">F32 output rows [<paramref name="n"/> * M] row-major.</param>
    /// <param name="bias">Optional per-expert bias [numExperts * M] F32 added to every output cell; <c>null</c> = none.</param>
    /// <param name="m">Per-expert weight row count (output dim).</param>
    /// <param name="k">Per-expert weight column count (must be a multiple of 32).</param>
    /// <param name="n">Number of output rows (typically <c>seqLen * topK</c>).</param>
    /// <param name="numExperts">Bank's first axis size - bounds-checks the index lookup.</param>
    /// <param name="xDiv">Activation row for output row <c>r</c> is <c>r / xDiv</c>: 1 = one quantized row per output row; <c>topK</c> = a
    /// single decode row broadcast to its topK expert slots.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer bank, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer y, VulkanDevice.Buffer? bias,
        int m, int k, int n, int numExperts, int xDiv = 1)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (xDiv <= 0) throw new ArgumentOutOfRangeException(nameof(xDiv));
        if ((k % GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {GroupSize}, got {k}", nameof(k));

        int xRows = (n + xDiv - 1) / xDiv;
        int blocksPerRow = k / GroupSize;
        long bankBytes = (long)numExperts * m * blocksPerRow * BlockBytes;
        if (bank.Size < bankBytes) throw new ArgumentException("bank buffer too small.", nameof(bank));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(xRows, k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(xRows, k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (indices.Size < (long)n * sizeof(int)) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (y.Size < (long)n * m * sizeof(float)) throw new ArgumentException("y buffer too small.", nameof(y));
        if (bias is not null && bias.Size < (long)numExperts * m * sizeof(float))
            throw new ArgumentException("bias buffer too small.", nameof(bias));

        Span<nint> buffers = stackalloc nint[BuffersPerSet]
        {
            bank.Handle, xq.Handle, xds.Handle, indices.Handle, y.Handle, (bias ?? xds).Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[7]
        {
            (uint)m, (uint)k, (uint)n, (uint)numExperts, (uint)blocksPerRow, (uint)xDiv, bias is null ? 0u : 1u,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
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
    }
}
