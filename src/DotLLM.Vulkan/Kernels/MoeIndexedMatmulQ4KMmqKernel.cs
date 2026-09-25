using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE indexed expert matmul over a packed Q4_K expert bank, dp4a (Q8_1-activation)
/// variant of <see cref="MoeIndexedMatmulQ4_KF32Kernel"/> (issue #383):
/// <c>y[n, m] = dequant_q4k(bank[indices[n], m, :]) dot dequant_q8_1(xq[n, :], xds[n, :])</c>.
/// </summary>
/// <remarks>
/// <para>
/// Root cause fixed: the F32 sibling is scalar (one thread, one output cell, plain
/// float multiply-add) — the same class of gap #139 closed for dense prefill matmuls
/// via a register-tiled dp4a MMQ kernel (<see cref="MatMulQ4KMmqKernel"/>). That
/// register-tiled design does not port here as-is: its tiling assumes every row in a
/// workgroup tile shares one weight matrix, but MoE rows independently route to
/// different experts. This kernel keeps the F32 sibling's exact one-thread-per-cell
/// dispatch shape (no weight-tile-reuse assumption, no architecture change) but swaps
/// the scalar per-element multiply for the same dp4a integer-dot math
/// <see cref="MatMulQ4KMmqKernel"/> already uses per output cell — valid per-thread
/// regardless of whether neighboring threads share a weight matrix.
/// </para>
/// <para>
/// Activations must be pre-quantized to Q8_1 row-wise (<see cref="QuantizeQ8_1RowsKernel"/>)
/// before this kernel runs. NOT bit-exact vs the F32-in kernel — validated against the
/// CPU F32 oracle with the same tolerance + argmax parity pattern as
/// <see cref="MatMulQ4KMmqKernel"/>.
/// </para>
/// </remarks>
public sealed class MoeIndexedMatmulQ4KMmqKernel : IDisposable
{
    /// <summary>Q4_K super-block: 2(d) + 2(dmin) + 12(scales) + 128(qs) = 144 bytes.</summary>
    public const int Q4_KBlockBytes = QuantFormat.Q4_KBlockBytes;

    /// <summary>Elements per Q4_K super-block.</summary>
    public const int Q4_KGroupSize = QuantFormat.KQuantGroupSize;

    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    // M, K, N, numExperts, blocksPerRow
    private const int PushConstantBytes = 5 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeIndexedMatmulQ4KMmqKernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 5);
    }

    /// <summary>
    /// Loads <c>moe_indexed_matmul_q4_k_q8_1.spv</c> from <paramref name="spvDir"/>.
    /// Returns <c>null</c> when the SPV is missing OR the device does not advertise
    /// <c>VK_KHR_shader_integer_dot_product</c> — the caller should fall back to
    /// <see cref="MoeIndexedMatmulQ4_KF32Kernel"/>.
    /// </summary>
    public static MoeIndexedMatmulQ4KMmqKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "moe_indexed_matmul_q4_k_q8_1.spv");
        if (!File.Exists(path))
            return null;

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[5];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            bindings[3] = new VkDescriptorBinding(3);
            bindings[4] = new VkDescriptorBinding(4);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        return new MoeIndexedMatmulQ4KMmqKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer bankQ4K, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, bankQ4K, xq, xds, indices, y, m, k, n, numExperts);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the indexed Q4_K expert-bank MMQ dispatch into <paramref name="cmdBuf"/>.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="bankQ4K">
    /// Raw Q4_K bank of <c>numExperts * M * (K/256) * 144</c> bytes, expert
    /// matrices contiguous, rows contiguous within each expert.
    /// </param>
    /// <param name="xq">Q8_1 packed-int8 activation rows, <see cref="QuantizeQ8_1RowsKernel.PackedBytes"/>.</param>
    /// <param name="xds">Q8_1 (scale, sum) activation rows, <see cref="QuantizeQ8_1RowsKernel.ScaleBytes"/>.</param>
    /// <param name="indices">int32 per-row expert index [<paramref name="n"/>].</param>
    /// <param name="y">F32 output rows [<paramref name="n"/> * M] row-major.</param>
    /// <param name="m">Per-expert weight row count (output dim).</param>
    /// <param name="k">Per-expert weight column count (must be a multiple of 256).</param>
    /// <param name="n">Number of output rows (typically <c>seqLen * topK</c>).</param>
    /// <param name="numExperts">Bank's first axis size — used for bounds-checking the index lookup.</param>
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
        if ((k % Q4_KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q4_KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q4_KGroupSize;
        long rowBytes = (long)blocksPerRow * Q4_KBlockBytes;
        long bankBytes = (long)numExperts * m * rowBytes;
        long idxBytes = (long)n * sizeof(int);
        long yBytes = (long)n * m * sizeof(float);
        if (bankQ4K.Size < bankBytes) throw new ArgumentException("bankQ4K buffer too small.", nameof(bankQ4K));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(n, k)) throw new ArgumentException("xq buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(n, k)) throw new ArgumentException("xds buffer too small.", nameof(xds));
        if (indices.Size < idxBytes) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (y.Size < yBytes) throw new ArgumentException("y buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[5]
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
            (uint)m,
            (uint)k,
            (uint)n,
            (uint)numExperts,
            (uint)blocksPerRow,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupX = (uint)((m + WorkgroupX - 1) / WorkgroupX);
        uint groupY = (uint)((n + WorkgroupY - 1) / WorkgroupY);
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
