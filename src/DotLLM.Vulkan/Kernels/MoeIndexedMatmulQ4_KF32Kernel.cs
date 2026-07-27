using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE indexed expert matmul over a packed Q4_K expert bank:
/// <c>y[n, m] = dequant_q4k(bank[indices[n], m, :]) dot x[n, :]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Per-row Q4_K dequantisation in the inner loop — the bank stays in its
/// 144-byte-per-super-block GGUF layout on device. Sibling of
/// <see cref="MoeIndexedMatmulQ6_KF32Kernel"/>; the dequant comes straight
/// from <see cref="MatMulQ4KGemvF32Kernel"/> (and matches
/// <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeQ4_KScalar</c> byte-for-byte)
/// and the indexed expert lookup comes straight from
/// <see cref="MoeIndexedMatmulF32Kernel"/>.
/// </para>
/// <para>
/// Keeps the Gemma-4 fused <c>gate_up</c> experts (split into separate
/// gate/up Q4_K banks at upload) quantized on device — required for the
/// real 26B, whose F32-dequantised experts would consume tens of GB.
/// </para>
/// </remarks>
public sealed class MoeIndexedMatmulQ4_KF32Kernel : IDisposable
{
    /// <summary>Q4_K super-block: 2(d) + 2(dmin) + 12(scales) + 128(qs) = 144 bytes.</summary>
    public const int Q4_KBlockBytes = 144;

    /// <summary>Elements per Q4_K super-block.</summary>
    public const int Q4_KGroupSize = 256;

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

    private MoeIndexedMatmulQ4_KF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>Loads <c>moe_indexed_matmul_q4_k_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeIndexedMatmulQ4_KF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_indexed_matmul_q4_k_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            bindings[3] = new VkDescriptorBinding(3);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MoeIndexedMatmulQ4_KF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer bankQ4K, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, bankQ4K, x, indices, y, m, k, n, numExperts);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the indexed Q4_K expert-bank matmul dispatch into <paramref name="cmdBuf"/>.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="bankQ4K">
    /// Raw Q4_K bank of <c>numExperts * M * (K/256) * 144</c> bytes, expert
    /// matrices contiguous, rows contiguous within each expert.
    /// </param>
    /// <param name="x">F32 input rows [<paramref name="n"/> * K] row-major.</param>
    /// <param name="indices">int32 per-row expert index [<paramref name="n"/>].</param>
    /// <param name="y">F32 output rows [<paramref name="n"/> * M] row-major.</param>
    /// <param name="m">Per-expert weight row count (output dim).</param>
    /// <param name="k">Per-expert weight column count (must be a multiple of 256).</param>
    /// <param name="n">Number of output rows (typically <c>seqLen * topK</c>).</param>
    /// <param name="numExperts">Bank's first axis size — used for bounds-checking the index lookup.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer bankQ4K, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
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
        long xBytes = (long)n * k * sizeof(float);
        long idxBytes = (long)n * sizeof(int);
        long yBytes = (long)n * m * sizeof(float);
        if (bankQ4K.Size < bankBytes) throw new ArgumentException("bankQ4K buffer too small.", nameof(bankQ4K));
        if (x.Size < xBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (indices.Size < idxBytes) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (y.Size < yBytes) throw new ArgumentException("y buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[4]
        {
            bankQ4K.Handle, x.Handle, indices.Handle, y.Handle,
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
