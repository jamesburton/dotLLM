using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Tiled (shared-memory) variant of the MoE indexed expert matmul.
/// Same numerics + same buffer/push-constant layout as
/// <see cref="MoeIndexedMatmulF32Kernel"/>; differs only in the on-device
/// dispatch shape and the use of workgroup shared memory along K.
/// </summary>
/// <remarks>
/// <para>
/// Strategy: one workgroup per output row (<c>gl_WorkGroupID.y = n</c>) so
/// every thread in the WG shares the same expert index <c>indices[n]</c>.
/// TILE_M = 16 threads per WG cover TILE_M consecutive m-values; threads
/// march K in chunks of TILE_K = 16, cooperatively staging
/// <c>x[n, kchunk]</c> and <c>bank[idx, m_tile, kchunk]</c> into shared
/// memory before the inner dot product. This amortises the global-memory
/// load of <c>x[n, :]</c> across the TILE_M output cells in the WG, which
/// is the bottleneck on prefill where N (<c>seqLen * topK</c>) is large.
/// </para>
/// <para>
/// Decode (N small, single-token) prefers the scalar variant — at N=1
/// the GEMV-style scalar kernel needs fewer dispatches and is not bound
/// by the same x-row reload that the tile amortises. Routing between
/// scalar and tiled is the caller's responsibility (see
/// <c>VulkanTransformerModel.RecordMoeLayer</c>).
/// </para>
/// </remarks>
public sealed class MoeIndexedMatmulTiledF32Kernel : IDisposable
{
    /// <summary>Tile width along the output (m) axis. Equal to local_size_x in the shader.</summary>
    public const int TileM = 16;
    /// <summary>Tile width along the contraction (k) axis. Internal to the shader.</summary>
    public const int TileK = 16;

    // Same push constants as the scalar variant: M, K, N, numExperts (all u32).
    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeIndexedMatmulTiledF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 4);
    }

    /// <summary>Loads <c>moe_indexed_matmul_tiled_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeIndexedMatmulTiledF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_indexed_matmul_tiled_f32.spv");
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
        return new MoeIndexedMatmulTiledF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer bank, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, bank, x, indices, y, m, k, n, numExperts);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the tiled indexed matmul dispatch into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="bank">F32 weight bank [<paramref name="numExperts"/> × M × K] row-major.</param>
    /// <param name="x">F32 input rows [<paramref name="n"/> × K] row-major.</param>
    /// <param name="indices">int32 per-row expert index [<paramref name="n"/>].</param>
    /// <param name="y">F32 output rows [<paramref name="n"/> × M] row-major.</param>
    /// <param name="m">Per-expert weight row count (output dim).</param>
    /// <param name="k">Per-expert weight column count (contraction dim).</param>
    /// <param name="n">Number of output rows (typically <c>seqLen * topK</c>).</param>
    /// <param name="numExperts">Bank's first axis size — used for bounds-checking the index lookup.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer bank, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));

        long bankBytes = (long)numExperts * m * k * sizeof(float);
        long xBytes = (long)n * k * sizeof(float);
        long idxBytes = (long)n * sizeof(int);
        long yBytes = (long)n * m * sizeof(float);
        if (bank.Size < bankBytes) throw new ArgumentException("bank buffer too small.", nameof(bank));
        if (x.Size < xBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (indices.Size < idxBytes) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (y.Size < yBytes) throw new ArgumentException("y buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[4]
        {
            bank.Handle, x.Handle, indices.Handle, y.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)m);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)k);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)n);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)numExperts);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // One WG per (m_tile, n). groupX = ceil(M / TILE_M), groupY = N.
        // Note: groupY is N (not ceil(N / TILE_N)) — there is no N-axis tiling
        // here; one WG handles a single output row to keep the per-row expert
        // index local to a single workgroup.
        uint groupX = (uint)((m + TileM - 1) / TileM);
        uint groupY = (uint)n;
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
