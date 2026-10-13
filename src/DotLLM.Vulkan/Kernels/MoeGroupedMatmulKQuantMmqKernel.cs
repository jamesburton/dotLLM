using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Grouped MoE expert projection over packed Q4_K / Q5_K expert banks using dp4a integer dot products on Q8_1-quantized
/// activations (issue #986). Two tile variants: 64×64 (best when rows/expert ≥ 48) and 64×32 (better at moderate prompt
/// lengths where most experts get &lt; 32 routed rows).
/// </summary>
/// <remarks>
/// <para>
/// Activations must be pre-quantized to Q8_1 by <see cref="QuantizeQ8_1RowsKernel"/> AFTER the
/// <see cref="MoeExpandGroupByExpertF32Kernel"/> sort, so within a tile all N rows share the same expert.
/// </para>
/// <para>
/// Supports both legacy 3-D grid dispatch and indirect dispatch via <see cref="MoeBuildTileListKernel"/>.
/// </para>
/// </remarks>
public sealed class MoeGroupedMatmulKQuantMmqKernel : IDisposable
{
    /// <summary>K must be a multiple of this (one super-block = 256 elements).</summary>
    public const int KGroup = 256;
    private const int TileM = 64;
    private const int PushConstantBytes = 7 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly int _blockBytes;
    private readonly int _tileN;
    private bool _disposed;

    private MoeGroupedMatmulKQuantMmqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, int blockBytes, int tileN)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _blockBytes = blockBytes;
        _tileN = tileN;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 5);
    }

    private static string SpvName(MoeGroupedKQuant q, int tileN) => (q, tileN) switch
    {
        (MoeGroupedKQuant.Q4_K, 32) => "moe_grouped_matmul_q4_k_mmq_n32.spv",
        (MoeGroupedKQuant.Q4_K, _)  => "moe_grouped_matmul_q4_k_mmq.spv",
        (MoeGroupedKQuant.Q5_K, 32) => "moe_grouped_matmul_q5_k_mmq_n32.spv",
        (MoeGroupedKQuant.Q5_K, _)  => "moe_grouped_matmul_q5_k_mmq.spv",
        _ => throw new ArgumentOutOfRangeException(nameof(q), q, "Only Q4_K and Q5_K have MMQ grouped kernels."),
    };

    private static int BlockBytesOf(MoeGroupedKQuant q) => q switch { MoeGroupedKQuant.Q4_K => 144, MoeGroupedKQuant.Q5_K => 176, _ => throw new ArgumentOutOfRangeException(nameof(q)) };

    /// <summary>Resolves the tile-N size: <c>DOTLLM_VK_MOE_GROUPED_MMQ_TILE=32|64</c> overrides; default 32 (better at pp≤1024).</summary>
    private static int ResolveTileN(string spvDir, MoeGroupedKQuant q)
    {
        if (int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VK_MOE_GROUPED_MMQ_TILE"), out int t) && (t == 32 || t == 64))
            return File.Exists(Path.Combine(spvDir, SpvName(q, t))) ? t : 32;
        return File.Exists(Path.Combine(spvDir, SpvName(q, 32))) ? 32 : 64;
    }

    /// <summary>Whether <paramref name="device"/> has integer dot product and at least one SPIR-V variant is present.</summary>
    public static bool IsSupportedOn(VulkanDevice device, string spvDir, MoeGroupedKQuant quant)
        => device.HasIntegerDotProduct
           && (File.Exists(Path.Combine(spvDir, SpvName(quant, 32))) || File.Exists(Path.Combine(spvDir, SpvName(quant, 64))));

    /// <summary>Creates the kernel for <paramref name="quant"/>.</summary>
    public static MoeGroupedMatmulKQuantMmqKernel Create(VulkanDevice device, string spvDir, MoeGroupedKQuant quant)
    {
        if (!device.HasIntegerDotProduct)
            throw new InvalidOperationException("MoeGroupedMatmulKQuantMmqKernel requires integer dot product (dp4a) support.");
        int tileN = ResolveTileN(spvDir, quant);
        string path = Path.Combine(spvDir, SpvName(quant, tileN));
        if (!File.Exists(path)) throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[5];
            for (int i = 0; i < bindings.Length; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes, requiredSubgroupSize: 0);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        return new MoeGroupedMatmulKQuantMmqKernel(device, module, pipeline, pool, BlockBytesOf(quant), tileN);
    }

    internal nint PipelineHandle => _pipeline.Pipeline;

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Token rows per workgroup (32 or 64).</summary>
    public int RowTile => _tileN;

    /// <summary>Weight-row tiles: <c>ceil(m / 64)</c>.</summary>
    public int MTiles(int m) => (m + TileM - 1) / TileM;

    /// <summary>Synchronous launch for tests.</summary>
    public void Launch(VulkanDevice.Buffer bank, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer offsets, VulkanDevice.Buffer output,
        int m, int k, int rows, int numExperts, int maxRowsPerExpert = 0)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, bank, xq, xds, offsets, output, m, k, rows, numExperts, maxRowsPerExpert);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the grouped dp4a matmul: <c>output[packedRow, m] = xq8[packedRow, :] . bank[expert(packedRow), m, :]</c>.</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer offsets, VulkanDevice.Buffer output,
        int m, int k, int rows, int numExperts, int maxRowsPerExpert = 0)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (k <= 0 || (k % KGroup) != 0) throw new ArgumentException($"k must be a positive multiple of {KGroup}, got {k}", nameof(k));

        int blocksPerRow = k / KGroup;
        long rowBytes = (long)blocksPerRow * _blockBytes;
        if (bank.Size < (long)numExperts * m * rowBytes) throw new ArgumentException("bank buffer too small.", nameof(bank));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(rows, k)) throw new ArgumentException("xq buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(rows, k)) throw new ArgumentException("xds buffer too small.", nameof(xds));
        if (offsets.Size < (long)(numExperts + 1) * sizeof(uint)) throw new ArgumentException("offsets buffer too small.", nameof(offsets));
        if (output.Size < (long)rows * m * sizeof(float)) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[5] { bank.Handle, xq.Handle, xds.Handle, offsets.Handle, output.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[7] { (uint)m, (uint)k, (uint)rows, (uint)numExperts, (uint)blocksPerRow, (uint)(rowBytes / 4), 0u };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);

        int rowTiles = maxRowsPerExpert > 0 ? Math.Min(maxRowsPerExpert, rows) : rows;
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + TileM - 1) / TileM), (uint)((rowTiles + _tileN - 1) / _tileN), (uint)numExperts);
    }

    /// <summary>
    /// Indirect variant using the tile work list from <see cref="MoeBuildTileListKernel"/>. A
    /// <see cref="KernelSupport.ComputeToIndirectAndComputeBarrier"/> must separate the build from this call.
    /// </summary>
    public unsafe void RecordIndirect(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer offsetsAndTiles, VulkanDevice.Buffer output,
        VulkanDevice.Buffer dispatchArgs, int dispatchArgsOffset, int m, int k, int rows, int numExperts)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (k <= 0 || (k % KGroup) != 0) throw new ArgumentException($"k must be a positive multiple of {KGroup}, got {k}", nameof(k));
        if (dispatchArgs.Size < dispatchArgsOffset + 3 * sizeof(uint)) throw new ArgumentException("dispatchArgs buffer too small.", nameof(dispatchArgs));

        int blocksPerRow = k / KGroup;
        long rowBytes = (long)blocksPerRow * _blockBytes;
        if (bank.Size < (long)numExperts * m * rowBytes) throw new ArgumentException("bank buffer too small.", nameof(bank));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(rows, k)) throw new ArgumentException("xq buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(rows, k)) throw new ArgumentException("xds buffer too small.", nameof(xds));
        if (offsetsAndTiles.Size < MoeBuildTileListKernel.OffsetsBufferUints(numExperts, rows) * sizeof(uint))
            throw new ArgumentException("offsets/tile-list buffer too small.", nameof(offsetsAndTiles));
        if (output.Size < (long)rows * m * sizeof(float)) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[5] { bank.Handle, xq.Handle, xds.Handle, offsetsAndTiles.Handle, output.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[7] { (uint)m, (uint)k, (uint)rows, (uint)numExperts, (uint)blocksPerRow, (uint)(rowBytes / 4), (uint)(numExperts + 1) };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        VulkanApi.vkCmdDispatchIndirect(cmdBuf, dispatchArgs.Handle, (ulong)dispatchArgsOffset);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptorPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
