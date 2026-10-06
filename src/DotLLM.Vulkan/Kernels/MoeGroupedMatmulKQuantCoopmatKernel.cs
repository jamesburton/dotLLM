using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>Packed K-quant bank type served by <see cref="MoeGroupedMatmulKQuantCoopmatKernel"/>.</summary>
public enum MoeGroupedKQuant
{
    /// <summary>Q4_K, 144-byte super-blocks.</summary>
    Q4_K,
    /// <summary>Q5_K, 176-byte super-blocks.</summary>
    Q5_K,
    /// <summary>Q6_K, 210-byte super-blocks.</summary>
    Q6_K,
}

/// <summary>
/// Grouped MoE expert projection over PACKED Q4_K / Q5_K / Q6_K expert banks using cooperative matrices (issue #637). Rows are grouped by expert
/// (see <see cref="MoeExpertOffsetsKernel"/> / <see cref="MoeExpandGroupByExpertF32Kernel"/>); each 16-row tile dequantises its expert's weight
/// sub-blocks into the F16 A operand once, instead of the indexed MMQ kernel re-reading the weights per routed row.
/// </summary>
public sealed class MoeGroupedMatmulKQuantCoopmatKernel : IDisposable
{
    /// <summary>K must be a multiple of this (one super-block).</summary>
    public const int KGroup = 256;
    private const int TileN = 16;
    // Output rows per workgroup: 16 for the one-subgroup shader; 64 for the _m64 shaders (4 wave64 subgroups, each one
    // 16-row M tile, sharing one staged B tile).
    private readonly int _tileM;
    private readonly bool _rowPair;
    private const int PushConstantBytes = 7 * sizeof(uint);   // + tileList

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly int _blockBytes;
    private bool _disposed;

    private MoeGroupedMatmulKQuantCoopmatKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, int blockBytes, int tileM, bool rowPair)
    {
        _tileM = tileM;
        _rowPair = rowPair;
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _blockBytes = blockBytes;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    private static string SpvName(MoeGroupedKQuant q) => q switch
    {
        MoeGroupedKQuant.Q4_K => "moe_grouped_matmul_q4_k_coopmat.spv",
        MoeGroupedKQuant.Q5_K => "moe_grouped_matmul_q5_k_coopmat.spv",
        _ => "moe_grouped_matmul_q6_k_coopmat.spv",
    };

    /// <summary>
    /// Output rows per workgroup. The multi-subgroup shaders assume wave64 (subgroup = 64 threads = one 16-row M tile), so they are chosen only
    /// on a wave64 device. Default 64; <c>DOTLLM_VK_MOE_GROUPED_TILE</c> = 16 | 64 overrides (16 = the single-subgroup shader).
    /// </summary>
    private static int ResolveTileM(VulkanDevice device, string spvDir, MoeGroupedKQuant q)
    {
        int tile = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_VK_MOE_GROUPED_TILE"), out int t) ? t : 64;
        if (tile == 16 || device.SubgroupSize != 64) return 16;
        return File.Exists(Path.Combine(spvDir, SpvNameTile(q, tile))) ? tile : 16;
    }

    private static string SpvNameTile(MoeGroupedKQuant q, int tile)
        => tile == 16 ? SpvName(q) : SpvName(q).Replace("_coopmat.spv", $"_coopmat_m{tile}.spv", StringComparison.Ordinal);

    private static int BlockBytesOf(MoeGroupedKQuant q) => q switch { MoeGroupedKQuant.Q4_K => 144, MoeGroupedKQuant.Q5_K => 176, _ => 210 };

    /// <summary>Whether <paramref name="device"/> has cooperative matrices and the SPIR-V is present.</summary>
    public static bool IsSupportedOn(VulkanDevice device, string spvDir, MoeGroupedKQuant quant)
        => device.HasCooperativeMatrix && File.Exists(Path.Combine(spvDir, SpvName(quant)));

    /// <summary>Creates the kernel for <paramref name="quant"/>; <paramref name="tileMOverride"/> (16 or 64) pins the tile shape for tests.</summary>
    public static MoeGroupedMatmulKQuantCoopmatKernel Create(VulkanDevice device, string spvDir, MoeGroupedKQuant quant, int tileMOverride = 0, bool rowPair = false)
    {
        if (!device.HasCooperativeMatrix)
            throw new InvalidOperationException("MoeGroupedMatmulKQuantCoopmatKernel requires VK_KHR_cooperative_matrix support.");
        int tileM = tileMOverride > 0 && File.Exists(Path.Combine(spvDir, SpvNameTile(quant, tileMOverride))) ? tileMOverride : ResolveTileM(device, spvDir, quant);
        string path = Path.Combine(spvDir, SpvNameTile(quant, tileM));
        // Row-pair shaders exist only for the 4-subgroup (64-row) form and need an INDIRECT launch built with 32-row tiles; anything else keeps the 16-row shader.
        if (rowPair && tileM == 64 && File.Exists(path.Replace("_m64.spv", "_m64r2.spv", StringComparison.Ordinal))) path = path.Replace("_m64.spv", "_m64r2.spv", StringComparison.Ordinal);
        else rowPair = false;
        if (!File.Exists(path)) throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            for (int i = 0; i < bindings.Length; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes, requiredSubgroupSize: 0);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MoeGroupedMatmulKQuantCoopmatKernel(device, module, pipeline, pool, BlockBytesOf(quant), tileM, rowPair);
    }

    /// <summary>Raw pipeline handle, for driver shader-statistics diagnostics.</summary>
    internal nint PipelineHandle => _pipeline.Pipeline;

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch; used by unit tests.</summary>
    public void Launch(VulkanDevice.Buffer bank, VulkanDevice.Buffer packedInput, VulkanDevice.Buffer offsets, VulkanDevice.Buffer output,
        int m, int k, int rows, int numExperts, int maxRowsPerExpert = 0)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, bank, packedInput, offsets, output, m, k, rows, numExperts, maxRowsPerExpert);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the grouped matmul (<paramref name="maxRowsPerExpert"/> bounds the row-tile grid; 0 = <paramref name="rows"/>; a token routes to an expert at most once, so a MoE layer can pass its token count): <c>output[packedRow, m] = packedInput[packedRow, :] . bank[expert(packedRow), m, :]</c>.</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer packedInput, VulkanDevice.Buffer offsets,
        VulkanDevice.Buffer output, int m, int k, int rows, int numExperts, int maxRowsPerExpert = 0)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (k <= 0 || (k % KGroup) != 0) throw new ArgumentException($"k must be a positive multiple of {KGroup}, got {k}", nameof(k));

        int blocksPerRow = k / KGroup;
        long rowBytes = (long)blocksPerRow * _blockBytes;
        if (bank.Size < (long)numExperts * m * rowBytes) throw new ArgumentException("bank buffer too small.", nameof(bank));
        if (packedInput.Size < (long)rows * k * sizeof(float)) throw new ArgumentException("packedInput buffer too small.", nameof(packedInput));
        if (offsets.Size < (long)(numExperts + 1) * sizeof(uint)) throw new ArgumentException("offsets buffer too small.", nameof(offsets));
        if (output.Size < (long)rows * m * sizeof(float)) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[4] { bank.Handle, packedInput.Handle, offsets.Handle, output.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[7] { (uint)m, (uint)k, (uint)rows, (uint)numExperts, (uint)blocksPerRow, (uint)(rowBytes / 4), 0u };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);

        int rowTiles = maxRowsPerExpert > 0 ? Math.Min(maxRowsPerExpert, rows) : rows;
        // Row tiles are pessimistic (all rows could land on one expert); tiles past an expert's count early-out.
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + _tileM - 1) / _tileM), (uint)((rowTiles + TileN - 1) / TileN), (uint)numExperts);
    }

    /// <summary>
    /// Indirect variant: launches ONLY the real (expert, 16-row tile) pairs. <paramref name="offsetsAndTiles"/> carries the group offsets
    /// in <c>[0, numExperts]</c> and the work list written by <see cref="MoeBuildTileListKernel"/> from <c>numExperts + 1</c> on;
    /// <paramref name="dispatchArgs"/> holds its <c>(mTiles, tileCount, 1)</c> at <paramref name="dispatchArgsOffset"/> (0 or <see cref="MoeBuildTileListKernel.ArgsStrideBytes"/>). A
    /// <see cref="KernelSupport.ComputeToIndirectAndComputeBarrier"/> must separate the build from this call. The legacy
    /// <see cref="Record"/> grid launches every expert x every possible row tile (~30x the work at serving prompt sizes).
    /// </summary>
    public unsafe void RecordIndirect(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer packedInput, VulkanDevice.Buffer offsetsAndTiles,
        VulkanDevice.Buffer output, VulkanDevice.Buffer dispatchArgs, int dispatchArgsOffset, int m, int k, int rows, int numExperts)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (k <= 0 || (k % KGroup) != 0) throw new ArgumentException($"k must be a positive multiple of {KGroup}, got {k}", nameof(k));
        if (dispatchArgs.Size < dispatchArgsOffset + 3 * sizeof(uint)) throw new ArgumentException("dispatchArgs buffer too small.", nameof(dispatchArgs));

        int blocksPerRow = k / KGroup;
        long rowBytes = (long)blocksPerRow * _blockBytes;
        if (bank.Size < (long)numExperts * m * rowBytes) throw new ArgumentException("bank buffer too small.", nameof(bank));
        if (packedInput.Size < (long)rows * k * sizeof(float)) throw new ArgumentException("packedInput buffer too small.", nameof(packedInput));
        if (offsetsAndTiles.Size < MoeBuildTileListKernel.OffsetsBufferUints(numExperts, rows) * sizeof(uint))
            throw new ArgumentException("offsets/tile-list buffer too small.", nameof(offsetsAndTiles));
        if (output.Size < (long)rows * m * sizeof(float)) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[4] { bank.Handle, packedInput.Handle, offsetsAndTiles.Handle, output.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[7] { (uint)m, (uint)k, (uint)rows, (uint)numExperts, (uint)blocksPerRow, (uint)(rowBytes / 4), (uint)(numExperts + 1) };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        VulkanApi.vkCmdDispatchIndirect(cmdBuf, dispatchArgs.Handle, (ulong)dispatchArgsOffset);
    }

    /// <summary>Token rows per workgroup: 32 for the row-pair shaders (build the tile list with this), else 16.</summary>
    public int RowTile => _rowPair ? 2 * TileN : TileN;

    /// <summary>Weight-row tiles per workgroup column (the x extent of the grid): <c>ceil(m / tileM)</c>.</summary>
    public int MTiles(int m) => (m + _tileM - 1) / _tileM;

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
