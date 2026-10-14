using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Grouped MoE expert projection over packed Q4_K / Q5_K expert banks using dp4a integer dot products on Q8_1-quantized
/// activations (issue #986). Two tile variants: 64×64 (best when avg rows/expert ≥ ~48, i.e. seqLen &gt; 1024) and 64×32
/// (better at moderate prompt lengths where most experts get &lt; 32 routed rows). When both SPIR-V variants are present,
/// <see cref="SelectRowTile"/> picks at dispatch time based on sequence length.
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
    private const int SeqLenCrossover = 1024;

    private readonly VulkanDevice _device;
    private readonly int _blockBytes;
    private readonly Variant? _v32;
    private readonly Variant? _v64;
    private bool _disposed;

    private sealed class Variant : IDisposable
    {
        public readonly VulkanModule Module;
        public readonly ComputePipeline Pipeline;
        public readonly nint DescriptorPool;
        public readonly DescriptorSetCache Cache;
        public readonly int TileN;
        private readonly nint _deviceHandle;

        public Variant(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, int tileN)
        {
            _deviceHandle = device.Handle;
            Module = module;
            Pipeline = pipeline;
            DescriptorPool = pool;
            TileN = tileN;
            Cache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 5);
        }

        public void InvalidateCache() => Cache.Reset();

        public void Dispose()
        {
            if (DescriptorPool != 0) VulkanApi.vkDestroyDescriptorPool(_deviceHandle, DescriptorPool, 0);
            Pipeline.Dispose();
            Module.Dispose();
        }
    }

    private MoeGroupedMatmulKQuantMmqKernel(VulkanDevice device, int blockBytes, Variant? v32, Variant? v64)
    {
        _device = device;
        _blockBytes = blockBytes;
        _v32 = v32;
        _v64 = v64;
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

    private static Variant? TryCreateVariant(VulkanDevice device, string spvDir, MoeGroupedKQuant q, int tileN)
    {
        string path = Path.Combine(spvDir, SpvName(q, tileN));
        if (!File.Exists(path)) return null;
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
        return new Variant(device, module, pipeline, pool, tileN);
    }

    /// <summary>Whether <paramref name="device"/> has integer dot product and at least one SPIR-V variant is present.</summary>
    public static bool IsSupportedOn(VulkanDevice device, string spvDir, MoeGroupedKQuant quant)
        => device.HasIntegerDotProduct
           && (File.Exists(Path.Combine(spvDir, SpvName(quant, 32))) || File.Exists(Path.Combine(spvDir, SpvName(quant, 64))));

    /// <summary>Creates the kernel for <paramref name="quant"/>, loading all available tile variants.</summary>
    public static MoeGroupedMatmulKQuantMmqKernel Create(VulkanDevice device, string spvDir, MoeGroupedKQuant quant)
    {
        if (!device.HasIntegerDotProduct)
            throw new InvalidOperationException("MoeGroupedMatmulKQuantMmqKernel requires integer dot product (dp4a) support.");
        var v32 = TryCreateVariant(device, spvDir, quant, 32);
        var v64 = TryCreateVariant(device, spvDir, quant, 64);
        if (v32 is null && v64 is null)
            throw new FileNotFoundException($"No MMQ SPIR-V found for {quant} in {spvDir}.");
        return new MoeGroupedMatmulKQuantMmqKernel(device, BlockBytesOf(quant), v32, v64);
    }

    internal void InvalidateDescriptorCache()
    {
        _v32?.InvalidateCache();
        _v64?.InvalidateCache();
    }

    /// <summary>Whether both 32-row and 64-row variants are loaded (adaptive mode).</summary>
    public bool IsAdaptive => _v32 is not null && _v64 is not null;

    /// <summary>Selects the best row tile for the given sequence length. Returns 32 for short prompts, 64 for long.</summary>
    public int SelectRowTile(int seqLen)
    {
        if (_v32 is null) return 64;
        if (_v64 is null) return 32;
        return seqLen <= SeqLenCrossover ? 32 : 64;
    }

    /// <summary>Weight-row tiles: <c>ceil(m / 64)</c>.</summary>
    public int MTiles(int m) => (m + TileM - 1) / TileM;

    private Variant GetVariant(int tileN) => tileN switch
    {
        32 => _v32 ?? throw new InvalidOperationException("32-row variant not loaded."),
        64 => _v64 ?? throw new InvalidOperationException("64-row variant not loaded."),
        _ => throw new ArgumentOutOfRangeException(nameof(tileN)),
    };

    /// <summary>Records the grouped dp4a matmul with the specified tile size.</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer offsets, VulkanDevice.Buffer output,
        int m, int k, int rows, int numExperts, int tileN, int maxRowsPerExpert = 0)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (k <= 0 || (k % KGroup) != 0) throw new ArgumentException($"k must be a positive multiple of {KGroup}, got {k}", nameof(k));

        var v = GetVariant(tileN);
        int blocksPerRow = k / KGroup;
        long rowBytes = (long)blocksPerRow * _blockBytes;
        if (bank.Size < (long)numExperts * m * rowBytes) throw new ArgumentException("bank buffer too small.", nameof(bank));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(rows, k)) throw new ArgumentException("xq buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(rows, k)) throw new ArgumentException("xds buffer too small.", nameof(xds));
        if (offsets.Size < (long)(numExperts + 1) * sizeof(uint)) throw new ArgumentException("offsets buffer too small.", nameof(offsets));
        if (output.Size < (long)rows * m * sizeof(float)) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[5] { bank.Handle, xq.Handle, xds.Handle, offsets.Handle, output.Handle };
        nint descriptorSet = v.Cache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, v.Pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, v.Pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[7] { (uint)m, (uint)k, (uint)rows, (uint)numExperts, (uint)blocksPerRow, (uint)(rowBytes / 4), 0u };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, v.Pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);

        int rowTiles = maxRowsPerExpert > 0 ? Math.Min(maxRowsPerExpert, rows) : rows;
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + TileM - 1) / TileM), (uint)((rowTiles + tileN - 1) / tileN), (uint)numExperts);
    }

    /// <summary>
    /// Indirect variant using the tile work list from <see cref="MoeBuildTileListKernel"/>. A
    /// <see cref="KernelSupport.ComputeToIndirectAndComputeBarrier"/> must separate the build from this call.
    /// </summary>
    public unsafe void RecordIndirect(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer offsetsAndTiles, VulkanDevice.Buffer output,
        VulkanDevice.Buffer dispatchArgs, int dispatchArgsOffset, int m, int k, int rows, int numExperts, int tileN)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (k <= 0 || (k % KGroup) != 0) throw new ArgumentException($"k must be a positive multiple of {KGroup}, got {k}", nameof(k));
        if (dispatchArgs.Size < dispatchArgsOffset + 3 * sizeof(uint)) throw new ArgumentException("dispatchArgs buffer too small.", nameof(dispatchArgs));

        var v = GetVariant(tileN);
        int blocksPerRow = k / KGroup;
        long rowBytes = (long)blocksPerRow * _blockBytes;
        if (bank.Size < (long)numExperts * m * rowBytes) throw new ArgumentException("bank buffer too small.", nameof(bank));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(rows, k)) throw new ArgumentException("xq buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(rows, k)) throw new ArgumentException("xds buffer too small.", nameof(xds));
        if (offsetsAndTiles.Size < MoeBuildTileListKernel.OffsetsBufferUints(numExperts, rows) * sizeof(uint))
            throw new ArgumentException("offsets/tile-list buffer too small.", nameof(offsetsAndTiles));
        if (output.Size < (long)rows * m * sizeof(float)) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[5] { bank.Handle, xq.Handle, xds.Handle, offsetsAndTiles.Handle, output.Handle };
        nint descriptorSet = v.Cache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, v.Pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, v.Pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[7] { (uint)m, (uint)k, (uint)rows, (uint)numExperts, (uint)blocksPerRow, (uint)(rowBytes / 4), (uint)(numExperts + 1) };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, v.Pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        VulkanApi.vkCmdDispatchIndirect(cmdBuf, dispatchArgs.Handle, (ulong)dispatchArgsOffset);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _v32?.Dispose();
        _v64?.Dispose();
    }
}
