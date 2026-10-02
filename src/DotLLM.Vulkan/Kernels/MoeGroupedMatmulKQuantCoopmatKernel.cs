using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>Packed K-quant bank type served by <see cref="MoeGroupedMatmulKQuantCoopmatKernel"/>.</summary>
public enum MoeGroupedKQuant
{
    /// <summary>Q4_K, 144-byte super-blocks.</summary>
    Q4_K,
    /// <summary>Q5_K, 176-byte super-blocks.</summary>
    Q5_K,
}

/// <summary>
/// Grouped MoE expert projection over PACKED Q4_K / Q5_K expert banks using cooperative matrices (issue #637). Rows are grouped by expert
/// (see <see cref="MoeExpertOffsetsKernel"/> / <see cref="MoeExpandGroupByExpertF32Kernel"/>); each 16-row tile dequantises its expert's weight
/// sub-blocks into the F16 A operand once, instead of the indexed MMQ kernel re-reading the weights per routed row.
/// </summary>
public sealed class MoeGroupedMatmulKQuantCoopmatKernel : IDisposable
{
    /// <summary>K must be a multiple of this (one super-block).</summary>
    public const int KGroup = 256;
    private const int TileM = 16, TileN = 16;
    private const int PushConstantBytes = 6 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly int _blockBytes;
    private bool _disposed;

    private MoeGroupedMatmulKQuantCoopmatKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, int blockBytes)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _blockBytes = blockBytes;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    private static string SpvName(MoeGroupedKQuant q) => q == MoeGroupedKQuant.Q4_K
        ? "moe_grouped_matmul_q4_k_coopmat.spv" : "moe_grouped_matmul_q5_k_coopmat.spv";

    /// <summary>Whether <paramref name="device"/> has cooperative matrices and the SPIR-V is present.</summary>
    public static bool IsSupportedOn(VulkanDevice device, string spvDir, MoeGroupedKQuant quant)
        => device.HasCooperativeMatrix && File.Exists(Path.Combine(spvDir, SpvName(quant)));

    /// <summary>Creates the kernel for <paramref name="quant"/>.</summary>
    public static MoeGroupedMatmulKQuantCoopmatKernel Create(VulkanDevice device, string spvDir, MoeGroupedKQuant quant)
    {
        if (!device.HasCooperativeMatrix)
            throw new InvalidOperationException("MoeGroupedMatmulKQuantCoopmatKernel requires VK_KHR_cooperative_matrix support.");
        string path = Path.Combine(spvDir, SpvName(quant));
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
        return new MoeGroupedMatmulKQuantCoopmatKernel(device, module, pipeline, pool, quant == MoeGroupedKQuant.Q4_K ? 144 : 176);
    }

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

        Span<uint> pc = stackalloc uint[6] { (uint)m, (uint)k, (uint)rows, (uint)numExperts, (uint)blocksPerRow, (uint)(rowBytes / 4) };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);

        int rowTiles = maxRowsPerExpert > 0 ? Math.Min(maxRowsPerExpert, rows) : rows;
        // Row tiles are pessimistic (all rows could land on one expert); tiles past an expert's count early-out.
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + TileM - 1) / TileM), (uint)((rowTiles + TileN - 1) / TileN), (uint)numExperts);
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
