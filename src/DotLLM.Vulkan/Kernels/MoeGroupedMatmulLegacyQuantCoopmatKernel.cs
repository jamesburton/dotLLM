using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>Legacy (32-element block) quant type served by <see cref="MoeGroupedMatmulLegacyQuantCoopmatKernel"/>.</summary>
public enum MoeGroupedLegacyQuant
{
    /// <summary>Q5_1, 24-byte blocks (fp16 d, fp16 m, 5th bits, nibbles).</summary>
    Q5_1,
    /// <summary>Q8_0, 34-byte blocks (fp16 d, 32 x int8).</summary>
    Q8_0,
}

/// <summary>
/// Grouped MoE expert projection over PACKED Q5_1 / Q8_0 expert banks using cooperative matrices (issue #773): the legacy-quant sibling of
/// <see cref="MoeGroupedMatmulKQuantCoopmatKernel"/> (Gemma-4's <c>ffn_down_exps</c>). Same row grouping, tile-list and
/// dispatch contract (16-row tiles, 64 output rows per workgroup, wave64), plus an optional per-expert output scale
/// (the Gemma-4 <c>ffn_down_exps.scale</c> folded into the accumulator, Q5_1 only; Q8_0 banks are pre-folded at upload).
/// K must be a multiple of <see cref="KGroup"/> (64: one staging round of two blocks).
/// </summary>
public sealed class MoeGroupedMatmulLegacyQuantCoopmatKernel : IDisposable
{
    /// <summary>K must be a multiple of this (one staging round = two 32-element blocks).</summary>
    public const int KGroup = 64;
    private const int TileN = 16;
    private const int TileM = 64;
    private const int PushConstantBytes = 8 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly MoeGroupedLegacyQuant _quant;
    private bool _disposed;

    private MoeGroupedMatmulLegacyQuantCoopmatKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, MoeGroupedLegacyQuant quant)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _quant = quant;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 5);
    }

    private static string SpvName(MoeGroupedLegacyQuant q)
        => q == MoeGroupedLegacyQuant.Q5_1 ? "moe_grouped_matmul_q5_1_coopmat_m64.spv" : "moe_grouped_matmul_q8_0_coopmat_m64.spv";

    private static int BlockBytesOf(MoeGroupedLegacyQuant q) => q == MoeGroupedLegacyQuant.Q5_1 ? 24 : 34;

    /// <summary>Whether <paramref name="device"/> has cooperative matrices, wave64 (the shader assumes 4 x 64-thread subgroups) and the SPIR-V is present.</summary>
    public static bool IsSupportedOn(VulkanDevice device, string spvDir, MoeGroupedLegacyQuant quant)
        => device.HasCooperativeMatrix && device.SubgroupSize == 64 && File.Exists(Path.Combine(spvDir, SpvName(quant)));

    /// <summary>Creates the kernel for <paramref name="quant"/>.</summary>
    public static MoeGroupedMatmulLegacyQuantCoopmatKernel Create(VulkanDevice device, string spvDir, MoeGroupedLegacyQuant quant)
    {
        if (!IsSupportedOn(device, spvDir, quant))
            throw new InvalidOperationException("MoeGroupedMatmulLegacyQuantCoopmatKernel requires VK_KHR_cooperative_matrix, wave64 and the SPIR-V.");
        var module = VulkanModule.LoadFromFile(device, Path.Combine(spvDir, SpvName(quant)));
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
        return new MoeGroupedMatmulLegacyQuantCoopmatKernel(device, module, pipeline, pool, quant);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Token rows per workgroup (always 16: there is no row-pair variant).</summary>
    public int RowTile => TileN;

    /// <summary>Weight-row tiles per workgroup column (the x extent of the grid): <c>ceil(m / 64)</c>.</summary>
    public int MTiles(int m) => (m + TileM - 1) / TileM;

    /// <summary>Synchronous launch on the legacy 3-D grid; used by unit tests.</summary>
    public void Launch(VulkanDevice.Buffer bank, VulkanDevice.Buffer packedInput, VulkanDevice.Buffer offsets, VulkanDevice.Buffer output,
        VulkanDevice.Buffer scale, bool applyScale, int m, int k, int rows, int numExperts, int maxRowsPerExpert = 0)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, bank, packedInput, offsets, output, scale, applyScale, m, k, rows, numExperts, maxRowsPerExpert);
        ctx.SubmitAndWait();
    }

    private void Validate(VulkanDevice.Buffer bank, VulkanDevice.Buffer packedInput, VulkanDevice.Buffer output, int m, int k, int rows, int numExperts, out long rowBytes)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (k <= 0 || (k % KGroup) != 0) throw new ArgumentException($"k must be a positive multiple of {KGroup}, got {k}", nameof(k));
        rowBytes = (long)(k / 32) * BlockBytesOf(_quant);
        if (bank.Size < (long)numExperts * m * rowBytes) throw new ArgumentException("bank buffer too small.", nameof(bank));
        if (packedInput.Size < (long)rows * k * sizeof(float)) throw new ArgumentException("packedInput buffer too small.", nameof(packedInput));
        if (output.Size < (long)rows * m * sizeof(float)) throw new ArgumentException("output buffer too small.", nameof(output));
    }

    /// <summary>Records the grouped matmul on the legacy grid (<paramref name="maxRowsPerExpert"/> bounds the row-tile grid; 0 = <paramref name="rows"/>): <c>output[packedRow, m] = scale? . packedInput[packedRow, :] . bank[expert(packedRow), m, :]</c>.</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer packedInput, VulkanDevice.Buffer offsets,
        VulkanDevice.Buffer output, VulkanDevice.Buffer scale, bool applyScale, int m, int k, int rows, int numExperts, int maxRowsPerExpert = 0)
    {
        Validate(bank, packedInput, output, m, k, rows, numExperts, out long rowBytes);
        if (offsets.Size < (long)(numExperts + 1) * sizeof(uint)) throw new ArgumentException("offsets buffer too small.", nameof(offsets));
        Bind(cmdBuf, bank, packedInput, offsets, output, scale, applyScale, m, k, rows, numExperts, rowBytes, tileList: 0);
        int rowTiles = maxRowsPerExpert > 0 ? Math.Min(maxRowsPerExpert, rows) : rows;
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)MTiles(m), (uint)((rowTiles + TileN - 1) / TileN), (uint)numExperts);
    }

    /// <summary>
    /// Indirect variant (see <see cref="MoeGroupedMatmulKQuantCoopmatKernel.RecordIndirect"/>): launches only the real (expert, 16-row tile)
    /// pairs from the list <see cref="MoeBuildTileListKernel"/> wrote after the offsets in <paramref name="offsetsAndTiles"/>.
    /// </summary>
    public unsafe void RecordIndirect(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer packedInput, VulkanDevice.Buffer offsetsAndTiles,
        VulkanDevice.Buffer output, VulkanDevice.Buffer scale, bool applyScale, VulkanDevice.Buffer dispatchArgs, int dispatchArgsOffset,
        int m, int k, int rows, int numExperts)
    {
        Validate(bank, packedInput, output, m, k, rows, numExperts, out long rowBytes);
        if (dispatchArgs.Size < dispatchArgsOffset + 3 * sizeof(uint)) throw new ArgumentException("dispatchArgs buffer too small.", nameof(dispatchArgs));
        if (offsetsAndTiles.Size < MoeBuildTileListKernel.OffsetsBufferUints(numExperts, rows) * sizeof(uint))
            throw new ArgumentException("offsets/tile-list buffer too small.", nameof(offsetsAndTiles));
        Bind(cmdBuf, bank, packedInput, offsetsAndTiles, output, scale, applyScale, m, k, rows, numExperts, rowBytes, tileList: (uint)(numExperts + 1));
        VulkanApi.vkCmdDispatchIndirect(cmdBuf, dispatchArgs.Handle, (ulong)dispatchArgsOffset);
    }

    private unsafe void Bind(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer packedInput, VulkanDevice.Buffer offsets, VulkanDevice.Buffer output,
        VulkanDevice.Buffer scale, bool applyScale, int m, int k, int rows, int numExperts, long rowBytes, uint tileList)
    {
        if (applyScale && scale.Size < (long)numExperts * sizeof(float)) throw new ArgumentException("scale buffer too small.", nameof(scale));
        Span<nint> buffers = stackalloc nint[5] { bank.Handle, packedInput.Handle, offsets.Handle, output.Handle, scale.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);
        // The Q5_1 shader strides rows in uints, the Q8_0 shader (34-byte blocks, not word aligned) in bytes.
        uint rowStride = _quant == MoeGroupedLegacyQuant.Q5_1 ? (uint)(rowBytes / 4) : (uint)rowBytes;
        Span<uint> pc = stackalloc uint[8] { (uint)m, (uint)k, (uint)rows, (uint)numExperts, (uint)(k / 32), rowStride, tileList, applyScale ? 1u : 0u };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
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
