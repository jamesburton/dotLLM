using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Grouped MoE expert projection over a PACKED Q8_0 expert bank with <b>F16 activations in and out</b> (issue #1471, nomic-bert-moe on the Vulkan
/// encoder). The F16-activation twin of <see cref="MoeGroupedMatmulLegacyQuantCoopmatKernel"/> (Q8_0): same 16-row tiles, 64 output rows per
/// workgroup, wave64, bank read in place (34-byte blocks dequantised into the LDS A tile), same tile-list indirect dispatch; the packed input
/// is <c>f16vec4</c>, the output <c>float16_t</c> (one F16 rounding of the F32 coopmat accumulator), and the up projection can fuse the tanh GELU.
/// K must be a multiple of 64.
/// </summary>
internal sealed class MoeGroupedMatmulQ8_0F16ActKernel : IDisposable
{
    /// <summary>K must be a multiple of this (one staging round = two 32-element blocks).</summary>
    public const int KGroup = 64;
    private const int TileM = 64;
    private const int PushConstantBytes = 8 * sizeof(uint);
    private const string Spv = "moe_grouped_matmul_q8_0_coopmat_m64_hh.spv";

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeGroupedMatmulQ8_0F16ActKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device; _module = module; _pipeline = pipeline; _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>Whether <paramref name="device"/> has cooperative matrices, wave64 and the SPIR-V is present.</summary>
    public static bool IsSupportedOn(VulkanDevice device, string spvDir)
        => device.HasCooperativeMatrix && device.SubgroupSize == 64 && File.Exists(Path.Combine(spvDir, Spv));

    /// <summary>Creates the kernel.</summary>
    public static MoeGroupedMatmulQ8_0F16ActKernel Create(VulkanDevice device, string spvDir)
    {
        if (!IsSupportedOn(device, spvDir))
            throw new InvalidOperationException("MoeGroupedMatmulQ8_0F16ActKernel requires VK_KHR_cooperative_matrix, wave64 and the SPIR-V.");
        var module = VulkanModule.LoadFromFile(device, Path.Combine(spvDir, Spv));
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            for (int i = 0; i < bindings.Length; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes, requiredSubgroupSize: 0);
        }
        catch { module.Dispose(); throw; }
        return new MoeGroupedMatmulQ8_0F16ActKernel(device, module, pipeline, KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4));
    }

    /// <summary>Weight-row tiles per workgroup column (the x extent the tile-list build must be given): <c>ceil(m / 64)</c>.</summary>
    public static int MTiles(int m) => (m + TileM - 1) / TileM;

    /// <summary>Bytes of one Q8_0 weight row of <paramref name="k"/> elements.</summary>
    public static long RowBytes(int k) => (long)(k / 32) * 34;

    /// <summary>
    /// Records the grouped matmul over the tile list in <paramref name="offsetsAndTiles"/> (<see cref="MoeBuildTileListKernel"/>), launched
    /// through <paramref name="dispatchArgs"/> at <paramref name="dispatchArgsOffset"/>:
    /// <c>output[packedRow, m] = act?(packedInput[packedRow, :] . bank[expert(packedRow), m, :])</c>.
    /// </summary>
    public unsafe void RecordIndirect(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer packedInput, VulkanDevice.Buffer offsetsAndTiles,
        VulkanDevice.Buffer output, VulkanDevice.Buffer dispatchArgs, int dispatchArgsOffset, int m, int k, int rows, int numExperts, bool gelu)
    {
        if (k <= 0 || k % KGroup != 0) throw new ArgumentException($"k must be a positive multiple of {KGroup}, got {k}", nameof(k));
        if (bank.Size < (long)numExperts * m * RowBytes(k)) throw new ArgumentException("bank buffer too small.", nameof(bank));
        if (packedInput.Size < (long)rows * k * sizeof(ushort)) throw new ArgumentException("packedInput buffer too small.", nameof(packedInput));
        if (output.Size < (long)rows * m * sizeof(ushort)) throw new ArgumentException("output buffer too small.", nameof(output));
        if (offsetsAndTiles.Size < MoeBuildTileListKernel.OffsetsBufferUints(numExperts, rows) * sizeof(uint))
            throw new ArgumentException("offsets/tile-list buffer too small.", nameof(offsetsAndTiles));
        Span<nint> buffers = stackalloc nint[4] { bank.Handle, packedInput.Handle, offsetsAndTiles.Handle, output.Handle };
        nint set = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[8] { (uint)m, (uint)k, (uint)rows, (uint)numExperts, (uint)(k / 32), (uint)RowBytes(k), (uint)(numExperts + 1), gelu ? 1u : 0u };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
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
