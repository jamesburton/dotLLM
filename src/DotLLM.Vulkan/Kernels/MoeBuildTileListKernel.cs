using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Builds the grouped-by-expert MoE work list (one entry per 16-row tile that actually exists) and the indirect dispatch
/// arguments for it, from the group offsets. See <c>moe_build_tile_list.comp</c> for why: the launch grid the grouped GEMMs used
/// before (every expert x every possible row tile) was ~30x larger than the work.
/// </summary>
public sealed class MoeBuildTileListKernel : IDisposable
{
    private const int PushConstantBytes = 3 * sizeof(uint);   // numExperts, mTiles0, mTiles1

    /// <summary>Byte distance between the two dispatch-argument triples the build writes.</summary>
    public const int ArgsStrideBytes = 3 * sizeof(uint);

    /// <summary>Rows per tile the list is built for (must match the grouped GEMM shaders' row tile).</summary>
    public const int TileRows = 16;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeBuildTileListKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 2);
    }

    /// <summary>True when the SPIR-V is present.</summary>
    public static bool IsSupportedOn(string spvDir) => File.Exists(Path.Combine(spvDir, "moe_build_tile_list.spv"));

    /// <summary>Creates the kernel.</summary>
    public static MoeBuildTileListKernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_build_tile_list.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");
        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[2];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 2);
        return new MoeBuildTileListKernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Number of uints the offsets buffer must hold for <paramref name="numExperts"/> experts and <paramref name="rows"/> routed rows.</summary>
    public static long OffsetsBufferUints(int numExperts, int rows)
        => (long)(numExperts + 1) + 2L * ((long)numExperts + (rows + TileRows - 1) / TileRows);

    /// <summary>
    /// Records the build. <paramref name="offsetsAndTiles"/> holds the group offsets in <c>[0, numExperts]</c> and receives the tile
    /// list from <c>numExperts + 1</c> on (size it with <see cref="OffsetsBufferUints"/>); <paramref name="args"/> receives two
    /// dispatch argument triples (6 uints): <c>(mTiles0, tileCount, 1)</c> at byte offset 0 and <c>(mTiles1, tileCount, 1)</c> at
    /// <see cref="ArgsStrideBytes"/> (gate/up and down projections have different weight-row extents).
    /// A <see cref="KernelSupport.ComputeToIndirectAndComputeBarrier"/> must separate this from the dispatches that consume them.
    /// </summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer offsetsAndTiles, VulkanDevice.Buffer args, int numExperts, int mTiles0, int mTiles1)
    {
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (mTiles0 <= 0) throw new ArgumentOutOfRangeException(nameof(mTiles0));
        if (mTiles1 <= 0) throw new ArgumentOutOfRangeException(nameof(mTiles1));
        if (args.Size < 6 * sizeof(uint)) throw new ArgumentException("args buffer too small.", nameof(args));

        Span<nint> buffers = stackalloc nint[2] { offsetsAndTiles.Handle, args.Handle };
        nint set = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[3] { (uint)numExperts, (uint)mTiles0, (uint)mTiles1 };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, 1, 1, 1);
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
