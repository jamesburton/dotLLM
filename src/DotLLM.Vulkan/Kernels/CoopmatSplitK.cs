using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Split-K driver for the blocked 128x128x4 coopmat GEMMs (Q4_K / Q5_K / Q6_K / Q8_0). The base kernels launch one workgroup per
/// 128x128 output tile, so a small-<c>m</c> GEMM at <c>n = 128</c> (the ffn-down and attention/GDN output projections: 20-32
/// tiles) leaves half of a 40-CU part idle and runs at ~9-13 TFLOPS against ~21 for full grids. This runs the K range as
/// <c>splits</c> z-slices that each write a private slab of a scratch buffer, then sums the slabs into the real output with a
/// small reduce pass. Measured 1.1-2.2x on exactly those shapes and neutral-to-negative on full grids, hence
/// <see cref="ChooseSplits"/> only engages below ~40 tiles. Opt out with <c>DOTLLM_VK_GEMM_SPLITK=0</c>.
/// </summary>
/// <remarks>
/// The scratch is allocated ONCE at its worst case (the policy bounds <c>splits * m * n</c>) because a mid-recording growth
/// would free a buffer an earlier, still-unsubmitted dispatch references. Every split dispatch ends in a compute barrier, so
/// two split GEMMs on the same instance can never overlap on the scratch.
/// </remarks>
internal sealed class CoopmatSplitK : IDisposable
{
    /// <summary>SPIR-V of the slab reduction.</summary>
    public const string ReduceSpvFileName = "splitk_reduce_f32.spv";

    /// <summary>Env var that disables split-K (<c>0</c>) for A/B runs.</summary>
    public const string EnvVar = "DOTLLM_VK_GEMM_SPLITK";

    private const int Tile = 128;
    private const int MinChunksPerSplit = 16;              // keep each slice long enough to amortise its prologue/epilogue
    private const int GemmPushBytes = 6 * sizeof(uint);    // M, K, N, blocksPerRow, rowUints, chunksPerSplit
    private const int ReducePushBytes = 2 * sizeof(uint);  // count, splits
    private const int ReduceWorkgroup = 256;

    /// <summary>Scratch floats: the policy never exceeds 40 tiles x 2 slabs, 20 tiles x 4 or 8 tiles x 8 = 1.31 M floats.</summary>
    private const long ScratchFloats = 40L * Tile * Tile * 2;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _gemmModule, _reduceModule;
    private readonly ComputePipeline _gemm, _reduce;
    private readonly nint _gemmPool, _reducePool;
    private readonly DescriptorSetCache _gemmCache, _reduceCache;
    private readonly VulkanDevice.Buffer _scratch;
    private readonly int _blockBytes, _groupSize;
    private bool _disposed;

    private CoopmatSplitK(VulkanDevice device, VulkanModule gm, ComputePipeline g, VulkanModule rm, ComputePipeline r,
                          VulkanDevice.Buffer scratch, int blockBytes, int groupSize)
    {
        _device = device;
        _gemmModule = gm; _gemm = g; _reduceModule = rm; _reduce = r;
        _scratch = scratch;
        _blockBytes = blockBytes; _groupSize = groupSize;
        _gemmPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        _reducePool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 2);
        _gemmCache = new DescriptorSetCache(device, _gemmPool, g, buffersPerSet: 3);
        _reduceCache = new DescriptorSetCache(device, _reducePool, r, buffersPerSet: 2);
    }

    /// <summary>
    /// Creates the driver for a quant, or returns null when disabled by env, the twin SPIR-V is missing, or the device cannot run
    /// the blocked template (cooperative matrix + native wave64, same gate as the base kernel).
    /// </summary>
    /// <param name="device">Device.</param>
    /// <param name="spvDir">SPIR-V directory.</param>
    /// <param name="gemmSpvFileName">The <c>*_splitk.spv</c> twin.</param>
    /// <param name="blockBytes">Bytes per quant block (weight row stride unit).</param>
    /// <param name="groupSize">Elements per quant block (256 for K-quants, 32 for Q8_0).</param>
    public static CoopmatSplitK? TryCreate(VulkanDevice device, string spvDir, string gemmSpvFileName, int blockBytes, int groupSize)
    {
        if (Environment.GetEnvironmentVariable(EnvVar) == "0") return null;
        if (!device.HasCooperativeMatrix || device.SubgroupSize != 64) return null;
        string gp = Path.Combine(spvDir, gemmSpvFileName), rp = Path.Combine(spvDir, ReduceSpvFileName);
        if (!File.Exists(gp) || !File.Exists(rp)) return null;

        VulkanModule? gm = null, rm = null;
        try
        {
            gm = VulkanModule.LoadFromFile(device, gp);
            rm = VulkanModule.LoadFromFile(device, rp);
            Span<VkDescriptorBinding> b3 = stackalloc VkDescriptorBinding[3];
            for (int i = 0; i < 3; i++) b3[i] = new VkDescriptorBinding((uint)i);
            var g = gm.CreateComputePipeline("main", b3, GemmPushBytes, requiredSubgroupSize: 0);
            Span<VkDescriptorBinding> b2 = stackalloc VkDescriptorBinding[2];
            for (int i = 0; i < 2; i++) b2[i] = new VkDescriptorBinding((uint)i);
            var r = rm.CreateComputePipeline("main", b2, ReducePushBytes);
            var scratch = device.AllocateDeviceLocal(ScratchFloats * sizeof(float));
            return new CoopmatSplitK(device, gm, g, rm, r, scratch, blockBytes, groupSize);
        }
        catch
        {
            gm?.Dispose();
            rm?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Number of K slices for an <c>m x n</c> output with <paramref name="chunks"/> BK=32 chunks along K; 1 means "do not split".
    /// Thresholds from the Q4_K sweep on gfx1151 (40 CUs): at n=128 s4 gave 2.2x on 20 tiles, s2 1.2x on 32-40 tiles, and every
    /// split lost on 64+ tiles (the slab write + reduce is pure overhead once the grid is full).
    /// </summary>
    public static int ChooseSplits(int m, int n, int chunks)
    {
        long tiles = (long)((m + Tile - 1) / Tile) * ((n + Tile - 1) / Tile);
        int splits = tiles <= 8 ? 8 : tiles <= 20 ? 4 : tiles <= 40 ? 2 : 1;
        while (splits > 1 && chunks / splits < MinChunksPerSplit) splits >>= 1;
        return splits;
    }

    internal void InvalidateDescriptorCache()
    {
        _gemmCache.Reset();
        _reduceCache.Reset();
    }

    /// <summary>Records GEMM + barrier + reduce + barrier. <paramref name="splits"/> must come from <see cref="ChooseSplits"/>.</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer weights, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
                              int m, int k, int n, int splits)
    {
        if (splits < 2) throw new ArgumentOutOfRangeException(nameof(splits));
        int blocksPerRow = k / _groupSize;
        int chunks = _groupSize == 32 ? blocksPerRow : blocksPerRow * 8;
        int chunksPerSplit = (chunks + splits - 1) / splits;
        int used = (chunks + chunksPerSplit - 1) / chunksPerSplit;          // a trailing empty slice would reduce garbage
        long count = (long)n * m;
        if (used * count > ScratchFloats)
            throw new InvalidOperationException($"split-K scratch overflow: {used} x {count} floats > {ScratchFloats} (policy bounds tiles).");
        long rowBytes = (long)blocksPerRow * _blockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        Span<nint> bufs = stackalloc nint[3] { weights.Handle, inputB.Handle, _scratch.Handle };
        nint set = _gemmCache.GetOrCreate(bufs);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _gemm.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _gemm.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[6] { (uint)m, (uint)k, (uint)n, (uint)blocksPerRow, (uint)rowUints, (uint)chunksPerSplit };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _gemm.Layout, VkShaderStageFlags.Compute, 0, GemmPushBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + Tile - 1) / Tile), (uint)((n + Tile - 1) / Tile), (uint)used);

        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        Span<nint> rb = stackalloc nint[2] { _scratch.Handle, outputC.Handle };
        nint rset = _reduceCache.GetOrCreate(rb);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _reduce.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _reduce.Layout, 0, 1, rset, 0, 0);
        Span<uint> rpc = stackalloc uint[2] { (uint)count, (uint)used };
        fixed (uint* p = rpc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _reduce.Layout, VkShaderStageFlags.Compute, 0, ReducePushBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((count + ReduceWorkgroup - 1) / ReduceWorkgroup), 1, 1);

        // WAR protection for the shared scratch: the next split GEMM on this instance must not start before this reduce
        // has finished reading it.
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_gemmPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _gemmPool, 0);
        if (_reducePool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _reducePool, 0);
        _gemm.Dispose();
        _reduce.Dispose();
        _gemmModule.Dispose();
        _reduceModule.Dispose();
        _scratch.Dispose();
    }
}
