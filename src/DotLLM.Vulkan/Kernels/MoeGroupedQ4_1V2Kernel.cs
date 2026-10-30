using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Grouped Q4_1 MoE expert GEMMs, v2 (issue #1161): <c>moe_grouped_q4_1_gateup_v2</c> (gate + up + SwiGLU, activations gathered from the
/// token rows through the expert permutation) and <c>moe_grouped_q4_1_down_v2</c> (down projection written straight to the (token, slot)
/// rows, with the per-row power-of-two scale undone in the epilogue). Work items come from <see cref="MoeBuildTileListKernel"/> with
/// <see cref="RowTile"/> = 64 token rows, so each expert's weights are streamed once per 64 rows.
/// </summary>
/// <remarks>
/// Replaces, for a Q4_1 bank, the v1 sequence broadcast -> grouped copy -> gate GEMM -> up GEMM -> SwiGLU -> down GEMM -> row-scale -> ungroup
/// with gate/up GEMM -> down GEMM. Numerics are those of <c>moe_grouped_matmul_q4_1_coopmat_m64</c> (F16 operands, F32 accumulation in
/// the same K order). Requires cooperative matrices and native wave64 (the epilogue maps one 16-column fragment per 64-lane wave).
/// </remarks>
public sealed class MoeGroupedQ4_1V2Kernel : IDisposable
{
    /// <summary>SPIR-V of the gate + up + SwiGLU pass.</summary>
    public const string GateUpSpvFileName = "moe_grouped_q4_1_gateup_v2.spv";

    /// <summary>SPIR-V of the down pass.</summary>
    public const string DownSpvFileName = "moe_grouped_q4_1_down_v2.spv";

    /// <summary><c>DOTLLM_VK_MOE_Q4_1_V2=0</c> keeps the v1 grouped path (A/B).</summary>
    public const string EnvVar = "DOTLLM_VK_MOE_Q4_1_V2";

    /// <summary>Token rows per work item (the tile list's <c>tileRows</c>).</summary>
    public const int RowTile = 64;

    /// <summary>K granularity (two Q4_1 blocks per staging round).</summary>
    public const int KGroup = 64;

    private const int GateUpMTile = 64;
    private const int DownMTile = 128;
    private const int Bindings = 7;
    private const int PushConstantBytes = 9 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _gateUpModule, _downModule;
    private readonly ComputePipeline _gateUp, _down;
    private readonly nint _gateUpPool, _downPool;
    private readonly DescriptorSetCache _gateUpCache, _downCache;
    private bool _disposed;

    private MoeGroupedQ4_1V2Kernel(VulkanDevice device, VulkanModule gum, ComputePipeline gu, VulkanModule dm, ComputePipeline d)
    {
        _device = device;
        _gateUpModule = gum; _gateUp = gu; _downModule = dm; _down = d;
        _gateUpPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: Bindings);
        _downPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: Bindings);
        _gateUpCache = new DescriptorSetCache(device, _gateUpPool, gu, buffersPerSet: Bindings);
        _downCache = new DescriptorSetCache(device, _downPool, d, buffersPerSet: Bindings);
    }

    /// <summary>Whether the kernels can run here: coopmat, native wave64, both SPIR-V files, not disabled by <see cref="EnvVar"/>.</summary>
    /// <param name="device">Device.</param>
    /// <param name="spvDir">SPIR-V directory.</param>
    /// <returns>True when <see cref="Create"/> succeeds.</returns>
    public static bool IsSupportedOn(VulkanDevice device, string spvDir)
        => Environment.GetEnvironmentVariable(EnvVar) != "0"
           && device.HasCooperativeMatrix && device.SubgroupSize == 64
           && File.Exists(Path.Combine(spvDir, GateUpSpvFileName)) && File.Exists(Path.Combine(spvDir, DownSpvFileName));

    /// <summary>Loads both pipelines. Check <see cref="IsSupportedOn"/> first.</summary>
    /// <param name="device">Device.</param>
    /// <param name="spvDir">SPIR-V directory.</param>
    /// <returns>The kernel pair.</returns>
    public static MoeGroupedQ4_1V2Kernel Create(VulkanDevice device, string spvDir)
    {
        VulkanModule? gum = null, dm = null;
        ComputePipeline? gu = null;
        try
        {
            Span<VkDescriptorBinding> b = stackalloc VkDescriptorBinding[Bindings];
            for (int i = 0; i < Bindings; i++) b[i] = new VkDescriptorBinding((uint)i);
            gum = VulkanModule.LoadFromFile(device, Path.Combine(spvDir, GateUpSpvFileName));
            gu = gum.CreateComputePipeline("main", b, PushConstantBytes, requiredSubgroupSize: 0);
            dm = VulkanModule.LoadFromFile(device, Path.Combine(spvDir, DownSpvFileName));
            var d = dm.CreateComputePipeline("main", b, PushConstantBytes, requiredSubgroupSize: 0);
            return new MoeGroupedQ4_1V2Kernel(device, gum, gu, dm, d);
        }
        catch
        {
            gu?.Dispose();
            gum?.Dispose();
            dm?.Dispose();
            throw;
        }
    }

    /// <summary>Weight-row tiles of the gate/up pass for <paramref name="m"/> output columns (the tile list's first grid width).</summary>
    /// <param name="m">Expert intermediate width.</param>
    /// <returns>Workgroups along x.</returns>
    public static int GateUpMTiles(int m) => (m + GateUpMTile - 1) / GateUpMTile;

    /// <summary>Weight-row tiles of the down pass for <paramref name="m"/> output columns (the tile list's second grid width).</summary>
    /// <param name="m">Hidden size.</param>
    /// <returns>Workgroups along x.</returns>
    public static int DownMTiles(int m) => (m + DownMTile - 1) / DownMTile;

    internal void InvalidateDescriptorCache()
    {
        _gateUpCache.Reset();
        _downCache.Reset();
    }

    /// <summary>
    /// Records the gate + up + SwiGLU pass: <c>act[p, :] = silu(x[perm[p] / topK] . Wg[e]) * (x[perm[p] / topK] . Wu[e])</c> for every packed
    /// row <c>p</c> of expert <c>e</c>, in packed order.
    /// </summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="gateBank">Q4_1 gate bank <c>[E, m, k/32 blocks]</c>.</param>
    /// <param name="upBank">Q4_1 up bank, same shape.</param>
    /// <param name="x">Normalised token rows <c>[rows / topK, k]</c> F32.</param>
    /// <param name="offsetsAndTiles">Expert offsets followed by the tile list (built with <see cref="RowTile"/>).</param>
    /// <param name="perm">Packed row -> (token * topK + slot).</param>
    /// <param name="act">Output <c>[rows, m]</c> F32, packed order.</param>
    /// <param name="dispatchArgs">Indirect dispatch arguments from the tile-list build.</param>
    /// <param name="argsOffset">Byte offset of this pass's arguments.</param>
    /// <param name="m">Expert intermediate width.</param>
    /// <param name="k">Hidden size (multiple of 32).</param>
    /// <param name="rows">Packed rows (tokens * topK).</param>
    /// <param name="numExperts">Experts in the bank.</param>
    /// <param name="topK">Slots per token.</param>
    public void RecordGateUpIndirect(nint cmdBuf, VulkanDevice.Buffer gateBank, VulkanDevice.Buffer upBank, VulkanDevice.Buffer x,
        VulkanDevice.Buffer offsetsAndTiles, VulkanDevice.Buffer perm, VulkanDevice.Buffer act, VulkanDevice.Buffer dispatchArgs, int argsOffset,
        int m, int k, int rows, int numExperts, int topK)
    {
        if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));
        Validate(gateBank, m, k, numExperts);
        Validate(upBank, m, k, numExperts);
        if (x.Size < (long)(rows / topK) * k * sizeof(float)) throw new ArgumentException("x buffer too small.", nameof(x));
        if (act.Size < (long)rows * m * sizeof(float)) throw new ArgumentException("act buffer too small.", nameof(act));
        Record(cmdBuf, _gateUp, _gateUpCache, gateBank, upBank, x, offsetsAndTiles, perm, act, act, dispatchArgs, argsOffset,
            m, k, rows, numExperts, (uint)topK, applyScale: false);
    }

    /// <summary>
    /// Records the down pass: <c>y[perm[p], :] = (act[p] . Wd[e]) * rowScale[p]</c> (the scale only when <paramref name="rowScale"/> is given).
    /// </summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="downBank">Q4_1 down bank <c>[E, m, k/32 blocks]</c>.</param>
    /// <param name="act">Packed activation rows <c>[rows, k]</c> F32.</param>
    /// <param name="offsetsAndTiles">Expert offsets followed by the tile list.</param>
    /// <param name="perm">Packed row -> (token * topK + slot).</param>
    /// <param name="y">Output <c>[rows, m]</c> F32 in (token, slot) order.</param>
    /// <param name="rowScale">Per packed row factor, or null.</param>
    /// <param name="dispatchArgs">Indirect dispatch arguments from the tile-list build.</param>
    /// <param name="argsOffset">Byte offset of this pass's arguments.</param>
    /// <param name="m">Hidden size.</param>
    /// <param name="k">Expert intermediate width (multiple of 32).</param>
    /// <param name="rows">Packed rows.</param>
    /// <param name="numExperts">Experts in the bank.</param>
    public void RecordDownIndirect(nint cmdBuf, VulkanDevice.Buffer downBank, VulkanDevice.Buffer act, VulkanDevice.Buffer offsetsAndTiles,
        VulkanDevice.Buffer perm, VulkanDevice.Buffer y, VulkanDevice.Buffer? rowScale, VulkanDevice.Buffer dispatchArgs, int argsOffset,
        int m, int k, int rows, int numExperts)
    {
        Validate(downBank, m, k, numExperts);
        if (act.Size < (long)rows * k * sizeof(float)) throw new ArgumentException("act buffer too small.", nameof(act));
        if (y.Size < (long)rows * m * sizeof(float)) throw new ArgumentException("y buffer too small.", nameof(y));
        Record(cmdBuf, _down, _downCache, downBank, downBank, act, offsetsAndTiles, perm, y, rowScale ?? y, dispatchArgs, argsOffset,
            m, k, rows, numExperts, 1u, applyScale: rowScale is not null);
    }

    private static void Validate(VulkanDevice.Buffer bank, int m, int k, int numExperts)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0 || (k % KGroup) != 0) throw new ArgumentException($"k must be a positive multiple of {KGroup}, got {k}", nameof(k));
        long bankBytes = (long)numExperts * m * (k / 32) * 20;
        if (bankBytes > uint.MaxValue) throw new ArgumentException("bank exceeds 32-bit addressing.", nameof(bank));
        if (bank.Size < bankBytes) throw new ArgumentException("bank buffer too small.", nameof(bank));
    }

    private unsafe void Record(nint cmdBuf, ComputePipeline pipeline, DescriptorSetCache cache,
        VulkanDevice.Buffer w0, VulkanDevice.Buffer w1, VulkanDevice.Buffer x, VulkanDevice.Buffer offsets, VulkanDevice.Buffer perm,
        VulkanDevice.Buffer y, VulkanDevice.Buffer scale, VulkanDevice.Buffer dispatchArgs, int argsOffset,
        int m, int k, int rows, int numExperts, uint topK, bool applyScale)
    {
        if (offsets.Size < MoeBuildTileListKernel.OffsetsBufferUints(numExperts, rows) * sizeof(uint))
            throw new ArgumentException("offsets/tile-list buffer too small.", nameof(offsets));
        if (perm.Size < (long)rows * sizeof(uint)) throw new ArgumentException("perm buffer too small.", nameof(perm));
        if (dispatchArgs.Size < argsOffset + 3 * sizeof(uint)) throw new ArgumentException("dispatchArgs buffer too small.", nameof(dispatchArgs));
        Span<nint> buffers = stackalloc nint[Bindings] { w0.Handle, w1.Handle, x.Handle, offsets.Handle, perm.Handle, y.Handle, scale.Handle };
        nint set = cache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, pipeline.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[9]
        {
            (uint)m, (uint)k, (uint)rows, (uint)numExperts, (uint)(k / 32), (uint)((k / 32) * 20), (uint)(numExperts + 1), topK, applyScale ? 1u : 0u,
        };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatchIndirect(cmdBuf, dispatchArgs.Handle, (ulong)argsOffset);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_gateUpPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _gateUpPool, 0);
        if (_downPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _downPool, 0);
        _gateUp.Dispose();
        _down.Dispose();
        _gateUpModule.Dispose();
        _downModule.Dispose();
    }
}
