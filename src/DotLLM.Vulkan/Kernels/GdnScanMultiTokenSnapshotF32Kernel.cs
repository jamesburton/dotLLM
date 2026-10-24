using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// A multi-token GDN scan (the <see cref="GdnScanMultiTokenF32Kernel.Variant.RowSplit"/> or
/// <see cref="GdnScanMultiTokenF32Kernel.Variant.LdsFused"/> arm, matching the model's decode scan, #1220)
/// plus per-row state snapshots (issue #473): after each of the first <c>snapRows</c> tokens the
/// whole <c>[NVHead, DState, DState]</c> state is also written to a snapshot buffer, row <c>t</c>
/// at element offset <c>t · NVHead · DState²</c>.
/// </summary>
/// <remarks>
/// Used only by a speculative verify forward, so a partial rejection can roll the recurrence back
/// to row <c>accepted</c> with a copy instead of replaying the accepted prefix. A separate pipeline
/// rather than an extra binding on the shipping kernel, so every non-verify forward runs exactly
/// the code it ran before. Bit-exact against the arm it mirrors: the snapshot is the f32 the arm already
/// holds, and the arithmetic is unchanged. The arms differ from each other by re-association ULPs, so the
/// verify must mirror the arm decode runs (row-split by default) for a rolled-back state to equal plain decode's.
/// </remarks>
public sealed class GdnScanMultiTokenSnapshotF32Kernel : IDisposable
{
    // n_v_head, n_k_head, d_state, v_heads_per_k_head, seq_len, snap_rows (all u32)
    private const int PushConstantBytes = 6 * sizeof(uint);
    private const int Bindings = 8;
    private const int ColsPerGroup = 32;   // must match COLS in the LDS shader
    private const int RowSplitColsPerGroup = 16;   // must match COLS in the row-split shader (which needs d_state == 128)
    private const string SpvName = "gdn_scan_multi_token_lds_fused_snap_f32.spv";
    private const string RowSplitSpvName = "gdn_scan_multi_token_rs_snap_f32.spv";

    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly VulkanDevice _device;
    private readonly bool _rowSplit;
    private bool _disposed;

    /// <summary>True when this is the row-split arm (d_state 128 only).</summary>
    public bool IsRowSplit => _rowSplit;

    private GdnScanMultiTokenSnapshotF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, bool rowSplit)
    {
        _device = device;
        _rowSplit = rowSplit;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: Bindings);
    }

    /// <summary>True when the snapshot kernel's SPIR-V is present in <paramref name="spvDir"/>.</summary>
    public static bool IsAvailable(string spvDir) => File.Exists(Path.Combine(spvDir, SpvName));

    /// <summary>Loads the pipeline from <paramref name="spvDir"/>.</summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V.</param>
    /// <param name="decodeVariant">The arm the model's decode scan runs; the snapshot kernel mirrors it when it can (row-split at
    /// <paramref name="dState"/> 128 when its SPIR-V is present, LDS-fused otherwise).</param>
    /// <param name="dState">The GDN state width.</param>
    /// <returns>The kernel.</returns>
    public static GdnScanMultiTokenSnapshotF32Kernel Create(VulkanDevice device, string spvDir,
        GdnScanMultiTokenF32Kernel.Variant decodeVariant = GdnScanMultiTokenF32Kernel.Variant.LdsFused, int dState = 128)
    {
        bool rowSplit = decodeVariant == GdnScanMultiTokenF32Kernel.Variant.RowSplit && dState == 128
                        && File.Exists(Path.Combine(spvDir, RowSplitSpvName));
        string path = Path.Combine(spvDir, rowSplit ? RowSplitSpvName : SpvName);
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[Bindings];
            for (int i = 0; i < Bindings; i++) bindings[i] = new VkDescriptorBinding((uint)i);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: Bindings);
        return new GdnScanMultiTokenSnapshotF32Kernel(device, module, pipeline, pool, rowSplit);
    }

    /// <summary>Drops every cached descriptor set; call when bound buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Records the scan. Same buffers and shapes as <see cref="GdnScanMultiTokenF32Kernel.Record"/>,
    /// plus <paramref name="snapshots"/>, which must hold at least
    /// <c>snapRows · nVHead · dState²</c> floats.
    /// </summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer state, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v,
        VulkanDevice.Buffer g, VulkanDevice.Buffer beta, VulkanDevice.Buffer output,
        VulkanDevice.Buffer snapshots, int snapRows,
        int seqLen, int nVHead, int nKHead, int dState)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nVHead <= 0) throw new ArgumentOutOfRangeException(nameof(nVHead));
        if (nKHead <= 0) throw new ArgumentOutOfRangeException(nameof(nKHead));
        if (dState <= 0 || dState > 128) throw new ArgumentOutOfRangeException(nameof(dState));
        if ((uint)snapRows > (uint)seqLen) throw new ArgumentOutOfRangeException(nameof(snapRows));
        if (_rowSplit && dState != 128) throw new ArgumentOutOfRangeException(nameof(dState), "The row-split arm needs d_state == 128.");
        if (nVHead % nKHead != 0)
            throw new ArgumentException($"nVHead ({nVHead}) must be a multiple of nKHead ({nKHead}).");
        if (snapshots.Size < (long)snapRows * nVHead * dState * dState * sizeof(float))
            throw new ArgumentException("snapshot buffer too small.", nameof(snapshots));

        Span<nint> buffers = stackalloc nint[Bindings]
        {
            state.Handle, q.Handle, k.Handle, v.Handle,
            g.Handle, beta.Handle, output.Handle, snapshots.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pc = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc, (uint)nVHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc[4..], (uint)nKHead);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc[8..], (uint)dState);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc[12..], (uint)(nVHead / nKHead));
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc[16..], (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc[20..], (uint)snapRows);
        fixed (byte* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        int cols = _rowSplit ? RowSplitColsPerGroup : ColsPerGroup;
        uint groupsY = (uint)((dState + cols - 1) / cols);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)nVHead, groupsY, 1);
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
