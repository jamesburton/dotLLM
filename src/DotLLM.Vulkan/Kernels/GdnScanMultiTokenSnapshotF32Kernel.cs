using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// The shipping multi-token GDN scan (<see cref="GdnScanMultiTokenF32Kernel.Variant.LdsFused"/>)
/// plus per-row state snapshots (issue #473): after each of the first <c>snapRows</c> tokens the
/// whole <c>[NVHead, DState, DState]</c> state is also written to a snapshot buffer, row <c>t</c>
/// at element offset <c>t · NVHead · DState²</c>.
/// </summary>
/// <remarks>
/// Used only by a speculative verify forward, so a partial rejection can roll the recurrence back
/// to row <c>accepted</c> with a copy instead of replaying the accepted prefix. A separate pipeline
/// rather than an extra binding on the shipping kernel, so every non-verify forward runs exactly
/// the code it ran before. Bit-exact against the shipping kernel: the snapshot is the f32 already
/// stored to LDS, and the arithmetic is unchanged.
/// </remarks>
public sealed class GdnScanMultiTokenSnapshotF32Kernel : IDisposable
{
    // n_v_head, n_k_head, d_state, v_heads_per_k_head, seq_len, snap_rows (all u32)
    private const int PushConstantBytes = 6 * sizeof(uint);
    private const int Bindings = 8;
    private const int ColsPerGroup = 32;   // must match COLS in the shader
    private const string SpvName = "gdn_scan_multi_token_lds_fused_snap_f32.spv";

    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly VulkanDevice _device;
    private bool _disposed;

    private GdnScanMultiTokenSnapshotF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: Bindings);
    }

    /// <summary>True when the snapshot kernel's SPIR-V is present in <paramref name="spvDir"/>.</summary>
    public static bool IsAvailable(string spvDir) => File.Exists(Path.Combine(spvDir, SpvName));

    /// <summary>Loads the pipeline from <paramref name="spvDir"/>.</summary>
    public static GdnScanMultiTokenSnapshotF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, SpvName);
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
        return new GdnScanMultiTokenSnapshotF32Kernel(device, module, pipeline, pool);
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

        uint groupsY = (uint)((dState + ColsPerGroup - 1) / ColsPerGroup);
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
