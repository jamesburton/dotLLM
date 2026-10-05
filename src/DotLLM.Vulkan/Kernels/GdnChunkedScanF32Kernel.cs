using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Chunked (WY-form) Gated-DeltaNet multi-token scan (issue #703): stage 1 prepares, in parallel across 64-token chunks and heads, everything
/// that does not depend on the incoming state (<c>U</c>, <c>W</c>, <c>P</c>, cumulative log-decay); stage 2 walks the chunks sequentially with
/// 3 barriers per CHUNK instead of per token. Same bindings/semantics as <see cref="GdnScanMultiTokenF32Kernel"/>; <c>d_state</c> must be 128.
/// Not bit-exact (different summation order and a triangular solve), F32-rounding-level drift. The kernel owns its scratch, grown on demand, so
/// a growing call invalidates descriptor sets (callers record all uses of one forward before the next grow).
/// </summary>
public sealed class GdnChunkedScanF32Kernel : IDisposable
{
    /// <summary>Tokens per chunk (baked into the shaders).</summary>
    public const int ChunkTokens = 64;
    private const int DState = 128;
    private const int PushConstantBytes = 4 * sizeof(int);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _prepModule, _scanModule;
    private readonly ComputePipeline _prep, _scan;
    private readonly nint _prepPool, _scanPool;
    private readonly DescriptorSetCache _prepCache, _scanCache;
    private VulkanDevice.Buffer? _u, _w, _p, _lg;
    private long _scratchSlots;
    private bool _disposed;
    /// <summary>Diagnostic: bit 0 = run stage 1, bit 1 = run stage 2 (probe attribution only).</summary>
    internal int StageMask { get; set; } = 3;

    private GdnChunkedScanF32Kernel(VulkanDevice device, VulkanModule pm, ComputePipeline prep, VulkanModule sm, ComputePipeline scan)
    {
        _device = device;
        _prepModule = pm; _prep = prep; _scanModule = sm; _scan = scan;
        _prepPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 9);
        _scanPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 8);
        _prepCache = new DescriptorSetCache(device, _prepPool, prep, buffersPerSet: 9);
        _scanCache = new DescriptorSetCache(device, _scanPool, scan, buffersPerSet: 8);
    }

    /// <summary>True when both SPIR-V files are present.</summary>
    public static bool IsSupportedOn(string spvDir)
        => File.Exists(Path.Combine(spvDir, "gdn_chunk_prep_f32.spv")) && File.Exists(Path.Combine(spvDir, "gdn_chunk_scan_f32.spv"));

    /// <summary>Creates the kernel pair.</summary>
    public static GdnChunkedScanF32Kernel Create(VulkanDevice device, string spvDir)
    {
        var pm = VulkanModule.LoadFromFile(device, Path.Combine(spvDir, "gdn_chunk_prep_f32.spv"));
        var sm = VulkanModule.LoadFromFile(device, Path.Combine(spvDir, "gdn_chunk_scan_f32.spv"));
        Span<VkDescriptorBinding> b9 = stackalloc VkDescriptorBinding[9];
        for (int i = 0; i < 9; i++) b9[i] = new VkDescriptorBinding((uint)i);
        Span<VkDescriptorBinding> b8 = stackalloc VkDescriptorBinding[8];
        for (int i = 0; i < 8; i++) b8[i] = new VkDescriptorBinding((uint)i);
        var prep = pm.CreateComputePipeline("main", b9, PushConstantBytes);
        var scan = sm.CreateComputePipeline("main", b8, PushConstantBytes);
        return new GdnChunkedScanF32Kernel(device, pm, prep, sm, scan);
    }

    internal void InvalidateDescriptorCache() { _prepCache.Reset(); _scanCache.Reset(); }

    private void EnsureScratch(int nVHead, int nChunks)
    {
        long slots = (long)nVHead * nChunks;
        if (_u is not null && _scratchSlots >= slots) return;
        _u?.Dispose(); _w?.Dispose(); _p?.Dispose(); _lg?.Dispose();
        _u = _device.AllocateDeviceLocal(slots * ChunkTokens * DState * sizeof(float));
        _w = _device.AllocateDeviceLocal(slots * ChunkTokens * DState * sizeof(float));
        _p = _device.AllocateDeviceLocal(slots * ChunkTokens * ChunkTokens * sizeof(float));
        _lg = _device.AllocateDeviceLocal(slots * ChunkTokens * sizeof(float));
        _scratchSlots = slots;
        InvalidateDescriptorCache();
    }

    /// <summary>Synchronous launch (tests).</summary>
    public void Launch(VulkanDevice.Buffer state, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v,
        VulkanDevice.Buffer g, VulkanDevice.Buffer beta, VulkanDevice.Buffer output, int seqLen, int nVHead, int nKHead)
    {
        EnsureScratch(nVHead, (seqLen + ChunkTokens - 1) / ChunkTokens);
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, state, q, k, v, g, beta, output, seqLen, nVHead, nKHead);
        ctx.SubmitAndWait();
    }

    /// <summary>Ensures scratch for a forward of up to <paramref name="seqLen"/> tokens BEFORE any command buffer is recorded.</summary>
    public void Reserve(int seqLen, int nVHead) => EnsureScratch(nVHead, (seqLen + ChunkTokens - 1) / ChunkTokens);

    /// <summary>Records both stages (with the compute barrier between them). Scratch must already cover the call (see <see cref="Reserve"/>).</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer state, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v,
        VulkanDevice.Buffer g, VulkanDevice.Buffer beta, VulkanDevice.Buffer output, int seqLen, int nVHead, int nKHead)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nVHead % nKHead != 0) throw new ArgumentException("nVHead must be a multiple of nKHead.");
        int nChunks = (seqLen + ChunkTokens - 1) / ChunkTokens;
        if (_u is null || _scratchSlots < (long)nVHead * nChunks) throw new InvalidOperationException("scratch not reserved for this call.");

        Span<int> pc = stackalloc int[4] { nVHead, nKHead, seqLen, nChunks };
        Span<nint> b9 = stackalloc nint[9] { q.Handle, k.Handle, v.Handle, g.Handle, beta.Handle, _u.Handle, _w!.Handle, _p!.Handle, _lg!.Handle };
        nint set1 = _prepCache.GetOrCreate(b9);
        if ((StageMask & 1) != 0) {
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _prep.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _prep.Layout, 0, 1, set1, 0, 0);
        fixed (int* p = pc) VulkanApi.vkCmdPushConstants(cmdBuf, _prep.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)nChunks, (uint)nVHead, 1);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
        if ((StageMask & 2) == 0) return;

        Span<nint> b8 = stackalloc nint[8] { state.Handle, q.Handle, k.Handle, _u.Handle, _w.Handle, _p.Handle, _lg.Handle, output.Handle };
        nint set2 = _scanCache.GetOrCreate(b8);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _scan.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _scan.Layout, 0, 1, set2, 0, 0);
        fixed (int* p = pc) VulkanApi.vkCmdPushConstants(cmdBuf, _scan.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)nVHead, DState / 16, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _u?.Dispose(); _w?.Dispose(); _p?.Dispose(); _lg?.Dispose();
        if (_prepPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _prepPool, 0);
        if (_scanPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _scanPool, 0);
        _prep.Dispose(); _scan.Dispose(); _prepModule.Dispose(); _scanModule.Dispose();
    }
}
