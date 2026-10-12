using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Split-KV (Flash-Decoding) FP32 attention for a SMALL BATCH OF ROWS that continue one sequence at consecutive positions - the
/// speculative-decoding verify forward (2..<see cref="MaxRows"/> rows, causal). Row <c>r</c> sits at <c>positionOffset + r</c> and attends to the
/// <c>positionOffset + r + 1</c> keys up to and including itself.
/// </summary>
/// <remarks>
/// <para>
/// Each row is computed by the same shader code, with the same split count and the same chunk boundaries, as the 1-row decode kernel
/// (<see cref="VulkanSplitKvAttentionKernel"/>) would use for a decode step at that length, so a row is meant to be bit-identical to that
/// decode step (the f16 coopmat flash attention that every <c>seqQ &gt; 1</c> forward used before rounds Q, K and V to f16 and accumulates in a different order).
/// The rows are independent workgroups that all read the shared KV prefix (one pass over the KV bytes per row, served from the caches after the first).
/// </para>
/// <para>Requires every row to split (<see cref="CanServe"/>), which only fails at contexts of 16 tokens or fewer; the caller falls back to the flash kernels there.</para>
/// </remarks>
public sealed class VulkanSplitKvRowsAttentionKernel : IDisposable
{
    /// <summary>Most rows one dispatch serves (bounds the partial scratch).</summary>
    public const int MaxRows = 8;

    private const int SplitPushConstantBytes = 15 * sizeof(uint);
    private const int MergePushConstantBytes = 7 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _splitModule, _mergeModule;
    private readonly ComputePipeline _splitPipeline, _mergePipeline;
    private readonly nint _splitPool, _mergePool;
    private readonly DescriptorSetCache _splitCache, _mergeCache;
    private VulkanDevice.Buffer? _partOut, _partMS;
    private long _partOutFloats, _partMSFloats;
    private bool _disposed;

    /// <summary>Dispatch counter (test hook: proves the path ran).</summary>
    internal long RecordCount { get; private set; }

    private VulkanSplitKvRowsAttentionKernel(VulkanDevice device, VulkanModule splitModule, ComputePipeline splitPipeline, nint splitPool,
        VulkanModule mergeModule, ComputePipeline mergePipeline, nint mergePool)
    {
        _device = device;
        _splitModule = splitModule; _splitPipeline = splitPipeline; _splitPool = splitPool;
        _mergeModule = mergeModule; _mergePipeline = mergePipeline; _mergePool = mergePool;
        _splitCache = new DescriptorSetCache(device, splitPool, splitPipeline, buffersPerSet: 5);
        _mergeCache = new DescriptorSetCache(device, mergePool, mergePipeline, buffersPerSet: 3);
    }

    /// <summary>Loads the rows split + merge SPIR-V; null when either is missing or the pipeline cannot be created.</summary>
    public static VulkanSplitKvRowsAttentionKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        string splitPath = Path.Combine(spvDir, "attention_f32_splitkv_rows.spv");
        string mergePath = Path.Combine(spvDir, "attention_f32_splitkv_rows_merge.spv");
        if (!File.Exists(splitPath) || !File.Exists(mergePath)) return null;
        VulkanModule? splitModule = null, mergeModule = null;
        ComputePipeline? splitPipeline = null, mergePipeline = null;
        try
        {
            splitModule = VulkanModule.LoadFromFile(device, splitPath);
            splitPipeline = CreatePipeline(splitModule, 5, SplitPushConstantBytes);
            mergeModule = VulkanModule.LoadFromFile(device, mergePath);
            mergePipeline = CreatePipeline(mergeModule, 3, MergePushConstantBytes);
        }
        catch (VulkanException)
        {
            mergePipeline?.Dispose(); mergeModule?.Dispose(); splitPipeline?.Dispose(); splitModule?.Dispose();
            return null;
        }
        nint splitPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        nint mergePool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new VulkanSplitKvRowsAttentionKernel(device, splitModule, splitPipeline, splitPool, mergeModule, mergePipeline, mergePool);
    }

    private static ComputePipeline CreatePipeline(VulkanModule module, int bindingCount, int pushConstantBytes)
    {
        Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[bindingCount];
        for (int i = 0; i < bindingCount; i++) bindings[i] = new VkDescriptorBinding((uint)i);
        return module.CreateComputePipeline("main", bindings, (uint)pushConstantBytes);
    }

    /// <summary>True when this kernel can serve <paramref name="seqQ"/> rows starting at <paramref name="positionOffset"/>: every row must split exactly as the 1-row kernel does (&gt;= 2 splits).</summary>
    public static bool CanServe(int seqQ, int positionOffset, int numHeads)
        => seqQ >= 2 && seqQ <= MaxRows && VulkanSplitKvAttentionKernel.WouldSplit(positionOffset + 1, numHeads);

    /// <summary>Drops every cached descriptor set.</summary>
    internal void InvalidateDescriptorCache() { _splitCache.Reset(); _mergeCache.Reset(); }

    /// <summary>
    /// Records the two passes. Buffers as for <see cref="VulkanSplitKvAttentionKernel.Record"/> with <c>seqQ</c> rows: Q / output are <c>[seqQ, numHeads * headDim]</c>,
    /// K / V hold at least <c>positionOffset + seqQ</c> positions.
    /// </summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        int seqQ, int numHeads, int numKvHeads, int headDim, int positionOffset)
    {
        if (!CanServe(seqQ, positionOffset, numHeads)) throw new ArgumentException("rows not servable (see CanServe).");
        if (numHeads % numKvHeads != 0) throw new ArgumentException("numHeads must be divisible by numKvHeads.");
        if (headDim <= 0 || headDim > VulkanSplitKvAttentionKernel.MaxHeadDim) throw new ArgumentOutOfRangeException(nameof(headDim));
        int seqKv = positionOffset + seqQ;
        long qBytes = (long)seqQ * numHeads * headDim * sizeof(float);
        long kvBytes = (long)seqKv * numKvHeads * headDim * sizeof(float);
        if (q.Size < qBytes || output.Size < qBytes) throw new ArgumentException("Q / output buffer too small.");
        if (k.Size < kvBytes || v.Size < kvBytes) throw new ArgumentException("K / V buffer too small.");

        int maxSplits = VulkanSplitKvAttentionKernel.MaxSplitsFor(numHeads);   // every row's split count is <= this
        EnsureScratch(numHeads, maxSplits);
        RecordCount++;

        if (_device.ActiveHazards is null) KernelSupport.ComputeToComputeBarrier(cmdBuf);

        Span<nint> splitBuffers = stackalloc nint[5] { q.Handle, k.Handle, v.Handle, _partOut!.Handle, _partMS!.Handle };
        nint splitSet = _splitCache.GetOrCreate(splitBuffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _splitPipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _splitPipeline.Layout, 0, 1, splitSet, 0, 0);
        Span<uint> spc = stackalloc uint[15];
        spc[0] = (uint)seqQ;
        spc[1] = (uint)seqKv;
        spc[2] = (uint)numHeads;
        spc[3] = (uint)numKvHeads;
        spc[4] = (uint)headDim;
        spc[5] = (uint)positionOffset;
        spc[6] = 0;                                   // slidingWindow
        spc[7] = 0;                                   // alibi
        spc[8] = BitConverter.SingleToUInt32Bits(0f); // softCap
        spc[9] = BitConverter.SingleToUInt32Bits(0f); // scaleOverride
        spc[10] = 0;                                  // causal
        spc[11] = 0;
        spc[12] = (uint)maxSplits;
        spc[13] = (uint)VulkanSplitKvAttentionKernel.TargetWorkgroups;
        spc[14] = (uint)VulkanSplitKvAttentionKernel.MinKvPerSplit;
        fixed (uint* p = spc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _splitPipeline.Layout, VkShaderStageFlags.Compute, 0, SplitPushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)(seqQ * numHeads * maxSplits), 1, 1);

        if (_device.ActiveHazards is null) KernelSupport.ComputeToComputeBarrier(cmdBuf);

        Span<nint> mergeBuffers = stackalloc nint[3] { _partOut!.Handle, _partMS!.Handle, output.Handle };
        nint mergeSet = _mergeCache.GetOrCreate(mergeBuffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _mergePipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _mergePipeline.Layout, 0, 1, mergeSet, 0, 0);
        Span<uint> mpc = stackalloc uint[7];
        mpc[0] = (uint)numHeads;
        mpc[1] = (uint)headDim;
        mpc[2] = (uint)maxSplits;
        mpc[3] = (uint)positionOffset;
        mpc[4] = (uint)VulkanSplitKvAttentionKernel.TargetWorkgroups;
        mpc[5] = (uint)VulkanSplitKvAttentionKernel.MinKvPerSplit;
        mpc[6] = (uint)seqQ;
        fixed (uint* p = mpc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _mergePipeline.Layout, VkShaderStageFlags.Compute, 0, MergePushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)(seqQ * numHeads), 1, 1);
    }

    private void EnsureScratch(int numHeads, int maxSplits)
    {
        // Sized on bounds (MaxRows, the largest head dim), never on this call's values, so it never grows while a command buffer is open.
        long needOut = (long)MaxRows * numHeads * maxSplits * VulkanSplitKvAttentionKernel.MaxHeadDim;
        long needMS = (long)MaxRows * numHeads * maxSplits * 2;
        if (_partOut is not null && needOut <= _partOutFloats && _partMS is not null && needMS <= _partMSFloats) return;
        _partOut?.Dispose(); _partMS?.Dispose();
        _partOut = _device.AllocateDeviceLocal(needOut * sizeof(float));
        _partMS = _device.AllocateDeviceLocal(needMS * sizeof(float));
        _partOutFloats = needOut; _partMSFloats = needMS;
        InvalidateDescriptorCache();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_splitPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _splitPool, 0);
        if (_mergePool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _mergePool, 0);
        _partOut?.Dispose(); _partMS?.Dispose();
        _mergePipeline.Dispose(); _mergeModule.Dispose(); _splitPipeline.Dispose(); _splitModule.Dispose();
    }
}
