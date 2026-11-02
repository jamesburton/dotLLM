using DotLLM.Core.Attention;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Causal scaled-dot-product attention with a position per query row (issue #1531 step 3): row <c>t</c> attends cache rows
/// <c>[0, positions[t]]</c> (and the sliding window, if any). The query rows need not be consecutive and the KV length may exceed
/// the last query position, so a non-contiguous subset of an already-filled cache is recomputed in ONE dispatch.
/// </summary>
/// <remarks>
/// This is the per-token online-softmax kernel (<c>attention_f32.glsl</c> compiled with <c>ATTN_QPOS</c>), one workgroup per
/// (query row, head): the contiguous-row flash/coopmat prefill kernels keep their <c>positionOffset + row</c> mask. It is meant for
/// the small selected subsets of KV reuse, not for bulk prefill. Reads F32 or F16 KV in place.
/// </remarks>
public sealed class AttentionQueryPositionsKernel : IDisposable
{
    /// <summary>Largest supported head dimension (shader bound).</summary>
    public const int MaxHeadDim = AttentionF32Kernel.MaxHeadDim;

    private const int PushConstantBytes = 12 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _pool;
    private readonly DescriptorSetCache _cache;
    private bool _disposed;

    /// <summary>KV element format this instance reads.</summary>
    public VulkanKvStorage KvStorage { get; }

    private AttentionQueryPositionsKernel(VulkanDevice device, string spvDir, VulkanKvStorage storage)
    {
        _device = device;
        KvStorage = storage;
        string path = Path.Combine(spvDir, storage.ShaderName("attention_f32_qpos") + ".spv");
        if (!File.Exists(path)) throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}.");
        _module = VulkanModule.LoadFromFile(device, path);
        var bindings = new VkDescriptorBinding[]
        {
            new(0), new(1), new(2), new(3), new(4),
        };
        _pipeline = _module.CreateComputePipeline("main", bindings, PushConstantBytes);
        _pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        _cache = new DescriptorSetCache(device, _pool, _pipeline, buffersPerSet: 5);
    }

    /// <summary>Loads the pipeline for <paramref name="storage"/> KV from <paramref name="spvDir"/>.</summary>
    public static AttentionQueryPositionsKernel Create(VulkanDevice device, string spvDir, VulkanKvStorage storage = VulkanKvStorage.F32)
        => new(device, spvDir, storage);

    /// <summary>Drops cached descriptor sets; call when scratch buffers were re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _cache.Reset();

    /// <summary>Records the attention dispatch. <paramref name="queryPositions"/> is an int32 device buffer with one position per query row.</summary>
    public unsafe void Record(nint cmdBuf,
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output, VulkanDevice.Buffer queryPositions,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int slidingWindow = 0, float softCap = 0.0f, float scaleOverride = 0.0f)
    {
        if (seqQ <= 0) throw new ArgumentOutOfRangeException(nameof(seqQ));
        if (seqKv <= 0) throw new ArgumentOutOfRangeException(nameof(seqKv));
        if (numHeads <= 0 || numKvHeads <= 0 || numHeads % numKvHeads != 0) throw new ArgumentException("numHeads must be a positive multiple of numKvHeads.");
        if (headDim <= 0 || headDim > MaxHeadDim) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (slidingWindow < 0 || softCap < 0 || scaleOverride < 0) throw new ArgumentOutOfRangeException(nameof(slidingWindow));
        long qBytes = (long)seqQ * numHeads * headDim * sizeof(float);
        long kvBytes = (long)seqKv * numKvHeads * headDim * KvStorage.ElementBytes();
        if (q.Size < qBytes || output.Size < qBytes) throw new ArgumentException("Q/output buffer too small.");
        if (k.Size < kvBytes || v.Size < kvBytes) throw new ArgumentException("K/V buffer too small.");
        if (queryPositions.Size < (long)seqQ * sizeof(int)) throw new ArgumentException("Positions buffer too small.", nameof(queryPositions));

        Span<nint> buffers = stackalloc nint[5] { q.Handle, k.Handle, v.Handle, output.Handle, queryPositions.Handle };
        nint set = _cache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);

        // Same push-constant block as attention_f32.comp: positionOffset (slot 5) is unused with ATTN_QPOS, maskMode is Causal.
        Span<uint> pc = stackalloc uint[12];
        pc[0] = (uint)seqQ; pc[1] = (uint)seqKv; pc[2] = (uint)numHeads; pc[3] = (uint)numKvHeads; pc[4] = (uint)headDim;
        pc[5] = 0; pc[6] = (uint)slidingWindow; pc[7] = 0;
        pc[8] = BitConverter.SingleToUInt32Bits(softCap); pc[9] = BitConverter.SingleToUInt32Bits(scaleOverride);
        pc[10] = (uint)AttentionMaskMode.Causal; pc[11] = 0;
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)seqQ * (uint)numHeads, 1, 1);
    }

    /// <summary>Synchronous wrapper over <see cref="Record"/>.</summary>
    public void Launch(VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output, VulkanDevice.Buffer queryPositions,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int slidingWindow = 0, float softCap = 0.0f, float scaleOverride = 0.0f)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, q, k, v, output, queryPositions, seqQ, seqKv, numHeads, numKvHeads, headDim, slidingWindow, softCap, scaleOverride);
        ctx.SubmitAndWait();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
