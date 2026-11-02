using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Query-relevance probe on the device (issue #1531 step 4; ProphetKV stage I). Attached to a <see cref="VulkanTransformerModel"/>, it
/// records at every layer of a cached forward, after the query rows' K/V were written: the post-RoPE queries' causal attention
/// probabilities over the cache's keys, accumulated per context key as the mean over layers, query rows and heads. The scores never
/// leave the device until <see cref="ReadScores"/>; the Q tensor is never read back.
/// </summary>
/// <remarks>
/// Two deterministic passes per layer (no float atomics): a per-(row, head) log-sum-exp, then one thread per key that sums
/// <c>exp(q·k·scale − lse)</c> over the rows that can see it. K is read in place (F32 or F16 cache). Usage:
/// <code>
/// probe.Begin(contextLength, queryRows, numHeads);
/// model.AttachQueryRelevanceProbe(probe);
/// model.Forward(queryIds, queryPositions, 0, cache);   // positions contiguous from contextLength, or any (qpos kernel)
/// model.AttachQueryRelevanceProbe(null);
/// float[] score = probe.ReadScores();                  // [contextLength]
/// cache.Rollback(contextLength);                       // drop the query rows' K/V, as the CPU reference does
/// </code>
/// Attention <c>scale</c> is the model's (1/√headDim, or its override). Only the query rows of one <c>Forward</c> are scored per
/// <see cref="Begin"/>; chunked query forwards keep accumulating until the next <see cref="Begin"/>.
/// </remarks>
public sealed class VulkanQueryRelevanceProbe : IDisposable
{
    private const int LsePushBytes = 6 * sizeof(uint);
    private readonly VulkanDevice _device;
    private readonly VulkanModule _lseModule, _accumModule;
    private readonly ComputePipeline _lsePipeline, _accumPipeline;
    private readonly nint _pool;
    private readonly DescriptorSetCache _lseCache, _accumCache;
    private VulkanDevice.Buffer? _acc, _lse;
    private int _nKeys;
    private double _rowLayerCount;
    private bool _disposed;

    /// <summary>KV element format the probe reads (must match the cache it probes).</summary>
    public VulkanKvStorage KvStorage { get; }

    private VulkanQueryRelevanceProbe(VulkanDevice device, string spvDir, VulkanKvStorage storage)
    {
        _device = device;
        KvStorage = storage;
        _lseModule = Load(device, spvDir, storage.ShaderName("query_relevance_lse_f32"));
        _accumModule = Load(device, spvDir, storage.ShaderName("query_relevance_accum_f32"));
        _lsePipeline = _lseModule.CreateComputePipeline("main", Bindings(5), LsePushBytes);
        _accumPipeline = _accumModule.CreateComputePipeline("main", Bindings(6), LsePushBytes);
        _pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 6);
        _lseCache = new DescriptorSetCache(device, _pool, _lsePipeline, buffersPerSet: 5);
        _accumCache = new DescriptorSetCache(device, _pool, _accumPipeline, buffersPerSet: 6);
    }

    private static VulkanModule Load(VulkanDevice device, string spvDir, string name)
    {
        string path = Path.Combine(spvDir, name + ".spv");
        if (!File.Exists(path)) throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}.");
        return VulkanModule.LoadFromFile(device, path);
    }

    private static VkDescriptorBinding[] Bindings(int n)
    {
        var b = new VkDescriptorBinding[n];
        for (int i = 0; i < n; i++) b[i] = new VkDescriptorBinding((uint)i);
        return b;
    }

    /// <summary>Creates the probe for caches of <paramref name="storage"/> format.</summary>
    public static VulkanQueryRelevanceProbe Create(VulkanDevice device, string spvDir, VulkanKvStorage storage = VulkanKvStorage.F32)
        => new(device, spvDir, storage);

    /// <summary>
    /// Starts a probe over keys <c>[0, contextKeys)</c> for a forward of up to <paramref name="maxQueryRows"/> rows: zeroes the scores and sizes the scratch.
    /// Call before the forward, outside any recording.
    /// </summary>
    public void Begin(int contextKeys, int maxQueryRows, int numHeads)
    {
        if (contextKeys <= 0) throw new ArgumentOutOfRangeException(nameof(contextKeys));
        if (maxQueryRows <= 0 || numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(maxQueryRows));
        _acc?.Dispose();
        _lse?.Dispose();
        _lseCache.Reset();
        _accumCache.Reset();
        _acc = _device.Allocate((long)contextKeys * sizeof(float));
        _lse = _device.Allocate((long)maxQueryRows * numHeads * sizeof(float));
        _device.Upload(new float[contextKeys], _acc);
        _nKeys = contextKeys;
        _rowLayerCount = 0;
    }

    /// <summary>Mean attention probability per context key over layers, query rows and heads; sums to at most 1 per key-mass of the context.</summary>
    public float[] ReadScores()
    {
        if (_acc is null) throw new InvalidOperationException("Begin was not called.");
        var scores = new float[_nKeys];
        _device.Download(_acc, scores);
        if (_rowLayerCount > 0)
            for (int j = 0; j < scores.Length; j++) scores[j] = (float)(scores[j] / _rowLayerCount);
        return scores;
    }

    /// <summary>Records one layer's probe into <paramref name="cmdBuf"/>: after the layer's attention read of <paramref name="q"/> and <paramref name="k"/>, before their next write.</summary>
    internal unsafe void RecordLayer(nint cmdBuf, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer positions,
        int rows, int numHeads, int numKvHeads, int headDim, float scale)
    {
        if (_acc is null || _lse is null) throw new InvalidOperationException("Begin was not called.");
        if ((long)rows * numHeads * sizeof(float) > _lse.Size) throw new ArgumentOutOfRangeException(nameof(rows), "more query rows than Begin sized for.");
        Span<byte> pc = stackalloc byte[LsePushBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc[0..], (uint)rows);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc[4..], (uint)numHeads);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc[8..], (uint)numKvHeads);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc[12..], (uint)headDim);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc[16..], (uint)_nKeys);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pc[20..], scale);

        Span<nint> b5 = stackalloc nint[5] { q.Handle, k.Handle, v.Handle, positions.Handle, _lse.Handle };
        Dispatch(cmdBuf, _lsePipeline, _lseCache.GetOrCreate(b5), pc, (uint)(rows * numHeads));
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        Span<nint> b6 = stackalloc nint[6] { q.Handle, k.Handle, v.Handle, positions.Handle, _lse.Handle, _acc.Handle };
        Dispatch(cmdBuf, _accumPipeline, _accumCache.GetOrCreate(b6), pc, (uint)((_nKeys + 63) / 64));
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        _rowLayerCount += rows;
    }

    private static unsafe void Dispatch(nint cmdBuf, ComputePipeline pipeline, nint set, ReadOnlySpan<byte> pc, uint groups)
    {
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, pipeline.Layout, 0, 1, set, 0, 0);
        fixed (byte* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, pipeline.Layout, VkShaderStageFlags.Compute, 0, LsePushBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);
    }

    /// <summary>Synchronous single-layer probe for tests and offline use: records and submits one layer over explicit buffers.</summary>
    internal void ProbeLayer(VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer positions,
        int rows, int numHeads, int numKvHeads, int headDim, float scale)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        RecordLayer(ctx.CommandBuffer, q, k, v, positions, rows, numHeads, numKvHeads, headDim, scale);
        ctx.SubmitAndWait();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _acc?.Dispose();
        _lse?.Dispose();
        if (_pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
        _lsePipeline.Dispose(); _accumPipeline.Dispose();
        _lseModule.Dispose(); _accumModule.Dispose();
    }
}
