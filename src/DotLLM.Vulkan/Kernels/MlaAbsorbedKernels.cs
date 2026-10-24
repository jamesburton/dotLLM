using System.Buffers.Binary;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Common plumbing for the four absorbed-MLA compute kernels (issue #913): one SPIR-V module, one pipeline, one
/// descriptor-set cache, a fixed push-constant block. Each concrete kernel only supplies its shader name, binding count and
/// push-constant size and writes its own <c>Record</c>.
/// </summary>
public abstract class MlaAbsorbedKernelBase : IDisposable
{
    /// <summary>Total dispatches recorded by every absorbed-MLA kernel since process start (tests assert the path ran).</summary>
    public static long TotalDispatches;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _pool;
    private readonly DescriptorSetCache _cache;
    private readonly int _pushBytes;
    private bool _disposed;

    /// <summary>Dispatches recorded by this kernel instance.</summary>
    public long Dispatches { get; private set; }

    private protected MlaAbsorbedKernelBase(VulkanDevice device, string spvDir, string spvName, int bindings, int pushBytes)
    {
        string path = Path.Combine(spvDir, spvName + ".spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");
        _device = device;
        _pushBytes = pushBytes;
        _module = VulkanModule.LoadFromFile(device, path);
        try
        {
            Span<VkDescriptorBinding> b = stackalloc VkDescriptorBinding[bindings];
            for (int i = 0; i < bindings; i++) b[i] = new VkDescriptorBinding((uint)i);
            _pipeline = _module.CreateComputePipeline(entryPoint: "main", bindings: b, pushConstantBytes: (uint)pushBytes);
        }
        catch
        {
            _module.Dispose();
            throw;
        }
        _pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)bindings);
        _cache = new DescriptorSetCache(device, _pool, _pipeline, buffersPerSet: bindings);
    }

    /// <summary>Drops cached descriptor sets (call after scratch buffers were re-allocated).</summary>
    internal virtual void InvalidateDescriptorCache() => _cache.Reset();

    private protected unsafe void Dispatch(
        nint cmdBuf, ReadOnlySpan<nint> buffers, ReadOnlySpan<uint> push, uint groupsX, uint groupsY)
    {
        if (push.Length * sizeof(uint) != _pushBytes)
            throw new InvalidOperationException("push-constant size mismatch");
        nint set = _cache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        fixed (uint* p = push)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, (uint)_pushBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, groupsX, groupsY, 1);
        Dispatches++;
        Interlocked.Increment(ref TotalDispatches);
    }

    private protected static uint F(float v) => BitConverter.SingleToUInt32Bits(v);

    /// <inheritdoc/>
    public virtual void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}

/// <summary>
/// Builds one latent-cache row per token: <c>[rmsnorm(c_kv) * w | rope(k_pe)]</c> from the raw kv_a_mqa output, writing
/// straight into the cache (or the cache-less scratch). See <c>mla_kv_prep_f32.comp</c>.
/// </summary>
public sealed class MlaKvPrepKernel : MlaAbsorbedKernelBase
{
    /// <summary>Loads <c>mla_kv_prep_f32.spv</c>.</summary>
    public static MlaKvPrepKernel Create(VulkanDevice device, string spvDir) => new(device, spvDir);
    private MlaKvPrepKernel(VulkanDevice device, string spvDir) : base(device, spvDir, "mla_kv_prep_f32", 4, 6 * sizeof(uint)) { }

    /// <summary>Records the dispatch. <paramref name="rowOffset"/> is the cache row of token 0.</summary>
    public void Record(nint cmdBuf, VulkanDevice.Buffer kvA, VulkanDevice.Buffer normWeight, VulkanDevice.Buffer positions,
        VulkanDevice.Buffer cache, int seqLen, int rowOffset, int kvLoraRank, int qkRope, float eps, float theta)
    {
        Span<nint> bufs = [kvA.Handle, normWeight.Handle, positions.Handle, cache.Handle];
        Span<uint> pc = [(uint)seqLen, (uint)rowOffset, (uint)kvLoraRank, (uint)qkRope, F(eps), F(theta)];
        Dispatch(cmdBuf, bufs, pc, (uint)seqLen, 1);
    }
}

/// <summary>
/// Per-head batched Q8_0 GEMM (<c>mla_head_gemm_q8.comp</c>): query absorption (q_nope x W_UK) with the q_pe rope pack, and
/// the value up-projection (latent x W_UV).
/// </summary>
public sealed class MlaHeadGemmQ8Kernel : MlaAbsorbedKernelBase
{
    /// <summary>Rows per workgroup (one thread per weight row).</summary>
    public const int RowsPerGroup = 128;
    /// <summary>Tokens per workgroup.</summary>
    public const int TokensPerGroup = 8;
    /// <summary>Largest supported contraction length K.</summary>
    public const int MaxK = 512;

    /// <summary>Loads <c>mla_head_gemm_q8.spv</c>.</summary>
    public static MlaHeadGemmQ8Kernel Create(VulkanDevice device, string spvDir) => new(device, spvDir);
    private MlaHeadGemmQ8Kernel(VulkanDevice device, string spvDir) : base(device, spvDir, "mla_head_gemm_q8", 4, 12 * sizeof(uint)) { }

    /// <summary>
    /// <c>out[t,h,m] = sum_k W[h*M+m,k] * x[t,h,k]</c>. When <paramref name="ropeDim"/> &gt; 0 the q_pe slice at
    /// <paramref name="ropeSrcOff"/> of every x head row is rotated into <paramref name="ropeDstOff"/> of the out head row.
    /// </summary>
    public void Record(nint cmdBuf, VulkanDevice.Buffer weights, VulkanDevice.Buffer x, VulkanDevice.Buffer output,
        VulkanDevice.Buffer positions, int m, int k, int numHeads, int seqLen,
        int xTokStride, int xHeadStride, int outTokStride, int outHeadStride,
        int ropeDim, int ropeSrcOff, int ropeDstOff, float theta)
    {
        if (k > MaxK || (k & 31) != 0) throw new ArgumentOutOfRangeException(nameof(k), $"K must be a multiple of 32 and <= {MaxK}.");
        Span<nint> bufs = [weights.Handle, x.Handle, output.Handle, positions.Handle];
        Span<uint> pc =
        [
            (uint)m, (uint)k, (uint)numHeads, (uint)seqLen, (uint)xTokStride, (uint)xHeadStride,
            (uint)outTokStride, (uint)outHeadStride, (uint)ropeDim, (uint)ropeSrcOff, (uint)ropeDstOff, F(theta),
        ];
        uint rowTiles = (uint)((m + RowsPerGroup - 1) / RowsPerGroup);
        Dispatch(cmdBuf, bufs, pc, (uint)numHeads * rowTiles, (uint)((seqLen + TokensPerGroup - 1) / TokensPerGroup));
    }
}

/// <summary>
/// Attention over the absorbed-MLA latent cache (<c>attention_mla_absorbed_f32.comp</c>) plus its split-KV merge.
/// </summary>
public sealed class MlaAbsorbedAttentionKernel : MlaAbsorbedKernelBase
{
    /// <summary>Rows (token x head) per workgroup.</summary>
    public const int RowsPerGroup = 8;
    /// <summary>Keys per inner tile.</summary>
    public const int KeysPerTile = 64;
    /// <summary>Latent width the shader is compiled for (kv_lora_rank).</summary>
    public const int LatentDim = 512;
    /// <summary>Cache row width the shader is compiled for (kv_lora_rank + qk_rope).</summary>
    public const int CacheDim = 576;

    private readonly MergeKernel _merge;

    /// <summary>Loads <c>attention_mla_absorbed_f32.spv</c> and its merge shader.</summary>
    public static MlaAbsorbedAttentionKernel Create(VulkanDevice device, string spvDir) => new(device, spvDir);
    private MlaAbsorbedAttentionKernel(VulkanDevice device, string spvDir)
        : base(device, spvDir, "attention_mla_absorbed_f32", 6, 8 * sizeof(uint))
    {
        _merge = new MergeKernel(device, spvDir);
    }

    internal override void InvalidateDescriptorCache() { base.InvalidateDescriptorCache(); _merge.InvalidateDescriptorCache(); }

    /// <summary>Number of split-KV partitions the heuristic picks for <paramref name="seqKv"/> keys.</summary>
    public static int ChooseSplits(int seqKv, int maxSplits)
    {
        int s = (seqKv + 63) / 64;                 // ~one tile per split at least
        return Math.Clamp(s, 1, maxSplits);
    }

    /// <summary>
    /// Records the attention dispatch. With <paramref name="numSplits"/> &gt; 1 the caller must follow with a compute
    /// barrier and <see cref="RecordMerge"/>.
    /// </summary>
    public void Record(nint cmdBuf, VulkanDevice.Buffer qAbs, VulkanDevice.Buffer cache, VulkanDevice.Buffer output,
        VulkanDevice.Buffer partO, VulkanDevice.Buffer partML, int seqQ, int seqKv, int numHeads,
        int positionOffset, float scale, int numSplits, int splitRows)
    {
        int kvPerSplit = numSplits > 1 ? (seqKv + numSplits - 1) / numSplits : 0;
        if (numSplits > 1) kvPerSplit = (kvPerSplit + 63) / 64 * 64;       // tile-aligned splits
        Span<nint> bufs = [qAbs.Handle, cache.Handle, cache.Handle, output.Handle, partO.Handle, partML.Handle];
        Span<uint> pc =
        [
            (uint)seqQ, (uint)seqKv, (uint)numHeads, (uint)positionOffset, F(scale),
            (uint)numSplits, (uint)kvPerSplit, (uint)splitRows,
        ];
        uint rowTiles = (uint)((seqQ * numHeads + RowsPerGroup - 1) / RowsPerGroup);
        Dispatch(cmdBuf, bufs, pc, rowTiles * (uint)numSplits, 1);
    }

    /// <summary>Records the split-KV merge: one workgroup per (token, head) row.</summary>
    public void RecordMerge(nint cmdBuf, VulkanDevice.Buffer partO, VulkanDevice.Buffer partML, VulkanDevice.Buffer output,
        int totalRows, int numSplits, int splitRows)
        => _merge.Record(cmdBuf, partO, partML, output, totalRows, numSplits, splitRows);

    private sealed class MergeKernel : MlaAbsorbedKernelBase
    {
        public MergeKernel(VulkanDevice device, string spvDir)
            : base(device, spvDir, "attention_mla_absorbed_merge_f32", 3, 3 * sizeof(uint)) { }
        public void Record(nint cmdBuf, VulkanDevice.Buffer partO, VulkanDevice.Buffer partML, VulkanDevice.Buffer output,
            int totalRows, int numSplits, int splitRows)
        {
            Span<nint> bufs = [partO.Handle, partML.Handle, output.Handle];
            Span<uint> pc = [(uint)totalRows, (uint)numSplits, (uint)splitRows];
            Dispatch(cmdBuf, bufs, pc, (uint)totalRows, 1);
        }
    }

    /// <inheritdoc/>
    public override void Dispose()
    {
        _merge.Dispose();
        base.Dispose();
    }
}
