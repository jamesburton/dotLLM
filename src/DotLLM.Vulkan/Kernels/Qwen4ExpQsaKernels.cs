using System.Buffers.Binary;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Qwen4-Exp QSA (query-sparse attention) device kernels (issue #819): indexer-key pooling, block scoring, exact top-k block
/// selection and the gather attention over the selected blocks. Semantics are the CPU oracle's
/// (<see cref="DotLLM.Cpu.Kernels.Qwen4ExpQsa"/>): block <c>b</c> pools raw keys <c>R*b..R*b+R-1</c> (mean, RMSNorm, RoPE at <c>R*b</c>);
/// a query at position <c>p</c> scores its <c>floor((p+1)/R)</c> complete blocks with <c>sum_h relu(q_h . k_b) / sqrt(D)</c>, keeps the top
/// <c>budget</c> blocks (ties to the lower index) and attends their tokens plus the tail of the incomplete block. Queries with at most
/// <c>budget</c> complete blocks attend densely and skip scoring.
/// </summary>
public sealed class Qwen4ExpQsaKernels : IDisposable
{
    private const int PoolPcBytes = 7 * 4;
    private const int ScorePcBytes = 9 * 4;
    private const int SelectPcBytes = 6 * 4;
    private const int AttnPcBytes = 11 * 4;
    private const int MergePcBytes = 4 * 4;

    /// <summary>Key tokens per scoring workgroup tile (matches the shader).</summary>
    public const int ScoreTileBlocks = 128;

    private sealed class Stage : IDisposable
    {
        public readonly VulkanModule Module;
        public readonly ComputePipeline Pipeline;
        public readonly nint Pool;
        public readonly DescriptorSetCache Cache;
        private readonly VulkanDevice _device;

        public Stage(VulkanDevice device, string spvDir, string name, int buffers, int pcBytes)
        {
            _device = device;
            string path = Path.Combine(spvDir, name + ".spv");
            if (!File.Exists(path))
                throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh after installing the Vulkan SDK.");
            Module = VulkanModule.LoadFromFile(device, path);
            try
            {
                Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[buffers];
                for (int i = 0; i < buffers; i++) bindings[i] = new VkDescriptorBinding((uint)i);
                Pipeline = Module.CreateComputePipeline("main", bindings, (uint)pcBytes);
            }
            catch { Module.Dispose(); throw; }
            Pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)buffers);
            Cache = new DescriptorSetCache(device, Pool, Pipeline, buffersPerSet: buffers);
        }

        public unsafe void Dispatch(nint cmd, ReadOnlySpan<nint> buffers, ReadOnlySpan<byte> pc, uint gx, uint gy)
        {
            nint set = Cache.GetOrCreate(buffers);
            VulkanApi.vkCmdBindPipeline(cmd, VkPipelineBindPoint.Compute, Pipeline.Pipeline);
            VulkanApi.vkCmdBindDescriptorSets(cmd, VkPipelineBindPoint.Compute, Pipeline.Layout, 0, 1, set, 0, 0);
            fixed (byte* p = pc)
                VulkanApi.vkCmdPushConstants(cmd, Pipeline.Layout, VkShaderStageFlags.Compute, 0, (uint)pc.Length, (nint)p);
            VulkanApi.vkCmdDispatch(cmd, gx, gy, 1);
        }

        public void Dispose()
        {
            if (Pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, Pool, 0);
            Pipeline.Dispose();
            Module.Dispose();
        }
    }

    private readonly Stage _pool, _score, _select, _attn, _merge;
    private bool _disposed;

    private Qwen4ExpQsaKernels(Stage pool, Stage score, Stage select, Stage attn, Stage merge)
    { _pool = pool; _score = score; _select = select; _attn = attn; _merge = merge; }

    /// <summary>Loads the five <c>qsa_*_f32.spv</c> blobs from <paramref name="spvDir"/>.</summary>
    public static Qwen4ExpQsaKernels Create(VulkanDevice device, string spvDir)
    {
        var made = new List<Stage>();
        try
        {
            Stage Make(string n, int b, int pc) { var s = new Stage(device, spvDir, n, b, pc); made.Add(s); return s; }
            return new Qwen4ExpQsaKernels(Make("qsa_pool_f32", 3, PoolPcBytes), Make("qsa_score_f32", 3, ScorePcBytes),
                Make("qsa_select_f32", 2, SelectPcBytes), Make("qsa_attention_f32", 6, AttnPcBytes), Make("qsa_merge_f32", 3, MergePcBytes));
        }
        catch { foreach (var s in made) s.Dispose(); throw; }
    }

    /// <summary>Drops every cached descriptor set (call after re-creating scratch buffers).</summary>
    public void InvalidateDescriptorCache()
    {
        _pool.Cache.Reset(); _score.Cache.Reset(); _select.Cache.Reset(); _attn.Cache.Reset(); _merge.Cache.Reset();
    }

    private static void U(Span<byte> pc, int i, uint v) => BinaryPrimitives.WriteUInt32LittleEndian(pc[(i * 4)..], v);
    private static void F(Span<byte> pc, int i, float v) => BinaryPrimitives.WriteSingleLittleEndian(pc[(i * 4)..], v);

    /// <summary>
    /// Pools blocks <c>firstBlock .. firstBlock+numBlocks-1</c> from <paramref name="raw"/> <c>[positions, dim]</c> into <paramref name="pooled"/>
    /// <c>[blocks, dim]</c>: mean of the block's raw keys, RMSNorm with <paramref name="gamma"/>, NeoX RoPE (leading <paramref name="ropeDim"/> dims) at the block's first position.
    /// </summary>
    public void RecordPool(nint cmd, VulkanDevice.Buffer raw, VulkanDevice.Buffer gamma, VulkanDevice.Buffer pooled,
        int firstBlock, int numBlocks, int dim, int ropeDim, int blockSize, float eps, float theta)
    {
        if (numBlocks <= 0) return;
        if (dim > 512) throw new ArgumentOutOfRangeException(nameof(dim), "indexer key width above 512 is not supported.");
        Span<byte> pc = stackalloc byte[PoolPcBytes];
        U(pc, 0, (uint)firstBlock); U(pc, 1, (uint)numBlocks); U(pc, 2, (uint)dim); U(pc, 3, (uint)ropeDim);
        U(pc, 4, (uint)blockSize); F(pc, 5, eps); F(pc, 6, theta);
        _pool.Dispatch(cmd, [raw.Handle, gamma.Handle, pooled.Handle], pc, (uint)numBlocks, 1);
    }

    /// <summary>Scores the complete blocks of queries <c>qBase .. qBase+qCount-1</c> whose block count exceeds the budget (others untouched).</summary>
    /// <remarks><c>maxBlocks</c> is an upper bound of the complete blocks visible to any query in the range (it sizes the grid).</remarks>
    public void RecordScore(nint cmd, VulkanDevice.Buffer iq, VulkanDevice.Buffer pooled, VulkanDevice.Buffer scores,
        int qBase, int qCount, int firstPos, int heads, int dim, int blockSize, int budgetBlocks, int nbCap, int maxBlocks)
    {
        if (heads * dim > 1024) throw new ArgumentOutOfRangeException(nameof(dim), "indexer heads*dim above 1024 is not supported.");
        Span<byte> pc = stackalloc byte[ScorePcBytes];
        U(pc, 0, (uint)qBase); U(pc, 1, (uint)qCount); U(pc, 2, (uint)firstPos); U(pc, 3, (uint)heads); U(pc, 4, (uint)dim);
        U(pc, 5, (uint)blockSize); U(pc, 6, (uint)budgetBlocks); U(pc, 7, (uint)nbCap); F(pc, 8, 1.0f / MathF.Sqrt(dim));
        _score.Dispatch(cmd, [iq.Handle, pooled.Handle, scores.Handle], pc, (uint)((maxBlocks + ScoreTileBlocks - 1) / ScoreTileBlocks), (uint)qCount);
    }

    /// <summary>Selects the top <paramref name="budgetBlocks"/> blocks per scored query into <paramref name="sel"/> <c>[qCount, budgetBlocks]</c> (ascending ids).</summary>
    public void RecordSelect(nint cmd, VulkanDevice.Buffer scores, VulkanDevice.Buffer sel,
        int qBase, int qCount, int firstPos, int blockSize, int budgetBlocks, int nbCap)
    {
        Span<byte> pc = stackalloc byte[SelectPcBytes];
        U(pc, 0, (uint)qBase); U(pc, 1, (uint)qCount); U(pc, 2, (uint)firstPos); U(pc, 3, (uint)blockSize);
        U(pc, 4, (uint)budgetBlocks); U(pc, 5, (uint)nbCap);
        _select.Dispatch(cmd, [scores.Handle, sel.Handle], pc, (uint)qCount, 1);
    }

    /// <summary>
    /// Attention of queries <c>qBase .. qBase+qCount-1</c> (rows of <paramref name="q"/> / <paramref name="output"/>) over their selected blocks plus tail,
    /// in <paramref name="numSplits"/> KV splits merged into <paramref name="output"/>. <paramref name="partOut"/> needs
    /// <c>qCount*numHeads*numSplits*headDim</c> floats, <paramref name="partMs"/> twice <c>qCount*numHeads*numSplits</c>.
    /// </summary>
    public void RecordAttention(nint cmd, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer sel,
        VulkanDevice.Buffer partOut, VulkanDevice.Buffer partMs, VulkanDevice.Buffer output,
        int qBase, int qCount, int firstPos, int numHeads, int numKvHeads, int headDim, int blockSize, int budgetBlocks, int numSplits)
    {
        if (headDim > 512) throw new ArgumentOutOfRangeException(nameof(headDim));
        Span<byte> pc = stackalloc byte[AttnPcBytes];
        U(pc, 0, (uint)qBase); U(pc, 1, (uint)qCount); U(pc, 2, (uint)firstPos); U(pc, 3, (uint)numHeads); U(pc, 4, (uint)numKvHeads);
        U(pc, 5, (uint)headDim); U(pc, 6, (uint)blockSize); U(pc, 7, (uint)budgetBlocks); U(pc, 8, (uint)numSplits);
        U(pc, 9, (uint)budgetBlocks); F(pc, 10, 1.0f / MathF.Sqrt(headDim));
        _attn.Dispatch(cmd, [q.Handle, k.Handle, v.Handle, sel.Handle, partOut.Handle, partMs.Handle], pc, (uint)(numHeads * numSplits), (uint)qCount);
        KernelSupport.ComputeToComputeBarrier(cmd);
        Span<byte> mc = stackalloc byte[MergePcBytes];
        U(mc, 0, (uint)qBase); U(mc, 1, (uint)numHeads); U(mc, 2, (uint)headDim); U(mc, 3, (uint)numSplits);
        _merge.Dispatch(cmd, [partOut.Handle, partMs.Handle, output.Handle], mc, (uint)numHeads, (uint)qCount);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _pool.Dispose(); _score.Dispose(); _select.Dispose(); _attn.Dispose(); _merge.Dispose();
    }
}
