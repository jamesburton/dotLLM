using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Minimal compute-kernel wrapper for the small gpt-oss glue shaders (#737): one SPIR-V module,
/// N storage-buffer bindings, a flat push-constant block of 32-bit words. Keeps the four tiny
/// gpt-oss kernels (sink attention, OAI SwiGLU, per-expert bias add, raw-top-k router) from each
/// needing a 150-line kernel class.
/// </summary>
internal sealed class SimpleComputeKernel : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _pool;
    private readonly DescriptorSetCache _cache;
    private readonly int _buffers;
    private readonly int _pushBytes;
    private bool _disposed;

    private SimpleComputeKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline,
        nint pool, int buffers, int pushBytes)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _pool = pool;
        _buffers = buffers;
        _pushBytes = pushBytes;
        _cache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: buffers);
    }

    public static SimpleComputeKernel Create(VulkanDevice device, string spvDir, string spvName,
        int buffers, int pushWords)
    {
        string path = Path.Combine(spvDir, spvName + ".spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[buffers];
            for (int i = 0; i < buffers; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, (uint)(pushWords * sizeof(uint)));
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)buffers);
        return new SimpleComputeKernel(device, module, pipeline, pool, buffers, pushWords * sizeof(uint));
    }

    internal void InvalidateDescriptorCache() => _cache.Reset();

    public unsafe void Record(nint cmdBuf, ReadOnlySpan<nint> bufferHandles, ReadOnlySpan<uint> pushWords,
        uint groupsX, uint groupsY = 1)
    {
        if (bufferHandles.Length != _buffers)
            throw new ArgumentException($"Expected {_buffers} buffers, got {bufferHandles.Length}.", nameof(bufferHandles));
        if (pushWords.Length * sizeof(uint) != _pushBytes)
            throw new ArgumentException($"Expected {_pushBytes / sizeof(uint)} push words, got {pushWords.Length}.", nameof(pushWords));

        Span<nint> handles = stackalloc nint[_buffers];
        bufferHandles.CopyTo(handles);
        nint set = _cache.GetOrCreate(handles);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        fixed (uint* pc = pushWords)
        {
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, (uint)_pushBytes, (nint)pc);
        }
        VulkanApi.vkCmdDispatch(cmdBuf, groupsX, groupsY, 1);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_pool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}

/// <summary>
/// The gpt-oss-specific Vulkan kernels (#737) that have no counterpart in the shared kernel set:
/// per-head attention sinks, clamped OAI SwiGLU, per-expert bias add and the raw-top-k-then-softmax
/// router. Created only for <c>Architecture.GptOss</c> models.
/// </summary>
internal sealed class VulkanGptOssKernels : IDisposable
{
    /// <summary>llama.cpp swiglu_oai alpha.</summary>
    public const float SwiGluOaiAlpha = 1.702f;
    /// <summary>llama.cpp swiglu_oai clamp limit.</summary>
    public const float SwiGluOaiLimit = 7.0f;

    /// <summary>Max head dim the sink attention shader supports (shared-memory sized).</summary>
    public const int MaxHeadDim = 512;

    private const int Wg256 = 256;

    private readonly SimpleComputeKernel _attnSinks;
    private readonly SimpleComputeKernel _swigluOai;
    private readonly SimpleComputeKernel _expertBias;
    private readonly SimpleComputeKernel _topkRaw;
    private readonly MoeIndexedMatmulMxfp4F32Kernel _mxfp4;
    private readonly SimpleComputeKernel _ropeYarn;

    /// <summary>Dispatch counters (debug/perturbation proof that each path actually ran, #737).</summary>
    public long SinkAttentionDispatches, SwiGluOaiDispatches, ExpertBiasDispatches,
        RawTopKDispatches, Mxfp4MatmulDispatches, RopeYarnDispatches;

    private VulkanGptOssKernels(SimpleComputeKernel attnSinks, SimpleComputeKernel swigluOai,
        SimpleComputeKernel expertBias, SimpleComputeKernel topkRaw, MoeIndexedMatmulMxfp4F32Kernel mxfp4,
        SimpleComputeKernel ropeYarn)
    {
        _ropeYarn = ropeYarn;
        _attnSinks = attnSinks;
        _swigluOai = swigluOai;
        _expertBias = expertBias;
        _topkRaw = topkRaw;
        _mxfp4 = mxfp4;
    }

    /// <summary>The MXFP4 indexed-expert matmul kernel.</summary>
    public MoeIndexedMatmulMxfp4F32Kernel IndexedMxfp4 => _mxfp4;

    public static VulkanGptOssKernels Create(VulkanDevice device, string spvDir)
    {
        var attn = SimpleComputeKernel.Create(device, spvDir, "attention_sinks_f32", buffers: 5, pushWords: 12);
        var swi = SimpleComputeKernel.Create(device, spvDir, "swiglu_oai_f32", buffers: 3, pushWords: 3);
        var bias = SimpleComputeKernel.Create(device, spvDir, "moe_expert_bias_add_f32", buffers: 3, pushWords: 3);
        var topk = SimpleComputeKernel.Create(device, spvDir, "moe_topk_rawsoftmax_f32", buffers: 3, pushWords: 3);
        var mx = MoeIndexedMatmulMxfp4F32Kernel.Create(device, spvDir);
        var yarn = SimpleComputeKernel.Create(device, spvDir, "rope_yarn_f32", buffers: 4, pushWords: 8);
        return new VulkanGptOssKernels(attn, swi, bias, topk, mx, yarn);
    }

    internal void InvalidateDescriptorCaches()
    {
        _attnSinks.InvalidateDescriptorCache();
        _swigluOai.InvalidateDescriptorCache();
        _expertBias.InvalidateDescriptorCache();
        _topkRaw.InvalidateDescriptorCache();
        _mxfp4.InvalidateDescriptorCache();
        _ropeYarn.InvalidateDescriptorCache();
    }

    /// <summary>
    /// Causal GQA attention with per-head attention sinks (and optional sliding window). Same
    /// buffer/push layout as <see cref="AttentionF32Kernel"/> plus <paramref name="sinks"/>
    /// <c>[numHeads]</c>. Always the shared-memory reference path: sinks are only wired here.
    /// </summary>
    public void RecordAttentionSinks(nint cmdBuf,
        VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer output,
        VulkanDevice.Buffer sinks,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset, int slidingWindow, float scaleOverride = 0.0f)
    {
        if (seqQ <= 0) throw new ArgumentOutOfRangeException(nameof(seqQ));
        if (seqKv <= 0) throw new ArgumentOutOfRangeException(nameof(seqKv));
        if (numHeads % numKvHeads != 0) throw new ArgumentException("numHeads must be divisible by numKvHeads.");
        if (headDim <= 0 || headDim > MaxHeadDim)
            throw new ArgumentException($"headDim ({headDim}) outside (0, {MaxHeadDim}].", nameof(headDim));
        if (sinks.Size < (long)numHeads * sizeof(float))
            throw new ArgumentException("Sinks buffer too small.", nameof(sinks));

        ReadOnlySpan<nint> bufs = [q.Handle, k.Handle, v.Handle, output.Handle, sinks.Handle];
        ReadOnlySpan<uint> pc =
        [
            (uint)seqQ, (uint)seqKv, (uint)numHeads, (uint)numKvHeads,
            (uint)headDim, (uint)positionOffset, (uint)slidingWindow, 0u /*alibi*/,
            BitConverter.SingleToUInt32Bits(0.0f) /*softCap*/, BitConverter.SingleToUInt32Bits(scaleOverride),
            0u /*causal*/, 0u,
        ];
        SinkAttentionDispatches++;
        _attnSinks.Record(cmdBuf, bufs, pc, (uint)seqQ * (uint)numHeads);
    }

    /// <summary>gpt-oss clamped SwiGLU: <c>result[i] = swiglu_oai(gate[i], up[i])</c> over <paramref name="n"/> elements.</summary>
    public void RecordSwiGluOai(nint cmdBuf, VulkanDevice.Buffer gate, VulkanDevice.Buffer up,
        VulkanDevice.Buffer result, int n)
    {
        ReadOnlySpan<nint> bufs = [gate.Handle, up.Handle, result.Handle];
        ReadOnlySpan<uint> pc =
        [
            (uint)n, BitConverter.SingleToUInt32Bits(SwiGluOaiAlpha), BitConverter.SingleToUInt32Bits(SwiGluOaiLimit),
        ];
        SwiGluOaiDispatches++;
        _swigluOai.Record(cmdBuf, bufs, pc, (uint)((n + Wg256 - 1) / Wg256));
    }

    /// <summary><c>y[r, :] += bias[indices[r], :]</c> for <paramref name="rows"/> rows of width <paramref name="dim"/>.</summary>
    public void RecordExpertBiasAdd(nint cmdBuf, VulkanDevice.Buffer y, VulkanDevice.Buffer bias,
        VulkanDevice.Buffer indices, int rows, int dim, int numExperts)
    {
        ReadOnlySpan<nint> bufs = [y.Handle, bias.Handle, indices.Handle];
        ReadOnlySpan<uint> pc = [(uint)rows, (uint)dim, (uint)numExperts];
        long total = (long)rows * dim;
        ExpertBiasDispatches++;
        _expertBias.Record(cmdBuf, bufs, pc, (uint)((total + Wg256 - 1) / Wg256));
    }

    /// <summary>
    /// Router gating: top-<paramref name="k"/> on raw logits then softmax over the selected logits.
    /// </summary>
    public void RecordTopKRawSoftmax(nint cmdBuf, VulkanDevice.Buffer logits, VulkanDevice.Buffer indices,
        VulkanDevice.Buffer weights, int seqLen, int numExperts, int k)
    {
        if (numExperts > 512) throw new ArgumentException("numExperts must be <= 512.", nameof(numExperts));
        if (k <= 0 || k > 16) throw new ArgumentException("k must be in [1, 16].", nameof(k));
        ReadOnlySpan<nint> bufs = [logits.Handle, indices.Handle, weights.Handle];
        ReadOnlySpan<uint> pc = [(uint)seqLen, (uint)numExperts, (uint)k];
        RawTopKDispatches++;
        _topkRaw.Record(cmdBuf, bufs, pc, (uint)seqLen);
    }

    /// <summary>
    /// Dense-YaRN RoPE: rotates Q/K in place using the host-built inverse-frequency table
    /// <paramref name="invFreq"/> <c>[ropeDim/2]</c> and scales cos/sin by <paramref name="mscale"/>.
    /// </summary>
    public void RecordRopeYarn(nint cmdBuf, VulkanDevice.Buffer q, VulkanDevice.Buffer k,
        VulkanDevice.Buffer positions, VulkanDevice.Buffer invFreq,
        int seqLen, int numHeads, int numKvHeads, int headDim, int ropeDim, bool neox, float mscale)
    {
        if (ropeDim <= 0 || (ropeDim & 1) != 0 || ropeDim > headDim)
            throw new ArgumentException($"ropeDim {ropeDim} invalid for headDim {headDim}.", nameof(ropeDim));
        ReadOnlySpan<nint> bufs = [q.Handle, k.Handle, positions.Handle, invFreq.Handle];
        ReadOnlySpan<uint> pc =
        [
            (uint)seqLen, (uint)numHeads, (uint)numKvHeads, (uint)headDim, (uint)ropeDim,
            neox ? 1u : 0u, BitConverter.SingleToUInt32Bits(mscale), (uint)(ropeDim / 2),
        ];
        long maxPairs = (long)seqLen * Math.Max(numHeads, numKvHeads) * (ropeDim / 2);
        RopeYarnDispatches++;
        _ropeYarn.Record(cmdBuf, bufs, pc, (uint)((maxPairs + Wg256 - 1) / Wg256));
    }

    /// <summary>Counted wrapper over the MXFP4 indexed matmul.</summary>
    public void RecordIndexedMxfp4(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer x,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer y, int m, int k, int n, int numExperts)
    {
        Mxfp4MatmulDispatches++;
        _mxfp4.Record(cmdBuf, bank, x, indices, y, m, k, n, numExperts);
    }

    public void Dispose()
    {
        _attnSinks.Dispose();
        _swigluOai.Dispose();
        _expertBias.Dispose();
        _topkRaw.Dispose();
        _mxfp4.Dispose();
        _ropeYarn.Dispose();
    }
}
