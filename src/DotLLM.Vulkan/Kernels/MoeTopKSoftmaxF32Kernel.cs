using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE router top-k softmax kernel — for each token, full softmax over
/// router logits then picks the top-k entries (stable on ties, lower
/// index wins). Optional renormalisation of the top-k weights to sum
/// to 1.0 (Mixtral / Qwen3-MoE convention; Qwen1.5-MoE uses raw probs).
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the routing path in
/// <c>DotLLM.Cpu.Kernels.MoeSwiGluMlp.SelectTopK</c>. The top-k selection
/// is sequential (single-threaded inside each workgroup) to preserve the
/// lower-index-wins-on-tie property — a parallel reduction would not.
/// For typical Mixtral / Qwen-MoE / Phi-3.5-MoE shapes (numExperts up
/// to 64, k up to 8) the cost is trivial.
/// </para>
/// <para>
/// Build block of the Vulkan MoE forward path (issue #4): the matmul
/// kernels (gate projection, expert MLPs) are reused; this is the only
/// genuinely new compute kernel. Forward-pass orchestration is a
/// follow-up.
/// </para>
/// </remarks>
public sealed class MoeTopKSoftmaxF32Kernel : IDisposable
{
    /// <summary>Compile-time upper bound on numExperts (mirrors <c>MAX_EXPERTS</c> in the shader).</summary>
    public const int MaxExperts = 256;

    /// <summary>Compile-time upper bound on top-k (mirrors <c>MAX_K</c> in the shader).</summary>
    public const int MaxK = 16;

    private const int WorkgroupSize = 64;
    // seqLen, numExperts, k, normTopKProb (all u32)
    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeTopKSoftmaxF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>moe_topk_softmax_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeTopKSoftmaxF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_topk_softmax_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new MoeTopKSoftmaxF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer logits, VulkanDevice.Buffer indices, VulkanDevice.Buffer weights,
        int seqLen, int numExperts, int k, bool normTopKProb)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, logits, indices, weights, seqLen, numExperts, k, normTopKProb);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the dispatch into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="logits">F32 router logits [seqLen × numExperts] row-major.</param>
    /// <param name="indices">int32 top-k indices output [seqLen × k] row-major.</param>
    /// <param name="weights">F32 top-k weights output [seqLen × k] row-major.</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="numExperts">Total experts per layer (must be ≤ <see cref="MaxExperts"/>).</param>
    /// <param name="k">Top-k count (1 ≤ k ≤ <see cref="MaxK"/> ≤ numExperts).</param>
    /// <param name="normTopKProb">When <c>true</c>, divides the picked weights by their sum.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer logits, VulkanDevice.Buffer indices, VulkanDevice.Buffer weights,
        int seqLen, int numExperts, int k, bool normTopKProb)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (numExperts <= 0 || numExperts > MaxExperts)
            throw new ArgumentOutOfRangeException(nameof(numExperts),
                $"numExperts must be in [1, {MaxExperts}], got {numExperts}.");
        if (k <= 0 || k > MaxK || k > numExperts)
            throw new ArgumentOutOfRangeException(nameof(k),
                $"k must be in [1, min({MaxK}, numExperts)], got {k} (numExperts={numExperts}).");

        long logitBytes = (long)seqLen * numExperts * sizeof(float);
        long idxBytes = (long)seqLen * k * sizeof(int);
        long wtBytes = (long)seqLen * k * sizeof(float);
        if (logits.Size < logitBytes) throw new ArgumentException("logits buffer too small.", nameof(logits));
        if (indices.Size < idxBytes) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (weights.Size < wtBytes) throw new ArgumentException("weights buffer too small.", nameof(weights));

        Span<nint> buffers = stackalloc nint[3] { logits.Handle, indices.Handle, weights.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)numExperts);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)k);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], normTopKProb ? 1u : 0u);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // One workgroup per token.
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)seqLen, 1, 1);
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
