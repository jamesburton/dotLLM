using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE router top-k with <b>sigmoid</b> scoring and a selection bias (DeepSeek-V3 / GLM-4.7-Flash,
/// llama.cpp <c>LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID</c>). Wraps <c>moe_topk_sigmoid_bias_f32.spv</c>.
/// </summary>
/// <remarks>
/// Per token: <c>p = sigmoid(logits)</c>; top-k is chosen on <c>p + bias</c> (lower index wins ties);
/// the emitted weights are the <b>unbiased</b> <c>p</c> of the chosen experts, optionally divided
/// by their sum (clamped at 6.1035e-5) and then multiplied by <c>weightsScale</c>. Mirrors
/// <c>MoeSwiGluMlp.Route(sigmoidGating: true)</c> on the CPU (#742).
/// </remarks>
public sealed class MoeTopKSigmoidBiasF32Kernel : IDisposable
{
    /// <summary>Compile-time upper bound on numExperts (mirrors <c>MAX_EXPERTS</c> in the shader).</summary>
    public const int MaxExperts = 256;

    /// <summary>Compile-time upper bound on top-k (mirrors <c>MAX_K</c> in the shader).</summary>
    public const int MaxK = 16;

    private const int PushConstantBytes = 5 * sizeof(uint); // seqLen, numExperts, k, norm (u32) + scale (f32)

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeTopKSigmoidBiasF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>Loads <c>moe_topk_sigmoid_bias_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeTopKSigmoidBiasF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_topk_sigmoid_bias_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            for (int i = 0; i < 4; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MoeTopKSigmoidBiasF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer logits, VulkanDevice.Buffer bias,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer weights,
        int seqLen, int numExperts, int k, bool normTopKProb, float weightsScale)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, logits, bias, indices, weights, seqLen, numExperts, k, normTopKProb, weightsScale);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the dispatch into <paramref name="cmdBuf"/> (one workgroup per token).</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="logits">F32 router logits [seqLen × numExperts].</param>
    /// <param name="bias">F32 selection bias [numExperts] (<c>exp_probs_b</c>).</param>
    /// <param name="indices">int32 top-k indices output [seqLen × k].</param>
    /// <param name="weights">F32 top-k weights output [seqLen × k].</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="numExperts">Experts per layer (≤ <see cref="MaxExperts"/>).</param>
    /// <param name="k">Top-k (1 ≤ k ≤ <see cref="MaxK"/> ≤ numExperts).</param>
    /// <param name="normTopKProb">Divide the picked weights by their (clamped) sum.</param>
    /// <param name="weightsScale">Multiplier applied last (<c>expert_weights_scale</c>).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer logits, VulkanDevice.Buffer bias,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer weights,
        int seqLen, int numExperts, int k, bool normTopKProb, float weightsScale)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (numExperts <= 0 || numExperts > MaxExperts)
            throw new ArgumentOutOfRangeException(nameof(numExperts),
                $"numExperts must be in [1, {MaxExperts}], got {numExperts}.");
        if (k <= 0 || k > MaxK || k > numExperts)
            throw new ArgumentOutOfRangeException(nameof(k),
                $"k must be in [1, min({MaxK}, numExperts)], got {k} (numExperts={numExperts}).");
        if (logits.Size < (long)seqLen * numExperts * sizeof(float))
            throw new ArgumentException("logits buffer too small.", nameof(logits));
        if (bias.Size < (long)numExperts * sizeof(float))
            throw new ArgumentException("bias buffer too small.", nameof(bias));
        if (indices.Size < (long)seqLen * k * sizeof(int))
            throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (weights.Size < (long)seqLen * k * sizeof(float))
            throw new ArgumentException("weights buffer too small.", nameof(weights));

        Span<nint> buffers = stackalloc nint[4] { logits.Handle, bias.Handle, indices.Handle, weights.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<byte> pc = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc, (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc[4..], (uint)numExperts);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc[8..], (uint)k);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pc[12..], normTopKProb ? 1u : 0u);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pc[16..], weightsScale);
        fixed (byte* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        }

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
