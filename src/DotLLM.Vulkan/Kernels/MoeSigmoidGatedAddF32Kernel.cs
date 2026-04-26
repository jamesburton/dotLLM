using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE sigmoid-gated add — fuses a per-token sigmoid over precomputed gate
/// logits and a weighted in-place accumulate into a single dispatch:
/// <code>
///     scale[t]  = sigmoid(gateLogits[t])
///     out[t, h] += scale[t] * b[t, h]
/// </code>
/// </summary>
/// <remarks>
/// Used by the Qwen1.5-MoE shared-expert path to fold the per-token sigmoid
/// gate (<c>shared_expert_gate.weight</c>, [hidden]) onto the running
/// shared-expert sum before merging it back into the routed output.
/// Mirrors the per-token loop in <c>MoeSwiGluMlp.ExecuteCoreGrouped</c>
/// (CPU reference — see the "Per-token sigmoid gate logit" comment).
/// </remarks>
public sealed class MoeSigmoidGatedAddF32Kernel : IDisposable
{
    private const int WorkgroupSize = 64;
    // seqLen, hiddenSize (u32)
    private const int PushConstantBytes = 2 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeSigmoidGatedAddF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>moe_sigmoid_gated_add_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeSigmoidGatedAddF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_sigmoid_gated_add_f32.spv");
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
        return new MoeSigmoidGatedAddF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer output, VulkanDevice.Buffer b, VulkanDevice.Buffer gateLogits,
        int seqLen, int hiddenSize)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, output, b, gateLogits, seqLen, hiddenSize);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the gated-add dispatch into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="output">Read-modify-write buffer [<paramref name="seqLen"/> × <paramref name="hiddenSize"/>] F32 row-major.</param>
    /// <param name="b">Per-token addend [<paramref name="seqLen"/> × <paramref name="hiddenSize"/>] F32 row-major.</param>
    /// <param name="gateLogits">Per-token gate logits [<paramref name="seqLen"/>] F32 (pre-sigmoid).</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="hiddenSize">Per-token feature dim.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer output, VulkanDevice.Buffer b, VulkanDevice.Buffer gateLogits,
        int seqLen, int hiddenSize)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));

        long rowsBytes = (long)seqLen * hiddenSize * sizeof(float);
        long logitBytes = (long)seqLen * sizeof(float);
        if (output.Size < rowsBytes) throw new ArgumentException("output buffer too small.", nameof(output));
        if (b.Size < rowsBytes) throw new ArgumentException("b buffer too small.", nameof(b));
        if (gateLogits.Size < logitBytes) throw new ArgumentException("gateLogits buffer too small.", nameof(gateLogits));

        Span<nint> buffers = stackalloc nint[3] { output.Handle, b.Handle, gateLogits.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)hiddenSize);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
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
