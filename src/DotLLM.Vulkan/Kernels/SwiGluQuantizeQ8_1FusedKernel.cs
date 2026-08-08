using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Fused SwiGLU activation + Q8_1 activation quantization for the decode
/// path (issue #71, sibling of <see cref="RmsNormQuantizeQ8_1FusedKernel"/>
/// / issue #145): one dispatch producing BOTH the gated F32 row
/// (<c>siluOut = silu(gate) * up</c>) AND its Q8_1 quantization (packed int8
/// <c>xq</c> + per-32-block (scale, sum) <c>xds</c>) for the down_proj dp4a
/// MMVQ GEMV. Replaces the standalone <see cref="SwiGluF32Kernel"/> →
/// barrier → <see cref="QuantizeQ8_1Kernel"/> pair that runs before every
/// SwiGLU-activated down_proj MMVQ GEMV at seqLen==1, saving one dispatch +
/// one full-pipeline barrier per dense-FFN decode layer.
/// </summary>
/// <remarks>
/// <para>
/// Unlike the RMSNorm+quantize fusion, this one needs no on-shader
/// reduction — SwiGLU is pointwise, so there is no barrier inside the
/// shader either; every 32-block is computed and quantized independently.
/// </para>
/// <para>
/// Dispatch: a single workgroup over all of <c>n</c> (decode ⇒ one
/// intermediate-size row). The gated F32 row is still written because the
/// down_proj LoRA delta reads it.
/// </para>
/// <para>
/// Only applies to plain SwiGLU (Llama/Mistral/Qwen-family MLP); GeGLU
/// (Gemma) and squared-ReLU-GLU (BitNet) keep the standalone
/// activation-kernel + quantize pair — a natural follow-up would add sibling
/// fused shaders for those gates using the same pattern.
/// </para>
/// </remarks>
public sealed class SwiGluQuantizeQ8_1FusedKernel : IDisposable
{
    /// <summary>Elements per Q8_1 block (must match <see cref="QuantizeQ8_1Kernel.GroupSize"/>).</summary>
    public const int GroupSize = 32;

    private const int PushConstantBytes = 2 * sizeof(uint); // n, blocksPerRow

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private SwiGluQuantizeQ8_1FusedKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 5);
    }

    /// <summary>
    /// Loads <c>swiglu_quantize_q8_1.spv</c> from <paramref name="spvDir"/> and
    /// builds the pipeline. Returns <c>null</c> when the SPV is missing (older
    /// builds) — the caller falls back to the standalone SwiGLU + quantize pair.
    /// </summary>
    public static SwiGluQuantizeQ8_1FusedKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "swiglu_quantize_q8_1.spv");
        if (!File.Exists(path))
            return null;

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[5];
            for (int i = 0; i < 5; i++)
                bindings[i] = new VkDescriptorBinding((uint)i);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        return new SwiGluQuantizeQ8_1FusedKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Synchronous one-shot launch (parity tests). Production callers use
    /// <see cref="Record"/> inside the batched forward command buffer.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer gate, VulkanDevice.Buffer up, VulkanDevice.Buffer siluOut,
        VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, gate, up, siluOut, xq, xds, n);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the fused SwiGLU + Q8_1 quantization into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="gate">FP32 pre-activation gate row, <c>[n]</c>.</param>
    /// <param name="up">FP32 up-projection row, <c>[n]</c>.</param>
    /// <param name="siluOut">Output gated FP32 row, <c>[n]</c>.</param>
    /// <param name="xq">Output packed-int8 buffer, ≥ <see cref="QuantizeQ8_1Kernel.PackedBytes"/> bytes.</param>
    /// <param name="xds">Output (scale, sum) buffer, ≥ <see cref="QuantizeQ8_1Kernel.ScaleBytes"/> bytes.</param>
    /// <param name="n">Row length (must be a multiple of 32).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer gate, VulkanDevice.Buffer up, VulkanDevice.Buffer siluOut,
        VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        int n)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((n % GroupSize) != 0)
            throw new ArgumentException($"n must be a multiple of {GroupSize}, got {n}", nameof(n));

        long rowBytes = (long)n * sizeof(float);
        if (gate.Size < rowBytes) throw new ArgumentException("Gate buffer too small.", nameof(gate));
        if (up.Size < rowBytes) throw new ArgumentException("Up buffer too small.", nameof(up));
        if (siluOut.Size < rowBytes) throw new ArgumentException("SiluOut buffer too small.", nameof(siluOut));
        if (xq.Size < QuantizeQ8_1Kernel.PackedBytes(n)) throw new ArgumentException("Packed output buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1Kernel.ScaleBytes(n)) throw new ArgumentException("Scale output buffer too small.", nameof(xds));

        Span<nint> buffers = stackalloc nint[5]
            { gate.Handle, up.Handle, siluOut.Handle, xq.Handle, xds.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[2] { (uint)n, (uint)(n / GroupSize) };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // Single workgroup over all of n (decode path: one activation row).
        VulkanApi.vkCmdDispatch(cmdBuf, 1, 1, 1);
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
