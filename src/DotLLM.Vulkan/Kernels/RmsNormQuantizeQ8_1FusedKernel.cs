using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Fused RMSNorm + Q8_1 activation quantization for the decode path
/// (issue #145): one dispatch producing BOTH the normalized F32 row
/// (<c>normOut = (x / rms(x)) * weight</c>) AND its Q8_1 quantization
/// (packed int8 <c>xq</c> + per-32-block (scale, sum) <c>xds</c>) for the
/// dp4a MMVQ GEMVs. Replaces the standalone
/// <see cref="RmsNormF32Kernel"/> → barrier → <see cref="QuantizeQ8_1Kernel"/>
/// pair, saving one dispatch + one full-pipeline barrier per quantized GEMV
/// group (two groups per dense layer). The GPU analogue of the CPU backend's
/// <c>FusedOps.RmsNormQuantize</c>.
/// </summary>
/// <remarks>
/// <para>
/// Bit-parity contract with the standalone pair: the shader copies the
/// subgroup sum-of-squares reduction verbatim from <c>rmsnorm_f32_sg.comp</c>
/// and the quantization verbatim from <c>quantize_q8_1.comp</c>, so on a
/// device where the standalone rmsnorm takes its subgroup path the fused
/// outputs are bit-identical. Callers must only create this kernel when
/// <see cref="VulkanDevice.HasSubgroupArithmetic"/> is true (the shader
/// requires the subgroup-arithmetic extension).
/// </para>
/// <para>
/// Dispatch: a single workgroup over all of <c>n</c> (decode ⇒ one activation
/// row). The normalized F32 row is still written because consumers besides
/// the GEMVs read it (LoRA deltas).
/// </para>
/// </remarks>
public sealed class RmsNormQuantizeQ8_1FusedKernel : IDisposable
{
    /// <summary>Elements per Q8_1 block (must match <see cref="QuantizeQ8_1Kernel.GroupSize"/>).</summary>
    public const int GroupSize = 32;

    private const int PushConstantBytes = 3 * sizeof(uint); // n, eps, blocksPerRow

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private RmsNormQuantizeQ8_1FusedKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 5);
    }

    /// <summary>
    /// Loads <c>rmsnorm_quantize_q8_1.spv</c> from <paramref name="spvDir"/> and
    /// builds the pipeline. Returns <c>null</c> when the SPV is missing (older
    /// builds) or the device lacks subgroup arithmetic, so the caller falls back
    /// to the standalone rmsnorm + quantize pair.
    /// </summary>
    public static RmsNormQuantizeQ8_1FusedKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasSubgroupArithmetic)
            return null;

        string path = Path.Combine(spvDir, "rmsnorm_quantize_q8_1.spv");
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
        return new RmsNormQuantizeQ8_1FusedKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Synchronous one-shot launch (parity tests). Production callers use
    /// <see cref="Record"/> inside the batched forward command buffer.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer input, VulkanDevice.Buffer weight, VulkanDevice.Buffer normOut,
        VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        int n, float eps)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, input, weight, normOut, xq, xds, n, eps);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the fused rmsnorm + Q8_1 quantization into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="input">FP32 hidden-state row, <c>[n]</c>.</param>
    /// <param name="weight">FP32 per-feature norm scale, <c>[n]</c>.</param>
    /// <param name="normOut">Output normalized FP32 row, <c>[n]</c>.</param>
    /// <param name="xq">Output packed-int8 buffer, ≥ <see cref="QuantizeQ8_1Kernel.PackedBytes"/> bytes.</param>
    /// <param name="xds">Output (scale, sum) buffer, ≥ <see cref="QuantizeQ8_1Kernel.ScaleBytes"/> bytes.</param>
    /// <param name="n">Row length (must be a multiple of 32).</param>
    /// <param name="eps">Epsilon under the square root.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer input, VulkanDevice.Buffer weight, VulkanDevice.Buffer normOut,
        VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        int n, float eps)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((n % GroupSize) != 0)
            throw new ArgumentException($"n must be a multiple of {GroupSize}, got {n}", nameof(n));

        long rowBytes = (long)n * sizeof(float);
        if (input.Size < rowBytes) throw new ArgumentException("Input buffer too small.", nameof(input));
        if (weight.Size < rowBytes) throw new ArgumentException("Weight buffer too small.", nameof(weight));
        if (normOut.Size < rowBytes) throw new ArgumentException("Norm output buffer too small.", nameof(normOut));
        if (xq.Size < QuantizeQ8_1Kernel.PackedBytes(n)) throw new ArgumentException("Packed output buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1Kernel.ScaleBytes(n)) throw new ArgumentException("Scale output buffer too small.", nameof(xds));

        Span<nint> buffers = stackalloc nint[5]
            { input.Handle, weight.Handle, normOut.Handle, xq.Handle, xds.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)n);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pcBytes[4..], eps);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)(n / GroupSize));
        fixed (byte* pcPtr = pcBytes)
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
