using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q8_0 MMVQ decode-path GEMV: <c>y[M] = W_q8[M,K] @ x[K]</c> via integer dp4a.
/// The activation <c>x</c> must already be quantized to Q8_1
/// (<see cref="QuantizeQ8_1Kernel"/>); this kernel runs
/// <c>dotPacked4x8AccSatEXT</c> (4×int8 → int32 saturating accumulate) against
/// the int8 weights, then scales each 32-block by <c>d_weight · d_activation</c>.
/// </summary>
/// <remarks>
/// <para>
/// Replaces the F32-in <see cref="MatMulQ8_0Kernel"/> on the decode path
/// (seqLen==1) on devices that advertise
/// <see cref="VulkanDevice.HasIntegerDotProduct"/>. Halves the activation read
/// traffic (int8 vs the implicit fp32 of the F32-in path) and removes the fp
/// multiply from the inner loop — the decode GEMV is bandwidth-bound, so the
/// integer-dot path attacks the ceiling directly.
/// </para>
/// <para>
/// Result is NOT bit-exact vs the F32-in GEMV (the activation is int8-quantized
/// first). Per-32-block scaling keeps the error at the int8-activation-quant
/// level — validated against the CPU F32 oracle with an argmax-exact +
/// tolerance parity test.
/// </para>
/// <para>
/// Dispatch: one workgroup per output row, 128 threads, shared-memory
/// reduction. Same weight layout (34-byte Q8_0 block) and 2-mod-4 phase funnel
/// as <see cref="MatMulQ8_0Kernel"/>.
/// </para>
/// </remarks>
public sealed class MatMulQ8_0MmvqKernel : IDisposable
{
    /// <summary>Q8_0 block: 2 bytes fp16 scale + 32 signed int8 values.</summary>
    public const int Q8_0BlockBytes = 34;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = 32;

    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ8_0MmvqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 4);
    }

    /// <summary>
    /// Loads <c>matmul_q8_0_mmvq.spv</c> from <paramref name="spvDir"/> and
    /// builds the pipeline. Returns <c>null</c> when the SPV is missing OR when
    /// the device does not advertise integer-dot-product support — the router
    /// falls back to <see cref="MatMulQ8_0Kernel"/> in either case.
    /// </summary>
    public static MatMulQ8_0MmvqKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_q8_0_mmvq.spv");
        if (!File.Exists(path))
            return null;

        // Pin the decode GEMV to wave32 on devices that support a required
        // compute subgroup size (issue #54 / #330). On RDNA3.5 (gfx1151) the
        // driver defaults compute to wave64; forcing wave32 PER-KERNEL here —
        // never globally — matches llama.cpp's K-quant decode strategy. Gated
        // on device support so it cleanly falls back to the driver default
        // (subgroupSize=0 = unset) elsewhere, and on an env opt-out for A/B.
        uint requiredSubgroupSize = Wave32SubgroupControl.RequiredSubgroupSizeFor(device);

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            bindings[3] = new VkDescriptorBinding(3);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindings,
                pushConstantBytes: PushConstantBytes,
                requiredSubgroupSize: requiredSubgroupSize);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MatMulQ8_0MmvqKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Records the MMVQ GEMV into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="weightsQ8">Raw Q8_0 blob of <c>M * (K/32) * 34</c> bytes.</param>
    /// <param name="xq">Packed-int8 quantized activations (<see cref="QuantizeQ8_1Kernel"/>), <c>K/4</c> uints.</param>
    /// <param name="xds">Per-block (scale, sum) of the quantized activations, <c>K/32</c> vec2.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 32).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q8_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q8_0BlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ8.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ8.Size}.", nameof(weightsQ8));
        if (xq.Size < QuantizeQ8_1Kernel.PackedBytes(k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1Kernel.ScaleBytes(k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[4] { weightsQ8.Handle, xq.Handle, xds.Handle, y.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4]
        {
            (uint)m, (uint)k, (uint)blocksPerRow, (uint)rowUints,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, 1, 1);
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
