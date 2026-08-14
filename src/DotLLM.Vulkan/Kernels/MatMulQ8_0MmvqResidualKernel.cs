using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Residual-fused variant of <see cref="MatMulQ8_0MmvqKernel"/> (issue #379):
/// <c>y[m] = (W_q8[M,K] @ x[K])[m] + residual[m]</c> in a single dispatch.
/// </summary>
/// <remarks>
/// <para>
/// Replaces the (MMVQ decode GEMV → barrier → <see cref="AddKernel"/>) pair
/// at residual-connection call sites (o_proj, down_proj). A same-session GPU
/// benchmark (<c>VulkanResidualAddOverheadBench</c>) measured the separate
/// add dispatch costing ~78% of the matmul's own dispatch time at SmolLM-135M
/// decode scale — fusing it into the matmul's final store removes that
/// dispatch and its barrier entirely.
/// </para>
/// <para>
/// The reduction (subgroup dp4a + subgroupAdd) is byte-for-byte identical to
/// <see cref="MatMulQ8_0MmvqKernel"/>; only the final store at the elected
/// lane changes from <c>y[m] = total</c> to <c>y[m] = total + residual[m]</c>.
/// This keeps the fused result bit-exact vs running the unfused kernel then
/// a separate <see cref="AddKernel"/> pass (same operands, same operation,
/// same order).
/// </para>
/// </remarks>
public sealed class MatMulQ8_0MmvqResidualKernel : IDisposable
{
    /// <summary>Q8_0 block: 2 bytes fp16 scale + 32 signed int8 values.</summary>
    public const int Q8_0BlockBytes = QuantFormat.Q8_0BlockBytes;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = QuantFormat.LegacyGroupSize;

    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ8_0MmvqResidualKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 5);
    }

    /// <summary>
    /// Loads <c>matmul_q8_0_mmvq_residual.spv</c> from <paramref name="spvDir"/>
    /// and builds the pipeline. Returns <c>null</c> when the SPV is missing OR
    /// when the device does not advertise integer-dot-product support — the
    /// router falls back to the unfused <see cref="MatMulQ8_0MmvqKernel"/> +
    /// <see cref="AddKernel"/> pair in either case.
    /// </summary>
    public static MatMulQ8_0MmvqResidualKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_q8_0_mmvq_residual.spv");
        if (!File.Exists(path))
            return null;

        // Same wave32 pin as the non-fused MMVQ kernel (issue #54 / #330) —
        // this shader is the identical reduction with a fused final store.
        uint requiredSubgroupSize = Wave32SubgroupControl.RequiredSubgroupSizeFor(device);

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[5];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            bindings[3] = new VkDescriptorBinding(3);
            bindings[4] = new VkDescriptorBinding(4);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        return new MatMulQ8_0MmvqResidualKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Records the residual-fused MMVQ GEMV into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="weightsQ8">Raw Q8_0 blob of <c>M * (K/32) * 34</c> bytes.</param>
    /// <param name="xq">Packed-int8 quantized activations (<see cref="QuantizeQ8_1Kernel"/>), <c>K/4</c> uints.</param>
    /// <param name="xds">Per-block (scale, sum) of the quantized activations, <c>K/32</c> vec2.</param>
    /// <param name="residual">FP32 residual buffer of length <paramref name="m"/>, added to the matmul result.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 32).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer residual, VulkanDevice.Buffer y,
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
        if (residual.Size < (long)m * sizeof(float))
            throw new ArgumentException("Residual buffer too small.", nameof(residual));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[5]
        {
            weightsQ8.Handle, xq.Handle, xds.Handle, residual.Handle, y.Handle,
        };
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
