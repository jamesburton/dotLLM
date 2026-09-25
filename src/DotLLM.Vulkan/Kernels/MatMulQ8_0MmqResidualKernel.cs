using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Residual-fused variant of <see cref="MatMulQ8_0MmqKernel"/> (issue #379):
/// <c>C[N, M] = (B[N, K] @ W_q8[M, K]^T) + residual[N, M]</c> in a single
/// dispatch.
/// </summary>
/// <remarks>
/// <para>
/// The compute-bound prefill analogue of
/// <see cref="MatMulQ8_0MmvqResidualKernel"/>. Replaces the (MMQ prefill GEMM
/// → barrier → <see cref="AddKernel"/>) pair at residual-connection call
/// sites (o_proj, down_proj), eliminating one dispatch and one pipeline
/// barrier per fused site per layer.
/// </para>
/// <para>
/// The dp4a register-tile accumulation is byte-for-byte identical to
/// <see cref="MatMulQ8_0MmqKernel"/>; only the final per-cell store changes
/// from <c>c[idx] = acc</c> to <c>c[idx] = acc + residual[idx]</c>. This
/// keeps the fused result bit-exact vs running the unfused kernel then a
/// separate <see cref="AddKernel"/> pass (same operands, same operation,
/// same order).
/// </para>
/// </remarks>
public sealed class MatMulQ8_0MmqResidualKernel : IDisposable
{
    /// <summary>Q8_0 block: 2 bytes fp16 scale + 32 signed int8 values.</summary>
    public const int Q8_0BlockBytes = QuantFormat.Q8_0BlockBytes;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = QuantFormat.LegacyGroupSize;

    private const int TileM = 64;
    private const int TileN = 64;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ8_0MmqResidualKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 5);
    }

    /// <summary>
    /// Loads <c>matmul_q8_0_mmq_residual.spv</c> from <paramref name="spvDir"/>
    /// and builds the pipeline. Returns <c>null</c> when the SPV is missing OR
    /// when the device does not advertise integer-dot-product support — the
    /// router falls back to the unfused <see cref="MatMulQ8_0MmqKernel"/> +
    /// <see cref="AddKernel"/> pair in either case.
    /// </summary>
    public static MatMulQ8_0MmqResidualKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_q8_0_mmq_residual.spv");
        if (!File.Exists(path))
            return null;

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
                pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        return new MatMulQ8_0MmqResidualKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the residual-fused MMQ Q8_0 GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsQ8">Raw Q8_0 blob of <c>M * (K / 32) * 34</c> bytes, rows contiguous.</param>
    /// <param name="xq">Packed-int8 quantized activations (<see cref="QuantizeQ8_1RowsKernel"/>), <c>N * K/4</c> uints.</param>
    /// <param name="xds">Per-block (scale, sum) of the quantized activations, <c>N * K/32</c> vec2.</param>
    /// <param name="residual">FP32 residual <c>[N, M]</c> row-major, added to the matmul result.</param>
    /// <param name="outputC">FP32 output <c>[N, M]</c> row-major.</param>
    /// <param name="m">Output dimension (number of weight rows).</param>
    /// <param name="k">Contraction dimension (must be a multiple of 32).</param>
    /// <param name="n">Batch size (number of input tokens).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer residual, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % Q8_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q8_0BlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        long outputMin = (long)n * m * sizeof(float);
        if (weightsQ8.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ8.Size}.", nameof(weightsQ8));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(n, k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(n, k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (residual.Size < outputMin)
            throw new ArgumentException("Residual buffer too small.", nameof(residual));
        if (outputC.Size < outputMin)
            throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[5]
        {
            weightsQ8.Handle, xq.Handle, xds.Handle, residual.Handle, outputC.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5]
        {
            (uint)m, (uint)k, (uint)n, (uint)blocksPerRow, (uint)rowUints,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupsX = (uint)((m + TileM - 1) / TileM);
        uint groupsY = (uint)((n + TileN - 1) / TileN);
        VulkanApi.vkCmdDispatch(cmdBuf, groupsX, groupsY, 1);
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
