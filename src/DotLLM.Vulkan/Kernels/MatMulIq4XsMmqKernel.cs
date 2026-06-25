using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ4_XS MMQ prefill-path batched GEMM: <c>C[N, M] = B[N, K] @ W_iq4xs[M, K]^T</c>
/// via integer dp4a. The activation <c>B</c> must already be Q8_1 row-wise
/// (<see cref="QuantizeQ8_1RowsKernel"/>); this kernel decodes IQ4_XS nibbles through
/// the 16-entry int8 codebook (embedded in the shader) into shared memory, runs
/// <c>dotPacked4x8AccSatEXT</c> per 32-element sub-block, and scales by
/// <c>dl·d_x</c> (dl = d·(ls−32), no min term).
/// </summary>
/// <remarks>
/// The compute-bound seqLen&gt;1 analogue of <see cref="MatMulIq4XsMmvqKernel"/> (decode);
/// the first IQ-family prefill MMQ — structurally the codebook sibling of
/// <see cref="MatMulQ4KMmqKernel"/> without the min term. Replaces the dequant→FP GEMM
/// (<see cref="MatMulIq4XsGemmF32Kernel"/>) on the seqLen&gt;1 path; falls back to it when
/// the device lacks integer-dot-product support. NOT bit-exact (activation int8-quant);
/// validated against the CPU F32 oracle. Workgroup <c>(16,16,1)</c>.
/// </remarks>
public sealed class MatMulIq4XsMmqKernel : IDisposable
{
    /// <summary>IQ4_XS super-block: d + scales_h + scales_l[4] + qs[128].</summary>
    public const int Iq4XsBlockBytes = 136;

    /// <summary>Elements per IQ4_XS super-block.</summary>
    public const int Iq4XsGroupSize = 256;

    private const int TileM = 16;
    private const int TileN = 16;
    private const int PushConstantBytes = 5 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulIq4XsMmqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 4);
    }

    /// <summary>
    /// Loads <c>matmul_iq4_xs_mmq.spv</c> from <paramref name="spvDir"/> and builds the
    /// pipeline. Returns <c>null</c> when the SPV is missing OR the device lacks
    /// integer-dot-product support — the router falls back to
    /// <see cref="MatMulIq4XsGemmF32Kernel"/>.
    /// </summary>
    public static MatMulIq4XsMmqKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_iq4_xs_mmq.spv");
        if (!File.Exists(path))
            return null;

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
                pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MatMulIq4XsMmqKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the MMQ GEMM synchronously (one-shot submit + fence wait).</summary>
    public void Launch(
        VulkanDevice.Buffer weightsIq4Xs, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsIq4Xs, xq, xds, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the MMQ IQ4_XS GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsIq4Xs">Raw IQ4_XS blob of <c>M * (K / 256) * 136</c> bytes, rows contiguous.</param>
    /// <param name="xq">Packed-int8 quantized activations (<see cref="QuantizeQ8_1RowsKernel"/>), <c>N * K/4</c> uints.</param>
    /// <param name="xds">Per-block (scale, sum) of the quantized activations, <c>N * K/32</c> vec2.</param>
    /// <param name="outputC">FP32 output <c>[N, M]</c> row-major.</param>
    /// <param name="m">Output dimension (number of weight rows).</param>
    /// <param name="k">Contraction dimension (must be a multiple of 256).</param>
    /// <param name="n">Batch size (number of input tokens).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsIq4Xs, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % Iq4XsGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq4XsGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq4XsGroupSize;
        long rowBytes = (long)blocksPerRow * Iq4XsBlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsIq4Xs.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsIq4Xs.Size}.", nameof(weightsIq4Xs));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(n, k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(n, k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (outputC.Size < (long)n * m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[4] { weightsIq4Xs.Handle, xq.Handle, xds.Handle, outputC.Handle };
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
