using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q8_0 MMQ prefill-path batched GEMM: <c>C[N, M] = B[N, K] @ W_q8[M, K]^T</c>
/// via integer dp4a. The activation <c>B</c> must already be quantized to Q8_1
/// row-wise (<see cref="QuantizeQ8_1RowsKernel"/>); this kernel runs
/// <c>dotPacked4x8AccSatEXT</c> (4×int8 → int32 saturating accumulate) against
/// the int8 Q8_0 weights, then scales each 32-block by
/// <c>d_weight · d_activation</c>.
/// </summary>
/// <remarks>
/// <para>
/// The compute-bound prefill analogue of <see cref="MatMulQ8_0MmvqKernel"/>.
/// Replaces the dequant→FP GEMM (<see cref="MatMulQ8_0GemmKernel"/> /
/// <see cref="MatMulQ8_0GemmCoopmatKernel"/>) on the seqLen&gt;1 path on devices
/// that advertise <see cref="VulkanDevice.HasIntegerDotProduct"/>: the integer
/// dot keeps the inner loop int8 (no per-element dequant FMA) and halves
/// activation read traffic, which is where the literature 2–4.5× prefill gains
/// come from on iGPU-class hardware.
/// </para>
/// <para>
/// Result is NOT bit-exact vs the F32-in GEMM (the activation is int8-quantized
/// first). Per-32-block scaling keeps the error at the int8-activation-quant
/// level — validated against the CPU F32 oracle with an argmax-exact +
/// tolerance parity test.
/// </para>
/// <para>
/// Dispatch: 2-D grid, workgroup <c>(16, 16, 1)</c> — one 64×64 output tile of
/// <c>C</c> per workgroup (issue #366 register-tiled rewrite, mirroring the
/// #139 Q4_K/Q6_K/IQ4_XS 64×64 tiling that this kernel was NOT included in at
/// the time — #139's scope was iq4_xs/q4_k/q6_k only). Each thread computes a
/// 4×4 register tile (strided by 16). The 64-row weight tile is funnel-read
/// into shared memory as packed int8 once per K-block and reused across 64
/// tokens; the 64-token activation tile is reused across 64 weight rows. Same
/// 34-byte Q8_0 block layout and 2-mod-4 phase funnel as
/// <see cref="MatMulQ8_0GemmKernel"/>.
/// </para>
/// </remarks>
public sealed class MatMulQ8_0MmqKernel : IDisposable
{
    /// <summary>Q8_0 block: 2 bytes fp16 scale + 32 signed int8 values.</summary>
    public const int Q8_0BlockBytes = 34;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = 32;

    private const int TileM = 64;
    private const int TileN = 64;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ8_0MmqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>
    /// Loads <c>matmul_q8_0_mmq.spv</c> from <paramref name="spvDir"/> and builds
    /// the pipeline. Returns <c>null</c> when the SPV is missing OR when the
    /// device does not advertise integer-dot-product support — the router falls
    /// back to <see cref="MatMulQ8_0GemmCoopmatKernel"/> /
    /// <see cref="MatMulQ8_0GemmKernel"/> in either case.
    /// </summary>
    public static MatMulQ8_0MmqKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_q8_0_mmq.spv");
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
        return new MatMulQ8_0MmqKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the MMQ GEMM synchronously (wraps <see cref="Record"/> with a
    /// one-shot submit + fence wait). Use <see cref="Record"/> directly on the
    /// forward-pass command buffer in production.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ8, xq, xds, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the MMQ Q8_0 GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsQ8">Raw Q8_0 blob of <c>M * (K / 32) * 34</c> bytes, rows contiguous.</param>
    /// <param name="xq">Packed-int8 quantized activations (<see cref="QuantizeQ8_1RowsKernel"/>), <c>N * K/4</c> uints.</param>
    /// <param name="xds">Per-block (scale, sum) of the quantized activations, <c>N * K/32</c> vec2.</param>
    /// <param name="outputC">FP32 output <c>[N, M]</c> row-major.</param>
    /// <param name="m">Output dimension (number of weight rows).</param>
    /// <param name="k">Contraction dimension (must be a multiple of 32).</param>
    /// <param name="n">Batch size (number of input tokens).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer outputC,
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
        if (weightsQ8.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ8.Size}.", nameof(weightsQ8));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(n, k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(n, k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (outputC.Size < (long)n * m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[4] { weightsQ8.Handle, xq.Handle, xds.Handle, outputC.Handle };
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
