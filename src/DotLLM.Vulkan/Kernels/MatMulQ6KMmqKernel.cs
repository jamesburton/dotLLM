using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q6_K MMQ prefill-path batched GEMM: <c>C[N, M] = B[N, K] @ W_q6k[M, K]^T</c>
/// via integer dp4a. The activation <c>B</c> must already be quantized to Q8_1
/// row-wise (<see cref="QuantizeQ8_1RowsKernel"/>); this kernel decodes the Q6_K
/// ql/qh into signed <c>(q6−32)</c> int8 in shared memory, runs
/// <c>dotPacked4x8AccSatEXT</c> per 16-element sub-block, and scales by
/// <c>d·scale·d_x</c>. Q6_K is symmetric — no min term.
/// </summary>
/// <remarks>
/// <para>
/// The compute-bound seqLen&gt;1 analogue of <see cref="MatMulQ6KMmvqKernel"/> (decode)
/// and <see cref="MatMulQ4KMmqKernel"/> (Q4_K prefill). Replaces the dequant→FP GEMM
/// (<see cref="MatMulQ6KGemmF32Kernel"/>) on the seqLen&gt;1 path — Q6_K is the
/// <c>ffn_down</c>/<c>attn_v</c> type in Q4_K_M models, the part #340 left on the slow
/// path.
/// </para>
/// <para>
/// NOT bit-exact vs the F32-in GEMM (the activation is int8-quantized first). Validated
/// against the CPU F32 oracle with an argmax-exact + tolerance parity test.
/// </para>
/// <para>
/// Dispatch: 2-D grid, workgroup <c>(16, 16, 1)</c> — one 16×16 output cell of <c>C</c>
/// per workgroup. The 16-row weight tile is ql/qh-decoded into shared int8 once per
/// 256-element super-block and reused across 16 tokens.
/// </para>
/// </remarks>
public sealed class MatMulQ6KMmqKernel : IDisposable
{
    /// <summary>Q6_K super-block: ql[128] + qh[64] + scales[16] + fp16 d.</summary>
    public const int Q6KBlockBytes = QuantFormat.Q6_KBlockBytes;

    /// <summary>Elements per Q6_K super-block.</summary>
    public const int Q6KGroupSize = QuantFormat.KQuantGroupSize;

    // issue #139: 64×64 output tile per workgroup (16×16 threads × 4×4 register
    // tile each) — must match TILE_M/TILE_N in the shader.
    private const int TileM = 64;
    private const int TileN = 64;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ6KMmqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>
    /// Loads <c>matmul_q6_k_mmq.spv</c> from <paramref name="spvDir"/> and builds the
    /// pipeline. Returns <c>null</c> when the SPV is missing OR when the device does not
    /// advertise integer-dot-product support — the router falls back to
    /// <see cref="MatMulQ6KGemmF32Kernel"/>.
    /// </summary>
    public static MatMulQ6KMmqKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_q6_k_mmq.spv");
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
        return new MatMulQ6KMmqKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the MMQ GEMM synchronously (wraps <see cref="Record"/> with a one-shot
    /// submit + fence wait). Use <see cref="Record"/> directly on the forward-pass command
    /// buffer in production.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ6K, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ6K, xq, xds, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the MMQ Q6_K GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsQ6K">Raw Q6_K blob of <c>M * (K / 256) * 210</c> bytes, rows contiguous.</param>
    /// <param name="xq">Packed-int8 quantized activations (<see cref="QuantizeQ8_1RowsKernel"/>), <c>N * K/4</c> uints.</param>
    /// <param name="xds">Per-block (scale, sum) of the quantized activations, <c>N * K/32</c> vec2.</param>
    /// <param name="outputC">FP32 output <c>[N, M]</c> row-major.</param>
    /// <param name="m">Output dimension (number of weight rows).</param>
    /// <param name="k">Contraction dimension (must be a multiple of 256).</param>
    /// <param name="n">Batch size (number of input tokens).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ6K, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % Q6KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q6KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q6KGroupSize;
        long rowBytes = (long)blocksPerRow * Q6KBlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ6K.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ6K.Size}.", nameof(weightsQ6K));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(n, k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(n, k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (outputC.Size < (long)n * m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[4] { weightsQ6K.Handle, xq.Handle, xds.Handle, outputC.Handle };
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
