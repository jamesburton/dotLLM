using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q8_0 batched GEMM via DP4a (<c>VK_KHR_shader_integer_dot_product</c>):
/// <c>C[N, M] = B[N, K] @ W_q8[M, K]^T</c>, both operands kept in INT8.
/// </summary>
/// <remarks>
/// <para>
/// Tiled-GEMM analogue of <see cref="MatMulQ8_0Dp4aPqKernel"/> (decode GEMV).
/// Targets compute-bound prefill (<c>seqLen = N &gt; 1</c>) on devices that
/// accelerate <c>dotPacked4x8AccSatEXT</c> but lack
/// <c>VK_KHR_cooperative_matrix</c> — e.g. Intel Arc Xe-LPG. Binding and
/// push-constant layout match <see cref="MatMulQ8_0GemmKernel"/> /
/// <see cref="MatMulQ8_0GemmCoopmatKernel"/> (3 storage buffers; M, K, N,
/// blocksPerRow, rowUints) so it is a drop-in dispatch alternative.
/// </para>
/// <para>
/// <b>Probe-scope kernel.</b> The shader handles perfect-multiple shapes only
/// (<c>M % 16 == 0</c>, <c>N % 16 == 0</c>, <c>K % 32 == 0</c>); it exists to
/// measure the DP4a-GEMM compute ceiling against the scalar GEMM on Arc. The
/// activation is re-quantized to INT8 per 32-block in-shader, so results carry
/// the activation's own Q8 rounding on top of the weight's — compare at
/// coopmat-grade tolerance, not the scalar GEMM's 1e-4 bar.
/// </para>
/// </remarks>
public sealed class MatMulQ8_0GemmDp4aKernel : IDisposable
{
    /// <summary>Q8_0 block: 2 bytes fp16 scale + 32 signed int8 values.</summary>
    public const int Q8_0BlockBytes = 34;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = 32;

    private const int TileM = 16;
    private const int TileN = 16;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ8_0GemmDp4aKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>
    /// Loads <c>matmul_q8_0_gemm_dp4a.spv</c> and creates the pipeline. Requires
    /// <see cref="VulkanDevice.HasIntegerDotProduct"/> — throws
    /// <see cref="InvalidOperationException"/> otherwise.
    /// </summary>
    /// <param name="device">Vulkan device with integer-dot-product support.</param>
    /// <param name="spvDir">Directory containing the compiled SPIR-V blobs.</param>
    /// <returns>An initialized kernel.</returns>
    public static MatMulQ8_0GemmDp4aKernel Create(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            throw new InvalidOperationException(
                "MatMulQ8_0GemmDp4aKernel requires VK_KHR_shader_integer_dot_product support. " +
                "Check VulkanDevice.HasIntegerDotProduct before calling Create().");

        string path = Path.Combine(spvDir, "matmul_q8_0_gemm_dp4a.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
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
        return new MatMulQ8_0GemmDp4aKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the DP4a GEMM synchronously (one-shot submit + fence wait).
    /// Use <see cref="Record"/> directly on a shared command buffer in production.
    /// </summary>
    /// <param name="weightsQ8">Raw Q8_0 blob, rows contiguous.</param>
    /// <param name="inputB">FP32 input <c>[N, K]</c> row-major.</param>
    /// <param name="outputC">FP32 output <c>[N, M]</c> row-major.</param>
    /// <param name="m">Output dimension (number of weight rows; multiple of 16).</param>
    /// <param name="k">Contraction dimension (multiple of 32).</param>
    /// <param name="n">Batch size (multiple of 16).</param>
    public void Launch(
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ8, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the DP4a Q8_0 GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsQ8">Raw Q8_0 blob of <c>M * (K / 32) * 34</c> bytes, rows contiguous.</param>
    /// <param name="inputB">FP32 input <c>[N, K]</c> row-major.</param>
    /// <param name="outputC">FP32 output <c>[N, M]</c> row-major.</param>
    /// <param name="m">Output dimension (number of weight rows; multiple of 16).</param>
    /// <param name="k">Contraction dimension (multiple of 32).</param>
    /// <param name="n">Batch size (multiple of 16).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % Q8_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));
        if ((m % TileM) != 0)
            throw new ArgumentException($"probe kernel requires m % {TileM} == 0, got {m}", nameof(m));
        if ((n % TileN) != 0)
            throw new ArgumentException($"probe kernel requires n % {TileN} == 0, got {n}", nameof(n));

        int blocksPerRow = k / Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q8_0BlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ8.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ8.Size}.",
                nameof(weightsQ8));
        long bMin = (long)n * k * sizeof(float);
        long cMin = (long)n * m * sizeof(float);
        if (inputB.Size < bMin) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < cMin) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[3] { weightsQ8.Handle, inputB.Handle, outputC.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5]
        {
            (uint)m,
            (uint)k,
            (uint)n,
            (uint)blocksPerRow,
            (uint)rowUints,
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
