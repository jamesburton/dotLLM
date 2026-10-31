using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q4_0 MMQ prefill-path batched GEMM (issue #1447): <c>C[N, M] = B[N, K] @ W_q4_0[M, K]^T</c> via integer dp4a, per 32-element block
/// <c>d * d_x * dot(q - 8, x)</c> (llama.cpp's <c>vec_dot_q4_0_q8_1</c>, the CPU packed dot of #1159). The activation <c>B</c>
/// must already be Q8_1 row-wise (<see cref="QuantizeQ8_1RowsKernel"/>).
/// </summary>
/// <remarks>
/// The dp4a fallback of <see cref="MatMulMmqCm1Kernel"/> (int8 cooperative matrices, AMD) for devices with integer dot products only.
/// Dispatch: 2-D grid, workgroup <c>(16, 16, 1)</c>, one 64x64 output tile per workgroup (the <see cref="MatMulIq4NlMmqKernel"/> tiling).
/// </remarks>
public sealed class MatMulQ4_0MmqKernel : IDisposable
{
    /// <summary>Q4_0 block: fp16 d + fp16 m + qs[16].</summary>
    public const int Q4_0BlockBytes = QuantFormat.Q4_0BlockBytes;

    /// <summary>Elements per Q4_0 block.</summary>
    public const int Q4_0GroupSize = QuantFormat.LegacyGroupSize;

    private const int TileM = 64;
    private const int TileN = 64;
    private const int PushConstantBytes = 5 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ4_0MmqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>Loads <c>matmul_q4_0_mmq.spv</c>.</summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the compiled shaders.</param>
    /// <returns>The kernel, or <see langword="null"/> when the SPIR-V is missing or the device has no integer dot product.</returns>
    public static MatMulQ4_0MmqKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_q4_0_mmq.spv");
        if (!File.Exists(path))
            return null;

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            for (int i = 0; i < bindings.Length; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline(entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MatMulQ4_0MmqKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the MMQ GEMM synchronously (one-shot submit + fence wait).</summary>
    /// <param name="weightsQ4_0">Packed Q4_0 weight rows.</param>
    /// <param name="xq">Q8_1 int8 activations.</param>
    /// <param name="xds">Q8_1 per-block (d, d*sum).</param>
    /// <param name="outputC">F32 <c>[n, m]</c> output.</param>
    /// <param name="m">Weight rows.</param>
    /// <param name="k">Contraction length (multiple of 32).</param>
    /// <param name="n">Token rows.</param>
    public void Launch(
        VulkanDevice.Buffer weightsQ4_0, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ4_0, xq, xds, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the MMQ Q4_0 GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Command buffer in the recording state.</param>
    /// <param name="weightsQ4_0">Packed Q4_0 weight rows.</param>
    /// <param name="xq">Q8_1 int8 activations (<see cref="QuantizeQ8_1RowsKernel"/> layout).</param>
    /// <param name="xds">Q8_1 per-block (d, d*sum).</param>
    /// <param name="outputC">F32 <c>[n, m]</c> output.</param>
    /// <param name="m">Weight rows.</param>
    /// <param name="k">Contraction length (multiple of 32).</param>
    /// <param name="n">Token rows.</param>
    /// <exception cref="ArgumentException">A dimension or buffer size is invalid.</exception>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ4_0, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % Q4_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q4_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q4_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q4_0BlockBytes;
        // 18-byte blocks are not uint-aligned: the shader addresses bytes.
        if (rowBytes > int.MaxValue / 2) throw new ArgumentException("Row too long.", nameof(k));

        long weightsMin = (long)m * rowBytes;
        if (weightsQ4_0.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ4_0.Size}.", nameof(weightsQ4_0));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(n, k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(n, k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (outputC.Size < (long)n * m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[4] { weightsQ4_0.Handle, xq.Handle, xds.Handle, outputC.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5]
        {
            (uint)m, (uint)k, (uint)n, (uint)blocksPerRow, (uint)rowBytes,
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
