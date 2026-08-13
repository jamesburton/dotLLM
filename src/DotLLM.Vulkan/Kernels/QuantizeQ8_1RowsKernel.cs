using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Multi-row activation quantizer: FP32 <c>B[N, K]</c> → Q8_1, one quantized
/// row per input row (per-32-block int8 + fp32 scale + fp32 block-sum). The
/// prefill (seqLen&gt;1) analogue of <see cref="QuantizeQ8_1Kernel"/>; feeds the
/// dp4a MMQ GEMM (<see cref="MatMulQ8_0MmqKernel"/>).
/// </summary>
/// <remarks>
/// <para>
/// Output is two parallel buffers, row-major over <c>N</c>:
/// <list type="bullet">
///   <item><c>xq</c>: <c>uint[N * K/4]</c> — 4 signed int8 lanes per uint, row
///     <c>r</c> at <c>r*(K/4)</c>.</item>
///   <item><c>xds</c>: <c>vec2[N * K/32]</c> — (scale d, sum s) per 32-block,
///     row <c>r</c> at <c>r*(K/32)</c>.</item>
/// </list>
/// </para>
/// <para>
/// Dispatch: <c>N</c> workgroups (one per row); 256 threads grid-stride over
/// the row's 32-blocks. Mirrors <see cref="QuantizeQ8_1Kernel"/> exactly except
/// for the per-row base offsets and the N-wide dispatch.
/// </para>
/// </remarks>
public sealed class QuantizeQ8_1RowsKernel : IDisposable
{
    /// <summary>Elements per Q8_1 block.</summary>
    public const int GroupSize = QuantFormat.LegacyGroupSize;

    private const int PushConstantBytes = 3 * sizeof(uint); // K, blocksPerRow, N

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private QuantizeQ8_1RowsKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>
    /// Loads <c>quantize_q8_1_rows.spv</c> from <paramref name="spvDir"/> and
    /// builds the pipeline. Returns <c>null</c> when the SPV is missing (older
    /// builds) so the router falls back to the F32-in Q8_0 GEMM.
    /// </summary>
    public static QuantizeQ8_1RowsKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "quantize_q8_1_rows.spv");
        if (!File.Exists(path))
            return null;

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
        return new QuantizeQ8_1RowsKernel(device, module, pipeline, pool);
    }

    /// <summary>Bytes the packed-int8 output buffer must hold for <paramref name="n"/> rows of <paramref name="k"/>.</summary>
    public static long PackedBytes(int n, int k) => (long)n * (k / 4) * sizeof(uint);

    /// <summary>Bytes the (scale, sum) output buffer must hold for <paramref name="n"/> rows of <paramref name="k"/>.</summary>
    public static long ScaleBytes(int n, int k) => (long)n * (k / GroupSize) * 2 * sizeof(float);

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Records the multi-row activation quantization into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="b">FP32 input <c>[N, K]</c> row-major.</param>
    /// <param name="xq">Output packed-int8 buffer, ≥ <see cref="PackedBytes"/> bytes.</param>
    /// <param name="xds">Output (scale, sum) buffer, ≥ <see cref="ScaleBytes"/> bytes.</param>
    /// <param name="n">Number of rows.</param>
    /// <param name="k">Activation length per row (must be a multiple of 32).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer b, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        int n, int k)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / GroupSize;
        if (b.Size < (long)n * k * sizeof(float)) throw new ArgumentException("Input buffer too small.", nameof(b));
        if (xq.Size < PackedBytes(n, k)) throw new ArgumentException("Packed output buffer too small.", nameof(xq));
        if (xds.Size < ScaleBytes(n, k)) throw new ArgumentException("Scale output buffer too small.", nameof(xds));

        Span<nint> buffers = stackalloc nint[3] { b.Handle, xq.Handle, xds.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[3] { (uint)k, (uint)blocksPerRow, (uint)n };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // One workgroup per row.
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)n, 1, 1);
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
