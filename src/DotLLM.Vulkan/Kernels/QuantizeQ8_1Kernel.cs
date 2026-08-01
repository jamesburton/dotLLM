using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Quantizes an FP32 activation row <c>x[K]</c> to Q8_1 on the device:
/// per-32-block int8 values + an fp32 scale and an fp32 block-sum. Feeds the
/// dp4a MMVQ decode GEMV (<see cref="MatMulQ8_0MmvqKernel"/>), turning the
/// weight×activation dot product into an integer dp4a.
/// </summary>
/// <remarks>
/// <para>
/// Output is two parallel buffers (so the MMVQ inner loop streams contiguous
/// packed int8 and reads the scales separately):
/// <list type="bullet">
///   <item><c>xq</c>: <c>uint[K/4]</c> — 4 signed int8 lanes per uint.</item>
///   <item><c>xds</c>: <c>vec2[K/32]</c> — (scale d, sum s) per 32-block.</item>
/// </list>
/// The block-sum <c>s = d * sum(qs)</c> is unused by the symmetric Q8_0 weight
/// path but is required by asymmetric (K-quant) weight formats' min term; it is
/// produced here so the same activation buffer is reusable by those kernels.
/// </para>
/// <para>
/// Dispatch: a single workgroup over all of <c>K</c> (decode path =&gt; one
/// activation row, K small). 256 threads grid-stride over the 32-blocks.
/// </para>
/// </remarks>
public sealed class QuantizeQ8_1Kernel : IDisposable
{
    /// <summary>Elements per Q8_1 block.</summary>
    public const int GroupSize = 32;

    private const int PushConstantBytes = 2 * sizeof(uint); // K, blocksPerRow

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private QuantizeQ8_1Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>
    /// Loads <c>quantize_q8_1.spv</c> from <paramref name="spvDir"/> and builds
    /// the pipeline. Returns <c>null</c> when the SPV is missing (older builds)
    /// so the router falls back to the F32-in Q8_0 GEMV.
    /// </summary>
    public static QuantizeQ8_1Kernel? TryCreate(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "quantize_q8_1.spv");
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
        return new QuantizeQ8_1Kernel(device, module, pipeline, pool);
    }

    /// <summary>Bytes the packed-int8 output buffer must hold for the given K.</summary>
    public static long PackedBytes(int k) => (long)(k / 4) * sizeof(uint);

    /// <summary>Bytes the (scale, sum) output buffer must hold for the given K.</summary>
    public static long ScaleBytes(int k) => (long)(k / GroupSize) * 2 * sizeof(float);

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Records the activation quantization into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="x">FP32 activation buffer of length <paramref name="k"/>.</param>
    /// <param name="xq">Output packed-int8 buffer, ≥ <see cref="PackedBytes"/> bytes.</param>
    /// <param name="xds">Output (scale, sum) buffer, ≥ <see cref="ScaleBytes"/> bytes.</param>
    /// <param name="k">Activation length (must be a multiple of 32).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer x, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        int k)
    {
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / GroupSize;
        if (x.Size < (long)k * sizeof(float)) throw new ArgumentException("Input buffer too small.", nameof(x));
        if (xq.Size < PackedBytes(k)) throw new ArgumentException("Packed output buffer too small.", nameof(xq));
        if (xds.Size < ScaleBytes(k)) throw new ArgumentException("Scale output buffer too small.", nameof(xds));

        Span<nint> buffers = stackalloc nint[3] { x.Handle, xq.Handle, xds.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[2] { (uint)k, (uint)blocksPerRow };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // Single workgroup over all of K (decode-path: one activation row).
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
