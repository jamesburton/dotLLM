using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Issue #496 — quantizes an FP32 activation batch <c>x[ncols, K]</c> to the permuted int8 form
/// the int8 PQ2_0 GEMV (<see cref="MatMulPQ2_0Int8GemvKernel"/>) consumes. The Vulkan twin of
/// CUDA #485's <c>pq2_0_dp4a_quantize_x</c>.
/// </summary>
/// <remarks>
/// <para>
/// Outputs two parallel buffers, both <b>compacted</b> (column <c>s</c> starts at element
/// <c>s*K</c> regardless of the input offset):
/// <list type="bullet">
///   <item><c>xq</c>: <c>uint[ncols*K/4]</c> — 4 signed int8 per uint, permuted within every
///     16-element chunk by a 4x4 byte transpose so that
///     <c>(codeWord &gt;&gt; 2i) &amp; 0x03030303</c> lines up against word <c>i</c> with no byte
///     shuffles in the GEMV.</item>
///   <item><c>xmeta</c>: <c>uvec2[ncols*K/32]</c> — <c>(bits(d), sum q)</c> per 32-block. The
///     GEMV seeds its dot chain at <c>-sum q</c>, which is what makes the <c>code - 1</c> offset
///     free.</item>
/// </list>
/// </para>
/// <para>
/// Numerics mirror the CPU W2A8 activation quantizer
/// (<c>DotLLM.Cpu.Kernels.MatMul.QuantizeF32ToQ8_0</c>): per-32 max-abs, <c>scale = amax/127</c>,
/// <c>q = roundEven(x/scale)</c> clamped to ±127, and the dot uses the <b>half-rounded</b> scale.
/// Unlike the CUDA kernel this is not promised byte-exact — GLSL has no correctly-rounded divide
/// and <c>packHalf2x16</c>'s rounding mode is unspecified — so
/// <c>VulkanQuantizePQ2_0Int8KernelTests</c> measures the mismatch rate against the CPU quantizer
/// directly and the GEMV parity bounds are set from that.
/// </para>
/// </remarks>
public sealed class QuantizePQ2_0Int8Kernel : IDisposable
{
    /// <summary>Elements per activation quantization block.</summary>
    public const int GroupSize = 32;

    private const int PushConstantBytes = 4 * sizeof(uint); // K, blocksPerCol, totalBlocks, xOff
    private const int WorkgroupSize = 256;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private QuantizePQ2_0Int8Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Packed-int8 bytes needed for <paramref name="columns"/> rows of <paramref name="k"/> elements.</summary>
    public static long PackedBytes(int k, int columns) => (long)k * columns;

    /// <summary>Metadata bytes (two uints per 32-block) for <paramref name="columns"/> rows of <paramref name="k"/> elements.</summary>
    public static long MetaBytes(int k, int columns) => (long)(k / GroupSize) * columns * 2 * sizeof(uint);

    /// <summary>
    /// Loads <c>quantize_pq2_0_int8.spv</c> from <paramref name="spvDir"/>. Returns <c>null</c>
    /// when the SPV is missing (older builds), so the caller falls back to the float GEMV.
    /// </summary>
    /// <param name="device">Target device.</param>
    /// <param name="spvDir">Directory containing the compiled SPIR-V modules.</param>
    public static QuantizePQ2_0Int8Kernel? TryCreate(VulkanDevice device, string spvDir)
    {
        ArgumentNullException.ThrowIfNull(device);
        string path = Path.Combine(spvDir, "quantize_pq2_0_int8.spv");
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
        return new QuantizePQ2_0Int8Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the activation quantization into <paramref name="cmdBuf"/>.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="x">FP32 activations; this dispatch reads <c>columns*k</c> floats from <paramref name="xOffsetElements"/>.</param>
    /// <param name="xq">Output packed-int8 buffer, ≥ <see cref="PackedBytes"/> bytes.</param>
    /// <param name="xmeta">Output (scale bits, sum) buffer, ≥ <see cref="MetaBytes"/> bytes.</param>
    /// <param name="k">Elements per column; must be a multiple of <see cref="GroupSize"/>.</param>
    /// <param name="columns">Token count.</param>
    /// <param name="xOffsetElements">First float of column 0's activation row.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer x, VulkanDevice.Buffer xq, VulkanDevice.Buffer xmeta,
        int k, int columns, int xOffsetElements = 0)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(xq);
        ArgumentNullException.ThrowIfNull(xmeta);
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (columns <= 0) throw new ArgumentOutOfRangeException(nameof(columns));
        if (xOffsetElements < 0) throw new ArgumentOutOfRangeException(nameof(xOffsetElements));
        if ((k % GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {GroupSize}, got {k}", nameof(k));
        if (x.Size < ((long)xOffsetElements + (long)columns * k) * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (xq.Size < PackedBytes(k, columns))
            throw new ArgumentException("Packed output buffer too small.", nameof(xq));
        if (xmeta.Size < MetaBytes(k, columns))
            throw new ArgumentException("Metadata output buffer too small.", nameof(xmeta));

        int blocksPerCol = k / GroupSize;
        int totalBlocks = blocksPerCol * columns;

        Span<nint> buffers = stackalloc nint[3] { x.Handle, xq.Handle, xmeta.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4]
        {
            (uint)k,
            (uint)blocksPerCol,
            (uint)totalBlocks,
            (uint)xOffsetElements,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        }

        uint groups = (uint)((totalBlocks + WorkgroupSize - 1) / WorkgroupSize);
        VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);
    }

    /// <summary>Dispatches the quantization synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer x, VulkanDevice.Buffer xq, VulkanDevice.Buffer xmeta,
        int k, int columns, int xOffsetElements = 0)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, x, xq, xmeta, k, columns, xOffsetElements);
        ctx.SubmitAndWait();
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
