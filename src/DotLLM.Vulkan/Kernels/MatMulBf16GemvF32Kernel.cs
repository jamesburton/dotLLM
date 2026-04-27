using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// BF16 native decode-path GEMV: <c>y[M] = W_bf16[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Weights stored on device as IEEE-754-truncated brain float (1 sign / 8
/// exponent / 7 mantissa bits, 2 bytes/element), row-major <c>[M, K]</c>.
/// Activations / output FP32; accumulation FP32. Companion to the BF16
/// prefill GEMM (<see cref="MatMulBf16GemmF32Kernel"/>) and the F16 siblings.
/// </para>
/// <para>
/// BF16 unpack: SPIR-V / GLSL has no native bfloat16 — the shader reconstructs
/// F32 via <c>uintBitsToFloat(bf16_bits &lt;&lt; 16)</c>. BF16 is the top 16
/// bits of the F32 binary representation, so the shift-and-reinterpret is
/// lossless. No <c>VK_KHR_cooperative_matrix</c> path for BF16 — that
/// extension exposes F16 / Sint8 operands on mainstream drivers, not BF16.
/// The decode-path GEMV stays on the scalar workgroup-reduce path; the
/// prefill GEMM stays on the scalar tiled path.
/// </para>
/// <para>
/// Accuracy: BF16 has ~7-bit mantissa (vs F16's ~10), so per-element drift
/// is larger. Kernel-parity tests use a slightly looser tolerance than F16
/// (abs 1e-2 / rel 5e-3 vs F16's abs 5e-3 / rel 1e-3) to account for this;
/// see <c>VulkanMatMulBf16GemvF32KernelTests</c> for the rationale.
/// </para>
/// <para>
/// Alignment requirement: <c>k</c> must be a multiple of 2 (each storage
/// uint holds two BF16 elements). Real-model K is always at least
/// <c>head_dim ≥ 32</c>, so this never binds in practice.
/// </para>
/// </remarks>
public sealed class MatMulBf16GemvF32Kernel : IDisposable
{
    /// <summary>Bytes per BF16 element on device.</summary>
    public const int Bf16ElementBytes = 2;

    private const int WorkgroupSize = 128;
    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, pairsPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulBf16GemvF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_bf16_gemv_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulBf16GemvF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_bf16_gemv_f32.spv");
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
        return new MatMulBf16GemvF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the BF16 GEMV synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsBf16, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsBf16, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the BF16 GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsBf16, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k & 1) != 0)
            throw new ArgumentException($"k must be a multiple of 2, got {k}", nameof(k));

        int pairsPerRow = k / 2;
        long rowBytes = (long)k * Bf16ElementBytes;
        int rowUints = pairsPerRow;

        long weightsMin = (long)m * rowBytes;
        if (weightsBf16.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsBf16.Size}.",
                nameof(weightsBf16));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsBf16.Handle, x.Handle, y.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4]
        {
            (uint)m,
            (uint)k,
            (uint)pairsPerRow,
            (uint)rowUints,
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
