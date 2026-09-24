using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q8_0 decode-path GEMV: <c>y[M] = W_q8[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Weight layout mirrors the CPU kernel <c>DotLLM.Cpu.Kernels.MatMul.GemvQ8_0</c>
/// and the CUDA kernel <c>quantized_gemv_q8_0</c>: each 32 contiguous columns of
/// a row form one Q8_0 block of 34 bytes — 2 bytes fp16 scale followed by
/// 32 signed int8 quantized values.
/// </para>
/// <para>
/// The activation vector <c>x</c> is FP32 (not pre-quantized) — this kernel is
/// the N=1 decode-path; prefill / batched paths that can amortize the
/// quantization of <c>x</c> are future work (matches how <c>GemmQ8_0</c>
/// delegates to <c>GemvQ8_0</c> when N==1 on the CPU side).
/// </para>
/// <para>
/// Dispatch: one workgroup per output row. Two variants share the bindings and push constants:
/// <list type="bullet">
/// <item><description><c>matmul_q8_0_coalesced.spv</c> (#471; the default when the device has
/// subgroup arithmetic): lane = K-position, so a wave reads one contiguous run of weights per
/// iteration, and a subgroupAdd reduction.</description></item>
/// <item><description><c>matmul_q8_0.spv</c>: lane = whole block, 128 threads, shared-memory
/// reduction, no subgroup intrinsics. Its lanes read words a block stride (34 bytes) apart.
/// </description></item>
/// </list>
/// Both use the FP32 activation directly and differ only in float summation order.
/// <c>DOTLLM_VK_Q8_0_GEMV_COALESCED=0</c> selects the block-per-lane variant.
/// </para>
/// </remarks>
public sealed class MatMulQ8_0Kernel : IDisposable
{
    /// <summary>Q8_0 block: 2 bytes fp16 scale + 32 signed int8 values.</summary>
    public const int Q8_0BlockBytes = QuantFormat.Q8_0BlockBytes;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = QuantFormat.LegacyGroupSize;

    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints

    private static readonly bool CoalescedDisabledFromEnv =
        Environment.GetEnvironmentVariable("DOTLLM_VK_Q8_0_GEMV_COALESCED") == "0";

    /// <summary>
    /// Test hook for a same-process A/B (#471): <see langword="true"/> forces the coalesced
    /// variant, <see langword="false"/> the block-per-lane one, and <see langword="null"/> (the
    /// default) keeps the choice made at creation. Ignored when the coalesced variant was not
    /// created.
    /// </summary>
    internal static bool? CoalescedOverride { get; set; }

    private readonly VulkanDevice _device;
    private readonly Variant _blockPerLane;
    private readonly Variant? _coalesced;
    private readonly bool _useCoalesced;
    private bool _disposed;

    private MatMulQ8_0Kernel(VulkanDevice device, Variant blockPerLane, Variant? coalesced, bool useCoalesced)
    {
        _device = device;
        _blockPerLane = blockPerLane;
        _coalesced = coalesced;
        _useCoalesced = useCoalesced && coalesced is not null;
    }

    /// <summary>True when <see cref="Record"/> dispatches the coalesced variant, absent a test override.</summary>
    public bool UsesCoalesced => _useCoalesced;

    /// <summary>The variant the next <see cref="Record"/> dispatches, honouring the test override — for profiler census lines.</summary>
    internal string VariantName
        => _coalesced is not null && (CoalescedOverride ?? _useCoalesced) ? "coalesced" : "block";

    /// <summary>
    /// Loads <c>matmul_q8_0.spv</c> and, when the device has subgroup arithmetic and the blob is
    /// present, <c>matmul_q8_0_coalesced.spv</c>, and creates the pipelines.
    /// </summary>
    public static MatMulQ8_0Kernel Create(VulkanDevice device, string spvDir)
        => Create(device, spvDir, coalesced: null);

    /// <summary>
    /// As <see cref="Create(VulkanDevice, string)"/>, with the variant pinned for tests:
    /// <see langword="true"/> requires the coalesced variant (throws if the device cannot run it),
    /// <see langword="false"/> selects the block-per-lane one, <see langword="null"/> is the default policy.
    /// </summary>
    internal static MatMulQ8_0Kernel Create(VulkanDevice device, string spvDir, bool? coalesced)
    {
        var kernel = CreateCore(device, spvDir, coalesced ?? !CoalescedDisabledFromEnv);
        if (coalesced == true && !kernel.UsesCoalesced)
        {
            kernel.Dispose();
            throw new NotSupportedException("The coalesced Q8_0 GEMV needs subgroup arithmetic and matmul_q8_0_coalesced.spv.");
        }
        return kernel;
    }

    private static MatMulQ8_0Kernel CreateCore(VulkanDevice device, string spvDir, bool useCoalesced)
    {
        string path = Path.Combine(spvDir, "matmul_q8_0.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var blockPerLane = Variant.Load(device, path);
        Variant? coalesced = null;
        string coalescedPath = Path.Combine(spvDir, "matmul_q8_0_coalesced.spv");
        if (device.HasSubgroupArithmetic && File.Exists(coalescedPath))
        {
            try
            {
                coalesced = Variant.Load(device, coalescedPath);
            }
            catch
            {
                blockPerLane.Dispose();
                throw;
            }
        }

        return new MatMulQ8_0Kernel(device, blockPerLane, coalesced, useCoalesced);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache()
    {
        _blockPerLane.Cache.Reset();
        _coalesced?.Cache.Reset();
    }

    /// <summary>
    /// Dispatches the GEMV: <c>y[M] = W[M,K] @ x[K]</c> with FP16-scaled int8 weights.
    /// Synchronous — returns after <c>vkQueueWaitIdle</c>. Legacy wrapper around
    /// <see cref="Record"/>.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ8, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the GEMV into <paramref name="cmdBuf"/> without submitting.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsQ8">
    /// Raw Q8_0 blob of <c>M * (K/32) * 34</c> bytes, rows contiguous.
    /// </param>
    /// <param name="x">FP32 activation buffer of length <paramref name="k"/>.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 32).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q8_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q8_0BlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ8.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ8.Size}.",
                nameof(weightsQ8));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        bool coalesced = _coalesced is not null && (CoalescedOverride ?? _useCoalesced);
        var v = coalesced ? _coalesced! : _blockPerLane;

        Span<nint> buffers = stackalloc nint[3] { weightsQ8.Handle, x.Handle, y.Handle };
        nint descriptorSet = v.Cache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, v.Pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, v.Pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4]
        {
            (uint)m,
            (uint)k,
            (uint)blocksPerRow,
            (uint)rowUints,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, v.Pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _coalesced?.Dispose();
        _blockPerLane.Dispose();
    }

    /// <summary>One SPIR-V variant: module, pipeline, descriptor pool and its set cache.</summary>
    private sealed class Variant : IDisposable
    {
        private readonly VulkanDevice _device;
        private readonly VulkanModule _module;
        private readonly nint _pool;

        public ComputePipeline Pipeline { get; }
        public DescriptorSetCache Cache { get; }

        private Variant(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
        {
            _device = device;
            _module = module;
            _pool = pool;
            Pipeline = pipeline;
            Cache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
        }

        public static Variant Load(VulkanDevice device, string path)
        {
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
            return new Variant(device, module, pipeline, pool);
        }

        public void Dispose()
        {
            if (_pool != 0)
                VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
            Pipeline.Dispose();
            _module.Dispose();
        }
    }
}
