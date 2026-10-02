using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// F32 matrix multiplication: <c>C[N,M] = B[N,K] @ A[M,K]^T</c>.
/// </summary>
/// <remarks>
/// Semantic parity with <c>DotLLM.Cpu.Kernels.MatMul.GemmF32</c>:
/// <list type="bullet">
///   <item><c>A</c> is row-major <c>[M,K]</c> weight matrix.</item>
///   <item><c>B</c> is row-major <c>[N,K]</c> input matrix (one row per token).</item>
///   <item><c>C</c> is row-major <c>[N,M]</c> output matrix; <c>C[t,m] = dot(A[m,:], B[t,:])</c>.</item>
/// </list>
/// Dispatch is a 2-D grid with one thread per output cell; workgroup size
/// <c>(16, 16, 1)</c>. No cache-blocked / cooperative-matrix variant yet — that
/// arrives with milestone 8 of the Vulkan roadmap.
/// </remarks>
public sealed class MatMulF32Kernel : IDisposable
{
    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    private const int PushConstantBytes = 3 * sizeof(uint); // M, K, N

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    // Single-row (decode) variant: one workgroup per output row, vec4 loads, shared reduction. Same descriptor-set layout and
    // push constants as _pipeline, so the descriptor cache is shared. Null when the SPIR-V is absent.
    private readonly VulkanModule? _gemvModule;
    private readonly ComputePipeline? _gemvPipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool,
                            VulkanModule? gemvModule, ComputePipeline? gemvPipeline)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _gemvModule = gemvModule;
        _gemvPipeline = gemvPipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_f32.spv");
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

        VulkanModule? gemvModule = null;
        ComputePipeline? gemvPipeline = null;
        string gemvPath = Path.Combine(spvDir, "matmul_f32_gemv.spv");
        if (File.Exists(gemvPath)
            && !string.Equals(Environment.GetEnvironmentVariable("DOTLLM_VK_F32_GEMV"), "0", StringComparison.Ordinal))
        {
            gemvModule = VulkanModule.LoadFromFile(device, gemvPath);
            Span<VkDescriptorBinding> gb = stackalloc VkDescriptorBinding[3];
            gb[0] = new VkDescriptorBinding(0);
            gb[1] = new VkDescriptorBinding(1);
            gb[2] = new VkDescriptorBinding(2);
            gemvPipeline = gemvModule.CreateComputePipeline(
                entryPoint: "main", bindings: gb, pushConstantBytes: PushConstantBytes);
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new MatMulF32Kernel(device, module, pipeline, pool, gemvModule, gemvPipeline);
    }

    /// <summary>
    /// Drops every cached descriptor set and resets the underlying pool.
    /// Call when the caller has externally invalidated the buffers bound
    /// to cached sets — e.g. <see cref="VulkanForwardState.EnsureCapacity"/>
    /// re-allocated the scratch buffers that previous descriptor sets
    /// pointed at. Do NOT call this on every forward; the cache's whole
    /// point is to survive across forwards.
    /// </summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the matmul: <c>C[N,M] = B[N,K] @ A[M,K]^T</c>.
    /// Synchronous — the call returns after <c>vkQueueWaitIdle</c>. Legacy
    /// wrapper around <see cref="Record"/> for unit tests and standalone
    /// use; production forward pass uses <see cref="Record"/> directly.
    /// </summary>
    public void Launch(VulkanDevice.Buffer weightsA, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
                       int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsA, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the matmul into <paramref name="cmdBuf"/> without submitting.
    /// The caller owns the command buffer, the submission fence, and the
    /// surrounding pipeline barriers (none needed before a sequence of
    /// compute dispatches against the same buffer set aside from the
    /// standard SHADER_WRITE → SHADER_READ between kernels).
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsA">Row-major <c>[M,K]</c> FP32 weights.</param>
    /// <param name="inputB">Row-major <c>[N,K]</c> FP32 inputs.</param>
    /// <param name="outputC">Row-major <c>[N,M]</c> FP32 outputs.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Contraction dimension.</param>
    /// <param name="n">Batch size (number of input rows).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsA, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));

        long aMin = (long)m * k * sizeof(float);
        long bMin = (long)n * k * sizeof(float);
        long cMin = (long)n * m * sizeof(float);
        if (weightsA.Size < aMin) throw new ArgumentException("Weights buffer too small.", nameof(weightsA));
        if (inputB.Size < bMin) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < cMin) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[3] { weightsA.Handle, inputB.Handle, outputC.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        bool gemv = n == 1 && (k & 3) == 0 && _gemvPipeline is not null;
        var pipe = gemv ? _gemvPipeline! : _pipeline;
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipe.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, pipe.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[3] { (uint)m, (uint)k, (uint)n };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, pipe.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        if (gemv)
        {
            VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, 1, 1);
            return;
        }

        uint groupsX = (uint)((m + WorkgroupX - 1) / WorkgroupX);
        uint groupsY = (uint)((n + WorkgroupY - 1) / WorkgroupY);
        VulkanApi.vkCmdDispatch(cmdBuf, groupsX, groupsY, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_descriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _gemvPipeline?.Dispose();
        _gemvModule?.Dispose();
        _pipeline.Dispose();
        _module.Dispose();
    }
}
