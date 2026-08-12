using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q5_0 prefill-path batched GEMM: <c>C[N, M] = B[N, K] @ W_q5_0[M, K]^T</c>.
/// </summary>
/// <remarks>
/// Companion to <see cref="MatMulQ5_0GemvF32Kernel"/> (decode-path GEMV). Same
/// byte layout: each weight row holds <c>(K / 32)</c> Q5_0 blocks of 22 bytes
/// each (fp16 d, 32-bit qh holding the 5th bit of every element, 16 bytes of
/// 4-bit qs). Output tile 16×16, K-chunk = 32 elements (one Q5_0 block) —
/// tile shape and shared-memory staging pattern copied verbatim from
/// <see cref="MatMulQ8_0GemmKernel"/> (the Q8_0 GEMM kernel's 32-element
/// direct-block staging pattern, not a new tile shape). No coopmat path —
/// follow-up ticket.
/// </remarks>
public sealed class MatMulQ5_0GemmF32Kernel : IDisposable
{
    /// <summary>Q5_0 block: 2 (d) + 4 (qh) + 16 (qs) = 22 bytes.</summary>
    public const int Q5_0BlockBytes = 22;

    /// <summary>Elements per Q5_0 block.</summary>
    public const int Q5_0GroupSize = 32;

    private const int TileM = 16;
    private const int TileN = 16;
    private const int PushConstantBytes = 5 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ5_0GemmF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_q5_0_f32_gemm.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulQ5_0GemmF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_q5_0_f32_gemm.spv");
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
        return new MatMulQ5_0GemmF32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the batched GEMM synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ5_0, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ5_0, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the Q5_0 GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ5_0, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % Q5_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q5_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q5_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q5_0BlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ5_0.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ5_0.Size}.",
                nameof(weightsQ5_0));
        long bMin = (long)n * k * sizeof(float);
        long cMin = (long)n * m * sizeof(float);
        if (inputB.Size < bMin) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < cMin) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[3] { weightsQ5_0.Handle, inputB.Handle, outputC.Handle };
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
