using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ2_S prefill-path batched GEMM: <c>C[N, M] = B[N, K] @ W_iq2s[M, K]^T</c>.
/// </summary>
public sealed class MatMulIq2SGemmF32Kernel : IDisposable
{
    /// <summary>IQ2 super-block size in bytes.</summary>
    public const int IQ2_SBlockBytes = 82;
    /// <summary>Elements per IQ2 super-block.</summary>
    public const int IQ2_SGroupSize = 256;

    private const int TileM = 16;
    private const int TileN = 16;
    private const int PushConstantBytes = 5 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly Iq2Codebooks _codebooks;
    private readonly bool _ownsCodebooks;
    private bool _disposed;

    private MatMulIq2SGemmF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool,
        Iq2Codebooks codebooks, bool ownsCodebooks)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 4);
        _codebooks = codebooks;
        _ownsCodebooks = ownsCodebooks;
    }

    /// <summary>Loads the matching .spv, allocates the shared codebook SSBOs, and creates the pipeline.</summary>
    public static MatMulIq2SGemmF32Kernel Create(VulkanDevice device, string spvDir)
    {
        var codebooks = Iq2Codebooks.Create(device);
        try { return CreateInternal(device, spvDir, codebooks, ownsCodebooks: true); }
        catch { codebooks.Dispose(); throw; }
    }

    internal static MatMulIq2SGemmF32Kernel CreateWithCodebooks(VulkanDevice device, string spvDir, Iq2Codebooks codebooks)
        => CreateInternal(device, spvDir, codebooks, ownsCodebooks: false);

    private static MatMulIq2SGemmF32Kernel CreateInternal(VulkanDevice device, string spvDir, Iq2Codebooks codebooks, bool ownsCodebooks)
    {
        string path = Path.Combine(spvDir, "matmul_iq2_s_f32_gemm.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            for (int i = 0; i < 4; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MatMulIq2SGemmF32Kernel(device, module, pipeline, pool, codebooks, ownsCodebooks);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the GEMM synchronously.</summary>
    public void Launch(VulkanDevice.Buffer weights, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weights, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weights, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % IQ2_SGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {IQ2_SGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / IQ2_SGroupSize;
        long rowBytes = (long)blocksPerRow * IQ2_SBlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weights.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weights.Size}.", nameof(weights));
        long bMin = (long)n * k * sizeof(float);
        long cMin = (long)n * m * sizeof(float);
        if (inputB.Size < bMin) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < cMin) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[4]
        {
            weights.Handle, inputB.Handle, outputC.Handle, _codebooks.Iq2SGrid.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5] { (uint)m, (uint)k, (uint)n, (uint)blocksPerRow, (uint)rowUints };
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
        if (_ownsCodebooks) _codebooks.Dispose();
    }
}
