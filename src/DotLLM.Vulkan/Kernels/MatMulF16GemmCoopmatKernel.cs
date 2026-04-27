using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// F16 native batched GEMM via <c>VK_KHR_cooperative_matrix</c>:
/// <c>C[N, M] = B[N, K] @ W_f16[M, K]^T</c>.
/// </summary>
/// <remarks>
/// <para>
/// Semantic and binding parity with <see cref="MatMulF16GemmF32Kernel"/> —
/// same descriptor layout (3 storage buffers: F16 weights, F32 input, F32
/// output), same push constants (M, K, N, pairsPerRow, rowUints), same weight
/// byte format. Drift versus the scalar kernel is the standard F16-vs-F32
/// staging delta — within abs 5e-3 / rel 1e-3 of the scalar reference at
/// the K shapes the parity tests cover.
/// </para>
/// <para>
/// Availability: this kernel requires the physical device to advertise
/// <c>VK_KHR_cooperative_matrix</c> with a 16x16x16 F16xF16->F32 subgroup
/// tile. Callers must check <see cref="VulkanDevice.HasCooperativeMatrix"/>
/// before calling <see cref="Create"/> — otherwise an exception is thrown.
/// The orchestrator wires runtime dispatch selection (coopmat vs scalar) in
/// <c>RecordMatmul</c>.
/// </para>
/// <para>
/// Dispatch: 2-D grid, workgroup <c>(64, 1, 1)</c> — one subgroup per
/// output tile. Tile shape: 16 rows x 16 cols of C, K stepped 32 at a time
/// with TK=16 (two coopMatMulAdd per chunk).
/// </para>
/// </remarks>
public sealed class MatMulF16GemmCoopmatKernel : IDisposable
{
    /// <summary>Bytes per F16 element on device.</summary>
    public const int F16ElementBytes = 2;

    /// <summary>K must be a multiple of this value.</summary>
    public const int KChunk = 32;

    private const int TileM = 16;
    private const int TileN = 16;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, pairsPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulF16GemmCoopmatKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>
    /// Loads <c>matmul_f16_gemm_coopmat.spv</c> from the given directory and
    /// creates the pipeline. Requires
    /// <see cref="VulkanDevice.HasCooperativeMatrix"/> to be <c>true</c> —
    /// throws <see cref="InvalidOperationException"/> otherwise so the caller
    /// can fall back to <see cref="MatMulF16GemmF32Kernel"/>.
    /// </summary>
    public static MatMulF16GemmCoopmatKernel Create(VulkanDevice device, string spvDir)
    {
        if (!device.HasCooperativeMatrix)
            throw new InvalidOperationException(
                "MatMulF16GemmCoopmatKernel requires VK_KHR_cooperative_matrix support. " +
                "Check VulkanDevice.HasCooperativeMatrix before calling Create() and fall " +
                "back to MatMulF16GemmF32Kernel when it is false.");

        string path = Path.Combine(spvDir, "matmul_f16_gemm_coopmat.spv");
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
        return new MatMulF16GemmCoopmatKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the coopmat GEMM synchronously (wraps <see cref="Record"/>
    /// with a one-shot submit + fence wait).
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer weightsF16, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsF16, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the coopmat F16 GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsF16, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % KChunk) != 0)
            throw new ArgumentException($"k must be a multiple of {KChunk}, got {k}", nameof(k));

        int pairsPerRow = k / 2;
        long rowBytes = (long)k * F16ElementBytes;
        int rowUints = pairsPerRow;

        long weightsMin = (long)m * rowBytes;
        if (weightsF16.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsF16.Size}.",
                nameof(weightsF16));
        long bMin = (long)n * k * sizeof(float);
        long cMin = (long)n * m * sizeof(float);
        if (inputB.Size < bMin) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < cMin) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[3] { weightsF16.Handle, inputB.Handle, outputC.Handle };
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
            (uint)pairsPerRow,
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
