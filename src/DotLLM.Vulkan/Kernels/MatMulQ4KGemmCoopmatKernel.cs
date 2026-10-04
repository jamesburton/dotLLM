using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q4_K batched GEMM on the shared 128x128, BK=32, four-wave64-subgroup cooperative-matrix tile
/// (issue #570): <c>C[N, M] = B[N, K] @ W_q4k[M, K]^T</c>, weights dequantised on the fly into F16
/// operands with an F32 accumulator. Same bindings and push constants as
/// <see cref="MatMulQ4KGemmF32Kernel"/>, which remains the fallback.
/// </summary>
/// <remarks>
/// <b>wave64 ONLY, and that is a correctness gate.</b> The shader declares <c>local_size_x = 256</c>
/// and lays four subgroups out as a 2x2 grid; on a 32-wide device those threads form eight
/// subgroups and ids 4-7 index past the grid (see <c>Q8_0GemmCoopmatVariant.Blocked128x128x4</c>).
/// Check <see cref="IsSupportedOn"/> before <see cref="Create"/>.
/// </remarks>
public sealed class MatMulQ4KGemmCoopmatKernel : IDisposable
{
    /// <summary>Q4_K super-block: 144 bytes.</summary>
    public const int Q4KBlockBytes = QuantFormat.Q4_KBlockBytes;

    /// <summary>Elements per Q4_K super-block.</summary>
    public const int Q4KGroupSize = QuantFormat.KQuantGroupSize;

    /// <summary>SPIR-V file name of the shader.</summary>
    public const string SpvFileName = "matmul_q4_k_gemm_coopmat_128x128x4.spv";

    /// <summary>Environment variable that disables this kernel so the tiled F32 GEMM is used (A/B).</summary>
    public const string LegacyEnvVar = "DOTLLM_VK_Q4_K_GEMM_LEGACY";

    private const int Tile = 128;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private CoopmatSplitK? _splitK;
    private bool _disposed;

    private MatMulQ4KGemmCoopmatKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>
    /// Whether the kernel can run on <paramref name="device"/>: cooperative matrix present, native
    /// wave64, the SPIR-V compiled, and not disabled by <see cref="LegacyEnvVar"/>.
    /// </summary>
    public static bool IsSupportedOn(VulkanDevice device, string spvDir)
        => Environment.GetEnvironmentVariable(LegacyEnvVar) != "1"
           && device.HasCooperativeMatrix
           && device.SubgroupSize == 64
           && File.Exists(Path.Combine(spvDir, SpvFileName));

    /// <summary>Loads the SPIR-V and creates the pipeline. Check <see cref="IsSupportedOn"/> first.</summary>
    public static MatMulQ4KGemmCoopmatKernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, SpvFileName);
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
                entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes,
                requiredSubgroupSize: 0);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        var kernel = new MatMulQ4KGemmCoopmatKernel(device, module, pipeline, pool);
        kernel._splitK = CoopmatSplitK.TryCreate(device, spvDir, "matmul_q4_k_gemm_coopmat_128x128x4_splitk.spv",
            QuantFormat.Q4_KBlockBytes, QuantFormat.KQuantGroupSize);
        return kernel;
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache()
    {
        _descriptorCache.Reset();
        _splitK?.InvalidateDescriptorCache();
    }

    /// <summary>Dispatches synchronously (one-shot submit + fence wait); production uses <see cref="Record"/>.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ4K, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ4K, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="weightsQ4K">Raw Q4_K blob of <c>M * (K / 256) * 144</c> bytes, rows contiguous.</param>
    /// <param name="inputB">FP32 input <c>[N, K]</c>.</param>
    /// <param name="outputC">FP32 output <c>[N, M]</c>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Contraction dimension (multiple of 256).</param>
    /// <param name="n">Batch size.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ4K, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % Q4KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q4KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q4KGroupSize;
        long rowBytes = (long)blocksPerRow * Q4KBlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        if (weightsQ4K.Size < (long)m * rowBytes)
            throw new ArgumentException("Weights buffer too small.", nameof(weightsQ4K));
        if (inputB.Size < (long)n * k * sizeof(float)) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < (long)n * m * sizeof(float)) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        // Underfilled grids (small m at small n) run split-K: see CoopmatSplitK.
        if (_splitK is not null)
        {
            int splits = CoopmatSplitK.ChooseSplits(m, n, k / QuantFormat.KQuantGroupSize * 8);
            if (splits > 1)
            {
                _splitK.Record(cmdBuf, weightsQ4K, inputB, outputC, m, k, n, splits);
                return;
            }
        }

        Span<nint> buffers = stackalloc nint[3] { weightsQ4K.Handle, inputB.Handle, outputC.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5] { (uint)m, (uint)k, (uint)n, (uint)blocksPerRow, (uint)rowUints };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + Tile - 1) / Tile), (uint)((n + Tile - 1) / Tile), 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _splitK?.Dispose();
        if (_descriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
