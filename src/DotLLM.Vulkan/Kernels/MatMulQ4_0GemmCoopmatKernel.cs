using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q4_0 batched GEMM on the shared 128x128, BK=32, four-wave64-subgroup cooperative-matrix tile
/// (issue #1018, templated from the Q5_0 kernel of #568): <c>C[N, M] = B[N, K] @ W_q4_0[M, K]^T</c>, weights dequantised on the fly into F16
/// operands with an F32 accumulator. There is no scalar F32 Q4_0 GEMM: a device
/// without cooperative matrices / wave64 does not get native Q4_0 projections (the LFM2 loader transcodes to Q8_0 instead).
/// </summary>
/// <remarks>
/// <b>wave64 ONLY, and that is a correctness gate.</b> The shader declares <c>local_size_x = 256</c>
/// and lays four subgroups out as a 2x2 grid; on a 32-wide device those threads form eight
/// subgroups and ids 4-7 index past the grid (see <c>Q8_0GemmCoopmatVariant.Blocked128x128x4</c>).
/// Check <see cref="IsSupportedOn"/> before <see cref="Create"/>.
/// </remarks>
public sealed class MatMulQ4_0GemmCoopmatKernel : IDisposable
{
    /// <summary>Q4_0 block: fp16 scale + 16 nibble bytes.</summary>
    public const int Q4_0BlockBytes = QuantFormat.Q4_0BlockBytes;

    /// <summary>Elements per Q4_0 block.</summary>
    public const int Q4_0GroupSize = QuantFormat.LegacyGroupSize;

    /// <summary>SPIR-V file name of the shader.</summary>
    public const string SpvFileName = "matmul_q4_0_f32_gemm_coopmat_128x128x4.spv";

    /// <summary>Environment variable that disables this kernel so the caller declines the native path (A/B).</summary>
    public const string LegacyEnvVar = "DOTLLM_VK_Q4_0_GEMM_LEGACY";

    private const int Tile = 128;

    /// <summary>SPIR-V of the 64x64 small-grid twin (issue #1161; only the Q4_1 instantiation ships).</summary>
    public const string SmallSpvFileName = "matmul_q4_0_f32_gemm_coopmat_64x64x4.spv";

    /// <summary><c>DOTLLM_VK_Q4_GEMM_SMALL=0</c> keeps every shape on the 128x128 tile (A/B).</summary>
    public const string SmallEnvVar = "DOTLLM_VK_Q4_GEMM_SMALL";

    // Below this many 128x128 tiles (~1.6 per CU on a 40-CU part) the 64x64 twin runs: 4x the workgroups, no stored partials.
    private const int SmallGridMaxTiles = 64;
    private const int SmallTile = 64;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private VulkanModule? _smallModule;
    private ComputePipeline? _smallPipeline;
    private nint _smallPool;
    private DescriptorSetCache? _smallCache;
    private bool _disposed;

    /// <summary>The legacy 4-bit format this instance reads: <see cref="QuantizationType.Q4_0"/> (18 B blocks) or
    /// <see cref="QuantizationType.Q4_1"/> (20 B blocks, issue #1171: same kernel shape, value <c>d * nibble + m</c>).</summary>
    public QuantizationType Quant { get; private init; } = QuantizationType.Q4_0;

    private int BlockBytes => Quant == QuantizationType.Q4_1 ? QuantFormat.Q4_1BlockBytes : QuantFormat.Q4_0BlockBytes;

    private static string ShaderFor(string q4_0Name, QuantizationType quant) => quant switch
    {
        QuantizationType.Q4_0 => q4_0Name,
        QuantizationType.Q4_1 => q4_0Name.Replace("q4_0", "q4_1", StringComparison.Ordinal),
        _ => throw new ArgumentOutOfRangeException(nameof(quant), quant, "Only Q4_0 and Q4_1 are served by this kernel."),
    };

    private MatMulQ4_0GemmCoopmatKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
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
    /// <param name="device">Vulkan device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <param name="quant"><see cref="QuantizationType.Q4_0"/> (default) or <see cref="QuantizationType.Q4_1"/>.</param>
    /// <returns>True when <see cref="Create"/> will succeed and the kernel should be used.</returns>
    public static bool IsSupportedOn(VulkanDevice device, string spvDir, QuantizationType quant = QuantizationType.Q4_0)
        => Environment.GetEnvironmentVariable(LegacyEnvVar) != "1"
           && device.HasCooperativeMatrix
           && device.SubgroupSize == 64
           && File.Exists(Path.Combine(spvDir, ShaderFor(SpvFileName, quant)));

    /// <summary>Loads the SPIR-V and creates the pipeline. Check <see cref="IsSupportedOn"/> first.</summary>
    /// <param name="device">Vulkan device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <param name="quant"><see cref="QuantizationType.Q4_0"/> (default) or <see cref="QuantizationType.Q4_1"/>.</param>
    /// <returns>The kernel.</returns>
    public static MatMulQ4_0GemmCoopmatKernel Create(VulkanDevice device, string spvDir, QuantizationType quant = QuantizationType.Q4_0)
    {
        string path = Path.Combine(spvDir, ShaderFor(SpvFileName, quant));
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
        var kernel = new MatMulQ4_0GemmCoopmatKernel(device, module, pipeline, pool) { Quant = quant };
        string smallPath = Path.Combine(spvDir, ShaderFor(SmallSpvFileName, quant));
        if (Environment.GetEnvironmentVariable(SmallEnvVar) != "0" && File.Exists(smallPath))
        {
            var sm = VulkanModule.LoadFromFile(device, smallPath);
            try
            {
                Span<VkDescriptorBinding> b = stackalloc VkDescriptorBinding[3];
                for (int i = 0; i < 3; i++) b[i] = new VkDescriptorBinding((uint)i);
                kernel._smallPipeline = sm.CreateComputePipeline("main", b, PushConstantBytes, requiredSubgroupSize: 0);
                kernel._smallModule = sm;
                kernel._smallPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
                kernel._smallCache = new DescriptorSetCache(device, kernel._smallPool, kernel._smallPipeline, buffersPerSet: 3);
            }
            catch (VulkanException)
            {
                sm.Dispose();
            }
        }
        return kernel;
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache()
    {
        _descriptorCache.Reset();
        _smallCache?.Reset();
    }

    /// <summary>Whether <see cref="Record"/> takes the 64x64 twin for this shape.</summary>
    /// <param name="m">Output dimension.</param>
    /// <param name="n">Batch size.</param>
    /// <returns>True for small grids when the twin loaded.</returns>
    public bool UsesSmallTile(int m, int n)
        => _smallPipeline is not null && (long)((m + Tile - 1) / Tile) * ((n + Tile - 1) / Tile) < SmallGridMaxTiles;

    /// <summary>Dispatches synchronously (one-shot submit + fence wait); production uses <see cref="Record"/>.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ4_0, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ4_0, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="weightsQ4_0">Raw Q4_0 blob of <c>M * (K / 32) * 18</c> bytes, rows contiguous.</param>
    /// <param name="inputB">FP32 input <c>[N, K]</c>.</param>
    /// <param name="outputC">FP32 output <c>[N, M]</c>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Contraction dimension (multiple of 32).</param>
    /// <param name="n">Batch size.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ4_0, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % Q4_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q4_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q4_0GroupSize;
        long rowBytes = (long)blocksPerRow * BlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        if (weightsQ4_0.Size < (long)m * rowBytes)
            throw new ArgumentException("Weights buffer too small.", nameof(weightsQ4_0));
        if (inputB.Size < (long)n * k * sizeof(float)) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < (long)n * m * sizeof(float)) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        bool small = UsesSmallTile(m, n);
        var pipeline = small ? _smallPipeline! : _pipeline;
        int tile = small ? SmallTile : Tile;
        Span<nint> buffers = stackalloc nint[3] { weightsQ4_0.Handle, inputB.Handle, outputC.Handle };
        nint descriptorSet = (small ? _smallCache! : _descriptorCache).GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5] { (uint)m, (uint)k, (uint)n, (uint)blocksPerRow, (uint)rowUints };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + tile - 1) / tile), (uint)((n + tile - 1) / tile), 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        if (_smallPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _smallPool, 0);
        _smallPipeline?.Dispose();
        _smallModule?.Dispose();
        _pipeline.Dispose();
        _module.Dispose();
    }
}
