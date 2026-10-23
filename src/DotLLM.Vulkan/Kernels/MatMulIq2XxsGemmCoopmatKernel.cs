using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Native IQ2_XXS batched GEMM on the shared 128x128, BK=32, four-wave64-subgroup cooperative-matrix tile (issue #1069):
/// <c>C[N, M] = B[N, K] @ W_iq2xxs[M, K]ᵀ</c>. Each workgroup decodes its IQ2_XXS sub-blocks (grid + ksigns codebooks) straight into
/// the F16 shared-memory operand, so unlike #1060's prefill path there is no whole-matrix F16 staging buffer.
/// </summary>
/// <remarks>
/// <para><b>Numerics.</b> The staged operand is computed exactly as <c>iq2_xxs_dequant_f16.comp</c> computes it, so the result
/// equals #1060's F16 path up to F32 accumulation order. See <c>native/vulkan/shaders/matmul_iq2_xxs_gemm_coopmat_128x128x4.comp</c>.</para>
/// <para><b>wave64 ONLY, and that is a correctness gate</b> (as for every shader on the blocked template, see
/// <see cref="MatMulQ4_0GemmCoopmatKernel"/>). Check <see cref="IsSupportedOn"/> before <see cref="Create"/>.</para>
/// </remarks>
public sealed class MatMulIq2XxsGemmCoopmatKernel : IDisposable
{
    /// <summary>SPIR-V file name of the shader.</summary>
    public const string SpvFileName = "matmul_iq2_xxs_gemm_coopmat_128x128x4.spv";

    /// <summary>Environment variable that disables this kernel so callers fall back to the F16 or scalar IQ2_XXS GEMM (A/B).</summary>
    public const string LegacyEnvVar = "DOTLLM_VK_IQ2_XXS_GEMM_LEGACY";

    /// <summary>IQ2_XXS super-block bytes: fp16 d + 8 x (4 grid indices, 4 sign indices | scale).</summary>
    public const int BlockBytes = 66;

    /// <summary>Elements per IQ2_XXS super-block.</summary>
    public const int GroupSize = 256;

    private const int Tile = 128;
    private const int Bindings = 5;                         // W, B, C, grid, ksigns
    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, N, blocksPerRow

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly Iq2Codebooks _codebooks;
    private readonly bool _ownsCodebooks;
    private bool _disposed;

    /// <summary>Dispatches recorded by every instance, read by tests to prove the native path ran.</summary>
    internal static long Dispatches;

    private MatMulIq2XxsGemmCoopmatKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool,
        Iq2Codebooks codebooks, bool ownsCodebooks)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: Bindings);
        _codebooks = codebooks;
        _ownsCodebooks = ownsCodebooks;
    }

    /// <summary>
    /// Whether the kernel can run on <paramref name="device"/>: cooperative matrix present, native wave64, the SPIR-V compiled,
    /// and not disabled by <see cref="LegacyEnvVar"/>.
    /// </summary>
    /// <param name="device">The Vulkan device.</param>
    /// <param name="spvDir">Directory holding the compiled shaders.</param>
    /// <returns>True when <see cref="Create"/> may be called.</returns>
    public static bool IsSupportedOn(VulkanDevice device, string spvDir)
        => Environment.GetEnvironmentVariable(LegacyEnvVar) != "1"
           && device.HasCooperativeMatrix
           && device.SubgroupSize == 64
           && File.Exists(Path.Combine(spvDir, SpvFileName));

    /// <summary>Creates the kernel on shared codebooks when <see cref="IsSupportedOn"/> holds, otherwise returns null.</summary>
    /// <param name="device">The Vulkan device.</param>
    /// <param name="spvDir">Directory holding the compiled shaders.</param>
    /// <param name="codebooks">The IQ2 grid / ksigns tables; owned by the caller and outliving the kernel.</param>
    /// <returns>The kernel, or null on an unsupported device.</returns>
    internal static MatMulIq2XxsGemmCoopmatKernel? TryCreate(VulkanDevice device, string spvDir, Iq2Codebooks codebooks)
        => IsSupportedOn(device, spvDir) ? CreateCore(device, spvDir, codebooks, ownsCodebooks: false) : null;

    /// <summary>Loads the SPIR-V, uploads its own codebooks and creates the pipeline. Check <see cref="IsSupportedOn"/> first.</summary>
    /// <param name="device">The Vulkan device.</param>
    /// <param name="spvDir">Directory holding the compiled shaders.</param>
    /// <returns>The kernel.</returns>
    /// <exception cref="FileNotFoundException">The SPIR-V is missing.</exception>
    public static MatMulIq2XxsGemmCoopmatKernel Create(VulkanDevice device, string spvDir)
    {
        var codebooks = Iq2Codebooks.Create(device);
        try
        {
            return CreateCore(device, spvDir, codebooks, ownsCodebooks: true);
        }
        catch
        {
            codebooks.Dispose();
            throw;
        }
    }

    private static MatMulIq2XxsGemmCoopmatKernel CreateCore(VulkanDevice device, string spvDir, Iq2Codebooks codebooks, bool ownsCodebooks)
    {
        string path = Path.Combine(spvDir, SpvFileName);
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[Bindings];
            for (int i = 0; i < Bindings; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes, requiredSubgroupSize: 0);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: Bindings);
        return new MatMulIq2XxsGemmCoopmatKernel(device, module, pipeline, pool, codebooks, ownsCodebooks);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches synchronously (one-shot submit + fence wait); production uses <see cref="Record"/>.</summary>
    /// <param name="weights">IQ2_XXS weights, <paramref name="m"/> rows of <c>k / 256</c> super-blocks.</param>
    /// <param name="input">FP32 input <c>[n, k]</c>.</param>
    /// <param name="output">FP32 output <c>[n, m]</c>.</param>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Contraction length; a multiple of 256.</param>
    /// <param name="n">Token rows.</param>
    public void Launch(VulkanDevice.Buffer weights, VulkanDevice.Buffer input, VulkanDevice.Buffer output, int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weights, input, output, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="weights">IQ2_XXS weights, <paramref name="m"/> rows of <c>k / 256</c> super-blocks.</param>
    /// <param name="input">FP32 input <c>[n, k]</c>.</param>
    /// <param name="output">FP32 output <c>[n, m]</c>.</param>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Contraction length; a multiple of 256.</param>
    /// <param name="n">Token rows.</param>
    /// <exception cref="ArgumentException">A dimension or buffer size is invalid.</exception>
    public unsafe void Record(nint cmdBuf,
        VulkanDevice.Buffer weights, VulkanDevice.Buffer input, VulkanDevice.Buffer output, int m, int k, int n)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(m);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);
        if (k % GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / GroupSize;
        long weightsMin = (long)m * blocksPerRow * BlockBytes;
        if (weights.Size < weightsMin)
            throw new ArgumentException($"Weights buffer too small: need >= {weightsMin} bytes, got {weights.Size}.", nameof(weights));
        if (input.Size < (long)n * k * sizeof(float)) throw new ArgumentException("Input buffer too small.", nameof(input));
        if (output.Size < (long)n * m * sizeof(float)) throw new ArgumentException("Output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[Bindings]
        {
            weights.Handle, input.Handle, output.Handle, _codebooks.Iq2XxsGrid.Handle, _codebooks.Ksigns.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4] { (uint)m, (uint)k, (uint)n, (uint)blocksPerRow };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + Tile - 1) / Tile), (uint)((n + Tile - 1) / Tile), 1);
        Interlocked.Increment(ref Dispatches);
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
