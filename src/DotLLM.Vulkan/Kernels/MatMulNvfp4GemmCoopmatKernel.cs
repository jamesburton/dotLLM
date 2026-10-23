using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Native NVFP4 batched GEMM on the shared 128x128, BK=32, four-wave64-subgroup cooperative-matrix tile (issue #1064):
/// <c>C[N, M] = scale · (B[N, K] @ W_nvfp4[M, K]ᵀ)</c>. Each workgroup decodes its NVFP4 blocks (E2M1 codes × UE4M3 sub-block
/// scales) straight into the F16 shared-memory operand, so unlike #1060's prefill path there is no whole-matrix F16 staging buffer.
/// </summary>
/// <remarks>
/// <para><b>Numerics.</b> The staged operand is the unscaled NVFP4 value, exact in F16; accumulation is F32; the per-tensor
/// <c>X.scale</c> multiplies the F32 accumulator once in the epilogue and the output is F32. See
/// <c>native/vulkan/shaders/matmul_nvfp4_gemm_coopmat_128x128x4.comp</c>.</para>
/// <para><b>wave64 ONLY, and that is a correctness gate</b> (as for every shader on the blocked template, see
/// <see cref="MatMulQ4_0GemmCoopmatKernel"/>). Check <see cref="IsSupportedOn"/> before <see cref="Create"/>.</para>
/// </remarks>
public sealed class MatMulNvfp4GemmCoopmatKernel : IDisposable
{
    /// <summary>SPIR-V file name of the shader.</summary>
    public const string SpvFileName = "matmul_nvfp4_gemm_coopmat_128x128x4.spv";

    /// <summary>Environment variable that disables this kernel so callers fall back to the F16 or scalar NVFP4 GEMM (A/B).</summary>
    public const string LegacyEnvVar = "DOTLLM_VK_NVFP4_GEMM_LEGACY";

    private const int Tile = 128;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, scale

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    /// <summary>Dispatches recorded by every instance, read by tests to prove the native path ran.</summary>
    internal static long Dispatches;

    private MatMulNvfp4GemmCoopmatKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
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

    /// <summary>Creates the kernel when <see cref="IsSupportedOn"/> holds, otherwise returns null.</summary>
    /// <param name="device">The Vulkan device.</param>
    /// <param name="spvDir">Directory holding the compiled shaders.</param>
    /// <returns>The kernel, or null on an unsupported device.</returns>
    public static MatMulNvfp4GemmCoopmatKernel? TryCreate(VulkanDevice device, string spvDir)
        => IsSupportedOn(device, spvDir) ? Create(device, spvDir) : null;

    /// <summary>Loads the SPIR-V and creates the pipeline. Check <see cref="IsSupportedOn"/> first.</summary>
    /// <param name="device">The Vulkan device.</param>
    /// <param name="spvDir">Directory holding the compiled shaders.</param>
    /// <returns>The kernel.</returns>
    /// <exception cref="FileNotFoundException">The SPIR-V is missing.</exception>
    public static MatMulNvfp4GemmCoopmatKernel Create(VulkanDevice device, string spvDir)
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
                entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes, requiredSubgroupSize: 0);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new MatMulNvfp4GemmCoopmatKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches synchronously (one-shot submit + fence wait); production uses <see cref="Record"/>.</summary>
    /// <param name="weights">Packed NVFP4 weights, <paramref name="m"/> rows of <c>k / 64</c> blocks.</param>
    /// <param name="input">FP32 input <c>[n, k]</c>.</param>
    /// <param name="output">FP32 output <c>[n, m]</c>.</param>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Contraction length; a multiple of 64.</param>
    /// <param name="n">Token rows.</param>
    /// <param name="scale">Per-tensor output scale (1 when the GGUF has no <c>X.scale</c>).</param>
    public void Launch(VulkanDevice.Buffer weights, VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int m, int k, int n, float scale)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weights, input, output, m, k, n, scale);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="weights">Packed NVFP4 weights, <paramref name="m"/> rows of <c>k / 64</c> blocks.</param>
    /// <param name="input">FP32 input <c>[n, k]</c>.</param>
    /// <param name="output">FP32 output <c>[n, m]</c>.</param>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Contraction length; a multiple of 64.</param>
    /// <param name="n">Token rows.</param>
    /// <param name="scale">Per-tensor output scale (1 when the GGUF has no <c>X.scale</c>); applied once, to the F32 accumulator.</param>
    /// <exception cref="ArgumentException">A dimension or buffer size is invalid.</exception>
    public unsafe void Record(nint cmdBuf,
        VulkanDevice.Buffer weights, VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int m, int k, int n, float scale)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(m);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);
        if (k % QuantFormat.Nvfp4GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {QuantFormat.Nvfp4GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / QuantFormat.Nvfp4GroupSize;
        long weightsMin = (long)m * blocksPerRow * QuantFormat.Nvfp4BlockBytes;
        if (weights.Size < weightsMin)
            throw new ArgumentException($"Weights buffer too small: need >= {weightsMin} bytes, got {weights.Size}.", nameof(weights));
        if (input.Size < (long)n * k * sizeof(float)) throw new ArgumentException("Input buffer too small.", nameof(input));
        if (output.Size < (long)n * m * sizeof(float)) throw new ArgumentException("Output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[3] { weights.Handle, input.Handle, output.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5]
        {
            (uint)m, (uint)k, (uint)n, (uint)blocksPerRow, BitConverter.SingleToUInt32Bits(scale),
        };
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
    }
}
