using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// NVFP4 matmul (issue #1038): <c>C[N, M] = scale · (B[N, K] @ W_nvfp4[M, K]ᵀ)</c>, with the GEMV
/// (<c>matmul_nvfp4_f32_gemv.spv</c>) for <c>N == 1</c> and the 16x16-tiled GEMM
/// (<c>matmul_nvfp4_f32_gemm.spv</c>) otherwise.
/// </summary>
/// <remarks>
/// <para>Weights stay packed on device in the GGUF byte layout (36-byte blocks of 64: four UE4M3 sub-block
/// scales then 32 E2M1 nibble bytes; see <c>native/vulkan/shaders/nvfp4_decode.glsl</c>). Activations are
/// FP32, so this matches the CPU <c>Dequantize.DequantizeNvfp4</c> + dense dot to rounding; the CPU's packed
/// path additionally rounds activations to Q8_0, as llama.cpp's CPU backend does.</para>
/// <para>The <c>scale</c> argument of <see cref="Record"/> is the weight's per-tensor <c>X.scale</c>, applied
/// to every output as llama.cpp's <c>build_lora_mm</c> does; pass 1 when the GGUF has none.</para>
/// </remarks>
public sealed class MatMulNvfp4F32Kernel : IDisposable
{
    private const int TileM = 16;
    private const int TileN = 16;
    private const int GemvPushBytes = 4 * sizeof(uint); // M, K, blocksPerRow, scale
    private const int GemmPushBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, scale

    private readonly VulkanDevice _device;
    private readonly Stage _gemv;
    private readonly Stage _gemm;
    private bool _disposed;

    private MatMulNvfp4F32Kernel(VulkanDevice device, Stage gemv, Stage gemm)
    {
        _device = device;
        _gemv = gemv;
        _gemm = gemm;
    }

    /// <summary>Loads both NVFP4 SPIR-V modules from <paramref name="spvDir"/> and creates their pipelines.</summary>
    /// <param name="device">The Vulkan device.</param>
    /// <param name="spvDir">Directory holding the compiled shaders.</param>
    /// <returns>The kernel.</returns>
    public static MatMulNvfp4F32Kernel Create(VulkanDevice device, string spvDir)
    {
        var gemv = Stage.Load(device, spvDir, "matmul_nvfp4_f32_gemv.spv", GemvPushBytes);
        try
        {
            var gemm = Stage.Load(device, spvDir, "matmul_nvfp4_f32_gemm.spv", GemmPushBytes);
            return new MatMulNvfp4F32Kernel(device, gemv, gemm);
        }
        catch
        {
            gemv.Dispose();
            throw;
        }
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache()
    {
        _gemv.Cache.Reset();
        _gemm.Cache.Reset();
    }

    /// <summary>Dispatches synchronously.</summary>
    /// <param name="weights">Packed NVFP4 weights, <paramref name="m"/> rows of <c>k / 64</c> blocks.</param>
    /// <param name="input">FP32 input, <paramref name="n"/> rows of <paramref name="k"/>.</param>
    /// <param name="output">FP32 output, <paramref name="n"/> rows of <paramref name="m"/>.</param>
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

    /// <summary>Records the NVFP4 matmul into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Command buffer in the recording state.</param>
    /// <param name="weights">Packed NVFP4 weights, <paramref name="m"/> rows of <c>k / 64</c> blocks.</param>
    /// <param name="input">FP32 input, <paramref name="n"/> rows of <paramref name="k"/>.</param>
    /// <param name="output">FP32 output, <paramref name="n"/> rows of <paramref name="m"/>.</param>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Contraction length; a multiple of 64.</param>
    /// <param name="n">Token rows.</param>
    /// <param name="scale">Per-tensor output scale (1 when the GGUF has no <c>X.scale</c>).</param>
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
        if (input.Size < (long)n * k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(input));
        if (output.Size < (long)n * m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(output));

        bool gemv = n == 1;
        Stage stage = gemv ? _gemv : _gemm;
        Span<nint> buffers = stackalloc nint[3] { weights.Handle, input.Handle, output.Handle };
        nint descriptorSet = stage.Cache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, stage.Pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, stage.Pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        uint scaleBits = BitConverter.SingleToUInt32Bits(scale);
        Span<uint> pc = gemv
            ? stackalloc uint[4] { (uint)m, (uint)k, (uint)blocksPerRow, scaleBits }
            : stackalloc uint[5] { (uint)m, (uint)k, (uint)n, (uint)blocksPerRow, scaleBits };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(cmdBuf, stage.Pipeline.Layout, VkShaderStageFlags.Compute,
                0, (uint)(pc.Length * sizeof(uint)), (nint)pcPtr);
        }

        if (gemv)
            VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, 1, 1);
        else
            VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + TileM - 1) / TileM), (uint)((n + TileN - 1) / TileN), 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _gemm.Dispose();
        _gemv.Dispose();
    }

    /// <summary>One shader module + pipeline + descriptor pool/cache.</summary>
    private sealed class Stage : IDisposable
    {
        private readonly VulkanDevice _device;
        private readonly VulkanModule _module;
        private readonly nint _pool;

        private Stage(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
        {
            _device = device;
            _module = module;
            Pipeline = pipeline;
            _pool = pool;
            Cache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
        }

        public ComputePipeline Pipeline { get; }

        public DescriptorSetCache Cache { get; }

        public static Stage Load(VulkanDevice device, string spvDir, string file, int pushBytes)
        {
            string path = Path.Combine(spvDir, file);
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

            var module = VulkanModule.LoadFromFile(device, path);
            try
            {
                Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
                bindings[0] = new VkDescriptorBinding(0);
                bindings[1] = new VkDescriptorBinding(1);
                bindings[2] = new VkDescriptorBinding(2);
                var pipeline = module.CreateComputePipeline(entryPoint: "main", bindings: bindings, pushConstantBytes: (uint)pushBytes);
                nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
                return new Stage(device, module, pipeline, pool);
            }
            catch
            {
                module.Dispose();
                throw;
            }
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
