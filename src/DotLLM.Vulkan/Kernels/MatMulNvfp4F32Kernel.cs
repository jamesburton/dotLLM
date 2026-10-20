using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// NVFP4 matmul (issue #1038): <c>C[N, M] = scale · (B[N, K] @ W_nvfp4[M, K]ᵀ)</c>, with a GEMV for <c>N == 1</c> and the
/// 16x16-tiled GEMM (<c>matmul_nvfp4_f32_gemm.spv</c>) otherwise. For <c>N == 1</c>, in order of preference (#1066):
/// <list type="number">
/// <item>the integer-dot MMVQ (<c>matmul_nvfp4_mmvq.spv</c>): the activation row is quantised to Q8_1 into a scratch owned here,
/// then dotted as int8 against the E2M1 codes; used for weights of at least <see cref="DefaultMmvqMinWeightBytes"/> (by default) on devices with
/// integer dot product;</item>
/// <item>the coalesced multi-row FP32 GEMV (<c>matmul_nvfp4_gemv_coalesced.spv</c>) on devices with subgroup arithmetic;</item>
/// <item>the one-block-per-lane FP32 GEMV (<c>matmul_nvfp4_f32_gemv.spv</c>, #1038).</item>
/// </list>
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

    /// <summary>SPIR-V of the coalesced multi-row GEMV (#1066).</summary>
    public const string CoalescedGemvSpvFileName = "matmul_nvfp4_gemv_coalesced.spv";

    /// <summary>Weight rows per workgroup of <see cref="CoalescedGemvSpvFileName"/> (its <c>ROWS</c>).</summary>
    public const int CoalescedGemvRows = 1;

    /// <summary>Environment variable that forces the #1038 one-block-per-lane GEMV (A/B); it also disables the MMVQ.</summary>
    public const string LegacyGemvEnvVar = "DOTLLM_VK_NVFP4_GEMV_LEGACY";

    /// <summary>SPIR-V of the integer-dot MMVQ (#1066).</summary>
    public const string MmvqSpvFileName = "matmul_nvfp4_mmvq.spv";

    /// <summary>Weight rows per workgroup of <see cref="MmvqSpvFileName"/> (its <c>ROWS</c>).</summary>
    public const int MmvqRows = 1;

    /// <summary>Environment variable that disables the MMVQ (A/B against the FP32 GEMVs).</summary>
    public const string DisableMmvqEnvVar = "DOTLLM_VK_NVFP4_MMVQ_DISABLE";

    /// <summary>Environment variable overriding <see cref="DefaultMmvqMinWeightBytes"/>.</summary>
    public const string MmvqMinBytesEnvVar = "DOTLLM_VK_NVFP4_MMVQ_MIN_BYTES";

    /// <summary>
    /// Smallest weight matrix routed to the MMVQ: below it the extra quantise dispatch and barrier cost more than the int8 dot saves
    /// (CUDA's #1061 gate is the same 4 MiB). Bolt's NVFP4 projections are 19 and 43 MB.
    /// </summary>
    public const long DefaultMmvqMinWeightBytes = 4L << 20;

    /// <summary>Largest K the MMVQ's activation scratch holds; a wider projection falls back to the FP32 GEMV.</summary>
    public const int MmvqMaxK = 1 << 16;

    private readonly VulkanDevice _device;
    private readonly Stage _gemv;
    private readonly Stage _gemm;
    private readonly Stage? _gemvCoalesced;
    private readonly int _gemvCoalescedRows;
    private readonly Mmvq? _mmvq;
    private bool _disposed;

    /// <summary>GEMV dispatches that took the coalesced kernel, read by tests to prove which GEMV ran.</summary>
    internal static long CoalescedGemvDispatches;

    /// <summary>GEMV dispatches that took the integer-dot MMVQ, read by tests to prove which GEMV ran.</summary>
    internal static long MmvqDispatches;

    private MatMulNvfp4F32Kernel(VulkanDevice device, Stage gemv, Stage gemm, Stage? gemvCoalesced, int gemvCoalescedRows, Mmvq? mmvq)
    {
        _device = device;
        _gemv = gemv;
        _gemm = gemm;
        _gemvCoalesced = gemvCoalesced;
        _gemvCoalescedRows = gemvCoalescedRows;
        _mmvq = mmvq;
    }

    /// <summary>Whether <c>N == 1</c> dispatches (below the MMVQ gate) take the coalesced multi-row GEMV (#1066).</summary>
    public bool UsesCoalescedGemv => _gemvCoalesced is not null;

    /// <summary>Whether the integer-dot MMVQ is available for large enough weights (#1066).</summary>
    public bool HasMmvq => _mmvq is not null;

    /// <summary>Loads both NVFP4 SPIR-V modules from <paramref name="spvDir"/> and creates their pipelines.</summary>
    /// <param name="device">The Vulkan device.</param>
    /// <param name="spvDir">Directory holding the compiled shaders.</param>
    /// <returns>The kernel.</returns>
    public static MatMulNvfp4F32Kernel Create(VulkanDevice device, string spvDir)
    {
        bool legacy = Environment.GetEnvironmentVariable(LegacyGemvEnvVar) == "1";
        bool coalesced = device.HasSubgroupArithmetic && !legacy
            && File.Exists(Path.Combine(spvDir, CoalescedGemvSpvFileName));
        bool mmvq = coalesced && device.HasIntegerDotProduct
            && Environment.GetEnvironmentVariable(DisableMmvqEnvVar) != "1"
            && File.Exists(Path.Combine(spvDir, MmvqSpvFileName));
        long minBytes = long.TryParse(Environment.GetEnvironmentVariable(MmvqMinBytesEnvVar), out long v) && v >= 0
            ? v : DefaultMmvqMinWeightBytes;
        return Create(device, spvDir, coalesced ? CoalescedGemvSpvFileName : null, CoalescedGemvRows,
            mmvq ? MmvqSpvFileName : null, MmvqRows, minBytes);
    }

    /// <summary>
    /// Creates the kernel with an explicit coalesced-GEMV module, so a benchmark can A/B builds of the shader with a different
    /// <c>ROWS</c>; <paramref name="coalescedGemvSpv"/> null keeps the #1038 GEMV.
    /// </summary>
    /// <param name="device">The Vulkan device.</param>
    /// <param name="spvDir">Directory holding the compiled shaders.</param>
    /// <param name="coalescedGemvSpv">File name (in <paramref name="spvDir"/>) of a <c>matmul_nvfp4_gemv_coalesced</c> build, or null.</param>
    /// <param name="coalescedGemvRows">The <c>ROWS</c> that build was compiled with.</param>
    /// <param name="mmvqSpv">File name (in <paramref name="spvDir"/>) of a <c>matmul_nvfp4_mmvq</c> build, or null for no MMVQ.</param>
    /// <param name="mmvqRows">The <c>ROWS</c> that MMVQ build was compiled with.</param>
    /// <param name="mmvqMinWeightBytes">Smallest weight matrix routed to the MMVQ.</param>
    /// <returns>The kernel.</returns>
    internal static MatMulNvfp4F32Kernel Create(VulkanDevice device, string spvDir, string? coalescedGemvSpv, int coalescedGemvRows,
        string? mmvqSpv = null, int mmvqRows = MmvqRows, long mmvqMinWeightBytes = DefaultMmvqMinWeightBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(coalescedGemvRows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(mmvqRows);
        Stage? gemv = null, gemm = null, gemvCoalesced = null;
        Mmvq? mmvq = null;
        try
        {
            gemv = Stage.Load(device, spvDir, "matmul_nvfp4_f32_gemv.spv", GemvPushBytes);
            gemm = Stage.Load(device, spvDir, "matmul_nvfp4_f32_gemm.spv", GemmPushBytes);
            if (coalescedGemvSpv is not null)
                gemvCoalesced = Stage.Load(device, spvDir, coalescedGemvSpv, GemvPushBytes);
            if (mmvqSpv is not null)
                mmvq = Mmvq.Create(device, spvDir, mmvqSpv, mmvqRows, mmvqMinWeightBytes);
            return new MatMulNvfp4F32Kernel(device, gemv, gemm, gemvCoalesced, coalescedGemvRows, mmvq);
        }
        catch
        {
            mmvq?.Dispose();
            gemvCoalesced?.Dispose();
            gemm?.Dispose();
            gemv?.Dispose();
            throw;
        }
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache()
    {
        _gemv.Cache.Reset();
        _gemm.Cache.Reset();
        _gemvCoalesced?.Cache.Reset();
        _mmvq?.InvalidateDescriptorCache();
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
        if (gemv && _mmvq is not null && _mmvq.Serves(weightsMin, k))
        {
            _mmvq.Record(cmdBuf, weights, input, output, m, k, blocksPerRow, scale);
            Interlocked.Increment(ref MmvqDispatches);
            return;
        }

        Stage stage = gemv ? _gemvCoalesced ?? _gemv : _gemm;
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

        if (gemv && _gemvCoalesced is not null)
        {
            VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + _gemvCoalescedRows - 1) / _gemvCoalescedRows), 1, 1);
            Interlocked.Increment(ref CoalescedGemvDispatches);
        }
        else if (gemv)
            VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, 1, 1);
        else
            VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + TileM - 1) / TileM), (uint)((n + TileN - 1) / TileN), 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _mmvq?.Dispose();
        _gemvCoalesced?.Dispose();
        _gemm.Dispose();
        _gemv.Dispose();
    }

    /// <summary>
    /// The integer-dot decode path (#1066): quantises the activation row to Q8_1 into a fixed-capacity scratch owned here, then runs
    /// <c>matmul_nvfp4_mmvq</c>. Each call starts with a compute-to-compute barrier (the previous MMVQ's read of the scratch against
    /// this quantise's write) and puts one between quantise and MMVQ. The capacity is fixed so no buffer is freed mid-recording.
    /// </summary>
    private sealed class Mmvq : IDisposable
    {
        private readonly QuantizeQ8_1Kernel _quantize;
        private readonly Stage _stage;
        private readonly int _rows;
        private readonly long _minWeightBytes;
        private readonly VulkanDevice.Buffer _xq;
        private readonly VulkanDevice.Buffer _xds;

        private Mmvq(QuantizeQ8_1Kernel quantize, Stage stage, int rows, long minWeightBytes, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds)
        {
            _quantize = quantize;
            _stage = stage;
            _rows = rows;
            _minWeightBytes = minWeightBytes;
            _xq = xq;
            _xds = xds;
        }

        public static Mmvq Create(VulkanDevice device, string spvDir, string spv, int rows, long minWeightBytes)
        {
            var quantize = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
                ?? throw new FileNotFoundException("quantize_q8_1.spv is missing; the NVFP4 MMVQ needs it.");
            Stage? stage = null;
            VulkanDevice.Buffer? xq = null;
            try
            {
                stage = Stage.Load(device, spvDir, spv, GemvPushBytes, buffers: 4);
                xq = device.AllocateDeviceLocal(QuantizeQ8_1Kernel.PackedBytes(MmvqMaxK));
                var xds = device.AllocateDeviceLocal(QuantizeQ8_1Kernel.ScaleBytes(MmvqMaxK));
                return new Mmvq(quantize, stage, rows, minWeightBytes, xq, xds);
            }
            catch
            {
                xq?.Dispose();
                stage?.Dispose();
                quantize.Dispose();
                throw;
            }
        }

        public bool Serves(long weightBytes, int k) => weightBytes >= _minWeightBytes && k <= MmvqMaxK;

        public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer weights, VulkanDevice.Buffer input, VulkanDevice.Buffer output,
            int m, int k, int blocksPerRow, float scale)
        {
            KernelSupport.ComputeToComputeBarrier(cmdBuf);   // previous MMVQ's read of the scratch vs this quantise's write
            _quantize.Record(cmdBuf, input, _xq, _xds, k);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);   // quantise write vs MMVQ read

            Span<nint> buffers = stackalloc nint[4] { weights.Handle, _xq.Handle, _xds.Handle, output.Handle };
            nint descriptorSet = _stage.Cache.GetOrCreate(buffers);
            VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _stage.Pipeline.Pipeline);
            VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _stage.Pipeline.Layout, 0, 1, descriptorSet, 0, 0);
            Span<uint> pc = stackalloc uint[4] { (uint)m, (uint)k, (uint)blocksPerRow, BitConverter.SingleToUInt32Bits(scale) };
            fixed (uint* pcPtr = pc)
            {
                VulkanApi.vkCmdPushConstants(cmdBuf, _stage.Pipeline.Layout, VkShaderStageFlags.Compute, 0, GemvPushBytes, (nint)pcPtr);
            }
            VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + _rows - 1) / _rows), 1, 1);
        }

        public void InvalidateDescriptorCache()
        {
            _stage.Cache.Reset();
            _quantize.InvalidateDescriptorCache();
        }

        public void Dispose()
        {
            _xds.Dispose();
            _xq.Dispose();
            _stage.Dispose();
            _quantize.Dispose();
        }
    }

    /// <summary>One shader module + pipeline + descriptor pool/cache.</summary>
    private sealed class Stage : IDisposable
    {
        private readonly VulkanDevice _device;
        private readonly VulkanModule _module;
        private readonly nint _pool;

        private Stage(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, int buffers)
        {
            _device = device;
            _module = module;
            Pipeline = pipeline;
            _pool = pool;
            Cache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: buffers);
        }

        public ComputePipeline Pipeline { get; }

        public DescriptorSetCache Cache { get; }

        public static Stage Load(VulkanDevice device, string spvDir, string file, int pushBytes, int buffers = 3)
        {
            string path = Path.Combine(spvDir, file);
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

            var module = VulkanModule.LoadFromFile(device, path);
            try
            {
                Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[buffers];
                for (int i = 0; i < buffers; i++) bindings[i] = new VkDescriptorBinding((uint)i);
                var pipeline = module.CreateComputePipeline(entryPoint: "main", bindings: bindings, pushConstantBytes: (uint)pushBytes);
                nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)buffers);
                return new Stage(device, module, pipeline, pool, buffers);
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
