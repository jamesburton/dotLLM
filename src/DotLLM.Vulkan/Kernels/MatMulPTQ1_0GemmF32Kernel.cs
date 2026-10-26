using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// PTQ1_0 (PrismML Bonsai dense-trit ternary, GGUF type 143) prefill-path GEMM:
/// <c>C[t, m] = Σ_g scale(m,g) · Σ_{e∈g} trit(m,g,e) · B[t, 128·g + e]</c>.
/// <c>B</c> is <c>[N, K]</c> token-major, <c>C</c> is <c>[N, M]</c> token-major.
/// </summary>
/// <remarks>
/// <b>Correctness-first, not tiled.</b> The shader is the GEMV body with a second dispatch
/// dimension: one workgroup per (row, token) pair, so it re-reads the whole weight row once per
/// token and moves N× the weight traffic a blocked kernel would. PQ2_0's prefill path has a
/// register-block variant and two coopmat variants selected per device; PTQ1_0 has none of that.
///
/// <para>This exists so that the keep-packed upload arm has a matching dispatch arm at <i>every</i>
/// sequence length. An upload arm without a dispatch arm is not a missing optimization, it is a
/// throw at inference time — and because both hybrid models share the weights class, it is a throw
/// for the model nobody was testing. That is precisely how PQ2_0 broke. Prefill throughput through
/// this kernel will be poor; that is a known, deliberate state, and a tiled PTQ1_0 GEMM sharing the
/// same decode is the obvious follow-on.</para>
///
/// <para>Weight layout is documented on <see cref="MatMulPTQ1_0GemvF32Kernel"/> — 28-byte groups,
/// <c>qs[24] + qh[2] + fp16 scale</c>, scale LAST.</para>
/// </remarks>
public sealed class MatMulPTQ1_0GemmF32Kernel : IDisposable
{
    /// <summary>PTQ1_0 group: qs[24] + qh[2] + 2 bytes fp16 scale — the scale comes last.</summary>
    public const int PTQ1_0GroupBytes = QuantFormat.PTQ1_0BlockBytes;

    /// <summary>Elements per PTQ1_0 group.</summary>
    public const int PTQ1_0GroupSize = QuantFormat.TernaryGroupSize;

    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulPTQ1_0GemmF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_ptq1_0_f32_gemm.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulPTQ1_0GemmF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_ptq1_0_f32_gemm.spv");
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
        return new MatMulPTQ1_0GemmF32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the GEMM synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsPTQ1_0, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsPTQ1_0, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the PTQ1_0 GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsPTQ1_0, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % PTQ1_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {PTQ1_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / PTQ1_0GroupSize;
        long rowBytes = (long)blocksPerRow * PTQ1_0GroupBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsPTQ1_0.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes (m·(k/128)·28), got {weightsPTQ1_0.Size}.",
                nameof(weightsPTQ1_0));
        if (inputB.Size < (long)n * k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < (long)n * m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[3] { weightsPTQ1_0.Handle, inputB.Handle, outputC.Handle };
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

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, (uint)n, 1);
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
