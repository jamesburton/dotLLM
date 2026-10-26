using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// PTQ1_0 (PrismML Bonsai dense-trit ternary, GGUF type 143) decode-path GEMV:
/// <c>y[m] = Σ_g scale(m,g) · Σ_{e∈g} trit(m,g,e) · x[128·g + e]</c>.
/// </summary>
/// <remarks>
/// Weight layout mirrors the CPU oracle <c>DotLLM.Cpu.Kernels.MatMul.UnpackPTQ1_0Row</c> /
/// <c>Dequantize.DequantizePTQ1_0</c>, which is itself gated bit-exact against the PrismML
/// fork's <c>dequantize_row_ptq1_0</c> (issue #436): each 128 contiguous columns of a row form
/// one 28-byte group — <c>qs[24]</c> (five base-3 trits per byte), <c>qh[2]</c> (four trits per
/// byte), then the little-endian fp16 group scale. <b>The scale is the LAST field</b>, unlike
/// PQ2_0 where it leads the block; that one offset is the difference between a correct decode
/// and one that reads every group's scale out of the next group's payload. Element order within
/// a group is trit-major over byte chunks, not positional — see the shader header, which quotes
/// the fork's own <c>ptq1_0.glsl</c> mapping.
/// Row stride is <c>(K/128)·28</c> bytes (1.75 bpw against PQ2_0's 2.125); there is no
/// per-tensor tail scale. Activation <c>x</c> and output <c>y</c> are FP32. One workgroup per
/// output row, 128 threads, shared-memory tree reduce. Correctness-first baseline — GEMV/decode
/// only, no coopmat/dp4a MMQ path, and not yet performance-characterised.
/// </remarks>
public sealed class MatMulPTQ1_0GemvF32Kernel : IDisposable
{
    /// <summary>PTQ1_0 group: qs[24] + qh[2] + 2 bytes fp16 scale — the scale comes last.</summary>
    public const int PTQ1_0GroupBytes = QuantFormat.PTQ1_0BlockBytes;

    /// <summary>Elements per PTQ1_0 group.</summary>
    public const int PTQ1_0GroupSize = QuantFormat.TernaryGroupSize;

    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulPTQ1_0GemvF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_ptq1_0_f32_gemv.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulPTQ1_0GemvF32Kernel Create(VulkanDevice device, string spvDir)
        => Create(device, spvDir, "matmul_ptq1_0_f32_gemv.spv");

    /// <summary>
    /// Loads the named SPIR-V from <paramref name="spvDir"/> and creates the pipeline. The
    /// <paramref name="spvFileName"/> overload exists so a benchmark can A/B successive kernel
    /// variants side by side in one process.
    /// </summary>
    public static MatMulPTQ1_0GemvF32Kernel Create(VulkanDevice device, string spvDir, string spvFileName)
    {
        string path = Path.Combine(spvDir, spvFileName);
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
        return new MatMulPTQ1_0GemvF32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the GEMV synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsPTQ1_0, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsPTQ1_0, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the PTQ1_0 GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsPTQ1_0, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % PTQ1_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {PTQ1_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / PTQ1_0GroupSize;
        long rowBytes = (long)blocksPerRow * PTQ1_0GroupBytes;
        // 28 is a multiple of 4, so every group starts uint-aligned — but the fp16 scale sits at
        // +26 within the group, so it always straddles a uint boundary. rowUints rounds up so a
        // straddling two-byte read at the very end of the row stays in bounds.
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsPTQ1_0.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes (m·(k/128)·28), got {weightsPTQ1_0.Size}.",
                nameof(weightsPTQ1_0));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsPTQ1_0.Handle, x.Handle, y.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4]
        {
            (uint)m,
            (uint)k,
            (uint)blocksPerRow,
            (uint)rowUints,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, 1, 1);
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
