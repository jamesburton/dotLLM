using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ1_S decode-path GEMV: <c>y[M] = W_iq1s[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Weight layout mirrors the CPU oracle <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeIQ1_S</c>
/// and llama.cpp's <c>block_iq1_s</c>: each 256 contiguous columns of a row form one
/// 50-byte super-block — 2 bytes fp16 <c>d</c>, 32 bytes <c>qs</c> (low 8 bits of grid
/// index per 8-element group), and 8 uint16 <c>qh</c> entries (per 32-element sub-block:
/// 3-bit scale, sign-of-delta bit, and four 3-bit grid-index high parts). The 11-bit grid
/// index selects from the 2048-entry signed-int8 codebook (each entry packs 8 ternary
/// {-1, 0, +1} values into a uint64).
/// </para>
/// <para>
/// Activation <c>x</c> is FP32; output <c>y</c> is FP32. One workgroup per output row,
/// 128 threads, shared-memory tree reduce — same dispatch shape as the IQ4_XS / K-quant
/// GEMVs.
/// </para>
/// </remarks>
public sealed class MatMulIq1SGemvF32Kernel : IDisposable
{
    /// <summary>IQ1_S super-block: 2 + 32 + 16 = 50 bytes.</summary>
    public const int IQ1_SBlockBytes = 50;

    /// <summary>Elements per IQ1_S super-block.</summary>
    public const int IQ1_SGroupSize = 256;

    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulIq1SGemvF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_iq1_s_f32_gemv.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulIq1SGemvF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_iq1_s_f32_gemv.spv");
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
        return new MatMulIq1SGemvF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the GEMV synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsIq1S, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsIq1S, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the IQ1_S GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsIq1S, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % IQ1_SGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {IQ1_SGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / IQ1_SGroupSize;
        long rowBytes = (long)blocksPerRow * IQ1_SBlockBytes;
        // 50 is not divisible by 4 — every super-block straddles uint boundaries past
        // the first. rowUints rounds up to cover safe straddle reads.
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsIq1S.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsIq1S.Size}.",
                nameof(weightsIq1S));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsIq1S.Handle, x.Handle, y.Handle };
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
