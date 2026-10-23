using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q4_0 decode-path GEMV: <c>y[M] = W_q4_0[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Weight layout mirrors the CPU oracle <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeQ4_0</c>
/// and llama.cpp's <c>block_q4_0</c>: each 32 contiguous columns of a row form one
/// 18-byte block - 2 bytes fp16 <c>d</c> and 16 bytes <c>qs</c> holding 32 4-bit
/// values (low nibble = element <c>j</c>, high nibble = element <c>j + 16</c>);
/// the weight is <c>d * (nibble - 8)</c>.
/// </para>
/// <para>
/// Activation <c>x</c> is FP32; output <c>y</c> is FP32. One workgroup per output
/// row, 128 threads, shared-memory tree reduce — same dispatch shape as the
/// K-quant GEMVs.
/// </para>
/// </remarks>
public sealed class MatMulQ4_0GemvF32Kernel : IDisposable
{
    /// <summary>Q4_0 block: 2 + 16 = 18 bytes.</summary>
    public const int Q4_0BlockBytes = QuantFormat.Q4_0BlockBytes;

    /// <summary>Elements per Q4_0 block.</summary>
    public const int Q4_0GroupSize = QuantFormat.LegacyGroupSize;

    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ4_0GemvF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_q4_0_f32_gemv.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulQ4_0GemvF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_q4_0_f32_gemv.spv");
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
        return new MatMulQ4_0GemvF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the GEMV synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ4_0, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ4_0, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the Q4_0 GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ4_0, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q4_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q4_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q4_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q4_0BlockBytes;
        // 18 is not divisible by 4 — every block pair (36 bytes) is aligned, but a single
        // block crosses a uint boundary. rowUints rounds up to cover safe straddle reads.
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ4_0.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ4_0.Size}.",
                nameof(weightsQ4_0));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsQ4_0.Handle, x.Handle, y.Handle };
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
