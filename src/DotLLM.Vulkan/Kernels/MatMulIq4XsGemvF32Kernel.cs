using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ4_XS decode-path GEMV: <c>y[M] = W_iq4xs[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Weight layout mirrors the CPU oracle <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeIQ4_XS</c>
/// and llama.cpp's <c>block_iq4_xs</c>: each 256 contiguous columns of a row form one
/// 136-byte super-block — 2 bytes fp16 <c>d</c>, 2 bytes <c>scales_h</c>, 4 bytes
/// <c>scales_l</c> (8 x 6-bit sub-block scales packed across those 6 bytes), and
/// 128 bytes <c>qs</c> (8 sub-blocks of 16 bytes / 32 nibbles each). Sub-block
/// scale encodes a signed effective scale <c>ls - 32</c> where
/// <c>ls = (scales_l[ib/2] >> (4*(ib&amp;1))) &amp; 0xF | ((scales_h >> 2*ib) &amp; 0x3) &lt;&lt; 4</c>.
/// The 4-bit qs nibble is an index into the same signed-int8 <c>kvalues_iq4nl[16]</c>
/// lookup as IQ4_NL.
/// </para>
/// <para>
/// Activation <c>x</c> and output <c>y</c> are FP32. One workgroup per output row,
/// 128 threads, shared-memory tree reduce.
/// </para>
/// </remarks>
public sealed class MatMulIq4XsGemvF32Kernel : IDisposable
{
    /// <summary>IQ4_XS super-block: 2 + 2 + 4 + 128 = 136 bytes.</summary>
    public const int IQ4_XSBlockBytes = 136;

    /// <summary>Elements per IQ4_XS super-block.</summary>
    public const int IQ4_XSGroupSize = 256;

    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulIq4XsGemvF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_iq4_xs_f32_gemv.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulIq4XsGemvF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_iq4_xs_f32_gemv.spv");
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
        return new MatMulIq4XsGemvF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the GEMV synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer weightsIq4Xs, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsIq4Xs, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the IQ4_XS GEMV into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsIq4Xs, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % IQ4_XSGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {IQ4_XSGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / IQ4_XSGroupSize;
        long rowBytes = (long)blocksPerRow * IQ4_XSBlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsIq4Xs.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsIq4Xs.Size}.",
                nameof(weightsIq4Xs));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsIq4Xs.Handle, x.Handle, y.Handle };
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
