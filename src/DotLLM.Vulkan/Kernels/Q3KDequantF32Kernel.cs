using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q3_K → FP32 dequantization. Reads a tightly-packed Q3_K blob and produces a
/// contiguous FP32 buffer.
/// </summary>
/// <remarks>
/// Layout matches <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ3_KScalar</c>
/// and llama.cpp's <c>block_q3_K</c>: 110 bytes per 256 elements (32 bytes
/// hmask, 64 bytes 2-bit qs, 12 bytes packed 6-bit scales, fp16 <c>d</c>).
/// One workgroup per super-block, 256 threads, one element per thread.
/// </remarks>
public sealed class Q3KDequantF32Kernel : VulkanComputeKernelBase
{
    /// <summary>Q3_K super-block: 32 + 64 + 12 + 2 = 110 bytes.</summary>
    public const int Q3_KBlockBytes = 110;

    /// <summary>Elements per Q3_K super-block.</summary>
    public const int Q3_KGroupSize = 256;

    private const int PushConstantBytes = 2 * sizeof(uint);

    private Q3KDequantF32Kernel(VulkanDevice device, string spvDir)
        : base(device, spvDir, "q3_k_dequant_f32.spv", buffersPerSet: 2, pushConstantBytes: PushConstantBytes)
    {
    }

    /// <summary>Loads <c>q3_k_dequant_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static Q3KDequantF32Kernel Create(VulkanDevice device, string spvDir) => new(device, spvDir);

    /// <summary>Dispatches the dequant synchronously.</summary>
    public void Launch(VulkanDevice.Buffer src, VulkanDevice.Buffer dst, int totalBlocks)
    {
        using var ctx = Device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, src, dst, totalBlocks);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the dequant into <paramref name="cmdBuf"/> without submitting.</summary>
    public void Record(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, int totalBlocks)
    {
        if (totalBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(totalBlocks));

        long srcMin = (long)totalBlocks * Q3_KBlockBytes;
        long dstMin = (long)totalBlocks * Q3_KGroupSize * sizeof(float);
        if (src.Size < srcMin)
            throw new ArgumentException($"Source buffer too small: need >= {srcMin} bytes.", nameof(src));
        if (dst.Size < dstMin)
            throw new ArgumentException($"Destination buffer too small: need >= {dstMin} bytes.", nameof(dst));

        int srcUints = (int)((srcMin + 3) / 4);

        Span<nint> buffers = stackalloc nint[2] { src.Handle, dst.Handle };
        Span<uint> pc = stackalloc uint[2] { (uint)totalBlocks, (uint)srcUints };
        BindAndPush(cmdBuf, buffers, pc);

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)totalBlocks, 1, 1);
    }
}
