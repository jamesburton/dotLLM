using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q2_K → FP32 dequantization. Reads a tightly-packed Q2_K blob and produces a
/// contiguous FP32 buffer.
/// </summary>
/// <remarks>
/// Layout matches <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ2_K</c> and
/// llama.cpp's <c>block_q2_K</c>: 84 bytes per 256 elements (16 bytes packed
/// per-sub-block <c>scale|dmCoef</c> nibbles, 64 bytes of 2-bit quants, fp16
/// <c>d</c> + <c>dmin</c>). One workgroup per super-block, 256 threads, one
/// element per thread.
/// </remarks>
public sealed class Q2KDequantF32Kernel : VulkanComputeKernelBase
{
    /// <summary>Q2_K super-block: 16 + 64 + 2 + 2 = 84 bytes.</summary>
    public const int Q2_KBlockBytes = 84;

    /// <summary>Elements per Q2_K super-block.</summary>
    public const int Q2_KGroupSize = 256;

    private const int PushConstantBytes = 2 * sizeof(uint);

    private Q2KDequantF32Kernel(VulkanDevice device, string spvDir)
        : base(device, spvDir, "q2_k_dequant_f32.spv", buffersPerSet: 2, pushConstantBytes: PushConstantBytes)
    {
    }

    /// <summary>Loads <c>q2_k_dequant_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static Q2KDequantF32Kernel Create(VulkanDevice device, string spvDir) => new(device, spvDir);

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

        long srcMin = (long)totalBlocks * Q2_KBlockBytes;
        long dstMin = (long)totalBlocks * Q2_KGroupSize * sizeof(float);
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
