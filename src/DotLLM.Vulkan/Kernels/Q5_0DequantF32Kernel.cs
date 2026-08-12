using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q5_0 → FP32 dequantization. Reads a tightly-packed Q5_0 blob and produces a
/// contiguous FP32 buffer.
/// </summary>
/// <remarks>
/// Layout matches <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeQ5_0Scalar</c> and
/// llama.cpp's <c>block_q5_0</c>: 22 bytes per 32 elements (fp16 <c>d</c>, 4-byte
/// <c>qh</c> holding the 5th bit of every element, 16 bytes of packed 4-bit
/// <c>qs</c> nibbles). One workgroup covers 8 blocks (256 elements), 256
/// threads, one element per thread.
/// </remarks>
public sealed class Q5_0DequantF32Kernel : VulkanComputeKernelBase
{
    /// <summary>Q5_0 block: 2 (fp16 d) + 4 (qh) + 16 (qs) = 22 bytes.</summary>
    public const int Q5_0BlockBytes = 22;

    /// <summary>Elements per Q5_0 block.</summary>
    public const int Q5_0GroupSize = 32;

    /// <summary>Blocks handled per workgroup dispatch.</summary>
    private const int BlocksPerWorkgroup = 8;

    private const int PushConstantBytes = 2 * sizeof(uint);

    private Q5_0DequantF32Kernel(VulkanDevice device, string spvDir)
        : base(device, spvDir, "q5_0_dequant_f32.spv", buffersPerSet: 2, pushConstantBytes: PushConstantBytes)
    {
    }

    /// <summary>Loads <c>q5_0_dequant_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static Q5_0DequantF32Kernel Create(VulkanDevice device, string spvDir) => new(device, spvDir);

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

        long srcMin = (long)totalBlocks * Q5_0BlockBytes;
        long dstMin = (long)totalBlocks * Q5_0GroupSize * sizeof(float);
        if (src.Size < srcMin)
            throw new ArgumentException($"Source buffer too small: need >= {srcMin} bytes.", nameof(src));
        if (dst.Size < dstMin)
            throw new ArgumentException($"Destination buffer too small: need >= {dstMin} bytes.", nameof(dst));

        int srcUints = (int)((srcMin + 3) / 4);

        Span<nint> buffers = stackalloc nint[2] { src.Handle, dst.Handle };
        Span<uint> pc = stackalloc uint[2] { (uint)totalBlocks, (uint)srcUints };
        BindAndPush(cmdBuf, buffers, pc);

        uint groupCount = (uint)((totalBlocks + BlocksPerWorkgroup - 1) / BlocksPerWorkgroup);
        VulkanApi.vkCmdDispatch(cmdBuf, groupCount, 1, 1);
    }
}
