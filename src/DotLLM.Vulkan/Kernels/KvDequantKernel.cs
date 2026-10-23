using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Dispatches a KV-cache dequantize compute shader (split Q8_0 or Q4_0 → FP32).
/// Reads <c>scales[blockIdx]</c> + <c>quants[blockIdx × N + ...]</c> and writes
/// contiguous FP32 output.
/// </summary>
public sealed class KvDequantKernel : VulkanComputeKernelBase
{
    private const int BindingCount = 3; // scales, quants, dst
    private const int PushBytes = 3 * sizeof(uint);
    private const int WorkgroupSize = 256;

    private KvDequantKernel(VulkanDevice device, string spvDir, string spvFile)
        : base(device, spvDir, spvFile, BindingCount, PushBytes)
    {
    }

    /// <summary>Creates a kernel for the given quantization type.</summary>
    public static KvDequantKernel Create(VulkanDevice device, string spvDir, KvCacheDType dtype) => dtype switch
    {
        KvCacheDType.Q8_0 => new(device, spvDir, "kv_dequant_q8_0_f32.spv"),
        KvCacheDType.Q4_0 => new(device, spvDir, "kv_dequant_q4_0_f32.spv"),
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "KvDequantKernel supports Q8_0 and Q4_0.")
    };

    /// <summary>
    /// Records the dequant dispatch into <paramref name="cmdBuf"/>. One thread per block,
    /// dispatched as <c>ceil(totalBlocks / 256)</c> workgroups.
    /// </summary>
    /// <param name="cmdBuf">Active command buffer.</param>
    /// <param name="scales">Source float-scale buffer (read from <paramref name="srcOffset"/>).</param>
    /// <param name="quants">Source packed-quant buffer (read from <c>srcOffset × quantUintsPerBlock</c>).</param>
    /// <param name="dst">FP32 destination buffer (written at <paramref name="dstOffset"/>).</param>
    /// <param name="totalBlocks">Number of 32-element blocks to dequantize.</param>
    /// <param name="srcOffset">Block offset in <paramref name="scales"/>/<paramref name="quants"/>.</param>
    /// <param name="dstOffset">Element offset in <paramref name="dst"/>.</param>
    public unsafe void Record(nint cmdBuf,
        VulkanDevice.Buffer scales, VulkanDevice.Buffer quants, VulkanDevice.Buffer dst,
        int totalBlocks, int srcOffset, int dstOffset)
    {
        if (totalBlocks <= 0) return;

        Span<nint> buffers = stackalloc nint[BindingCount]
            { scales.Handle, quants.Handle, dst.Handle };
        Span<uint> pc = stackalloc uint[3]
            { (uint)totalBlocks, (uint)srcOffset, (uint)dstOffset };
        BindAndPush(cmdBuf, buffers, pc);

        uint groups = ((uint)totalBlocks + WorkgroupSize - 1) / WorkgroupSize;
        VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);
    }

    /// <summary>Synchronous dequant (one-shot submit). For tests.</summary>
    public void Launch(VulkanDevice.Buffer scales, VulkanDevice.Buffer quants, VulkanDevice.Buffer dst,
                       int totalBlocks, int srcOffset, int dstOffset)
    {
        using var ctx = Device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, scales, quants, dst, totalBlocks, srcOffset, dstOffset);
        ctx.SubmitAndWait();
    }
}
