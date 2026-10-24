using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Dispatches a KV-cache quantize compute shader (FP32 → split Q8_0 or Q4_0).
/// Split layout: <c>scales[blockIdx]</c> (float) + <c>quants[blockIdx × N + ...]</c>
/// (packed int8 or nibbles as uint).
/// </summary>
public sealed class KvQuantizeKernel : VulkanComputeKernelBase
{
    private const int BindingCount = 3; // src, scales, quants
    private const int PushBytes = 3 * sizeof(uint);
    private const int WorkgroupSize = 256;

    private readonly int _quantUintsPerBlock;

    private KvQuantizeKernel(VulkanDevice device, string spvDir, string spvFile, int quantUintsPerBlock)
        : base(device, spvDir, spvFile, BindingCount, PushBytes)
    {
        _quantUintsPerBlock = quantUintsPerBlock;
    }

    /// <summary>Creates a kernel for the given quantization type.</summary>
    public static KvQuantizeKernel Create(VulkanDevice device, string spvDir, KvCacheDType dtype) => dtype switch
    {
        KvCacheDType.Q8_0 => new(device, spvDir, "kv_quantize_q8_0_f32.spv", quantUintsPerBlock: 8),
        KvCacheDType.Q4_0 => new(device, spvDir, "kv_quantize_q4_0_f32.spv", quantUintsPerBlock: 4),
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "KvQuantizeKernel supports Q8_0 and Q4_0.")
    };

    /// <summary>Packed uint count per 32-element block in the quants buffer (8 for Q8_0, 4 for Q4_0).</summary>
    public int QuantUintsPerBlock => _quantUintsPerBlock;

    /// <summary>
    /// Records the quantize dispatch into <paramref name="cmdBuf"/>. One thread per block,
    /// dispatched as <c>ceil(totalBlocks / 256)</c> workgroups.
    /// </summary>
    /// <param name="cmdBuf">Active command buffer.</param>
    /// <param name="src">FP32 source buffer (read from <paramref name="srcOffset"/>).</param>
    /// <param name="scales">Destination float-scale buffer (written at <paramref name="dstOffset"/>).</param>
    /// <param name="quants">Destination packed-quant buffer (written at <c>dstOffset × quantUintsPerBlock</c>).</param>
    /// <param name="totalBlocks">Number of 32-element blocks to quantize.</param>
    /// <param name="srcOffset">Element offset in <paramref name="src"/>.</param>
    /// <param name="dstOffset">Block offset in <paramref name="scales"/>/<paramref name="quants"/>.</param>
    public unsafe void Record(nint cmdBuf,
        VulkanDevice.Buffer src, VulkanDevice.Buffer scales, VulkanDevice.Buffer quants,
        int totalBlocks, int srcOffset, int dstOffset)
    {
        if (totalBlocks <= 0) return;

        Span<nint> buffers = stackalloc nint[BindingCount]
            { src.Handle, scales.Handle, quants.Handle };
        Span<uint> pc = stackalloc uint[3]
            { (uint)totalBlocks, (uint)srcOffset, (uint)dstOffset };
        BindAndPush(cmdBuf, buffers, pc);

        uint groups = ((uint)totalBlocks + WorkgroupSize - 1) / WorkgroupSize;
        VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);
    }

    /// <summary>Synchronous quantize (one-shot submit). For tests.</summary>
    public void Launch(VulkanDevice.Buffer src, VulkanDevice.Buffer scales, VulkanDevice.Buffer quants,
                       int totalBlocks, int srcOffset, int dstOffset)
    {
        using var ctx = Device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, src, scales, quants, totalBlocks, srcOffset, dstOffset);
        ctx.SubmitAndWait();
    }
}
