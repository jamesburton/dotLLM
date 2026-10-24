using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// The DFlash drafter's own Vulkan kernels (issue #1193), the twins of <c>dflash_f32.cu</c>: the DFlash2 grouped convolution, the
/// context-ring write (F32 rows to F16 slots), the fused per-head RMSNorm + NeoX RoPE and the block attention over the F16 ring.
/// Everything else the drafter runs (projections, row norms, SwiGLU, adds, the target's embedding and head) is the trunk's.
/// </summary>
internal sealed class VulkanDFlashKernels : IDisposable
{
    /// <summary>Largest head dimension the norm/RoPE and attention shaders cover (two dims per thread of 128).</summary>
    public const int MaxHeadDim = 256;

    private readonly Op _conv, _ringWrite, _normRope, _attention;

    private VulkanDFlashKernels(VulkanDevice device, string spvDir)
    {
        _conv = new Op(device, spvDir, "dflash_conv_f32.spv", 4, 5);
        _ringWrite = new Op(device, spvDir, "dflash_ring_write_f32.spv", 4, 7);
        _normRope = new Op(device, spvDir, "dflash_norm_rope_f32.spv", 3, 3);
        _attention = new Op(device, spvDir, "dflash_attention_f32.spv", 6, 11);
    }

    /// <summary>Loads the four pipelines from <paramref name="spvDir"/>.</summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <returns>The kernels.</returns>
    public static VulkanDFlashKernels Create(VulkanDevice device, string spvDir) => new(device, spvDir);

    /// <summary>DFlash2 grouped dynamic causal convolution, out of place.</summary>
    public void RecordConv(nint cmd, VulkanDevice.Buffer x, VulkanDevice.Buffer dyn, VulkanDevice.Buffer baseKernel, VulkanDevice.Buffer y,
        int rows, int hidden, int kernel, int group, int side)
        => _conv.Record(cmd, [x.Handle, dyn.Handle, baseKernel.Handle, y.Handle],
            [(uint)rows, (uint)hidden, (uint)kernel, (uint)group, (uint)side], Groups((long)rows * hidden, 256), 1);

    /// <summary>Writes K/V rows <c>[srcRow0, srcRow0 + rows)</c> to the F16 ring slots <c>(pos0 + r) % cap</c> of one layer.</summary>
    public void RecordRingWrite(nint cmd, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer ringK, VulkanDevice.Buffer ringV,
        int rows, int kvDim, int srcRow0, int pos0, int cap, long ringKByteOffset, long ringVByteOffset)
        => _ringWrite.Record(cmd, [k.Handle, v.Handle, ringK.Handle, ringV.Handle],
            [(uint)rows, (uint)kvDim, (uint)srcRow0, (uint)pos0, (uint)cap, Words(ringKByteOffset), Words(ringVByteOffset)],
            Groups((long)rows * kvDim / 2, 256), 1);

    /// <summary>Per-head RMSNorm then NeoX RoPE, in place; <paramref name="cosSin"/> holds <c>[rows, hd/2, (cos, sin)]</c>.</summary>
    public void RecordNormRope(nint cmd, VulkanDevice.Buffer x, VulkanDevice.Buffer norm, VulkanDevice.Buffer cosSin,
        int rows, int heads, int headDim, float eps)
        => _normRope.Record(cmd, [x.Handle, norm.Handle, cosSin.Handle],
            [(uint)heads, (uint)headDim, BitConverter.SingleToUInt32Bits(eps)], (uint)heads, (uint)rows);

    /// <summary>Block attention of <paramref name="b"/> queries over ring positions <c>[lo, start)</c> and the block's own K/V.</summary>
    public void RecordAttention(nint cmd, VulkanDevice.Buffer q, VulkanDevice.Buffer kb, VulkanDevice.Buffer vb,
        VulkanDevice.Buffer ringK, VulkanDevice.Buffer ringV, VulkanDevice.Buffer output,
        int b, int heads, int kvHeads, int headDim, int cap, int lo, int start, int window, float scale,
        long ringKByteOffset, long ringVByteOffset)
        => _attention.Record(cmd, [q.Handle, kb.Handle, vb.Handle, ringK.Handle, ringV.Handle, output.Handle],
            [(uint)b, (uint)heads, (uint)kvHeads, (uint)headDim, (uint)cap, (uint)lo, (uint)start, (uint)window,
             BitConverter.SingleToUInt32Bits(scale), Words(ringKByteOffset), Words(ringVByteOffset)],
            (uint)(b * heads), 1);

    private static uint Groups(long items, int size) => (uint)((items + size - 1) / size);

    private static uint Words(long byteOffset)
        => (byteOffset & 3) == 0 && byteOffset / 4 <= uint.MaxValue
            ? (uint)(byteOffset / 4)
            : throw new ArgumentOutOfRangeException(nameof(byteOffset), "Ring offsets must be whole 32-bit words.");

    /// <summary>Drops every cached descriptor set (a drafter buffer was freed; its handle may be recycled).</summary>
    public void InvalidateDescriptorCaches()
    {
        _conv.InvalidateDescriptorCache();
        _ringWrite.InvalidateDescriptorCache();
        _normRope.InvalidateDescriptorCache();
        _attention.InvalidateDescriptorCache();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _conv.Dispose();
        _ringWrite.Dispose();
        _normRope.Dispose();
        _attention.Dispose();
    }

    private sealed class Op(VulkanDevice device, string spvDir, string spv, int buffers, int pushWords)
        : VulkanComputeKernelBase(device, spvDir, spv, buffers, pushWords * sizeof(uint))
    {
        public void Record(nint cmd, ReadOnlySpan<nint> buffers, ReadOnlySpan<uint> push, uint groupsX, uint groupsY)
        {
            BindAndPush(cmd, buffers, push);
            VulkanApi.vkCmdDispatch(cmd, groupsX, groupsY, 1);
        }
    }
}
