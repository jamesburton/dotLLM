using System;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q4_K MMVQ decode-path GEMV: <c>y[M] = W_q4k[M,K] @ x[K]</c> via integer dp4a.
/// The activation <c>x</c> must already be quantized to Q8_1
/// (<see cref="QuantizeQ8_1Kernel"/>); this kernel runs
/// <c>dotPacked4x8AccSatEXT</c> (4×int8 → int32 saturating accumulate) against
/// the 4-bit weight nibbles, then applies the Q4_K super-block scale/min:
/// <c>d·scale·(d_x·dot) − dmin·min·s</c> per 32-element sub-block.
/// </summary>
/// <remarks>
/// <para>
/// Replaces the F32-in <see cref="MatMulQ4KGemvF32Kernel"/> on the decode path
/// (seqLen==1) on devices that advertise
/// <see cref="VulkanDevice.HasIntegerDotProduct"/>. Q4_K decode is the largest
/// remaining gap vs llama.cpp; the float dequant+madd of the F32-in GEMV was a
/// big part of it, and the integer-dot path removes it.
/// </para>
/// <para>
/// Result is NOT bit-exact vs the F32-in GEMV (the activation is int8-quantized
/// first). Per-32-block scaling keeps the error at the int8-activation-quant
/// level — validated against the CPU F32 oracle with an argmax-exact +
/// tolerance parity test.
/// </para>
/// <para>
/// Dispatch: one workgroup per output row, 128 threads, shared-memory reduction
/// (mirrors the F32-in Q4_K GEMV). Same 144-byte super-block layout.
/// </para>
/// </remarks>
public sealed class MatMulQ4KMmvqKernel : IDisposable
{
    /// <summary>Q4_K super-block: 144 bytes for 256 elements.</summary>
    public const int Q4KBlockBytes = 144;

    /// <summary>Elements per Q4_K super-block.</summary>
    public const int Q4KGroupSize = 256;

    /// <summary>Workgroup width — must match the shader's <c>local_size_x</c>. One
    /// wave32 subgroup per output row (issue #338 coalesced lane=K-position GEMV).</summary>
    private const int WorkgroupSize = 32;

    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, pad

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ4KMmvqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>
    /// Loads <c>matmul_q4_k_mmvq.spv</c> from <paramref name="spvDir"/> and
    /// builds the pipeline. Returns <c>null</c> when the SPV is missing OR when
    /// the device does not advertise integer-dot-product support — the router
    /// falls back to <see cref="MatMulQ4KGemvF32Kernel"/> in either case.
    /// </summary>
    public static MatMulQ4KMmvqKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_q4_k_mmvq.spv");
        if (!File.Exists(path))
            return null;

        // Pin the decode GEMV to wave32 on devices that support a required
        // compute subgroup size (issue #54 / #330). On RDNA3.5 (gfx1151) the
        // driver defaults compute to wave64; forcing wave32 PER-KERNEL here —
        // never globally — matches llama.cpp's K-quant decode strategy. Gated
        // on device support so it cleanly falls back to the driver default
        // (subgroupSize=0 = unset) elsewhere, and on an env opt-out for A/B.
        uint requiredSubgroupSize = Wave32SubgroupControl.RequiredSubgroupSizeFor(device);

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            bindings[3] = new VkDescriptorBinding(3);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindings,
                pushConstantBytes: PushConstantBytes,
                requiredSubgroupSize: requiredSubgroupSize);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MatMulQ4KMmvqKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Records the Q4_K MMVQ GEMV into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="weightsQ4K">Raw Q4_K blob of <c>M * (K/256) * 144</c> bytes.</param>
    /// <param name="xq">Packed-int8 quantized activations (<see cref="QuantizeQ8_1Kernel"/>), <c>K/4</c> uints.</param>
    /// <param name="xds">Per-block (scale, sum) of the quantized activations, <c>K/32</c> vec2.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 256).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ4K, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q4KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q4KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q4KGroupSize;
        long rowBytes = (long)blocksPerRow * Q4KBlockBytes;

        long weightsMin = (long)m * rowBytes;
        if (weightsQ4K.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ4K.Size}.", nameof(weightsQ4K));
        if (xq.Size < QuantizeQ8_1Kernel.PackedBytes(k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1Kernel.ScaleBytes(k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[4] { weightsQ4K.Handle, xq.Handle, xds.Handle, y.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        // One workgroup per output row (the shader uses gl_WorkGroupID.x as the
        // row index and reduces 128 partials in shared memory).
        Span<uint> pc = stackalloc uint[4]
        {
            (uint)m, (uint)k, (uint)blocksPerRow, 0u,
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
