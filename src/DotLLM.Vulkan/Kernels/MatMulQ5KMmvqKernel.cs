using System;
using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q5_K MMVQ decode-path GEMV: <c>y[M] = W_q5k[M,K] @ x[K]</c> via integer dp4a.
/// The activation <c>x</c> must already be quantized to Q8_1
/// (<see cref="QuantizeQ8_1Kernel"/>); this kernel runs
/// <c>dotPacked4x8AccSatEXT</c> against the 5-bit weights (4-bit qs nibble +
/// 1-bit qh), then applies the Q5_K super-block scale/min:
/// <c>d·scale·(d_x·dot) − dmin·min·s</c> per 32-element sub-block.
/// </summary>
/// <remarks>
/// <para>
/// NEW dp4a path for Q5_K — previously only the F32-dequant
/// <see cref="MatMulQ5KGemvF32Kernel"/> existed for the decode path. Sibling of
/// the Q4_K / Q6_K MMVQ kernels (issue #338), completing the K-quant decode MMVQ
/// coverage. Created only when the device advertises
/// <see cref="VulkanDevice.HasIntegerDotProduct"/>; the router falls back to the
/// F32-in GEMV otherwise.
/// </para>
/// <para>
/// Result is NOT bit-exact vs the F32-in GEMV (the activation is int8-quantized
/// first) — validated against the CPU F32 oracle with an argmax + tolerance test.
/// </para>
/// <para>
/// Dispatch: one workgroup (= one wave32 subgroup) per output row, coalesced
/// lane=K-position layout. Same 176-byte super-block layout.
/// </para>
/// </remarks>
public sealed class MatMulQ5KMmvqKernel : IDisposable
{
    /// <summary>Q5_K super-block: 176 bytes for 256 elements.</summary>
    public const int Q5KBlockBytes = QuantFormat.Q5_KBlockBytes;

    /// <summary>Elements per Q5_K super-block.</summary>
    public const int Q5KGroupSize = QuantFormat.KQuantGroupSize;

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

    private MatMulQ5KMmvqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>
    /// Loads <c>matmul_q5_k_mmvq.spv</c> from <paramref name="spvDir"/> and builds
    /// the pipeline. Returns <c>null</c> when the SPV is missing OR when the device
    /// does not advertise integer-dot-product support — the router falls back to
    /// <see cref="MatMulQ5KGemvF32Kernel"/> in either case.
    /// </summary>
    public static MatMulQ5KMmvqKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_q5_k_mmvq.spv");
        if (!File.Exists(path))
            return null;

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
        return new MatMulQ5KMmvqKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Records the Q5_K MMVQ GEMV into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="weightsQ5K">Raw Q5_K blob of <c>M * (K/256) * 176</c> bytes.</param>
    /// <param name="xq">Packed-int8 quantized activations (<see cref="QuantizeQ8_1Kernel"/>), <c>K/4</c> uints.</param>
    /// <param name="xds">Per-block (scale, sum) of the quantized activations, <c>K/32</c> vec2.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 256).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ5K, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q5KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q5KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q5KGroupSize;
        long rowBytes = (long)blocksPerRow * Q5KBlockBytes;

        long weightsMin = (long)m * rowBytes;
        if (weightsQ5K.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ5K.Size}.", nameof(weightsQ5K));
        if (xq.Size < QuantizeQ8_1Kernel.PackedBytes(k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1Kernel.ScaleBytes(k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[4] { weightsQ5K.Handle, xq.Handle, xds.Handle, y.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

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
