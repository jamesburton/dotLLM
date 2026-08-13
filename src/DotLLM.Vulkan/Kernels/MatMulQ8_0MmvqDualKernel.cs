using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Fused dual-output Q8_0 MMVQ decode GEMV (issue #71): computes two
/// independent same-<c>K</c> GEMVs sharing one pre-quantized Q8_1 activation
/// in a SINGLE dispatch — <c>ya = Wa_q8 @ x</c> and <c>yb = Wb_q8 @ x</c>.
/// </summary>
/// <remarks>
/// <para>
/// Extends the shared-activation-quant optimisation already used by
/// <c>RecordSharedInputMmvqGroup</c> (one <see cref="QuantizeQ8_1Kernel"/>
/// dispatch feeding N barrier-free <see cref="MatMulQ8_0MmvqKernel"/>
/// dispatches) one step further for same-<c>K</c> PAIRS: instead of two
/// separate <c>vkCmdDispatch</c> + pipeline-bind pairs, one dispatch covers
/// <c>Ma + Mb</c> output rows, workgroup <c>m &lt; Ma</c> computing a row of
/// A and <c>m &gt;= Ma</c> computing a row of B. On an underfilled decode GPU
/// (SmolLM-class hidden=576) this removes one dispatch boundary per fused
/// pair — the FFN gate_proj/up_proj pair is the natural target (both read
/// the post-ffn-norm activation, both K = hidden_size).
/// </para>
/// <para>
/// Per-row numerics are copied verbatim from <c>matmul_q8_0_mmvq.comp</c>
/// (coalesced lane=K-position layout), so each output row is bit-identical
/// to running <see cref="MatMulQ8_0MmvqKernel"/> twice against the same
/// inputs — this kernel changes dispatch shape only, not arithmetic.
/// </para>
/// </remarks>
public sealed class MatMulQ8_0MmvqDualKernel : IDisposable
{
    /// <summary>Q8_0 block: 2 bytes fp16 scale + 32 signed int8 values.</summary>
    public const int Q8_0BlockBytes = QuantFormat.Q8_0BlockBytes;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = QuantFormat.LegacyGroupSize;

    private const int PushConstantBytes = 3 * sizeof(uint); // Ma, Mb, blocksPerRow

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ8_0MmvqDualKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 6);
    }

    /// <summary>
    /// Loads <c>matmul_q8_0_mmvq_dual.spv</c> from <paramref name="spvDir"/> and
    /// builds the pipeline. Returns <c>null</c> when the SPV is missing OR when
    /// the device does not advertise integer-dot-product support — callers fall
    /// back to two separate <see cref="MatMulQ8_0MmvqKernel"/> dispatches.
    /// </summary>
    public static MatMulQ8_0MmvqDualKernel? TryCreate(
        VulkanDevice device, string spvDir, uint? requiredSubgroupSizeOverride = null)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_q8_0_mmvq_dual.spv");
        if (!File.Exists(path))
            return null;

        // Same wave-width pinning policy as MatMulQ8_0MmvqKernel — the per-row
        // body is copied verbatim, so it needs the same subgroup=32 contract.
        uint requiredSubgroupSize =
            requiredSubgroupSizeOverride ?? Wave32SubgroupControl.RequiredSubgroupSizeFor(device);

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[6];
            for (int i = 0; i < 6; i++)
                bindings[i] = new VkDescriptorBinding((uint)i);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 6);
        return new MatMulQ8_0MmvqDualKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Records the fused dual-output MMVQ GEMV into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="weightsA">Raw Q8_0 blob A of <c>Ma * (K/32) * 34</c> bytes.</param>
    /// <param name="weightsB">Raw Q8_0 blob B of <c>Mb * (K/32) * 34</c> bytes.</param>
    /// <param name="xq">Packed-int8 quantized activations shared by both GEMVs, <c>K/4</c> uints.</param>
    /// <param name="xds">Per-block (scale, sum) of the quantized activations, <c>K/32</c> vec2.</param>
    /// <param name="ya">FP32 output buffer of length <paramref name="mA"/>.</param>
    /// <param name="yb">FP32 output buffer of length <paramref name="mB"/>.</param>
    /// <param name="mA">Output dimension of A.</param>
    /// <param name="mB">Output dimension of B.</param>
    /// <param name="k">Shared input dimension (must be a multiple of 32).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsA, VulkanDevice.Buffer weightsB,
        VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer ya, VulkanDevice.Buffer yb,
        int mA, int mB, int k)
    {
        if (mA <= 0) throw new ArgumentOutOfRangeException(nameof(mA));
        if (mB <= 0) throw new ArgumentOutOfRangeException(nameof(mB));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q8_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q8_0BlockBytes;

        if (weightsA.Size < (long)mA * rowBytes)
            throw new ArgumentException("Weight buffer A too small.", nameof(weightsA));
        if (weightsB.Size < (long)mB * rowBytes)
            throw new ArgumentException("Weight buffer B too small.", nameof(weightsB));
        if (xq.Size < QuantizeQ8_1Kernel.PackedBytes(k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1Kernel.ScaleBytes(k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (ya.Size < (long)mA * sizeof(float))
            throw new ArgumentException("Output buffer A too small.", nameof(ya));
        if (yb.Size < (long)mB * sizeof(float))
            throw new ArgumentException("Output buffer B too small.", nameof(yb));

        Span<nint> buffers = stackalloc nint[6]
        {
            weightsA.Handle, weightsB.Handle, xq.Handle, xds.Handle, ya.Handle, yb.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[3] { (uint)mA, (uint)mB, (uint)blocksPerRow };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)(mA + mB), 1, 1);
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
