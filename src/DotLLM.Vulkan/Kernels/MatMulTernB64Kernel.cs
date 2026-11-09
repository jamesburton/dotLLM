using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// <see cref="QuantizationType.TERN_B64_F16"/> MMVQ matmul (issue #1706, <c>--tensor-overlay</c>): dense <c>y[n, m] = sum_k w[m, k] * x[n, k]</c> and MoE-indexed
/// (<c>w</c> chosen per row by an expert id through a slot table) in one shader, <c>matmul_tern_b64_mmvq.comp</c>. The weights are read PACKED
/// (18 B per 64 weights = 2.25 bits/weight; never expanded on the device); the activation is the house Q8_1 row quantisation
/// (<see cref="QuantizeQ8_1RowsKernel"/>), so the arithmetic contract matches the CPU <c>TernB64F16.GemmQ8</c>: one int32 dot per 16 columns folded with
/// the 64-column fp16 scale and the per-32 activation scale. Created only when the device advertises integer dot product.
/// </summary>
public sealed class MatMulTernB64Kernel : IDisposable
{
    /// <summary>Bytes per block (64 weights).</summary>
    public const int BlockBytes = QuantFormat.TernB64F16BlockBytes;

    /// <summary>Weights per block.</summary>
    public const int GroupSize = QuantFormat.TernB64GroupSize;

    private const int BuffersPerSet = 6;
    private const int PushConstantBytes = 8 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulTernB64Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: BuffersPerSet);
    }

    /// <summary>Loads <c>matmul_tern_b64_mmvq.spv</c>; null when the SPIR-V is missing or the device lacks integer-dot-product support.</summary>
    /// <param name="device">Target device.</param>
    /// <param name="spvDir">Directory holding the compiled shaders.</param>
    public static MatMulTernB64Kernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct) return null;
        string path = Path.Combine(spvDir, "matmul_tern_b64_mmvq.spv");
        if (!File.Exists(path)) return null;

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[BuffersPerSet];
            for (int i = 0; i < BuffersPerSet; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            // No subgroup assumption: the shader reduces through shared memory, so no required-subgroup-size pin is needed.
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)BuffersPerSet);
        return new MatMulTernB64Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the dense matmul: <c>y[n, m] = W[m, :] . x[n, :]</c> for <paramref name="n"/> activation rows.</summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="weights">Packed TERN_B64_F16 rows, <c>m * (k/64) * 18</c> bytes.</param>
    /// <param name="xq">Q8_1 packed int8 activations <c>[n, k/4]</c> uints.</param>
    /// <param name="xds">Q8_1 (d, s) per 32 block <c>[n, k/32]</c> vec2.</param>
    /// <param name="y">F32 output <c>[n, m]</c>.</param>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Contraction length (multiple of 64).</param>
    /// <param name="n">Activation rows.</param>
    public void Record(nint cmdBuf, VulkanDevice.Buffer weights, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer y, int m, int k, int n)
        => RecordCore(cmdBuf, weights, xq, xds, xq, xq, y, m, k, n, numExperts: 1, xDiv: 1, indexed: false);

    /// <summary>
    /// Records the indexed matmul: row <c>r</c> (of <paramref name="n"/>) with expert <c>e = indices[r]</c> and <c>s = slotOf[e] &gt;= 0</c> gets
    /// <c>y[r, m] = W[s * M + m, :] . x[r / xDiv, :]</c>; rows whose expert has no ternary slot are not written.
    /// </summary>
    /// <param name="cmdBuf">Open command buffer.</param>
    /// <param name="bank">Packed TERN_B64_F16 expert tensors back to back, <c>numSlots * m * (k/64) * 18</c> bytes.</param>
    /// <param name="xq">Q8_1 packed int8 activations.</param>
    /// <param name="xds">Q8_1 scales.</param>
    /// <param name="indices">int32 expert id per row <c>[n]</c>.</param>
    /// <param name="slotOf">int32 per expert: slot in <paramref name="bank"/>, or -1 <c>[numExperts]</c>.</param>
    /// <param name="y">F32 output <c>[n, m]</c> (cells of non-ternary rows keep their previous value).</param>
    /// <param name="m">Per-expert output rows.</param>
    /// <param name="k">Contraction length (multiple of 64).</param>
    /// <param name="n">Number of (token, slot) rows.</param>
    /// <param name="numExperts">Experts in the layer.</param>
    /// <param name="xDiv">Activation row = r / xDiv (1 per-row input; topK = one row per token broadcast to its topK slots).</param>
    public void RecordIndexed(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer slotOf, VulkanDevice.Buffer y, int m, int k, int n, int numExperts, int xDiv)
        => RecordCore(cmdBuf, bank, xq, xds, indices, slotOf, y, m, k, n, numExperts, xDiv, indexed: true);

    private unsafe void RecordCore(nint cmdBuf, VulkanDevice.Buffer weights, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer indices, VulkanDevice.Buffer slotOf, VulkanDevice.Buffer y, int m, int k, int n, int numExperts, int xDiv, bool indexed)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0 || (k % GroupSize) != 0)
            throw new ArgumentException($"k must be a positive multiple of {GroupSize}, got {k}", nameof(k));
        if (xDiv <= 0) throw new ArgumentOutOfRangeException(nameof(xDiv));
        if (n > 65535) throw new ArgumentOutOfRangeException(nameof(n), "dispatch grid y limit");

        int blocksPerRow = k / GroupSize;
        long rowBytes = (long)blocksPerRow * BlockBytes;
        if (!indexed && weights.Size < (long)m * rowBytes) throw new ArgumentException("Weights buffer too small.", nameof(weights));
        int xRows = (n + xDiv - 1) / xDiv;
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(xRows, k)) throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(xRows, k)) throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (y.Size < (long)n * m * sizeof(float)) throw new ArgumentException("y buffer too small.", nameof(y));
        if (indexed)
        {
            if (indices.Size < (long)n * sizeof(int)) throw new ArgumentException("indices buffer too small.", nameof(indices));
            if (slotOf.Size < (long)numExperts * sizeof(int)) throw new ArgumentException("slotOf buffer too small.", nameof(slotOf));
        }

        Span<nint> buffers = stackalloc nint[BuffersPerSet]
        {
            weights.Handle, xq.Handle, xds.Handle, indices.Handle, slotOf.Handle, y.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);
        Span<uint> pc = stackalloc uint[8]
        {
            (uint)m, (uint)k, (uint)n, (uint)numExperts, (uint)blocksPerRow, (uint)xDiv, indexed ? 1u : 0u, 0u,
        };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, (uint)n, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptorPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
