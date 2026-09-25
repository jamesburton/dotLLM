using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ4_NL MMQ prefill-path batched GEMM: <c>C[N, M] = B[N, K] @ W_iq4nl[M, K]^T</c>
/// via integer dp4a. Decodes IQ4_NL nibbles through the embedded 16-entry int8 codebook
/// into shared memory, dp4a per 32-element block scaled by <c>d·d_x</c> — no sub-block
/// scale, no min term (the simplest MMQ). The activation <c>B</c> must already be Q8_1
/// row-wise (<see cref="QuantizeQ8_1RowsKernel"/>).
/// </summary>
/// <remarks>
/// The seqLen&gt;1 analogue of <see cref="MatMulIq4NlMmvqKernel"/> (decode). Replaces the
/// dequant→FP GEMM (<see cref="MatMulIq4NlGemmF32Kernel"/>) on the seqLen&gt;1 path; falls
/// back to it when the device lacks integer-dot-product support. NOT bit-exact (activation
/// int8-quant); validated against the CPU F32 oracle.
/// </remarks>
/// <remarks>
/// Dispatch: 2-D grid, workgroup <c>(16, 16, 1)</c> — one 64×64 output tile of <c>C</c>
/// per workgroup (issue #367 register-tiled rewrite, direct port of #366's Q8_0 tiling —
/// same single-32-block-per-row layout, only the weight decode differs). Each thread
/// computes a 4×4 register tile (strided by 16).
/// </remarks>
public sealed class MatMulIq4NlMmqKernel : IDisposable
{
    /// <summary>IQ4_NL block: fp16 d + qs[16].</summary>
    public const int Iq4NlBlockBytes = QuantFormat.IQ4_NLBlockBytes;

    /// <summary>Elements per IQ4_NL block.</summary>
    public const int Iq4NlGroupSize = QuantFormat.LegacyGroupSize;

    private const int TileM = 64;
    private const int TileN = 64;
    private const int PushConstantBytes = 5 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulIq4NlMmqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>Loads <c>matmul_iq4_nl_mmq.spv</c>; null when missing or no integer-dot support.</summary>
    public static MatMulIq4NlMmqKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_iq4_nl_mmq.spv");
        if (!File.Exists(path))
            return null;

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
                pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MatMulIq4NlMmqKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the MMQ GEMM synchronously (one-shot submit + fence wait).</summary>
    public void Launch(
        VulkanDevice.Buffer weightsIq4Nl, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsIq4Nl, xq, xds, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the MMQ IQ4_NL GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsIq4Nl, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % Iq4NlGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq4NlGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq4NlGroupSize;
        long rowBytes = (long)blocksPerRow * Iq4NlBlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsIq4Nl.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsIq4Nl.Size}.", nameof(weightsIq4Nl));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(n, k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(n, k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (outputC.Size < (long)n * m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[4] { weightsIq4Nl.Handle, xq.Handle, xds.Handle, outputC.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5]
        {
            (uint)m, (uint)k, (uint)n, (uint)blocksPerRow, (uint)rowUints,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupsX = (uint)((m + TileM - 1) / TileM);
        uint groupsY = (uint)((n + TileN - 1) / TileN);
        VulkanApi.vkCmdDispatch(cmdBuf, groupsX, groupsY, 1);
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
