using System;
using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ2_XXS MMQ prefill-path batched GEMM: <c>C[N, M] = B[N, K] @ W_iq2xxs[M, K]^T</c>
/// via integer dp4a. Decodes each weight from the shared 256-entry <c>iq2xxs_grid</c> +
/// 128-entry <c>ksigns</c> codebooks (<see cref="Iq2Codebooks"/>) as sign·grid (int8) into
/// shared memory, dp4a per 32-element sub-block scaled by <c>db·d_x</c> (no min term). The
/// activation <c>B</c> must already be Q8_1 row-wise (<see cref="QuantizeQ8_1RowsKernel"/>).
/// 6 bindings: weight, xq, xds, C, grid, ksigns.
/// </summary>
/// <remarks>
/// The seqLen&gt;1 analogue of <see cref="MatMulIq2XxsMmvqKernel"/> (decode). Replaces the
/// dequant→FP GEMM (<see cref="MatMulIq2XxsGemmF32Kernel"/>) on the seqLen&gt;1 path; falls
/// back to it when the device lacks integer-dot-product support. NOT bit-exact (activation
/// int8-quant); validated against the CPU F32 oracle. Workgroup <c>(16,16,1)</c>.
/// </remarks>
public sealed class MatMulIq2XxsMmqKernel : IDisposable
{
    /// <summary>IQ2_XXS super-block bytes.</summary>
    public const int Iq2XxsBlockBytes = QuantFormat.IQ2_XXSBlockBytes;
    /// <summary>Elements per IQ2_XXS super-block.</summary>
    public const int Iq2XxsGroupSize = QuantFormat.KQuantGroupSize;

    private const int TileM = 16;
    private const int TileN = 16;
    private const int PushConstantBytes = 5 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly Iq2Codebooks _codebooks;
    private bool _disposed;

    private MatMulIq2XxsMmqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, Iq2Codebooks codebooks)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 6);
        _codebooks = codebooks;
    }

    /// <summary>Loads <c>matmul_iq2_xxs_mmq.spv</c> and reuses the shared <paramref name="codebooks"/>;
    /// null when the SPV is missing or the device lacks integer-dot-product support.</summary>
    internal static MatMulIq2XxsMmqKernel? TryCreate(VulkanDevice device, string spvDir, Iq2Codebooks codebooks)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_iq2_xxs_mmq.spv");
        if (!File.Exists(path))
            return null;

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[6];
            for (int i = 0; i < 6; i++) bindings[i] = new VkDescriptorBinding((uint)i);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 6);
        return new MatMulIq2XxsMmqKernel(device, module, pipeline, pool, codebooks);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the MMQ IQ2_XXS GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weights, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % Iq2XxsGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq2XxsGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq2XxsGroupSize;
        long rowBytes = (long)blocksPerRow * Iq2XxsBlockBytes;

        if (weights.Size < (long)m * rowBytes)
            throw new ArgumentException("Weights buffer too small.", nameof(weights));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(n, k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(n, k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (outputC.Size < (long)n * m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[6]
        {
            weights.Handle, xq.Handle, xds.Handle, outputC.Handle,
            _codebooks.Iq2XxsGrid.Handle, _codebooks.Ksigns.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5] { (uint)m, (uint)k, (uint)n, (uint)blocksPerRow, 0u };
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
        // _codebooks are shared/owned by the F32 kernel set — do not dispose here.
    }
}
