using System;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ2_S MMVQ decode-path GEMV via integer dp4a (issue #339). The activation must
/// already be Q8_1 (<see cref="QuantizeQ8_1Kernel"/>); each weight is reconstructed
/// from the shared 1024-entry <c>iq2s_grid</c> (<see cref="Iq2Codebooks"/>) and an
/// explicit per-pair 8-bit sign mask as sign·grid, which is int8 → fed to dp4a.
/// 5 bindings (no ksigns): weight, xq, xds, y, grid. Created only when the device
/// advertises <see cref="VulkanDevice.HasIntegerDotProduct"/>; the router falls back
/// to <see cref="MatMulIq2SGemvF32Kernel"/> otherwise. NOT bit-exact — argmax + tolerance.
/// </summary>
public sealed class MatMulIq2SMmvqKernel : IDisposable
{
    /// <summary>IQ2_S super-block bytes.</summary>
    public const int Iq2SBlockBytes = 82;
    /// <summary>Elements per IQ2_S super-block.</summary>
    public const int Iq2SGroupSize = 256;

    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly Iq2Codebooks _codebooks;
    private bool _disposed;

    private MatMulIq2SMmvqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, Iq2Codebooks codebooks)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 5);
        _codebooks = codebooks;
    }

    /// <summary>Loads <c>matmul_iq2_s_mmvq.spv</c> and reuses the shared <paramref name="codebooks"/>;
    /// null when the SPV is missing or the device lacks integer-dot-product support.</summary>
    internal static MatMulIq2SMmvqKernel? TryCreate(VulkanDevice device, string spvDir, Iq2Codebooks codebooks)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_iq2_s_mmvq.spv");
        if (!File.Exists(path))
            return null;

        uint requiredSubgroupSize = Wave32SubgroupControl.RequiredSubgroupSizeFor(device);

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[5];
            for (int i = 0; i < 5; i++) bindings[i] = new VkDescriptorBinding((uint)i);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        return new MatMulIq2SMmvqKernel(device, module, pipeline, pool, codebooks);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the IQ2_S MMVQ GEMV into <paramref name="cmdBuf"/>.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weights, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Iq2SGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq2SGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq2SGroupSize;
        long rowBytes = (long)blocksPerRow * Iq2SBlockBytes;

        if (weights.Size < (long)m * rowBytes)
            throw new ArgumentException("Weights buffer too small.", nameof(weights));
        if (xq.Size < QuantizeQ8_1Kernel.PackedBytes(k))
            throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1Kernel.ScaleBytes(k))
            throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[5]
        {
            weights.Handle, xq.Handle, xds.Handle, y.Handle,
            _codebooks.Iq2SGrid.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4] { (uint)m, (uint)k, (uint)blocksPerRow, 0u };
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
        // _codebooks are shared/owned by the F32 kernel set — do not dispose here.
    }
}
