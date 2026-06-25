using System;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ3_S MMVQ decode-path GEMV via integer dp4a (issue #339). The activation must
/// already be Q8_1 (<see cref="QuantizeQ8_1Kernel"/>); each pair's 8 weights are
/// reconstructed from two 4-byte <c>iq3s_grid</c> entries (9-bit index) + an explicit
/// per-pair 8-bit sign mask (<see cref="Iq3Codebooks"/>) as sign·grid (int8) → fed to
/// dp4a. 5 bindings (no ksigns): weight, xq, xds, y, grid. Created only when the device
/// advertises <see cref="VulkanDevice.HasIntegerDotProduct"/>; the router falls back to
/// <see cref="MatMulIq3SGemvF32Kernel"/> otherwise. NOT bit-exact — argmax + tolerance.
/// </summary>
public sealed class MatMulIq3SMmvqKernel : IDisposable
{
    /// <summary>IQ3_S super-block bytes.</summary>
    public const int Iq3SBlockBytes = 110;
    /// <summary>Elements per IQ3_S super-block.</summary>
    public const int Iq3SGroupSize = 256;

    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly Iq3Codebooks _codebooks;
    private bool _disposed;

    private MatMulIq3SMmvqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, Iq3Codebooks codebooks)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 5);
        _codebooks = codebooks;
    }

    /// <summary>Loads <c>matmul_iq3_s_mmvq.spv</c> and reuses the shared <paramref name="codebooks"/>;
    /// null when the SPV is missing or the device lacks integer-dot-product support.</summary>
    internal static MatMulIq3SMmvqKernel? TryCreate(VulkanDevice device, string spvDir, Iq3Codebooks codebooks)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_iq3_s_mmvq.spv");
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
        return new MatMulIq3SMmvqKernel(device, module, pipeline, pool, codebooks);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the IQ3_S MMVQ GEMV into <paramref name="cmdBuf"/>.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weights, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Iq3SGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq3SGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq3SGroupSize;
        long rowBytes = (long)blocksPerRow * Iq3SBlockBytes;

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
            _codebooks.Iq3SGrid.Handle,
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
