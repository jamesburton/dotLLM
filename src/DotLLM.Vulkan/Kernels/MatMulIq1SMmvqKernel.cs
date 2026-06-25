using System;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// IQ1_S MMVQ decode-path GEMV via integer dp4a (issue #339). The activation must
/// already be Q8_1 (<see cref="QuantizeQ8_1Kernel"/>); each weight is reconstructed
/// as <c>dl·(g + delta)</c> where <c>g ∈ {-1,0,+1}</c> comes from the SSBO-bound
/// 2048-entry <c>iq1s_grid</c> (<see cref="Iq1Codebooks"/>) — the ternary lanes feed
/// dp4a verbatim and the per-sub-block <c>delta</c> term is <c>delta·d_x·Σxq</c>.
/// 5 bindings: weight, xq, xds, y, grid. Created only when the device advertises
/// <see cref="VulkanDevice.HasIntegerDotProduct"/>; the router falls back to
/// <see cref="MatMulIq1SGemvF32Kernel"/> otherwise. NOT bit-exact — argmax + tolerance.
/// </summary>
public sealed class MatMulIq1SMmvqKernel : IDisposable
{
    /// <summary>IQ1_S super-block bytes.</summary>
    public const int Iq1SBlockBytes = 50;
    /// <summary>Elements per IQ1_S super-block.</summary>
    public const int Iq1SGroupSize = 256;

    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly Iq1Codebooks _codebooks;
    private readonly bool _ownsCodebooks;
    private bool _disposed;

    private MatMulIq1SMmvqKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, Iq1Codebooks codebooks, bool ownsCodebooks)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 5);
        _codebooks = codebooks;
        _ownsCodebooks = ownsCodebooks;
    }

    /// <summary>Loads <c>matmul_iq1_s_mmvq.spv</c>, allocates its own <see cref="Iq1Codebooks"/>
    /// (no other kernel shares it), and creates the pipeline; null when the SPV is missing or
    /// the device lacks integer-dot-product support.</summary>
    internal static MatMulIq1SMmvqKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            return null;

        string path = Path.Combine(spvDir, "matmul_iq1_s_mmvq.spv");
        if (!File.Exists(path))
            return null;

        var codebooks = Iq1Codebooks.Create(device);
        try
        {
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
            return new MatMulIq1SMmvqKernel(device, module, pipeline, pool, codebooks, ownsCodebooks: true);
        }
        catch
        {
            codebooks.Dispose();
            throw;
        }
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the IQ1_S MMVQ GEMV into <paramref name="cmdBuf"/>.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weights, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Iq1SGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq1SGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq1SGroupSize;
        long rowBytes = (long)blocksPerRow * Iq1SBlockBytes;

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
            _codebooks.Iq1SGrid.Handle,
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
        if (_ownsCodebooks) _codebooks.Dispose();
    }
}
