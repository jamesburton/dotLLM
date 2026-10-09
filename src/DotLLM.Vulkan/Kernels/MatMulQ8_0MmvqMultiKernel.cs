using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q8_0 MMVQ multi-column GEMV (#876): <c>Y[N, M] = X_q8_1[N, K] @ W_q8[M, K]^T</c> for N in 2..<see cref="MaxColumns"/>. Activations come
/// from <see cref="QuantizeQ8_1RowsKernel"/> (row-major <c>[N][K/4]</c> / <c>[N][K/32]</c>). Column c is bit-equal to
/// <see cref="MatMulQ8_0MmvqKernel"/> run on row c alone: each weight word is read once and dotted against every column in the same order.
/// </summary>
public sealed class MatMulQ8_0MmvqMultiKernel : IDisposable
{
    /// <summary>Largest supported column count.</summary>
    public const int MaxColumns = 8;

    private const int PushConstantBytes = 4 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline?[] _pipelines = new ComputePipeline?[MaxColumns + 1];
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ8_0MmvqMultiKernel(VulkanDevice device, VulkanModule module, nint pool, uint subgroup)
    {
        _device = device; _module = module; _descriptorPool = pool;
        for (int n = 2; n <= MaxColumns; n++)
            _pipelines[n] = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: [new VkDescriptorBinding(0), new VkDescriptorBinding(1), new VkDescriptorBinding(2), new VkDescriptorBinding(3)],
                pushConstantBytes: PushConstantBytes,
                requiredSubgroupSize: subgroup,
                specConstants: [(uint)n]);
        _descriptorCache = new DescriptorSetCache(device, pool, _pipelines[2]!, buffersPerSet: 4);
    }

    /// <summary>Loads <c>matmul_q8_0_mmvq_multi.spv</c>; null when it is missing or the device has no integer dot product.</summary>
    public static MatMulQ8_0MmvqMultiKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct) return null;
        string path = Path.Combine(spvDir, "matmul_q8_0_mmvq_multi.spv");
        if (!File.Exists(path)) return null;
        var module = VulkanModule.LoadFromFile(device, path);
        try
        {
            nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
            return new MatMulQ8_0MmvqMultiKernel(device, module, pool, Wave32SubgroupControl.RequiredSubgroupSizeFor(device));
        }
        catch { module.Dispose(); throw; }
    }

    /// <summary>True when this kernel serves <paramref name="n"/> rows of contraction <paramref name="k"/>.</summary>
    public static bool Accepts(int n, int k) => n >= 2 && n <= MaxColumns && (k % 32) == 0;

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the multi-column MMVQ GEMV (output layout <c>[n, m]</c>).</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer y, int m, int k, int n)
    {
        if (!Accepts(n, k)) throw new ArgumentOutOfRangeException(nameof(n), $"n={n}, k={k}");
        int blocksPerRow = k / 32;
        long rowBytes = (long)blocksPerRow * QuantFormat.Q8_0BlockBytes;
        if (weightsQ8.Size < (long)m * rowBytes) throw new ArgumentException("Weights buffer too small.", nameof(weightsQ8));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(n, k)) throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(n, k)) throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (y.Size < (long)n * m * sizeof(float)) throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[4] { weightsQ8.Handle, xq.Handle, xds.Handle, y.Handle };
        nint set = _descriptorCache.GetOrCreate(buffers);
        var pipe = _pipelines[n]!;
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipe.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, pipe.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[4] { (uint)m, (uint)k, (uint)blocksPerRow, (uint)((rowBytes + 3) / 4) };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, pipe.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptorPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        foreach (var p in _pipelines) p?.Dispose();
        _module.Dispose();
    }
}
