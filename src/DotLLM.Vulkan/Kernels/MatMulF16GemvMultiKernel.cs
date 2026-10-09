using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// F16 multi-column GEMV (#876): <c>Y[N, M] = X[N, K] @ W_f16[M, K]^T</c> for <c>N</c> in 2..<see cref="MaxColumns"/>, K even.
/// One workgroup per output row streams the weight row once for all N activation rows (see <c>matmul_f16_gemv_multi.comp</c>);
/// the 64x32 tiled GEMM it replaces at these sizes ran thin projections (M = 4 inject, M = 48 alpha/beta, M = 512 router) on a handful
/// of workgroups.
/// </summary>
public sealed class MatMulF16GemvMultiKernel : IDisposable
{
    /// <summary>Largest supported column count.</summary>
    public const int MaxColumns = 8;

    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, pairsPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline?[] _pipelines = new ComputePipeline?[MaxColumns + 1];
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulF16GemvMultiKernel(VulkanDevice device, VulkanModule module, nint pool)
    {
        _device = device; _module = module; _descriptorPool = pool;
        for (int n = 2; n <= MaxColumns; n++)
            _pipelines[n] = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: [new VkDescriptorBinding(0), new VkDescriptorBinding(1), new VkDescriptorBinding(2)],
                pushConstantBytes: PushConstantBytes,
                specConstants: [(uint)n]);
        _descriptorCache = new DescriptorSetCache(device, pool, _pipelines[2]!, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_f16_gemv_multi.spv</c>; null when it is missing.</summary>
    public static MatMulF16GemvMultiKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_f16_gemv_multi.spv");
        if (!File.Exists(path)) return null;
        var module = VulkanModule.LoadFromFile(device, path);
        try
        {
            nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
            return new MatMulF16GemvMultiKernel(device, module, pool);
        }
        catch { module.Dispose(); throw; }
    }

    /// <summary>True when this kernel serves a matmul with <paramref name="n"/> rows and contraction <paramref name="k"/>.</summary>
    public static bool Accepts(int n, int k) => n >= 2 && n <= MaxColumns && (k & 1) == 0;

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the multi-column GEMV (output layout <c>[n, m]</c>).</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer weightsA, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC, int m, int k, int n)
    {
        if (!Accepts(n, k)) throw new ArgumentOutOfRangeException(nameof(n), $"n={n}, k={k}");
        if (weightsA.Size < (long)m * k * 2) throw new ArgumentException("Weights buffer too small.", nameof(weightsA));
        if (inputB.Size < (long)n * k * sizeof(float)) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < (long)n * m * sizeof(float)) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[3] { weightsA.Handle, inputB.Handle, outputC.Handle };
        nint set = _descriptorCache.GetOrCreate(buffers);
        var pipe = _pipelines[n]!;
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipe.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, pipe.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[4] { (uint)m, (uint)k, (uint)(k / 2), (uint)(k / 2) };
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
