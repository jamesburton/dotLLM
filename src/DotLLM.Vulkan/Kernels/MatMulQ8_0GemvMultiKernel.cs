using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q8_0 multi-column GEMV with FP32 activations (#1185): <c>Y[N, M] = X[N, K] @ dequant(W_q8_0[M, K])^T</c> for <c>N</c> in
/// 2..<see cref="MaxColumns"/>, K a multiple of 32. The coalesced single-row Q8_0 GEMV with N accumulators (see
/// <c>matmul_q8_0_gemv_multi.comp</c>): each column reduces in the single-row kernel's order, so a K+1-row MTP verify reproduces the
/// 1-row decode's logits exactly, which the Q8_1-activation MMVQ multi kernel does not.
/// </summary>
public sealed class MatMulQ8_0GemvMultiKernel : IDisposable
{
    /// <summary>Largest supported column count.</summary>
    public const int MaxColumns = 8;

    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline?[] _pipelines = new ComputePipeline?[MaxColumns + 1];
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ8_0GemvMultiKernel(VulkanDevice device, VulkanModule module, nint pool)
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

    /// <summary>Loads <c>matmul_q8_0_gemv_multi.spv</c>; null when it is missing or the device lacks subgroup arithmetic.</summary>
    public static MatMulQ8_0GemvMultiKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_q8_0_gemv_multi.spv");
        if (!device.HasSubgroupArithmetic || !File.Exists(path)) return null;
        var module = VulkanModule.LoadFromFile(device, path);
        try
        {
            nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
            return new MatMulQ8_0GemvMultiKernel(device, module, pool);
        }
        catch { module.Dispose(); throw; }
    }

    /// <summary>True when this kernel serves a matmul with <paramref name="n"/> rows and contraction <paramref name="k"/>.</summary>
    public static bool Accepts(int n, int k) => n >= 2 && n <= MaxColumns && k > 0 && (k % 32) == 0;

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the multi-column GEMV (output layout <c>[n, m]</c>).</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer weightsA, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC, int m, int k, int n)
    {
        if (!Accepts(n, k)) throw new ArgumentOutOfRangeException(nameof(n), $"n={n}, k={k}");
        if (weightsA.Size < (long)m * (k / 32) * 34) throw new ArgumentException("Weights buffer too small.", nameof(weightsA));
        if (inputB.Size < (long)n * k * sizeof(float)) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < (long)n * m * sizeof(float)) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[3] { weightsA.Handle, inputB.Handle, outputC.Handle };
        nint set = _descriptorCache.GetOrCreate(buffers);
        var pipe = _pipelines[n]!;
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipe.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, pipe.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[4] { (uint)m, (uint)k, (uint)(k / 32), 0 };
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
