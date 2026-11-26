using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// F16-output Q8_0 wide GEMV for the Edge0 latent D and U at 9..16 rows (#2004): <c>Y[N, M] (F16) = X[N, K] @ dequant(W_q8_0[M, K])^T</c>.
/// It is <see cref="MatMulQ8_0GemvMultiKernel"/>'s wide pipeline (the F32 version these shapes ran on before #1986's all-F16 latent) with the Edge0
/// storage rule applied: D reads the F32 normed activations, U reads the F16 <c>z</c>; both accumulate in F32 registers in the unchanged fixed
/// order and round ONCE at the store, so every output is the F32 kernel's value rounded to F16. No F32 scratch is written.
/// </summary>
public sealed class Edge0WideGemvKernel : IDisposable
{
    /// <summary>Smallest column count served (the 2..8 rows run on the MMVQ small-row kernel).</summary>
    public const int MinColumns = 9;
    /// <summary>Largest column count served.</summary>
    public const int MaxColumns = MatMulQ8_0GemvMultiKernel.WideMaxColumns;

    private const int PushConstantBytes = 4 * sizeof(uint);   // M, K, blocksPerRow, unused

    private readonly VulkanDevice _device;
    private readonly bool _f16Input;
    private readonly int _rowsPerWorkgroup;
    private readonly VulkanModule _module;
    private readonly ComputePipeline?[] _pipelines = new ComputePipeline?[MaxColumns + 1];
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private Edge0WideGemvKernel(VulkanDevice device, VulkanModule module, nint pool, bool f16Input, int nr)
    {
        _device = device; _module = module; _descriptorPool = pool; _f16Input = f16Input; _rowsPerWorkgroup = nr;
        for (int n = MinColumns; n <= MaxColumns; n++)
            _pipelines[n] = module.CreateComputePipeline("main", [new VkDescriptorBinding(0), new VkDescriptorBinding(1), new VkDescriptorBinding(2)],
                PushConstantBytes, specConstants: [(uint)n, (uint)nr]);
        _descriptorCache = new DescriptorSetCache(device, pool, _pipelines[MinColumns]!, buffersPerSet: 3);
    }

    /// <summary>Loads <c>edge0_gemv_wide_d.spv</c> (F32 in, F16 out) or <c>edge0_gemv_wide_u.spv</c> (F16 in, F16 out); null when missing or the device lacks 16-bit storage.</summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">SPIR-V directory.</param>
    /// <param name="f16Input">U variant (F16 activations) when true, D variant (F32 activations) when false.</param>
    /// <param name="rowsPerWorkgroup">Output rows per workgroup, 1..4 (the shader stores one lane per output, NR x 16 &lt;= 64).</param>
    public static Edge0WideGemvKernel? TryCreate(VulkanDevice device, string spvDir, bool f16Input, int rowsPerWorkgroup = 4)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(rowsPerWorkgroup, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(rowsPerWorkgroup, 4);
        string path = Path.Combine(spvDir, f16Input ? "edge0_gemv_wide_u.spv" : "edge0_gemv_wide_d.spv");
        if (!File.Exists(path)) return null;
        var module = VulkanModule.LoadFromFile(device, path);
        try
        {
            nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
            return new Edge0WideGemvKernel(device, module, pool, f16Input, rowsPerWorkgroup);
        }
        catch { module.Dispose(); throw; }
    }

    /// <summary>True when this kernel serves <paramref name="n"/> rows of contraction <paramref name="k"/>.</summary>
    public static bool Accepts(int n, int k) => n >= MinColumns && n <= MaxColumns && k > 0 && (k % 32) == 0;

    /// <summary>Drops cached descriptor sets (scratch buffers were re-allocated).</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records <c>Y[n, m] (F16) = X[n, k] @ W[m, k]^T</c>; X is F32 for D, F16 for U.</summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer weights, VulkanDevice.Buffer x, VulkanDevice.Buffer y, int m, int k, int n)
    {
        if (!Accepts(n, k)) throw new ArgumentOutOfRangeException(nameof(n), $"n={n}, k={k}");
        if (weights.Size < (long)m * (k / 32) * 34) throw new ArgumentException("Weights buffer too small.", nameof(weights));
        if (x.Size < (long)n * k * (_f16Input ? sizeof(ushort) : sizeof(float))) throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)n * m * sizeof(ushort)) throw new ArgumentException("Output (F16) buffer too small.", nameof(y));
        Span<nint> buffers = stackalloc nint[3] { weights.Handle, x.Handle, y.Handle };
        nint set = _descriptorCache.GetOrCreate(buffers);
        var pipe = _pipelines[n]!;
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipe.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, pipe.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[4] { (uint)m, (uint)k, (uint)(k / 32), 0 };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, pipe.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + _rowsPerWorkgroup - 1) / _rowsPerWorkgroup), 1, 1);
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
