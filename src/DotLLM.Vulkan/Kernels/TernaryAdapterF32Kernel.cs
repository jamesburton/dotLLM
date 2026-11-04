using DotLLM.Models.Adapters.Ternary;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// In-place ternary residual adapter (#1618) for one layer's circuits: <c>h' = h + sum_c dq_c * W_up_c * relu(W_down_c * x8)</c>
/// with the integer pipeline of the CPU reference (see <c>ternary_adapter_residual_f32.comp</c>). One workgroup per token row.
/// A pipeline is specialised per padded model width and scale storage (spec constants), cached for the kernel's lifetime.
/// </summary>
public sealed class TernaryAdapterF32Kernel : IDisposable
{
    /// <summary>Largest summed rank per layer (shared-memory strips are 256 wide; mirrors <c>TernaryResidualHook.MaxRankPerLayer</c>).</summary>
    public const int MaxRank = 256;

    /// <summary>Largest supported residual width (the shared int8 row plus the strips must fit the 32 KiB LDS every Vulkan device guarantees).</summary>
    public const int MaxHiddenSize = 6144;

    private const int PushConstantBytes = 8 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly nint _pool;
    private readonly Dictionary<(int dPad, TernaryScaleStorage scales), (ComputePipeline pipe, DescriptorSetCache cache)> _variants = new();
    private bool _disposed;

    private TernaryAdapterF32Kernel(VulkanDevice device, VulkanModule module, nint pool)
    {
        _device = device; _module = module; _pool = pool;
    }

    /// <summary>Loads <c>ternary_adapter_residual_f32.spv</c>; throws when it is missing.</summary>
    public static TernaryAdapterF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "ternary_adapter_residual_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.ps1.");
        var module = VulkanModule.LoadFromFile(device, path);
        try { return new TernaryAdapterF32Kernel(device, module, KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4)); }
        catch { module.Dispose(); throw; }
    }

    /// <summary>Words (<c>uint</c>) per packed mask row for width <paramref name="d"/>: <c>ceil(d/64) * 4</c> (plus/minus x lo/hi per 64 columns).</summary>
    public static int WordsPerRow(int d) => TernaryDevicePack.WordsPerRow(d);

    private (ComputePipeline pipe, DescriptorSetCache cache) Variant(int d, TernaryScaleStorage scales)
    {
        int dPad = (d + 63) & ~63;
        if (!_variants.TryGetValue((dPad, scales), out var v))
        {
            var pipe = _module.CreateComputePipeline(
                entryPoint: "main",
                bindings: [new VkDescriptorBinding(0), new VkDescriptorBinding(1), new VkDescriptorBinding(2), new VkDescriptorBinding(3)],
                pushConstantBytes: PushConstantBytes,
                specConstants: [(uint)scales, (uint)dPad]);
            v = (pipe, new DescriptorSetCache(_device, _pool, pipe, buffersPerSet: 4));
            _variants[(dPad, scales)] = v;
        }
        return v;
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache()
    {
        foreach (var v in _variants.Values) v.cache.Reset();
    }

    /// <summary>
    /// Records the in-place update of <paramref name="nRows"/> residual rows of width <paramref name="d"/>. Buffer layouts are
    /// documented in the shader header; <paramref name="upScaleBase"/>/<paramref name="upScaleStride"/> are scale-element indices.
    /// </summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer residual, VulkanDevice.Buffer masks, VulkanDevice.Buffer scales,
        VulkanDevice.Buffer meta, int nRows, int d, int nCirc, int totalRank, int upScaleBase, int upScaleStride,
        TernaryScaleStorage scaleStorage)
    {
        if (nRows <= 0) throw new ArgumentOutOfRangeException(nameof(nRows));
        if (d <= 0 || d > MaxHiddenSize) throw new ArgumentOutOfRangeException(nameof(d), $"d={d} (max {MaxHiddenSize})");
        if (nCirc <= 0 || totalRank < nCirc || totalRank > MaxRank) throw new ArgumentOutOfRangeException(nameof(totalRank));
        if (residual.Size < (long)nRows * d * sizeof(float)) throw new ArgumentException("residual buffer too small.", nameof(residual));

        var (pipe, cache) = Variant(d, scaleStorage);
        Span<nint> buffers = stackalloc nint[4] { residual.Handle, masks.Handle, scales.Handle, meta.Handle };
        nint set = cache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, pipe.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, pipe.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[8]
        {
            (uint)d, (uint)nRows, (uint)nCirc, (uint)totalRank, (uint)WordsPerRow(d), (uint)upScaleBase, (uint)upScaleStride, 127u,
        };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, pipe.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)nRows, 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var v in _variants.Values) v.pipe.Dispose();
        _variants.Clear();
        if (_pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
        _module.Dispose();
    }
}
