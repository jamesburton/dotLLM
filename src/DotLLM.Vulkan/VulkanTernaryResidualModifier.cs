using DotLLM.Models.Adapters.Ternary;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Device-resident ternary residual adapters (#1618): the Vulkan counterpart of <c>TernaryResidualHook</c>. Packages are
/// converted once into the in-place plus/minus bitmask layout (2 bits per weight, the same bytes the CPU hook uses) and
/// uploaded; the forward then records <see cref="TernaryAdapterF32Kernel"/> after each targeted layer, in place on the
/// device residual. Several adapters at one layer are concatenated into one pass (registration order, summed rank capped at
/// <see cref="TernaryAdapterF32Kernel.MaxRank"/>) exactly like the CPU composer. Scales stay metadata (F32 by default, F16 optional).
/// </summary>
public sealed class VulkanTernaryResidualModifier : IVulkanResidualModifier, IDisposable
{
    private sealed class LayerPack : IDisposable
    {
        public required VulkanDevice.Buffer Masks, Scales, Meta;
        public int NCirc, TotalRank, UpScaleBase, UpScaleStride;
        public void Dispose() { Masks.Dispose(); Scales.Dispose(); Meta.Dispose(); }
    }

    private readonly TernaryAdapterF32Kernel _kernel;
    private readonly Dictionary<int, LayerPack> _layers = new();
    private readonly int _d;
    private readonly TernaryScaleStorage _scaleStorage;
    private bool _disposed;

    private VulkanTernaryResidualModifier(TernaryAdapterF32Kernel kernel, int d, TernaryScaleStorage storage)
    {
        _kernel = kernel; _d = d; _scaleStorage = storage;
    }

    /// <summary>Residual width the adapters were built for.</summary>
    public int DModel => _d;

    /// <summary>Layers that carry a circuit (ascending).</summary>
    public IReadOnlyCollection<int> Layers => _layers.Keys.Order().ToArray();

    /// <summary>Summed rank applied after <paramref name="layer"/> (0 when none).</summary>
    public int RankAt(int layer) => _layers.TryGetValue(layer, out var p) ? p.TotalRank : 0;

    /// <summary>
    /// Builds the device packs for <paramref name="packages"/> (registration order). All packages must share one D;
    /// the summed rank per layer is capped at <see cref="TernaryAdapterF32Kernel.MaxRank"/>.
    /// </summary>
    public static VulkanTernaryResidualModifier Create(VulkanDevice device, string spvDir,
        IReadOnlyList<NsAdapterPackage> packages, TernaryScaleStorage scaleStorage = TernaryScaleStorage.F32)
    {
        ArgumentNullException.ThrowIfNull(packages);
        if (packages.Count == 0) throw new ArgumentException("no adapter packages given", nameof(packages));
        int d = packages[0].Manifest.InputDimension;
        if (d <= 0 || d > TernaryAdapterF32Kernel.MaxHiddenSize)
            throw new NotSupportedException($"ternary adapter D={d} exceeds the Vulkan kernel limit {TernaryAdapterF32Kernel.MaxHiddenSize}");
        foreach (var p in packages)
        {
            p.ValidateDimensions();
            if (p.Manifest.InputDimension != d)
                throw new InvalidOperationException($"adapter '{p.Manifest.Domain}' has D={p.Manifest.InputDimension}, expected D={d}");
        }

        var kernel = TernaryAdapterF32Kernel.Create(device, spvDir);
        var self = new VulkanTernaryResidualModifier(kernel, d, scaleStorage);
        try
        {
            foreach (var group in packages.GroupBy(p => p.Manifest.TargetLayer).OrderBy(g => g.Key))
            {
                var list = group.ToList();
                int total = list.Sum(p => p.Manifest.Rank);
                if (total > TernaryAdapterF32Kernel.MaxRank)
                    throw new NotSupportedException($"summed ternary rank {total} at layer {group.Key} exceeds {TernaryAdapterF32Kernel.MaxRank}");
                self._layers[group.Key] = BuildPack(device, list, d, total, scaleStorage);
            }
        }
        catch { self.Dispose(); throw; }
        return self;
    }

    private static LayerPack BuildPack(VulkanDevice device, List<NsAdapterPackage> list, int d, int total, TernaryScaleStorage storage)
    {
        var pk = TernaryDevicePack.Build(list, d, storage);
        VulkanDevice.Buffer? bm = null, bs = null, bt = null;
        try
        {
            bm = device.AllocateDeviceLocal(pk.Masks.Length);
            bs = device.AllocateDeviceLocal(pk.Scales.Length);
            bt = device.AllocateDeviceLocal(pk.Meta.Length);
            device.Upload(pk.Masks, bm);
            device.Upload(pk.Scales, bs);
            device.Upload(pk.Meta, bt);
            return new LayerPack
            {
                Masks = bm, Scales = bs, Meta = bt, NCirc = pk.Circuits, TotalRank = pk.TotalRank,
                UpScaleBase = pk.UpScaleBase, UpScaleStride = pk.UpScaleStride,
            };
        }
        catch { bm?.Dispose(); bs?.Dispose(); bt?.Dispose(); throw; }
    }

    /// <inheritdoc/>
    public bool AppliesTo(int layerIndex) => _layers.ContainsKey(layerIndex);

    /// <summary>Throws when the adapters cannot attach to a model with this geometry.</summary>
    public void ValidateFor(int hiddenSize, int numLayers)
    {
        if (hiddenSize != _d) throw new InvalidOperationException($"ternary adapter D={_d} != model hidden size {hiddenSize}");
        int max = _layers.Count == 0 ? -1 : _layers.Keys.Max();
        if (max >= numLayers) throw new InvalidOperationException($"ternary adapter targets layer {max}, model has {numLayers} layers");
    }

    /// <inheritdoc/>
    public void Record(nint cmdBuf, VulkanDevice.Buffer residual, int layerIndex, int tokenCount, int hiddenSize)
    {
        if (!_layers.TryGetValue(layerIndex, out var p)) return;
        if (hiddenSize != _d) throw new InvalidOperationException($"hidden size {hiddenSize} != adapter D={_d}");
        _kernel.Record(cmdBuf, residual, p.Masks, p.Scales, p.Meta, tokenCount, _d, p.NCirc, p.TotalRank,
            p.UpScaleBase, p.UpScaleStride, _scaleStorage);
    }

    /// <inheritdoc/>
    public void InvalidateDescriptorCache() => _kernel.InvalidateDescriptorCache();

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var p in _layers.Values) p.Dispose();
        _layers.Clear();
        _kernel.Dispose();
    }
}
