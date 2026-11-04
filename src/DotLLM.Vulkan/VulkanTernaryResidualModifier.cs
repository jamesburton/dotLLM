using System.Runtime.InteropServices;
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
    public static unsafe VulkanTernaryResidualModifier Create(VulkanDevice device, string spvDir,
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

    private static unsafe LayerPack BuildPack(VulkanDevice device, List<NsAdapterPackage> list, int d, int total, TernaryScaleStorage storage)
    {
        int wprU = TernaryAdapterF32Kernel.WordsPerRow(d);
        long maskBytes = 2L * total * wprU * sizeof(uint);
        int upBase = (total + 1) & ~1;
        int upStride = (d + 1) & ~1;
        long scaleElems = upBase + (long)list.Count * upStride;
        long scaleBytes = storage == TernaryScaleStorage.F16 ? scaleElems * 2 : scaleElems * 4;
        scaleBytes = (scaleBytes + 3) & ~3L;
        long metaBytes = (long)list.Count * 4 * sizeof(uint);

        void* maskMem = NativeMemory.AllocZeroed((nuint)maskBytes);
        void* scaleMem = NativeMemory.AllocZeroed((nuint)scaleBytes);
        void* metaMem = NativeMemory.AllocZeroed((nuint)metaBytes);
        try
        {
            uint* masks = (uint*)maskMem;
            uint* meta = (uint*)metaMem;
            uint* upRows = masks + (long)total * wprU;
            int off = 0;
            for (int c = 0; c < list.Count; c++)
            {
                var pkg = list[c];
                int rank = pkg.Manifest.Rank;
                meta[c * 4 + 0] = (uint)rank;
                meta[c * 4 + 1] = (uint)off;
                meta[c * 4 + 2] = BitConverter.SingleToUInt32Bits(pkg.Manifest.Alpha);
                meta[c * 4 + 3] = BitConverter.SingleToUInt32Bits(pkg.Manifest.ActivationThreshold);
                for (int j = 0; j < rank; j++)
                {
                    PutScale(scaleMem, storage, off + j, pkg.Down.Scales[j]);
                    uint* row = masks + (long)(off + j) * wprU;
                    for (int col = 0; col < d; col++)
                    {
                        sbyte w = pkg.Down[j, col];
                        if (w != 0) SetBit(row, col, w < 0);
                    }
                }
                for (int r = 0; r < d; r++)
                {
                    PutScale(scaleMem, storage, upBase + c * upStride + r, pkg.Up.Scales[r]);
                    for (int j = 0; j < rank; j++)
                    {
                        sbyte w = pkg.Up[r, j];
                        if (w != 0) SetBit(upRows + (long)(off + j) * wprU, r, w < 0);   // UpT row j = column j of W_up
                    }
                }
                off += rank;
            }

            VulkanDevice.Buffer? bm = null, bs = null, bt = null;
            try
            {
                bm = device.AllocateDeviceLocal(maskBytes);
                bs = device.AllocateDeviceLocal(scaleBytes);
                bt = device.AllocateDeviceLocal(metaBytes);
                device.Upload(new ReadOnlySpan<byte>(maskMem, (int)maskBytes), bm);
                device.Upload(new ReadOnlySpan<byte>(scaleMem, (int)scaleBytes), bs);
                device.Upload(new ReadOnlySpan<byte>(metaMem, (int)metaBytes), bt);
                return new LayerPack
                {
                    Masks = bm, Scales = bs, Meta = bt, NCirc = list.Count, TotalRank = total,
                    UpScaleBase = upBase, UpScaleStride = upStride,
                };
            }
            catch { bm?.Dispose(); bs?.Dispose(); bt?.Dispose(); throw; }
        }
        finally { NativeMemory.Free(maskMem); NativeMemory.Free(scaleMem); NativeMemory.Free(metaMem); }
    }

    private static unsafe void SetBit(uint* row, int col, bool minus)
    {
        // block of 64 columns = [plusLo, plusHi, minusLo, minusHi]
        int word = (col >> 6) * 4 + ((col >> 5) & 1) + (minus ? 2 : 0);
        row[word] |= 1u << (col & 31);
    }

    private static unsafe void PutScale(void* mem, TernaryScaleStorage storage, long idx, float v)
    {
        if (storage == TernaryScaleStorage.F16) ((Half*)mem)[idx] = (Half)v;
        else ((float*)mem)[idx] = v;
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
