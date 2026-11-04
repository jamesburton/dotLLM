using System.Runtime.InteropServices;
using DotLLM.Core.Diagnostics;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Adapters.Ternary;

/// <summary>
/// One layer's ternary residual circuit: packed W_down (rank x d) and packed W_up transposed (rank x d),
/// plus a per-rank-row scale. Pointers are owned by the enclosing <see cref="TernaryResidualAdapter"/>.
/// </summary>
internal readonly unsafe struct TernaryLayerData
{
    public TernaryLayerData(int layer, int rank, ulong* down, ulong* up, float* rankAlpha)
    {
        Layer = layer; Rank = rank; Down = down; Up = up; RankAlpha = rankAlpha;
    }

    public int Layer { get; }
    public int Rank { get; }
    public ulong* Down { get; }
    public ulong* Up { get; }
    public float* RankAlpha { get; }
}

/// <summary>
/// Zero-allocation ternary residual adapter <c>h' = x + alpha * W_up * sigma(W_down * x)</c> applied after
/// selected transformer layers through <see cref="IResidualModifier"/> (see docs/TERNARY_ADAPTER.md).
/// Weights ({-1,0,+1}) are 2-bit packed bitmasks read in place (memory-mapped for <c>.nsadapter</c> files);
/// the per-token path allocates nothing (the rank-r intermediate lives in a preallocated native buffer).
/// Not thread-safe: one forward thread per adapter instance.
/// </summary>
public sealed unsafe class TernaryResidualAdapter : IResidualModifier, IDisposable
{
    private readonly TernaryLayerData[] _byLayer; // indexed by layer, Rank == 0 => not adapted
    private readonly List<IDisposable> _owners;
    private readonly List<nint> _nativeBlocks;
    private readonly float* _h;
    private readonly int _maxRank;
    private bool _disposed;

    internal TernaryResidualAdapter(string name, int dModel, bool relu, TernaryLayerData[] layers,
        List<IDisposable> owners, List<nint> nativeBlocks, TernaryKernelIsa isa = TernaryKernelIsa.Auto)
    {
        Name = name;
        DModel = dModel;
        Relu = relu;
        Isa = isa;
        _owners = owners;
        _nativeBlocks = nativeBlocks;
        int maxLayer = -1;
        foreach (var l in layers) maxLayer = Math.Max(maxLayer, l.Layer);
        _byLayer = new TernaryLayerData[maxLayer + 1];
        foreach (var l in layers)
        {
            if (_byLayer[l.Layer].Rank != 0)
                throw new InvalidDataException($"Duplicate layer {l.Layer} in ternary adapter '{name}'.");
            _byLayer[l.Layer] = l;
            _maxRank = Math.Max(_maxRank, l.Rank);
            TotalRank += l.Rank;
            LayerCount++;
        }
        _h = (float*)NativeMemory.AlignedAlloc((nuint)Math.Max(_maxRank, 1) * sizeof(float), 64);
    }

    /// <summary>Adapter name (manifest <c>name</c>).</summary>
    public string Name { get; }

    /// <summary>Residual width the adapter was synthesised for.</summary>
    public int DModel { get; }

    /// <summary>True for ReLU sigma, false for identity.</summary>
    public bool Relu { get; }

    /// <summary>Number of adapted layers.</summary>
    public int LayerCount { get; }

    /// <summary>Sum of ranks over all adapted layers.</summary>
    public int TotalRank { get; }

    /// <summary>Kernel ISA used (tests force one; default widest available).</summary>
    public TernaryKernelIsa Isa { get; set; }

    internal TernaryLayerData[] Layers => _byLayer;

    /// <summary>Rank applied after <paramref name="layer"/> (0 if the layer is not adapted).</summary>
    public int RankAt(int layer) => (uint)layer < (uint)_byLayer.Length ? _byLayer[layer].Rank : 0;

    /// <summary>Throws when this adapter cannot attach to a model with the given geometry.</summary>
    public void ValidateFor(int hiddenSize, int numLayers)
    {
        if (hiddenSize != DModel)
            throw new InvalidOperationException(
                $"Ternary adapter '{Name}' was built for d_model={DModel}, model hidden size is {hiddenSize}.");
        if (_byLayer.Length > numLayers)
            throw new InvalidOperationException(
                $"Ternary adapter '{Name}' targets layer {_byLayer.Length - 1}, model has {numLayers} layers.");
    }

    /// <inheritdoc/>
    public bool AppliesTo(int layerIndex) => (uint)layerIndex < (uint)_byLayer.Length && _byLayer[layerIndex].Rank != 0;

    /// <inheritdoc/>
    public void ModifyResidual(Span<float> residual, int layerIndex, int tokenCount)
    {
        if ((uint)layerIndex >= (uint)_byLayer.Length) return;
        ref readonly TernaryLayerData l = ref _byLayer[layerIndex];
        if (l.Rank == 0) return;
        if (residual.Length < (long)tokenCount * DModel)
            throw new ArgumentException("Residual span shorter than tokenCount * d_model.", nameof(residual));
        fixed (float* p = residual)
        {
            for (int t = 0; t < tokenCount; t++)
                TernaryAdapterKernel.ApplyRow(p + (long)t * DModel, DModel, l.Rank, l.Down, l.Up, l.RankAlpha, Relu, _h, Isa);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        NativeMemory.AlignedFree(_h);
        foreach (nint b in _nativeBlocks) NativeMemory.AlignedFree((void*)b);
        foreach (var o in _owners) o.Dispose();
    }
}
