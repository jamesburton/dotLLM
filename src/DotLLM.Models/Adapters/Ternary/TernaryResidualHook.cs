using System.Runtime.InteropServices;
using DotLLM.Core.Diagnostics;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Adapters.Ternary;

/// <summary>In-memory storage of the per-row scales of a ternary adapter (#1620, #1107).</summary>
public enum TernaryScaleFormat
{
    /// <summary>IEEE binary32, exactly as in the package (bit-identical to the reference).</summary>
    F32 = 0,
    /// <summary>IEEE binary16, rounded to nearest even at load; used only when every non-zero |scale| is in the F16 normal range.</summary>
    F16 = 1,
}

/// <summary>
/// Loads and injects ternary residual adapters (<see cref="NsAdapterPackage"/>) after transformer layers
/// (<see cref="HookPoint.PostLayer"/>). Implemented on <see cref="IResidualModifier"/> rather than
/// <see cref="IInferenceHook"/>: that interface can only replace an activation by returning a freshly allocated
/// <c>float[]</c>, whereas the runner mutates the residual rows in place and allocates nothing per token
/// (see docs/DIAGNOSTICS.md). Several adapters may be registered, also at the same layer: circuits of one layer share
/// one int8 quantization of the residual and are evaluated in one pass in registration order (the rank-concatenation
/// pattern of <c>LoraComposer</c>, but per-adapter ReLU/int8 requantization and per-row F32 scales are preserved,
/// which a single concatenated LoRA-style matrix cannot represent). Not thread-safe: one forward thread.
/// </summary>
public sealed unsafe class TernaryResidualHook : IResidualModifier, IDisposable
{
    /// <summary>Largest summed rank accepted at one layer (mirrors <c>CudaForwardState.MaxLoraRank</c>).</summary>
    public const int MaxRankPerLayer = 256;

    private readonly int _d;
    private readonly List<(int layer, TernaryCircuit circuit)> _circuits = new();
    private TernaryCircuit*[] _byLayerPtr = [];
    private int[] _byLayerCount = [];
    private int[] _byLayerRank = [];
    private nint _circuitMem;
    private sbyte* _x8, _q8;
    private float* _f;
    private bool _disposed;

    /// <summary>Creates an empty hook for residual width <paramref name="dModel"/>.</summary>
    public TernaryResidualHook(int dModel)
    {
        if (dModel <= 0) throw new ArgumentOutOfRangeException(nameof(dModel));
        _d = dModel;
        Rebuild();
    }

    /// <summary>
    /// How the per-row scales of adapters registered AFTER this is set are held in memory. <see cref="TernaryScaleFormat.F32"/> (default) is
    /// bit-identical to the reference. <see cref="TernaryScaleFormat.F16"/> halves the scale metadata (RNE rounding at load; measured sufficient in
    /// docs/TERNARY_ADAPTER.md) but only for circuits whose every non-zero |scale| lies in the F16 normal range
    /// [<see cref="MinF16Scale"/>, <see cref="MaxF16Scale"/>]; any other circuit silently stays F32 (see <see cref="ScaleFormatAt"/>).
    /// </summary>
    public TernaryScaleFormat ScaleFormat { get; set; } = TernaryScaleFormat.F32;

    /// <summary>Smallest |scale| held in F16 (smallest normal binary16, 2^-14); below it F16 is subnormal and loses precision.</summary>
    public const float MinF16Scale = 6.103515625e-5f;
    /// <summary>Largest |scale| held in F16 (65504); above it F16 overflows.</summary>
    public const float MaxF16Scale = 65504f;

    /// <summary>The scale format actually used by circuit <paramref name="index"/> (registration order).</summary>
    public TernaryScaleFormat ScaleFormatAt(int index) => _circuits[index].circuit.DownScaleH is null ? TernaryScaleFormat.F32 : TernaryScaleFormat.F16;

    private static bool F16Representable(float[] a)
    {
        foreach (float s in a)
        {
            float m = Math.Abs(s);
            if (m == 0f) continue;
            if (!(m >= MinF16Scale && m <= MaxF16Scale)) return false;   // also rejects NaN/Inf
        }
        return true;
    }

    /// <summary>ISA override (tests); default widest available.</summary>
    public TernaryKernelIsa Isa { get; set; }

    /// <summary>Convenience: loads each package path and registers it (in order).</summary>
    public static TernaryResidualHook Load(IEnumerable<string> paths)
    {
        var pkgs = paths.Select(NsAdapterPackage.Load).ToList();
        if (pkgs.Count == 0) throw new ArgumentException("no adapter paths given");
        var hook = new TernaryResidualHook(pkgs[0].Manifest.InputDimension);
        try { foreach (var p in pkgs) hook.Add(p); }
        catch { hook.Dispose(); throw; }
        return hook;
    }

    /// <summary>Registers a package at its manifest <c>TargetLayer</c>. Weights are copied into the in-place bitmask layout (2 bits/weight).</summary>
    public void Add(NsAdapterPackage pkg)
    {
        ArgumentNullException.ThrowIfNull(pkg);
        pkg.ValidateDimensions();
        int layer = pkg.Manifest.TargetLayer, rank = pkg.Manifest.Rank;
        if (pkg.Manifest.InputDimension != _d)
            throw new InvalidOperationException($"adapter '{pkg.Manifest.Domain}' has D={pkg.Manifest.InputDimension}, hook is for D={_d}");
        int layerRank = _circuits.Where(c => c.layer == layer).Sum(c => c.circuit.Rank) + rank;
        if (layerRank > MaxRankPerLayer)
            throw new NotSupportedException($"summed ternary rank {layerRank} at layer {layer} exceeds {MaxRankPerLayer}");

        int wpr = TernaryAdapterKernel.WordsPerRow(_d);
        var ci = new TernaryCircuit
        {
            Rank = rank,
            Alpha = pkg.Manifest.Alpha,
            Threshold = pkg.Manifest.ActivationThreshold,
            Down = (ulong*)AllocZero((long)rank * wpr * 8),
            UpT = (ulong*)AllocZero((long)rank * wpr * 8),
            DownScale = (float*)AllocZero((long)rank * 4),
            UpScale = (float*)AllocZero((long)((_d + 7) & ~7) * 4),   // padded with zeros to a multiple of 8
        };
        for (int j = 0; j < rank; j++)
        {
            ci.DownScale[j] = pkg.Down.Scales[j];
            for (int c = 0; c < _d; c++)
            {
                sbyte w = pkg.Down[j, c];
                if (w != 0) ci.Down[(long)j * wpr + (c >> 6) * 2 + (w < 0 ? 1 : 0)] |= 1UL << (c & 63);
            }
        }
        for (int dRow = 0; dRow < _d; dRow++)
        {
            ci.UpScale[dRow] = pkg.Up.Scales[dRow];
            for (int j = 0; j < rank; j++)
            {
                sbyte w = pkg.Up[dRow, j];
                if (w != 0) ci.UpT[(long)j * wpr + (dRow >> 6) * 2 + (w < 0 ? 1 : 0)] |= 1UL << (dRow & 63);
            }
        }
        if (ScaleFormat == TernaryScaleFormat.F16 && F16Representable(pkg.Down.Scales) && F16Representable(pkg.Up.Scales))
        {
            ci.DownScaleH = (ushort*)AllocZero((long)rank * 2);
            ci.UpScaleH = (ushort*)AllocZero((long)((_d + 7) & ~7) * 2);
            for (int j = 0; j < rank; j++) ci.DownScaleH[j] = BitConverter.HalfToUInt16Bits((Half)pkg.Down.Scales[j]);
            for (int r = 0; r < _d; r++) ci.UpScaleH[r] = BitConverter.HalfToUInt16Bits((Half)pkg.Up.Scales[r]);
            NativeMemory.AlignedFree(ci.DownScale); NativeMemory.AlignedFree(ci.UpScale);
            ci.DownScale = null; ci.UpScale = null;
        }
        _circuits.Add((layer, ci));
        Rebuild();
    }

    /// <summary>Number of registered circuits.</summary>
    public int CircuitCount => _circuits.Count;

    /// <summary>Summed rank applied after <paramref name="layer"/>.</summary>
    public int RankAt(int layer) => (uint)layer < (uint)_byLayerRank.Length ? _byLayerRank[layer] : 0;

    /// <summary>Residual width.</summary>
    public int DModel => _d;

    /// <summary>Throws when the hook cannot attach to a model with this geometry.</summary>
    public void ValidateFor(int hiddenSize, int numLayers)
    {
        if (hiddenSize != _d) throw new InvalidOperationException($"ternary adapter D={_d} != model hidden size {hiddenSize}");
        if (_byLayerCount.Length > numLayers)
            throw new InvalidOperationException($"ternary adapter targets layer {_byLayerCount.Length - 1}, model has {numLayers} layers");
    }

    /// <inheritdoc/>
    public bool AppliesTo(int layerIndex) => (uint)layerIndex < (uint)_byLayerCount.Length && _byLayerCount[layerIndex] != 0;

    /// <inheritdoc/>
    public void ModifyResidual(Span<float> residual, int layerIndex, int tokenCount)
    {
        if (!AppliesTo(layerIndex)) return;
        if (residual.Length < (long)tokenCount * _d) throw new ArgumentException("residual shorter than tokenCount * d_model", nameof(residual));
        TernaryCircuit* cs = _byLayerPtr[layerIndex];
        int n = _byLayerCount[layerIndex];
        fixed (float* p = residual)
            for (int t = 0; t < tokenCount; t++)
                TernaryAdapterKernel.ApplyRow(p + (long)t * _d, _d, cs, n, _x8, _q8, _f, Isa);
    }

    private static void* AllocZero(long bytes)
    {
        void* p = NativeMemory.AlignedAlloc((nuint)Math.Max(bytes, 64), 64);
        NativeMemory.Clear(p, (nuint)Math.Max(bytes, 64));
        return p;
    }

    private void Rebuild()
    {
        if (_circuitMem != 0) NativeMemory.Free((void*)_circuitMem);
        if (_x8 is not null) NativeMemory.AlignedFree(_x8);
        if (_q8 is not null) NativeMemory.AlignedFree(_q8);
        if (_f is not null) NativeMemory.AlignedFree(_f);
        int maxLayer = _circuits.Count == 0 ? -1 : _circuits.Max(c => c.layer);
        _byLayerPtr = new TernaryCircuit*[maxLayer + 1];
        _byLayerCount = new int[maxLayer + 1];
        _byLayerRank = new int[maxLayer + 1];
        _circuitMem = (nint)NativeMemory.Alloc((nuint)Math.Max(_circuits.Count, 1), (nuint)sizeof(TernaryCircuit));
        TernaryCircuit* cur = (TernaryCircuit*)_circuitMem;
        int maxRank = 1, maxCount = 1;
        for (int layer = 0; layer <= maxLayer; layer++)
        {
            int start = (int)(cur - (TernaryCircuit*)_circuitMem);
            int count = 0, rank = 0;
            foreach (var (l, c) in _circuits)
            {
                if (l != layer) continue;
                *cur++ = c; count++; rank += c.Rank;
            }
            _byLayerPtr[layer] = (TernaryCircuit*)_circuitMem + start;
            _byLayerCount[layer] = count;
            _byLayerRank[layer] = rank;
            maxRank = Math.Max(maxRank, rank); maxCount = Math.Max(maxCount, count);
        }
        _x8 = (sbyte*)AllocZero(TernaryAdapterKernel.PaddedCols(_d));
        _q8 = (sbyte*)AllocZero(maxRank);
        _f = (float*)AllocZero((long)(maxRank + maxCount) * 4);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var (_, c) in _circuits)
        {
            NativeMemory.AlignedFree(c.Down); NativeMemory.AlignedFree(c.UpT);
            NativeMemory.AlignedFree(c.DownScale); NativeMemory.AlignedFree(c.UpScale);
            NativeMemory.AlignedFree(c.DownScaleH); NativeMemory.AlignedFree(c.UpScaleH);
        }
        _circuits.Clear();
        if (_circuitMem != 0) NativeMemory.Free((void*)_circuitMem);
        NativeMemory.AlignedFree(_x8); NativeMemory.AlignedFree(_q8); NativeMemory.AlignedFree(_f);
    }
}
