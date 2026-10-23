using DotLLM.Core.Models;

namespace DotLLM.Core.Attention;

/// <summary>
/// Per-layer KV-cache geometry — the width (in elements) of one cached K (or V)
/// row for each transformer layer.
/// </summary>
/// <remarks>
/// For every dense / GQA / MoE model this is <b>uniform</b>: one stride
/// (<c>numKvHeads * headDim</c>) repeated across all layers. Gemma-4 is the
/// exception — its sliding-window and global (full-attention) layers carry
/// different KV-head counts and head dims, so each layer's cached K/V row is a
/// different width (e.g. sliding <c>8*256</c> vs global <c>2*512</c>). A KV cache
/// that assumes a single scalar stride mis-addresses one of the two layer classes.
/// <para>
/// The <see cref="IsUniform"/> / <see cref="UniformStride"/> fast path lets hot
/// loops keep a single scalar local and run the exact same offset arithmetic as a
/// scalar cache when the model is uniform (≈ every model except Gemma-4), so the
/// generalisation is byte-identical for non-Gemma architectures.
/// </para>
/// </remarks>
public readonly struct KvGeometry
{
    private readonly int[] _kvStridePerLayer;
    private readonly int[]? _slotForLayer;

    private KvGeometry(int[] kvStridePerLayer, bool isUniform, int uniformStride, int[]? slotForLayer = null)
    {
        _kvStridePerLayer = kvStridePerLayer;
        IsUniform = isUniform;
        UniformStride = uniformStride;
        _slotForLayer = slotForLayer;
    }

    /// <summary>
    /// Number of KV <b>slots</b> (buffers) this geometry describes. Equals the model's layer count for every dense / GQA / MoE
    /// model; for a hybrid (GDN / SSM + attention) model it is the number of <em>attention</em> layers only
    /// (see <see cref="HasLayerSlotMap"/>, <see cref="SlotOfLayer"/>).
    /// </summary>
    public int LayerCount => _kvStridePerLayer.Length;

    /// <summary>
    /// True when the slots are a compacted subset of the model's layers (hybrid models: slots only for attention layers).
    /// <see cref="KvStrideOf"/> and every cache indexer then take the <b>slot</b>, not the model layer index.
    /// </summary>
    public bool HasLayerSlotMap => _slotForLayer is not null;

    /// <summary>Number of model layers the slot map spans (== <see cref="LayerCount"/> when there is no slot map).</summary>
    public int ModelLayerCount => _slotForLayer?.Length ?? _kvStridePerLayer.Length;

    /// <summary>
    /// The cache slot holding model layer <paramref name="modelLayer"/>'s K/V, or -1 if that layer has none (a recurrent
    /// layer of a hybrid). Identity when there is no slot map.
    /// </summary>
    public int SlotOfLayer(int modelLayer) => _slotForLayer is null ? modelLayer : _slotForLayer[modelLayer];

    /// <summary>
    /// True when every layer shares the same KV row width. Hot paths may then use
    /// <see cref="UniformStride"/> directly instead of indexing per layer.
    /// </summary>
    public bool IsUniform { get; }

    /// <summary>
    /// The shared per-layer KV row width when <see cref="IsUniform"/> is true
    /// (also equals <c>KvStrideOf(0)</c>); 0 when the geometry is non-uniform.
    /// </summary>
    public int UniformStride { get; }

    /// <summary>The cached K/V row width (in elements) for <paramref name="layer"/>.</summary>
    public int KvStrideOf(int layer) => _kvStridePerLayer[layer];

    /// <summary>
    /// Builds a uniform geometry: <paramref name="numKvHeads"/> * <paramref name="headDim"/>
    /// repeated across <paramref name="numLayers"/> layers — the byte-identical
    /// equivalent of a scalar <c>_kvStride</c>.
    /// </summary>
    public static KvGeometry Uniform(int numLayers, int numKvHeads, int headDim)
    {
        if (numLayers <= 0)
            throw new System.ArgumentOutOfRangeException(nameof(numLayers), numLayers, "numLayers must be positive.");
        if (numKvHeads <= 0)
            throw new System.ArgumentOutOfRangeException(nameof(numKvHeads), numKvHeads, "numKvHeads must be positive.");
        if (headDim <= 0)
            throw new System.ArgumentOutOfRangeException(nameof(headDim), headDim, "headDim must be positive.");

        int stride = numKvHeads * headDim;
        var strides = new int[numLayers];
        System.Array.Fill(strides, stride);
        return new KvGeometry(strides, isUniform: true, uniformStride: stride);
    }

    /// <summary>
    /// Builds a geometry from explicit per-layer KV row widths. The array is copied;
    /// uniformity is detected automatically (so a degenerate all-equal array still
    /// takes the <see cref="IsUniform"/> fast path).
    /// </summary>
    public static KvGeometry PerLayer(int[] kvStridePerLayer)
    {
        System.ArgumentNullException.ThrowIfNull(kvStridePerLayer);
        if (kvStridePerLayer.Length == 0)
            throw new System.ArgumentException("At least one layer stride is required.", nameof(kvStridePerLayer));

        int first = kvStridePerLayer[0];
        bool uniform = true;
        for (int i = 0; i < kvStridePerLayer.Length; i++)
        {
            int s = kvStridePerLayer[i];
            if (s <= 0)
                throw new System.ArgumentOutOfRangeException(nameof(kvStridePerLayer), s, "Each layer stride must be positive.");
            if (s != first)
                uniform = false;
        }

        var copy = (int[])kvStridePerLayer.Clone();
        return new KvGeometry(copy, uniform, uniform ? first : 0);
    }

    /// <summary>
    /// Number of KV slots a cache for <paramref name="config"/> needs: <c>NumLayers</c> for non-hybrid models, the number of
    /// <see cref="HybridLayerKind.Attention"/> layers (min 1) for a model with a <see cref="ModelConfig.HybridLayout"/>.
    /// Allocation-free; equals <c>FromConfig(config).LayerCount</c>.
    /// </summary>
    public static int SlotCount(ModelConfig config)
    {
        System.ArgumentNullException.ThrowIfNull(config);
        if (config.HybridLayout is not { } layout)
            return config.NumLayers;
        int n = 0;
        int limit = System.Math.Min(config.NumLayers, layout.LayerKind.Length);
        for (int l = 0; l < limit; l++)
            if (layout.LayerKind[l] == HybridLayerKind.Attention) n++;
        return n > 0 ? n : 1;
    }

    /// <summary>
    /// Derives the KV geometry for <paramref name="config"/>: each layer's stride is
    /// <c>GetLayerKvHeads(l) * GetLayerHeadDim(l)</c>. For a hybrid model (<see cref="ModelConfig.HybridLayout"/> set) only the
    /// <see cref="HybridLayerKind.Attention"/> layers get a slot, in layer order (slot k = k-th attention layer, the
    /// <c>kvSlotForLayer</c> convention every hybrid forward pass already uses), and <see cref="HasLayerSlotMap"/> is true. Returns a uniform geometry for
    /// every non-Gemma-4 model (where both resolve to the model-wide defaults), so the
    /// scalar addressing path is preserved. This is the single helper every backend
    /// cache factory should call instead of re-deriving per-layer strides.
    /// </summary>
    public static KvGeometry FromConfig(ModelConfig config)
    {
        System.ArgumentNullException.ThrowIfNull(config);
        int n = config.NumLayers;
        if (config.HybridLayout is { } layout)
        {
            int limit = System.Math.Min(n, layout.LayerKind.Length);
            var map = new int[n];
            var slotStrides = new System.Collections.Generic.List<int>();
            for (int l = 0; l < n; l++)
            {
                if (l < limit && layout.LayerKind[l] == HybridLayerKind.Attention)
                {
                    map[l] = slotStrides.Count;
                    slotStrides.Add(config.GetLayerKvHeads(l) * config.GetLayerHeadDim(l));
                }
                else map[l] = -1;
            }
            if (slotStrides.Count == 0)
            {
                // No attention layer at all: keep one minimal slot so the cache object is constructible.
                slotStrides.Add(config.GetLayerKvHeads(0) * config.GetLayerHeadDim(0));
            }
            var g = PerLayer(slotStrides.ToArray());
            return new KvGeometry(g._kvStridePerLayer, g.IsUniform, g.UniformStride, map);
        }
        var strides = new int[n];
        for (int l = 0; l < n; l++)
            strides[l] = config.GetLayerKvHeads(l) * config.GetLayerHeadDim(l);
        return PerLayer(strides);
    }
}
