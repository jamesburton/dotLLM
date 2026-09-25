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

    private KvGeometry(int[] kvStridePerLayer, bool isUniform, int uniformStride)
    {
        _kvStridePerLayer = kvStridePerLayer;
        IsUniform = isUniform;
        UniformStride = uniformStride;
    }

    /// <summary>Number of layers this geometry describes.</summary>
    public int LayerCount => _kvStridePerLayer.Length;

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
    /// Derives the KV geometry for <paramref name="config"/>: each layer's stride is
    /// <c>GetLayerKvHeads(l) * GetLayerHeadDim(l)</c>. Returns a uniform geometry for
    /// every non-Gemma-4 model (where both resolve to the model-wide defaults), so the
    /// scalar addressing path is preserved. This is the single helper every backend
    /// cache factory should call instead of re-deriving per-layer strides.
    /// </summary>
    public static KvGeometry FromConfig(ModelConfig config)
    {
        System.ArgumentNullException.ThrowIfNull(config);
        int n = config.NumLayers;
        var strides = new int[n];
        for (int l = 0; l < n; l++)
            strides[l] = config.GetLayerKvHeads(l) * config.GetLayerHeadDim(l);
        return PerLayer(strides);
    }
}
