using System.Numerics.Tensors;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;

namespace DotLLM.ChunkReuse.Selectors;

/// <summary>
/// CacheBlend's runnable deviation proxy: the K/V deviation at one early layer (default layer 1) between the reused
/// (isolated-chunk, RoPE-shifted) rows and freshly computed rows. Layer 0's K/V depend only on token and position, so they
/// are exact after the shift; the first layer that sees cross-chunk attention is layer 1. The fresh rows come from running
/// layers <c>[0, layer]</c> over the whole context on a separate scratch cache (C4: <c>ForwardLayerWindow</c> with a cache), so
/// the fresh layer-1 K/V equal the full-prefill layer-1 K/V, and the stale cache is never modified (no free fix-up for the
/// selector that uses this). Cost: <c>(layer + 1) / NumLayers</c> of a full prefill.
/// </summary>
public sealed class Layer1KvDeviationSource : ITokenScoreSource
{
    private readonly TransformerModel _model;
    private readonly SimpleKvCache _stale;
    private readonly int _layer;
    private readonly int _block;
    private ChunkedContext? _lastContext;
    private float[]? _lastScores;

    /// <summary>Binds the source to a model and the stale (naive-concatenation) cache of one context.</summary>
    /// <param name="model">The CPU model that encoded the chunks.</param>
    /// <param name="staleCache">The assembled naive-concatenation cache, holding the context at <c>[0, L)</c>. Read only.</param>
    /// <param name="layer">Layer whose K/V deviation is measured; at least 1.</param>
    /// <param name="blockTokens">Prefill block size for the probe.</param>
    public Layer1KvDeviationSource(TransformerModel model, SimpleKvCache staleCache, int layer = 1, int blockTokens = 512)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(staleCache);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layer);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(layer, model.Config.NumLayers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockTokens);
        _model = model;
        _stale = staleCache;
        _layer = layer;
        _block = blockTokens;
    }

    /// <summary>Per token: ‖(K, V)<sub>fresh</sub> − (K, V)<sub>stale</sub>‖₂ at the probe layer. Memoised per context instance.</summary>
    public float[] Score(ChunkedContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (ReferenceEquals(context, _lastContext)) return _lastScores!;
        int n = context.Length;
        if (_stale.CurrentLength != n)
            throw new ArgumentException($"The stale cache holds {_stale.CurrentLength} positions; the context has {n}.", nameof(context));

        ModelConfig config = _model.Config;
        int[] ids = context.TokenIds;
        using var fresh = new SimpleKvCache(KvGeometry.FromConfig(config, compactSlidingWindows: false), n);
        float[] hidden = new float[(long)Math.Min(_block, n) * config.HiddenSize];
        for (int off = 0; off < n; off += _block)
        {
            int len = Math.Min(_block, n - off);
            int[] positions = new int[len];
            for (int i = 0; i < len; i++) positions[i] = off + i;
            _model.ForwardLayerWindow(ids.AsSpan(off, len), ReadOnlySpan<float>.Empty, positions, 0, _layer + 1,
                hidden.AsSpan(0, len * config.HiddenSize), fresh);
        }

        _lastScores = RowDeviation(fresh, _stale, n, [_layer]);
        _lastContext = context;
        return _lastScores;
    }

    /// <summary>‖(K,V)<sub>a</sub> − (K,V)<sub>b</sub>‖₂ per row over <paramref name="layers"/> (double accumulation).</summary>
    internal static float[] RowDeviation(SimpleKvCache a, SimpleKvCache b, int rows, IEnumerable<int> layers)
    {
        var sq = new double[rows];
        foreach (int l in layers)
        {
            int stride = a.KvStrideOf(l);
            if (b.KvStrideOf(l) != stride)
                throw new ArgumentException($"Layer {l}: caches have different KV strides.");
            ReadOnlySpan<float> ka = a.KeysSpan(l), kb = b.KeysSpan(l), va = a.ValuesSpan(l), vb = b.ValuesSpan(l);
            for (int r = 0; r < rows; r++)
            {
                int o = r * stride;
                double s = 0;
                for (int i = 0; i < stride; i++)
                {
                    double dk = ka[o + i] - kb[o + i], dv = va[o + i] - vb[o + i];
                    s += dk * dk + dv * dv;
                }
                sq[r] += s;
            }
        }
        var dev = new float[rows];
        for (int r = 0; r < rows; r++) dev[r] = (float)Math.Sqrt(sq[r]);
        return dev;
    }
}

/// <summary>
/// Upper-bound deviation: ‖(K, V)<sub>full prefill</sub> − (K, V)<sub>reused</sub>‖₂ per token summed (in squares) over every
/// layer. It needs the full-prefill K/V — the very computation chunk reuse avoids — so it is NOT runnable at serving time; it
/// shows how far a deviation-ranked selector could go with perfect deviation information (it is not the optimal selection).
/// </summary>
public sealed class OracleKvDeviationSource : ITokenScoreSource
{
    private readonly SimpleKvCache _reference;
    private readonly SimpleKvCache _stale;
    private ChunkedContext? _lastContext;
    private float[]? _lastScores;

    /// <summary>Binds the source to the full-prefill and the naive-concatenation caches of one context.</summary>
    /// <param name="referenceCache">Full prefill of the context at <c>[0, L)</c>.</param>
    /// <param name="staleCache">Naive concatenation of the same context.</param>
    public OracleKvDeviationSource(SimpleKvCache referenceCache, SimpleKvCache staleCache)
    {
        ArgumentNullException.ThrowIfNull(referenceCache);
        ArgumentNullException.ThrowIfNull(staleCache);
        _reference = referenceCache;
        _stale = staleCache;
    }

    /// <inheritdoc/>
    public float[] Score(ChunkedContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (ReferenceEquals(context, _lastContext)) return _lastScores!;
        int n = context.Length;
        if (_reference.CurrentLength < n || _stale.CurrentLength < n)
            throw new ArgumentException($"Both caches must hold the {n}-token context.", nameof(context));
        _lastScores = Layer1KvDeviationSource.RowDeviation(_reference, _stale, n, Enumerable.Range(0, _reference.NumLayers));
        _lastContext = context;
        return _lastScores;
    }
}

/// <summary>
/// ProphetKV Stage I (arXiv 2602.02579): the query tokens are forwarded after the context over the reused (stale) K/V, and at
/// every layer the attention probabilities of each query row are read back (C7: <c>TransformerModel.AttentionQueryProbe</c>,
/// post-RoPE Q + the cache's K, same scale and causal mask as the model). Score of context token <c>j</c> = mean over layers,
/// query rows and heads of <c>p(j)</c>. The query rows' K/V written during the probe are dropped (<c>Rollback(L)</c>), so the
/// stale cache's context rows are unchanged.
/// </summary>
public sealed class QueryAttentionSource : ITokenScoreSource
{
    private readonly TransformerModel _model;
    private readonly SimpleKvCache _stale;
    private ChunkedContext? _lastContext;
    private float[]? _lastScores;

    /// <summary>Binds the source to a model and the stale cache of one context.</summary>
    /// <param name="model">The CPU model.</param>
    /// <param name="staleCache">The naive-concatenation cache holding the context at <c>[0, L)</c>, with room for the query.</param>
    public QueryAttentionSource(TransformerModel model, SimpleKvCache staleCache)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(staleCache);
        if (model.Config.AttnLogitSoftcap is not null || model.Config.SlidingWindowSize is not null || model.Config.PerLayerSlidingWindow is not null)
            throw new NotSupportedException("Query attention scores support plain causal softmax attention only (no soft-cap, no sliding window).");
        _model = model;
        _stale = staleCache;
    }

    /// <inheritdoc/>
    public float[] Score(ChunkedContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (ReferenceEquals(context, _lastContext)) return _lastScores!;
        int n = context.Length;
        int[] query = context.QueryTokenIds;
        if (query.Length == 0)
            throw new ArgumentException("Query relevance needs query tokens.", nameof(context));
        if (_stale.CurrentLength != n)
            throw new ArgumentException($"The stale cache holds {_stale.CurrentLength} positions; the context has {n}.", nameof(context));
        if (_stale.MaxLength < n + query.Length)
            throw new ArgumentException($"The stale cache holds at most {_stale.MaxLength} positions; context + query need {n + query.Length}.");

        var acc = new double[n];
        int rows = 0;
        float[] probs = [];
        _model.AttentionQueryProbe = (int layer, ReadOnlySpan<float> q, ReadOnlySpan<float> attnOut, ReadOnlySpan<int> positions,
            int numHeads, int numKvHeads, int headDim, float scale, IKvCache cache) =>
        {
            ReadOnlySpan<float> keys = ((SimpleKvCache)cache).KeysSpan(layer);
            int qStride = numHeads * headDim;
            for (int r = 0; r < positions.Length; r++)
            {
                int visible = positions[r] + 1;
                if (probs.Length < numHeads * visible) probs = new float[numHeads * visible];
                QueryAttention.Probabilities(q.Slice(r * qStride, qStride), keys, visible, numHeads, numKvHeads, headDim, scale, probs);
                for (int h = 0; h < numHeads; h++)
                {
                    int o = h * visible;
                    for (int j = 0; j < n; j++) acc[j] += probs[o + j] / numHeads;
                }
                rows++;
            }
        };
        try
        {
            int[] positions = new int[query.Length];
            for (int i = 0; i < query.Length; i++) positions[i] = n + i;
            using (_model.Forward(query, positions, -1, _stale, lastTokenLogitsOnly: true)) { }
        }
        finally
        {
            _model.AttentionQueryProbe = null;
            _stale.Rollback(n);
        }

        var scores = new float[n];
        for (int j = 0; j < n; j++) scores[j] = (float)(acc[j] / rows);
        _lastScores = scores;
        _lastContext = context;
        return scores;
    }
}

/// <summary>Causal softmax attention probabilities of one query row over cached keys (GQA), as the CPU attention computes them.</summary>
public static class QueryAttention
{
    /// <summary>
    /// <c>probs[h, j] = softmax_j(scale · q_h · k_{j, h / (numHeads / numKvHeads)})</c> for <c>j &lt; visibleKeys</c>.
    /// </summary>
    /// <param name="q">One query row, <c>[numHeads × headDim]</c>, post-RoPE.</param>
    /// <param name="keys">Cached keys, <c>[≥ visibleKeys, numKvHeads × headDim]</c>.</param>
    /// <param name="visibleKeys">Keys the row may attend to (causal: its position + 1).</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">KV heads.</param>
    /// <param name="headDim">Head dimension.</param>
    /// <param name="scale">Score scale.</param>
    /// <param name="probs">Output <c>[numHeads × visibleKeys]</c> (row stride <paramref name="visibleKeys"/>).</param>
    public static void Probabilities(ReadOnlySpan<float> q, ReadOnlySpan<float> keys, int visibleKeys, int numHeads, int numKvHeads,
        int headDim, float scale, Span<float> probs)
    {
        if (numKvHeads <= 0 || numHeads % numKvHeads != 0)
            throw new ArgumentException($"numHeads {numHeads} is not a multiple of numKvHeads {numKvHeads}.");
        int group = numHeads / numKvHeads;
        int kvStride = numKvHeads * headDim;
        for (int h = 0; h < numHeads; h++)
        {
            ReadOnlySpan<float> qh = q.Slice(h * headDim, headDim);
            int kvOff = h / group * headDim;
            Span<float> p = probs.Slice(h * visibleKeys, visibleKeys);
            float max = float.NegativeInfinity;
            for (int j = 0; j < visibleKeys; j++)
            {
                float s = scale * TensorPrimitives.Dot(qh, keys.Slice(j * kvStride + kvOff, headDim));
                p[j] = s;
                if (s > max) max = s;
            }
            double sum = 0;
            for (int j = 0; j < visibleKeys; j++)
            {
                float e = MathF.Exp(p[j] - max);
                p[j] = e;
                sum += e;
            }
            float inv = (float)(1.0 / sum);
            for (int j = 0; j < visibleKeys; j++) p[j] *= inv;
        }
    }
}
