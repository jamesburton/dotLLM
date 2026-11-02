using DotLLM.Engine.KvCache;

namespace DotLLM.ChunkReuse.Deviation;

/// <summary>
/// Per-layer, per-token K and V deviation between a reference cache (full prefill) and a reused cache (assembled chunk K/V
/// after RoPE position alignment, <see cref="ChunkAssembler"/>), over the rows <c>[0, Tokens)</c> (E3, Task 10).
/// <para>
/// For layer <c>l</c> and token <c>t</c>: <c>KeyL2 = ‖K_ref − K_reused‖₂</c> and <c>ValueL2 = ‖V_ref − V_reused‖₂</c> over the
/// whole KV row (all KV heads); the relative forms divide by the <b>reference</b> row norm (<c>‖K_ref‖₂</c>,
/// <c>‖V_ref‖₂</c>), because layers differ in scale. Combined = <c>‖(ΔK, ΔV)‖₂</c> and its relative form
/// <c>‖(ΔK, ΔV)‖₂ / ‖(K_ref, V_ref)‖₂</c>. Accumulation is in double, storage in float. A zero reference norm gives a relative
/// deviation of 0 when the rows are equal and +∞ otherwise.
/// </para>
/// </summary>
public sealed class KvDeviation
{
    private readonly float[] _dk, _dv, _nk, _nv; // [layer * Tokens + token]

    private KvDeviation(int layers, int tokens, float[] dk, float[] dv, float[] nk, float[] nv)
    {
        Layers = layers;
        Tokens = tokens;
        _dk = dk;
        _dv = dv;
        _nk = nk;
        _nv = nv;
    }

    /// <summary>Number of layers.</summary>
    public int Layers { get; }

    /// <summary>Number of token rows compared.</summary>
    public int Tokens { get; }

    /// <summary>‖K_ref − K_reused‖₂ of token <paramref name="token"/> at layer <paramref name="layer"/>.</summary>
    public float KeyL2(int layer, int token) => _dk[Index(layer, token)];

    /// <summary>‖V_ref − V_reused‖₂.</summary>
    public float ValueL2(int layer, int token) => _dv[Index(layer, token)];

    /// <summary>‖K_ref‖₂ (the reference row norm).</summary>
    public float KeyReferenceNorm(int layer, int token) => _nk[Index(layer, token)];

    /// <summary>‖V_ref‖₂.</summary>
    public float ValueReferenceNorm(int layer, int token) => _nv[Index(layer, token)];

    /// <summary><see cref="KeyL2"/> / <see cref="KeyReferenceNorm"/>.</summary>
    public float KeyRelative(int layer, int token) => Ratio(KeyL2(layer, token), KeyReferenceNorm(layer, token));

    /// <summary><see cref="ValueL2"/> / <see cref="ValueReferenceNorm"/>.</summary>
    public float ValueRelative(int layer, int token) => Ratio(ValueL2(layer, token), ValueReferenceNorm(layer, token));

    /// <summary>‖(ΔK, ΔV)‖₂.</summary>
    public float CombinedL2(int layer, int token) => Hypot(KeyL2(layer, token), ValueL2(layer, token));

    /// <summary>‖(ΔK, ΔV)‖₂ / ‖(K_ref, V_ref)‖₂.</summary>
    public float CombinedRelative(int layer, int token)
        => Ratio(CombinedL2(layer, token), Hypot(KeyReferenceNorm(layer, token), ValueReferenceNorm(layer, token)));

    /// <summary>
    /// Fills <paramref name="keys"/> and <paramref name="values"/> (each <c>rows × stride</c> floats) with rows <c>[0, rows)</c> of
    /// <paramref name="layer"/>, for <see cref="Compute(int, int, Func{int, int}, KvRowsReader, KvRowsReader)"/>.
    /// </summary>
    public delegate void KvRowsReader(int layer, int rows, Span<float> keys, Span<float> values);

    /// <summary>Compares rows <c>[0, <paramref name="rows"/>)</c> of every layer of two caches with the same geometry.</summary>
    /// <param name="reference">The full-prefill cache.</param>
    /// <param name="reused">The assembled (reused) cache.</param>
    /// <param name="rows">Rows to compare; both caches must hold at least this many.</param>
    /// <exception cref="ArgumentException">The caches differ in layers or strides, or hold fewer than <paramref name="rows"/> rows.</exception>
    public static KvDeviation Compute(SimpleKvCache reference, SimpleKvCache reused, int rows)
    {
        ArgumentNullException.ThrowIfNull(reference);
        ArgumentNullException.ThrowIfNull(reused);
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        if (reference.CurrentLength < rows || reused.CurrentLength < rows)
            throw new ArgumentException($"Both caches must hold {rows} rows (reference {reference.CurrentLength}, reused {reused.CurrentLength}).");
        if (reference.NumLayers != reused.NumLayers)
            throw new ArgumentException($"Layer counts differ ({reference.NumLayers} vs {reused.NumLayers}).");
        for (int l = 0; l < reference.NumLayers; l++)
            if (reused.KvStrideOf(l) != reference.KvStrideOf(l))
                throw new ArgumentException($"Layer {l}: KV strides differ ({reference.KvStrideOf(l)} vs {reused.KvStrideOf(l)}).");

        static KvRowsReader Of(SimpleKvCache c) => (layer, n, keys, values) =>
        {
            int count = n * c.KvStrideOf(layer);
            c.KeysSpan(layer)[..count].CopyTo(keys);
            c.ValuesSpan(layer)[..count].CopyTo(values);
        };
        return Compute(reference.NumLayers, rows, reference.KvStrideOf, Of(reference), Of(reused));
    }

    /// <summary>
    /// Compares rows <c>[0, <paramref name="rows"/>)</c> of every layer of two caches read through <paramref name="reference"/>
    /// and <paramref name="reused"/> (any backend; e.g. <c>IChunkBackend.ReadRows</c>). Same arithmetic as the
    /// <see cref="SimpleKvCache"/> overload; one layer is held in memory per side at a time.
    /// </summary>
    /// <param name="layers">Layer count.</param>
    /// <param name="rows">Rows compared per layer.</param>
    /// <param name="strideOf">KV row stride (floats) of a layer; the same for both sides.</param>
    /// <param name="reference">Reads the full-prefill (reference) rows.</param>
    /// <param name="reused">Reads the compared rows.</param>
    public static KvDeviation Compute(int layers, int rows, Func<int, int> strideOf, KvRowsReader reference, KvRowsReader reused)
    {
        ArgumentNullException.ThrowIfNull(strideOf);
        ArgumentNullException.ThrowIfNull(reference);
        ArgumentNullException.ThrowIfNull(reused);
        ArgumentOutOfRangeException.ThrowIfNegative(layers);
        ArgumentOutOfRangeException.ThrowIfNegative(rows);

        long size = (long)layers * rows;
        float[] dk = new float[size], dv = new float[size], nk = new float[size], nv = new float[size];
        float[] kr = [], ku = [], vr = [], vu = [];
        for (int l = 0; l < layers; l++)
        {
            int stride = strideOf(l);
            int count = rows * stride;
            if (kr.Length != count)
            {
                kr = new float[count];
                ku = new float[count];
                vr = new float[count];
                vu = new float[count];
            }
            reference(l, rows, kr, vr);
            reused(l, rows, ku, vu);
            for (int t = 0; t < rows; t++)
            {
                int o = t * stride;
                double sdk = 0, sdv = 0, snk = 0, snv = 0;
                for (int i = 0; i < stride; i++)
                {
                    double a = kr[o + i], b = ku[o + i], c = vr[o + i], d = vu[o + i];
                    sdk += (a - b) * (a - b);
                    sdv += (c - d) * (c - d);
                    snk += a * a;
                    snv += c * c;
                }
                long x = (long)l * rows + t;
                dk[x] = (float)Math.Sqrt(sdk);
                dv[x] = (float)Math.Sqrt(sdv);
                nk[x] = (float)Math.Sqrt(snk);
                nv[x] = (float)Math.Sqrt(snv);
            }
        }
        return new KvDeviation(layers, rows, dk, dv, nk, nv);
    }

    private int Index(int layer, int token)
    {
        if ((uint)layer >= (uint)Layers) throw new ArgumentOutOfRangeException(nameof(layer));
        if ((uint)token >= (uint)Tokens) throw new ArgumentOutOfRangeException(nameof(token));
        return layer * Tokens + token;
    }

    private static float Hypot(float a, float b) => (float)Math.Sqrt((double)a * a + (double)b * b);

    private static float Ratio(float deviation, float norm)
        => norm > 0 ? deviation / norm : deviation == 0 ? 0f : float.PositiveInfinity;
}
