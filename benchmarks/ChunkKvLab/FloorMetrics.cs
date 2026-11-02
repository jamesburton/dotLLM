using DotLLM.ChunkReuse.Backends;
using DotLLM.ChunkReuse.Deviation;
using DotLLM.Core.Attention;
using DotLLM.Engine.KvCache;

namespace ChunkKvLab;

/// <summary>
/// Metric helpers of the <c>floor</c> command (Phase G Task 2): per-row relative L2 of K and V per layer between two caches of
/// any backend (read through <see cref="IChunkBackend.ReadRows"/> into <see cref="KvDeviation"/>), nearest-rank percentiles, and
/// the per-row logits comparison (KL, top-1, masked-call NLL and exact match) used for teacher-forced continuation blocks.
/// </summary>
public static class FloorMetrics
{
    /// <summary>Count, mean, nearest-rank 99th percentile and maximum of a set of values.</summary>
    public readonly record struct Summary(int Count, double Mean, double P99, double Max);

    /// <summary>
    /// Per-row comparison of a candidate's teacher-forced logits with the reference's: <see cref="Kl"/> = KL(P_ref ‖ P) per row
    /// (nats), <see cref="Top1Agree"/> per row, and over the first <c>maskCount</c> rows the summed NLL of the targets and the
    /// greedy exact match on each side.
    /// </summary>
    public sealed record LogitsComparison(double[] Kl, bool[] Top1Agree, double ReferenceMaskNll, double MaskNll, bool ReferenceMaskExact,
        bool MaskExact);

    /// <summary>
    /// Nearest-rank percentile: the value at rank <c>ceil(percent / 100 × n)</c> (1-based) of the ascending values; no
    /// interpolation. The input is not reordered.
    /// </summary>
    /// <param name="values">At least one value.</param>
    /// <param name="percent">In (0, 100].</param>
    public static double Percentile(ReadOnlySpan<float> values, double percent)
    {
        if (values.IsEmpty) throw new ArgumentException("No values.", nameof(values));
        if (!(percent > 0 && percent <= 100)) throw new ArgumentOutOfRangeException(nameof(percent), percent, "Must lie in (0, 100].");
        float[] sorted = values.ToArray();
        Array.Sort(sorted);
        int rank = (int)Math.Ceiling(percent / 100.0 * sorted.Length);
        return sorted[Math.Clamp(rank, 1, sorted.Length) - 1];
    }

    /// <summary>Count, mean (accumulated in double), nearest-rank p99 and maximum.</summary>
    public static Summary Summarize(ReadOnlySpan<float> values)
    {
        if (values.IsEmpty) throw new ArgumentException("No values.", nameof(values));
        double sum = 0, max = double.NegativeInfinity;
        foreach (float v in values)
        {
            sum += v;
            if (v > max) max = v;
        }
        return new Summary(values.Length, sum / values.Length, Percentile(values, 99), max);
    }

    /// <summary>
    /// Per-layer, per-row K and V deviation of <paramref name="cache"/> from <paramref name="reference"/> over rows <c>[0, rows)</c>,
    /// each read through its own backend (so a CPU cache can be compared with a Vulkan one). Relative values divide by the
    /// <b>reference</b> row norm (<see cref="KvDeviation"/>).
    /// </summary>
    public static KvDeviation Deviation(IChunkBackend referenceBackend, IKvCache reference, IChunkBackend backend, IKvCache cache, int rows,
        KvGeometry geometry)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        if (reference.CurrentLength < rows || cache.CurrentLength < rows)
            throw new ArgumentException($"Both caches must hold {rows} rows (reference {reference.CurrentLength}, other {cache.CurrentLength}).");
        return KvDeviation.Compute(geometry.LayerCount, rows, geometry.KvStrideOf,
            (l, n, k, v) => referenceBackend.ReadRows(reference, l, 0, n, k, v),
            (l, n, k, v) => backend.ReadRows(cache, l, 0, n, k, v));
    }

    /// <summary>K relative L2 (‖ΔK‖ / ‖K_ref‖) of every row of <paramref name="layer"/>.</summary>
    public static float[] KeyRelativeRows(KvDeviation d, int layer)
    {
        var r = new float[d.Tokens];
        for (int t = 0; t < r.Length; t++) r[t] = d.KeyRelative(layer, t);
        return r;
    }

    /// <summary>V relative L2 (‖ΔV‖ / ‖V_ref‖) of every row of <paramref name="layer"/>.</summary>
    public static float[] ValueRelativeRows(KvDeviation d, int layer)
    {
        var r = new float[d.Tokens];
        for (int t = 0; t < r.Length; t++) r[t] = d.ValueRelative(layer, t);
        return r;
    }

    /// <summary>
    /// Compares row-major <c>[targets.Length, vocab]</c> logits: per-row KL(ref ‖ rows) and top-1 agreement; NLL sum and greedy
    /// exact match of the first <paramref name="maskCount"/> targets on each side.
    /// </summary>
    public static LogitsComparison CompareLogits(ReadOnlySpan<float> reference, ReadOnlySpan<float> rows, int vocab, ReadOnlySpan<int> targets,
        int maskCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocab);
        int n = targets.Length;
        if (reference.Length != rows.Length || reference.Length != (long)n * vocab || n == 0)
            throw new ArgumentException($"Logits must be [targets {n} x vocab {vocab}] on both sides ({reference.Length} vs {rows.Length}).");
        ArgumentOutOfRangeException.ThrowIfNegative(maskCount);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(maskCount, n);

        var kl = new double[n];
        var agree = new bool[n];
        double refNll = 0, nll = 0;
        bool refEm = true, em = true;
        for (int j = 0; j < n; j++)
        {
            ReadOnlySpan<float> r = reference.Slice(j * vocab, vocab), c = rows.Slice(j * vocab, vocab);
            kl[j] = Metrics.KlDivergence(r, c);
            int argRef = Metrics.ArgMax(r), arg = Metrics.ArgMax(c);
            agree[j] = argRef == arg;
            if (j < maskCount)
            {
                refNll += Nll(r, targets[j]);
                nll += Nll(c, targets[j]);
                refEm &= argRef == targets[j];
                em &= arg == targets[j];
            }
        }
        return new LogitsComparison(kl, agree, refNll, nll, refEm, em);
    }

    private static double Nll(ReadOnlySpan<float> row, int target)
    {
        double max = double.NegativeInfinity;
        foreach (float v in row) if (v > max) max = v;
        double s = 0;
        foreach (float v in row) s += Math.Exp(v - max);
        return max + Math.Log(s) - row[target];
    }
}

/// <summary>Pools values under string keys (e.g. size|pair|layer|kind) and summarises each pool over all its rows.</summary>
public sealed class FloorAccumulator
{
    private readonly Dictionary<string, List<float>> _pools = new(StringComparer.Ordinal);

    /// <summary>The keys added so far.</summary>
    public IEnumerable<string> Keys => _pools.Keys;

    /// <summary>Appends <paramref name="values"/> to the pool of <paramref name="key"/>.</summary>
    public void Add(string key, ReadOnlySpan<float> values)
    {
        if (!_pools.TryGetValue(key, out List<float>? pool)) _pools[key] = pool = [];
        foreach (float v in values) pool.Add(v);
    }

    /// <summary>Summary over every value added under <paramref name="key"/>.</summary>
    /// <exception cref="KeyNotFoundException">Nothing was added under <paramref name="key"/>.</exception>
    public FloorMetrics.Summary Summary(string key) => FloorMetrics.Summarize(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_pools[key]));
}
