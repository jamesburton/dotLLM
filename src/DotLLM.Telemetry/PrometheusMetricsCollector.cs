using System.Collections.Concurrent;
using System.Diagnostics.Metrics;
using System.Globalization;
using System.Text;

namespace DotLLM.Telemetry;

/// <summary>
/// In-process Prometheus aggregator (#1344): a <see cref="MeterListener"/> over every <c>DotLLM.*</c>
/// <see cref="Meter"/> that folds measurements into counters, gauges and fixed-bucket histograms and renders them
/// in the Prometheus text exposition format (version 0.0.4). No OpenTelemetry exporter or HTTP stack is involved,
/// so a scrape endpoint costs nothing beyond this listener.
/// </summary>
/// <remarks>
/// <para>
/// Subscribing a listener is what flips <c>Instrument.Enabled</c> to true, so the engine's hot-path
/// <c>if (x.Enabled)</c> sites only do work once a collector exists (the server creates one only when
/// <c>/metrics</c> is enabled).
/// </para>
/// <para>
/// Name mapping: dots become underscores; a unit suffix is appended unless the name already carries it
/// (<c>ms</c> becomes <c>milliseconds</c>, <c>By</c> becomes <c>bytes</c>); monotonic counters get <c>_total</c>. Histogram buckets are chosen per unit
/// (see <see cref="BucketsFor"/>). Observable gauges keep their sentinel values (for example <c>-1</c> for
/// "provider not wired").
/// </para>
/// </remarks>
public sealed class PrometheusMetricsCollector : IDisposable
{
    private static readonly double[] s_msBuckets = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000, 120000];
    private static readonly double[] s_rateBuckets = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000];
    private static readonly double[] s_defaultBuckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10];

    internal enum Kind { Counter, Gauge, Histogram }

    private sealed class Family
    {
        public required string Name;
        public required string Help;
        public required Kind Kind;
        public required double[] Bounds;
        public readonly ConcurrentDictionary<string, Series> Series = new();
    }

    private sealed class Series
    {
        public required string Labels;          // already rendered: {a="b",c="d"} or ""
        public double Value;                    // counter / gauge
        public long[]? Buckets;                 // histogram, non-cumulative, length = bounds + 1
        public double Sum;
        public long Count;
        public readonly object Gate = new();
    }

    private readonly MeterListener _listener = new();
    private readonly ConcurrentDictionary<Instrument, Family> _families = new();
    private readonly Func<string, bool> _meterFilter;
    private bool _disposed;

    /// <summary>Creates and starts a collector over meters whose name starts with <c>DotLLM.</c>.</summary>
    public PrometheusMetricsCollector() : this(static n => n.StartsWith("DotLLM.", StringComparison.Ordinal)) { }

    /// <summary>Creates and starts a collector over the meters accepted by <paramref name="meterFilter"/>.</summary>
    /// <param name="meterFilter">Selects meters by name.</param>
    public PrometheusMetricsCollector(Func<string, bool> meterFilter)
    {
        _meterFilter = meterFilter;
        _listener.InstrumentPublished = (instrument, listener) =>
        {
            if (!_meterFilter(instrument.Meter.Name)) return;
            if (TryCreateFamily(instrument) is not { } family) return;
            _families[instrument] = family;
            listener.EnableMeasurementEvents(instrument);
        };
        _listener.SetMeasurementEventCallback<long>((i, v, t, _) => Record(i, v, t));
        _listener.SetMeasurementEventCallback<int>((i, v, t, _) => Record(i, v, t));
        _listener.SetMeasurementEventCallback<short>((i, v, t, _) => Record(i, v, t));
        _listener.SetMeasurementEventCallback<byte>((i, v, t, _) => Record(i, v, t));
        _listener.SetMeasurementEventCallback<double>((i, v, t, _) => Record(i, v, t));
        _listener.SetMeasurementEventCallback<float>((i, v, t, _) => Record(i, v, t));
        _listener.Start();
    }

    private static Family? TryCreateFamily(Instrument instrument)
    {
        Kind kind;
        if (instrument is Histogram<long> or Histogram<double> or Histogram<int> or Histogram<float>) kind = Kind.Histogram;
        else if (instrument.GetType().Name.Contains("UpDownCounter", StringComparison.Ordinal)) kind = Kind.Gauge;
        else if (instrument.IsObservable && instrument.GetType().Name.Contains("Gauge", StringComparison.Ordinal)) kind = Kind.Gauge;
        else if (instrument.GetType().Name.Contains("Counter", StringComparison.Ordinal)) kind = Kind.Counter;
        else if (instrument.GetType().Name.Contains("Gauge", StringComparison.Ordinal)) kind = Kind.Gauge;
        else return null;

        string name = MapName(instrument.Name, instrument.Unit, kind);
        return new Family
        {
            Name = name,
            Help = EscapeHelp(instrument.Description ?? instrument.Name),
            Kind = kind,
            Bounds = kind == Kind.Histogram ? BucketsFor(instrument.Unit, instrument.Name) : [],
        };
    }

    /// <summary>Histogram upper bounds for an instrument, chosen from its unit (falling back to the name).</summary>
    internal static double[] BucketsFor(string? unit, string name)
    {
        if (unit == "ms" || name.EndsWith("_ms", StringComparison.Ordinal)) return s_msBuckets;
        if (unit is "tokens/s" or "{token}/s") return s_rateBuckets;
        return s_defaultBuckets;
    }

    internal static string MapName(string name, string? unit, Kind kind)
    {
        var sb = new StringBuilder(name.Length + 16);
        foreach (char c in name)
            sb.Append(char.IsAsciiLetterOrDigit(c) || c == '_' || c == ':' ? c : '_');
        if (sb.Length > 0 && char.IsAsciiDigit(sb[0])) sb.Insert(0, '_');
        string baseName = sb.ToString();

        string? suffix = unit switch
        {
            "ms" => "milliseconds",
            "us" => "microseconds",
            "s" => "seconds",
            "By" => "bytes",
            "tokens/s" => "tokens_per_second",
            "tokens" => "tokens",
            "experts" => "experts",
            "slots" => "slots",
            _ => null,
        };
        if (suffix is not null && !baseName.Contains(suffix, StringComparison.Ordinal)
            && !(unit == "ms" && baseName.EndsWith("_ms", StringComparison.Ordinal))
            && !(unit == "us" && baseName.EndsWith("_us", StringComparison.Ordinal))
            && !(unit == "tokens/s" && baseName.Contains("tokens_per_second", StringComparison.Ordinal)))
            baseName += "_" + suffix;

        if (kind == Kind.Counter && !baseName.EndsWith("_total", StringComparison.Ordinal))
            baseName += "_total";
        return baseName;
    }

    private void Record<T>(Instrument instrument, T measurement, ReadOnlySpan<KeyValuePair<string, object?>> tags) where T : struct, System.Numerics.INumber<T>
    {
        if (!_families.TryGetValue(instrument, out var family)) return;
        double value = double.CreateChecked(measurement);
        var series = family.Series.GetOrAdd(RenderLabels(tags), static (key, _) => new Series { Labels = key }, 0);

        lock (series.Gate)
        {
            switch (family.Kind)
            {
                case Kind.Counter:
                    if (instrument.IsObservable) series.Value = value; else series.Value += value;
                    break;
                case Kind.Gauge:
                    if (instrument.IsObservable || !instrument.GetType().Name.Contains("UpDownCounter", StringComparison.Ordinal))
                        series.Value = value;
                    else
                        series.Value += value;
                    break;
                case Kind.Histogram:
                    series.Buckets ??= new long[family.Bounds.Length + 1];
                    int idx = Array.BinarySearch(family.Bounds, value);
                    if (idx < 0) idx = ~idx;          // first bound >= value (le semantics); == Length means +Inf only
                    series.Buckets[idx]++;
                    series.Sum += value;
                    series.Count++;
                    break;
            }
        }
    }

    private static string RenderLabels(ReadOnlySpan<KeyValuePair<string, object?>> tags)
    {
        if (tags.IsEmpty) return "";
        var pairs = new List<(string K, string V)>(tags.Length);
        foreach (var t in tags)
            pairs.Add((SanitizeLabelName(t.Key), Convert.ToString(t.Value, CultureInfo.InvariantCulture) ?? ""));
        pairs.Sort(static (a, b) => string.CompareOrdinal(a.K, b.K));
        var sb = new StringBuilder("{");
        for (int i = 0; i < pairs.Count; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append(pairs[i].K).Append("=\"");
            AppendEscapedLabelValue(sb, pairs[i].V);
            sb.Append('"');
        }
        return sb.Append('}').ToString();
    }

    private static string SanitizeLabelName(string key)
    {
        var sb = new StringBuilder(key.Length);
        foreach (char c in key) sb.Append(char.IsAsciiLetterOrDigit(c) || c == '_' ? c : '_');
        if (sb.Length == 0 || char.IsAsciiDigit(sb[0])) sb.Insert(0, '_');
        return sb.ToString();
    }

    private static void AppendEscapedLabelValue(StringBuilder sb, string v)
    {
        foreach (char c in v)
        {
            switch (c)
            {
                case '\\': sb.Append("\\\\"); break;
                case '"': sb.Append("\\\""); break;
                case '\n': sb.Append("\\n"); break;
                default: sb.Append(c); break;
            }
        }
    }

    private static string EscapeHelp(string s) => s.Replace("\\", "\\\\", StringComparison.Ordinal).Replace("\n", "\\n", StringComparison.Ordinal);

    /// <summary>Formats a sample value: integers without a decimal point, <c>NaN</c>/<c>+Inf</c>/<c>-Inf</c> per the spec.</summary>
    internal static string FormatValue(double v)
    {
        if (double.IsNaN(v)) return "NaN";
        if (double.IsPositiveInfinity(v)) return "+Inf";
        if (double.IsNegativeInfinity(v)) return "-Inf";
        return v.ToString("R", CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Polls observable instruments, then renders every family in the Prometheus text exposition format.
    /// Families with no samples yet are omitted (a family appears on its first measurement).
    /// </summary>
    /// <returns>The exposition text, ending in a newline.</returns>
    public string Render()
    {
        if (!_disposed) _listener.RecordObservableInstruments();

        var sb = new StringBuilder(4096);
        // Several instruments can map to one name (e.g. same instrument name on two meters); render grouped by name.
        foreach (var group in _families.Values.GroupBy(f => f.Name).OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            var first = group.First();
            if (!group.Any(f => !f.Series.IsEmpty)) continue;
            sb.Append("# HELP ").Append(group.Key).Append(' ').Append(first.Help).Append('\n');
            sb.Append("# TYPE ").Append(group.Key).Append(' ')
              .Append(first.Kind switch { Kind.Counter => "counter", Kind.Histogram => "histogram", _ => "gauge" }).Append('\n');

            foreach (var family in group)
            foreach (var series in family.Series.Values.OrderBy(s => s.Labels, StringComparer.Ordinal))
            {
                lock (series.Gate)
                {
                    if (family.Kind != Kind.Histogram)
                    {
                        sb.Append(group.Key).Append(series.Labels).Append(' ').Append(FormatValue(series.Value)).Append('\n');
                        continue;
                    }

                    long cumulative = 0;
                    for (int i = 0; i < family.Bounds.Length; i++)
                    {
                        cumulative += series.Buckets is null ? 0 : series.Buckets[i];
                        sb.Append(group.Key).Append("_bucket").Append(WithLe(series.Labels, FormatValue(family.Bounds[i])))
                          .Append(' ').Append(cumulative.ToString(CultureInfo.InvariantCulture)).Append('\n');
                    }
                    sb.Append(group.Key).Append("_bucket").Append(WithLe(series.Labels, "+Inf"))
                      .Append(' ').Append(series.Count.ToString(CultureInfo.InvariantCulture)).Append('\n');
                    sb.Append(group.Key).Append("_sum").Append(series.Labels).Append(' ').Append(FormatValue(series.Sum)).Append('\n');
                    sb.Append(group.Key).Append("_count").Append(series.Labels).Append(' ').Append(series.Count.ToString(CultureInfo.InvariantCulture)).Append('\n');
                }
            }
        }
        return sb.ToString();
    }

    private static string WithLe(string labels, string le)
        => labels.Length == 0 ? $"{{le=\"{le}\"}}" : $"{labels[..^1]},le=\"{le}\"}}";

    /// <summary>Escapes a value for use inside a hand-written label set (backslash, quote, newline).</summary>
    public static string EscapeLabelValue(string value)
    {
        var sb = new StringBuilder(value.Length);
        AppendEscapedLabelValue(sb, value);
        return sb.ToString();
    }

    /// <summary>Stops listening.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _listener.Dispose();
    }
}
