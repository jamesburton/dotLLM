using System.Globalization;

namespace DotLLM.Cli.Benchmarking;

/// <summary>
/// Pure statistics helpers for the <c>bench</c> command: median / min aggregation
/// over per-repetition measurements with the warm-up repetition discarded.
/// Kept free of any I/O or model dependency so the math is unit-testable.
/// </summary>
public static class BenchStats
{
    /// <summary>
    /// Median of <paramref name="values"/>. Even counts average the two middle
    /// elements. Throws on an empty input.
    /// </summary>
    public static double Median(IReadOnlyList<double> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Count == 0)
            throw new ArgumentException("Median of an empty sequence is undefined.", nameof(values));

        double[] sorted = values.ToArray();
        Array.Sort(sorted);
        int mid = sorted.Length / 2;
        return sorted.Length % 2 == 1
            ? sorted[mid]
            : (sorted[mid - 1] + sorted[mid]) / 2.0;
    }

    /// <summary>Minimum of <paramref name="values"/>. Throws on an empty input.</summary>
    public static double Min(IReadOnlyList<double> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Count == 0)
            throw new ArgumentException("Min of an empty sequence is undefined.", nameof(values));

        double min = double.PositiveInfinity;
        for (int i = 0; i < values.Count; i++)
            if (values[i] < min) min = values[i];
        return min;
    }

    /// <summary>
    /// Discards the warm-up repetition(s) from a raw measurement list: the first
    /// <paramref name="warmupCount"/> entries are dropped, the rest returned in order.
    /// </summary>
    public static IReadOnlyList<T> DiscardWarmup<T>(IReadOnlyList<T> reps, int warmupCount = 1)
    {
        ArgumentNullException.ThrowIfNull(reps);
        ArgumentOutOfRangeException.ThrowIfNegative(warmupCount);
        if (reps.Count <= warmupCount)
            throw new ArgumentException(
                $"Need more than {warmupCount} repetitions to discard {warmupCount} warm-up rep(s); got {reps.Count}.",
                nameof(reps));
        return reps.Skip(warmupCount).ToArray();
    }

    /// <summary>
    /// Synthesizes a prompt of exactly <paramref name="targetTokens"/> token ids by
    /// tiling <paramref name="seedTokens"/> (truncating the final repetition).
    /// </summary>
    public static int[] TilePrompt(ReadOnlySpan<int> seedTokens, int targetTokens)
    {
        if (seedTokens.Length == 0)
            throw new ArgumentException("Seed prompt produced no tokens.", nameof(seedTokens));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetTokens);

        var prompt = new int[targetTokens];
        for (int i = 0; i < targetTokens; i++)
            prompt[i] = seedTokens[i % seedTokens.Length];
        return prompt;
    }

    /// <summary>
    /// Formats a throughput value for CSV / display with precision scaled to magnitude
    /// (matches the hand-entered convention in <c>benchmarks/perf-matrix/results.csv</c>:
    /// whole numbers above 100 tok/s, one decimal above 10, two below).
    /// </summary>
    public static string FormatTokS(double tokPerSec)
    {
        if (double.IsNaN(tokPerSec) || double.IsInfinity(tokPerSec)) return "";
        return tokPerSec switch
        {
            >= 100 => tokPerSec.ToString("F0", CultureInfo.InvariantCulture),
            >= 10 => tokPerSec.ToString("F1", CultureInfo.InvariantCulture),
            >= 0.1 => tokPerSec.ToString("F2", CultureInfo.InvariantCulture),
            _ => tokPerSec.ToString("F3", CultureInfo.InvariantCulture),
        };
    }
}
