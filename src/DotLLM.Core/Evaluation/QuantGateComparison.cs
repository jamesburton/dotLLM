namespace DotLLM.Core.Evaluation;

/// <summary>Which quantity the quantization gate compares across backends.</summary>
public enum QuantGateMetric
{
    /// <summary>Relative difference of perplexity. Continuous with <c>quant_matrix.tsv</c>.</summary>
    Perplexity,

    /// <summary>Absolute difference of mean negative log-likelihood, in nats. Scale-free.</summary>
    Nats,

    /// <summary>Assert both independently; report both.</summary>
    Both,
}

/// <summary>Bounds a cross-backend comparison must satisfy.</summary>
/// <param name="MaxNats">Maximum tolerated |ΔNLL| in nats.</param>
/// <param name="MaxPerplexityRelative">Maximum tolerated relative perplexity difference (0.02 = 2%).</param>
public readonly record struct QuantGateThresholds(double MaxNats, double MaxPerplexityRelative)
{
    /// <summary>
    /// Defaults derived from measured data in <c>.docs/corpora/quant_matrix.tsv</c>: the observed
    /// healthy band across 24 types is 0.03%–1.2% relative perplexity, so 2% leaves headroom
    /// without reaching the 23-nat gap that #254 produced. The nats bound is set at 0.05 because
    /// the widest healthy three-way spread measured (Q2_K) is 0.018 nats.
    /// </summary>
    public static QuantGateThresholds Default => new(0.05, 0.02);
}

/// <summary>Outcome of one cross-backend comparison.</summary>
/// <param name="Passed">Whether every asserted arm was within bounds.</param>
/// <param name="NatsDelta">Absolute difference of mean NLL, in nats.</param>
/// <param name="PerplexityRelative">Relative perplexity difference, as a fraction.</param>
/// <param name="Detail">Human-readable summary naming both measured quantities.</param>
public readonly record struct QuantGateVerdict(
    bool Passed, double NatsDelta, double PerplexityRelative, string Detail);

/// <summary>
/// Compares two <see cref="PerplexityResult"/> values scored on identical tokens by different
/// backends (#256).
/// </summary>
/// <remarks>
/// <para>
/// <b>The metric is cross-backend spread on identical tensors, never absolute perplexity.</b> The
/// ≥1B fixtures are <c>--pure</c> requantized from Q8_0, so low-bit ones legitimately collapse:
/// <c>--pure</c> Q2_K scores ~7.1e6 while three backends agree to 0.018 nats. A magnitude
/// threshold would false-alarm on every such fixture.
/// </para>
/// <para>
/// <b>Why nats is the preferred arm.</b> Perplexity is <c>exp(NLL)</c>, so at those magnitudes a
/// negligible numeric difference is amplified into a large percentage. In nats the same
/// comparison is scale-free, and the defect this gate exists to catch (#254) is still 23 nats out.
/// </para>
/// </remarks>
public static class QuantGateComparison
{
    /// <summary>Environment variable selecting the comparison metric.</summary>
    public const string ModeEnvVar = "DOTLLM_QUANT_GATE_MODE";

    /// <summary>Parses a metric name. <see langword="null"/> or empty selects <see cref="QuantGateMetric.Both"/>.</summary>
    /// <param name="raw">Mode name, case-insensitive: <c>perplexity</c>, <c>nats</c> or <c>both</c>.</param>
    /// <returns>The parsed metric.</returns>
    /// <exception cref="ArgumentException">The value is not a recognised mode.</exception>
    public static QuantGateMetric ResolveMode(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
            return QuantGateMetric.Both;

        // Deliberately throws rather than defaulting: a toggle that silently ignores a typo
        // reports a metric nobody selected, and the run looks like it passed.
        return raw.Trim().ToLowerInvariant() switch
        {
            "perplexity" or "ppl" => QuantGateMetric.Perplexity,
            "nats" or "nll" => QuantGateMetric.Nats,
            "both" => QuantGateMetric.Both,
            _ => throw new ArgumentException(
                $"Unknown {ModeEnvVar} value '{raw}'. Expected 'perplexity', 'nats' or 'both'.", nameof(raw)),
        };
    }

    /// <summary>Reads <see cref="ModeEnvVar"/> and parses it via <see cref="ResolveMode"/>.</summary>
    /// <returns>The configured metric, or <see cref="QuantGateMetric.Both"/> when unset.</returns>
    public static QuantGateMetric ResolveModeFromEnvironment()
        => ResolveMode(Environment.GetEnvironmentVariable(ModeEnvVar));

    /// <summary>Compares two results scored on identical tokens.</summary>
    /// <param name="reference">Result from the reference backend.</param>
    /// <param name="candidate">Result from the backend under test.</param>
    /// <param name="metric">Which arm(s) to assert.</param>
    /// <param name="thresholds">Bounds to apply.</param>
    /// <returns>The verdict, carrying both measured quantities regardless of <paramref name="metric"/>.</returns>
    public static QuantGateVerdict Compare(
        PerplexityResult reference, PerplexityResult candidate,
        QuantGateMetric metric, QuantGateThresholds thresholds)
    {
        // Guard first: NaN fails every relational test, so an unguarded bound check would read a
        // NaN-emitting kernel as "within tolerance".
        if (!IsFinite(reference) || !IsFinite(candidate))
        {
            return new QuantGateVerdict(false, double.NaN, double.NaN,
                $"non-finite score: reference ppl={reference.Perplexity} nll={reference.MeanNegativeLogLikelihood}, " +
                $"candidate ppl={candidate.Perplexity} nll={candidate.MeanNegativeLogLikelihood}");
        }

        double natsDelta = Math.Abs(reference.MeanNegativeLogLikelihood - candidate.MeanNegativeLogLikelihood);

        // Denominator is the smaller magnitude, so the reported spread is the conservative one.
        double denominator = Math.Min(Math.Abs(reference.Perplexity), Math.Abs(candidate.Perplexity));
        double pplRelative = denominator > 0
            ? Math.Abs(reference.Perplexity - candidate.Perplexity) / denominator
            : double.PositiveInfinity;

        bool natsOk = natsDelta <= thresholds.MaxNats;
        bool pplOk = pplRelative <= thresholds.MaxPerplexityRelative;

        bool passed = metric switch
        {
            QuantGateMetric.Nats => natsOk,
            QuantGateMetric.Perplexity => pplOk,
            QuantGateMetric.Both => natsOk && pplOk,
            _ => throw new ArgumentOutOfRangeException(nameof(metric), metric, "Unhandled metric."),
        };

        string detail =
            $"|ΔNLL| = {natsDelta:F6} nats (bound {thresholds.MaxNats:F3}, {(natsOk ? "ok" : "BREACH")}); " +
            $"perplexity spread = {pplRelative * 100:F4}% (bound {thresholds.MaxPerplexityRelative * 100:F2}%, " +
            $"{(pplOk ? "ok" : "BREACH")}); metric = {metric}";

        return new QuantGateVerdict(passed, natsDelta, pplRelative, detail);
    }

    private static bool IsFinite(PerplexityResult r)
        => double.IsFinite(r.Perplexity) && double.IsFinite(r.MeanNegativeLogLikelihood);
}
