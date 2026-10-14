namespace DotLLM.Cli.Benchmarking;

/// <summary>
/// One timed benchmark repetition: a fresh-KV-cache prefill of
/// <see cref="PromptTokens"/> tokens followed by <see cref="DecodeTokens"/>
/// greedy decode steps. Sampling (argmax) time is excluded from both figures.
/// </summary>
/// <param name="PrefillMs">Wall time of the single prefill forward pass, in milliseconds.</param>
/// <param name="DecodeMs">Total wall time of all decode forward passes, in milliseconds.</param>
/// <param name="PromptTokens">Actual prompt token count fed to the prefill.</param>
/// <param name="DecodeTokens">Number of greedy decode steps timed.</param>
public readonly record struct BenchRep(double PrefillMs, double DecodeMs, int PromptTokens, int DecodeTokens)
{
    /// <summary>Prefill throughput in tokens/second.</summary>
    public double PrefillTokS => PrefillMs > 0 ? PromptTokens / (PrefillMs / 1000.0) : 0;

    /// <summary>Decode throughput in tokens/second.</summary>
    public double DecodeTokS => DecodeMs > 0 ? DecodeTokens / (DecodeMs / 1000.0) : 0;
}

/// <summary>
/// Aggregated result of a benchmark run: the warm-up repetition (discarded from
/// statistics), the measured repetitions, and median / best summaries.
/// </summary>
public sealed class BenchResult
{
    /// <summary>The discarded warm-up repetition (JIT / shader-compile / page-in cost).</summary>
    public required BenchRep Warmup { get; init; }

    /// <summary>Measured repetitions (warm-up excluded), in run order.</summary>
    public required IReadOnlyList<BenchRep> Reps { get; init; }

    /// <summary>Model load wall time in milliseconds — excluded from every repetition.</summary>
    public required double LoadMs { get; init; }

    /// <summary>Actual prompt token count per repetition.</summary>
    public required int PromptTokens { get; init; }

    /// <summary>Greedy decode steps timed per repetition.</summary>
    public required int DecodeTokens { get; init; }

    /// <summary>
    /// Extra synthetic context tokens appended (untimed) after the prefill and before
    /// the timed decode. Decode therefore runs at context depth
    /// <see cref="PromptTokens"/> + <see cref="Depth"/>.
    /// </summary>
    public required int Depth { get; init; }

    /// <summary>Context depth the timed decode ran at (<c>PromptTokens + Depth</c>).</summary>
    public int DecodeCtxDepth => PromptTokens + Depth;

    /// <summary>Median prefill throughput across measured reps, tokens/second.</summary>
    public double PrefillTokSMedian => BenchStats.Median(Reps.Select(r => r.PrefillTokS).ToArray());

    /// <summary>Best prefill throughput (fastest rep), tokens/second.</summary>
    public double PrefillTokSBest => Reps.Max(r => r.PrefillTokS);

    /// <summary>Median decode throughput across measured reps, tokens/second.</summary>
    public double DecodeTokSMedian => BenchStats.Median(Reps.Select(r => r.DecodeTokS).ToArray());

    /// <summary>Best decode throughput (fastest rep), tokens/second.</summary>
    public double DecodeTokSBest => Reps.Max(r => r.DecodeTokS);

    /// <summary>Median prefill wall time across measured reps, milliseconds.</summary>
    public double PrefillMsMedian => BenchStats.Median(Reps.Select(r => r.PrefillMs).ToArray());

    /// <summary>Minimum prefill wall time across measured reps, milliseconds.</summary>
    public double PrefillMsMin => BenchStats.Min(Reps.Select(r => r.PrefillMs).ToArray());

    /// <summary>Median decode wall time across measured reps, milliseconds.</summary>
    public double DecodeMsMedian => BenchStats.Median(Reps.Select(r => r.DecodeMs).ToArray());

    /// <summary>Minimum decode wall time across measured reps, milliseconds.</summary>
    public double DecodeMsMin => BenchStats.Min(Reps.Select(r => r.DecodeMs).ToArray());
}
