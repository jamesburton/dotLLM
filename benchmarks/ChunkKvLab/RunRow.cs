using System.Globalization;

namespace ChunkKvLab;

/// <summary>One E1 result row (one method on one seed and context size).</summary>
/// <param name="Method"><c>full</c>, <c>naive_concat</c> or <c>prefix_reuse</c>.</param>
/// <param name="Seed">Context seed.</param>
/// <param name="ContextTokens">Context length in tokens (chunks × chunk size).</param>
/// <param name="Top1Agreement">Fraction of teacher-forced continuation positions whose argmax equals the reference's.</param>
/// <param name="MeanKl">Mean KL(reference ‖ method) in nats over the teacher-forced continuation positions.</param>
/// <param name="ExactMatch">Common leading run of the method's and the reference's greedy tokens, divided by N.</param>
/// <param name="TtftMs">Cache build (prefill / assembly / reuse + suffix prefill) plus the first-token forward, wall ms (CPU, indicative).</param>
/// <param name="PrefillTokensComputed">Context tokens run through the model for this request.</param>
/// <param name="CacheBytes">Bytes of precomputed KV (F32) read from earlier work: 0 for full, all chunk entries for naive, the matched prefix for prefix reuse.</param>
public sealed record RunRow(
    string Method, int Seed, int ContextTokens, double Top1Agreement, double MeanKl, double ExactMatch, double TtftMs,
    long PrefillTokensComputed, long CacheBytes)
{
    /// <summary>CSV header matching <see cref="ToCsv"/>.</summary>
    public const string CsvHeader = "Method,Seed,ContextTokens,Top1Agreement,MeanKl,ExactMatch,TtftMs,PrefillTokensComputed,CacheBytes";

    /// <summary>Invariant-culture CSV line.</summary>
    public string ToCsv() => string.Join(',',
        Method, Seed.ToString(CultureInfo.InvariantCulture), ContextTokens.ToString(CultureInfo.InvariantCulture),
        Top1Agreement.ToString("R", CultureInfo.InvariantCulture), MeanKl.ToString("R", CultureInfo.InvariantCulture),
        ExactMatch.ToString("R", CultureInfo.InvariantCulture), TtftMs.ToString("F1", CultureInfo.InvariantCulture),
        PrefillTokensComputed.ToString(CultureInfo.InvariantCulture), CacheBytes.ToString(CultureInfo.InvariantCulture));
}
