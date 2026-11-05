using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Tokenizers;

namespace DotLLM.Server.ChunkCache;

/// <summary>A located, token-verified chunk: where its ids sit in the prompt's ids, and whether the store holds it.</summary>
/// <param name="TokenStart">Index of the chunk's first id in the whole-prompt ids.</param>
/// <param name="TokenCount">Its id count (<c>ChunkTokenization.Encode(text).Length</c>).</param>
/// <param name="Hit">Whether the store holds an entry under the chunk's key.</param>
internal readonly record struct PlannedSpan(int TokenStart, int TokenCount, bool Hit);

/// <summary>
/// One read annotation's outcome, for per-chunk status (an authenticated caller's <c>report: "per_chunk"</c> and
/// <c>explain</c>): <c>hit</c> / <c>miss</c> for a verified chunk (whether it lay inside the restored prefix is decided
/// once that is known), else <c>ignored</c> with the §5.8 code that dropped it (null when nothing more specific applies).
/// </summary>
/// <param name="Index">The annotation's position in document order.</param>
/// <param name="Id">Its <c>id</c> (the caller's own label), when read.</param>
/// <param name="Status"><c>hit</c>, <c>miss</c> or <c>ignored</c>.</param>
/// <param name="Code">Why it was ignored.</param>
/// <param name="TokenStart">Verified chunks: index of the first id in the prompt's ids; else -1.</param>
/// <param name="TokenCount">Verified chunks: id count; else 0.</param>
internal readonly record struct SpanStatus(int Index, string? Id, string Status, string? Code, int TokenStart, int TokenCount)
{
    /// <summary>Status of a verified chunk the store holds.</summary>
    public const string Hit = "hit";

    /// <summary>Status of a verified chunk the store does not hold.</summary>
    public const string Miss = "miss";

    /// <summary>Status of an annotation treated as plain text.</summary>
    public const string Ignored = "ignored";

    /// <summary>Status of a verified chunk wholly inside the restored exact prefix.</summary>
    public const string Prefix = "prefix";
}

/// <summary>What <see cref="ChunkCachePipeline.Apply"/> decided for one request, before generation.</summary>
/// <param name="Requested">The requested policy.</param>
/// <param name="Reason">Why the applied policy (always exact) differs from the requested one, or null.</param>
/// <param name="Shadow">Whether a shadow plan is reported.</param>
/// <param name="Annotated">Annotations in the request.</param>
/// <param name="Spans">The verified chunks (hit or miss; <c>prefix</c> is decided once the restored prefix is known).</param>
/// <param name="Warnings">Parser, locator and pipeline warnings, merged per code in order of first occurrence.</param>
/// <param name="Error">With <c>strict</c>: the error for the first defect; else null.</param>
/// <param name="Report">The requested report level (<c>none</c>, <c>summary</c>, <c>per_chunk</c>).</param>
/// <param name="Backend">The backend tag the cost model is chosen by.</param>
/// <param name="PromptTokens">Ids of the whole rendered prompt (the plain request's tokenisation).</param>
/// <param name="Budget">The shadow plan's recompute budget: the request's <c>recompute.budget</c>, else the default (ruling 50).</param>
internal sealed record ChunkCachePlan(
    ChunkPolicy Requested, string? Reason, bool Shadow, int Annotated, IReadOnlyList<PlannedSpan> Spans,
    IReadOnlyList<ChunkWarning> Warnings, ChunkCacheError? Error, string Report, string Backend, int PromptTokens,
    double Budget = ChunkCacheReport.DefaultPlanBudget)
{
    /// <summary>Every read annotation's outcome, in document order (see <see cref="SpanStatus"/>).</summary>
    public IReadOnlyList<SpanStatus> Statuses { get; init; } = [];
}

/// <summary>
/// Runs <c>exact</c> / <c>shadow</c> for one annotated request (spec §5.5, §5.9; Phase I Task 13). Nothing here changes
/// what the engine computes: the prompt is tokenised exactly as a plain request (§5.3 item 7), and annotations only
/// produce counts and, under <c>shadow</c>, a plan.
/// </summary>
/// <remarks>
/// <para>
/// <b>Per located span</b> (Task 11), in prompt order, at most <see cref="MaxPlannedSpans"/> per request: its ids are
/// <c>ChunkTokenization.Encode(text)</c> (ruling 21, Task 5b); shorter than <c>min_chunk_tokens</c> is
/// <c>chunk_ignored_too_short</c>; then the <b>token check</b>: equal text is not equal tokens at a span's edges (BPE merges
/// with the text around it, a SentencePiece leading space, a live reasoning tag in assistant content), so a span counts
/// only when its ids occur in the whole-prompt ids exactly where its characters are; else <c>chunk_template_altered</c>;
/// then the store lookup under <see cref="ServerModelIdentity.KeyOf"/> in the caller's (server-assigned) namespace.
/// </para>
/// <para>
/// <b>The token check</b> walks the prompt once and only ever counts a span whose ids equal the prompt's ids at the chosen
/// index. With a token boundary proven at the end of the previous verified span, the glue up to the next span is encoded
/// alone (<c>EncodeRaw</c>) and compared with the prompt's ids there. A tokenizer that adds a leading-space piece to the
/// start of an encode (SentencePiece <c>add_dummy_prefix</c>) makes the glue's first piece differ while the rest match, so
/// the glue is also accepted when everything after its first piece matches within two ids of the boundary. Otherwise the
/// prefix up to the span is encoded (<c>Encode</c>, as the prompt) and must be a prefix of the prompt's ids: that proves a
/// token boundary at the span start, and differing ids there are <c>chunk_template_altered</c>. Prefix encodes are capped
/// at <see cref="PrefixBudgetFactor"/> × the prompt length (+64 Ki chars); a span the walk cannot place once the cap is
/// spent, and every span past <see cref="MaxPlannedSpans"/>, is ignored with <c>chunk_verification_skipped</c> (ruling 49:
/// a warning only, never a strict error). So the work before the scheduler queue is linear in the prompt, plus at most
/// <see cref="MaxPlannedSpans"/> store header reads.
/// </para>
/// <para>
/// <b>Strict mode</b>: the error is the first of the request-level defects (<c>chunk_model_unsupported</c>,
/// <c>chunk_policy_unavailable</c>, <c>chunk_store_unavailable</c>), else the span-level defect of the span first in
/// document order (ruling 4: locator and token-check defects alike). Warning-only codes (<c>chunk_ignored_too_short</c>,
/// <c>chunk_verification_skipped</c>) never become an error. Parser errors were returned before the prompt was rendered.
/// </para>
/// </remarks>
internal static class ChunkCachePipeline
{
    /// <summary>Prefix encodes for the token check stop after this many times the prompt length (plus a constant).</summary>
    internal const int PrefixBudgetFactor = 4;

    /// <summary>Most located spans tokenised, verified and looked up per request; the rest are <c>chunk_verification_skipped</c>.</summary>
    internal const int MaxPlannedSpans = 1024;

    private const int PrefixBudgetConstant = 64 * 1024;

    /// <summary>How far past the boundary the glue's tail may start when its first piece differs (a leading-space piece).</summary>
    private const int LeadingPieceSlack = 2;

    /// <summary>Builds the plan.</summary>
    /// <param name="parsed">The parser's result (Task 10).</param>
    /// <param name="located">The locator's result (Task 11) for <c>parsed.Spans</c>; null when not run (unsupported model, no
    /// accepted span).</param>
    /// <param name="identity">The served model's identity.</param>
    /// <param name="tokenizer">The model's tokenizer.</param>
    /// <param name="prompt">The rendered prompt.</param>
    /// <param name="promptIds">The prompt's ids exactly as the plain request tokenises it (<c>tokenizer.Encode(prompt)</c>).</param>
    /// <param name="store">The chunk store, or null when none is configured or usable.</param>
    /// <param name="storeUnavailable">A store is configured but unusable: misses plus <c>chunk_store_unavailable</c>.</param>
    /// <returns>The plan.</returns>
    public static ChunkCachePlan Apply(ParsedChunkRequest parsed, LocatedSpans? located, ServerModelIdentity identity,
        ITokenizer tokenizer, string prompt, IReadOnlyList<int> promptIds, IChunkStore? store, bool storeUnavailable)
    {
        ArgumentNullException.ThrowIfNull(parsed);
        ArgumentNullException.ThrowIfNull(identity);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(promptIds);

        var warnings = new WarningList(parsed.Warnings);
        var requestErrors = new List<string>();          // request-level codes, in precedence order
        var spanErrors = new List<(int Doc, string Code)>(); // span-level codes with the span's document index
        var spans = new List<PlannedSpan>();
        bool shadow = parsed.Requested == ChunkPolicy.Shadow;
        string? reason = null;
        string backend = identity.Supported ? identity.Backend : "cpu";
        double budget = parsed.RecomputeBudget ?? ChunkCacheReport.DefaultPlanBudget;

        var outcomes = new SpanStatus?[parsed.Spans.Count];
        if (!identity.Supported)
        {
            // Every policy degrades to exact; every annotation is plain text (ruling 41: reason only when not exact).
            for (int i = 0; i < outcomes.Length; i++)
                outcomes[i] = Ignored(parsed.Spans[i], ChunkCacheCodes.ModelUnsupported);
            warnings.Add(ChunkCacheCodes.ModelUnsupported, 1);
            requestErrors.Add(ChunkCacheCodes.ModelUnsupported);
            if (parsed.Requested != ChunkPolicy.Exact)
                reason = ChunkCacheCodes.ModelUnsupported;
            shadow = false;
        }
        else
        {
            if (parsed.Requested is ChunkPolicy.Balanced or ChunkPolicy.Fast or ChunkPolicy.Auto)
            {
                // Applied reuse arrives in Phase I5: parse, then serve exact with the reason (ruling 42: and one warning).
                reason = ChunkCacheCodes.PolicyUnavailable;
                warnings.Add(ChunkCacheCodes.PolicyUnavailable, 1);
                requestErrors.Add(ChunkCacheCodes.PolicyUnavailable);
            }

            if (located is not null)
            {
                var docIndex = new Dictionary<ChunkSpanRef, int>(ReferenceEqualityComparer.Instance);
                for (int i = 0; i < parsed.Spans.Count; i++)
                    docIndex.TryAdd(parsed.Spans[i], i);

                foreach (var w in located.Warnings)
                    warnings.Add(w.Code, w.Count);
                for (int i = 0; i < located.Codes.Count; i++)
                {
                    if (located.Codes[i] is not { } code)
                        continue;
                    if (i < outcomes.Length)
                        outcomes[i] = Ignored(parsed.Spans[i], code);
                    if (!ChunkCacheCodes.IsWarningOnly(code))
                        spanErrors.Add((i, code));
                }

                PlanSpans(parsed, located, identity, tokenizer, prompt, promptIds, store, storeUnavailable, spans, warnings,
                    spanErrors, docIndex, outcomes, out bool lookedUp);
                if (storeUnavailable && lookedUp)
                {
                    warnings.Add(ChunkCacheCodes.StoreUnavailable, 1);
                    requestErrors.Add(ChunkCacheCodes.StoreUnavailable);
                }
            }
        }

        ChunkCacheError? error = null;
        if (parsed.Strict)
        {
            string? code = requestErrors.Count > 0 ? requestErrors[0]
                : spanErrors.Count > 0 ? spanErrors.MinBy(e => e.Doc).Code
                : null;
            if (code is not null)
                error = ChunkCacheCodes.ErrorFor(code, MessageFor(code));
        }

        return new ChunkCachePlan(parsed.Requested, reason, shadow, parsed.Annotated, spans, warnings.ToList(), error,
            parsed.Report, backend, promptIds.Count, budget)
        {
            Statuses = StatusesOf(parsed, outcomes),
        };
    }

    /// <summary>Every read annotation's outcome in document order: the accepted spans' and the parser's dropped ones.</summary>
    /// <param name="parsed">The parse.</param>
    /// <param name="outcomes">Per accepted span (index into <c>parsed.Spans</c>), its outcome; null = ignored, no code.</param>
    /// <returns>The statuses.</returns>
    internal static IReadOnlyList<SpanStatus> StatusesOf(ParsedChunkRequest parsed, IReadOnlyList<SpanStatus?> outcomes)
    {
        var all = new List<SpanStatus>(parsed.Spans.Count + parsed.Defects.Count);
        for (int i = 0; i < parsed.Spans.Count; i++)
            all.Add(i < outcomes.Count && outcomes[i] is { } o ? o : Ignored(parsed.Spans[i], null));
        foreach (var d in parsed.Defects)
            all.Add(new SpanStatus(d.Index, d.Id, SpanStatus.Ignored, d.Code, -1, 0));
        all.Sort((a, b) => a.Index.CompareTo(b.Index));
        return all;
    }

    private static SpanStatus Ignored(ChunkSpanRef span, string? code) =>
        new(span.Index, span.Id, SpanStatus.Ignored, code, -1, 0);

    private static void PlanSpans(ParsedChunkRequest parsed, LocatedSpans located, ServerModelIdentity identity,
        ITokenizer tokenizer, string prompt, IReadOnlyList<int> promptIds, IChunkStore? store, bool storeUnavailable,
        List<PlannedSpan> spans, WarningList warnings, List<(int Doc, string Code)> spanErrors,
        Dictionary<ChunkSpanRef, int> docIndex, SpanStatus?[] outcomes, out bool lookedUp)
    {
        lookedUp = false;
        var order = located.Located
            .Where(l => l.RenderedStartChar >= 0 && l.RenderedStartChar < l.RenderedEndChar && l.RenderedEndChar <= prompt.Length)
            .OrderBy(l => l.RenderedStartChar)
            .ToList();
        int malformedRanges = located.Located.Count - order.Count;
        if (malformedRanges > 0)
        {
            warnings.Add(ChunkCacheCodes.VerificationSkipped, malformedRanges); // a locator bug: never counted, never an error
            foreach (var l in located.Located.Except(order))
                Set(l.Span, Ignored(l.Span, ChunkCacheCodes.VerificationSkipped));
        }

        void Set(ChunkSpanRef span, SpanStatus status)
        {
            if (docIndex.TryGetValue(span, out int d) && d < outcomes.Length)
                outcomes[d] = status;
        }

        var checker = new TokenChecker(tokenizer, prompt, promptIds);
        int planned = 0;
        foreach (var l in order)
        {
            if (planned++ >= MaxPlannedSpans)
            {
                warnings.Add(ChunkCacheCodes.VerificationSkipped, 1);
                Set(l.Span, Ignored(l.Span, ChunkCacheCodes.VerificationSkipped));
                continue;
            }

            int[] ids = ChunkTokenization.Encode(tokenizer, l.Span.Text);
            if (ids.Length == 0 || ids.Length < parsed.MinChunkTokens)
            {
                warnings.Add(ChunkCacheCodes.IgnoredTooShort, 1);
                Set(l.Span, Ignored(l.Span, ChunkCacheCodes.IgnoredTooShort));
                continue;
            }

            var (start, outcome) = checker.Locate(l.RenderedStartChar, l.RenderedEndChar, ids);
            if (outcome == Outcome.Skipped)
            {
                warnings.Add(ChunkCacheCodes.VerificationSkipped, 1);
                Set(l.Span, Ignored(l.Span, ChunkCacheCodes.VerificationSkipped));
                continue;
            }
            if (outcome == Outcome.Altered)
            {
                warnings.Add(ChunkCacheCodes.TemplateAltered, 1);
                Set(l.Span, Ignored(l.Span, ChunkCacheCodes.TemplateAltered));
                spanErrors.Add((docIndex.TryGetValue(l.Span, out int d) ? d : int.MaxValue, ChunkCacheCodes.TemplateAltered));
                continue;
            }

            bool hit = false;
            if (store is not null || storeUnavailable)
            {
                lookedUp = true;
                if (store is not null && !storeUnavailable)
                    hit = Contains(store, identity, ids, parsed.Namespace);
            }
            spans.Add(new PlannedSpan(start, ids.Length, hit));
            Set(l.Span, new SpanStatus(l.Span.Index, l.Span.Id, hit ? SpanStatus.Hit : SpanStatus.Miss, null, start, ids.Length));
        }
    }

    // A store failure is a miss (spec 5.7), never an exception on the request path.
    private static bool Contains(IChunkStore store, ServerModelIdentity identity, int[] ids, string ns)
    {
        try
        {
            return store.Contains(identity.KeyOf(ids, ns));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or InvalidOperationException)
        {
            return false;
        }
    }

    private static string MessageFor(string code) => code switch
    {
        ChunkCacheCodes.ModelUnsupported => "chunk_cache: the served model or backend cannot reuse chunks (recurrent, hybrid or without chunk assembly)",
        ChunkCacheCodes.PolicyUnavailable => "chunk_cache: only the exact and shadow policies are available on this server",
        ChunkCacheCodes.StoreUnavailable => "chunk_cache: the chunk store is unavailable",
        ChunkCacheCodes.TemplateAltered => "chunk_cache: a chunk span is not reproduced verbatim (characters and token ids) in the rendered prompt",
        ChunkCacheCodes.SpanInvalid => "chunk_cache: a chunk span is not where its position says",
        _ => "chunk_cache: " + code,
    };

    internal enum Outcome : byte { Verified, Altered, Skipped }

    /// <summary>The token check (see the class remarks).</summary>
    internal sealed class TokenChecker(ITokenizer tokenizer, string prompt, IReadOnlyList<int> promptIds)
    {
        private readonly long _budget = (long)PrefixBudgetFactor * prompt.Length + PrefixBudgetConstant;
        private long _spent;
        private int _syncChar = -1; // a char position of the prompt with a known token boundary ...
        private int _syncToken = -1; // ... at this index of the prompt's ids

        /// <summary>Characters encoded by prefix fallbacks so far (tests).</summary>
        internal long PrefixCharsEncoded => _spent;

        /// <summary>Where the span at chars [start, end) with <paramref name="ids"/> sits in the prompt's ids.</summary>
        public (int Start, Outcome Outcome) Locate(int start, int end, int[] ids)
        {
            // 1. Glue from the last known boundary, encoded alone.
            if (_syncChar >= 0 && _syncChar <= start)
            {
                int[] glue = start > _syncChar ? tokenizer.EncodeRaw(prompt.Substring(_syncChar, start - _syncChar)) : [];
                if (Matches(_syncToken, glue))
                {
                    // The prompt's ids before the span are proven: the span starts at a token boundary here.
                    int at = _syncToken + glue.Length;
                    if (Matches(at, ids))
                        return Verified(at, end, ids);
                    _syncChar = -1;
                    return (-1, Outcome.Altered);
                }
                if (glue.Length > 0)
                {
                    // A leading piece the encoder adds to every encode (SentencePiece's dummy space prefix): everything after
                    // the glue's first piece must match, starting within LeadingPieceSlack ids of the boundary.
                    var tail = glue.AsSpan(1);
                    for (int d = 0; d <= LeadingPieceSlack; d++)
                    {
                        int from = _syncToken + d;
                        if (Matches(from, tail) && Matches(from + tail.Length, ids))
                            return Verified(from + tail.Length, end, ids);
                    }
                }
            }

            // 2. Prefix fallback: proves (or disproves) a token boundary at the span start.
            if (_spent + start > _budget)
            {
                _syncChar = -1;
                return (-1, Outcome.Skipped);
            }
            _spent += start;
            int[] prefix = start > 0 ? tokenizer.Encode(prompt[..start]) : [];
            if (Matches(0, prefix) && Matches(prefix.Length, ids))
                return Verified(prefix.Length, end, ids);
            _syncChar = -1;
            return (-1, Outcome.Altered);
        }

        private (int, Outcome) Verified(int at, int end, int[] ids)
        {
            _syncChar = end;
            _syncToken = at + ids.Length;
            return (at, Outcome.Verified);
        }

        private bool Matches(int offset, ReadOnlySpan<int> ids)
        {
            if (offset < 0 || offset + ids.Length > promptIds.Count)
                return false;
            for (int i = 0; i < ids.Length; i++)
                if (promptIds[offset + i] != ids[i])
                    return false;
            return true;
        }
    }

    /// <summary>Warning codes with summed counts, in order of first occurrence.</summary>
    private sealed class WarningList
    {
        private readonly List<string> _order = [];
        private readonly Dictionary<string, int> _counts = new(StringComparer.Ordinal);

        public WarningList(IEnumerable<ChunkWarning> initial)
        {
            foreach (var w in initial)
                Add(w.Code, w.Count);
        }

        public void Add(string code, int count)
        {
            if (_counts.TryGetValue(code, out int c))
            {
                _counts[code] = (int)Math.Min(int.MaxValue, (long)c + count);
                return;
            }
            _order.Add(code);
            _counts[code] = count;
        }

        public List<ChunkWarning> ToList() => _order.Select(c => new ChunkWarning(c, _counts[c])).ToList();
    }
}
