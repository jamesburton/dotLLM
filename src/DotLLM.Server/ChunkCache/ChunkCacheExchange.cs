using DotLLM.Engine;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;

namespace DotLLM.Server.ChunkCache;

/// <summary>
/// The chunk-cache side of one annotated inference request, as the endpoints use it: parse (done by the caller), then
/// <see cref="Prepare"/> after the prompt is rendered, then <see cref="Finish"/> once the engine reports its prefix reuse.
/// It only reads the request, the prompt and the store: generation runs exactly as for the unannotated request.
/// </summary>
internal sealed class ChunkCacheExchange
{
    private ChunkCacheReportDto? _report;
    private bool _finished;

    private ChunkCacheExchange(ParsedChunkRequest parsed, ChunkCaller caller)
    {
        Parsed = parsed;
        Caller = caller;
    }

    /// <summary>The parse.</summary>
    public ParsedChunkRequest Parsed { get; }

    /// <summary>
    /// Who is asking (<see cref="ChunkCacheAuth"/>). An unauthenticated caller's chunks are never looked up in the store
    /// (all misses, no warning: the store is a membership oracle) and it gets the summary even for <c>per_chunk</c>.
    /// </summary>
    public ChunkCaller Caller { get; }

    /// <summary>The plan, once <see cref="Prepare"/> ran.</summary>
    public ChunkCachePlan? Plan { get; private set; }

    /// <summary>Starts an exchange; null when there is nothing to do (flag off or no annotation).</summary>
    /// <param name="parsed">The parser's result (parsed with <paramref name="caller"/>'s namespace).</param>
    /// <param name="caller">The caller.</param>
    /// <returns>The exchange or null.</returns>
    public static ChunkCacheExchange? Begin(ParsedChunkRequest? parsed, ChunkCaller caller)
    {
        ArgumentNullException.ThrowIfNull(caller);
        // One source of truth: lookups use parsed.Namespace, the lookup gate uses the caller. A call site that parsed in
        // another namespace than the caller's is a bug (the message never names either namespace).
        if (parsed is not null && !string.Equals(parsed.Namespace, caller.Namespace, StringComparison.Ordinal))
            throw new ArgumentException("The parse and the caller name different chunk namespaces.", nameof(caller));
        return parsed is null ? null : new ChunkCacheExchange(parsed, caller);
    }

    /// <summary>
    /// Locates the accepted spans in the rendered prompt, verifies their ids and looks them up in the store. Never throws on
    /// request content: an unexpected failure ignores every annotation (counts only, no warning) and is logged without content.
    /// </summary>
    /// <param name="state">The server state (tokenizer, template, store, model identity).</param>
    /// <param name="messages">The messages exactly as rendered.</param>
    /// <param name="engineMessageOffset">1 when a profile system prompt was prepended to the converter's messages, else 0.</param>
    /// <param name="options">The template options exactly as rendered.</param>
    /// <param name="prompt">The rendered prompt.</param>
    /// <param name="promptIds">The prompt's ids when the caller already has <c>tokenizer.Encode(prompt)</c> (explain); null encodes here.</param>
    /// <returns>The strict-mode error, or null.</returns>
    public ChunkCacheError? Prepare(ServerState state, IReadOnlyList<ChatMessage> messages, int engineMessageOffset,
        ChatTemplateOptions options, string prompt, int[]? promptIds = null)
    {
        ArgumentNullException.ThrowIfNull(state);
        string backend = "cpu";
        try
        {
            var runtime = state.ChunkCacheRuntime;
            var identity = runtime.IdentityFor(state); // never throws; a failure is an unsupported identity, cached per model
            backend = identity.Supported ? identity.Backend : "cpu";
            ITokenizer tokenizer = state.Tokenizer!;
            int[] ids = promptIds ?? tokenizer.Encode(prompt);
            LocatedSpans? located = identity.Supported && Parsed.Spans.Count > 0
                ? ChunkSpanLocator.Locate(state.ChatTemplate!, messages, options, prompt, Parsed.Spans, engineMessageOffset)
                : null;
            bool unavailable = false;
            var store = identity.Supported && Caller.Authenticated ? runtime.Store(state.Options.ChunkCache, out unavailable) : null;
            Plan = ChunkCachePipeline.Apply(Parsed, located, identity, tokenizer, prompt, ids, store, unavailable);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            ServerLog.Info($"chunk_cache: planning failed ({ex.GetType().Name}); annotations ignored");
            Plan = new ChunkCachePlan(Parsed.Requested, null, false, Parsed.Annotated, [], Parsed.Warnings, null,
                Parsed.Report, backend, 0);
        }
        return Plan.Error;
    }

    /// <summary>
    /// The response <c>chunk_cache</c> object, built once (later calls return the same object) and counted in
    /// <c>/metrics</c>. Null when the request asked for <c>report: "none"</c>.
    /// </summary>
    /// <param name="promptTokens">Prompt tokens the response reports.</param>
    /// <param name="timings">The engine's timings (prefix reuse and prefill time); null when it reported none.</param>
    /// <returns>The report or null.</returns>
    public ChunkCacheReportDto? Finish(int promptTokens, InferenceTimings? timings)
    {
        if (_finished)
            return _report;
        _finished = true;
        var plan = Plan ?? new ChunkCachePlan(Parsed.Requested, null, false, Parsed.Annotated, [], Parsed.Warnings, null,
            Parsed.Report, "cpu", 0);
        int prompt = promptTokens > 0 ? promptTokens : plan.PromptTokens;
        int cached = Math.Clamp(timings?.CachedTokenCount ?? 0, 0, prompt);
        ChunkCacheReportDto report;
        try
        {
            report = ChunkCacheReport.Build(plan, prompt, cached, timings?.PrefillTimeMs ?? 0, ChunkCostModel.For(plan.Backend));
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            // The response is already generated: never fail it over the report.
            ServerLog.Info($"chunk_cache: report failed ({ex.GetType().Name})");
            return null;
        }
        ChunkCacheMetrics.Record(report.Chunks, report.Tokens);
        if (string.Equals(plan.Report, "per_chunk", StringComparison.Ordinal) && Caller.Authenticated)
            report = report with { PerChunk = PerChunk(plan.Statuses, cached) };
        _report = string.Equals(plan.Report, "none", StringComparison.Ordinal) ? null : report;
        return _report;
    }

    /// <summary>
    /// Per-chunk status DTOs: a verified chunk wholly inside the first <paramref name="cachedTokens"/> prompt ids (the exact
    /// prefix the prompt cache restored) is <c>prefix</c>, as in the counts.
    /// </summary>
    /// <param name="statuses">The plan's statuses.</param>
    /// <param name="cachedTokens">Exact rows restored (0 when not known, e.g. for <c>explain</c>).</param>
    /// <returns>The DTOs, in document order.</returns>
    internal static ChunkCacheChunkStatusDto[] PerChunk(IReadOnlyList<SpanStatus> statuses, int cachedTokens) =>
        statuses.Select(s => new ChunkCacheChunkStatusDto
        {
            Id = s.Id,
            Status = s.Status != SpanStatus.Ignored && s.TokenStart >= 0 && (long)s.TokenStart + s.TokenCount <= cachedTokens
                ? SpanStatus.Prefix
                : s.Status,
            Why = s.Status == SpanStatus.Ignored ? s.Code : null,
        }).ToArray();

    /// <summary>The timings extension of an annotated request: <c>chunk_n</c> and <c>chunk_recomputed_n</c> present (0 in v1).</summary>
    /// <param name="timings">The timings DTO, or null.</param>
    /// <returns>The DTO with the chunk fields, or null.</returns>
    public static TimingsDto? WithChunkTimings(TimingsDto? timings) =>
        timings is null ? null : timings with { ChunkN = 0, ChunkRecomputedN = 0 };
}
