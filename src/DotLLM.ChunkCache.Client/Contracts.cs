// SPDX-License-Identifier: MIT
using System.Text.Json.Nodes;

namespace DotLLM.ChunkCache.Client;

// The harness-facing abstractions of the chunk-cache integration spec, section 8.2, verbatim. These are .NET-side
// types: offsets here are UTF-16 chars; the body handler converts them to the wire's UTF-8 byte offsets.

/// <summary>
/// One chunk inside one message part, identified by its exact text. Offsets are UTF-16 chars here (.NET strings); the
/// body handler converts them to the wire's UTF-8 byte offsets.
/// </summary>
/// <param name="Id">The caller's label (for example <c>path:startLine-endLine@offset</c>). Telemetry only; never used for lookup. At most 256 chars on the wire.</param>
/// <param name="Hash"><c>"sha256:" + lowercase hex</c> of the UTF-8 bytes of the span text (<see cref="ChunkHash.Of"/>).</param>
/// <param name="Start">UTF-16 offset of the span within the part text.</param>
/// <param name="End">UTF-16 end offset (exclusive) within the part text; <see langword="null"/> means to the end of the part.</param>
/// <param name="Chunker">A wire-eligible (text-first) chunker name, for example <c>symbol-512</c>.</param>
/// <param name="Source">Provenance for telemetry; never read by the server for lookup.</param>
/// <param name="Volatile">Hint: volatile chunks are not queued for background encode.</param>
/// <param name="NoStore">Never persist this chunk's K/V or sighting (secrets).</param>
public sealed record ChunkSpan(
    string Id, string Hash,
    int Start = 0, int? End = null,
    string? Chunker = null, ChunkSourceInfo? Source = null,
    bool Volatile = false, bool NoStore = false);

/// <summary>Where a chunk came from (telemetry only).</summary>
/// <param name="Path">Source path, relative to the harness's source root.</param>
/// <param name="StartLine">First line (1-based).</param>
/// <param name="EndLine">Last line (1-based, inclusive).</param>
/// <param name="Revision">An optional revision (commit, content hash).</param>
public sealed record ChunkSourceInfo(string Path, int StartLine, int EndLine, string? Revision);

/// <summary>
/// Reads a file the harness is about to show the model and returns the text plus its chunk spans, chunked with a
/// wire-eligible (text-first) chunker over the same normalised text the store indexes.
/// </summary>
public interface IChunkSource
{
    /// <summary>Reads <paramref name="path"/>, optionally limited to <paramref name="window"/>.</summary>
    ValueTask<ChunkedText> ReadAsync(string path, LineRange? window, CancellationToken ct);
}

/// <summary>A file's text plus the spans that may be annotated.</summary>
/// <param name="Text">The text to show the model (LF-normalised, as indexed).</param>
/// <param name="Spans">Spans within <paramref name="Text"/>.</param>
/// <param name="ContentSha256">SHA-256 of the file content the spans were computed from.</param>
/// <param name="Notes">Human-readable notes, e.g. "2 chunks cut by the window: not annotated".</param>
public sealed record ChunkedText(string Text, IReadOnlyList<ChunkSpan> Spans, string ContentSha256,
                                 IReadOnlyList<string> Notes);

/// <summary>An inclusive range of 1-based line numbers.</summary>
/// <param name="Start">First line.</param>
/// <param name="End">Last line.</param>
public readonly record struct LineRange(int Start, int End);

/// <summary>
/// Source-change notifications. Identity is content, so this is an optimisation (re-read and prewarm early); without
/// it, stale chunks simply miss.
/// </summary>
public interface IInvalidationSignal
{
    /// <summary>Streams invalidations until <paramref name="ct"/> is cancelled.</summary>
    IAsyncEnumerable<ChunkInvalidation> ReadAllAsync(CancellationToken ct);
}

/// <summary>One source change.</summary>
/// <param name="Path">The changed path.</param>
/// <param name="Kind">What happened.</param>
/// <param name="NewContentSha256">The new content hash, when known.</param>
/// <param name="Origin">Who noticed: <c>"edit-tool"</c>, <c>"fs-watch"</c>, <c>"git"</c>.</param>
public sealed record ChunkInvalidation(string Path, InvalidationKind Kind, string? NewContentSha256, string Origin);

/// <summary>Kinds of source change.</summary>
public enum InvalidationKind
{
    /// <summary>Content changed.</summary>
    Modified,

    /// <summary>The file is gone.</summary>
    Deleted,

    /// <summary>The file moved.</summary>
    Renamed,

    /// <summary>Something changed; re-read to find out.</summary>
    Unknown,
}

/// <summary>Chooses the request policy. The default implementation (<see cref="ExactPolicy"/>) returns <c>Policy = "exact"</c>.</summary>
public interface IRecomputePolicy
{
    /// <summary>Decides the <c>chunk_cache</c> request options for one request.</summary>
    ChunkRequestOptions Decide(ChunkRequestContext context);
}

/// <summary>What a <see cref="IRecomputePolicy"/> may consider.</summary>
/// <param name="ModelId">The served model id.</param>
/// <param name="PromptTokensEstimate">Estimated prompt tokens.</param>
/// <param name="ChunkTokens">Estimated tokens inside annotated spans.</param>
/// <param name="Server">The server's capabilities, or <see langword="null"/> when the convention is absent.</param>
/// <param name="IsSubAgent">Whether the request comes from a sub-agent.</param>
/// <param name="AfterCompaction">Whether the context was just compacted.</param>
/// <param name="UserOptedIntoApproximation">Whether the user explicitly allowed lossy reuse.</param>
public sealed record ChunkRequestContext(string ModelId, int PromptTokensEstimate, int ChunkTokens,
    ChunkCacheCapabilities? Server, bool IsSubAgent, bool AfterCompaction, bool UserOptedIntoApproximation);

/// <summary>The request-level <c>chunk_cache</c> options (spec 5.2). Defaults are the wire defaults.</summary>
/// <param name="Policy"><c>exact</c> (default), <c>shadow</c>, <c>balanced</c>, <c>fast</c> or <c>auto</c>.</param>
/// <param name="Selector"><c>chunk_starts</c> or <c>query_relevance</c>; <see langword="null"/> = server default.</param>
/// <param name="Budget">Recompute budget in [0, 1]; <see langword="null"/> = server default.</param>
/// <param name="OnMiss"><c>prefill</c> (default) or <c>encode</c>.</param>
/// <param name="Strict">Errors instead of degradation.</param>
/// <param name="Report"><c>none</c>, <c>summary</c> (default, counts only) or <c>per_chunk</c>.</param>
public sealed record ChunkRequestOptions(string Policy = "exact", string? Selector = null, double? Budget = null,
    string OnMiss = "prefill", bool Strict = false, string Report = "summary");

/// <summary>Control plane over the server's <c>/v1/chunk-cache</c> endpoints (plus local index operations).</summary>
public interface IChunkCacheClient
{
    /// <summary>
    /// Probes <c>GET /v1/chunk-cache</c>. <see langword="null"/> = convention absent. Implementations may throw for a
    /// present server that refuses or fails; contract ruling 16 makes <b>any</b> exception mean "not capable" as well, so
    /// the caller strips annotations in both cases. <see cref="ChunkCacheProbe.ProbeAsync"/> applies that rule.
    /// </summary>
    ValueTask<ChunkCacheCapabilities?> GetCapabilitiesAsync(CancellationToken ct);

    /// <summary>Selects chunks from a local index (a local operation; the server has no select endpoint).</summary>
    ValueTask<SelectResult> SelectAsync(SelectRequest request, CancellationToken ct);

    /// <summary>Queues isolated encodes on the server. Sends text only, never hashes.</summary>
    ValueTask<PrewarmResult> PrewarmAsync(PrewarmRequest request, CancellationToken ct);

    /// <summary>Index and store status (a local operation; the server has no status endpoint).</summary>
    ValueTask<StatusResult> StatusAsync(StatusRequest request, CancellationToken ct);

    /// <summary>Evicts through the server, which holds the write lock.</summary>
    ValueTask<EvictResult> EvictAsync(EvictRequest request, CancellationToken ct);

    /// <summary>A dry run of the policy for a full request body (text only).</summary>
    ValueTask<ExplainResult> ExplainAsync(JsonObject requestBody, CancellationToken ct);
}

/// <summary>
/// The default <see cref="IRecomputePolicy"/>: always the wire defaults (<c>exact</c>, <c>prefill</c>, <c>summary</c>,
/// not strict, server-default selector and budget), whatever the context says. Nothing lossy without an explicit opt-in,
/// and opting in is a different policy, not this one.
/// </summary>
public sealed class ExactPolicy : IRecomputePolicy
{
    private static readonly ChunkRequestOptions s_defaults = new();

    /// <summary>A shared instance.</summary>
    public static ExactPolicy Instance { get; } = new();

    /// <inheritdoc />
    public ChunkRequestOptions Decide(ChunkRequestContext context) => s_defaults;
}
