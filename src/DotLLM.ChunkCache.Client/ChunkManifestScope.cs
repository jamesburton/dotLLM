// SPDX-License-Identifier: MIT
using System.Text.Json.Nodes;

namespace DotLLM.ChunkCache.Client;

/// <summary>A registered part text and its spans (UTF-16 offsets into <paramref name="Text"/>).</summary>
/// <param name="Text">The text as registered.</param>
/// <param name="Spans">Its spans.</param>
public sealed record ChunkPart(string Text, IReadOnlyList<ChunkSpan> Spans);

/// <summary>What the body handler did with one request (the last attempt, when retried).</summary>
/// <param name="Annotated">Whether a <c>chunk_cache</c> manifest was sent.</param>
/// <param name="SkipReason">Why not, when <paramref name="Annotated"/> is false (see <see cref="ChunkSkipReasons"/>).</param>
/// <param name="RegisteredSpans">Spans the decorator handed over.</param>
/// <param name="SentSpans">Annotations in the manifest.</param>
/// <param name="NotLocated">Spans whose registered text was not found in any serialised part.</param>
/// <param name="HashMismatch">Spans whose text in the serialised body no longer hashes to the registered hash.</param>
/// <param name="Invalid">Spans out of range, empty, cutting a surrogate pair, or in a part holding a lone surrogate.</param>
/// <param name="Overlapping">Spans dropped because they overlap an earlier span of the same part.</param>
/// <param name="OverLimit">Spans dropped beyond the server's <c>max_chunks_per_request</c>.</param>
/// <param name="FellBackToPlain">A non-strict request was refused with a <c>chunk_*</c> error and resent without annotations.</param>
public sealed record ChunkManifestOutcome(bool Annotated, string? SkipReason, int RegisteredSpans, int SentSpans,
    int NotLocated, int HashMismatch, int Invalid, int Overlapping, int OverLimit, bool FellBackToPlain);

/// <summary>Values of <see cref="ChunkManifestOutcome.SkipReason"/>.</summary>
public static class ChunkSkipReasons
{
    /// <summary>The capability probe failed or the convention is absent (contract ruling 16).</summary>
    public const string NotCapable = "not_capable";

    /// <summary>The body already carries <c>chunk_cache</c> or an inline <c>chunk</c> (left as is).</summary>
    public const string AlreadyAnnotated = "already_annotated";

    /// <summary>No registered span survived location and re-hashing.</summary>
    public const string NoSpans = "no_spans";

    /// <summary>The body is not a JSON object with <c>messages</c>, or is encoded (e.g. gzip).</summary>
    public const string NotJson = "not_json";
}

/// <summary>
/// The hand-off from <see cref="ChunkAnnotatingChatClient"/> to <see cref="ChunkCacheBodyHandler"/> over an
/// <see cref="AsyncLocal{T}"/> (spec 8.3): the registered parts of this call and the request options go down; the
/// response's <c>chunk_cache</c> report and the handler's outcome come back on this object.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>The parts are a snapshot taken when the call starts, so every retry of the call (an HTTP-level retry, or the
/// SDK's own) derives the same manifest from the same body, whatever happens to the registry meanwhile.</item>
/// <item>The handler captures the scope object when the request is sent; a streamed report is written to it later,
/// while the response is read, without needing the AsyncLocal to flow into the reader.</item>
/// <item>Retries overwrite <see cref="Outcome"/>; it describes the last attempt.</item>
/// </list>
/// </remarks>
public sealed class ChunkManifestScope : IDisposable
{
    private static readonly AsyncLocal<ChunkManifestScope?> s_current = new();

    private readonly ChunkManifestScope? _previous;
    private JsonObject? _report;
    private ChunkManifestOutcome? _outcome;
    private bool _disposed;

    private ChunkManifestScope(IReadOnlyList<ChunkPart> parts, ChunkRequestOptions options, ChunkManifestScope? previous)
    {
        Parts = parts;
        Options = options;
        _previous = previous;
    }

    /// <summary>The scope of the current asynchronous flow, if any.</summary>
    public static ChunkManifestScope? Current => s_current.Value;

    /// <summary>Opens a scope for one model call and makes it current; dispose it when the call ends.</summary>
    /// <param name="parts">Registered part texts found in the call's messages (copied).</param>
    /// <param name="options">The request-level options.</param>
    public static ChunkManifestScope Begin(IReadOnlyList<ChunkPart> parts, ChunkRequestOptions options)
    {
        ArgumentNullException.ThrowIfNull(parts);
        ArgumentNullException.ThrowIfNull(options);
        var scope = new ChunkManifestScope([.. parts], options, s_current.Value);
        s_current.Value = scope;
        return scope;
    }

    /// <summary>The registered parts found in this call's messages.</summary>
    public IReadOnlyList<ChunkPart> Parts { get; }

    /// <summary>The request-level options.</summary>
    public ChunkRequestOptions Options { get; }

    /// <summary>
    /// The response's top-level <c>chunk_cache</c> object, once seen (blocking: when the response returns; streaming:
    /// as soon as the event carrying it has been read).
    /// </summary>
    public JsonObject? ResponseReport => Volatile.Read(ref _report);

    /// <summary>What the handler did on the last attempt; <see langword="null"/> if no request went through the handler.</summary>
    public ChunkManifestOutcome? Outcome => Volatile.Read(ref _outcome);

    /// <summary>Makes <paramref name="scope"/> current again (a streaming enumeration resumed on the caller's context).</summary>
    internal static void Enter(ChunkManifestScope scope) => s_current.Value = scope;

    internal void SetReport(JsonObject report) => Volatile.Write(ref _report, report);

    internal void SetOutcome(ChunkManifestOutcome outcome) => Volatile.Write(ref _outcome, outcome);

    /// <summary>Restores the scope that was current when this one began.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (ReferenceEquals(s_current.Value, this))
            s_current.Value = _previous;
    }
}
