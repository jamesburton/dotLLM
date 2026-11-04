namespace DotLLM.Server.ChunkCache;

/// <summary>Constants of the <c>chunk_cache</c> wire convention, v1 (spec §5.2, §5.3).</summary>
internal static class ChunkCacheWire
{
    /// <summary>The only convention version this server speaks.</summary>
    public const int Version = 1;

    /// <summary>Longest accepted <c>id</c>, in Unicode scalar values (characters).</summary>
    public const int MaxIdLength = 256;

    /// <summary>Default of <c>min_chunk_tokens</c>.</summary>
    public const int DefaultMinChunkTokens = 32;

    /// <summary>
    /// The single-user namespace (<c>ChunkNamespace.Default</c> in <c>DotLLM.ChunkReuse</c>, copied until the server
    /// references it). The caller's namespace is server-assigned: until authentication exists (a later task), every
    /// caller gets this one.
    /// </summary>
    public const string DefaultNamespace = "";

    /// <summary>Prefix of every <c>hash</c>: <c>"sha256:"</c> + 64 lowercase hex digits.</summary>
    public const string HashPrefix = "sha256:";

    /// <summary>
    /// The text-first chunkers (spec §5.3 item 4) a span may name. A copy of <c>WireChunking.Eligible</c> in
    /// <c>DotLLM.ChunkReuse</c> until the server references that assembly; <c>lines-512</c> joins it in a later task.
    /// </summary>
    public static IReadOnlySet<string> WireChunkers { get; } = new HashSet<string>(StringComparer.Ordinal) { "symbol-512" };
}

/// <summary>The requested reuse policy (spec §5.5). Only <see cref="Exact"/> and <see cref="Shadow"/> run in v1.</summary>
internal enum ChunkPolicy
{
    /// <summary>Output identical to the unannotated request (the default).</summary>
    Exact,

    /// <summary>As <see cref="Exact"/>, plus a reuse plan that is computed but not applied.</summary>
    Shadow,

    /// <summary>Reuse with recompute; degrades to <see cref="Exact"/> until applied reuse ships.</summary>
    Balanced,

    /// <summary>Lossy reuse; degrades to <see cref="Exact"/> until applied reuse ships.</summary>
    Fast,

    /// <summary>Reuse when both gates pass; degrades to <see cref="Exact"/> until applied reuse ships.</summary>
    Auto,
}

/// <summary>
/// Where a span lies in the engine's message list: the <see cref="DotLLM.Tokenizers.ChatMessage"/> array the surface's
/// converter produced (<c>RequestConverter.ToMessages</c> / <c>AnthropicConverter.ToMessages</c>), <b>before</b>
/// <c>ProfileSystemPrompt.Apply</c> may prepend a profile system message. Offsets are UTF-16 chars into that message's
/// <c>Content</c>, end exclusive.
/// </summary>
/// <param name="EngineMessage">Index into the converter's output.</param>
/// <param name="StartChar">First char of the span in that message's content.</param>
/// <param name="EndChar">One past the last char.</param>
internal readonly record struct EngineSpan(int EngineMessage, int StartChar, int EndChar);

/// <summary>An accepted chunk annotation, validated against the text the server received.</summary>
/// <param name="Message">Wire index into the request's <c>messages</c>.</param>
/// <param name="Part">Wire index into that message's content (0 for string content).</param>
/// <param name="StartByte">UTF-8 byte offset of the span in the part's decoded text.</param>
/// <param name="EndByte">One past the span's last byte.</param>
/// <param name="Id">The caller's label; telemetry only, never used for lookup.</param>
/// <param name="Hash">The verified hash (<c>sha256:</c> + lowercase hex of the span's UTF-8 bytes).</param>
/// <param name="Chunker">The named chunker, when given (always wire-eligible here).</param>
/// <param name="Volatile"><c>stability: "volatile"</c>.</param>
/// <param name="NoStore"><c>no_store: true</c>.</param>
/// <param name="Text">The span text, exactly as it sits in the engine message.</param>
/// <param name="Engine">The span in the engine's message list.</param>
/// <param name="InnerPart">
/// Anthropic only: index of a text block inside a <c>tool_result</c>'s content array (inline placement), else -1.
/// </param>
internal sealed record ChunkSpanRef(
    int Message, int Part, int StartByte, int EndByte, string Id, string Hash, string? Chunker, bool Volatile,
    bool NoStore, string Text, EngineSpan Engine, int InnerPart = -1)
{
    // Never print the span text or its hash (spec 10: chunk content stays out of logs).
    private bool PrintMembers(System.Text.StringBuilder builder)
    {
        builder.Append(System.Globalization.CultureInfo.InvariantCulture,
            $"Message = {Message}, Part = {Part}, InnerPart = {InnerPart}, StartByte = {StartByte}, EndByte = {EndByte}, Engine = {Engine}, TextLength = {Text.Length}");
        return true;
    }
}

/// <summary>A warning code and how many annotations (or requests, for request-level codes) it covers.</summary>
internal sealed record ChunkWarning(string Code, int Count);

/// <summary>
/// Why one annotation was dropped. <see cref="Index"/> counts annotations in document order, inline first then
/// manifest (the contract's <c>expected.json</c> numbering); <see cref="Kind"/> is the finer cause
/// (<c>overlap</c>, <c>multibyte_boundary</c>, <c>part_out_of_range</c>, <c>non_text</c>, <c>range_invalid</c>,
/// <c>hash_mismatch</c>, <c>id_invalid</c>, <c>malformed</c>, <c>version</c>, <c>positioning</c>,
/// <c>not_wire_eligible</c>, <c>too_many</c>, <c>not_carried</c>).
/// </summary>
internal sealed record ChunkDefect(int Index, string Code, string Kind);

/// <summary>A strict-mode error: the HTTP status, the error type and the §5.8 code.</summary>
internal sealed record ChunkCacheError(int Status, string Type, string Code, string Message);

/// <summary>
/// The parsed <c>chunk_cache</c> annotations of one request. Produced only with <c>--chunk-cache</c> on and at least
/// one annotation (a <c>chunk_cache</c> object or an inline <c>chunk</c>).
/// </summary>
/// <param name="Requested">The requested policy (unknown values parse as <see cref="ChunkPolicy.Exact"/> with a warning).</param>
/// <param name="Spans">The accepted spans, in document order.</param>
/// <param name="QueryText">The query (spec §5.6): <c>query.text</c>, the text of <c>query.message</c>, or the last user message.</param>
/// <param name="Strict"><c>strict: true</c>.</param>
/// <param name="Report"><c>none</c>, <c>summary</c> (default) or <c>per_chunk</c>.</param>
/// <param name="Namespace">The store namespace, from the caller's credentials (never from the body).</param>
/// <param name="Warnings">Warning codes with counts, in order of first occurrence.</param>
/// <param name="Error">With <see cref="Strict"/>: the error for the first defect in document order; else null.</param>
internal sealed record ParsedChunkRequest(
    ChunkPolicy Requested, IReadOnlyList<ChunkSpanRef> Spans, string? QueryText, bool Strict, string Report,
    string Namespace, IReadOnlyList<ChunkWarning> Warnings, ChunkCacheError? Error)
{
    /// <summary>Annotations seen (accepted or not); the denominator of <c>chunks.annotated</c>.</summary>
    public int Annotated { get; init; }

    /// <summary><c>min_chunk_tokens</c> (default 32).</summary>
    public int MinChunkTokens { get; init; } = ChunkCacheWire.DefaultMinChunkTokens;

    /// <summary><c>on_miss</c>: <c>prefill</c> (default) or <c>encode</c>.</summary>
    public string OnMiss { get; init; } = "prefill";

    /// <summary>
    /// <c>recompute.budget</c> (0.0–1.0) when the request gave one: the shadow plan's budget (ruling 50); null = the
    /// server default.
    /// </summary>
    public double? RecomputeBudget { get; init; }

    /// <summary>Every dropped annotation with its cause, in document order.</summary>
    public IReadOnlyList<ChunkDefect> Defects { get; init; } = [];

    // Never print the namespace, the query or span text (spec 10); counts and codes only.
    private bool PrintMembers(System.Text.StringBuilder builder)
    {
        builder.Append(System.Globalization.CultureInfo.InvariantCulture,
            $"Requested = {Requested}, Spans = {Spans.Count}, Annotated = {Annotated}, Strict = {Strict}, Report = {Report}, Warnings = [{string.Join(", ", Warnings.Select(w => $"{w.Code}x{w.Count}"))}], Error = {Error?.Code}");
        return true;
    }
}
