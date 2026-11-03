// SPDX-License-Identifier: MIT
using System.Text.Json.Serialization;

namespace DotLLM.ChunkCache.Client;

// The control-plane tool schemas of spec 7.2, snake_case on the wire (ChunkCacheJsonContext). Inputs keep the tools'
// common `store` / `server` parameters so the CLI and the MCP server can bind them; the HTTP client sends only the
// fields the server endpoint takes.
//
// JSON DTO rule (dotLLM CLAUDE.md, #462): no `init` property carries an initializer. Optional inputs are nullable and
// their defaults are resolved by the `Effective*` accessors, which are [JsonIgnore].

/// <summary>Input of <c>chunkcache_select</c> (spec 7.2): <c>ContextSelector.Select</c> over a local index.</summary>
public sealed record SelectRequest
{
    /// <summary>The store root; default from config.</summary>
    public string? Store { get; init; }

    /// <summary>The dotLLM base URL; when given, <c>stored</c> is answered by the server.</summary>
    public string? Server { get; init; }

    /// <summary>The solution (<c>.sln</c>/<c>.slnx</c>/<c>.csproj</c>) or folder whose index to use.</summary>
    public string? Path { get; init; }

    /// <summary>The chunker whose index to use; <see cref="EffectiveChunker"/> when absent.</summary>
    public string? Chunker { get; init; }

    /// <summary>Path globs to select.</summary>
    public IReadOnlyList<string>? Paths { get; init; }

    /// <summary>Path globs to exclude.</summary>
    public IReadOnlyList<string>? Exclude { get; init; }

    /// <summary>A symbol to start from.</summary>
    public string? Symbol { get; init; }

    /// <summary>Reference-graph hops from <see cref="Symbol"/> (default 1, at most 100).</summary>
    public int? Hops { get; init; }

    /// <summary>Only chunks of files changed since this git revision.</summary>
    public string? ChangedSince { get; init; }

    /// <summary>Trim the selection to this many tokens.</summary>
    public int? TokenBudget { get; init; }

    /// <summary>Include each chunk's text (default false).</summary>
    public bool? IncludeText { get; init; }

    /// <summary><c>rank</c> or <c>document</c> (default).</summary>
    public string? Order { get; init; }

    /// <summary><see cref="Chunker"/> or <c>symbol-512</c>.</summary>
    [JsonIgnore] public string EffectiveChunker => Chunker ?? ChunkCacheDefaults.Chunker;

    /// <summary><see cref="Hops"/> or 1.</summary>
    [JsonIgnore] public int EffectiveHops => Hops ?? 1;

    /// <summary><see cref="IncludeText"/> or false.</summary>
    [JsonIgnore] public bool EffectiveIncludeText => IncludeText ?? false;

    /// <summary><see cref="Order"/> or <c>document</c>.</summary>
    [JsonIgnore] public string EffectiveOrder => Order ?? "document";
}

/// <summary>Output of <c>chunkcache_select</c>.</summary>
public sealed record SelectResult
{
    /// <summary>The selected chunks.</summary>
    public IReadOnlyList<SelectedChunk>? Chunks { get; init; }

    /// <summary>Tokens in the selection.</summary>
    public int SelectedTokens { get; init; }

    /// <summary>Tokens in all candidates before trimming.</summary>
    public int CandidateTokens { get; init; }

    /// <summary>Candidates trimmed by the budget.</summary>
    public int Trimmed { get; init; }

    /// <summary>Patterns that matched nothing.</summary>
    public IReadOnlyList<string>? UnmatchedPatterns { get; init; }

    /// <summary>Notes.</summary>
    public IReadOnlyList<string>? Notes { get; init; }
}

/// <summary>One chunk of a <see cref="SelectResult"/>.</summary>
public sealed record SelectedChunk
{
    /// <summary>The chunk label.</summary>
    public string? Id { get; init; }

    /// <summary>Source path.</summary>
    public string? Path { get; init; }

    /// <summary>First line.</summary>
    public int StartLine { get; init; }

    /// <summary>Last line.</summary>
    public int EndLine { get; init; }

    /// <summary>Token count.</summary>
    public int Tokens { get; init; }

    /// <summary>The wire hash of the chunk's LF-normalised <c>plain</c> text.</summary>
    public string? Hash { get; init; }

    /// <summary>Why it was selected.</summary>
    public string? Reason { get; init; }

    /// <summary>Detail for <see cref="Reason"/>.</summary>
    public string? Detail { get; init; }

    /// <summary>Reference-graph hop.</summary>
    public int? Hop { get; init; }

    /// <summary>Whether the server holds K/V for it (only when a server was given).</summary>
    public bool? Stored { get; init; }

    /// <summary>The chunk text (only with <c>include_text</c>).</summary>
    public string? Text { get; init; }
}

/// <summary>Input of <c>chunkcache_prewarm</c>: text only. There is no hash-to-key map, and a hash-only API would be a membership oracle.</summary>
public sealed record PrewarmRequest
{
    /// <summary>The store root used to resolve <see cref="Select"/> locally.</summary>
    public string? Store { get; init; }

    /// <summary>The dotLLM base URL (tool-level; the HTTP client already has one).</summary>
    public string? Server { get; init; }

    /// <summary>A select input; its chunk texts are read locally and sent as text.</summary>
    public SelectRequest? Select { get; init; }

    /// <summary>Texts to encode.</summary>
    public IReadOnlyList<PrewarmText>? Texts { get; init; }

    /// <summary>Stop queueing after this many tokens.</summary>
    public int? MaxTokens { get; init; }
}

/// <summary>One text to prewarm.</summary>
public sealed record PrewarmText
{
    /// <summary>The caller's label (telemetry only).</summary>
    public string? Id { get; init; }

    /// <summary>The exact chunk text.</summary>
    public string? Text { get; init; }

    /// <summary>
    /// Optional, advisory provenance (contract ruling 14): when present the server applies its deny-list to
    /// <c>source.path</c>, but a client can omit it, so the server never relies on it.
    /// </summary>
    public PrewarmSource? Source { get; init; }
}

/// <summary>A prewarm item's <c>source</c>: <c>{path}</c> only (contract ruling 14).</summary>
public sealed record PrewarmSource
{
    /// <summary>The source path the text came from.</summary>
    public string? Path { get; init; }
}

/// <summary>Output of <c>chunkcache_prewarm</c>.</summary>
public sealed record PrewarmResult
{
    /// <summary>Texts received.</summary>
    public int Requested { get; init; }

    /// <summary>Texts whose K/V is already stored.</summary>
    public int AlreadyStored { get; init; }

    /// <summary>Texts queued for encode.</summary>
    public int Queued { get; init; }

    /// <summary>Tokens queued.</summary>
    public long QueuedTokens { get; init; }

    /// <summary>Texts refused (deny-list, budget, ...).</summary>
    public int Refused { get; init; }

    /// <summary>Why, one entry per refusal reason.</summary>
    public IReadOnlyList<string>? RefusedReasons { get; init; }
}

/// <summary>Input of <c>chunkcache_status</c> (a local index operation).</summary>
public sealed record StatusRequest
{
    /// <summary>The store root.</summary>
    public string? Store { get; init; }

    /// <summary>The dotLLM base URL.</summary>
    public string? Server { get; init; }

    /// <summary>The solution or folder.</summary>
    public string? Path { get; init; }

    /// <summary>The chunker; <see cref="EffectiveChunker"/> when absent.</summary>
    public string? Chunker { get; init; }

    /// <summary>List each missing chunk (default false).</summary>
    public bool? ListMissing { get; init; }

    /// <summary><see cref="Chunker"/> or <c>symbol-512</c>.</summary>
    [JsonIgnore] public string EffectiveChunker => Chunker ?? ChunkCacheDefaults.Chunker;

    /// <summary><see cref="ListMissing"/> or false.</summary>
    [JsonIgnore] public bool EffectiveListMissing => ListMissing ?? false;
}

/// <summary>Output of <c>chunkcache_status</c>.</summary>
public sealed record StatusResult
{
    /// <summary>Chunks in the index.</summary>
    public int TotalChunks { get; init; }

    /// <summary>Indexed chunks without stored K/V.</summary>
    public int MissingChunks { get; init; }

    /// <summary>Tokens in <see cref="MissingChunks"/>.</summary>
    public long MissingTokens { get; init; }

    /// <summary>Whether the index matches the files on disk.</summary>
    public bool UpToDate { get; init; }

    /// <summary>Files changed since indexing.</summary>
    public IReadOnlyList<string>? StaleFiles { get; init; }

    /// <summary>Files not yet indexed.</summary>
    public IReadOnlyList<string>? NewFiles { get; init; }

    /// <summary>Indexed files that no longer exist.</summary>
    public IReadOnlyList<string>? DeletedFiles { get; init; }

    /// <summary>The missing chunks (only with <c>list_missing</c>).</summary>
    public IReadOnlyList<MissingChunk>? Missing { get; init; }

    /// <summary>Bytes in the store.</summary>
    public long StoreBytes { get; init; }

    /// <summary>The store budget.</summary>
    public long BudgetBytes { get; init; }

    /// <summary>Stored entries.</summary>
    public long Entries { get; init; }
}

/// <summary>One missing chunk of a <see cref="StatusResult"/>.</summary>
public sealed record MissingChunk
{
    /// <summary>Source path.</summary>
    public string? Path { get; init; }

    /// <summary>First line.</summary>
    public int StartLine { get; init; }

    /// <summary>Last line.</summary>
    public int EndLine { get; init; }

    /// <summary>Token count.</summary>
    public int Tokens { get; init; }
}

/// <summary>Input of <c>chunkcache_evict</c>, executed by the server (<c>POST /v1/chunk-cache/evict</c>).</summary>
public sealed record EvictRequest
{
    /// <summary>The store root (tool-level).</summary>
    public string? Store { get; init; }

    /// <summary>The dotLLM base URL (tool-level).</summary>
    public string? Server { get; init; }

    /// <summary><c>orphans</c> (default), <c>path</c>, <c>namespace</c> or <c>all</c>.</summary>
    public string? Mode { get; init; }

    /// <summary>Glob; <c>mode=path</c> forgets every entry first seen under it.</summary>
    public string? Path { get; init; }

    /// <summary>The namespace for <c>mode=namespace</c>.</summary>
    public string? Namespace { get; init; }

    /// <summary>Report only (default <see langword="true"/>).</summary>
    public bool? DryRun { get; init; }

    /// <summary><see cref="Mode"/> or <c>orphans</c>.</summary>
    [JsonIgnore] public string EffectiveMode => Mode ?? "orphans";

    /// <summary><see cref="DryRun"/> or <see langword="true"/>: nothing is deleted unless the caller says so.</summary>
    [JsonIgnore] public bool EffectiveDryRun => DryRun ?? true;
}

/// <summary>Output of <c>chunkcache_evict</c>.</summary>
public sealed record EvictResult
{
    /// <summary>Entries evicted (or that would be, under a dry run).</summary>
    public int Evicted { get; init; }

    /// <summary>Bytes freed (or that would be).</summary>
    public long FreedBytes { get; init; }

    /// <summary>Entries kept because an index or the ledger references them.</summary>
    public int KeptReferenced { get; init; }

    /// <summary>Whether this was a dry run.</summary>
    public bool? DryRun { get; init; }
}

/// <summary>Output of <c>chunkcache_explain</c> (<c>POST /v1/chunk-cache/explain</c>).</summary>
public sealed record ExplainResult
{
    /// <summary>The policy the server would apply.</summary>
    public string? PolicyWouldApply { get; init; }

    /// <summary>Why, when it differs from the requested one.</summary>
    public string? Reason { get; init; }

    /// <summary>Per-chunk status (only for an authenticated caller in its own namespace).</summary>
    public IReadOnlyList<ExplainChunk>? Chunks { get; init; }

    /// <summary>Predicted timings.</summary>
    public ExplainPredictedMs? PredictedMs { get; init; }

    /// <summary>Warning codes (spec 5.8).</summary>
    public IReadOnlyList<string>? Warnings { get; init; }
}

/// <summary>One chunk of an <see cref="ExplainResult"/>.</summary>
public sealed record ExplainChunk
{
    /// <summary>The chunk's id.</summary>
    public string? Id { get; init; }

    /// <summary><c>prefix</c>, <c>hit</c>, <c>miss</c> or <c>ignored</c>.</summary>
    public string? Status { get; init; }

    /// <summary>Why (hash mismatch, template altered, not wire-eligible, too short, other model, CRLF vs index).</summary>
    public string? Why { get; init; }
}

/// <summary><c>predicted_ms</c> of an <see cref="ExplainResult"/>.</summary>
public sealed record ExplainPredictedMs
{
    /// <summary>Predicted time with reuse.</summary>
    public double? Reuse { get; init; }

    /// <summary>Predicted full prefill time.</summary>
    public double? FullPrefill { get; init; }
}

/// <summary>Defaults shared by the control-plane inputs.</summary>
public static class ChunkCacheDefaults
{
    /// <summary>The default wire chunker.</summary>
    public const string Chunker = "symbol-512";
}

// ---- HTTP request bodies: only what each endpoint takes (no tool-level `store` / `server`). ----

/// <summary>Body of <c>POST /v1/chunk-cache/prewarm</c>.</summary>
internal sealed record PrewarmBody
{
    public IReadOnlyList<PrewarmText>? Texts { get; init; }

    public int? MaxTokens { get; init; }
}

/// <summary>Body of <c>POST /v1/chunk-cache/evict</c>; mode and dry_run are always sent resolved.</summary>
internal sealed record EvictBody
{
    public string? Mode { get; init; }

    public string? Path { get; init; }

    public string? Namespace { get; init; }

    public bool? DryRun { get; init; }
}

/// <summary>The OpenAI-style error envelope dotLLM uses: <c>{"error": {"message", "type", "code"}}</c>.</summary>
internal sealed record ErrorEnvelope
{
    public ErrorBody? Error { get; init; }
}

/// <summary>The inner error object.</summary>
internal sealed record ErrorBody
{
    public string? Message { get; init; }

    public string? Type { get; init; }

    public string? Code { get; init; }
}
