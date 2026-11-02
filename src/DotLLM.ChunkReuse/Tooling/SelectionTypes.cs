using System.Collections;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.Store;

namespace DotLLM.ChunkReuse.Tooling;

/// <summary>
/// What to put in a context (<see cref="ContextSelector.Select"/>). The criteria are combined as a <b>union</b>: a chunk is a
/// candidate when it matches <see cref="Paths"/>, lies in the neighbourhood of <see cref="Symbol"/>, or is in a file changed
/// since <see cref="ChangedSince"/>. <see cref="Exclude"/> and <see cref="TokenBudget"/> then
/// restrict the union. At least one of <see cref="Paths"/>, <see cref="Symbol"/> and <see cref="ChangedSince"/> is required.
/// </summary>
public sealed record SelectionRequest
{
    /// <summary>
    /// Path globs (see <see cref="PathGlob"/>) over the corpus paths (<see cref="CodeCorpus.SourceFile.Path"/>, relative to the corpus root).
    /// A pattern starting with <c>!</c> removes what earlier and later include patterns matched (path criterion only).
    /// </summary>
    public IReadOnlyList<string> Paths { get; init; } = [];

    /// <summary>Path globs whose files are never selected (by any criterion) and never traversed by <see cref="Hops"/>.</summary>
    public IReadOnlyList<string> Exclude { get; init; } = [];

    /// <summary>
    /// A declared symbol: a simple name (<c>Select</c>) or a dotted suffix of its qualified name (<c>ContextSelector.Select</c>,
    /// <c>DotLLM.ChunkReuse.Tooling.ContextSelector</c>). Generic arguments (<c>List&lt;T&gt;</c>), a trailing parameter list
    /// (<c>Select(…)</c>) and <c>global::</c> are ignored; matching is case-sensitive. Every declaration that matches (overloads,
    /// partial parts, same name in several types) is a definition.
    /// </summary>
    public string? Symbol { get; init; }

    /// <summary>Reference-graph hops from the definition chunks of <see cref="Symbol"/> (0 = the definition only). Default 1, at most <see cref="ContextSelector.MaxHops"/>.</summary>
    public int Hops { get; init; } = 1;

    /// <summary>A git revision (branch, tag, SHA, <c>HEAD~3</c>); every chunk of a file that differs from it in the working tree (or is untracked) is selected.</summary>
    public string? ChangedSince { get; init; }

    /// <summary>Maximum total tokens of the selection (null: no limit). Trimming order: see <see cref="ContextSelector"/>.</summary>
    public int? TokenBudget { get; init; }

    /// <summary>Edge kinds the neighbourhood walk follows (null: <see cref="SymbolEdgeKind.Call"/>, <see cref="SymbolEdgeKind.Use"/>, <see cref="SymbolEdgeKind.Definition"/>; imports are left out because a namespace links everything that uses it).</summary>
    public IReadOnlyCollection<SymbolEdgeKind>? EdgeKinds { get; init; }

    /// <summary>The <see cref="ChunkRef.Kind"/> to stamp on the result (the graph does not know its chunker).</summary>
    public ChunkKind ChunkKind { get; init; } = ChunkKind.Other;
}

/// <summary>Why a chunk was selected (flags: a chunk can match several criteria).</summary>
[Flags]
public enum SelectionReason
{
    /// <summary>No reason (not used in results).</summary>
    None = 0,

    /// <summary>Holds the name of a declaration that matches <see cref="SelectionRequest.Symbol"/>.</summary>
    Definition = 1,

    /// <summary>Overlaps such a declaration but does not hold its name (the rest of a long method, the members of a matched type).</summary>
    DefinitionBody = 2,

    /// <summary>In a file changed since <see cref="SelectionRequest.ChangedSince"/>.</summary>
    Changed = 4,

    /// <summary>In a file matched by <see cref="SelectionRequest.Paths"/>.</summary>
    Path = 8,

    /// <summary>References (calls, uses) a chunk of the previous hop.</summary>
    Caller = 16,

    /// <summary>Is referenced by a chunk of the previous hop (declares what it calls or uses).</summary>
    Callee = 32,
}

/// <summary>Priorities of <see cref="SelectedChunk.Priority"/>: higher is kept longer under a <see cref="SelectionRequest.TokenBudget"/>.</summary>
public static class SelectionPriority
{
    /// <summary><see cref="SelectionReason.Definition"/>.</summary>
    public const int Definition = 1000;

    /// <summary><see cref="SelectionReason.DefinitionBody"/>.</summary>
    public const int DefinitionBody = 900;

    /// <summary><see cref="SelectionReason.Changed"/>.</summary>
    public const int Changed = 800;

    /// <summary><see cref="SelectionReason.Path"/>.</summary>
    public const int Path = 700;

    /// <summary>A neighbour (<see cref="SelectionReason.Caller"/> / <see cref="SelectionReason.Callee"/>) at <paramref name="hop"/> ≥ 1: <c>600 - hop</c>.</summary>
    public static int Neighbour(int hop) => 600 - hop;
}

/// <summary>A selected (or trimmed) chunk.</summary>
/// <param name="Chunk">The chunk (with <see cref="ChunkRef.TokenCount"/>; <see cref="ChunkRef.KeyHex"/> is null).</param>
/// <param name="Reason">The reason that gave <see cref="Priority"/> (one flag).</param>
/// <param name="Priority">See <see cref="SelectionPriority"/>.</param>
/// <param name="Detail">Human-readable reason, e.g. <c>definition of A.B.Run</c> or <c>calls hop-0 chunks (hop 1, 3 edges)</c>.</param>
public sealed record SelectedChunk(ChunkRef Chunk, SelectionReason Reason, int Priority, string Detail)
{
    /// <summary>Every criterion the chunk matched (includes <see cref="Reason"/>).</summary>
    public SelectionReason Reasons { get; init; }

    /// <summary>Hop distance from the symbol's definition chunks (0 for those), or -1 when not reached by the symbol walk.</summary>
    public int Hop { get; init; } = -1;

    /// <summary>For neighbours: reference edges linking the chunk to the previous hop (the tie-break within a priority); else 0.</summary>
    public int Weight { get; init; }

    /// <summary>The chunk's token count.</summary>
    public int Tokens => Chunk.TokenCount ?? 0;
}

/// <summary>A declaration that matched <see cref="SelectionRequest.Symbol"/>.</summary>
/// <param name="QualifiedName">Namespace, containing types and name, dot-separated.</param>
/// <param name="Kind">Declaration kind: <c>type</c>, <c>method</c>, <c>constructor</c>, <c>property</c>, <c>event</c>, <c>field</c>, <c>enum-member</c>, <c>delegate</c>.</param>
/// <param name="FilePath">Scan path of the declaring file.</param>
/// <param name="Line">1-based line of the declaration's name.</param>
/// <param name="NameOffset">Character offset of the declaration's name.</param>
/// <param name="StartOffset">Start of the declaration (without leading trivia).</param>
/// <param name="EndOffset">End of the declaration (exclusive).</param>
public sealed record SymbolDefinition(string QualifiedName, string Kind, string FilePath, int Line, int NameOffset, int StartOffset, int EndOffset);

/// <summary>
/// The selected chunks in rank order (priority descending, then weight descending, then path ordinal, then start offset),
/// plus what was trimmed and why some criteria matched nothing.
/// </summary>
public sealed class SelectionResult : IReadOnlyList<SelectedChunk>
{
    internal SelectionResult(IReadOnlyList<SelectedChunk> selected, IReadOnlyList<SelectedChunk> trimmed, int? budget,
        IReadOnlyList<SymbolDefinition> definitions, IReadOnlyList<string> unmatchedPatterns, IReadOnlyList<string> unmatchedChangedFiles,
        IReadOnlyList<string> notes)
    {
        Selected = selected;
        Trimmed = trimmed;
        TokenBudget = budget;
        SymbolDefinitions = definitions;
        UnmatchedPatterns = unmatchedPatterns;
        UnmatchedChangedFiles = unmatchedChangedFiles;
        Notes = notes;
        SelectedTokens = selected.Sum(s => (long)s.Tokens);
        CandidateTokens = SelectedTokens + trimmed.Sum(s => (long)s.Tokens);
    }

    /// <summary>The selection, rank order.</summary>
    public IReadOnlyList<SelectedChunk> Selected { get; }

    /// <summary>Candidates dropped by <see cref="TokenBudget"/>, rank order (the first was the first not to fit).</summary>
    public IReadOnlyList<SelectedChunk> Trimmed { get; }

    /// <summary>The request's budget.</summary>
    public int? TokenBudget { get; }

    /// <summary>Tokens of <see cref="Selected"/>.</summary>
    public long SelectedTokens { get; }

    /// <summary>Tokens of every candidate (selected + trimmed).</summary>
    public long CandidateTokens { get; }

    /// <summary>Declarations that matched <see cref="SelectionRequest.Symbol"/> (ordered by path, then offset).</summary>
    public IReadOnlyList<SymbolDefinition> SymbolDefinitions { get; }

    /// <summary>Include patterns of <see cref="SelectionRequest.Paths"/> that matched no selectable file.</summary>
    public IReadOnlyList<string> UnmatchedPatterns { get; }

    /// <summary>Changed files (relative to the corpus root, '/' separated, or absolute on another volume) that have no chunk in the corpus: deleted, non-C#, excluded from the corpus, or outside it.</summary>
    public IReadOnlyList<string> UnmatchedChangedFiles { get; }

    /// <summary>Other remarks (definitions removed by <see cref="SelectionRequest.Exclude"/>, and so on).</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <inheritdoc/>
    public int Count => Selected.Count;

    /// <inheritdoc/>
    public SelectedChunk this[int index] => Selected[index];

    /// <inheritdoc/>
    public IEnumerator<SelectedChunk> GetEnumerator() => Selected.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>The selection in document order (path ordinal, then start offset), the order to assemble a context in.</summary>
    public IReadOnlyList<SelectedChunk> InDocumentOrder() =>
        Selected.OrderBy(s => s.Chunk.FilePath, StringComparer.Ordinal).ThenBy(s => s.Chunk.StartOffset).ToArray();
}

/// <summary><see cref="SelectionRequest.Symbol"/> has no declaration in the scan (or none that the graph's chunks cover).</summary>
public sealed class SymbolNotFoundException : Exception
{
    /// <summary>Creates the exception.</summary>
    public SymbolNotFoundException(string symbol, string message, IReadOnlyList<string> suggestions) : base(message)
    {
        Symbol = symbol;
        Suggestions = suggestions;
    }

    /// <summary>The requested symbol.</summary>
    public string Symbol { get; }

    /// <summary>Qualified names of declarations with a similar name (case-insensitive match of the last segment).</summary>
    public IReadOnlyList<string> Suggestions { get; }
}
