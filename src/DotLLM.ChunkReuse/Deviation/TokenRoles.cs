using DotLLM.ChunkReuse.Graph;

namespace DotLLM.ChunkReuse.Deviation;

/// <summary>
/// Multi-label token flags for E3. A token can carry several (a definition name in a chunk's first token is both
/// <see cref="ChunkStart1"/> and <see cref="Definition"/>); <see cref="TokenRoles.Primary"/> collapses them to one role.
/// </summary>
[Flags]
public enum TokenRoleFlags
{
    /// <summary>No flag.</summary>
    None = 0,

    /// <summary>The first token of a chunk (chunk-local offset 0).</summary>
    ChunkStart1 = 1 << 0,

    /// <summary>One of the first 4 tokens of a chunk.</summary>
    ChunkStart4 = 1 << 1,

    /// <summary>One of the first 16 tokens of a chunk.</summary>
    ChunkStart16 = 1 << 2,

    /// <summary>A declaration name: the definition end of a graph edge, or either end of a <see cref="SymbolEdgeKind.Definition"/> edge.</summary>
    Definition = 1 << 3,

    /// <summary>A reference: the use end of a <see cref="SymbolEdgeKind.Use"/> or <see cref="SymbolEdgeKind.Call"/> edge.</summary>
    Use = 1 << 4,

    /// <summary>Either end of a <see cref="SymbolEdgeKind.Import"/> edge (a using directive, or the namespace name it binds to).</summary>
    Import = 1 << 5,

    /// <summary>The token is an end of at least one graph edge whose other end lies in a different chunk of the corpus.</summary>
    CrossChunk = 1 << 6,

    /// <summary>The token is an end of an edge between two different chunks of this context (<see cref="ReferenceGraph.EdgesForContext(ChunkedContext, IReadOnlyList{CodeCorpus.Chunk})"/>).</summary>
    InContextCross = 1 << 7,

    /// <summary>The token is the definition end of a query-prefix edge (<see cref="ReferenceGraph.QueryEdges"/>) into this context.</summary>
    QueryTarget = 1 << 8,

    /// <summary>A query row (forwarded after the context in both arms), not a context row.</summary>
    Query = 1 << 9,
}

/// <summary>Primary (single-label) token role of E3.</summary>
public enum TokenRole
{
    /// <summary>First token of a chunk (<see cref="TokenRoleFlags.ChunkStart1"/>).</summary>
    ChunkStart,

    /// <summary>A declaration name (<see cref="TokenRoleFlags.Definition"/>).</summary>
    DefinitionSite,

    /// <summary>A reference (<see cref="TokenRoleFlags.Use"/>: Use and Call edges).</summary>
    UseSite,

    /// <summary>A using directive / imported namespace (<see cref="TokenRoleFlags.Import"/>).</summary>
    Import,

    /// <summary>Any other context token.</summary>
    Other,

    /// <summary>A query row at the use end of a query-prefix edge.</summary>
    QuerySite,

    /// <summary>Any other query row.</summary>
    Query,
}

/// <summary>Token-role labelling for E3.</summary>
public static class TokenRoles
{
    /// <summary>
    /// Primary role by priority ChunkStart &gt; DefinitionSite &gt; UseSite &gt; Import &gt; Other. ChunkStart means offset 0 only
    /// (<see cref="TokenRoleFlags.ChunkStart1"/>); the first-4/16 flags and the bookkeeping flags do not decide the role. Query
    /// rows are <see cref="TokenRole.QuerySite"/> when they carry a use/definition/import flag, else <see cref="TokenRole.Query"/>.
    /// </summary>
    public static TokenRole Primary(TokenRoleFlags flags)
    {
        if ((flags & TokenRoleFlags.Query) != 0)
            return (flags & (TokenRoleFlags.Use | TokenRoleFlags.Definition | TokenRoleFlags.Import)) != 0 ? TokenRole.QuerySite : TokenRole.Query;
        if ((flags & TokenRoleFlags.ChunkStart1) != 0) return TokenRole.ChunkStart;
        if ((flags & TokenRoleFlags.Definition) != 0) return TokenRole.DefinitionSite;
        if ((flags & TokenRoleFlags.Use) != 0) return TokenRole.UseSite;
        if ((flags & TokenRoleFlags.Import) != 0) return TokenRole.Import;
        return TokenRole.Other;
    }

    /// <summary>The role flag of one end of a graph edge.</summary>
    /// <param name="kind">Edge kind.</param>
    /// <param name="useEnd">True for the use end, false for the definition end.</param>
    public static TokenRoleFlags EdgeEndFlag(SymbolEdgeKind kind, bool useEnd) => kind switch
    {
        SymbolEdgeKind.Import => TokenRoleFlags.Import,
        SymbolEdgeKind.Definition => TokenRoleFlags.Definition,
        SymbolEdgeKind.Use or SymbolEdgeKind.Call => useEnd ? TokenRoleFlags.Use : TokenRoleFlags.Definition,
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, null),
    };

    /// <summary>
    /// Flags for <paramref name="contextLength"/> context rows followed by <paramref name="queryLength"/> query rows: first-k
    /// flags from each token's offset in its chunk, <see cref="TokenRoleFlags.Query"/> on the query rows, and every mark OR-ed
    /// into its row.
    /// </summary>
    /// <param name="chunkStarts">Ascending chunk start indices; the first is 0.</param>
    /// <param name="contextLength">Context tokens.</param>
    /// <param name="queryLength">Query rows after the context.</param>
    /// <param name="marks">(row, flags) pairs; rows in <c>[0, contextLength + queryLength)</c>.</param>
    public static TokenRoleFlags[] Label(IReadOnlyList<int> chunkStarts, int contextLength, int queryLength,
        IEnumerable<(int Token, TokenRoleFlags Flags)> marks)
    {
        ArgumentNullException.ThrowIfNull(chunkStarts);
        ArgumentNullException.ThrowIfNull(marks);
        ArgumentOutOfRangeException.ThrowIfNegative(contextLength);
        ArgumentOutOfRangeException.ThrowIfNegative(queryLength);
        var f = new TokenRoleFlags[contextLength + queryLength];
        for (int c = 0; c < chunkStarts.Count; c++)
        {
            int start = chunkStarts[c];
            int end = c + 1 < chunkStarts.Count ? chunkStarts[c + 1] : contextLength;
            if (start < 0 || end > contextLength || start > end)
                throw new ArgumentException($"Chunk {c} spans [{start}, {end}) outside the context of {contextLength}.", nameof(chunkStarts));
            for (int t = start; t < end; t++)
            {
                int offset = t - start;
                if (offset < 1) f[t] |= TokenRoleFlags.ChunkStart1;
                if (offset < 4) f[t] |= TokenRoleFlags.ChunkStart4;
                if (offset < 16) f[t] |= TokenRoleFlags.ChunkStart16;
            }
        }
        for (int t = contextLength; t < f.Length; t++) f[t] |= TokenRoleFlags.Query;
        foreach ((int token, TokenRoleFlags flags) in marks)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(token, nameof(marks));
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(token, f.Length, nameof(marks));
            f[token] |= flags;
        }
        return f;
    }
}
