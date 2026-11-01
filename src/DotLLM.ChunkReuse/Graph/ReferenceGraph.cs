using DotLLM.ChunkReuse.CodeCorpus;

namespace DotLLM.ChunkReuse.Graph;

/// <summary>What a <see cref="SymbolEdge"/> links.</summary>
public enum SymbolEdgeKind
{
    /// <summary>A declaration's name → the same symbol's ordinal-first declaration (later parts of a partial type, method or property).</summary>
    Definition,

    /// <summary>A reference to a type, field, property, event, or method group → its declaration name.</summary>
    Use,

    /// <summary>
    /// A using directive's name → the imported namespace's name in its ordinal-first declaring file (or the type, for
    /// <c>using static</c> / aliases). The use end is the first character of the whole directive name, the definition end the
    /// last segment's name, so for a dotted namespace (<c>using A.B;</c>) the two ends spell different identifiers (A vs B).
    /// </summary>
    Import,

    /// <summary>An invoked method's name (or a created type's name, for an explicit constructor; or an attribute name) → the method's declaration name.</summary>
    Call,
}

/// <summary>A reference edge in context coordinates: both ends are token indices in <c>[0, ChunkedContext.Length)</c>.</summary>
/// <param name="UseTokenIndex">Token holding the first character of the referencing name.</param>
/// <param name="DefinitionTokenIndex">Token holding the first character of the declaration's name.</param>
/// <param name="Kind">The edge kind.</param>
public readonly record struct SymbolEdge(int UseTokenIndex, int DefinitionTokenIndex, SymbolEdgeKind Kind);

/// <summary>A reference edge in graph coordinates: chunk indices into <see cref="ReferenceGraph.Chunks"/> and chunk-local token indices.</summary>
/// <param name="UseChunk">Chunk of the referencing name.</param>
/// <param name="UseToken">Token of that chunk holding the name's first character.</param>
/// <param name="DefinitionChunk">Chunk of the declaration's name.</param>
/// <param name="DefinitionToken">Token of that chunk holding the declaration name's first character.</param>
/// <param name="Kind">The edge kind.</param>
public readonly record struct ChunkEdge(int UseChunk, int UseToken, int DefinitionChunk, int DefinitionToken, SymbolEdgeKind Kind);

/// <summary>An edge from a character range of a file (a query) to a chunk: <see cref="ReferenceGraph.QueryEdges"/>.</summary>
/// <param name="UseOffset">File character offset of the referencing name's first character.</param>
/// <param name="DefinitionChunk">Chunk of the declaration's name (index into <see cref="ReferenceGraph.Chunks"/>).</param>
/// <param name="DefinitionToken">Token of that chunk holding the declaration name's first character.</param>
/// <param name="Kind">The edge kind.</param>
public readonly record struct QueryEdge(int UseOffset, int DefinitionChunk, int DefinitionToken, SymbolEdgeKind Kind);

/// <summary>
/// Symbol reference graph over a chunked code corpus (built by <see cref="RoslynGraphBuilder"/> when the chunks are encoded):
/// every edge in chunk-local token coordinates, including edges inside one chunk. Request-time selection asks
/// <see cref="EdgesForContext(ChunkedContext)"/> for the edges that cross chunks of one context, which costs a lookup per
/// context chunk plus the edges of those chunks.
/// </summary>
/// <remarks>
/// Edges are sorted by (use chunk, use token, definition chunk, definition token, kind) and distinct. Immutable.
/// </remarks>
public sealed class ReferenceGraph
{
    private readonly Chunk[] _chunks;
    private readonly ChunkEdge[] _edges;
    private readonly int[] _useOffsets; // file character offset of each edge's use name, parallel to _edges
    private readonly int[] _edgeStart; // edges of use chunk c are _edges[_edgeStart[c] .. _edgeStart[c + 1])
    private readonly Dictionary<string, int> _chunkById;
    private readonly Dictionary<long, List<int>> _chunksByHash; // content hash → chunk indices (collisions resolved by comparing ids)

    internal ReferenceGraph(IReadOnlyList<Chunk> chunks, ChunkEdge[] sortedEdges, int[] useOffsets, int mappingMismatchChunks)
    {
        _chunks = chunks.ToArray();
        _edges = sortedEdges;
        _useOffsets = useOffsets;
        MappingMismatchChunks = mappingMismatchChunks;
        _edgeStart = new int[_chunks.Length + 1];
        foreach (ChunkEdge e in _edges) _edgeStart[e.UseChunk + 1]++;
        for (int c = 0; c < _chunks.Length; c++) _edgeStart[c + 1] += _edgeStart[c];

        _chunkById = new Dictionary<string, int>(StringComparer.Ordinal);
        _chunksByHash = [];
        for (int c = 0; c < _chunks.Length; c++)
        {
            _chunkById.TryAdd(_chunks[c].Id, c);
            long h = ContentHash(_chunks[c].TokenIds);
            if (!_chunksByHash.TryGetValue(h, out List<int>? list)) _chunksByHash[h] = list = [];
            list.Add(c);
        }
    }

    /// <summary>The chunks, in the order given to the builder; <see cref="ChunkEdge"/> chunk indices refer to it.</summary>
    public IReadOnlyList<Chunk> Chunks => _chunks;

    /// <summary>Every edge, in graph coordinates, sorted and distinct.</summary>
    public IReadOnlyList<ChunkEdge> Edges => _edges;

    /// <summary>
    /// Chunks holding an edge end whose decoded tokens do not spell the chunk's slice of the file (token-sliced chunks cut
    /// inside a multi-byte character, see <see cref="SymbolChunker"/> and <see cref="FixedSizeChunker"/>): edge ends in them
    /// can be off by a token. Zero for a tokenizer that round-trips each chunk. Chunks without an edge end are not checked.
    /// </summary>
    public int MappingMismatchChunks { get; }

    /// <summary>
    /// Edges whose use name starts in the file characters [<paramref name="start"/>, <paramref name="end"/>) of
    /// <paramref name="filePath"/>, with that start as a file offset. For a query that is a prefix of a corpus file (a Task 8
    /// <see cref="CodeTask"/>), pass the prefix's range, which ends where the masked call begins, so the masked call — the
    /// answer — is excluded. Map <see cref="QueryEdge.UseOffset"/> into the query's own tokenisation by character offset (the
    /// file's chunk tokens are not the query's tokens); keep the edges whose <see cref="QueryEdge.DefinitionChunk"/> is in the
    /// context. Sorted by (offset, definition chunk, definition token, kind).
    /// </summary>
    public IReadOnlyList<QueryEdge> QueryEdges(string filePath, int start, int end)
    {
        ArgumentNullException.ThrowIfNull(filePath);
        var result = new List<QueryEdge>();
        for (int c = 0; c < _chunks.Length; c++)
        {
            Chunk chunk = _chunks[c];
            if (chunk.EndOffset <= start || chunk.StartOffset >= end || !string.Equals(chunk.FilePath, filePath, StringComparison.Ordinal)) continue;
            for (int i = _edgeStart[c]; i < _edgeStart[c + 1]; i++)
            {
                int offset = _useOffsets[i];
                if (offset >= start && offset < end)
                    result.Add(new QueryEdge(offset, _edges[i].DefinitionChunk, _edges[i].DefinitionToken, _edges[i].Kind));
            }
        }
        result.Sort((x, y) =>
        {
            int d = x.UseOffset.CompareTo(y.UseOffset);
            if (d == 0) d = x.DefinitionChunk.CompareTo(y.DefinitionChunk);
            if (d == 0) d = x.DefinitionToken.CompareTo(y.DefinitionToken);
            return d != 0 ? d : x.Kind.CompareTo(y.Kind);
        });
        return result;
    }

    /// <summary>Index into <see cref="Chunks"/> of the chunk with this <see cref="Chunk.Id"/>, or -1.</summary>
    public int IndexOf(string chunkId)
    {
        ArgumentNullException.ThrowIfNull(chunkId);
        return _chunkById.TryGetValue(chunkId, out int c) ? c : -1;
    }

    /// <summary>Edges whose use end lies in chunk <paramref name="chunk"/>.</summary>
    public ReadOnlySpan<ChunkEdge> EdgesFrom(int chunk) => _edges.AsSpan(_edgeStart[chunk], _edgeStart[chunk + 1] - _edgeStart[chunk]);

    /// <summary>
    /// Graph chunk index of each context chunk, matched by token-id content, or -1 when no graph chunk has that content or
    /// several do (identical text in different places can bind to different symbols, so an ambiguous chunk is left
    /// unidentified rather than guessed).
    /// </summary>
    public int[] MatchChunks(ChunkedContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        var match = new int[context.ChunkCount];
        for (int k = 0; k < match.Length; k++)
        {
            ReadOnlySpan<int> ids = context.Chunk(k);
            match[k] = -1;
            if (!_chunksByHash.TryGetValue(ContentHash(ids), out List<int>? candidates)) continue;
            foreach (int c in candidates)
            {
                if (!ids.SequenceEqual(_chunks[c].TokenIds)) continue;
                if (match[k] >= 0)
                {
                    match[k] = -1; // a second chunk with the same content: ambiguous
                    break;
                }
                match[k] = c;
            }
        }
        return match;
    }

    /// <summary>
    /// The edges between different chunks of <paramref name="context"/>, in context coordinates (each end offset by its chunk's
    /// placement, <see cref="ChunkedContext.ChunkStarts"/>). Context chunks are identified by content
    /// (<see cref="MatchChunks"/>); unidentified chunks contribute no edges, and edges whose other end is outside the context
    /// are dropped. Sorted by (use, definition, kind), distinct.
    /// </summary>
    public IReadOnlyList<SymbolEdge> EdgesForContext(ChunkedContext context) => Collect(context, MatchChunks(context));

    /// <summary>
    /// As <see cref="EdgesForContext(ChunkedContext)"/>, with the context chunks given explicitly (matched by
    /// <see cref="Chunk.Id"/>), so chunks with duplicate content are still identified.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="chunks"/> does not describe the context chunk by chunk, or a chunk is not in the graph.</exception>
    public IReadOnlyList<SymbolEdge> EdgesForContext(ChunkedContext context, IReadOnlyList<Chunk> chunks)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(chunks);
        if (chunks.Count != context.ChunkCount)
            throw new ArgumentException($"{chunks.Count} chunks for a context of {context.ChunkCount}.", nameof(chunks));
        var match = new int[chunks.Count];
        for (int k = 0; k < match.Length; k++)
        {
            if (!context.Chunk(k).SequenceEqual(chunks[k].TokenIds))
                throw new ArgumentException($"Chunk {k} ({chunks[k].Id}) does not match the context's chunk {k}.", nameof(chunks));
            match[k] = _chunkById.TryGetValue(chunks[k].Id, out int c)
                ? c
                : throw new ArgumentException($"Chunk {chunks[k].Id} is not in the graph.", nameof(chunks));
        }
        return Collect(context, match);
    }

    private SymbolEdge[] Collect(ChunkedContext context, int[] match)
    {
        int[] starts = context.ChunkStarts;
        // Graph chunk → its context position (the first, if a chunk is placed twice).
        var placed = new Dictionary<int, int>(match.Length);
        for (int k = 0; k < match.Length; k++)
            if (match[k] >= 0) placed.TryAdd(match[k], k);

        var result = new List<SymbolEdge>();
        foreach ((int c, int k) in placed)
        {
            foreach (ChunkEdge e in EdgesFrom(c))
            {
                if (e.DefinitionChunk == c || !placed.TryGetValue(e.DefinitionChunk, out int dk)) continue;
                result.Add(new SymbolEdge(starts[k] + e.UseToken, starts[dk] + e.DefinitionToken, e.Kind));
            }
        }
        result.Sort((x, y) =>
        {
            int d = x.UseTokenIndex.CompareTo(y.UseTokenIndex);
            if (d == 0) d = x.DefinitionTokenIndex.CompareTo(y.DefinitionTokenIndex);
            return d != 0 ? d : x.Kind.CompareTo(y.Kind);
        });
        return result.Distinct().ToArray();
    }

    private static long ContentHash(ReadOnlySpan<int> ids)
    {
        var h = new HashCode();
        h.AddBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(ids));
        return ((long)h.ToHashCode() << 32) ^ ids.Length;
    }
}
