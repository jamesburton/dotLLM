using DotLLM.ChunkReuse.CodeCorpus;

namespace DotLLM.ChunkReuse.Store;

/// <summary>How a chunk's boundaries were chosen.</summary>
public enum ChunkKind
{
    /// <summary>Fixed-size token slices of the file (<see cref="FixedSizeChunker"/>); boundaries can fall mid-line.</summary>
    Fixed = 0,

    /// <summary>One member per chunk from the Roslyn syntax tree (<see cref="SymbolChunker"/>); whole lines.</summary>
    Symbol = 1,
}

/// <summary>
/// A lightweight reference to one chunk of a source file: where it is (path, 1-based inclusive lines, character offsets) and,
/// when known, its size and store key. It carries no token ids, so it is cheap to keep for a whole repository, to persist and
/// to pass between the index, the selection tools and the CLI; the chunk's K/V is looked up in an <see cref="IChunkStore"/>
/// by <see cref="KeyHex"/>.
/// </summary>
/// <param name="FilePath">Corpus-relative path, '/' separated (as <see cref="SourceFile.Path"/>).</param>
/// <param name="StartLine">Line of the chunk's first character (1-based).</param>
/// <param name="EndLine">Line of the chunk's last character (1-based, inclusive).</param>
/// <param name="StartOffset">Character offset of the chunk's first character in the file text (inclusive).</param>
/// <param name="EndOffset">Character offset just past the chunk's last character (exclusive).</param>
/// <param name="Kind">The chunking that produced it.</param>
public sealed record ChunkRef(string FilePath, int StartLine, int EndLine, int StartOffset, int EndOffset, ChunkKind Kind)
{
    /// <summary>Number of token ids in the chunk, or null when not known (e.g. a selection made before tokenisation).</summary>
    public int? TokenCount { get; init; }

    /// <summary>The chunk's <see cref="ChunkKey.Hex"/>, or null when not known.</summary>
    public string? KeyHex { get; init; }

    /// <summary>The key, when <see cref="KeyHex"/> is known.</summary>
    public ChunkKey? Key => KeyHex is null ? null : new ChunkKey(KeyHex);

    /// <summary>Same id as <see cref="Chunk.Id"/>: path, line range and start offset.</summary>
    public string Id => $"{FilePath}:{StartLine}-{EndLine}@{StartOffset}";

    /// <summary>A reference to <paramref name="chunk"/>, with its token count and (optionally) its key.</summary>
    /// <param name="chunk">The chunk.</param>
    /// <param name="kind">The chunking that produced it.</param>
    /// <param name="key">Its store key, if computed.</param>
    /// <returns>The reference.</returns>
    public static ChunkRef From(Chunk chunk, ChunkKind kind, ChunkKey? key = null)
    {
        ArgumentNullException.ThrowIfNull(chunk);
        return new ChunkRef(chunk.FilePath, chunk.StartLine, chunk.EndLine, chunk.StartOffset, chunk.EndOffset, kind)
        {
            TokenCount = chunk.TokenIds.Length,
            KeyHex = key?.Hex,
        };
    }

    /// <summary>The kind of the chunks <paramref name="chunker"/> produces (from its name: <c>symbol-*</c> or <c>fixed-*</c>).</summary>
    /// <param name="chunker">A chunker.</param>
    /// <returns>Its kind.</returns>
    /// <exception cref="ArgumentException">The chunker is neither a symbol nor a fixed-size chunker.</exception>
    public static ChunkKind KindOf(IChunker chunker)
    {
        ArgumentNullException.ThrowIfNull(chunker);
        if (chunker.Name.StartsWith("symbol-", StringComparison.Ordinal)) return ChunkKind.Symbol;
        if (chunker.Name.StartsWith("fixed-", StringComparison.Ordinal)) return ChunkKind.Fixed;
        throw new ArgumentException($"Chunker '{chunker.Name}' is neither symbol-* nor fixed-*.", nameof(chunker));
    }
}
