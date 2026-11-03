namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>A source file of a code corpus: its corpus-relative path ('/' separators) and its text (LF line ends).</summary>
/// <param name="Path">Corpus-relative path, '/' separated; unique within a corpus.</param>
/// <param name="Text">File text. Line numbers count '\n' only, so callers normalise CRLF first.</param>
public sealed record SourceFile(string Path, string Text);

/// <summary>
/// A chunk of one source file with its token ids, fixed once at chunking time (never re-tokenised later). Lines are 1-based
/// and inclusive; a line is terminated by '\n'.
/// </summary>
/// <param name="FilePath">Path of the file the chunk belongs to.</param>
/// <param name="StartLine">Line of the chunk's first character.</param>
/// <param name="EndLine">Line of the chunk's last character (its '\n' when it ends a line).</param>
/// <param name="TokenIds">The chunk's token ids.</param>
/// <remarks>
/// Record equality compares <see cref="TokenIds"/> by reference; compare chunks by <see cref="Id"/> or by value explicitly.
/// </remarks>
public sealed record Chunk(string FilePath, int StartLine, int EndLine, int[] TokenIds)
{
    /// <summary>Character offset of the chunk's first character in the file text (inclusive).</summary>
    public int StartOffset { get; init; }

    /// <summary>Character offset just past the chunk's last character (exclusive).</summary>
    public int EndOffset { get; init; }

    /// <summary>
    /// Whether the chunk may be sent on the wire as text: its text, tokenised alone, gives exactly <see cref="TokenIds"/> by
    /// construction (default true). Token-id slices are not (<c>SymbolChunker</c>'s pieces of an over-long line, every
    /// <see cref="FixedSizeChunker"/> chunk): a slice can end inside a multi-byte character or a BPE merge. Whether the
    /// <i>chunker</i> is allowed on the wire is <see cref="WireChunking.IsWireChunker"/>.
    /// </summary>
    public bool WireEligible { get; init; } = true;

    /// <summary>Stable id within a chunked corpus: path, line range and start offset.</summary>
    public string Id => $"{FilePath}:{StartLine}-{EndLine}@{StartOffset}";
}

/// <summary>Splits a source file into chunks whose concatenated text is the file text.</summary>
public interface IChunker
{
    /// <summary>Short name including the size parameter, e.g. <c>symbol-512</c>; part of the corpus hash.</summary>
    string Name { get; }

    /// <summary>Upper bound on <see cref="Chunk.TokenIds"/> length of every chunk this chunker returns.</summary>
    int MaxChunkTokens { get; }

    /// <summary>
    /// Chunks <paramref name="file"/> in text order. The chunks cover the file without overlap (their texts concatenate to the
    /// file text); an empty file has no chunks.
    /// </summary>
    IReadOnlyList<Chunk> Chunk(SourceFile file);
}

/// <summary>'\n'-only line index of a text (1-based lines).</summary>
internal sealed class LineIndex
{
    private readonly int[] _starts;

    public LineIndex(string text)
    {
        var starts = new List<int> { 0 };
        for (int i = 0; i < text.Length; i++)
            if (text[i] == '\n' && i + 1 < text.Length) starts.Add(i + 1);
        _starts = starts.ToArray();
    }

    /// <summary>1-based line of character <paramref name="offset"/>.</summary>
    public int LineOf(int offset)
    {
        int i = Array.BinarySearch(_starts, offset);
        return (i >= 0 ? i : ~i - 1) + 1;
    }

    /// <summary>Offset of the start of the line holding <paramref name="offset"/>.</summary>
    public int LineStartOf(int offset) => _starts[LineOf(offset) - 1];

    /// <summary>Start offsets of every line whose start lies strictly inside (<paramref name="start"/>, <paramref name="end"/>).</summary>
    public IEnumerable<int> LineStartsInside(int start, int end)
    {
        int i = LineOf(start); // index of the next line (0-based = 1-based line number)
        for (; i < _starts.Length && _starts[i] < end; i++)
            if (_starts[i] > start) yield return _starts[i];
    }
}
