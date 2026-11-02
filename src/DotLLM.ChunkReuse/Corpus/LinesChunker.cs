using DotLLM.Tokenizers;

namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>
/// Text-first fixed-size chunks, <c>lines-N</c> (E4c ran this chunker as <c>fixed-text-512</c>; every chunk round-trips by
/// construction, but <c>lines-512</c> is on the wire allow-list only from Phase I Task 19, see
/// <see cref="WireChunking.PendingE4c"/>): the file <b>text</b> is cut into pieces of at most
/// <see cref="MaxChunkTokens"/> tokens each, and every piece is tokenised alone (<see cref="ChunkTokenization.Encode"/>, no BOS; every size probe too, so a
/// piece whose special-token spellings encode as several plain tokens never exceeds the limit).
/// A server that receives a piece as text and tokenises it reproduces exactly the chunk's ids, which is not true of
/// <see cref="FixedSizeChunker"/>'s id slices (BPE merges across a slice boundary, slices ending inside a multi-byte UTF-8
/// character).
/// </summary>
/// <remarks>
/// Greedy, left to right. From the current start, the furthest code-point boundary whose piece tokenises to at most N ids is
/// found by binary search (each probe tokenises the candidate piece alone; only probed pieces that fit are ever accepted, so the
/// bound holds even where token count is not monotone in length). The cut is the last line end ('\n') inside that piece when
/// there is one; otherwise that character boundary, moved back so that it splits neither a surrogate pair (a 4-byte UTF-8
/// sequence; BMP characters are whole <see cref="char"/>s) nor a "\r\n". When the rest of the file fits, it is the last chunk.
/// Text is used as given (CRLF is kept); offsets and lines are exact.
/// </remarks>
public sealed class LinesChunker : IChunker
{
    private readonly ITokenizer _tokenizer;

    /// <summary>Creates the chunker.</summary>
    /// <param name="tokenizer">The model's tokenizer.</param>
    /// <param name="chunkTokens">Maximum tokens per chunk (default 512, as <see cref="FixedSizeChunker"/>).</param>
    public LinesChunker(ITokenizer tokenizer, int chunkTokens = 512)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunkTokens);
        _tokenizer = tokenizer;
        MaxChunkTokens = chunkTokens;
    }

    /// <inheritdoc />
    public string Name => $"lines-{MaxChunkTokens}";

    /// <inheritdoc />
    public int MaxChunkTokens { get; }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">A single character tokenises to more than <see cref="MaxChunkTokens"/> ids.</exception>
    public IReadOnlyList<Chunk> Chunk(SourceFile file)
    {
        ArgumentNullException.ThrowIfNull(file);
        string text = file.Text;
        var chunks = new List<Chunk>();
        if (text.Length == 0) return chunks;
        var lines = new LineIndex(text);
        int start = 0;
        while (start < text.Length)
        {
            (int end, int[] ids) = NextCut(text, start);
            chunks.Add(new Chunk(file.Path, lines.LineOf(start), lines.LineOf(end - 1), ids) { StartOffset = start, EndOffset = end });
            start = end;
        }
        return chunks;
    }

    private (int End, int[] Ids) NextCut(string text, int start)
    {
        int n = MaxChunkTokens;
        int[] rest = ChunkTokenization.Encode(_tokenizer, text[start..]);
        if (rest.Length <= n) return (text.Length, rest);

        // Furthest code-point boundary e in (start, text.Length) with Tokens(start..e) <= n.
        int lo = start, hi = text.Length;          // invariant: lo fits (or is start), hi does not
        int[]? loIds = null;
        while (hi - lo > 1)
        {
            int mid = lo + (hi - lo) / 2;
            if (SplitsPair(text, mid))
            {
                mid--;                          // back to the pair's start, or past the pair when that is lo
                if (mid <= lo) mid += 2;
                if (mid >= hi) break;           // the only boundary left in (lo, hi) is hi itself
            }
            int[] ids = ChunkTokenization.Encode(_tokenizer, text[start..mid]);
            if (ids.Length <= n) { lo = mid; loIds = ids; }
            else hi = mid;
        }
        if (loIds is null)
            throw new InvalidOperationException(
                $"The character at offset {start} tokenises to more than {n} tokens on its own; it cannot be chunked without splitting it.");

        // Prefer the last line end inside the piece (the furthest one that fits when token count is not monotone).
        for (int i = lo - 1; i >= start; i--)
        {
            if (text[i] != '\n') continue;
            if (i + 1 == lo) return (lo, loIds);
            int[] ids = ChunkTokenization.Encode(_tokenizer, text[start..(i + 1)]);
            if (ids.Length <= n) return (i + 1, ids);
        }

        // No line end fits: a character cut, never between '\r' and '\n'.
        for (int end = lo; end - 1 > start && text[end - 1] == '\r' && text[end] == '\n';)
        {
            end--;
            if (SplitsPair(text, end)) continue;
            int[] ids = ChunkTokenization.Encode(_tokenizer, text[start..end]);
            if (ids.Length <= n) return (end, ids);
        }
        return (lo, loIds);
    }

    /// <summary>Whether a cut at <paramref name="pos"/> falls between a high and a low surrogate.</summary>
    private static bool SplitsPair(string text, int pos) =>
        pos > 0 && pos < text.Length && char.IsHighSurrogate(text[pos - 1]) && char.IsLowSurrogate(text[pos]);
}
