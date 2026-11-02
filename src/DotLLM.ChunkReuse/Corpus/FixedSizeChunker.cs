using DotLLM.Tokenizers;

namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>
/// Fixed-size token chunks (the E1/E2 chunking, but per file and keeping the tail): the file is tokenised exactly once
/// (<see cref="ITokenizer.EncodeRaw"/>, no BOS) and cut into consecutive slices of <see cref="MaxChunkTokens"/> ids; the last
/// slice may be shorter. Chunk boundaries can fall mid-line, so two neighbouring chunks may share a line.
/// </summary>
/// <remarks>
/// Lines and character offsets come from decoding the ids before each boundary. For a byte-level BPE that round-trips
/// (<c>Decode(Encode(t)) == t</c>) they are exact, except that a boundary inside a multi-byte UTF-8 character shifts the
/// offset by that character.
/// </remarks>
public sealed class FixedSizeChunker : IChunker
{
    private readonly ITokenizer _tokenizer;

    /// <summary>Creates the chunker.</summary>
    /// <param name="tokenizer">The model's tokenizer.</param>
    /// <param name="chunkTokens">Tokens per chunk (default 512, as in E1/E2).</param>
    public FixedSizeChunker(ITokenizer tokenizer, int chunkTokens = 512)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunkTokens);
        _tokenizer = tokenizer;
        MaxChunkTokens = chunkTokens;
    }

    /// <inheritdoc />
    public string Name => $"fixed-{MaxChunkTokens}";

    /// <inheritdoc />
    public int MaxChunkTokens { get; }

    /// <inheritdoc />
    public IReadOnlyList<Chunk> Chunk(SourceFile file)
    {
        ArgumentNullException.ThrowIfNull(file);
        if (file.Text.Length == 0) return [];
        int[] ids = _tokenizer.EncodeRaw(file.Text);
        var chunks = new List<Chunk>((ids.Length + MaxChunkTokens - 1) / MaxChunkTokens);
        int startOffset = 0, startLine = 1;
        for (int start = 0; start < ids.Length; start += MaxChunkTokens)
        {
            int[] slice = ids.AsSpan(start, Math.Min(MaxChunkTokens, ids.Length - start)).ToArray();
            string text = _tokenizer.Decode(slice);
            int newlines = text.AsSpan().Count('\n');
            int endLine = startLine + newlines - (text.EndsWith('\n') ? 1 : 0);
            // Decode-based offsets can drift from the text (a character the tokenizer drops or replaces on decode, e.g. a
            // Unicode noncharacter such as U+FDD0, or a cut inside a multi-byte character): clamp, and let the last chunk end at
            // the end of the text, so the chunks always tile the file (ChunkIndex's contract).
            bool last = start + MaxChunkTokens >= ids.Length;
            int endOffset = last ? file.Text.Length : Math.Min(file.Text.Length, startOffset + text.Length);
            chunks.Add(new Chunk(file.Path, startLine, Math.Max(startLine, endLine), slice) { StartOffset = startOffset, EndOffset = endOffset });
            startOffset = endOffset;
            startLine += newlines;
        }
        return chunks;
    }
}
