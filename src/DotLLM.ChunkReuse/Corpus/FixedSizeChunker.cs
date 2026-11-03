using DotLLM.Tokenizers;

namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>
/// Fixed-size token chunks (the E1/E2 chunking, but per file and keeping the tail): the file is tokenised exactly once
/// (<see cref="ChunkTokenization.Encode"/>, no BOS) and cut into consecutive slices of <see cref="MaxChunkTokens"/> ids; the last
/// slice may be shorter. Chunk boundaries can fall mid-line, so two neighbouring chunks may share a line. Lab-only: no chunk is
/// <see cref="Chunk.WireEligible"/>.
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
        int[] ids = ChunkTokenization.Encode(_tokenizer, file.Text);
        var chunks = new List<Chunk>((ids.Length + MaxChunkTokens - 1) / MaxChunkTokens);
        int startOffset = 0, startLine = 1;
        for (int start = 0; start < ids.Length; start += MaxChunkTokens)
        {
            int[] slice = ids.AsSpan(start, Math.Min(MaxChunkTokens, ids.Length - start)).ToArray();
            string text = _tokenizer.Decode(slice);
            int newlines = text.AsSpan().Count('\n');
            int endLine = startLine + newlines - (text.EndsWith('\n') ? 1 : 0);
            // Offsets come from the decoded slice, which can differ from the text: the encoder strips characters (dotLLM's
            // ControlTokenGuard removes U+FDD0 on every encode, so the ids never represent it), and a cut inside a multi-byte
            // character decodes to U+FFFD. Advance over the text along the decoded characters, skipping stripped markers, and let
            // the last chunk end at the end of the text, so the chunks always tile the file (ChunkIndex's contract).
            bool last = start + MaxChunkTokens >= ids.Length;
            int endOffset = last ? file.Text.Length : Advance(file.Text, startOffset, text);
            // Id slices are never wire-eligible: a slice can end inside a multi-byte character or a BPE merge (E4c).
            chunks.Add(new Chunk(file.Path, startLine, Math.Max(startLine, endLine), slice) { StartOffset = startOffset, EndOffset = endOffset, WireEligible = false });
            startOffset = endOffset;
            startLine += newlines;
        }
        return chunks;
    }

    /// <summary>The text offset after <paramref name="decoded"/>, read from <paramref name="offset"/>: one text character per
    /// decoded character, skipping text characters the encoder strips (U+FDD0) that the decoded text does not have.</summary>
    private static int Advance(string text, int offset, string decoded)
    {
        int p = offset;
        foreach (char c in decoded)
        {
            while (p < text.Length && text[p] != c && text[p] == StrippedOnEncode) p++;
            if (p >= text.Length) break;
            p++;
        }
        return p;
    }

    /// <summary>The character dotLLM's encoder strips (<c>ControlTokenGuard.Marker</c>, U+FDD0).</summary>
    private const char StrippedOnEncode = (char)0xFDD0;
}
