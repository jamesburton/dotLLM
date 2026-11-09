using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.Tokenizers;

namespace DotLLM.ChunkCache;

/// <summary>
/// Which chunks the <c>chunk_cache</c> wire may carry (integration spec §5.3 item 4, L6; Phase I Task 17). A chunk is wire-eligible
/// only if ALL THREE hold: its chunker is on the allow-list (<see cref="WireChunking.IsWireChunker"/>), it is a text piece and
/// not a token-id slice (<see cref="Chunk.WireEligible"/>), and its text tokenised alone with
/// <see cref="ChunkTokenization.Encode"/> (what a server receiving the text computes; a bare <c>EncodeRaw</c> would parse a
/// spelled-out control token such as <c>&lt;think&gt;</c> as the special) gives exactly its token ids.
/// </summary>
internal static class WireEligibility
{
    /// <summary>The number of chunks of <paramref name="chunks"/> that are NOT wire-eligible.</summary>
    /// <param name="chunkerName"><see cref="IChunker.Name"/> of the chunker that cut them.</param>
    /// <param name="chunks">Chunks (of one or several files).</param>
    /// <param name="textOf">The text the wire would carry for a chunk (its character range of the file).</param>
    /// <param name="tokenizer">The model's tokenizer.</param>
    public static int CountIneligible(string chunkerName, IReadOnlyList<Chunk> chunks, Func<Chunk, string> textOf, ITokenizer tokenizer)
    {
        if (!WireChunking.IsWireChunker(chunkerName)) return chunks.Count; // the whole chunker is off the wire
        var pieces = new List<Chunk>(chunks.Count);
        int slices = 0;
        foreach (Chunk c in chunks)
        {
            if (c.WireEligible) pieces.Add(c);
            else slices++;
        }
        return slices + WireChunking.VerifyRoundTrip(pieces, textOf, tokenizer).Failed;
    }

    /// <summary>Whether this one chunk may be sent (the per-chunk form of <see cref="CountIneligible"/>).</summary>
    public static bool IsEligible(string chunkerName, Chunk chunk, string text, ITokenizer tokenizer) =>
        WireChunking.IsWireChunker(chunkerName) && chunk.WireEligible
        && ChunkTokenization.Encode(tokenizer, text).AsSpan().SequenceEqual(chunk.TokenIds);
}
