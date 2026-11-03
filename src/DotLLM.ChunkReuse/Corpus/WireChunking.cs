using System.Collections.Frozen;
using System.Security.Cryptography;
using System.Text;
using DotLLM.Tokenizers;

namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>
/// Which chunks may travel on the <c>chunk_cache</c> wire (integration spec §5.3 item 4, L6) and the wire hash. The wire carries
/// chunk <b>text</b>; the server tokenises it, so only chunkers that cut text first and tokenise each piece alone are allowed,
/// and within them only chunks that are <see cref="Chunk.WireEligible"/> (not token-id slices).
/// </summary>
public static class WireChunking
{
    /// <summary>Chunker names allowed on the wire: <c>symbol-512</c>.</summary>
    public static IReadOnlySet<string> Eligible { get; } = FrozenSet.Create(StringComparer.Ordinal, "symbol-512");

    /// <summary>
    /// Chunker names whose wire eligibility is granted on correctness (E4c: <c>lines-512</c> round-trips by construction) but
    /// that join <see cref="Eligible"/> only in Phase I Task 19, after the chunk tokenisation matches the chat path (Task 5b).
    /// Not wire chunkers until then.
    /// </summary>
    public static IReadOnlySet<string> PendingE4c { get; } = FrozenSet.Create(StringComparer.Ordinal, "lines-512");

    /// <summary>Whether chunks of the chunker named <paramref name="chunkerName"/> may be sent on the wire.</summary>
    /// <param name="chunkerName"><see cref="IChunker.Name"/>.</param>
    /// <returns><see cref="Eligible"/> contains the name (ordinal).</returns>
    public static bool IsWireChunker(string chunkerName)
    {
        ArgumentNullException.ThrowIfNull(chunkerName);
        return Eligible.Contains(chunkerName);
    }

    /// <summary>
    /// The wire hash of a chunk text: <c>"sha256:"</c> + lowercase hex of its UTF-8 bytes (spec §5.2). A lone surrogate is
    /// encoded as U+FFFD (the <see cref="Encoding.UTF8"/> default; contract ruling 17).
    /// </summary>
    /// <param name="text">The chunk text exactly as sent (CRLF, BOM and trailing whitespace all count).</param>
    /// <returns>The hash string.</returns>
    public static string Hash(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return "sha256:" + Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text)));
    }

    /// <summary>
    /// Checks, per chunk, that its text tokenised alone (<see cref="ChunkTokenization.Encode"/>) gives exactly its ids (what a
    /// server receiving the text would compute). The caller marks failures not wire-eligible.
    /// </summary>
    /// <param name="chunks">Chunks to check.</param>
    /// <param name="textOf">The text the wire would carry for a chunk: its character range of the file
    /// (e.g. <c>Corpus.TextOf</c>), not its decoded ids.</param>
    /// <param name="tokenizer">The model's tokenizer.</param>
    /// <returns>Counts and the failing chunks' <see cref="Chunk.Id"/>s, in input order.</returns>
    public static RoundTripReport VerifyRoundTrip(IReadOnlyList<Chunk> chunks, Func<Chunk, string> textOf, ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(textOf);
        ArgumentNullException.ThrowIfNull(tokenizer);
        var failed = new List<string>();
        foreach (Chunk c in chunks)
        {
            int[] ids = ChunkTokenization.Encode(tokenizer, textOf(c));
            if (!ids.AsSpan().SequenceEqual(c.TokenIds)) failed.Add(c.Id);
        }
        return new RoundTripReport(chunks.Count, failed.Count, failed);
    }
}

/// <summary>Result of <see cref="WireChunking.VerifyRoundTrip"/>.</summary>
/// <param name="Checked">Chunks checked.</param>
/// <param name="Failed">Chunks whose text does not tokenise to their ids.</param>
/// <param name="FailedIds">Their <see cref="Chunk.Id"/>s.</param>
public sealed record RoundTripReport(int Checked, int Failed, IReadOnlyList<string> FailedIds);
