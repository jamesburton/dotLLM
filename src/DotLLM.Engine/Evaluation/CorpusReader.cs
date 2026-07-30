using System.Text;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Evaluation;

/// <summary>Streams a text corpus into tokens without materializing the whole file or token array.</summary>
/// <remarks>
/// Streaming is a design constraint rather than an optimisation. On unified-memory parts a large
/// VRAM carve-out leaves host RAM scarce, and a standard perplexity corpus tokenizes to hundreds of
/// thousands of ints — held alongside the weights, that is exactly the pressure this harness must
/// not add.
/// </remarks>
public static class CorpusReader
{
    /// <summary>
    /// Reads <paramref name="reader"/> in character chunks, tokenizes each chunk, and yields token
    /// ids in order, stopping after <paramref name="maxTokens"/> (<c>0</c> = unbounded).
    /// </summary>
    /// <remarks>
    /// Chunks are cut at the last whitespace so a token is never split across a boundary; the
    /// remainder is carried into the next chunk, and the final carry is flushed whole.
    /// </remarks>
    /// <param name="reader">Corpus source.</param>
    /// <param name="tokenizer">Tokenizer whose vocabulary the ids belong to.</param>
    /// <param name="maxTokens">Upper bound on emitted tokens; <c>0</c> for unbounded.</param>
    /// <param name="charChunkSize">Characters read per chunk.</param>
    public static IEnumerable<int> StreamTokens(
        TextReader reader, ITokenizer tokenizer, int maxTokens = 0, int charChunkSize = 65536)
    {
        ArgumentNullException.ThrowIfNull(reader);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentOutOfRangeException.ThrowIfLessThan(charChunkSize, 1);

        var buffer = new char[charChunkSize];
        var carry = new StringBuilder();
        int emitted = 0;

        while (true)
        {
            int read = reader.Read(buffer, 0, buffer.Length);
            if (read == 0) break;

            carry.Append(buffer, 0, read);
            string pending = carry.ToString();

            int cut = pending.LastIndexOf(' ');
            if (cut < 0) continue;   // no safe split point yet; keep accumulating

            string ready = pending[..cut];
            carry.Clear();
            carry.Append(pending[(cut + 1)..]);

            foreach (int id in tokenizer.Encode(ready))
            {
                yield return id;
                if (maxTokens > 0 && ++emitted >= maxTokens) yield break;
            }
        }

        if (carry.Length > 0)
        {
            foreach (int id in tokenizer.Encode(carry.ToString()))
            {
                yield return id;
                if (maxTokens > 0 && ++emitted >= maxTokens) yield break;
            }
        }
    }
}
