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
    /// <param name="bosTokenId">
    /// BOS id to prepend to the stream, or <c>-1</c> for none. Derived from the vocab via
    /// <c>GgufAddBosResolver</c> rather than chosen by the caller: llama.cpp prepends BOS when the
    /// vocab asks for it, and a stream that omits it is offset by one token against llama.cpp's
    /// at every chunk boundary (issue #515).
    /// </param>
    public static IEnumerable<int> StreamTokens(
        TextReader reader, ITokenizer tokenizer, int maxTokens = 0, int charChunkSize = 65536,
        int bosTokenId = -1)
    {
        ArgumentNullException.ThrowIfNull(reader);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentOutOfRangeException.ThrowIfLessThan(charChunkSize, 1);

        var buffer = new char[charChunkSize];
        var carry = new StringBuilder();
        int emitted = 0;

        // Prepended to the stream BEFORE chunking, which is what llama.cpp does and therefore
        // what makes chunk N cover the same text on both sides (issue #515/#516). It counts
        // toward maxTokens for the same reason: it is simply the stream's first token.
        if (bosTokenId >= 0)
        {
            yield return bosTokenId;
            if (maxTokens > 0 && ++emitted >= maxTokens) yield break;
        }

        while (true)
        {
            int read = reader.Read(buffer, 0, buffer.Length);
            if (read == 0) break;

            carry.Append(buffer, 0, read);
            string pending = carry.ToString();

            int cut = pending.LastIndexOf(' ');
            if (cut < 0) continue;   // no safe split point yet; keep accumulating

            // The separating space is carried INTO the next chunk, not dropped. GPT-2-style BPE
            // encodes a leading space as part of the following token, so dropping it silently
            // changes the token stream — and therefore the perplexity — versus tokenizing the
            // corpus in one pass.
            string ready = pending[..cut];
            carry.Clear();
            carry.Append(pending[cut..]);

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
