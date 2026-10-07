using System.Runtime.CompilerServices;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Samplers.StopConditions;

/// <summary>
/// Resolves a tokenizer's end-of-generation (EOG) token set: the declared EOS id plus the well-known
/// end-of-turn control tokens that exist in its vocabulary as single tokens.
/// </summary>
/// <remarks>
/// llama.cpp stops on every EOG token, not only <c>eos_token_id</c>. The distinction matters for
/// Gemma-4: the GGUF declares <c>eos_token_id = 106</c> (<c>&lt;turn|&gt;</c>), but a tool-call turn ends with
/// <c>&lt;eos&gt;</c> (id 1) instead, so stopping on the declared id alone ran every tool call to
/// <c>max_tokens</c> and leaked <c>&lt;eos&gt;&lt;eos&gt;...</c> into the response (#776).
/// A candidate is accepted only when some vocabulary entry's text is exactly the candidate, so a vocabulary
/// without that token adds nothing (found by scanning token text: the real Gemma-4 vocabulary does NOT
/// pre-split <c>&lt;eos&gt;</c>, so encoding the candidate would miss it). Deliberately omitted:
/// <c>&lt;|end|&gt;</c> and <c>&lt;|endoftext|&gt;</c>, which delimit messages / pad rather than end a turn in
/// several families.
/// </remarks>
public static class EndOfGenerationTokens
{
    private static readonly string[] Candidates =
        ["<eos>", "<end_of_turn>", "<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<turn|>", "<|call|>", "<|return|>"];

    private static readonly ConditionalWeakTable<ITokenizer, int[]> Cache = new();

    /// <summary>Returns the distinct EOG token ids for <paramref name="tokenizer"/> (always includes the declared EOS id).</summary>
    public static int[] Resolve(ITokenizer tokenizer)
        => Cache.GetValue(tokenizer, Compute);

    /// <summary>Builds the stop condition that ends generation on any EOG token.</summary>
    public static EosStopCondition CreateStopCondition(ITokenizer tokenizer)
        => new(Resolve(tokenizer));

    private static int[] Compute(ITokenizer tokenizer)
    {
        var ids = new List<int> { tokenizer.EosTokenId };

        // Ids the model file itself declares as end-of-turn / end-of-message (GGUF eot / eom).
        foreach (int extra in tokenizer.ExtraEndOfGenerationTokenIds)
            if (extra >= 0 && !ids.Contains(extra))
                ids.Add(extra);

        // Scan the vocabulary by token TEXT rather than encoding the candidates: Gemma-4's "<eos>" (id 1) is
        // not pre-split as a special token, so Encode("<eos>") yields the literal characters, not [1].
        // One pass per tokenizer (cached), ~262k string compares for a 256k vocabulary.
        int vocab = tokenizer.VocabSize;
        int endId = -1, returnId = -1, callId = -1;     // OpenAI Harmony markers, for the <|end|> rule below
        for (int id = 0; id < vocab; id++)
        {
            string text;
            try
            {
                text = tokenizer.DecodeToken(id);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                // A probe, not a request: ids a tokenizer cannot decode (test doubles, holes in the
                // vocabulary) simply contribute nothing.
                continue;
            }

            switch (text)
            {
                case "<|end|>": endId = id; break;
                case "<|return|>": returnId = id; break;
                case "<|call|>": callId = id; break;
            }

            if (text.Length is >= 5 and <= 14 && text[0] == '<' && Array.IndexOf(Candidates, text) >= 0
                && !ids.Contains(id))
                ids.Add(id);
        }

        // llama.cpp (llama-vocab.cpp): when a vocabulary has BOTH <|return|> and <|call|> it is a Harmony (gpt-oss)
        // vocabulary, where <|end|> closes the analysis message and the turn continues with the final / commentary
        // channel. A model file that declared <|end|> as eot/eom would end the turn after the thinking (#798), so it
        // is never end-of-generation there, whatever the file says.
        if (endId >= 0 && returnId >= 0 && callId >= 0 && endId != tokenizer.EosTokenId)
            ids.Remove(endId);

        return ids.ToArray();
    }
}
