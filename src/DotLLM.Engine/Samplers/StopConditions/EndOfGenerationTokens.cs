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
/// A candidate is accepted only when it encodes to exactly one token that decodes back to itself, so
/// on a vocabulary where e.g. <c>&lt;eos&gt;</c> is ordinary text nothing is added. Deliberately omitted:
/// <c>&lt;|end|&gt;</c> and <c>&lt;|endoftext|&gt;</c>, which delimit messages / pad rather than end a turn in
/// several families.
/// </remarks>
public static class EndOfGenerationTokens
{
    private static readonly string[] Candidates =
        ["<eos>", "<end_of_turn>", "<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<turn|>"];

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
        foreach (string candidate in Candidates)
        {
            try
            {
                int[] enc = tokenizer.Encode(candidate);
                if (enc.Length == 1 && enc[0] >= 0 && !ids.Contains(enc[0])
                    && string.Equals(tokenizer.DecodeToken(enc[0]), candidate, StringComparison.Ordinal))
                    ids.Add(enc[0]);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                // A probe, not a request: a tokenizer that cannot encode the candidate (test doubles that
                // parse ids from text, vocabularies without byte fallback, ...) simply contributes nothing.
            }
        }

        return ids.ToArray();
    }
}
