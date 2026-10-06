using System.Text.Json;

namespace DotLLM.Server.Models;

/// <summary>
/// Parses the OpenAI <c>input</c> union of <c>POST /v1/embeddings</c> into a list of
/// per-item token sequences.
/// </summary>
/// <remarks>
/// The four accepted shapes, and how they are disambiguated:
/// <list type="bullet">
///   <item><description><c>"text"</c> — a single string.</description></item>
///   <item><description><c>["a","b"]</c> — array whose first element is a string ⇒ many texts.</description></item>
///   <item><description><c>[1,2,3]</c> — array whose first element is a number ⇒ <b>one</b>
///     pre-tokenised sequence.</description></item>
///   <item><description><c>[[1,2],[3,4]]</c> — array whose first element is an array ⇒ many
///     pre-tokenised sequences.</description></item>
/// </list>
/// Mixed-kind arrays, empty arrays, empty strings, empty token sequences and non-integer or
/// out-of-vocabulary token ids are all rejected — a caller that meant one of them is better served
/// by a 400 than by a silently plausible vector.
/// </remarks>
public static class EmbeddingInputParser
{
    /// <summary>Outcome of parsing: either the token sequences, or an error message for a 400.</summary>
    /// <param name="Sequences">One token-id array per input item, in input order.</param>
    /// <param name="Error">Non-null when parsing failed; the message for the 400 response.</param>
    public readonly record struct Result(IReadOnlyList<int[]>? Sequences, string? Error)
    {
        /// <summary><c>true</c> when parsing succeeded.</summary>
        public bool Ok => Error is null;
    }

    /// <summary>
    /// Parses <paramref name="input"/>, tokenising strings with <paramref name="encode"/> and
    /// range-checking pre-tokenised ids against <paramref name="vocabSize"/>.
    /// </summary>
    /// <param name="input">The raw <c>input</c> element.</param>
    /// <param name="encode">Tokenizer callback for string items.</param>
    /// <param name="vocabSize">Vocabulary size used to range-check pre-tokenised ids.</param>
    public static Result Parse(JsonElement input, Func<string, int[]> encode, int vocabSize)
    {
        switch (input.ValueKind)
        {
            case JsonValueKind.Undefined:
            case JsonValueKind.Null:
                return new Result(null, "'input' is required.");

            case JsonValueKind.String:
                {
                    string text = input.GetString() ?? "";
                    if (text.Length == 0)
                        return new Result(null, "'input' must not be an empty string.");
                    int[] tokens = encode(text);
                    if (tokens.Length == 0)
                        return new Result(null, "'input' tokenised to zero tokens.");
                    return new Result([tokens], null);
                }

            case JsonValueKind.Array:
                break;

            default:
                return new Result(null,
                    "'input' must be a string, an array of strings, an array of token ids, or an array of token-id arrays.");
        }

        int count = input.GetArrayLength();
        if (count == 0)
            return new Result(null, "'input' must not be an empty array.");

        JsonValueKind firstKind = JsonValueKind.Undefined;
        foreach (var element in input.EnumerateArray())
        {
            firstKind = element.ValueKind;
            break;
        }

        switch (firstKind)
        {
            case JsonValueKind.String:
                {
                    var sequences = new List<int[]>(count);
                    int index = 0;
                    foreach (var element in input.EnumerateArray())
                    {
                        if (element.ValueKind != JsonValueKind.String)
                            return new Result(null, $"'input[{index}]' is {element.ValueKind}, expected a string (the array's first element was a string).");
                        string text = element.GetString() ?? "";
                        if (text.Length == 0)
                            return new Result(null, $"'input[{index}]' must not be an empty string.");
                        int[] tokens = encode(text);
                        if (tokens.Length == 0)
                            return new Result(null, $"'input[{index}]' tokenised to zero tokens.");
                        sequences.Add(tokens);
                        index++;
                    }
                    return new Result(sequences, null);
                }

            case JsonValueKind.Number:
                {
                    // A flat number array is ONE pre-tokenised sequence.
                    var result = ReadTokenArray(input, vocabSize, "input");
                    return result.Error is not null ? new Result(null, result.Error) : new Result([result.Tokens!], null);
                }

            case JsonValueKind.Array:
                {
                    var sequences = new List<int[]>(count);
                    int index = 0;
                    foreach (var element in input.EnumerateArray())
                    {
                        if (element.ValueKind != JsonValueKind.Array)
                            return new Result(null, $"'input[{index}]' is {element.ValueKind}, expected an array of token ids (the array's first element was an array).");
                        var read = ReadTokenArray(element, vocabSize, $"input[{index}]");
                        if (read.Error is not null)
                            return new Result(null, read.Error);
                        sequences.Add(read.Tokens!);
                        index++;
                    }
                    return new Result(sequences, null);
                }

            default:
                return new Result(null,
                    $"'input[0]' is {firstKind}; an 'input' array must hold strings, token ids, or token-id arrays.");
        }
    }

    private static (int[]? Tokens, string? Error) ReadTokenArray(JsonElement array, int vocabSize, string path)
    {
        int n = array.GetArrayLength();
        if (n == 0)
            return (null, $"'{path}' must not be an empty token array.");

        var tokens = new int[n];
        int i = 0;
        foreach (var element in array.EnumerateArray())
        {
            if (element.ValueKind != JsonValueKind.Number || !element.TryGetInt32(out int id))
                return (null, $"'{path}[{i}]' is not an integer token id.");
            if (id < 0 || id >= vocabSize)
                return (null, $"'{path}[{i}]' = {id} is outside the model vocabulary [0, {vocabSize}).");
            tokens[i++] = id;
        }

        return (tokens, null);
    }
}
