using System.Collections.Concurrent;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Tokenizers;

/// <summary>
/// Token-id parity tests for the gemma-4 merge-ranked BPE tokenizer against
/// llama.cpp's <c>llama-tokenize</c> oracle (issue #142). Expected id sequences were
/// pinned from <c>llama-tokenize --ids --no-escape</c> (build b-llamacpp-vulkan,
/// 2026-07-16) with the leading BOS (2) stripped — <see cref="BpeTokenizer.Encode"/>
/// does not add BOS. The corpus covers plain English, C# code, accented/CJK text,
/// emoji, leading/trailing/multiple spaces, newlines/tabs, long newline runs,
/// digits/punctuation, special tokens (<c>&lt;|turn&gt;</c>/<c>&lt;turn|&gt;</c>), and
/// the empty string.
/// </summary>
/// <remarks>
/// The 26B-A4B and E4B releases carry byte-identical tokenizer metadata (262144
/// tokens, 514906 merges) — verified during pinning — so both models are asserted
/// against the same expected ids. Skipped per model when the GGUF is absent.
/// </remarks>
public sealed class Gemma4TokenizerParityTests
{
    private const string Path26B =
        "C:/Users/james/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/" +
        "snapshots/3365c68df1a83799b846d05324ebfadbb8cc70b3/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf";
    private const string PathE4B =
        "C:/Users/james/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/" +
        "snapshots/653803f092503c04a65164346f3208a36e707693/gemma-4-E4B-it-Q4_K_M.gguf";

    /// <summary>Tokenizers are expensive to build (262k-entry trie) — cache per model path.</summary>
    private static readonly ConcurrentDictionary<string, BpeTokenizer> Cache = new();

    private static string? Resolve(string model) => model switch
    {
        "26B" => File.Exists(Path26B) ? Path26B : null,
        "E4B" => File.Exists(PathE4B) ? PathE4B : null,
        _ => null,
    };

    private static BpeTokenizer LoadTokenizer(string path) =>
        Cache.GetOrAdd(path, p =>
        {
            using var gguf = GgufFile.Open(p);
            return GgufBpeTokenizerFactory.Load(gguf.Metadata);
        });

    public static IEnumerable<object[]> ParityCorpus()
    {
        foreach (string model in new[] { "26B", "E4B" })
        {
            // plain english
            yield return new object[] { model, "The quick brown fox jumps over the lazy dog.", new int[] { 818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 236761 } };
            // greeting with punctuation
            yield return new object[] { model, "Hello, world! How are you today?", new int[] { 9259, 236764, 1902, 236888, 2088, 659, 611, 3124, 236881 } };
            // C# code snippet
            yield return new object[] { model, "public static int[] Encode(string text)\n{\n    if (text.Length == 0)\n        return [];\n    var result = new List<int>();\n    for (int i = 0; i < text.Length; i++) { result.Add(i * 2); }\n    return result.ToArray();\n}", new int[] { 2153, 2002, 801, 3805, 156512, 236769, 2383, 1816, 236768, 107, 236782, 107, 140, 584, 568, 1005, 236761, 8615, 1251, 236743, 236771, 236768, 107, 144, 2060, 12224, 107, 140, 1967, 1354, 578, 861, 4361, 236820, 720, 9589, 107, 140, 1708, 568, 720, 858, 578, 236743, 236771, 236793, 858, 655, 1816, 236761, 8615, 236793, 858, 4419, 642, 1354, 236761, 3218, 236769, 236747, 808, 236743, 236778, 626, 682, 107, 140, 2060, 1354, 236761, 102395, 1086, 107, 236783 } };
            // multilingual accents (the SentencePiece fallback got this WRONG pre-#142)
            yield return new object[] { model, "Café naïve façade über Straße żółw język", new int[] { 160319, 236859, 120362, 96008, 8046, 80176, 207635, 236765, 119116, 90838 } };
            // CJK
            yield return new object[] { model, "人工知能は未来を変える。한국어 테스트입니다.", new int[] { 66447, 237669, 237230, 237048, 37922, 126836, 15982, 236924, 114216, 237430, 112196, 15245, 236761 } };
            // emoji
            yield return new object[] { model, "Hello 😀🔥 world 🌍✨ test 🤖", new int[] { 9259, 163543, 240785, 1902, 236743, 244906, 239794, 1594, 236743, 246143 } };
            // leading spaces
            yield return new object[] { model, "   leading spaces", new int[] { 139, 26016, 9952 } };
            // trailing spaces
            yield return new object[] { model, "trailing spaces   ", new int[] { 136697, 9952, 139 } };
            // multiple interior spaces
            yield return new object[] { model, "a  b   c    d", new int[] { 236746, 138, 236763, 139, 236755, 140, 236753 } };
            // newlines and tabs (the SentencePiece fallback got this WRONG pre-#142)
            yield return new object[] { model, "line1\nline2\n\nline3\n\n\n\tindented\ttabs\t\n", new int[] { 1257, 236770, 107, 1257, 236778, 108, 1257, 236800, 109, 255968, 724, 16764, 255968, 39218, 255968, 107 } };
            // long newline run (whole-run vocab shortcut, llama.cpp PR #21343)
            yield return new object[] { model, "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n", new int[] { 126 } };
            // gemma-3-style turn markers — NOT in the gemma-4 vocab, must BPE-split
            yield return new object[] { model, "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n", new int[] { 236820, 3041, 236779, 1340, 236779, 887, 236813, 2364, 107, 9259, 236820, 643, 236779, 1340, 236779, 887, 236813, 107, 236820, 3041, 236779, 1340, 236779, 887, 236813, 4368, 107 } };
            // mixed whitespace + accents (the SentencePiece fallback got this WRONG pre-#142)
            yield return new object[] { model, " élève  à\n la  plage ", new int[] { 13124, 39512, 138, 236937, 107, 759, 138, 675, 676, 236743 } };
            // digits and operators
            yield return new object[] { model, "3.14159 == pi; x != 42 && y <= 7 || z >= 100", new int[] { 236800, 236761, 236770, 236812, 236770, 236810, 236819, 1251, 4604, 236793, 1123, 2843, 236743, 236812, 236778, 2732, 570, 6605, 236743, 236832, 3943, 904, 6867, 236743, 236770, 236771, 236771 } };
            // single space
            yield return new object[] { model, " ", new int[] { 236743 } };
            // single newline
            yield return new object[] { model, "\n", new int[] { 107 } };
            // empty string
            yield return new object[] { model, "", Array.Empty<int>() };
            // real gemma-4 special tokens <|turn> (105) / <turn|> (106)
            yield return new object[] { model, "<|turn>user\nHello there<turn|>\n<|turn>model\n", new int[] { 105, 2364, 107, 9259, 993, 106, 107, 105, 4368, 107 } };
        }
    }

    [SkippableTheory]
    [MemberData(nameof(ParityCorpus))]
    public void Encode_MatchesLlamaCppOracle(string model, string text, int[] expected)
    {
        string? path = Resolve(model);
        Skip.If(path is null, $"gemma-4 {model} GGUF not present in the HF hub cache.");

        BpeTokenizer tok = LoadTokenizer(path!);
        int[] got = tok.Encode(text);
        Assert.Equal(expected, got);
    }

    [SkippableFact]
    public void Load_RoutesToMergeRankedBpe_NotSentencePieceFallback()
    {
        string? path = Resolve("E4B");
        Skip.If(path is null, "gemma-4 E4B GGUF not present in the HF hub cache.");

        // Discriminating probe: SentencePiece longest-match and merge-ranked BPE
        // agree on simple prompts but diverge on this accented input. The pinned
        // ids are the llama.cpp oracle output; the pre-#142 fallback produced
        // [236780, 21793, 39512, ...] instead.
        BpeTokenizer tok = LoadTokenizer(path!);
        Assert.Equal(new[] { 160319, 236859 }, tok.Encode("Café")[..2]);
    }

    [SkippableFact]
    public void Decode_RoundtripsParityCorpus()
    {
        string? path = Resolve("E4B");
        Skip.If(path is null, "gemma-4 E4B GGUF not present in the HF hub cache.");

        BpeTokenizer tok = LoadTokenizer(path!);
        foreach (object[] row in ParityCorpus())
        {
            if ((string)row[0] != "E4B") continue;
            string text = (string)row[1];
            if (text.Contains("<|turn>")) continue; // special ids decode to their literal text
            Assert.Equal(text, tok.Decode(tok.Encode(text)));
        }
    }
}
