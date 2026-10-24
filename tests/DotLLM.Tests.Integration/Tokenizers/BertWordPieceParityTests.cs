using System.Text.Json;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.WordPiece;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Tokenizers;

/// <summary>
/// WordPiece id parity (issue #738). Expected ids come from HuggingFace <c>tokenizers</c> loading the
/// real <c>sentence-transformers/all-MiniLM-L6-v2</c> <c>tokenizer.json</c> (fixture
/// <c>bert-tokenizer-minilm-hf-expected.json</c>); the same ids must come out of dotLLM's GGUF path
/// (llama.cpp's "▁"-prefixed WPM vocabulary) and its <c>tokenizer.json</c> path.
/// </summary>
public sealed class BertWordPieceParityTests(ITestOutputHelper output)
{
    private const string Description = "all-MiniLM-L6-v2 f16 GGUF (bert, WordPiece)";

    internal static FixtureLocation MiniLmGguf => TestFixtureResolver.ResolveFile(
        "DOTLLM_MINILM_GGUF", "second-state", "All-MiniLM-L6-v2-Embedding-GGUF",
        "all-MiniLM-L6-v2-ggml-model-f16.gguf");

    private static (string Text, int[] Ids)[] LoadExpected()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Embeddings", "bert-tokenizer-minilm-hf-expected.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.EnumerateArray()
            .Select(e => (e.GetProperty("text").GetString()!,
                          e.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray()))
            .ToArray();
    }

    [SkippableFact]
    public void Gguf_wordpiece_ids_match_huggingface()
    {
        var loc = MiniLmGguf;
        Skip.If(!loc.Found, loc.SkipMessage(Description));

        using var gguf = GgufFile.Open(loc.Path!);
        Assert.True(GgufTokenizerFactory.IsWordPiece(gguf.Metadata));
        var tokenizer = GgufTokenizerFactory.LoadWordPiece(gguf.Metadata);

        int bad = 0;
        foreach (var (text, expected) in LoadExpected())
        {
            int[] actual = tokenizer.Encode(text);
            if (!expected.SequenceEqual(actual))
            {
                bad++;
                output.WriteLine($"DIFF '{text}'\n  hf : [{string.Join(",", expected)}]\n  got: [{string.Join(",", actual)}]");
            }
        }
        Assert.Equal(0, bad);
    }

    [SkippableFact]
    public void Tokenizer_json_ids_match_huggingface()
    {
        var loc = TestFixtureResolver.ResolveFile(
            "DOTLLM_MINILM_TOKENIZER_JSON", "sentence-transformers", "all-MiniLM-L6-v2", "tokenizer.json");
        Skip.If(!loc.Found, loc.SkipMessage("all-MiniLM-L6-v2 tokenizer.json"));

        var tokenizer = HfWordPieceLoader.Parse(File.ReadAllText(loc.Path!));
        foreach (var (text, expected) in LoadExpected())
            Assert.True(expected.SequenceEqual(tokenizer.Encode(text)), $"tokenizer.json ids differ for '{text}'");
    }
}
