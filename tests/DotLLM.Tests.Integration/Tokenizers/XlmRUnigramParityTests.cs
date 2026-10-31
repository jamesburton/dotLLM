using System.Text.Json;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Unigram;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Tokenizers;

/// <summary>
/// XLM-R SentencePiece Unigram parity (issue #1401). Fixture <c>hf-xlmr-multilingual-e5-small.json</c> holds, per text, the
/// ids from HuggingFace <c>AutoTokenizer</c> (<c>intfloat/multilingual-e5-small</c>, the gate) and from llama.cpp b9016
/// <c>/tokenize</c> on the same GGUF. 63 curated multilingual strings (Latin/diacritics, Cyrillic, Arabic, Devanagari,
/// Thai/CJK/Japanese, emoji, NFKC cases, whitespace edge cases, special-token literals) plus 150 random vocab/odd-codepoint
/// strings match HF token for token. A further 23 random strings, group <c>hf-grapheme-quirk</c>, differ: HF's
/// precompiled-charsmap normaliser works per grapheme cluster and drops the tail of a cluster after a
/// compatibility-mapped first character (U+FB01 + U+0301 gives "fi"), where SentencePiece and llama.cpp keep it.
/// dotLLM follows llama.cpp there (3000/3000 identical over the full random set), and the test pins both sides.
/// </summary>
public sealed class XlmRUnigramParityTests(ITestOutputHelper output)
{
    private const string Description = "multilingual-e5-small f16 GGUF (bert, XLM-R Unigram)";

    internal static FixtureLocation Gguf => TestFixtureResolver.ResolveFile(
        "DOTLLM_MLE5_F16_GGUF", "hdrrayan", "multilingual-e5-small-gguf", "multilingual-e5-small-f16.gguf");

    private sealed record Case(string Text, int[] Hf, int[] Llama, string Group);

    private static Case[] Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Embeddings", "hf-xlmr-multilingual-e5-small.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("cases").EnumerateArray()
            .Select(e => new Case(
                e.GetProperty("text").GetString()!,
                e.GetProperty("hf").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                e.GetProperty("llama").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                e.GetProperty("group").GetString()!))
            .ToArray();
    }

    [SkippableFact]
    public void Gguf_dispatches_to_the_unigram_tokenizer()
    {
        var loc = Gguf;
        Skip.If(!loc.Found, loc.SkipMessage(Description));
        using var gguf = GgufFile.Open(loc.Path!);
        Assert.True(GgufTokenizerFactory.IsUnigram(gguf.Metadata));
        Assert.IsType<UnigramTokenizer>(GgufTokenizerFactory.Load(gguf.Metadata));
    }

    [SkippableFact]
    public void Ids_match_huggingface_token_for_token()
    {
        var loc = Gguf;
        Skip.If(!loc.Found, loc.SkipMessage(Description));
        using var gguf = GgufFile.Open(loc.Path!);
        var tokenizer = GgufTokenizerFactory.LoadUnigram(gguf.Metadata);

        int bad = 0, checkedCount = 0;
        foreach (var c in Load().Where(c => c.Group != "hf-grapheme-quirk"))
        {
            checkedCount++;
            int[] actual = tokenizer.Encode(c.Text);
            if (!c.Hf.SequenceEqual(actual))
            {
                bad++;
                output.WriteLine($"DIFF '{c.Text}'\n  hf : [{string.Join(",", c.Hf)}]\n  got: [{string.Join(",", actual)}]");
            }
        }
        output.WriteLine($"{checkedCount} texts, {bad} differ from HF");
        Assert.Equal(0, bad);
    }

    [SkippableFact]
    public void Ids_match_llamacpp_including_where_huggingface_differs()
    {
        var loc = Gguf;
        Skip.If(!loc.Found, loc.SkipMessage(Description));
        using var gguf = GgufFile.Open(loc.Path!);
        var tokenizer = GgufTokenizerFactory.LoadUnigram(gguf.Metadata);

        var cases = Load();
        Assert.Contains(cases, c => c.Group == "hf-grapheme-quirk");
        foreach (var c in cases)
        {
            Assert.True(c.Llama.SequenceEqual(tokenizer.Encode(c.Text)), $"ids differ from llama.cpp for '{c.Text}'");
            if (c.Group == "hf-grapheme-quirk")
                Assert.False(c.Hf.SequenceEqual(c.Llama), "fixture says HF differs here");
        }
    }
}
