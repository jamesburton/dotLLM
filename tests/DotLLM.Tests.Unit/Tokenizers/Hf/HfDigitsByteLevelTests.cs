using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Tokenizers.Hf;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.Hf;

public sealed class HfDigitsByteLevelTests
{
    private const string PreTokenizer = """
        {"type":"Sequence","pretokenizers":[
          {"type":"Digits","individual_digits":true},
          {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}
        """;

    [Theory]
    [InlineData("pretokenizers")]
    [InlineData("pre_tokenizers")]
    public void Parser_RecognizesOnlyTheQualifiedDigitsPipeline(string sequenceKey)
    {
        var fixture = Synthetic(PreTokenizer.Replace("pretokenizers", sequenceKey, StringComparison.Ordinal));
        HfTokenizerSpec spec = HfTokenizerJsonParser.Parse(fixture.Json);
        Assert.Equal(HfPreTokenizerKind.DigitsByteLevel, spec.PreTokenizerKind);
        Assert.True(spec.ByteLevelUseRegex);
        Assert.False(spec.ByteLevelAddPrefixSpace);
        Assert.Null(spec.ByteLevelSplitRegex);
    }

    [Fact]
    public void Encoding_PreservesWhitespaceBoundariesAndDoesNotAddPrefixSpace()
    {
        var fixture = Synthetic(PreTokenizer);
        ITokenizer tokenizer = HfBpeTokenizerFactory.Create(fixture.Json);
        Assert.Equal(new[] { 65, fixture.Id("  "), 49, 50, 10, 66 }, tokenizer.Encode("A  12\nB"));
        Assert.Equal(new[] { 49, 50 }, tokenizer.Encode("12"));
        Assert.Equal(new[] { 32, 49, 50 }, tokenizer.Encode(" 12"));
        Assert.Equal(new[] { 120 }, tokenizer.Encode("x"));
        Assert.Equal(new[] { fixture.Id(" x") }, tokenizer.Encode(" x"));
        Assert.Equal(new[] { 104, 105, fixture.Id("!?"), 49, 50 }, tokenizer.Encode("hi!?12"));
        Assert.Equal("A  12\nB", tokenizer.Decode(tokenizer.Encode("A  12\nB")));
    }

    [Theory]
    [InlineData("١", "٢")]
    [InlineData("𝟙", "𝟚")]
    public void Encoding_IsolatesUnicodeNumbersEvenWhenTheVocabularyCanMergeThem(string first, string second)
    {
        var fixture = Synthetic(PreTokenizer);
        ITokenizer tokenizer = HfBpeTokenizerFactory.Create(fixture.Json);
        Assert.Equal(new[] { fixture.Id(first), fixture.Id(second) }, tokenizer.Encode(first + second));
        Assert.Equal(new[] { 65, fixture.Id("  "), fixture.Id(first), fixture.Id(second) }, tokenizer.Encode("A  " + first + second));
        Assert.Equal(first + second, tokenizer.Decode(tokenizer.Encode(first + second)));
    }

    [Fact]
    public void SharedGgufSmolLmPipeline_AlsoIsolatesAstralNumbers()
    {
        var fixture = Synthetic(PreTokenizer);
        var tokenizer = BpeTokenizer.CreateTiktoken(
            fixture.Vocab.OrderBy(item => item.Value).Select(item => item.Key).ToArray(),
            fixture.Merges.Select(pair => string.Join(' ', pair)).ToArray(), null, -1, -1, "smollm");
        Assert.Equal(new[] { 65, fixture.Id("  "), fixture.Id("𝟙"), fixture.Id("𝟚") }, tokenizer.Encode("A  𝟙𝟚"));
    }

    [Theory]
    [InlineData("grouped-digits")]
    [InlineData("missing-digits-option")]
    [InlineData("malformed-digits-option")]
    [InlineData("prefix-space")]
    [InlineData("no-regex")]
    [InlineData("no-trim")]
    [InlineData("missing-trim")]
    [InlineData("extra-digit-option")]
    [InlineData("extra-byte-option")]
    [InlineData("reversed")]
    [InlineData("extra-stage")]
    public void UnsupportedDigitsOptionsAndArrangements_AreRejected(string variant)
    {
        JsonObject pre = JsonNode.Parse(PreTokenizer)!.AsObject();
        JsonArray steps = pre["pretokenizers"]!.AsArray();
        switch (variant)
        {
            case "grouped-digits": steps[0]!["individual_digits"] = false; break;
            case "missing-digits-option": steps[0]!.AsObject().Remove("individual_digits"); break;
            case "malformed-digits-option": steps[0]!["individual_digits"] = "true"; break;
            case "prefix-space": steps[1]!["add_prefix_space"] = true; break;
            case "no-regex": steps[1]!["use_regex"] = false; break;
            case "no-trim": steps[1]!["trim_offsets"] = false; break;
            case "missing-trim": steps[1]!.AsObject().Remove("trim_offsets"); break;
            case "extra-digit-option": steps[0]!["unknown_option"] = true; break;
            case "extra-byte-option": steps[1]!["unknown_option"] = true; break;
            case "reversed": pre["pretokenizers"] = new JsonArray(steps[1]!.DeepClone(), steps[0]!.DeepClone()); break;
            case "extra-stage": steps.Add(steps[1]!.DeepClone()); break;
            default: throw new ArgumentOutOfRangeException(nameof(variant));
        }
        var fixture = Synthetic(pre.ToJsonString());
        Assert.Equal(HfPreTokenizerKind.Sequence, HfTokenizerJsonParser.Parse(fixture.Json).PreTokenizerKind);
        Assert.Throws<InvalidDataException>(() => HfBpeTokenizerFactory.Create(fixture.Json));
    }

    // HF tokenizers==0.21.4, official SmolLM2-135M-Instruct revision
    // 12fd25f77366fa6b3b4b768ec3050bf629380bac, encode(add_special_tokens=False).
    // The repeated spaces before digits distinguish this pipeline on the REAL vocab;
    // plain "1234" alone does not distinguish it from a wrong GPT-2 substitution.
    private static readonly (string Text, int[] Ids)[] RealReference =
    [
        ("", []),
        ("Hello world", [19556, 905]),
        ("A  1234\nB", [49, 256, 33, 34, 35, 36, 198, 50]),
        (" 12  34   56", [216, 33, 34, 256, 35, 36, 333, 37, 38]),
        ("v1234.56", [102, 33, 34, 35, 36, 30, 37, 38]),
        ("2026-11-14, 09:30!", [34, 32, 34, 38, 29, 33, 33, 29, 33, 36, 28, 216, 32, 41, 42, 35, 32, 17]),
        ("\t\n  12345\r\nZ", [197, 11181, 33, 34, 35, 36, 37, 201, 198, 74]),
        ("café١٢٣４５６😀", [83, 1939, 2756, 164, 111, 164, 112, 164, 113, 8083, 238, 8083, 239, 8083, 240, 10813, 242, 218]),
        ("e\u0301  १२३", [85, 151, 219, 256, 9310, 117, 9310, 118, 9310, 119]),
        ("They're  123, we'll  456.", [4948, 2316, 256, 33, 34, 35, 28, 392, 3060, 256, 36, 37, 38, 30]),
        ("𝟙𝟚  123", [23159, 249, 243, 23159, 249, 244, 256, 33, 34, 35]),
        ("A  𝟙𝟚", [49, 256, 23159, 249, 243, 23159, 249, 244]),
        ("²³ⅣⅤ", [19133, 39050, 173, 223, 113, 173, 223, 114]),
    ];

    [SkippableFact]
    public void RealSmolLm_MatchesPinnedHuggingFaceIdsAndRoundTrips()
    {
        string json = ReadRealTokenizer();
        ITokenizer tokenizer = HfBpeTokenizerFactory.Create(json);
        foreach ((string text, int[] ids) in RealReference)
        {
            Assert.Equal(ids, tokenizer.Encode(text));
            Assert.Equal(text, tokenizer.Decode(ids));
        }
    }

    [SkippableFact]
    public void RealSmolLm_GenericGpt2SubstitutionChangesMultiDigitBoundaries()
    {
        string json = ReadRealTokenizer();
        JsonNode wrong = JsonNode.Parse(json)!;
        wrong["pre_tokenizer"] = wrong["pre_tokenizer"]!["pretokenizers"]![1]!.DeepClone();
        const string text = "A  1234\nB";
        int[] correct = HfBpeTokenizerFactory.Create(json).Encode(text);
        int[] substituted = HfBpeTokenizerFactory.Create(wrong.ToJsonString()).Encode(text);
        Assert.Equal(new[] { 49, 256, 33, 34, 35, 36, 198, 50 }, correct);
        Assert.Equal(new[] { 49, 216, 216, 33, 34, 35, 36, 198, 50 }, substituted);
        Assert.False(correct.SequenceEqual(substituted));
    }

    private static string ReadRealTokenizer()
    {
        string path = Environment.GetEnvironmentVariable("DOTLLM_TEST_SMOLLM2_TOKENIZER_JSON")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "models", "HuggingFaceTB", "SmolLM2-135M-Instruct-F16-derived", "tokenizer.json");
        if (Environment.GetEnvironmentVariable("DOTLLM_TEST_HF_TOKENIZER_REQUIRED") is "1")
            Assert.True(File.Exists(path), $"Required SmolLM2 tokenizer fixture is absent: {path}");
        Skip.IfNot(File.Exists(path), "Official SmolLM2 tokenizer fixture is not installed.");
        byte[] bytes = File.ReadAllBytes(path);
        Assert.Equal("9CA9ACDDB6525A194EC8AC7A87F24FBBA7232A9A15FFA1AF0C1224FCD888E47C", Convert.ToHexString(SHA256.HashData(bytes)));
        return Encoding.UTF8.GetString(bytes);
    }

    private sealed record Fixture(string Json, Dictionary<string, int> Vocab, List<string[]> Merges)
    {
        public int Id(string text) => Vocab[ByteLevelPreTokenizer.ApplyByteLevel(text)];
    }

    private static Fixture Synthetic(string preTokenizer)
    {
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < 256; i++) vocab[ByteLevelPreTokenizer.BytesToUnicode[i].ToString()] = i;
        var merges = new List<string[]>();
        void Pair(string left, string right)
        {
            if (vocab.ContainsKey(left + right)) return;
            vocab.Add(left + right, vocab.Count);
            merges.Add([left, right]);
        }
        void Token(string text)
        {
            string mapped = ByteLevelPreTokenizer.ApplyByteLevel(text), prefix = mapped[..1];
            for (int i = 1; i < mapped.Length; i++) { Pair(prefix, mapped[i].ToString()); prefix += mapped[i]; }
        }
        foreach (string text in new[] { "12", "  ", " 1", "!?", " x", "١", "٢", "𝟙", "𝟚" }) Token(text);
        Pair(ByteLevelPreTokenizer.ApplyByteLevel("١"), ByteLevelPreTokenizer.ApplyByteLevel("٢"));
        Pair(ByteLevelPreTokenizer.ApplyByteLevel("𝟙"), ByteLevelPreTokenizer.ApplyByteLevel("𝟚"));
        string json = JsonSerializer.Serialize(new
        {
            pre_tokenizer = JsonNode.Parse(preTokenizer), decoder = new { type = "ByteLevel" },
            model = new { type = "BPE", vocab, merges },
        });
        return new Fixture(json, vocab, merges);
    }
}
