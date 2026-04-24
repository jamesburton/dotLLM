using System.Text;
using System.Text.Json;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Hf;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.Hf;

/// <summary>
/// End-to-end round-trip tests for the ByteLevel (GPT-2 / Qwen / Granite)
/// pipeline in the HF <c>tokenizer.json</c> adapter. Each test loads a real
/// checkpoint's <c>tokenizer.json</c> from disk and validates encode/decode
/// parity against token IDs produced by the reference implementation
/// (<c>transformers.AutoTokenizer</c>). Expected IDs are baked as
/// constants — no Python is invoked at test time.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generating the expected IDs.</b> Run:
/// </para>
/// <code>
/// from transformers import AutoTokenizer
/// tok = AutoTokenizer.from_pretrained('C:/temp/dotllm-qwen25-0.5b')
/// tok.encode('Hello world', add_special_tokens=False)
/// </code>
/// <para>
/// <b>Covered checkpoints:</b> Qwen2.5-0.5B (Sequence[Split, ByteLevel] + NFC),
/// IBM Granite-3-2b-instruct (standalone ByteLevel with use_regex=true),
/// Phi-3.5-mini and TinyLlama-1.1B (existing Metaspace path — regression
/// guard after the factory refactor).
/// </para>
/// <para>
/// Tests are skipped when the corresponding <c>C:/temp/dotllm-*</c>
/// directory is absent, so the suite runs cleanly on CI without the
/// downloads.
/// </para>
/// </remarks>
public class HfByteLevelBpeTokenizerTests
{
    private const string QwenDir = "C:/temp/dotllm-qwen25-0.5b";
    private const string GraniteDir = "C:/temp/dotllm-granite3-2b-inst";
    private const string Phi35Dir = "C:/temp/dotllm-phi35-mini";
    private const string TinyLlamaDir = "C:/temp/dotllm-tinyllama";

    // ---------------------------------------------------------------------
    // Baked HF AutoTokenizer.encode(add_special_tokens=False) baselines.
    // ---------------------------------------------------------------------

    private static readonly (string Input, int[] Expected)[] QwenExpected =
    [
        ("Hello world",              [9707, 1879]),
        ("The capital of France is", [785, 6722, 315, 9625, 374]),
        ("",                         []),
        (" ",                        [220]),
        ("éàü",                      [963, 6362, 2391]),
        ("Hello!",                   [9707, 0]),
        ("\U0001F600",               [141334]), // 😀
    ];

    private static readonly (string Input, int[] Expected)[] GraniteExpected =
    [
        ("Hello world",              [8279, 5788]),
        ("The capital of France is", [1318, 18926, 432, 45600, 438]),
        ("",                         []),
        (" ",                        [225]),
        ("éàü",                      [1309, 6048, 2006]),
        ("Hello!",                   [8279, 19]),
        ("\U0001F600",               [36628, 227]),
    ];

    private static readonly (string Input, int[] Expected)[] Phi35Expected =
    [
        ("Hello world",              [15043, 3186]),
        ("The capital of France is", [450, 7483, 310, 3444, 338]),
        ("",                         []),
        (" ",                        [29871]),
        ("éàü",                      [904, 30001, 29993]),
        ("Hello!",                   [15043, 29991]),
    ];

    private static readonly (string Input, int[] Expected)[] TinyLlamaExpected =
    [
        ("Hello world",              [15043, 3186]),
        ("The capital of France is", [450, 7483, 310, 3444, 338]),
        ("",                         []),
        (" ",                        [29871]),
        ("éàü",                      [904, 30001, 29993]),
        ("Hello!",                   [15043, 29991]),
    ];

    // =====================================================================
    // Parser: pipeline detection round-trip.
    // =====================================================================

    [SkippableFact]
    public void Parser_Qwen25_DetectsSequenceSplitByteLevel()
    {
        SkipIfMissing(QwenDir);
        HfTokenizerSpec spec = LoadSpec(QwenDir);

        Assert.Equal(HfPreTokenizerKind.ByteLevel, spec.PreTokenizerKind);
        Assert.False(string.IsNullOrEmpty(spec.ByteLevelSplitRegex));
        // Spot-check the Qwen2 split pattern: the (?i:'s|...) head is unique.
        Assert.Contains("(?i:", spec.ByteLevelSplitRegex!);
        Assert.Equal(HfNormalizerKind.Nfc, spec.NormalizerKind);
        Assert.Contains(HfDecoderStage.ByteLevel, spec.DecoderStages);
    }

    [SkippableFact]
    public void Parser_Granite3_DetectsStandaloneByteLevel()
    {
        SkipIfMissing(GraniteDir);
        HfTokenizerSpec spec = LoadSpec(GraniteDir);

        Assert.Equal(HfPreTokenizerKind.ByteLevel, spec.PreTokenizerKind);
        // Standalone ByteLevel: no Split regex, use_regex=true drives the default GPT-2 pattern.
        Assert.True(string.IsNullOrEmpty(spec.ByteLevelSplitRegex));
        Assert.True(spec.ByteLevelUseRegex ?? false);
        Assert.Contains(HfDecoderStage.ByteLevel, spec.DecoderStages);
    }

    [SkippableFact]
    public void Parser_TinyLlama_DetectsMetaspacePath()
    {
        SkipIfMissing(TinyLlamaDir);
        HfTokenizerSpec spec = LoadSpec(TinyLlamaDir);

        // TinyLlama ships pre_tokenizer=null but a Metaspace-style decoder.
        // We still route it through the SPM path by virtue of PreTokenizerKind=None.
        Assert.Equal(HfPreTokenizerKind.None, spec.PreTokenizerKind);
        Assert.True(spec.ByteFallback);
        Assert.Contains(HfDecoderStage.ByteFallback, spec.DecoderStages);
    }

    // =====================================================================
    // Encode: exact-ID parity with HF AutoTokenizer.
    // =====================================================================

    [SkippableFact]
    public void Encode_Qwen25_MatchesHfReferenceIds()
    {
        SkipIfMissing(QwenDir);
        ITokenizer tok = HfBpeTokenizerFactory.TryLoadFromDirectory(QwenDir)!;
        AssertEncodeMatches(tok, QwenExpected);
    }

    [SkippableFact]
    public void Encode_Granite3_MatchesHfReferenceIds()
    {
        SkipIfMissing(GraniteDir);
        ITokenizer tok = HfBpeTokenizerFactory.TryLoadFromDirectory(GraniteDir)!;
        AssertEncodeMatches(tok, GraniteExpected);
    }

    [SkippableFact]
    public void Encode_Phi35_MatchesHfReferenceIds_MetaspaceRegressionGuard()
    {
        SkipIfMissing(Phi35Dir);
        ITokenizer tok = HfBpeTokenizerFactory.TryLoadFromDirectory(Phi35Dir)!;
        AssertEncodeMatches(tok, Phi35Expected);
    }

    [SkippableFact]
    public void Encode_TinyLlama_MatchesHfReferenceIds_MetaspaceRegressionGuard()
    {
        SkipIfMissing(TinyLlamaDir);
        ITokenizer tok = HfBpeTokenizerFactory.TryLoadFromDirectory(TinyLlamaDir)!;
        AssertEncodeMatches(tok, TinyLlamaExpected);
    }

    // =====================================================================
    // Decode round-trip.
    // =====================================================================

    [SkippableFact]
    public void Decode_Qwen25_RoundTripsAllSamples()
    {
        SkipIfMissing(QwenDir);
        ITokenizer tok = HfBpeTokenizerFactory.TryLoadFromDirectory(QwenDir)!;
        AssertDecodeRoundTrips(tok, QwenExpected);
    }

    [SkippableFact]
    public void Decode_Granite3_RoundTripsAllSamples()
    {
        SkipIfMissing(GraniteDir);
        ITokenizer tok = HfBpeTokenizerFactory.TryLoadFromDirectory(GraniteDir)!;
        AssertDecodeRoundTrips(tok, GraniteExpected);
    }

    // =====================================================================
    // ByteLevel + Split Sequence composition explicit coverage.
    // =====================================================================

    [Fact]
    public void ParseSequenceSplitByteLevel_Synthetic_CapturesRegexAndFlags()
    {
        // A hand-rolled tokenizer.json with exactly the Qwen2 composition,
        // excluding the huge vocab — just enough for the parser to recognise
        // the pipeline shape.
        const string json = """
        {
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              { "type": "Split",
                "pattern": { "Regex": "\\s+" },
                "behavior": "Isolated",
                "invert": false },
              { "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": false }
            ]
          },
          "decoder": { "type": "ByteLevel" },
          "normalizer": { "type": "NFC" },
          "model": { "type": "BPE", "vocab": {}, "merges": [] }
        }
        """;

        HfTokenizerSpec spec = HfTokenizerJsonParser.Parse(json);
        Assert.Equal(HfPreTokenizerKind.ByteLevel, spec.PreTokenizerKind);
        Assert.Equal(@"\s+", spec.ByteLevelSplitRegex);
        Assert.False(spec.ByteLevelUseRegex ?? true);
        Assert.False(spec.ByteLevelAddPrefixSpace ?? true);
        Assert.Equal(HfNormalizerKind.Nfc, spec.NormalizerKind);
        Assert.Single(spec.DecoderStages);
        Assert.Equal(HfDecoderStage.ByteLevel, spec.DecoderStages[0]);
    }

    [Fact]
    public void ParseStandaloneByteLevel_CapturesUseRegexFlag()
    {
        const string json = """
        {
          "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
          },
          "decoder": { "type": "ByteLevel" },
          "model": { "type": "BPE", "vocab": {}, "merges": [] }
        }
        """;

        HfTokenizerSpec spec = HfTokenizerJsonParser.Parse(json);
        Assert.Equal(HfPreTokenizerKind.ByteLevel, spec.PreTokenizerKind);
        Assert.Null(spec.ByteLevelSplitRegex); // no Sequence wrapper
        Assert.True(spec.ByteLevelUseRegex);
        Assert.False(spec.ByteLevelAddPrefixSpace);
    }

    [Fact]
    public void ParseSequence_UnknownComposition_ReturnsSequence()
    {
        // A Sequence of [Digits, ByteLevel] is not the supported [Split, ByteLevel]
        // shape — parser must surface it as Sequence so the factory rejects it.
        const string json = """
        {
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              { "type": "Digits" },
              { "type": "ByteLevel", "use_regex": false }
            ]
          },
          "model": { "type": "BPE", "vocab": {}, "merges": [] }
        }
        """;

        HfTokenizerSpec spec = HfTokenizerJsonParser.Parse(json);
        Assert.Equal(HfPreTokenizerKind.Sequence, spec.PreTokenizerKind);
    }

    // =====================================================================
    // NormalizingTokenizer — short-circuits when input is already normalized.
    // Regression guard for the P0.1 follow-up that avoids the allocation from
    // string.Normalize(form) when text.IsNormalized(form) would be true.
    // =====================================================================

    [Fact]
    public void NormalizingTokenizer_AlreadyNormalizedAscii_SkipsNormalizeCall()
    {
        var inner = new RecordingTokenizer();
        var decorated = new NormalizingTokenizer(inner, NormalizationForm.FormC);

        // ASCII is always NFC-normalized — the decorator must pass the
        // original string reference through untouched (no Normalize allocation).
        const string Ascii = "Hello, world!";
        _ = decorated.Encode(Ascii);

        Assert.Single(inner.ReceivedInputs);
        Assert.Same(Ascii, inner.ReceivedInputs[0]);
    }

    [Fact]
    public void NormalizingTokenizer_EmptyString_SkipsNormalizeCall()
    {
        var inner = new RecordingTokenizer();
        var decorated = new NormalizingTokenizer(inner, NormalizationForm.FormC);

        string empty = string.Empty;
        _ = decorated.Encode(empty);

        Assert.Single(inner.ReceivedInputs);
        Assert.Same(empty, inner.ReceivedInputs[0]);
    }

    [Fact]
    public void NormalizingTokenizer_NonNormalizedInput_StillNormalizes()
    {
        var inner = new RecordingTokenizer();
        var decorated = new NormalizingTokenizer(inner, NormalizationForm.FormC);

        // "e" + combining acute accent (U+0301) is NFD, NOT NFC. The
        // decorator must call Normalize and forward the NFC result.
        string nfd = "é";
        Assert.False(nfd.IsNormalized(NormalizationForm.FormC));

        _ = decorated.Encode(nfd);

        Assert.Single(inner.ReceivedInputs);
        Assert.NotSame(nfd, inner.ReceivedInputs[0]);
        Assert.Equal(nfd.Normalize(NormalizationForm.FormC), inner.ReceivedInputs[0]);
    }

    /// <summary>
    /// Test-only <see cref="ITokenizer"/> that records each input string it
    /// receives via <see cref="Encode"/> so the surrounding test can assert
    /// reference-identity (i.e. whether the decorator skipped the Normalize
    /// allocation).
    /// </summary>
    private sealed class RecordingTokenizer : ITokenizer
    {
        public List<string> ReceivedInputs { get; } = new();

        public int VocabSize => 0;
        public int BosTokenId => -1;
        public int EosTokenId => -1;

        public int[] Encode(string text)
        {
            ReceivedInputs.Add(text);
            return [];
        }

        public string Decode(ReadOnlySpan<int> tokenIds) => string.Empty;
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => string.Empty;
        public string DecodeToken(int tokenId) => string.Empty;
        public int CountTokens(string text) => 0;
    }

    // =====================================================================
    // Helpers.
    // =====================================================================

    private static void AssertEncodeMatches(ITokenizer tok, (string Input, int[] Expected)[] samples)
    {
        foreach ((string input, int[] expected) in samples)
        {
            int[] actual = tok.Encode(input);
            Assert.True(
                expected.AsSpan().SequenceEqual(actual),
                $"Encode mismatch for \"{input}\": expected [{string.Join(",", expected)}], got [{string.Join(",", actual)}]");
        }
    }

    private static void AssertDecodeRoundTrips(ITokenizer tok, (string Input, int[] Expected)[] samples)
    {
        foreach ((string input, int[] expected) in samples)
        {
            if (expected.Length == 0)
                continue; // empty → nothing to decode
            string decoded = tok.Decode(expected);
            Assert.Equal(input, decoded);
        }
    }

    private static HfTokenizerSpec LoadSpec(string dir)
    {
        string path = Path.Combine(dir, "tokenizer.json");
        return HfTokenizerJsonParser.Parse(File.ReadAllText(path));
    }

    private static void SkipIfMissing(string dir)
    {
        Skip.IfNot(
            File.Exists(Path.Combine(dir, "tokenizer.json")),
            $"tokenizer.json not present at {dir} — skip.");
    }
}
