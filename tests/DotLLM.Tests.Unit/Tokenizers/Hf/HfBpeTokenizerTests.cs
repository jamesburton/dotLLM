using System.Text;
using System.Text.Json;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Tokenizers.Hf;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.Hf;

/// <summary>
/// Unit tests for the HuggingFace <c>tokenizer.json</c> adapter. Synthetic
/// vocabularies only — the real-checkpoint round-trip lives in the integration
/// suite.
/// </summary>
public class HfBpeTokenizerTests
{
    // Llama-2-shaped synthetic tokenizer.json. Vocab layout:
    //   <unk>=0, <s>=1, </s>=2, <0x00>..<0xFF>=3..258,
    //   ▁=259, H=260, e=261, l=262, o=263, w=264, r=265, d=266, space=267,
    //   ▁H=268, ▁He=269, ▁Hell=270, ▁Hello=271,
    //   ▁w=272, ▁wo=273, ▁wor=274, ▁worl=275, ▁world=276, ll=277.
    private static readonly string MinimalTokenizerJson = BuildMinimalTokenizerJson();

    private static string BuildMinimalTokenizerJson()
    {
        var sb = new StringBuilder();
        sb.Append("""
        {
          "version": "1.0",
          "added_tokens": [
            {"id":0, "content":"<unk>", "special":true},
            {"id":1, "content":"<s>",   "special":true},
            {"id":2, "content":"</s>",  "special":true}
          ],
          "normalizer": null,
          "pre_tokenizer": {"type":"Metaspace","replacement":"\u2581","prepend_scheme":"first","split":false},
          "decoder": {"type":"Sequence","decoders":[
            {"type":"Replace","pattern":{"String":"\u2581"},"content":" "},
            {"type":"ByteFallback"},{"type":"Fuse"},
            {"type":"Strip","content":" ","start":1,"stop":0}]},
          "model": {
            "type":"BPE",
            "byte_fallback": true,
            "vocab": {
              "<unk>":0, "<s>":1, "</s>":2,
        """);
        // Full 256 byte tokens at IDs 3..258.
        for (int b = 0; b < 256; b++)
            sb.Append($"\"<0x{b:X2}>\":{3 + b},");
        sb.Append(""""
              "\u2581":259,
              "H":260, "e":261, "l":262, "o":263,
              "w":264, "r":265, "d":266, " ":267,
              "\u2581H":268, "\u2581He":269, "\u2581Hell":270, "\u2581Hello":271,
              "\u2581w":272, "\u2581wo":273, "\u2581wor":274, "\u2581worl":275, "\u2581world":276,
              "ll":277
            },
            "merges": [
              ["\u2581","H"],
              ["\u2581H","e"],
              ["l","l"],
              ["\u2581He","ll"],
              ["\u2581Hell","o"],
              ["\u2581","w"],
              ["\u2581w","o"],
              ["\u2581wo","r"],
              ["\u2581wor","l"],
              ["\u2581worl","d"]
            ]
          }
        }
        """");
        return sb.ToString();
    }

    [Fact]
    public void ParseVocab_ExtractsAllEntries()
    {
        using JsonDocument doc = JsonDocument.Parse(MinimalTokenizerJson);
        var vocab = HfTokenizerJsonParser.ParseVocab(doc.RootElement);

        Assert.Equal(0, vocab["<unk>"]);
        Assert.Equal(1, vocab["<s>"]);
        Assert.Equal(260, vocab["H"]);
        Assert.Equal(271, vocab["\u2581Hello"]);
    }

    [Fact]
    public void ParseMerges_ArrayFormat_Works()
    {
        using JsonDocument doc = JsonDocument.Parse(MinimalTokenizerJson);
        var merges = HfTokenizerJsonParser.ParseMerges(doc.RootElement);

        Assert.Equal(10, merges.Count);
        Assert.Equal(("\u2581", "H"), merges[0]);
        Assert.Equal(("\u2581Hell", "o"), merges[4]);
    }

    [Fact]
    public void ParseMerges_StringFormat_Works()
    {
        // Legacy tokenizers (< 0.15) wrote merges as space-separated strings.
        const string legacy = """
        {
          "model": {
            "type": "BPE",
            "vocab": {"a":0,"b":1,"ab":2,"c":3,"bc":4},
            "merges": ["a b", "b c"]
          }
        }
        """;
        using JsonDocument doc = JsonDocument.Parse(legacy);
        var merges = HfTokenizerJsonParser.ParseMerges(doc.RootElement);

        Assert.Equal(2, merges.Count);
        Assert.Equal(("a", "b"), merges[0]);
        Assert.Equal(("b", "c"), merges[1]);
    }

    [Fact]
    public void ParseAddedTokens_CapturesSpecialFlag()
    {
        using JsonDocument doc = JsonDocument.Parse(MinimalTokenizerJson);
        var added = HfTokenizerJsonParser.ParseAddedTokens(doc.RootElement);

        Assert.Equal(3, added.Count);
        Assert.Contains(added, a => a.Id == 1 && a.Content == "<s>" && a.Special);
    }

    [Fact]
    public void Parse_CaptureMetaspaceConfig()
    {
        HfTokenizerSpec spec = HfTokenizerJsonParser.Parse(MinimalTokenizerJson);

        Assert.Equal(HfPreTokenizerKind.Metaspace, spec.PreTokenizerKind);
        Assert.Equal("\u2581", spec.MetaspaceReplacement);
        Assert.Equal("first", spec.MetaspacePrependScheme);
        Assert.True(spec.ByteFallback);
        Assert.Contains(HfDecoderStage.ByteFallback, spec.DecoderStages);
        Assert.Contains(HfDecoderStage.Replace, spec.DecoderStages);
    }

    [Fact]
    public void EncodeDecode_RoundTrip_Ascii()
    {
        BpeTokenizer tok = HfBpeTokenizerFactory.Create(MinimalTokenizerJson);

        int[] ids = tok.Encode("Hello world");
        Assert.NotEmpty(ids);
        foreach (int id in ids) Assert.InRange(id, 0, tok.VocabSize - 1);

        string decoded = tok.Decode(ids);
        Assert.Equal("Hello world", decoded);

        // Merged tokens should dominate: we expect exactly [▁Hello, ▁world] (IDs 271, 276).
        Assert.Equal(new[] { 271, 276 }, ids);
    }

    [Fact]
    public void EncodeDecode_RoundTrip_NonAscii_UsesByteFallback()
    {
        // Non-ASCII string with chars outside the tiny vocab — must hit the byte-fallback path.
        BpeTokenizer tok = HfBpeTokenizerFactory.Create(MinimalTokenizerJson);

        const string input = "Hello \u00e9"; // é = 0xC3 0xA9 in UTF-8
        int[] ids = tok.Encode(input);
        Assert.NotEmpty(ids);

        // Expect the byte tokens <0xC3> and <0xA9> somewhere in the sequence.
        // They are vocab IDs 0xC3+3 = 198 and 0xA9+3 = 172.
        Assert.Contains(198, ids);
        Assert.Contains(172, ids);

        string decoded = tok.Decode(ids);
        Assert.Equal(input, decoded);
    }

    [Fact]
    public void SpecialTokens_AreLookupNotSplit()
    {
        BpeTokenizer tok = HfBpeTokenizerFactory.Create(MinimalTokenizerJson);

        int[] ids = tok.Encode("<s>");
        // <s> is a special token (length > 1) → emitted directly as ID 1.
        Assert.Equal(new[] { 1 }, ids);

        int[] withText = tok.Encode("<s>Hello");
        Assert.Equal(1, withText[0]);
        // The continuation "Hello" is a post-special segment: no ▁ prepend
        // (prepend_scheme = "first"), so it tokenises via raw chars + merges.
        // With only the "l l → ll" merge firing, we expect H, e, ll, o.
        Assert.Equal(new[] { 1, 260, 261, 277, 263 }, withText);
    }

    [Fact]
    public void BosEos_AutoDetected_FromAddedTokens()
    {
        BpeTokenizer tok = HfBpeTokenizerFactory.Create(MinimalTokenizerJson);
        Assert.Equal(1, tok.BosTokenId);
        Assert.Equal(2, tok.EosTokenId);
    }

    [Fact]
    public void Create_UnsupportedPreTokenizer_Throws()
    {
        const string byteLevel = """
        {
          "pre_tokenizer": {"type":"ByteLevel"},
          "model": {"type":"BPE","vocab": {"a":0},"merges":[]}
        }
        """;
        Assert.Throws<InvalidDataException>(() =>
            HfBpeTokenizerFactory.Create(byteLevel));
    }
}
