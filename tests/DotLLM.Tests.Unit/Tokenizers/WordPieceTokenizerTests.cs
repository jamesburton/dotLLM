using DotLLM.Tokenizers.WordPiece;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers;

public sealed class WordPieceTokenizerTests
{
    // Canonical (HF-style) vocab: ids are positions.
    private static readonly string[] Vocab =
    [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",     // 0-4
        "hello", "world", "un", "##aff", "##able",         // 5-9
        ",", "!", "cafe", "play", "##ing",                 // 10-14
        "北", "京", "a", "$", "[", "]",            // 15-20
    ];

    private static WordPieceTokenizer Make(bool lower = true, bool? strip = null)
        => new(Vocab, clsId: 2, sepId: 3, unkId: 1, padId: 0, lowerCase: lower, stripAccents: strip);

    [Fact]
    public void Encode_wraps_with_cls_and_sep()
        => Assert.Equal([2, 5, 10, 6, 11, 3], Make().Encode("Hello, WORLD!"));

    [Fact]
    public void Greedy_longest_match_uses_continuation_pieces()
        => Assert.Equal([2, 7, 8, 9, 3], Make().Encode("unaffable"));

    [Fact]
    public void Unsegmentable_word_becomes_a_single_unk()
        => Assert.Equal([2, 1, 3], Make().Encode("unaffablex"));

    [Fact]
    public void Accents_are_stripped_when_lowercasing()
        => Assert.Equal([2, 12, 3], Make().Encode("Café"));

    [Fact]
    public void Accents_survive_when_stripping_is_disabled_and_become_unk()
        => Assert.Equal([2, 1, 3], Make(lower: true, strip: false).Encode("Café"));

    [Fact]
    public void Cjk_ideographs_are_split_into_single_character_words()
        => Assert.Equal([2, 15, 16, 3], Make().Encode("北京"));

    [Fact]
    public void Ascii_symbols_count_as_punctuation()
        => Assert.Equal([2, 17, 18, 17, 3], Make().Encode("a$a"));

    [Fact]
    public void Literal_special_tokens_in_text_map_to_their_ids()
        => Assert.Equal([2, 5, 4, 6, 3], Make().Encode("hello [MASK] world"));

    [Fact]
    public void Control_and_replacement_characters_are_dropped_and_whitespace_normalised()
        => Assert.Equal([2, 5, 6, 3], Make().Encode("hello\u0000�\t\n  world"));

    [Fact]
    public void Overlong_word_is_unk()
    {
        var tok = new WordPieceTokenizer(Vocab, 2, 3, 1, maxCharsPerWord: 5);
        Assert.Equal([2, 1, 3], tok.Encode("aaaaaa"));
        Assert.Equal([2, 17, 3], tok.Encode("a"));
    }

    [Fact]
    public void Empty_input_is_just_cls_sep()
        => Assert.Equal([2, 3], Make().Encode(""));

    [Fact]
    public void Encode_without_special_tokens_omits_the_wrapper()
        => Assert.Equal([5, 6], Make().Encode("hello world", addSpecialTokens: false));

    [Fact]
    public void Decode_joins_continuations()
        => Assert.Equal("hello unaffable", Make().Decode([5, 7, 8, 9]));

    [Fact]
    public void Gguf_vocab_with_prefix_marker_converts_to_the_canonical_form()
    {
        // llama.cpp WPM form: word-initial = "▁x", continuation = bare, bracketed specials bare.
        string[] gguf =
        [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "▁hello", "▁un", "aff", "able", "▁,",
        ];
        var tok = WordPieceTokenizer.FromGgufTokens(gguf, clsId: 2, sepId: 3, unkId: 1);
        Assert.Equal([2, 4, 8, 5, 6, 7, 3], tok.Encode("hello, unaffable"));
        Assert.Equal(2, tok.BosTokenId);
        Assert.Equal(3, tok.EosTokenId);
    }

    [Fact]
    public void Gguf_special_ids_resolve_by_string_when_keys_are_absent()
    {
        string[] gguf = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "▁a"];
        var tok = WordPieceTokenizer.FromGgufTokens(gguf);
        Assert.Equal(2, tok.ClsTokenId);
        Assert.Equal(3, tok.SepTokenId);
        Assert.Equal(1, tok.UnkTokenId);
    }

    [Fact]
    public void Hf_tokenizer_json_round_trips()
    {
        const string json = """
        {
          "added_tokens": [ {"id":0,"content":"[PAD]","special":true}, {"id":1,"content":"[UNK]","special":true},
                            {"id":2,"content":"[CLS]","special":true}, {"id":3,"content":"[SEP]","special":true} ],
          "normalizer": {"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},
          "model": {"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,
                    "vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"play":4,"##ing":5,"cafe":6}}
        }
        """;
        Assert.True(HfWordPieceLoader.IsWordPiece(json));
        var tok = HfWordPieceLoader.Parse(json);
        Assert.Equal([2, 4, 5, 6, 3], tok.Encode("Playing Café"));
    }

    [Fact]
    public void Hf_loader_rejects_non_wordpiece()
        => Assert.Throws<InvalidDataException>(() => HfWordPieceLoader.Parse("{\"model\":{\"type\":\"BPE\"}}"));
}
