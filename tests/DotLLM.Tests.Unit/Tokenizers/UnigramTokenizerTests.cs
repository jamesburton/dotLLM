using System.Text;
using DotLLM.Tokenizers.Unigram;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers;

public sealed class UnigramTokenizerTests
{
    // ids: 0 <s>, 1 </s>, 2 <unk>, 3.. normal pieces ("▁" is the SentencePiece word boundary).
    private static readonly (string Text, float Score, int Type)[] Pieces =
    [
        ("<s>", 0, 3), ("</s>", 0, 3), ("<unk>", 0, 2),
        ("▁hello", -3f, 1),   // 3
        ("▁hell", -4f, 1),    // 4
        ("o", -2f, 1),             // 5
        ("▁", -5f, 1),        // 6
        ("▁world", -3.5f, 1), // 7  longest match, but...
        ("▁w", -1.5f, 1),     // 8  ...▁w + orld = -2.5 beats it
        ("orld", -1f, 1),          // 9
        ("▁fish", -3f, 1),    // 10
        ("▁ab", -3f, 1),      // 11
        ("▁" + "1", -3f, 1),  // 12
        ("[PAD0]", -10000f, 5),    // 13 unused: never matchable, must not poison the unknown score
    ];

    private static UnigramTokenizer Make(byte[]? charsMap = null, bool addSpacePrefix = true, bool removeExtra = true)
        => new(Pieces.Select(p => p.Text).ToArray(), Pieces.Select(p => p.Score).ToArray(), Pieces.Select(p => p.Type).ToArray(),
               unkId: 2, bosId: 0, eosId: 1, padId: -1, addSpacePrefix: addSpacePrefix, removeExtraWhitespaces: removeExtra,
               precompiledCharsMap: charsMap ?? []);

    [Fact]
    public void Encode_frames_with_bos_and_eos_and_prefixes_the_word_boundary()
        => Assert.Equal([0, 3, 1], Make().Encode("hello"));

    [Fact]
    public void EncodeRaw_has_no_framing()
        => Assert.Equal([3], Make().EncodeRaw("hello"));

    [Fact]
    public void Viterbi_beats_longest_match()
    {
        // "▁world" is one piece (-3.5) but "▁w"+"orld" scores -2.5: a greedy longest-match tokenizer would get this wrong.
        var ids = Make().Encode("hello world");
        Assert.Equal([0, 3, 8, 9, 1], ids);
        Assert.NotEqual([0, 3, 7, 1], ids);
    }

    [Fact]
    public void Whitespace_runs_collapse_and_ends_are_trimmed()
    {
        var t = Make();
        Assert.Equal(t.Encode("hello world"), t.Encode("  hello \t\n world   ".Replace('\t', ' ').Replace('\n', ' ')));
        Assert.Equal([0, 1], t.Encode("   "));
        Assert.Equal([0, 1], t.Encode(""));
    }

    [Fact]
    public void Consecutive_unknown_characters_fuse_into_one_unk()
        // "▁" then two unknown emoji -> a single <unk>.
        => Assert.Equal([0, 6, 2, 1], Make().Encode("\U0001F600\U0001F600"));

    [Fact]
    public void Unknown_character_between_known_pieces_keeps_both_sides()
        => Assert.Equal([0, 3, 6, 2, 1], Make().Encode("hello \U0001F600"));

    [Fact]
    public void Unused_pieces_are_not_matchable()
        // The literal text of an UNUSED piece must not segment as that piece (id 13).
        => Assert.DoesNotContain(13, Make().Encode("[PAD0]"));

    [Fact]
    public void Literal_control_tokens_in_the_text_become_their_ids()
        => Assert.Equal([0, 3, 1, 3, 1], Make().Encode("hello</s>hello"));

    [Fact]
    public void Decode_restores_spaces()
        => Assert.Equal("hello world", Make().Decode([0, 3, 7, 1]));

    [Fact]
    public void Charsmap_longest_rule_rewrites_input_before_segmentation()
    {
        var map = CharsMapBuilder.Build(
            ("ﬁ", "fi"),      // ligature -> "fi"   (NFKC)
            ("①", "1"),       // circled digit one -> "1"
            ("­", ""));       // soft hyphen -> deleted
        var t = Make(map);

        Assert.Equal([0, 10, 1], t.Encode("ﬁsh"));         // "ﬁsh" -> "fish"
        Assert.Equal([0, 12, 1], t.Encode("①"));           // "①"   -> "1"
        Assert.Equal([0, 11, 1], t.Encode("a­b"));         // "a<SHY>b" -> "ab"
        // Without the table the same input is NOT rewritten (discriminates the charsmap from .NET NFKC).
        Assert.NotEqual([0, 10, 1], Make().Encode("ﬁsh"));
    }

    [Fact]
    public void Charsmap_blob_is_validated()
    {
        Assert.Throws<InvalidDataException>(() => Make([1, 2]));
        Assert.Throws<InvalidDataException>(() => Make([0x10, 0, 0, 0, 1, 2]));   // declares 16 trie bytes, has 2
    }

    /// <summary>Builds a minimal darts-clone double array for single-codepoint rules with distinct lead bytes.</summary>
    private static class CharsMapBuilder
    {
        public static byte[] Build(params (string Key, string Replacement)[] rules)
        {
            var units = new uint[0x20000];
            var used = new HashSet<uint> { 0 };
            const uint rootBase = 0x400;
            units[0] = rootBase << 10;

            var pool = new List<byte> { 0 };   // offset 0 = "" (deletion)
            for (int r = 0; r < rules.Length; r++)
            {
                byte[] key = Encoding.UTF8.GetBytes(rules[r].Key);
                int poolOffset = 0;
                if (rules[r].Replacement.Length > 0)
                {
                    poolOffset = pool.Count;
                    pool.AddRange(Encoding.UTF8.GetBytes(rules[r].Replacement));
                    pool.Add(0);
                }

                uint cur = rootBase;
                for (int i = 0; i < key.Length; i++)
                {
                    uint idx = cur ^ key[i];
                    Assert.True(used.Add(idx), "test charsmap slot collision");
                    uint b = 1u << (8 + r * 3 + i);
                    bool last = i == key.Length - 1;
                    units[idx] = (b << 10) | key[i] | (last ? 1u << 8 : 0);
                    cur = idx ^ b;
                }
                Assert.True(used.Add(cur), "test charsmap leaf collision");
                units[cur] = (uint)poolOffset;   // leaf unit: value = pool offset
            }

            int count = (int)used.Max() + 1;
            var blob = new List<byte>();
            blob.AddRange(BitConverter.GetBytes((uint)(count * 4)));
            for (int i = 0; i < count; i++) blob.AddRange(BitConverter.GetBytes(units[i]));
            blob.AddRange(pool);
            return blob.ToArray();
        }
    }
}
