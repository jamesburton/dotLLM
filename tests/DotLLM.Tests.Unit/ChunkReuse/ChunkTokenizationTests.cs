using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Phase I Task 5b (contract ruling 21): chunk text is tokenised as <c>EncodeRaw(MarkUntrustedSpecials(text))</c>, the way the
/// chat path tokenises caller content (#1339: the template marks special-token spellings with U+FDD0, the encoder strips the
/// mark and encodes the spelling as plain text). Bare <see cref="ITokenizer.EncodeRaw"/> parses a spelling such as
/// <c>&lt;think&gt;</c> into its control id, so a chunk containing one would get ids the server never computes.
/// </summary>
public sealed class ChunkTokenizationTests
{
    // ------------------------------------------------------------------------------------------------ synthetic vocabulary

    private static readonly string[] Specials = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>"];

    private static readonly Lazy<BpeTokenizer> Vocab = new(() =>
    {
        // One piece per printable ASCII character and '\n' (no merges), plus the specials as control tokens (type 3): every
        // plain character is one id, every parsed special is one id, so counts tell the two encodings apart exactly.
        var tokens = new List<string> { "<unk>" };
        var types = new List<int> { 2 };
        for (char c = ' '; c <= '~'; c++)
        {
            tokens.Add(c == ' ' ? "▁" : c.ToString());
            types.Add(1);
        }
        tokens.Add("\n");
        types.Add(1);
        tokens.Add("\r");
        types.Add(1);
        foreach (string s in Specials)
        {
            tokens.Add(s);
            types.Add(3);
        }
        return BpeTokenizer.CreateSentencePiece(tokens.ToArray(), new float[tokens.Count], types.ToArray(),
            bosId: tokens.IndexOf("<|im_start|>"), eosId: tokens.IndexOf("<|im_end|>"), addBosSpace: false);
    });

    private static BpeTokenizer Tok => Vocab.Value;

    private static int SpecialId(string s) => Assert.Single(Tok.EncodeRaw(s));

    // A Qwen-style ChatML template; the tokenizer is the #1339 guard, as the server wires it.
    private const string ChatMl =
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        + "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    private const string ThinkChunk = "    // Models emit <think>...</think> around a scratchpad.\n    int x = 1;\n";

    private static int IndexOf(int[] haystack, int[] needle)
    {
        for (int i = 0; i + needle.Length <= haystack.Length; i++)
            if (haystack.AsSpan(i, needle.Length).SequenceEqual(needle)) return i;
        return -1;
    }

    private static int[] ChatPathIds(ITokenizer tok, IChatTemplate template, string content)
        => tok.Encode(template.Apply([new ChatMessage { Role = "user", Content = content }], new ChatTemplateOptions()));

    // ------------------------------------------------------------------------------------------------ the encoding itself

    [Fact]
    public void ChunkWithThinkTag_StoreAndChatPathIdsEqual()
    {
        var template = new JinjaChatTemplate(ChatMl, "<|im_start|>", "<|im_end|>", Tok);
        int[] prompt = ChatPathIds(Tok, template, ThinkChunk);
        int[] chunk = ChunkTokenization.Encode(Tok, ThinkChunk);

        Assert.True(IndexOf(prompt, chunk) >= 0, "the chunk's ids appear verbatim in the chat path's prompt ids");
        Assert.DoesNotContain(SpecialId("<think>"), chunk);
        Assert.Equal(ThinkChunk.Length, chunk.Length); // one id per character: the spellings are plain text
        // Substitution: the old encoding parses the spellings, and those ids are not in the prompt.
        int[] raw = Tok.EncodeRaw(ThinkChunk);
        Assert.Contains(SpecialId("<think>"), raw);
        Assert.Equal(-1, IndexOf(prompt, raw));
    }

    [Fact]
    public void PlainChunk_IdsUnchanged()
    {
        const string plain = "public int Add(int a, int b)\n{\n    return a + b; // a < b, x > y, <tag>\n}\n";
        Assert.Equal(Tok.EncodeRaw(plain), ChunkTokenization.Encode(Tok, plain));
        ITokenizer chars = new CharTokenizer();
        Assert.Equal(chars.EncodeRaw(plain), ChunkTokenization.Encode(chars, plain));
        Assert.Empty(ChunkTokenization.Encode(Tok, ""));
    }

    [Fact]
    public void PreMarkedOrCallerMarkerText_EncodesAsTheChatPathDoes()
    {
        // Marking is idempotent once the spellings are broken, and a caller's own U+FDD0 is dropped at encode (#1339, F10).
        string marked = Tok.MarkUntrustedSpecials(ThinkChunk);
        Assert.NotEqual(ThinkChunk, marked);
        Assert.Equal(ChunkTokenization.Encode(Tok, ThinkChunk), ChunkTokenization.Encode(Tok, marked));
        Assert.Equal(ChunkTokenization.Encode(Tok, "ab"), ChunkTokenization.Encode(Tok, "a﷐b"));
    }

    [Fact]
    public void Encode_NeverAddsBos_EvenWhenTheTokenizerFramesEncode()
    {
        // The chunk is mid-prompt text: no BOS, whatever Encode does on its own (kills "Encode instead of EncodeRaw").
        var tok = new BosFramingTokenizer(Tok);
        Assert.Equal(tok.BosTokenId, tok.Encode("int x;")[0]);
        Assert.Equal(Tok.EncodeRaw(Tok.MarkUntrustedSpecials(ThinkChunk)), ChunkTokenization.Encode(tok, ThinkChunk));
        Assert.DoesNotContain(tok.BosTokenId, ChunkTokenization.Encode(tok, "int x;"));
    }

    /// <summary>Delegates to <see cref="Tok"/>, but <see cref="Encode"/> prepends BOS (as a Llama-style tokenizer does).</summary>
    private sealed class BosFramingTokenizer(BpeTokenizer inner) : ITokenizer
    {
        public int[] Encode(string text) => [BosTokenId, .. inner.EncodeRaw(text)];
        public int[] EncodeRaw(string text) => inner.EncodeRaw(text);
        public string MarkUntrustedSpecials(string text, IReadOnlyList<string>? exemptPrefixes = null) => inner.MarkUntrustedSpecials(text, exemptPrefixes);
        public string Decode(ReadOnlySpan<int> tokenIds) => inner.Decode(tokenIds);
        public string DecodeToken(int tokenId) => inner.DecodeToken(tokenId);
        public int VocabSize => inner.VocabSize;
        public int BosTokenId => 0;
        public int EosTokenId => inner.EosTokenId;
        public int CountTokens(string text) => Encode(text).Length;
    }

    // ------------------------------------------------------------------------------------------------ every chunker uses it

    private const string CodeWithThink = """
        namespace Demo;

        public static class Reasoning
        {
            // Reasoning models emit <think>...</think> around their scratchpad.
            public static string Strip(string s) => s.Replace("<think>", "").Replace("</think>", "");

            public static int Two() => 2;
        }

        """;

    private static void AssertChunksUseChunkTokenization(SourceFile file, IReadOnlyList<Chunk> chunks)
    {
        int withSpelling = 0;
        foreach (Chunk c in chunks)
        {
            string text = file.Text[c.StartOffset..c.EndOffset];
            Assert.Equal(ChunkTokenization.Encode(Tok, text), c.TokenIds);
            if (text.Contains("<think>", StringComparison.Ordinal))
            {
                withSpelling++;
                Assert.NotEqual(Tok.EncodeRaw(text), c.TokenIds);
            }
        }
        Assert.True(withSpelling > 0, "the fixture puts a special spelling inside a chunk");
    }

    [Fact]
    public void SymbolChunker_UsesChunkTokenization()
    {
        var file = new SourceFile("Reasoning.cs", CodeWithThink.ReplaceLineEndings("\n"));
        AssertChunksUseChunkTokenization(file, new SymbolChunker(Tok, 512).Chunk(file));
    }

    [Fact]
    public void SymbolChunker_PackedPieces_UseChunkTokenization()
    {
        // A member over the limit whose lines all fit: re-split by Pack at statements / line starts (not sliced).
        const string code = """
            namespace Demo;

            public static class R
            {
                public static int M()
                {
                    // <think> a
                    // </think> b
                    int x = 1;
                    // <think> c
                    return x;
                }
            }

            """;
        var file = new SourceFile("R.cs", code.ReplaceLineEndings("\n"));
        Assert.All(file.Text.Split('\n'), l => Assert.True(ChunkTokenization.Encode(Tok, l + "\n").Length < 30));
        IReadOnlyList<Chunk> chunks = new SymbolChunker(Tok, 30).Chunk(file);
        Assert.True(chunks.Count >= 3);
        Assert.All(chunks, c => Assert.True(c.WireEligible, "no slices here"));
        AssertChunksUseChunkTokenization(file, chunks);
    }

    [Fact]
    public void SymbolChunker_OverlongLineSlices_UseChunkTokenization()
    {
        // A line over the limit is cut into id slices of the whole line's ids: those ids are the marked encoding too.
        var file = new SourceFile("Reasoning.cs", CodeWithThink.ReplaceLineEndings("\n"));
        IReadOnlyList<Chunk> chunks = new SymbolChunker(Tok, 40).Chunk(file);
        string[] lines = file.Text.Split('\n');
        int lineNo = 1 + Array.FindIndex(lines, l => l.Contains("Strip", StringComparison.Ordinal));
        string line = lines[lineNo - 1] + "\n";
        List<Chunk> pieces = chunks.Where(c => c.StartLine == lineNo && c.EndLine == lineNo).ToList();
        Assert.True(pieces.Count >= 2);
        Assert.All(pieces, c => Assert.False(c.WireEligible));
        Assert.Equal(ChunkTokenization.Encode(Tok, line), pieces.SelectMany(c => c.TokenIds));
    }

    [Fact]
    public void LinesChunker_UsesChunkTokenization()
    {
        var file = new SourceFile("Reasoning.cs", CodeWithThink.ReplaceLineEndings("\n"));
        IReadOnlyList<Chunk> chunks = new LinesChunker(Tok, 64).Chunk(file);
        Assert.True(chunks.Count >= 3);
        AssertChunksUseChunkTokenization(file, chunks);
    }

    [Fact]
    public void LinesChunker_CrlfBackOff_UsesChunkTokenization()
    {
        // The furthest fitting cut falls between '\r' and '\n' with no line end inside the piece: the chunker backs off one
        // character and re-probes; that probe's ids are the chunk's ids.
        const int n = 20;
        string text = "<think>" + new string('y', n - 8) + "\r\n" + "tail line here\r\n";
        Assert.Equal(n, ChunkTokenization.Encode(Tok, text[..n]).Length);   // up to and including '\r'
        var file = new SourceFile("crlf.txt", text);
        IReadOnlyList<Chunk> chunks = new LinesChunker(Tok, n).Chunk(file);
        Assert.Equal(n - 1, chunks[0].EndOffset);                           // not between '\r' and '\n'
        foreach (Chunk c in chunks)
            Assert.Equal(ChunkTokenization.Encode(Tok, text[c.StartOffset..c.EndOffset]), c.TokenIds);
    }

    [Fact]
    public void FixedSizeChunker_UsesChunkTokenization()
    {
        var file = new SourceFile("Reasoning.cs", CodeWithThink.ReplaceLineEndings("\n"));
        IReadOnlyList<Chunk> chunks = new FixedSizeChunker(Tok, 64).Chunk(file);
        Assert.Equal(ChunkTokenization.Encode(Tok, file.Text), chunks.SelectMany(c => c.TokenIds));
        Assert.DoesNotContain(SpecialId("<think>"), chunks.SelectMany(c => c.TokenIds));
        // Offsets still tile the text exactly (one id per character here).
        Assert.Equal(file.Text, string.Concat(chunks.Select(c => file.Text[c.StartOffset..c.EndOffset])));
    }

    [Fact]
    public void LinesChunker_SpecialSpellingNearLimit_StaysWithinN()
    {
        // Each "<think>" is 1 id when parsed but 7 marked. With N = 40 a line of 20 spellings is 20 parsed ids (fits) but 140
        // marked ids: the size probes must use the marked encoding, or a piece over N is emitted.
        string text = "x\n" + string.Concat(Enumerable.Repeat("<think>", 20)) + "\nend\n";
        var file = new SourceFile("t.txt", text);
        const int n = 40;
        IReadOnlyList<Chunk> chunks = new LinesChunker(Tok, n).Chunk(file);
        Assert.All(chunks, c => Assert.InRange(c.TokenIds.Length, 1, n));
        Assert.Equal(text, string.Concat(chunks.Select(c => text[c.StartOffset..c.EndOffset])));
        Assert.Equal(0, WireChunking.VerifyRoundTrip(chunks, c => text[c.StartOffset..c.EndOffset], Tok).Failed);
    }

    // ------------------------------------------------------------------------------------------------ round trip

    [Fact]
    public void VerifyRoundTrip_SpecialSpellingChunk_Passes()
    {
        var file = new SourceFile("Reasoning.cs", CodeWithThink.ReplaceLineEndings("\n"));
        IReadOnlyList<Chunk> chunks = new LinesChunker(Tok, 64).Chunk(file);
        string TextOf(Chunk c) => file.Text[c.StartOffset..c.EndOffset];
        RoundTripReport ok = WireChunking.VerifyRoundTrip(chunks, TextOf, Tok);
        Assert.Equal((chunks.Count, 0), (ok.Checked, ok.Failed));

        // A chunk carrying the old (parsed) ids is a failure: the server would compute different ids from its text.
        Chunk think = chunks.First(c => TextOf(c).Contains("<think>", StringComparison.Ordinal));
        Chunk stale = think with { TokenIds = Tok.EncodeRaw(TextOf(think)) };
        RoundTripReport bad = WireChunking.VerifyRoundTrip([stale], TextOf, Tok);
        Assert.Equal([stale.Id], bad.FailedIds);
    }

    // ------------------------------------------------------------------------------------------------ real tokenizer (CPU, metadata only)

    private static readonly string? QwenGguf = TestFixtureResolver.ResolveFile([], "bartowski", "Qwen_Qwen3-4B-Instruct-2507-GGUF",
        ["Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "Qwen_Qwen3-4B-Instruct-2507-IQ4_XS.gguf", "Qwen_Qwen3-4B-Instruct-2507-*.gguf"], null).Path;

    /// <summary>
    /// Golden on the real Qwen3-4B-Instruct-2507 tokenizer and GGUF chat template, built as the server builds them: the
    /// codesharp line that holds <c>&lt;think&gt;...&lt;/think&gt;</c> (CompactionSummaryValidation.cs:57 in the frozen corpus).
    /// </summary>
    [SkippableFact]
    public void RealQwen3_ChunkWithThinkTag_StoreAndChatPathIdsEqual()
    {
        Skip.If(QwenGguf is null, "Qwen3-4B-Instruct-2507 GGUF not in the local model stores / HF cache.");
        using var gguf = GgufFile.Open(QwenGguf!);
        BpeTokenizer tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        JinjaChatTemplate template = GgufChatTemplateFactory.TryCreate(gguf.Metadata, tok)!;
        const string chunkText =
            "    // Reasoning models emit <think>...</think> around their scratchpad. Anthropic's summarized-thinking\n"
            + "    // opt-in (ThinkingSummaryHandler) is Anthropic-only, so a reasoning model reached through any other\n";

        int[] prompt = ChatPathIds(tok, template, chunkText);
        int[] chunk = ChunkTokenization.Encode(tok, chunkText);
        Assert.True(IndexOf(prompt, chunk) >= 0, "the chunk's ids appear verbatim in the chat path's prompt ids");

        int think = Assert.Single(tok.EncodeRaw("<think>"));
        int[] raw = tok.EncodeRaw(chunkText);
        Assert.Contains(think, raw);
        Assert.DoesNotContain(think, chunk);
        Assert.Equal(-1, IndexOf(prompt, raw));

        // A plain line keeps its ids (existing store entries stay valid).
        const string plain = "    public const int TrivialExcerptChars = 2_000;\n";
        Assert.Equal(tok.EncodeRaw(plain), ChunkTokenization.Encode(tok, plain));
    }
}
