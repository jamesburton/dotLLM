using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers;

/// <summary>
/// Issue #373 — the <c>tokenizer.ggml.pre</c> policy must mirror llama.cpp's
/// <c>llama_vocab</c>: absent → GPT-2 default pipeline (llama.cpp's "missing
/// pre-tokenizer type, using: 'default'"), unknown → throw (llama.cpp:
/// <c>throw std::runtime_error("unknown pre-tokenizer type")</c>). Before this
/// fix an unknown value silently disabled pre-tokenization entirely — BPE merges
/// then crossed boundaries the model was trained to respect, diverging from
/// llama.cpp with no error and no warning.
/// </summary>
public class PreTokenizerPolicyTests
{
    private const string OptOutVar = "DOTLLM_ALLOW_UNKNOWN_PRETOKENIZER";

    /// <summary>
    /// 256 GPT-2 byte tokens plus a merged "34" token and the single merge
    /// "3 4". Pipeline-discriminating: the Llama-3 pipeline's <c>\p{N}{1,3}</c>
    /// splits "1234" into segments "123"|"4", so the 3+4 merge cannot apply;
    /// with NO pre-tokenization (the old unknown-pre behavior) "1234" is one
    /// segment and 3+4 merges. Token count 4 vs 3 discriminates the two.
    /// </summary>
    private static BpeTokenizer Build(string? preType) => Build(preType, "34", "3 4");

    /// <summary>
    /// Builds the same 256-byte GPT-2 vocabulary with one extra merged token, so a
    /// single merge either fires or is blocked purely by where the pre-tokenizer put
    /// its segment boundary.
    /// </summary>
    /// <param name="preType">GGUF <c>tokenizer.ggml.pre</c> value under test.</param>
    /// <param name="mergedToken">The extra vocabulary entry, in byte-level (GPT-2
    /// <c>bytes_to_unicode</c>) spelling.</param>
    /// <param name="merge">The single merge rule producing <paramref name="mergedToken"/>.</param>
    private static BpeTokenizer Build(string? preType, string mergedToken, string merge)
        => BpeTokenizer.CreateTiktoken(Vocab(mergedToken), merges: [merge], tokenTypes: null,
            bosId: 0, eosId: 0, preTokenizerType: preType);

    /// <summary>The 256 byte tokens plus one merged entry.</summary>
    private static string[] Vocab(string mergedToken)
    {
        string[] tokens = new string[257];
        for (int i = 0; i < 256; i++) tokens[i] = ByteToUnicode[i].ToString();
        tokens[256] = mergedToken;
        return tokens;
    }

    /// <summary>GPT-2 <c>bytes_to_unicode</c> table — printable bytes map to themselves.</summary>
    private static readonly char[] ByteToUnicode = BuildByteToUnicode();

    private static char[] BuildByteToUnicode()
    {
        char[] map = new char[256];
        for (int b = 33; b <= 126; b++) map[b] = (char)b;
        for (int b = 161; b <= 172; b++) map[b] = (char)b;
        for (int b = 174; b <= 255; b++) map[b] = (char)b;
        int n = 0;
        for (int b = 0; b < 256; b++)
            if (map[b] == 0) map[b] = (char)(0x100 + n++);
        return map;
    }

    [Theory]
    [InlineData("pixtral")]       // Nemotron-3.5-Lightning; llama.cpp → LLAMA_VOCAB_PRE_TYPE_LLAMA3
    [InlineData("falcon3")]
    [InlineData("falcon-h1")]
    [InlineData("llama-v3")]
    public void Llama3FamilyAliases_UseTheLlama3Pipeline(string preType)
    {
        int[] viaAlias = Build(preType).Encode("1234");
        int[] viaLlama3 = Build("llama3").Encode("1234");

        Assert.Equal(viaLlama3, viaAlias);
        // Discriminates against the pre-#373 behavior (unknown → NO pre-tokenization):
        // without the digit-group split the "3 4" merge applies and this is 3 tokens.
        Assert.Equal(4, viaAlias.Length);
    }

    /// <summary>
    /// Tekken (Mistral NeMo family; NVIDIA Nemotron-Nano-9B-v2 ships it) splits
    /// EVERY digit into its own segment (bare <c>\p{N}</c>), unlike gpt2
    /// (<c> ?\p{N}+</c> groups runs) and llama3 (<c>\p{N}{1,3}</c> groups up to
    /// three). With a "3 4" merge in the vocab, "34" therefore stays two tokens
    /// under tekken but merges to one under gpt2 — discriminating tekken both
    /// from "missing" and from either wrong-pipeline routing.
    /// </summary>
    [Fact]
    public void Tekken_SplitsEveryDigit_UnlikeGpt2AndLlama3()
    {
        int[] tekken = Build("tekken").Encode("34");
        Assert.Equal(2, tekken.Length);      // "3","4" — merge blocked by per-digit split
        Assert.Single(Build("gpt2").Encode("34"));   // digit run grouped → "34" merges
        Assert.Single(Build("llama3").Encode("34")); // {1,3} group → "34" merges
    }

    /// <summary>
    /// Issue #397 — <c>qwen2</c> is the value on <b>every</b> Qwen2/Qwen3 GGUF, and it
    /// differs from llama3 in exactly one place: the digit alternative is a bare
    /// <c>\p{N}</c>, not <c>\p{N}{1,3}</c>. With a "3 4" merge in the vocab, "34"
    /// therefore stays two tokens under qwen2 and collapses to one under both gpt2
    /// (<c> ?\p{N}+</c>) and llama3 — so this fails if qwen2 is routed to either.
    /// </summary>
    [Theory]
    [InlineData("qwen2")]
    [InlineData("qwen35")]
    [InlineData("deepseek-r1-qwen")] // llama.cpp: -> LLAMA_VOCAB_PRE_TYPE_QWEN2
    [InlineData("kormo")]
    [InlineData("f2llmv2")]
    [InlineData("megrez")]
    public void QwenPipelines_SplitEveryDigit_UnlikeGpt2AndLlama3(string preType)
    {
        Assert.Equal(2, Build(preType).Encode("34").Length);
        Assert.Single(Build("gpt2").Encode("34"));
        Assert.Single(Build("llama3").Encode("34"));
    }

    /// <summary>
    /// The discriminator between <c>qwen35</c> and <c>qwen2</c> (they share the bare
    /// <c>\p{N}</c>, so the digit test above cannot tell them apart): qwen35's letter
    /// run is <c>[\p{L}\p{M}]+</c>, so a decomposed "e" + U+0301 COMBINING ACUTE stays
    /// one segment, while qwen2/llama3/gpt2 all use <c>\p{L}+</c> and split the mark
    /// off into the punctuation alternative. The vocab carries the byte-level merge
    /// "e" + 0xCC (the first UTF-8 byte of U+0301), which can only fire inside a single
    /// segment — 2 tokens under qwen35, 3 under everything else.
    /// </summary>
    [Fact]
    public void Qwen35_KeepsCombiningMarksInTheLetterRun_UnlikeQwen2Gpt2AndLlama3()
    {
        // Built from code points rather than source literals so the test cannot be
        // defeated by an editor silently NFC-composing the file.
        string input = "e" + (char)0x0301;   // e + COMBINING ACUTE ACCENT (UTF-8: 65 CC 81)
        string merged = "e" + (char)0x00CC;   // bytes_to_unicode spelling of the bytes 65 CC
        string merge = "e " + (char)0x00CC;

        Assert.Equal(2, Build("qwen35", merged, merge).Encode(input).Length);
        Assert.Equal(3, Build("qwen2", merged, merge).Encode(input).Length);
        Assert.Equal(3, Build("llama3", merged, merge).Encode(input).Length);
        Assert.Equal(3, Build("gpt2", merged, merge).Encode(input).Length);
    }

    [Fact]
    public void UnknownPreType_Throws_NamingTheValue()
    {
        var ex = Assert.Throws<InvalidDataException>(() => Build("no-such-pre-tokenizer"));
        Assert.Contains("no-such-pre-tokenizer", ex.Message, StringComparison.Ordinal);
        Assert.Contains(OptOutVar, ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// Separates <c>qwen35</c> from <c>tekken</c> — the one pairing the real-GGUF parity suite
    /// cannot cover. Both split every digit and both fold combining marks into the letter run,
    /// so they differ only at a case boundary: qwen35's <c>[\p{L}\p{M}]+</c> runs straight through
    /// it, while tekken's <c>[\p{Lu}…]*[\p{Ll}…]+</c> / <c>[\p{Lu}…]+[\p{Ll}…]*</c> pair must end
    /// the segment before an upper-case letter that follows a lower-case one. With an "a B" merge
    /// in the vocab, "aB" is therefore 1 token under qwen35 and 2 under tekken.
    /// </summary>
    /// <remarks>
    /// Deliberately unit-level: neither Qwen vocabulary contains a merge that crosses a case
    /// boundary, so at integration level both routings emit identical ids and the assertion would
    /// be vacuous.
    /// </remarks>
    [Fact]
    public void Qwen35_KeepsCaseBoundariesInsideTheLetterRun_UnlikeTekken()
    {
        Assert.Single(Build("qwen35", "aB", "a B").Encode("aB"));
        Assert.Equal(2, Build("tekken", "aB", "a B").Encode("aB").Length);
    }

    /// <summary>
    /// Negative control for <see cref="UnknownPreType_Throws_NamingTheValue"/>, required by
    /// #373. The pre-fix behaviour — unknown <c>pre</c> resolved to a <c>null</c> regex, i.e.
    /// NO pre-tokenization — is reconstructed explicitly here via
    /// <see cref="BpeTokenizer.CreateTiktokenWithRegex"/> and shown to (a) load without
    /// complaint and (b) produce a materially different token stream. That is exactly what the
    /// old silent path did on an unknown value: it "passed", and mis-tokenized.
    /// </summary>
    [Fact]
    public void NegativeControl_TheOldSilentNoPreTokenizationPath_LoadsQuietlyAndMisTokenizes()
    {
        var silent = BpeTokenizer.CreateTiktokenWithRegex(
            Vocab("34"), merges: ["3 4"], tokenTypes: null, bosId: 0, eosId: 0, preRegex: null);

        // (a) No throw, no diagnostic — the failure mode #373 removed.
        // (b) One segment for the whole input, so the "3 4" merge fires where every
        //     real pipeline forbids it: 1 token vs 2 under qwen2, 3 vs 4 under llama3.
        Assert.Single(silent.Encode("34"));
        Assert.Equal(2, Build("qwen2").Encode("34").Length);
        Assert.Equal(3, silent.Encode("1234").Length);
        Assert.Equal(4, Build("llama3").Encode("1234").Length);
    }

    [Fact]
    public void UnknownPreType_WithOptOut_FallsBackToGpt2Default_NotToNone()
    {
        string? saved = Environment.GetEnvironmentVariable(OptOutVar);
        try
        {
            Environment.SetEnvironmentVariable(OptOutVar, "1");
            int[] viaOptOut = Build("no-such-pre-tokenizer").Encode("1234");
            // Fallback is the GPT-2 default pipeline (' ?\p{N}+' keeps "1234" one
            // number segment, so the 3+4 merge applies within it) — identical to
            // an explicit "gpt2", NOT to the old "no pre-tokenization".
            Assert.Equal(Build("gpt2").Encode("1234"), viaOptOut);
        }
        finally
        {
            Environment.SetEnvironmentVariable(OptOutVar, saved);
        }
    }

    [Fact]
    public void AbsentPreType_UsesGpt2Default_MatchingLlamaCpp()
    {
        // llama.cpp: missing tokenizer.ggml.pre → warn + 'default' (the GPT-2
        // pipeline). It must NOT mean "no pre-tokenization".
        Assert.Equal(Build("gpt2").Encode("hello world 1234"),
                     Build(null).Encode("hello world 1234"));
    }

    [Fact]
    public void GgufFactory_UnknownPre_SurfacesTheThrow()
    {
        // End-to-end through the GGUF metadata path ChatCommand/RunCommand use.
        var entries = new Dictionary<string, GgufMetadataValue>
        {
            ["tokenizer.ggml.model"]        = new(GgufValueType.String, "gpt2"),
            ["tokenizer.ggml.tokens"]       = new(GgufValueType.Array,
                Enumerable.Range(0, 256).Select(i => ((char)(0x100 + i)).ToString()).ToArray()),
            ["tokenizer.ggml.merges"]       = new(GgufValueType.Array, Array.Empty<string>()),
            ["tokenizer.ggml.bos_token_id"] = new(GgufValueType.UInt32, 0u),
            ["tokenizer.ggml.eos_token_id"] = new(GgufValueType.UInt32, 0u),
            ["tokenizer.ggml.pre"]          = new(GgufValueType.String, "definitely-unknown"),
        };

        var ex = Assert.Throws<InvalidDataException>(
            () => GgufBpeTokenizerFactory.Load(new GgufMetadata(entries)));
        Assert.Contains("definitely-unknown", ex.Message, StringComparison.Ordinal);
    }
}
