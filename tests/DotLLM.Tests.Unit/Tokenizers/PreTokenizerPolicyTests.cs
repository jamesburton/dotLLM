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
    private static BpeTokenizer Build(string? preType)
    {
        char[] byteToUnicode = new char[256];
        for (int b = 33; b <= 126; b++) byteToUnicode[b] = (char)b;
        for (int b = 161; b <= 172; b++) byteToUnicode[b] = (char)b;
        for (int b = 174; b <= 255; b++) byteToUnicode[b] = (char)b;
        int n = 0;
        for (int b = 0; b < 256; b++)
            if (byteToUnicode[b] == 0) byteToUnicode[b] = (char)(0x100 + n++);

        string[] tokens = new string[257];
        for (int i = 0; i < 256; i++) tokens[i] = byteToUnicode[i].ToString();
        tokens[256] = "34";

        return BpeTokenizer.CreateTiktoken(tokens, merges: ["3 4"], tokenTypes: null,
            bosId: 0, eosId: 0, preTokenizerType: preType);
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

    [Fact]
    public void UnknownPreType_Throws_NamingTheValue()
    {
        var ex = Assert.Throws<InvalidDataException>(() => Build("no-such-pre-tokenizer"));
        Assert.Contains("no-such-pre-tokenizer", ex.Message, StringComparison.Ordinal);
        Assert.Contains(OptOutVar, ex.Message, StringComparison.Ordinal);
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
