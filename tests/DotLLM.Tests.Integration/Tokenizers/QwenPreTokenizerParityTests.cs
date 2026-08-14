using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Tokenizers;

/// <summary>
/// Issue #397 — token-id parity against llama.cpp for the two Qwen
/// <c>tokenizer.ggml.pre</c> pipelines, on real GGUFs.
/// </summary>
/// <remarks>
/// <para><b>Why real weights.</b> The unit-level tests in
/// <c>DotLLM.Tests.Unit.Tokenizers.PreTokenizerPolicyTests</c> prove the routing is
/// <i>distinguishable</i> from gpt2/llama3 on a synthetic vocabulary. Only a real vocab +
/// merge table can show the segmentation is byte-for-byte the one llama.cpp produces, which
/// is what the perplexity numbers in #237/#397 actually depend on.</para>
/// <para><b>Provenance of the expected ids.</b> Produced with the official
/// <c>ggml-org/llama.cpp</c> release <b>b10434</b> Windows CPU build:
/// <c>llama-tokenize.exe -m &lt;gguf&gt; -f &lt;fixture&gt; --ids --no-bos</c>, with the
/// fixture file written as raw UTF-8 with no trailing newline. Upstream llama.cpp cannot
/// open the Ternary-Bonsai Q2_0 tensors (a non-upstream quantization type), so its ids were
/// taken from a tensor-less copy of that file's header + KV section — the same vocabulary,
/// merges and <c>tokenizer.ggml.pre</c>, no tensor data. That scratch file is not checked in;
/// regenerating it is a header/KV byte copy with <c>tensor_count</c> patched to 0.</para>
/// <para>Both fixtures resolve through <see cref="KnownTestFixtures"/> (env override, then the
/// dotLLM test cache, then the HF hub cache — #308). A missing fixture <b>skips loudly</b>,
/// naming every probed path.</para>
/// </remarks>
public sealed class QwenPreTokenizerParityTests
{
    /// <summary>
    /// Exercises every alternative the two pipelines disagree on: contractions, a digit run
    /// (bare <c>\p{N}</c> splits it per digit), a decomposed combining acute and a decomposed
    /// combining diaeresis (qwen35 keeps them inside the letter run, qwen2 does not), an
    /// em dash, a slash, and a newline + tab run.
    /// </summary>
    /// <remarks>Built from code points so no source-file re-encoding or NFC normalization can
    /// silently change what is being tokenized.</remarks>
    private static string Fixture =>
        "It's 2026: cafe" + (char)0x0301 +
        " costs 1234.56 " + (char)0x2014 +
        " nai" + (char)0x0308 + "ve/OK\n\tdone";

    /// <summary>llama.cpp b10434, <c>qwen2.5-0.5b-instruct-q8_0.gguf</c> (pre = <c>qwen2</c>).</summary>
    private static readonly int[] Qwen2ExpectedIds =
    [
        2132, 594, 220, 17, 15, 17, 21, 25, 40930, 53839, 7049, 220, 16, 17, 18, 19,
        13, 20, 21, 1959, 308, 2143, 136, 230, 586, 14, 3925, 198, 40495,
    ];

    /// <summary>llama.cpp b10434, <c>Ternary-Bonsai-27B-Q2_0.gguf</c> vocab (pre = <c>qwen35</c>).</summary>
    private static readonly int[] Qwen35ExpectedIds =
    [
        2064, 579, 220, 17, 15, 17, 21, 25, 39579, 52033, 6829, 220, 16, 17, 18, 19,
        13, 20, 21, 1892, 238883, 136, 230, 571, 14, 3793, 198, 39157,
    ];

    [SkippableFact]
    public void Qwen2Gguf_TokenIds_MatchLlamaCpp()
    {
        FixtureLocation loc = KnownTestFixtures.Qwen2_5_0_5B_Q8_0;
        Skip.If(!loc.Found, loc.SkipMessage(KnownTestFixtures.Qwen2_5_0_5BDescription));

        using var gguf = GgufFile.Open(loc.Path!);
        Assert.Equal("qwen2", gguf.Metadata.GetStringOrDefault("tokenizer.ggml.pre"));

        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        Assert.Equal(Qwen2ExpectedIds, tokenizer.Encode(Fixture));
    }

    [SkippableFact]
    public void Qwen35Gguf_TokenIds_MatchLlamaCpp()
    {
        FixtureLocation loc = KnownTestFixtures.TernaryBonsai27B_Q2_0;
        Skip.If(!loc.Found, loc.SkipMessage(KnownTestFixtures.TernaryBonsai27BDescription));

        using var gguf = GgufFile.Open(loc.Path!);
        Assert.Equal("qwen35", gguf.Metadata.GetStringOrDefault("tokenizer.ggml.pre"));

        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        Assert.Equal(Qwen35ExpectedIds, tokenizer.Encode(Fixture));
    }
}
