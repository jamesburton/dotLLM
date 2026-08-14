using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Tokenizers;

/// <summary>
/// Issue #397 — real-GGUF gate for the two Qwen <c>tokenizer.ggml.pre</c> pipelines, asserting
/// <b>two independent properties</b> that are easy to conflate:
/// </summary>
/// <remarks>
/// <para><b>1. Parity.</b> dotLLM's token ids equal llama.cpp's on a fixture string. The expected
/// ids come from llama.cpp and nothing else — deriving them from dotLLM would make this a
/// self-consistency test wearing a parity test's name.</para>
/// <para><b>2. Discrimination.</b> Re-tokenizing the <i>same real vocabulary</i> under every other
/// pipeline must produce a <i>different</i> id stream. Without this a parity test passes unchanged
/// when the value is routed to the wrong pipeline, which is exactly what the first version of this
/// file did: measured on these two vocabularies, the original fixture was byte-identical under
/// qwen2, qwen35, llama3 <i>and</i> tekken, so it could only ever have caught a fall-back to gpt2.
/// Two properties of these vocabularies caused that, and the fixture below is built to defeat both:
/// <list type="bullet">
///   <item><description>Neither vocabulary contains a single ASCII multi-digit token, so
///     <c>\p{N}</c> vs <c>\p{N}{1,3}</c> is unobservable in ids no matter how many ASCII digits the
///     fixture has. The <b>fullwidth digits</b> U+FF11 U+FF10 are the only multi-digit tokens in
///     either vocabulary, so they are what separates qwen2/qwen35 from llama3.</description></item>
///   <item><description>Latin decomposed marks never merge with letter bytes here, so
///     <c>café</c>/<c>naïve</c>/<c>Zürich</c> tokenize identically under qwen2 and qwen35. A
///     <b>Thai</b> cluster (U+0E17 U+0E35 U+0E48 — base plus vowel sign plus tone mark) does merge,
///     so it is what separates qwen35's <c>[\p{L}\p{M}]+</c> from qwen2's <c>\p{L}+</c>.</description></item>
/// </list></para>
/// <para><b>Provenance of the expected ids.</b> Official <c>ggml-org/llama.cpp</c> release
/// <b>b10434</b> Windows CPU build:
/// <c>llama-tokenize.exe -m &lt;gguf&gt; -f &lt;fixture&gt; --ids --no-bos</c>, the fixture written as
/// raw UTF-8 with no trailing newline. Upstream llama.cpp cannot open the Ternary-Bonsai Q2_0
/// tensors (a non-upstream quantization type), so its ids were taken from a tensor-less copy of
/// that file's header + KV section — same vocabulary, merges and <c>tokenizer.ggml.pre</c>, no
/// tensor data. That scratch file is not checked in; regenerating it is a header/KV byte copy with
/// <c>tensor_count</c> patched to 0. The dotLLM side of both tests is a normal end-to-end
/// <see cref="GgufFile.Open"/> on the real GGUF.</para>
/// <para>Fixtures resolve through <see cref="KnownTestFixtures"/> (env override, then the dotLLM
/// test cache, then the HF hub cache — #308); a missing one <b>skips loudly</b>, naming every
/// probed path. No weights enter the repository.</para>
/// <para><b>Not covered here:</b> <c>qwen35</c> vs <c>tekken</c>. Those two agree exactly on both
/// vocabularies even with this fixture, because neither contains a merge that crosses an
/// upper/lower-case boundary — the only place their expressions differ. That discrimination is
/// therefore proved at unit level instead, on a vocabulary built to contain such a merge
/// (<c>PreTokenizerPolicyTests.Qwen35_KeepsCaseBoundariesInsideTheLetterRun_UnlikeTekken</c>).</para>
/// </remarks>
public sealed class QwenPreTokenizerParityTests
{
    /// <summary>
    /// Exercises the alternatives these pipelines actually disagree on <i>in these vocabularies</i>
    /// (verified, not assumed — see the class remarks): a contraction, fullwidth digits, a Thai
    /// base+mark cluster, an ASCII digit run, a decomposed combining acute and diaeresis, an em
    /// dash, a slash, and a newline + tab run.
    /// </summary>
    /// <remarks>Built from code points so no source re-encoding or NFC normalization can silently
    /// change what is being tokenized.</remarks>
    private static string Fixture =>
        "It's " + (char)0xFF11 + (char)0xFF10 +                      // fullwidth 1 0
        " " + (char)0x0E17 + (char)0x0E35 + (char)0x0E48 +           // Thai base + vowel sign + tone
        " 2026: cafe" + (char)0x0301 +                               // + COMBINING ACUTE
        " 1234.56 " + (char)0x2014 +                                 // + EM DASH
        " nai" + (char)0x0308 + "ve/OK\n\tdone";                     // + COMBINING DIAERESIS

    /// <summary>llama.cpp b10434, <c>qwen2.5-0.5b-instruct-q8_0.gguf</c> (pre = <c>qwen2</c>).</summary>
    private static readonly int[] Qwen2ExpectedIds =
    [
        2132, 594, 220, 20109, 26022, 220, 35884, 47171, 220, 17, 15, 17, 21, 25, 40930,
        53839, 220, 16, 17, 18, 19, 13, 20, 21, 1959, 308, 2143, 136, 230, 586, 14, 3925,
        198, 40495,
    ];

    /// <summary>llama.cpp b10434, <c>Ternary-Bonsai-27B-Q2_0.gguf</c> vocab (pre = <c>qwen35</c>).</summary>
    private static readonly int[] Qwen35ExpectedIds =
    [
        2064, 579, 220, 19496, 25191, 149496, 220, 17, 15, 17, 21, 25, 39579, 52033, 220,
        16, 17, 18, 19, 13, 20, 21, 1892, 238883, 136, 230, 571, 14, 3793, 198, 39157,
    ];

    /// <summary>
    /// Pipelines a mis-routing could plausibly land on. <c>tekken</c> is deliberately absent —
    /// see the class remarks.
    /// </summary>
    private static readonly string[] WrongPipelines = ["gpt2", "llama3", "starcoder", "gpt-4o"];

    [SkippableFact]
    public void Qwen2Gguf_TokenIds_MatchLlamaCpp_AndDifferUnderEveryOtherPipeline()
    {
        FixtureLocation loc = KnownTestFixtures.Qwen2_5_0_5B_Q8_0;
        Skip.If(!loc.Found, loc.SkipMessage(KnownTestFixtures.Qwen2_5_0_5BDescription));

        using var gguf = GgufFile.Open(loc.Path!);
        Assert.Equal("qwen2", gguf.Metadata.GetStringOrDefault("tokenizer.ggml.pre"));

        // Property 1 — parity with llama.cpp, through the production factory path.
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        Assert.Equal(Qwen2ExpectedIds, tokenizer.Encode(Fixture));

        // Property 2 — a wrong pipeline on this same vocabulary is actually observable.
        AssertDiscriminates(gguf.Metadata, "qwen2", [.. WrongPipelines, "qwen35"]);
    }

    [SkippableFact]
    public void Qwen35Gguf_TokenIds_MatchLlamaCpp_AndDifferUnderEveryOtherPipeline()
    {
        FixtureLocation loc = KnownTestFixtures.TernaryBonsai27B_Q2_0;
        Skip.If(!loc.Found, loc.SkipMessage(KnownTestFixtures.TernaryBonsai27BDescription));

        using var gguf = GgufFile.Open(loc.Path!);
        Assert.Equal("qwen35", gguf.Metadata.GetStringOrDefault("tokenizer.ggml.pre"));

        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        Assert.Equal(Qwen35ExpectedIds, tokenizer.Encode(Fixture));

        AssertDiscriminates(gguf.Metadata, "qwen35", [.. WrongPipelines, "qwen2"]);
    }

    /// <summary>
    /// Re-tokenizes <see cref="Fixture"/> with the model's real vocabulary and merge table but a
    /// forced <c>tokenizer.ggml.pre</c>, and asserts every wrong pipeline yields a different id
    /// stream than the correct one. This is what makes the parity assertion above load-bearing:
    /// it proves the fixture can tell the pipelines apart at all.
    /// </summary>
    /// <param name="metadata">Metadata of the real GGUF under test.</param>
    /// <param name="correctPre">The <c>pre</c> value the file actually declares.</param>
    /// <param name="wrongPres">Pipelines that must all disagree with <paramref name="correctPre"/>.</param>
    private static void AssertDiscriminates(
        GgufMetadata metadata, string correctPre, string[] wrongPres)
    {
        string[] tokens = metadata.GetStringArray("tokenizer.ggml.tokens");
        string[] merges = metadata.ContainsKey("tokenizer.ggml.merges")
            ? metadata.GetStringArray("tokenizer.ggml.merges")
            : [];
        int[]? tokenTypes = metadata.ContainsKey("tokenizer.ggml.token_type")
            ? metadata.GetInt32Array("tokenizer.ggml.token_type")
            : null;

        int[] Encode(string pre) => BpeTokenizer
            .CreateTiktoken(tokens, merges, tokenTypes, bosId: 0, eosId: 0, preTokenizerType: pre)
            .Encode(Fixture);

        int[] correct = Encode(correctPre);
        foreach (string wrong in wrongPres)
        {
            Assert.False(
                correct.AsSpan().SequenceEqual(Encode(wrong)),
                $"Fixture cannot discriminate '{correctPre}' from '{wrong}' on this vocabulary — " +
                "the parity assertion would pass even if the pre value were routed to the wrong " +
                "pipeline. Extend the fixture (see this class's remarks for how the current " +
                "discriminators were chosen).");
        }
    }
}
