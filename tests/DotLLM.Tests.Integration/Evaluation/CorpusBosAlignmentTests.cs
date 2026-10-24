using DotLLM.Engine.Evaluation;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Evaluation;

/// <summary>
/// Issue #516: the corpus token stream must be <b>identical</b> to llama.cpp's, id for id.
/// </summary>
/// <remarks>
/// <para>
/// This is the acceptance criterion for #516, and it is deliberately a token-stream comparison
/// rather than a perplexity comparison. A "perplexity is close to llama.cpp" assertion passed
/// throughout the life of the bug it guards: dotLLM omitted the BOS that llama.cpp prepends, every
/// chunk boundary sat one token off, and the two engines aggregated over different text. On a
/// healthy Q8_0 model that cost ~0.1% and looked like agreement; on a degraded Q3_K model it cost
/// ~4% and was reported as a kernel divergence for two sessions (issue #515).
/// </para>
/// <para>
/// The two fixtures discriminate in opposite directions, which is the point — one of them fails
/// for a build that never prepends BOS, the other for a build that always does:
/// </para>
/// <list type="bullet">
/// <item><b>Llama-3.2-1B</b> — <c>pre = "llama-bpe"</c> and <b>no</b> <c>add_bos_token</c> key;
/// llama.cpp prepends BOS from the pre-type default.</item>
/// <item><b>SmolLM-135M</b> — <c>pre = "smollm"</c> with <c>add_bos_token = false</c>; llama.cpp
/// prepends nothing.</item>
/// </list>
/// <para>
/// The oracle files hold the exact ids llama.cpp scored, extracted from the header its
/// <c>--kl-divergence-base</c> writes (see <c>docs/PERPLEXITY.md</c>). They live in the shared
/// cache under <c>~/.dotllm/test-cache/token-oracles/</c> because regenerating them means a
/// multi-GB <c>kld.bin</c>.
/// </para>
/// </remarks>
public sealed class CorpusBosAlignmentTests
{
    private readonly ITestOutputHelper _output;

    public CorpusBosAlignmentTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Llama32_CorpusStream_MatchesLlamaCppIds_IncludingPrependedBos()
    {
        AssertCorpusStreamMatchesOracle(
            TestFixtureResolver.ResolveFile(
                ["DOTLLM_LLAMA32_1B_PURE_Q8_0_GGUF"], "quant-ladder", "Llama-3.2-1B-pure",
                ["Llama-3.2-1B-pure-Q8_0.gguf"],
                extraDirectories: [QuantLadderDir("Llama-3.2-1B-pure")]),
            "Llama-3.2-1B pure Q8_0 GGUF",
            "llama-3.2-1b-wiki-test-lf-64x512.txt",
            expectedLeadingBos: true);
    }

    [SkippableFact]
    public void SmolLm_CorpusStream_MatchesLlamaCppIds_WithNoBos()
    {
        AssertCorpusStreamMatchesOracle(
            TestFixtureResolver.ResolveFile(
                ["DOTLLM_SMOLLM_135M_PURE_Q8_0_GGUF"], "quant-ladder", "SmolLM-135M-pure",
                ["SmolLM-135M-pure-Q8_0.gguf"],
                extraDirectories: [QuantLadderDir("SmolLM-135M-pure")]),
            "SmolLM-135M pure Q8_0 GGUF",
            "smollm-135m-wiki-test-lf-4x512.txt",
            expectedLeadingBos: false);
    }

    private static string QuantLadderDir(string name) => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "quant-ladder", name);

    private void AssertCorpusStreamMatchesOracle(
        FixtureLocation fixture, string description, string oracleFileName, bool expectedLeadingBos)
    {
        Skip.If(!fixture.Found, fixture.SkipMessage(description));

        string corpus = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "test-cache", "corpora", "wikitext-2-raw", "wiki.test.lf.raw");
        Skip.If(!File.Exists(corpus), $"LF corpus not found at {corpus} (scripts/make_lf_corpus.py)");

        string oracle = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "test-cache", "token-oracles", oracleFileName);
        Skip.If(!File.Exists(oracle), $"Token oracle not found at {oracle}");

        int[] expected = File.ReadAllText(oracle)
            .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            .Select(int.Parse)
            .ToArray();

        using GgufFile gguf = GgufFile.Open(fixture.Path!);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        bool addBos = GgufAddBosResolver.Resolve(gguf.Metadata);

        Assert.Equal(expectedLeadingBos, addBos);
        Assert.Equal(expectedLeadingBos, expected[0] == tokenizer.BosTokenId);

        using var reader = new StreamReader(corpus);
        int[] actual = CorpusReader
            .StreamTokens(reader, tokenizer, maxTokens: expected.Length,
                bosTokenId: addBos ? tokenizer.BosTokenId : -1)
            .ToArray();

        _output.WriteLine($"{description}: add_bos={addBos}, {actual.Length} ids vs {expected.Length} expected");

        Assert.Equal(expected.Length, actual.Length);
        int firstMismatch = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i]) { firstMismatch = i; break; }
        }

        if (firstMismatch >= 0)
        {
            // A one-token shift is the signature of the #515 defect, so name it rather than
            // dumping two id lists and leaving the reader to spot it.
            int shifted = 0;
            for (int i = 0; i + 1 < expected.Length; i++)
                if (expected[i + 1] == actual[i]) shifted++;

            _output.WriteLine($"first mismatch at {firstMismatch}: expected {expected[firstMismatch]}, got {actual[firstMismatch]}");
            _output.WriteLine($"shift-by-one matches: {shifted}/{expected.Length - 1}");
        }

        Assert.Equal(-1, firstMismatch);
    }
}
