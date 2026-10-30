using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Issue #1276: the re-published Ternary Bonsai 2 small models declare
/// <c>prism.hadamard.version = 2</c> (tied lm_head = the rotated <c>token_embd</c>). dotLLM's CPU
/// greedy decode must reproduce <b>PrismML's own runtime</b> on the identical file.
/// </summary>
/// <remarks>
/// <para>
/// <b>Oracle capture (2026-10-25):</b> PrismML-Eng/llama.cpp branch <c>prism</c> at
/// <c>c96baf4a</c>, CPU-only build (<c>-DGGML_CUDA=OFF</c>), on
/// <c>Ternary-Bonsai-2-2B-PQ2_0.gguf</c>:
/// <c>llama-completion -m … -p "&lt;prompt&gt;" -n &lt;N&gt; --temp 0 -no-cnv</c>. The continuations are
/// compared as exact text (trailing whitespace trimmed): N greedy tokens reproduced verbatim.
/// </para>
/// <para>
/// <b>PTQ1_0 (#1280):</b> the same fork build on <c>Ternary-Bonsai-2-2B-PTQ1_0.gguf</c> (2026-10-26) produces the
/// identical continuation for both probes (the packs decode to the same ternary weights), so one reference serves
/// both packings.
/// </para>
/// <para>
/// Gated on the 2B fixtures (<c>DOTLLM_BONSAI2_2B_GGUF</c> / <c>DOTLLM_BONSAI2_2B_PTQ1_0_GGUF</c>, or the
/// <c>prism-ml/Ternary-Bonsai-2-2B-gguf</c> cache / model store); skips when absent.
/// </para>
/// </remarks>
public sealed class Bonsai2SmallReferenceParityTests
{
    // (prompt, greedy tokens, PrismML fork continuation) — see remarks for the capture.
    private static readonly (string Prompt, int MaxTokens, string Reference)[] ProbeSet =
    [
        (
            "The capital of France is", 32,
            " Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\n" +
            "The capital of France is Paris.\nThe capital of France is"
        ),
        ("Water boils at a temperature of", 24, " 100°C.\nA. 100°C\nB. 100°C\nC"),
    ];

    /// <summary>Every probe against every packing of the 2B checkpoint.</summary>
    public static TheoryData<QuantizationType, string, int, string> Probes
    {
        get
        {
            var data = new TheoryData<QuantizationType, string, int, string>();
            foreach (var pack in new[] { QuantizationType.PQ2_0, QuantizationType.PTQ1_0 })
            {
                foreach (var (prompt, maxTokens, reference) in ProbeSet)
                    data.Add(pack, prompt, maxTokens, reference);
            }

            return data;
        }
    }

    private readonly ITestOutputHelper _output;

    public Bonsai2SmallReferenceParityTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [MemberData(nameof(Probes))]
    public void GreedyDecode_MatchesPrismMLReference(QuantizationType pack, string prompt, int maxTokens, string reference)
    {
        FixtureLocation fixture = pack == QuantizationType.PTQ1_0
            ? KnownTestFixtures.TernaryBonsai2_2B_PTQ1_0
            : KnownTestFixtures.TernaryBonsai2_2B_PQ2_0;
        Skip.If(!fixture.Found, fixture.SkipMessage(KnownTestFixtures.TernaryBonsai2_2BDescription));

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.True(config.HadamardFold is { TiedOutput: true }, "fixture is not a prism.hadamard v2 (tied) file");
        Assert.Equal(pack, gguf.TensorsByName["token_embd.weight"].QuantizationType);

        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        var options = new InferenceOptions { Temperature = 0f, MaxTokens = maxTokens, RepetitionPenalty = 1.0f };
        string text = new TextGenerator(model, tokenizer).Generate(prompt, options).Text;
        _output.WriteLine($"dotLLM [{pack}]: \"{text}\"");

        Assert.Equal(reference.TrimEnd(), text.TrimEnd());
    }
}
