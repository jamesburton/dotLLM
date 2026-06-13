using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// BF16-vs-integer Q8_0 outer-product A/B accuracy measurement on Llama-3.2-1B-Instruct — the same
/// contract as <see cref="OuterProductBf16AccuracyTests"/> (SmolLM-135M) at Llama's much larger matmul
/// shapes, where bf16's 8-bit-mantissa rounding accumulates over deeper reductions (k up to 8192).
/// Reports the full-logits divergence plus inner / integer / bf16 perplexity on a fixed passage, so the
/// bf16 accuracy cost — the other half of the speed/quality trade — is substantiated on the same model
/// the prefill benchmark times, not just on the 135M proxy. Skips unless AVX512-BF16 is present (net11
/// on Zen4/Zen5/Strix); see <see cref="OuterProductBf16Accuracy"/> for the shared driver and rationale.
/// </summary>
[Collection("Llama32Instruct")]
public class Llama32OuterProductBf16AccuracyTests
{
    private readonly Llama32InstructFixture _fixture;

    public Llama32OuterProductBf16AccuracyTests(Llama32InstructFixture fixture)
    {
        _fixture = fixture;
    }

    [SkippableTheory]
    [InlineData("The capital of France is Paris and the weather today")]
    public void Bf16OuterProduct_WithinToleranceOfInteger_AndReportsPerplexity(string prompt)
        => OuterProductBf16Accuracy.AssertWithinToleranceOfInteger(_fixture.FilePath, prompt);
}
