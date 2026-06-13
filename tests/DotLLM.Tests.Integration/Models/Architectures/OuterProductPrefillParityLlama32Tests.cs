using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Same-process A/B parity test for the Q8_0 outer-product prefill path on Llama-3.2-1B-Instruct.
/// A second architecture with larger and different projection dims than SmolLM-135M, so it exercises
/// different M (output-row) / K tile shapes through <c>OuterProductGemmQ8_0</c>. See
/// <see cref="OuterProductPrefillParity"/> for the shared driver and rationale.
/// </summary>
[Collection("Llama32Instruct")]
public class OuterProductPrefillParityLlama32Tests
{
    private readonly Llama32InstructFixture _fixture;

    public OuterProductPrefillParityLlama32Tests(Llama32InstructFixture fixture)
    {
        _fixture = fixture;
    }

    [Theory]
    [InlineData("Hello world")]
    [InlineData("The capital of France")]
    [InlineData("The capital of France is Paris and the weather today")]
    public void OuterProductPrefill_MatchesInnerProduct_FullLogits(string prompt)
        => OuterProductPrefillParity.AssertMatchesInnerProduct(_fixture.FilePath, prompt);
}
