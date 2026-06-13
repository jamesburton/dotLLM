using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Same-process A/B parity test for the Q8_0 outer-product prefill path on SmolLM-135M.
/// See <see cref="OuterProductPrefillParity"/> for the shared driver and rationale.
/// </summary>
[Collection("SmallModel")]
public class OuterProductPrefillParityTests
{
    private readonly SmallModelFixture _fixture;

    public OuterProductPrefillParityTests(SmallModelFixture fixture)
    {
        _fixture = fixture;
    }

    // Prompts chosen so the token count N lands on different remainder-tile boundaries of the
    // outer-product dispatcher (AVX2 steps by 3, AVX-512 by 6, then single-token tails):
    //   N small (2)   → no full 3-tile, all single-token tail path
    //   N=3-ish       → exact 3-tile, no tail
    //   N larger (>9) → multiple full tiles + a 1- or 2-token tail
    // This guards the "N not a multiple of 6/3" remainder handling in OuterProductGemmQ8_0.
    [Theory]
    [InlineData("Hello world")]
    [InlineData("The capital of France")]
    [InlineData("The capital of France is Paris and the weather today")]
    public void OuterProductPrefill_MatchesInnerProduct_FullLogits(string prompt)
        => OuterProductPrefillParity.AssertMatchesInnerProduct(_fixture.FilePath, prompt);
}
