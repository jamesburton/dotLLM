using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Same-process A/B parity test for the Q8_0 outer-product prefill path on Llama-3.2-1B-Instruct.
/// Identical contract to <see cref="OuterProductPrefillParityTests"/> (SmolLM-135M) but at Llama's
/// much larger matmul shapes (hidden 2048, intermediate 8192 — 4.5× SmolLM's accumulation depth),
/// where inner-product vs outer-product reduction-order divergence is largest. See
/// <see cref="OuterProductPrefillParity"/> for the shared driver and the scale-normalized tolerance
/// rationale; the discriminating correctness proof at these shapes is the scalar-ground-truth arbiter
/// (<c>OuterProductGemmTests.GroundTruth_GemmAndOuter_AtRealisticShapes</c>, extended to k=2048/8192).
/// </summary>
[Collection("Llama32Instruct")]
public class Llama32OuterProductPrefillParityTests
{
    private readonly Llama32InstructFixture _fixture;

    public Llama32OuterProductPrefillParityTests(Llama32InstructFixture fixture)
    {
        _fixture = fixture;
    }

    // Same boundary-probing prompts as the SmolLM parity test: token counts that land on different
    // remainder tiles of the outer-product dispatcher (single-token tail, exact 3-tile, multi-tile + tail).
    [Theory]
    [InlineData("Hello world")]
    [InlineData("The capital of France")]
    [InlineData("The capital of France is Paris and the weather today")]
    public void OuterProductPrefill_MatchesInnerProduct_FullLogits(string prompt)
        => OuterProductPrefillParity.AssertMatchesInnerProduct(_fixture.FilePath, prompt);
}
