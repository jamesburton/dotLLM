using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Decode roofline probe for SmolLM2-135M (hidden 576) — the <b>small-model</b> point of the
/// <see cref="DecodeThreadScalingSweep"/> §5 sweep, and the gating measurement of the whole
/// threading investigation. Its decode matmuls are tiny, so SpinWait dispatch coordination cost
/// dominates and 32T historically collapsed (10.6 ms vs 32 µs per 30-dispatch burst). Confirming the
/// collapse still bites under current code is what justifies keeping a small-model cap floor; if it
/// no longer collapses, the risk picture for raising the default changes entirely.
/// See <c>.docs/decode-threading-investigation.md</c> §5.1/§5.3.
/// </summary>
/// <remarks>Opt-in via <c>DOTLLM_RUN_PREFILL_BENCH</c> (loads a ~145 MB model; never runs in CI).</remarks>
[Collection("SmolLM2Instruct")]
public class SmolLM2DecodeRooflineProbe
{
    private readonly SmolLM2InstructFixture _fixture;

    public SmolLM2DecodeRooflineProbe(SmolLM2InstructFixture fixture)
    {
        _fixture = fixture;
    }

    [SkippableFact]
    public void DecodeThreadScaling_RevealsBound()
    {
        Skip.If(string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DOTLLM_RUN_PREFILL_BENCH")),
            "Decode roofline probe is opt-in — set DOTLLM_RUN_PREFILL_BENCH=1 to run.");

        DecodeThreadScalingSweep.RunKneeSweep(_fixture.FilePath);
    }

    [SkippableFact]
    public void DecodeProductionConfigPaths_PinningNotSlower()
    {
        Skip.If(string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DOTLLM_RUN_PREFILL_BENCH")),
            "Decode production-path probe is opt-in — set DOTLLM_RUN_PREFILL_BENCH=1 to run.");

        DecodeThreadScalingSweep.RunProductionConfigPaths(_fixture.FilePath);
    }
}
