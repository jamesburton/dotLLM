using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Decode roofline probe for Bielik-1.5B-v3.0 (hidden ~1536–2048) — the <b>mid-model</b> point of the
/// <see cref="DecodeThreadScalingSweep"/> §5 sweep. Sits between SmolLM-135M (collapses at 32T) and
/// Llama-3.2-1B (scales to 32T); the decode-thread count at which 32T flips from loss to win on this
/// model is the crossover that sets the threshold for any work-size-adaptive dispatch gate.
/// See <c>.docs/decode-threading-investigation.md</c> §5.2.
/// </summary>
/// <remarks>Opt-in via <c>DOTLLM_RUN_PREFILL_BENCH</c> (loads a ~1.6 GB model; never runs in CI).</remarks>
[Collection("BielikQ8Model")]
public class BielikDecodeRooflineProbe
{
    private readonly BielikQ8ModelFixture _fixture;

    public BielikDecodeRooflineProbe(BielikQ8ModelFixture fixture)
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
