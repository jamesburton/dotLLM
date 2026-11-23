using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// #1922 review I3 / I1: the pure gates in front of the dense v2 decode kernel and the step graph. No GPU needed.
/// <list type="bullet">
/// <item><see cref="CudaDenseDecodeV2.Eligible"/> is the only thing keeping sliding-window layers, attention sinks, the FP16 window
/// ring and non-newest-row queries off a kernel that has none of them; every clause has a case that fails only that clause.</item>
/// <item><see cref="CudaTransformerModel.StepGraphAdmitsQuantized"/> keeps the windowed quantized cache (host-side eviction
/// bookkeeping advanced during capture) and the dequant-scratch oracle out of the step graph.</item>
/// </list>
/// </summary>
public sealed class CudaDenseDecodeGateTests
{
    // Baseline: an eligible decode step at depth 1000 with a 512-row threshold.
    private const int Min = 512;

    [Theory]
    //          enabled seqLen seqKv  offset slidingWindow sinks  window  expected   (the clause each false case fails)
    [InlineData(true,   1,     1000,  999,   0,            false, 0,      true)]
    [InlineData(true,   1,     512,   511,   0,            false, 0,      true)]    // exactly at the threshold
    [InlineData(true,   1,     1000,  999,   -1,           false, 0,      true)]    // "no window" spelled -1
    [InlineData(false,  1,     1000,  999,   0,            false, 0,      false)]   // kill switch
    [InlineData(true,   2,     1000,  999,   0,            false, 0,      false)]   // not a decode step
    [InlineData(true,   1,     511,   510,   0,            false, 0,      false)]   // below the threshold
    [InlineData(true,   1,     1000,  998,   0,            false, 0,      false)]   // query is not the newest row
    [InlineData(true,   1,     1000,  999,   4096,         false, 0,      false)]   // sliding-window layer
    [InlineData(true,   1,     1000,  999,   0,            true,  0,      false)]   // attention sinks
    [InlineData(true,   1,     1000,  999,   0,            false, 64,     false)]   // FP16 window ring
    public void DenseV2_Eligible_EachClauseRejectsAlone(bool enabled, int seqLen, int seqKv, int offset, int slidingWindow,
        bool sinks, int window, bool expected)
        => Assert.Equal(expected, CudaDenseDecodeV2.Eligible(enabled, seqLen, seqKv, offset, slidingWindow, sinks, window, Min));

    [Theory]
    [InlineData(false, 0, true)]    // pure quantized: captured
    [InlineData(false, 16, false)]  // windowed: eviction bookkeeping is not idempotent under a part-failed capture
    [InlineData(true, 0, false)]    // test-only dequant-scratch oracle
    public void StepGraph_AdmitsOnlyPureQuantizedCaches(bool referenceScratch, int window, bool expected)
        => Assert.Equal(expected, CudaTransformerModel.StepGraphAdmitsQuantized(referenceScratch, window));
}
