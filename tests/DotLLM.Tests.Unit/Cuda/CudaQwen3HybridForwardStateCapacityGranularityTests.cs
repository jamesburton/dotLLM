using DotLLM.Cuda.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU-only tests for the capacity-rounding math issue #188 introduced in
/// <see cref="CudaQwen3HybridDenseForwardState.RoundUpCapacity"/> and
/// <see cref="CudaQwen3MoeHybridForwardState.RoundUpCapacity"/>. Both methods are pure
/// (no CUDA calls), so this discriminates the sizing policy itself without needing a GPU or
/// constructing either state class (whose constructors allocate real device memory).
/// </summary>
/// <remarks>
/// Issue #188: <c>EnsureCapacity</c> used to round every requested length up to the next
/// power of two via <c>BitOperations.RoundUpToPowerOf2</c>, which meant a request one token past
/// a power-of-two boundary (e.g. seqLen=1025) doubled the allocation (cap=2048) -- exactly what
/// let <c>dotllm bench --depth 1536</c> (which rounded to cap=2048) land on 0 MiB free VRAM on a
/// 12 GB card. The fix keeps pow2 rounding below <see cref="CudaQwen3HybridDenseForwardState.CapacityGranularity"/>
/// tokens (cheap in absolute bytes, preserves amortization for small varying-length calls) and
/// switches to fixed-granularity rounding above it (bounds worst-case waste to at most
/// <c>CapacityGranularity - 1</c> tokens regardless of scale). These tests pin down both regimes
/// and the regime boundary itself.
/// </remarks>
public sealed class CudaQwen3HybridForwardStateCapacityGranularityTests
{
    // Both classes' RoundUpCapacity are independent copies of the same policy (issue #188 --
    // "apply the fix to BOTH forward-state classes" mirrors #185's cross-backend rule) --
    // asserted to agree via the theory below, then exercised individually so a future edit to
    // just one of the two copies fails immediately instead of silently diverging.

    [Theory]
    // ── Below-granularity regime: unchanged pow2 rounding ──
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 4)]
    [InlineData(8, 8)]
    [InlineData(200, 256)]
    [InlineData(256, 256)] // exactly at the regime boundary -- still the pow2 branch (seqLen <= granularity)
    // ── Above-granularity regime: fixed-granularity rounding ──
    [InlineData(257, 512)] // one token over the boundary: old pow2 scheme would ALSO give 512 here,
                            // so this alone doesn't discriminate -- see the next cases.
    [InlineData(513, 768)] // old pow2 scheme: RoundUpToPowerOf2(513) = 1024 (waste=511).
                            // New scheme: ceil(513/256)*256 = 768 (waste=255) -- exactly half the waste.
    [InlineData(768, 768)] // exact multiple of granularity -- zero waste either scheme reaches only
                            // by coincidence; old pow2 scheme actually gives 1024 here (waste=256),
                            // this is the depth=768 case from issue #188's own repro.
    [InlineData(1024, 1024)] // exact power of two AND exact multiple of granularity -- both regimes agree.
    [InlineData(1025, 1280)] // one past a pow2 boundary: old scheme gives 2048 (waste=1023, ~2x).
                             // New scheme gives 1280 (waste=255) -- the core issue #188 fix.
    [InlineData(1536, 1536)] // issue #188's own repro depth: old scheme rounds to 2048 (waste=512,
                             // landing on 0 MiB free VRAM on a 12 GB card); new scheme is exact.
    [InlineData(2048, 2048)] // exact power of two AND exact multiple of granularity -- no rounding
                             // waste is available to reclaim here under ANY monotonic scheme; the
                             // depth=2048 repro case needed BenchRunner-side chunking (see
                             // DotLLM.Cli's BenchRunner.DepthExtensionChunkSize), not a sizing-policy
                             // change, to get comfortable headroom.
    public void DenseAndMoeRoundUpCapacity_Agree_AndMatchExpected(int seqLen, int expectedCap)
    {
        int denseCap = CudaQwen3HybridDenseForwardState.RoundUpCapacity(seqLen);
        int moeCap = CudaQwen3MoeHybridForwardState.RoundUpCapacity(seqLen);

        Assert.Equal(expectedCap, denseCap);
        Assert.Equal(expectedCap, moeCap);
    }

    [Theory]
    [InlineData(257)]
    [InlineData(1000)]
    [InlineData(1537)]
    [InlineData(4097)]
    public void RoundUpCapacity_AboveGranularity_WasteIsBoundedByGranularityMinusOne(int seqLen)
    {
        int denseCap = CudaQwen3HybridDenseForwardState.RoundUpCapacity(seqLen);
        int moeCap = CudaQwen3MoeHybridForwardState.RoundUpCapacity(seqLen);

        Assert.True(denseCap >= seqLen);
        Assert.True(denseCap - seqLen < CudaQwen3HybridDenseForwardState.CapacityGranularity);
        Assert.True(moeCap >= seqLen);
        Assert.True(moeCap - seqLen < CudaQwen3MoeHybridForwardState.CapacityGranularity);
    }

    [Fact]
    public void RoundUpCapacity_NeverReturnsLessThanRequested()
    {
        foreach (int seqLen in new[] { 1, 2, 7, 8, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4096, 4097 })
        {
            Assert.True(CudaQwen3HybridDenseForwardState.RoundUpCapacity(seqLen) >= seqLen);
            Assert.True(CudaQwen3MoeHybridForwardState.RoundUpCapacity(seqLen) >= seqLen);
        }
    }
}
