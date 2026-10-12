using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The qwen4exp MTP-head attach guard (#820): the arithmetic estimate and the second-device canary decision. No GPU needed. The figures are
/// the ones measured on the 127 GiB Strix Halo with UD-Q4_K_XL (resident 77.86 GiB, head + snapshots 2.00, 8K context 0.46, other processes 1.22).
/// </summary>
public sealed class Qwen4ExpMtpAttachEstimateTests
{
    private const long GiB = 1L << 30;

    private static Qwen4ExpMtpAttachEstimate Q4kxl(long capacity = 83_323L << 20, long others = 1_250L << 20) =>
        new(LiveBytes: 79_729L << 20, HeadBytes: 2 * GiB, ContextBytes: 470L << 20, ScratchBytes: 20L << 20, OtherProcessBytes: others, CapacityBytes: capacity);

    [Fact]
    public void Needed_IsTheItemisedSum_WithNoFlatGiBForScratchThatIsAlreadyResident()
    {
        var e = Q4kxl();
        long expected = e.LiveBytes + e.HeadBytes + e.ContextBytes + e.ScratchBytes + e.OtherProcessBytes + Qwen4ExpMtpAttachEstimate.DefaultMarginBytes;
        Assert.Equal(expected, e.NeededBytes);
        // The pre-#820 estimate added 1 GiB + the full-capacity K/V on top of a trunk that already held its scratch: 82.5 GiB. The sum here is below that.
        Assert.True(e.NeededBytes < 82L * GiB + GiB / 2, e.Describe());
    }

    [Fact]
    public void Arithmetic_AloneStaysConservative_WhenTheOsLimitIsAssumed()
    {
        // 77.86 + 2.00 + 0.46 + 0.02 + 0.25 + 1.22 = 81.8 GiB against 0.64 x 127.1 = 81.37: the arithmetic still declines. Only a probe can say the real limit is higher.
        var e = Q4kxl();
        Assert.False(e.Fits);
        Assert.False(e.Decide(null).Fits);
        Assert.False(e.Decide(new HeadroomProbeResult(HeadroomOutcome.Unavailable, e.ProbeBytes, 0, 0, "no second device")).Fits);
    }

    [Fact]
    public void ConclusiveProbe_Overrides_TheArithmetic_BothWays()
    {
        var e = Q4kxl();
        var pass = new HeadroomProbeResult(HeadroomOutcome.Fits, e.ProbeBytes, e.ProbeBytes, 800, null);
        var v = e.Decide(pass);
        Assert.True(v.Fits);
        Assert.Contains("second device", v.Reason, StringComparison.Ordinal);

        // The OS limit drifted down (81.4 -> 73.9 GiB measured on the same idle box): the arithmetic against 0.64 x RAM still fits, the probe does not.
        var roomy = Q4kxl(capacity: 95 * GiB);
        Assert.True(roomy.Fits);
        var fail = new HeadroomProbeResult(HeadroomOutcome.DoesNotFit, roomy.ProbeBytes, 1_400L << 20, 300, "VK_ERROR_OUT_OF_DEVICE_MEMORY");
        var refused = roomy.Decide(fail);
        Assert.False(refused.Fits);
        Assert.Contains("1.37 GiB", refused.Reason, StringComparison.Ordinal);   // how far the probe got
        Assert.Contains("77.", refused.Reason, StringComparison.Ordinal);        // what is resident
    }

    [Fact]
    public void ProbeBytes_CoversEverythingTheAttachAdds_PlusMarginAndSlack()
    {
        var e = Q4kxl();
        Assert.Equal(e.HeadBytes + e.ContextBytes + e.ScratchBytes + e.MarginBytes + Qwen4ExpMtpAttachEstimate.ProbeSlackBytes, e.ProbeBytes);
        // ~3.0 GiB for the real file: small enough to allocate and touch in about a second.
        Assert.InRange(e.ProbeBytes, 2L * GiB, 4L * GiB);
    }
}

/// <summary>The canary loop (#820) against a fake device: chunking, the failing point, and what counts as "no information".</summary>
public sealed class VulkanHeadroomCanaryTests
{
    private sealed class FakeTarget(long capacity, Exception? failure = null) : IHeadroomCanaryTarget
    {
        public long Held;
        public bool Disposed;

        public void AllocateAndTouch(long bytes)
        {
            if (failure is not null && Held + bytes > capacity) throw failure;
            Held += bytes;
        }

        public void Dispose() => Disposed = true;
    }

    private static VulkanException Oom() => new(-2, "VK_ERROR_OUT_OF_DEVICE_MEMORY: vkQueueSubmit");

    [Fact]
    public void Fits_WhenEveryChunkIsAllocatedAndTouched_AndThePartialLastChunkIsExact()
    {
        var t = new FakeTarget(long.MaxValue);
        var r = VulkanHeadroomCanary.Run(t, bytes: 1000, chunk: 300);
        Assert.Equal(HeadroomOutcome.Fits, r.Outcome);
        Assert.Equal(1000, t.Held);
        Assert.Equal(1000, r.ReachedBytes);
        Assert.True(t.Disposed);
    }

    [Fact]
    public void DoesNotFit_ReportsHowFarItGot_AndStillDisposesTheDevice()
    {
        var t = new FakeTarget(capacity: 700, failure: Oom());
        var r = VulkanHeadroomCanary.Run(t, bytes: 1000, chunk: 300);
        Assert.Equal(HeadroomOutcome.DoesNotFit, r.Outcome);
        Assert.Equal(600, r.ReachedBytes);       // two whole chunks fit, the third crossed the limit
        Assert.Equal(1000, r.RequestedBytes);
        Assert.Contains("OUT_OF_DEVICE_MEMORY", r.Detail, StringComparison.Ordinal);
        Assert.True(t.Disposed);
    }

    [Fact]
    public void UnexpectedFailure_IsNoInformation_NotARefusal()
    {
        var t = new FakeTarget(capacity: 0, failure: new InvalidOperationException("driver hiccup"));
        var r = VulkanHeadroomCanary.Run(t, bytes: 1000, chunk: 300);
        Assert.Equal(HeadroomOutcome.Unavailable, r.Outcome);   // the arithmetic decides; a flaky probe must not switch speculation off by itself
        Assert.True(t.Disposed);
    }

    [Fact]
    public void ZeroBytes_FitsWithoutTouchingTheDevice()
    {
        var t = new FakeTarget(0, Oom());
        var r = VulkanHeadroomCanary.Run(t, bytes: 0, chunk: 300);
        Assert.Equal(HeadroomOutcome.Fits, r.Outcome);
        Assert.Equal(0, t.Held);
    }
}
