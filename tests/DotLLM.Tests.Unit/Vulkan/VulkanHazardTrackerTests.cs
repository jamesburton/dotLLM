using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Hazard-scoped barrier tracker tests (issue #144).
/// </summary>
/// <remarks>
/// <para>
/// <see cref="SpirvReflectionTests"/> covers the shader-derived read/write
/// masks; this class covers the RAW/WAR/WAW epoch logic itself. The tracker
/// records real <c>vkCmdPipelineBarrier</c> commands, so the logic tests run
/// against an open command buffer on a real device (skipped when no Vulkan
/// driver is present); the buffer "handles" passed to the guards are opaque
/// dictionary keys and never reach the driver.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanHazardTrackerTests
{
    // Opaque per-test buffer identities.
    private const nint W1 = 0x1000; // weights (never written)
    private const nint W2 = 0x2000;
    private const nint X = 0x3000;
    private const nint O1 = 0x4000;
    private const nint O2 = 0x5000;

    private static void RunWithTracker(Action<VulkanHazardTracker, nint> body)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");

        using var device = VulkanDevice.Create();
        using var submit = device.CreateSubmitContext();
        submit.Begin();
        var tracker = new VulkanHazardTracker();
        tracker.Begin(submit.CommandBuffer);
        body(tracker, submit.CommandBuffer);
        submit.SubmitAndWait(); // Barrier-only command buffer — valid to submit.
    }

    private static long Barriers => ProfileCounters.HazardBarriers;

    /// <summary>reads = buffers with mask bit clear, writes = bit set (bit i = index i).</summary>
    private static void Dispatch(VulkanHazardTracker t, uint writesMask, params nint[] buffers)
        => t.OnDispatch(buffers, writesMask);

    [SkippableFact]
    public void ReadAfterWrite_EmitsExactlyOneBarrier()
    {
        RunWithTracker((t, cmdBuf) =>
        {
            long before = Barriers;
            Dispatch(t, 0b10, X, O1);      // reads X, writes O1
            Assert.Equal(before, Barriers); // first op: nothing pending
            Dispatch(t, 0b10, O1, O2);     // reads O1 (pending write) → barrier
            Assert.Equal(before + 1, Barriers);
            Dispatch(t, 0b10, O1, X);      // reads O1 again — already synced
            Assert.Equal(before + 1, Barriers);
        });
    }

    [SkippableFact]
    public void IndependentDispatches_ShareNoBarrier()
    {
        RunWithTracker((t, cmdBuf) =>
        {
            long before = Barriers;
            // Q/K/V-style fan-out: three dispatches read the same input and
            // write disjoint outputs — no barriers between them.
            Dispatch(t, 0b100, X, W1, O1);
            Dispatch(t, 0b100, X, W1, O2);
            Dispatch(t, 0b100, X, W2, W2 + 4); // distinct dummy output
            Assert.Equal(before, Barriers);
        });
    }

    [SkippableFact]
    public void BarrierIsBatched_OverMultiplePendingWrites()
    {
        RunWithTracker((t, cmdBuf) =>
        {
            long before = Barriers;
            Dispatch(t, 0b1, O1);          // writes O1
            Dispatch(t, 0b1, O2);          // writes O2 (independent)
            Assert.Equal(before, Barriers);
            Dispatch(t, 0b100, O1, O2, X); // reads BOTH pending writes → ONE barrier
            Assert.Equal(before + 1, Barriers);
        });
    }

    [SkippableFact]
    public void WriteAfterRead_EmitsBarrier()
    {
        RunWithTracker((t, cmdBuf) =>
        {
            long before = Barriers;
            Dispatch(t, 0b10, X, O1); // reads X
            Dispatch(t, 0b1, X);      // writes X while read pending → WAR barrier
            Assert.Equal(before + 1, Barriers);
        });
    }

    [SkippableFact]
    public void WriteAfterWrite_EmitsBarrier()
    {
        RunWithTracker((t, cmdBuf) =>
        {
            long before = Barriers;
            Dispatch(t, 0b1, O1);
            Dispatch(t, 0b1, O1);
            Assert.Equal(before + 1, Barriers);
        });
    }

    [SkippableFact]
    public void Transfer_ParticipatesInHazards()
    {
        RunWithTracker((t, cmdBuf) =>
        {
            long before = Barriers;
            t.OnTransfer(W1, O1);       // copy weights → O1
            Dispatch(t, 0b10, W1, O2);  // reads W1 (only read by the copy) — no barrier
            Assert.Equal(before, Barriers);
            Dispatch(t, 0b10, O1, X);   // reads the copy's destination → barrier
            Assert.Equal(before + 1, Barriers);
        });
    }

    [SkippableFact]
    public void Begin_ResetsEpochState()
    {
        RunWithTracker((t, cmdBuf) =>
        {
            Dispatch(t, 0b1, O1);
            long before = Barriers;
            // Simulate the next forward: state cleared, pending write forgotten
            // (the inter-forward fence made it visible).
            t.Begin(cmdBuf);
            Dispatch(t, 0b10, O1, O2);
            Assert.Equal(before, Barriers);
        });
    }
}
