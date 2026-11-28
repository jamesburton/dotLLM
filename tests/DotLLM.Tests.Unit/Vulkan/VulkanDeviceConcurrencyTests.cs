using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Several threads reach one <see cref="VulkanDevice"/> at once in the server (a request thread's text forward, an image encode on a <c>Task.Run</c> worker, the scheduler, load-time uploads;
/// issue #2068). Vulkan needs <c>vkQueueSubmit</c> externally synchronised on the shared queue, command pools externally synchronised for recording, and the device-wide hazard tracker
/// must not be armed or disarmed across threads. These tests pin each of those without a model.
/// </summary>
public sealed class VulkanDeviceConcurrencyTests
{
    private readonly ITestOutputHelper _output;
    public VulkanDeviceConcurrencyTests(ITestOutputHelper output) => _output = output;

    private static VulkanDevice Device()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");
        return VulkanDevice.Create();
    }

    /// <summary>
    /// Four threads, each with its own <see cref="VulkanDevice.SubmitContext"/>, submit empty command buffers in a loop; the critical section is widened by a hook so an unsynchronised
    /// submit is observed with certainty rather than by luck. The device records how many threads were ever inside <c>vkQueueSubmit</c> together: it must be exactly 1.
    /// </summary>
    [SkippableFact]
    public void ConcurrentSubmitsFromManyThreads_NeverOverlapOnTheQueue()
    {
        using var dev = Device();
        dev.QueueSubmitWindowHook = () => Thread.Sleep(1);
        const int threads = 4, iterations = 25;
        var errors = new List<Exception>();
        var workers = Enumerable.Range(0, threads).Select(_ => new Thread(() =>
        {
            try
            {
                using var ctx = dev.CreateSubmitContext();
                for (int i = 0; i < iterations; i++) { ctx.Begin(); ctx.SubmitAndWait(); }
            }
            catch (Exception e) { lock (errors) errors.Add(e); }
        })).ToList();
        foreach (var t in workers) { t.IsBackground = true; t.Start(); }
        foreach (var t in workers) Assert.True(t.Join(TimeSpan.FromSeconds(60)), "a submitting thread did not finish (device deadlock)");
        _output.WriteLine($"max threads inside vkQueueSubmit at once: {dev.MaxConcurrentQueueSubmits} ({threads} threads x {iterations} submits)");
        Assert.Empty(errors);
        Assert.Equal(1, dev.MaxConcurrentQueueSubmits);
    }

    /// <summary>One-shot copies and the staging-slot command buffers (load-time uploads on one thread while a forward submits on another) go through the same serialised submit.</summary>
    [SkippableFact]
    public void OneShotCopiesAndContextSubmits_ShareOneSerialisedQueue()
    {
        using var dev = Device();
        dev.QueueSubmitWindowHook = () => Thread.Sleep(1);
        using var a = dev.AllocateDeviceLocal(4096);
        using var b = dev.AllocateDeviceLocal(4096);
        var errors = new List<Exception>();
        var copier = new Thread(() => { try { for (int i = 0; i < 20; i++) dev.CopyBufferSynchronous(a, b, 4096); } catch (Exception e) { lock (errors) errors.Add(e); } });
        var submitter = new Thread(() => { try { using var ctx = dev.CreateSubmitContext(); for (int i = 0; i < 20; i++) { ctx.Begin(); ctx.SubmitAndWait(); } } catch (Exception e) { lock (errors) errors.Add(e); } });
        copier.IsBackground = true; submitter.IsBackground = true;
        copier.Start(); submitter.Start();
        Assert.True(copier.Join(TimeSpan.FromSeconds(60)) && submitter.Join(TimeSpan.FromSeconds(60)), "a submitting thread did not finish (device deadlock)");
        Assert.Empty(errors);
        Assert.Equal(1, dev.MaxConcurrentQueueSubmits);
    }

    /// <summary>
    /// The forward pass arms <c>ActiveHazards</c> on its own thread; another thread's <c>Begin</c> / <c>SubmitAndWait</c> (an image encode) used to null the shared field, silently disarming the
    /// forward's tracking mid-recording. Now only the arming thread sees or clears its tracker.
    /// </summary>
    [SkippableFact]
    public void HazardTracker_ArmedOnOneThread_SurvivesAnotherThreadsSubmit_AndIsInvisibleToIt()
    {
        using var dev = Device();
        var tracker = new VulkanHazardTracker();
        using var armed = new ManualResetEventSlim(); using var otherDone = new ManualResetEventSlim();
        var errors = new List<Exception>();
        object? seenByOther = "unset"; bool survived = false;
        var forward = new Thread(() =>
        {
            try
            {
                dev.ActiveHazards = tracker;
                armed.Set();
                if (!otherDone.Wait(TimeSpan.FromSeconds(60))) throw new TimeoutException("the encode thread never finished");
                survived = ReferenceEquals(dev.ActiveHazards, tracker);
            }
            catch (Exception e) { lock (errors) errors.Add(e); }
            finally { armed.Set(); }
        });
        var encode = new Thread(() =>
        {
            try
            {
                if (!armed.Wait(TimeSpan.FromSeconds(60))) throw new TimeoutException("the forward thread never armed");
                using var ctx = dev.CreateSubmitContext();
                ctx.Begin();                           // disarms "its own" (none): must not touch the forward thread's tracker
                seenByOther = dev.ActiveHazards;
                ctx.SubmitAndWait();
            }
            catch (Exception e) { lock (errors) errors.Add(e); }
            finally { otherDone.Set(); }
        });
        forward.IsBackground = true; encode.IsBackground = true;
        forward.Start(); encode.Start();
        Assert.True(forward.Join(TimeSpan.FromSeconds(90)) && encode.Join(TimeSpan.FromSeconds(90)), "a hazard-tracker thread did not finish");
        Assert.Empty(errors);
        Assert.Null(seenByOther);
        Assert.True(survived, "another thread's Begin / SubmitAndWait disarmed this thread's hazard tracker");
    }

    /// <summary>
    /// Two threads arm their own trackers on one device (two models sharing it). Each must still see its own tracker afterwards: a device-wide last-writer-wins slot returned null to the
    /// first armer once the second armed, silently dropping its barriers mid-recording.
    /// </summary>
    [SkippableFact]
    public void HazardTracker_TwoThreadsArming_EachKeepsItsOwnTracker()
    {
        using var dev = Device();
        var ta = new VulkanHazardTracker(); var tb = new VulkanHazardTracker();
        using var aArmed = new ManualResetEventSlim(); using var bArmed = new ManualResetEventSlim();
        var errors = new List<Exception>();
        object? seenByA = "unset", seenByB = "unset";
        var a = new Thread(() =>
        {
            try
            {
                dev.ActiveHazards = ta;
                aArmed.Set();
                if (!bArmed.Wait(TimeSpan.FromSeconds(60))) throw new TimeoutException("thread B never armed");
                seenByA = dev.ActiveHazards;
            }
            catch (Exception e) { lock (errors) errors.Add(e); }
            finally { aArmed.Set(); }
        });
        var b = new Thread(() =>
        {
            try
            {
                if (!aArmed.Wait(TimeSpan.FromSeconds(60))) throw new TimeoutException("thread A never armed");
                dev.ActiveHazards = tb;
                seenByB = dev.ActiveHazards;
            }
            catch (Exception e) { lock (errors) errors.Add(e); }
            finally { bArmed.Set(); }
        });
        a.IsBackground = true; b.IsBackground = true;
        a.Start(); b.Start();
        Assert.True(a.Join(TimeSpan.FromSeconds(90)) && b.Join(TimeSpan.FromSeconds(90)), "a hazard-tracker thread did not finish");
        Assert.Empty(errors);
        Assert.Same(ta, seenByA);
        Assert.Same(tb, seenByB);
        Assert.Null(dev.ActiveHazards);   // and the test thread, which never armed, sees neither
    }

    /// <summary>The arming thread's own Begin / submit still disarms (an aborted recording must not leak tracking into the next one).</summary>
    [SkippableFact]
    public void HazardTracker_IsDisarmedByTheArmingThreadsOwnSubmit()
    {
        using var dev = Device();
        dev.ActiveHazards = new VulkanHazardTracker();
        using var ctx = dev.CreateSubmitContext();
        ctx.Begin();
        Assert.Null(dev.ActiveHazards);
        dev.ActiveHazards = new VulkanHazardTracker();
        ctx.SubmitAndWait();
        Assert.Null(dev.ActiveHazards);
    }
}
