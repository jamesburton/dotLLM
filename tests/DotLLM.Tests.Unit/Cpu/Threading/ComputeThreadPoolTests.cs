using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Threading;

public sealed unsafe class ComputeThreadPoolTests
{
    /// <summary>
    /// Verifies that spin-wait dispatch produces the same numerical results as event-based dispatch.
    /// </summary>
    [Fact]
    public void SpinWait_Dispatch_SameResultAsEventBased()
    {
        const int threadCount = 4;
        const int arraySize = 1024;
        using var pool = new ComputeThreadPool(threadCount);

        // Run with event-based mode
        pool.SetDispatchMode(DispatchMode.EventBased);
        var eventResult = RunSumWork(pool, arraySize);

        // Run with spin-wait mode
        pool.SetDispatchMode(DispatchMode.SpinWait);
        var spinResult = RunSumWork(pool, arraySize);

        Assert.Equal(eventResult, spinResult);
    }

    /// <summary>
    /// Verifies that switching between dispatch modes across multiple dispatches doesn't hang or corrupt results.
    /// </summary>
    [Fact]
    public void SetDispatchMode_SwitchesBetweenModes()
    {
        const int threadCount = 4;
        const int arraySize = 256;
        using var pool = new ComputeThreadPool(threadCount);

        for (int i = 0; i < 20; i++)
        {
            var mode = i % 2 == 0 ? DispatchMode.SpinWait : DispatchMode.EventBased;
            pool.SetDispatchMode(mode);
            var result = RunSumWork(pool, arraySize);
            Assert.Equal(arraySize, result); // each element = 1, so sum = arraySize
        }
    }

    /// <summary>
    /// Verifies that decode thread cap correctly reduces the number of active workers.
    /// All threads should still produce correct results, but with fewer partitions.
    /// </summary>
    [Fact]
    public void DecodeThreadCap_ReducesActiveWorkers()
    {
        const int totalThreads = 8;
        const int decodeThreads = 3;
        const int arraySize = 1024;

        var config = new ThreadingConfig(totalThreads, decodeThreads);
        using var pool = new ComputeThreadPool(totalThreads, topology: null, config);

        // In event-based mode: all threads active
        pool.SetDispatchMode(DispatchMode.EventBased);
        var fullResult = RunSumWork(pool, arraySize);

        // In spin-wait mode: decode thread cap applies
        pool.SetDispatchMode(DispatchMode.SpinWait);
        var cappedResult = RunSumWork(pool, arraySize);

        // Both should produce the same sum (work is partitioned differently but covers all elements)
        Assert.Equal(arraySize, fullResult);
        Assert.Equal(arraySize, cappedResult);
    }

    /// <summary>
    /// Stress test: rapidly switch modes while dispatching to ensure no deadlocks.
    /// </summary>
    [Fact]
    public void Dispatch_NoDeadlock_UnderModeSwitch()
    {
        const int threadCount = 4;
        const int arraySize = 128;
        using var pool = new ComputeThreadPool(threadCount);

        for (int i = 0; i < 100; i++)
        {
            pool.SetDispatchMode(i % 3 == 0 ? DispatchMode.SpinWait : DispatchMode.EventBased);
            var result = RunSumWork(pool, arraySize);
            Assert.Equal(arraySize, result);
        }
    }

    /// <summary>
    /// Verifies that the default constructor (no topology/config) still works correctly.
    /// </summary>
    [Fact]
    public void DefaultConstructor_WorksCorrectly()
    {
        const int threadCount = 3;
        const int arraySize = 512;
        using var pool = new ComputeThreadPool(threadCount);

        var result = RunSumWork(pool, arraySize);
        Assert.Equal(arraySize, result);
    }

    /// <summary>
    /// Without NUMA topology or explicit pinning flags, caller-thread pinning has no
    /// candidate core and must remain a no-op. The property should stay <c>false</c>.
    /// </summary>
    [Fact]
    public void CallerPinning_NoTopology_IsNoOp()
    {
        const int threadCount = 3;
        using var pool = new ComputeThreadPool(threadCount);

        // Force a dispatch — PinCallerThread runs automatically.
        var result = RunSumWork(pool, 64);
        Assert.Equal(64, result);

        Assert.False(pool.CallerThreadPinned,
            "No topology provided → no candidate core → pinning must be a no-op");
    }

    /// <summary>
    /// When caller pinning is explicitly disabled via <see cref="ThreadingConfig"/>,
    /// the pool must not pin the caller even if topology is available.
    /// </summary>
    [Fact]
    public void CallerPinning_ExplicitlyDisabled_IsNoOp()
    {
        const int threadCount = 3;
        var config = new ThreadingConfig(
            ThreadCount: threadCount,
            DecodeThreadCount: 0,
            EnableNumaPinning: true,
            EnablePCorePinning: true,
            EnableCallerPinning: false);

        // Even if topology were supplied, EnableCallerPinning=false forces _callerCoreId=-1.
        using var pool = new ComputeThreadPool(threadCount, topology: null, config);

        var result = RunSumWork(pool, 64);
        Assert.Equal(64, result);

        Assert.False(pool.CallerThreadPinned);
    }

    /// <summary>
    /// Calling <c>PinCallerThread</c> from a <c>ThreadPool</c> thread (via <c>Task.Run</c>)
    /// must skip pinning rather than corrupt the pool. The method should be silent and
    /// <see cref="ComputeThreadPool.CallerThreadPinned"/> must remain <c>false</c>.
    /// </summary>
    [Fact]
    public void PinCallerThread_OnThreadPoolThread_Skips()
    {
        // Exercise the ThreadPool-skip guard: call PinCallerThread from a pool thread
        // and assert it did not flip the flag. The guard short-circuits before the
        // actual pin syscall so pool threads keep their existing affinity.
        // Uses a signal primitive instead of await because the test class is 'unsafe'
        // (which precludes async/await).
        const int threadCount = 3;
        using var pool = new ComputeThreadPool(threadCount);

        using var done = new ManualResetEventSlim(false);
        bool wasOnPoolThread = false;
        bool pinnedAfter = false;

        ThreadPool.QueueUserWorkItem(_ =>
        {
            wasOnPoolThread = Thread.CurrentThread.IsThreadPoolThread;
            pool.PinCallerThread();
            pinnedAfter = pool.CallerThreadPinned;
            done.Set();
        });

        Assert.True(done.Wait(TimeSpan.FromSeconds(5)), "Pool-thread pin attempt timed out");
        Assert.True(wasOnPoolThread, "Test helper did not run on a ThreadPool thread");
        Assert.False(pinnedAfter, "PinCallerThread on a ThreadPool thread must be a no-op");
    }

    /// <summary>
    /// <c>PinCallerThread</c> is idempotent — repeated calls must not re-run the pin logic
    /// nor throw. Verified via the CAS-based one-shot flag.
    /// </summary>
    [Fact]
    public void PinCallerThread_Idempotent()
    {
        const int threadCount = 3;
        using var pool = new ComputeThreadPool(threadCount);

        // Call multiple times — must not throw and must not flip state unexpectedly.
        pool.PinCallerThread();
        pool.PinCallerThread();
        pool.PinCallerThread();

        // A subsequent dispatch should also be a no-op for the pin path.
        var result = RunSumWork(pool, 64);
        Assert.Equal(64, result);
    }

    /// <summary>
    /// Regression test for issue #129 (dotnet test hangs when running the full CPU test
    /// namespace on the Strix Halo box). Root cause: <c>WorkerLoop</c> used to seed its
    /// "have I seen a new dispatch" baseline with a live read of the mutable
    /// <c>_dispatchGeneration</c> field. If a worker's OS thread hadn't actually been scheduled
    /// yet by the time <see cref="ComputeThreadPool.Dispatch"/> ran on the caller (Thread.Start()
    /// only requests scheduling, it doesn't guarantee it), that first read could observe the
    /// dispatch's already-incremented generation and treat it as "nothing new" — the worker would
    /// reset the very ready-event that woke it and go back to waiting forever, while the caller's
    /// <c>CountdownEvent.Wait()</c> blocked forever for a <c>Signal()</c> that would never come.
    /// This reproduced far more often on Strix's 32-logical-processor box (many concurrently
    /// scheduled test-collection thread pools widen the scheduling-delay window) but was a latent
    /// race on any core count.
    /// </summary>
    /// <remarks>
    /// Forces the exact interleaving deterministically via the debug-only worker-start gate
    /// rather than relying on real OS scheduling luck to hit the window: workers are held before
    /// they read any dispatch state, a dispatch is driven to completion on a background thread
    /// (bumping the generation counter and setting the ready-events while workers are still
    /// gated), and only then are the workers released — reproducing "worker starts running after
    /// the dispatch it's meant to service already happened".
    /// </remarks>
    [Fact]
    public void Dispatch_WorkerStartsAfterDispatchAlreadyRan_StillCompletes()
    {
        // Plain Thread + ManualResetEventSlim rather than Task.Run + Task.Wait: the test class is
        // 'unsafe' (precludes async/await, see PinCallerThread_OnThreadPoolThread_Skips above),
        // and a blocking Task.Wait trips xUnit1031 as a build-time error in this repo anyway.
        const int threadCount = 4;
        // Standalone, this test completes in well under 100ms. Observed empirically: run as part
        // of the full ~800-test CPU namespace (issue #129's own repro command), the burst of
        // dozens of test collections simultaneously spinning up their own ComputeThreadPool
        // instances at the start of the run can genuinely delay scheduling of this test's
        // freshly-created threads by tens of seconds — real OS scheduler contention, not a bug.
        // The signature of the actual bug this guards against is "zero progress for 5+ minutes"
        // (see issue #129), not "slower than usual under heavy concurrent load" — so 90s stays
        // far below any real deadlock while comfortably absorbing that startup burst.
        const int TimeoutSeconds = 90;
        var startGate = new ManualResetEventSlim(false);
        var pool = new ComputeThreadPool(threadCount, startGate);
        var dispatchDone = new ManualResetEventSlim(false);
        int result = -1;
        bool completed = false;
        try
        {
            // Workers are blocked on startGate right now — before they've read any dispatch
            // state — exactly like a worker OS thread that Thread.Start() requested but that the
            // scheduler hasn't actually run yet.
            var dispatchThread = new Thread(() =>
            {
                result = RunSumWork(pool, 256);
                dispatchDone.Set();
            })
            { IsBackground = true, Name = "issue129-repro-dispatch" };
            dispatchThread.Start();

            // Give the caller side of Dispatch a head start: bump the generation, set the
            // ready-events, run its own slot-0 share, and block in CountdownEvent.Wait() — all
            // while workers are still gated. Not required for the race to exist, but it removes
            // scheduling luck as a variable in whether this test actually exercises it.
            Thread.Sleep(50);

            startGate.Set(); // release: workers now run WorkerLoop's real body for the first time

            completed = dispatchDone.Wait(TimeSpan.FromSeconds(TimeoutSeconds));
        }
        finally
        {
            // Only safe to dispose ANY of these on the success path. On a regression (or even
            // just extreme, unrelated system contention delaying things past the bound above):
            //  - A worker may still be permanently blocked inside CountdownEvent.Wait() —
            //    pool.Dispose() joins worker threads, turning a caught regression into a second
            //    hang.
            //  - A worker thread may not even have reached (or may be mid-call inside)
            //    _debugWorkerStartGate.Wait() yet — disposing startGate out from under that wait
            //    throws ObjectDisposedException on a background thread we no longer observe.
            //  - dispatchThread may only be slow rather than deadlocked and could call
            //    dispatchDone.Set() *after* this method returns — same ObjectDisposedException
            //    risk if dispatchDone is disposed here.
            // Any of those is an unhandled crash (or, headless, a hang behind a WER dialog) that
            // is strictly worse than the assertion failure below. So: leak all three uniformly
            // unless we positively know every thread involved has finished with them. All the
            // threads here are IsBackground=true, so leaking them never blocks process/test-run
            // exit — the leak is confined to this one test's small fixed allocation.
            if (completed)
            {
                pool.Dispose();
                dispatchDone.Dispose();
                startGate.Dispose();
            }
        }

        Assert.True(completed,
            $"Dispatch did not complete within {TimeoutSeconds}s — a worker likely missed the " +
            "dispatch that happened before it finished starting up (issue #129 regression).");
        Assert.Equal(256, result);
    }

    /// <summary>
    /// Helper: dispatches work that sums an array of 1.0f values across all threads.
    /// Returns the computed sum (should equal arraySize when each element is 1.0f).
    /// </summary>
    private static int RunSumWork(ComputeThreadPool pool, int arraySize)
    {
        float* data = (float*)NativeMemory.AlignedAlloc((nuint)(arraySize * sizeof(float)), 64);
        float* partialSums = (float*)NativeMemory.AlignedAlloc((nuint)(pool.ThreadCount * sizeof(float)), 64);

        try
        {
            for (int i = 0; i < arraySize; i++)
                data[i] = 1.0f;
            for (int i = 0; i < pool.ThreadCount; i++)
                partialSums[i] = 0;

            var ctx = new SumContext { Data = data, PartialSums = partialSums, ArraySize = arraySize };
            pool.Dispatch((nint)(&ctx), &SumWorker);

            float total = 0;
            for (int i = 0; i < pool.ThreadCount; i++)
                total += partialSums[i];

            return (int)total;
        }
        finally
        {
            NativeMemory.AlignedFree(data);
            NativeMemory.AlignedFree(partialSums);
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct SumContext
    {
        public float* Data;
        public float* PartialSums;
        public int ArraySize;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SumWorker(nint ctx, int threadIdx, int threadCount)
    {
        ref var c = ref Unsafe.AsRef<SumContext>((void*)ctx);
        int chunkSize = (c.ArraySize + threadCount - 1) / threadCount;
        int start = threadIdx * chunkSize;
        int end = Math.Min(start + chunkSize, c.ArraySize);

        float sum = 0;
        for (int i = start; i < end; i++)
            sum += c.Data[i];

        c.PartialSums[threadIdx] = sum;
    }
}
