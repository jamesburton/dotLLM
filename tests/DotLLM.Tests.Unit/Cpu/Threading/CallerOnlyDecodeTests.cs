using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Threading;

/// <summary>
/// An explicit <see cref="ThreadingConfig.DecodeThreadCount"/> of 1 must run SpinWait (decode) dispatches on the caller
/// alone, and switching back to EventBased (prefill) must restore every worker (#1272, upstream kkokosa/dotLLM#440).
/// </summary>
public sealed unsafe class CallerOnlyDecodeTests
{
    [Fact]
    public void ExplicitSingleThreadDecode_UsesOnlyCallerAndRestoresPrefillWorkers()
    {
        var config = new ThreadingConfig(4, DecodeThreadCount: 1);
        Assert.Equal(1, config.EffectiveDecodeThreadCount(4));
        using var pool = new ComputeThreadPool(4, topology: null, config);

        // Start workers before exercising mode transitions; first-dispatch startup
        // has its own independent regression fix.
        Assert.True(SpinWait.SpinUntil(() => WorkersAreWaiting(pool), TimeSpan.FromSeconds(5)));
        for (int iteration = 0; iteration < 20; iteration++)
        {
            AssertDispatch(pool, DispatchMode.EventBased, expectedThreads: 4);
            AssertDispatch(pool, DispatchMode.SpinWait, expectedThreads: 1);
        }
    }

    private static bool WorkersAreWaiting(ComputeThreadPool pool)
    {
        var field = typeof(ComputeThreadPool).GetField("_workers",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
        var workers = (Thread[])field.GetValue(pool)!;
        return workers.All(worker => (worker.ThreadState & ThreadState.WaitSleepJoin) == ThreadState.WaitSleepJoin);
    }

    private static void AssertDispatch(ComputeThreadPool pool, DispatchMode mode, int expectedThreads)
    {
        Span<int> counts = stackalloc int[4];
        counts.Clear();
        fixed (int* pointer = counts)
        {
            var context = new WorkContext { Counts = pointer, ExpectedThreads = expectedThreads };
            pool.SetDispatchMode(mode);
            pool.Dispatch((nint)(&context), &RecordPartition);
        }
        for (int i = 0; i < counts.Length; i++)
            Assert.Equal(i < expectedThreads ? 1 : 0, counts[i]);
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct WorkContext
    {
        public int* Counts;
        public int ExpectedThreads;
    }

    private static void RecordPartition(nint pointer, int threadIndex, int threadCount)
    {
        ref var context = ref Unsafe.AsRef<WorkContext>((void*)pointer);
        // Record both unexpected participants and an incorrect partition count.
        context.Counts[threadIndex] = threadCount == context.ExpectedThreads ? 1 : -1;
    }
}
