namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Occupies the thread pool's worker threads for the lifetime of the returned handle, the way a busy CI runner does when many
/// test classes run in parallel (xUnit runs classes concurrently, and a test that blocks its own thread makes it worse). Code that
/// hands a timed action to <see cref="Task.Run(Action)"/> and then waits on wall-clock time must still be correct when that
/// action starts seconds late: tests use this to prove they do not depend on pool timing.
/// </summary>
internal static class ThreadPoolStarver
{
    public static IDisposable Starve()
    {
        ThreadPool.GetMinThreads(out int workers, out _);
        var gate = new ManualResetEventSlim(false);
        using var started = new CountdownEvent(workers);
        // More blocking tasks than the pool's minimum: the pool adds a thread only about every 500 ms beyond the minimum.
        for (int i = 0; i < workers * 2 + 4; i++)
            ThreadPool.UnsafeQueueUserWorkItem(_ =>
            {
                if (!started.IsSet) { try { started.Signal(); } catch (InvalidOperationException) { } }
                gate.Wait();
            }, null);
        started.Wait(TimeSpan.FromSeconds(10));
        return new Release(gate);
    }

    private sealed class Release(ManualResetEventSlim gate) : IDisposable
    {
        public void Dispose() => gate.Set();
    }
}
