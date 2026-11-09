using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Backstop for tests that prove "the call ended although its work is still blocked". Such a test holds the work on a gate that
/// only the test releases, so a correct implementation returns whatever the machine load, and a broken one (a timeout that never
/// fires, a thread joined without a limit) never returns. The test then fails here instead of hanging the run. The limit is far
/// above any scheduling delay a loaded runner produces (the slowest seen was 23 s) and is only ever reached by a regression, so it
/// is not a threshold on the behaviour under test.
/// </summary>
internal static class HangGuard
{
    /// <summary>How long a call that must return may take before the test calls it hung.</summary>
    public static readonly TimeSpan Limit = TimeSpan.FromMinutes(2);

    /// <summary>Awaits <paramref name="task"/>, failing the test when it has not finished after <see cref="Limit"/>.</summary>
    public static async Task<T> ReturnsAsync<T>(Task<T> task, string what)
    {
        using var stop = new CancellationTokenSource();
        Task finished = await Task.WhenAny(task, Task.Delay(Limit, stop.Token));
        if (!ReferenceEquals(finished, task))
            Assert.Fail($"{what} did not return within {Limit.TotalSeconds:F0} s although its work is held; the call is waiting for the work.");
        await stop.CancelAsync();
        return await task;
    }
}
