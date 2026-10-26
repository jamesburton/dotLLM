using System.Threading.Tasks;
using DotLLM.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

/// <summary>
/// #1291: the repeat guard ends a scheduled request as <see cref="FinishReason.Length"/> on the N-th identical token in a
/// row, not on N-1, and never when disabled (0). Partial over <c>ContinuousBatchSchedulerTests</c> for its fixture.
/// </summary>
public sealed partial class ContinuousBatchSchedulerTests
{
    private static InferenceRequest GuardedRequest(int maxTokens, int guard)
    {
        var r = MakeRequest(promptLen: 2, maxTokens: maxTokens);
        return r with { Options = r.Options with { RepeatGuardLimit = guard } };
    }

    [Fact]
    public async Task RepeatGuard_TripsOnTheNthIdenticalToken()
    {
        using var fix = new TestFixture(tokenScript: TokenScript.Constant(9));

        var handle = fix.Scheduler.Submit(GuardedRequest(maxTokens: 10, guard: 4));
        DriveUntilIdle(fix.Scheduler);
        var response = await handle.Completion;

        Assert.Equal(FinishReason.Length, response.FinishReason);
        Assert.Equal(4, response.GeneratedTokenCount);
    }

    [Fact]
    public async Task RepeatGuard_DoesNotTripOnRunsOfNMinusOne()
    {
        // Runs of three 9s broken by a 5: never four in a row.
        using var fix = new TestFixture(tokenScript: new TokenScript(step => step % 4 == 3 ? 5 : 9));

        var handle = fix.Scheduler.Submit(GuardedRequest(maxTokens: 10, guard: 4));
        DriveUntilIdle(fix.Scheduler);
        var response = await handle.Completion;

        Assert.Equal(10, response.GeneratedTokenCount);
    }

    [Fact]
    public async Task RepeatGuard_ZeroDisables()
    {
        using var fix = new TestFixture(tokenScript: TokenScript.Constant(9));

        var handle = fix.Scheduler.Submit(GuardedRequest(maxTokens: 10, guard: 0));
        DriveUntilIdle(fix.Scheduler);
        var response = await handle.Completion;

        Assert.Equal(FinishReason.Length, response.FinishReason);
        Assert.Equal(10, response.GeneratedTokenCount);
    }
}
