using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

public sealed partial class ContinuousBatchSchedulerTests
{
    /// <summary>
    /// #1892: a model with a hard per-forward limit (Vulkan in-place TurboQuant) is never handed a prompt longer than it. The scheduler
    /// used to forward every admitted prompt in one pass and ignore the model's preference, so a long prompt reached the model's
    /// own guard. With the limit it prefills in passes of at most the limit and still completes.
    /// </summary>
    [Fact]
    public async Task Prefill_LongerThanTheModelsLimit_IsSplitAtTheLimit()
    {
        using var fix = new TestFixture(tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 100));
        fix.Model.MaxPrefillForwardTokens = 3;
        var handle = fix.Scheduler.Submit(MakeRequest(promptLen: 10, maxTokens: 4));
        DriveUntilIdle(fix.Scheduler);

        var response = await handle.Completion;
        Assert.Equal(4, response.GeneratedTokenCount);
        var prefill = fix.Model.ForwardLengths.Take(4).ToArray();
        Assert.Equal(new[] { 3, 3, 3, 1 }, prefill);                       // 10 prompt tokens in passes of at most 3
        Assert.All(fix.Model.ForwardLengths, n => Assert.True(n <= 3, $"a forward of {n} rows exceeded the limit"));
    }

    [Fact]
    public async Task Prefill_WithNoLimit_StaysOnePass()
    {
        using var fix = new TestFixture(tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 100));
        var handle = fix.Scheduler.Submit(MakeRequest(promptLen: 10, maxTokens: 2));
        DriveUntilIdle(fix.Scheduler);
        await handle.Completion;
        Assert.Equal(10, fix.Model.ForwardLengths[0]);
    }

    /// <summary>Two prompts in one step would take the fused <c>ForwardBatch</c> path, which hands each the whole suffix: past the limit it must not.</summary>
    [Fact]
    public async Task TwoPrompts_OverTheLimit_DoNotTakeTheFusedBatch()
    {
        using var fix = new TestFixture(tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 100));
        fix.Model.MaxPrefillForwardTokens = 4;
        var a = fix.Scheduler.Submit(MakeRequest(promptLen: 9, maxTokens: 2));
        var b = fix.Scheduler.Submit(MakeRequest(promptLen: 7, maxTokens: 2));
        DriveUntilIdle(fix.Scheduler);
        await a.Completion;
        await b.Completion;
        Assert.All(fix.Model.ForwardLengths, n => Assert.True(n <= 4, $"a forward of {n} rows exceeded the limit"));
    }
}
