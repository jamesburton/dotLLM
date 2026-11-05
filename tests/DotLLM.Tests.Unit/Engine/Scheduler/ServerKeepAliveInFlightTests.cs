using System;
using System.Threading;
using System.Threading.Tasks;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

/// <summary>
/// #1606: the keep-alive idle sweep must never unload the model under a running generation, on ANY request path, and keep-alive
/// counts from the moment the last request FINISHED. Hermetic: fake model, manual clock, no sleeping, no real weights.
/// Scheduler paths (no gate) and gated paths (<c>ExecuteAsync</c> x2, <c>ExecuteOnModelAsync</c>) are each held mid-flight,
/// the clock is moved far past the keep-alive, and the sweep is run.
/// </summary>
public sealed partial class ContinuousBatchSchedulerServiceTests
{
    private const double KeepAliveSeconds = 10;
    private static readonly TimeSpan PastKeepAlive = TimeSpan.FromSeconds(KeepAliveSeconds * 5);

    private sealed class ManualClock : TimeProvider
    {
        private DateTimeOffset _now = new(2026, 11, 4, 12, 0, 0, TimeSpan.Zero);
        public override DateTimeOffset GetUtcNow() => _now;
        public void Advance(TimeSpan by) => _now += by;
    }

    private static ServerState NewServerState(ServiceFixture fix, ManualClock? clock = null)
    {
        clock ??= new ManualClock();
        var state = new ServerState
        {
            Options = new ServerOptions
            {
                Model = "", ModelId = "fake", Device = "cpu", UsePaged = false, PromptCacheEnabled = false, Warmup = WarmupOptions.Disabled,
            },
            Residency = new ModelResidencyManager { MaxResidentModels = 1, DefaultKeepAliveSeconds = KeepAliveSeconds },
            Clock = clock,
            Model = fix.Model,
            Tokenizer = fix.Tokenizer,
            Generator = new TextGenerator(fix.Model, fix.Tokenizer),
            Scheduler = fix.Service,
            IsReady = true,
        };
        state.LastUsedUtc = clock.GetUtcNow();
        return state;
    }

    // ---- scheduler (gate-free) paths -------------------------------------------------------------------------------

    [Fact]
    public async Task Sweep_SchedulerNonStreaming_NotUnloadedMidFlight_UnloadedAfterFinishPlusKeepAlive()
    {
        using var fix = new ServiceFixture(emitToken: EosTokenId, afterNTokens: 3);
        var clock = new ManualClock();
        using var state = NewServerState(fix, clock);

        // No run loop yet: the request is genuinely in flight and cannot finish.
        var pending = state.EnqueueTrackedAsync(fix.Service, MakeRequest(promptLen: 3, maxTokens: 8), CancellationToken.None);
        Assert.Equal(1, state.InFlightRequests);

        clock.Advance(PastKeepAlive);
        await state.SweepIdleAsync();
        Assert.True(state.IsReady);
        Assert.NotNull(state.Model);

        using var loopCts = new CancellationTokenSource();
        state.SchedulerLoopCts = loopCts;   // as in the server: the sweep's unload stops the loop it owns
        state.SchedulerLoopTask = Task.Run(() => fix.Service.RunLoopAsync(loopCts.Token));
        await pending.WaitAsync(Deadline);
        Assert.Equal(0, state.InFlightRequests);

        // Keep-alive counts from completion: right after finishing, still resident...
        await state.SweepIdleAsync();
        Assert.True(state.IsReady);
        // ...and unloaded once a full keep-alive has passed since the finish.
        clock.Advance(PastKeepAlive);
        await state.SweepIdleAsync();
        Assert.False(state.IsReady);
        Assert.Null(state.Model);
    }

    [Fact]
    public async Task Sweep_SchedulerStreaming_NotUnloadedMidFlight_UnloadedAfterFinishPlusKeepAlive()
    {
        using var fix = new ServiceFixture(emitToken: EosTokenId, afterNTokens: 3);
        var clock = new ManualClock();
        using var state = NewServerState(fix, clock);
        using var loopCts = new CancellationTokenSource();
        state.SchedulerLoopCts = loopCts;   // as in the server: the sweep's unload stops the loop it owns
        state.SchedulerLoopTask = Task.Run(() => fix.Service.RunLoopAsync(loopCts.Token));

        var request = MakeRequest(promptLen: 3, maxTokens: 8);
        var source = ServerState.SchedulerStream(state, fix.Service, new DefaultHttpContext(), request.TokenIds, request.Options);
        await using var stream = source(CancellationToken.None).GetAsyncEnumerator();
        Assert.True(await stream.MoveNextAsync().AsTask().WaitAsync(Deadline));   // mid-stream, consumer paused
        Assert.Equal(1, state.InFlightRequests);

        clock.Advance(PastKeepAlive);
        await state.SweepIdleAsync().WaitAsync(Deadline);
        Assert.True(state.IsReady);

        while (await stream.MoveNextAsync().AsTask().WaitAsync(Deadline)) { }
        await stream.DisposeAsync();
        Assert.Equal(0, state.InFlightRequests);

        await state.SweepIdleAsync().WaitAsync(Deadline);
        Assert.True(state.IsReady);
        clock.Advance(PastKeepAlive);
        await state.SweepIdleAsync().WaitAsync(Deadline);
        Assert.False(state.IsReady);
    }

    [Fact]
    public async Task SchedulerStream_AbandonedOrCancelled_ReleasesTheInFlightCount()
    {
        using var fix = new ServiceFixture(emitToken: 5, afterNTokens: 1000);
        using var state = NewServerState(fix);
        using var loopCts = new CancellationTokenSource();
        var loopTask = Task.Run(() => fix.Service.RunLoopAsync(loopCts.Token));
        var request = MakeRequest(promptLen: 3, maxTokens: 50);

        // Abandoned after the first token.
        await foreach (var _ in ServerState.SchedulerStream(state, fix.Service, new DefaultHttpContext(), request.TokenIds, request.Options)(CancellationToken.None))
            break;
        Assert.Equal(0, state.InFlightRequests);

        // Cancelled mid-stream.
        using var cts = new CancellationTokenSource();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var _ in ServerState.SchedulerStream(state, fix.Service, new DefaultHttpContext(), request.TokenIds, request.Options)(cts.Token))
                cts.Cancel();
        });
        Assert.Equal(0, state.InFlightRequests);

        loopCts.Cancel();
        await loopTask;
    }

    [Fact]
    public async Task EnqueueTracked_CancelledRequest_ReleasesTheInFlightCount_AndStampsLastUsed()
    {
        using var fix = new ServiceFixture(emitToken: EosTokenId, afterNTokens: 3);
        var clock = new ManualClock();
        using var state = NewServerState(fix, clock);
        var start = state.LastUsedUtc;

        using var cts = new CancellationTokenSource();
        var pending = state.EnqueueTrackedAsync(fix.Service, MakeRequest(3, 8), cts.Token);   // no loop: stays pending
        Assert.Equal(1, state.InFlightRequests);
        clock.Advance(TimeSpan.FromSeconds(7));
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending);

        Assert.Equal(0, state.InFlightRequests);
        Assert.Equal(start + TimeSpan.FromSeconds(7), state.LastUsedUtc);
    }

    // ---- gated (direct-generator) paths ----------------------------------------------------------------------------

    public static TheoryData<string> GatedPaths => new() { "Execute", "ExecuteForModel", "ExecuteOnModel" };

    private static Task RunGated(ServerState state, string path, Func<Task> work) => path switch
    {
        "Execute" => state.ExecuteAsync(work, CancellationToken.None),
        "ExecuteForModel" => state.ExecuteAsync("fake", _ => work(), CancellationToken.None),
        _ => state.ExecuteOnModelAsync("fake", work, CancellationToken.None),
    };

    [Theory]
    [MemberData(nameof(GatedPaths))]
    public async Task Sweep_GatedPath_NotUnloadedMidFlight_UnloadedAfterFinishPlusKeepAlive(string path)
    {
        using var fix = new ServiceFixture(emitToken: EosTokenId, afterNTokens: 3);
        var clock = new ManualClock();
        using var state = NewServerState(fix, clock);
        state.Scheduler = null;   // the direct-generator path
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        Task running = RunGated(state, path, () => { entered.SetResult(); return release.Task; });
        await entered.Task.WaitAsync(Deadline);
        Assert.Equal(1, state.InFlightRequests);

        clock.Advance(PastKeepAlive);
        await state.SweepIdleAsync();
        Assert.True(state.IsReady);

        release.SetResult();
        await running.WaitAsync(Deadline);
        Assert.Equal(0, state.InFlightRequests);

        // The generation outlived the keep-alive; the stamp is the FINISH time, so the model is not unloaded straight away.
        await state.SweepIdleAsync();
        Assert.True(state.IsReady);
        clock.Advance(PastKeepAlive);
        await state.SweepIdleAsync();
        Assert.False(state.IsReady);
    }

    [Theory]
    [MemberData(nameof(GatedPaths))]
    public async Task GatedPath_ThrowingWork_StillReleasesInFlightCount(string path)
    {
        using var fix = new ServiceFixture(emitToken: EosTokenId, afterNTokens: 3);
        using var state = NewServerState(fix);
        state.Scheduler = null;

        await Assert.ThrowsAsync<InvalidOperationException>(() => RunGated(state, path, () => throw new InvalidOperationException("boom")));
        Assert.Equal(0, state.InFlightRequests);
    }

    // ---- requests vs the gate / unload -----------------------------------------------------------------------------

    [Fact]
    public async Task RequestQueuedBehindARunningOne_WaitsForTheGate_AndOnlyTheRunningOneCountsAsInFlight()
    {
        using var fix = new ServiceFixture(emitToken: EosTokenId, afterNTokens: 3);
        using var state = NewServerState(fix, new ManualClock());
        state.Scheduler = null;

        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var first = state.ExecuteAsync(() => { entered.SetResult(); return release.Task; }, CancellationToken.None);
        await entered.Task.WaitAsync(Deadline);

        bool secondRan = false;
        var second = state.ExecuteAsync(() => { secondRan = true; return Task.CompletedTask; }, CancellationToken.None);
        await Task.Delay(50);
        Assert.False(secondRan);
        Assert.Equal(1, state.InFlightRequests);

        release.SetResult();
        await Task.WhenAll(first, second).WaitAsync(Deadline);
        Assert.True(secondRan);
        Assert.Equal(0, state.InFlightRequests);
    }

    [Fact]
    public async Task RequestArrivingAfterTheSweepUnloaded_NeverRunsOnTheDisposedModel()
    {
        using var fix = new ServiceFixture(emitToken: EosTokenId, afterNTokens: 3);
        var clock = new ManualClock();
        using var state = NewServerState(fix, clock);
        state.Scheduler = null;
        clock.Advance(PastKeepAlive);
        await state.SweepIdleAsync();
        Assert.False(state.IsReady);

        // The request waits for the (finished) unload, finds the model gone, and fails cleanly (there is no path to reload
        // from in this fake) instead of touching freed state; its work never runs and nothing is left counted in flight.
        bool ran = false;
        await Assert.ThrowsAsync<ModelUnavailableException>(() =>
            state.ExecuteAsync("fake", _ => { ran = true; return Task.CompletedTask; }, CancellationToken.None));
        Assert.False(ran);
        Assert.Equal(0, state.InFlightRequests);
    }

    // ---- unchanged behaviour ---------------------------------------------------------------------------------------

    [Fact]
    public async Task Sweep_IdleModel_StillUnloads_AndNegativeKeepAliveNeverDoes()
    {
        using var fix = new ServiceFixture(emitToken: EosTokenId, afterNTokens: 3);
        var clock = new ManualClock();
        using var state = NewServerState(fix, clock);

        await state.SweepIdleAsync();
        Assert.True(state.IsReady);                  // not yet expired
        clock.Advance(PastKeepAlive);
        state.KeepAliveSecondsOverride = -1;
        await state.SweepIdleAsync();
        Assert.True(state.IsReady);                  // never expires
        state.KeepAliveSecondsOverride = null;
        await state.SweepIdleAsync();
        Assert.False(state.IsReady);                 // idle past keep-alive: unloaded
    }
}
