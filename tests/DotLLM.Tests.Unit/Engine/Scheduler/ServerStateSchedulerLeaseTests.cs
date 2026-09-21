using System;
using System.Threading;
using System.Threading.Tasks;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

/// <summary>
/// #461, follow-up 1: the scheduler lease must be taken by <see cref="ServerState.ExecuteAsync"/>
/// itself, not by each caller remembering to.
/// </summary>
/// <remarks>
/// <para>
/// The request gate does not protect the model. A running scheduler drives forwards on the same
/// model from its background loop, deliberately outside that gate, and the model's scratch buffers
/// are shared mutable state — a direct-generator forward beside a scheduler step tears down the
/// shared <c>ComputeThreadPool</c> and the <b>process dies</b>. #451 fixed that from the embeddings
/// side; the defect was never embeddings-specific.
/// </para>
/// <para>
/// The exposure is wider than the issue records: <b>streaming chat and completions always take the
/// direct-generator path</b>, so every streaming request overlapping a batched one was racing the
/// loop. It only bites under a mixed workload — an idle loop performs no forward — which is why it
/// first surfaced in a test that deliberately overlapped two endpoints, rather than in ordinary use.
/// </para>
/// <para>
/// Declared <c>partial</c> over <c>ContinuousBatchSchedulerServiceTests</c> to reuse its mock model
/// and service fixture rather than standing up a second, subtly different one.
/// </para>
/// </remarks>
public sealed partial class ContinuousBatchSchedulerServiceTests
{
    /// <summary>
    /// While <see cref="ServerState.ExecuteAsync"/> is running work, the scheduler's model lease
    /// is unavailable — which is exactly what stops its run loop entering a step. Against the
    /// pre-fix <c>ExecuteAsync</c> (request gate only) the lease is free throughout, and this
    /// fails on the first assertion.
    /// </summary>
    [Fact]
    public async Task ExecuteAsync_HoldsTheSchedulerModelLease_ForTheDurationOfTheWork()
    {
        using var fix = new ServiceFixture(emitToken: 5, afterNTokens: 1);
        using var state = new ServerState
        {
            Options = new ServerOptions { Model = "test" },
            Scheduler = fix.Service,
        };

        var workStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var releaseWork = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        Task executing = state.ExecuteAsync(async () =>
        {
            workStarted.SetResult();
            await releaseWork.Task;
        }, CancellationToken.None);

        await workStarted.Task;

        // The lease must NOT be grantable while the work is in flight. A timeout is the only way
        // to assert a negative here; it is generous enough not to be flaky and short enough not to
        // stall the suite.
        using (var probe = new CancellationTokenSource(TimeSpan.FromMilliseconds(500)))
        {
            await Assert.ThrowsAnyAsync<OperationCanceledException>(
                async () => (await fix.Service.AcquireModelAsync(probe.Token)).Dispose());
        }

        releaseWork.SetResult();
        await executing;

        // ...and must be grantable immediately once the work completes, so the fix serialises
        // rather than deadlocking. Without this half, a lease that was never released would also
        // satisfy the assertion above.
        using var after = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        using var lease = await fix.Service.AcquireModelAsync(after.Token);
        Assert.NotNull(lease);
    }

    /// <summary>
    /// The lease is released on a failure path too. A work delegate that throws must not leave the
    /// scheduler permanently unable to step — that would turn one failed request into a server
    /// that never generates again.
    /// </summary>
    [Fact]
    public async Task ExecuteAsync_ReleasesTheLease_WhenTheWorkThrows()
    {
        using var fix = new ServiceFixture(emitToken: 5, afterNTokens: 1);
        using var state = new ServerState
        {
            Options = new ServerOptions { Model = "test" },
            Scheduler = fix.Service,
        };

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            state.ExecuteAsync(() => throw new InvalidOperationException("boom"), CancellationToken.None));

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        using var lease = await fix.Service.AcquireModelAsync(cts.Token);
        Assert.NotNull(lease);
    }

    /// <summary>
    /// With no scheduler there is nothing to serialise against, and <c>ExecuteAsync</c> must not
    /// acquire or require anything extra — the direct-generator-only configuration (quantized KV,
    /// hybrid/CUDA models) is the common one and must keep working unchanged.
    /// </summary>
    [Fact]
    public async Task ExecuteAsync_WithoutAScheduler_StillRunsTheWork()
    {
        using var state = new ServerState { Options = new ServerOptions { Model = "test" } };

        bool ran = false;
        await state.ExecuteAsync(() => { ran = true; return Task.CompletedTask; }, CancellationToken.None);

        Assert.True(ran);
    }
}
