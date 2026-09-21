using DotLLM.Tray.Hosting;
using Xunit;

namespace DotLLM.Tray.Tests;

/// <summary>
/// Covers the tray's lifecycle state machine — the part of #455's acceptance criteria that says
/// killing the tray must never orphan a server, and vice versa.
/// </summary>
/// <remarks>
/// <b>Discrimination (#417), by mutation.</b> Mutants applied, observed, and reverted:
/// <list type="number">
///   <item>
///     <b>Died.</b> Neutering the pre-spawn <c>IsHealthyAsync</c> check in <c>StartAsync</c> (so
///     it always spawns) failed three tests:
///     <see cref="Start_AttachesToAnAlreadyRunningServerInsteadOfSpawningASecond"/>,
///     <see cref="Stop_RefusesToKillAnAttachedServerItDoesNotOwn"/> and
///     <see cref="Restart_OfAnAttachedServer_ChangesNothing"/> — the attach behaviour is guarded
///     from three angles.
///   </item>
///   <item>
///     <b>Survived, and the survivor is recorded rather than papered over.</b> Deleting the
///     explicit <c>handle.Kill()</c> from the start-timeout path alone changed nothing, because
///     <c>ReleaseChild</c> disposes the handle and disposal kills. The call is redundant
///     belt-and-braces, not an unguarded invariant.
///   </item>
///   <item>
///     <b>Died.</b> Removing <i>both</i> reap paths (the <c>handle.Kill()</c> above and
///     <c>child.Dispose()</c> in <c>ReleaseChild</c>) failed
///     <see cref="Start_ThatTimesOut_KillsTheHalfStartedChildRatherThanLeakingIt"/> and
///     <see cref="Dispose_KillsAnOwnedChild"/>. So the orphan property itself <i>is</i> guarded;
///     what mutant 2 showed is that it is guarded twice over.
///   </item>
/// </list>
/// Note what these tests cannot reach: the kernel-level half of the guarantee lives in
/// <see cref="WindowsJobObject"/> and is verified only by the manual kill-the-tray test in
/// <c>docs/TRAY.md</c>.
/// </remarks>
public sealed class ServerSupervisorTests
{
    private static ServerLaunchSpec Spec() => new(@"C:\dotllm\dotllm.exe", ["serve", "--allow-model-admin"]);

    private static ServerSupervisor Create(
        FakeProcessRunner runner, FakeHealthProbe probe, TimeSpan? startTimeout = null) =>
        new(runner, probe, Spec,
            startTimeout ?? TimeSpan.FromMilliseconds(200),
            TimeSpan.FromMilliseconds(1),
            // No real sleeping: the poll loop's pacing is not what is under test.
            (_, _) => Task.CompletedTask);

    [Fact]
    public async Task Start_AttachesToAnAlreadyRunningServerInsteadOfSpawningASecond()
    {
        // The acceptance criterion "must handle 'already running'". Spawning anyway would produce
        // a process that fails to bind the port and dies, and a tray that believes it owns it.
        var runner = new FakeProcessRunner();
        var probe = new FakeHealthProbe { Healthy = true, Ready = true };
        using var supervisor = Create(runner, probe);

        var status = await supervisor.StartAsync(CancellationToken.None);

        Assert.Equal(ServerState.RunningAttached, status.State);
        Assert.True(status.IsRunning);
        Assert.False(status.IsOwned);
        Assert.Empty(runner.Specs);
        Assert.NotNull(status.Detail);
    }

    [Fact]
    public async Task Start_SpawnsWhenNothingIsAnswering_AndBecomesOwnedOnceHealthy()
    {
        var runner = new FakeProcessRunner();
        var probe = new FakeHealthProbe { Healthy = false, BecomeHealthyAfter = 2 };
        using var supervisor = Create(runner, probe);

        var status = await supervisor.StartAsync(CancellationToken.None);

        Assert.Equal(ServerState.RunningOwned, status.State);
        Assert.True(status.IsOwned);
        var spec = Assert.Single(runner.Specs);
        Assert.Contains("--allow-model-admin", spec.Arguments);
        Assert.Equal(Assert.Single(runner.Handles).Id, status.ProcessId);
    }

    [Fact]
    public async Task Start_ThatTimesOut_KillsTheHalfStartedChildRatherThanLeakingIt()
    {
        var runner = new FakeProcessRunner();
        var probe = new FakeHealthProbe { Healthy = false };
        using var supervisor = Create(runner, probe, TimeSpan.FromMilliseconds(20));

        var status = await supervisor.StartAsync(CancellationToken.None);

        Assert.Equal(ServerState.Failed, status.State);
        Assert.True(Assert.Single(runner.Handles).WasKilled);
        Assert.Contains("did not answer /health", status.Detail!, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Start_WhenTheChildDiesDuringStartup_ReportsFailedWithTheExitCode()
    {
        var runner = new FakeProcessRunner();
        var probe = new FakeHealthProbe { Healthy = false };
        using var supervisor = Create(runner, probe, TimeSpan.FromSeconds(5));

        // On the second probe — the first one inside the wait loop — the child dies, simulating a
        // port-bind failure. Deterministic: no racing against a background task.
        probe.OnProbe = count =>
        {
            if (count == 2)
                runner.Handles[0].SimulateExit(70);
        };

        var status = await supervisor.StartAsync(CancellationToken.None);

        Assert.Equal(ServerState.Failed, status.State);
        Assert.Contains("exited", status.Detail!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Start_WhenTheRunnerThrows_ReportsFailedRatherThanPropagating()
    {
        // A missing dotllm.exe is a normal, user-fixable condition. Throwing out of a tray menu
        // click would take the tray down with it.
        var runner = new FakeProcessRunner { StartFailure = new FileNotFoundException("dotllm.exe not found") };
        var probe = new FakeHealthProbe { Healthy = false };
        using var supervisor = Create(runner, probe);

        var status = await supervisor.StartAsync(CancellationToken.None);

        Assert.Equal(ServerState.Failed, status.State);
        Assert.Contains("dotllm.exe", status.Detail!, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Stop_KillsAnOwnedChild()
    {
        var runner = new FakeProcessRunner();
        var probe = new FakeHealthProbe { Healthy = false, BecomeHealthyAfter = 1 };
        using var supervisor = Create(runner, probe);
        await supervisor.StartAsync(CancellationToken.None);

        probe.Healthy = false;
        var status = await supervisor.StopAsync(CancellationToken.None);

        Assert.Equal(ServerState.Stopped, status.State);
        Assert.True(Assert.Single(runner.Handles).WasKilled);
    }

    [Fact]
    public async Task Stop_RefusesToKillAnAttachedServerItDoesNotOwn()
    {
        // #454 exposes no shutdown route, and the tray will NOT go looking for the listening pid.
        // The honest outcome is to leave it running and say why.
        var runner = new FakeProcessRunner();
        var probe = new FakeHealthProbe { Healthy = true, Ready = true };
        using var supervisor = Create(runner, probe);
        await supervisor.StartAsync(CancellationToken.None);

        var status = await supervisor.StopAsync(CancellationToken.None);

        Assert.Equal(ServerState.RunningAttached, status.State);
        Assert.Empty(runner.Handles);
        Assert.Contains("will not", status.Detail!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task AnOwnedChildDyingOnItsOwn_MovesTheTrayToFailed()
    {
        // The converse orphan: the server dies, the tray must not keep showing green.
        var runner = new FakeProcessRunner();
        var probe = new FakeHealthProbe { Healthy = false, BecomeHealthyAfter = 1 };
        using var supervisor = Create(runner, probe);
        await supervisor.StartAsync(CancellationToken.None);
        Assert.Equal(ServerState.RunningOwned, supervisor.Status.State);

        runner.Handles[0].SimulateExit(139);

        Assert.Equal(ServerState.Failed, supervisor.Status.State);
        Assert.Contains("exited unexpectedly", supervisor.Status.Detail!, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Dispose_KillsAnOwnedChild()
    {
        var runner = new FakeProcessRunner();
        var probe = new FakeHealthProbe { Healthy = false, BecomeHealthyAfter = 1 };
        var supervisor = Create(runner, probe);
        await supervisor.StartAsync(CancellationToken.None);

        supervisor.Dispose();

        Assert.True(Assert.Single(runner.Handles).WasDisposed);
        Assert.True(runner.Handles[0].WasKilled);
    }

    [Fact]
    public async Task Restart_StopsThenStartsAFreshChild()
    {
        var runner = new FakeProcessRunner();
        var probe = new FakeHealthProbe { Healthy = false, BecomeHealthyAfter = 1 };
        using var supervisor = Create(runner, probe);
        await supervisor.StartAsync(CancellationToken.None);

        // The restarted server is down until its own spawn answers: false for the stop-path
        // refresh and the pre-spawn probe, true thereafter.
        probe.Healthy = false;
        probe.BecomeHealthyAfter = probe.HealthProbes + 2;
        var status = await supervisor.RestartAsync(CancellationToken.None);

        Assert.Equal(ServerState.RunningOwned, status.State);
        Assert.Equal(2, runner.Handles.Count);
        Assert.True(runner.Handles[0].WasKilled);
        Assert.False(runner.Handles[1].WasKilled);
    }

    [Fact]
    public async Task Restart_OfAnAttachedServer_ChangesNothing()
    {
        var runner = new FakeProcessRunner();
        var probe = new FakeHealthProbe { Healthy = true };
        using var supervisor = Create(runner, probe);
        await supervisor.StartAsync(CancellationToken.None);

        var status = await supervisor.RestartAsync(CancellationToken.None);

        Assert.Equal(ServerState.RunningAttached, status.State);
        Assert.Empty(runner.Handles);
    }

    [Fact]
    public async Task Refresh_HealthyWithoutAModel_IsRunningNotFailed()
    {
        // `dotllm serve` starts happily with no model. /health ok + /ready 503 is normal.
        var runner = new FakeProcessRunner();
        var probe = new FakeHealthProbe { Healthy = true, Ready = false };
        using var supervisor = Create(runner, probe);

        var status = await supervisor.RefreshAsync(CancellationToken.None);

        Assert.True(status.IsRunning);
        Assert.False(status.IsModelLoaded);
    }

    [Fact]
    public async Task StatusChanged_FiresOnTransitionsOnly()
    {
        var runner = new FakeProcessRunner();
        var probe = new FakeHealthProbe { Healthy = true, Ready = true };
        using var supervisor = Create(runner, probe);

        var seen = new List<ServerState>();
        supervisor.StatusChanged += (_, status) => seen.Add(status.State);

        await supervisor.RefreshAsync(CancellationToken.None);
        await supervisor.RefreshAsync(CancellationToken.None);
        await supervisor.RefreshAsync(CancellationToken.None);

        // Three identical refreshes; the tray icon must not be told to redraw three times.
        Assert.Equal([ServerState.RunningAttached], seen);
    }

    [Fact]
    public async Task StartTwice_DoesNotSpawnASecondChild()
    {
        var runner = new FakeProcessRunner();
        var probe = new FakeHealthProbe { Healthy = false, BecomeHealthyAfter = 1 };
        using var supervisor = Create(runner, probe);

        await supervisor.StartAsync(CancellationToken.None);
        await supervisor.StartAsync(CancellationToken.None);

        Assert.Single(runner.Handles);
    }
}
