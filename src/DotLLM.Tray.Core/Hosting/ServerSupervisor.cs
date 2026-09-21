namespace DotLLM.Tray.Hosting;

/// <summary>Live state of the server the tray is pointed at.</summary>
public enum ServerState
{
    /// <summary>Not probed yet.</summary>
    Unknown,

    /// <summary>Nothing is answering and the tray owns no process.</summary>
    Stopped,

    /// <summary>A child has been spawned; waiting for it to answer <c>/health</c>.</summary>
    Starting,

    /// <summary>Healthy, and the tray started it — the tray may stop it.</summary>
    RunningOwned,

    /// <summary>Healthy, but something else started it — the tray must not stop it.</summary>
    RunningAttached,

    /// <summary>An owned child is being terminated.</summary>
    Stopping,

    /// <summary>A start attempt failed, or an owned child exited unexpectedly.</summary>
    Failed,
}

/// <summary>A state change, for the tray icon and menu.</summary>
/// <param name="State">The new state.</param>
/// <param name="IsModelLoaded">Whether <c>/ready</c> reports a model is loaded.</param>
/// <param name="ProcessId">The owned child's pid, when there is one.</param>
/// <param name="Detail">Human-readable explanation, for a failure or an attach.</param>
public sealed record ServerStatus(
    ServerState State,
    bool IsModelLoaded = false,
    int? ProcessId = null,
    string? Detail = null)
{
    /// <summary>Whether the server is answering requests, however it was started.</summary>
    public bool IsRunning => State is ServerState.RunningOwned or ServerState.RunningAttached;

    /// <summary>Whether the tray started this server and is therefore entitled to stop it.</summary>
    public bool IsOwned => State is ServerState.RunningOwned or ServerState.Starting or ServerState.Stopping;
}

/// <summary>Answers "is a server answering at the tray's base address?".</summary>
public interface IServerHealthProbe
{
    /// <summary>True when <c>GET /health</c> succeeds.</summary>
    /// <param name="ct">Cancellation token.</param>
    Task<bool> IsHealthyAsync(CancellationToken ct);

    /// <summary>True when <c>GET /ready</c> succeeds — the server has a model loaded.</summary>
    /// <param name="ct">Cancellation token.</param>
    Task<bool> IsReadyAsync(CancellationToken ct);
}

/// <summary>
/// The tray's server lifecycle state machine: attach-or-spawn, stop, restart, and exit-detection.
/// </summary>
/// <remarks>
/// <para>
/// <b>Ownership is the central idea.</b> The tray never assumes it started the server it is
/// talking to. <see cref="StartAsync"/> probes <c>/health</c> first and, if something answers,
/// <i>attaches</i> — it does not spawn a second server that would fail to bind the port and leave
/// a confusing dead process behind. An attached server is never stopped by the tray: the tray did
/// not start it, has no shutdown endpoint to ask it politely (see the class remarks on
/// <see cref="StopAsync"/>), and killing a process it does not own would be exactly the
/// reach-around the tray's client-only contract forbids.
/// </para>
/// <para>
/// The other half of the orphan guarantee lives in <see cref="WindowsJobObject"/>: the tray
/// joining a kill-on-close job makes "tray dies → server dies" a kernel invariant rather than
/// something this class has to get right on every exit path. What this class adds is the converse
/// — "server dies → tray notices" — via <see cref="IServerProcessHandle.Exited"/>, so a crashed
/// backend shows as <see cref="ServerState.Failed"/> instead of a tray that still claims green.
/// </para>
/// </remarks>
public sealed class ServerSupervisor : IDisposable
{
    private readonly IServerProcessRunner _runner;
    private readonly IServerHealthProbe _probe;
    private readonly Func<ServerLaunchSpec> _specFactory;
    private readonly TimeSpan _startTimeout;
    private readonly TimeSpan _pollInterval;
    private readonly Func<TimeSpan, CancellationToken, Task> _delay;
    private readonly SemaphoreSlim _gate = new(1, 1);

    private IServerProcessHandle? _child;
    private ServerStatus _status = new(ServerState.Unknown);
    private int _disposed;

    /// <summary>Creates a supervisor.</summary>
    /// <param name="runner">Spawns server processes.</param>
    /// <param name="probe">Answers health/readiness questions.</param>
    /// <param name="specFactory">Produces the launch spec at start time, so settings edits take effect on the next start.</param>
    /// <param name="startTimeout">How long to wait for a spawned child to answer <c>/health</c>.</param>
    /// <param name="pollInterval">Gap between health polls while starting.</param>
    /// <param name="delay">Delay function, injectable so tests do not sleep in real time.</param>
    public ServerSupervisor(
        IServerProcessRunner runner,
        IServerHealthProbe probe,
        Func<ServerLaunchSpec> specFactory,
        TimeSpan? startTimeout = null,
        TimeSpan? pollInterval = null,
        Func<TimeSpan, CancellationToken, Task>? delay = null)
    {
        _runner = runner ?? throw new ArgumentNullException(nameof(runner));
        _probe = probe ?? throw new ArgumentNullException(nameof(probe));
        _specFactory = specFactory ?? throw new ArgumentNullException(nameof(specFactory));
        _startTimeout = startTimeout ?? TimeSpan.FromSeconds(60);
        _pollInterval = pollInterval ?? TimeSpan.FromMilliseconds(250);
        _delay = delay ?? Task.Delay;
    }

    /// <summary>Raised whenever the status changes.</summary>
    public event EventHandler<ServerStatus>? StatusChanged;

    /// <summary>The current status.</summary>
    public ServerStatus Status => Volatile.Read(ref _status);

    /// <summary>
    /// Re-probes and updates the status without starting or stopping anything.
    /// </summary>
    /// <remarks>
    /// Distinguishes healthy-with-a-model from healthy-but-bare: the server may be started without
    /// a model and load one later, so <c>/health</c> ok with <c>/ready</c> 503 is a normal running
    /// state, not a failure.
    /// </remarks>
    /// <param name="ct">Cancellation token.</param>
    public async Task<ServerStatus> RefreshAsync(CancellationToken ct = default)
    {
        var healthy = await _probe.IsHealthyAsync(ct).ConfigureAwait(false);
        var child = _child;
        var childAlive = child is { HasExited: false };

        if (!healthy)
        {
            // An owned child that is alive but not yet answering is still starting, not stopped.
            if (childAlive && Status.State == ServerState.Starting)
                return Status;

            return Publish(childAlive
                ? new ServerStatus(ServerState.Starting, ProcessId: child!.Id)
                : new ServerStatus(ServerState.Stopped));
        }

        var ready = await _probe.IsReadyAsync(ct).ConfigureAwait(false);
        return Publish(childAlive
            ? new ServerStatus(ServerState.RunningOwned, ready, child!.Id)
            : new ServerStatus(
                ServerState.RunningAttached, ready,
                Detail: "Attached to a server this tray did not start."));
    }

    /// <summary>
    /// Attaches to a running server, or starts one.
    /// </summary>
    /// <remarks>
    /// Probing first is what makes "a server is already running" a supported case rather than a
    /// port-bind crash. The probe is the same <c>/health</c> check the #454 test harness uses to
    /// refuse to start over a live server.
    /// </remarks>
    /// <param name="ct">Cancellation token.</param>
    public async Task<ServerStatus> StartAsync(CancellationToken ct = default)
    {
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (_child is { HasExited: false })
                return await RefreshAsync(ct).ConfigureAwait(false);

            if (await _probe.IsHealthyAsync(ct).ConfigureAwait(false))
            {
                var ready = await _probe.IsReadyAsync(ct).ConfigureAwait(false);
                return Publish(new ServerStatus(
                    ServerState.RunningAttached, ready,
                    Detail: "A server was already answering at this address; attached to it "
                          + "instead of starting a second one."));
            }

            ReleaseChild();

            IServerProcessHandle handle;
            try
            {
                handle = _runner.Start(_specFactory());
            }
            catch (Exception ex)
            {
                return Publish(new ServerStatus(ServerState.Failed, Detail: ex.Message));
            }

            _child = handle;
            handle.Exited += OnChildExited;
            Publish(new ServerStatus(ServerState.Starting, ProcessId: handle.Id));

            return await WaitForHealthyAsync(handle, ct).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>
    /// Stops the server, but only when the tray owns it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For an attached server this returns the current status unchanged with a <see cref="ServerStatus.Detail"/>
    /// explaining why. The tray deliberately does <b>not</b> find the listening pid and kill it:
    /// that process may be a developer's <c>dotnet run</c>, and the tray has no mandate over it.
    /// </para>
    /// <para>
    /// <b>Known gap.</b> Even for an owned child, "stop" is a hard kill, because #454 exposes no
    /// shutdown route. In-flight generations are cut; in-flight <i>downloads</i> are not lost
    /// (they resume from their <c>.incomplete</c> file). A gated <c>POST /v1/admin/shutdown</c>
    /// that drains behind the request gate — the same sequence <c>UnloadAsync</c> already uses —
    /// would close this, and is reported as the change needed in #454.
    /// </para>
    /// </remarks>
    /// <param name="ct">Cancellation token.</param>
    public async Task<ServerStatus> StopAsync(CancellationToken ct = default)
    {
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            var child = _child;
            if (child is null || child.HasExited)
            {
                ReleaseChild();
                var status = await RefreshAsync(ct).ConfigureAwait(false);
                return status.State == ServerState.RunningAttached
                    ? Publish(status with
                    {
                        Detail = "This server was started outside the tray, so the tray will not "
                               + "stop it. Stop it where it was started.",
                    })
                    : status;
            }

            Publish(new ServerStatus(ServerState.Stopping, ProcessId: child.Id));
            child.Kill();
            await child.WaitForExitAsync(TimeSpan.FromSeconds(10), ct).ConfigureAwait(false);
            ReleaseChild();
            return Publish(new ServerStatus(ServerState.Stopped));
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>
    /// Stops an owned server and starts a fresh one, picking up any launch-option edits.
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    public async Task<ServerStatus> RestartAsync(CancellationToken ct = default)
    {
        var stopped = await StopAsync(ct).ConfigureAwait(false);
        if (stopped.State == ServerState.RunningAttached)
            return stopped;
        return await StartAsync(ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Kills an owned child, if any. Belt-and-braces alongside the job object, which is what
    /// actually guarantees no orphan when the tray does not get to run this.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        ReleaseChild();
        _gate.Dispose();
    }

    private async Task<ServerStatus> WaitForHealthyAsync(IServerProcessHandle handle, CancellationToken ct)
    {
        var deadline = DateTime.UtcNow + _startTimeout;
        while (DateTime.UtcNow < deadline)
        {
            if (handle.HasExited)
            {
                ReleaseChild();
                return Publish(new ServerStatus(
                    ServerState.Failed,
                    Detail: $"The server exited during startup (exit code {handle.ExitCode?.ToString() ?? "unknown"})."));
            }

            if (await _probe.IsHealthyAsync(ct).ConfigureAwait(false))
            {
                var ready = await _probe.IsReadyAsync(ct).ConfigureAwait(false);
                return Publish(new ServerStatus(ServerState.RunningOwned, ready, handle.Id));
            }

            await _delay(_pollInterval, ct).ConfigureAwait(false);
        }

        // Timed out. The child is killed rather than left running: a half-started server the tray
        // has given up tracking is exactly the orphan this class exists to prevent.
        handle.Kill();
        ReleaseChild();
        return Publish(new ServerStatus(
            ServerState.Failed,
            Detail: $"The server did not answer /health within {_startTimeout.TotalSeconds:0} seconds."));
    }

    private void OnChildExited(object? sender, EventArgs e)
    {
        if (Volatile.Read(ref _disposed) != 0)
            return;
        if (Status.State is ServerState.Stopping or ServerState.Stopped)
            return;
        Publish(new ServerStatus(ServerState.Failed, Detail: "The server process exited unexpectedly."));
    }

    private void ReleaseChild()
    {
        var child = Interlocked.Exchange(ref _child, null);
        if (child is null)
            return;
        child.Exited -= OnChildExited;
        child.Dispose();
    }

    private ServerStatus Publish(ServerStatus status)
    {
        var previous = Interlocked.Exchange(ref _status, status);
        if (previous != status)
            StatusChanged?.Invoke(this, status);
        return status;
    }
}
