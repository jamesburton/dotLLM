using DotLLM.Tray.Hosting;

namespace DotLLM.Tray.Tests;

/// <summary>A process handle that never touches the OS.</summary>
internal sealed class FakeProcessHandle : IServerProcessHandle
{
    private bool _exited;

    internal FakeProcessHandle(int id) => Id = id;

    public int Id { get; }

    public bool HasExited => _exited;

    public int? ExitCode { get; private set; }

    public event EventHandler? Exited;

    /// <summary>Whether <see cref="Kill"/> was called.</summary>
    internal bool WasKilled { get; private set; }

    /// <summary>Whether <see cref="Dispose"/> was called.</summary>
    internal bool WasDisposed { get; private set; }

    public void Kill()
    {
        WasKilled = true;
        SimulateExit(1);
    }

    public Task<bool> WaitForExitAsync(TimeSpan timeout, CancellationToken ct = default) =>
        Task.FromResult(_exited);

    public void Dispose()
    {
        WasDisposed = true;
        Kill();
    }

    /// <summary>Simulates the process going away, raising <see cref="Exited"/> once.</summary>
    internal void SimulateExit(int exitCode)
    {
        if (_exited)
            return;
        _exited = true;
        ExitCode = exitCode;
        Exited?.Invoke(this, EventArgs.Empty);
    }
}

/// <summary>A runner that hands out <see cref="FakeProcessHandle"/>s and records the specs.</summary>
internal sealed class FakeProcessRunner : IServerProcessRunner
{
    private int _nextId = 1000;

    /// <summary>Every spec passed to <see cref="Start"/>, in order.</summary>
    internal List<ServerLaunchSpec> Specs { get; } = [];

    /// <summary>Every handle produced, in order.</summary>
    internal List<FakeProcessHandle> Handles { get; } = [];

    /// <summary>When set, <see cref="Start"/> throws this instead of starting.</summary>
    internal Exception? StartFailure { get; set; }

    public IServerProcessHandle Start(ServerLaunchSpec spec)
    {
        Specs.Add(spec);
        if (StartFailure is not null)
            throw StartFailure;
        var handle = new FakeProcessHandle(Interlocked.Increment(ref _nextId));
        Handles.Add(handle);
        return handle;
    }
}

/// <summary>A scriptable health probe.</summary>
internal sealed class FakeHealthProbe : IServerHealthProbe
{
    /// <summary>What <see cref="IsHealthyAsync"/> returns.</summary>
    internal bool Healthy { get; set; }

    /// <summary>What <see cref="IsReadyAsync"/> returns.</summary>
    internal bool Ready { get; set; }

    /// <summary>Number of health probes performed.</summary>
    internal int HealthProbes { get; private set; }

    /// <summary>When set, the probe flips <see cref="Healthy"/> to true after this many calls.</summary>
    internal int BecomeHealthyAfter { get; set; } = -1;

    /// <summary>Invoked on every health probe, so a test can perturb the world deterministically.</summary>
    internal Action<int>? OnProbe { get; set; }

    public Task<bool> IsHealthyAsync(CancellationToken ct)
    {
        HealthProbes++;
        OnProbe?.Invoke(HealthProbes);
        if (BecomeHealthyAfter >= 0 && HealthProbes > BecomeHealthyAfter)
            Healthy = true;
        return Task.FromResult(Healthy);
    }

    public Task<bool> IsReadyAsync(CancellationToken ct) => Task.FromResult(Ready);
}
