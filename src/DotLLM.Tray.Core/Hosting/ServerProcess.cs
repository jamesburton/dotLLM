using System.Diagnostics;
using System.Runtime.Versioning;
using System.Text;

namespace DotLLM.Tray.Hosting;

/// <summary>
/// Everything needed to launch a <c>dotllm serve</c> child process.
/// </summary>
/// <param name="ExecutablePath">Absolute path of the dotLLM executable.</param>
/// <param name="Arguments">Argument list, already split (never a single joined string).</param>
/// <param name="WorkingDirectory">Working directory for the child, or null for the tray's own.</param>
/// <param name="LogFilePath">File the child's stdout/stderr are drained to, or null to discard.</param>
public sealed record ServerLaunchSpec(
    string ExecutablePath,
    IReadOnlyList<string> Arguments,
    string? WorkingDirectory = null,
    string? LogFilePath = null);

/// <summary>A launched server process the tray owns.</summary>
public interface IServerProcessHandle : IDisposable
{
    /// <summary>OS process id.</summary>
    int Id { get; }

    /// <summary>Whether the process has exited.</summary>
    bool HasExited { get; }

    /// <summary>Exit code once exited, otherwise null.</summary>
    int? ExitCode { get; }

    /// <summary>Raised when the process exits, however it exits.</summary>
    event EventHandler? Exited;

    /// <summary>Terminates the process. Idempotent; a no-op once exited.</summary>
    void Kill();

    /// <summary>Waits for exit. Returns false on timeout.</summary>
    /// <param name="timeout">How long to wait.</param>
    /// <param name="ct">Cancellation token.</param>
    Task<bool> WaitForExitAsync(TimeSpan timeout, CancellationToken ct = default);
}

/// <summary>Launches server processes. Abstracted so the supervisor is testable without spawning.</summary>
public interface IServerProcessRunner
{
    /// <summary>Starts a process for the given spec.</summary>
    /// <param name="spec">What to launch.</param>
    IServerProcessHandle Start(ServerLaunchSpec spec);
}

/// <summary>
/// Real <see cref="IServerProcessRunner"/>: <see cref="Process"/> plus a drained log.
/// </summary>
/// <remarks>
/// <para>
/// Orphan prevention is <b>not</b> done here. It is done once, at tray startup, by
/// <see cref="WindowsJobObject.AssignCurrentProcess"/>: the tray itself joins a kill-on-close job
/// object, and every process it starts inherits that membership. Assigning each child after
/// <see cref="Process.Start()"/> would leave a window in which a child exists outside the job, and
/// closing that window needs <c>CREATE_SUSPENDED</c>, which <see cref="Process"/> cannot express.
/// </para>
/// <para>
/// stdout/stderr are redirected and drained asynchronously to a log file. Redirecting without
/// draining deadlocks the child once its pipe buffer fills (~4 KB), which for a server that logs
/// per request means it wedges shortly after the first few requests.
/// </para>
/// </remarks>
[SupportedOSPlatform("windows")]
public sealed class ProcessServerProcessRunner : IServerProcessRunner
{
    /// <inheritdoc />
    public IServerProcessHandle Start(ServerLaunchSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);

        var info = new ProcessStartInfo
        {
            FileName = spec.ExecutablePath,
            WorkingDirectory = spec.WorkingDirectory ?? Path.GetDirectoryName(spec.ExecutablePath) ?? "",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };

        foreach (var argument in spec.Arguments)
            info.ArgumentList.Add(argument);

        var process = new Process { StartInfo = info, EnableRaisingEvents = true };
        try
        {
            if (!process.Start())
                throw new InvalidOperationException($"Failed to start '{spec.ExecutablePath}'.");
        }
        catch
        {
            process.Dispose();
            throw;
        }

        return new ProcessHandle(process, spec.LogFilePath);
    }

    private sealed class ProcessHandle : IServerProcessHandle
    {
        private readonly Process _process;
        private readonly StreamWriter? _log;
        private readonly object _logLock = new();
        private int _disposed;

        internal ProcessHandle(Process process, string? logFilePath)
        {
            _process = process;
            _process.Exited += OnExited;

            if (logFilePath is not null)
            {
                try
                {
                    var directory = Path.GetDirectoryName(logFilePath);
                    if (!string.IsNullOrEmpty(directory))
                        Directory.CreateDirectory(directory);
                    _log = new StreamWriter(
                        new FileStream(logFilePath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite))
                    {
                        AutoFlush = true,
                    };
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    // A log we cannot open must not stop the server from running. The pipes are
                    // still drained below, which is the part that matters for correctness.
                    _log = null;
                }
            }

            _process.OutputDataReceived += (_, e) => Write(e.Data);
            _process.ErrorDataReceived += (_, e) => Write(e.Data);
            _process.BeginOutputReadLine();
            _process.BeginErrorReadLine();
        }

        public int Id => _process.Id;

        public bool HasExited
        {
            get
            {
                try
                {
                    return _process.HasExited;
                }
                catch (InvalidOperationException)
                {
                    return true;
                }
            }
        }

        public int? ExitCode
        {
            get
            {
                try
                {
                    return _process.HasExited ? _process.ExitCode : null;
                }
                catch (InvalidOperationException)
                {
                    return null;
                }
            }
        }

        public event EventHandler? Exited;

        public void Kill()
        {
            try
            {
                if (!_process.HasExited)
                    _process.Kill(entireProcessTree: true);
            }
            catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception)
            {
                // Already gone, or exiting as we asked. Either way there is nothing to kill.
            }
        }

        public async Task<bool> WaitForExitAsync(TimeSpan timeout, CancellationToken ct = default)
        {
            using var timeoutSource = CancellationTokenSource.CreateLinkedTokenSource(ct);
            timeoutSource.CancelAfter(timeout);
            try
            {
                await _process.WaitForExitAsync(timeoutSource.Token).ConfigureAwait(false);
                return true;
            }
            catch (OperationCanceledException) when (!ct.IsCancellationRequested)
            {
                return false;
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;

            _process.Exited -= OnExited;
            Kill();
            _process.Dispose();
            lock (_logLock)
                _log?.Dispose();
        }

        private void OnExited(object? sender, EventArgs e) => Exited?.Invoke(this, EventArgs.Empty);

        private void Write(string? line)
        {
            if (line is null)
                return;
            lock (_logLock)
            {
                if (_disposed != 0)
                    return;
                try
                {
                    _log?.WriteLine(line);
                }
                catch (Exception ex) when (ex is IOException or ObjectDisposedException)
                {
                    // Losing a log line must never take the server down.
                }
            }
        }
    }
}
