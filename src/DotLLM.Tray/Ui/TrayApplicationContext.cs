using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Globalization;
using System.Runtime.Versioning;
using System.Windows.Forms;
using DotLLM.Tray.Api;
using DotLLM.Tray.Autostart;
using DotLLM.Tray.Config;
using DotLLM.Tray.Hosting;
using DotLLM.Tray.Updates;

namespace DotLLM.Tray.Ui;

/// <summary>
/// The tray icon and its menu — the whole visible surface of the application.
/// </summary>
/// <remarks>
/// <para>
/// This class is deliberately thin. Every decision it makes (what to launch, whether to attach,
/// whether autostart is on, whether a release is newer, what the server said) belongs to a
/// testable service in <c>DotLLM.Tray.Core</c>; what lives here is menu construction, marshalling
/// to the UI thread, and turning exceptions into message boxes. That split is why the suite in
/// <c>tests/DotLLM.Tray.Tests</c> can cover the behaviour at all — see <c>docs/TRAY.md</c> for
/// what remains manual-only.
/// </para>
/// <para>
/// Every action is an HTTP call. The tray references neither DotLLM.Engine nor DotLLM.Server, so
/// there is no in-process path to reach for even by accident.
/// </para>
/// </remarks>
[SupportedOSPlatform("windows")]
internal sealed class TrayApplicationContext : ApplicationContext
{
    private readonly TraySettingsStore _settingsStore = new(TraySettingsStore.DefaultPath);
    private readonly TrayIcons _icons = new();
    private readonly NotifyIcon _notifyIcon;
    private readonly System.Windows.Forms.Timer _refreshTimer;
    private readonly AutostartManager _autostart = new(new RegistryAutostartStore());
    private readonly bool _orphanProtection;
    private readonly HttpClient _updateHttp;

    private TraySettings _settings;
    private HttpClient _apiHttp;
    private DotLlmApiClient _api;
    private ServerSupervisor _supervisor;
    private int _refreshing;
    private bool _disposed;

    /// <summary>Creates the tray context and shows the icon.</summary>
    /// <param name="orphanProtection">
    /// Whether the kill-on-close job object was established. Surfaced in the UI rather than
    /// assumed, so a user on a locked-down machine learns that a hard-killed tray could leave the
    /// server running.
    /// </param>
    internal TrayApplicationContext(bool orphanProtection)
    {
        _orphanProtection = orphanProtection;
        _settings = _settingsStore.Load();

        _updateHttp = new HttpClient();
        // GitHub rejects requests without one.
        _updateHttp.DefaultRequestHeaders.UserAgent.ParseAdd("dotllm-tray");

        (_apiHttp, _api, _supervisor) = BuildStack(_settings);

        _notifyIcon = new NotifyIcon
        {
            Icon = _icons.For(ServerState.Unknown),
            Text = "dotLLM",
            Visible = true,
            ContextMenuStrip = new ContextMenuStrip(),
        };
        _notifyIcon.ContextMenuStrip.Opening += (_, _) => RebuildMenu();
        _notifyIcon.DoubleClick += (_, _) => OpenWebUi();

        // A poll rather than a push: the server has no event channel, and /health is cheap.
        // 3 s keeps "time to auto-unload" readable without hammering a loaded server.
        _refreshTimer = new System.Windows.Forms.Timer { Interval = 3000 };
        _refreshTimer.Tick += async (_, _) => await RefreshAsync().ConfigureAwait(true);
        _refreshTimer.Start();

        _ = InitializeAsync();
    }

    private (HttpClient Http, DotLlmApiClient Api, ServerSupervisor Supervisor) BuildStack(TraySettings settings)
    {
        // Ten minutes, because POST /v1/models/unload legitimately blocks behind an in-flight
        // generation and POST /v1/models/load blocks for as long as a model takes to load. The
        // health probes below deliberately do NOT inherit it.
        var http = new HttpClient { BaseAddress = settings.BaseAddress, Timeout = TimeSpan.FromMinutes(10) };
        var api = new DotLlmApiClient(http);
        var supervisor = new ServerSupervisor(
            new ProcessServerProcessRunner(),
            new ApiHealthProbe(api),
            () => BuildLaunchSpec(settings),
            // `dotllm serve` loads the model BEFORE Kestrel starts listening (ServeCommand calls
            // ServerStartup.LoadModel, then app.RunAsync), so with a startup model /health does
            // not answer until the load and its warm-up passes finish. A 60 s budget would kill a
            // legitimately loading 27B model and report it as a hang. A dead child is caught by
            // the exit check, not by this timeout, so a generous budget costs nothing.
            startTimeout: string.IsNullOrWhiteSpace(settings.StartupModel)
                ? TimeSpan.FromSeconds(60)
                : TimeSpan.FromMinutes(15));
        supervisor.StatusChanged += (_, status) => BeginInvokeOnUi(() => ApplyStatus(status));
        return (http, api, supervisor);
    }

    private static ServerLaunchSpec BuildLaunchSpec(TraySettings settings)
    {
        var executable = DotLlmExecutableLocator.Locate(settings.ExecutablePath)
            ?? throw new FileNotFoundException(
                "Could not find dotllm.exe. Set its path in tray Settings, put it beside "
                + "dotllm-tray.exe, or add it to PATH.");

        var logDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "dotLLM", "tray");

        return new ServerLaunchSpec(
            executable,
            ServerLaunchSpecBuilder.BuildArguments(settings.ToLaunchOptions()),
            WorkingDirectory: null,
            LogFilePath: Path.Combine(
                logDirectory,
                "server-" + DateTime.Now.ToString("yyyyMMdd", CultureInfo.InvariantCulture) + ".log"));
    }

    private async Task InitializeAsync()
    {
        if (_settings.StartServerOnLaunch)
            await StartServerAsync().ConfigureAwait(true);
        else
            await RefreshAsync().ConfigureAwait(true);

        if (_settings.CheckForUpdates)
            await CheckForUpdatesAsync(announceWhenCurrent: false).ConfigureAwait(true);
    }

    private async Task RefreshAsync()
    {
        // One refresh at a time. The probes are budgeted to 3 s, but a slow machine can still
        // overlap a 3 s tick, and overlapping refreshes publish out of order.
        if (Interlocked.Exchange(ref _refreshing, 1) != 0)
            return;

        try
        {
            await _supervisor.RefreshAsync().ConfigureAwait(true);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            // A refresh that cannot reach the server is information, not a crash; the next tick
            // will try again and ApplyStatus already renders the disconnected state.
        }
        finally
        {
            Volatile.Write(ref _refreshing, 0);
        }
    }

    private void ApplyStatus(ServerStatus status)
    {
        if (_disposed)
            return;
        _notifyIcon.Icon = _icons.For(status.State);
        var text = "dotLLM — " + TrayIcons.Describe(status);
        // The shell truncates NotifyIcon.Text at 127 characters and throws above 63 on some
        // older shells; keep it comfortably short.
        _notifyIcon.Text = text.Length > 120 ? text[..120] : text;
    }

    // ────────────────────────────────── menu ──────────────────────────────────

    private void RebuildMenu()
    {
        var menu = _notifyIcon.ContextMenuStrip!;
        menu.Items.Clear();

        var status = _supervisor.Status;

        menu.Items.Add(new ToolStripMenuItem(TrayIcons.Describe(status)) { Enabled = false });
        if (status.Detail is { Length: > 0 } detail)
            menu.Items.Add(new ToolStripMenuItem(Ellipsize(detail, 70)) { Enabled = false });
        if (!_orphanProtection)
        {
            menu.Items.Add(new ToolStripMenuItem("⚠ Orphan protection unavailable on this machine")
            {
                Enabled = false,
            });
        }

        menu.Items.Add(new ToolStripSeparator());

        menu.Items.Add(Item("&Start server", !status.IsRunning, async () => await StartServerAsync().ConfigureAwait(true)));

        // Stop/Restart are enabled only for a server the tray owns. #454 has no shutdown route,
        // and the tray will not go hunting for someone else's pid — see ServerSupervisor.StopAsync.
        var owned = status.State == ServerState.RunningOwned;
        var stop = Item("Sto&p server", owned, async () => await _supervisor.StopAsync().ConfigureAwait(true));
        var restart = Item("&Restart server", owned, async () => await _supervisor.RestartAsync().ConfigureAwait(true));
        if (status.State == ServerState.RunningAttached)
        {
            stop.ToolTipText = restart.ToolTipText =
                "This server was started outside the tray. Stop it where it was started.";
        }

        menu.Items.Add(stop);
        menu.Items.Add(restart);

        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add(Item("Open &Web UI", status.IsRunning, OpenWebUi));
        menu.Items.Add(Item("&Copy base URL", true, CopyBaseUrl));

        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add(Item("&Models…", status.IsRunning, () => ShowDialog(new ModelsForm(_api))));
        menu.Items.Add(Item("Se&ttings…", true, ShowSettings));

        menu.Items.Add(new ToolStripSeparator());
        var autostart = new ToolStripMenuItem("Start dotLLM at &login")
        {
            // The registry is the source of truth, re-read on every menu open: the user may have
            // removed the entry from Task Manager since the tray started.
            Checked = _autostart.IsEnabled(),
            CheckOnClick = false,
        };
        autostart.Click += (_, _) => ToggleAutostart();
        menu.Items.Add(autostart);

        menu.Items.Add(Item(
            "Check for &updates…", true,
            async () => await CheckForUpdatesAsync(announceWhenCurrent: true).ConfigureAwait(true)));

        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add(Item("E&xit", true, ExitTray));
    }

    private static ToolStripMenuItem Item(string text, bool enabled, Action onClick)
    {
        var item = new ToolStripMenuItem(text) { Enabled = enabled };
        item.Click += (_, _) => onClick();
        return item;
    }

    private static ToolStripMenuItem Item(string text, bool enabled, Func<Task> onClick)
    {
        var item = new ToolStripMenuItem(text) { Enabled = enabled };
        item.Click += async (_, _) => await onClick().ConfigureAwait(true);
        return item;
    }

    // ───────────────────────────────── actions ────────────────────────────────

    private async Task StartServerAsync()
    {
        var status = await _supervisor.StartAsync().ConfigureAwait(true);
        if (status.State == ServerState.Failed)
            Warn("Could not start the dotLLM server.\n\n" + status.Detail);
    }

    private void OpenWebUi() => OpenUrl(_settings.BaseAddress.ToString());

    private void CopyBaseUrl()
    {
        try
        {
            Clipboard.SetText(_settings.BaseAddress.ToString());
        }
        catch (ExternalException)
        {
            // Another process can hold the clipboard open; not worth an error dialog.
        }
    }

    private void ShowSettings()
    {
        using var form = new SettingsForm(_settings, _api, _supervisor.Status.IsRunning);
        if (form.ShowDialog() != DialogResult.OK)
            return;

        var updated = form.Result.Normalized();
        var addressChanged = updated.BaseAddress != _settings.BaseAddress;

        _settings = updated;
        _settingsStore.Save(_settings);

        if (!addressChanged)
            return;

        // The tray is now pointed somewhere else, so the old client and supervisor are stale.
        // The previously-owned child is stopped first: leaving it running while the tray watches
        // a different port is precisely the orphan this app promises not to create.
        var old = _supervisor;
        var oldHttp = _apiHttp;
        old.Dispose();
        oldHttp.Dispose();
        (_apiHttp, _api, _supervisor) = BuildStack(_settings);
        _ = RefreshAsync();
    }

    private void ToggleAutostart()
    {
        try
        {
            if (_autostart.IsEnabled())
            {
                _autostart.Disable();
            }
            else
            {
                var path = Environment.ProcessPath;
                if (string.IsNullOrEmpty(path))
                {
                    Warn("Could not determine this application's path, so autostart cannot be enabled.");
                    return;
                }

                // Under `dotnet dotllm-tray.dll` the host process IS dotnet.exe, so registering
                // ProcessPath would put a bare `"...\dotnet.exe"` in the Run key — a logon entry
                // that launches the SDK and exits. The shipped build is a single-file exe, so this
                // only bites a developer running from source; refuse rather than write a Run value
                // that silently does nothing.
                if (Path.GetFileName(path).Equals("dotnet.exe", StringComparison.OrdinalIgnoreCase))
                {
                    Warn("Autostart needs the published dotllm-tray.exe. This instance is hosted by "
                       + "dotnet.exe, so registering it would create a startup entry that does nothing.");
                    return;
                }

                _autostart.Enable(path);
            }

            _settings = _settings with { AutostartLastKnown = _autostart.IsEnabled() };
            _settingsStore.Save(_settings);
        }
        catch (Exception ex) when (ex is UnauthorizedAccessException or System.Security.SecurityException)
        {
            Warn("Could not change the autostart setting:\n\n" + ex.Message);
        }
    }

    private async Task CheckForUpdatesAsync(bool announceWhenCurrent)
    {
        var current = UpdateChecker.CurrentVersion(typeof(TrayApplicationContext).Assembly);
        var result = await new UpdateChecker(_updateHttp)
            .CheckAsync(current, _settings.IncludePrereleases)
            .ConfigureAwait(true);

        if (!result.UpdateAvailable)
        {
            if (announceWhenCurrent)
                Inform($"dotLLM {current} is up to date.");
            return;
        }

        // Check, changelog, and a link. The tray does NOT download or swap the executable:
        // the release archives are unsigned, so a self-applied update would hand the user a
        // binary SmartScreen flags and could leave a half-replaced install if Defender
        // quarantines it mid-swap. See docs/TRAY.md.
        using var form = new UpdateForm(result);
        if (form.ShowDialog() == DialogResult.OK && result.ReleaseUrl is { Length: > 0 } url)
            OpenUrl(url);
    }

    private void ExitTray()
    {
        // Stopping first is the ordinary, polite path; the job object is what covers the paths
        // this method never gets to run on.
        _ = _supervisor.StopAsync();
        _notifyIcon.Visible = false;
        ExitThread();
    }

    // ───────────────────────────────── plumbing ───────────────────────────────

    private void BeginInvokeOnUi(Action action)
    {
        if (_disposed)
            return;
        var target = _notifyIcon.ContextMenuStrip;
        if (target is { IsHandleCreated: true } && target.InvokeRequired)
            target.BeginInvoke(action);
        else
            action();
    }

    private static void OpenUrl(string url)
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or FileNotFoundException)
        {
            MessageBox.Show(
                "Could not open " + url, "dotLLM", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }
    }

    private static void ShowDialog(Form form)
    {
        using (form)
            form.ShowDialog();
    }

    private static void Warn(string message) =>
        MessageBox.Show(message, "dotLLM", MessageBoxButtons.OK, MessageBoxIcon.Warning);

    private static void Inform(string message) =>
        MessageBox.Show(message, "dotLLM", MessageBoxButtons.OK, MessageBoxIcon.Information);

    private static string Ellipsize(string text, int maximum) =>
        text.Length <= maximum ? text : text[..(maximum - 1)] + "…";

    protected override void Dispose(bool disposing)
    {
        if (disposing && !_disposed)
        {
            _disposed = true;
            _refreshTimer.Stop();
            _refreshTimer.Dispose();
            _notifyIcon.Visible = false;
            _notifyIcon.ContextMenuStrip?.Dispose();
            _notifyIcon.Dispose();
            _icons.Dispose();
            // Disposing the supervisor reaps an owned child. The job object is the backstop for
            // the exits that never reach here.
            _supervisor.Dispose();
            _apiHttp.Dispose();
            _updateHttp.Dispose();
        }

        base.Dispose(disposing);
    }

    /// <summary>Bridges <see cref="IServerHealthProbe"/> onto the HTTP client.</summary>
    private sealed class ApiHealthProbe(DotLlmApiClient api) : IServerHealthProbe
    {
        public Task<bool> IsHealthyAsync(CancellationToken ct) => api.IsHealthyAsync(ct);

        public Task<bool> IsReadyAsync(CancellationToken ct) => api.IsReadyAsync(ct);
    }
}
