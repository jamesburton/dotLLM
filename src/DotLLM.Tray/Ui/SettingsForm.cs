using System.Globalization;
using System.Runtime.Versioning;
using System.Windows.Forms;
using DotLLM.Tray.Api;
using DotLLM.Tray.Config;

namespace DotLLM.Tray.Ui;

/// <summary>
/// Settings, split into the three tiers that actually exist, each labelled with when it applies.
/// </summary>
/// <remarks>
/// <list type="bullet">
///   <item><b>Live</b> — written with <c>PUT /v1/settings</c> and in force immediately: keep-alive
///   default, max resident models, residency byte budget, idle sweep interval.</item>
///   <item><b>On next server start</b> — the tray's own <c>tray.json</c>, baked into the child's
///   command line: host, port, executable path, startup model, device, GPU layers, KV cache
///   types.</item>
///   <item><b>Per load</b> — sent on <c>POST /v1/models/load</c>; the values here are only the
///   defaults a tray-started server begins with.</item>
/// </list>
/// Presenting the launch-time fields as though they were live would be the single most misleading
/// thing this dialog could do, so each group says so in its caption.
/// </remarks>
[SupportedOSPlatform("windows")]
internal sealed class SettingsForm : Form
{
    private readonly DotLlmApiClient _api;
    private readonly bool _serverRunning;

    private readonly TextBox _host = new();
    private readonly NumericUpDown _port = new() { Minimum = 1, Maximum = 65535 };
    private readonly TextBox _executablePath = new();
    private readonly TextBox _startupModel = new();
    private readonly ComboBox _device = new() { DropDownStyle = ComboBoxStyle.DropDown };
    private readonly NumericUpDown _gpuLayers = new() { Minimum = 0, Maximum = 999 };
    private readonly ComboBox _cacheTypeK = NewCacheCombo();
    private readonly ComboBox _cacheTypeV = NewCacheCombo();
    private readonly CheckBox _startOnLaunch = new() { Text = "Start a server when the tray launches", AutoSize = true };
    private readonly CheckBox _checkForUpdates = new() { Text = "Check GitHub for new releases (opt-in)", AutoSize = true };
    private readonly CheckBox _includePrereleases = new() { Text = "Include prereleases", AutoSize = true };

    private readonly NumericUpDown _keepAlive = new() { Minimum = -1, Maximum = 86400, DecimalPlaces = 0 };
    private readonly NumericUpDown _maxResident = new() { Minimum = 1, Maximum = 64 };
    private readonly NumericUpDown _budgetGb = new() { Minimum = 0, Maximum = 4096, DecimalPlaces = 1, Increment = 0.5m };
    private readonly NumericUpDown _sweepInterval = new() { Minimum = 0.1m, Maximum = 3600, DecimalPlaces = 1, Increment = 0.5m };
    private readonly Label _liveStatus = new() { AutoSize = true, MaximumSize = new System.Drawing.Size(560, 0) };
    private readonly Button _applyLive = new() { Text = "Apply live settings now", AutoSize = true };

    /// <summary>Creates the dialog.</summary>
    /// <param name="settings">Current tray settings.</param>
    /// <param name="api">Client for the live-settings and devices routes.</param>
    /// <param name="serverRunning">Whether a server is answering; the live group is useless without one.</param>
    internal SettingsForm(TraySettings settings, DotLlmApiClient api, bool serverRunning)
    {
        _api = api;
        _serverRunning = serverRunning;
        Result = settings;

        Text = "dotLLM — Settings";
        Width = 640;
        Height = 760;
        StartPosition = FormStartPosition.CenterScreen;
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MinimizeBox = false;
        MaximizeBox = false;

        _host.Text = settings.Host;
        _port.Value = settings.Port;
        _executablePath.Text = settings.ExecutablePath ?? "";
        _startupModel.Text = settings.StartupModel ?? "";
        _device.Text = settings.Device ?? "";
        _gpuLayers.Value = settings.GpuLayers ?? 0;
        _cacheTypeK.Text = settings.CacheTypeK ?? "f32";
        _cacheTypeV.Text = settings.CacheTypeV ?? "f32";
        _startOnLaunch.Checked = settings.StartServerOnLaunch;
        _checkForUpdates.Checked = settings.CheckForUpdates;
        _includePrereleases.Checked = settings.IncludePrereleases;

        Controls.Add(BuildLayout());

        Shown += async (_, _) => await LoadServerSideAsync().ConfigureAwait(true);
    }

    /// <summary>The settings as edited. Valid once the dialog returns <see cref="DialogResult.OK"/>.</summary>
    internal TraySettings Result { get; private set; }

    private Control BuildLayout()
    {
        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 4,
            Padding = new Padding(10),
            AutoScroll = true,
        };
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));

        root.Controls.Add(LiveGroup(), 0, 0);
        root.Controls.Add(LaunchGroup(), 0, 1);
        root.Controls.Add(TrayGroup(), 0, 2);
        root.Controls.Add(DialogButtons(), 0, 3);
        return root;
    }

    private GroupBox LiveGroup()
    {
        var box = new GroupBox
        {
            // The caption is load-bearing: these four take effect the moment Apply is pressed.
            Text = "Server — live (applied immediately, no restart)",
            Dock = DockStyle.Top,
            AutoSize = true,
            Padding = new Padding(8),
        };

        var grid = NewGrid();
        AddRow(grid, "Keep-alive (s, -1 = never)", _keepAlive);
        AddRow(grid, "Max resident models", _maxResident);
        AddRow(grid, "Residency budget (GB, 0 = unlimited)", _budgetGb);
        AddRow(grid, "Idle sweep interval (s)", _sweepInterval);

        _applyLive.Enabled = _serverRunning;
        _applyLive.Click += async (_, _) => await ApplyLiveAsync().ConfigureAwait(true);
        grid.Controls.Add(_applyLive, 1, grid.RowCount);
        grid.RowCount++;
        grid.Controls.Add(_liveStatus, 1, grid.RowCount);
        grid.RowCount++;

        if (!_serverRunning)
            _liveStatus.Text = "No server is answering, so live settings cannot be read or applied.";

        box.Controls.Add(grid);
        return box;
    }

    private GroupBox LaunchGroup()
    {
        var box = new GroupBox
        {
            Text = "Server — applied when the tray next starts a server",
            Dock = DockStyle.Top,
            AutoSize = true,
            Padding = new Padding(8),
        };

        var grid = NewGrid();
        AddRow(grid, "Host", _host);
        AddRow(grid, "Port", _port);
        AddRow(grid, "dotllm.exe path (blank = auto-detect)", _executablePath);
        AddRow(grid, "Startup model (blank = none)", _startupModel);
        AddRow(grid, "Device", _device);
        AddRow(grid, "GPU layers", _gpuLayers);
        AddRow(grid, "KV cache type (K)", _cacheTypeK);
        AddRow(grid, "KV cache type (V)", _cacheTypeV);
        box.Controls.Add(grid);
        return box;
    }

    private GroupBox TrayGroup()
    {
        var box = new GroupBox { Text = "Tray", Dock = DockStyle.Top, AutoSize = true, Padding = new Padding(8) };
        var stack = new FlowLayoutPanel { FlowDirection = FlowDirection.TopDown, AutoSize = true, Dock = DockStyle.Top };
        stack.Controls.Add(_startOnLaunch);
        stack.Controls.Add(_checkForUpdates);
        stack.Controls.Add(_includePrereleases);
        stack.Controls.Add(new Label
        {
            AutoSize = true,
            MaximumSize = new System.Drawing.Size(560, 0),
            Text = "Updates are checked only, never applied automatically: the tray shows the "
                 + "changelog and opens the release page. Start at login is toggled from the tray menu.",
        });
        box.Controls.Add(stack);
        return box;
    }

    private Control DialogButtons()
    {
        var panel = new FlowLayoutPanel { FlowDirection = FlowDirection.RightToLeft, Dock = DockStyle.Top, AutoSize = true };
        var cancel = new Button { Text = "Cancel", DialogResult = DialogResult.Cancel, AutoSize = true };
        var ok = new Button { Text = "OK", AutoSize = true };
        ok.Click += (_, _) =>
        {
            Result = Result with
            {
                Host = string.IsNullOrWhiteSpace(_host.Text) ? "localhost" : _host.Text.Trim(),
                Port = (int)_port.Value,
                ExecutablePath = Blank(_executablePath.Text),
                StartupModel = Blank(_startupModel.Text),
                Device = Blank(_device.Text),
                GpuLayers = _gpuLayers.Value == 0 ? null : (int)_gpuLayers.Value,
                CacheTypeK = Blank(_cacheTypeK.Text),
                CacheTypeV = Blank(_cacheTypeV.Text),
                StartServerOnLaunch = _startOnLaunch.Checked,
                CheckForUpdates = _checkForUpdates.Checked,
                IncludePrereleases = _includePrereleases.Checked,
            };
            DialogResult = DialogResult.OK;
            Close();
        };

        panel.Controls.Add(cancel);
        panel.Controls.Add(ok);
        AcceptButton = ok;
        CancelButton = cancel;
        return panel;
    }

    private async Task LoadServerSideAsync()
    {
        if (!_serverRunning)
            return;

        try
        {
            var settings = await _api.GetSettingsAsync().ConfigureAwait(true);
            _keepAlive.Value = Clamp(_keepAlive, (decimal)settings.KeepAliveSeconds);
            _maxResident.Value = Clamp(_maxResident, settings.MaxResidentModels);
            _budgetGb.Value = Clamp(_budgetGb, settings.ResidentMemoryBudgetBytes / 1024m / 1024m / 1024m);
            _sweepInterval.Value = Clamp(_sweepInterval, (decimal)settings.IdleSweepIntervalSeconds);

            if (!settings.ModelAdminApiEnabled)
            {
                // The gate is a startup flag, so this is not fixable from here.
                _applyLive.Enabled = false;
                _liveStatus.Text = "This server was started without --allow-model-admin, so settings "
                                 + "are read-only. Restart it with that flag to change them.";
            }

            await LoadDevicesAsync().ConfigureAwait(true);
        }
        catch (DotLlmApiException ex)
        {
            _liveStatus.Text = ex.Message;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            _liveStatus.Text = "The server is not responding.";
        }
    }

    private async Task LoadDevicesAsync()
    {
        var devices = await _api.GetDevicesAsync().ConfigureAwait(true);
        var current = _device.Text;
        _device.Items.Clear();

        foreach (var backend in devices.Backends)
        {
            // Gated on Servable, not Available. Vulkan reports available-but-not-servable, and
            // offering it would produce a load that silently lands on the CPU.
            if (!backend.Servable)
                continue;
            foreach (var device in backend.Devices)
            {
                if (device.DeviceString is { Length: > 0 } deviceString)
                    _device.Items.Add(deviceString);
            }
        }

        _device.Text = current;

        var unservable = devices.Backends
            .Where(b => b.Available && !b.Servable)
            .Select(b => b.Name)
            .ToArray();
        if (unservable.Length > 0)
        {
            _liveStatus.Text = (_liveStatus.Text + " ").TrimStart()
                + $"Present but not usable for serving: {string.Join(", ", unservable)}.";
        }
    }

    private async Task ApplyLiveAsync()
    {
        _applyLive.Enabled = false;
        try
        {
            var result = await _api.UpdateSettingsAsync(new TraySettingsUpdate
            {
                KeepAliveSeconds = (double)_keepAlive.Value,
                MaxResidentModels = (int)_maxResident.Value,
                ResidentMemoryBudgetBytes = (long)(_budgetGb.Value * 1024m * 1024m * 1024m),
                IdleSweepIntervalSeconds = (double)_sweepInterval.Value,
            }).ConfigureAwait(true);

            var message = "Applied: " + string.Join(", ", result.Applied);
            if (result.Evicted.Length > 0)
                message += ". Evicted by the new budget: " + string.Join(", ", result.Evicted);
            if (result.RestartRequired.Length > 0)
                message += ". Needs a restart: " + string.Join(", ", result.RestartRequired);
            _liveStatus.Text = message;
        }
        catch (DotLlmApiException ex)
        {
            _liveStatus.Text = ex.Message;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            _liveStatus.Text = "The server is not responding.";
        }
        finally
        {
            _applyLive.Enabled = _serverRunning;
        }
    }

    private static decimal Clamp(NumericUpDown control, decimal value) =>
        Math.Clamp(value, control.Minimum, control.Maximum);

    private static string? Blank(string? text) => string.IsNullOrWhiteSpace(text) ? null : text.Trim();

    private static ComboBox NewCacheCombo()
    {
        var combo = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList };
        combo.Items.AddRange(["f32", "q8_0", "q4_0"]);
        return combo;
    }

    private static TableLayoutPanel NewGrid() =>
        new() { ColumnCount = 2, AutoSize = true, Dock = DockStyle.Top, RowCount = 0 };

    private static void AddRow(TableLayoutPanel grid, string label, Control control)
    {
        control.Width = 260;
        control.Anchor = AnchorStyles.Left;
        grid.Controls.Add(new Label { Text = label, AutoSize = true, Anchor = AnchorStyles.Left }, 0, grid.RowCount);
        grid.Controls.Add(control, 1, grid.RowCount);
        grid.RowCount++;
    }

}
