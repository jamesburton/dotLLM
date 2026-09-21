using System.Globalization;
using System.Runtime.Versioning;
using System.Windows.Forms;
using DotLLM.Tray.Api;

namespace DotLLM.Tray.Ui;

/// <summary>
/// Model management: resident models with footprint and time-to-auto-unload, locally available
/// models, load/unload/enable/disable, and Hugging Face pulls with progress.
/// </summary>
/// <remarks>
/// Every button is one call on <see cref="DotLlmApiClient"/>. The form holds no server state of
/// its own; after any mutation it re-reads the two listings, because the server is the only place
/// that knows what actually happened (an unload waits behind an in-flight generation, a disable
/// leaves an active model loaded, a settings change can evict).
/// </remarks>
[SupportedOSPlatform("windows")]
internal sealed class ModelsForm : Form
{
    private readonly DotLlmApiClient _api;
    private readonly ListView _resident = NewListView();
    private readonly ListView _available = NewListView();
    private readonly ListView _pulls = NewListView();
    private readonly TextBox _repoId = new() { PlaceholderText = "org/repo (Hugging Face)" };
    private readonly TextBox _filename = new() { PlaceholderText = "model-Q4_K_M.gguf" };
    private readonly Label _statusLabel = new() { AutoSize = true, Text = "" };
    private readonly System.Windows.Forms.Timer _poll = new() { Interval = 1500 };
    private bool _refreshing;

    /// <summary>Creates the dialog.</summary>
    /// <param name="api">The client every action goes through.</param>
    internal ModelsForm(DotLlmApiClient api)
    {
        _api = api;

        Text = "dotLLM — Models";
        Width = 940;
        Height = 700;
        StartPosition = FormStartPosition.CenterScreen;
        MinimizeBox = false;

        _resident.Columns.Add("Model", 280);
        _resident.Columns.Add("Active", 60);
        _resident.Columns.Add("Size", 90);
        _resident.Columns.Add("Idle", 80);
        _resident.Columns.Add("Keep-alive", 90);
        _resident.Columns.Add("Auto-unload in", 120);

        _available.Columns.Add("Model id", 280);
        _available.Columns.Add("Repo", 240);
        _available.Columns.Add("Size", 90);
        _available.Columns.Add("Enabled", 80);

        _pulls.Columns.Add("Job", 200);
        _pulls.Columns.Add("File", 280);
        _pulls.Columns.Add("Status", 100);
        _pulls.Columns.Add("Progress", 140);

        Controls.Add(BuildLayout());
        _poll.Tick += async (_, _) => await ReloadAsync().ConfigureAwait(true);

        Shown += async (_, _) =>
        {
            await ReloadAsync().ConfigureAwait(true);
            _poll.Start();
        };
    }

    private Control BuildLayout()
    {
        var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 5, Padding = new Padding(10) };
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 34));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 34));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 32));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));

        root.Controls.Add(Group("Resident (loaded now)", _resident, ResidentButtons()), 0, 0);
        root.Controls.Add(Group("Available locally", _available, AvailableButtons()), 0, 1);
        root.Controls.Add(PullBar(), 0, 2);
        root.Controls.Add(Group("Downloads", _pulls, PullButtons()), 0, 3);
        root.Controls.Add(_statusLabel, 0, 4);
        return root;
    }

    private static GroupBox Group(string title, Control list, Control buttons)
    {
        var box = new GroupBox { Text = title, Dock = DockStyle.Fill };
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2 };
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        layout.Controls.Add(list, 0, 0);
        layout.Controls.Add(buttons, 0, 1);
        box.Controls.Add(layout);
        return box;
    }

    private Control ResidentButtons()
    {
        var panel = NewButtonPanel();
        panel.Controls.Add(Button("Unload", async () =>
        {
            if (SelectedKey(_resident) is not { } key)
                return;
            // The server waits behind an in-flight generation rather than interrupting it, so
            // this can take as long as the current response. Say so instead of appearing hung.
            SetStatus($"Unloading {key} (waits for any in-flight generation)…");
            var result = await _api.UnloadModelAsync(new TrayUnloadRequest { Model = key }).ConfigureAwait(true);
            SetStatus(result.Unloaded.Length > 0
                ? "Unloaded " + string.Join(", ", result.Unloaded)
                : "Nothing was resident under that key.");
        }));

        panel.Controls.Add(Button("Unload all", async () =>
        {
            SetStatus("Unloading every resident model…");
            var result = await _api.UnloadModelAsync(new TrayUnloadRequest { All = true }).ConfigureAwait(true);
            SetStatus("Unloaded " + (result.Unloaded.Length == 0 ? "nothing" : string.Join(", ", result.Unloaded)));
        }));

        return panel;
    }

    private Control AvailableButtons()
    {
        var panel = NewButtonPanel();

        panel.Controls.Add(Button("Load", async () =>
        {
            if (_available.SelectedItems.Count == 0)
                return;
            var model = (TrayAvailableModel)_available.SelectedItems[0].Tag!;
            SetStatus($"Loading {model.ModelId}…");
            await _api.LoadModelAsync(new TrayLoadRequest { Model = model.FullPath }).ConfigureAwait(true);
            SetStatus("Loaded " + model.ModelId);
        }));

        panel.Controls.Add(Button("Enable", async () =>
        {
            if (SelectedKey(_available) is not { } key)
                return;
            var result = await _api.EnableModelAsync(key).ConfigureAwait(true);
            SetStatus($"{result.Model} is now loadable.");
        }));

        panel.Controls.Add(Button("Disable", async () =>
        {
            if (SelectedKey(_available) is not { } key)
                return;
            var result = await _api.DisableModelAsync(key).ConfigureAwait(true);
            // Two things a user will otherwise get wrong: disabling does not unload, and the
            // curation is in-memory only.
            SetStatus(result.StillLoaded
                ? $"{result.Model} is disabled but still loaded and serving — use Unload to free it. "
                  + "(Disabled models reset when the server restarts.)"
                : $"{result.Model} is disabled. (Disabled models reset when the server restarts.)");
        }));

        return panel;
    }

    private Control PullBar()
    {
        var panel = new FlowLayoutPanel { Dock = DockStyle.Fill, AutoSize = true, WrapContents = false };
        _repoId.Width = 300;
        _filename.Width = 300;
        panel.Controls.Add(new Label { Text = "Pull:", AutoSize = true, Padding = new Padding(0, 6, 4, 0) });
        panel.Controls.Add(_repoId);
        panel.Controls.Add(_filename);
        panel.Controls.Add(Button("Start download", async () =>
        {
            if (string.IsNullOrWhiteSpace(_repoId.Text) || string.IsNullOrWhiteSpace(_filename.Text))
            {
                SetStatus("Both a repo id and a filename are required.");
                return;
            }

            // Non-streaming start: the job outlives this dialog either way (only DELETE cancels),
            // and the Downloads list polls GET /v1/models/pull, which also re-attaches to jobs
            // started before the tray was opened.
            var job = await _api.StartPullAsync(new TrayPullRequest
            {
                RepoId = _repoId.Text.Trim(),
                Filename = _filename.Text.Trim(),
            }).ConfigureAwait(true);

            SetStatus($"Download {job.Id} started. It continues even if you close this window.");
        }));
        return panel;
    }

    private Control PullButtons()
    {
        var panel = NewButtonPanel();
        panel.Controls.Add(Button("Cancel download", async () =>
        {
            if (_pulls.SelectedItems.Count == 0)
                return;
            var job = (TrayPullJob)_pulls.SelectedItems[0].Tag!;
            // DELETE is the only thing that stops a download. The partial file is kept for resume.
            await _api.CancelPullAsync(job.Id).ConfigureAwait(true);
            SetStatus($"Cancelling {job.Id}. The partial file is kept, so a later pull resumes it.");
        }));
        return panel;
    }

    private async Task ReloadAsync()
    {
        if (_refreshing)
            return;
        _refreshing = true;
        try
        {
            var resident = await _api.GetResidentModelsAsync().ConfigureAwait(true);
            var available = await _api.GetAvailableModelsAsync().ConfigureAwait(true);
            var pulls = await _api.GetPullJobsAsync().ConfigureAwait(true);

            Fill(_resident, resident.Data, model =>
            [
                model.Id,
                model.IsActive ? "yes" : "",
                FormatBytes(model.SizeBytes),
                FormatSeconds(model.IdleSeconds),
                model.KeepAliveSeconds < 0 ? "never" : FormatSeconds(model.KeepAliveSeconds),
                // #455 asks for this explicitly. Null means the keep-alive never expires.
                model.ExpiresInSeconds is { } expires ? FormatSeconds(expires) : "never",
            ]);

            Fill(_available, available.Models, model =>
            [
                model.ModelId,
                model.RepoId,
                FormatBytes(model.SizeBytes),
                model.Enabled ? "yes" : "no",
            ]);

            Fill(_pulls, pulls.Jobs, job =>
            [
                job.Id,
                job.Filename,
                job.Status,
                job.Percent is { } percent
                    ? percent.ToString("0.0", CultureInfo.CurrentCulture) + "%"
                    : FormatBytes(job.BytesDownloaded),
            ]);
        }
        catch (DotLlmApiException ex)
        {
            SetStatus(ex.Message);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            SetStatus("The server is not responding.");
        }
        finally
        {
            _refreshing = false;
        }
    }

    private static void Fill<T>(ListView view, IReadOnlyList<T> items, Func<T, string[]> project)
    {
        var selected = view.SelectedItems.Count > 0 ? view.SelectedItems[0].SubItems[0].Text : null;
        view.BeginUpdate();
        view.Items.Clear();
        foreach (var item in items)
        {
            var row = new ListViewItem(project(item)) { Tag = item };
            view.Items.Add(row);
            if (selected is not null && row.SubItems[0].Text == selected)
                row.Selected = true;
        }

        view.EndUpdate();
    }

    private static string? SelectedKey(ListView view) =>
        view.SelectedItems.Count == 0 ? null : view.SelectedItems[0].SubItems[0].Text;

    private void SetStatus(string message) => _statusLabel.Text = message;

    private static ListView NewListView() => new()
    {
        Dock = DockStyle.Fill,
        View = View.Details,
        FullRowSelect = true,
        MultiSelect = false,
        HideSelection = false,
    };

    private static FlowLayoutPanel NewButtonPanel() =>
        new() { Dock = DockStyle.Fill, AutoSize = true, WrapContents = false };

    private Button Button(string text, Func<Task> onClick)
    {
        var button = new Button { Text = text, AutoSize = true };
        button.Click += async (_, _) =>
        {
            button.Enabled = false;
            try
            {
                await onClick().ConfigureAwait(true);
                await ReloadAsync().ConfigureAwait(true);
            }
            catch (DotLlmApiException ex)
            {
                // Includes the admin gate's 403, whose body names the flag to start with.
                SetStatus(ex.Message);
            }
            catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
            {
                SetStatus("The server is not responding.");
            }
            finally
            {
                button.Enabled = true;
            }
        };
        return button;
    }

    private static string FormatBytes(long bytes)
    {
        if (bytes <= 0)
            return "";
        string[] units = ["B", "KB", "MB", "GB", "TB"];
        double value = bytes;
        var unit = 0;
        while (value >= 1024 && unit < units.Length - 1)
        {
            value /= 1024;
            unit++;
        }

        return value.ToString("0.#", CultureInfo.CurrentCulture) + " " + units[unit];
    }

    private static string FormatSeconds(double seconds)
    {
        if (seconds < 0)
            return "never";
        var span = TimeSpan.FromSeconds(seconds);
        if (span.TotalHours >= 1)
            return span.ToString(@"h\:mm\:ss", CultureInfo.CurrentCulture);
        return span.ToString(@"m\:ss", CultureInfo.CurrentCulture);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _poll.Stop();
            _poll.Dispose();
        }

        base.Dispose(disposing);
    }
}
