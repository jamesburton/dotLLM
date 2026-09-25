using System.Runtime.Versioning;
using System.Windows.Forms;
using DotLLM.Tray.Updates;

namespace DotLLM.Tray.Ui;

/// <summary>
/// Shows a newer release and its changelog, and offers to open the release page.
/// </summary>
/// <remarks>
/// There is no "install" button, and its absence is deliberate. <c>.github/workflows/release.yml</c>
/// publishes unsigned self-contained archives; an in-place self-update would hand the user a
/// binary SmartScreen marks as an unrecognized app, and a Defender quarantine partway through the
/// swap would leave a broken install with no tray left to repair it. Until the project has an
/// Authenticode certificate, "show the changelog and open the page" is the honest option.
/// <c>docs/TRAY.md</c> records the decision and what would change it.
/// </remarks>
[SupportedOSPlatform("windows")]
internal sealed class UpdateForm : Form
{
    /// <summary>Creates the dialog for a check result that found an update.</summary>
    /// <param name="result">The update-check outcome.</param>
    internal UpdateForm(UpdateCheckResult result)
    {
        ArgumentNullException.ThrowIfNull(result);

        Text = "dotLLM — Update available";
        Width = 620;
        Height = 520;
        StartPosition = FormStartPosition.CenterScreen;
        MinimizeBox = false;
        MaximizeBox = false;

        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 4,
            Padding = new Padding(12),
        };
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));

        root.Controls.Add(
            new Label
            {
                AutoSize = true,
                Text = $"dotLLM {result.LatestVersion} is available. You are running {result.CurrentVersion}.",
            },
            0, 0);

        root.Controls.Add(
            new TextBox
            {
                Dock = DockStyle.Fill,
                Multiline = true,
                ReadOnly = true,
                ScrollBars = ScrollBars.Vertical,
                Text = string.IsNullOrWhiteSpace(result.Changelog)
                    ? "(No release notes were published for this release.)"
                    : result.Changelog!.ReplaceLineEndings(),
            },
            0, 1);

        root.Controls.Add(
            new Label
            {
                AutoSize = true,
                MaximumSize = new System.Drawing.Size(570, 0),
                Text = "The tray does not install updates itself. Releases are not code-signed, so "
                     + "an automatic replacement would be flagged by SmartScreen and could be "
                     + "quarantined mid-swap. Download it from the release page and replace the "
                     + "files yourself; the new version is in use the next time dotLLM starts.",
            },
            0, 2);

        var buttons = new FlowLayoutPanel { FlowDirection = FlowDirection.RightToLeft, Dock = DockStyle.Top, AutoSize = true };
        var later = new Button { Text = "Later", DialogResult = DialogResult.Cancel, AutoSize = true };
        var open = new Button { Text = "Open release page", DialogResult = DialogResult.OK, AutoSize = true };
        buttons.Controls.Add(later);
        buttons.Controls.Add(open);
        root.Controls.Add(buttons, 0, 3);

        AcceptButton = open;
        CancelButton = later;
        Controls.Add(root);
    }
}
