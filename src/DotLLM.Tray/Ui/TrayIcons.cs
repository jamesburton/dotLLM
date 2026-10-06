using System.Drawing;
using System.Drawing.Drawing2D;
using System.Runtime.Versioning;
using DotLLM.Tray.Hosting;

namespace DotLLM.Tray.Ui;

/// <summary>
/// Draws the tray icon for each server state.
/// </summary>
/// <remarks>
/// Drawn rather than shipped as .ico resources so the four states stay visually consistent and
/// the icon scales to whatever the shell asks for (96–200% DPI produce 16, 20, 24 and 32 px
/// requests). Each icon is generated once and cached: <see cref="Icon.FromHandle"/> hands back a
/// wrapper over an HICON that must outlive every assignment to <c>NotifyIcon.Icon</c>.
/// </remarks>
[SupportedOSPlatform("windows")]
internal sealed partial class TrayIcons : IDisposable
{
    private readonly Dictionary<ServerState, Icon> _cache = [];
    private readonly List<nint> _handles = [];
    private bool _disposed;

    /// <summary>The icon for a state, created on first use.</summary>
    /// <param name="state">Server state to depict.</param>
    internal Icon For(ServerState state)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_cache.TryGetValue(state, out var cached))
            return cached;

        var icon = Create(ColorFor(state));
        _cache[state] = icon;
        return icon;
    }

    /// <summary>The status colour for a state.</summary>
    /// <param name="state">Server state.</param>
    internal static Color ColorFor(ServerState state) => state switch
    {
        ServerState.RunningOwned => Color.FromArgb(0x2E, 0xA0, 0x43),   // green — ours, healthy
        ServerState.RunningAttached => Color.FromArgb(0x1F, 0x6F, 0xEB), // blue — healthy, not ours
        ServerState.Starting or ServerState.Stopping => Color.FromArgb(0xD2, 0x9A, 0x22), // amber
        ServerState.Failed => Color.FromArgb(0xD1, 0x24, 0x2F),          // red
        _ => Color.FromArgb(0x8B, 0x94, 0x9E),                            // grey — stopped/unknown
    };

    /// <summary>A short label for a state, for the tooltip and the menu header.</summary>
    /// <param name="status">Current status.</param>
    internal static string Describe(ServerStatus status) => status.State switch
    {
        ServerState.RunningOwned => status.IsModelLoaded
            ? "Running (started by the tray) — model loaded"
            : "Running (started by the tray) — no model loaded",
        ServerState.RunningAttached => status.IsModelLoaded
            ? "Running (started elsewhere) — model loaded"
            : "Running (started elsewhere) — no model loaded",
        ServerState.Starting => "Starting…",
        ServerState.Stopping => "Stopping…",
        ServerState.Failed => "Failed",
        ServerState.Stopped => "Stopped",
        _ => "Unknown",
    };

    private Icon Create(Color color)
    {
        using var bitmap = new Bitmap(32, 32);
        using (var graphics = Graphics.FromImage(bitmap))
        {
            graphics.SmoothingMode = SmoothingMode.AntiAlias;
            graphics.Clear(Color.Transparent);
            using var fill = new SolidBrush(color);
            graphics.FillEllipse(fill, 3, 3, 26, 26);
            using var rim = new Pen(Color.FromArgb(0x60, 0, 0, 0), 2f);
            graphics.DrawEllipse(rim, 3, 3, 26, 26);
        }

        // Icon.FromHandle does not own the HICON, so it is tracked and destroyed in Dispose.
        var handle = bitmap.GetHicon();
        _handles.Add(handle);
        return Icon.FromHandle(handle);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        foreach (var icon in _cache.Values)
            icon.Dispose();
        _cache.Clear();

        foreach (var handle in _handles)
            NativeMethods.DestroyIcon(handle);
        _handles.Clear();
    }

    private static partial class NativeMethods
    {
        [System.Runtime.InteropServices.LibraryImport("user32.dll", SetLastError = true)]
        [return: System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.Bool)]
        internal static partial bool DestroyIcon(nint handle);
    }
}
