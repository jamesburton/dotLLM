using System.Runtime.Versioning;
using Microsoft.Win32;

namespace DotLLM.Tray.Autostart;

/// <summary>Reads and writes the per-user autostart entry. Abstracted so the logic is testable.</summary>
public interface IAutostartStore
{
    /// <summary>The stored command line for <paramref name="valueName"/>, or null when absent.</summary>
    /// <param name="valueName">Entry name.</param>
    string? Read(string valueName);

    /// <summary>Creates or replaces the entry.</summary>
    /// <param name="valueName">Entry name.</param>
    /// <param name="commandLine">Command line to run at logon.</param>
    void Write(string valueName, string commandLine);

    /// <summary>Removes the entry. A no-op when absent.</summary>
    /// <param name="valueName">Entry name.</param>
    void Delete(string valueName);
}

/// <summary>
/// The real store: <c>HKCU\Software\Microsoft\Windows\CurrentVersion\Run</c>.
/// </summary>
/// <remarks>
/// Per-user (<c>HKCU</c>), never <c>HKLM</c>: an all-users autostart needs elevation, would start
/// the tray for accounts that never asked for it, and could not be removed by the user who enabled
/// it. The <c>Run</c> key is also the entry Task Manager's Startup tab lists and can disable,
/// which means the user has a way out that does not involve the tray.
/// </remarks>
[SupportedOSPlatform("windows")]
public sealed class RegistryAutostartStore : IAutostartStore
{
    /// <summary>The registry sub-key holding per-user logon entries.</summary>
    public const string RunKeyPath = @"Software\Microsoft\Windows\CurrentVersion\Run";

    /// <inheritdoc />
    public string? Read(string valueName)
    {
        using var key = Registry.CurrentUser.OpenSubKey(RunKeyPath, writable: false);
        return key?.GetValue(valueName) as string;
    }

    /// <inheritdoc />
    public void Write(string valueName, string commandLine)
    {
        using var key = Registry.CurrentUser.CreateSubKey(RunKeyPath, writable: true);
        key.SetValue(valueName, commandLine, RegistryValueKind.String);
    }

    /// <inheritdoc />
    public void Delete(string valueName)
    {
        using var key = Registry.CurrentUser.OpenSubKey(RunKeyPath, writable: true);
        key?.DeleteValue(valueName, throwOnMissingValue: false);
    }
}

/// <summary>
/// Turns "start dotLLM when I log in" on and off.
/// </summary>
/// <remarks>
/// <para>
/// <b>Off until explicitly enabled.</b> Nothing in this type runs on construction and nothing
/// writes the registry except <see cref="Enable"/>. A fresh install therefore has no autostart
/// entry at all — not a disabled one — which is the acceptance criterion #455 states.
/// </para>
/// <para>
/// <b>The registry is the source of truth, not the tray's settings file.</b> A user can remove the
/// entry from Task Manager's Startup tab or with <c>regedit</c> without the tray ever running, so
/// a cached flag in <c>tray.json</c> would drift and the UI would lie. <see cref="IsEnabled"/>
/// always reads the key.
/// </para>
/// </remarks>
public sealed class AutostartManager
{
    /// <summary>The <c>Run</c> value name this tray owns.</summary>
    public const string DefaultValueName = "dotLLM Tray";

    private readonly IAutostartStore _store;
    private readonly string _valueName;

    /// <summary>Creates a manager over a store.</summary>
    /// <param name="store">Where autostart entries live.</param>
    /// <param name="valueName">Entry name. Overridable so tests can use a throwaway name.</param>
    public AutostartManager(IAutostartStore store, string valueName = DefaultValueName)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        if (string.IsNullOrWhiteSpace(valueName))
            throw new ArgumentException("Value name must not be empty.", nameof(valueName));
        _valueName = valueName;
    }

    /// <summary>The entry name this manager reads and writes.</summary>
    public string ValueName => _valueName;

    /// <summary>Whether an autostart entry currently exists.</summary>
    public bool IsEnabled() => !string.IsNullOrWhiteSpace(_store.Read(_valueName));

    /// <summary>The command line currently registered, or null when autostart is off.</summary>
    public string? CurrentCommandLine() => _store.Read(_valueName);

    /// <summary>
    /// Enables autostart for <paramref name="executablePath"/>.
    /// </summary>
    /// <remarks>
    /// The path is quoted unconditionally. An unquoted path containing a space — and the default
    /// install location under <c>C:\Program Files</c> does — is parsed by <c>CreateProcess</c> as
    /// a different executable plus arguments, which is both a startup failure and the classic
    /// unquoted-service-path hijack. Re-enabling with a new path overwrites in place rather than
    /// leaving a stale second entry.
    /// </remarks>
    /// <param name="executablePath">Absolute path of the tray executable.</param>
    /// <param name="arguments">Extra arguments, already individually quoted if needed.</param>
    public void Enable(string executablePath, string? arguments = null)
    {
        if (string.IsNullOrWhiteSpace(executablePath))
            throw new ArgumentException("Executable path must not be empty.", nameof(executablePath));

        var command = Quote(executablePath);
        if (!string.IsNullOrWhiteSpace(arguments))
            command = command + " " + arguments.Trim();

        _store.Write(_valueName, command);
    }

    /// <summary>
    /// Disables autostart by removing the entry entirely.
    /// </summary>
    /// <remarks>
    /// Removed, not blanked: an empty <c>Run</c> value still shows in Task Manager's Startup list,
    /// so leaving one behind would fail "cleanly removable". Idempotent.
    /// </remarks>
    public void Disable() => _store.Delete(_valueName);

    /// <summary>Quotes a path for a <c>Run</c> command line.</summary>
    /// <param name="path">The path to quote.</param>
    public static string Quote(string path) =>
        path.StartsWith('"') && path.EndsWith('"') ? path : "\"" + path + "\"";
}
