using DotLLM.Tray.Config;
using Xunit;

namespace DotLLM.Tray.Tests;

/// <summary>Covers the tray's own preferences file.</summary>
/// <remarks>
/// <b>Discrimination (#417), by mutation.</b> Flipping <c>CheckForUpdates</c>'s default to true
/// failed <see cref="Defaults_AreOptInForEverythingThatReachesTheNetworkOrTheRegistry"/>; removing
/// the <c>JsonException</c> arm from <c>Load</c>'s catch failed
/// <see cref="Load_OfACorruptFile_ReturnsDefaultsRatherThanThrowing"/> with an unhandled
/// <c>JsonException</c>. Both reverted.
/// </remarks>
public sealed class TraySettingsStoreTests : IDisposable
{
    private readonly string _directory = Path.Combine(
        Path.GetTempPath(), "dotllm-tray-tests", Guid.NewGuid().ToString("N"));

    private string SettingsPath => Path.Combine(_directory, "tray.json");

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_directory))
                Directory.Delete(_directory, recursive: true);
        }
        catch (IOException)
        {
            // Best effort; a leaked temp directory is not worth failing a test over.
        }
    }

    [Fact]
    public void Defaults_AreOptInForEverythingThatReachesTheNetworkOrTheRegistry()
    {
        var settings = new TraySettings();

        // #455: the update check is opt-in and must make no request until turned on.
        Assert.False(settings.CheckForUpdates);
        Assert.False(settings.IncludePrereleases);
        // Autostart's truth is the registry, but nothing here may imply it is on.
        Assert.False(settings.AutostartLastKnown);
        // A freshly-installed tray attaches to whatever is running; it does not launch a server
        // behind the user's back.
        Assert.False(settings.StartServerOnLaunch);
        Assert.Equal("localhost", settings.Host);
        Assert.Equal(8080, settings.Port);
    }

    [Fact]
    public void Load_OfAMissingFile_ReturnsDefaults()
    {
        var store = new TraySettingsStore(SettingsPath);
        Assert.Equal(8080, store.Load().Port);
    }

    [Fact]
    public void SaveThenLoad_RoundTrips()
    {
        var store = new TraySettingsStore(SettingsPath);
        var settings = new TraySettings
        {
            Host = "127.0.0.1",
            Port = 9123,
            ExecutablePath = @"C:\dotLLM\dotllm.exe",
            StartupModel = "qwen2.5-3b",
            Device = "cpu",
            GpuLayers = 12,
            CacheTypeK = "q8_0",
            CacheTypeV = "q8_0",
            StartServerOnLaunch = true,
            CheckForUpdates = true,
            IncludePrereleases = true,
            AutostartLastKnown = true,
        };

        store.Save(settings);
        var loaded = store.Load();

        Assert.Equal(settings, loaded);
        Assert.True(File.Exists(SettingsPath));
    }

    [Fact]
    public void Save_CreatesTheDirectory()
    {
        new TraySettingsStore(SettingsPath).Save(new TraySettings());
        Assert.True(Directory.Exists(_directory));
    }

    [Fact]
    public void Load_OfACorruptFile_ReturnsDefaultsRatherThanThrowing()
    {
        // A tray that refuses to start over a bad JSON byte offers the user no way to fix it —
        // the only surface that could is the tray.
        Directory.CreateDirectory(_directory);
        File.WriteAllText(SettingsPath, "{ this is not json");

        var loaded = new TraySettingsStore(SettingsPath).Load();

        Assert.Equal(8080, loaded.Port);
        Assert.False(loaded.CheckForUpdates);
    }

    [Fact]
    public void Load_OfAPartialFile_KeepsDefaultsForTheMissingFields()
    {
        Directory.CreateDirectory(_directory);
        File.WriteAllText(SettingsPath, """{"port":9999}""");

        var loaded = new TraySettingsStore(SettingsPath).Load();

        Assert.Equal(9999, loaded.Port);
        Assert.Equal("localhost", loaded.Host);
        Assert.False(loaded.CheckForUpdates);
    }

    [Fact]
    public void DefaultPath_IsUnderTheRoamingApplicationDataFolder()
    {
        var path = TraySettingsStore.DefaultPath;
        Assert.EndsWith(Path.Combine("dotLLM", "tray.json"), path, StringComparison.OrdinalIgnoreCase);
        Assert.StartsWith(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            path,
            StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Constructor_RejectsAnEmptyPath() =>
        Assert.Throws<ArgumentException>(() => new TraySettingsStore(""));
}
