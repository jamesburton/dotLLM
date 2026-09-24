using System.Text.Json;
using System.Text.Json.Serialization;
using DotLLM.Tray.Hosting;

namespace DotLLM.Tray.Config;

/// <summary>
/// The tray's own persisted preferences.
/// </summary>
/// <remarks>
/// Deliberately small. Anything the <i>server</i> owns lives on the server and is read back over
/// <c>GET /v1/settings</c>; duplicating it here would create two sources of truth that drift the
/// moment someone uses the CLI or curl. What is here is either a tray preference (update checks)
/// or a launch-time parameter that has nowhere else to live until the next server start.
/// </remarks>
public sealed record TraySettings
{
    /// <summary>
    /// Host the tray-owned server binds, and the host the tray talks to. Nullable with no
    /// initializer on purpose: source generation drops an initializer on an <c>init</c> property,
    /// so a <c>= "localhost"</c> here would be a default that never actually applied to a loaded
    /// file while looking like it did. <see cref="Normalized"/> is where the default lives.
    /// </summary>
    [JsonPropertyName("host")]
    public string? Host { get; init; }

    /// <summary>Port the tray-owned server binds. Nullable for the same reason as
    /// <see cref="Host"/>; resolved by <see cref="Normalized"/>.</summary>
    [JsonPropertyName("port")]
    public int? Port { get; init; }

    /// <summary>Explicit path to <c>dotllm.exe</c>, or null to discover it.</summary>
    [JsonPropertyName("executable_path")]
    public string? ExecutablePath { get; init; }

    /// <summary>Model loaded when the tray starts a server, or null to start bare.</summary>
    [JsonPropertyName("startup_model")]
    public string? StartupModel { get; init; }

    /// <summary>Device for a tray-started server (<c>cpu</c>, <c>gpu</c>, <c>gpu:0</c>).</summary>
    [JsonPropertyName("device")]
    public string? Device { get; init; }

    /// <summary>GPU layers for a tray-started server.</summary>
    [JsonPropertyName("gpu_layers")]
    public int? GpuLayers { get; init; }

    /// <summary>KV-cache key type for a tray-started server.</summary>
    [JsonPropertyName("cache_type_k")]
    public string? CacheTypeK { get; init; }

    /// <summary>KV-cache value type for a tray-started server.</summary>
    [JsonPropertyName("cache_type_v")]
    public string? CacheTypeV { get; init; }

    /// <summary>Whether the tray starts a server as soon as it launches.</summary>
    [JsonPropertyName("start_server_on_launch")]
    public bool StartServerOnLaunch { get; init; }

    /// <summary>
    /// Whether the tray contacts GitHub to look for a newer release.
    /// <b>False by default</b> — the update check is opt-in and makes no network request until it
    /// is turned on.
    /// </summary>
    [JsonPropertyName("check_for_updates")]
    public bool CheckForUpdates { get; init; }

    /// <summary>Whether the update check considers prereleases.</summary>
    [JsonPropertyName("include_prereleases")]
    public bool IncludePrereleases { get; init; }

    /// <summary>
    /// Mirror of the autostart state, for display only. <see cref="Autostart.AutostartManager"/>
    /// reads the registry, which is authoritative; this field is never trusted for a decision.
    /// </summary>
    [JsonPropertyName("autostart_last_known")]
    public bool AutostartLastKnown { get; init; }

    /// <summary>Base URL the tray uses for every API call.</summary>
    public Uri BaseAddress => ServerLaunchSpecBuilder.BuildBaseAddress(ToLaunchOptions());

    /// <summary>
    /// Fills in any field a partial or older settings file left absent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Necessary because <see cref="TraySettingsJsonContext"/> is source-generated and the
    /// generated deserializer does <b>not</b> run property initializers on <c>init</c> members the
    /// JSON omits. Verified, not assumed: see
    /// <c>TraySettingsStoreTests.Load_OfAPartialFile_KeepsDefaultsForTheMissingFields</c>, which
    /// caught exactly this.
    /// </para>
    /// <para>
    /// The trigger is the <c>init</c> accessor, not a <c>required</c> member — this record has
    /// none and is affected anyway. Measured four ways in
    /// <c>TrayJsonContextDefaultsTests</c>; the earlier note here, and the project rule it came
    /// from, both named <c>required</c> as the cause, which was a correlate.
    /// </para>
    /// </remarks>
    public TraySettings Normalized() => this with
    {
        Host = string.IsNullOrWhiteSpace(Host) ? "localhost" : Host,
        Port = Port is > 0 and < 65536 ? Port : 8080,
    };

    /// <summary>The host to use, with the default applied. Safe on an un-normalized instance.</summary>
    [JsonIgnore]
    public string EffectiveHost => string.IsNullOrWhiteSpace(Host) ? "localhost" : Host;

    /// <summary>The port to use, with the default applied. Safe on an un-normalized instance.</summary>
    [JsonIgnore]
    public int EffectivePort => Port is > 0 and < 65536 ? Port.Value : 8080;

    /// <summary>Projects these settings onto the launch options for a tray-owned server.</summary>
    public ServerLaunchOptions ToLaunchOptions() => new()
    {
        Host = EffectiveHost,
        Port = EffectivePort,
        Model = StartupModel,
        Device = Device,
        GpuLayers = GpuLayers,
        CacheTypeK = CacheTypeK,
        CacheTypeV = CacheTypeV,
    };
}

/// <summary>Serialization context for <see cref="TraySettings"/>.</summary>
[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(TraySettings))]
public sealed partial class TraySettingsJsonContext : JsonSerializerContext;

/// <summary>Loads and saves <see cref="TraySettings"/> as JSON.</summary>
public sealed class TraySettingsStore
{
    private readonly string _path;

    /// <summary>Creates a store over an explicit file path.</summary>
    /// <param name="path">Absolute path of the settings file.</param>
    public TraySettingsStore(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path must not be empty.", nameof(path));
        _path = path;
    }

    /// <summary>The file this store reads and writes.</summary>
    public string Path => _path;

    /// <summary>The default location: <c>%APPDATA%\dotLLM\tray.json</c>.</summary>
    public static string DefaultPath => System.IO.Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "dotLLM", "tray.json");

    /// <summary>
    /// Reads the settings, returning defaults when the file is missing or unreadable.
    /// </summary>
    /// <remarks>
    /// A corrupt settings file returns defaults rather than throwing. A tray that refuses to start
    /// because of a bad JSON byte is unusable and offers the user no way to fix it — the only
    /// surface that could is the tray itself.
    /// </remarks>
    public TraySettings Load()
    {
        try
        {
            if (!File.Exists(_path))
                return new TraySettings().Normalized();
            var json = File.ReadAllText(_path);
            var settings = JsonSerializer.Deserialize(json, TraySettingsJsonContext.Default.TraySettings);
            return settings?.Normalized() ?? new TraySettings();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
        {
            return new TraySettings().Normalized();
        }
    }

    /// <summary>Writes the settings, creating the directory if needed.</summary>
    /// <param name="settings">Settings to persist.</param>
    public void Save(TraySettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        var directory = System.IO.Path.GetDirectoryName(_path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        File.WriteAllText(_path, JsonSerializer.Serialize(settings, TraySettingsJsonContext.Default.TraySettings));
    }
}
