using System.Globalization;

namespace DotLLM.Tray.Hosting;

/// <summary>
/// Parameters the tray passes to a server it launches. These are <b>not</b> live settings: they
/// are baked into the child's command line and only change on a restart.
/// </summary>
/// <remarks>
/// Kept distinct from <see cref="Api.TraySettingsDto"/> on purpose. The tray has three tiers of
/// setting and conflating them is the easiest way to lie to the user:
/// <list type="number">
///   <item>Live, over <c>PUT /v1/settings</c>: keep-alive default, max resident, budget, sweep.</item>
///   <item>Launch-time, this type: host, port, admin gate, initial model, initial device.</item>
///   <item>Per-load, on <c>POST /v1/models/load</c>: device, GPU layers, KV cache types, keep-alive override.</item>
/// </list>
/// </remarks>
public sealed record ServerLaunchOptions
{
    /// <summary>Host to bind. Defaults to loopback.</summary>
    public string Host { get; init; } = "localhost";

    /// <summary>Port to bind.</summary>
    public int Port { get; init; } = 8080;

    /// <summary>Model to load at startup, or null to start bare and load from the tray.</summary>
    public string? Model { get; init; }

    /// <summary>Initial device (<c>cpu</c>, <c>gpu</c>, <c>gpu:0</c>), or null for the server default.</summary>
    public string? Device { get; init; }

    /// <summary>Initial GPU layer count, or null for the server default.</summary>
    public int? GpuLayers { get; init; }

    /// <summary>Initial KV-cache key type, or null for the server default.</summary>
    public string? CacheTypeK { get; init; }

    /// <summary>Initial KV-cache value type, or null for the server default.</summary>
    public string? CacheTypeV { get; init; }

    /// <summary>Server-wide default keep-alive in seconds, or null for the server default.</summary>
    public double? KeepAliveSeconds { get; init; }

    /// <summary>Maximum resident models, or null for the server default.</summary>
    public int? MaxResidentModels { get; init; }

    /// <summary>Residency byte budget, or null for the server default.</summary>
    public long? ResidentMemoryBudgetBytes { get; init; }

    /// <summary>Extra arguments appended verbatim, for anything the tray does not model.</summary>
    public IReadOnlyList<string> ExtraArguments { get; init; } = [];
}

/// <summary>Builds the <c>dotllm serve</c> command line for a tray-owned server.</summary>
public static class ServerLaunchSpecBuilder
{
    /// <summary>
    /// Renders <paramref name="options"/> as an argument list.
    /// </summary>
    /// <remarks>
    /// <c>--allow-model-admin</c> is <b>always</b> emitted and is not configurable. A tray-owned
    /// server exists to be administered by the tray; without the flag every write route answers
    /// 403 and the tray's model and settings controls are dead buttons against a server it started
    /// itself. (An <i>attached</i> external server is a different matter — the tray cannot choose
    /// its flags and must degrade gracefully instead.)
    /// <c>--no-browser</c> is likewise always emitted: the tray owns the "open the Web UI" verb,
    /// and a browser window appearing on login would defeat the point of a background tray.
    /// </remarks>
    /// <param name="options">Launch parameters.</param>
    public static IReadOnlyList<string> BuildArguments(ServerLaunchOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        var args = new List<string> { "serve" };

        // The positional [model] argument must come before the options.
        if (!string.IsNullOrWhiteSpace(options.Model))
            args.Add(options.Model);

        args.Add("--host");
        args.Add(options.Host);
        args.Add("--port");
        args.Add(options.Port.ToString(CultureInfo.InvariantCulture));
        args.Add("--allow-model-admin");
        args.Add("--no-browser");

        if (!string.IsNullOrWhiteSpace(options.Device))
        {
            args.Add("--device");
            args.Add(options.Device);
        }

        if (options.GpuLayers is { } gpuLayers)
        {
            args.Add("--gpu-layers");
            args.Add(gpuLayers.ToString(CultureInfo.InvariantCulture));
        }

        if (!string.IsNullOrWhiteSpace(options.CacheTypeK))
        {
            args.Add("--cache-type-k");
            args.Add(options.CacheTypeK);
        }

        if (!string.IsNullOrWhiteSpace(options.CacheTypeV))
        {
            args.Add("--cache-type-v");
            args.Add(options.CacheTypeV);
        }

        if (options.KeepAliveSeconds is { } keepAlive)
        {
            args.Add("--keep-alive");
            args.Add(keepAlive.ToString(CultureInfo.InvariantCulture));
        }

        if (options.MaxResidentModels is { } maxResident)
        {
            args.Add("--max-resident-models");
            args.Add(maxResident.ToString(CultureInfo.InvariantCulture));
        }

        if (options.ResidentMemoryBudgetBytes is { } budget)
        {
            args.Add("--resident-memory-budget");
            args.Add(budget.ToString(CultureInfo.InvariantCulture));
        }

        args.AddRange(options.ExtraArguments);
        return args;
    }

    /// <summary>Builds the base URL a server launched with these options will listen on.</summary>
    /// <param name="options">Launch parameters.</param>
    public static Uri BuildBaseAddress(ServerLaunchOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var host = options.Host is "0.0.0.0" or "*" or "+" ? "localhost" : options.Host;
        return new Uri($"http://{host}:{options.Port.ToString(CultureInfo.InvariantCulture)}/");
    }
}

/// <summary>Finds the dotLLM executable the tray should launch.</summary>
public static class DotLlmExecutableLocator
{
    /// <summary>Executable file name on Windows.</summary>
    public const string ExecutableName = "dotllm.exe";

    /// <summary>
    /// Resolves the executable: an explicit configured path wins, then a sibling of the tray's own
    /// executable, then <c>PATH</c>. Returns null when nothing is found.
    /// </summary>
    /// <remarks>
    /// The sibling probe uses <see cref="Environment.ProcessPath"/> rather than
    /// <c>Assembly.Location</c>, which is empty in a single-file publish — the very packaging the
    /// tray ships as. It matches the release archive layout, where <c>dotllm.exe</c> and the tray
    /// sit in the same extracted folder.
    /// </remarks>
    /// <param name="configuredPath">A user-configured absolute path, or null.</param>
    /// <param name="fileExists">File-existence probe, injectable for tests.</param>
    /// <param name="processPath">The tray's own executable path; defaults to <see cref="Environment.ProcessPath"/>.</param>
    /// <param name="pathVariable">The <c>PATH</c> value; defaults to the environment's.</param>
    public static string? Locate(
        string? configuredPath,
        Func<string, bool>? fileExists = null,
        string? processPath = null,
        string? pathVariable = null)
    {
        var exists = fileExists ?? File.Exists;

        if (!string.IsNullOrWhiteSpace(configuredPath) && exists(configuredPath))
            return configuredPath;

        var self = processPath ?? Environment.ProcessPath;
        if (!string.IsNullOrEmpty(self))
        {
            var directory = Path.GetDirectoryName(self);
            if (!string.IsNullOrEmpty(directory))
            {
                var sibling = Path.Combine(directory, ExecutableName);
                if (exists(sibling))
                    return sibling;
            }
        }

        var path = pathVariable ?? Environment.GetEnvironmentVariable("PATH");
        if (!string.IsNullOrEmpty(path))
        {
            foreach (var entry in path.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
            {
                var candidate = Path.Combine(entry.Trim(), ExecutableName);
                if (exists(candidate))
                    return candidate;
            }
        }

        return null;
    }
}
