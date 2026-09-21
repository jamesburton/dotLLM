using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.Tray.Updates;

/// <summary>A GitHub release, as much of it as the update check needs.</summary>
public sealed record GitHubRelease
{
    /// <summary>The git tag, e.g. <c>v0.4.0</c>.</summary>
    [JsonPropertyName("tag_name")]
    public string TagName { get; init; } = "";

    /// <summary>Display name.</summary>
    [JsonPropertyName("name")]
    public string? Name { get; init; }

    /// <summary>Release notes in Markdown — the changelog shown to the user.</summary>
    [JsonPropertyName("body")]
    public string? Body { get; init; }

    /// <summary>Whether GitHub marks this as a prerelease.</summary>
    [JsonPropertyName("prerelease")]
    public bool Prerelease { get; init; }

    /// <summary>Whether the release is still a draft.</summary>
    [JsonPropertyName("draft")]
    public bool Draft { get; init; }

    /// <summary>Browser URL of the release page.</summary>
    [JsonPropertyName("html_url")]
    public string HtmlUrl { get; init; } = "";
}

/// <summary>Serialization context for the release feed.</summary>
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
[JsonSerializable(typeof(GitHubRelease))]
[JsonSerializable(typeof(GitHubRelease[]))]
public sealed partial class UpdateJsonContext : JsonSerializerContext;

/// <summary>Outcome of an update check.</summary>
/// <param name="UpdateAvailable">Whether a newer release than the running version exists.</param>
/// <param name="CurrentVersion">The version the tray is running.</param>
/// <param name="LatestVersion">The newest eligible release found, or null when none was.</param>
/// <param name="Changelog">The release notes for <paramref name="LatestVersion"/>.</param>
/// <param name="ReleaseUrl">Browser URL of that release.</param>
public sealed record UpdateCheckResult(
    bool UpdateAvailable,
    string CurrentVersion,
    string? LatestVersion,
    string? Changelog,
    string? ReleaseUrl);

/// <summary>
/// Checks the GitHub releases feed for a newer dotLLM.
/// </summary>
/// <remarks>
/// <para>
/// <b>This type checks and reports. It never downloads and never applies.</b> That is a
/// deliberate v1 boundary, not an unfinished feature: the release pipeline
/// (<c>.github/workflows/release.yml</c>) publishes unsigned self-contained archives, so an
/// auto-applied update would hand the user an unsigned executable that SmartScreen flags as
/// "unrecognized app" and that Defender may quarantine mid-swap, leaving a half-replaced install.
/// Until the project has an Authenticode certificate, "open the release page and let the user
/// decide" is the only honest path. See <c>docs/TRAY.md</c>.
/// </para>
/// <para>
/// The check is opt-in (<see cref="Config.TraySettings.CheckForUpdates"/>, default false), so the
/// tray makes no network request to GitHub unless the user asks it to.
/// </para>
/// </remarks>
public sealed class UpdateChecker
{
    private readonly HttpClient _http;
    private readonly string _releasesUrl;

    /// <summary>The default releases endpoint for the dotLLM repository.</summary>
    public const string DefaultReleasesUrl = "https://api.github.com/repos/kkokosa/dotLLM/releases";

    /// <summary>Creates a checker.</summary>
    /// <param name="http">Transport. GitHub requires a User-Agent; the caller sets it.</param>
    /// <param name="releasesUrl">Releases feed URL. Overridable for tests.</param>
    public UpdateChecker(HttpClient http, string releasesUrl = DefaultReleasesUrl)
    {
        _http = http ?? throw new ArgumentNullException(nameof(http));
        _releasesUrl = releasesUrl;
    }

    /// <summary>
    /// Fetches the release feed and compares it against <paramref name="currentVersion"/>.
    /// </summary>
    /// <param name="currentVersion">The running version; see <see cref="CurrentVersion"/>.</param>
    /// <param name="includePrereleases">
    /// When false (the default) prereleases are ignored — unless the running version is itself a
    /// prerelease, in which case suppressing them would strand a preview user with no upgrade path.
    /// </param>
    /// <param name="ct">Cancellation token.</param>
    public async Task<UpdateCheckResult> CheckAsync(
        string currentVersion, bool includePrereleases = false, CancellationToken ct = default)
    {
        var current = ReleaseVersion.Parse(currentVersion);
        GitHubRelease[]? releases;
        try
        {
            releases = await _http
                .GetFromJsonAsync(_releasesUrl, UpdateJsonContext.Default.GitHubReleaseArray, ct)
                .ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is HttpRequestException or JsonException or NotSupportedException)
        {
            // An unreachable or malformed feed is not an update. Reporting "no update" is the
            // safe answer; reporting a fabricated one would prompt a needless reinstall.
            return new UpdateCheckResult(false, currentVersion, null, null, null);
        }

        if (releases is null || releases.Length == 0 || current is null)
            return new UpdateCheckResult(false, currentVersion, null, null, null);

        var allowPrerelease = includePrereleases || current.IsPrerelease;

        GitHubRelease? best = null;
        ReleaseVersion? bestVersion = null;
        foreach (var release in releases)
        {
            if (release.Draft)
                continue;

            var version = ReleaseVersion.Parse(release.TagName);
            if (version is null)
                continue;
            if (!allowPrerelease && (release.Prerelease || version.IsPrerelease))
                continue;
            if (bestVersion is not null && version.CompareTo(bestVersion) <= 0)
                continue;

            best = release;
            bestVersion = version;
        }

        if (best is null || bestVersion is null)
            return new UpdateCheckResult(false, currentVersion, null, null, null);

        var newer = bestVersion.CompareTo(current) > 0;
        return new UpdateCheckResult(
            newer,
            currentVersion,
            bestVersion.Original,
            newer ? best.Body : null,
            newer ? best.HtmlUrl : null);
    }

    /// <summary>
    /// The running tray version, from <c>AssemblyInformationalVersionAttribute</c> (which MinVer
    /// stamps), falling back to the assembly version.
    /// </summary>
    /// <param name="assembly">Assembly to read; defaults to the one declaring this type.</param>
    public static string CurrentVersion(System.Reflection.Assembly? assembly = null)
    {
        var target = assembly ?? typeof(UpdateChecker).Assembly;
        var informational = target
            .GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
            .OfType<System.Reflection.AssemblyInformationalVersionAttribute>()
            .FirstOrDefault()?.InformationalVersion;

        if (!string.IsNullOrWhiteSpace(informational))
            return informational;

        return target.GetName().Version?.ToString() ?? "0.0.0";
    }
}
