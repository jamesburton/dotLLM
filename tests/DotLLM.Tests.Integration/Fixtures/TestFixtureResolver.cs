using System.Diagnostics.CodeAnalysis;
using System.Globalization;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Outcome of a fixture lookup: either a resolved path, or the full list of locations
/// that were probed (so the skip message can name every one of them — issue #308).
/// </summary>
internal sealed class FixtureLocation
{
    internal FixtureLocation(string? path, IReadOnlyList<string> probed)
    {
        Path = path;
        Probed = probed;
    }

    /// <summary>Resolved absolute path (file or directory), or <c>null</c> when not found.</summary>
    public string? Path { get; }

    /// <summary>Every location that was probed, in probe order.</summary>
    public IReadOnlyList<string> Probed { get; }

    /// <summary><c>true</c> when the fixture was found.</summary>
    [MemberNotNullWhen(true, nameof(Path))]
    public bool Found => Path is not null;

    /// <summary>
    /// Human-readable reason for skipping, naming <paramref name="description"/> and every
    /// probed location. Pass straight to <c>Skip.If(!loc.Found, loc.SkipMessage(...))</c>.
    /// </summary>
    public string SkipMessage(string description)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append(CultureInfo.InvariantCulture, $"Fixture not found: {description}. Probed:");
        foreach (string p in Probed)
            sb.Append(CultureInfo.InvariantCulture, $"{Environment.NewLine}  - {p}");
        return sb.ToString();
    }
}

/// <summary>
/// Shared fixture-path resolver for real-model test suites (issue #308).
/// </summary>
/// <remarks>
/// <para>Suites used to read a single <c>DOTLLM_*_GGUF</c> environment variable with no
/// fallback, so they skipped even when the model was already on disk in one of the caches the
/// rest of the suite uses. This resolver probes, in order:</para>
/// <list type="number">
///   <item><description>the explicit environment override (file path, or a directory
///     containing one of the candidate file names);</description></item>
///   <item><description>the dotLLM test cache — <c>$DOTLLM_TEST_CACHE_DIR</c> or
///     <c>~/.dotllm/test-cache/</c> — laid out as <c>{org}/{repo}/{file}</c>, plus the
///     interchangeable CLI store <c>~/.dotllm/models/{org}/{repo}/</c>;</description></item>
///   <item><description>the Hugging Face hub cache —
///     <c>{hub}/models--{org}--{repo}/snapshots/*/</c>, searched recursively because
///     multi-quant GGUF repos nest files under a per-quant subdirectory.</description></item>
/// </list>
/// <para>Every probe is recorded so a genuine miss reports <b>skipped</b> with a message
/// naming each path tried, rather than a bare "fixture not found".</para>
/// </remarks>
internal static class TestFixtureResolver
{
    /// <summary>Environment variable naming the Hugging Face hub cache directory directly.</summary>
    private const string HfHubCacheEnvVar = "HF_HUB_CACHE";

    /// <summary>Legacy alias for <see cref="HfHubCacheEnvVar"/>.</summary>
    private const string LegacyHfHubCacheEnvVar = "HUGGINGFACE_HUB_CACHE";

    /// <summary>Environment variable naming the Hugging Face home directory (hub cache is <c>{HF_HOME}/hub</c>).</summary>
    private const string HfHomeEnvVar = "HF_HOME";

    /// <summary>
    /// Resolves a fixture <b>file</b>.
    /// </summary>
    /// <param name="envVars">Explicit override environment variables, highest priority first.
    /// A value may point at the file itself or at a directory containing one of
    /// <paramref name="fileNames"/>.</param>
    /// <param name="org">Hugging Face repo owner (e.g. <c>microsoft</c>).</param>
    /// <param name="repo">Hugging Face repo name (e.g. <c>bitnet-b1.58-2B-4T-gguf</c>).</param>
    /// <param name="fileNames">Candidate file names, in preference order.</param>
    /// <param name="extraDirectories">Additional directories probed after the caches — for
    /// legacy conventional paths a suite historically documented.</param>
    public static FixtureLocation ResolveFile(
        IReadOnlyList<string> envVars,
        string org,
        string repo,
        IReadOnlyList<string> fileNames,
        IReadOnlyList<string>? extraDirectories)
    {
        ArgumentNullException.ThrowIfNull(envVars);
        ArgumentNullException.ThrowIfNull(fileNames);

        var probed = new List<string>();

        foreach (string envVar in envVars)
        {
            string? raw = Environment.GetEnvironmentVariable(envVar);
            if (string.IsNullOrWhiteSpace(raw))
            {
                probed.Add($"${envVar} (not set)");
                continue;
            }

            probed.Add($"${envVar}={raw}");
            if (File.Exists(raw))
                return new FixtureLocation(System.IO.Path.GetFullPath(raw), probed);

            if (Directory.Exists(raw))
            {
                string? hit = FirstExisting(raw, fileNames, probed);
                if (hit is not null)
                    return new FixtureLocation(hit, probed);
            }
        }

        foreach (string dir in DotLlmCacheDirectories(org, repo).Concat(extraDirectories ?? []))
        {
            string? hit = FirstExisting(dir, fileNames, probed);
            if (hit is not null)
                return new FixtureLocation(hit, probed);
        }

        foreach (string snapshot in HuggingFaceSnapshotDirectories(org, repo, probed))
        {
            string? hit = FirstExistingRecursive(snapshot, fileNames);
            if (hit is not null)
                return new FixtureLocation(hit, probed);
        }

        return new FixtureLocation(null, probed);
    }

    /// <summary>
    /// Convenience overload for the common single-override case.
    /// </summary>
    public static FixtureLocation ResolveFile(string envVar, string org, string repo, params string[] fileNames)
        => ResolveFile([envVar], org, repo, fileNames, extraDirectories: null);

    /// <summary>
    /// Resolves a fixture <b>directory</b> — e.g. a Hugging Face safetensors snapshot. A
    /// candidate directory qualifies when every name in <paramref name="requiredEntries"/>
    /// exists directly inside it (pass <c>config.json</c> and friends).
    /// </summary>
    /// <param name="envVars">Explicit override environment variables, highest priority first.
    /// A value may point at the directory itself or at a file inside it.</param>
    /// <param name="org">Hugging Face repo owner.</param>
    /// <param name="repo">Hugging Face repo name.</param>
    /// <param name="requiredEntries">Entry names that must all be present. Empty means
    /// "any existing directory qualifies".</param>
    /// <param name="extraDirectories">Additional directories probed after the caches — for
    /// legacy conventional paths a suite historically documented.</param>
    public static FixtureLocation ResolveDirectory(
        IReadOnlyList<string> envVars,
        string org,
        string repo,
        IReadOnlyList<string> requiredEntries,
        IReadOnlyList<string>? extraDirectories)
    {
        ArgumentNullException.ThrowIfNull(envVars);
        ArgumentNullException.ThrowIfNull(requiredEntries);

        var probed = new List<string>();

        foreach (string envVar in envVars)
        {
            string? raw = Environment.GetEnvironmentVariable(envVar);
            if (string.IsNullOrWhiteSpace(raw))
            {
                probed.Add($"${envVar} (not set)");
                continue;
            }

            probed.Add($"${envVar}={raw}");
            string candidate = File.Exists(raw)
                ? System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(raw))!
                : raw;
            if (QualifiesAsDirectory(candidate, requiredEntries))
                return new FixtureLocation(System.IO.Path.GetFullPath(candidate), probed);
        }

        foreach (string dir in DotLlmCacheDirectories(org, repo).Concat(extraDirectories ?? []))
        {
            probed.Add(dir);
            if (QualifiesAsDirectory(dir, requiredEntries))
                return new FixtureLocation(System.IO.Path.GetFullPath(dir), probed);
        }

        foreach (string snapshot in HuggingFaceSnapshotDirectories(org, repo, probed))
        {
            if (QualifiesAsDirectory(snapshot, requiredEntries))
                return new FixtureLocation(snapshot, probed);
        }

        return new FixtureLocation(null, probed);
    }

    /// <summary>Convenience overload for the common single-override case.</summary>
    public static FixtureLocation ResolveDirectory(string envVar, string org, string repo, params string[] requiredEntries)
        => ResolveDirectory([envVar], org, repo, requiredEntries, extraDirectories: null);

    // ── Probe sources ──

    /// <summary>
    /// <c>{cacheRoot}/{org}/{repo}</c> for the test cache and the interchangeable
    /// <c>~/.dotllm/models/</c> CLI store.
    /// </summary>
    private static IEnumerable<string> DotLlmCacheDirectories(string org, string repo)
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        yield return System.IO.Path.Combine(TestModelDownloader.CacheDirectory, org, repo);
        yield return System.IO.Path.Combine(home, ".dotllm", "models", org, repo);
    }

    /// <summary>
    /// Hugging Face hub snapshot directories for <c>models--{org}--{repo}</c>, newest first.
    /// Records the hub roots it looked at in <paramref name="probed"/>.
    /// </summary>
    private static IEnumerable<string> HuggingFaceSnapshotDirectories(string org, string repo, List<string> probed)
    {
        foreach (string hub in HuggingFaceHubRoots())
        {
            string repoDir = System.IO.Path.Combine(hub, $"models--{org}--{repo}");
            string snapshots = System.IO.Path.Combine(repoDir, "snapshots");
            if (!Directory.Exists(snapshots))
            {
                probed.Add(System.IO.Path.Combine(snapshots, "*"));
                continue;
            }

            string[] dirs;
            try
            {
                dirs = Directory.GetDirectories(snapshots);
            }
            catch (IOException)
            {
                continue;
            }
            catch (UnauthorizedAccessException)
            {
                continue;
            }

            Array.Sort(dirs, StringComparer.Ordinal);
            foreach (string d in dirs)
            {
                probed.Add(d);
                yield return d;
            }
        }
    }

    /// <summary>
    /// The Hugging Face hub cache root, following <c>huggingface_hub</c>'s own precedence:
    /// <c>HF_HUB_CACHE</c> (or its legacy alias) wins outright, else <c>{HF_HOME}/hub</c>, else
    /// <c>~/.cache/huggingface/hub</c>. These <b>replace</b> the default rather than adding to
    /// it — otherwise pointing the variables at an empty directory would not actually redirect
    /// the lookup, and a fixture could never be hidden from a test run.
    /// </summary>
    private static IEnumerable<string> HuggingFaceHubRoots()
    {
        foreach (string envVar in new[] { HfHubCacheEnvVar, LegacyHfHubCacheEnvVar })
        {
            string? raw = Environment.GetEnvironmentVariable(envVar);
            if (!string.IsNullOrWhiteSpace(raw))
            {
                yield return System.IO.Path.GetFullPath(raw);
                yield break;
            }
        }

        string? hfHome = Environment.GetEnvironmentVariable(HfHomeEnvVar);
        if (!string.IsNullOrWhiteSpace(hfHome))
        {
            yield return System.IO.Path.Combine(System.IO.Path.GetFullPath(hfHome), "hub");
            yield break;
        }

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        yield return System.IO.Path.Combine(home, ".cache", "huggingface", "hub");
    }

    // ── Probing primitives ──

    private static string? FirstExisting(string dir, IReadOnlyList<string> fileNames, List<string> probed)
    {
        if (fileNames.Count == 0)
        {
            probed.Add(dir);
            return null;
        }

        foreach (string name in fileNames)
        {
            string full = System.IO.Path.Combine(dir, name);
            probed.Add(full);
            if (File.Exists(full))
                return System.IO.Path.GetFullPath(full);
        }

        return null;
    }

    /// <summary>
    /// Multi-quant GGUF repos nest files one level down (<c>snapshots/{sha}/UD-Q4_K_M/*.gguf</c>),
    /// so the hub probe searches the whole snapshot rather than just its root.
    /// </summary>
    private static string? FirstExistingRecursive(string dir, IReadOnlyList<string> fileNames)
    {
        foreach (string name in fileNames)
        {
            string direct = System.IO.Path.Combine(dir, name);
            if (File.Exists(direct))
                return System.IO.Path.GetFullPath(direct);

            string[] hits;
            try
            {
                hits = Directory.GetFiles(dir, name, SearchOption.AllDirectories);
            }
            catch (IOException)
            {
                continue;
            }
            catch (UnauthorizedAccessException)
            {
                continue;
            }

            if (hits.Length > 0)
            {
                Array.Sort(hits, StringComparer.Ordinal);
                return System.IO.Path.GetFullPath(hits[0]);
            }
        }

        return null;
    }

    private static bool QualifiesAsDirectory(string dir, IReadOnlyList<string> requiredEntries)
    {
        if (!Directory.Exists(dir))
            return false;

        foreach (string entry in requiredEntries)
        {
            string full = System.IO.Path.Combine(dir, entry);
            if (!File.Exists(full) && !Directory.Exists(full))
                return false;
        }

        return true;
    }
}
