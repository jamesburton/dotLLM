using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace DotLLM.Server.Tuning;

/// <summary>How a saved result relates to the current model, device and build.</summary>
public enum TuningMatchKind
{
    /// <summary>Everything matches.</summary>
    Hit,
    /// <summary>Only stale-level fields differ (driver, build, runtime, environment overrides, tuner protocol). Applied with a warning.</summary>
    Stale,
    /// <summary>The run did not finish. Applied with a warning.</summary>
    Partial,
    /// <summary>No usable result: absent, unreadable, checksum mismatch, or a miss-level field differs. Defaults run.</summary>
    Miss,
}

/// <summary>The outcome of <see cref="TuningStore.Find"/>.</summary>
/// <param name="Kind">The match kind.</param>
/// <param name="Result">The saved result for Hit, Stale and Partial; null on Miss.</param>
/// <param name="Path">The file looked at.</param>
/// <param name="Reason">Why (a miss field, or old and new values for a stale one); empty for a hit.</param>
public sealed record TuningMatch(TuningMatchKind Kind, TuningResult? Result, string Path, string Reason);

/// <summary>
/// Saved tuning results (#1902): <c>~/.dotllm/tuning/&lt;model-hash&gt;/&lt;scope&gt;-&lt;profile&gt;.json</c>, checksummed, matched
/// field by field against the current key.
/// </summary>
/// <param name="root">Store root; null is <c>DOTLLM_TUNING_DIR</c>, else <c>~/.dotllm/tuning</c>.</param>
public sealed class TuningStore(string? root = null)
{
    /// <summary>The file layout version; part of the key.</summary>
    public const int SchemaVersion = 1;

    /// <summary>The tuner protocol version; bumped when the measurement or gate rules change. Nothing measures yet, so this is 1.</summary>
    public const int ProtocolVersion = 1;

    private readonly string _root = root ?? DefaultRoot();

    /// <summary>The default store root.</summary>
    public static string DefaultRoot() =>
        Environment.GetEnvironmentVariable("DOTLLM_TUNING_DIR") is { Length: > 0 } d
            ? d
            : System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "tuning");

    /// <summary>The file a key's result lives in.</summary>
    public string PathFor(TuningKey key) =>
        System.IO.Path.Combine(_root, FileSafe(key.ModelHash ?? "unknown"), FileSafe(key.Scope ?? "unknown") + "-" + FileSafe(key.Profile ?? "chat") + ".json");

    private static string FileSafe(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (char c in s) sb.Append(char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.' ? c : '-');
        return sb.ToString();
    }

    /// <summary>Writes <paramref name="result"/> with its checksum, atomically (temp file, then move).</summary>
    public void Save(TuningResult result)
    {
        ArgumentNullException.ThrowIfNull(result.Key);
        var sealedResult = result with { Checksum = ComputeChecksum(result) };
        string path = PathFor(result.Key);
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(path)!);
        string tmp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        var indented = new TuningJsonContext(new JsonSerializerOptions(TuningJsonContext.Default.Options) { WriteIndented = true });
        File.WriteAllText(tmp, JsonSerializer.Serialize(sealedResult, indented.TuningResult));
        File.Move(tmp, path, overwrite: true);
    }

    /// <summary>SHA-256 over the canonical (compact, declared-order) JSON of the result with its checksum cleared.</summary>
    public static string ComputeChecksum(TuningResult r) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(
            JsonSerializer.Serialize(r with { Checksum = null }, TuningJsonContext.Default.TuningResult)))).ToLowerInvariant();

    /// <summary>Finds and matches the saved result for <paramref name="current"/>. Never throws for file content.</summary>
    public TuningMatch Find(TuningKey current)
    {
        string path = PathFor(current);
        try
        {
            if (!File.Exists(path)) return new(TuningMatchKind.Miss, null, path, "no saved result");
            TuningResult? r = JsonSerializer.Deserialize(File.ReadAllText(path), TuningJsonContext.Default.TuningResult);
            if (r?.Key is null || r.Settings is null) return new(TuningMatchKind.Miss, null, path, "saved result is incomplete (no key or settings)");
            if (r.Checksum is null || !string.Equals(r.Checksum, ComputeChecksum(r), StringComparison.OrdinalIgnoreCase))
                return new(TuningMatchKind.Miss, null, path, "checksum mismatch (file edited or damaged)");

            string? miss = MissReason(r.Key, current);
            if (miss is not null) return new(TuningMatchKind.Miss, null, path, miss);

            var stale = StaleReasons(r.Key, current);
            if (stale.Count > 0)
            {
                string reason = string.Join("; ", stale) + (r.Partial == true ? "; the run was also incomplete" : "");
                return new(TuningMatchKind.Stale, r, path, reason);
            }
            return r.Partial == true
                ? new(TuningMatchKind.Partial, r, path, "the tuning run did not finish")
                : new(TuningMatchKind.Hit, r, path, "");
        }
        catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException or NotSupportedException or ArgumentException)
        {
            return new(TuningMatchKind.Miss, null, path, "saved result unreadable: " + ex.Message);
        }
    }

    private static string? MissReason(TuningKey saved, TuningKey cur)
    {
        if (saved.SchemaVersion != SchemaVersion || cur.SchemaVersion != saved.SchemaVersion)
            return $"schema version {saved.SchemaVersion?.ToString() ?? "none"} (this build reads {SchemaVersion})";
        if (saved.ModelHash != cur.ModelHash) return "different model";
        if (saved.ModelModifiers != cur.ModelModifiers) return "different model modifiers (overlay, ternary adapters, MTP head or DFlash drafter)";
        if (saved.Scope != cur.Scope) return $"different device scope ({saved.Scope} vs {cur.Scope})";
        if (!DevicesEqual(saved.Devices, cur.Devices)) return "different device identity";
        if (saved.Profile != cur.Profile) return "different profile";
        if (saved.ProfileHash != cur.ProfileHash) return "profile definition changed";
        if (saved.KvQualityBudget != cur.KvQualityBudget) return $"different KV quality budget ({saved.KvQualityBudget} vs {cur.KvQualityBudget})";
        return null;
    }

    private static bool DevicesEqual(IReadOnlyList<DeviceFingerprint>? a, IReadOnlyList<DeviceFingerprint>? b)
    {
        if (a is null || b is null) return a is null && b is null;
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++)
        {
            // Identity only: driver, API and OS build are stale-level.
            if (a[i] with { DriverVersion = null, DriverId = null, DriverInfo = null, ApiVersion = null, OsBuild = null }
                != b[i] with { DriverVersion = null, DriverId = null, DriverInfo = null, ApiVersion = null, OsBuild = null })
                return false;
        }
        return true;
    }

    private static List<string> StaleReasons(TuningKey saved, TuningKey cur)
    {
        var reasons = new List<string>();
        var sd = saved.Devices ?? [];
        var cd = cur.Devices ?? [];
        for (int i = 0; i < Math.Min(sd.Count, cd.Count); i++)
        {
            Diff(reasons, "driver version", sd[i].DriverVersion, cd[i].DriverVersion);
            Diff(reasons, "driver id", sd[i].DriverId, cd[i].DriverId);
            Diff(reasons, "driver info", sd[i].DriverInfo, cd[i].DriverInfo);
            Diff(reasons, "graphics API version", sd[i].ApiVersion, cd[i].ApiVersion);
            Diff(reasons, "OS build", sd[i].OsBuild, cd[i].OsBuild);
        }
        var sp = saved.Perf ?? new PerfIdentity();
        var cp = cur.Perf ?? new PerfIdentity();
        Diff(reasons, "dotLLM release", sp.ReleaseVersion, cp.ReleaseVersion);
        Diff(reasons, "kernel files", sp.KernelFilesHash, cp.KernelFilesHash);
        Diff(reasons, "kernel assemblies", sp.KernelAssembliesHash, cp.KernelAssembliesHash);
        Diff(reasons, ".NET runtime", sp.RuntimeVersion, cp.RuntimeVersion);
        if (saved.EnvOverridesHash != cur.EnvOverridesHash)
        {
            var was = saved.EnvOverrides ?? [];
            var now = cur.EnvOverrides ?? [];
            var changed = was.Except(now).Concat(now.Except(was))
                .Select(p => p.Split('=')[0]).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToList();
            reasons.Add("environment overrides differ" + (changed.Count > 0
                ? ": " + string.Join(", ", changed.Select(n => $"{n} (tuned with {Value(was, n)}, now {Value(now, n)})"))
                : ""));
        }
        if (saved.ProtocolVersion != cur.ProtocolVersion)
            reasons.Add($"tuner protocol {saved.ProtocolVersion?.ToString() ?? "none"} (now {cur.ProtocolVersion?.ToString() ?? "none"})");
        return reasons;
    }

    private static string Value(IReadOnlyList<string> pairs, string name)
    {
        string? p = pairs.FirstOrDefault(x => x.StartsWith(name + "=", StringComparison.Ordinal));
        return p is null ? "unset" : p[(name.Length + 1)..];
    }

    private static void Diff(List<string> reasons, string what, string? was, string? now)
    {
        if (!string.Equals(was, now, StringComparison.Ordinal))
            reasons.Add($"{what} changed (tuned with {was ?? "none"}, now {now ?? "none"})");
    }
}
