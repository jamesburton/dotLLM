using System.Diagnostics;
using System.Reflection;
using System.Text.RegularExpressions;

namespace DotLLM.Cli.Benchmarking;

/// <summary>
/// Environment / metadata helpers for the <c>bench</c> command: quant-label and
/// model-name inference from GGUF file names, and the point-in-time runtime
/// version (commit) used for the <c>runtime_version</c> results.csv column.
/// </summary>
public static partial class BenchEnvironment
{
    [GeneratedRegex(@"[.\-]((?:I?Q|IQ|BF|F|MXFP|UD-Q)[\w\-]+?)\.gguf$", RegexOptions.IgnoreCase)]
    private static partial Regex QuantSuffixRegex();

    /// <summary>
    /// Infers the quantization label from a GGUF file name (e.g.
    /// <c>SmolLM-135M.Q8_0.gguf</c> → <c>Q8_0</c>). An explicit
    /// <paramref name="quantFlag"/> wins. Returns <c>unknown</c> when neither matches.
    /// </summary>
    public static string InferQuantLabel(string ggufPath, string? quantFlag = null)
    {
        if (!string.IsNullOrEmpty(quantFlag))
            return quantFlag;
        var match = QuantSuffixRegex().Match(Path.GetFileName(ggufPath));
        return match.Success ? match.Groups[1].Value : "unknown";
    }

    /// <summary>
    /// Derives the results.csv <c>model</c> column from a GGUF file name: extension
    /// removed, and a trailing <c>.{quant}</c> / <c>-{quant}</c> suffix stripped when
    /// it matches <paramref name="quant"/> (e.g. <c>SmolLM-135M.Q8_0.gguf</c> +
    /// <c>Q8_0</c> → <c>SmolLM-135M</c>).
    /// </summary>
    public static string InferModelName(string ggufPath, string quant)
    {
        string name = Path.GetFileNameWithoutExtension(ggufPath);
        if (!string.IsNullOrEmpty(quant) && quant != "unknown")
        {
            foreach (char sep in new[] { '.', '-' })
            {
                string suffix = sep + quant;
                if (name.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                    return name[..^suffix.Length];
            }
        }
        return name;
    }

    /// <summary>
    /// Point-in-time runtime version for the results.csv <c>runtime_version</c> column
    /// (convention: <c>dev-96a892bd</c>). Resolution order: the
    /// <c>DOTLLM_BENCH_COMMIT</c> environment variable, <c>git</c> in the current
    /// directory (branch tag + short SHA), the assembly informational version's
    /// <c>+sha</c> suffix, then <c>unknown</c>.
    /// </summary>
    public static string ResolveRuntimeVersion()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_BENCH_COMMIT");
        if (!string.IsNullOrWhiteSpace(env))
            return env.Trim();

        string? sha = TryGit("rev-parse --short=8 HEAD");
        if (sha is not null)
        {
            string? branch = TryGit("rev-parse --abbrev-ref HEAD");
            return $"{TagFromBranch(branch)}-{sha}";
        }

        // MinVer / SourceLink embed the commit after '+' in the informational version.
        string? info = Assembly.GetExecutingAssembly()
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        int plus = info?.IndexOf('+') ?? -1;
        if (info is not null && plus >= 0 && plus + 1 < info.Length)
        {
            string hash = info[(plus + 1)..];
            return "rel-" + hash[..Math.Min(8, hash.Length)];
        }

        return "unknown";
    }

    /// <summary>Compact branch tag: <c>dev</c>/<c>main</c> pass through; <c>issue/140-foo</c> → <c>issue140</c>.</summary>
    public static string TagFromBranch(string? branch)
    {
        if (string.IsNullOrWhiteSpace(branch) || branch == "HEAD")
            return "dev";
        branch = branch.Trim();
        if (branch is "dev" or "main")
            return branch;
        var m = Regex.Match(branch, @"^issue/(\d+)");
        if (m.Success)
            return "issue" + m.Groups[1].Value;
        string sanitized = Regex.Replace(branch, @"[^A-Za-z0-9]", "");
        return sanitized.Length == 0 ? "dev" : sanitized[..Math.Min(12, sanitized.Length)];
    }

    private static string? TryGit(string args)
    {
        try
        {
            var psi = new ProcessStartInfo("git", args)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using var proc = Process.Start(psi);
            if (proc is null) return null;
            string output = proc.StandardOutput.ReadToEnd().Trim();
            if (!proc.WaitForExit(3000)) { try { proc.Kill(); } catch { } return null; }
            return proc.ExitCode == 0 && output.Length > 0 ? output : null;
        }
        catch
        {
            return null;
        }
    }
}
