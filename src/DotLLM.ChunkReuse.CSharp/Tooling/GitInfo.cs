using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;

namespace DotLLM.ChunkReuse.Tooling;

/// <summary>Git queries the selection tools need (<see cref="SelectionRequest.ChangedSince"/>); faked in tests.</summary>
public interface IGitInfo
{
    /// <summary>
    /// Files that differ between <paramref name="gitRef"/> and the working tree, as <b>absolute</b> paths: modified, added and
    /// deleted tracked files, and (for <see cref="GitCliInfo"/> by default) untracked files that are not ignored.
    /// </summary>
    /// <param name="gitRef">A revision, already checked with <see cref="GitCliInfo.ValidateRef"/>.</param>
    /// <param name="sinceMergeBase">Compare against the merge base of <paramref name="gitRef"/> and <c>HEAD</c> (see <see cref="SelectionRequest.ChangedSinceMergeBase"/>).</param>
    IReadOnlyList<string> ChangedFiles(string gitRef, bool sinceMergeBase = false);
}

/// <summary>A git command failed (not a repository, not a commit, no merge base, git missing, timeout).</summary>
public sealed class GitException : Exception
{
    /// <summary>Creates the exception.</summary>
    public GitException(string message) : base(message) { }

    /// <summary>Creates the exception.</summary>
    public GitException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>
/// <see cref="IGitInfo"/> over the git command line.
/// </summary>
/// <remarks>
/// <para>
/// <b>No injection.</b> Arguments go through <see cref="ProcessStartInfo.ArgumentList"/>, never a shell or a joined command
/// line, so quotes, <c>;</c>, <c>&amp;</c>, <c>$()</c> and spaces are plain characters of one argument. A revision that starts
/// with '-' (it would be read as an option, e.g. <c>--output=FILE</c> writes a file) or holds control characters is rejected
/// by <see cref="ValidateRef"/> before git runs. It is then resolved with <c>git rev-parse --verify --quiet
/// [--end-of-options] &lt;ref&gt;^{commit}</c>, and only the resulting object id reaches <c>git diff</c>, followed by
/// <c>--</c>. <c>--end-of-options</c> needs git 2.30 or later (<see cref="MinimumGitForEndOfOptions"/>); with an older or
/// unrecognised git it is left out, which is still safe because no revision starting with '-' gets that far.
/// </para>
/// <para>
/// <b>Paths.</b> Changed files are made absolute under the working directory's spelling of the repository top level
/// (working directory + <c>rev-parse --show-cdup</c>), not under <c>--show-toplevel</c>, which resolves junctions, symlinks
/// and <c>subst</c> drives. So a solution opened through a link gets paths under that link. A path that would leave the top
/// level is dropped. Submodules appear as their gitlink path (one entry, no files inside them); untracked files inside a
/// submodule are not listed.
/// </para>
/// <para>
/// <b>Commands</b> (in the top level): <c>diff --name-only -z --no-renames --no-ext-diff --no-textconv &lt;sha&gt; --</c>,
/// where <c>&lt;sha&gt;</c> is the revision or, with <c>sinceMergeBase</c>, <c>merge-base &lt;sha&gt; HEAD</c>; and, with
/// <see cref="IncludeUntracked"/>, <c>ls-files --others --exclude-standard -z</c>. NUL-separated output keeps unusual file
/// names (spaces, non-ASCII, quotes) exact.
/// </para>
/// <para>
/// <b>Environment.</b> Every inherited <c>GIT_*</c> variable is removed from the child's environment
/// (<see cref="SanitizeEnvironment"/>): <c>GIT_DIR</c>, <c>GIT_WORK_TREE</c>, <c>GIT_INDEX_FILE</c>,
/// <c>GIT_OBJECT_DIRECTORY</c>, <c>GIT_CEILING_DIRECTORIES</c>, <c>GIT_CONFIG_*</c>, <c>GIT_EXTERNAL_DIFF</c> and the like
/// would otherwise point git at another repository or index when the tool runs from a hook or wrapper. Git then runs with
/// <c>GIT_TERMINAL_PROMPT=0</c>, <c>GIT_OPTIONAL_LOCKS=0</c> and <c>core.fsmonitor=false</c>, and is killed (its own process
/// tree, by handle) after the timeout.
/// </para>
/// </remarks>
public sealed partial class GitCliInfo : IGitInfo
{
    private readonly string _workingDirectory;
    private readonly string _git;
    private readonly TimeSpan _timeout;
    private Version? _version;
    private bool _versionKnown;

    /// <summary>Creates it.</summary>
    /// <param name="workingDirectory">A directory inside the repository (e.g. the solution directory).</param>
    /// <param name="gitExecutable">The git executable (default <c>git</c> on PATH).</param>
    /// <param name="timeout">Per-command timeout (default 60 s).</param>
    public GitCliInfo(string workingDirectory, string gitExecutable = "git", TimeSpan? timeout = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(workingDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(gitExecutable);
        _workingDirectory = Path.GetFullPath(workingDirectory);
        _git = gitExecutable;
        _timeout = timeout ?? TimeSpan.FromSeconds(60);
    }

    /// <summary>The first git version whose <c>rev-parse</c> accepts <c>--end-of-options</c> (2.30).</summary>
    public static Version MinimumGitForEndOfOptions { get; } = new(2, 30);

    /// <summary>Also report untracked, not-ignored files (new files not yet added). Default true.</summary>
    public bool IncludeUntracked { get; init; } = true;

    /// <summary>The git version (<c>git version</c>), or null when its output is not recognised. Runs git once.</summary>
    /// <exception cref="GitException">Git could not be started or timed out.</exception>
    public Version? GitVersion
    {
        get
        {
            if (_versionKnown) return _version;
            (int code, string output, _) = Run(_workingDirectory, "version");
            _version = code == 0 ? ParseVersion(output) : null;
            _versionKnown = true;
            return _version;
        }
    }

    /// <summary>Parses <c>git version 2.53.0.windows.2</c> (major and minor, plus the patch when numeric); null if not recognised.</summary>
    public static Version? ParseVersion(string output)
    {
        ArgumentNullException.ThrowIfNull(output);
        Match m = VersionPattern().Match(output);
        if (!m.Success) return null;
        int major = int.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
        int minor = int.Parse(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
        return m.Groups[3].Success ? new Version(major, minor, int.Parse(m.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture)) : new Version(major, minor);
    }

    [GeneratedRegex(@"^git version (\d+)\.(\d+)(?:\.(\d+))?", RegexOptions.CultureInvariant)]
    private static partial Regex VersionPattern();

    /// <summary>The <c>rev-parse</c> arguments that resolve <paramref name="gitRef"/> to a commit; <c>--end-of-options</c> only for git ≥ 2.30.</summary>
    /// <exception cref="ArgumentException">See <see cref="ValidateRef"/>.</exception>
    internal static string[] RevParseArguments(string gitRef, Version? version)
    {
        ValidateRef(gitRef);
        return version is not null && version >= MinimumGitForEndOfOptions
            ? ["rev-parse", "--verify", "--quiet", "--end-of-options", gitRef + "^{commit}"]
            : ["rev-parse", "--verify", "--quiet", gitRef + "^{commit}"];
    }

    /// <summary>
    /// Removes every <c>GIT_*</c> variable (case-insensitive) from <paramref name="environment"/> and sets
    /// <c>GIT_TERMINAL_PROMPT=0</c> and <c>GIT_OPTIONAL_LOCKS=0</c>.
    /// </summary>
    internal static void SanitizeEnvironment(IDictionary<string, string?> environment)
    {
        foreach (string key in environment.Keys.Where(k => k.StartsWith("GIT_", StringComparison.OrdinalIgnoreCase)).ToArray())
            environment.Remove(key);
        environment["GIT_TERMINAL_PROMPT"] = "0";
        environment["GIT_OPTIONAL_LOCKS"] = "0";
    }

    /// <summary>
    /// Rejects a revision that git could read as an option or that is malformed: null, empty or blank, leading or trailing
    /// white space, a leading '-', or a control character (NUL, newline, …). Everything else is left to git.
    /// </summary>
    /// <exception cref="ArgumentException">The revision is rejected.</exception>
    public static void ValidateRef(string gitRef)
    {
        ArgumentNullException.ThrowIfNull(gitRef);
        if (string.IsNullOrWhiteSpace(gitRef))
            throw new ArgumentException("The git revision is empty.", nameof(gitRef));
        if (gitRef.Trim().Length != gitRef.Length)
            throw new ArgumentException($"The git revision '{Printable(gitRef)}' has leading or trailing white space.", nameof(gitRef));
        if (gitRef.StartsWith('-'))
            throw new ArgumentException($"The git revision '{Printable(gitRef)}' starts with '-', so git would read it as an option.", nameof(gitRef));
        if (gitRef.Any(char.IsControl))
            throw new ArgumentException($"The git revision '{Printable(gitRef)}' contains a control character.", nameof(gitRef));
    }

    /// <inheritdoc/>
    /// <exception cref="ArgumentException">See <see cref="ValidateRef"/>.</exception>
    /// <exception cref="DirectoryNotFoundException">The working directory does not exist.</exception>
    /// <exception cref="GitException">Not a repository, not a commit, no merge base, or git failed.</exception>
    public IReadOnlyList<string> ChangedFiles(string gitRef, bool sinceMergeBase = false)
    {
        ValidateRef(gitRef);
        if (!Directory.Exists(_workingDirectory))
            throw new DirectoryNotFoundException($"Git working directory not found: {_workingDirectory}");

        // The top level in the caller's spelling: --show-toplevel would resolve junctions, symlinks and subst drives.
        (int code, string output, string error) = Run(_workingDirectory, "rev-parse", "--show-cdup");
        if (code != 0)
            throw new GitException($"{_workingDirectory} is not in a git repository: {error.Trim()}");
        string top = Path.GetFullPath(Path.Combine(_workingDirectory, output.Trim()));

        (code, output, error) = Run(top, RevParseArguments(gitRef, GitVersion));
        string sha = output.Trim();
        if (code != 0 || !IsObjectId(sha))
            throw new GitException($"'{gitRef}' is not a commit in the repository {top}. {error.Trim()}".TrimEnd());
        if (sinceMergeBase)
        {
            (code, output, error) = Run(top, "merge-base", sha, "HEAD");
            string mergeBase = output.Trim();
            if (code != 0 || !IsObjectId(mergeBase))
                throw new GitException($"'{gitRef}' and HEAD have no merge base in the repository {top}. {error.Trim()}".TrimEnd());
            sha = mergeBase;
        }

        (code, output, error) = Run(top, "diff", "--name-only", "-z", "--no-renames", "--no-ext-diff", "--no-textconv", sha, "--");
        if (code != 0) throw new GitException($"git diff against {sha} failed in {top}: {error.Trim()}");
        var names = new List<string>(Split(output));
        if (IncludeUntracked)
        {
            (code, output, error) = Run(top, "ls-files", "--others", "--exclude-standard", "-z");
            if (code != 0) throw new GitException($"git ls-files failed in {top}: {error.Trim()}");
            names.AddRange(Split(output));
        }

        string topWithSep = Path.EndsInDirectorySeparator(top) ? top : top + Path.DirectorySeparatorChar;
        StringComparison cmp = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        return names
            .Select(n => Path.GetFullPath(Path.Combine(top, n)))
            .Where(p => p.StartsWith(topWithSep, cmp))
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToArray();
    }

    private static bool IsObjectId(string s) => s.Length is 40 or 64 && s.All(Uri.IsHexDigit);

    private static IEnumerable<string> Split(string nulSeparated) => nulSeparated.Split('\0', StringSplitOptions.RemoveEmptyEntries);

    private static string Printable(string s) => new(s.Select(c => char.IsControl(c) ? '?' : c).ToArray());

    /// <summary>Most characters read from git's stdout or stderr; more kills git and throws <see cref="GitException"/> (a hostile or huge repository must not fill memory).</summary>
    internal int MaxOutputChars { get; init; } = 64 * 1024 * 1024;

    private static async Task<string> ReadCapped(StreamReader reader, Process process, int maxChars, string verb)
    {
        var sb = new StringBuilder();
        var buffer = new char[8192];
        int n;
        while ((n = await reader.ReadAsync(buffer, 0, buffer.Length).ConfigureAwait(false)) > 0)
        {
            if (sb.Length + n > maxChars)
            {
                try { process.Kill(entireProcessTree: true); } catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception) { }
                throw new GitException($"git {verb} printed more than {maxChars} characters; stopped.");
            }
            sb.Append(buffer, 0, n);
        }
        return sb.ToString();
    }

    private (int Code, string Output, string Error) Run(string directory, params string[] args)
    {
        var psi = new ProcessStartInfo(_git)
        {
            WorkingDirectory = directory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = new UTF8Encoding(false),
            StandardErrorEncoding = new UTF8Encoding(false),
        };
        SanitizeEnvironment(psi.Environment);
        foreach (string a in new[] { "-c", "core.fsmonitor=false", "-c", "core.quotepath=off" }) psi.ArgumentList.Add(a);
        foreach (string a in args) psi.ArgumentList.Add(a);

        Process process;
        try
        {
            process = Process.Start(psi) ?? throw new GitException($"Could not start '{_git}'.");
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or DirectoryNotFoundException or InvalidOperationException
                                        or PlatformNotSupportedException)
        {
            throw new GitException($"Could not start '{_git}' in {directory} (is git installed and on PATH?): {ex.Message}", ex);
        }
        using (process)
        {
            process.StandardInput.Close();
            Task<string> stdout = ReadCapped(process.StandardOutput, process, MaxOutputChars, args[0]);
            Task<string> stderr = ReadCapped(process.StandardError, process, MaxOutputChars, args[0]);
            if (!process.WaitForExit(_timeout))
            {
                try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
                throw new GitException($"git {args[0]} timed out after {_timeout.TotalSeconds:F0} s in {directory}.");
            }
            process.WaitForExit();
            return (process.ExitCode, stdout.GetAwaiter().GetResult(), stderr.GetAwaiter().GetResult());
        }
    }
}
