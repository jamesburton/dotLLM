using System.Diagnostics;
using System.Text;

namespace DotLLM.ChunkReuse.Tooling;

/// <summary>Git queries the selection tools need (<see cref="SelectionRequest.ChangedSince"/>); faked in tests.</summary>
public interface IGitInfo
{
    /// <summary>
    /// Files that differ between <paramref name="gitRef"/> and the working tree, as <b>absolute</b> paths: modified, added and
    /// deleted tracked files, and (for <see cref="GitCliInfo"/> by default) untracked files that are not ignored.
    /// </summary>
    /// <param name="gitRef">A revision, already checked with <see cref="GitCliInfo.ValidateRef"/>.</param>
    IReadOnlyList<string> ChangedFiles(string gitRef);
}

/// <summary>A git command failed (not a repository, not a commit, git missing, timeout).</summary>
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
/// by <see cref="ValidateRef"/> before git runs; it is then resolved with
/// <c>git rev-parse --verify --quiet --end-of-options &lt;ref&gt;^{commit}</c>, and only the resulting object id reaches
/// <c>git diff</c>, followed by <c>--</c>.
/// </para>
/// <para>
/// <b>Commands.</b> <c>rev-parse --show-toplevel</c> (from the working directory), then, in the top level:
/// <c>diff --name-only -z --no-renames --no-ext-diff --no-textconv &lt;sha&gt; --</c> and, with <see cref="IncludeUntracked"/>,
/// <c>ls-files --others --exclude-standard -z</c>. NUL-separated output keeps unusual file names (spaces, non-ASCII, quotes)
/// exact. Paths are made absolute against the top level; a path that would leave it is dropped. Git runs with
/// <c>GIT_TERMINAL_PROMPT=0</c>, <c>GIT_OPTIONAL_LOCKS=0</c> and <c>core.fsmonitor=false</c> (no hook program is started) and is
/// killed (its own process tree, by handle) after the timeout.
/// </para>
/// </remarks>
public sealed class GitCliInfo : IGitInfo
{
    private readonly string _workingDirectory;
    private readonly string _git;
    private readonly TimeSpan _timeout;

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

    /// <summary>Also report untracked, not-ignored files (new files not yet added). Default true.</summary>
    public bool IncludeUntracked { get; init; } = true;

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
    /// <exception cref="GitException">Not a repository, not a commit, or git failed.</exception>
    public IReadOnlyList<string> ChangedFiles(string gitRef)
    {
        ValidateRef(gitRef);
        if (!Directory.Exists(_workingDirectory))
            throw new DirectoryNotFoundException($"Git working directory not found: {_workingDirectory}");

        (int code, string output, string error) = Run(_workingDirectory, "rev-parse", "--show-toplevel");
        if (code != 0)
            throw new GitException($"{_workingDirectory} is not in a git repository: {error.Trim()}");
        string top = Path.GetFullPath(output.Trim());

        (code, output, error) = Run(top, "rev-parse", "--verify", "--quiet", "--end-of-options", gitRef + "^{commit}");
        string sha = output.Trim();
        if (code != 0 || sha.Length is not (40 or 64) || !sha.All(Uri.IsHexDigit))
            throw new GitException($"'{gitRef}' is not a commit in the repository {top}. {error.Trim()}".TrimEnd());

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

    private static IEnumerable<string> Split(string nulSeparated) => nulSeparated.Split('\0', StringSplitOptions.RemoveEmptyEntries);

    private static string Printable(string s) => new(s.Select(c => char.IsControl(c) ? '?' : c).ToArray());

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
        psi.Environment["GIT_TERMINAL_PROMPT"] = "0";
        psi.Environment["GIT_OPTIONAL_LOCKS"] = "0";
        foreach (string a in new[] { "-c", "core.fsmonitor=false", "-c", "core.quotepath=off" }) psi.ArgumentList.Add(a);
        foreach (string a in args) psi.ArgumentList.Add(a);

        Process process;
        try
        {
            process = Process.Start(psi) ?? throw new GitException($"Could not start '{_git}'.");
        }
        catch (System.ComponentModel.Win32Exception ex)
        {
            throw new GitException($"Could not start '{_git}' (is git installed and on PATH?): {ex.Message}", ex);
        }
        using (process)
        {
            process.StandardInput.Close();
            Task<string> stdout = process.StandardOutput.ReadToEndAsync();
            Task<string> stderr = process.StandardError.ReadToEndAsync();
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
