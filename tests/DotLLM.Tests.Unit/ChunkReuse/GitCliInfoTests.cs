using System.Diagnostics;
using DotLLM.ChunkReuse.Tooling;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>Directory links for tests: a junction on Windows (no admin needed), a symlink elsewhere.</summary>
internal static class TestLinks
{
    public static bool TryCreateDirectoryLink(string link, string target)
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                var psi = new ProcessStartInfo("cmd") { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
                foreach (string a in new[] { "/c", "mklink", "/J", link, target }) psi.ArgumentList.Add(a);
                using Process p = Process.Start(psi)!;
                p.StandardOutput.ReadToEnd();
                p.StandardError.ReadToEnd();
                p.WaitForExit();
                return p.ExitCode == 0 && Directory.Exists(link);
            }
            Directory.CreateSymbolicLink(link, target);
            return Directory.Exists(link);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.ComponentModel.Win32Exception)
        {
            return false;
        }
    }

    /// <summary>Deletes a temp tree: links are removed without following them, read-only git objects are made writable.</summary>
    public static void DeleteTree(string dir)
    {
        if (!Directory.Exists(dir)) return;
        foreach (string d in Directory.EnumerateDirectories(dir))
        {
            if (new DirectoryInfo(d).LinkTarget is not null) Directory.Delete(d); // the link only
            else DeleteTree(d);
        }
        foreach (string f in Directory.EnumerateFiles(dir)) File.SetAttributes(f, FileAttributes.Normal);
        Directory.Delete(dir, recursive: true);
    }
}

/// <summary>A temporary git repository with one commit (tag <c>v1</c>): <c>src/a.cs</c>, <c>src/keep.cs</c>, <c>src/sub/sp ace ü.cs</c>, <c>.gitignore</c>.</summary>
internal sealed class TempGitRepo : IDisposable
{
    public TempGitRepo()
    {
        Dir = Path.Combine(Path.GetTempPath(), "gitcli-" + Guid.NewGuid().ToString("N"));
        Repo = Path.Combine(Dir, "repo");
        Directory.CreateDirectory(Path.Combine(Repo, "src", "sub"));
        Git("init", "-q");
        Write("src/a.cs", "class A { }\n");
        Write("src/keep.cs", "class K { }\n");
        Write("src/sub/sp ace ü.cs", "class U { }\n");
        Write(".gitignore", "*.ignored.cs\n");
        Git("add", "-A");
        Git("commit", "-q", "-m", "one");
        Git("tag", "v1");
    }

    public string Dir { get; }

    public string Repo { get; }

    public void Dispose() => TestLinks.DeleteTree(Dir);

    public void Write(string rel, string text) => File.WriteAllText(Path.Combine(Repo, rel), text);

    public string Abs(string rel) => Path.GetFullPath(Path.Combine(Repo, rel));

    public void Git(params string[] args) => RunGit(Repo, args);

    public static void RunGit(string dir, params string[] args)
    {
        var psi = new ProcessStartInfo("git") { WorkingDirectory = dir, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        foreach (string a in new[] { "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false" })
            psi.ArgumentList.Add(a);
        foreach (string a in args) psi.ArgumentList.Add(a);
        using Process p = Process.Start(psi)!;
        string err = p.StandardError.ReadToEnd();
        p.StandardOutput.ReadToEnd();
        p.WaitForExit();
        Assert.True(p.ExitCode == 0, $"git {string.Join(' ', args)}: {err}");
    }
}

/// <summary><see cref="GitCliInfo"/> against real temporary git repositories (needs <c>git</c> on PATH).</summary>
public sealed class GitCliInfoTests : IDisposable
{
    private readonly TempGitRepo _r = new();

    public void Dispose() => _r.Dispose();

    [Fact]
    public void ChangedFiles_ModifiedAndUntracked_AsAbsolutePaths_IgnoredLeftOut()
    {
        _r.Write("src/a.cs", "class A { int x; }\n");
        _r.Write("src/sub/sp ace ü.cs", "class U { int y; }\n");
        _r.Write("src/new.cs", "class N { }\n");
        _r.Write("src/x.ignored.cs", "class I { }\n");
        var git = new GitCliInfo(_r.Repo);
        IReadOnlyList<string> changed = git.ChangedFiles("HEAD");
        Assert.Equal(new[] { _r.Abs("src/a.cs"), _r.Abs("src/new.cs"), _r.Abs("src/sub/sp ace ü.cs") }.Order(StringComparer.Ordinal), changed.Order(StringComparer.Ordinal));
        Assert.Equal(changed, git.ChangedFiles("v1")); // a tag
        Assert.Equal(changed, new GitCliInfo(Path.Combine(_r.Repo, "src", "sub")).ChangedFiles("HEAD")); // from a subdirectory: whole repository
        Assert.DoesNotContain(_r.Abs("src/new.cs"), new GitCliInfo(_r.Repo) { IncludeUntracked = false }.ChangedFiles("HEAD"));
    }

    [Fact]
    public void ChangedFiles_CommittedChangesAndDeletes_SinceAnOlderRef()
    {
        _r.Write("src/a.cs", "class A { int z; }\n");
        File.Delete(Path.Combine(_r.Repo, "src", "keep.cs"));
        _r.Git("add", "-A");
        _r.Git("commit", "-q", "-m", "two");
        var git = new GitCliInfo(_r.Repo);
        Assert.Empty(git.ChangedFiles("HEAD"));
        Assert.Equal(new[] { _r.Abs("src/a.cs"), _r.Abs("src/keep.cs") }, git.ChangedFiles("HEAD~1").Order(StringComparer.Ordinal));
    }

    [Fact]
    public void ChangedFiles_SinceMergeBase_LeavesOutWhatTheOtherBranchChanged()
    {
        // other: v1 → changes keep.cs. This branch: v1 → changes a.cs (committed) and sub file (working tree).
        _r.Git("branch", "other", "v1");
        _r.Git("checkout", "-q", "other");
        _r.Write("src/keep.cs", "class K { int moved; }\n");
        _r.Git("commit", "-q", "-am", "other moves on");
        _r.Git("checkout", "-q", "-");
        _r.Write("src/a.cs", "class A { int mine; }\n");
        _r.Git("commit", "-q", "-am", "mine");
        _r.Write("src/sub/sp ace ü.cs", "class U { int wip; }\n");

        var git = new GitCliInfo(_r.Repo);
        // Two-dot (working tree vs other): also "changes" keep.cs, which only the other branch touched.
        Assert.Equal(new[] { _r.Abs("src/a.cs"), _r.Abs("src/keep.cs"), _r.Abs("src/sub/sp ace ü.cs") }, git.ChangedFiles("other").Order(StringComparer.Ordinal));
        // Merge base (other...HEAD, plus the working tree): only this branch's changes.
        Assert.Equal(new[] { _r.Abs("src/a.cs"), _r.Abs("src/sub/sp ace ü.cs") }, git.ChangedFiles("other", sinceMergeBase: true).Order(StringComparer.Ordinal));
        // Unrelated history: no merge base is a clear error.
        _r.Git("checkout", "-q", "--orphan", "lonely");
        _r.Git("commit", "-q", "-m", "orphan");
        Assert.Contains("merge base", Assert.Throws<GitException>(() => git.ChangedFiles("other", sinceMergeBase: true)).Message, StringComparison.Ordinal);
    }

    [SkippableFact]
    public void ChangedFiles_ThroughALinkedDirectory_AreUnderTheLinkNotTheTarget()
    {
        string link = Path.Combine(_r.Dir, "link");
        Skip.IfNot(TestLinks.TryCreateDirectoryLink(link, _r.Repo), "Cannot create a directory junction or symlink here.");
        _r.Write("src/a.cs", "class A { int x; }\n");
        IReadOnlyList<string> changed = new GitCliInfo(link).ChangedFiles("HEAD");
        Assert.Equal([Path.Combine(link, "src", "a.cs")], changed);
        // From a subdirectory reached through the link.
        Assert.Equal(changed, new GitCliInfo(Path.Combine(link, "src", "sub")).ChangedFiles("HEAD"));
    }

    [Theory]
    [InlineData("--output=PWNED")]
    [InlineData("--output")]
    [InlineData("-o")]
    [InlineData("--exec=whoami")]
    [InlineData("-")]
    public void ChangedFiles_OptionLikeRef_IsRejected_AndWritesNothing(string option)
    {
        var git = new GitCliInfo(_r.Repo);
        Assert.ThrowsAny<ArgumentException>(() => git.ChangedFiles(option));
        Assert.ThrowsAny<ArgumentException>(() => git.ChangedFiles(option, sinceMergeBase: true));
        Assert.False(File.Exists(Path.Combine(_r.Repo, "PWNED")));
        Assert.False(File.Exists(Path.Combine(_r.Repo, "src", "PWNED")));
    }

    [Theory]
    [InlineData("HEAD --output=PWNED")]
    [InlineData("HEAD;echo PWNED>PWNED")]
    [InlineData("HEAD & echo PWNED > PWNED")]
    [InlineData("$(touch PWNED)")]
    [InlineData("\"HEAD\" --output=PWNED")]
    [InlineData("no-such-branch")]
    public void ChangedFiles_RefThatIsNotACommit_IsAGitError_AndWritesNothing(string gitRef)
    {
        var git = new GitCliInfo(_r.Repo);
        var ex = Assert.Throws<GitException>(() => git.ChangedFiles(gitRef));
        Assert.Contains("not a commit", ex.Message, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Combine(_r.Repo, "PWNED")));
    }

    [Fact]
    public void ChangedFiles_NotARepository_IsAGitError()
    {
        string plain = Path.Combine(_r.Dir, "plain");
        Directory.CreateDirectory(plain);
        // A git repository above the temp directory would make this test meaningless; the temp dir is not one on this box.
        var ex = Assert.Throws<GitException>(() => new GitCliInfo(plain).ChangedFiles("HEAD"));
        Assert.Contains("repository", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Throws<DirectoryNotFoundException>(() => new GitCliInfo(Path.Combine(_r.Dir, "missing")).ChangedFiles("HEAD"));
    }

    [Fact]
    public void ChangedFiles_GitMissing_OrTimedOut_IsAGitError()
    {
        var missing = Assert.Throws<GitException>(() => new GitCliInfo(_r.Repo, gitExecutable: "no-such-git-" + Guid.NewGuid().ToString("N")).ChangedFiles("HEAD"));
        Assert.Contains("Could not start", missing.Message, StringComparison.Ordinal);
        var slow = Assert.Throws<GitException>(() => new GitCliInfo(_r.Repo, timeout: TimeSpan.FromTicks(1)).ChangedFiles("HEAD"));
        Assert.Contains("timed out", slow.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("git version 2.53.0.windows.2", 2, 53)]
    [InlineData("git version 2.25.1", 2, 25)]
    [InlineData("git version 2.30.0\n", 2, 30)]
    [InlineData("git version 3.0", 3, 0)]
    public void ParseVersion_ReadsMajorAndMinor(string output, int major, int minor)
    {
        Version v = GitCliInfo.ParseVersion(output)!;
        Assert.Equal((major, minor), (v.Major, v.Minor));
    }

    [Theory]
    [InlineData("")]
    [InlineData("hub version 2.14")]
    [InlineData("git version x.y")]
    public void ParseVersion_Unknown_IsNull(string output) => Assert.Null(GitCliInfo.ParseVersion(output));

    [Fact]
    public void RevParse_UsesEndOfOptions_OnlyFromGit230()
    {
        Assert.Contains("--end-of-options", GitCliInfo.RevParseArguments("main", new Version(2, 30)));
        Assert.Contains("--end-of-options", GitCliInfo.RevParseArguments("main", new Version(2, 53, 0)));
        Assert.DoesNotContain("--end-of-options", GitCliInfo.RevParseArguments("main", new Version(2, 29, 9)));
        Assert.DoesNotContain("--end-of-options", GitCliInfo.RevParseArguments("main", null));
        Assert.Equal("main^{commit}", GitCliInfo.RevParseArguments("main", null)[^1]);
        // Older git is still safe: the option-like refs never reach it.
        Assert.ThrowsAny<ArgumentException>(() => GitCliInfo.RevParseArguments("--output=x", null));
        Assert.True(GitCliInfo.MinimumGitForEndOfOptions == new Version(2, 30));
        Assert.NotNull(new GitCliInfo(_r.Repo).GitVersion);
    }

    [Fact]
    public void SanitizeEnvironment_RemovesEveryGitVariable_AndSetsNonInteractiveOnes()
    {
        var env = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase)
        {
            ["GIT_DIR"] = "x", ["git_work_tree"] = "x", ["GIT_INDEX_FILE"] = "x", ["GIT_OBJECT_DIRECTORY"] = "x", ["GIT_CEILING_DIRECTORIES"] = "x",
            ["GIT_EXTERNAL_DIFF"] = "x", ["GIT_CONFIG_COUNT"] = "1", ["PATH"] = "p", ["HOME"] = "h",
        };
        GitCliInfo.SanitizeEnvironment(env);
        Assert.Equal(new[] { "GIT_OPTIONAL_LOCKS", "GIT_TERMINAL_PROMPT", "HOME", "PATH" }, env.Keys.Order(StringComparer.Ordinal));
        Assert.Equal(("0", "0"), (env["GIT_TERMINAL_PROMPT"], env["GIT_OPTIONAL_LOCKS"]));
    }
}

[CollectionDefinition(nameof(GitEnvironmentCollection), DisableParallelization = true)]
public sealed class GitEnvironmentCollection;

/// <summary>Process-wide GIT_* variables (as set by a git hook or wrapper) must not redirect <see cref="GitCliInfo"/>. Not parallel: it sets environment variables.</summary>
[Collection(nameof(GitEnvironmentCollection))]
public sealed class GitCliInfoEnvironmentTests : IDisposable
{
    private static readonly string[] Redirecting = ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_CEILING_DIRECTORIES"];
    private readonly TempGitRepo _r = new();
    private readonly TempGitRepo _other = new();

    public void Dispose()
    {
        _r.Dispose();
        _other.Dispose();
    }

    [Fact]
    public void ChangedFiles_IgnoresInheritedGitDirWorkTreeAndIndex()
    {
        _r.Write("src/a.cs", "class A { int x; }\n");
        var saved = Redirecting.ToDictionary(k => k, Environment.GetEnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable("GIT_DIR", Path.Combine(_other.Repo, ".git"));
            Environment.SetEnvironmentVariable("GIT_WORK_TREE", _other.Repo);
            Environment.SetEnvironmentVariable("GIT_INDEX_FILE", Path.Combine(_other.Dir, "no-such-index"));
            Environment.SetEnvironmentVariable("GIT_CEILING_DIRECTORIES", _r.Dir);
            Assert.Equal([_r.Abs("src/a.cs")], new GitCliInfo(_r.Repo).ChangedFiles("HEAD"));
        }
        finally
        {
            foreach ((string k, string? v) in saved) Environment.SetEnvironmentVariable(k, v);
        }
    }
}
