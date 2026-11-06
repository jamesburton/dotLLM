using System.Text;
using DotLLM.ChunkReuse.Tooling;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// A fake git executable in a temp directory (a .cmd file on Windows, a /bin/sh script elsewhere): it appends the arguments of
/// every call to <c>args.log</c> and prints the canned answer for the sub-command (<c>out.&lt;key&gt;</c>, bytes as written, so
/// NUL-separated lists work). No real git is needed.
/// </summary>
internal sealed class FakeGitDir : IDisposable
{
    public const string Prefix = "-c core.fsmonitor=false -c core.quotepath=off ";
    public const string Sha = "0123456789abcdef0123456789abcdef01234567";
    public const string OtherSha = "fedcba9876543210fedcba9876543210fedcba98";

    private const string CmdScript = """
        @echo off
        setlocal EnableDelayedExpansion
        set "ARGS=%*"
        >>"%~dp0args.log" echo(!ARGS!
        set "KEY="
        if not "!ARGS:show-cdup=!"=="!ARGS!" set "KEY=cdup"
        if not "!ARGS:merge-base=!"=="!ARGS!" set "KEY=mergebase"
        if not "!ARGS:--verify=!"=="!ARGS!" set "KEY=revparse"
        if not "!ARGS: diff =!"=="!ARGS!" set "KEY=diff"
        if not "!ARGS:ls-files=!"=="!ARGS!" set "KEY=lsfiles"
        if not "!ARGS:version=!"=="!ARGS!" set "KEY=version"
        if defined KEY if exist "%~dp0out.!KEY!" type "%~dp0out.!KEY!"
        exit /b 0
        """;

    private const string ShScript = """
        #!/bin/sh
        d=$(dirname "$0")
        args="$*"
        printf '%s\n' "$args" >> "$d/args.log"
        key=
        case "$args" in *show-cdup*) key=cdup ;; esac
        case "$args" in *merge-base*) key=mergebase ;; esac
        case "$args" in *--verify*) key=revparse ;; esac
        case "$args" in *" diff "*) key=diff ;; esac
        case "$args" in *ls-files*) key=lsfiles ;; esac
        case "$args" in *version*) key=version ;; esac
        if [ -n "$key" ] && [ -f "$d/out.$key" ]; then cat "$d/out.$key"; fi
        exit 0
        """;

    public FakeGitDir()
    {
        Dir = Path.Combine(Path.GetTempPath(), "fakegit-" + Guid.NewGuid().ToString("N"));
        Top = Path.Combine(Dir, "repo");
        Work = Path.Combine(Top, "sub");
        Directory.CreateDirectory(Work);
        Directory.CreateDirectory(Path.Combine(Dir, "bin"));
        if (OperatingSystem.IsWindows())
        {
            Exe = Path.Combine(Dir, "bin", "fakegit.cmd");
            File.WriteAllText(Exe, CmdScript.ReplaceLineEndings("\r\n") + "\r\n", new UTF8Encoding(false));
        }
        else
        {
            Exe = Path.Combine(Dir, "bin", "fakegit");
            File.WriteAllText(Exe, ShScript.ReplaceLineEndings("\n") + "\n", new UTF8Encoding(false));
            File.SetUnixFileMode(Exe, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        }
        // Defaults: git 2.45, the working directory is one level below the top, a commit, nothing changed.
        Answer("version", "git version 2.45.0\n");
        Answer("cdup", "../\n");
        Answer("revparse", Sha + "\n");
        Answer("mergebase", OtherSha + "\n");
        Answer("diff", "");
        Answer("lsfiles", "");
    }

    public string Dir { get; }

    /// <summary>The repository top level (one level above <see cref="Work"/>).</summary>
    public string Top { get; }

    /// <summary>The directory handed to <see cref="GitCliInfo"/>.</summary>
    public string Work { get; }

    public string Exe { get; }

    public void Answer(string key, string text) => File.WriteAllBytes(Path.Combine(Dir, "bin", "out." + key), new UTF8Encoding(false).GetBytes(text));

    /// <summary>One entry per git call so far (the arguments, space-joined).</summary>
    public string[] Calls
    {
        get
        {
            string log = Path.Combine(Dir, "bin", "args.log");
            return File.Exists(log) ? File.ReadAllLines(log).Where(l => l.Length > 0).ToArray() : [];
        }
    }

    public string Abs(string rel) => Path.GetFullPath(Path.Combine(Top, rel));

    public IReadOnlyList<string> Changed(string gitRef, bool sinceMergeBase = false)
    {
        var git = new GitCliInfo(Work, Exe);
        // Writing a script and running it at once can fail with "text file busy" on Linux when a parallel test forks meanwhile.
        for (int attempt = 0; ; attempt++)
        {
            try
            {
                return git.ChangedFiles(gitRef, sinceMergeBase);
            }
            catch (GitException ex) when (attempt < 5 && ex.Message.Contains("Could not start", StringComparison.Ordinal))
            {
                Thread.Sleep(100);
            }
        }
    }

    public void Dispose() => TestLinks.DeleteTree(Dir);
}

/// <summary>What <see cref="GitCliInfo"/> passes to git and how it treats the answers, observed through a fake git executable.</summary>
public sealed class GitInfoArgumentTests : IDisposable
{
    private readonly FakeGitDir _g = new();

    public void Dispose() => _g.Dispose();

    [Fact]
    public void EveryGitCall_DisablesFsmonitor_AndQuotepath()
    {
        _g.Answer("diff", "src/a.cs\0");
        _g.Answer("lsfiles", "src/b.cs\0");
        Assert.Equal([_g.Abs("src/a.cs"), _g.Abs("src/b.cs")], _g.Changed("HEAD", sinceMergeBase: true));

        string[] calls = _g.Calls;
        // Control: the fake really saw all six sub-commands (version, show-cdup, rev-parse, merge-base, diff, ls-files).
        Assert.Equal(6, calls.Length);
        foreach (string sub in new[] { " version", " --show-cdup"," rev-parse ", " merge-base ", " diff ", " ls-files" })
            Assert.Contains(calls, c => c.Contains(sub, StringComparison.Ordinal));
        Assert.All(calls, c => Assert.StartsWith(FakeGitDir.Prefix, c, StringComparison.Ordinal));
    }

    [Fact]
    public void Diff_IsRun_WithoutExternalDiffOrTextconv_AgainstTheResolvedSha()
    {
        _g.Changed("HEAD");
        string[] calls = _g.Calls;
        // Control: the diff call was made at all, with the commit the fake resolved HEAD to.
        string diff = Assert.Single(calls, c => c.Contains(" diff ", StringComparison.Ordinal));
        Assert.Equal(FakeGitDir.Prefix + "diff --name-only -z --no-renames --no-ext-diff --no-textconv " + FakeGitDir.Sha + " --", diff);
        Assert.Equal(FakeGitDir.Prefix + "ls-files --others --exclude-standard -z", Assert.Single(calls, c => c.Contains(" ls-files", StringComparison.Ordinal)));

        // With the merge base, that answer is what the diff compares against.
        _g.Changed("HEAD", sinceMergeBase: true);
        string[] after = _g.Calls[calls.Length..];
        Assert.Equal(FakeGitDir.Prefix + "merge-base " + FakeGitDir.Sha + " HEAD", Assert.Single(after, c => c.Contains(" merge-base ", StringComparison.Ordinal)));
        Assert.EndsWith(" --no-ext-diff --no-textconv " + FakeGitDir.OtherSha + " --", Assert.Single(after, c => c.Contains(" diff ", StringComparison.Ordinal)), StringComparison.Ordinal);
    }

    [Fact]
    public void RevParse_UsesEndOfOptions_OnlyWhenTheGitVersionAllowsIt()
    {
        _g.Changed("HEAD");
        // (cmd treats '^' in an argument as its escape character, so the ^{commit} suffix is not compared here.)
        string recent = Assert.Single(_g.Calls, c => c.Contains(" rev-parse ", StringComparison.Ordinal) && c.Contains("--verify", StringComparison.Ordinal));
        Assert.StartsWith(FakeGitDir.Prefix + "rev-parse --verify --quiet --end-of-options HEAD", recent, StringComparison.Ordinal);

        using var old = new FakeGitDir();
        old.Answer("version", "git version 2.29.1\n");
        old.Changed("HEAD");
        string older = Assert.Single(old.Calls, c => c.Contains("--verify", StringComparison.Ordinal));
        Assert.StartsWith(FakeGitDir.Prefix + "rev-parse --verify --quiet HEAD", older, StringComparison.Ordinal);
        Assert.DoesNotContain("--end-of-options", older, StringComparison.Ordinal);
    }

    [Fact]
    public void Paths_OutsideTheTopLevel_AreDropped_AndDuplicatesCollapse()
    {
        // Precondition: the "outside" spellings really leave the top level (and one is a sibling that merely shares its prefix).
        string top = _g.Top.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        Assert.False(_g.Abs("../outside.cs").StartsWith(top, StringComparison.Ordinal));
        Assert.False(_g.Abs("../repo-evil/x.cs").StartsWith(top, StringComparison.Ordinal));
        Assert.False(_g.Abs("../../up.cs").StartsWith(top, StringComparison.Ordinal));

        _g.Answer("diff", "src/a.cs\0../outside.cs\0../repo-evil/x.cs\0src/../src/a.cs\0src/b.cs\0../../up.cs\0");
        _g.Answer("lsfiles", "src/b.cs\0src/a.cs\0new.cs\0../outside.cs\0..\0");
        IReadOnlyList<string> changed = _g.Changed("HEAD");

        // Control: the answers reached the code (the repository top is the working directory's parent, via show-cdup).
        Assert.Contains(" --no-textconv " + FakeGitDir.Sha + " --", Assert.Single(_g.Calls, c => c.Contains(" diff ", StringComparison.Ordinal)), StringComparison.Ordinal);
        Assert.Equal([_g.Abs("new.cs"), _g.Abs("src/a.cs"), _g.Abs("src/b.cs")], changed);
    }

    public static TheoryData<string> NotObjectIds => new()
    {
        "not-a-sha",
        new string('z', 40),
        new string('0', 39) + "g",
        new string('a', 39),
        new string('a', 41),
        new string('a', 63),
        "",
    };

    [Theory]
    [MemberData(nameof(NotObjectIds))]
    public void RevParseAnswer_ThatIsNotAnObjectId_IsAGitError_AndNothingElseRuns(string answer)
    {
        _g.Answer("revparse", answer + "\n");
        var ex = Assert.Throws<GitException>(() => _g.Changed("HEAD"));
        Assert.Contains("not a commit", ex.Message, StringComparison.Ordinal);
        string[] calls = _g.Calls;
        // Control: rev-parse ran and answered with this value; the diff never started.
        Assert.Contains(calls, c => c.Contains("--verify", StringComparison.Ordinal));
        Assert.DoesNotContain(calls, c => c.Contains(" diff ", StringComparison.Ordinal));
    }

    [Theory]
    [MemberData(nameof(NotObjectIds))]
    public void MergeBaseAnswer_ThatIsNotAnObjectId_IsAGitError_AndNothingElseRuns(string answer)
    {
        _g.Answer("mergebase", answer + "\n");
        var ex = Assert.Throws<GitException>(() => _g.Changed("HEAD", sinceMergeBase: true));
        Assert.Contains("merge base", ex.Message, StringComparison.Ordinal);
        string[] calls = _g.Calls;
        Assert.Contains(calls, c => c.Contains(" merge-base ", StringComparison.Ordinal));
        Assert.DoesNotContain(calls, c => c.Contains(" diff ", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("0123456789abcdef0123456789abcdef01234567")]
    [InlineData("0123456789ABCDEF0123456789ABCDEF01234567")]
    [InlineData("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")] // SHA-256 object format
    public void RevParseAnswer_ThatIsAnObjectId_IsUsedForTheDiff(string answer)
    {
        _g.Answer("revparse", answer + "\n");
        _g.Answer("diff", "src/a.cs\0");
        Assert.Equal([_g.Abs("src/a.cs")], _g.Changed("HEAD"));
        Assert.EndsWith(" --no-textconv " + answer + " --", Assert.Single(_g.Calls, c => c.Contains(" diff ", StringComparison.Ordinal)), StringComparison.Ordinal);
    }
}
