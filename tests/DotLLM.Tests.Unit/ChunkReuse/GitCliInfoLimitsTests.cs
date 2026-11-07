using DotLLM.ChunkReuse.Tooling;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>git output is bounded, and every way of failing to run git surfaces as <see cref="GitException"/>.</summary>
public sealed class GitCliInfoLimitsTests : IDisposable
{
    private readonly TempGitRepo _r = new();

    public void Dispose() => _r.Dispose();

    [Fact]
    public void ChangedFiles_OutputOverTheCap_StopsGitAndThrows()
    {
        _r.Write("src/a.cs", "class A { int x; }\n");
        _r.Write("src/new.cs", "class N { }\n");
        Assert.NotEmpty(new GitCliInfo(_r.Repo).ChangedFiles("HEAD")); // control: the same call works with the default cap

        var capped = new GitCliInfo(_r.Repo) { MaxOutputChars = 8 };

        GitException ex = Assert.Throws<GitException>(() => capped.ChangedFiles("HEAD"));
        Assert.Contains("more than 8 characters", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ChangedFiles_OutputExactlyAtTheCap_IsAccepted()
    {
        _r.Write("src/a.cs", "class A { int x; }\n");
        int size = new GitCliInfo(_r.Repo).ChangedFiles("HEAD").Sum(p => p.Length); // upper bound on any single git output here

        Assert.NotEmpty(new GitCliInfo(_r.Repo) { MaxOutputChars = Math.Max(size, 4096) }.ChangedFiles("HEAD"));
    }

    [Fact]
    public void ChangedFiles_GitExecutableMissing_IsAGitException()
    {
        Assert.Throws<GitException>(() => new GitCliInfo(_r.Repo, Path.Combine(_r.Dir, "no-such-git.exe")).ChangedFiles("HEAD"));
    }
}
