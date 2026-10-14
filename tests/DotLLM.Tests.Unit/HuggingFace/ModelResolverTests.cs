using DotLLM.HuggingFace;
using Xunit;

namespace DotLLM.Tests.Unit.HuggingFace;

/// <summary>Issue #714: one resolver for every place a model is named.</summary>
public sealed class ModelResolverTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-714-" + Guid.NewGuid().ToString("N"));
    private string Hub => Path.Combine(_root, "hub");
    private string Mirror => Path.Combine(_root, "models");

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); } catch { }
    }

    /// <summary>Writes a file of <paramref name="size"/> bytes into the hub-cache layout (blob + snapshot link) and optionally the mirror.</summary>
    private string AddHub(string repo, string file, int size, bool mirror = false, char fill = 'x')
    {
        string repoDir = HubCache.RepoDirectory(repo, Hub);
        string blob = Path.Combine(repoDir, "blobs", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.GetDirectoryName(blob)!);
        File.WriteAllBytes(blob, Enumerable.Repeat((byte)fill, size).ToArray());
        string snap = HubCache.SnapshotFilePath(repo, "c0ffee", file, Hub);
        Directory.CreateDirectory(Path.GetDirectoryName(snap)!);
        HubCache.LinkOrCopy(blob, snap);
        if (mirror)
        {
            string m = HubCache.MirrorPath(repo, file, Mirror);
            Directory.CreateDirectory(Path.GetDirectoryName(m)!);
            HubCache.LinkOrCopy(blob, m);
        }
        return snap;
    }

    [Theory]
    [InlineData("owner/repo", "owner/repo", null, null)]
    [InlineData("owner/repo:Q4_K_M", "owner/repo", null, "Q4_K_M")]
    [InlineData("owner/repo:latest", "owner/repo", null, null)]
    [InlineData("owner/repo/file.gguf", "owner/repo", "file.gguf", null)]
    [InlineData("hf.co/owner/repo:Q8_0", "owner/repo", null, "Q8_0")]
    [InlineData("hf://owner/repo", "owner/repo", null, null)]
    [InlineData("https://huggingface.co/owner/repo", "owner/repo", null, null)]
    [InlineData("owner/repo/sub/file.gguf", "owner/repo", "sub/file.gguf", null)]
    public void Parse_RepoForms(string arg, string repo, string? file, string? tag)
    {
        var r = ModelResolver.Parse(arg);
        Assert.Equal(repo, r.RepoId);
        Assert.Equal(file, r.Filename);
        Assert.Equal(tag, r.Tag);
    }

    [Theory]
    [InlineData("my-model")]
    [InlineData("Tev1-4B-experimental-Q4_K_M")]
    [InlineData(@"C:\models\x.gguf")]
    [InlineData("/models/x.gguf")]
    public void Parse_NamesAndPathsAreNotRepos(string arg) => Assert.False(ModelResolver.Parse(arg).IsRepo);

    [Fact]
    public void Enumerate_FindsHubCacheOnlyModels_AndListsLinkedFilesOnce()
    {
        AddHub("acme/only-hub", "a-Q4_K_M.gguf", 100);
        AddHub("acme/both", "b-Q8_0.gguf", 200, mirror: true);

        var all = ModelResolver.EnumerateLocal(Mirror, Hub);

        Assert.Contains(all, m => m.RepoId == "acme/only-hub" && m.Filename == "a-Q4_K_M.gguf");
        Assert.Single(all, m => m.RepoId == "acme/both");   // mirror hardlink + snapshot link = one model
        Assert.Equal(2, all.Count);
    }

    [Fact]
    public void Enumerate_SkipsMultimodalProjectors_AndDanglingLinks()
    {
        AddHub("acme/vl", "mmproj-F16.gguf", 50);
        AddHub("acme/vl", "vl-Q4_K_M.gguf", 100);
        // A zero-byte snapshot entry is a dangling/partial link, not a model.
        string dangling = HubCache.SnapshotFilePath("acme/vl", "c0ffee", "empty.gguf", Hub);
        File.WriteAllBytes(dangling, []);

        var files = ModelResolver.EnumerateLocal(Mirror, Hub).Select(m => m.Filename).ToList();
        Assert.Equal(["vl-Q4_K_M.gguf"], files);
    }

    [SkippableFact]
    public void Enumerate_FollowsSymlinkedSnapshots_LikeHfHubOnWindows()
    {
        string repoDir = HubCache.RepoDirectory("acme/linked", Hub);
        string blob = Path.Combine(repoDir, "blobs", "abc");
        Directory.CreateDirectory(Path.GetDirectoryName(blob)!);
        File.WriteAllBytes(blob, new byte[1234]);
        string snap = HubCache.SnapshotFilePath("acme/linked", "c0ffee", "l-Q4_K_M.gguf", Hub);
        Directory.CreateDirectory(Path.GetDirectoryName(snap)!);
        try { File.CreateSymbolicLink(snap, blob); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { Skip.If(true, "symlinks need developer mode / elevation here"); }

        var m = Assert.Single(ModelResolver.EnumerateLocal(Mirror, Hub));
        Assert.Equal(1234, m.SizeBytes);   // the target's size, not the link's
        Assert.Equal(snap, ModelResolver.ResolveLocal("acme/linked", null, Mirror, Hub));
    }

    [Fact]
    public void ResolveLocal_RepoPicksQ4KM_NotTheLargestFile()
    {
        AddHub("acme/multi", "m-Q8_0.gguf", 900);
        string q4 = AddHub("acme/multi", "m-Q4_K_M.gguf", 300);
        AddHub("acme/multi", "m-Q2_K.gguf", 100);

        Assert.Equal(q4, ModelResolver.ResolveLocal("acme/multi", null, Mirror, Hub));
    }

    [Fact]
    public void ResolveLocal_TagAndLegacyQuantSelectTheMatchingFile()
    {
        AddHub("acme/multi", "m-Q4_K_M.gguf", 300);
        string q8 = AddHub("acme/multi", "m-Q8_0.gguf", 900);

        Assert.Equal(q8, ModelResolver.ResolveLocal("acme/multi:Q8_0", null, Mirror, Hub));
        Assert.Equal(q8, ModelResolver.ResolveLocal("acme/multi", "q8_0", Mirror, Hub));
        Assert.Null(ModelResolver.ResolveLocal("acme/multi:Q3_K_L", null, Mirror, Hub));
    }

    [Fact]
    public void ResolveLocal_BareNameMatchesFileStem_AsReportedByModelsEndpoint()
    {
        string p = AddHub("acme/named", "Tev1-4B-Q4_K_M.gguf", 100);
        Assert.Equal(p, ModelResolver.ResolveLocal("Tev1-4B-Q4_K_M", null, Mirror, Hub));
        Assert.Equal(p, ModelResolver.ResolveLocal("tev1-4b-q4_k_m.gguf", null, Mirror, Hub));
        Assert.Null(ModelResolver.ResolveLocal("nonexistent", null, Mirror, Hub));
    }

    [Fact]
    public void ResolveLocal_ExplicitFileInRepo_AndDirectPath()
    {
        string p = AddHub("acme/multi", "m-Q8_0.gguf", 900);
        AddHub("acme/multi", "m-Q4_K_M.gguf", 300);
        Assert.Equal(p, ModelResolver.ResolveLocal("acme/multi/m-Q8_0.gguf", null, Mirror, Hub));

        string direct = Path.Combine(_root, "x.gguf");
        File.WriteAllText(direct, "g");
        Assert.Equal(Path.GetFullPath(direct), ModelResolver.ResolveLocal(direct, null, Mirror, Hub));
    }

    [Fact]
    public void ChooseFile_NeverPicksALaterShard()
    {
        AddHub("acme/split", "big-Q4_K_M-00001-of-00002.gguf", 100);
        AddHub("acme/split", "big-Q4_K_M-00002-of-00002.gguf", 500);
        var all = ModelResolver.EnumerateLocal(Mirror, Hub);
        Assert.EndsWith("00001-of-00002.gguf", ModelResolver.ChooseFile(all, null)!.Filename);
    }

    [Fact]
    public void ChooseRemoteFile_AppliesTagThenPreference()
    {
        var files = new (string, long)[] { ("a-Q8_0.gguf", 9), ("a-Q4_K_M.gguf", 4), ("mmproj-a-F16.gguf", 1), ("README.md", 1) };
        Assert.Equal("a-Q4_K_M.gguf", ModelResolver.ChooseRemoteFile(files, null));
        Assert.Equal("a-Q8_0.gguf", ModelResolver.ChooseRemoteFile(files, "q8_0"));
        Assert.Null(ModelResolver.ChooseRemoteFile(files, "IQ1_S"));
    }

    [Fact]
    public async Task PullAsync_RejectsNonRepoReferences()
    {
        using var client = new HuggingFaceClient();
        using var downloader = new HuggingFaceDownloader();
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            ModelResolver.PullAsync(ModelResolver.Parse("just-a-name"), null, client, downloader, null, default));
    }

    [Fact]
    public void DeleteLocal_RemovesMirrorSnapshotAndBlob_AndLeavesOtherFilesAlone()
    {
        AddHub("acme/del", "keep-Q8_0.gguf", 700, mirror: true, fill: 'k');
        AddHub("acme/del", "gone-Q4_K_M.gguf", 300, mirror: true, fill: 'g');
        var gone = ModelResolver.EnumerateLocal(Mirror, Hub).Single(m => m.Filename == "gone-Q4_K_M.gguf");

        long freed = ModelResolver.DeleteLocal(gone, Mirror, Hub);

        Assert.Equal(300, freed);
        var left = ModelResolver.EnumerateLocal(Mirror, Hub);
        Assert.Equal(["keep-Q8_0.gguf"], left.Select(m => m.Filename));
        Assert.Single(Directory.GetFiles(Path.Combine(HubCache.RepoDirectory("acme/del", Hub), "blobs")));   // only keep's blob remains
    }

    [Fact]
    public void DeleteLocal_LastFileRemovesTheRepoFolders()
    {
        AddHub("acme/solo", "s-Q4_K_M.gguf", 100, mirror: true);
        var m = ModelResolver.EnumerateLocal(Mirror, Hub).Single();
        ModelResolver.DeleteLocal(m, Mirror, Hub);
        Assert.False(Directory.Exists(HubCache.RepoDirectory("acme/solo", Hub)));
        Assert.False(Directory.Exists(Path.Combine(Mirror, "acme", "solo")));
    }
}
