using DotLLM.HuggingFace;
using Xunit;

namespace DotLLM.Tests.Unit.HuggingFace;

/// <summary>Issue #714: one resolver for every place a model is named.</summary>
[Collection("SequentialFileIO")]   // reads the process-global model-store env vars that SequentialFileIO tests mutate (#1395)
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

    [Fact]
    public void IdFor_QualifiesSameNamedFilesByRepo_AndTheIdResolvesBackToItsOwnFile()
    {
        // Two repos shipping the same bare quant filename were both listed as "q4_k_m", indistinguishable to a client.
        string a = AddHub("acme/alpha", "q4_k_m.gguf", 100);
        string b = AddHub("other/beta", "q4_k_m.gguf", 200);

        var ids = ModelResolver.EnumerateLocal(Mirror, Hub).Select(ModelResolver.IdFor).ToList();
        Assert.Equal(["acme/alpha/q4_k_m.gguf", "other/beta/q4_k_m.gguf"], ids);

        Assert.Equal(a, ModelResolver.ResolveLocal("acme/alpha/q4_k_m.gguf", null, Mirror, Hub));
        Assert.Equal(b, ModelResolver.ResolveLocal("other/beta/q4_k_m.gguf", null, Mirror, Hub));
        Assert.Equal("acme/alpha/q4_k_m.gguf", ModelResolver.IdForPath(a, Mirror, Hub));
        Assert.Equal("other/beta/q4_k_m.gguf", ModelResolver.IdForPath(b, Mirror, Hub));
    }

    [Fact]
    public void IdForPath_FileOutsideTheStores_IsItsFullPath_NeverABareStem()
    {
        string loose = Path.Combine(_root, "loose", "my-model.gguf");
        Directory.CreateDirectory(Path.GetDirectoryName(loose)!);
        File.WriteAllBytes(loose, new byte[10]);

        // #1021: a bare stem would be persisted by clients and could later resolve to a different (store) file of that name.
        string id = ModelResolver.IdForPath(loose, Mirror, Hub);
        Assert.Equal(Path.GetFullPath(loose), id);
        Assert.Equal(Path.GetFullPath(loose), ModelResolver.ResolveLocal(id, null, Mirror, Hub));
    }

    /// <summary>
    /// #1021: two repos' <c>q4_k_m.gguf</c> under one bare name. It used to resolve to the larger file (a model the client never named,
    /// which then hung loading); it must fail and list both repo-qualified ids instead.
    /// </summary>
    [Theory]
    [InlineData("q4_k_m")]
    [InlineData("Q4_K_M")]
    [InlineData("q4_k_m.gguf")]
    public void ResolveLocal_BareNameMatchingTwoRepos_IsAmbiguous_AndListsEveryCandidate(string bare)
    {
        AddHub("Hcompany/Holo4-27B-GGUF", "q4_k_m.gguf", 100);
        AddHub("Hcompany/Holo4-35B-A3B-GGUF", "q4_k_m.gguf", 200);

        var ex = Assert.Throws<AmbiguousModelReferenceException>(() => ModelResolver.ResolveLocal(bare, null, Mirror, Hub));

        Assert.Equal(bare, ex.Reference);
        Assert.Equal(["Hcompany/Holo4-27B-GGUF/q4_k_m.gguf", "Hcompany/Holo4-35B-A3B-GGUF/q4_k_m.gguf"], ex.Candidates);
        Assert.Contains("Hcompany/Holo4-27B-GGUF/q4_k_m.gguf", ex.Message);
        Assert.Contains("Hcompany/Holo4-35B-A3B-GGUF/q4_k_m.gguf", ex.Message);
    }

    [Fact]
    public void ResolveLocal_AmbiguousBareName_EachCandidateIdStillResolvesToItsOwnFile()
    {
        string small = AddHub("Hcompany/Holo4-27B-GGUF", "q4_k_m.gguf", 100);
        string large = AddHub("Hcompany/Holo4-35B-A3B-GGUF", "q4_k_m.gguf", 200);

        var ex = Assert.Throws<AmbiguousModelReferenceException>(() => ModelResolver.ResolveLocal("q4_k_m", null, Mirror, Hub));

        // The listed ids are the way out: each must resolve, and to the right file.
        Assert.Equal(small, ModelResolver.ResolveLocal(ex.Candidates[0], null, Mirror, Hub));
        Assert.Equal(large, ModelResolver.ResolveLocal(ex.Candidates[1], null, Mirror, Hub));
    }

    [Fact]
    public void ResolveLocal_SameFileReachableThroughMirrorAndHub_IsOneModel_NotAmbiguous()
    {
        // The flat mirror and the hub snapshot are two paths to ONE model - that must not count as two candidates.
        AddHub("acme/linked", "q6_k.gguf", 100, mirror: true);

        Assert.NotNull(ModelResolver.ResolveLocal("q6_k", null, Mirror, Hub));   // either path to the one file; no exception
    }

    [Fact]
    public void ResolveLocal_AmbiguityIsPerName_NotPerSharedQuant()
    {
        // Only files sharing the requested NAME compete: model-Q4_K_M is a different name, so it neither joins nor breaks the tie.
        AddHub("acme/a", "model-Q4_K_M.gguf", 100);
        string q8 = AddHub("acme/b", "model-Q8_0.gguf", 100);
        AddHub("acme/c", "model-Q8_0.gguf", 100);

        var ex = Assert.Throws<AmbiguousModelReferenceException>(() => ModelResolver.ResolveLocal("model-Q8_0", null, Mirror, Hub));
        Assert.Equal(2, ex.Candidates.Count);
        Assert.NotNull(ModelResolver.ResolveLocal("model-Q4_K_M", null, Mirror, Hub));
        Assert.Equal(q8, ModelResolver.ResolveLocal("acme/b/model-Q8_0.gguf", null, Mirror, Hub));
    }

    [Fact]
    public void ResolveLocal_RepoReference_KeepsDefaultQuantChoice_NotAnAmbiguity()
    {
        // owner/repo names ONE model with several quants: choosing the default quant is documented behaviour, not a guess.
        string q4 = AddHub("acme/one-model", "m-Q4_K_M.gguf", 100);
        AddHub("acme/one-model", "m-Q8_0.gguf", 300);

        Assert.Equal(q4, ModelResolver.ResolveLocal("acme/one-model", null, Mirror, Hub));
    }

    [Fact]
    public void ResolveLocal_BareStemStillResolves_ForClientsHoldingAnOldId()
    {
        string only = AddHub("acme/solo-stem", "q5_k_m.gguf", 100);

        Assert.Equal(only, ModelResolver.ResolveLocal("q5_k_m", null, Mirror, Hub));
    }

    /// <summary>
    /// Review of #1021: the flat mirror keeps only the file NAME while the hub snapshot keeps its subfolder, so one file was listed twice
    /// and a bare name reported it as ambiguous with itself.
    /// </summary>
    [Fact]
    public void SubfolderSnapshotPlusFlatMirror_IsOneModel_ListedOnce_AndNotAmbiguousWithItself()
    {
        AddHub("acme/split", "Q4_K_M/model-Q4_K_M.gguf", 100, mirror: true);   // mirror: models/acme/split/model-Q4_K_M.gguf

        var listed = ModelResolver.EnumerateLocal(Mirror, Hub);
        Assert.Single(listed);
        Assert.Equal("acme/split/Q4_K_M/model-Q4_K_M.gguf", ModelResolver.IdFor(listed[0]));   // the structural (snapshot) id wins
        Assert.NotNull(ModelResolver.ResolveLocal("model-Q4_K_M", null, Mirror, Hub));         // no self-ambiguity
        Assert.NotNull(ModelResolver.ResolveLocal("acme/split/model-Q4_K_M.gguf", null, Mirror, Hub));   // the flat id still resolves
    }

    /// <summary>Review of #1021: the id must not depend on which copy of a file was resolved (it was a stem for the losing copy).</summary>
    [Fact]
    public void IdForPath_IsTheSameForEitherCopyOfAFile()
    {
        string snap = AddHub("acme/split", "Q4_K_M/model-Q4_K_M.gguf", 100, mirror: true);
        string mirror = HubCache.MirrorPath("acme/split", "Q4_K_M/model-Q4_K_M.gguf", Mirror);
        Assert.True(File.Exists(mirror));

        Assert.Equal("acme/split/Q4_K_M/model-Q4_K_M.gguf", ModelResolver.IdForPath(snap, Mirror, Hub));
        Assert.Equal("acme/split/Q4_K_M/model-Q4_K_M.gguf", ModelResolver.IdForPath(mirror, Mirror, Hub));
    }

    [Fact]
    public void Ambiguity_HasItsOwnErrorCode()
    {
        var ex = new AmbiguousModelReferenceException("q4_k_m", ["a/b/q4_k_m.gguf", "c/d/q4_k_m.gguf"]);
        Assert.Equal("ambiguous_model", AmbiguousModelReferenceException.CodeFor(ex.Message));
        Assert.Equal("model_not_found", AmbiguousModelReferenceException.CodeFor("Model not found: x"));
    }

    /// <summary>A file in snapshot <paramref name="commit"/>, last written at <paramref name="writtenUtc"/>; optionally hard-linked into the mirror.</summary>
    private string AddRevision(string repo, string commit, string file, int size, DateTime writtenUtc, bool mirror = false)
    {
        string blob = Path.Combine(HubCache.RepoDirectory(repo, Hub), "blobs", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.GetDirectoryName(blob)!);
        File.WriteAllBytes(blob, new byte[size]);
        File.SetLastWriteTimeUtc(blob, writtenUtc);
        string snap = HubCache.SnapshotFilePath(repo, commit, file, Hub);
        Directory.CreateDirectory(Path.GetDirectoryName(snap)!);
        HubCache.LinkOrCopy(blob, snap);
        File.SetLastWriteTimeUtc(snap, writtenUtc);
        if (mirror)
        {
            string m = HubCache.MirrorPath(repo, file, Mirror);
            Directory.CreateDirectory(Path.GetDirectoryName(m)!);
            if (File.Exists(m)) File.Delete(m);
            HubCache.LinkOrCopy(blob, m);
            File.SetLastWriteTimeUtc(m, writtenUtc);
        }
        return snap;
    }

    /// <summary>
    /// Review round 2 of #1021: a re-upload leaves the old snapshot behind, usually with a different size. Both revisions share one id,
    /// so they must be ONE entry and every resolution path must reach the NEWEST - a size-including key listed both and let enumeration
    /// order pick, which could silently load the stale revision.
    /// </summary>
    [Fact]
    public void TwoSnapshotRevisionsOfOneFile_AreOneModel_AndEveryPathResolvesTheNewest()
    {
        AddRevision("acme/rev", "0ld", "Q4_K_M/m-Q4_K_M.gguf", 300, DateTime.UtcNow.AddDays(-3));            // stale and LARGER
        string fresh = AddRevision("acme/rev", "n3w", "Q4_K_M/m-Q4_K_M.gguf", 200, DateTime.UtcNow, mirror: true);

        var listed = ModelResolver.EnumerateLocal(Mirror, Hub).Where(m => m.RepoId == "acme/rev").ToList();
        Assert.Single(listed);
        Assert.Equal(fresh, listed[0].FullPath);

        foreach (string reference in new[] { "m-Q4_K_M", "m-Q4_K_M.gguf", "acme/rev/Q4_K_M/m-Q4_K_M.gguf", "acme/rev", "acme/rev:Q4_K_M", "acme/rev/m-Q4_K_M.gguf" })
            Assert.Equal(200, ModelResolver.FileLength(ModelResolver.ResolveLocal(reference, null, Mirror, Hub)!));
    }

    /// <summary>Review round 2 of #1021: equal name and size in DIFFERENT subfolders are different files, not one.</summary>
    [Fact]
    public void SameNameAndSize_InDifferentSubfolders_AreTwoModels()
    {
        AddRevision("acme/two", "c0ffee", "a/x.gguf", 100, DateTime.UtcNow);
        AddRevision("acme/two", "c0ffee", "b/x.gguf", 100, DateTime.UtcNow);

        Assert.Equal(2, ModelResolver.EnumerateLocal(Mirror, Hub).Count(m => m.RepoId == "acme/two"));
    }
}
