using System.Runtime.Versioning;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Remaining permission and path items of #1664 (from the #1676 review): shard directories and temp files that become chunk, index,
/// ledger and registry files are owner-only even under a pre-existing group-readable root; only the store directory itself is made
/// private, not its missing parents; and a <c>..</c> after a link follows the link's target.
/// </summary>
public sealed class StoreModesTests : IDisposable
{
    private static readonly RopeParameters Rope = new(128, 5_000_000f, RoPEType.NeoX);
    private const UnixFileMode Dir700 = UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute;
    private const UnixFileMode File600 = UnixFileMode.UserRead | UnixFileMode.UserWrite;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-modes-" + Guid.NewGuid().ToString("N"));

    public StoreModesTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
    }

    [SkippableFact]
    [UnsupportedOSPlatform("windows")]
    public void ExistingGroupReadableRoot_ShardDirectoriesAndChunkFilesAreStillOwnerOnly()
    {
        Skip.If(OperatingSystem.IsWindows(), "POSIX modes");
        string root = Path.Combine(_scratch, "store");
        Directory.CreateDirectory(root);
        UnixFileMode shared = Dir700 | UnixFileMode.GroupRead | UnixFileMode.GroupExecute | UnixFileMode.OtherRead | UnixFileMode.OtherExecute;
        File.SetUnixFileMode(root, shared);
        var store = new FileChunkStore(root);
        ChunkKey key = ChunkKey.Create("fp-a", "tok", Rope, "cpu", KvDtype.F32, [1, 2]);

        store.Put(key, new ChunkKvEntry("fp-a", [1, 2], [[1f, -2f]], [[0.5f, 4f]]));

        Assert.Equal(shared, File.GetUnixFileMode(root)); // control: the operator's root is untouched
        string chunk = Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories).Single(f => !f.EndsWith(".lock", StringComparison.Ordinal));
        Assert.Equal(File600, File.GetUnixFileMode(chunk));
        Assert.Equal(Dir700, File.GetUnixFileMode(Path.GetDirectoryName(chunk)!));
    }

    [SkippableFact]
    [UnsupportedOSPlatform("windows")]
    public void ExistingGroupReadableRoot_IndexLedgerAndRegistryFilesAreOwnerOnly()
    {
        Skip.If(OperatingSystem.IsWindows(), "POSIX modes");
        string root = Path.Combine(_scratch, "store");
        Directory.CreateDirectory(root);
        File.SetUnixFileMode(root, Dir700 | UnixFileMode.GroupRead | UnixFileMode.GroupExecute);
        using StoreLockHandle held = ChunkIndex.AcquireStoreLock(root, TimeSpan.Zero);

        ServerLedger.Load(root).Save(held);
        StoreRegistry.Load(root).Save(held);
        new ChunkIndex(ids => ChunkKey.Create("fp-a", "tok", Rope, "cpu", KvDtype.F32, ids)).Save(Path.Combine(root, "index-fixed-10.json"));

        string[] saved = Directory.EnumerateFiles(root).Where(f => !f.EndsWith(".lock", StringComparison.Ordinal)).ToArray();
        Assert.NotEmpty(saved);
        Assert.All(saved, f => Assert.Equal(File600, File.GetUnixFileMode(f)));
    }

    [SkippableFact]
    [UnsupportedOSPlatform("windows")]
    public void CreateStoreDirectory_MissingParents_KeepTheDefaultMode_OnlyTheStoreIsPrivate()
    {
        Skip.If(OperatingSystem.IsWindows(), "POSIX modes");
        string plain = Path.Combine(_scratch, "plain");
        Directory.CreateDirectory(plain);
        UnixFileMode normal = File.GetUnixFileMode(plain);
        Skip.If(normal == Dir700, "umask 077: a default directory is already owner-only");
        string store = Path.Combine(_scratch, "a", "b", "store");

        StorePaths.CreateStoreDirectory(store);

        Assert.Equal(Dir700, File.GetUnixFileMode(store));
        Assert.Equal(normal, File.GetUnixFileMode(Path.Combine(_scratch, "a", "b")));
        Assert.Equal(normal, File.GetUnixFileMode(Path.Combine(_scratch, "a")));
    }

    [SkippableFact]
    public void CanonicalRoot_DotDotAfterALink_FollowsTheLinksTarget_NotItsParent()
    {
        string target = Path.Combine(_scratch, "real", "deep");
        Directory.CreateDirectory(target);
        string link = Path.Combine(_scratch, "link");
        bool linked;
        try { Directory.CreateSymbolicLink(link, target); linked = Directory.Exists(link); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException) { linked = false; }
        Skip.If(!linked, "cannot create a directory symlink here");
        // The OS resolves "link/.." to the parent of the link's TARGET (real), a lexical collapse would give the link's parent (_scratch).
        string viaLink = link + Path.DirectorySeparatorChar + ".." + Path.DirectorySeparatorChar + "store";

        string canonical = StorePaths.CanonicalRoot(viaLink);

        Assert.Equal(StorePaths.CanonicalRoot(Path.Combine(_scratch, "real", "store")), canonical);
        Assert.NotEqual(StorePaths.CanonicalRoot(Path.Combine(_scratch, "store")), canonical);
    }

    [Fact]
    public void CanonicalRoot_PlainDotsAndTrailingSeparator_AreNormalised()
    {
        Directory.CreateDirectory(Path.Combine(_scratch, "x"));
        string odd = Path.Combine(_scratch, "x", ".", "..", "x") + Path.DirectorySeparatorChar;

        Assert.Equal(StorePaths.CanonicalRoot(Path.Combine(_scratch, "x")), StorePaths.CanonicalRoot(odd));
    }
}
