using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>#1664: a rename into the store is followed by a directory sync (POSIX fsync of the directory entry; the hook fires on every OS).</summary>
public sealed class DirectorySyncTests : IDisposable
{
    private static readonly RopeParameters Rope = new(128, 5_000_000f, RoPEType.NeoX);
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-dirsync-" + Guid.NewGuid().ToString("N"));

    public DirectorySyncTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        StorePaths.DirectorySyncedForTest = null;
        try { Directory.Delete(_scratch, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
    }

    [Fact]
    public void ChunkPut_SyncsTheShardDirectoryItRenamedInto()
    {
        var synced = new List<string>();
        StorePaths.DirectorySyncedForTest = synced.Add;
        var store = new FileChunkStore(Path.Combine(_scratch, "store"));
        ChunkKey key = ChunkKey.Create("fp-a", "tok", Rope, "cpu", KvDtype.F32, [1, 2]);

        store.Put(key, new ChunkKvEntry("fp-a", [1, 2], [[1f, -2f]], [[0.5f, 4f]]));

        string chunk = Directory.EnumerateFiles(Path.Combine(_scratch, "store"), "*", SearchOption.AllDirectories).Single(f => !f.EndsWith(".lock", StringComparison.Ordinal));
        Assert.Contains(Path.GetDirectoryName(chunk)!, synced);
    }

    [Fact]
    public void IndexAndLedgerAndRegistrySave_SyncTheirDirectory()
    {
        string root = Path.Combine(_scratch, "store2");
        Directory.CreateDirectory(root);
        using StoreLockHandle held = ChunkIndex.AcquireStoreLock(root, TimeSpan.Zero);
        var synced = new List<string>();
        StorePaths.DirectorySyncedForTest = synced.Add;

        ServerLedger.Load(root).Save(held);
        int afterLedger = synced.Count;
        StoreRegistry.Load(root).Save(held);

        Assert.True(afterLedger >= 1, "ledger save did not sync its directory");
        Assert.True(synced.Count > afterLedger, "registry save did not sync its directory");
        Assert.All(synced, d => Assert.True(Directory.Exists(d)));
    }

    [Fact]
    public void SyncDirectory_OnARealDirectory_DoesNotThrow_AndOnAMissingOneIsIgnored()
    {
        StorePaths.SyncDirectory(_scratch);
        StorePaths.SyncDirectory(Path.Combine(_scratch, "missing"));
    }
}
