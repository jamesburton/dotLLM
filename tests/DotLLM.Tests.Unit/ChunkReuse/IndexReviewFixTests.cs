using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.ChunkReuse.Tooling;
using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Review fixes for the index / lock / ledger / registry PR: the test hooks and retry budgets are per async flow (a process-wide
/// value raced between test classes running in parallel and made CI fail), released lock handles are not kept by the flow, the
/// registry's bad-copy name survives a '~' in the directory, <see cref="PathGlob"/> rejects '..' anywhere, and the
/// referenced-key scan skips exactly the caller's own index file.
/// </summary>
public sealed class IndexReviewFixTests : IDisposable
{
    private static readonly RopeParameters Rope = new(128, 5_000_000f, RoPEType.NeoX);
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-indexfix-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best effort */ }
        catch (UnauthorizedAccessException) { }
    }

    private static ChunkKey KeyA(ReadOnlySpan<int> ids) => ChunkKey.Create("fp-a", "tok", Rope, "cpu", KvDtype.F32, ids);

    // ------------------------------------------------------------------------------------------------ per-flow hooks

    /// <summary>What another async flow (another test class on another thread) sees: the context is not inherited.</summary>
    private static T InOtherFlow<T>(Func<T> read)
    {
        T result = default!;
        using (ExecutionContext.SuppressFlow())
        {
            var thread = new Thread(() => result = read());
            thread.Start();
            thread.Join();
        }
        return result;
    }

    [Fact]
    public void BeforeOrphanDeletes_IsPerFlow()
    {
        ChunkIndex.BeforeOrphanDeletes = _ => { };
        try
        {
            Assert.NotNull(ChunkIndex.BeforeOrphanDeletes);
            Assert.Null(InOtherFlow(() => ChunkIndex.BeforeOrphanDeletes));
        }
        finally
        {
            ChunkIndex.BeforeOrphanDeletes = null;
        }
    }

    [Fact]
    public void AfterMigrationRename_IsPerFlow()
    {
        ChunkIndex.AfterMigrationRename = () => { };
        try
        {
            Assert.NotNull(ChunkIndex.AfterMigrationRename);
            Assert.Null(InOtherFlow(() => ChunkIndex.AfterMigrationRename));
        }
        finally
        {
            ChunkIndex.AfterMigrationRename = null;
        }
    }

    [Fact]
    public void SaveRetryBudget_IsPerFlow_AndKeepsItsDefault()
    {
        TimeSpan defaultBudget = InOtherFlow(() => ChunkIndex.SaveRetryBudget);
        Assert.Equal(TimeSpan.FromSeconds(2), defaultBudget);
        ChunkIndex.SaveRetryBudget = TimeSpan.FromMilliseconds(7);
        try
        {
            Assert.Equal(TimeSpan.FromMilliseconds(7), ChunkIndex.SaveRetryBudget);
            Assert.Equal(defaultBudget, InOtherFlow(() => ChunkIndex.SaveRetryBudget));
        }
        finally
        {
            ChunkIndex.SaveRetryBudget = defaultBudget;
        }
    }

    [Fact]
    public void RegistryReadRetryBudget_IsPerFlow_AndKeepsItsDefault()
    {
        TimeSpan defaultBudget = InOtherFlow(() => StoreRegistry.ReadRetryBudget);
        Assert.Equal(TimeSpan.FromSeconds(2), defaultBudget);
        StoreRegistry.ReadRetryBudget = TimeSpan.FromMilliseconds(7);
        try
        {
            Assert.Equal(TimeSpan.FromMilliseconds(7), StoreRegistry.ReadRetryBudget);
            Assert.Equal(defaultBudget, InOtherFlow(() => StoreRegistry.ReadRetryBudget));
        }
        finally
        {
            StoreRegistry.ReadRetryBudget = defaultBudget;
        }
    }

    // ------------------------------------------------------------------------------------------------ flow lock list

    /// <summary>
    /// A long-lived flow (a server's request loop) takes and releases the store lock over and over: the per-flow list must not keep
    /// every released handle (it is copied on each acquire, so keeping them is unbounded growth and O(n) per acquire).
    /// </summary>
    [Fact]
    public void StoreLocks_ReleasedHandlesAreNotKeptByTheFlow()
    {
        Directory.CreateDirectory(_root);
        for (int i = 0; i < 300; i++)
            using (ChunkIndex.AcquireStoreLock(_root, TimeSpan.Zero)) { }
        Assert.True(ChunkIndex.FlowStoreLocksTracked <= 1, $"the flow tracks {ChunkIndex.FlowStoreLocksTracked} handles after 300 released locks");
    }

    // ------------------------------------------------------------------------------------------------ registry names

    [Theory]
    [InlineData("index-fixed-10.0123456789ab.0123abcd.default.json", true)]
    [InlineData("index-symbol-512.json", true)]
    [InlineData("index-fixed-10..json", false)]           // '..' anywhere is refused (conservative; one path component, so no traversal)
    [InlineData("index-a/b.json", false)]                  // forward slash
    [InlineData("index-a\\b.json", false)]                 // backslash
    [InlineData("index-a:b.json", false)]                  // drive / stream separator
    [InlineData("..\\index-a.json", false)]
    [InlineData("sub/index-a.json", false)]
    [InlineData("index-a.txt", false)]
    [InlineData("notindex-a.json", false)]
    [InlineData("", false)]
    [InlineData(null, false)]
    public void IsPlainIndexFileName_AcceptsOnlyAPlainIndexFileName(string? name, bool expected)
    {
        if (name is not null && name.Contains('\\', StringComparison.Ordinal))
            Assert.Contains('\\', name); // control: the backslash reached the string
        Assert.Equal(expected, StoreRegistry.IsPlainIndexFileName(name));
    }

    [Fact]
    public void BadCopyPath_AppendsTheCounter_NotCutAtATildeInTheDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "PROGRA~1", "chunkcache.json");
        DateTime t = new(2026, 11, 6, 12, 30, 15, 123, DateTimeKind.Utc);
        string stamped = path + StoreRegistry.BadCopyPrefix + "20261106T123015123Z";
        Assert.Equal(stamped, StoreRegistry.BadCopyPath(path, t, _ => false));
        Assert.Equal(stamped + "~1", StoreRegistry.BadCopyPath(path, t, p => p == stamped));
        Assert.Equal(stamped + "~2", StoreRegistry.BadCopyPath(path, t, p => p == stamped || p == stamped + "~1"));
        Assert.Contains("PROGRA~1", StoreRegistry.BadCopyPath(path, t, p => p == stamped), StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------------------------------------ PathGlob

    [Theory]
    [InlineData("a/../../x")]
    [InlineData("a/../x")]
    [InlineData("../x")]
    [InlineData("a/..")]
    public void PathGlob_NeverMatchesAPathWithADotDotSegment(string path)
    {
        Assert.False(PathGlob.Parse("**").IsMatch(path), path);
        Assert.False(PathGlob.Parse("*.cs").IsMatch(path + "/f.cs"), path); // still a '..' segment before the file name
        Assert.True(PathGlob.Parse("**").IsMatch("a/b/x"), "control: an ordinary path still matches");
        Assert.True(PathGlob.Parse("**").IsMatch("a/b..c/x"), "control: '..' inside a name is not a segment");
    }

    // ------------------------------------------------------------------------------------------------ referenced keys

    /// <summary>
    /// The scan skips exactly the caller's own index file. Windows compares paths case-insensitively (one file); elsewhere two
    /// files differing only in case are two indexes, and skipping the other one would drop its keys from the protected set.
    /// </summary>
    [Fact]
    public void ReferencedKeysUnder_SkipsOnlyTheCallersOwnIndexFile()
    {
        Directory.CreateDirectory(_root);
        var chunker = new FixedSizeChunker(new CharTokenizer(), chunkTokens: 10);
        ChunkKey keyOf(ReadOnlySpan<int> ids) => KeyA(ids);

        var mine = new ChunkIndex(keyOf);
        mine.Update([new SourceFile("A.txt", "012345678\n")], chunker);
        var other = new ChunkIndex(keyOf);
        other.Update([new SourceFile("B.txt", "ABCDEFGHI\n")], chunker);
        ChunkKey mineKey = new(mine.ChunksOf("A.txt")[0].KeyHex!);
        ChunkKey otherKey = new(other.ChunksOf("B.txt")[0].KeyHex!);
        Assert.NotEqual(mineKey, otherKey);

        // Both names match the scan pattern 'index-*.json' (it is case-sensitive on Linux) and differ only in case after the prefix.
        string lower = Path.Combine(_root, "index-case.json");
        string upper = Path.Combine(_root, "index-CASE.json");
        mine.Save(lower);
        if (OperatingSystem.IsWindows())
        {
            // One file under two spellings: passing the other spelling skips it (and the keys are gone from the set).
            Assert.True(File.Exists(upper));
            Assert.DoesNotContain(mineKey, ChunkIndex.ReferencedKeysUnder(_root, upper));
            Assert.Contains(mineKey, ChunkIndex.ReferencedKeysUnder(_root));
        }
        else
        {
            other.Save(upper);
            Assert.NotEqual(Path.GetFileName(lower), Path.GetFileName(upper));
            IReadOnlySet<ChunkKey> keys = ChunkIndex.ReferencedKeysUnder(_root, lower);
            Assert.DoesNotContain(mineKey, keys);
            Assert.Contains(otherKey, keys); // the differently-cased file is another index: its keys stay protected
        }
    }

    [Fact]
    public void IndexSave_CleanupOfTheTempFileNeverMasksTheSaveFailure()
    {
        // The temp-file delete in Save's finally is guarded (IOException / UnauthorizedAccessException are swallowed). Observable
        // part: when the destination cannot be replaced (a directory sits at the index path), the exception is the save's own.
        Directory.CreateDirectory(_root);
        string path = Path.Combine(_root, "index-blocked.0123456789ab.0123abcd.default.json");
        Directory.CreateDirectory(path);
        var index = new ChunkIndex(KeyA);
        index.Update([new SourceFile("A.txt", "012345678\n")], new FixedSizeChunker(new CharTokenizer(), chunkTokens: 10));

        Exception ex = Assert.ThrowsAny<Exception>(() => index.Save(path));

        Assert.True(ex is IOException or UnauthorizedAccessException, ex.GetType().Name);
        Assert.Empty(Directory.GetFiles(_root, "*.tmp"));
    }
}
