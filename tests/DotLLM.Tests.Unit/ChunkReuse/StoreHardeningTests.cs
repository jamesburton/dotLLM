using System.Diagnostics;
using System.Runtime.Versioning;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Store hardening (issue #1664): lock-file permissions, access denied is not "busy", the store lock is not re-entered by a
/// <see cref="LockingChunkStore"/>, store roots are compared by their real path, and the owner lock's probe and error
/// classification.
/// </summary>
public sealed class StoreHardeningTests : IDisposable
{
    private static readonly RopeParameters Rope = new(128, 5_000_000f, RoPEType.NeoX);
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-hardening-" + Guid.NewGuid().ToString("N"));
    private string Root => Path.Combine(_scratch, "store");

    public StoreHardeningTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try
        {
            foreach (string f in Directory.EnumerateFiles(_scratch, "*", SearchOption.AllDirectories))
                File.SetAttributes(f, FileAttributes.Normal);
            Directory.Delete(_scratch, recursive: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
    }

    private static ChunkKey Key(ReadOnlySpan<int> ids) => ChunkKey.Create("fp-a", "tok", Rope, "cpu", KvDtype.F32, ids);

    private static ChunkKvEntry Entry(int seed) => new("fp-a", [seed, seed + 1], [[1f, -2f]], [[0.5f, 4f]]);

    // ------------------------------------------------------------------------------------------------ access denied

    /// <summary>A lock file that cannot be opened for writing is a configuration error: it must not look like a lock timeout.</summary>
    [Fact]
    public void StoreLock_LockFileNotWritable_FailsAtOnce_NotAsATimeout()
    {
        Directory.CreateDirectory(Root);
        string lockFile = Path.Combine(Root, ChunkIndex.StoreLockFileName);
        File.WriteAllText(lockFile, "");
        File.SetAttributes(lockFile, FileAttributes.ReadOnly);

        var sw = Stopwatch.StartNew();
        Exception ex = Assert.ThrowsAny<Exception>(() => ChunkIndex.AcquireStoreLock(Root, TimeSpan.FromSeconds(20)));

        Assert.IsAssignableFrom<UnauthorizedAccessException>(ex);
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(5), $"took {sw.Elapsed}: access denied was retried as if busy");
    }

    [Fact]
    public void IndexLock_LockFileNotWritable_FailsAtOnce_NotAsATimeout()
    {
        Directory.CreateDirectory(Root);
        string index = Path.Combine(Root, "index-fixed-10.json");
        string lockFile = Path.ChangeExtension(index, ChunkIndex.IndexLockSuffix);
        File.WriteAllText(lockFile, "");
        File.SetAttributes(lockFile, FileAttributes.ReadOnly);

        var sw = Stopwatch.StartNew();
        Exception ex = Assert.ThrowsAny<Exception>(() => ChunkIndex.AcquireIndexLockFor(index, TimeSpan.FromSeconds(20)));

        Assert.IsAssignableFrom<UnauthorizedAccessException>(ex);
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(5), $"took {sw.Elapsed}: access denied was retried as if busy");
    }

    [Fact]
    public void StoreLock_HeldByAnotherHandle_StillWaitsAndTimesOut()
    {
        // Control for the two tests above: a genuine sharing conflict is still retried until the timeout.
        Directory.CreateDirectory(Root);
        using var other = new FileStream(Path.Combine(Root, ChunkIndex.StoreLockFileName), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);

        var sw = Stopwatch.StartNew();
        Assert.Throws<ChunkStoreLockTimeoutException>(() => ChunkIndex.AcquireStoreLock(Root, TimeSpan.FromMilliseconds(300)));
        Assert.True(sw.Elapsed >= TimeSpan.FromMilliseconds(250), $"gave up after {sw.Elapsed}");
    }

    // ------------------------------------------------------------------------------------------------ self-deadlock

    [Fact]
    public void LockingChunkStore_InsideTheFlowsStoreLock_ThrowsAtOnce_NotAfterTheTimeout()
    {
        var files = new FileChunkStore(Root);
        var locking = new LockingChunkStore(files, Root, TimeSpan.FromSeconds(20));
        ChunkKey key = Key([1, 2]);
        var sw = Stopwatch.StartNew();
        using (ChunkIndex.AcquireStoreLock(Root, TimeSpan.Zero))
        {
            Assert.Throws<InvalidOperationException>(() => locking.Put(key, Entry(1)));
            Assert.Throws<InvalidOperationException>(() => locking.Evict(key));
        }
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(5), $"took {sw.Elapsed}: the decorator waited for a lock its own flow holds");
        Assert.False(files.Contains(key));

        // Control: outside the scope it writes and evicts as before.
        locking.Put(key, Entry(1));
        Assert.True(files.Contains(key));
        locking.Evict(key);
        Assert.False(files.Contains(key));
    }

    [Fact]
    public void LockingChunkStore_OtherFlowHoldsTheLock_StillTimesOut()
    {
        // Control: only the caller's own flow is detected; another holder is a normal wait + timeout.
        var locking = new LockingChunkStore(new FileChunkStore(Root), Root, TimeSpan.FromMilliseconds(200));
        using var other = new FileStream(Path.Combine(Root, ChunkIndex.StoreLockFileName), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);

        Assert.Throws<ChunkStoreLockTimeoutException>(() => locking.Put(Key([1]), Entry(1)));
    }

    // ------------------------------------------------------------------------------------------------ root identity

    /// <summary>The same store root through a symlink or junction is the same root: the lock order check and the held-lock proof use real paths.</summary>
    [SkippableFact]
    public void StoreRoot_ThroughALink_IsTheSameRoot_ForLockOrderAndProof()
    {
        Directory.CreateDirectory(Root);
        string link = Path.Combine(_scratch, "link");
        Skip.If(!TryCreateDirectoryLink(link, Root), "cannot create a directory link here");
        string viaLink = Path.Combine(link, "index-fixed-10.json");

        using StoreLockHandle held = ChunkIndex.AcquireStoreLock(Root, TimeSpan.Zero);

        // Lock order: an index lock of the same store, spelled through the link, is refused like the plain spelling.
        Assert.Throws<InvalidOperationException>(() => ChunkIndex.AcquireIndexLockFor(Path.Combine(Root, "index-fixed-10.json"), TimeSpan.Zero));
        Assert.Throws<InvalidOperationException>(() => ChunkIndex.AcquireIndexLockFor(viaLink, TimeSpan.Zero));

        // Proof: the handle taken through one spelling proves the lock for the other.
        held.EnsureHeldFor(link);
        using StoreLockHandle viaLinkHandle = ReleaseAndTake(held, link);
        viaLinkHandle.EnsureHeldFor(Root);
    }

    private static StoreLockHandle ReleaseAndTake(StoreLockHandle held, string root)
    {
        held.Dispose();
        return ChunkIndex.AcquireStoreLock(root, TimeSpan.Zero);
    }

    private static bool TryCreateDirectoryLink(string link, string target)
    {
        try
        {
            Directory.CreateSymbolicLink(link, target);
            return Directory.Exists(link);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException) { }
        if (!OperatingSystem.IsWindows()) return false;
        try
        {
            using var p = Process.Start(new ProcessStartInfo("cmd.exe", $"/c mklink /J \"{link}\" \"{target}\"")
            {
                UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true,
            })!;
            p.StandardOutput.ReadToEnd();
            p.StandardError.ReadToEnd();
            p.WaitForExit();
            return p.ExitCode == 0 && Directory.Exists(link);
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return false;
        }
    }

    // ------------------------------------------------------------------------------------------------ permissions (POSIX)

    [SkippableFact]
    [UnsupportedOSPlatform("windows")]
    public void NewStoreRoot_IsPrivate_AndLockFilesAreOwnerOnly()
    {
        Skip.If(OperatingSystem.IsWindows(), "POSIX modes; on Windows a new directory inherits its parent's ACL (documented)");
        string root = Path.Combine(_scratch, "fresh", "store");

        using (ChunkIndex.AcquireStoreLock(root, TimeSpan.Zero)) { }
        using (ChunkIndex.AcquireIndexLockFor(Path.Combine(root, "index-x.json"), TimeSpan.Zero)) { }
        using (StoreOwnerLock.TryAcquire(root)) { }

        Assert.Equal(UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute, File.GetUnixFileMode(root));
        Assert.Equal(UnixFileMode.UserRead | UnixFileMode.UserWrite, File.GetUnixFileMode(Path.Combine(root, ChunkIndex.StoreLockFileName)));
        Assert.Equal(UnixFileMode.UserRead | UnixFileMode.UserWrite, File.GetUnixFileMode(Path.Combine(root, "index-x.lock")));
        Assert.Equal(UnixFileMode.UserRead | UnixFileMode.UserWrite, File.GetUnixFileMode(Path.Combine(root, StoreOwnerLock.FileName)));
    }

    [SkippableFact]
    [UnsupportedOSPlatform("windows")]
    public void ExistingStoreRoot_KeepsItsMode()
    {
        Skip.If(OperatingSystem.IsWindows(), "POSIX modes");
        Directory.CreateDirectory(Root);
        UnixFileMode shared = UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute | UnixFileMode.GroupRead | UnixFileMode.GroupExecute;
        File.SetUnixFileMode(Root, shared);

        using (ChunkIndex.AcquireStoreLock(Root, TimeSpan.Zero)) { }

        Assert.Equal(shared, File.GetUnixFileMode(Root)); // an operator's choice for an existing store is not overridden
    }

    // ------------------------------------------------------------------------------------------------ owner lock

    [Fact]
    public void IsSharingViolation_OnlyForARealLockConflict()
    {
        // Windows: ERROR_SHARING_VIOLATION (32) / ERROR_LOCK_VIOLATION (33). Linux: EAGAIN (11), what .NET reports for a flock
        // conflict; macOS / BSD: EWOULDBLOCK (35). Anything else (disk full, I/O error) is not "another owner".
        int conflict = OperatingSystem.IsWindows() ? unchecked((int)0x80070020) : OperatingSystem.IsMacOS() ? 35 : 11;
        int diskFull = OperatingSystem.IsWindows() ? unchecked((int)0x80070070) : 28;
        int ioError = OperatingSystem.IsWindows() ? unchecked((int)0x80070017) : 5;

        Assert.True(StoreOwnerLock.IsSharingViolation(new IOException("busy", conflict)));
        Assert.False(StoreOwnerLock.IsSharingViolation(new IOException("full", diskFull)));
        Assert.False(StoreOwnerLock.IsSharingViolation(new IOException("io", ioError)));
        Assert.False(StoreOwnerLock.IsSharingViolation(new FileNotFoundException("gone")));
    }

    /// <summary>A probe (<see cref="StoreOwnerLock.IsHeldByAnother"/>) opens the lock file exclusively for a moment: it must not make a starting server refuse.</summary>
    [Fact]
    public void TryAcquire_WhileAProbeHoldsTheFileBriefly_Succeeds()
    {
        Directory.CreateDirectory(Root);
        string path = Path.Combine(Root, StoreOwnerLock.FileName);
        var probe = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None); // what a probe holds
        int busy = 0;
        StoreOwnerLock.AfterBusyAttempt = () => { busy++; probe.Dispose(); }; // the probe ends while the acquirer waits
        try
        {
            using StoreOwnerLock? owner = StoreOwnerLock.TryAcquire(Root);
            Assert.NotNull(owner);
            Assert.True(busy >= 1, "control: the first attempt did meet the probe");
        }
        finally
        {
            StoreOwnerLock.AfterBusyAttempt = null;
            probe.Dispose();
        }
    }

    [Fact]
    public void TryAcquire_HeldByARealOwner_ReturnsNullAfterAShortGrace()
    {
        Directory.CreateDirectory(Root);
        using var owner = new FileStream(Path.Combine(Root, StoreOwnerLock.FileName), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);

        var sw = Stopwatch.StartNew();
        Assert.Null(StoreOwnerLock.TryAcquire(Root));
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(2), $"took {sw.Elapsed}: the grace must stay short");
    }
}
