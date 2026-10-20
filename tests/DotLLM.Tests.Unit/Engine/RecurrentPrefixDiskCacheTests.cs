using DotLLM.Core.Attention;
using System.Security.AccessControl;
using System.Security.Principal;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

public sealed class RecurrentPrefixDiskCacheTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "dotllm-disk-cache-" + Guid.NewGuid().ToString("N"));
    private static int[] Tokens(int id) => Enumerable.Repeat(id, 20).ToArray();
    public void Dispose() { if (Directory.Exists(_directory)) Directory.Delete(_directory, true); }
    private RecurrentPrefixSnapshotCache Cache(Provider provider, long budget = 32768) => new(provider,
        new() { MaxEntries = 3, HostBudgetBytes = 64, DiskDirectory = _directory, DiskBudgetBytes = budget });

    [Fact]
    public void ThreeConversations_WithOneHostSlot_RestoreEveryPrefixAfterWarmup()
    {
        var provider = new Provider();
        using var cache = Cache(provider);
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        for (int id = 1; id <= 3; id++) { provider.State = id * 11; Assert.True(cache.TryCapture(Tokens(id), kv, 19)); }
        Assert.Equal(3, cache.Statistics.Entries);
        for (int turn = 0; turn < 3; turn++)
            for (int id = 1; id <= 3; id++)
            {
                Assert.Equal(19, cache.TryRestore(Tokens(id), kv, out var tier, out _));
                Assert.Equal(id * 11, provider.State);
                Assert.Contains(tier, new[] { "host", "disk" });
                Assert.True(cache.Statistics.DiskBytes <= cache.Statistics.DiskBudgetBytes);
            }
        Assert.True(cache.Statistics.DiskHits >= 3);
        Assert.Equal(64, provider.PeakHostBytes);
        cache.Clear();
        Assert.Equal(0, provider.HostBytes);
        Assert.Empty(Directory.EnumerateFiles(_directory, "*", SearchOption.AllDirectories));
    }

    [Theory]
    [InlineData("payload")]
    [InlineData("header")]
    [InlineData("truncated")]
    [InlineData("trailing")]
    public void CorruptPayload_IsMissBeforeProviderImportOrLiveMutation(string mutation)
    {
        var provider = new Provider();
        using var cache = Cache(provider);
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        provider.State = 11; Assert.True(cache.TryCapture(Tokens(1), kv, 19));
        provider.State = 22; Assert.True(cache.TryCapture(Tokens(2), kv, 19));
        string path = Assert.Single(Directory.EnumerateFiles(_directory, "*.snapshot", SearchOption.AllDirectories));
        using (var file = new FileStream(path, FileMode.Open, FileAccess.ReadWrite))
        {
            if (mutation == "truncated") file.SetLength(file.Length - 1);
            else
            {
                file.Position = mutation == "header" ? 0 : mutation == "trailing" ? file.Length : file.Length - 1;
                file.WriteByte(255);
            }
        }
        provider.State = 77;
        Assert.Equal(0, cache.TryRestore(Tokens(1), kv, out _, out _));
        Assert.Equal(77, provider.State);
        Assert.Equal(0, provider.Imports);
        Assert.Equal(0, cache.Statistics.DiskEntries);
        Assert.False(File.Exists(path));
    }

    [Fact]
    public void IdentityChange_CleansOwnedFilesButPreservesUnrelatedAndOtherInstanceFiles()
    {
        var provider = new Provider();
        using var cache = Cache(provider);
        using var second = Cache(new Provider());
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        Assert.True(cache.TryCapture(Tokens(1), kv, 19)); Assert.True(cache.TryCapture(Tokens(2), kv, 19));
        Assert.True(second.TryCapture(Tokens(1), kv, 19)); Assert.True(second.TryCapture(Tokens(2), kv, 19));
        string unrelated = Path.Combine(_directory, "conversation.txt"); File.WriteAllText(unrelated, "keep");
        provider.PrefixSnapshotIdentity = "changed-model-or-rope";
        Assert.Equal(0, cache.TryRestore(Tokens(1), kv, out _, out _));
        Assert.Single(Directory.EnumerateFiles(_directory, "*.snapshot", SearchOption.AllDirectories));
        Assert.Equal("keep", File.ReadAllText(unrelated));
        Assert.Equal(19, second.TryRestore(Tokens(1), kv, out var tier, out _)); Assert.Equal("disk", tier);
    }

    [Theory]
    [InlineData(1)] // measurement fails without touching disk
    [InlineData(2)] // real write fails after creating and partially writing the temp file
    public void FailedSpill_DoesNotPublishPartialFileOrExceedHostBudget(int failedWrite)
    {
        var provider = new Provider();
        using var cache = Cache(provider);
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        Assert.True(cache.TryCapture(Tokens(1), kv, 19));
        provider.FailWriteOnCall = failedWrite; provider.State = 22;
        Assert.True(cache.TryCapture(Tokens(2), kv, 19));
        Assert.Equal(19, cache.TryRestore(Tokens(2), kv, out _, out _));
        Assert.Equal(22, provider.State);
        Assert.Equal(64, provider.PeakHostBytes);
        Assert.True(Directory.Exists(_directory));
        Assert.Empty(Directory.EnumerateFiles(_directory, "*", SearchOption.AllDirectories));
    }

    [Fact]
    public void DiskBudget_CountsHeadersAndEvictsOldestOwnedEntry()
    {
        var provider = new Provider();
        const long fileBytes = 4096 + 8;
        using var cache = Cache(provider, fileBytes);
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        for (int id = 1; id <= 3; id++) { provider.State = id * 11; Assert.True(cache.TryCapture(Tokens(id), kv, 19)); }
        Assert.Equal(fileBytes, cache.Statistics.DiskBytes);
        Assert.Equal(1, cache.Statistics.DiskEntries);
        Assert.Equal(0, cache.TryRestore(Tokens(1), kv, out _, out _));
        Assert.Equal(19, cache.TryRestore(Tokens(2), kv, out var tier, out _));
        Assert.Equal("disk", tier); Assert.Equal(22, provider.State);
        Assert.True(cache.Statistics.DiskBytes <= fileBytes);
        Assert.Equal(64, provider.PeakHostBytes);
    }

    [Fact]
    public void BudgetSmallerThanHeader_DropsColdEntryWithoutWriting()
    {
        var provider = new Provider();
        using var cache = Cache(provider, 4095);
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        Assert.True(cache.TryCapture(Tokens(1), kv, 19)); Assert.True(cache.TryCapture(Tokens(2), kv, 19));
        Assert.Equal(0, cache.Statistics.DiskBytes);
        Assert.Empty(Directory.EnumerateFiles(_directory, "*", SearchOption.AllDirectories));
        Assert.Equal(0, cache.TryRestore(Tokens(1), kv, out _, out _));
    }

    [Fact]
    public void Store_ReservesTemporaryBytesBeforeCallingProviderWrite()
    {
        var provider = new Provider();
        using var cache = Cache(provider);
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        long observed = -1;
        provider.OnWrite = count => { if (count == 2) observed = cache.Statistics.DiskBytes; };
        Assert.True(cache.TryCapture(Tokens(1), kv, 19)); Assert.True(cache.TryCapture(Tokens(2), kv, 19));
        Assert.Equal(4096 + 8, observed);
        Assert.Equal(4096 + 8, cache.Statistics.DiskBytes);
    }

    [Fact]
    public void IncompatibleProviderImport_IsReleasedAndHandledAsMiss()
    {
        var provider = new Provider();
        using var cache = Cache(provider);
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        Assert.True(cache.TryCapture(Tokens(1), kv, 19)); Assert.True(cache.TryCapture(Tokens(2), kv, 19));
        provider.InvalidImport = true; provider.State = 77;
        Assert.Equal(0, cache.TryRestore(Tokens(1), kv, out _, out _));
        Assert.Equal(77, provider.State);
        Assert.Equal(0, provider.HostBytes);
        Assert.Equal(64, provider.PeakHostBytes);
    }

    [Fact]
    public void SnapshotDirectory_IsRestrictedToCurrentUser()
    {
        using var cache = Cache(new Provider());
        string session = Assert.Single(Directory.EnumerateDirectories(_directory));
        if (OperatingSystem.IsWindows())
        {
            var security = new DirectoryInfo(session).GetAccessControl();
            var user = WindowsIdentity.GetCurrent().User;
            Assert.True(security.AreAccessRulesProtected);
            var rules = security.GetAccessRules(true, true, typeof(SecurityIdentifier)).Cast<FileSystemAccessRule>().ToArray();
            Assert.NotEmpty(rules);
            foreach (var rule in rules)
            {
                Assert.Equal(user, rule.IdentityReference);
                Assert.Equal(AccessControlType.Allow, rule.AccessControlType);
            }
        }
        else
        {
            Assert.Equal(UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute, File.GetUnixFileMode(session));
        }
    }

    [Fact]
    public void ExistingParentDirectory_PreservesPermissionsAndUnrelatedFiles()
    {
        Directory.CreateDirectory(_directory);
        string unrelated = Path.Combine(_directory, "keep.txt"); File.WriteAllText(unrelated, "existing");
        string? windowsBefore = OperatingSystem.IsWindows()
            ? new DirectoryInfo(_directory).GetAccessControl().GetSecurityDescriptorSddlForm(AccessControlSections.Access) : null;
        UnixFileMode unixBefore = OperatingSystem.IsWindows() ? default : File.GetUnixFileMode(_directory);
        using (var cache = Cache(new Provider()))
        {
            if (OperatingSystem.IsWindows())
                Assert.Equal(windowsBefore, new DirectoryInfo(_directory).GetAccessControl().GetSecurityDescriptorSddlForm(AccessControlSections.Access));
            else Assert.Equal(unixBefore, File.GetUnixFileMode(_directory));
        }
        Assert.Equal("existing", File.ReadAllText(unrelated));
    }

    [Fact]
    public void ZeroHostBudget_SpillsDeviceSnapshotsAndImportsBackToDevice()
    {
        var provider = new Provider { DeviceLimit = 1 };
        using var cache = new RecurrentPrefixSnapshotCache(provider,
            new() { MaxEntries = 3, HostBudgetBytes = 0, DiskDirectory = _directory, DiskBudgetBytes = 32768 });
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        for (int id = 1; id <= 3; id++) { provider.State = id * 11; Assert.True(cache.TryCapture(Tokens(id), kv, 19)); }
        Assert.Equal(3, cache.Statistics.Entries);
        for (int id = 1; id <= 3; id++)
        {
            Assert.Equal(19, cache.TryRestore(Tokens(id), kv, out var tier, out _));
            Assert.Equal("disk", tier); Assert.Equal(id * 11, provider.State);
            Assert.Equal(0, provider.HostBytes);
        }
        Assert.Equal(0, provider.PeakHostBytes);
        Assert.Equal(64, provider.DeviceBytes);
    }

    [Fact]
    public void DeviceImport_ThatConsumesRequestHeadroom_FallsBackToReservedHostBudget()
    {
        var provider = new Provider { DeviceLimit = 1 };
        using var cache = Cache(provider);
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        for (int id = 1; id <= 3; id++) { provider.State = id * 11; Assert.True(cache.TryCapture(Tokens(id), kv, 19)); }
        provider.RequireEmptyDeviceForRequest = true;
        Assert.Equal(19, cache.TryRestore(Tokens(2), kv, out var tier, out _));
        Assert.Equal("disk", tier); Assert.Equal(22, provider.State);
        Assert.Equal(0, provider.DeviceBytes);
        Assert.Equal(64, provider.HostBytes);
        Assert.Equal(64, provider.PeakHostBytes);
    }

    [Fact]
    public void RequestAdmission_ProtectsRequestedDiskPrefixFromColdDeviceReclamation()
    {
        var provider = new Provider { DeviceLimit = 1 };
        using var cache = Cache(provider, 4096 + 8);
        using var kv = new SimpleKvCache(1, 1, 4, 64);
        for (int id = 1; id <= 3; id++) { provider.State = id * 11; Assert.True(cache.TryCapture(Tokens(id), kv, 19)); }
        Assert.Equal(1, cache.Statistics.DiskEntries); // B disk, A device, C host
        provider.RequireEmptyDeviceForRequest = true;
        Assert.Equal(19, cache.TryRestore(Tokens(2), kv, out var tier, out _));
        Assert.Equal("disk", tier); Assert.Equal(22, provider.State);
        Assert.Equal(0, provider.DeviceBytes);
        Assert.Equal(64, provider.HostBytes);
        Assert.True(cache.Statistics.DiskBytes <= 4096 + 8);
    }

    internal sealed class Provider : IPersistableRecurrentPrefixSnapshotProvider
    {
        public int State, Imports;
        public long HostBytes, PeakHostBytes;
        public long DeviceBytes;
        public int DeviceLimit;
        public bool RequireEmptyDeviceForRequest;
        public int FailWriteOnCall, Writes;
        public bool InvalidImport;
        public Action<int>? OnWrite;
        public string PrefixSnapshotIdentity { get; set; } = "model-path-kv-layout-rope-mtp";
        public bool SupportsPrefixSnapshotStorage(RecurrentPrefixSnapshotStorage storage) => storage == RecurrentPrefixSnapshotStorage.Host || DeviceLimit > 0;
        public bool HasPrefixSnapshotRequestHeadroom(IKvCache cache) => !RequireEmptyDeviceForRequest || DeviceBytes == 0;
        public long EstimatePrefixSnapshotBytes(IKvCache kv, int length) => 64;
        public IRecurrentPrefixSnapshot CapturePrefixSnapshot(IKvCache kv, int length, RecurrentPrefixSnapshotStorage storage) => Allocate(length, State, storage);
        private Snapshot Allocate(int length, int state, RecurrentPrefixSnapshotStorage storage)
        {
            if (storage == RecurrentPrefixSnapshotStorage.Device)
            {
                if (DeviceBytes / 64 >= DeviceLimit) throw new IOException("device admission refused");
                DeviceBytes += 64;
            }
            else { HostBytes += 64; PeakHostBytes = Math.Max(PeakHostBytes, HostBytes); }
            return new Snapshot(this, length, state, storage);
        }
        public IRecurrentPrefixSnapshot TransferPrefixSnapshot(IRecurrentPrefixSnapshot source, RecurrentPrefixSnapshotStorage storage) => throw new NotSupportedException();
        public void RestorePrefixSnapshot(IRecurrentPrefixSnapshot snapshot, IKvCache kv) => State = ((Snapshot)snapshot).State;
        public IRecurrentPrefixSnapshot ReadPrefixSnapshot(Stream source, RecurrentPrefixSnapshotStorage storage, long maxBytes)
        {
            if (maxBytes < 64) throw new InvalidDataException();
            using var reader = new BinaryReader(source, System.Text.Encoding.UTF8, leaveOpen: true);
            int length = reader.ReadInt32(), state = reader.ReadInt32();
            Imports++;
            return Allocate(InvalidImport ? length + 1 : length, state, storage);
        }
        private sealed class Snapshot(Provider owner, int length, int state, RecurrentPrefixSnapshotStorage storage) : IPersistableRecurrentPrefixSnapshot
        {
            private bool _disposed;
            public int State => state;
            public long ByteCount => 64;
            public int PrefixLength => length;
            public string Identity => owner.PrefixSnapshotIdentity;
            public RecurrentPrefixSnapshotStorage Storage => storage;
            public void WriteTo(Stream destination)
            {
                using var writer = new BinaryWriter(destination, System.Text.Encoding.UTF8, leaveOpen: true);
                owner.Writes++;
                owner.OnWrite?.Invoke(owner.Writes);
                writer.Write(length);
                if (owner.FailWriteOnCall == owner.Writes) throw new IOException("injected spill failure");
                writer.Write(state);
            }
            public void Dispose()
            {
                if (_disposed) return;
                _disposed = true;
                if (storage == RecurrentPrefixSnapshotStorage.Host) owner.HostBytes -= 64;
                else owner.DeviceBytes -= 64;
            }
        }
    }
}
