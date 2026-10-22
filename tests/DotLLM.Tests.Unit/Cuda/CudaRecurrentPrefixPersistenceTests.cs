using DotLLM.Core.Attention;
using DotLLM.Engine.PromptCache;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaRecurrentPrefixPersistenceTests
{
    [SkippableTheory]
    [InlineData(KvCacheDType.F32, KvCacheDType.F32, RecurrentPrefixSnapshotStorage.Host)]
    [InlineData(KvCacheDType.F32, KvCacheDType.F32, RecurrentPrefixSnapshotStorage.Device)]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q4_0, RecurrentPrefixSnapshotStorage.Host)]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q4_0, RecurrentPrefixSnapshotStorage.Device)]
    public void FileRoundTrip_PreservesExactContinuationAcrossOverwrittenLiveState(KvCacheDType keys, KvCacheDType values, RecurrentPrefixSnapshotStorage storage)
    {
        using var fixture = new Fixture(keys, values);
        var provider = Assert.IsAssignableFrom<IPersistableRecurrentPrefixSnapshotProvider>(fixture.Model);
        using var kv = fixture.Model.CreateKvCache(64);
        fixture.Model.Forward([1, 2, 3, 4], [0, 1, 2, 3], -1, kv).Dispose();
        using var original = provider.CapturePrefixSnapshot(kv, 4, storage);
        var persistent = Assert.IsAssignableFrom<IPersistableRecurrentPrefixSnapshot>(original);
        float[] expected = Next(fixture.Model, kv, 5, 4);
        string path = Path.Combine(Path.GetTempPath(), $"prefix-payload-{Guid.NewGuid():N}.bin");
        try
        {
            using var file = new FileStream(path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 4096, FileOptions.Asynchronous | FileOptions.SequentialScan);
            using (var forward = new ForwardOnlyWriter(file)) persistent.WriteTo(forward);
            file.Flush(); long serializedBytes = file.Length;
            Assert.InRange(serializedBytes, original.ByteCount, original.ByteCount + 65536);
            fixture.Model.ResetSequenceState();
            using var other = fixture.Model.CreateKvCache(64);
            fixture.Model.Forward([7, 6, 5], [0, 1, 2], -1, other).Dispose();
            file.Position = 0;
            using var restored = provider.ReadPrefixSnapshot(file, storage, original.ByteCount);
            Assert.Equal(serializedBytes, file.Position);
            Assert.Equal(original.ByteCount, restored.ByteCount); Assert.Equal(original.Identity, restored.Identity);
            Assert.Equal(4, restored.PrefixLength); Assert.Equal(storage, restored.Storage);
            provider.RestorePrefixSnapshot(restored, other);
            Assert.Equal(expected, Next(fixture.Model, other, 5, 4));
        }
        finally { File.Delete(path); }
    }

    [SkippableFact]
    public void CorruptTruncatedOversizedForeignAndDisposedPayloads_AreRejected()
    {
        using var fixture = new Fixture();
        var provider = Assert.IsAssignableFrom<IPersistableRecurrentPrefixSnapshotProvider>(fixture.Model);
        using var kv = fixture.Model.CreateKvCache(64);
        fixture.Model.Forward([1, 2, 3, 4], [0, 1, 2, 3], -1, kv).Dispose();
        using var original = provider.CapturePrefixSnapshot(kv, 4, RecurrentPrefixSnapshotStorage.Host);
        var persistent = Assert.IsAssignableFrom<IPersistableRecurrentPrefixSnapshot>(original);
        using var output = new MemoryStream(); persistent.WriteTo(output);
        byte[] bytes = output.ToArray();
        Assert.Throws<InvalidDataException>(() => provider.ReadPrefixSnapshot(new MemoryStream(bytes), RecurrentPrefixSnapshotStorage.Host, original.ByteCount - 1));
        byte[] corrupt = (byte[])bytes.Clone(); corrupt[corrupt.Length / 2] ^= 0x40;
        Assert.Throws<InvalidDataException>(() => provider.ReadPrefixSnapshot(new MemoryStream(corrupt), RecurrentPrefixSnapshotStorage.Host, original.ByteCount));
        Assert.Throws<InvalidDataException>(() => provider.ReadPrefixSnapshot(new MemoryStream(bytes[..^1]), RecurrentPrefixSnapshotStorage.Host, original.ByteCount));
        byte[] header = (byte[])bytes.Clone(); header[0] ^= 1;
        Assert.Throws<InvalidDataException>(() => provider.ReadPrefixSnapshot(new MemoryStream(header), RecurrentPrefixSnapshotStorage.Host, original.ByteCount));
        using var other = new Fixture();
        var otherProvider = Assert.IsAssignableFrom<IPersistableRecurrentPrefixSnapshotProvider>(other.Model);
        Assert.Throws<InvalidDataException>(() => otherProvider.ReadPrefixSnapshot(new MemoryStream(bytes), RecurrentPrefixSnapshotStorage.Host, original.ByteCount));
        original.Dispose();
        Assert.Throws<ObjectDisposedException>(() => persistent.WriteTo(new MemoryStream()));
    }

    [SkippableFact]
    public void FailedReloadAfterValidation_LeavesLiveStateAndFutureImportsUsable()
    {
        using var fixture = new Fixture();
        var provider = Assert.IsAssignableFrom<IPersistableRecurrentPrefixSnapshotProvider>(fixture.Model);
        using var kv = fixture.Model.CreateKvCache(64);
        fixture.Model.Forward([1, 2, 3, 4], [0, 1, 2, 3], -1, kv).Dispose();
        using var original = provider.CapturePrefixSnapshot(kv, 4, RecurrentPrefixSnapshotStorage.Host);
        using var output = new MemoryStream(); ((IPersistableRecurrentPrefixSnapshot)original).WriteTo(output);
        byte[] bytes = output.ToArray();
        float[] expected = Next(fixture.Model, kv, 5, 4);
        provider.RestorePrefixSnapshot(original, kv);
        using var broken = new FailOnReloadStream(bytes);
        Assert.Throws<IOException>(() => provider.ReadPrefixSnapshot(broken, RecurrentPrefixSnapshotStorage.Host, original.ByteCount));
        Assert.Equal(expected, Next(fixture.Model, kv, 5, 4));
        using var good = provider.ReadPrefixSnapshot(new MemoryStream(bytes), RecurrentPrefixSnapshotStorage.Host, original.ByteCount);
        provider.RestorePrefixSnapshot(good, kv); Assert.Equal(expected, Next(fixture.Model, kv, 5, 4));
    }

    [SkippableFact]
    public void EngineDiskSpill_RestoresActualCudaStateWithExactColdLogits()
    {
        using var fixture = new Fixture(KvCacheDType.Q8_0, KvCacheDType.Q4_0);
        var provider = new HostOnlyProvider(Assert.IsAssignableFrom<IPersistableRecurrentPrefixSnapshotProvider>(fixture.Model));
        using var kv = fixture.Model.CreateKvCache(64);
        fixture.Model.Forward([1, 2, 3, 4], [0, 1, 2, 3], -1, kv).Dispose();
        long bytes = provider.EstimatePrefixSnapshotBytes(kv, 4);
        string directory = Path.Combine(Path.GetTempPath(), $"cuda-disk-tier-{Guid.NewGuid():N}");
        int[][] histories = [[1, 2, 3, 4], [7, 6, 5, 4], [2, 4, 6, 8]];
        var expected = new List<float[]>();
        try
        {
            using var cache = new RecurrentPrefixSnapshotCache(provider, new RecurrentPrefixCacheOptions
            {
                MaxEntries = 3, HostBudgetBytes = bytes, DiskDirectory = directory,
                DiskBudgetBytes = 3 * (bytes + 131072),
            });
            foreach (int[] history in histories)
            {
                fixture.Model.ResetSequenceState(); kv.Rollback(0);
                fixture.Model.Forward(history, [0, 1, 2, 3], -1, kv).Dispose();
                Assert.True(cache.TryCapture(history, kv, 4));
                expected.Add(Next(fixture.Model, kv, 5, 4));
            }
            Assert.Equal(3, cache.Statistics.Entries);
            Assert.Equal(2, cache.Statistics.DiskEntries);
            foreach (int index in new[] { 0, 1, 2, 0, 1, 2 })
            {
                int[] tokens = [.. histories[index], 5];
                fixture.Model.ResetSequenceState(); kv.Rollback(0);
                Assert.Equal(4, cache.TryRestore(tokens, kv, out string? tier, out _));
                Assert.Equal("disk", tier);
                Assert.Equal(expected[index], Next(fixture.Model, kv, 5, 4));
                Assert.InRange(cache.Statistics.HostBytes, 0, bytes);
            }
            Assert.Equal(6, cache.Statistics.DiskHits);
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true); }
    }

    private sealed class HostOnlyProvider(IPersistableRecurrentPrefixSnapshotProvider inner) : IPersistableRecurrentPrefixSnapshotProvider
    {
        public string PrefixSnapshotIdentity => inner.PrefixSnapshotIdentity;
        public bool SupportsPrefixSnapshotStorage(RecurrentPrefixSnapshotStorage storage) => storage == RecurrentPrefixSnapshotStorage.Host;
        public bool HasPrefixSnapshotRequestHeadroom(IKvCache kvCache) => inner.HasPrefixSnapshotRequestHeadroom(kvCache);
        public long EstimatePrefixSnapshotBytes(IKvCache kvCache, int prefixLength) => inner.EstimatePrefixSnapshotBytes(kvCache, prefixLength);
        public IRecurrentPrefixSnapshot CapturePrefixSnapshot(IKvCache kvCache, int prefixLength, RecurrentPrefixSnapshotStorage storage) => inner.CapturePrefixSnapshot(kvCache, prefixLength, storage);
        public IRecurrentPrefixSnapshot TransferPrefixSnapshot(IRecurrentPrefixSnapshot source, RecurrentPrefixSnapshotStorage storage) => inner.TransferPrefixSnapshot(source, storage);
        public void RestorePrefixSnapshot(IRecurrentPrefixSnapshot snapshot, IKvCache kvCache) => inner.RestorePrefixSnapshot(snapshot, kvCache);
        public IRecurrentPrefixSnapshot ReadPrefixSnapshot(Stream source, RecurrentPrefixSnapshotStorage storage, long maxBytes) => inner.ReadPrefixSnapshot(source, storage, maxBytes);
    }
    private static unsafe float[] Next(CudaQwen3HybridDenseTransformerModel model, DotLLM.Core.Attention.IKvCache kv, int token, int position)
    {
        using var logits = model.Forward([token], [position], -1, kv);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, (int)logits.ElementCount).ToArray();
    }

    private sealed class Fixture : IDisposable
    {
        private readonly string _path;
        private readonly GgufFile _gguf;
        public CudaQwen3HybridDenseTransformerModel Model { get; }
        public Fixture(KvCacheDType keys = KvCacheDType.F32, KvCacheDType values = KvCacheDType.F32)
        {
            Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
            string ptx = Path.Combine(AppContext.BaseDirectory, "ptx");
            Skip.IfNot(Directory.Exists(ptx), "PTX files missing");
            _path = Path.Combine(Path.GetTempPath(), $"prefix-persist-{Guid.NewGuid():N}.gguf");
            SyntheticQwen35HybridDenseMtpGguf.Write(_path, withMtp: false, contextLength: 64, pq2_0Projections: true);
            _gguf = GgufFile.Open(_path);
            var config = GgufModelConfigExtractor.Extract(_gguf.Metadata);
            Model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(_gguf, config, 0, ptx);
            Model.ConfigureKvStorage(keys, values);
        }
        public void Dispose() { Model.Dispose(); _gguf.Dispose(); File.Delete(_path); }
    }

    private sealed class ForwardOnlyWriter(Stream inner) : Stream
    {
        public override bool CanRead => false; public override bool CanSeek => false; public override bool CanWrite => true;
        public override long Length => throw new NotSupportedException(); public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() => inner.Flush();
        public override void Write(byte[] buffer, int offset, int count) => inner.Write(buffer, offset, count);
        public override void Write(ReadOnlySpan<byte> buffer) => inner.Write(buffer);
        public override ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default) => inner.WriteAsync(buffer, cancellationToken);
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException(); public override void SetLength(long value) => throw new NotSupportedException();
    }

    private sealed class FailOnReloadStream(byte[] bytes) : Stream
    {
        private readonly MemoryStream _inner = new(bytes, writable: false);
        private bool _fail;
        public override bool CanRead => true; public override bool CanSeek => true; public override bool CanWrite => false;
        public override long Length => _inner.Length;
        public override long Position { get => _inner.Position; set { if (_inner.Position == _inner.Length && value < _inner.Position) _fail = true; _inner.Position = value; } }
        public override long Seek(long offset, SeekOrigin origin) { long position = origin switch { SeekOrigin.Begin => offset, SeekOrigin.Current => Position + offset, _ => Length + offset }; Position = position; return position; }
        public override int Read(byte[] buffer, int offset, int count) { if (_fail) throw new IOException("injected second-pass read failure"); return _inner.Read(buffer, offset, count); }
        public override int Read(Span<byte> buffer) { if (_fail) throw new IOException("injected second-pass read failure"); return _inner.Read(buffer); }
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) => _fail ? ValueTask.FromException<int>(new IOException("injected second-pass read failure")) : _inner.ReadAsync(buffer, cancellationToken);
        public override void Flush() { }
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException(); public override void SetLength(long value) => throw new NotSupportedException();
        protected override void Dispose(bool disposing) { if (disposing) _inner.Dispose(); base.Dispose(disposing); }
    }
}
