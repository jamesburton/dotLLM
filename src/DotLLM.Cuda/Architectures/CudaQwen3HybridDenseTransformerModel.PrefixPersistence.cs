using System.Buffers;
using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using DotLLM.Core.Models;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

public sealed unsafe partial class CudaQwen3HybridDenseTransformerModel : IPersistableRecurrentPrefixSnapshotProvider
{
    private const uint PrefixPayloadMagic = 0x43505344; // DSPC, little-endian
    private const int PrefixPayloadVersion = 1;
    private const int PrefixPayloadMaxHeader = 65536;
    private const int PrefixIoChunk = 4 * 1024 * 1024;

    private long[] PersistenceSegmentSizes(int length)
    {
        int stride = checked(Config.NumKvHeads * Config.HeadDim);
        var sizes = new long[checked(_attentionLayerCount * 2 + 2)];
        for (int i = 0; i < _attentionLayerCount; i++)
        {
            sizes[2 * i] = checked(length * QuantRowBytes(_kvKeyDType, stride));
            sizes[2 * i + 1] = checked(length * QuantRowBytes(_kvValueDType, stride));
        }
        sizes[^2] = checked((long)_gdnCache.NumGdnLayers * _gdnCache.ConvStateElements * sizeof(float));
        sizes[^1] = checked((long)_gdnCache.NumGdnLayers * _gdnCache.GdnStateElements * sizeof(float));
        return sizes;
    }

    private byte[] PrefixSourceFingerprint()
    {
        using var metadata = new MemoryStream();
        using (var writer = new BinaryWriter(metadata, Encoding.UTF8, leaveOpen: true))
        {
            var paths = _gguf?.ShardPaths;
            writer.Write(paths?.Count ?? 0);
            if (paths is not null)
                foreach (string path in paths)
                {
                    string full = Path.GetFullPath(path);
                    writer.Write(OperatingSystem.IsWindows() ? full.ToUpperInvariant() : full);
                    var info = new FileInfo(full);
                    writer.Write(info.Length); writer.Write(info.LastWriteTimeUtc.Ticks);
                }
        }
        return SHA256.HashData(metadata.GetBuffer().AsSpan(0, checked((int)metadata.Length)));
    }

    private byte[] PrefixConfigurationFingerprint()
    {
        using var metadata = new MemoryStream();
        using (var writer = new BinaryWriter(metadata, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write((int)Config.Architecture); writer.Write(Config.NumLayers); writer.Write(Config.HiddenSize);
            writer.Write(Config.VocabSize); writer.Write(Config.IntermediateSize); writer.Write(Config.NumAttentionHeads);
            writer.Write(Config.NumKvHeads); writer.Write(Config.HeadDim); writer.Write(Config.MaxSequenceLength);
            writer.Write(Config.NormEpsilon); writer.Write((int)_kvKeyDType); writer.Write((int)_kvValueDType);
            writer.Write(_attentionLayerCount); writer.Write(_kvSlotForLayer.Length);
            foreach (int slot in _kvSlotForLayer) writer.Write(slot);
            var layout = Config.HybridLayout;
            writer.Write(layout is not null);
            if (layout is not null)
                for (int i = 0; i < layout.LayerKind.Length; i++)
                { writer.Write((int)layout.LayerKind[i]); writer.Write(layout.HeadCountKv[i]); writer.Write(layout.FeedForwardLength[i]); }
            writer.Write(_gdn.FullAttnInterval); writer.Write(_gdn.NVHead); writer.Write(_gdn.NKHead);
            writer.Write(_gdn.DState); writer.Write(_gdn.DInner); writer.Write(_gdn.DConv);
            writer.Write(_ropeDim); writer.Write(_ropeTheta); writer.Write(_mtpHead is not null);
            writer.Write(Config.RoPEConfig.HasValue);
            if (Config.RoPEConfig is { } rope)
            {
                writer.Write(rope.Theta); writer.Write(rope.DimensionCount); writer.Write((int)rope.Type);
                writer.Write((int)rope.ScalingType); writer.Write(rope.ScalingFactor); writer.Write(rope.OrigMaxSeqLen);
                writer.Write(rope.AttnFactor); writer.Write(rope.BetaFast); writer.Write(rope.BetaSlow);
            }
        }
        return SHA256.HashData(metadata.GetBuffer().AsSpan(0, checked((int)metadata.Length)));
    }

    private byte[] BuildPrefixPayloadHeader(CudaPrefixSnapshot snapshot)
    {
        byte[] identity = Encoding.UTF8.GetBytes(snapshot.Identity);
        long[] sizes = PersistenceSegmentSizes(snapshot.PrefixLength);
        if (identity.Length is < 1 or > 1024 || sizes.Sum() != snapshot.ByteCount)
            throw new InvalidDataException("Snapshot identity or layout is not serializable.");
        using var buffer = new MemoryStream();
        using (var writer = new BinaryWriter(buffer, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(PrefixPayloadMagic); writer.Write(PrefixPayloadVersion); writer.Write(0);
            writer.Write(identity.Length); writer.Write(identity);
            writer.Write(PrefixSourceFingerprint()); writer.Write(PrefixConfigurationFingerprint());
            writer.Write(snapshot.PrefixLength); writer.Write(snapshot.ByteCount); writer.Write(sizes.Length);
            foreach (long size in sizes) writer.Write(size);
        }
        byte[] header = buffer.ToArray();
        if (header.Length > PrefixPayloadMaxHeader) throw new InvalidDataException("Snapshot header exceeds its limit.");
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(8, 4), header.Length);
        return header;
    }

    private (byte[] Header, int Length, long Bytes) ReadPrefixPayloadHeader(Stream source, long maxBytes)
    {
        Span<byte> start = stackalloc byte[12]; source.ReadExactly(start);
        int headerBytes = BinaryPrimitives.ReadInt32LittleEndian(start[8..]);
        if (BinaryPrimitives.ReadUInt32LittleEndian(start) != PrefixPayloadMagic
            || BinaryPrimitives.ReadInt32LittleEndian(start[4..]) != PrefixPayloadVersion
            || headerBytes is < 112 or > PrefixPayloadMaxHeader)
            throw new InvalidDataException("Invalid CUDA prefix snapshot header.");
        byte[] header = new byte[headerBytes]; start.CopyTo(header); source.ReadExactly(header.AsSpan(12));
        try
        {
            using var reader = new BinaryReader(new MemoryStream(header, writable: false), Encoding.UTF8);
            reader.BaseStream.Position = 12;
            int identityBytes = reader.ReadInt32();
            if (identityBytes is < 1 or > 1024) throw new InvalidDataException("Invalid snapshot identity size.");
            string identity = new UTF8Encoding(false, true).GetString(reader.ReadBytes(identityBytes));
            if (identity != PrefixSnapshotIdentity) throw new InvalidDataException("Snapshot belongs to a different model instance or settings.");
            if (!CryptographicOperations.FixedTimeEquals(reader.ReadBytes(32), PrefixSourceFingerprint())
                || !CryptographicOperations.FixedTimeEquals(reader.ReadBytes(32), PrefixConfigurationFingerprint()))
                throw new InvalidDataException("Snapshot source or forward configuration mismatch.");
            int length = reader.ReadInt32(); long bytes = reader.ReadInt64(); int count = reader.ReadInt32();
            if (length <= 0 || length > Config.MaxSequenceLength || bytes <= 0 || bytes > maxBytes)
                throw new InvalidDataException("Snapshot token or byte budget is invalid.");
            long[] sizes = PersistenceSegmentSizes(length);
            if (count != sizes.Length || bytes != sizes.Sum()) throw new InvalidDataException("Snapshot geometry mismatch.");
            foreach (long size in sizes)
                if (reader.ReadInt64() != size) throw new InvalidDataException("Snapshot segment geometry mismatch.");
            if (reader.BaseStream.Position != header.Length) throw new InvalidDataException("Unexpected snapshot metadata.");
            return (header, length, bytes);
        }
        catch (Exception exception) when (exception is EndOfStreamException or DecoderFallbackException or OverflowException)
        { throw new InvalidDataException("Malformed snapshot metadata.", exception); }
    }

    /// <inheritdoc/>
    public IRecurrentPrefixSnapshot ReadPrefixSnapshot(Stream source, RecurrentPrefixSnapshotStorage storage, long maxBytes)
    {
        ArgumentNullException.ThrowIfNull(source);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!source.CanRead || !source.CanSeek) throw new NotSupportedException("Snapshot import needs a readable, seekable stream for validation before allocation.");
        if (!SupportsPrefixSnapshotStorage(storage)) throw new NotSupportedException("Snapshot storage is unavailable.");
        var saved = ReadPrefixPayloadHeader(source, maxBytes);
        long dataStart = source.Position;
        if (saved.Bytes > source.Length - dataStart - 32) throw new InvalidDataException("Truncated snapshot payload.");
        // No GPU or pinned snapshot allocation until the complete envelope and payload have passed validation.
        byte[] checksum;
        using (var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256))
        {
            hash.AppendData(saved.Header);
            Span<byte> scratch = stackalloc byte[65536];
            for (long remaining = saved.Bytes; remaining > 0;)
            {
                int count = (int)Math.Min(scratch.Length, remaining);
                source.ReadExactly(scratch[..count]); hash.AppendData(scratch[..count]); remaining -= count;
            }
            checksum = new byte[32]; source.ReadExactly(checksum);
            if (!CryptographicOperations.FixedTimeEquals(hash.GetHashAndReset(), checksum))
                throw new InvalidDataException("Snapshot checksum mismatch.");
        }
        source.Position = dataStart;
        _context.MakeCurrent();
        var snapshot = AllocatePrefixSnapshot(saved.Length, saved.Bytes, storage);
        try
        {
            using var memory = storage == RecurrentPrefixSnapshotStorage.Device
                ? PrefixIoMemory.AllocatePinned((int)Math.Min(PrefixIoChunk, saved.Bytes)) : new PrefixIoMemory();
            using var copy = storage == RecurrentPrefixSnapshotStorage.Device ? CudaStream.Create() : null;
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256); hash.AppendData(saved.Header);
            try
            {
                for (long offset = 0; offset < saved.Bytes;)
                {
                    int count = (int)Math.Min(PrefixIoChunk, saved.Bytes - offset);
                    memory.Window(storage == RecurrentPrefixSnapshotStorage.Host ? snapshot.Data + (nint)offset : memory.OwnedPointer, count);
                    ReadPrefixBytesAsync(source, memory.Memory);
                    hash.AppendData(memory.GetSpan());
                    if (copy is not null)
                    {
                        CopyPrefixBytes(snapshot.Data + (nint)offset, RecurrentPrefixSnapshotStorage.Device,
                            memory.OwnedPointer, RecurrentPrefixSnapshotStorage.Host, count, copy.Handle);
                        copy.Synchronize();
                    }
                    offset += count;
                }
            }
            finally { copy?.Synchronize(); }
            Span<byte> footer = stackalloc byte[32]; source.ReadExactly(footer);
            // Recheck after the second read: a changed file cannot expose a different independently allocated state.
            if (!CryptographicOperations.FixedTimeEquals(footer, checksum)
                || !CryptographicOperations.FixedTimeEquals(hash.GetHashAndReset(), checksum))
                throw new InvalidDataException("Snapshot changed while loading.");
            return snapshot;
        }
        catch
        {
            try { snapshot.Dispose(); } catch (Exception) { /* Preserve the original I/O failure. */ }
            throw;
        }
    }

    private void WritePrefixSnapshot(CudaPrefixSnapshot snapshot, Stream destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        RequirePrefixSnapshot(snapshot);
        if (!destination.CanWrite) throw new NotSupportedException("Snapshot destination must be writable.");
        byte[] header = BuildPrefixPayloadHeader(snapshot);
        _context.MakeCurrent();
        using var memory = snapshot.Storage == RecurrentPrefixSnapshotStorage.Device
            ? PrefixIoMemory.AllocatePinned((int)Math.Min(PrefixIoChunk, snapshot.ByteCount)) : new PrefixIoMemory();
        using var copy = snapshot.Storage == RecurrentPrefixSnapshotStorage.Device ? CudaStream.Create() : null;
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256); hash.AppendData(header);
        destination.Write(header);
        try
        {
            for (long offset = 0; offset < snapshot.ByteCount;)
            {
                int count = (int)Math.Min(PrefixIoChunk, snapshot.ByteCount - offset);
                memory.Window(copy is null ? snapshot.Data + (nint)offset : memory.OwnedPointer, count);
                if (copy is not null)
                {
                    CopyPrefixBytes(memory.OwnedPointer, RecurrentPrefixSnapshotStorage.Host,
                        snapshot.Data + (nint)offset, RecurrentPrefixSnapshotStorage.Device, count, copy.Handle);
                    copy.Synchronize();
                }
                hash.AppendData(memory.GetSpan());
                destination.WriteAsync(memory.Memory).AsTask().GetAwaiter().GetResult();
                offset += count;
            }
        }
        finally { copy?.Synchronize(); }
        destination.Write(hash.GetHashAndReset());
    }

    private static void ReadPrefixBytesAsync(Stream source, Memory<byte> memory)
    {
        while (!memory.IsEmpty)
        {
            int read = source.ReadAsync(memory).AsTask().GetAwaiter().GetResult();
            if (read <= 0) throw new EndOfStreamException("Truncated snapshot payload during import.");
            memory = memory[read..];
        }
    }

    // A pointer-backed Memory exposes CUDA-pinned pages to asynchronous FileStream I/O without a managed tensor copy.
    private sealed class PrefixIoMemory : MemoryManager<byte>
    {
        private nint _pointer;
        private int _length;
        private int _capacity;
        private bool _disposed;
        public nint OwnedPointer { get; private set; }
        public static PrefixIoMemory AllocatePinned(int bytes)
        {
            CudaDriverApi.cuMemHostAlloc(out nint pointer, (nuint)bytes, 0).ThrowOnError();
            return new PrefixIoMemory { OwnedPointer = pointer, _pointer = pointer, _length = bytes, _capacity = bytes };
        }
        public void Window(nint pointer, int bytes)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (bytes < 0 || (OwnedPointer != 0 && (pointer != OwnedPointer || bytes > _capacity))) throw new ArgumentOutOfRangeException(nameof(bytes));
            _pointer = pointer; _length = bytes;
        }
        public override Span<byte> GetSpan() { ObjectDisposedException.ThrowIf(_disposed, this); return new Span<byte>((void*)_pointer, _length); }
        public override MemoryHandle Pin(int elementIndex = 0)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if ((uint)elementIndex > (uint)_length) throw new ArgumentOutOfRangeException(nameof(elementIndex));
            return new MemoryHandle((void*)(_pointer + elementIndex), default, this);
        }
        public override void Unpin() { }
        protected override void Dispose(bool disposing)
        {
            if (_disposed) return;
            _disposed = true; _pointer = 0; _length = 0;
            nint owned = OwnedPointer; OwnedPointer = 0;
            if (owned != 0) CudaTeardownDiagnostics.RecordDestroy("prefix I/O pinned staging", CudaDriverApi.cuMemFreeHost(owned));
        }
    }

    private sealed partial class CudaPrefixSnapshot : IPersistableRecurrentPrefixSnapshot
    {
        public void WriteTo(Stream destination) => Owner.WritePrefixSnapshot(this, destination);
    }
}
