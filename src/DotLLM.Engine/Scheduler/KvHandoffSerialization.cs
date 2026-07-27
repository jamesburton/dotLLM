using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;

namespace DotLLM.Engine.Scheduler;

/// <summary>
/// Stream (de)serialization of a sequence's KV-cache contents — the cross-<em>process</em> leg of a
/// disaggregated prefill→decode handoff. <see cref="Export"/> writes the per-layer K/V of an
/// <see cref="IHostStagedKvCache"/> to any <see cref="Stream"/> (named pipe, socket, file);
/// <see cref="Import"/> rebuilds them into a fresh cache on the other side. The transport is the
/// caller's choice — this type owns only the wire format, so the same bytes work over an in-memory
/// pipe between two schedulers in one process, a named pipe between two processes on one box, or a
/// socket between two boxes.
/// </summary>
/// <remarks>
/// <para>
/// This is the address-space-crossing sibling of <see cref="StagedKvHandoffTransfer"/> (same
/// <see cref="IHostStagedKvCache"/> staging seam, same per-layer walk): where the staged transfer
/// hands host floats directly to the destination cache, this writes them to a stream because the
/// destination lives in another process. Replica <em>placement</em> (which device each process binds)
/// stays caller-side, exactly as in-process: each process builds its model on its own device.
/// </para>
/// <para>
/// <b>Wire format v1 (little-endian):</b> header <c>magic u32 ("DKVH") · version u16 · reserved u16 ·
/// numLayers i32 · currentLength i32</c>, then per layer <c>elementCount i32 · keys f32[count] ·
/// values f32[count]</c>. Payload floats are raw little-endian IEEE-754 — every supported target is
/// little-endian, and the header magic would be detectably byte-swapped on a mismatched peer.
/// </para>
/// </remarks>
public static class KvHandoffSerialization
{
    private const uint Magic = 0x48564B44; // "DKVH" little-endian
    private const ushort Version = 1;
    private const int HeaderBytes = 4 + 2 + 2 + 4 + 4;

    /// <summary>
    /// Writes <paramref name="source"/>'s first <paramref name="numLayers"/> layers of K/V (at its
    /// current visible length) to <paramref name="stream"/>. The source must implement
    /// <see cref="IHostStagedKvCache"/> (the same staging capability the in-process staged transfer
    /// needs). The stream is not disposed; the caller owns flushing/closing its transport.
    /// </summary>
    public static void Export(IKvCache source, int numLayers, Stream stream)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(stream);
        if (source is not IHostStagedKvCache staged)
            throw new NotSupportedException(
                $"Source KV-cache {source.GetType().Name} does not implement {nameof(IHostStagedKvCache)}.");

        Span<byte> header = stackalloc byte[HeaderBytes];
        BinaryPrimitives.WriteUInt32LittleEndian(header, Magic);
        BinaryPrimitives.WriteUInt16LittleEndian(header[4..], Version);
        BinaryPrimitives.WriteUInt16LittleEndian(header[6..], 0); // reserved
        BinaryPrimitives.WriteInt32LittleEndian(header[8..], numLayers);
        BinaryPrimitives.WriteInt32LittleEndian(header[12..], source.CurrentLength);
        stream.Write(header);

        if (source.CurrentLength == 0) return;

        Span<byte> countBuf = stackalloc byte[4];
        for (int layer = 0; layer < numLayers; layer++)
        {
            // Per-layer rent: KV row width can differ per layer (e.g. Gemma-4 sliding vs global layers).
            int count = staged.StagedLayerElementCount(layer);
            BinaryPrimitives.WriteInt32LittleEndian(countBuf, count);
            stream.Write(countBuf);

            float[] keys = ArrayPool<float>.Shared.Rent(count);
            float[] values = ArrayPool<float>.Shared.Rent(count);
            try
            {
                staged.DownloadLayer(layer, keys.AsSpan(0, count), values.AsSpan(0, count));
                stream.Write(MemoryMarshal.AsBytes(keys.AsSpan(0, count)));
                stream.Write(MemoryMarshal.AsBytes(values.AsSpan(0, count)));
            }
            finally
            {
                ArrayPool<float>.Shared.Return(keys);
                ArrayPool<float>.Shared.Return(values);
            }
        }
    }

    /// <summary>
    /// Reads one exported KV payload from <paramref name="stream"/> into the empty
    /// <paramref name="destination"/> cache (which must implement <see cref="IHostStagedKvCache"/>),
    /// advancing its visible length to the exported sequence length. Handles partial reads (pipes,
    /// sockets) via exact-read semantics. Throws <see cref="InvalidDataException"/> on a header
    /// mismatch (wrong magic/version, or a layer count differing from <paramref name="numLayers"/>).
    /// </summary>
    public static void Import(Stream stream, IKvCache destination, int numLayers)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(destination);
        if (destination is not IHostStagedKvCache staged)
            throw new NotSupportedException(
                $"Destination KV-cache {destination.GetType().Name} does not implement {nameof(IHostStagedKvCache)}.");

        Span<byte> header = stackalloc byte[HeaderBytes];
        stream.ReadExactly(header);
        if (BinaryPrimitives.ReadUInt32LittleEndian(header) != Magic)
            throw new InvalidDataException("KV handoff stream does not start with the DKVH magic.");
        ushort version = BinaryPrimitives.ReadUInt16LittleEndian(header[4..]);
        if (version != Version)
            throw new InvalidDataException($"KV handoff wire version {version} is not supported (expected {Version}).");
        int wireLayers = BinaryPrimitives.ReadInt32LittleEndian(header[8..]);
        if (wireLayers != numLayers)
            throw new InvalidDataException($"KV handoff carries {wireLayers} layers; this replica expects {numLayers}.");
        int length = BinaryPrimitives.ReadInt32LittleEndian(header[12..]);
        if (length < 0 || length > destination.MaxLength)
            throw new InvalidDataException(
                $"KV handoff length {length} does not fit destination cache (MaxLength {destination.MaxLength}).");
        if (length == 0) return;

        Span<byte> countBuf = stackalloc byte[4];
        for (int layer = 0; layer < numLayers; layer++)
        {
            stream.ReadExactly(countBuf);
            int count = BinaryPrimitives.ReadInt32LittleEndian(countBuf);
            if (count < 0 || count % length != 0)
                throw new InvalidDataException($"KV handoff layer {layer} element count {count} is not a multiple of length {length}.");

            float[] keys = ArrayPool<float>.Shared.Rent(count);
            float[] values = ArrayPool<float>.Shared.Rent(count);
            try
            {
                stream.ReadExactly(MemoryMarshal.AsBytes(keys.AsSpan(0, count)));
                stream.ReadExactly(MemoryMarshal.AsBytes(values.AsSpan(0, count)));
                // UploadLayer validates count == length × kvStride(layer) for this cache, so a stride
                // mismatch between the two replicas' models fails loudly here rather than corrupting.
                staged.UploadLayer(layer, length, keys.AsSpan(0, count), values.AsSpan(0, count));
            }
            finally
            {
                ArrayPool<float>.Shared.Return(keys);
                ArrayPool<float>.Shared.Return(values);
            }
        }
    }
}

/// <summary>
/// Handoff strategy that routes the KV contents through the <see cref="KvHandoffSerialization"/> wire
/// format (export → stream → import) instead of handing host floats across directly. Functionally
/// equivalent to <see cref="StagedKvHandoffTransfer"/> in one process, but every byte travels the exact
/// serialized path a cross-process deployment uses — plug it into a <see cref="DisaggregatedScheduler"/>
/// to prove token parity through the wire format itself.
/// </summary>
public sealed class SerializedKvHandoffTransfer : IKvHandoffTransfer
{
    /// <summary>Shared singleton — the strategy is stateless.</summary>
    public static readonly SerializedKvHandoffTransfer Instance = new();

    /// <inheritdoc/>
    public IKvCache Transfer(IKvCache source, ModelConfig config, Func<ModelConfig, int, IKvCache> destinationFactory)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destinationFactory);

        IKvCache destination = destinationFactory(config, source.MaxLength);
        try
        {
            using var stream = new MemoryStream();
            KvHandoffSerialization.Export(source, config.NumLayers, stream);
            stream.Position = 0;
            KvHandoffSerialization.Import(stream, destination, config.NumLayers);
        }
        catch
        {
            destination.Dispose();
            throw;
        }

        source.Dispose();
        return destination;
    }
}
