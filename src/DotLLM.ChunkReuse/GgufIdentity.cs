using System.Buffers.Binary;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using DotLLM.Models.Gguf;

namespace DotLLM.ChunkReuse;

/// <summary>
/// Content-based identities of a GGUF for chunk-store keys (<see cref="ChunkFingerprint.Compute"/> model id and tokenizer id), so
/// that a re-quantized, re-downloaded or edited file with the same name is a store miss rather than a wrong hit.
/// </summary>
public static class GgufIdentity
{
    /// <summary>Bytes of each tensor's data that <see cref="ModelId"/> hashes.</summary>
    public const int TensorSampleBytes = 4096;

    /// <summary>
    /// SHA-256 over the GGUF header (everything before the tensor data: magic, version, all metadata, the tensor table with names,
    /// types, shapes and offsets) and the first <see cref="TensorSampleBytes"/> bytes of every tensor's data. Cheap (a few MB of
    /// reads for a multi-GB file) and independent of the file name.
    /// </summary>
    /// <remarks>
    /// Not a full-weight hash: two files with identical metadata and tensor table that differ only after the first 4 KiB of every
    /// changed tensor get the same id. Weights from another quantization, imatrix, fine-tune or conversion change the metadata, the
    /// tensor types or the leading blocks in practice; when that is not enough, pass a full weights SHA-256 or a hub revision as the
    /// model id instead. Split (multi-shard) GGUFs are refused.
    /// </remarks>
    /// <param name="gguf">The opened file (for the data-section offset and the tensor table).</param>
    /// <param name="path">Path of the same file.</param>
    /// <returns><c>gguf-sha256:</c> followed by 64 lowercase hex characters.</returns>
    /// <exception cref="NotSupportedException">The GGUF is a split set.</exception>
    public static string ModelId(GgufFile gguf, string path)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (gguf.Metadata.TryGetValue("split.count", out GgufMetadataValue split) && Convert.ToInt64(split.Value, CultureInfo.InvariantCulture) > 1)
            throw new NotSupportedException("GgufIdentity.ModelId does not support split GGUF sets; pass a content hash of all shards instead.");

        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
        long dataStart = gguf.DataSectionOffset;
        long fileLength = fs.Length;
        if (dataStart <= 0 || dataStart > fileLength) throw new InvalidDataException($"GGUF data offset {dataStart} outside the file ({fileLength} B).");
        AppendRange(hash, fs, 0, dataStart);

        // Tensor data samples in offset order; each sample stops at the next tensor's start (or the end of the file).
        long[] offsets = gguf.Tensors.Select(t => dataStart + (long)t.DataOffset).Distinct().Order().ToArray();
        for (int i = 0; i < offsets.Length; i++)
        {
            long end = i + 1 < offsets.Length ? offsets[i + 1] : fileLength;
            AppendRange(hash, fs, offsets[i], Math.Min(TensorSampleBytes, Math.Max(0, end - offsets[i])));
        }
        return "gguf-sha256:" + Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    /// <summary>
    /// SHA-256 over the <c>tokenizer.ggml.*</c> metadata (model, pre-tokenizer, vocabulary, token types, scores, merges, special ids,
    /// add-BOS/EOS flags) in key order, length-prefixed. These decide how text maps to token ids; the chat template and other keys
    /// do not take part.
    /// </summary>
    /// <param name="metadata">The GGUF metadata.</param>
    /// <returns><c>gguf-tokenizer-sha256:</c> followed by 64 lowercase hex characters.</returns>
    public static string TokenizerId(GgufMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (string key in metadata.Keys.Where(k => k.StartsWith("tokenizer.ggml.", StringComparison.Ordinal)).Order(StringComparer.Ordinal))
        {
            GgufMetadataValue v = metadata.TryGetValue(key, out GgufMetadataValue value) ? value : default;
            AppendString(hash, key);
            AppendInt(hash, (int)v.Type);
            AppendValue(hash, v.Value);
        }
        return "gguf-tokenizer-sha256:" + Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    private static void AppendValue(IncrementalHash hash, object? value)
    {
        switch (value)
        {
            case null:
                AppendString(hash, "null");
                break;
            case string s:
                AppendString(hash, "s");
                AppendString(hash, s);
                break;
            case string[] a:
                AppendString(hash, "s[]");
                AppendInt(hash, a.Length);
                foreach (string s in a) AppendString(hash, s ?? "");
                break;
            case Array a when a.GetType().GetElementType() is { IsPrimitive: true } elem:
                AppendString(hash, elem.FullName + "[]");
                byte[] bytes = new byte[Buffer.ByteLength(a)];
                Buffer.BlockCopy(a, 0, bytes, 0, bytes.Length);
                AppendInt(hash, bytes.Length);
                hash.AppendData(bytes);
                break;
            case IConvertible c when value.GetType().IsPrimitive:
                AppendString(hash, value.GetType().FullName!);
                AppendString(hash, c.ToString(CultureInfo.InvariantCulture));
                break;
            default:
                throw new NotSupportedException($"Unexpected GGUF tokenizer metadata value type {value.GetType()}.");
        }
    }

    private static void AppendRange(IncrementalHash hash, FileStream fs, long offset, long count)
    {
        byte[] buffer = new byte[(int)Math.Min(count, 1 << 20)];
        fs.Position = offset;
        long left = count;
        while (left > 0)
        {
            int n = fs.Read(buffer, 0, (int)Math.Min(buffer.Length, left));
            if (n <= 0) throw new EndOfStreamException($"GGUF ends before byte {offset + count}.");
            hash.AppendData(buffer, 0, n);
            left -= n;
        }
    }

    private static void AppendInt(IncrementalHash hash, int value)
    {
        Span<byte> b = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(b, value);
        hash.AppendData(b);
    }

    private static void AppendString(IncrementalHash hash, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        AppendInt(hash, bytes.Length);
        hash.AppendData(bytes);
    }
}
