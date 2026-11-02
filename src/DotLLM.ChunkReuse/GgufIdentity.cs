using System.Buffers.Binary;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using DotLLM.Models.Gguf;

namespace DotLLM.ChunkReuse;

/// <summary>
/// Content-based identities of a GGUF for chunk-store keys (<see cref="ChunkFingerprint.Compute(DotLLM.Core.Models.ModelConfig, string, string, string, string, int)"/> model id and tokenizer id), so
/// that a re-quantized, re-downloaded or edited file with the same name is a store miss rather than a wrong hit.
/// </summary>
public static class GgufIdentity
{
    /// <summary>Bytes hashed at the start and at the end of each tensor's data by the sampled <see cref="ModelId"/>.</summary>
    public const int TensorSampleBytes = 4096;

    /// <summary>
    /// Content identity of a GGUF file or split set. For every shard in order (<see cref="GgufFile.ShardPaths"/>) it hashes the shard
    /// index, the shard length, the shard header (everything before its tensor data: magic, version, all metadata, the tensor table
    /// with names, types, shapes and offsets), then either the first and last <see cref="TensorSampleBytes"/> of every tensor's data
    /// (default: a few MB of reads for a multi-GB model) or, with <paramref name="fullHash"/>, every byte. Independent of the file
    /// names.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>What the sampled id does not catch.</b> Two files with identical metadata (including <c>general.name</c>), an identical
    /// tensor table and the same length whose weights differ only strictly inside some tensors, more than 4 KiB from both ends,
    /// get the same id. Dense changes (a full fine-tune, a merge, a LoRA merged into attention/MLP, another quantization or imatrix)
    /// alter the head of every changed tensor and are caught. Row-sparse edits in the middle of a tensor are not: for example a
    /// fine-tune that trains only some rows of <c>token_embd</c> or <c>output</c> other than the last ones, a surgical edit of a few
    /// rows, a tool that rewrites only interior blocks, or an adversarial file. Edits to the last rows (e.g. trained reserved or
    /// special tokens at the end of the vocabulary) fall in the sampled tail and are caught. Use <paramref name="fullHash"/> (or a
    /// hub revision) when that matters, e.g. for a persistent store shared across model updates.
    /// </para>
    /// <para>
    /// Each tensor's extent runs from its offset to the next tensor's offset in the same shard (or the end of the shard), so the
    /// tail sample may include up to the alignment's worth of padding. Identify the file before any writable mapping (e.g. a
    /// Vulkan import alias) can modify it.
    /// </para>
    /// </remarks>
    /// <param name="gguf">The opened file or split set.</param>
    /// <param name="fullHash">Hash every byte of every shard instead of sampling the tensor data.</param>
    /// <returns><c>gguf-sha256:</c> (sampled) or <c>gguf-full-sha256:</c> (full) followed by 64 lowercase hex characters.</returns>
    public static string ModelId(GgufFile gguf, bool fullHash = false)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        IReadOnlyList<string> shards = gguf.ShardPaths;
        var shardIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < shards.Count; i++) shardIndex[shards[i]] = i;
        var offsetsByShard = shards.Select(_ => new List<long>()).ToArray();
        foreach (GgufTensorDescriptor t in gguf.Tensors)
        {
            (string path, long offset) = gguf.GetTensorFileLocation(t.Name);
            offsetsByShard[shardIndex[path]].Add(offset);
        }

        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        AppendInt(hash, shards.Count);
        for (int s = 0; s < shards.Count; s++)
        {
            // ReadWrite sharing: this process may already hold a writable alias of the file (Vulkan zero-copy import).
            using var fs = new FileStream(shards[s], FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            long length = fs.Length;
            AppendInt(hash, s);
            AppendLong(hash, length);
            if (fullHash)
            {
                AppendRange(hash, fs, 0, length);
                continue;
            }

            long[] offsets = offsetsByShard[s].Distinct().Order().ToArray();
            long dataStart = s == 0 ? gguf.DataSectionOffset : offsets.Length > 0 ? offsets[0] : length;
            if (dataStart <= 0 || dataStart > length)
                throw new InvalidDataException($"GGUF shard {shards[s]}: data offset {dataStart} outside the file ({length} B).");
            AppendRange(hash, fs, 0, dataStart);
            for (int i = 0; i < offsets.Length; i++)
            {
                long end = i + 1 < offsets.Length ? offsets[i + 1] : length;
                long extent = Math.Max(0, end - offsets[i]);
                long n = Math.Min(TensorSampleBytes, extent);
                AppendRange(hash, fs, offsets[i], n);       // head
                AppendRange(hash, fs, offsets[i] + extent - n, n); // tail
            }
        }
        return (fullHash ? "gguf-full-sha256:" : "gguf-sha256:") + Convert.ToHexStringLower(hash.GetHashAndReset());
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
            // Raw element bytes in host order (Buffer.BlockCopy is native-endian); GgufReader and FileChunkStore assume a
            // little-endian host, so ids are defined for little-endian hosts only.
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

    private static void AppendLong(IncrementalHash hash, long value)
    {
        Span<byte> b = stackalloc byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(b, value);
        hash.AppendData(b);
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
