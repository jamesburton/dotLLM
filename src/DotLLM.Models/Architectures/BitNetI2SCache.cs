using System;
using System.Collections.Generic;
using System.IO;

namespace DotLLM.Models.Architectures;

/// <summary>
/// On-disk cache for BitNet ternary <see cref="Core.Configuration.QuantizationType.I2_S"/>
/// weights. The HF BitNet safetensors loader quantizes every linear projection from bf16 to
/// I2_S at load time (per-tensor absmean → ternary → 2-bit pack); for a 2B model this online
/// quantization dominates load latency. Since the packing is deterministic in the source
/// weights, the packed bytes can be cached to disk keyed by (model identity, tensor name) and
/// reused verbatim on subsequent loads, eliminating the re-quantization cost.
/// <para>
/// Correctness is keyed entirely by the caller-supplied <c>modelKey</c> (which folds in the
/// source checkpoint identity <em>and</em> a quantizer-format version): a cache entry is only
/// reused for a byte-identical source tensor produced by the same packer. The cache is
/// otherwise a dumb content store.
/// </para>
/// </summary>
internal static class BitNetI2SCache
{
    /// <summary>
    /// Writes freshly-quantized I2_S <paramref name="payload"/> bytes for
    /// <paramref name="tensorName"/> into the cache under <paramref name="cacheDir"/> /
    /// <paramref name="modelKey"/>. The write is staged through a temp file and atomically
    /// moved into place so a crash mid-write never leaves a truncated entry.
    /// </summary>
    public static void Store(string cacheDir, string modelKey, string tensorName, ReadOnlySpan<byte> payload)
    {
        string path = EntryPath(cacheDir, modelKey, tensorName);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);

        string tmp = path + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            using (var fs = new FileStream(tmp, FileMode.Create, FileAccess.Write, FileShare.None))
                fs.Write(payload);
            File.Move(tmp, path, overwrite: true);
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    /// <summary>
    /// Attempts to load cached I2_S bytes for <paramref name="tensorName"/> into
    /// <paramref name="dest"/>. Returns <c>true</c> only when a valid entry filling exactly
    /// <paramref name="dest"/> exists.
    /// </summary>
    public static bool TryLoad(string cacheDir, string modelKey, string tensorName, Span<byte> dest)
    {
        string path = EntryPath(cacheDir, modelKey, tensorName);
        if (!File.Exists(path)) return false;

        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            // The entry must fill dest exactly — a shorter file can't (ReadExactly returns
            // false) and a longer file is a stale/incompatible entry we must reject rather
            // than read a truncated prefix from.
            if (fs.Length != dest.Length) return false;
            return ReadExactly(fs, dest);
        }
        catch (IOException)
        {
            return false;
        }
    }

    /// <summary>
    /// Derives a stable cache key identifying a specific source checkpoint quantized by a
    /// specific packer version. Folds in the raw <paramref name="configJsonUtf8"/>, each shard's
    /// name and byte length (cheap identity — no full-content hash of multi-GB weight files), and
    /// <paramref name="quantizerVersion"/> (bump to invalidate every prior cache when the packing
    /// semantics change). Callers should pass <paramref name="shards"/> in a stable order
    /// (e.g. sorted by name) so the key does not depend on filesystem enumeration order.
    /// </summary>
    public static string ComputeModelKey(
        ReadOnlySpan<byte> configJsonUtf8,
        IReadOnlyList<(string Name, long Length)> shards,
        int quantizerVersion)
    {
        using var hash = System.Security.Cryptography.IncrementalHash.CreateHash(
            System.Security.Cryptography.HashAlgorithmName.SHA256);

        Span<byte> num = stackalloc byte[8];
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(num, quantizerVersion);
        hash.AppendData(num[..4]);
        hash.AppendData(configJsonUtf8);
        foreach (var (name, length) in shards)
        {
            hash.AppendData(System.Text.Encoding.UTF8.GetBytes(name));
            System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(num, length);
            hash.AppendData(num);
        }

        Span<byte> digest = stackalloc byte[32];
        hash.GetCurrentHash(digest);
        return Convert.ToHexString(digest);
    }

    /// <summary>Quantizer packing version folded into the cache key. Bump when
    /// <see cref="Cpu.Kernels.BitNetQuantize.QuantizeToI2S"/>'s output layout changes so every
    /// prior on-disk entry is invalidated.</summary>
    public const int QuantizerVersion = 1;

    private static bool ReadExactly(Stream s, Span<byte> buffer)
    {
        int total = 0;
        while (total < buffer.Length)
        {
            int n = s.Read(buffer[total..]);
            if (n == 0) return false;
            total += n;
        }
        return true;
    }

    private static string EntryPath(string cacheDir, string modelKey, string tensorName)
        => Path.Combine(cacheDir, Sanitize(modelKey), Sanitize(tensorName) + ".i2s");

    private static string Sanitize(string name)
    {
        Span<char> buf = stackalloc char[name.Length];
        for (int i = 0; i < name.Length; i++)
        {
            char c = name[i];
            buf[i] = c is '/' or '\\' or ':' || Array.IndexOf(Path.GetInvalidFileNameChars(), c) >= 0
                ? '_'
                : c;
        }
        return new string(buf);
    }
}

/// <summary>
/// Per-load handle over a <see cref="BitNetI2SCache"/> directory bound to one model's cache key.
/// The BitNet safetensors loader consults it per linear projection: a hit returns the cached
/// packed I2_S bytes (skipping the bf16→f32 upcast and the absmean/round/pack quantization); a
/// miss quantizes and stores. Counters make the hit/miss/store behaviour observable for tests
/// and diagnostics. Not thread-safe — the BitNet layer load loop is sequential.
/// </summary>
internal sealed class BitNetI2SCacheContext
{
    private readonly string _cacheDir;
    private readonly string _modelKey;

    public BitNetI2SCacheContext(string cacheDir, string modelKey)
    {
        _cacheDir = cacheDir ?? throw new ArgumentNullException(nameof(cacheDir));
        _modelKey = modelKey ?? throw new ArgumentNullException(nameof(modelKey));
    }

    public int Hits { get; private set; }
    public int Misses { get; private set; }
    public int Stores { get; private set; }

    /// <summary>Fills <paramref name="dest"/> from cache if a valid entry exists, tallying a
    /// hit or miss.</summary>
    public bool TryLoad(string tensorName, Span<byte> dest)
    {
        if (BitNetI2SCache.TryLoad(_cacheDir, _modelKey, tensorName, dest))
        {
            Hits++;
            return true;
        }
        Misses++;
        return false;
    }

    /// <summary>Persists freshly-quantized <paramref name="payload"/> bytes, tallying a store.</summary>
    public void Store(string tensorName, ReadOnlySpan<byte> payload)
    {
        BitNetI2SCache.Store(_cacheDir, _modelKey, tensorName, payload);
        Stores++;
    }
}
