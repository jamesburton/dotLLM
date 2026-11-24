using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;

namespace DotLLM.Core.Identity;

/// <summary>
/// Order-stable identity of whatever changes the weights a forward pass runs on without changing the base file: tensor overlays,
/// ternary adapters, per-request LoRAs, residual modifiers (#1915). Shared by the chunk-cache identity and (when #1931 lands) the
/// <c>CacheProvenanceTag.ActiveModifiersDigest</c>: <see cref="Compute"/> is byte-for-byte the algorithm of
/// <c>DotLLM.Engine.ActiveModifiersDigest.Compute</c> (#1931), so the two give equal values for equal sets and one can delegate to the other.
/// </summary>
public static class ModifierSetDigest
{
    /// <summary>Modifier kind of a tensor overlay (<c>--tensor-overlay</c>).</summary>
    public const string KindOverlay = "overlay";

    /// <summary>Modifier kind of a ternary residual adapter (<c>--ternary-adapter</c>).</summary>
    public const string KindTernaryAdapter = "ternary-adapter";

    /// <summary>Modifier kind of a LoRA adapter.</summary>
    public const string KindLora = "lora";

    /// <summary>
    /// Canonical digest of a set of modifiers: entries sorted ordinally by (kind, id, content digest), each component
    /// length-prefixed so field boundaries cannot collide, lowercase SHA-256 hex of the stream. The same set in any order gives the
    /// same digest; the empty set gives a fixed value (callers that need "no modifiers" check for emptiness first).
    /// </summary>
    /// <param name="modifiers">
    /// (kind, id, contentDigest) per active modifier. <c>contentDigest</c> is what identifies the modifier. <c>id</c> is an optional
    /// disambiguator that is part of the digest: chunk identity (#1915) passes the EMPTY string for every kind, so that the same weights under
    /// another name or path are the same identity. A caller that wants a renamed adapter to count as a different modifier (e.g. a
    /// provenance tag that records which adapter by name) may pass the adapter's name as <c>id</c>, but then its digest is not comparable
    /// with chunk identity's. This is the canonical helper; other digests of the same sets should delegate to it with explicit semantics.
    /// </param>
    /// <returns>The digest.</returns>
    public static string Compute(IEnumerable<(string Kind, string Id, string ContentDigest)> modifiers)
    {
        ArgumentNullException.ThrowIfNull(modifiers);
        var sorted = modifiers.ToArray();
        Array.Sort(sorted, static (a, b) =>
        {
            int c = string.CompareOrdinal(a.Kind, b.Kind);
            if (c != 0) return c;
            c = string.CompareOrdinal(a.Id, b.Id);
            return c != 0 ? c : string.CompareOrdinal(a.ContentDigest, b.ContentDigest);
        });

        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var len = new byte[4];
        void Add(string s)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(s ?? throw new ArgumentNullException(nameof(modifiers), "A modifier component is null."));
            BinaryPrimitives.WriteInt32LittleEndian(len, bytes.Length);
            sha.AppendData(len);
            sha.AppendData(bytes);
        }
        BinaryPrimitives.WriteInt32LittleEndian(len, sorted.Length);
        sha.AppendData(len);
        foreach (var (kind, id, digest) in sorted) { Add(kind); Add(id); Add(digest); }
        return Convert.ToHexStringLower(sha.GetHashAndReset());
    }

    /// <summary>SHA-256 (lowercase hex) of a file's bytes, streamed. Independent of the file's name and location.</summary>
    /// <param name="path">The file.</param>
    /// <returns>The digest.</returns>
    public static string OfFile(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete, 1 << 16);
        return Convert.ToHexStringLower(SHA256.HashData(fs));
    }

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<ILoraAdapter, string> LoraDigests = new();

    /// <summary>
    /// Content digest of a loaded LoRA adapter: rank, alpha, and EVERY stored factor pair (<see cref="ILoraAdapter.EnumerateWeights"/>), in
    /// (layer, projection name, region) order: its coordinates, dims, dtypes and the bytes of both factors, each length-prefixed. Nothing
    /// is looked up by assumed name, so MoE expert tensors, <c>kv_b_proj</c> and any other target are covered. The adapter's name is NOT
    /// part of it (the same weights under another name are the same adapter). Cached per adapter instance (adapters are immutable
    /// once loaded); reloading under the same name is a new instance and is hashed again.
    /// </summary>
    /// <param name="adapter">The adapter.</param>
    /// <returns>The digest.</returns>
    /// <exception cref="NotSupportedException">The adapter cannot enumerate its tensors, so its content cannot be proven.</exception>
    public static unsafe string OfLora(ILoraAdapter adapter)
    {
        ArgumentNullException.ThrowIfNull(adapter);
        if (LoraDigests.TryGetValue(adapter, out string? cached))
            return cached;

        var entries = adapter.EnumerateWeights()
            ?? throw new NotSupportedException($"{adapter.GetType().Name} cannot enumerate its tensors; its content cannot be hashed.");
        var sorted = entries.OrderBy(e => e.Layer).ThenBy(e => e.Proj, StringComparer.Ordinal).ThenBy(e => (int)e.Region).ToArray();
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        byte[] w = new byte[8];
        void Int(long v) { BinaryPrimitives.WriteInt64LittleEndian(w, v); sha.AppendData(w); }
        void Bytes(nint p, long n)
        {
            Int(n);
            const int chunk = 1 << 20;
            for (long off = 0; off < n; off += chunk)
                sha.AppendData(new ReadOnlySpan<byte>((byte*)p + off, (int)Math.Min(chunk, n - off)));
        }
        Int(adapter.Rank);
        Int(BitConverter.SingleToInt32Bits(adapter.Alpha));
        Int(sorted.Length);
        foreach (var (layer, proj, region, lw) in sorted)
        {
            byte[] name = Encoding.UTF8.GetBytes(proj);
            Int(layer); Int(name.Length); sha.AppendData(name); Int((int)region);
            Int(lw.InputDim); Int(lw.OutputDim);
            Int((int)lw.WeightDType); Int((int)lw.ResolvedAWeightDType);
            long r = adapter.Rank;
            Bytes(lw.AHandle, ElementBytes(lw.ResolvedAWeightDType, (long)lw.OutputDim * r, lw.OutputDim, r));
            Bytes(lw.BHandle, ElementBytes(lw.WeightDType, r * lw.InputDim, r, lw.InputDim));
        }
        string digest = Convert.ToHexStringLower(sha.GetHashAndReset());
        LoraDigests.TryAdd(adapter, digest);
        return digest;
    }

    private static long ElementBytes(LoraWeightDType dtype, long elements, long rows, long cols) => dtype switch
    {
        LoraWeightDType.F32 => elements * 4,
        LoraWeightDType.F16 or LoraWeightDType.BF16 => elements * 2,
        // Q8_0: B is [rank, inputDim], rows of (inputDim/32) blocks.
        LoraWeightDType.Q8_0 => rows * LoraAdapter.Q8_0ByteSize(cols),
        _ => throw new NotSupportedException($"LoRA dtype {dtype} has no known byte size."),
    };
}
