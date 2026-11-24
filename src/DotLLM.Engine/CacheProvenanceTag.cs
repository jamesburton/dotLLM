using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;

namespace DotLLM.Engine;

/// <summary>
/// Identity of everything that decides whether cached KV / recurrent state is still valid for a forward pass: base weights,
/// tokenizer, numeric precision, the active weight modifiers and the external knowledge snapshot. Two tags are interchangeable
/// only when every field is equal (issue #1915: a base GGUF fingerprint alone misses overlays, adapters and LoRAs). The strings are
/// the contract; the engine does not interpret them.
/// </summary>
public sealed record CacheProvenanceTag
{
    /// <summary>Creates a tag. No field may be null.</summary>
    /// <param name="modelDigest">Content digest of the base weights.</param>
    /// <param name="tokenizerRevision">Tokenizer revision / digest.</param>
    /// <param name="precision">Quantization / KV / compute precision identifier.</param>
    /// <param name="activeModifiersDigest">Digest of the active overlays, residual modifiers, adapters and LoRAs; see <see cref="Engine.ActiveModifiersDigest.Compute"/>.</param>
    /// <param name="knowledgeSnapshotId">External knowledge / snapshot revision identifier.</param>
    public CacheProvenanceTag(string modelDigest, string tokenizerRevision, string precision,
        string activeModifiersDigest, string knowledgeSnapshotId)
    {
        ModelDigest = modelDigest ?? throw new ArgumentNullException(nameof(modelDigest));
        TokenizerRevision = tokenizerRevision ?? throw new ArgumentNullException(nameof(tokenizerRevision));
        Precision = precision ?? throw new ArgumentNullException(nameof(precision));
        ActiveModifiersDigest = activeModifiersDigest ?? throw new ArgumentNullException(nameof(activeModifiersDigest));
        KnowledgeSnapshotId = knowledgeSnapshotId ?? throw new ArgumentNullException(nameof(knowledgeSnapshotId));
    }

    /// <summary>Content digest of the base weights.</summary>
    public string ModelDigest
    {
        get => _ModelDigest;
        init => _ModelDigest = value ?? throw new ArgumentNullException(nameof(ModelDigest));
    }
    private readonly string _ModelDigest = null!;

    /// <summary>Tokenizer revision / digest.</summary>
    public string TokenizerRevision
    {
        get => _TokenizerRevision;
        init => _TokenizerRevision = value ?? throw new ArgumentNullException(nameof(TokenizerRevision));
    }
    private readonly string _TokenizerRevision = null!;

    /// <summary>Quantization / KV / compute precision identifier.</summary>
    public string Precision
    {
        get => _Precision;
        init => _Precision = value ?? throw new ArgumentNullException(nameof(Precision));
    }
    private readonly string _Precision = null!;

    /// <summary>Digest of the active weight overlays, residual modifiers, ternary adapters and per-request LoRAs.</summary>
    public string ActiveModifiersDigest
    {
        get => _ActiveModifiersDigest;
        init => _ActiveModifiersDigest = value ?? throw new ArgumentNullException(nameof(ActiveModifiersDigest));
    }
    private readonly string _ActiveModifiersDigest = null!;

    /// <summary>External knowledge / snapshot revision identifier.</summary>
    public string KnowledgeSnapshotId
    {
        get => _KnowledgeSnapshotId;
        init => _KnowledgeSnapshotId = value ?? throw new ArgumentNullException(nameof(KnowledgeSnapshotId));
    }
    private readonly string _KnowledgeSnapshotId = null!;

    /// <summary>Names of the fields whose values differ from <paramref name="other"/> (ordinal comparison), in declaration order; empty when equal.</summary>
    public IReadOnlyList<string> Diff(CacheProvenanceTag other)
    {
        ArgumentNullException.ThrowIfNull(other);
        var diff = new List<string>(5);
        if (!string.Equals(ModelDigest, other.ModelDigest, StringComparison.Ordinal)) diff.Add(nameof(ModelDigest));
        if (!string.Equals(TokenizerRevision, other.TokenizerRevision, StringComparison.Ordinal)) diff.Add(nameof(TokenizerRevision));
        if (!string.Equals(Precision, other.Precision, StringComparison.Ordinal)) diff.Add(nameof(Precision));
        if (!string.Equals(ActiveModifiersDigest, other.ActiveModifiersDigest, StringComparison.Ordinal)) diff.Add(nameof(ActiveModifiersDigest));
        if (!string.Equals(KnowledgeSnapshotId, other.KnowledgeSnapshotId, StringComparison.Ordinal)) diff.Add(nameof(KnowledgeSnapshotId));
        return diff;
    }
}

/// <summary>Builds the order-stable <see cref="CacheProvenanceTag.ActiveModifiersDigest"/> value.</summary>
public static class ActiveModifiersDigest
{
    /// <summary>
    /// Canonical digest of a set of active modifiers: entries are sorted ordinally by (kind, id, content digest), each component is
    /// length-prefixed (so field boundaries cannot collide), and the result is the lowercase SHA-256 hex of the stream. The same
    /// set in any order gives the same digest; an empty set gives a fixed digest.
    /// </summary>
    /// <param name="modifiers">(kind, id, contentDigest) per active modifier, e.g. ("lora", "my-adapter", sha256-of-weights).</param>
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
}
