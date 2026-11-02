using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;

namespace DotLLM.ChunkReuse.Store;

/// <summary>Counts of one store-backed assembly.</summary>
/// <param name="Hits">Chunk placements served without encoding: from the store, or from an entry encoded earlier in the same call
/// (a chunk that repeats in the context is encoded at most once).</param>
/// <param name="Misses">Chunk placements that had to be encoded (absent, damaged, foreign or stale in the store).</param>
/// <param name="EncodedTokens">Tokens run through the model to encode the misses.</param>
public readonly record struct AssemblyStats(int Hits, int Misses, long EncodedTokens);

/// <summary>Result of <see cref="StoredContextAssembler.Assemble"/>.</summary>
/// <param name="Context">The token-level context: the chunks in order, then the query.</param>
/// <param name="Stats">Hit, miss and encoded-token counts.</param>
/// <param name="Entries">The entry used for each chunk, in context order (what selective recompute needs).</param>
public sealed record StoredAssembly(ChunkedContext Context, AssemblyStats Stats, IReadOnlyList<ChunkKvEntry> Entries)
{
    /// <summary>Entries the store returned that did not belong to their key (wrong fingerprint, tokens or shape); evicted and replaced.</summary>
    public int Rejected { get; init; }

    /// <summary>Misses whose <see cref="IChunkStore.Put"/> threw <see cref="ChunkStoreBusyException"/>; their entries were used but not stored.</summary>
    public int StoreFailures { get; init; }

    /// <summary>Deconstructs into the <c>(Context, Stats)</c> pair.</summary>
    /// <param name="context">The context.</param>
    /// <param name="stats">The counts.</param>
    public void Deconstruct(out ChunkedContext context, out AssemblyStats stats)
    {
        context = Context;
        stats = Stats;
    }
}

/// <summary>
/// Assembles a context from per-chunk K/V held in an <see cref="IChunkStore"/>: each chunk is looked up by its
/// <see cref="ChunkKey"/>; a miss is encoded with <see cref="ChunkEncoder"/> and stored; the entries are then placed with
/// <see cref="ChunkAssembler"/>. The result equals assembling from fresh encodes, bit for bit, because the store round trip is
/// bit-exact and a hit is only accepted when it carries this assembler's fingerprint, the chunk's exact token ids and the model's
/// per-layer K/V shape.
/// </summary>
/// <remarks>
/// <para>
/// Store failures are misses, not errors: a damaged, truncated, foreign-key or old-format file reads as a miss (the store's own
/// checks), and a structurally valid entry that fails the checks above is evicted and replaced. A Put that hits
/// <see cref="ChunkStoreBusyException"/> (another reader holds a damaged file open) uses the freshly encoded entry without storing
/// it; a later call repairs the entry. Any other store exception propagates.
/// </para>
/// <para>
/// Only the CPU encoder exists, so entries are keyed with backend <see cref="Backend"/> and dtype <see cref="KvDtype.F32"/>.
/// The instance holds no mutable state, so concurrent calls are safe as long as each uses its own <see cref="IModel"/> (a CPU model's
/// forward is not re-entrant); several calls may share one store root.
/// </para>
/// </remarks>
public sealed class StoredContextAssembler
{
    /// <summary>Backend tag in the keys this assembler reads and writes (the encoder is the CPU oracle).</summary>
    public const string Backend = "cpu";

    private readonly ChunkEncoder _encoder;

    /// <summary>Creates an assembler for one model identity.</summary>
    /// <param name="modelFingerprint">The real model fingerprint (<see cref="ChunkFingerprint.Compute"/>) of the model passed to <see cref="Assemble"/>.</param>
    /// <param name="tokenizerId">Tokenizer identity, part of the key.</param>
    public StoredContextAssembler(string modelFingerprint, string tokenizerId)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelFingerprint);
        ArgumentNullException.ThrowIfNull(tokenizerId);
        ModelFingerprint = modelFingerprint;
        TokenizerId = tokenizerId;
        _encoder = new ChunkEncoder(modelFingerprint);
    }

    /// <summary>The model fingerprint every entry must carry.</summary>
    public string ModelFingerprint { get; }

    /// <summary>The tokenizer identity hashed into keys.</summary>
    public string TokenizerId { get; }

    /// <summary>The store key of a chunk with <paramref name="tokenIds"/> for a model with <paramref name="config"/>.</summary>
    /// <param name="config">Model configuration (RoPE parameters).</param>
    /// <param name="tokenIds">The chunk's token ids.</param>
    /// <returns>The key.</returns>
    public ChunkKey KeyOf(ModelConfig config, ReadOnlySpan<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(config);
        return ChunkKey.Create(ModelFingerprint, TokenizerId, RopeParameters.FromConfig(config), Backend, KvDtype.F32, tokenIds);
    }

    /// <summary>
    /// Resolves every chunk from <paramref name="store"/> (encoding and storing misses) and, when <paramref name="destination"/> is
    /// given, appends the chunks' K/V to it at running offsets (<see cref="ChunkAssembler.Assemble"/>).
    /// </summary>
    /// <param name="chunks">Chunks in context order; none empty.</param>
    /// <param name="model">The model with fingerprint <see cref="ModelFingerprint"/>; encodes misses (CPU <c>TransformerModel</c>).</param>
    /// <param name="store">The chunk store.</param>
    /// <param name="destination">Optional exact host-F32 cache to assemble into; left unchanged when no chunk fits.</param>
    /// <param name="query">Tokens that follow the context, recorded in <see cref="StoredAssembly.Context"/>.</param>
    /// <returns>The context, the counts and the entry used for each chunk.</returns>
    /// <exception cref="ArgumentException">There are no chunks, a chunk is null or empty, or the chunks do not fit the destination.</exception>
    /// <exception cref="NotSupportedException">The model's RoPE variant or the destination cache is not supported.</exception>
    public StoredAssembly Assemble(IReadOnlyList<Chunk> chunks, IModel model, IChunkStore store, IKvCache? destination = null,
        ReadOnlySpan<int> query = default)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(store);

        // Validates the chunks (non-empty list, no null or empty chunk) before the store is touched.
        var tokenIds = new int[chunks.Count][];
        for (int c = 0; c < chunks.Count; c++)
            tokenIds[c] = chunks[c]?.TokenIds ?? throw new ArgumentException($"Chunk {c} is null.", nameof(chunks));
        var context = new ChunkedContext(tokenIds, query);

        ModelConfig config = model.Config;
        var assembler = new ChunkAssembler(config); // refuses unsupported RoPE before any encode
        var geometry = KvGeometry.FromConfig(config, compactSlidingWindows: false);

        var resolved = new Dictionary<ChunkKey, ChunkKvEntry>();
        var entries = new ChunkKvEntry[chunks.Count];
        int hits = 0, misses = 0, rejected = 0, storeFailures = 0;
        long encodedTokens = 0;
        for (int c = 0; c < chunks.Count; c++)
        {
            int[] ids = tokenIds[c];
            ChunkKey key = KeyOf(config, ids);
            if (resolved.TryGetValue(key, out ChunkKvEntry? known))
            {
                entries[c] = known;
                hits++;
                continue;
            }

            if (store.TryGet(key, out ChunkKvEntry stored))
            {
                if (Belongs(stored, ids, geometry))
                {
                    resolved[key] = entries[c] = stored;
                    hits++;
                    continue;
                }
                // Well formed but not this key's content: drop it so Put (which keeps a valid entry) can replace it.
                rejected++;
                store.Evict(key);
            }

            ChunkKvEntry encoded = _encoder.Encode(model, ids);
            misses++;
            encodedTokens += ids.Length;
            try
            {
                store.Put(key, encoded);
            }
            catch (ChunkStoreBusyException)
            {
                storeFailures++; // a reader holds a damaged file; use the fresh entry, a later call stores it
            }
            resolved[key] = entries[c] = encoded;
        }

        if (destination is not null)
            assembler.Assemble(destination, entries, ModelFingerprint);

        return new StoredAssembly(context, new AssemblyStats(hits, misses, encodedTokens), entries)
        {
            Rejected = rejected,
            StoreFailures = storeFailures,
        };
    }

    // A hit must carry this model's fingerprint, exactly the chunk's tokens and one [n, stride(l)] K and V array per KV layer.
    private bool Belongs(ChunkKvEntry entry, int[] tokenIds, KvGeometry geometry)
    {
        if (!string.Equals(entry.ModelFingerprint, ModelFingerprint, StringComparison.Ordinal)) return false;
        if (entry.TokenIds is null || !entry.TokenIds.AsSpan().SequenceEqual(tokenIds)) return false;
        int layers = geometry.LayerCount;
        if (entry.KeysPerLayer is null || entry.ValuesPerLayer is null
            || entry.KeysPerLayer.Length != layers || entry.ValuesPerLayer.Length != layers) return false;
        for (int l = 0; l < layers; l++)
        {
            long expected = (long)tokenIds.Length * geometry.KvStrideOf(l);
            if (entry.KeysPerLayer[l]?.Length != expected || entry.ValuesPerLayer[l]?.Length != expected) return false;
        }
        return true;
    }
}
