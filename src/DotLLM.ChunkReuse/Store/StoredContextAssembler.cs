using DotLLM.ChunkReuse.Backends;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;

namespace DotLLM.ChunkReuse.Store;

/// <summary>Counts of one store-backed assembly. <c>Hits + Misses + Repeats</c> is the number of chunks.</summary>
/// <param name="Hits">Chunks served from the store (valid entry found under the chunk's key).</param>
/// <param name="Misses">Chunks that had to be encoded (absent, damaged, foreign, stale or rejected in the store).</param>
/// <param name="EncodedTokens">Tokens run through the model to encode the misses.</param>
/// <param name="Repeats">Chunks whose key already occurred earlier in the same call; they reuse that entry (neither a store read
/// nor an encode).</param>
public readonly record struct AssemblyStats(int Hits, int Misses, long EncodedTokens, int Repeats);

/// <summary>Result of <see cref="StoredContextAssembler.Assemble"/>.</summary>
/// <param name="Context">The token-level context: the chunks in order, then the query. Its <see cref="ChunkedContext.ChunkStarts"/>
/// are relative to the context; in the destination cache chunk <c>c</c> starts at row <c>DestinationOffset + ChunkStarts[c]</c>.</param>
/// <param name="Stats">Hit, miss, repeat and encoded-token counts.</param>
/// <param name="Entries">The entry used for each chunk, in context order (what selective recompute needs).</param>
public sealed record StoredAssembly(ChunkedContext Context, AssemblyStats Stats, IReadOnlyList<ChunkKvEntry> Entries)
{
    /// <summary>Entries the store returned that did not belong to their key (wrong fingerprint, tokens or shape); evicted and replaced
    /// when the store allowed it.</summary>
    public int Rejected { get; init; }

    /// <summary>Misses whose entry could not be written: <see cref="IChunkStore.Put"/> threw <see cref="ChunkStoreBusyException"/>,
    /// or evicting a rejected entry failed with an I/O or access error. Their fresh entries were used but not stored.</summary>
    public int StoreFailures { get; init; }

    /// <summary>The destination's length before the call (0 without a destination): the row the first chunk was written at.</summary>
    public int DestinationOffset { get; init; }

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
/// <b>Identity is the caller's responsibility.</b> The key and the hit check use the fingerprint given here, not one read from the
/// model: with <see cref="StoredContextAssembler(string, string)"/> nothing ties the fingerprint to the <see cref="IModel"/> passed
/// to <see cref="Assemble"/>, and a mismatched pair writes one model's K/V under another model's keys, which every later reader of
/// those keys accepts (store poisoning). <see cref="ForModel(ModelConfig, string, string, string)"/> keeps the fingerprint inputs and refuses a model whose configuration
/// (shape, RoPE, NoPE layers, KV layout) does not reproduce the fingerprint; it cannot see the weights, so the model id must itself
/// be content-based (e.g. <see cref="GgufIdentity.ModelId"/>, a weights hash or a hub revision), never a file name.
/// </para>
/// <para>
/// Entries are isolated encodes: each chunk alone at positions <c>0..n-1</c>. They are keyed with
/// <see cref="ChunkPositioning.Isolated"/>, so K/V computed with preceding context (a recompute output) stored under a
/// <see cref="ChunkPositioning.Contextual"/> key never answer here.
/// </para>
/// <para>
/// Store failures are misses, not errors: a damaged, truncated, foreign-key or old-format file reads as a miss (the store's own
/// checks), and a structurally valid entry that fails the checks above is evicted and replaced. When the store cannot write (Put
/// throws <see cref="ChunkStoreBusyException"/>, or evicting a rejected entry throws <see cref="IOException"/> or
/// <see cref="UnauthorizedAccessException"/>) the freshly encoded entry is used without storing it; a later call repairs the entry.
/// A store-lock timeout (<see cref="ChunkStoreLockTimeoutException"/>, e.g. from <see cref="LockingChunkStore"/>) is not a busy
/// entry: it propagates, so the caller stops at once (CLI <c>store_busy</c>, server <c>chunk_store_unavailable</c>) instead of
/// encoding every remaining chunk and waiting the timeout for each. Any other store exception propagates.
/// </para>
/// <para>
/// Entries are keyed with the backend's tag (<see cref="Backend"/>; <see cref="IChunkBackend.Tag"/>) and dtype <see cref="KvDtype.F32"/>,
/// encoded through that backend and stamped with its tag. An entry whose <see cref="ChunkKvEntry.Backend"/> differs from
/// <see cref="Backend"/> is rejected and replaced like any other wrong entry, so CPU and Vulkan K/V never mix (by key, and by the
/// <c>.ckv</c> header tag). The constructor and <see cref="ForModel(ModelConfig, string, string, string)"/> without a backend mean the
/// CPU backend (<see cref="CpuChunkBackend"/>), with the same fingerprint and keys as before the backend seam existed.
/// Each call works in one store partition (<see cref="ChunkNamespace"/>; the <c>namespace</c> argument of <see cref="Assemble"/>
/// and <see cref="KeyOf"/>), hashed into every key: one namespace's entries are invisible to another's lookups.
/// The instance holds no mutable state, so concurrent calls are safe as long as each uses its own <see cref="IModel"/> (a CPU model's
/// forward is not re-entrant); several calls may share one store root.
/// </para>
/// </remarks>
public sealed class StoredContextAssembler
{
    private readonly ChunkEncoder _encoder;
    private readonly (string ModelId, string Quantization)? _identity;
    private readonly IChunkBackend? _backend; // null: the CPU backend of each model's config

    /// <summary>Creates a CPU assembler for one model identity, given as a precomputed fingerprint (see the class remarks on identity).</summary>
    /// <param name="modelFingerprint">The real model fingerprint (<see cref="ChunkFingerprint.Compute(DotLLM.Core.Models.ModelConfig, string, string, string, string, int)"/>) of every model passed to
    /// <see cref="Assemble"/>; not checked against the model.</param>
    /// <param name="tokenizerId">Tokenizer identity, part of the key (content-based, e.g. <see cref="GgufIdentity.TokenizerId"/>).</param>
    public StoredContextAssembler(string modelFingerprint, string tokenizerId)
        : this(modelFingerprint, tokenizerId, null, null)
    {
    }

    /// <summary>As <see cref="StoredContextAssembler(string, string)"/>, encoding and placing through <paramref name="backend"/>.</summary>
    /// <param name="modelFingerprint">The real model fingerprint of every model passed to <see cref="Assemble"/>, computed for this
    /// backend (<see cref="ChunkFingerprint.Compute(DotLLM.Core.Models.ModelConfig, string, string, string, string, int)"/> with its tag); not checked.</param>
    /// <param name="tokenizerId">Tokenizer identity, part of the key.</param>
    /// <param name="backend">The chunk backend; its <see cref="IChunkBackend.Tag"/> is in every key and every entry.</param>
    public StoredContextAssembler(string modelFingerprint, string tokenizerId, IChunkBackend backend)
        : this(modelFingerprint, tokenizerId, null, backend ?? throw new ArgumentNullException(nameof(backend)))
    {
    }

    private StoredContextAssembler(string modelFingerprint, string tokenizerId, (string, string)? identity, IChunkBackend? backend)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelFingerprint);
        ArgumentNullException.ThrowIfNull(tokenizerId);
        if (backend is not null && !FileChunkStore.IsWellFormedBackendTag(backend.Tag))
            throw new ArgumentException($"Backend tag '{backend.Tag}' cannot be stored.", nameof(backend));
        ModelFingerprint = modelFingerprint;
        TokenizerId = tokenizerId;
        _identity = identity;
        _backend = backend;
        _encoder = new ChunkEncoder(modelFingerprint);
    }

    /// <summary>
    /// Creates a CPU assembler from the fingerprint inputs. <see cref="Assemble"/> then recomputes the fingerprint from each model's
    /// configuration and refuses a model that does not reproduce it, before any store read or encode.
    /// </summary>
    /// <param name="config">The model configuration.</param>
    /// <param name="modelId">Content-based model identity (<see cref="GgufIdentity.ModelId"/>, a weights hash or a hub revision).</param>
    /// <param name="quantization">Weight quantization label.</param>
    /// <param name="tokenizerId">Content-based tokenizer identity (<see cref="GgufIdentity.TokenizerId"/>).</param>
    /// <returns>The assembler.</returns>
    public static StoredContextAssembler ForModel(ModelConfig config, string modelId, string quantization, string tokenizerId)
        => Create(config, modelId, quantization, tokenizerId, null);

    /// <summary>As <see cref="ForModel(ModelConfig, string, string, string)"/>, on <paramref name="backend"/>: the fingerprint is computed
    /// with its tag (<see cref="ChunkFingerprint.Compute(DotLLM.Core.Models.ModelConfig, string, string, string, string, int)"/>).</summary>
    /// <param name="config">The model configuration.</param>
    /// <param name="modelId">Content-based model identity.</param>
    /// <param name="quantization">Weight quantization label.</param>
    /// <param name="tokenizerId">Content-based tokenizer identity.</param>
    /// <param name="backend">The chunk backend (for Vulkan: on the device of the models passed to <see cref="Assemble"/>).</param>
    /// <returns>The assembler.</returns>
    public static StoredContextAssembler ForModel(ModelConfig config, string modelId, string quantization, string tokenizerId, IChunkBackend backend)
        => Create(config, modelId, quantization, tokenizerId, backend ?? throw new ArgumentNullException(nameof(backend)));

    private static StoredContextAssembler Create(ModelConfig config, string modelId, string quantization, string tokenizerId, IChunkBackend? backend)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentException.ThrowIfNullOrEmpty(modelId);
        ArgumentNullException.ThrowIfNull(quantization);
        ArgumentNullException.ThrowIfNull(tokenizerId);
        string tag = backend?.Tag ?? ChunkBackendTags.Cpu;
        return new StoredContextAssembler(ChunkFingerprint.Compute(config, modelId, quantization, tokenizerId, tag), tokenizerId,
            (modelId, quantization), backend);
    }

    /// <summary>Backend tag in the keys this assembler reads and writes and on the entries it encodes (<c>cpu</c> by default).</summary>
    public string Backend => _backend?.Tag ?? ChunkBackendTags.Cpu;

    /// <summary>The model fingerprint every entry must carry.</summary>
    public string ModelFingerprint { get; }

    /// <summary>The tokenizer identity hashed into keys.</summary>
    public string TokenizerId { get; }

    /// <summary>The store key of a chunk with <paramref name="tokenIds"/> for a model with <paramref name="config"/>.</summary>
    /// <param name="config">Model configuration (RoPE parameters).</param>
    /// <param name="tokenIds">The chunk's token ids.</param>
    /// <param name="namespace">Store partition (<see cref="ChunkNamespace"/>); <see cref="ChunkNamespace.Default"/> by default.</param>
    /// <returns>The key (<see cref="ChunkPositioning.Isolated"/>).</returns>
    public ChunkKey KeyOf(ModelConfig config, ReadOnlySpan<int> tokenIds, string @namespace = ChunkNamespace.Default)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(@namespace);
        return ChunkKey.Create(ModelFingerprint, TokenizerId, RopeParameters.FromConfig(config), Backend, KvDtype.F32, tokenIds,
            @namespace: @namespace);
    }

    /// <summary>
    /// Resolves every chunk from <paramref name="store"/> (encoding and storing misses) and, when <paramref name="destination"/> is
    /// given, appends the chunks' K/V to it at running offsets from its current length (<see cref="ChunkAssembler.Assemble(DotLLM.Core.Attention.IKvCache, IReadOnlyList{ChunkKvEntry}, string)"/>).
    /// The chunks, the model identity and the destination (layer count, strides, capacity, then the backend's
    /// <see cref="IChunkBackend.RequireCache"/>) are validated before any store read or encode, and the destination is left unchanged
    /// whenever the call throws.
    /// </summary>
    /// <param name="chunks">Chunks in context order; none empty.</param>
    /// <param name="model">The model with fingerprint <see cref="ModelFingerprint"/>; encodes misses on the backend (CPU:
    /// <c>TransformerModel</c>; Vulkan: a Vulkan model on the backend's device).</param>
    /// <param name="store">The chunk store.</param>
    /// <param name="destination">Optional cache of the backend to append to (CPU: exact host-F32 <c>SimpleKvCache</c>).</param>
    /// <param name="query">Tokens that follow the context, recorded in <see cref="StoredAssembly.Context"/>.</param>
    /// <param name="namespace">Store partition every chunk is looked up and stored in (<see cref="ChunkNamespace"/>; the
    /// <see cref="ChunkNamespace.Default"/> partition by default). Entries of another namespace are never hits, rejects or evictions.</param>
    /// <returns>The context, the counts and the entry used for each chunk.</returns>
    /// <exception cref="ArgumentException">There are no chunks, a chunk is null or empty, the model does not match a
    /// <see cref="ForModel(ModelConfig, string, string, string)"/> identity, or the chunks do not fit the destination.</exception>
    /// <exception cref="NotSupportedException">The model's RoPE variant or the destination cache is not supported.</exception>
    public StoredAssembly Assemble(IReadOnlyList<Chunk> chunks, IModel model, IChunkStore store, IKvCache? destination = null,
        ReadOnlySpan<int> query = default, string @namespace = ChunkNamespace.Default)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(store);
        ChunkNamespace.Validate(@namespace, nameof(@namespace));

        // Validates the chunks (non-empty list, no null or empty chunk) before the store is touched.
        var tokenIds = new int[chunks.Count][];
        for (int c = 0; c < chunks.Count; c++)
            tokenIds[c] = chunks[c]?.TokenIds ?? throw new ArgumentException($"Chunk {c} is null.", nameof(chunks));
        var context = new ChunkedContext(tokenIds, query);

        ModelConfig config = model.Config;
        if (_identity is { } id)
        {
            string actual = ChunkFingerprint.Compute(config, id.ModelId, id.Quantization, TokenizerId, Backend);
            if (!string.Equals(actual, ModelFingerprint, StringComparison.Ordinal))
                throw new ArgumentException(
                    $"The model's configuration gives fingerprint {actual}, not this assembler's {ModelFingerprint} (another shape, RoPE or KV layout).",
                    nameof(model));
        }
        IChunkBackend backend = _backend ?? new CpuChunkBackend(config);
        var assembler = new ChunkAssembler(config, backend); // refuses unsupported RoPE (or a backend bound to other RoPE) before any encode
        var geometry = KvGeometry.FromConfig(config, compactSlidingWindows: false);
        int offset = 0;
        if (destination is not null)
        {
            ValidateDestination(destination, backend, config, geometry, context.Length);
            offset = destination.CurrentLength;
        }

        var resolved = new Dictionary<ChunkKey, ChunkKvEntry>();
        var entries = new ChunkKvEntry[chunks.Count];
        int hits = 0, misses = 0, repeats = 0, rejected = 0, storeFailures = 0;
        long encodedTokens = 0;
        for (int c = 0; c < chunks.Count; c++)
        {
            int[] ids = tokenIds[c];
            ChunkKey key = KeyOf(config, ids, @namespace);
            if (resolved.TryGetValue(key, out ChunkKvEntry? known))
            {
                entries[c] = known;
                repeats++;
                continue;
            }

            bool writable = true;
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
                try
                {
                    store.Evict(key);
                }
                catch (Exception e) when ((e is IOException or UnauthorizedAccessException) && e is not ChunkStoreLockTimeoutException)
                {
                    writable = false; // Put would keep the wrong entry; use the fresh one, a later call retries
                }
            }

            ChunkKvEntry encoded = _encoder.Encode(model, backend, ids);
            misses++;
            encodedTokens += ids.Length;
            if (writable)
            {
                try
                {
                    store.Put(key, encoded);
                }
                catch (ChunkStoreBusyException e) when (e is not ChunkStoreLockTimeoutException)
                {
                    writable = false; // a reader holds a damaged file; use the fresh entry, a later call stores it
                }
            }
            if (!writable) storeFailures++;
            resolved[key] = entries[c] = encoded;
        }

        if (destination is not null)
            assembler.Assemble(destination, entries, ModelFingerprint);

        return new StoredAssembly(context, new AssemblyStats(hits, misses, encodedTokens, repeats), entries)
        {
            Rejected = rejected,
            StoreFailures = storeFailures,
            DestinationOffset = offset,
        };
    }

    // The checks ChunkAssembler makes before writing, done before any encode so a bad destination costs nothing.
    private static void ValidateDestination(IKvCache destination, IChunkBackend backend, ModelConfig config, KvGeometry geometry, int tokens)
    {
        if (destination is not IPerLayerKvCache perLayer)
            throw new NotSupportedException(
                $"Chunk assembly needs a KV-cache with per-layer strides; {destination.GetType().Name} has none.");
        if (perLayer.LayerCount != config.NumLayers || perLayer.LayerCount != geometry.LayerCount)
            throw new NotSupportedException(
                $"The destination holds {perLayer.LayerCount} KV layers for a {config.NumLayers}-layer model; not supported.");
        for (int l = 0; l < geometry.LayerCount; l++)
        {
            if (perLayer.KvStrideOf(l) != geometry.KvStrideOf(l))
                throw new ArgumentException(
                    $"Destination layer {l} KV stride {perLayer.KvStrideOf(l)} differs from the model's {geometry.KvStrideOf(l)}.", nameof(destination));
        }
        if ((long)destination.CurrentLength + tokens > destination.MaxLength)
            throw new ArgumentException(
                $"The chunks need rows {destination.CurrentLength}..{destination.CurrentLength + tokens - 1} but the destination holds {destination.MaxLength}.",
                nameof(destination));
        backend.RequireCache(destination); // CPU: exact host-F32 SimpleKvCache; Vulkan: F32 VulkanKvCache
    }

    // A hit must carry this assembler's backend tag and this model's fingerprint, exactly the chunk's tokens and one [n, stride(l)] K and V array per KV layer.
    private bool Belongs(ChunkKvEntry entry, int[] tokenIds, KvGeometry geometry)
    {
        if (!string.Equals(entry.Backend, Backend, StringComparison.Ordinal)) return false;
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
