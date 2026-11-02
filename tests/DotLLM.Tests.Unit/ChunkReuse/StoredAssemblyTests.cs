using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Backends;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Vulkan;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Store-backed assembly (<see cref="StoredContextAssembler"/>) on the synthetic dense model (<see cref="SyntheticDenseGguf"/>,
/// F32 weights, CPU, single-threaded). The oracle is the store-less path: <see cref="ChunkEncoder.Encode"/> per chunk, then
/// <see cref="ChunkAssembler.Assemble"/>. Store-backed assembly must equal it bit for bit, cold or warm; a damaged, foreign or
/// stale entry is a miss that is re-encoded, never an error and never a wrong hit.
/// </summary>
public sealed class StoredAssemblyTests : IDisposable
{
    private const string Tokenizer = "tok-tiny";

    // Distinct token ids (vocab 48) so chunks are distinguishable; ChunkA appears twice in Context4.
    private static readonly int[] TokA = [3, 17, 5, 29, 11, 40];
    private static readonly int[] TokB = [8, 22, 1, 35];
    private static readonly int[] TokC = [12, 7, 44, 2, 19];
    private static readonly int[] TokD = [30, 6, 21];
    private static readonly int[] TokE = [10, 25, 33, 4, 16, 27]; // same length as TokA: a slot mix-up keeps shapes valid
    private static readonly int[] Query = [9, 14, 38];

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-storedasm-" + Guid.NewGuid().ToString("N"));
    private readonly string _root;
    private readonly ITestOutputHelper _output;

    public StoredAssemblyTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_scratch);
        _root = Path.Combine(_scratch, "store");
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { /* best effort: a still-mapped file on Windows */ }
        catch (UnauthorizedAccessException) { }
    }

    private static Chunk Ch(string file, int[] tokens) => new(file, 1, 1, (int[])tokens.Clone());

    private static IReadOnlyList<Chunk> Context3 => [Ch("a.cs", TokA), Ch("b.cs", TokB), Ch("c.cs", TokC)];

    // A repeats (the same text in two files): one entry, two placements.
    private static IReadOnlyList<Chunk> Context4 => [Ch("a.cs", TokA), Ch("b.cs", TokB), Ch("c.cs", TokC), Ch("copy-of-a.cs", TokA)];

    // ------------------------------------------------------------------------------------------------ parity

    [Fact]
    public void Assemble_FromStore_EqualsFreshEncode_BitExact()
    {
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        string fp = tiny.Fingerprint;
        // A and E have the same length and sit side by side, so placing an entry in the wrong slot passes every shape check
        // (mutant M23 in the Task 3 report).
        IReadOnlyList<Chunk> chunks = [Ch("a.cs", TokA), Ch("e.cs", TokE), Ch("b.cs", TokB), Ch("c.cs", TokC), Ch("copy-of-a.cs", TokA)];

        // Oracle: fresh encodes, no store.
        var encoder = new ChunkEncoder(fp);
        var fresh = chunks.Select(c => encoder.Encode(tiny.Model, c.TokenIds)).ToList();
        using SimpleKvCache freshCache = NewCache(tiny);
        new ChunkAssembler(tiny.Config).Assemble(freshCache, fresh, fp);

        var assembler = new StoredContextAssembler(fp, Tokenizer);

        // Cold: every distinct chunk is encoded and stored.
        using SimpleKvCache coldCache = NewCache(tiny);
        StoredAssembly cold = assembler.Assemble(chunks, tiny.Model, new FileChunkStore(_root), coldCache, Query);

        // Warm: a new store instance on the same root, so every entry comes from disk.
        using SimpleKvCache warmCache = NewCache(tiny);
        StoredAssembly warm = assembler.Assemble(chunks, tiny.Model, new FileChunkStore(_root), warmCache, Query);
        Assert.Equal(new AssemblyStats(Hits: 4, Misses: 0, EncodedTokens: 0, Repeats: 1), warm.Stats); // the second A is a repeat
        Assert.Equal(new AssemblyStats(Hits: 0, Misses: 4, EncodedTokens: TokA.Length + TokE.Length + TokB.Length + TokC.Length, Repeats: 1), cold.Stats);

        int length = chunks.Sum(c => c.TokenIds.Length);
        Assert.Equal(length, freshCache.CurrentLength);
        Assert.Equal(length, coldCache.CurrentLength);
        Assert.Equal(length, warmCache.CurrentLength);
        AssertCachesBitExact(tiny, freshCache, coldCache, length);
        AssertCachesBitExact(tiny, freshCache, warmCache, length);
        for (int i = 0; i < chunks.Count; i++)
        {
            AssertBitExact(fresh[i], cold.Entries[i]);
            AssertBitExact(fresh[i], warm.Entries[i]);
        }

        // Downstream: the query logits on the warm cache equal those on the fresh cache, bit for bit.
        float[] freshLogits = LastRow(tiny.Model, Query, length, freshCache);
        float[] warmLogits = LastRow(tiny.Model, Query, length, warmCache);
        Assert.True(MemoryMarshal.AsBytes(freshLogits.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(warmLogits.AsSpan())),
            "query logits on the warm store-assembled cache differ from the fresh-encode cache");

        // Fixture guard: the comparison discriminates. Swapping two chunks' K/V (same total length) changes the cache.
        using SimpleKvCache swapped = NewCache(tiny);
        new ChunkAssembler(tiny.Config).Assemble(swapped, [fresh[1], fresh[0], fresh[2], fresh[3], fresh[4]], fp);
        Assert.False(CachesBitEqual(tiny, freshCache, swapped, length), "fixture cannot tell chunk order apart");

        // The context is the token-level view of the same chunks, followed by the query.
        Assert.Equal(chunks.SelectMany(c => c.TokenIds).ToArray(), warm.Context.TokenIds);
        Assert.Equal(new[] { 0, 6, 12, 16, 21 }, warm.Context.ChunkStarts);
        Assert.Equal(Query, warm.Context.QueryTokenIds);
        _output.WriteLine($"cold {cold.Stats}; warm {warm.Stats}; {length} context tokens, bit-exact");
    }

    [Fact]
    public void Assemble_WithoutDestination_ReturnsEntriesOnly()
    {
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var assembler = new StoredContextAssembler(tiny.Fingerprint, Tokenizer);
        StoredAssembly r = assembler.Assemble(Context3, tiny.Model, new FileChunkStore(_root));
        Assert.Equal(3, r.Entries.Count);
        Assert.Empty(r.Context.QueryTokenIds);
        var (context, stats) = r;
        Assert.Same(r.Context, context);
        Assert.Equal(new AssemblyStats(0, 3, TokA.Length + TokB.Length + TokC.Length, 0), stats);
    }

    // ------------------------------------------------------------------------------------------------ hit / miss counts

    [Fact]
    public void Assemble_ColdThenWarm_HitMissCounts()
    {
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var counting = new CountingModel(tiny.Model);
        var assembler = new StoredContextAssembler(tiny.Fingerprint, Tokenizer);
        var store = new FileChunkStore(_root);

        // Cold: A, B, C are encoded once each; the second A is a repeat served from the first (no second encode, not a store hit).
        StoredAssembly cold = assembler.Assemble(Context4, counting, store);
        long distinct = TokA.Length + TokB.Length + TokC.Length;
        Assert.Equal(new AssemblyStats(Hits: 0, Misses: 3, EncodedTokens: distinct, Repeats: 1), cold.Stats);
        Assert.Equal(distinct, counting.ForwardTokens);
        Assert.Equal(3, store.Keys().Count());

        // Warm: nothing is encoded.
        counting.ForwardTokens = 0;
        StoredAssembly warm = assembler.Assemble(Context4, counting, new FileChunkStore(_root));
        Assert.Equal(new AssemblyStats(Hits: 3, Misses: 0, EncodedTokens: 0, Repeats: 1), warm.Stats);
        Assert.Equal(0, counting.ForwardTokens);

        // Partially warm: B evicted, D new -> two misses, two encodes, the rest hits.
        store.Evict(KeyOf(tiny, TokB));
        StoredAssembly partial = assembler.Assemble([Ch("a.cs", TokA), Ch("b.cs", TokB), Ch("d.cs", TokD), Ch("c.cs", TokC)], counting, store);
        Assert.Equal(new AssemblyStats(Hits: 2, Misses: 2, EncodedTokens: TokB.Length + TokD.Length, Repeats: 0), partial.Stats);
        Assert.Equal(TokB.Length + TokD.Length, counting.ForwardTokens);
        Assert.Equal(4, store.Keys().Count());
    }

    [Fact]
    public void Assemble_StoresUnderTheCanonicalKey()
    {
        // The key the assembler writes is ChunkKey.Create(fingerprint, tokenizer, RoPE from config, "cpu", F32, tokens), so other
        // tools (index/status) can find entries without the assembler.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var store = new FileChunkStore(_root);
        new StoredContextAssembler(tiny.Fingerprint, Tokenizer).Assemble(Context3, tiny.Model, store);
        var expected = new[] { TokA, TokB, TokC }.Select(t => KeyOf(tiny, t)).OrderBy(k => k.Hex, StringComparer.Ordinal);
        Assert.Equal(expected, store.Keys().OrderBy(k => k.Hex, StringComparer.Ordinal));
        Assert.Equal(KeyOf(tiny, TokA), new StoredContextAssembler(tiny.Fingerprint, Tokenizer).KeyOf(tiny.Config, TokA));
    }

    [Fact]
    public void Assemble_ModelChanged_AllMisses()
    {
        using var a = TinyModel.Load(_scratch, seed: 1);
        using var b = TinyModel.Load(_scratch, seed: 2); // same shape, other weights
        Assert.NotEqual(a.Fingerprint, b.Fingerprint);
        var store = new FileChunkStore(_root);
        new StoredContextAssembler(a.Fingerprint, Tokenizer).Assemble(Context3, a.Model, store);

        // Model B over the same store: same chunk text, different fingerprint -> all misses, never A's K/V.
        var asmB = new StoredContextAssembler(b.Fingerprint, Tokenizer);
        StoredAssembly rb = asmB.Assemble(Context3, b.Model, store);
        Assert.Equal(new AssemblyStats(0, 3, TokA.Length + TokB.Length + TokC.Length, 0), rb.Stats);
        var encB = new ChunkEncoder(b.Fingerprint);
        for (int i = 0; i < 3; i++) AssertBitExact(encB.Encode(b.Model, Context3[i].TokenIds), rb.Entries[i]);
        Assert.Equal(6, store.Keys().Count());

        // Tokenizer changed, model same -> misses as well.
        StoredAssembly rt = new StoredContextAssembler(a.Fingerprint, "tok-other").Assemble(Context3, a.Model, store);
        Assert.Equal(0, rt.Stats.Hits);
        Assert.Equal(3, rt.Stats.Misses);

        // Model A again: still all hits (B did not overwrite A's entries).
        StoredAssembly ra = new StoredContextAssembler(a.Fingerprint, Tokenizer).Assemble(Context3, a.Model, store);
        Assert.Equal(3, ra.Stats.Hits);
        var encA = new ChunkEncoder(a.Fingerprint);
        for (int i = 0; i < 3; i++) AssertBitExact(encA.Encode(a.Model, Context3[i].TokenIds), ra.Entries[i]);
    }

    // ------------------------------------------------------------------------------------------------ bad entries

    [Fact]
    public void Assemble_StaleSchemaEntry_IsMissNotError()
    {
        // An entry written under an older K/V schema version lives under another key: a miss, re-encoded under the current key.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var store = new FileChunkStore(_root);
        ChunkKvEntry encoded = new ChunkEncoder(tiny.Fingerprint).Encode(tiny.Model, TokA);
        ChunkKey stale = ChunkKey.Create(tiny.Fingerprint, Tokenizer, RopeParameters.FromConfig(tiny.Config), "cpu", KvDtype.F32, TokA,
            kvOutputVersion: ChunkReuseSchema.KvOutputVersion - 1);
        store.Put(stale, encoded);

        StoredAssembly r = new StoredContextAssembler(tiny.Fingerprint, Tokenizer).Assemble([Ch("a.cs", TokA)], tiny.Model, store);
        Assert.Equal(new AssemblyStats(0, 1, TokA.Length, 0), r.Stats);
        Assert.True(store.Contains(KeyOf(tiny, TokA)));
        Assert.True(store.Contains(stale)); // left alone; not this assembler's key
    }

    [Theory]
    [InlineData("version")]
    [InlineData("truncated")]
    [InlineData("other-key-file")]
    public void Assemble_CorruptOrForeignFile_IsMiss_ReencodedAndReplaced(string damage)
    {
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var store = new FileChunkStore(_root);
        var assembler = new StoredContextAssembler(tiny.Fingerprint, Tokenizer);
        assembler.Assemble(Context3, tiny.Model, store);
        ChunkKey keyB = KeyOf(tiny, TokB);
        string pathB = store.PathOf(keyB);
        switch (damage)
        {
            case "version":
                using (var fs = new FileStream(pathB, FileMode.Open, FileAccess.Write)) { fs.Position = 4; fs.Write(BitConverter.GetBytes(99)); }
                break;
            case "truncated":
                using (var fs = new FileStream(pathB, FileMode.Open, FileAccess.Write)) fs.SetLength(fs.Length - 7);
                break;
            case "other-key-file": // chunk A's (valid) file at chunk B's path
                File.Copy(store.PathOf(KeyOf(tiny, TokA)), pathB, overwrite: true);
                break;
        }
        Assert.False(store.Contains(keyB));

        StoredAssembly r = assembler.Assemble(Context3, tiny.Model, store);
        Assert.Equal(new AssemblyStats(Hits: 2, Misses: 1, EncodedTokens: TokB.Length, Repeats: 0), r.Stats);
        ChunkKvEntry freshB = new ChunkEncoder(tiny.Fingerprint).Encode(tiny.Model, TokB);
        AssertBitExact(freshB, r.Entries[1]);

        // Replaced on disk: a new instance reads the repaired entry.
        Assert.True(new FileChunkStore(_root).TryGet(keyB, out ChunkKvEntry repaired));
        AssertBitExact(freshB, repaired);
        Assert.Equal(3, assembler.Assemble(Context3, tiny.Model, store).Stats.Hits);
    }

    [Theory]
    [InlineData("tokens")]
    [InlineData("fingerprint")]
    [InlineData("layers")]
    [InlineData("stride")]
    public void Assemble_StructurallyValidButWrongEntry_IsMiss_EvictedAndReplaced(string wrong)
    {
        // A store may hand back a well-formed entry that does not belong to the key (a buggy or foreign IChunkStore). The
        // assembler checks fingerprint, token ids and per-layer shape itself, and replaces the entry.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var encoder = new ChunkEncoder(tiny.Fingerprint);
        ChunkKvEntry good = encoder.Encode(tiny.Model, TokB);
        ChunkKvEntry bad = wrong switch
        {
            "tokens" => encoder.Encode(tiny.Model, [8, 22, 1, 36]), // same length, one token different
            "fingerprint" => new ChunkEncoder("another-model").Encode(tiny.Model, TokB),
            "layers" => good with { KeysPerLayer = good.KeysPerLayer[..^1], ValuesPerLayer = good.ValuesPerLayer[..^1] },
            "stride" => good with
            {
                KeysPerLayer = good.KeysPerLayer.Select(k => k[..^TokB.Length]).ToArray(),
                ValuesPerLayer = good.ValuesPerLayer.Select(v => v[..^TokB.Length]).ToArray(),
            },
            _ => throw new ArgumentOutOfRangeException(nameof(wrong)),
        };
        ChunkKey keyB = KeyOf(tiny, TokB);
        var store = new DictionaryStore();
        store.Put(keyB, bad);

        using SimpleKvCache cache = NewCache(tiny);
        StoredAssembly r = new StoredContextAssembler(tiny.Fingerprint, Tokenizer).Assemble([Ch("b.cs", TokB)], tiny.Model, store, cache);
        Assert.Equal(new AssemblyStats(0, 1, TokB.Length, 0), r.Stats);
        Assert.Equal(1, r.Rejected);
        AssertBitExact(good, r.Entries[0]);
        Assert.Equal([keyB], store.Evicted);
        Assert.True(store.TryGet(keyB, out ChunkKvEntry stored));
        AssertBitExact(good, stored);
        Assert.Equal(TokB.Length, cache.CurrentLength);
    }

    // ------------------------------------------------------------------------------------------------ busy / concurrency

    [Fact]
    public void Assemble_StoreBusy_UsesFreshEntry_AndALaterCallRepairs()
    {
        // A corrupt entry is mapped by another reader, so Put cannot replace it (ChunkStoreBusyException). Assembly still
        // succeeds with the freshly encoded entry; the key stays a miss until a later call stores it.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var store = new FileChunkStore(_root);
        var assembler = new StoredContextAssembler(tiny.Fingerprint, Tokenizer);
        assembler.Assemble(Context3, tiny.Model, store);
        string pathB = store.PathOf(KeyOf(tiny, TokB));
        using (var fs = new FileStream(pathB, FileMode.Open, FileAccess.Write)) fs.SetLength(100);

        ChunkKvEntry freshB = new ChunkEncoder(tiny.Fingerprint).Encode(tiny.Model, TokB);
        var held = new FileStream(pathB, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
        var map = MemoryMappedFile.CreateFromFile(held, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: true);
        try
        {
            using SimpleKvCache cache = NewCache(tiny);
            StoredAssembly r = assembler.Assemble(Context3, tiny.Model, store, cache);
            Assert.Equal(new AssemblyStats(2, 1, TokB.Length, 0), r.Stats);
            Assert.Equal(1, r.StoreFailures);
            AssertBitExact(freshB, r.Entries[1]);
            Assert.Equal(TokA.Length + TokB.Length + TokC.Length, cache.CurrentLength);
            Assert.False(store.Contains(KeyOf(tiny, TokB)));
        }
        finally
        {
            map.Dispose();
            held.Dispose();
        }

        StoredAssembly repair = assembler.Assemble(Context3, tiny.Model, store);
        Assert.Equal(new AssemblyStats(2, 1, TokB.Length, 0), repair.Stats);
        Assert.Equal(0, repair.StoreFailures);
        Assert.Equal(3, assembler.Assemble(Context3, tiny.Model, store).Stats.Hits);
    }

    [Fact]
    public void Assemble_BusyFromStore_IsNotSwallowedForOtherIOErrors()
    {
        // Only ChunkStoreBusyException is tolerated on Put; any other store failure surfaces.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var store = new DictionaryStore { PutFailure = () => new IOException("disk full") };
        Assert.Throws<IOException>(() => new StoredContextAssembler(tiny.Fingerprint, Tokenizer).Assemble(Context3, tiny.Model, store));

        var busy = new DictionaryStore { PutFailure = () => new ChunkStoreBusyException("x", new IOException("held")) };
        StoredAssembly r = new StoredContextAssembler(tiny.Fingerprint, Tokenizer).Assemble(Context4, tiny.Model, busy);
        Assert.Equal(new AssemblyStats(0, 3, TokA.Length + TokB.Length + TokC.Length, 1), r.Stats); // the repeat of A is still not re-encoded
        Assert.Equal(3, r.StoreFailures);
    }

    [Fact]
    public void Assemble_ConcurrentSameChunks_NoCorruption()
    {
        // Several assemblers (own model instance and own store instance each, one root) assemble the same chunks at once.
        // Every result equals the fresh encode, and the store ends with exactly one valid entry per chunk and no temp files.
        const int Workers = 4;
        var models = Enumerable.Range(0, Workers).Select(_ => TinyModel.Load(_scratch, seed: 1)).ToArray();
        try
        {
            string fp = models[0].Fingerprint;
            var encoder = new ChunkEncoder(fp);
            var fresh = Context4.Select(c => encoder.Encode(models[0].Model, c.TokenIds)).ToArray();
            for (int round = 0; round < 3; round++)
            {
                using var barrier = new Barrier(Workers);
                var results = new StoredAssembly[Workers];
                var errors = new Exception?[Workers];
                var threads = Enumerable.Range(0, Workers).Select(w => new Thread(() =>
                {
                    var store = new FileChunkStore(_root);
                    barrier.SignalAndWait();
                    try { results[w] = new StoredContextAssembler(fp, Tokenizer).Assemble(Context4, models[w].Model, store); }
                    catch (Exception e) { errors[w] = e; } // an unhandled thread exception would kill the test host
                })).ToArray();
                foreach (var t in threads) t.Start();
                foreach (var t in threads) t.Join();
                Assert.All(errors, e => Assert.Null(e));

                foreach (StoredAssembly r in results)
                {
                    Assert.Equal(4, r.Stats.Hits + r.Stats.Misses + r.Stats.Repeats);
                    Assert.Equal(1, r.Stats.Repeats);
                    for (int i = 0; i < 4; i++) AssertBitExact(fresh[i], r.Entries[i]);
                }
                if (round == 0) Assert.True(results.Sum(r => r.Stats.Misses) >= 3); // every chunk encoded at least once
                if (round == 1) Assert.True(results.Sum(r => r.Stats.Misses) >= 1); // the corrupt C was re-encoded at least once
                if (round == 2) Assert.All(results, r => Assert.Equal(3, r.Stats.Hits));

                var check = new FileChunkStore(_root);
                Assert.Equal(3, check.Keys().Count());
                Assert.Equal(0, check.GetStats().TempFiles);
                foreach (int[] t in new[] { TokA, TokB, TokC })
                {
                    Assert.True(check.TryGet(KeyOf(models[0], t), out ChunkKvEntry e));
                    AssertBitExact(encoder.Encode(models[0].Model, t), e);
                }
                if (round == 0)
                {
                    // Next round starts from a store with one corrupt entry: concurrent repair.
                    using var fs = new FileStream(check.PathOf(KeyOf(models[0], TokC)), FileMode.Open, FileAccess.Write);
                    fs.SetLength(50);
                }
                _output.WriteLine($"round {round}: " + string.Join("; ", results.Select(r => r.Stats.ToString())));
            }
        }
        finally
        {
            foreach (var m in models) m.Dispose();
        }
    }

    // ------------------------------------------------------------------------------------------------ argument checks

    [Theory]
    [InlineData("capacity", typeof(ArgumentException))]
    [InlineData("prefilled-capacity", typeof(ArgumentException))]
    [InlineData("not-exact", typeof(NotSupportedException))]
    [InlineData("layers", typeof(NotSupportedException))]
    [InlineData("stride", typeof(ArgumentException))]
    public void Assemble_UnusableDestination_ThrowsBeforeAnyStoreReadOrEncode(string problem, Type expected)
    {
        // A destination that cannot take the chunks is refused up front: no store read, no encode (about 30 s per 2K context on
        // Qwen3-4B), nothing stored, and the destination is unchanged.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var counting = new CountingModel(tiny.Model);
        var store = new DictionaryStore();
        var geometry = KvGeometry.FromConfig(tiny.Config, compactSlidingWindows: false);
        int total = TokA.Length + TokB.Length + TokC.Length; // 15
        using var inner = new SimpleKvCache(geometry, problem == "capacity" ? total - 1 : SyntheticDenseGguf.Context);
        if (problem == "prefilled-capacity")
        {
            // 50 rows already used: 50 + 15 > 64.
            var filler = Enumerable.Range(0, 5).Select(_ => new ChunkEncoder(tiny.Fingerprint).Encode(tiny.Model, Enumerable.Range(0, 10).ToArray())).ToList();
            new ChunkAssembler(tiny.Config).Assemble(inner, filler, tiny.Fingerprint);
        }
        int before = inner.CurrentLength;
        IKvCache destination = problem switch
        {
            "not-exact" => new WrappedCache(inner, exact: false),
            "layers" => new WrappedCache(inner, extraLayers: 1),
            "stride" => new WrappedCache(inner, strideDelta: tiny.Config.HeadDim),
            _ => inner,
        };

        Exception ex = Assert.ThrowsAny<Exception>(() =>
            new StoredContextAssembler(tiny.Fingerprint, Tokenizer).Assemble(Context3, counting, store, destination));
        Assert.IsType(expected, ex);
        Assert.Equal(0, counting.ForwardTokens);
        Assert.Equal(0, store.Gets);
        Assert.Empty(store.Keys());
        Assert.Equal(before, inner.CurrentLength);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(49)]
    public void Assemble_DestinationExactFit_Succeeds(int prefilled)
    {
        // CurrentLength + context tokens == MaxLength is a fit, not an overflow.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var geometry = KvGeometry.FromConfig(tiny.Config, compactSlidingWindows: false);
        int total = TokA.Length + TokB.Length + TokC.Length;
        using var cache = new SimpleKvCache(geometry, prefilled + total);
        if (prefilled > 0)
        {
            int[] filler = Enumerable.Range(0, prefilled).Select(i => i % SyntheticDenseGguf.Vocab).ToArray();
            new ChunkAssembler(tiny.Config).Assemble(cache, [new ChunkEncoder(tiny.Fingerprint).Encode(tiny.Model, filler)], tiny.Fingerprint);
        }
        StoredAssembly r = new StoredContextAssembler(tiny.Fingerprint, Tokenizer).Assemble(Context3, tiny.Model, new DictionaryStore(), cache);
        Assert.Equal(prefilled, r.DestinationOffset);
        Assert.Equal(prefilled + total, cache.CurrentLength);
        Assert.Equal(cache.MaxLength, cache.CurrentLength);
    }

    [Fact]
    public void Assemble_NonEmptyDestination_AppendsAtItsLength_AndReportsTheOffset()
    {
        // Context.ChunkStarts are relative to the context; the destination rows are DestinationOffset + ChunkStarts[c]. The K/V
        // equal ChunkAssembler placing [D, A, B] in one call (keys re-rotated to their absolute positions).
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var encoder = new ChunkEncoder(tiny.Fingerprint);
        ChunkKvEntry d = encoder.Encode(tiny.Model, TokD), a = encoder.Encode(tiny.Model, TokA), b = encoder.Encode(tiny.Model, TokB);
        using SimpleKvCache expected = NewCache(tiny);
        new ChunkAssembler(tiny.Config).Assemble(expected, [d, a, b], tiny.Fingerprint);

        using SimpleKvCache actual = NewCache(tiny);
        new ChunkAssembler(tiny.Config).Assemble(actual, [d], tiny.Fingerprint);
        StoredAssembly r = new StoredContextAssembler(tiny.Fingerprint, Tokenizer)
            .Assemble([Ch("a.cs", TokA), Ch("b.cs", TokB)], tiny.Model, new FileChunkStore(_root), actual);
        Assert.Equal(TokD.Length, r.DestinationOffset);
        Assert.Equal(new[] { 0, TokA.Length }, r.Context.ChunkStarts);
        int length = TokD.Length + TokA.Length + TokB.Length;
        Assert.Equal(length, actual.CurrentLength);
        AssertCachesBitExact(tiny, expected, actual, length);

        // Without a destination the offset is 0.
        Assert.Equal(0, new StoredContextAssembler(tiny.Fingerprint, Tokenizer).Assemble([Ch("a.cs", TokA)], tiny.Model, new FileChunkStore(_root)).DestinationOffset);
    }

    [Fact]
    public void Assemble_EvictFailsOnRejectedEntry_IsMissNotError()
    {
        // A wrong entry that cannot be evicted: the assembler still uses a fresh encode, counts a store failure and does not throw.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var encoder = new ChunkEncoder(tiny.Fingerprint);
        ChunkKvEntry good = encoder.Encode(tiny.Model, TokB);
        ChunkKey keyB = KeyOf(tiny, TokB);
        foreach (Func<Exception> failure in new Func<Exception>[] { () => new IOException("in use"), () => new UnauthorizedAccessException("denied") })
        {
            var store = new DictionaryStore { EvictFailure = failure };
            store.Put(keyB, encoder.Encode(tiny.Model, [8, 22, 1, 36]));
            using SimpleKvCache cache = NewCache(tiny);
            StoredAssembly r = new StoredContextAssembler(tiny.Fingerprint, Tokenizer).Assemble([Ch("b.cs", TokB)], tiny.Model, store, cache);
            Assert.Equal(new AssemblyStats(0, 1, TokB.Length, 0), r.Stats);
            Assert.Equal(1, r.Rejected);
            Assert.Equal(1, r.StoreFailures);
            AssertBitExact(good, r.Entries[0]);
            Assert.Equal(TokB.Length, cache.CurrentLength);
        }

        // Other exception types from Evict are not store failures and propagate.
        var odd = new DictionaryStore { EvictFailure = () => new InvalidOperationException("bug") };
        odd.Put(keyB, encoder.Encode(tiny.Model, [8, 22, 1, 36]));
        Assert.Throws<InvalidOperationException>(() =>
            new StoredContextAssembler(tiny.Fingerprint, Tokenizer).Assemble([Ch("b.cs", TokB)], tiny.Model, odd));
    }

    [Fact]
    public void ForModel_RefusesAModelWhoseConfigDoesNotMatchTheIdentity()
    {
        // ForModel derives the fingerprint from (config, model id, quantization, tokenizer) and checks every model passed to
        // Assemble against it, so a model of another shape or RoPE cannot write its K/V under this identity's keys.
        using var qwen = TinyModel.Load(_scratch, seed: 1);
        using var llama = TinyModel.Load(_scratch, seed: 1, arch: "llama");
        var assembler = StoredContextAssembler.ForModel(qwen.Config, TinyModel.ModelIdOf(1), "F32", Tokenizer);
        Assert.Equal(qwen.Fingerprint, assembler.ModelFingerprint);

        var store = new DictionaryStore();
        var counting = new CountingModel(llama.Model);
        var ex = Assert.Throws<ArgumentException>(() => assembler.Assemble(Context3, counting, store));
        Assert.Contains("fingerprint", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(0, store.Gets);
        Assert.Equal(0, counting.ForwardTokens);
        Assert.Empty(store.Keys());

        // The matching model works and writes the same keys as the fingerprint constructor.
        StoredAssembly ok = assembler.Assemble(Context3, qwen.Model, store);
        Assert.Equal(3, ok.Stats.Misses);
        Assert.True(store.Contains(KeyOf(qwen, TokA)));
    }

    [Fact]
    public void Assemble_EntryUnderContextualPositioningKey_IsNotAHit()
    {
        // Store entries must be isolated encodes (positions 0..n-1, no preceding context). The positioning is part of the key, so
        // K/V stored as Contextual (e.g. a recompute output) never answers an isolated lookup, even with identical tokens.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var store = new DictionaryStore();
        ChunkKvEntry entry = new ChunkEncoder(tiny.Fingerprint).Encode(tiny.Model, TokA);
        ChunkKey contextual = ChunkKey.Create(tiny.Fingerprint, Tokenizer, RopeParameters.FromConfig(tiny.Config), "cpu", KvDtype.F32, TokA,
            positioning: ChunkPositioning.Contextual);
        Assert.NotEqual(KeyOf(tiny, TokA), contextual);
        store.Put(contextual, entry);

        StoredAssembly r = new StoredContextAssembler(tiny.Fingerprint, Tokenizer).Assemble([Ch("a.cs", TokA)], tiny.Model, store);
        Assert.Equal(new AssemblyStats(0, 1, TokA.Length, 0), r.Stats);
        Assert.True(store.Contains(KeyOf(tiny, TokA)));
    }

    [Fact]
    public void Assemble_InvalidChunks_ThrowBeforeTouchingTheStore()
    {
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var store = new DictionaryStore();
        var assembler = new StoredContextAssembler(tiny.Fingerprint, Tokenizer);
        Assert.Throws<ArgumentException>(() => assembler.Assemble([], tiny.Model, store));
        Assert.Throws<ArgumentException>(() => assembler.Assemble([Ch("a.cs", TokA), Ch("e.cs", [])], tiny.Model, store));
        Assert.Equal(0, store.Gets);
        Assert.Empty(store.Keys());
    }

    // ------------------------------------------------------------------------------------------------ backend tag (Phase I Task 1, L4)

    // Pinned before the backend seam was merged (dotLLM aa11975a, CPU only): the CPU fingerprint and key of the tiny model are
    // unchanged by it, so CPU stores written before format version 2 differ only in the file format, not in their keys.
    private const string GoldenCpuFingerprint = "b724e6231aebff3569a81501051d493e3b81fbacf26377d5ea52e5e5c9bbbf90";
    private const string GoldenCpuKeyOfTokA = "b7ee7e3b0d140948eb5172b2611978f8853ea040fd7504b1852361a27081ac6e";

    [Fact]
    public void StoreHeader_PersistsBackendTag()
    {
        // The .ckv header carries the backend tag: a new store instance reads back the tag the entry was written with.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        ChunkKvEntry cpu = new ChunkEncoder(tiny.Fingerprint).Encode(tiny.Model, TokA);
        ChunkKvEntry vulkan = cpu with { Backend = ChunkBackendTags.Vulkan };
        ChunkKey cpuKey = KeyOf(tiny, TokA);
        ChunkKey vulkanKey = ChunkKey.Create(tiny.Fingerprint, Tokenizer, RopeParameters.FromConfig(tiny.Config), ChunkBackendTags.Vulkan, KvDtype.F32, TokA);
        var store = new FileChunkStore(_root);
        store.Put(cpuKey, cpu);
        store.Put(vulkanKey, vulkan);

        var reopened = new FileChunkStore(_root);
        Assert.True(reopened.TryGet(vulkanKey, out ChunkKvEntry gotVulkan));
        Assert.Equal("vulkan", gotVulkan.Backend);
        AssertBitExact(vulkan, gotVulkan);
        Assert.True(reopened.TryGet(cpuKey, out ChunkKvEntry gotCpu));
        Assert.Equal("cpu", gotCpu.Backend);
        AssertBitExact(cpu, gotCpu);
    }

    [Fact]
    public void StoredAssembly_EntryBackendDiffers_IsRejectedAndReencoded()
    {
        // An entry under this assembler's key whose Backend is another backend's (same fingerprint, tokens and shape) is not a hit:
        // it is evicted and replaced by a fresh encode stamped with this assembler's backend.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        ChunkKvEntry fresh = new ChunkEncoder(tiny.Fingerprint).Encode(tiny.Model, TokA);
        ChunkKey key = KeyOf(tiny, TokA);
        var store = new DictionaryStore();
        store.Put(key, fresh with { Backend = ChunkBackendTags.Vulkan });

        using SimpleKvCache cache = NewCache(tiny);
        StoredAssembly r = new StoredContextAssembler(tiny.Fingerprint, Tokenizer).Assemble([Ch("a.cs", TokA)], tiny.Model, store, cache);
        Assert.Equal(new AssemblyStats(0, 1, TokA.Length, 0), r.Stats);
        Assert.Equal(1, r.Rejected);
        Assert.Equal([key], store.Evicted);
        Assert.Equal("cpu", r.Entries[0].Backend);
        AssertBitExact(fresh, r.Entries[0]);
        Assert.True(store.TryGet(key, out ChunkKvEntry replaced));
        Assert.Equal("cpu", replaced.Backend);
        Assert.Equal(TokA.Length, cache.CurrentLength);

        // The replacement is a hit.
        StoredAssembly again = new StoredContextAssembler(tiny.Fingerprint, Tokenizer).Assemble([Ch("a.cs", TokA)], tiny.Model, store);
        Assert.Equal(new AssemblyStats(1, 0, 0, 0), again.Stats);
        Assert.Equal(0, again.Rejected);
    }

    [Fact]
    public void StoredAssembly_HeaderBackendTampered_IsMissAndReencoded()
    {
        // A .ckv whose header tag was changed to another well-formed tag reads back with that tag (the store cannot know the key's
        // backend) and the assembler rejects it, re-encodes and rewrites the file with its own tag.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var assembler = new StoredContextAssembler(tiny.Fingerprint, Tokenizer);
        Assert.Equal(new AssemblyStats(0, 1, TokA.Length, 0), assembler.Assemble([Ch("a.cs", TokA)], tiny.Model, new FileChunkStore(_root)).Stats);

        var store = new FileChunkStore(_root);
        string path = store.PathOf(KeyOf(tiny, TokA));
        byte[] bytes = File.ReadAllBytes(path);
        int at = 16 + 64 + 4 + System.Text.Encoding.UTF8.GetByteCount(tiny.Fingerprint) + 4;
        Assert.Equal("cpu"u8.ToArray(), bytes[at..(at + 3)]);
        "gpu"u8.CopyTo(bytes.AsSpan(at));
        File.WriteAllBytes(path, bytes);

        StoredAssembly tampered = assembler.Assemble([Ch("a.cs", TokA)], tiny.Model, store);
        Assert.Equal(new AssemblyStats(0, 1, TokA.Length, 0), tampered.Stats);
        Assert.Equal(1, tampered.Rejected);
        Assert.Equal("cpu"u8.ToArray(), File.ReadAllBytes(path)[at..(at + 3)]);
        Assert.Equal(new AssemblyStats(1, 0, 0, 0), assembler.Assemble([Ch("a.cs", TokA)], tiny.Model, new FileChunkStore(_root)).Stats);
    }

    [Fact]
    public void StoredAssembly_DefaultCtor_KeysOnCpu()
    {
        // The constructors and ForModel overloads without a backend mean the CPU backend; their fingerprint and keys equal the
        // explicit CpuChunkBackend ones and the values pinned before the backend seam existed.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        Assert.Equal(GoldenCpuFingerprint, tiny.Fingerprint);
        Assert.Equal(GoldenCpuFingerprint, ChunkFingerprint.Compute(tiny.Config, TinyModel.ModelIdOf(1), "F32", Tokenizer, ChunkBackendTags.Cpu));

        var byFingerprint = new StoredContextAssembler(tiny.Fingerprint, Tokenizer);
        var byForModel = StoredContextAssembler.ForModel(tiny.Config, TinyModel.ModelIdOf(1), "F32", Tokenizer);
        var explicitCpu = new StoredContextAssembler(tiny.Fingerprint, Tokenizer, new CpuChunkBackend(tiny.Config));
        var forModelCpu = StoredContextAssembler.ForModel(tiny.Config, TinyModel.ModelIdOf(1), "F32", Tokenizer, new CpuChunkBackend(tiny.Config));
        ChunkKey canonical = ChunkKey.Create(GoldenCpuFingerprint, Tokenizer, RopeParameters.FromConfig(tiny.Config), "cpu", KvDtype.F32, TokA);
        Assert.Equal(GoldenCpuKeyOfTokA, canonical.Hex);
        foreach (StoredContextAssembler a in new[] { byFingerprint, byForModel, explicitCpu, forModelCpu })
        {
            Assert.Equal("cpu", a.Backend);
            Assert.Equal(GoldenCpuFingerprint, a.ModelFingerprint);
            Assert.Equal(canonical, a.KeyOf(tiny.Config, TokA));
        }

        // And the default assembler stores CPU-tagged entries under that key.
        var store = new DictionaryStore();
        byFingerprint.Assemble([Ch("a.cs", TokA)], tiny.Model, store);
        Assert.Equal([canonical], store.Keys());
        Assert.True(store.TryGet(canonical, out ChunkKvEntry e));
        Assert.Equal("cpu", e.Backend);
    }

    [Fact]
    public void StoredAssembly_OtherBackend_KeysOnItsTag_AndNeverHitsCpuEntries()
    {
        // CPU stand-in for the GPU test below: a backend with another tag (the CPU backend relabelled, so it runs anywhere) keys,
        // fingerprints and stamps with its tag. A store filled by the CPU backend gives it no hits, and the reverse; neither evicts
        // the other's entries.
        using var tiny = TinyModel.Load(_scratch, seed: 1);
        var other = new RetaggedBackend(new CpuChunkBackend(tiny.Config), ChunkBackendTags.Vulkan);
        var cpu = StoredContextAssembler.ForModel(tiny.Config, TinyModel.ModelIdOf(1), "F32", Tokenizer);
        var vk = StoredContextAssembler.ForModel(tiny.Config, TinyModel.ModelIdOf(1), "F32", Tokenizer, other);
        Assert.Equal("vulkan", vk.Backend);
        string vkFp = ChunkFingerprint.Compute(tiny.Config, TinyModel.ModelIdOf(1), "F32", Tokenizer, ChunkBackendTags.Vulkan);
        Assert.Equal(vkFp, vk.ModelFingerprint);
        Assert.NotEqual(cpu.ModelFingerprint, vk.ModelFingerprint);
        Assert.Equal(ChunkKey.Create(vkFp, Tokenizer, RopeParameters.FromConfig(tiny.Config), "vulkan", KvDtype.F32, TokA), vk.KeyOf(tiny.Config, TokA));

        int tokens = TokA.Length + TokB.Length + TokC.Length;
        foreach (bool cpuFirst in new[] { true, false })
        {
            var store = new DictionaryStore();
            StoredContextAssembler first = cpuFirst ? cpu : vk, second = cpuFirst ? vk : cpu;
            Assert.Equal(new AssemblyStats(0, 3, tokens, 0), first.Assemble(Context3, tiny.Model, store).Stats);
            StoredAssembly cross = second.Assemble(Context3, tiny.Model, store);
            Assert.Equal(new AssemblyStats(0, 3, tokens, 0), cross.Stats);
            Assert.Equal(0, cross.Rejected);
            Assert.Empty(store.Evicted);
            Assert.All(cross.Entries, e => Assert.Equal(second.Backend, e.Backend));
            Assert.Equal(6, store.Keys().Count());
            Assert.Equal(new AssemblyStats(3, 0, 0, 0), first.Assemble(Context3, tiny.Model, store).Stats);
            Assert.Equal(new AssemblyStats(3, 0, 0, 0), second.Assemble(Context3, tiny.Model, store).Stats);
        }

        // Even with the CPU fingerprint (the fingerprint constructor), the tag alone separates the keys: no hit, no reject.
        var sameFp = new StoredContextAssembler(cpu.ModelFingerprint, Tokenizer, other);
        var shared = new DictionaryStore();
        cpu.Assemble(Context3, tiny.Model, shared);
        StoredAssembly r = sameFp.Assemble(Context3, tiny.Model, shared);
        Assert.Equal(new AssemblyStats(0, 3, tokens, 0), r.Stats);
        Assert.Equal(0, r.Rejected);
        Assert.Empty(shared.Evicted);
        Assert.True(shared.Contains(KeyOf(tiny, TokA)));
    }

    /// <summary>A backend that behaves as <paramref name="inner"/> but carries another tag (a CPU stand-in for a second backend).</summary>
    internal sealed class RetaggedBackend(IChunkBackend inner, string tag) : IChunkBackend
    {
        public string Tag => tag;
        public int ForwardDeviceId => inner.ForwardDeviceId;
        public RopeParameters? Rope => inner.Rope;
        public IKvCache CreateCache(int maxSeqLen) => inner.CreateCache(maxSeqLen);
        public void RequireCache(IKvCache cache) => inner.RequireCache(cache);
        public void WriteRows(IKvCache cache, int layer, int startPos, ReadOnlySpan<float> k, ReadOnlySpan<float> v) => inner.WriteRows(cache, layer, startPos, k, v);
        public void ReadRows(IKvCache cache, int layer, int startPos, int count, Span<float> k, Span<float> v) => inner.ReadRows(cache, layer, startPos, count, k, v);
        public void CopyRows(IKvCache source, int layer, int srcStart, IKvCache destination, int dstStart, int count) => inner.CopyRows(source, layer, srcStart, destination, dstStart, count);
        public void ShiftKeys(IKvCache cache, int startPos, int count, int delta, RopeParameters rope) => inner.ShiftKeys(cache, startPos, count, delta, rope);
    }

    // ------------------------------------------------------------------------------------------------ helpers

    private static ChunkKey KeyOf(TinyModel tiny, int[] tokens)
        => ChunkKey.Create(tiny.Fingerprint, Tokenizer, RopeParameters.FromConfig(tiny.Config), "cpu", KvDtype.F32, tokens);

    private static SimpleKvCache NewCache(TinyModel tiny)
        => new(KvGeometry.FromConfig(tiny.Config, compactSlidingWindows: false), SyntheticDenseGguf.Context);

    private static unsafe float[] LastRow(IModel model, int[] ids, int start, IKvCache cache)
    {
        using ITensor logits = model.Forward(ids, Enumerable.Range(start, ids.Length).ToArray(), -1, cache, lastTokenLogitsOnly: true);
        int vocab = logits.Shape[1];
        int row = logits.Shape[0] - 1;
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)row * vocab, vocab).ToArray();
    }

    private static bool CachesBitEqual(TinyModel tiny, SimpleKvCache a, SimpleKvCache b, int rows)
    {
        for (int l = 0; l < a.NumLayers; l++)
        {
            int n = rows * a.KvStrideOf(l);
            if (!MemoryMarshal.AsBytes(a.KeysSpan(l)[..n]).SequenceEqual(MemoryMarshal.AsBytes(b.KeysSpan(l)[..n]))) return false;
            if (!MemoryMarshal.AsBytes(a.ValuesSpan(l)[..n]).SequenceEqual(MemoryMarshal.AsBytes(b.ValuesSpan(l)[..n]))) return false;
        }
        return a.NumLayers == tiny.Config.NumLayers && b.NumLayers == a.NumLayers;
    }

    private static void AssertCachesBitExact(TinyModel tiny, SimpleKvCache expected, SimpleKvCache actual, int rows)
        => Assert.True(CachesBitEqual(tiny, expected, actual, rows), "assembled K/V caches differ");

    private static void AssertBitExact(ChunkKvEntry expected, ChunkKvEntry actual)
    {
        Assert.Equal(expected.ModelFingerprint, actual.ModelFingerprint);
        Assert.Equal(expected.TokenIds, actual.TokenIds);
        Assert.Equal(expected.Backend, actual.Backend);
        Assert.Equal(expected.KeysPerLayer.Length, actual.KeysPerLayer.Length);
        for (int l = 0; l < expected.KeysPerLayer.Length; l++)
        {
            Assert.True(MemoryMarshal.AsBytes(expected.KeysPerLayer[l].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.KeysPerLayer[l].AsSpan())),
                $"layer {l} K bits differ");
            Assert.True(MemoryMarshal.AsBytes(expected.ValuesPerLayer[l].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.ValuesPerLayer[l].AsSpan())),
                $"layer {l} V bits differ");
        }
    }

    /// <summary>In-memory store that returns whatever was put (no validation), records evictions and can fail Put.</summary>
    private sealed class DictionaryStore : IChunkStore
    {
        private readonly Dictionary<ChunkKey, ChunkKvEntry> _entries = new();

        public Func<Exception>? PutFailure { get; init; }

        public List<ChunkKey> Evicted { get; } = [];

        public int Gets { get; private set; }

        public long TotalBytes => 0;

        public bool TryGet(ChunkKey key, out ChunkKvEntry entry)
        {
            Gets++;
            return _entries.TryGetValue(key, out entry!);
        }

        public void Put(ChunkKey key, ChunkKvEntry entry)
        {
            if (PutFailure is not null) throw PutFailure();
            _entries.TryAdd(key, entry); // a present entry is kept, as IChunkStore specifies
        }

        public bool Contains(ChunkKey key) => _entries.ContainsKey(key);

        public IEnumerable<ChunkKey> Keys() => _entries.Keys.ToList();

        public Func<Exception>? EvictFailure { get; init; }

        public void Evict(ChunkKey key)
        {
            if (EvictFailure is not null) throw EvictFailure();
            Evicted.Add(key);
            _entries.Remove(key);
        }
    }

    /// <summary>Counts the tokens forwarded through the wrapped model (an independent measure of encoding work).</summary>
    private sealed class CountingModel(IModel inner) : IModel
    {
        public long ForwardTokens { get; set; }

        public ModelConfig Config => inner.Config;

        public long ComputeMemoryBytes => inner.ComputeMemoryBytes;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        {
            ForwardTokens += tokenIds.Length;
            return inner.Forward(tokenIds, positions, deviceId);
        }

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        {
            ForwardTokens += tokenIds.Length;
            return inner.Forward(tokenIds, positions, deviceId, kvCache);
        }

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, bool lastTokenLogitsOnly)
        {
            ForwardTokens += tokenIds.Length;
            return inner.Forward(tokenIds, positions, deviceId, kvCache, lastTokenLogitsOnly);
        }

        public void Dispose() { }
    }

    /// <summary>
    /// Wraps a <see cref="SimpleKvCache"/> and can misreport exactness, layer count or a layer's stride, to check that the
    /// assembler validates the destination before any store read or encode.
    /// </summary>
    private sealed class WrappedCache(SimpleKvCache inner, bool exact = true, int extraLayers = 0, int strideDelta = 0) : IKvCache, IPerLayerKvCache
    {
        public int CurrentLength => inner.CurrentLength;
        public int MaxLength => inner.MaxLength;
        public bool SupportsExactF32PrefixCopy => exact && inner.SupportsExactF32PrefixCopy;
        public int LayerCount => inner.NumLayers + extraLayers;
        public int KvStrideOf(int layerIndex) => inner.KvStrideOf(layerIndex) + (layerIndex == 0 ? strideDelta : 0);
        public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex) => inner.Update(keys, values, positions, layerIndex);
        public ITensor GetKeys(int layerIndex) => inner.GetKeys(layerIndex);
        public ITensor GetValues(int layerIndex) => inner.GetValues(layerIndex);
        public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex) => inner.Update(keys, values, positions, layerIndex);
        public TensorRef GetKeysRef(int layerIndex) => inner.GetKeysRef(layerIndex);
        public TensorRef GetValuesRef(int layerIndex) => inner.GetValuesRef(layerIndex);
        public void Rollback(int length) => inner.Rollback(length);
        public void Dispose() { }
    }

    /// <summary>A loaded synthetic model and its fingerprint; keeps the mapped GGUF alive for the model's lifetime.</summary>
    private sealed class TinyModel : IDisposable
    {
        private readonly GgufFile _gguf;

        private TinyModel(GgufFile gguf, ModelConfig config, TransformerModel model, string fingerprint)
        {
            _gguf = gguf;
            Config = config;
            Model = model;
            Fingerprint = fingerprint;
        }

        public ModelConfig Config { get; }

        public TransformerModel Model { get; }

        public string Fingerprint { get; }

        public static TinyModel Load(string scratch, int seed, string arch = "qwen3")
        {
            string path = Path.Combine(scratch, $"{arch}-{seed}-{Guid.NewGuid():N}.gguf");
            File.WriteAllBytes(path, SyntheticDenseGguf.Build(arch, seed));
            var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            string fp = ChunkFingerprint.Compute(config, ModelIdOf(seed, arch), "F32", Tokenizer);
            return new TinyModel(gguf, config, TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded), fp);
        }

        public static string ModelIdOf(int seed, string arch = "qwen3") => $"tiny-{arch}-seed{seed}";

        public void Dispose()
        {
            Model.Dispose();
            _gguf.Dispose();
        }
    }
}

/// <summary>
/// GPU half of the backend tag tests (Phase I Task 1, L4): store-backed assembly on the Vulkan backend with the synthetic F32 model.
/// CPU and Vulkan entries never mix: a store filled through one backend gives no hit to the other, by key and by the header tag.
/// Kept small (three short chunks, one tiny model, no recompute) because the box has had device-lost resets on long submits.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class StoredAssemblyTestsGpu : IDisposable
{
    private const string Tokenizer = "tok-tiny";
    private const string ModelId = "tiny-qwen3-default-seed";
    private static readonly int[] TokA = [3, 17, 5, 29, 11, 40];
    private static readonly int[] TokB = [8, 22, 1, 35];
    private static readonly int[] TokC = [12, 7, 44, 2, 19];
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-storedasm-gpu-" + Guid.NewGuid().ToString("N"));

    public StoredAssemblyTestsGpu() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private static IReadOnlyList<Chunk> Context3 =>
        [new("a.cs", 1, 1, (int[])TokA.Clone()), new("b.cs", 1, 1, (int[])TokB.Clone()), new("c.cs", 1, 1, (int[])TokC.Clone())];

    [SkippableFact]
    public void StoredAssembly_Vulkan_CpuEntryNeverHits()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var tiny = BackendSeamTests.TinyModel.Load(_scratch, "qwen3");
        using VulkanDevice device = VulkanDevice.Create();
        using IModel vkModel = VulkanTransformerModel.LoadFromGguf(device, tiny.Gguf, tiny.Config, spvDir);
        using var vkBackend = new VulkanChunkBackend(device, spvDir, tiny.Config, ChunkModelFeatures.FromGguf(tiny.Gguf));
        var cpu = StoredContextAssembler.ForModel(tiny.Config, ModelId, "F32", Tokenizer);
        var vk = StoredContextAssembler.ForModel(tiny.Config, ModelId, "F32", Tokenizer, vkBackend);
        Assert.Equal("cpu", cpu.Backend);
        Assert.Equal("vulkan", vk.Backend);
        int tokens = TokA.Length + TokB.Length + TokC.Length;

        foreach (bool cpuFirst in new[] { true, false })
        {
            string root = Path.Combine(_scratch, cpuFirst ? "cpu-first" : "vulkan-first");
            (StoredContextAssembler first, IModel firstModel) = cpuFirst ? (cpu, (IModel)tiny.Model) : (vk, vkModel);
            (StoredContextAssembler second, IModel secondModel) = cpuFirst ? (vk, vkModel) : (cpu, (IModel)tiny.Model);

            Assert.Equal(new AssemblyStats(0, 3, tokens, 0), first.Assemble(Context3, firstModel, new FileChunkStore(root)).Stats);
            StoredAssembly cross = second.Assemble(Context3, secondModel, new FileChunkStore(root));
            Assert.Equal(new AssemblyStats(0, 3, tokens, 0), cross.Stats);
            Assert.Equal(0, cross.Rejected);
            Assert.All(cross.Entries, e => Assert.Equal(second.Backend, e.Backend));

            // Both sets of entries are on disk with their own header tag, and each backend now hits its own.
            var store = new FileChunkStore(root);
            Assert.Equal(6, store.Keys().Count());
            foreach (StoredContextAssembler a in new[] { cpu, vk })
            {
                Assert.True(store.TryGet(a.KeyOf(tiny.Config, TokB), out ChunkKvEntry e));
                Assert.Equal(a.Backend, e.Backend);
            }
            Assert.Equal(new AssemblyStats(3, 0, 0, 0), first.Assemble(Context3, firstModel, new FileChunkStore(root)).Stats);

            // Vulkan warm assembly into a device cache (positive control on the device path).
            using IKvCache destination = vkBackend.CreateCache(SyntheticDenseGguf.Context);
            StoredAssembly warmVk = vk.Assemble(Context3, vkModel, new FileChunkStore(root), destination);
            Assert.Equal(new AssemblyStats(3, 0, 0, 0), warmVk.Stats);
            Assert.Equal(tokens, destination.CurrentLength);
        }
    }
}