using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Content-addressed chunk key (<see cref="ChunkKey"/>) and the on-disk store (<see cref="FileChunkStore"/>). No model is needed:
/// entries are seeded random K/V, so every byte of a round trip is checkable.
/// </summary>
public sealed class ChunkStoreTests : IDisposable
{
    private const string Fingerprint = "fp-model-a";
    private const string Tokenizer = "tok-a";
    private const string Backend = "cpu";
    private const KvDtype Dtype = KvDtype.F32;
    private static readonly RopeParameters Rope = new(128, 5_000_000f, RoPEType.NeoX);
    private static readonly int[] Tokens = [3, 17, 5, 29, 11, 40, 8];

    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-chunkstore-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best effort */ }
        catch (UnauthorizedAccessException) { }
    }

    // ---------------------------------------------------------------- key

    [Theory]
    [InlineData("model")]
    [InlineData("tokenizer")]
    [InlineData("rope-dim")]
    [InlineData("rope-theta")]
    [InlineData("rope-type")]
    [InlineData("kv-output-version")]
    [InlineData("backend")]
    [InlineData("dtype")]
    [InlineData("tokens")]
    [InlineData("positioning")]
    public void Key_DiffersWhenModelTokenizerRopeOrEngineVersionChanges(string field)
    {
        ChunkKey baseKey = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        ChunkKey changed = field switch
        {
            "model" => ChunkKey.Create("fp-model-b", Tokenizer, Rope, Backend, Dtype, Tokens),
            "tokenizer" => ChunkKey.Create(Fingerprint, "tok-b", Rope, Backend, Dtype, Tokens),
            "rope-dim" => ChunkKey.Create(Fingerprint, Tokenizer, Rope with { RopeDim = 64 }, Backend, Dtype, Tokens),
            "rope-theta" => ChunkKey.Create(Fingerprint, Tokenizer, Rope with { Theta = 1_000_000f }, Backend, Dtype, Tokens),
            "rope-type" => ChunkKey.Create(Fingerprint, Tokenizer, Rope with { Type = RoPEType.Norm }, Backend, Dtype, Tokens),
            "kv-output-version" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, ChunkReuseSchema.KvOutputVersion + 1),
            "backend" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, "vulkan", Dtype, Tokens),
            "dtype" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, KvDtype.F16, Tokens),
            "tokens" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, [3, 17, 5, 29, 11, 40, 9]),
            "positioning" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, positioning: ChunkPositioning.Contextual),
            _ => throw new ArgumentOutOfRangeException(nameof(field)),
        };
        Assert.NotEqual(baseKey, changed);

        // A changed producer must be a store miss, never a wrong hit: same chunk text, different fingerprint.
        var store = new FileChunkStore(_root);
        store.Put(baseKey, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 1));
        Assert.False(store.Contains(changed));
        Assert.False(store.TryGet(changed, out _));
    }

    [Fact]
    public void Key_FieldBoundariesCannotBeForged()
    {
        // Length prefixes: moving characters between adjacent string fields must change the key.
        ChunkKey a = ChunkKey.Create("ab", "c", Rope, Backend, Dtype, Tokens);
        ChunkKey b = ChunkKey.Create("a", "bc", Rope, Backend, Dtype, Tokens);
        Assert.NotEqual(a, b);
        // The token count is encoded: a trailing zero token is not the same chunk.
        Assert.NotEqual(ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, [1, 0]),
                        ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, [1]));
    }

    // chunk-key/v3 (before the namespace field) of ChunkStoreTests' default inputs; v3 stores are rebuilt, never read.
    private const string GoldenV3Key = "b0f13424a15549bb98aeff929c34318dbb47586ed3056d1dc177a0e6ef1c80de";

    [Fact]
    public void Key_DefaultNamespace_GoldenV4()
    {
        // Pins the canonical encoding (scheme string, field order, widths, endianness). An accidental change would orphan every
        // persistent store (misses, not wrong hits); a deliberate one bumps the scheme string and updates this value.
        // chunk-key/v4 hashes the namespace (length-prefixed, right after the scheme) even when it is the default "", so the
        // default namespace never collapses to the v3 encoding.
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        // Both values cross-checked when pinned by an independent re-implementation of the encoding (Python hashlib/struct,
        // 2026-11-02), which also reproduces GoldenV3Key under the v3 encoding.
        Assert.Equal("8be78abfcdcf2194b9cf662cdc6e1384d1d0808c5f492b720d65ce487935cd69", key.Hex);
        Assert.Equal(key, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ChunkNamespace.Default));
        Assert.Equal(key, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ""));
        Assert.NotEqual(GoldenV3Key, key.Hex);
        Assert.Equal("2c4b35eac74ee8c32a65c5974f92fba80a4880b8024b21a935b8b79621be3a41",
            ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: "tenant-a").Hex);
    }

    [Fact]
    public void Key_DiffersByNamespace()
    {
        ChunkKey def = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        ChunkKey a = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: "tenant-a");
        ChunkKey b = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: "tenant-b");
        ChunkKey k1 = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ChunkNamespace.ForApiKey("key-1"));
        ChunkKey k2 = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ChunkNamespace.ForApiKey("key-2"));
        Assert.Equal(5, new[] { def, a, b, k1, k2 }.Distinct().Count());
        // Same namespace, same key.
        Assert.Equal(a, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: new string("tenant-a".AsSpan())));

        // Moving characters between the namespace and the fingerprint (the next field) changes the key. (This pair would differ
        // even without the namespace's length prefix, because the fingerprint carries its own; a real prefix-free forgery would
        // need the fingerprint's 4 length bytes to be valid namespace characters, i.e. a fingerprint of >= 0x2d2d2d2d bytes. The
        // prefix is pinned by the golden values instead.)
        Assert.NotEqual(ChunkKey.Create("b", Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: "a"),
                        ChunkKey.Create("ab", Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ""));

        // Another namespace is a store miss, never a hit on the other tenant's entry.
        var store = new FileChunkStore(_root);
        store.Put(a, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 1));
        foreach (ChunkKey other in new[] { def, b, k1 })
        {
            Assert.False(store.Contains(other));
            Assert.False(store.TryGet(other, out _));
        }
        Assert.True(store.Contains(a));

        Assert.Throws<ArgumentNullException>(() => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: null!));
    }

    [Fact]
    public void Namespace_LoneSurrogates_NeverShareAKey()
    {
        // The hazard: the default UTF-8 encoder replaces every lone surrogate with U+FFFD, so two different namespace strings
        // would hash to the same bytes and share every key.
        Assert.Equal(System.Text.Encoding.UTF8.GetBytes("\uD800"), System.Text.Encoding.UTF8.GetBytes("\uDC00"));
        foreach (string ns in new[] { "\uD800", "\uDC00", "a\uD800", "a\uDC00" })
        {
            Assert.False(ChunkNamespace.IsValid(ns));
            Assert.Throws<ArgumentException>(() => ChunkNamespace.Validate(ns));
            Assert.ThrowsAny<ArgumentException>(() => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ns));
        }
    }

    [Theory]
    [InlineData("")]
    [InlineData("a")]
    [InlineData("tenant-a")]
    [InlineData("a.b_c-9")]
    [InlineData("k-0123456789abcdef")]
    [InlineData("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")] // 64
    public void Namespace_Valid_IsAccepted(string ns)
    {
        Assert.True(ChunkNamespace.IsValid(ns));
        ChunkNamespace.Validate(ns);
        Assert.Equal(64, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ns).Hex.Length);
    }

    [Theory]
    [InlineData("Tenant")]
    [InlineData("K-0123456789ABCDEF")]
    [InlineData("a/b")]
    [InlineData("a\\b")]
    [InlineData("a b")]
    [InlineData(" ")]
    [InlineData("tenant\n")]
    [InlineData("a\0")]
    [InlineData("é")]
    // Lone surrogates are in Namespace_LoneSurrogates_NeverShareAKey: xUnit serialises them identically, so as InlineData they
    // would collapse into one theory case.
    [InlineData("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")] // 65
    [InlineData("default")] // reserved (Phase I Task 4b): the index file name tag of the empty Default namespace
    public void Namespace_Invalid_IsRefusedEverywhere(string ns)
    {
        // Rule: ^(|k-[0-9a-f]{16}|[a-z0-9._-]{1,64})$ (empty = Default). Enforced by ChunkNamespace and by ChunkKey.Create, so a
        // namespace that reaches a key can also go into a file name, a JSON ledger or a log line unescaped.
        Assert.False(ChunkNamespace.IsValid(ns));
        Assert.Throws<ArgumentException>(() => ChunkNamespace.Validate(ns));
        Assert.ThrowsAny<ArgumentException>(() => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ns));
    }

    [Fact]
    public void Namespace_Null_IsNotValid()
    {
        Assert.False(ChunkNamespace.IsValid(null));
        Assert.Throws<ArgumentNullException>(() => ChunkNamespace.Validate(null!));
        Assert.True(ChunkNamespace.IsValid(ChunkNamespace.Default));
        Assert.True(ChunkNamespace.IsValid(ChunkNamespace.ForApiKey("sk-test-123")));
    }

    [Fact]
    public void ForApiKey_IsStable_AndNeverEqualsDefault()
    {
        // "k-" + the first 16 lowercase hex of SHA-256(UTF-8 key); golden cross-checked with Python hashlib (2026-11-02).
        Assert.Equal("k-e0dbaa0c6455768b", ChunkNamespace.ForApiKey("sk-test-123"));
        Assert.Equal(ChunkNamespace.ForApiKey("sk-test-123"), ChunkNamespace.ForApiKey(new string("sk-test-123".AsSpan())));
        Assert.Equal("k-2bb80d537b1da3e3", ChunkNamespace.ForApiKey("secret"));
        Assert.Equal("", ChunkNamespace.Default);
        foreach (string apiKey in new[] { "sk-test-123", "secret", " ", "é" })
        {
            string ns = ChunkNamespace.ForApiKey(apiKey);
            Assert.NotEqual(ChunkNamespace.Default, ns);
            Assert.Matches("^k-[0-9a-f]{16}$", ns); // a digest, never the key itself
        }
        Assert.NotEqual(ChunkNamespace.ForApiKey("sk-test-123"), ChunkNamespace.ForApiKey("sk-test-124"));
        Assert.Throws<ArgumentNullException>(() => ChunkNamespace.ForApiKey(null!));
        Assert.Throws<ArgumentException>(() => ChunkNamespace.ForApiKey("")); // no key: the caller uses Default
    }

    [Fact]
    public void Key_SameInputs_SameKey()
    {
        ChunkKey a = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        ChunkKey b = ChunkKey.Create(new string(Fingerprint.AsSpan()), new string(Tokenizer.AsSpan()),
            new RopeParameters(128, 5_000_000f, RoPEType.NeoX), new string(Backend.AsSpan()), KvDtype.F32, (int[])Tokens.Clone());
        Assert.Equal(a, b);
        // The default schema version is the current constant.
        Assert.Equal(a, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, ChunkReuseSchema.KvOutputVersion));
        // The default positioning is Isolated.
        Assert.Equal(a, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, positioning: ChunkPositioning.Isolated));
        // The default namespace is ChunkNamespace.Default.
        Assert.Equal(a, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ChunkNamespace.Default));
        Assert.Equal(64, a.Hex.Length);
        Assert.Matches("^[0-9a-f]{64}$", a.Hex);
    }

    // ---------------------------------------------------------------- store

    [Fact]
    public void Store_RoundTrip_IsBitExact()
    {
        // Values include NaN payloads, signed zero, subnormals and infinities, so any conversion or tolerance shows up.
        ChunkKvEntry entry = NewEntry(Tokens, Fingerprint, layers: 3, stride: 6, seed: 7);
        entry.KeysPerLayer[1][0] = BitConverter.Int32BitsToSingle(0x7FC0_1234); // NaN with payload
        entry.KeysPerLayer[1][1] = -0.0f;
        entry.ValuesPerLayer[2][3] = float.Epsilon;
        entry.ValuesPerLayer[0][5] = float.NegativeInfinity;

        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        Assert.False(store.Contains(key));
        store.Put(key, entry);
        Assert.True(store.Contains(key));

        // A fresh instance on the same root: the entry comes from disk, not from memory.
        var reopened = new FileChunkStore(_root);
        Assert.True(reopened.TryGet(key, out ChunkKvEntry got));
        AssertBitExact(entry, got);
        Assert.Equal([key], reopened.Keys());
        Assert.Equal(new FileInfo(EntryPath(key)).Length, reopened.TotalBytes);
        Assert.True(reopened.TotalBytes > 3 * 2 * Tokens.Length * 6 * sizeof(float));
    }

    [Fact]
    public void Store_LayersWithDifferentStrides_RoundTrip()
    {
        // Per-layer KV strides may differ (e.g. sliding-window/global layer mixes); the header records each layer's length.
        var rng = new Random(3);
        int n = Tokens.Length;
        int[] strides = [4, 8, 2];
        float[][] k = strides.Select(s => RandomFloats(rng, n * s)).ToArray();
        float[][] v = strides.Select(s => RandomFloats(rng, n * s)).ToArray();
        var entry = new ChunkKvEntry(Fingerprint, Tokens, k, v);
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        store.Put(key, entry);
        Assert.True(store.TryGet(key, out ChunkKvEntry got));
        AssertBitExact(entry, got);
    }

    [Theory]
    [InlineData("header-start")]
    [InlineData("header-middle")]
    [InlineData("tokens")]
    [InlineData("keys")]
    [InlineData("values")]
    [InlineData("last-byte")]
    [InlineData("empty")]
    public void Store_TruncatedFile_IsTreatedAsMiss(string where)
    {
        ChunkKvEntry entry = NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5);
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        store.Put(key, entry);

        string path = EntryPath(key);
        long full = new FileInfo(path).Length;
        long payload = (long)Tokens.Length * sizeof(int) + 2L * 2 * Tokens.Length * 4 * sizeof(float);
        long header = full - payload;
        long keysStart = header + Tokens.Length * sizeof(int);
        long valuesStart = keysStart + 2L * Tokens.Length * 4 * sizeof(float);
        long cut = where switch
        {
            "header-start" => 3,
            "header-middle" => header / 2,
            "tokens" => header + 5,
            "keys" => keysStart + 13,
            "values" => valuesStart + 9,
            "last-byte" => full - 1,
            "empty" => 0,
            _ => throw new ArgumentOutOfRangeException(nameof(where)),
        };
        using (var fs = new FileStream(path, FileMode.Open, FileAccess.Write)) fs.SetLength(cut);

        Assert.False(store.Contains(key));
        Assert.False(store.TryGet(key, out _));

        // A Put after the damage repairs the entry.
        store.Put(key, entry);
        Assert.True(store.TryGet(key, out ChunkKvEntry got));
        AssertBitExact(entry, got);
    }

    [Fact]
    public void Store_TrailingGarbage_IsTreatedAsMiss()
    {
        ChunkKvEntry entry = NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5);
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        store.Put(key, entry);
        using (var fs = new FileStream(EntryPath(key), FileMode.Append, FileAccess.Write)) fs.WriteByte(0x42);
        Assert.False(store.Contains(key));
        Assert.False(store.TryGet(key, out _));
    }

    [Fact]
    public void Store_EntryUnderAnotherKeysName_IsTreatedAsMiss()
    {
        // The header records the key; a file copied or renamed to another key's path must not be served for that key.
        ChunkKvEntry entry = NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5);
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        ChunkKey other = ChunkKey.Create(Fingerprint, Tokenizer, Rope, "vulkan", Dtype, Tokens);
        var store = new FileChunkStore(_root);
        store.Put(key, entry);
        string otherPath = EntryPath(other);
        Directory.CreateDirectory(Path.GetDirectoryName(otherPath)!);
        File.Copy(EntryPath(key), otherPath);
        Assert.False(store.Contains(other));
        Assert.False(store.TryGet(other, out _));
    }

    [Fact]
    public void Store_ConcurrentPutSameKey_LeavesOneValidEntry()
    {
        ChunkKvEntry entry = NewEntry(Tokens, Fingerprint, layers: 4, stride: 64, seed: 11);
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);

        // 16 threads in this process, released together, each doing 20 Put+TryGet rounds on one key. What this proves: no Put
        // throws, no read that hits returns different bits, and afterwards there is exactly one file (the .ckv; every temp file
        // was removed). It does not by itself prove torn-write safety: an in-place writer is caught here by a sharing violation,
        // not by a torn read; torn files are caught by the exact-length check (truncation tests). Cross-process writers are not
        // tested.
        const int Writers = 16;
        using var gate = new ManualResetEventSlim(false);
        var errors = new System.Collections.Concurrent.ConcurrentQueue<Exception>();
        int badReads = 0;
        var threads = new Thread[Writers];
        for (int t = 0; t < Writers; t++)
        {
            threads[t] = new Thread(() =>
            {
                gate.Wait();
                for (int r = 0; r < 20; r++)
                {
                    try
                    {
                        store.Put(key, entry);
                        if (store.TryGet(key, out ChunkKvEntry seen) && !SameBits(entry, seen))
                            Interlocked.Increment(ref badReads);
                    }
                    catch (Exception e) { errors.Enqueue(e); }
                }
            });
            threads[t].Start();
        }
        gate.Set();
        foreach (Thread th in threads) th.Join();

        Assert.Empty(errors);
        Assert.Equal(0, badReads); // a read that hits returns the written bits
        string[] files = Directory.GetFiles(_root, "*", SearchOption.AllDirectories);
        Assert.Single(files);
        Assert.EndsWith(".ckv", files[0], StringComparison.Ordinal);
        Assert.True(store.TryGet(key, out ChunkKvEntry got));
        AssertBitExact(entry, got);
        Assert.Equal([key], store.Keys());
    }

    [Fact]
    public void Store_IdenticalChunksFromTwoFiles_OneEntry()
    {
        // The key has no file path in it: identical chunk tokens from two source files resolve to the same key and file.
        int[] fromFileA = [10, 20, 30, 40];
        int[] fromFileB = [10, 20, 30, 40];
        ChunkKey a = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, fromFileA);
        ChunkKey b = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, fromFileB);
        Assert.Equal(a, b);

        var store = new FileChunkStore(_root);
        ChunkKvEntry entry = NewEntry(fromFileA, Fingerprint, layers: 2, stride: 4, seed: 2);
        store.Put(a, entry);
        long afterFirst = store.TotalBytes;
        store.Put(b, NewEntry(fromFileB, Fingerprint, layers: 2, stride: 4, seed: 2));

        Assert.Equal([a], store.Keys());
        Assert.Equal(afterFirst, store.TotalBytes);
        Assert.Single(Directory.GetFiles(_root, "*", SearchOption.AllDirectories));
        Assert.True(store.TryGet(a, out ChunkKvEntry viaA));
        Assert.True(store.TryGet(b, out ChunkKvEntry viaB));
        AssertBitExact(entry, viaA);
        AssertBitExact(entry, viaB);

        // A different chunk is a second entry and TotalBytes grows.
        int[] other = [10, 20, 30, 41];
        ChunkKey c = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, other);
        store.Put(c, NewEntry(other, Fingerprint, layers: 2, stride: 4, seed: 3));
        Assert.Equal(2, store.Keys().Count());
        Assert.True(store.TotalBytes > afterFirst);
    }

    [Fact]
    public void Store_PutExistingValidKey_KeepsExistingFile()
    {
        // Content-addressed: a valid entry under a key is never rewritten (a rewrite could fail on Windows while a reader has
        // the file mapped, and is wasted I/O).
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        ChunkKvEntry first = NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5);
        store.Put(key, first);
        store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 6));
        Assert.True(store.TryGet(key, out ChunkKvEntry got));
        AssertBitExact(first, got);
    }

    [Fact]
    public void Store_FormatVersionMismatch_IsMiss()
    {
        ChunkKvEntry entry = NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5);
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        store.Put(key, entry);

        // Header layout: 4-byte magic, then the int32 format version.
        string path = EntryPath(key);
        byte[] bytes = File.ReadAllBytes(path);
        Assert.Equal(FileChunkStore.FormatVersion, BitConverter.ToInt32(bytes, 4));
        BitConverter.GetBytes(FileChunkStore.FormatVersion + 1).CopyTo(bytes, 4);
        File.WriteAllBytes(path, bytes);

        Assert.False(store.Contains(key));
        Assert.False(store.TryGet(key, out _));

        // Wrong magic is a miss too.
        BitConverter.GetBytes(FileChunkStore.FormatVersion).CopyTo(bytes, 4);
        bytes[0] ^= 0xFF;
        File.WriteAllBytes(path, bytes);
        Assert.False(store.TryGet(key, out _));
    }

    [Fact]
    public void Store_DtypeTagMismatch_IsMiss()
    {
        // Header: magic, version, then the int32 dtype tag. Only F32 rows are readable today; any other tag is a miss.
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5));
        string path = EntryPath(key);
        byte[] bytes = File.ReadAllBytes(path);
        Assert.Equal((int)KvDtype.F32, BitConverter.ToInt32(bytes, 8));
        BitConverter.GetBytes((int)KvDtype.F16).CopyTo(bytes, 8);
        File.WriteAllBytes(path, bytes);
        Assert.False(store.Contains(key));
        Assert.False(store.TryGet(key, out _));
    }

    // A version-1 file (no backend tag) as FileChunkStore wrote it before format version 2: tokens [3, 17], one layer, stride 1,
    // K [1, -2], V [0.5, 4], fingerprint "fp-model-a", under GoldenV1Key. Captured from the store at dotLLM aa11975a and re-derived
    // byte for byte from the documented v1 layout with Python struct (2026-11-02). GoldenV1Key is the chunk-key/v3 key of those
    // inputs (v1 files were written under v3 keys); the test places the file under that key's name.
    private const string GoldenV1Key = "121e11e83c5294bccd77b5be9ca1de2568a677c018039dbced6ed29896645ca4";
    private const string GoldenV1Hex =
        "434b5653010000000000000000000000"
        + "31323165313165383363353239346263636437376235626539636131646532353638613637376330313830333964626365643665643239383936363435636134"
        + "0a00000066702d6d6f64656c2d61"
        + "0200000001000000020000000200000018000000000000000300000011000000"
        + "0000803f000000c00000003f00008040";

    [Fact]
    public void Store_FormatV1File_IsMiss()
    {
        // Version-1 files carry no backend tag, so a reader cannot tell CPU from Vulkan K/V: they are misses (re-encoded by the
        // assembler), never read as "cpu".
        Assert.Equal(2, FileChunkStore.FormatVersion);
        int[] tokens = [3, 17];
        var key = new ChunkKey(GoldenV1Key); // a v3 key; under chunk-key/v4 these inputs key elsewhere
        Assert.NotEqual(key, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, tokens));
        byte[] v1 = Convert.FromHexString(GoldenV1Hex);
        Assert.Equal(142, v1.Length);

        var store = new FileChunkStore(_root);
        string path = EntryPath(key);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, v1);
        Assert.Equal([key], store.Keys()); // listed (a file under a key's name) but not readable
        Assert.False(store.Contains(key));
        Assert.False(store.TryGet(key, out _));
        Assert.False(store.TryReadFingerprint(key, out _));

        // Put replaces it with a version-2 entry that reads back with its tag.
        var entry = new ChunkKvEntry(Fingerprint, tokens, [[1f, -2f]], [[0.5f, 4f]]);
        store.Put(key, entry);
        Assert.True(new FileChunkStore(_root).TryGet(key, out ChunkKvEntry got));
        AssertBitExact(entry, got);
        Assert.Equal("cpu", got.Backend);
        Assert.Equal(2, BitConverter.ToInt32(File.ReadAllBytes(path), 4));
    }

    [Fact]
    public void Store_HeaderBackendTag_Layout()
    {
        // v2 header: magic, version, dtype, reserved, 64-byte key, int32 + fingerprint, then int32 + ASCII backend tag; the size
        // helper the CLI's estimate uses equals the real file size.
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, "vulkan", Dtype, Tokens);
        var store = new FileChunkStore(_root);
        ChunkKvEntry entry = NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5) with { Backend = "vulkan" };
        store.Put(key, entry);
        byte[] bytes = File.ReadAllBytes(EntryPath(key));
        int at = BackendTagOffset(Fingerprint);
        Assert.Equal(6, BitConverter.ToInt32(bytes, at));
        Assert.Equal("vulkan"u8.ToArray(), bytes[(at + 4)..(at + 10)]);
        long payload = Tokens.Length * sizeof(int) + 2L * 2 * Tokens.Length * 4 * sizeof(float);
        Assert.Equal(bytes.Length, FileChunkStore.HeaderBytes(Fingerprint, "vulkan", layers: 2) + payload);
        Assert.Equal(FileChunkStore.HeaderBytes(Fingerprint, "cpu", 2) + 3, FileChunkStore.HeaderBytes(Fingerprint, "vulkan", 2));
    }

    [Theory]
    [InlineData("upper")]      // "CPU": not a tag (lowercase ASCII only)
    [InlineData("nul")]        // "c\0u"
    [InlineData("invalid-utf8")]
    [InlineData("len-zero")]
    [InlineData("len-negative")]
    [InlineData("len-longer")] // the length runs into the token count
    [InlineData("len-huge")]
    public void Store_HeaderBackendTampered_IsMiss(string tamper)
    {
        // A damaged backend tag is a miss, like any other damaged header field. A well-formed tag is reported as written (see
        // Store_HeaderBackendTampered_WellFormedTag_IsReportedAsWritten) and the assembler rejects it when it is not its own.
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5));
        string path = EntryPath(key);
        byte[] bytes = File.ReadAllBytes(path);
        int at = BackendTagOffset(Fingerprint);
        Assert.Equal(3, BitConverter.ToInt32(bytes, at));
        Assert.Equal("cpu"u8.ToArray(), bytes[(at + 4)..(at + 7)]);
        Assert.True(store.TryGet(key, out _)); // the untampered file is valid

        switch (tamper)
        {
            case "upper": "CPU"u8.CopyTo(bytes.AsSpan(at + 4)); break;
            case "nul": bytes[at + 5] = 0; break;
            case "invalid-utf8": bytes[at + 4] = 0xFF; bytes[at + 5] = 0xFE; break;
            case "len-zero": BitConverter.GetBytes(0).CopyTo(bytes, at); break;
            case "len-negative": BitConverter.GetBytes(-1).CopyTo(bytes, at); break;
            case "len-longer": BitConverter.GetBytes(4).CopyTo(bytes, at); break;
            case "len-huge": BitConverter.GetBytes(1 << 20).CopyTo(bytes, at); break;
            default: throw new ArgumentOutOfRangeException(nameof(tamper));
        }
        File.WriteAllBytes(path, bytes);
        Assert.False(store.Contains(key));
        Assert.False(store.TryGet(key, out _));
        Assert.False(store.TryReadFingerprint(key, out _));
    }

    [Fact]
    public void Store_HeaderBackendTampered_WellFormedTag_IsReportedAsWritten()
    {
        // The store cannot tell which backend a key was computed for, so a well-formed foreign tag reads back as written; the
        // entry then carries that tag and StoredContextAssembler rejects it (StoredAssembly_HeaderBackendTampered_IsMissAndReencoded).
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5));
        string path = EntryPath(key);
        byte[] bytes = File.ReadAllBytes(path);
        "gpu"u8.CopyTo(bytes.AsSpan(BackendTagOffset(Fingerprint) + 4));
        File.WriteAllBytes(path, bytes);
        Assert.True(new FileChunkStore(_root).TryGet(key, out ChunkKvEntry got));
        Assert.Equal("gpu", got.Backend);
    }

    [Theory]
    [InlineData("")]
    [InlineData("CPU")]
    [InlineData("cpu ")]
    [InlineData("vülkan")]
    [InlineData("a-tag-that-is-much-longer-than-sixty-four-bytes-and-so-cannot-be-a-backend")]
    public void Store_PutRejectsMalformedBackendTag(string tag)
    {
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        Assert.Throws<ArgumentException>(() => store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5) with { Backend = tag }));
        Assert.Throws<ArgumentNullException>(() => store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5) with { Backend = null! }));
        Assert.False(store.Contains(key));
        // The tags in use are accepted.
        store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5) with { Backend = "vulkan" });
        Assert.True(store.Contains(key));
    }

    [Fact]
    public void Store_PutOverEntryWithOtherBackendTag_ReplacesIt()
    {
        // Put keeps a valid entry only when its header tag equals the new entry's; an entry with another tag under the key (a
        // writer bug or tampering) is replaced even by a writer that does not evict first. Same tag: kept as before.
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5) with { Backend = "vulkan" });
        ChunkKvEntry cpu = NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 6);
        store.Put(key, cpu);
        Assert.True(new FileChunkStore(_root).TryGet(key, out ChunkKvEntry got));
        AssertBitExact(cpu, got);
        Assert.Equal("cpu", got.Backend);

        store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 7));
        Assert.True(store.TryGet(key, out ChunkKvEntry kept));
        AssertBitExact(cpu, kept);
    }

    [Fact]
    public void Store_HeaderBackendTagLongerThanMax_IsMiss()
    {
        // A tag of MaxBackendTagBytes (64) is stored and read; a file whose tag is one byte longer (otherwise consistent: the tag
        // length, the tag bytes and the file length all agree) is a miss, as Put would never write it (mutant M13).
        string max = new('a', FileChunkStore.MaxBackendTagBytes);
        Assert.Equal(64, max.Length);
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, max, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5) with { Backend = max });
        Assert.True(store.TryGet(key, out ChunkKvEntry got));
        Assert.Equal(max, got.Backend);

        string path = EntryPath(key);
        byte[] bytes = File.ReadAllBytes(path);
        int at = BackendTagOffset(Fingerprint);
        Assert.Equal(64, BitConverter.ToInt32(bytes, at));
        BitConverter.GetBytes(65).CopyTo(bytes, at);
        byte[] longer = [.. bytes[..(at + 4)], (byte)'a', .. bytes[(at + 4)..]];
        File.WriteAllBytes(path, longer);
        Assert.False(store.Contains(key));
        Assert.False(store.TryGet(key, out _));
    }

    // magic + version + dtype + reserved, key, fingerprint length + bytes: the backend tag's int32 length follows.
    private static int BackendTagOffset(string fingerprint) => 16 + 64 + 4 + System.Text.Encoding.UTF8.GetByteCount(fingerprint);

    [Fact]
    public void Store_Evict_RemovesEntry_AndMissingEvictIsNoOp()
    {
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        ChunkKey other = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, [1, 2, 3]);
        var store = new FileChunkStore(_root);
        store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5));
        store.Put(other, NewEntry([1, 2, 3], Fingerprint, layers: 2, stride: 4, seed: 6));
        long before = store.TotalBytes;

        store.Evict(key);
        Assert.False(store.Contains(key));
        Assert.False(store.TryGet(key, out _));
        Assert.True(store.Contains(other));
        Assert.Equal([other], store.Keys());
        Assert.True(store.TotalBytes < before);

        store.Evict(key); // already gone
    }

    [Fact]
    public void Store_IgnoresTempAndForeignFiles()
    {
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5));
        long bytes = store.TotalBytes;
        string shard = Path.GetDirectoryName(EntryPath(key))!;
        File.WriteAllBytes(Path.Combine(shard, key.Hex + ".abc.tmp"), new byte[100]); // a crashed writer's leftover
        File.WriteAllBytes(Path.Combine(shard, "notes.txt"), new byte[10]);
        File.Copy(EntryPath(key), Path.Combine(shard, key.Hex + ".ckvx")); // longer extension, valid-looking name
        Assert.Equal([key], store.Keys());
        Assert.Equal(bytes, store.TotalBytes);
    }

    [Fact]
    public void Store_PutRejectsMalformedEntry()
    {
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        // K/V layer count mismatch and a row count that is not a multiple of the token count.
        Assert.Throws<ArgumentException>(() => store.Put(key,
            new ChunkKvEntry(Fingerprint, Tokens, [new float[Tokens.Length]], [new float[Tokens.Length], new float[Tokens.Length]])));
        Assert.Throws<ArgumentException>(() => store.Put(key,
            new ChunkKvEntry(Fingerprint, Tokens, [new float[Tokens.Length + 1]], [new float[Tokens.Length]])));
        Assert.False(store.Contains(key));
    }

    [Fact]
    public void Store_PutWhileReaderHoldsValidDestination_Succeeds()
    {
        // Another writer stores the same key between this Put's existence check and its rename, and a reader maps that file:
        // the rename fails, the destination is valid, so Put returns without an exception and without leaving a temp file.
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        ChunkKvEntry entry = NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5);
        var store = new FileChunkStore(_root);
        var other = new FileChunkStore(_root);
        MemoryMappedFile? map = null;
        FileStream? held = null;
        store.BeforeRenameForTest = path =>
        {
            other.Put(key, entry);
            (held, map) = MapForRead(path);
        };
        try
        {
            store.Put(key, entry); // must not throw
        }
        finally
        {
            map?.Dispose();
            held?.Dispose();
        }
        Assert.NotNull(held); // the hook ran, so the rename really met a mapped destination
        Assert.True(store.TryGet(key, out ChunkKvEntry got));
        AssertBitExact(entry, got);
        Assert.Single(Directory.GetFiles(_root, "*", SearchOption.AllDirectories));
    }

    [Fact]
    public void Store_RepairWhileReaderHoldsCorruptFile_ThrowsBusyWithoutRetrying()
    {
        // A corrupt file at the destination is mapped by a reader, so it cannot be replaced. Defined behaviour: Put throws
        // ChunkStoreBusyException after one rename attempt (no retry loop), removes its temp file and leaves the key a miss;
        // a later Put repairs the entry.
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        ChunkKvEntry entry = NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5);
        var store = new FileChunkStore(_root);
        store.Put(key, entry);
        string path = EntryPath(key);
        using (var fs = new FileStream(path, FileMode.Open, FileAccess.Write)) fs.SetLength(100);

        int renames = 0;
        store.BeforeRenameForTest = _ => renames++;
        var (held, map) = MapForRead(path);
        try
        {
            Assert.Throws<ChunkStoreBusyException>(() => store.Put(key, entry));
            Assert.Equal(1, renames);
            Assert.False(store.TryGet(key, out _));
            Assert.Single(Directory.GetFiles(_root, "*", SearchOption.AllDirectories)); // temp file removed
        }
        finally
        {
            map.Dispose();
            held.Dispose();
        }
        store.Put(key, entry);
        Assert.True(store.TryGet(key, out ChunkKvEntry got));
        AssertBitExact(entry, got);
    }

    [Fact]
    public void Store_SweepTempFiles_RemovesOnlyOldTemps_AndStatsCountThemSeparately()
    {
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5));
        long entryBytes = store.TotalBytes;
        string shard = Path.GetDirectoryName(EntryPath(key))!;
        string oldTemp = Path.Combine(shard, key.Hex + "." + Guid.NewGuid().ToString("N") + ".tmp");
        string newTemp = Path.Combine(shard, key.Hex + "." + Guid.NewGuid().ToString("N") + ".tmp");
        string foreign = Path.Combine(shard, "notes.tmp");
        File.WriteAllBytes(oldTemp, new byte[100]);
        File.WriteAllBytes(newTemp, new byte[30]);
        File.WriteAllBytes(foreign, new byte[7]);
        File.SetLastWriteTimeUtc(oldTemp, DateTime.UtcNow.AddHours(-2));
        File.SetLastWriteTimeUtc(foreign, DateTime.UtcNow.AddHours(-2));

        Assert.Equal(new StoreStats(Entries: 1, EntryBytes: entryBytes, TempFiles: 2, TempBytes: 130), store.GetStats());
        Assert.Equal(entryBytes, store.TotalBytes); // temp files are not entry bytes

        Assert.Equal(1, store.SweepTempFiles(TimeSpan.FromHours(1)));
        Assert.False(File.Exists(oldTemp));
        Assert.True(File.Exists(newTemp));  // may belong to a writer that is still running
        Assert.True(File.Exists(foreign));  // not a store temp name
        Assert.True(store.Contains(key));
        Assert.Equal(new StoreStats(1, entryBytes, 1, 30), store.GetStats());

        Assert.Equal(1, store.SweepTempFiles(TimeSpan.Zero));
        Assert.Equal(new StoreStats(1, entryBytes, 0, 0), store.GetStats());
        Assert.True(File.Exists(foreign));
    }

    [Fact]
    public void Store_DeletedShardOrRoot_EnumerationTolerates()
    {
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        ChunkKey other = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, [1, 2, 3]);
        var store = new FileChunkStore(_root);
        store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5));
        store.Put(other, NewEntry([1, 2, 3], Fingerprint, layers: 2, stride: 4, seed: 6));
        Assert.NotEqual(key.Hex[..2], other.Hex[..2]); // two shards

        Directory.Delete(Path.GetDirectoryName(EntryPath(key))!, recursive: true);
        Assert.Equal([other], store.Keys());
        Assert.False(store.Contains(key));
        store.Evict(key); // no-op

        Directory.Delete(_root, recursive: true);
        Assert.Empty(store.Keys());
        Assert.Equal(0, store.TotalBytes);
        Assert.Equal(new StoreStats(0, 0, 0, 0), store.GetStats());
        Assert.Equal(0, store.SweepTempFiles(TimeSpan.Zero));
        store.Put(key, NewEntry(Tokens, Fingerprint, layers: 2, stride: 4, seed: 5)); // recreates the directories
        Assert.True(store.Contains(key));
    }

    // ---------------------------------------------------------------- helpers

    private static (FileStream Stream, MemoryMappedFile Map) MapForRead(string path)
    {
        var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
        var map = MemoryMappedFile.CreateFromFile(fs, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: true);
        return (fs, map);
    }

    private string EntryPath(ChunkKey key) => Path.Combine(_root, key.Hex[..2], key.Hex + ".ckv");

    private static ChunkKvEntry NewEntry(int[] tokens, string fingerprint, int layers, int stride, int seed)
    {
        var rng = new Random(seed);
        var k = new float[layers][];
        var v = new float[layers][];
        for (int l = 0; l < layers; l++)
        {
            k[l] = RandomFloats(rng, tokens.Length * stride);
            v[l] = RandomFloats(rng, tokens.Length * stride);
        }
        return new ChunkKvEntry(fingerprint, (int[])tokens.Clone(), k, v);
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var a = new float[count];
        for (int i = 0; i < count; i++) a[i] = (float)(rng.NextDouble() * 8 - 4);
        return a;
    }

    private static bool SameBits(ChunkKvEntry a, ChunkKvEntry b)
    {
        if (!string.Equals(a.ModelFingerprint, b.ModelFingerprint, StringComparison.Ordinal) || !a.TokenIds.AsSpan().SequenceEqual(b.TokenIds)) return false;
        if (a.KeysPerLayer.Length != b.KeysPerLayer.Length || a.ValuesPerLayer.Length != b.ValuesPerLayer.Length) return false;
        for (int l = 0; l < a.KeysPerLayer.Length; l++)
        {
            if (!MemoryMarshal.AsBytes(a.KeysPerLayer[l].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(b.KeysPerLayer[l].AsSpan())))
                return false;
            if (!MemoryMarshal.AsBytes(a.ValuesPerLayer[l].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(b.ValuesPerLayer[l].AsSpan())))
                return false;
        }
        return true;
    }

    private static void AssertBitExact(ChunkKvEntry expected, ChunkKvEntry actual)
    {
        Assert.Equal(expected.ModelFingerprint, actual.ModelFingerprint);
        Assert.Equal(expected.TokenIds, actual.TokenIds);
        Assert.Equal(expected.KeysPerLayer.Length, actual.KeysPerLayer.Length);
        Assert.Equal(expected.ValuesPerLayer.Length, actual.ValuesPerLayer.Length);
        Assert.Equal(expected.Backend, actual.Backend);
        Assert.True(SameBits(expected, actual), "K/V bits differ");
    }
}
