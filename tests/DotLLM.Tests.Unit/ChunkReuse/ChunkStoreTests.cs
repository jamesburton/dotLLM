using System.Runtime.InteropServices;
using DotLLM.ChunkReuse;
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
    private const string Engine = "engine-1";
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
    [InlineData("engine")]
    [InlineData("backend")]
    [InlineData("dtype")]
    [InlineData("tokens")]
    public void Key_DiffersWhenModelTokenizerRopeOrEngineVersionChanges(string field)
    {
        ChunkKey baseKey = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
        ChunkKey changed = field switch
        {
            "model" => ChunkKey.Create("fp-model-b", Tokenizer, Rope, Engine, Backend, Dtype, Tokens),
            "tokenizer" => ChunkKey.Create(Fingerprint, "tok-b", Rope, Engine, Backend, Dtype, Tokens),
            "rope-dim" => ChunkKey.Create(Fingerprint, Tokenizer, Rope with { RopeDim = 64 }, Engine, Backend, Dtype, Tokens),
            "rope-theta" => ChunkKey.Create(Fingerprint, Tokenizer, Rope with { Theta = 1_000_000f }, Engine, Backend, Dtype, Tokens),
            "rope-type" => ChunkKey.Create(Fingerprint, Tokenizer, Rope with { Type = RoPEType.Norm }, Engine, Backend, Dtype, Tokens),
            "engine" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, "engine-2", Backend, Dtype, Tokens),
            "backend" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, "vulkan", Dtype, Tokens),
            "dtype" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, KvDtype.F16, Tokens),
            "tokens" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, [3, 17, 5, 29, 11, 40, 9]),
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
        ChunkKey a = ChunkKey.Create("ab", "c", Rope, Engine, Backend, Dtype, Tokens);
        ChunkKey b = ChunkKey.Create("a", "bc", Rope, Engine, Backend, Dtype, Tokens);
        Assert.NotEqual(a, b);
        // The token count is encoded: a trailing zero token is not the same chunk.
        Assert.NotEqual(ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, [1, 0]),
                        ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, [1]));
    }

    [Fact]
    public void Key_SameInputs_SameKey()
    {
        ChunkKey a = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
        ChunkKey b = ChunkKey.Create(new string(Fingerprint.AsSpan()), new string(Tokenizer.AsSpan()),
            new RopeParameters(128, 5_000_000f, RoPEType.NeoX), new string(Engine.AsSpan()),
            new string(Backend.AsSpan()), KvDtype.F32, (int[])Tokens.Clone());
        Assert.Equal(a, b);
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

        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
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
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
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
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
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
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
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
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
        ChunkKey other = ChunkKey.Create(Fingerprint, Tokenizer, Rope, "engine-2", Backend, Dtype, Tokens);
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
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);

        // Intra-process only: 16 writers released together, 20 rounds each, interleaved with readers.
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
        Assert.Equal(0, badReads); // a reader may miss mid-write, but never sees a torn entry
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
        ChunkKey a = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, fromFileA);
        ChunkKey b = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, fromFileB);
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
        ChunkKey c = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, other);
        store.Put(c, NewEntry(other, Fingerprint, layers: 2, stride: 4, seed: 3));
        Assert.Equal(2, store.Keys().Count());
        Assert.True(store.TotalBytes > afterFirst);
    }

    [Fact]
    public void Store_PutExistingValidKey_KeepsExistingFile()
    {
        // Content-addressed: a valid entry under a key is never rewritten (a rewrite could fail on Windows while a reader has
        // the file mapped, and is wasted I/O).
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
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
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
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
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
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

    [Fact]
    public void Store_Evict_RemovesEntry_AndMissingEvictIsNoOp()
    {
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
        ChunkKey other = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, [1, 2, 3]);
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
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
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
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Engine, Backend, Dtype, Tokens);
        var store = new FileChunkStore(_root);
        // K/V layer count mismatch and a row count that is not a multiple of the token count.
        Assert.Throws<ArgumentException>(() => store.Put(key,
            new ChunkKvEntry(Fingerprint, Tokens, [new float[Tokens.Length]], [new float[Tokens.Length], new float[Tokens.Length]])));
        Assert.Throws<ArgumentException>(() => store.Put(key,
            new ChunkKvEntry(Fingerprint, Tokens, [new float[Tokens.Length + 1]], [new float[Tokens.Length]])));
        Assert.False(store.Contains(key));
    }

    // ---------------------------------------------------------------- helpers

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
        Assert.True(SameBits(expected, actual), "K/V bits differ");
    }
}
