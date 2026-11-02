using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// <see cref="ChunkIndex"/>: incremental invalidation (an edit re-keys only the chunks whose tokens changed), status against a
/// store, deleted files with shared (deduplicated) entries, and JSON persistence. No model: keys come from token ids and a
/// fixed producer, and the store is an in-memory key set (or a <see cref="FileChunkStore"/> root for the index path).
/// </summary>
public sealed class ChunkIndexTests : IDisposable
{
    private static readonly RopeParameters Rope = new(128, 5_000_000f, RoPEType.NeoX);
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-chunkindex-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best effort */ }
        catch (UnauthorizedAccessException) { }
    }

    private static ChunkKey KeyA(ReadOnlySpan<int> ids) => ChunkKey.Create("fp-a", "tok", Rope, "cpu", KvDtype.F32, ids);
    private static ChunkKey KeyB(ReadOnlySpan<int> ids) => ChunkKey.Create("fp-b", "tok", Rope, "cpu", KvDtype.F32, ids);

    private const string Calc = """
        namespace Demo;

        public sealed class Calc
        {
            public int Add(int a, int b)
            {
                return a + b;
            }

            public int Mul(int a, int b)
            {
                return a * b;
            }

            public int Neg(int a)
            {
                return -a;
            }
        }

        """;

    private const string Other = """
        namespace Demo;

        public static class Other
        {
            public static int One() => 1;

            public static int Two() => 2;
        }

        """;

    private static IReadOnlyList<string> Ids(IEnumerable<ChunkRef> refs) => refs.Select(r => r.Id).Order(StringComparer.Ordinal).ToList();

    // ---------------------------------------------------------------- symbol chunks

    [Fact]
    public void Edit_InsideOneMethod_ChangesOnlyThatSymbolChunk()
    {
        var chunker = new SymbolChunker(new CharTokenizer(), maxChunkTokens: 4096);
        var index = new ChunkIndex(KeyA);
        IndexDiff first = index.Update([new SourceFile("Calc.cs", Calc), new SourceFile("Other.cs", Other)], chunker);
        Assert.Empty(first.Removed);
        Assert.Empty(first.Unchanged);
        int calcChunks = index.ChunksOf("Calc.cs").Count;
        Assert.Equal(3, calcChunks); // one per method (header glued to Add, closing brace to Neg)
        int total = first.Added.Count;

        // Edit Mul's body: one more line, so Neg moves down a line but keeps its tokens.
        string edited = Calc.Replace("return a * b;", "int p = a * b;\n        return p;", StringComparison.Ordinal);
        IndexDiff diff = index.Update([new SourceFile("Calc.cs", edited), new SourceFile("Other.cs", Other)], chunker);

        ChunkRef added = Assert.Single(diff.Added);
        ChunkRef removed = Assert.Single(diff.Removed);
        Assert.Equal("Calc.cs", added.FilePath);
        Assert.Equal(9, added.StartLine);     // the blank line before Mul belongs to Mul's chunk (leading trivia)
        Assert.Equal(14, added.EndLine);
        Assert.Equal(9, removed.StartLine);
        Assert.Equal(13, removed.EndLine);
        Assert.Equal(ChunkKind.Symbol, added.Kind);
        Assert.NotEqual(added.KeyHex, removed.KeyHex);
        Assert.Equal(total - 1, diff.Unchanged.Count);

        // Neg is unchanged (same key) although its line range moved: the unchanged ref carries the new position.
        ChunkRef neg = Assert.Single(diff.Unchanged, r => string.Equals(r.FilePath, "Calc.cs", StringComparison.Ordinal) && r.StartLine > 9);
        Assert.Equal(15, neg.StartLine);
        Assert.Equal(Ids(index.ChunksOf("Calc.cs")), Ids(diff.Unchanged.Where(r => string.Equals(r.FilePath, "Calc.cs", StringComparison.Ordinal)).Append(added)));
        Assert.Equal(ChunkKey.Create("fp-a", "tok", Rope, "cpu", KvDtype.F32, edited[added.StartOffset..added.EndOffset].Select(c => (int)c).ToArray()).Hex,
            added.KeyHex);
    }

    // ---------------------------------------------------------------- fixed chunks

    [Fact]
    public void Edit_ShiftsFixedChunks_FromEditPointOnly()
    {
        var chunker = new FixedSizeChunker(new CharTokenizer(), chunkTokens: 10);
        string text = string.Concat(Enumerable.Range(0, 10).Select(i => new string((char)('a' + i), 9) + "\n")); // 100 chars, 10 chunks
        var index = new ChunkIndex(KeyA);
        index.Update([new SourceFile("F.txt", text)], chunker);
        Assert.Equal(10, index.ChunksOf("F.txt").Count);

        // Insert one char at offset 45 (inside chunk 4): chunks 0-3 keep their tokens, 4.. shift.
        string inserted = text.Insert(45, "X");
        IndexDiff ins = index.Update([new SourceFile("F.txt", inserted)], chunker);
        Assert.Equal(4, ins.Unchanged.Count);
        Assert.All(ins.Unchanged, r => Assert.True(r.EndOffset <= 40));
        Assert.Equal(7, ins.Added.Count);            // 101 chars -> 11 chunks, 7 from offset 40
        Assert.Equal(6, ins.Removed.Count);          // old chunks 4..9
        Assert.All(ins.Added, r => Assert.True(r.StartOffset >= 40));
        Assert.All(ins.Added, r => Assert.Equal(ChunkKind.Fixed, r.Kind));

        // A same-length substitution at offset 45 of the original: only chunk 4 changes, the tail is not shifted.
        index.Update([new SourceFile("F.txt", text)], chunker);
        string substituted = text[..45] + "X" + text[46..];
        IndexDiff sub = index.Update([new SourceFile("F.txt", substituted)], chunker);
        ChunkRef only = Assert.Single(sub.Added);
        Assert.Equal(40, only.StartOffset);
        Assert.Equal(5, only.StartLine);
        Assert.Single(sub.Removed);
        Assert.Equal(9, sub.Unchanged.Count);
    }

    // ---------------------------------------------------------------- status

    [Fact]
    public void Status_ListsExactlyMissingChunks()
    {
        var chunker = new SymbolChunker(new CharTokenizer(), maxChunkTokens: 4096);
        var index = new ChunkIndex(KeyA);
        var files = new[] { new SourceFile("Calc.cs", Calc), new SourceFile("Other.cs", Other) };
        index.Update(files, chunker);
        List<ChunkRef> all = index.ChunksOf("Calc.cs").Concat(index.ChunksOf("Other.cs")).ToList();
        Assert.Equal(5, all.Count);

        // Store holds Calc's first and last chunk and Other's first: missing are Calc's middle (Mul) and Other's second.
        var store = new KeySetStore();
        foreach (ChunkRef r in new[] { all[0], all[2], all[3] }) store.Add(r.Key!.Value);
        StatusReport status = index.Status(store);

        Assert.Equal(5, status.TotalChunks);
        Assert.Equal(2, status.MissingChunks);
        Assert.Equal(all[1].TokenCount!.Value + all[4].TokenCount!.Value, status.MissingTokens);
        Assert.Equal(Ids([all[1], all[4]]), Ids(status.Files.SelectMany(f => f.Missing)));
        FileStatus calc = Assert.Single(status.Files, f => string.Equals(f.Path, "Calc.cs", StringComparison.Ordinal));
        Assert.Equal(FileState.Indexed, calc.State);
        Assert.Equal((9, 13), (calc.Missing[0].StartLine, calc.Missing[0].EndLine));
        Assert.False(status.IsUpToDate);

        // All stored: nothing missing and no file listed.
        store.Add(all[1].Key!.Value);
        store.Add(all[4].Key!.Value);
        Assert.True(index.Status(store).IsUpToDate);
        Assert.Empty(index.Status(store).Files);

        // Against the current tree: an edited file is stale, a new one is new, an absent one is deleted.
        var current = new[] { new SourceFile("Calc.cs", Calc + "// tail\n"), new SourceFile("New.cs", "class N {}\n") };
        StatusReport tree = index.Status(store, current);
        Assert.Equal(FileState.Stale, Assert.Single(tree.Files, f => string.Equals(f.Path, "Calc.cs", StringComparison.Ordinal)).State);
        Assert.Equal(FileState.New, Assert.Single(tree.Files, f => string.Equals(f.Path, "New.cs", StringComparison.Ordinal)).State);
        Assert.Equal(FileState.Deleted, Assert.Single(tree.Files, f => string.Equals(f.Path, "Other.cs", StringComparison.Ordinal)).State);
        Assert.False(tree.IsUpToDate);
    }

    // ---------------------------------------------------------------- deletion and dedup

    [Fact]
    public void DeletedFile_RemovesItsChunks_ButSharedEntriesStay()
    {
        var chunker = new FixedSizeChunker(new CharTokenizer(), chunkTokens: 10);
        var a = new SourceFile("A.txt", "012345678\nAAAAAAAAA\n");
        var b = new SourceFile("B.txt", "012345678\nBBBBBBBBB\n");
        var index = new ChunkIndex(KeyA);
        index.Update([a, b], chunker);
        string shared = index.ChunksOf("A.txt")[0].KeyHex!;
        Assert.Equal(shared, index.ChunksOf("B.txt")[0].KeyHex); // identical text in two files: one key
        Assert.Equal(3, index.DistinctKeyCount);

        var store = new KeySetStore();
        foreach (ChunkRef r in index.ChunksOf("A.txt").Concat(index.ChunksOf("B.txt"))) store.Add(r.Key!.Value);

        IndexDiff diff = index.Update([a], chunker); // B deleted
        Assert.Equal(["B.txt", "B.txt"], diff.Removed.Select(r => r.FilePath));
        Assert.Empty(diff.Added);
        Assert.Equal(2, diff.Unchanged.Count);
        ChunkKey orphan = Assert.Single(diff.OrphanedKeys); // the shared key is still referenced by A
        Assert.Equal(20, b.Text.Length);
        Assert.Equal(ChunkKey.Create("fp-a", "tok", Rope, "cpu", KvDtype.F32, b.Text[10..].Select(c => (int)c).ToArray()), orphan);
        Assert.NotEqual(shared, orphan.Hex);
        Assert.False(index.Contains("B.txt"));
        Assert.Equal(2, index.DistinctKeyCount);

        foreach (ChunkKey k in diff.OrphanedKeys) store.Evict(k);
        Assert.True(index.Status(store).IsUpToDate); // A still resolves to the shared entry

        // An update that names only some files and keeps the rest leaves B's absence alone and A untouched.
        IndexDiff partial = index.Update([b], chunker, removeMissingFiles: false);
        Assert.True(index.Contains("A.txt"));
        Assert.Equal(2, partial.Added.Count); // B is back (its chunks are new to the index)
        Assert.Empty(partial.Removed);
        Assert.Empty(partial.OrphanedKeys);
    }

    // ---------------------------------------------------------------- persistence

    [Fact]
    public void Persist_RoundTrip_AndProducerChangeDiscardsIndex()
    {
        var chunker = new SymbolChunker(new CharTokenizer(), maxChunkTokens: 4096);
        var store = new FileChunkStore(_root);
        string path = ChunkIndex.DefaultPath(store.Root, chunker.Name);
        Assert.Equal(store.Root, Path.GetDirectoryName(path));

        var index = new ChunkIndex(KeyA);
        var files = new[] { new SourceFile("Calc.cs", Calc), new SourceFile("Other.cs", Other) };
        index.Update(files, chunker);
        index.Save(path);
        Assert.Empty(store.Keys()); // the index file is not mistaken for an entry

        ChunkIndex loaded = ChunkIndex.Load(path, KeyA);
        Assert.Null(loaded.LoadNote);
        Assert.Equal(chunker.Name, loaded.ChunkerName);
        Assert.Equal(index.ChunksOf("Calc.cs"), loaded.ChunksOf("Calc.cs"));
        Assert.Equal(index.ChunksOf("Other.cs"), loaded.ChunksOf("Other.cs"));
        IndexDiff again = loaded.Update(files, chunker); // unchanged text: nothing re-keyed
        Assert.Empty(again.Added);
        Assert.Empty(again.Removed);
        Assert.Equal(5, again.Unchanged.Count);

        // Another producer (model fingerprint): the keys are meaningless, so the index loads empty with a note.
        ChunkIndex other = ChunkIndex.Load(path, KeyB);
        Assert.NotNull(other.LoadNote);
        Assert.Empty(other.Paths);
        Assert.Equal(5, other.Update(files, chunker).Added.Count);

        // A chunker change re-chunks every file even when the text is the same.
        IndexDiff rechunked = loaded.Update(files, new FixedSizeChunker(new CharTokenizer(), 16));
        Assert.Empty(rechunked.Unchanged);
        Assert.All(rechunked.Added, r => Assert.Equal(ChunkKind.Fixed, r.Kind));

        // Missing file: empty index, no note; corrupt file: empty index with a note.
        Assert.Null(ChunkIndex.Load(Path.Combine(_root, "absent.json"), KeyA).LoadNote);
        File.WriteAllText(path, "{ not json");
        Assert.NotNull(ChunkIndex.Load(path, KeyA).LoadNote);
    }

    // ---------------------------------------------------------------- review fixes (S3-S7, N1)

    [Fact]
    public void Edit_RepeatedBlockInOneFile_MatchesAsMultiset()
    {
        var chunker = new FixedSizeChunker(new CharTokenizer(), chunkTokens: 10);
        const string A = "aaaaaaaaa\n", B = "bbbbbbbbb\n";
        var index = new ChunkIndex(KeyA);
        index.Update([new SourceFile("R.txt", A + A + A + B)], chunker);
        Assert.Equal(2, index.DistinctKeyCount);

        // One more copy of the repeated block: exactly one chunk is new to the file, although its key was already there.
        IndexDiff more = index.Update([new SourceFile("R.txt", A + A + A + A + B)], chunker);
        ChunkRef added = Assert.Single(more.Added);
        Assert.Equal(KeyA(A.Select(c => (int)c).ToArray()).Hex, added.KeyHex);
        Assert.Equal(4, more.Unchanged.Count);
        Assert.Empty(more.Removed);

        // And one copy fewer than at the start: two of the four go, no orphan (the key is still used twice).
        IndexDiff fewer = index.Update([new SourceFile("R.txt", A + A + B)], chunker);
        Assert.Equal(2, fewer.Removed.Count);
        Assert.Empty(fewer.Added);
        Assert.Empty(fewer.OrphanedKeys);
    }

    [Fact]
    public void MoveAcrossFiles_InOneUpdate_IsAddedButNeedsNoEncodingAndIsNotOrphaned()
    {
        var chunker = new FixedSizeChunker(new CharTokenizer(), chunkTokens: 10);
        const string X = "XXXXXXXXX\n", A = "AAAAAAAAA\n", B = "BBBBBBBBB\n", Z = "ZZZZZZZZZ\n";
        var index = new ChunkIndex(KeyA);
        IndexDiff first = index.Update([new SourceFile("A.txt", X + A), new SourceFile("B.txt", B)], chunker);
        var store = new KeySetStore();
        foreach (ChunkRef r in first.Added) store.Add(r.Key!.Value);

        // X moves from A to B, and both files gain the same new block Z.
        IndexDiff move = index.Update([new SourceFile("A.txt", A + Z), new SourceFile("B.txt", B + X + Z)], chunker);
        Assert.Contains(move.Added, r => string.Equals(r.FilePath, "B.txt", StringComparison.Ordinal)
            && string.Equals(r.KeyHex, KeyA(X.Select(c => (int)c).ToArray()).Hex, StringComparison.Ordinal));
        Assert.Empty(move.OrphanedKeys); // X is still referenced (by B)
        ChunkRef need = Assert.Single(move.NeedsEncoding(store)); // X is stored; Z is needed once, not twice
        Assert.Equal(KeyA(Z.Select(c => (int)c).ToArray()).Hex, need.KeyHex);
        Assert.Equal(10, need.TokenCount);
    }

    [Fact]
    public void EvictableOrphans_RespectOtherIndexesUnderTheStore()
    {
        var store = new FileChunkStore(_root);
        var symbol = new SymbolChunker(new CharTokenizer(), maxChunkTokens: 4096);
        var fixedChunker = new FixedSizeChunker(new CharTokenizer(), chunkTokens: 4096);
        var file = new SourceFile("S.cs", "class C { }\n"); // one chunk under both chunkers: the same tokens, the same key
        var symIndex = new ChunkIndex(KeyA);
        var fixIndex = new ChunkIndex(KeyA);
        symIndex.Update([file], symbol);
        fixIndex.Update([file], fixedChunker);
        Assert.Equal(symIndex.ChunksOf("S.cs")[0].KeyHex, fixIndex.ChunksOf("S.cs")[0].KeyHex);
        symIndex.Save(ChunkIndex.DefaultPath(store.Root, symbol.Name));
        fixIndex.Save(ChunkIndex.DefaultPath(store.Root, fixedChunker.Name));

        IndexDiff deleted = symIndex.Update([], symbol);
        ChunkKey orphan = Assert.Single(deleted.OrphanedKeys); // orphaned within the symbol index...
        Assert.Contains(orphan, ChunkIndex.ReferencedKeysUnder(store.Root));
        Assert.Empty(symIndex.EvictableOrphans(deleted, store.Root)); // ...but the fixed index still uses it

        // The symbol index's own (stale) file does not protect the key; another index's file does.
        File.Delete(ChunkIndex.DefaultPath(store.Root, fixedChunker.Name));
        Assert.Equal([orphan], symIndex.EvictableOrphans(deleted, store.Root));

        // An unreadable index under the store: eviction cannot be decided, so it throws rather than guess.
        File.WriteAllText(Path.Combine(store.Root, "index-other-1.json"), "{ not json");
        Assert.Throws<InvalidDataException>(() => symIndex.EvictableOrphans(deleted, store.Root));
    }

    public static TheoryData<string> BadIndexFiles() => new()
    {
        "version", "file-null", "path-null", "sha-bad", "chunks-null", "chunk-null", "key-bad", "duplicate-path",
        "offset-inverted", "offset-gap", "line-inverted", "line-zero", "tokens-negative", "chunker-null", "path-rooted",
    };

    [Theory]
    [MemberData(nameof(BadIndexFiles))]
    public void Load_MalformedOrOtherVersion_IsEmptyWithNote(string defect)
    {
        Directory.CreateDirectory(_root);
        string path = Path.Combine(_root, "index-fixed-10.json");
        File.WriteAllText(path, IndexJson("valid"));
        ChunkIndex ok = ChunkIndex.Load(path, KeyA);
        Assert.Null(ok.LoadNote);
        Assert.Equal(2, ok.ChunksOf("A.txt").Count);

        File.WriteAllText(path, IndexJson(defect));
        ChunkIndex bad = ChunkIndex.Load(path, KeyA);
        Assert.NotNull(bad.LoadNote);
        Assert.Empty(bad.Paths);
        Assert.Equal(0, bad.DistinctKeyCount);
    }

    private static string IndexJson(string defect)
    {
        string producer = KeyA([]).Hex;
        string k1 = KeyA([1]).Hex, k2 = KeyA([2]).Hex, sha = new string('a', 64);
        string c1 = defect switch
        {
            "offset-inverted" => $$"""{"startLine":1,"endLine":1,"startOffset":0,"endOffset":0,"tokens":10,"key":"{{k1}}"}""",
            "line-inverted" => $$"""{"startLine":2,"endLine":1,"startOffset":0,"endOffset":10,"tokens":10,"key":"{{k1}}"}""",
            "line-zero" => $$"""{"startLine":0,"endLine":1,"startOffset":0,"endOffset":10,"tokens":10,"key":"{{k1}}"}""",
            "tokens-negative" => $$"""{"startLine":1,"endLine":1,"startOffset":0,"endOffset":10,"tokens":-1,"key":"{{k1}}"}""",
            "key-bad" => """{"startLine":1,"endLine":1,"startOffset":0,"endOffset":10,"tokens":10,"key":"XYZ"}""",
            "chunk-null" => "null",
            _ => $$"""{"startLine":1,"endLine":1,"startOffset":0,"endOffset":10,"tokens":10,"key":"{{k1}}"}""",
        };
        string c2 = defect is "offset-gap"
            ? $$"""{"startLine":2,"endLine":2,"startOffset":11,"endOffset":20,"tokens":10,"key":"{{k2}}"}"""
            : $$"""{"startLine":2,"endLine":2,"startOffset":10,"endOffset":20,"tokens":10,"key":"{{k2}}"}""";
        string chunks = defect is "chunks-null" ? "null" : $"[{c1},{c2}]";
        string filePath = defect switch { "path-null" => "null", "path-rooted" => "\"/abs/A.txt\"", _ => "\"A.txt\"" };
        string file = $$"""{"path":{{filePath}},"sha256":"{{(defect is "sha-bad" ? "nothex" : sha)}}","chunks":{{chunks}}}""";
        string files = defect switch
        {
            "file-null" => $"[{file},null]",
            "duplicate-path" => $$"""[{{file}},{"path":"A.txt","sha256":"{{sha}}","chunks":[]}]""",
            _ => $"[{file}]",
        };
        string chunker = defect is "chunker-null" ? "null" : "\"fixed-10\"";
        int version = defect is "version" ? ChunkIndex.FormatVersion + 1 : ChunkIndex.FormatVersion;
        return $$"""{"formatVersion":{{version}},"producer":"{{producer}}","chunker":{{chunker}},"files":{{files}}}""";
    }

    [Fact]
    public async Task Save_WhileAReaderHoldsTheIndex_Succeeds()
    {
        var chunker = new FixedSizeChunker(new CharTokenizer(), chunkTokens: 10);
        Directory.CreateDirectory(_root);
        string path = Path.Combine(_root, "index-fixed-10.json");
        var index = new ChunkIndex(KeyA);
        index.Update([new SourceFile("A.txt", "012345678\n")], chunker);
        index.Save(path);

        index.Update([new SourceFile("A.txt", "012345678\nmore\n")], chunker);
        Task save;
        using (FileStream reader = ChunkIndex.OpenForRead(path)) // the share mode Load uses
        {
            save = Task.Run(() => index.Save(path)); // waits while the reader holds the file
            await Task.Delay(150);
            Assert.Single(ChunkIndex.Load(path, KeyA).ChunksOf("A.txt")); // a second reader still sees the old file
        }
        await save.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(2, ChunkIndex.Load(path, KeyA).ChunksOf("A.txt").Count);
        Assert.Single(Directory.GetFiles(_root)); // no temp file left

        // A reader that never lets go: Save gives up with an error and the old file stays intact.
        TimeSpan budget = ChunkIndex.SaveRetryBudget;
        ChunkIndex.SaveRetryBudget = TimeSpan.FromMilliseconds(100);
        try
        {
            using FileStream stuck = ChunkIndex.OpenForRead(path);
            index.Update([new SourceFile("A.txt", "x\n")], chunker);
            if (OperatingSystem.IsWindows()) Assert.ThrowsAny<IOException>(() => index.Save(path));
            else index.Save(path); // POSIX rename replaces an open file; the reader keeps the old inode
        }
        finally
        {
            ChunkIndex.SaveRetryBudget = budget;
        }
        if (OperatingSystem.IsWindows()) Assert.Equal(2, ChunkIndex.Load(path, KeyA).ChunksOf("A.txt").Count);
        Assert.Single(Directory.GetFiles(_root));
    }

    [Fact]
    public void Paths_AreNormalised()
    {
        var chunker = new FixedSizeChunker(new CharTokenizer(), chunkTokens: 10);
        var index = new ChunkIndex(KeyA);
        index.Update([new SourceFile("src/A.cs", "0123456789")], chunker);

        IndexDiff again = index.Update([new SourceFile("src\\A.cs", "0123456789"), new SourceFile("./src/B.cs", "b")], chunker,
            removeMissingFiles: false);
        Assert.Empty(again.Removed);
        Assert.Equal(["src/A.cs", "src/B.cs"], index.Paths);
        Assert.Equal("src/B.cs", Assert.Single(again.Added).FilePath);
        Assert.Single(index.ChunksOf("src\\A.cs"));
        Assert.Equal("src/A.cs", ChunkIndex.NormalizePath(".\\src\\A.cs"));

        if (OperatingSystem.IsWindows())
        {
            IndexDiff cased = index.Update([new SourceFile("SRC/a.cs", "0123456789")], chunker, removeMissingFiles: false);
            Assert.Empty(cased.Added);
            Assert.Equal(2, index.Paths.Count);
            Assert.True(index.Contains("Src/A.CS"));
        }

        Assert.Throws<ArgumentException>(() => index.Update([new SourceFile("/abs/A.cs", "x")], chunker, removeMissingFiles: false));
        Assert.Throws<ArgumentException>(() => index.Update([new SourceFile("C:/abs/A.cs", "x")], chunker, removeMissingFiles: false));
        Assert.Throws<ArgumentException>(() => index.Update([new SourceFile("src/../A.cs", "x")], chunker, removeMissingFiles: false));
        Assert.Throws<ArgumentException>(() => index.Update([new SourceFile("src/A.cs", "x"), new SourceFile("src\\A.cs", "x")], chunker));
    }

    [Fact]
    public void ChunkRef_Json_RoundTrips_WithKindAsNameAndNoComputedProperties()
    {
        ChunkRef[] refs =
        [
            new("src/A.cs", 3, 9, 40, 180, ChunkKind.Symbol) { TokenCount = 37, KeyHex = KeyA([1, 2]).Hex },
            new("src/B.cs", 1, 4, 0, 60, ChunkKind.Fixed), // selection made before tokenisation: no count, no key
        ];
        string json = System.Text.Json.JsonSerializer.Serialize(refs.ToList(), ChunkRefJsonContext.Default.ListChunkRef);
        Assert.Contains("\"kind\":\"Symbol\"", json, StringComparison.Ordinal);
        Assert.DoesNotContain("\"id\"", json, StringComparison.Ordinal);
        Assert.DoesNotContain("\"key\"", json, StringComparison.Ordinal);
        List<ChunkRef> back = System.Text.Json.JsonSerializer.Deserialize(json, ChunkRefJsonContext.Default.ListChunkRef)!;
        Assert.Equal(refs, back);
    }

    [Fact]
    public void KindOf_UnknownChunker_IsOther_AndIndexes()
    {
        var lines = new LineChunker();
        Assert.Equal(ChunkKind.Other, ChunkRef.KindOf(lines));
        var index = new ChunkIndex(KeyA);
        IndexDiff diff = index.Update([new SourceFile("A.txt", "one\ntwo\n")], lines);
        Assert.Equal(2, diff.Added.Count);
        Assert.All(diff.Added, r => Assert.Equal(ChunkKind.Other, r.Kind));
        string path = Path.Combine(_root, "index-lines.json");
        index.Save(path);
        ChunkIndex loaded = ChunkIndex.Load(path, KeyA);
        Assert.Null(loaded.LoadNote);
        Assert.Equal(index.ChunksOf("A.txt"), loaded.ChunksOf("A.txt"));
    }

    /// <summary>One chunk per line (a chunker that is neither symbol nor fixed).</summary>
    private sealed class LineChunker : IChunker
    {
        public string Name => "lines";
        public int MaxChunkTokens => int.MaxValue;
        public IReadOnlyList<Chunk> Chunk(SourceFile file)
        {
            var chunks = new List<Chunk>();
            int start = 0, line = 1;
            while (start < file.Text.Length)
            {
                int end = file.Text.IndexOf('\n', start);
                end = end < 0 ? file.Text.Length : end + 1;
                chunks.Add(new Chunk(file.Path, line, line, file.Text[start..end].Select(c => (int)c).ToArray()) { StartOffset = start, EndOffset = end });
                start = end;
                line++;
            }
            return chunks;
        }
    }

    /// <summary>In-memory store of keys only (status needs <see cref="IChunkStore.Contains"/>).</summary>
    private sealed class KeySetStore : IChunkStore
    {
        private readonly HashSet<ChunkKey> _keys = [];
        public void Add(ChunkKey key) => _keys.Add(key);
        public bool TryGet(ChunkKey key, out ChunkKvEntry entry) { entry = null!; return false; }
        public void Put(ChunkKey key, ChunkKvEntry entry) => _keys.Add(key);
        public bool Contains(ChunkKey key) => _keys.Contains(key);
        public IEnumerable<ChunkKey> Keys() => _keys;
        public long TotalBytes => 0;
        public void Evict(ChunkKey key) => _keys.Remove(key);
    }
}
