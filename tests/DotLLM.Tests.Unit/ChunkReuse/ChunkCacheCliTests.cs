using System.Text;
using System.Text.Json;
using DotLLM.ChunkCache;
using DotLLM.ChunkReuse.Store;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// The <c>chunkcache</c> CLI (Phase S Task 7), run in process (<see cref="ChunkCacheCli.Run"/>) on the synthetic byte-level
/// qwen3 model (<see cref="SyntheticDenseGguf"/> with <c>byteLevelVocab</c>: its GGUF tokenizer encodes one token per byte), on a
/// small generated source directory and on the MiniSolution fixture (<see cref="SolutionScannerTests.MiniSolution"/>). The
/// oracles are independent of the CLI: the store's files, the index files on disk, the library's own
/// <see cref="ChunkIndex.Status"/>, and the model geometry for the size estimate.
/// </summary>
public sealed partial class ChunkCacheCliTests : IClassFixture<ChunkCacheCliTests.Fixture>
{
    private const int ChunkTokens = 64;
    private readonly Fixture _fx;
    private readonly ITestOutputHelper _log;

    public ChunkCacheCliTests(Fixture fx, ITestOutputHelper log)
    {
        _fx = fx;
        _log = log;
    }

    // ------------------------------------------------------------------------------------------------ usage and exit codes

    [Fact]
    public void Usage_ExitCodes()
    {
        Assert.Equal(2, Cli([]).Code);
        Assert.Equal(0, Cli(["--help"]).Code);
        Assert.Equal(2, Cli(["frobnicate"]).Code);
        foreach (string verb in new[] { "build", "select", "status", "evict", "prewarm", "explain" })
        {
            var help = Cli([verb, "--help"]);
            Assert.Equal(0, help.Code);
            Assert.Contains("chunkcache " + verb, help.Out, StringComparison.Ordinal);
            Assert.Equal(2, Cli([verb, "--no-such-option", "1"]).Code);
        }
        string store = NewDir("usage-store");
        Assert.Equal(2, Cli(["build", "--store", store, "--directory", _fx.TreeTemplate]).Code); // no --model
        Assert.Equal(2, Cli(["build", "--model", _fx.ModelA, "--store", store]).Code); // no source
        Assert.Equal(2, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", _fx.TreeTemplate, "--chunker", "lines"]).Code);
        Assert.Equal(2, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", _fx.TreeTemplate, "--max-chunks", "-1"]).Code);
        Assert.Equal(2, Cli(["build", "--model", _fx.ModelA, "--store", store, "--solution", "x.sln", "--directory", _fx.TreeTemplate]).Code);
        Assert.Equal(2, Cli(["select", "--model", _fx.ModelA, "--directory", _fx.TreeTemplate]).Code); // no criterion
        Assert.Equal(2, Cli(["evict", "--store", store, "--all-for-model"]).Code); // needs --model
        Assert.Equal(1, Cli(["build", "--model", Path.Combine(store, "missing.gguf"), "--store", store, "--directory", _fx.TreeTemplate]).Code);
        Assert.Equal(1, Cli(["status", "--model", _fx.ModelA, "--store", Path.Combine(store, "nope")]).Code);
        Assert.Equal(1, Cli(["evict", "--store", Path.Combine(store, "nope")]).Code);
        Assert.Equal(2, Cli(["select", "--model", _fx.ModelA, "--directory", _fx.TreeTemplate, "--path", "src/A.cs,src/B.cs"]).Code); // no comma lists
        Assert.Equal(2, Cli(["evict", "--store", store, "--threads", "4"]).Code); // model options only with --all-for-model
        Assert.Empty(Directory.EnumerateFileSystemEntries(store)); // usage errors write nothing

        // --json: a failure without a result still prints one JSON document on stdout.
        var usage = Cli(["build", "--model", _fx.ModelA, "--store", store, "--json"]);
        Assert.Equal(2, usage.Code);
        JsonElement err = JsonDocument.Parse(usage.Out).RootElement.GetProperty("error");
        Assert.Equal("usage", err.GetProperty("code").GetString());
        Assert.Equal(2, err.GetProperty("exit_code").GetInt32());
        var missing = Cli(["status", "--model", _fx.ModelA, "--store", Path.Combine(store, "nope"), "--json"]);
        Assert.Equal(1, missing.Code);
        Assert.Equal("error", JsonDocument.Parse(missing.Out).RootElement.GetProperty("error").GetProperty("code").GetString());
        Assert.DoesNotContain("   at ", missing.Err, StringComparison.Ordinal); // no stack trace without --verbose
        Assert.Contains("   at ", Cli(["status", "--model", _fx.ModelA, "--store", Path.Combine(store, "nope"), "--verbose"]).Err, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------------------------------------ build

    [Fact]
    public void Build_EncodesEveryMissingChunk_StoreMatchesIndex()
    {
        string tree = NewTree();
        string store = NewDir("store");
        JsonElement r = Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--json"]));
        Assert.Equal(0, r.GetProperty("exit").GetInt32());
        int distinct = r.GetProperty("distinct_keys").GetInt32();
        Assert.True(distinct > 10, $"fixture too small: {distinct} chunks");
        Assert.Equal(distinct, r.GetProperty("encoded_chunks").GetInt32());
        Assert.Equal(0, r.GetProperty("missing_after").GetInt32());
        Assert.Equal(r.GetProperty("distinct_tokens").GetInt64(), r.GetProperty("encoded_tokens").GetInt64());
        // symbol-64 is a lab chunker (only symbol-512 is on the wire): every one of its chunks counts as wire-ineligible.
        Assert.Equal(r.GetProperty("chunks").GetInt32(), r.GetProperty("wire_ineligible_chunks").GetInt32());

        // Independent oracle: the index file on disk, loaded by the library, against the store's files.
        var store2 = new FileChunkStore(store);
        string[] indexes = Directory.GetFiles(store, "index-*.json");
        Assert.Single(indexes);
        Assert.StartsWith("index-symbol-64.", Path.GetFileName(indexes[0]), StringComparison.Ordinal);
        using var model = OpenModel(_fx.ModelA);
        ChunkIndex index = ChunkIndex.Load(indexes[0], model.KeyOf);
        Assert.Null(index.LoadNote);
        Assert.Equal(distinct, index.DistinctKeyCount);
        Assert.Equal(index.ReferencedKeys().OrderBy(k => k.Hex).ToArray(), store2.Keys().OrderBy(k => k.Hex).ToArray());
        Assert.True(index.Status(store2).IsUpToDate);
        // The source and root are recorded for status/select.
        string config = File.ReadAllText(Path.Combine(store, StoreLayout.ConfigFileName));
        Assert.Contains("\"source_kind\": \"directory\"", config, StringComparison.Ordinal);
        Assert.Equal(0, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--strict"]).Code);
    }

    [Fact]
    public void Build_Rerun_IsAllHits()
    {
        string tree = NewTree();
        string store = NewDir("store");
        string[] args = ["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--json"];
        JsonElement first = Json(Cli(args));
        long bytes = new FileChunkStore(store).GetStats().EntryBytes;
        DateTime[] stamps = EntryStamps(store);
        JsonElement second = Json(Cli(args));
        Assert.Equal(0, second.GetProperty("exit").GetInt32());
        Assert.Equal(0, second.GetProperty("added").GetInt32());
        Assert.Equal(0, second.GetProperty("removed").GetInt32());
        Assert.Equal(first.GetProperty("chunks").GetInt32(), second.GetProperty("unchanged").GetInt32());
        Assert.Equal(second.GetProperty("distinct_keys").GetInt32(), second.GetProperty("already_stored").GetInt32());
        Assert.Equal(0, second.GetProperty("to_encode_chunks").GetInt32());
        Assert.Equal(0, second.GetProperty("encoded_chunks").GetInt32());
        Assert.Equal(0L, second.GetProperty("encoded_tokens").GetInt64());
        Assert.Equal(bytes, new FileChunkStore(store).GetStats().EntryBytes);
        Assert.Equal(stamps, EntryStamps(store)); // nothing rewritten
    }

    [Fact]
    public void Build_SkippedProject_ExitsNonZeroOnlyWithStrictFlag()
    {
        string sln = Path.Combine(_fx.Mini.Root, "MiniSolution.slnx");
        string store = NewDir("store-mini");
        // Without --strict: exit 0, the skipped projects are reported, the build proceeds.
        JsonElement lax = Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--solution", sln, "--chunk-tokens", "64", "--json"]));
        Assert.Equal(0, lax.GetProperty("exit").GetInt32());
        string[] skipped = lax.GetProperty("skipped_projects").EnumerateArray().Select(s => s.GetProperty("project").GetString()!).ToArray();
        Assert.Contains("Broken/Broken.csproj", skipped);
        Assert.Contains("Missing/Missing.csproj", skipped);
        Assert.Contains("Tool/Tool.vbproj", skipped);
        Assert.True(lax.GetProperty("encoded_chunks").GetInt32() > 0);

        // With --strict: exit 3 before anything is written.
        string strictStore = NewDir("store-mini-strict");
        var strict = Cli(["build", "--model", _fx.ModelA, "--store", strictStore, "--solution", sln, "--chunk-tokens", "64", "--strict", "--json"]);
        Assert.Equal(3, strict.Code);
        JsonElement sr = JsonDocument.Parse(strict.Out).RootElement;
        Assert.True(sr.GetProperty("strict_failed").GetBoolean());
        Assert.Empty(Directory.EnumerateFileSystemEntries(strictStore));

        // A project that loads cleanly (App and its reference Lib; Lib's extra target framework is informational): --strict passes.
        string appStore = NewDir("store-app-strict");
        JsonElement app = Json(Cli(["build", "--model", _fx.ModelA, "--store", appStore, "--solution", Path.Combine(_fx.Mini.Root, "App", "App.csproj"),
            "--chunk-tokens", "64", "--strict", "--json"]));
        Assert.Equal(0, app.GetProperty("exit").GetInt32());
        Assert.All(app.GetProperty("skipped_projects").EnumerateArray(), s => Assert.False(s.GetProperty("fails_strict").GetBoolean()));
        // App's files reach ../Lib and ../Shared: the root widens to the fixture root and paths stay relative to it.
        Assert.Equal(Path.TrimEndingDirectorySeparator(_fx.Mini.Root), Path.TrimEndingDirectorySeparator(app.GetProperty("root").GetString()!), ignoreCase: true);
    }

    [Fact]
    public void Build_DryRun_WritesNothing_AndEstimatesFromTheModelGeometry()
    {
        string tree = NewTree();
        string store = NewDir("store");
        JsonElement dry = Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--dry-run", "--json"]));
        Assert.Equal(0, dry.GetProperty("exit").GetInt32());
        Assert.True(dry.GetProperty("dry_run").GetBoolean());
        Assert.Empty(Directory.GetFiles(store, "index-*.json"));
        Assert.False(File.Exists(Path.Combine(store, StoreLayout.ConfigFileName)));
        Assert.Equal(0, new FileChunkStore(store).GetStats().Entries);

        // bytes/token/layer from the synthetic geometry: K and V, KV heads x head dim, F32.
        const int perLayer = 2 * SyntheticDenseGguf.Kv * SyntheticDenseGguf.HeadDim * sizeof(float);
        Assert.Equal(perLayer, dry.GetProperty("bytes_per_token_per_layer").GetDouble());
        Assert.Equal(SyntheticDenseGguf.Layers, dry.GetProperty("layers").GetInt32());
        JsonElement[] planned = dry.GetProperty("would_encode").EnumerateArray().ToArray();
        Assert.Equal(dry.GetProperty("distinct_keys").GetInt32(), planned.Length);
        long plannedTokens = planned.Sum(p => (long)p.GetProperty("tokens").GetInt32());
        long estimate = dry.GetProperty("estimated_bytes").GetInt64();
        Assert.InRange(estimate, plannedTokens * perLayer * SyntheticDenseGguf.Layers, plannedTokens * (perLayer * SyntheticDenseGguf.Layers + 4) + planned.Length * 400L);

        // The real build's store matches the estimate exactly (same header size).
        JsonElement real = Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--json"]));
        Assert.Equal(estimate, new FileChunkStore(store).GetStats().EntryBytes);
        Assert.Equal(estimate, real.GetProperty("store_bytes").GetInt64());
    }

    [Fact]
    public void Build_MaxChunks_EncodesAtMostN_StatusListsTheRest()
    {
        string tree = NewTree();
        string store = NewDir("store");
        JsonElement r = Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--max-chunks", "3", "--json"]));
        Assert.Equal(3, r.GetProperty("encoded_chunks").GetInt32());
        Assert.Equal(3, new FileChunkStore(store).GetStats().Entries);
        int distinct = r.GetProperty("distinct_keys").GetInt32();
        var status = Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--strict", "--json", "--list-missing"]);
        Assert.Equal(3, status.Code);
        JsonElement s = JsonDocument.Parse(status.Out).RootElement;
        Assert.False(s.GetProperty("up_to_date").GetBoolean());
        Assert.Equal(s.GetProperty("missing_chunks").GetInt32(), s.GetProperty("missing").GetArrayLength());
        Assert.True(s.GetProperty("missing_chunks").GetInt32() >= distinct - 3);
        Assert.All(s.GetProperty("missing").EnumerateArray(), m => Assert.True(m.GetProperty("start_line").GetInt32() >= 1));

        // A rerun resumes from Status (not from the diff, which is empty now): it finishes the store.
        JsonElement rest = Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--json"]));
        Assert.Equal(0, rest.GetProperty("added").GetInt32());
        Assert.Equal(distinct - 3, rest.GetProperty("encoded_chunks").GetInt32());
        Assert.Equal(0, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--strict"]).Code);
    }

    [Fact]
    public void Build_WithSelection_EncodesOnlySelectedChunks_ButIndexesTheTree()
    {
        string tree = NewTree();
        string store = NewDir("store");
        JsonElement r = Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--path", "src/Calc.cs", "--json"]));
        using var model = OpenModel(_fx.ModelA);
        ChunkIndex index = ChunkIndex.Load(Directory.GetFiles(store, "index-*.json").Single(), model.KeyOf);
        Assert.True(index.Paths.Count >= 3); // the whole tree is indexed
        var calcKeys = index.ChunksOf("src/Calc.cs").Select(c => c.KeyHex!).ToHashSet();
        Assert.Equal(calcKeys.Count, r.GetProperty("encoded_chunks").GetInt32());
        Assert.Equal(calcKeys.OrderBy(k => k), new FileChunkStore(store).Keys().Select(k => k.Hex).OrderBy(k => k));
    }

    [Fact]
    public void Build_OtherTreeRoot_IsRefusedWithoutReroot()
    {
        string store = NewDir("store");
        string a = NewTree(), b = NewTree();
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", a, "--chunk-tokens", "64"]).Code);
        var refused = Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", b, "--chunk-tokens", "64"]);
        Assert.Equal(1, refused.Code);
        Assert.Contains("--reroot", refused.Err, StringComparison.Ordinal);
        JsonElement moved = Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", b, "--chunk-tokens", "64", "--reroot", "--json"]));
        Assert.Equal(0, moved.GetProperty("encoded_chunks").GetInt32()); // same text, same keys: nothing re-encoded
    }

    // ------------------------------------------------------------------------------------------------ status

    [Fact]
    public void Status_AfterEdit_ListsTheStaleFile_AndARebuildEncodesOnlyItsChangedChunks()
    {
        string tree = NewTree();
        string store = NewDir("store");
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        int before = new FileChunkStore(store).GetStats().Entries;

        string calc = Path.Combine(tree, "src", "Calc.cs");
        File.WriteAllText(calc, File.ReadAllText(calc).Replace("return a * b;", "return checked(a * b);", StringComparison.Ordinal));
        File.WriteAllText(Path.Combine(tree, "src", "New.cs"), "class Fresh { int One() => 1; }\n");
        var storedBefore = new FileChunkStore(store).Keys().Select(k => k.Hex).ToHashSet(StringComparer.Ordinal);
        File.Delete(Path.Combine(tree, "src", "Util", "Text.cs"));

        var st = Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--strict", "--json"]);
        Assert.Equal(3, st.Code);
        JsonElement s = JsonDocument.Parse(st.Out).RootElement;
        Assert.Equal(["src/Calc.cs"], s.GetProperty("stale_files").EnumerateArray().Select(e => e.GetString()));
        Assert.Equal(["src/New.cs"], s.GetProperty("new_files").EnumerateArray().Select(e => e.GetString()));
        Assert.Equal(["src/Util/Text.cs"], s.GetProperty("deleted_files").EnumerateArray().Select(e => e.GetString()));
        Assert.Equal(0, s.GetProperty("missing_chunks").GetInt32()); // the store holds every indexed (old) chunk
        // --no-scan sees only the store: up to date.
        Assert.Equal(0, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--no-scan", "--strict"]).Code);

        // Symbol chunks: the rebuild encodes exactly the keys the new index has and the store lacked, and they all come from
        // the edited method's file and the new file (oracle: the index on disk, loaded by the library).
        JsonElement rb = Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--json"]));
        using var model = OpenModel(_fx.ModelA);
        ChunkIndex index = ChunkIndex.Load(Directory.GetFiles(store, "index-*.json").Single(), model.KeyOf);
        var fresh = index.Paths.SelectMany(index.ChunksOf).Where(c => !storedBefore.Contains(c.KeyHex!)).ToList();
        Assert.Equal(fresh.Select(c => c.KeyHex).Distinct(StringComparer.Ordinal).Count(), rb.GetProperty("encoded_chunks").GetInt32());
        Assert.All(fresh, c => Assert.Contains(c.FilePath, new[] { "src/Calc.cs", "src/New.cs" }));
        Assert.Contains(fresh, c => string.Equals(c.FilePath, "src/Calc.cs", StringComparison.Ordinal));
        Assert.True(fresh.Count(c => string.Equals(c.FilePath, "src/Calc.cs", StringComparison.Ordinal)) < index.ChunksOf("src/Calc.cs").Count); // not the whole file
        Assert.Equal(before + rb.GetProperty("encoded_chunks").GetInt32(), new FileChunkStore(store).GetStats().Entries); // a build evicts nothing
        Assert.Equal(0, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--strict"]).Code);
    }

    [Fact]
    public void Status_TextOutput_ListsMissingChunksByFileAndLineRange()
    {
        string tree = NewTree();
        string store = NewDir("store");
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--max-chunks", "0"]).Code);
        var st = Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--show", "1000"]);
        Assert.Equal(0, st.Code);
        Assert.Contains("NOT up to date", st.Out, StringComparison.Ordinal);
        using var model = OpenModel(_fx.ModelA);
        ChunkIndex index = ChunkIndex.Load(Directory.GetFiles(store, "index-*.json").Single(), model.KeyOf);
        foreach (ChunkRef c in index.ChunksOf("src/Shapes.cs"))
            Assert.Contains($"missing  src/Shapes.cs:{c.StartLine}-{c.EndLine} ({c.TokenCount} tokens)", st.Out, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------------------------------------ select

    [Fact]
    public void Select_PathSymbolBudget_AndStoredFlags()
    {
        string tree = NewTree();
        string store = NewDir("store");
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--path", "src/Calc.cs"]).Code);

        // Paths: every chunk of the file, in document order, with the stored flag from the store.
        JsonElement p = Json(Cli(["select", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--path", "src/Calc.cs", "--json"]));
        JsonElement[] chunks = p.GetProperty("chunks").EnumerateArray().ToArray();
        Assert.NotEmpty(chunks);
        Assert.All(chunks, c => Assert.Equal("src/Calc.cs", c.GetProperty("path").GetString()));
        Assert.All(chunks, c => Assert.True(c.GetProperty("stored").GetBoolean()));
        Assert.Equal(chunks.Select(c => c.GetProperty("start_line").GetInt32()).Order(), chunks.Select(c => c.GetProperty("start_line").GetInt32()));
        string calcText = File.ReadAllText(Path.Combine(tree, "src", "Calc.cs")).Replace("\r\n", "\n", StringComparison.Ordinal);
        JsonElement withText = Json(Cli(["select", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--path", "src/Calc.cs", "--include-text", "--json"]));
        string joined = string.Concat(withText.GetProperty("chunks").EnumerateArray().Select(c => c.GetProperty("text").GetString()));
        Assert.Equal(calcText, joined); // symbol chunks tile the file
        JsonElement firstChunk = withText.GetProperty("chunks")[0];
        Assert.Equal("sha256:" + Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(Encoding.UTF8.GetBytes(firstChunk.GetProperty("text").GetString()!))),
            firstChunk.GetProperty("hash").GetString());

        // Symbol neighbourhood: Circle.Area calls Calc.Mul -> Mul's definition, and its caller at hop 1.
        JsonElement sym = Json(Cli(["select", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--symbol", "Calc.Mul", "--hops", "1", "--json"]));
        JsonElement[] sc = sym.GetProperty("chunks").EnumerateArray().ToArray();
        Assert.Contains(sc, c => c.GetProperty("reason").GetString() == "definition" && c.GetProperty("path").GetString() == "src/Calc.cs");
        Assert.Contains(sc, c => c.GetProperty("reason").GetString() == "caller" && c.GetProperty("path").GetString() == "src/Shapes.cs" && c.GetProperty("hop").GetInt32() == 1);
        Assert.Contains(sc, c => c.GetProperty("path").GetString() == "src/Shapes.cs" && !c.GetProperty("stored").GetBoolean());

        // Order: Circle's definition (src/Shapes.cs) ranks first, but its callees in src/Calc.cs come first in document order.
        JsonElement[] rank = Json(Cli(["select", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--symbol", "Circle", "--order", "rank", "--json"]))
            .GetProperty("chunks").EnumerateArray().ToArray();
        JsonElement[] doc = Json(Cli(["select", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--symbol", "Circle", "--json"]))
            .GetProperty("chunks").EnumerateArray().ToArray();
        Assert.Equal("definition", rank[0].GetProperty("reason").GetString());
        Assert.Equal("src/Shapes.cs", rank[0].GetProperty("path").GetString());
        Assert.Equal("src/Calc.cs", doc[0].GetProperty("path").GetString());
        Assert.Equal(rank.Select(c => c.GetProperty("priority").GetInt32()).OrderDescending(), rank.Select(c => c.GetProperty("priority").GetInt32()));
        Assert.Equal(rank.Select(c => c.GetProperty("id").GetString()).Order(StringComparer.Ordinal), doc.Select(c => c.GetProperty("id").GetString()).Order(StringComparer.Ordinal));

        // Budget: strict prefix of the rank, trimmed count reported.
        long all = sym.GetProperty("selected_tokens").GetInt64();
        JsonElement budget = Json(Cli(["select", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--symbol", "Calc.Mul", "--budget", (all - 1).ToString(System.Globalization.CultureInfo.InvariantCulture), "--json", "--order", "rank"]));
        Assert.True(budget.GetProperty("selected_tokens").GetInt64() <= all - 1);
        Assert.True(budget.GetProperty("trimmed").GetInt32() >= 1);
        Assert.Equal("definition", budget.GetProperty("chunks")[0].GetProperty("reason").GetString());

        // Symbol not found: exit 1 with suggestions.
        var missing = Cli(["select", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--symbol", "Calc.Mull"]);
        Assert.Equal(1, missing.Code);
        Assert.Contains("Calc.Mul", missing.Err, StringComparison.Ordinal);
    }

    [Fact]
    public void Select_CrossProjectSymbol_OnASolutionWhoseRootWidens()
    {
        // App.csproj's files reach ../Lib and ../Shared: RelativizeScan widens the root; the graph must still resolve App's use
        // of Lib's Greeter (compilation trees re-keyed to index paths). (Greeter.Greet itself is declared under an inactive #if,
        // which the syntax-only symbol lookup does not see.)
        JsonElement sym = Json(Cli(["select", "--model", _fx.ModelA, "--solution", Path.Combine(_fx.Mini.Root, "App", "App.csproj"), "--chunk-tokens", "64",
            "--symbol", "Greeter", "--hops", "1", "--json"]));
        JsonElement[] sc = sym.GetProperty("chunks").EnumerateArray().ToArray();
        Assert.Contains(sc, c => c.GetProperty("reason").GetString() == "definition" && c.GetProperty("path").GetString() == "Lib/Greeter.cs");
        Assert.Contains(sc, c => c.GetProperty("reason").GetString() == "caller" && c.GetProperty("path").GetString() == "App/Program.cs");
    }

    // ------------------------------------------------------------------------------------------------ evict

    [Fact]
    public void Evict_DryRun_DoesNotDelete_ApplyDeletesOnlyOrphans()
    {
        string tree = NewTree();
        string store = NewDir("store");
        (HashSet<string> orphans, _) = MakeOrphan(tree, store);
        int entries = new FileChunkStore(store).GetStats().Entries;
        var fs = new FileChunkStore(store);

        JsonElement dry = Json(Cli(["evict", "--store", store, "--json"]));
        Assert.True(dry.GetProperty("dry_run").GetBoolean());
        Assert.Equal(orphans.Order(StringComparer.Ordinal), dry.GetProperty("keys").EnumerateArray().Select(k => k.GetString()!));
        Assert.Equal(entries - orphans.Count, dry.GetProperty("kept_referenced").GetInt32());
        Assert.Equal(orphans.Sum(o => new FileInfo(fs.PathOf(new ChunkKey(o))).Length), dry.GetProperty("freed_bytes").GetInt64());
        Assert.Equal(entries, fs.GetStats().Entries); // nothing deleted
        Assert.All(orphans, o => Assert.True(fs.Contains(new ChunkKey(o))));

        JsonElement applied = Json(Cli(["evict", "--store", store, "--apply", "--json"]));
        Assert.False(applied.GetProperty("dry_run").GetBoolean());
        Assert.Equal(orphans.Count, applied.GetProperty("evicted").GetInt32());
        Assert.All(orphans, o => Assert.False(fs.Contains(new ChunkKey(o))));
        Assert.Equal(entries - orphans.Count, fs.GetStats().Entries);
        Assert.Equal(0, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--strict"]).Code);
        Assert.Equal(0, Json(Cli(["evict", "--store", store, "--json"])).GetProperty("evicted").GetInt32());
    }

    [Fact]
    public void Evict_Apply_KeepsEntriesReferencedByAnyIndex()
    {
        string tree = NewTree();
        string store = NewDir("store");
        // Two indexes share the store: symbol and fixed chunks of the same tree.
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--chunker", "fixed"]).Code);
        // ... and a second model's index too.
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelB, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        Assert.Equal(3, Directory.GetFiles(store, "index-*.json").Length);
        int entries = new FileChunkStore(store).GetStats().Entries;

        // Delete the text of a file: its symbol chunks are orphaned in the symbol index; the fixed index still has the old text
        // (it was not rebuilt), so the fixed entries stay referenced, and so do model B's.
        File.Delete(Path.Combine(tree, "src", "Util", "Text.cs"));
        JsonElement rb = Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--json"]));
        Assert.True(rb.GetProperty("removed").GetInt32() > 0);
        JsonElement ev = Json(Cli(["evict", "--store", store, "--apply", "--json"]));
        int evicted = ev.GetProperty("evicted").GetInt32();
        Assert.True(evicted > 0);
        Assert.Equal(entries - evicted, new FileChunkStore(store).GetStats().Entries);
        // Every index is still complete (independent check with the library).
        foreach ((string modelPath, string chunker) in new[] { (_fx.ModelA, "symbol"), (_fx.ModelA, "fixed"), (_fx.ModelB, "symbol") })
            Assert.Equal(0, Cli(["status", "--model", modelPath, "--store", store, "--chunk-tokens", "64", "--chunker", chunker, "--no-scan", "--strict"]).Code);

        // An unreadable index blocks eviction: exit 1, nothing deleted.
        MakeOrphan(tree, store);
        int now = new FileChunkStore(store).GetStats().Entries;
        File.WriteAllText(Path.Combine(store, "index-stray.json"), "{ not json");
        Assert.Equal(1, Cli(["evict", "--store", store, "--apply"]).Code);
        Assert.Equal(now, new FileChunkStore(store).GetStats().Entries);
    }

    [Fact]
    public void Evict_KeepsServerLedgerKeys_AndAnUnreadableLedgerBlocksIt_StatusReportsTheLedger()
    {
        string tree = NewTree();
        string store = NewDir("store");
        (HashSet<string> orphans, _) = MakeOrphan(tree, store);
        var fs = new FileChunkStore(store);
        int entries = fs.GetStats().Entries;
        // The server wrote (and the ledger records) one of the unreferenced entries, in a tenant namespace.
        string serverKey = orphans.Order(StringComparer.Ordinal).First();
        ServerLedger ledger = ServerLedger.Load(store);
        ledger.Record(new ChunkKey(serverKey), "k-0123456789abcdef", "src/Calc.cs", 1, DateTimeOffset.UtcNow);
        using (StoreLockHandle held = ChunkIndex.AcquireStoreLock(store, TimeSpan.FromSeconds(30))) ledger.Save(held);

        JsonElement st = Json(Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--no-scan", "--json"]));
        Assert.Equal("ok", st.GetProperty("ledger").GetProperty("state").GetString());
        Assert.Equal(0, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--no-scan", "--strict"]).Code);
        Assert.Equal(1, st.GetProperty("ledger").GetProperty("rows").GetInt32());
        Assert.DoesNotContain("k-0123456789abcdef", st.ToString(), StringComparison.Ordinal); // no namespace values in output

        JsonElement dry = Json(Cli(["evict", "--store", store, "--json"]));
        string[] expected = orphans.Where(o => o != serverKey).Order(StringComparer.Ordinal).ToArray();
        Assert.Equal(expected, dry.GetProperty("keys").EnumerateArray().Select(k => k.GetString()!));
        Assert.Equal(entries - expected.Length, dry.GetProperty("kept_referenced").GetInt32());
        Assert.Contains("server ledger", Cli(["evict", "--store", store]).Out, StringComparison.Ordinal);

        // An unreadable ledger: status still answers (read-only) and says so; evict refuses, exit 1, nothing deleted.
        string ledgerFile = ServerLedger.DefaultPath(store);
        File.WriteAllText(ledgerFile, """{"formatVersion": 99, "rows": []}""");
        var stText = Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--no-scan"]);
        Assert.Equal(0, stText.Code);
        Assert.Contains("ledger UNREADABLE", stText.Out, StringComparison.Ordinal);
        JsonElement bad = Json(Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--no-scan", "--json"]));
        Assert.Equal("unreadable", bad.GetProperty("ledger").GetProperty("state").GetString());
        Assert.Contains("format version 99", bad.GetProperty("ledger").GetProperty("error").GetString(), StringComparison.Ordinal);
        var refused = Cli(["evict", "--store", store, "--apply"]);
        Assert.Equal(1, refused.Code);
        Assert.Contains(ledgerFile, refused.Err, StringComparison.Ordinal);
        Assert.Equal(1, Cli(["evict", "--store", store]).Code); // the dry run refuses too: it cannot list what it would delete
        var refusedJson = Cli(["evict", "--store", store, "--json"]);
        Assert.Equal(1, refusedJson.Code);
        Assert.Equal("ledger_unreadable", JsonDocument.Parse(refusedJson.Out).RootElement.GetProperty("error").GetProperty("code").GetString());
        // status --strict is not up to date while the ledger is unreadable (CI must notice it).
        Assert.Equal(3, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--no-scan", "--strict"]).Code);
        Assert.Equal(entries, fs.GetStats().Entries);
        Assert.All(orphans, o => Assert.True(fs.Contains(new ChunkKey(o))));

        // No ledger at all: the state is "absent" and the server's entry is an orphan like any other.
        File.Delete(ledgerFile);
        JsonElement none = Json(Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--no-scan", "--json"]));
        Assert.Equal("absent", none.GetProperty("ledger").GetProperty("state").GetString());
        Assert.Equal(orphans.Count, Json(Cli(["evict", "--store", store, "--json"])).GetProperty("evicted").GetInt32());
    }

    [Fact]
    public void Evict_AllForModel_RemovesOnlyThatModelsEntriesAndIndexes()
    {
        string tree = NewTree();
        string store = NewDir("store");
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        int a = new FileChunkStore(store).GetStats().Entries;
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelB, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        int ab = new FileChunkStore(store).GetStats().Entries;
        Assert.Equal(2 * a, ab); // same chunks, other model: other keys

        JsonElement dry = Json(Cli(["evict", "--store", store, "--mode", "all", "--model", _fx.ModelA, "--json"]));
        Assert.Equal(a, dry.GetProperty("evicted").GetInt32());
        Assert.Equal(ab, new FileChunkStore(store).GetStats().Entries);
        Assert.Single(dry.GetProperty("dropped_indexes").EnumerateArray());

        Assert.Equal(0, Cli(["evict", "--store", store, "--mode", "all", "--model", _fx.ModelA, "--apply"]).Code);
        Assert.Equal(ab - a, new FileChunkStore(store).GetStats().Entries);
        Assert.Single(Directory.GetFiles(store, "index-*.json"));
        Assert.Equal(0, Cli(["status", "--model", _fx.ModelB, "--store", store, "--chunk-tokens", "64", "--strict"]).Code);
        Assert.Equal(1, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64"]).Code); // its index is gone
    }

    // ------------------------------------------------------------------------------------------------ the store lock

    [Fact]
    public void Lock_HeldByAnotherWriter_BuildAndEvictApplyFail_DryRunsDoNot()
    {
        string tree = NewTree();
        string store = NewDir("store");
        (HashSet<string> orphans, _) = MakeOrphan(tree, store);
        string index = Directory.GetFiles(store, "index-*.json").Single();
        string[] keysBefore = new FileChunkStore(store).Keys().Select(k => k.Hex).Order(StringComparer.Ordinal).ToArray();
        // Another process's store lock: an exclusive handle on the lock file (not this flow's AcquireStoreLock, which would make
        // the in-process build's index lock a lock-order violation).
        using (new FileStream(Path.Combine(store, ChunkIndex.StoreLockFileName), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None))
        {
            File.WriteAllText(Path.Combine(tree, "src", "Late.cs"), "class Late { }\n");
            var build = Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--lock-timeout", "0.3", "--json"]);
            Assert.Equal(1, build.Code);
            Assert.Equal("store_busy", JsonDocument.Parse(build.Out).RootElement.GetProperty("error").GetProperty("code").GetString());
            // L3: the index-only update does not need the store lock, so the index was published (it lists Late.cs); the
            // registry save and every put do, so nothing was encoded.
            using (var m = OpenModel(_fx.ModelA))
                Assert.Contains("src/Late.cs", ChunkIndex.Load(index, m.KeyOf).Paths);
            Assert.Equal(keysBefore, new FileChunkStore(store).Keys().Select(k => k.Hex).Order(StringComparer.Ordinal).ToArray());
            Assert.Equal(1, Cli(["evict", "--store", store, "--apply", "--lock-timeout", "0.3"]).Code);
            Assert.All(orphans, o => Assert.True(new FileChunkStore(store).Contains(new ChunkKey(o))));
            Assert.Equal(0, Cli(["evict", "--store", store]).Code); // dry run: no lock
            Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--dry-run"]).Code);
        }
        Assert.Equal(0, Cli(["evict", "--store", store, "--apply", "--lock-timeout", "0.3"]).Code);
        Assert.All(orphans, o => Assert.False(new FileChunkStore(store).Contains(new ChunkKey(o))));
    }

    [Fact]
    public void Lock_EvictDuringOfflineBuild_IsRefusedByTheOwnerLock_AndKeepsEveryKey()
    {
        string tree = NewTree();
        string store = NewDir("store");
        (HashSet<string> orphans, string revert) = MakeOrphan(tree, store);
        // Revert the edit: the next build references the orphaned key again and skips encoding it (the store has it). The build
        // holds no store lock here (L3), but an offline build owns the store for its WHOLE run (Task 17): an offline eviction pass
        // right after its index save is refused at once (exit 1, store_owned), not blocked, and every key stays.
        File.WriteAllText(Path.Combine(tree, "src", "Calc.cs"), revert);
        int? evictCode = null;
        string? indexLockHeld = null;
        BuildCommand.TestHookAfterIndexSave = () =>
        {
            evictCode = Cli(["evict", "--store", store, "--apply", "--lock-timeout", "0.2"]).Code;
            // The build holds its index lock here: a second build of the same index would wait.
            string index = Directory.GetFiles(store, "index-*.json").Single();
            indexLockHeld = Record.Exception(() => ChunkIndex.AcquireIndexLockFor(index, TimeSpan.Zero).Dispose())?.GetType().Name;
        };
        try
        {
            Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        }
        finally
        {
            BuildCommand.TestHookAfterIndexSave = null;
        }
        Assert.Equal(1, evictCode); // refused by the owner lock, not blocked by the build
        Assert.Equal(nameof(ChunkStoreLockTimeoutException), indexLockHeld);
        Assert.All(orphans, o => Assert.True(new FileChunkStore(store).Contains(new ChunkKey(o))));
        Assert.Equal(0, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--strict"]).Code);
    }

    [Fact]
    public void Lock_EachPutTakesTheStoreLock_AHeldLockFailsTheBuildFast_StoreBusy()
    {
        string tree = NewTree();
        string store = NewDir("store");
        // Another process holds the store lock (past the timeout) after the build's publication check: the first put waits one
        // timeout and the build stops at once with store_busy (exit 1): no further chunk is encoded or waited for (review S1).
        // The index is published; nothing is half-written.
        FileStream? held = null;
        BuildCommand.TestHookBeforeEncode = () =>
            held = new FileStream(Path.Combine(store, ChunkIndex.StoreLockFileName), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        try
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var build = Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--lock-timeout", "1", "--json"]);
            sw.Stop();
            Assert.Equal(1, build.Code);
            JsonElement err = JsonDocument.Parse(build.Out).RootElement.GetProperty("error");
            Assert.Equal("store_busy", err.GetProperty("code").GetString());
            // More than ten chunks are missing; per-put waiting would take > 10 s. One wait is about 1 s.
            Assert.True(sw.Elapsed < TimeSpan.FromSeconds(6), $"build took {sw.Elapsed.TotalSeconds:F1} s: it waited per put");
            Assert.Empty(new FileChunkStore(store).Keys());
            Assert.Single(Directory.GetFiles(store, "index-*.json"));
        }
        finally
        {
            BuildCommand.TestHookBeforeEncode = null;
            held?.Dispose();
        }
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        Assert.Equal(0, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--strict"]).Code);
    }

    // ------------------------------------------------------------------------------------------------ index names (L9)

    [Fact]
    public void Build_MigratesAPhaseSStore_IndexRenamedOnce_RegistryKeepsRootAndSource()
    {
        string tree = NewTree();
        string store = NewDir("store");
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        using var model = OpenModel(_fx.ModelA);
        string current = Directory.GetFiles(store, "index-*.json").Single();
        Assert.Equal($"index-symbol-64.{ChunkIndex.ModelTag(model.ModelId)}.{model.Producer[..8]}.default.json", Path.GetFileName(current));

        // Turn it into a Phase S store: index-<chunker>.<producer12>.json and a version-1 chunkcache.json.
        string legacyName = $"index-symbol-64.{model.Producer[..12]}.json";
        File.Move(current, Path.Combine(store, legacyName));
        string esc(string s) => s.Replace("\\", "\\\\", StringComparison.Ordinal);
        File.WriteAllText(Path.Combine(store, StoreLayout.ConfigFileName), $$"""
            {
              "format_version": 1,
              "indexes": [
                { "index_file": "{{legacyName}}", "chunker": "symbol-64", "producer": "{{model.Producer}}", "model_fingerprint": "{{esc(model.Fingerprint)}}",
                  "model_id": "{{esc(model.ModelId)}}", "model_path": "{{esc(model.Path)}}", "source_kind": "directory", "source_path": "{{esc(tree)}}",
                  "root": "{{esc(tree)}}", "updated_utc": "2026-11-02T10:00:00Z" }
              ]
            }
            """);
        string[] legacyFiles = ChunkIndex.Load(Path.Combine(store, legacyName), model.KeyOf).Paths.ToArray();
        Assert.NotEmpty(legacyFiles);

        // Read-only verbs read it in place (no rename), and still find the recorded source (no --directory).
        Assert.Equal(0, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--strict"]).Code);
        Assert.True(File.Exists(Path.Combine(store, legacyName)));
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--dry-run"]).Code);
        Assert.True(File.Exists(Path.Combine(store, legacyName)));

        // The next build renames it once (nothing re-encoded) and moves the registry row with the CLI's fields.
        JsonElement r = Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--json"]));
        Assert.Equal(0, r.GetProperty("encoded_chunks").GetInt32());
        Assert.False(File.Exists(Path.Combine(store, legacyName)));
        Assert.True(legacyFiles.SequenceEqual(ChunkIndex.Load(current, model.KeyOf).Paths, StringComparer.Ordinal)); // the same index, at the current name
        Assert.Equal(Path.GetFileName(current), Path.GetFileName(Directory.GetFiles(store, "index-*.json").Single()));
        RegistryEntry row = Assert.Single(StoreRegistry.Load(store).Entries);
        Assert.Equal((Path.GetFileName(current), model.ModelId, tree), (row.File, row.ModelId, row.Tool!["root"]));
        Assert.Equal(0, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--strict"]).Code); // no --directory
    }

    [Fact]
    public void Build_AnotherProducersIndex_IsLeftAlone_ThisProducerWritesItsOwn()
    {
        // Review ruling S3: the producer is in the name, so another producer of this model (e.g. a Vulkan server's index under
        // the Task 4b name, which had no producer segment) is neither refused nor overwritten: the build writes its own file.
        string tree = NewTree();
        string store = NewDir("store");
        using var model = OpenModel(_fx.ModelA);
        string t4bName = Path.Combine(store, $"index-symbol-64.{ChunkIndex.ModelTag(model.ModelId)}.default.json");
        Directory.CreateDirectory(store);
        var other = new ChunkIndex(_ => new ChunkKey(new string('e', 64)));
        other.Save(t4bName);
        byte[] before = File.ReadAllBytes(t4bName);
        var build = Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]);
        Assert.Equal(0, build.Code);
        Assert.Contains("another producer", build.Err, StringComparison.Ordinal); // reported as left in place
        Assert.Equal(before, File.ReadAllBytes(t4bName));
        string mine = Path.Combine(store, $"index-symbol-64.{ChunkIndex.ModelTag(model.ModelId)}.{model.Producer[..8]}.default.json");
        Assert.Null(ChunkIndex.Load(mine, model.KeyOf).LoadNote);
        Assert.Equal(2, Directory.GetFiles(store, "index-*.json").Length);
    }

    [Fact]
    public void Build_NeverOverwritesAProducerPrefixCollision()
    {
        // Only 8 hex of the producer are in the name: an index at this producer's name whose recorded producer differs is
        // refused, never overwritten.
        string tree = NewTree();
        string store = NewDir("store");
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        string index = Directory.GetFiles(store, "index-*.json").Single();
        using var model = OpenModel(_fx.ModelA);
        string collider = model.Producer[..8] + new string(model.Producer[8] == '0' ? '1' : '0', 56);
        var other = new ChunkIndex(_ => new ChunkKey(collider));
        other.Save(index);
        byte[] before = File.ReadAllBytes(index);
        var build = Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]);
        Assert.Equal(1, build.Code);
        Assert.Contains("another producer", build.Err, StringComparison.Ordinal);
        Assert.Equal(before, File.ReadAllBytes(index));
    }

    [Fact]
    public void Lock_IndexLockHeld_BuildIsBusy_DryRunIsNot()
    {
        string tree = NewTree();
        string store = NewDir("store");
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        string index = Directory.GetFiles(store, "index-*.json").Single();
        byte[] before = File.ReadAllBytes(index);
        File.WriteAllText(Path.Combine(tree, "src", "Late.cs"), "class Late { }\n");
        using (ChunkIndex.AcquireIndexLockFor(index, TimeSpan.Zero))
        {
            var build = Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--lock-timeout", "0.3"]);
            Assert.Equal(1, build.Code);
            Assert.Contains("busy", build.Err, StringComparison.Ordinal);
            Assert.Equal(before, File.ReadAllBytes(index));
            Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--dry-run"]).Code);
            // The index lock does not block an eviction pass (it takes only the store lock).
            Assert.Equal(0, Cli(["evict", "--store", store, "--apply", "--lock-timeout", "0.3"]).Code);
        }
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
    }

    [Fact]
    public async Task Lock_ConcurrentBuildAndEvict_LeaveTheStoreComplete()
    {
        string tree = NewTree();
        string store = NewDir("store");
        for (int round = 0; round < 4; round++)
        {
            (_, string revert) = MakeOrphan(tree, store);
            File.WriteAllText(Path.Combine(tree, "src", "Calc.cs"), revert);
            Task<int> build = Task.Run(() => Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--lock-timeout", "60"]).Code);
            Task<int> evict = Task.Run(() => Cli(["evict", "--store", store, "--apply", "--lock-timeout", "60"]).Code);
            int[] codes = await Task.WhenAll(build, evict);
            Assert.Equal(new[] { 0, 0 }, codes);
            Assert.Equal(0, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--strict"]).Code);
        }
    }

    // ------------------------------------------------------------------------------------------------ paths

    [Fact]
    public void Paths_SpacesNonAsciiAndRelativeStore()
    {
        string work = NewDir("work dir");
        string tree = Path.Combine(work, "my src ü 日本");
        CopyTree(_fx.TreeTemplate, tree);
        File.WriteAllText(Path.Combine(tree, "src", "Ünïcødé Name.cs"), "namespace Demo;\npublic static class Überall { public static string Grüß() => \"grüß gott 日本\"; }\n");
        var b = Cli(["build", "--model", Path.GetRelativePath(work, _fx.ModelA), "--store", "rel store ä", "--directory", "my src ü 日本", "--chunk-tokens", "64", "--json"], work);
        JsonElement r = Json(b);
        Assert.Equal(0, r.GetProperty("exit").GetInt32());
        string storeFull = Path.Combine(work, "rel store ä");
        Assert.True(Directory.Exists(storeFull));
        Assert.Equal(Path.GetFullPath(tree), Path.TrimEndingDirectorySeparator(r.GetProperty("root").GetString()!));
        Assert.Equal(0, Cli(["status", "--model", _fx.ModelA, "--store", "rel store ä", "--chunk-tokens", "64", "--strict"], work).Code);
        JsonElement sel = Json(Cli(["select", "--model", _fx.ModelA, "--store", "rel store ä", "--chunk-tokens", "64", "--path", "src/Ü*", "--include-text", "--json"], work));
        JsonElement[] chunks = sel.GetProperty("chunks").EnumerateArray().ToArray();
        Assert.NotEmpty(chunks);
        Assert.All(chunks, c => Assert.Equal("src/Ünïcødé Name.cs", c.GetProperty("path").GetString()));
        Assert.Contains("日本", string.Concat(chunks.Select(c => c.GetProperty("text").GetString())), StringComparison.Ordinal);
        Assert.All(chunks, c => Assert.True(c.GetProperty("stored").GetBoolean()));
    }

    // ------------------------------------------------------------------------------------------------ fix round (review S1-S9)

    [Fact]
    public void Build_Selection_EncodesASelectedChunkWhoseTextAlsoOccursInAnUnselectedEarlierFile()
    {
        // a/Dup.cs and b/Dup.cs are identical: every chunk of b has a twin (same key) earlier in path order. Selecting only b
        // must still encode those keys (the selection is applied before deduplicating by key).
        string tree = NewTree();
        string dup = File.ReadAllText(Path.Combine(tree, "src", "Util", "Text.cs"));
        foreach (string d in new[] { "a", "b" })
        {
            Directory.CreateDirectory(Path.Combine(tree, d));
            File.WriteAllText(Path.Combine(tree, d, "Dup.cs"), dup);
        }
        string store = NewDir("store");
        JsonElement r = Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--path", "b/Dup.cs", "--json"]));
        using var model = OpenModel(_fx.ModelA);
        ChunkIndex index = ChunkIndex.Load(Directory.GetFiles(store, "index-*.json").Single(), model.KeyOf);
        var bKeys = index.ChunksOf("b/Dup.cs").Select(c => c.KeyHex!).ToHashSet(StringComparer.Ordinal);
        Assert.NotEmpty(bKeys);
        Assert.Equal(bKeys, index.ChunksOf("a/Dup.cs").Select(c => c.KeyHex!).ToHashSet(StringComparer.Ordinal)); // fixture: twins
        Assert.Equal(bKeys.Count, r.GetProperty("encoded_chunks").GetInt32());
        var fs = new FileChunkStore(store);
        Assert.All(bKeys, k => Assert.True(fs.Contains(new ChunkKey(k))));
    }

    [Fact]
    public void Evict_AllForModel_RefusesRegistryIndexNamesOutsideTheStore()
    {
        string tree = NewTree();
        string work = NewDir("evil");
        string store = Path.Combine(work, "store");
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        int entries = new FileChunkStore(store).GetStats().Entries;
        string victim = Path.Combine(work, "victim.json");
        string config = Path.Combine(store, StoreLayout.ConfigFileName);
        string original = File.ReadAllText(config);
        string indexName = Path.GetFileName(Directory.GetFiles(store, "index-*.json").Single());
        foreach (string evil in new[] { "..\\\\victim.json", "../victim.json", victim.Replace("\\", "\\\\", StringComparison.Ordinal), "sub\\\\index-x.json", "notindex.json" })
        {
            File.WriteAllText(victim, "keep me");
            File.WriteAllText(config, original.Replace($"\"{indexName}\"", $"\"{evil}\"", StringComparison.Ordinal));
            Assert.Contains(evil, File.ReadAllText(config), StringComparison.Ordinal);
            var r = Cli(["evict", "--store", store, "--mode", "all", "--model", _fx.ModelA, "--apply"]);
            Assert.Equal(1, r.Code);
            Assert.True(File.Exists(victim), $"'{evil}' deleted a file outside the store");
            Assert.Equal(entries, new FileChunkStore(store).GetStats().Entries); // nothing deleted
        }
    }

    [Fact]
    public async Task ChildProcess_StdoutIsUtf8WithoutBom_WithNonAsciiPathsAndText()
    {
        string tree = NewDir("ü tree 日本");
        CopyTree(_fx.TreeTemplate, tree);
        File.WriteAllText(Path.Combine(tree, "src", "Ünïcødé.cs"), "class Überall { string G() => \"grüß gott 日本\"; }\n");
        string exe = Path.Combine(AppContext.BaseDirectory, OperatingSystem.IsWindows() ? "chunkcache.exe" : "chunkcache");
        var psi = new System.Diagnostics.ProcessStartInfo(exe)
        {
            RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, WorkingDirectory = _fx.Scratch,
        };
        foreach (string a in new[] { "select", "--model", _fx.ModelA, "--directory", tree, "--chunk-tokens", "64", "--path", "src/Ü*", "--include-text", "--json" })
            psi.ArgumentList.Add(a);
        using var p = System.Diagnostics.Process.Start(psi)!;
        using var bytes = new MemoryStream();
        Task copy = p.StandardOutput.BaseStream.CopyToAsync(bytes);
        string err = await p.StandardError.ReadToEndAsync();
        await copy;
        await p.WaitForExitAsync();
        Assert.True(p.ExitCode == 0, err);
        byte[] raw = bytes.ToArray();
        Assert.Equal((byte)'{', raw[0]); // no BOM
        string text = new UTF8Encoding(false, throwOnInvalidBytes: true).GetString(raw);
        using JsonDocument doc = JsonDocument.Parse(text);
        JsonElement chunk = doc.RootElement.GetProperty("chunks")[0];
        Assert.Equal("src/Ünïcødé.cs", chunk.GetProperty("path").GetString());
        Assert.Contains("grüß gott 日本", chunk.GetProperty("text").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public void ReadOnlyVerbsAndDryRun_DoNotCreateTheStore()
    {
        string tree = NewTree();
        string parent = NewDir("ro");
        string store = Path.Combine(parent, "typo-store");
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--dry-run"]).Code);
        Assert.Equal(1, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64"]).Code);
        Assert.Equal(1, Cli(["select", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--path", "src/Calc.cs"]).Code);
        Assert.Equal(1, Cli(["evict", "--store", store]).Code);
        Assert.False(Directory.Exists(store));
    }

    [Fact]
    public void Build_StoreWriteFailure_ExitsOne()
    {
        // A damaged entry at a chunk's final path, held open by a reader that does not share delete: the store cannot replace
        // it (ChunkStoreBusyException), the build uses the fresh K/V for nothing and must report the failure.
        string tree = NewTree();
        string store = NewDir("store");
        JsonElement dry = Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--dry-run", "--json"]));
        string key = dry.GetProperty("would_encode")[0].GetProperty("key").GetString()!;
        string path = new FileChunkStore(store).PathOf(new ChunkKey(key));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, [1, 2, 3]);
        using (new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            var r = Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--json"]);
            Assert.Equal(1, r.Code);
            Assert.Equal(1, JsonDocument.Parse(r.Out).RootElement.GetProperty("store_failures").GetInt32());
        }
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code); // repaired
        Assert.Equal(0, Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--strict"]).Code);
    }

    // ------------------------------------------------------------------------------------------------ JSON schemas

    [Fact]
    public void Json_UsesTheMcpToolSchemaNames()
    {
        string tree = NewTree();
        string store = NewDir("store");
        AssertHas(Json(Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64", "--json"])),
            "files", "chunks", "tokens", "wire_ineligible_chunks", "denied_files", "added", "removed", "unchanged", "encode_queued_tokens", "skipped_projects", "elapsed_ms");
        JsonElement sel = Json(Cli(["select", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--symbol", "Calc.Mul", "--include-text", "--json"]));
        AssertHas(sel, "chunks", "selected_tokens", "candidate_tokens", "trimmed", "unmatched_patterns", "notes");
        AssertHas(sel.GetProperty("chunks")[0], "id", "path", "start_line", "end_line", "tokens", "hash", "reason", "detail", "hop", "stored", "text");
        JsonElement st = Json(Cli(["status", "--model", _fx.ModelA, "--store", store, "--chunk-tokens", "64", "--list-missing", "--json"]));
        AssertHas(st, "total_chunks", "missing_chunks", "missing_tokens", "up_to_date", "stale_files", "new_files", "deleted_files", "missing", "store_bytes", "entries");
        AssertHas(Json(Cli(["evict", "--store", store, "--json"])), "evicted", "freed_bytes", "kept_referenced", "dry_run");
    }

    // ------------------------------------------------------------------------------------------------ helpers

    private static void AssertHas(JsonElement e, params string[] names)
    {
        foreach (string n in names) Assert.True(e.TryGetProperty(n, out _), $"missing JSON property '{n}' in {e}");
    }

    /// <summary>
    /// Runs the CLI as the Phase S tests did: a local build/evict now has to say who writes, so this adds <c>--offline</c> to a
    /// <c>build</c> (unless it is a dry run, index-only or server-backed) and to an <c>evict</c>, and <c>--lab</c> to a
    /// <c>build</c> of a chunker that is not wire-eligible (<c>--chunk-tokens</c> other than 512, or <c>--chunker fixed</c>).
    /// The new Task 17 tests call <see cref="CliRaw"/> and spell their flags out.
    /// </summary>
    private (int Code, string Out, string Err) Cli(string[] args, string? cwd = null)
    {
        if (args.Length > 0 && !args.Contains("--help") && !args.Contains("--server"))
        {
            var withFlags = args.ToList();
            if (args[0] == "evict" && !args.Contains("--offline")) withFlags.Add("--offline");
            if (args[0] == "build")
            {
                if (!args.Contains("--offline") && !args.Contains("--index-only") && !args.Contains("--dry-run")) withFlags.Add("--offline");
                int tokens = Array.IndexOf(args, "--chunk-tokens");
                int kind = Array.IndexOf(args, "--chunker");
                bool offWire = (tokens >= 0 && tokens + 1 < args.Length && args[tokens + 1] != "512") || (kind >= 0 && kind + 1 < args.Length && args[kind + 1] == "fixed");
                if (offWire && !args.Contains("--lab")) withFlags.Add("--lab");
            }
            args = withFlags.ToArray();
        }
        return CliRaw(args, cwd);
    }

    private (int Code, string Out, string Err) CliRaw(string[] args, string? cwd = null, HttpMessageHandler? http = null, Func<string, string?>? env = null)
    {
        var o = new StringWriter();
        var e = new StringWriter();
        int code = ChunkCacheCli.Run(args, o, e, cwd ?? _fx.Scratch, http, env ?? (_ => null));
        _log.WriteLine($"> chunkcache {string.Join(' ', args)}  => {code}");
        if (e.ToString().Length > 0) _log.WriteLine(e.ToString());
        return (code, o.ToString(), e.ToString());
    }

    /// <summary>The JSON result with an <c>exit</c> property added (asserts it parses).</summary>
    private static JsonElement Json((int Code, string Out, string Err) r)
    {
        Assert.True(r.Code == 0, $"exit {r.Code}: {r.Err}");
        using JsonDocument doc = JsonDocument.Parse(r.Out);
        var dict = doc.RootElement.EnumerateObject().ToDictionary(p => p.Name, p => (object?)p.Value.Clone());
        dict["exit"] = r.Code;
        return JsonDocument.Parse(JsonSerializer.Serialize(dict)).RootElement.Clone();
    }

    private ModelHandle OpenModel(string path) =>
        ModelHandle.Open(new CliContext(TextWriter.Null, TextWriter.Null, _fx.Scratch), path, fullHash: false, threads: 1);

    private string NewDir(string name)
    {
        string d = Path.Combine(_fx.Scratch, name + "-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(d);
        return d;
    }

    private string NewTree()
    {
        string d = NewDir("tree");
        CopyTree(_fx.TreeTemplate, d);
        return d;
    }

    private static void CopyTree(string from, string to)
    {
        foreach (string f in Directory.EnumerateFiles(from, "*", SearchOption.AllDirectories))
        {
            string t = Path.Combine(to, Path.GetRelativePath(from, f));
            Directory.CreateDirectory(Path.GetDirectoryName(t)!);
            File.Copy(f, t);
        }
    }

    private static DateTime[] EntryStamps(string store) =>
        Directory.EnumerateFiles(store, "*.ckv", SearchOption.AllDirectories).Order(StringComparer.Ordinal).Select(File.GetLastWriteTimeUtc).ToArray();

    /// <summary>
    /// Builds <paramref name="tree"/>, edits one method of <c>src/Calc.cs</c> and builds again: the old chunk's key is now an
    /// orphan (no index references it; with 64-token symbol chunks the edited method may span two). Returns the store's
    /// unreferenced keys and the file's original text.
    /// </summary>
    private (HashSet<string> Orphans, string Original) MakeOrphan(string tree, string store)
    {
        string calc = Path.Combine(tree, "src", "Calc.cs");
        string original = File.ReadAllText(calc);
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        var before = new FileChunkStore(store).Keys().Select(k => k.Hex).ToHashSet();
        string edited = original.Replace("return a + b;", "return a + b + 0;", StringComparison.Ordinal);
        Assert.NotEqual(original, edited);
        File.WriteAllText(calc, edited);
        Assert.Equal(0, Cli(["build", "--model", _fx.ModelA, "--store", store, "--directory", tree, "--chunk-tokens", "64"]).Code);
        using var model = OpenModel(_fx.ModelA);
        var referenced = Directory.GetFiles(store, "index-*.json").SelectMany(f => ChunkIndex.Load(f, model.KeyOf).ReferencedKeys()).Select(k => k.Hex).ToHashSet();
        var orphans = before.Where(k => !referenced.Contains(k)).ToHashSet(StringComparer.Ordinal);
        Assert.NotEmpty(orphans);
        return (orphans, original);
    }

    /// <summary>Scratch directory, two tiny byte-level models, a generated source tree template and the restored MiniSolution.</summary>
    public sealed class Fixture : IAsyncLifetime
    {
        public string Scratch { get; } = Path.Combine(Path.GetTempPath(), "dotllm-chunkcache-" + Guid.NewGuid().ToString("N")[..12]);

        public string ModelA => Path.Combine(Scratch, "tiny-a.gguf");

        public string ModelB => Path.Combine(Scratch, "tiny-b.gguf");

        /// <summary>Like <see cref="ModelA"/> (same seed) but declaring the Qwen3 control tokens (type 3), <c>&lt;think&gt;</c> among them.</summary>
        public string ModelSpecials => Path.Combine(Scratch, "tiny-s.gguf");

        public string TreeTemplate => Path.Combine(Scratch, "template");

        public SolutionScannerTests.MiniSolution Mini { get; } = new();

        public async Task InitializeAsync()
        {
            Directory.CreateDirectory(Scratch);
            File.WriteAllBytes(ModelA, SyntheticDenseGguf.Build("qwen3", seed: 7, byteLevelVocab: true));
            File.WriteAllBytes(ModelB, SyntheticDenseGguf.Build("qwen3", seed: 8, byteLevelVocab: true));
            File.WriteAllBytes(ModelSpecials, SyntheticDenseGguf.Build("qwen3", seed: 7, byteLevelVocab: true, specials: DotLLM.Tests.Unit.Server.ChunkCache.ChunkCacheTestServer.Specials));
            WriteTree(TreeTemplate);
            await Mini.InitializeAsync();
        }

        public async Task DisposeAsync()
        {
            await Mini.DisposeAsync();
            try { Directory.Delete(Scratch, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }

        private static void WriteTree(string root)
        {
            void W(string rel, string text)
            {
                string p = Path.Combine(root, rel);
                Directory.CreateDirectory(Path.GetDirectoryName(p)!);
                File.WriteAllText(p, text.Replace("\r\n", "\n", StringComparison.Ordinal));
            }
            W("src/Calc.cs", """
                namespace Demo;

                /// <summary>Arithmetic helpers.</summary>
                public static class Calc
                {
                    /// <summary>Adds two numbers.</summary>
                    public static int Add(int a, int b)
                    {
                        return a + b;
                    }

                    /// <summary>Multiplies two numbers.</summary>
                    public static double Mul(double a, double b)
                    {
                        return a * b;
                    }

                    /// <summary>Squares a number by multiplying it with itself.</summary>
                    public static double Square(double x) => Mul(x, x);

                    /// <summary>Sums a list of integers one by one, using Add for each element.</summary>
                    public static int Sum(int[] values)
                    {
                        int total = 0;
                        foreach (int v in values) total = Add(total, v);
                        return total;
                    }
                }
                """);
            W("src/Shapes.cs", """
                namespace Demo;

                public sealed class Circle(double radius)
                {
                    public double Radius { get; } = radius;

                    public double Area() => Calc.Mul(System.Math.PI, Calc.Square(Radius));
                }

                public sealed class Rect(double w, double h)
                {
                    public double Area() => Calc.Mul(w, h);

                    public override string ToString() => $"Rect {w} x {h}";
                }
                """);
            W("src/Util/Text.cs", """
                namespace Demo.Util;

                public static class Text
                {
                    public static string Shout(string s) => s.ToUpperInvariant() + "!";

                    public static string Whisper(string s) => s.ToLowerInvariant() + "...";

                    public static int Words(string s) => s.Split(' ', System.StringSplitOptions.RemoveEmptyEntries).Length;
                }
                """);
        }
    }
}
