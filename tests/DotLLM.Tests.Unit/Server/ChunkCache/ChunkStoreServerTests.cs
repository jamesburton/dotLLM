using System.Diagnostics;
using System.Text;
using System.Text.Json.Nodes;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.ChunkCache;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// The sole writer (Phase I Task 15) in a real server over HTTP on the synthetic model (CPU): what it stores is exactly what
/// the <c>chunkcache</c> CLI would (same tokenisation, isolated encode, fingerprint and backend tag), annotated requests are
/// served exactly as unannotated ones while their misses are queued, startup ownership and the unreadable-ledger rule, and
/// that no request thread waits for the store lock.
/// </summary>
[Collection(ChunkCacheServerCollection.Name)]
public sealed class ChunkStoreServerTests : IDisposable
{
    private const string Key = "test-key-alpha-0123456789abcdef0123";

    private static readonly string ChunkA =
        "public static int Add(int a, int b)\n{\n    return a + b; // the sum of both operands\n}\n";

    private static readonly string User = "Read this file.\n// A.cs\n" + ChunkA + "What does Add return?";

    private static readonly (string Role, string Content)[] Conversation = [("system", "You are terse."), ("user", User)];

    private static readonly TimeSpan Drain = TimeSpan.FromSeconds(60);

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunk-writer-srv-" + Guid.NewGuid().ToString("N")[..12]);
    private readonly string _model;

    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public ChunkStoreServerTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
        _model = ChunkCacheTestServer.WriteModel(Path.Combine(_scratch, "tiny.gguf"));
    }

    /// <summary>
    /// Review S2: a server shut down while the worker is inside an encode does not free the model under it (it is left
    /// for the process exit), while a normal shutdown frees it.
    /// </summary>
    [Fact]
    public async Task ServerDispose_WorkerStillEncoding_ModelNotFreed()
    {
        string root = NewRoot();
        var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        var state = server.State;
        state.ChunkCacheRuntime.Dispose(); // replace the real worker by a slow one
        var slow = new ChunkStoreServiceTests.FakeHost { Block = new ManualResetEventSlim(false) };
        var writer = state.ChunkCacheRuntime.StartWriter(state, slow)!;
        writer.Queue.QuietPeriod = TimeSpan.Zero;
        writer.Queue.ShutdownTimeout = TimeSpan.FromMilliseconds(200);
        Assert.True(writer.Admit(ChunkStoreServiceTests.Span(ChunkA), ChunkStoreServiceTests.KeyFor(ChunkA), 80, new ChunkCaller(true, ""), true, "cpu").Queued);
        Assert.True(slow.Entered.Wait(TimeSpan.FromSeconds(10)));
        await server.DisposeAsync();
        Assert.True(state.ModelLeftForProcessExit);
        slow.Block.Set();
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
        while (!writer.WorkerStopped && DateTime.UtcNow < deadline) await Task.Delay(20);
        Assert.True(writer.WorkerStopped);

        var normal = await ChunkCacheTestServer.StartAsync(_model, Writing(NewRoot()));
        var normalState = normal.State;
        await normal.DisposeAsync();
        Assert.False(normalState.ModelLeftForProcessExit);
    }

    /// <summary>
    /// Review N8: an item whose key was computed for another model (a swap between admission and encode) is dropped by
    /// the worker's key re-check, never stored under the new model.
    /// </summary>
    [Fact]
    public async Task ModelSwap_ItemWithStaleKeyDropped()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(NewRoot()));
        var state = server.State;
        var (key, ids) = KeyOfChunkA(server);
        var host = new ServerChunkEncodeHost(state);
        var item = new ChunkEncodeItem(ChunkNamespace.Default, key, ChunkA, ids.Length, null);
        Assert.NotNull(host.Encode(item));
        var current = state.ChunkCacheRuntime.IdentityFor(state);
        state.ChunkCacheRuntime.IdentityOverride = ServerModelIdentity.Create(state.Config!, "another-model-id", current.Quantization, current.TokenizerId, current.Backend);
        try
        {
            Assert.Null(host.Encode(item));
        }
        finally
        {
            state.ChunkCacheRuntime.IdentityOverride = null;
        }
    }

    /// <summary>
    /// Review S4: the quiet-period clock moves with every request (scheduled or direct), not with the worker's own lease, and
    /// the busy probe sees a held gate.
    /// </summary>
    [Fact]
    public async Task ActivityStamp_MovesWithRequests_NotWithTheWorkersLease()
    {
        foreach (bool paged in new[] { true, false })
        {
            await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(NewRoot()), paged: paged);
            var state = server.State;
            long before = state.ActivityStamp();
            var (status, _) = await server.PostAsync("/v1/chat/completions", Body(null));
            Assert.Equal(200, status);
            long after = state.ActivityStamp();
            Assert.NotEqual(before, after);
            Assert.False(state.IsBusyForBackgroundWork());
            using (var lease = state.TryAcquireIdleModel())
            {
                Assert.NotNull(lease);
                Assert.True(state.IsBusyForBackgroundWork());
            }
            Assert.Equal(after, state.ActivityStamp());
        }
    }

    /// <summary>Review N6: the writer's counters and queue gauge are on <c>/metrics</c>, with fixed label sets.</summary>
    [Fact]
    public async Task WriterMetrics_Exported()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(NewRoot()));
        static long Value(string text, string series)
        {
            string? line = text.Split('\n').FirstOrDefault(l => l.StartsWith(series + " ", StringComparison.Ordinal));
            return line is null ? 0 : (long)double.Parse(line[(series.Length + 1)..], System.Globalization.CultureInfo.InvariantCulture);
        }
        const string Stored = "dotllm_chunk_store_writes_total{result=\"stored\"}";
        const string SeenOnce = "dotllm_chunk_store_skips_total{reason=\"not_seen_twice\"}";
        string before = await server.GetAsync("/metrics");
        await server.PostAsync("/v1/chat/completions", Body(Manifest()));
        await server.PostAsync("/v1/chat/completions", Body(Manifest()));
        Assert.True(Writer(server).Queue.WaitDrained(Drain));
        string after = await server.GetAsync("/metrics");
        Assert.True(Value(after, Stored) - Value(before, Stored) >= 1, after);
        Assert.True(Value(after, SeenOnce) - Value(before, SeenOnce) >= 1, after);
        Assert.Contains("dotllm_chunk_store_queue_depth", after, StringComparison.Ordinal);
    }

    /// <summary>
    /// Re-review nit 1 (a measurement, run on demand): the worst-case request wait on a real model is one chunk encode.
    /// Set <c>DOTLLM_CHUNK_TIMING_GGUF</c> to a GGUF (e.g. Qwen3-4B-Instruct-2507 Q4_K_M) and optionally
    /// <c>DOTLLM_CHUNK_TIMING_THREADS</c> (default 8); prints the CPU encode time of a 512- and a 2048-token chunk.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void EncodeTime_RealModel_Measured()
    {
        string? gguf = Environment.GetEnvironmentVariable("DOTLLM_CHUNK_TIMING_GGUF");
        Skip.If(string.IsNullOrEmpty(gguf) || !File.Exists(gguf), "DOTLLM_CHUNK_TIMING_GGUF not set");
        int threads = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_CHUNK_TIMING_THREADS"), out int t) ? t : 8;
        var options = new ServerOptions { Model = gguf!, ModelId = "timing", Device = "cpu", Threads = threads, UsePaged = false, ChunkCache = ChunkCacheOptions.Create(true, null) };
        using var state = ServerStartup.LoadModel(gguf!, options);
        var identity = state.ChunkCacheRuntime.IdentityFor(state);
        Assert.True(identity.Supported, identity.UnsupportedReason);
        var encoder = new ChunkEncoder(identity.Fingerprint);
        int[] text = ChunkTokenization.Encode(state.Tokenizer!, string.Concat(Enumerable.Repeat(ChunkA, 200)));
        encoder.Encode(state.Model!, text[..32]); // warm-up
        foreach (int n in new[] { 512, 2048 })
        {
            var sw = Stopwatch.StartNew();
            encoder.Encode(state.Model!, text[..n]);
            _output.WriteLine($"real model encode ({Path.GetFileName(gguf)}, CPU, {threads} threads): {n} tokens {sw.Elapsed.TotalSeconds:0.00} s");
        }
    }

    /// <summary>
    /// Review S4 (a measurement, not a guarantee): the longest a request can wait behind the worker is one chunk encode.
    /// Prints the synthetic model's encode time for ChunkA and for a 2048-token chunk.
    /// </summary>
    [Fact]
    public async Task EncodeTime_TinyModel_Measured()
    {
        await using var server = await ChunkCacheTestServer.StartAsync(_model, ChunkCacheOptions.Create(true, null));
        var identity = server.State.ChunkCacheRuntime.IdentityFor(server.State);
        var encoder = new ChunkEncoder(identity.Fingerprint);
        int[] small = ChunkTokenization.Encode(server.State.Tokenizer!, ChunkA);
        int ctx = server.State.Config!.MaxSequenceLength;
        int[] large = Enumerable.Range(0, Math.Min(2048, ctx)).Select(i => small[i % small.Length]).ToArray();
        encoder.Encode(server.State.Model!, small); // warm-up
        var sw = Stopwatch.StartNew();
        encoder.Encode(server.State.Model!, small);
        double smallMs = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        encoder.Encode(server.State.Model!, large);
        _output.WriteLine($"tiny model encode: {small.Length} tokens {smallMs:0.0} ms; {large.Length} tokens {sw.Elapsed.TotalMilliseconds:0.0} ms");
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private string NewRoot()
    {
        string root = Path.Combine(_scratch, "store-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        return root;
    }

    private static ChunkCacheOptions Writing(string root, string? keysFile = null) =>
        ChunkCacheOptions.Create(true, null, root, keysFile, storeWrite: true, storeBudget: "1GiB");

    private static JsonObject Manifest(string policy = "exact", string? onMiss = null)
    {
        var cc = new JsonObject
        {
            ["v"] = 1,
            ["policy"] = policy,
            ["chunks"] = new JsonArray(ChunkCacheTestServer.Entry("a", 1, User, ChunkA)),
        };
        if (onMiss is not null)
            cc["on_miss"] = onMiss;
        return cc;
    }

    private static string Body(JsonObject? cc, bool stream = false) => ChunkCacheTestServer.OpenAi(Conversation, cc, stream);

    private static JsonNode Report(string body) => JsonNode.Parse(body)!["chunk_cache"]!;

    private static int? WarningCount(JsonNode report, string code) =>
        report["warnings"]!.AsArray().FirstOrDefault(w => w!["code"]!.GetValue<string>() == code)?["count"]!.GetValue<int>();

    private static string? WarningDetail(JsonNode report, string code) =>
        report["warnings"]!.AsArray().FirstOrDefault(w => w!["code"]!.GetValue<string>() == code)?["detail"]?.GetValue<string>();

    private static ChunkStoreService Writer(ChunkCacheTestServer server) => server.State.ChunkCacheRuntime.Writer!;

    private static (ChunkKey Key, int[] Ids) KeyOfChunkA(ChunkCacheTestServer server)
    {
        var identity = server.State.ChunkCacheRuntime.IdentityFor(server.State);
        Assert.True(identity.Supported, identity.UnsupportedReason);
        int[] ids = ChunkTokenization.Encode(server.State.Tokenizer!, ChunkA);
        return (identity.KeyOf(ids, ChunkNamespace.Default), ids);
    }

    /// <summary>
    /// The second request with a chunk queues it; the worker stores exactly the CLI's entry for it (same ids, K/V bit for bit,
    /// fingerprint, backend tag <c>cpu</c>) with a ledger row in the default namespace; the next shadow request hits it.
    /// </summary>
    /// <summary>
    /// The backend is part of the key: an entry this CPU server wrote is a hit for it and a miss for a server whose identity
    /// differs only in the backend tag (what a Vulkan server over the same store sees), on the same store and request.
    /// </summary>
    [Fact]
    public async Task EntryWrittenByOneBackend_IsAMissForAnother_SameStoreSameRequest()
    {
        string root = NewRoot();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        await server.PostAsync("/v1/chat/completions", Body(Manifest()));
        await server.PostAsync("/v1/chat/completions", Body(Manifest()));
        Assert.True(Writer(server).Queue.WaitDrained(Drain));
        var (_, hit) = await server.PostAsync("/v1/chat/completions", Body(Manifest("shadow")));
        Assert.Equal(1, Report(hit)["chunks"]!["hit"]!.GetValue<int>());   // control: the writing backend hits

        var runtime = server.State.ChunkCacheRuntime;
        ServerModelIdentity cpu = runtime.IdentityFor(server.State);
        runtime.IdentityOverride = ServerModelIdentity.Create(server.State.Config!, cpu.ModelId, cpu.Quantization, cpu.TokenizerId, "vulkan");
        try
        {
            Assert.Equal(cpu.ModelId, runtime.IdentityFor(server.State).ModelId);
            Assert.NotEqual(cpu.KeyOf([1, 2, 3], ""), runtime.IdentityFor(server.State).KeyOf([1, 2, 3], ""));
            var (status, other) = await server.PostAsync("/v1/chat/completions", Body(Manifest("shadow")));
            Assert.Equal(200, status);
            Assert.Equal(0, Report(other)["chunks"]!["hit"]!.GetValue<int>());
            Assert.Equal(1, Report(other)["chunks"]!["miss"]!.GetValue<int>());
        }
        finally
        {
            runtime.IdentityOverride = null;
        }
    }

    [Fact]
    public async Task EntriesTaggedWithServerBackend()
    {
        string root = NewRoot();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        var (key, ids) = KeyOfChunkA(server);

        var (s1, b1) = await server.PostAsync("/v1/chat/completions", Body(Manifest()));
        Assert.Equal(200, s1);
        Assert.Equal(1, WarningCount(Report(b1), ChunkCacheCodes.EncodeSkipped)); // seen once
        Assert.Equal(0, Report(b1)["tokens"]!["encode_queued"]!.GetValue<int>());
        var (s2, b2) = await server.PostAsync("/v1/chat/completions", Body(Manifest()));
        Assert.Equal(200, s2);
        Assert.Equal(1, WarningCount(Report(b2), ChunkCacheCodes.EncodeQueued));
        Assert.Equal(ids.Length, Report(b2)["tokens"]!["encode_queued"]!.GetValue<int>());
        Assert.True(Writer(server).Queue.WaitDrained(Drain));

        var identity = server.State.ChunkCacheRuntime.IdentityFor(server.State);
        var store = new FileChunkStore(root);
        Assert.True(store.TryGet(key, out var entry));
        Assert.Equal("cpu", entry.Backend);
        Assert.Equal(identity.Backend, entry.Backend);
        Assert.Equal(identity.Fingerprint, entry.ModelFingerprint);
        Assert.Equal(ids, entry.TokenIds);
        var oracle = new ChunkEncoder(identity.Fingerprint).Encode(server.State.Model!, ids);
        for (int l = 0; l < oracle.KeysPerLayer.Length; l++)
        {
            Assert.Equal(oracle.KeysPerLayer[l], entry.KeysPerLayer[l]);
            Assert.Equal(oracle.ValuesPerLayer[l], entry.ValuesPerLayer[l]);
        }
        var row = Assert.Single(ServerLedger.Load(root).Rows);
        Assert.Equal(key.Hex, row.KeyHex);
        Assert.Equal(ChunkNamespace.Default, row.Namespace);
        Assert.Equal(new FileInfo(store.PathOf(key)).Length, row.Bytes);

        var (s3, b3) = await server.PostAsync("/v1/chat/completions", Body(Manifest("shadow")));
        Assert.Equal(200, s3);
        Assert.Equal(1, Report(b3)["chunks"]!["hit"]!.GetValue<int>());
        Assert.Null(WarningCount(Report(b3), ChunkCacheCodes.EncodeQueued)); // a hit is not queued again
    }

    /// <summary>
    /// <c>on_miss: "encode"</c> queues the first sighting while the request itself is served exactly as the unannotated one
    /// (blocking and streaming: same text and usage).
    /// </summary>
    [Fact]
    public async Task OnMissEncode_QueuesFirstSighting_ServesExact()
    {
        string root = NewRoot();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        var (key, _) = KeyOfChunkA(server);

        var (_, plain) = await server.PostAsync("/v1/chat/completions", Body(null));
        var (status, annotated) = await server.PostAsync("/v1/chat/completions", Body(Manifest(onMiss: "encode")));
        Assert.Equal(200, status);
        Assert.Equal(1, WarningCount(Report(annotated), ChunkCacheCodes.EncodeQueued));
        var p = JsonNode.Parse(plain)!;
        var a = JsonNode.Parse(annotated)!;
        Assert.Equal(p["choices"]!.ToJsonString(), a["choices"]!.ToJsonString());
        Assert.Equal(p["usage"]!["prompt_tokens"]!.GetValue<int>(), a["usage"]!["prompt_tokens"]!.GetValue<int>());
        Assert.Equal(p["usage"]!["completion_tokens"]!.GetValue<int>(), a["usage"]!["completion_tokens"]!.GetValue<int>());
        Assert.True(Writer(server).Queue.WaitDrained(Drain));
        Assert.True(new FileChunkStore(root).Contains(key));

        var (_, plainStream) = await server.PostAsync("/v1/chat/completions", Body(null, stream: true));
        var (_, annotatedStream) = await server.PostAsync("/v1/chat/completions", Body(Manifest(onMiss: "encode"), stream: true));
        string Text(string sse) => string.Concat(ChunkCacheTestServer.SseData(sse)
            .Select(d => d["choices"]?.AsArray().FirstOrDefault()?["delta"]?["content"]?.GetValue<string>() ?? ""));
        Assert.Equal(Text(plainStream), Text(annotatedStream));
    }

    /// <summary>A second server on the same store root refuses to start, before it listens.</summary>
    [Fact]
    public async Task SecondServerSameRoot_FailsFast()
    {
        string root = NewRoot();
        await using var first = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        var sw = Stopwatch.StartNew();
        var ex = await Assert.ThrowsAsync<InvalidServerOptionException>(() => ChunkCacheTestServer.StartAsync(_model, Writing(root)));
        Assert.Contains(StoreOwnerLock.FileName, ex.Message, StringComparison.Ordinal);
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(30));
        // A read-only server on the same root is fine (it never writes).
        await using var reader = await ChunkCacheTestServer.StartAsync(_model, ChunkCacheOptions.Create(true, null, root));
        Assert.Null(reader.State.ChunkCacheRuntime.Writer);
    }

    /// <summary>
    /// An unreadable ledger at startup: the server runs, the ledger file stays byte-identical across requests that would
    /// otherwise store, nothing is written, and responses warn <c>chunk_store_unavailable</c> with <c>ledger_unreadable</c>.
    /// </summary>
    [Fact]
    public async Task UnreadableLedgerAtStartup_WritesDisabled_FileUntouched()
    {
        string root = NewRoot();
        byte[] garbage = Encoding.UTF8.GetBytes("not json at all");
        File.WriteAllBytes(ServerLedger.DefaultPath(root), garbage);
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));

        for (int i = 0; i < 2; i++)
        {
            var (status, body) = await server.PostAsync("/v1/chat/completions", Body(Manifest(onMiss: "encode")));
            Assert.Equal(200, status);
            Assert.Equal(StoreProblems.LedgerUnreadable, WarningDetail(Report(body), ChunkCacheCodes.StoreUnavailable));
        }
        Assert.True(Writer(server).Queue.WaitDrained(Drain));
        Assert.Equal(garbage, File.ReadAllBytes(ServerLedger.DefaultPath(root)));
        Assert.Empty(new FileChunkStore(root).Keys());
    }

    /// <summary>
    /// With the store lock held by another writer, annotated requests still complete at their normal speed (well under the
    /// 10 s lock timeout) and report the queue; the worker then reports the timeout as <c>lock_timeout</c>.
    /// </summary>
    [Fact]
    public async Task RequestThread_NeverBlocksOnStoreLock()
    {
        string root = NewRoot();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        var (key, _) = KeyOfChunkA(server);
        await server.PostAsync("/v1/chat/completions", Body(null)); // warm-up
        using (var held = ChunkIndex.AcquireStoreLock(root, TimeSpan.FromSeconds(5)))
        {
            for (int i = 0; i < 2; i++)
            {
                var sw = Stopwatch.StartNew();
                var (status, body) = await server.PostAsync("/v1/chat/completions", Body(Manifest(onMiss: "encode")));
                Assert.Equal(200, status);
                Assert.True(sw.Elapsed < TimeSpan.FromSeconds(5), $"request took {sw.Elapsed}");
                Assert.Equal(1, WarningCount(Report(body), ChunkCacheCodes.EncodeQueued));
            }
            Assert.True(Writer(server).Queue.WaitDrained(Drain));
            Assert.Equal(StoreProblems.LockTimeout, Writer(server).Problem);
            Assert.False(new FileChunkStore(root).Contains(key));
        }
        var (_, after) = await server.PostAsync("/v1/chat/completions", Body(Manifest(onMiss: "encode")));
        Assert.Equal(StoreProblems.LockTimeout, WarningDetail(Report(after), ChunkCacheCodes.StoreUnavailable));
        Assert.True(Writer(server).Queue.WaitDrained(Drain));
        Assert.True(new FileChunkStore(root).Contains(key));
        Assert.Null(Writer(server).Problem);
    }

    /// <summary><c>explain</c> is a dry run: it never records a sighting or queues an encode.</summary>
    [Fact]
    public async Task Explain_NeverAdmits()
    {
        string root = NewRoot();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        string explain = new JsonObject { ["request"] = JsonNode.Parse(Body(Manifest(onMiss: "encode"))) }.ToJsonString();
        for (int i = 0; i < 3; i++)
        {
            var (status, _) = await server.PostAsync("/v1/chat/completions".Replace("chat/completions", "chunk-cache/explain"), explain);
            Assert.Equal(200, status);
        }
        Assert.True(Writer(server).Queue.WaitDrained(Drain));
        Assert.Equal(0, Writer(server).Sightings.Count);
        Assert.Empty(new FileChunkStore(root).Keys());
    }

    /// <summary>
    /// An unauthenticated inference caller (keys configured, none presented) has no store side effects: no sighting, no
    /// queue, no encode warnings; the same request with a key queues.
    /// </summary>
    [Fact]
    public async Task Unauthenticated_NoStoreSideEffects()
    {
        string root = NewRoot();
        string keys = Path.Combine(_scratch, "keys.txt");
        File.WriteAllLines(keys, [Key]);
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root, keys));
        for (int i = 0; i < 2; i++)
        {
            var (status, body) = await server.PostAsync("/v1/chat/completions", Body(Manifest(onMiss: "encode")));
            Assert.Equal(200, status);
            Assert.Null(WarningCount(Report(body), ChunkCacheCodes.EncodeQueued));
            Assert.Null(WarningCount(Report(body), ChunkCacheCodes.EncodeSkipped));
        }
        Assert.Equal(0, Writer(server).Sightings.Count);
        var (s, keyed) = await server.SendAsync(HttpMethod.Post, "/v1/chat/completions", Body(Manifest(onMiss: "encode")), ("X-API-Key", Key));
        Assert.Equal(200, s);
        Assert.Equal(1, WarningCount(Report(keyed), ChunkCacheCodes.EncodeQueued));
        Assert.True(Writer(server).Queue.WaitDrained(Drain));
        var row = Assert.Single(ServerLedger.Load(root).Rows);
        Assert.Equal(ChunkNamespace.ForApiKey(Key), row.Namespace);
    }

    /// <summary>
    /// The idle probe the worker uses: refused while a direct-generator request holds the gate, and while the scheduler has a
    /// queued sequence even with the gate free; granted on an idle server, and a request then waits for the lease.
    /// </summary>
    [Fact]
    public async Task IdleProbe_RefusedWhileARequestRunsOrIsScheduled()
    {
        string root = NewRoot();
        await using (var single = await ChunkCacheTestServer.StartAsync(_model, Writing(root), paged: false))
        {
            var running = new TaskCompletionSource();
            var release = new TaskCompletionSource();
            var work = single.State.ExecuteAsync(async () => { running.SetResult(); await release.Task; }, CancellationToken.None);
            await running.Task;
            Assert.Null(single.State.TryAcquireIdleModel());
            release.SetResult();
            await work;
            using (var lease = single.State.TryAcquireIdleModel())
                Assert.NotNull(lease);
        }

        string root2 = NewRoot();
        await using var paged = await ChunkCacheTestServer.StartAsync(_model, Writing(root2));
        await paged.PostAsync("/v1/chat/completions", Body(null)); // warm-up
        using (var lease = paged.State.TryAcquireIdleModel())
        {
            Assert.NotNull(lease);
            // While the worker holds the model, a request waits (it is not refused) and completes once the lease goes.
            var pending = paged.PostAsync("/v1/chat/completions", Body(null));
            await Task.Delay(300);
            Assert.False(pending.IsCompleted);
            lease!.Dispose();
            var (status, _) = await pending;
            Assert.Equal(200, status);
        }
    }

    /// <summary>
    /// The probe refuses while the continuous-batch scheduler has a queued sequence even with the gate and the model lock
    /// free (a scheduler whose loop is not running holds its sequence queued).
    /// </summary>
    [Fact]
    public async Task IdleProbe_RefusedWhileSchedulerHasWork()
    {
        string root = NewRoot();
        await using var paged = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        var state = paged.State;
        var running = state.Scheduler!;
        using var parked = new DotLLM.Engine.Scheduler.ContinuousBatchSchedulerService(state.Model!, state.Tokenizer!, state.KvCacheFactory!,
            registerTelemetryProviders: false);
        using var cancel = new CancellationTokenSource();
        state.Scheduler = parked; // no run loop: what it is given stays queued
        try
        {
            using (var idle = state.TryAcquireIdleModel())
                Assert.NotNull(idle);
            var queued = parked.EnqueueAsync(new InferenceRequest
            {
                TokenIds = state.Tokenizer!.Encode("hello"),
                Options = new InferenceOptions { MaxTokens = 2 },
            }, cancel.Token);
            Assert.False(parked.Inner.IsIdle);
            Assert.Null(state.TryAcquireIdleModel());
            cancel.Cancel();
            _ = queued.ContinueWith(static _ => { }, TaskScheduler.Default);
        }
        finally
        {
            state.Scheduler = running;
        }
    }

    /// <summary>
    /// A chunk holding a special-token spelling (<c>&lt;think&gt;</c>) is encoded by the server with
    /// <c>ChunkTokenization.Encode</c> (ruling 21: plain text, as on the chat path), so its entry lands under the key the
    /// CLI and the wire path compute.
    /// </summary>
    [Fact]
    public async Task ServerEncode_ChunkWithThinkTag_UsesChunkTokenization()
    {
        string root = NewRoot();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        string chunk = "/// The model writes <think> before it reasons, then </think> after it.\npublic const string Open = \"<think>\";\n";
        string user = "Look:\n" + chunk + "Why?";
        (string, string)[] conversation = [("user", user)];
        var cc = new JsonObject
        {
            ["v"] = 1,
            ["on_miss"] = "encode",
            ["chunks"] = new JsonArray(ChunkCacheTestServer.Entry("t", 0, user, chunk)),
        };
        var (status, body) = await server.PostAsync("/v1/chat/completions", ChunkCacheTestServer.OpenAi(conversation, cc));
        Assert.Equal(200, status);
        Assert.Equal(1, WarningCount(Report(body), ChunkCacheCodes.EncodeQueued));
        Assert.True(Writer(server).Queue.WaitDrained(Drain));

        var identity = server.State.ChunkCacheRuntime.IdentityFor(server.State);
        int[] ids = ChunkTokenization.Encode(server.State.Tokenizer!, chunk);
        Assert.DoesNotContain(server.State.Tokenizer!.EncodeRaw("<think>").Single(), ids);
        Assert.True(new FileChunkStore(root).TryGet(identity.KeyOf(ids, ChunkNamespace.Default), out var entry));
        Assert.Equal(ids, entry.TokenIds);
    }
}

/// <summary>
/// Phase I Task 15, GPU step: CPU and Vulkan entries never mix. A store written by a CPU server (its sole writer) gives
/// <c>hit: 0</c> in <c>shadow</c> on a Vulkan server for the same request (the backend is in the key), and the Vulkan
/// server, though started with <c>--chunk-store-write</c>, stores nothing (only CPU entries are written until the
/// per-backend schema question is settled). Tiny synthetic model, two generated tokens per request.
/// </summary>
[Trait("Category", "GPU")]
[Collection(ChunkCacheServerCollection.Name)]
public sealed class ChunkStoreServerGpuTests : IDisposable
{
    private static readonly string ChunkA =
        "public static int Add(int a, int b)\n{\n    return a + b; // the sum of both operands\n}\n";

    private static readonly string User = "Read this file.\n// A.cs\n" + ChunkA + "What does Add return?";

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunk-writer-gpu-" + Guid.NewGuid().ToString("N")[..12]);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private static string Body(string policy, string? onMiss = null)
    {
        var cc = new JsonObject
        {
            ["v"] = 1,
            ["policy"] = policy,
            ["chunks"] = new JsonArray(ChunkCacheTestServer.Entry("a", 1, User, ChunkA)),
        };
        if (onMiss is not null)
            cc["on_miss"] = onMiss;
        return ChunkCacheTestServer.OpenAi([("system", "You are terse."), ("user", User)], cc, maxTokens: 2);
    }

    [SkippableFact]
    public async Task VulkanServer_CpuWrittenStore_NoHits()
    {
        DotLLM.Tests.Unit.Vulkan.VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        string model = ChunkCacheTestServer.WriteModel(Path.Combine(_scratch, "tiny.gguf"));
        string root = Path.Combine(_scratch, "store");
        var options = ChunkCacheOptions.Create(true, null, root, storeWrite: true, storeBudget: "256MiB");

        await using (var cpu = await ChunkCacheTestServer.StartAsync(model, options))
        {
            var (status, _) = await cpu.PostAsync("/v1/chat/completions", Body("exact", onMiss: "encode"));
            Assert.Equal(200, status);
            Assert.True(cpu.State.ChunkCacheRuntime.Writer!.Queue.WaitDrained(TimeSpan.FromSeconds(60)));
            var (_, shadow) = await cpu.PostAsync("/v1/chat/completions", Body("shadow"));
            Assert.Equal(1, JsonNode.Parse(shadow)!["chunk_cache"]!["chunks"]!["hit"]!.GetValue<int>());
        }
        var keys = new FileChunkStore(root).Keys().Select(k => k.Hex).ToList();
        Assert.Single(keys);
        Assert.True(new FileChunkStore(root).TryGet(new ChunkKey(keys[0]), out var written));
        Assert.Equal("cpu", written.Backend);
        byte[] ledger = File.ReadAllBytes(ServerLedger.DefaultPath(root));

        await using (var vk = await ChunkCacheTestServer.StartAsync(model, options, device: "vulkan"))
        {
            var identity = vk.State.ChunkCacheRuntime.IdentityFor(vk.State);
            Assert.True(identity.Supported, identity.UnsupportedReason);
            Assert.Equal("vulkan", identity.Backend); // really on the GPU, not a CPU fallback
            for (int i = 0; i < 2; i++)
            {
                var (status, body) = await vk.PostAsync("/v1/chat/completions", Body("shadow", onMiss: "encode"));
                Assert.Equal(200, status);
                var report = JsonNode.Parse(body)!["chunk_cache"]!;
                Assert.Equal(0, report["chunks"]!["hit"]!.GetValue<int>());
                Assert.Equal(1, report["chunks"]!["miss"]!.GetValue<int>());
                Assert.Equal(0, report["plan"]!["tokens"]!["chunk_reused"]!.GetValue<int>());
                Assert.Contains(report["warnings"]!.AsArray(), w => w!["code"]!.GetValue<string>() == ChunkCacheCodes.EncodeSkipped);
                Assert.DoesNotContain(report["warnings"]!.AsArray(), w => w!["code"]!.GetValue<string>() == ChunkCacheCodes.StoreUnavailable);
            }
            Assert.True(vk.State.ChunkCacheRuntime.Writer!.Queue.WaitDrained(TimeSpan.FromSeconds(60)));
        }
        Assert.Equal(keys, new FileChunkStore(root).Keys().Select(k => k.Hex).ToList());
        Assert.Equal(ledger, File.ReadAllBytes(ServerLedger.DefaultPath(root)));
    }
}
