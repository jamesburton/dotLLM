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

    public ChunkStoreServerTests() => _model = ChunkCacheTestServer.WriteModel(Path.Combine(_scratch, "tiny.gguf"));

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

    /// <summary>The probe with the scheduler's queue non-empty and the gate free is refused (the scheduler check itself).</summary>
    [Fact]
    public async Task IdleProbe_RefusedWhileSchedulerHasWork()
    {
        string root = NewRoot();
        await using var paged = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        var scheduler = paged.State.Scheduler!;
        var modelLease = await scheduler.AcquireModelAsync();
        Task<InferenceResponse> queued;
        try
        {
            queued = scheduler.EnqueueAsync(new InferenceRequest
            {
                TokenIds = paged.State.Tokenizer!.Encode("hello"),
                Options = new InferenceOptions { MaxTokens = 2 },
            });
            Assert.Null(paged.State.TryAcquireIdleModel());
        }
        finally
        {
            modelLease.Dispose();
        }
        await queued;
    }
}
