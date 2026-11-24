using System.Text.Json.Nodes;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Backends;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Identity;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using DotLLM.Server.ChunkCache;
using DotLLM.Tests.Unit.Vulkan.Ternary;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// #1915: what changes the weights without changing the base GGUF (a per-request LoRA, <c>--tensor-overlay</c>,
/// <c>--ternary-adapter</c>) is part of the chunk identity, so such requests and servers live in their own key space: never a
/// wrong hit on base-model K/V, and base entries written before this existed stay valid for the base model.
/// </summary>
[Collection(ChunkCacheServerCollection.Name)]
public sealed class ChunkIdentityModifiersTests : IDisposable
{
    private static readonly string ChunkA =
        "public static int Add(int a, int b)\n{\n    return a + b; // the sum of both operands\n}\n";

    private static readonly string User = "Read this file.\n// A.cs\n" + ChunkA + "What does Add return?";

    private static int _nonce;

    /// <summary>
    /// A fresh system prompt per request: the prompt cache would otherwise restore the whole repeated prompt and report the chunk
    /// <c>prefix</c>, never <c>hit</c>/<c>miss</c> (a request that wants a chunk status must diverge before the chunk).
    /// </summary>
    private static (string Role, string Content)[] NextConversation() => [("system", "You are terse. #" + Interlocked.Increment(ref _nonce)), ("user", User)];

    private static readonly TimeSpan Drain = TimeSpan.FromSeconds(60);

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunk-identity-" + Guid.NewGuid().ToString("N")[..12]);
    private readonly string _model;

    public ChunkIdentityModifiersTests()
    {
        _model = ChunkCacheTestServer.WriteModel(Path.Combine(_scratch, "tiny.gguf"));
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    // ------------------------------------------------------------------------------------------------ helpers

    private string NewRoot()
    {
        string root = Path.Combine(_scratch, "store-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        return root;
    }

    private static ChunkCacheOptions Writing(string root) =>
        ChunkCacheOptions.Create(true, null, root, null, storeWrite: true, storeBudget: "1GiB");

    private static ChunkCacheOptions ReadOnly(string root) => ChunkCacheOptions.Create(true, null, root);

    private static JsonObject Manifest(string policy = "shadow", string? onMiss = null)
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

    private static string Body(JsonObject cc, string? adapter = null)
    {
        var o = JsonNode.Parse(ChunkCacheTestServer.OpenAi(NextConversation(), cc, false))!.AsObject();
        if (adapter is not null)
            o["lora_adapter"] = adapter;
        return o.ToJsonString();
    }

    private static JsonNode Chunks(string body) => JsonNode.Parse(body)!["chunk_cache"]!["chunks"]!;

    private static int Count(string body, string status) => Chunks(body)[status]?.GetValue<int>() ?? 0;

    private static async Task<(int Hit, int Miss)> Lookup(ChunkCacheTestServer server, string? adapter = null)
    {
        var (status, body) = await server.PostAsync("/v1/chat/completions", Body(Manifest(), adapter));
        Assert.Equal(200, status);
        return (Count(body, "hit"), Count(body, "miss"));
    }

    private static int[] IdsOfChunkA(ChunkCacheTestServer server) => ChunkTokenization.Encode(server.State.Tokenizer!, ChunkA);

    /// <summary>A deterministic LoRA over every layer's q_proj and v_proj of <paramref name="cfg"/>: content decided by <paramref name="seed"/>.</summary>
    private static unsafe LoraAdapter MakeLora(string name, ModelConfig cfg, int seed, float alpha = 8f)
    {
        const int Rank = 4;
        var adapter = new LoraAdapter(name, Rank, alpha, ["q_proj", "v_proj"]);
        var rng = new Random(seed);
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim, vOut = cfg.NumKvHeads * cfg.HeadDim;
        for (int layer = 0; layer < cfg.NumLayers; layer++)
        {
            foreach ((string proj, int dOut) in new[] { ("q_proj", qOut), ("v_proj", vOut) })
            {
                nint b = LoraAdapter.AllocAligned((long)Rank * cfg.HiddenSize);
                nint a = LoraAdapter.AllocAligned((long)dOut * Rank);
                for (long i = 0; i < (long)Rank * cfg.HiddenSize; i++) ((float*)b)[i] = (float)(rng.NextDouble() * 0.02 - 0.01);
                for (long i = 0; i < (long)dOut * Rank; i++) ((float*)a)[i] = (float)(rng.NextDouble() * 0.02 - 0.01);
                adapter.AddLayerWeights(layer, proj, new LoraLayerWeights(a, b, cfg.HiddenSize, dOut));
            }
        }
        return adapter;
    }

    /// <summary>Registers adapters whose content is the seed in their name: <c>s7</c>, <c>s7-again</c> (same content), <c>s9</c>.</summary>
    private static void RegisterLoras(ChunkCacheTestServer server)
    {
        var cfg = server.State.Config!;
        var registry = new LoraAdapterRegistry((name, path) => MakeLora(name, cfg, int.Parse(path)));
        registry.Load("s7", "7");
        registry.Load("s7-again", "7");
        registry.Load("s9", "9");
        server.State.LoraRegistry = registry;
    }

    /// <summary>Puts the chunk into <paramref name="store"/> under <paramref name="identity"/>'s key, encoded by the server's own model.</summary>
    private static void PutUnder(ChunkCacheTestServer server, IChunkStore store, ServerModelIdentity identity)
    {
        int[] ids = IdsOfChunkA(server);
        var entry = new ChunkEncoder(identity.Fingerprint).Encode(server.State.Model!, ids);
        store.Put(identity.KeyOf(ids, ChunkNamespace.Default), entry);
    }

    private static ILoraAdapter Adapter(ChunkCacheTestServer server, string name) => server.State.LoraRegistry!.Get(name)!;

    // ------------------------------------------------------------------------------------------------ the digest

    /// <summary>The same set in any order is one digest; any difference in the set (content, kind, id, membership) is another.</summary>
    [Fact]
    public void ModifierSetDigest_IsOrderStable_AndSetSensitive()
    {
        (string, string, string) a = ("lora", "", "d1"), b = ("overlay", "", "d2"), c = ("ternary-adapter", "", "d3");
        string abc = ModifierSetDigest.Compute([a, b, c]);
        Assert.Equal(abc, ModifierSetDigest.Compute([c, a, b]));
        Assert.Equal(abc, ModifierSetDigest.Compute([b, c, a]));
        Assert.NotEqual(abc, ModifierSetDigest.Compute([a, b]));
        Assert.NotEqual(abc, ModifierSetDigest.Compute([a, b, ("ternary-adapter", "", "d3x")]));
        Assert.NotEqual(abc, ModifierSetDigest.Compute([a, b, ("ternary-adapter", "id", "d3")]));
        Assert.NotEqual(ModifierSetDigest.Compute([("lora", "", "d")]), ModifierSetDigest.Compute([("overlay", "", "d")])); // kind matters
        Assert.NotEqual(ModifierSetDigest.Compute([a, a]), ModifierSetDigest.Compute([a])); // multiplicity matters
        // Field boundaries cannot be forged: ("ab","c") is not ("a","bc").
        Assert.NotEqual(ModifierSetDigest.Compute([("ab", "c", "d")]), ModifierSetDigest.Compute([("a", "bc", "d")]));
    }

    /// <summary>A LoRA's digest is its content: not its name, but every weight, its rank and alpha.</summary>
    [Fact]
    public unsafe void LoraDigest_IsContent_NotName_AndSeesEveryWeight()
    {
        var fx = new ModelConfigFixture();
        var cfg = fx.Config;
        string a = ModifierSetDigest.OfLora(MakeLora("one", cfg, 7), cfg);
        Assert.Equal(a, ModifierSetDigest.OfLora(MakeLora("another-name", cfg, 7), cfg)); // control: same content, other name
        Assert.NotEqual(a, ModifierSetDigest.OfLora(MakeLora("one", cfg, 8), cfg));
        Assert.NotEqual(a, ModifierSetDigest.OfLora(MakeLora("one", cfg, 7, alpha: 9f), cfg));

        // The very last weight byte decides too (a digest that sampled only the head or tail of a buffer would miss it).
        var tweaked = MakeLora("one", cfg, 7);
        var last = tweaked.GetLayerWeights(cfg.NumLayers - 1, "v_proj")!.Value;
        long aElems = (long)last.OutputDim * tweaked.Rank;
        ((float*)last.AHandle)[aElems - 1] += 1e-3f;
        Assert.NotEqual(a, ModifierSetDigest.OfLora(tweaked, cfg));
        var tweakedB = MakeLora("one", cfg, 7);
        var firstQ = tweakedB.GetLayerWeights(0, "q_proj")!.Value;
        ((float*)firstQ.BHandle)[0] += 1e-3f;
        Assert.NotEqual(a, ModifierSetDigest.OfLora(tweakedB, cfg));
    }

    /// <summary>Provides a ModelConfig without a server (the synthetic model's).</summary>
    private sealed class ModelConfigFixture
    {
        public ModelConfig Config { get; } = SyntheticConfig();

        private static ModelConfig SyntheticConfig()
        {
            string path = Path.Combine(Path.GetTempPath(), "dotllm-chunk-identity-cfg-" + Guid.NewGuid().ToString("N")[..8] + ".gguf");
            try
            {
                ChunkCacheTestServer.WriteModel(path);
                using var gguf = GgufFile.Open(path);
                return GgufModelConfigExtractor.Extract(gguf.Metadata);
            }
            finally
            {
                try { File.Delete(path); } catch (IOException) { }
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ fingerprint compatibility

    /// <summary>
    /// Old entries stay valid for the base model: with no modifiers the fingerprint is exactly the one written before #1915
    /// (<see cref="ChunkFingerprint.Compute(ModelConfig, string, string, string, string, int)"/>), so keys and stored files are unchanged and
    /// no index version bump is needed. Any modifier makes a different fingerprint, so an old entry never matches a modified request.
    /// </summary>
    [Fact]
    public void Fingerprint_NoModifiers_IsThePreChangeValue_AnyModifierChangesIt()
    {
        var cfg = new ModelConfigFixture().Config;
        string legacy = ChunkFingerprint.Compute(cfg, "model-id", "ftype-2", "tok-id", "cpu");
        Assert.Equal(legacy, ChunkFingerprint.WithModifiers(legacy, null));
        Assert.Equal(legacy, ChunkFingerprint.WithModifiers(legacy, ""));
        Assert.Equal(legacy, ServerModelIdentity.Create(cfg, "model-id", "ftype-2", "tok-id", "cpu").Fingerprint);

        var identity = ServerModelIdentity.Create(cfg, "model-id", "ftype-2", "tok-id", "cpu");
        Assert.Same(identity, identity.WithModifiers([])); // nothing to add: the same identity
        var lora = identity.WithModifiers([("lora", "", "d1")]);
        Assert.NotEqual(legacy, lora.Fingerprint);
        Assert.NotEqual(lora.Fingerprint, identity.WithModifiers([("lora", "", "d2")]).Fingerprint);
        Assert.Equal(lora.Fingerprint, identity.WithModifiers([("lora", "", "d1")]).Fingerprint); // same set: same key space
        int[] ids = [1, 2, 3];
        Assert.NotEqual(identity.KeyOf(ids, ""), lora.KeyOf(ids, ""));
        // Adding in two steps is the same set as adding at once, in any order (the digest sorts).
        Assert.Equal(identity.WithModifiers([("lora", "", "d1"), ("overlay", "", "o")]).Fingerprint,
            identity.WithModifiers([("overlay", "", "o")]).WithModifiers([("lora", "", "d1")]).Fingerprint);
        var unsupported = ServerModelIdentity.Unsupported("x");
        Assert.Same(unsupported, unsupported.WithModifiers([("lora", "", "d1")])); // nothing to key
    }

    // ------------------------------------------------------------------------------------------------ per-request LoRA, on the wire

    /// <summary>
    /// A base-model entry is a hit for a base request and a miss for every LoRA request; an entry stored under a LoRA's identity is a hit
    /// for that LoRA (and for the same weights loaded under another name) and a miss for another LoRA and for the base model.
    /// </summary>
    [Fact]
    public async Task LoraRequest_NeverHitsBaseEntries_SameLoraHits_OtherLoraMisses()
    {
        string root = NewRoot();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, ReadOnly(root));
        RegisterLoras(server);
        var runtime = server.State.ChunkCacheRuntime;
        var store = new FileChunkStore(root);
        PutUnder(server, store, runtime.IdentityFor(server.State));

        Assert.Equal((1, 0), await Lookup(server));                  // control: the base request hits its own entry
        Assert.Equal((0, 1), await Lookup(server, "s7"));            // a LoRA request misses on base K/V
        Assert.Equal((0, 1), await Lookup(server, "s9"));

        PutUnder(server, store, runtime.IdentityFor(server.State, Adapter(server, "s7")));
        Assert.Equal((1, 0), await Lookup(server, "s7"));            // the same LoRA now hits
        Assert.Equal((1, 0), await Lookup(server, "s7-again"));      // the same weights under another name: the same identity
        Assert.Equal((0, 1), await Lookup(server, "s9"));            // another LoRA misses on s7's entry
        Assert.Equal((1, 0), await Lookup(server));                  // and the base request still hits only its own

        PutUnder(server, store, runtime.IdentityFor(server.State, Adapter(server, "s9")));
        Assert.Equal((1, 0), await Lookup(server, "s9"));
        Assert.Equal((1, 0), await Lookup(server, "s7"));
        Assert.Equal(3, store.Keys().Count()); // one entry per identity: base, s7, s9
    }

    /// <summary>
    /// A LoRA request never admits: the background encoder runs the served model without the adapter, so what it would store under the
    /// adapter's keys would be the wrong K/V. No sighting, no queue, nothing stored; the next plain request is the chunk's first sighting.
    /// </summary>
    [Fact]
    public async Task LoraRequest_NothingAdmitted_NothingStored()
    {
        string root = NewRoot();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root));
        RegisterLoras(server);
        foreach (var cc in new[] { Manifest("exact"), Manifest("exact"), Manifest("exact", onMiss: "encode") })
        {
            var (status, body) = await server.PostAsync("/v1/chat/completions", Body(cc, "s7"));
            Assert.Equal(200, status);
            Assert.True(Count(body, "miss") == 1, body);
            var warnings = JsonNode.Parse(body)!["chunk_cache"]!["warnings"]!.AsArray().Select(w => w!["code"]!.GetValue<string>()).ToArray();
            Assert.DoesNotContain(ChunkCacheCodes.EncodeQueued, warnings);
            Assert.DoesNotContain(ChunkCacheCodes.EncodeSkipped, warnings);
            Assert.DoesNotContain(ChunkCacheCodes.ModelUnsupported, warnings); // supported now: looked up under its own identity
        }
        var writer = server.State.ChunkCacheRuntime.Writer!;
        Assert.Equal(0, writer.Sightings.Count);
        Assert.True(writer.Queue.WaitDrained(Drain));
        Assert.Empty(new FileChunkStore(root).Keys());
    }

    // ------------------------------------------------------------------------------------------------ server-level modifiers

    private string WriteTernaryAdapter(string name, ulong seed, int layer = 0)
    {
        string path = Path.Combine(_scratch, name);
        Directory.CreateDirectory(_scratch);
        VulkanTernaryAdapterKernelTests.Synth(rank: 4, d: 64, seed, alpha: 4f, thr: 0f, layer).Save(path);
        return path;
    }

    private static ServerState StateOf(ChunkCacheTestServer server, ServerOptions options) =>
        new() { Options = options, Model = server.State.Model, Config = server.State.Config, CurrentGguf = server.State.CurrentGguf };

    /// <summary>
    /// The server-level modifier set is content-addressed and order-stable: the same adapter files in another order, or under other
    /// paths, are the same identity; other content, a missing adapter, or the same bytes as an overlay instead are a different one.
    /// </summary>
    [Fact]
    public async Task ServerModifiers_AreContentAddressed_AndOrderStable()
    {
        string root = NewRoot();
        await using var server = await ChunkCacheTestServer.StartAsync(_model, ReadOnly(root));
        string a = WriteTernaryAdapter("a.nsadapter", 11, layer: 0), b = WriteTernaryAdapter("b.nsadapter", 22, layer: 1);
        string aCopy = Path.Combine(_scratch, "elsewhere-a.nsadapter");
        File.Copy(a, aCopy);
        string Fp(ServerOptions o) => new ChunkCacheRuntime().IdentityFor(StateOf(server, o)).Fingerprint;
        var baseOptions = new ServerOptions { Model = "m" };

        string baseFp = Fp(baseOptions);
        Assert.Equal(server.State.ChunkCacheRuntime.IdentityFor(server.State).Fingerprint, baseFp); // control: a base server's identity is the plain one
        string ab = Fp(baseOptions with { TernaryAdapterPaths = [a, b] });
        Assert.NotEqual(baseFp, ab);
        Assert.Equal(ab, Fp(baseOptions with { TernaryAdapterPaths = [b, a] }));       // order-stable
        Assert.Equal(ab, Fp(baseOptions with { TernaryAdapterPaths = [aCopy, b] }));   // path-independent
        Assert.NotEqual(ab, Fp(baseOptions with { TernaryAdapterPaths = [a] }));       // membership
        Assert.NotEqual(Fp(baseOptions with { TernaryAdapterPaths = [a] }), Fp(baseOptions with { TernaryAdapterPaths = [b] })); // content
        Assert.NotEqual(Fp(baseOptions with { TernaryAdapterPaths = [a] }), Fp(baseOptions with { TensorOverlayPath = a })); // kind
        Assert.NotEqual(baseFp, Fp(baseOptions with { TensorOverlayPath = a }));
        Assert.Equal(baseFp, Fp(baseOptions with { TernaryAdapterPaths = [] }));       // an empty list is a base server
        // An unreadable adapter file is never a 500 nor a base-keyed identity: unsupported.
        Assert.False(new ChunkCacheRuntime().IdentityFor(StateOf(server, baseOptions with { TernaryAdapterPaths = [Path.Combine(_scratch, "missing.nsadapter")] })).Supported);
    }

    /// <summary>
    /// A server with a ternary adapter keeps chunk reuse, in its own key space and with real K/V: it misses on the base server's
    /// entry, writes its own (different tensors: the adapter changes every later layer's K/V), hits it, and the base server afterwards
    /// still hits its own entry and not the adapter's.
    /// </summary>
    [Fact]
    public async Task TernaryAdapterServer_OwnKeySpace_RealTensorsDiffer_BothDirectionsMiss()
    {
        string root = NewRoot();
        string adapter = WriteTernaryAdapter("t.nsadapter", 5);
        Func<ServerOptions, ServerOptions> ternary = o => o with { TernaryAdapterPaths = [adapter] };

        ChunkKvEntry baseEntry;
        await using (var baseServer = await ChunkCacheTestServer.StartAsync(_model, Writing(root)))
        {
            await Warm(baseServer);
            Assert.Equal((1, 0), await Lookup(baseServer));
            Assert.True(new FileChunkStore(root).TryGet(KeyOf(baseServer), out baseEntry!));
        }

        ChunkKvEntry ternaryEntry;
        ChunkKey ternaryKey;
        await using (var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root), customize: ternary))
        {
            Assert.True(server.State.ChunkCacheRuntime.IdentityFor(server.State).Supported); // not the blanket refusal any more
            Assert.Equal((0, 1), await Lookup(server));      // the base entry is not a hit under the adapter
            await Warm(server);
            Assert.Equal((1, 0), await Lookup(server));      // its own entry is
            ternaryKey = KeyOf(server);
            Assert.True(new FileChunkStore(root).TryGet(ternaryKey, out ternaryEntry!));
        }

        Assert.NotEqual(KeyOf2(baseEntry), ternaryKey.Hex);
        Assert.NotEqual(baseEntry.ModelFingerprint, ternaryEntry.ModelFingerprint);
        Assert.Equal(baseEntry.TokenIds, ternaryEntry.TokenIds);
        // Real tensors: layer 0 is computed before the adapter's residual edit, every later layer after it.
        Assert.Equal(baseEntry.KeysPerLayer[0], ternaryEntry.KeysPerLayer[0]);
        for (int l = 1; l < baseEntry.KeysPerLayer.Length; l++)
            Assert.NotEqual(baseEntry.KeysPerLayer[l], ternaryEntry.KeysPerLayer[l]);

        await using (var again = await ChunkCacheTestServer.StartAsync(_model, Writing(root)))
        {
            Assert.Equal((1, 0), await Lookup(again));        // the base entry written before the adapter server existed is untouched
            Assert.Equal(2, new FileChunkStore(root).Keys().Count());
        }
    }

    /// <summary>The same for <c>--tensor-overlay</c>: its own key space, real K/V that differ, base entries not hit and still valid.</summary>
    [Fact]
    public async Task TensorOverlayServer_OwnKeySpace_RealTensorsDiffer_BothDirectionsMiss()
    {
        string root = NewRoot();
        string overlay = WriteOverlay();
        Func<ServerOptions, ServerOptions> overlaid = o => o with { TensorOverlayPath = overlay, OverlaySkipBaseHash = true };

        ChunkKvEntry baseEntry;
        await using (var baseServer = await ChunkCacheTestServer.StartAsync(_model, Writing(root)))
        {
            await Warm(baseServer);
            Assert.Equal((1, 0), await Lookup(baseServer));
            Assert.True(new FileChunkStore(root).TryGet(KeyOf(baseServer), out baseEntry!));
        }

        ChunkKvEntry overlayEntry;
        await using (var server = await ChunkCacheTestServer.StartAsync(_model, Writing(root), customize: overlaid))
        {
            Assert.NotNull(server.State.CurrentGguf!.Overlay); // control: the overlay is really applied to the served weights
            Assert.True(server.State.ChunkCacheRuntime.IdentityFor(server.State).Supported);
            Assert.Equal((0, 1), await Lookup(server));
            await Warm(server);
            Assert.Equal((1, 0), await Lookup(server));
            Assert.True(new FileChunkStore(root).TryGet(KeyOf(server), out overlayEntry!));
        }

        Assert.NotEqual(baseEntry.ModelFingerprint, overlayEntry.ModelFingerprint);
        // The overlay replaces layer 0's FFN: layer 0's K/V (attention before the FFN) match, later layers' do not.
        Assert.Equal(baseEntry.KeysPerLayer[0], overlayEntry.KeysPerLayer[0]);
        for (int l = 1; l < baseEntry.KeysPerLayer.Length; l++)
            Assert.NotEqual(baseEntry.KeysPerLayer[l], overlayEntry.KeysPerLayer[l]);

        await using (var again = await ChunkCacheTestServer.StartAsync(_model, Writing(root)))
        {
            Assert.Equal((1, 0), await Lookup(again));
            Assert.Equal(2, new FileChunkStore(root).Keys().Count());
        }
    }

    /// <summary>An entry written under the pre-#1915 recipe (plain fingerprint, no modifier fields) is a hit for a base server, unchanged.</summary>
    [Fact]
    public async Task OldEntry_KeyedWithoutModifiers_StillHitsTheBaseServer_NotAModifiedOne()
    {
        string root = NewRoot();
        string adapter = WriteTernaryAdapter("old.nsadapter", 3);
        await using (var server = await ChunkCacheTestServer.StartAsync(_model, ReadOnly(root)))
        {
            // Keyed exactly as the CLI and the pre-#1915 server did: no use of ServerModelIdentity or the modifier fold.
            var state = server.State;
            var gguf = state.CurrentGguf!;
            string fp = ChunkFingerprint.Compute(state.Config!, GgufIdentity.ModelId(gguf), GgufIdentity.QuantizationLabel(gguf.Metadata),
                GgufIdentity.TokenizerId(gguf.Metadata), ChunkBackendTags.Cpu);
            int[] ids = IdsOfChunkA(server);
            var key = ChunkKey.Create(fp, GgufIdentity.TokenizerId(gguf.Metadata), RopeParameters.FromConfig(state.Config!), ChunkBackendTags.Cpu,
                KvDtype.F32, ids, positioning: ChunkPositioning.Isolated, @namespace: ChunkNamespace.Default);
            new FileChunkStore(root).Put(key, new ChunkEncoder(fp).Encode(state.Model!, ids));
            Assert.Equal((1, 0), await Lookup(server));
        }
        await using var modified = await ChunkCacheTestServer.StartAsync(_model, ReadOnly(root), customize: o => o with { TernaryAdapterPaths = [adapter] });
        Assert.Equal((0, 1), await Lookup(modified));
    }

    // ------------------------------------------------------------------------------------------------ more helpers

    private static async Task Warm(ChunkCacheTestServer server)
    {
        for (int i = 0; i < 2; i++)
            await server.PostAsync("/v1/chat/completions", Body(Manifest("exact")));
        Assert.True(server.State.ChunkCacheRuntime.Writer!.Queue.WaitDrained(Drain));
    }

    private static ChunkKey KeyOf(ChunkCacheTestServer server) =>
        server.State.ChunkCacheRuntime.IdentityFor(server.State).KeyOf(IdsOfChunkA(server), ChunkNamespace.Default);

    private static string KeyOf2(ChunkKvEntry entry) => entry.ModelFingerprint;

    /// <summary>An overlay GGUF replacing layer 0's dense FFN gate with other F32 values (shape [hidden, ff] in GGUF order).</summary>
    private string WriteOverlay()
    {
        const int Hidden = 64, Ff = 96;
        var rng = new Random(99);
        float[] w = new float[Hidden * Ff];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() - 0.5) * 0.9f;
        var gw = new GgufWriter();
        gw.AddString("general.architecture", "dotllm-overlay");
        gw.AddTensor("blk.0.ffn_gate.weight", [Hidden, Ff], (uint)DotLLM.Core.Configuration.QuantizationType.F32,
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(w.AsSpan()).ToArray());
        string path = Path.Combine(_scratch, "overlay.gguf");
        Directory.CreateDirectory(_scratch);
        File.WriteAllBytes(path, gw.Build());
        return path;
    }
}
