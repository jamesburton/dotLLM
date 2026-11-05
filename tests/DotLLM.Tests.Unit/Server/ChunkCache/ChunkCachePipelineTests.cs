using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using DotLLM.Server.ChunkCache;
using DotLLM.Server.Models;
using DotLLM.Tests.Unit.ChunkReuse;
using DotLLM.Tests.Unit.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// <see cref="ChunkCachePipeline"/> and <see cref="ChunkCacheReport"/> (Phase I Task 13): request → parser (Task 10) →
/// render → locator (Task 11) → token check → store lookup → report, on the synthetic qwen3 GGUF's own tokenizer and Qwen3
/// template (byte-level, Qwen3 special tokens) with a real CPU-encoded store, plus a SentencePiece tokenizer with one merge
/// for the boundary case.
/// </summary>
public sealed class ChunkCachePipelineTests : IClassFixture<ChunkCachePipelineTests.Fixture>
{
    private readonly Fixture _fx;

    public ChunkCachePipelineTests(Fixture fx) => _fx = fx;

    // ------------------------------------------------------------------------------------------------ harness

    /// <summary>The synthetic model (GGUF tokenizer, template, CPU model) and its server identity; scratch store roots.</summary>
    public sealed class Fixture : IDisposable
    {
        public Fixture()
        {
            Directory.CreateDirectory(Scratch);
            ModelPath = ChunkCacheTestServer.WriteModel(Path.Combine(Scratch, "tiny.gguf"));
            Gguf = GgufFile.Open(ModelPath);
            Config = GgufModelConfigExtractor.Extract(Gguf.Metadata);
            Tok = GgufBpeTokenizerFactory.Load(Gguf.Metadata);
            Template = GgufChatTemplateFactory.TryCreate(Gguf.Metadata, Tok, Config.Architecture)!;
            Model = TransformerModel.LoadFromGguf(Gguf, Config, ThreadingConfig.SingleThreaded);
            Identity = ServerModelIdentity.FromGguf(Gguf, Config, "cpu");
        }

        public string Scratch { get; } = Path.Combine(Path.GetTempPath(), "dotllm-chunk-pipeline-" + Guid.NewGuid().ToString("N")[..12]);

        public string ModelPath { get; }

        public GgufFile Gguf { get; }

        public ModelConfig Config { get; }

        public BpeTokenizer Tok { get; }

        public JinjaChatTemplate Template { get; }

        public TransformerModel Model { get; }

        internal ServerModelIdentity Identity { get; }

        public string NewDir(string name)
        {
            string d = Path.Combine(Scratch, name + "-" + Guid.NewGuid().ToString("N")[..8]);
            Directory.CreateDirectory(d);
            return d;
        }

        /// <summary>Encodes <paramref name="text"/> with the CPU backend and puts it in <paramref name="store"/> under the server key.</summary>
        public void Put(IChunkStore store, string text, string ns = ChunkNamespace.Default)
        {
            int[] ids = ChunkTokenization.Encode(Tok, text);
            store.Put(Identity.KeyOf(ids, ns), new ChunkEncoder(Identity.Fingerprint).Encode(Model, ids));
        }

        public void Dispose()
        {
            Model.Dispose();
            Gguf.Dispose();
            try { Directory.Delete(Scratch, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    private sealed record Run(ChunkCachePlan Plan, string Prompt, int[] Ids, ParsedChunkRequest Parsed);

    /// <summary>The endpoint path for an OpenAI body: parse, convert, render, locate, plain tokenisation, then the pipeline.</summary>
    private static Run Apply(string body, ITokenizer tok, IChatTemplate template, ServerModelIdentity identity, IChunkStore? store,
        bool storeUnavailable = false)
    {
        var request = JsonSerializer.Deserialize(body, ServerJsonContext.Default.ChatCompletionRequest)!;
        var parsed = ChunkAnnotationParser.Parse(request, new ChunkCacheOptions(true, 4096), ChunkNamespace.Default)!;
        Assert.NotNull(parsed);
        var messages = RequestConverter.ToMessages(request.Messages!);
        var options = ReasoningSupport.BuildTemplateOptions(null, null, null, null, false);
        Assert.True(ReasoningSupport.TryApply(template, messages, options, out string prompt, out string? error, out _), error);
        var located = identity.Supported && parsed.Spans.Count > 0
            ? ChunkSpanLocator.Locate(template, messages, options, prompt, parsed.Spans, 0)
            : null;
        int[] ids = tok.Encode(prompt);
        return new Run(ChunkCachePipeline.Apply(parsed, located, identity, tok, prompt, ids, store, storeUnavailable), prompt, ids, parsed);
    }

    private Run Apply(string body, IChunkStore? store = null, bool storeUnavailable = false) =>
        Apply(body, _fx.Tok, _fx.Template, _fx.Identity, store, storeUnavailable);

    private static string Body((string Role, string Content)[] messages, string policy, (int Message, string Text)[] spans, string extra = "")
    {
        var cc = new JsonObject
        {
            ["v"] = 1,
            ["policy"] = policy,
            ["chunks"] = new JsonArray(spans.Select((s, i) => (JsonNode)ChunkCacheTestServer.Entry("c" + i, s.Message, messages[s.Message].Content, s.Text)).ToArray()),
        };
        if (extra.Length > 0)
            foreach (var p in JsonNode.Parse(extra)!.AsObject().ToList())
                cc[p.Key] = p.Value?.DeepClone();
        return ChunkCacheTestServer.OpenAi(messages, cc);
    }

    private static string Code(int i) => "public int Field" + i + " => " + i + "; // a line of code long enough for a chunk\n";

    private static readonly string[] Chunks = [Code(0), Code(1), Code(2), Code(3)];

    private static readonly string User = "Files:\n" + string.Join("// next\n", Chunks) + "Question?";

    private static (string, string)[] Conversation(string user) => [("system", "Be brief."), ("user", user)];

    private static (int, string)[] AllChunks() => Chunks.Select(c => (1, c)).ToArray();

    private static ChunkCacheReportDto Report(ChunkCachePlan plan, int cached = 0, double prefillMs = 100) =>
        ChunkCacheReport.Build(plan, plan.PromptTokens, cached, prefillMs, ChunkCostModel.For(plan.Backend));

    private static void AssertPartition(ChunkCacheReportDto r)
    {
        Assert.Equal(r.Chunks.Annotated, r.Chunks.Prefix + r.Chunks.Hit + r.Chunks.Miss + r.Chunks.Ignored);
        Assert.Equal(r.Tokens.Prompt, r.Tokens.PrefixReused + r.Tokens.PrefixReusedApprox + r.Tokens.ChunkReused + r.Tokens.Prefilled);
        Assert.True(r.Tokens.ChunkRecomputed <= r.Tokens.ChunkReused);
        Assert.Equal("exact", r.PolicyApplied);
        Assert.Equal(0, r.Tokens.ChunkReused);
        if (r.Plan is { } p)
        {
            Assert.Equal(p.Tokens.Prompt, p.Tokens.PrefixReused + p.Tokens.PrefixReusedApprox + p.Tokens.ChunkReused + p.Tokens.Prefilled);
            Assert.True(p.Tokens.ChunkRecomputed <= p.Tokens.ChunkReused);
            Assert.Equal(r.Tokens.Prompt, p.Tokens.Prompt);
            Assert.Equal(r.Tokens.PrefixReused, p.Tokens.PrefixReused);
        }
    }

    private static string[] Codes(IEnumerable<ChunkWarning> w) => w.Select(x => $"{x.Code}x{x.Count}").ToArray();

    // ------------------------------------------------------------------------------------------------ plan and counts

    /// <summary>
    /// A store pre-filled (CPU backend) for half the spans: <c>shadow</c> finds those as hits, the rest as misses, places
    /// every chunk at its ids in the prompt, and plans reuse for the hits only.
    /// </summary>
    [Fact]
    public void Shadow_PlanCountsHitsAndMisses()
    {
        var store = new FileChunkStore(_fx.NewDir("store"));
        _fx.Put(store, Chunks[0]);
        _fx.Put(store, Chunks[2]);

        var run = Apply(Body(Conversation(User), "shadow", AllChunks()), store);
        var plan = run.Plan;
        Assert.Null(plan.Error);
        Assert.True(plan.Shadow);
        Assert.Null(plan.Reason);
        Assert.Equal([true, false, true, false], plan.Spans.Select(s => s.Hit));
        Assert.Equal(run.Ids.Length, plan.PromptTokens);
        for (int i = 0; i < 4; i++)
        {
            int[] ids = ChunkTokenization.Encode(_fx.Tok, Chunks[i]);
            Assert.Equal(ids.Length, plan.Spans[i].TokenCount);
            Assert.Equal(ids, run.Ids.AsSpan(plan.Spans[i].TokenStart, ids.Length).ToArray());
        }

        var r = Report(plan);
        AssertPartition(r);
        Assert.Equal(new ChunkCacheChunksDto { Annotated = 4, Prefix = 0, Hit = 2, Miss = 2, Ignored = 0 }, r.Chunks);
        Assert.Equal(run.Ids.Length, r.Tokens.Prefilled);
        int n0 = plan.Spans[0].TokenCount, n2 = plan.Spans[2].TokenCount;
        Assert.Equal(n0 + n2, r.Plan!.Tokens.ChunkReused);
        Assert.Equal((int)Math.Ceiling(n0 * ChunkCacheReport.DefaultPlanBudget) + (int)Math.Ceiling(n2 * ChunkCacheReport.DefaultPlanBudget), r.Plan.Tokens.ChunkRecomputed);
        Assert.Equal(ChunkCacheReport.PlanSelector, r.Plan.Selector);
        Assert.Equal(ChunkCacheReport.DefaultPlanBudget, r.Plan.Budget);

        // The cost model: prefill at the request's measured rate; assembly and recompute from the backend's constants.
        var cost = ChunkCostModel.For("cpu");
        double rate = run.Ids.Length / 0.1; // 100 ms for the whole prompt
        double predicted = cost.AssemblyMs(n0 + n2)
            + cost.RecomputeMs(plan.Spans[0].TokenStart, (long)Math.Ceiling(n0 * ChunkCacheReport.DefaultPlanBudget))
            + cost.RecomputeMs(plan.Spans[2].TokenStart, (long)Math.Ceiling(n2 * ChunkCacheReport.DefaultPlanBudget))
            + ChunkCostModel.PrefillMs(run.Ids.Length - n0 - n2, rate);
        Assert.Equal((long)Math.Round(predicted), r.Plan.PredictedTtftMs);
        Assert.Equal(100, r.Plan.FullPrefillEstimateMs);
        Assert.Equal(100, r.TimingsMs.FullPrefillEstimate);
        Assert.Equal(100, r.TimingsMs.Prefill);

        // A restored exact prefix that ends where chunk 1 starts: chunk 0 is "prefix", not a hit; the plan reuses chunk 2 only.
        int cached = plan.Spans[1].TokenStart;
        var warm = Report(plan, cached);
        AssertPartition(warm);
        Assert.Equal(new ChunkCacheChunksDto { Annotated = 4, Prefix = 1, Hit = 1, Miss = 2, Ignored = 0 }, warm.Chunks);
        Assert.Equal(cached, warm.Tokens.PrefixReused);
        Assert.Equal(n2, warm.Plan!.Tokens.ChunkReused);

        // A restored prefix that ends INSIDE a chunk: the chunk straddles the boundary, so it is not "prefix" (only a chunk
        // wholly inside the prefix is). Inside chunk 1 (a miss): chunk 0 is prefix, chunk 1 still a miss. Inside chunk 2 (a
        // hit): chunks 0 and 1 are prefix, chunk 2 a hit whose plan reuses only the rows beyond the prefix.
        var inMiss = Report(plan, plan.Spans[1].TokenStart + 1);
        AssertPartition(inMiss);
        Assert.Equal(new ChunkCacheChunksDto { Annotated = 4, Prefix = 1, Hit = 1, Miss = 2, Ignored = 0 }, inMiss.Chunks);
        int inside = plan.Spans[2].TokenStart + 1;
        var inHit = Report(plan, inside);
        AssertPartition(inHit);
        Assert.Equal(new ChunkCacheChunksDto { Annotated = 4, Prefix = 2, Hit = 1, Miss = 1, Ignored = 0 }, inHit.Chunks);
        Assert.Equal(n2 - 1, inHit.Plan!.Tokens.ChunkReused);

        // exact counts the same hits (ruling 5) and plans nothing.
        var exact = Apply(Body(Conversation(User), "exact", AllChunks()), store).Plan;
        Assert.False(exact.Shadow);
        Assert.Equal([true, false, true, false], exact.Spans.Select(s => s.Hit));
        Assert.Null(Report(exact).Plan);
    }

    /// <summary>
    /// The report's shape is the contract's (response-blocking / response-stream-final / response-shadow): same keys, same
    /// JSON kinds, <c>recompute</c> omitted (ruling 10), <c>reason</c> written as null; and the partitions hold.
    /// </summary>
    [Fact]
    public void Report_PartitionHolds()
    {
        var store = new FileChunkStore(_fx.NewDir("store"));
        _fx.Put(store, Chunks[1]);
        var shadow = Apply(Body(Conversation(User), "shadow", AllChunks()), store).Plan;
        var exact = Apply(Body(Conversation(User), "exact", AllChunks()), store).Plan;

        foreach ((ChunkCachePlan plan, string fixture) in new[] { (exact, "response-blocking.json"), (exact, "response-stream-final.json"), (shadow, "response-shadow.json") })
        {
            foreach (int cached in new[] { 0, 7, plan.Spans[2].TokenStart + 1, plan.PromptTokens })
            {
                var r = Report(plan, cached);
                AssertPartition(r);
                var golden = JsonNode.Parse(ContractFixtures.Bytes(fixture))!["chunk_cache"]!;
                var mine = JsonNode.Parse(JsonSerializer.Serialize(r, ServerJsonContext.Default.ChunkCacheReportDto))!;
                AssertSameShape(golden, mine, "chunk_cache");
                Assert.Equal(JsonValueKind.Null, JsonSerializer.SerializeToElement(r, ServerJsonContext.Default.ChunkCacheReportDto).GetProperty("reason").ValueKind);
            }
        }

        // The anthropic-delta fixture (applied reuse, Phase I5) differs only by `recompute`, which v1 never emits.
        var delta = JsonNode.Parse(ContractFixtures.Bytes("response-anthropic-delta.json"))!["chunk_cache"]!.AsObject();
        Assert.NotNull(delta["recompute"]);
        delta.Remove("recompute");
        AssertSameShape(delta, JsonNode.Parse(JsonSerializer.Serialize(Report(exact), ServerJsonContext.Default.ChunkCacheReportDto))!, "delta");
    }

    private static void AssertSameShape(JsonNode? golden, JsonNode? mine, string path)
    {
        if (golden is JsonObject g)
        {
            var m = Assert.IsType<JsonObject>(mine);
            Assert.True(g.Select(p => p.Key).Order(StringComparer.Ordinal).SequenceEqual(m.Select(p => p.Key).Order(StringComparer.Ordinal)),
                $"{path}: keys {string.Join(",", m.Select(p => p.Key))} vs contract {string.Join(",", g.Select(p => p.Key))}");
            foreach (var p in g)
            {
                if (p.Key == "reason")
                    continue; // null or a string code
                AssertSameShape(p.Value, m[p.Key], path + "." + p.Key);
            }
        }
        else if (golden is JsonArray ga)
        {
            var ma = Assert.IsType<JsonArray>(mine);
            if (ga.Count > 0)
                foreach (var item in ma)
                    AssertSameShape(ga[0], item, path + "[]");
        }
        else if (golden is JsonValue gv)
        {
            var mv = Assert.IsAssignableFrom<JsonValue>(mine);
            Assert.True(gv.GetValueKind() == mv.GetValueKind()
                || (gv.GetValueKind() is JsonValueKind.Number && mv.GetValueKind() is JsonValueKind.Number), $"{path}: kind {mv.GetValueKind()} vs {gv.GetValueKind()}");
        }
    }

    /// <summary><c>balanced</c> / <c>fast</c> / <c>auto</c> are served <c>exact</c> with reason <c>chunk_policy_unavailable</c>; strict is a 400.</summary>
    [Theory]
    [InlineData("balanced")]
    [InlineData("fast")]
    [InlineData("auto")]
    public void Balanced_DegradesToExact_WithReason(string policy)
    {
        var plan = Apply(Body(Conversation(User), policy, AllChunks())).Plan;
        Assert.Null(plan.Error);
        Assert.False(plan.Shadow);
        Assert.Equal(ChunkCacheCodes.PolicyUnavailable, plan.Reason);
        Assert.Contains($"{ChunkCacheCodes.PolicyUnavailable}x1", Codes(plan.Warnings));
        var r = Report(plan);
        Assert.Equal(policy, r.PolicyRequested);
        Assert.Equal("exact", r.PolicyApplied);
        Assert.Equal(ChunkCacheCodes.PolicyUnavailable, r.Reason);
        Assert.Null(r.Plan);
        Assert.Equal(4, r.Chunks.Miss); // still counted (no store: every located chunk is a miss)

        var strict = Apply(Body(Conversation(User), policy, AllChunks(), """{"strict":true}""")).Plan;
        Assert.Equal(new ChunkCacheError(400, "invalid_request_error", ChunkCacheCodes.PolicyUnavailable, strict.Error!.Message), strict.Error);

        foreach (string same in new[] { "exact", "shadow" })
            Assert.Null(Report(Apply(Body(Conversation(User), same, AllChunks())).Plan).Reason); // ruling 6
    }

    /// <summary>Recurrent / hybrid models (and other unsupported cases) give <c>chunk_model_unsupported</c>: every annotation ignored, exact served.</summary>
    [Fact]
    public void RecurrentModel_ModelUnsupported()
    {
        // What the server classifies as unsupported.
        // (States not disposed: they borrow the fixture's model and GGUF.)
        Assert.Equal("recurrent or hybrid model", ChunkCacheRuntime.UnsupportedReason(
            new ServerState { Options = new ServerOptions { Model = "m" }, Model = new RecurrentModel(), Config = new RecurrentModel().Config }));
        Assert.Equal("backend without chunk assembly", ChunkCacheRuntime.UnsupportedReason(
            new ServerState { Options = new ServerOptions { Model = "m", Device = "cuda:0" }, Model = _fx.Model, Config = _fx.Config }));
        Assert.Null(ChunkCacheRuntime.UnsupportedReason(
            new ServerState { Options = new ServerOptions { Model = "m" }, Model = _fx.Model, Config = _fx.Config, CurrentGguf = _fx.Gguf }));

        var unsupported = ServerModelIdentity.Unsupported("recurrent or hybrid model");
        foreach (string policy in new[] { "exact", "shadow" })
        {
            var plan = Apply(Body(Conversation(User), policy, AllChunks()), _fx.Tok, _fx.Template, unsupported, null).Plan;
            Assert.Empty(plan.Spans);
            Assert.False(plan.Shadow);
            Assert.Equal([$"{ChunkCacheCodes.ModelUnsupported}x1"], Codes(plan.Warnings));
            Assert.Equal(policy == "exact" ? null : ChunkCacheCodes.ModelUnsupported, plan.Reason);
            var r = Report(plan);
            Assert.Equal(new ChunkCacheChunksDto { Annotated = 4, Ignored = 4 }, r.Chunks);
            AssertPartition(r);
        }
        var strict = Apply(Body(Conversation(User), "exact", AllChunks(), """{"strict":true}"""), _fx.Tok, _fx.Template, unsupported, null).Plan;
        Assert.Equal(ChunkCacheCodes.ModelUnsupported, strict.Error!.Code);
        Assert.Equal(400, strict.Error.Status);
    }

    private sealed class RecurrentModel : IModel
    {
        public ModelConfig Config { get; } = new ServerPinHttpTests.NonComputingModel(16).Config;
        public long ComputeMemoryBytes => 0;
        public bool RequiresPerSequenceState => true;
        public DotLLM.Core.Tensors.ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => throw new NotSupportedException();
        public DotLLM.Core.Tensors.ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, DotLLM.Core.Attention.IKvCache? kvCache) => throw new NotSupportedException();
        public void Dispose() { }
    }

    /// <summary><c>min_chunk_tokens</c> (default 32, counted with the server tokenizer): a shorter span is plain text with a warning.</summary>
    [Fact]
    public void ShortChunk_IgnoredTooShortWarning()
    {
        string s31 = new string('x', 30) + "\n"; // 31 byte-level tokens
        string s32 = new string('y', 31) + "\n"; // 32
        string user = "a:\n" + s31 + "b:\n" + s32 + "end";
        var plan = Apply(Body(Conversation(user), "exact", [(1, s31), (1, s32)])).Plan;
        Assert.Equal(31, ChunkTokenization.Encode(_fx.Tok, s31).Length);
        Assert.Single(plan.Spans);
        Assert.Equal(32, plan.Spans[0].TokenCount);
        Assert.Equal([$"{ChunkCacheCodes.IgnoredTooShort}x1"], Codes(plan.Warnings));
        Assert.Equal(1, Report(plan).Chunks.Ignored);
        Assert.Null(Apply(Body(Conversation(user), "exact", [(1, s31)], """{"strict":true}""")).Plan.Error); // warning only

        var lower = Apply(Body(Conversation(user), "exact", [(1, s31), (1, s32)], """{"min_chunk_tokens":8}""")).Plan;
        Assert.Equal(2, lower.Spans.Count);
        Assert.Empty(lower.Warnings);
    }

    // ------------------------------------------------------------------------------------------------ token-level verification

    /// <summary>
    /// Equal text is not equal tokens at a span's edge: a span glued to the text before it with no newline, where the
    /// tokenizer merges across the edge, is located by characters but its ids differ from the prompt's, so it is
    /// <c>chunk_template_altered</c> and not counted. After a newline glue the same text is verified.
    /// </summary>
    [Fact]
    public void Span_TokenIdsDifferAtBoundary_NotCountedReusable()
    {
        var (tok, template) = MergingTokenizer();
        string span = "some code that is long enough to count as a chunk\n";
        Assert.True(ChunkTokenization.Encode(tok, span).Length >= 32);
        var identity = _fx.Identity;

        string glued = "head:" + span + "tail";
        var g = Apply(Body(Conversation(glued), "shadow", [(1, span)]), tok, template, identity, null);
        Assert.Empty(g.Plan.Spans);
        Assert.Equal([$"{ChunkCacheCodes.TemplateAltered}x1"], Codes(g.Plan.Warnings));
        Assert.Equal(1, Report(g.Plan).Chunks.Ignored);
        Assert.Equal(ChunkCacheCodes.TemplateAltered,
            Apply(Body(Conversation(glued), "shadow", [(1, span)], """{"strict":true}"""), tok, template, identity, null).Plan.Error!.Code);
        // Strict precedence: a request-level defect (the policy) wins over a span-level one (the altered span).
        Assert.Equal(ChunkCacheCodes.PolicyUnavailable,
            Apply(Body(Conversation(glued), "balanced", [(1, span)], """{"strict":true}"""), tok, template, identity, null).Plan.Error!.Code);

        string clean = "head:\n" + span + "tail";
        var c = Apply(Body(Conversation(clean), "shadow", [(1, span)]), tok, template, identity, null);
        var planned = Assert.Single(c.Plan.Spans);
        Assert.Equal(ChunkTokenization.Encode(tok, span), c.Ids.AsSpan(planned.TokenStart, planned.TokenCount).ToArray());
        Assert.Empty(c.Plan.Warnings);
    }

    /// <summary>
    /// Ruling 38 / Task 11 review S2: a Qwen3 assistant span holding a live <c>&lt;think&gt;</c> is located by characters,
    /// but the prompt encodes the tag as its control token while the chunk encodes it as text: not reusable. The same text
    /// in a user message (where the template marks the spelling) is verified.
    /// </summary>
    [Fact]
    public void AssistantSpanWithThinkTag_VerifiedByIds()
    {
        string span = "<think> stays live here in the assistant turn, long enough\n";
        (string, string)[] assistant = [("user", "q1"), ("assistant", "ok:\n" + span + "end"), ("user", "q2")];
        var a = Apply(Body(assistant, "shadow", [(1, span)]));
        Assert.Empty(a.Plan.Spans);
        Assert.Equal([$"{ChunkCacheCodes.TemplateAltered}x1"], Codes(a.Plan.Warnings));
        int think = _fx.Tok.EncodeRaw("<think>").Single();
        Assert.Contains(think, a.Ids); // the live control token is in the prompt

        (string, string)[] user = [("user", "q1:\n" + span + "end")];
        var u = Apply(Body(user, "shadow", [(0, span)]));
        var planned = Assert.Single(u.Plan.Spans);
        Assert.DoesNotContain(think, u.Ids);
        Assert.Equal(ChunkTokenization.Encode(_fx.Tok, span), u.Ids.AsSpan(planned.TokenStart, planned.TokenCount).ToArray());
    }

    /// <summary>Warnings from the parser, the locator and the token check merge per code with counts (ruling 38).</summary>
    [Fact]
    public void Warnings_MergedPerCode_AcrossParserLocatorAndTokenCheck()
    {
        string marked = "has a caller marker \uFDD0 inside it, long enough to count\n";
        string assistantSpan = "<think> live in assistant content, long enough to count\n";
        (string, string)[] msgs = [("user", "a:\n" + Chunks[0] + "b:\n" + marked + "c:\n" + Chunks[1] + "end"), ("assistant", "x\n" + assistantSpan + "y"), ("user", "q")];
        var body = JsonNode.Parse(Body(msgs, "exact", [(0, Chunks[0]), (0, marked), (0, Chunks[1]), (1, assistantSpan)]))!;
        body["chunk_cache"]!["chunks"]![2]!["hash"] = ChunkAnnotationParserTests.Sha("wrong");
        var plan = Apply(body.ToJsonString()).Plan;
        Assert.Single(plan.Spans);
        Assert.Equal([$"{ChunkCacheCodes.HashMismatch}x1", $"{ChunkCacheCodes.TemplateAltered}x2"], Codes(plan.Warnings));
        var r = Report(plan);
        Assert.Equal(new ChunkCacheChunksDto { Annotated = 4, Miss = 1, Ignored = 3 }, r.Chunks);
        Assert.Equal([ChunkCacheCodes.HashMismatch, ChunkCacheCodes.TemplateAltered], r.Warnings.Select(w => w.Code));
    }

    /// <summary>A configured but unusable store: every chunk a miss, one <c>chunk_store_unavailable</c>; strict is a 503 api_error.</summary>
    [Fact]
    public void StoreUnavailable_MissesAndWarning_StrictIs503()
    {
        var plan = Apply(Body(Conversation(User), "shadow", AllChunks()), store: null, storeUnavailable: true).Plan;
        Assert.All(plan.Spans, s => Assert.False(s.Hit));
        Assert.Equal(4, plan.Spans.Count);
        Assert.Equal([$"{ChunkCacheCodes.StoreUnavailable}x1"], Codes(plan.Warnings));
        var strict = Apply(Body(Conversation(User), "shadow", AllChunks(), """{"strict":true}"""), store: null, storeUnavailable: true).Plan;
        Assert.Equal(new ChunkCacheError(503, "api_error", ChunkCacheCodes.StoreUnavailable, strict.Error!.Message), strict.Error);
        Assert.Empty(Apply(Body(Conversation(User), "shadow", AllChunks())).Plan.Warnings); // no store configured: no warning
    }

    /// <summary>The namespace is part of the key: an entry stored under another namespace is a miss.</summary>
    [Fact]
    public void Lookup_UsesTheCallersNamespace()
    {
        var store = new FileChunkStore(_fx.NewDir("store"));
        _fx.Put(store, Chunks[0], ns: "k-0123456789abcdef");
        Assert.All(Apply(Body(Conversation(User), "shadow", AllChunks()), store).Plan.Spans, s => Assert.False(s.Hit));
        _fx.Put(store, Chunks[0]);
        Assert.True(Apply(Body(Conversation(User), "shadow", AllChunks()), store).Plan.Spans[0].Hit);
    }

    /// <summary>
    /// Bounded work: a thousand spans in a long prompt cost a bounded number of tokenizer characters (one pass over the
    /// glue and the spans, plus a capped number of prefix encodes), whatever the layout.
    /// </summary>
    [Fact]
    public void TokenCheck_CostBound_ManySpans()
    {
        var sb = new StringBuilder("files:\n");
        var spans = new List<(int, string)>();
        for (int i = 0; i < 1000; i++)
        {
            string s = $"chunk {i:D5} with enough characters to pass the minimum\n";
            sb.Append(i % 2 == 0 ? "glue\n" : "glue:"); // half the spans glued without a newline
            sb.Append(s);
            spans.Add((1, s));
        }
        string user = sb.Append("end").ToString();
        var counting = new CountingTokenizer(_fx.Tok);
        var run = Apply(Body(Conversation(user), "shadow", spans.ToArray()), counting, _fx.Template, _fx.Identity, null);
        Assert.Equal(1000, run.Plan.Spans.Count); // byte-level: no merges, all verified (under MaxPlannedSpans)
        long promptChars = run.Prompt.Length;
        Assert.True(counting.EncodedChars <= 8 * promptChars + 200_000, $"{counting.EncodedChars} chars encoded for a {promptChars}-char prompt");

        // Worst case: every span glued so the tokenizer merges across its first edge (":" + "s"): the glue never matches
        // alone, every span needs the prefix fallback, and the prefix encodes stop at the cap (those spans: unverified).
        var (merging, template) = MergingTokenizer();
        var glued = new StringBuilder("files:\n");
        var gluedSpans = new List<(int, string)>();
        for (int i = 0; i < 1000; i++)
        {
            string s = $"span {i:D5} with enough characters to pass the minimum\n";
            glued.Append("glue:").Append(s);
            gluedSpans.Add((1, s));
        }
        var counting2 = new CountingTokenizer(merging);
        var worst = Apply(Body(Conversation(glued.Append("end").ToString()), "shadow", gluedSpans.ToArray()), counting2, template, _fx.Identity, null);
        Assert.Empty(worst.Plan.Spans);
        Assert.True(counting2.EncodedChars <= 8L * worst.Prompt.Length + 200_000, $"{counting2.EncodedChars} chars encoded for a {worst.Prompt.Length}-char prompt");
    }

    /// <summary>Never throws on content: a tokenizer failure while planning ignores every annotation.</summary>
    [Fact]
    public void Exchange_FailureWhilePlanning_IgnoresAnnotations_NeverThrows()
    {
        var state = new ServerState // not disposed: it borrows the fixture's model and GGUF
        {
            Options = new ServerOptions { Model = "m", ChunkCache = new ChunkCacheOptions(true, 4096) },
            Model = _fx.Model,
            Config = _fx.Config,
            Tokenizer = new ThrowingTokenizer(_fx.Tok),
            ChatTemplate = _fx.Template,
            CurrentGguf = _fx.Gguf,
        };
        var request = JsonSerializer.Deserialize(Body(Conversation(User), "shadow", AllChunks()), ServerJsonContext.Default.ChatCompletionRequest)!;
        var exchange = ChunkCacheExchange.Begin(ChunkAnnotationParser.Parse(request, state.Options.ChunkCache, ChunkNamespace.Default))!;
        var messages = RequestConverter.ToMessages(request.Messages!);
        var options = ReasoningSupport.BuildTemplateOptions(null, null, null, null, false);
        Assert.True(ReasoningSupport.TryApply(_fx.Template, messages, options, out string prompt, out _, out _));
        Assert.Null(exchange.Prepare(state, messages, 0, options, prompt));
        var report = exchange.Finish(100, null)!;
        Assert.Equal(new ChunkCacheChunksDto { Annotated = 4, Ignored = 4 }, report.Chunks);
        Assert.Same(report, exchange.Finish(100, null)); // built once
    }

    // ------------------------------------------------------------------------------------------------ fix round (review)

    /// <summary>Ruling 50: the plan's budget is 0.05 by default; <c>recompute.budget</c> overrides it; an invalid one is malformed.</summary>
    [Fact]
    public void Plan_Budget_DefaultsTo005_RequestOverrides_InvalidIsMalformed()
    {
        var store = new FileChunkStore(_fx.NewDir("store"));
        _fx.Put(store, Chunks[0]);
        var def = Report(Apply(Body(Conversation(User), "shadow", AllChunks()), store).Plan);
        Assert.Equal(0.05, def.Plan!.Budget);
        int n0 = ChunkTokenization.Encode(_fx.Tok, Chunks[0]).Length;
        Assert.Equal((int)Math.Ceiling(n0 * 0.05), def.Plan.Tokens.ChunkRecomputed);

        var run = Apply(Body(Conversation(User), "shadow", AllChunks(), """{"recompute":{"selector":"chunk_starts","budget":0.3}}"""), store);
        Assert.Empty(run.Plan.Warnings);
        var over = Report(run.Plan);
        Assert.Equal(0.3, over.Plan!.Budget);
        Assert.Equal((int)Math.Ceiling(n0 * 0.3), over.Plan.Tokens.ChunkRecomputed);
        Assert.Equal(0, Report(Apply(Body(Conversation(User), "shadow", AllChunks(), """{"recompute":{"budget":0}}"""), store).Plan).Plan!.Tokens.ChunkRecomputed);

        foreach (string bad in new[] { """{"recompute":{"budget":1.5}}""", """{"recompute":{"budget":-0.1}}""", """{"recompute":{"budget":"x"}}""", """{"recompute":5}""" })
        {
            var p = Apply(Body(Conversation(User), "shadow", AllChunks(), bad), store).Plan;
            Assert.Equal([$"{ChunkCacheCodes.AnnotationMalformed}x1"], Codes(p.Warnings));
            Assert.Equal(0.05, Report(p).Plan!.Budget);
        }
        var request = JsonSerializer.Deserialize(Body(Conversation(User), "shadow", AllChunks(), """{"strict":true,"recompute":{"budget":2}}"""),
            ServerJsonContext.Default.ChatCompletionRequest)!;
        Assert.Equal(ChunkCacheCodes.AnnotationMalformed, ChunkAnnotationParser.Parse(request, new ChunkCacheOptions(true, 4096), "")!.Error!.Code);
    }

    /// <summary>
    /// Ruling 48: <c>full_prefill_estimate</c> estimates the unannotated prefill of the non-prefix rows. Under exact it equals
    /// the measured prefill when the request prefilled something; with nothing measured it is the backend's default rate.
    /// </summary>
    [Fact]
    public void FullPrefillEstimate_IsAnEstimate_OfTheNonPrefixRows()
    {
        var plan = Apply(Body(Conversation(User), "shadow", AllChunks())).Plan;
        var cost = ChunkCostModel.For("cpu");
        var measured = Report(plan, cached: 10, prefillMs: 250);
        Assert.Equal(250, measured.TimingsMs.FullPrefillEstimate);
        Assert.Equal(250, measured.Plan!.FullPrefillEstimateMs);
        var unmeasured = Report(plan, cached: 10, prefillMs: 0);
        long expected = (long)Math.Round((plan.PromptTokens - 10) * 1000.0 / cost.PrefillTokensPerSec);
        Assert.Equal(expected, unmeasured.TimingsMs.FullPrefillEstimate);
        Assert.Equal(0, unmeasured.TimingsMs.Prefill);
    }

    /// <summary>
    /// SentencePiece with a dummy space prefix (<c>add_dummy_prefix</c>): every glue encoded alone gains a leading piece, so
    /// the glue path must tolerate it. 150 valid spans (each starting with a space, as such a tokenizer needs) are all
    /// verified, with no prefix encode after the first span, and a strict request is not an error.
    /// </summary>
    [Fact]
    public void SentencePieceLeadingSpace_ManyValidSpans_AllVerified_StrictOk()
    {
        var (tok, template) = MergingTokenizer(addBosSpace: true);
        var sb = new StringBuilder("files:\n");
        var spans = new List<(int, string)>();
        for (int i = 0; i < 150; i++)
        {
            string span = $" span {i:D4} that is long enough to count as a whole chunk\n";
            sb.Append(i % 3 == 0 ? "glue\n" : i % 3 == 1 ? "// next file\n" : "").Append(span);
            spans.Add((1, span));
        }
        string user = sb.Append("end").ToString();
        var counting = new CountingTokenizer(tok);
        var run = Apply(Body(Conversation(user), "shadow", spans.ToArray(), """{"strict":true}"""), counting, template, _fx.Identity, null);
        Assert.Null(run.Plan.Error);
        Assert.Empty(run.Plan.Warnings);
        Assert.Equal(150, run.Plan.Spans.Count);
        foreach (var (planned, (_, text)) in run.Plan.Spans.Zip(spans))
            Assert.Equal(ChunkTokenization.Encode(tok, text), run.Ids.AsSpan(planned.TokenStart, planned.TokenCount).ToArray());
        Assert.True(counting.EncodedChars <= 3L * run.Prompt.Length, $"{counting.EncodedChars} chars encoded for a {run.Prompt.Length}-char prompt");
    }

    /// <summary>
    /// Ruling 49: valid spans the check cannot place once its prefix budget is spent are ignored with
    /// <c>chunk_verification_skipped</c>, never <c>chunk_template_altered</c>, and a strict request is not an error. (A
    /// tokenizer whose glue never encodes alone as it does in the prompt forces the prefix fallback at every span.)
    /// </summary>
    [Fact]
    public void ValidSpansPastTheVerificationBudget_Skipped_NeverAltered_StrictOk()
    {
        var tok = new GlueBreakingTokenizer(_fx.Tok);
        var sb = new StringBuilder("files:\n");
        var spans = new List<(int, string)>();
        for (int i = 0; i < 300; i++)
        {
            string span = $"chunk {i:D4} that is long enough to count as a whole chunk with padding text\n";
            sb.Append("glue line ").Append(i).Append('\n').Append(span);
            spans.Add((1, span));
        }
        var run = Apply(Body(Conversation(sb.Append("end").ToString()), "shadow", spans.ToArray(), """{"strict":true}"""), tok, _fx.Template, _fx.Identity, null);
        Assert.Null(run.Plan.Error);
        int verified = run.Plan.Spans.Count;
        Assert.InRange(verified, 1, 299); // the budget verified some and skipped the rest
        Assert.Equal([$"{ChunkCacheCodes.VerificationSkipped}x{300 - verified}"], Codes(run.Plan.Warnings));
        var r = Report(run.Plan);
        Assert.Equal(300 - verified, r.Chunks.Ignored);
        AssertPartition(r);
    }

    /// <summary>Bounded planning: at most <see cref="ChunkCachePipeline.MaxPlannedSpans"/> spans are verified and looked up per
    /// request; the rest are <c>chunk_verification_skipped</c> (warning only).</summary>
    [Fact]
    public void ManySpans_CappedAtMaxPlannedSpans_RestSkipped()
    {
        int n = ChunkCachePipeline.MaxPlannedSpans + 40;
        var sb = new StringBuilder("files:\n");
        var spans = new List<(int, string)>();
        for (int i = 0; i < n; i++)
        {
            string span = $"chunk {i:D5} with enough characters to pass the minimum\n";
            sb.Append(span);
            spans.Add((1, span));
        }
        var store = new CountingStore();
        var run = Apply(Body(Conversation(sb.Append("end").ToString()), "exact", spans.ToArray(), """{"strict":true}"""), store);
        Assert.Null(run.Plan.Error);
        Assert.Equal(ChunkCachePipeline.MaxPlannedSpans, run.Plan.Spans.Count);
        Assert.Equal(ChunkCachePipeline.MaxPlannedSpans, store.ContainsCalls);
        Assert.Equal([$"{ChunkCacheCodes.VerificationSkipped}x40"], Codes(run.Plan.Warnings));
    }

    /// <summary>
    /// Ruling 4 (review S6): among span-level defects the strict error is the one of the span first in document order,
    /// whether the locator or the token check found it.
    /// </summary>
    [Fact]
    public void Strict_SpanLevelError_FollowsDocumentOrder()
    {
        var (tok, template) = MergingTokenizer();
        string altered = "some code that is long enough to count as a chunk\n"; // glued after ':' → ids differ
        string fine = "other code that is long enough to count as a chunk\n";
        string user = "head:" + altered + "x\n" + fine + "tail";
        string body = Body(Conversation(user), "shadow", [(1, altered), (1, fine)], """{"strict":true}""");
        var request = JsonSerializer.Deserialize(body, ServerJsonContext.Default.ChatCompletionRequest)!;
        var parsed = ChunkAnnotationParser.Parse(request, new ChunkCacheOptions(true, 4096), "")!;
        var messages = RequestConverter.ToMessages(request.Messages!);
        var options = ReasoningSupport.BuildTemplateOptions(null, null, null, null, false);
        Assert.True(ReasoningSupport.TryApply(template, messages, options, out string prompt, out _, out _));
        var located = ChunkSpanLocator.Locate(template, messages, options, prompt, parsed.Spans, 0);
        Assert.Equal(2, located.Located.Count);
        // Make the locator report the LATER span (document index 1) as invalid: the token-check defect of span 0 must win.
        var mixed = located with { Located = [located.Located[0]], Warnings = [new ChunkWarning(ChunkCacheCodes.SpanInvalid, 1)], Codes = [null, ChunkCacheCodes.SpanInvalid] };
        var plan = ChunkCachePipeline.Apply(parsed, mixed, _fx.Identity, tok, prompt, tok.Encode(prompt), null, false);
        Assert.Equal(ChunkCacheCodes.TemplateAltered, plan.Error!.Code);
        // And the other way round: a locator defect on span 0 wins over nothing later.
        var first = located with { Located = [located.Located[1]], Codes = [ChunkCacheCodes.SpanInvalid, null] };
        Assert.Equal(ChunkCacheCodes.SpanInvalid, ChunkCachePipeline.Apply(parsed, first, _fx.Identity, tok, prompt, tok.Encode(prompt), null, false).Error!.Code);
    }

    /// <summary>
    /// Ruling 4 with the REAL locator (not a hand-built <see cref="LocatedSpans"/>): a span the locator rejects is the strict
    /// error. <c>chunk_template_altered</c>: a span whose text carries a U+FDD0 marker (dropped at encode, so the text is not
    /// what the prompt encodes). <c>chunk_span_invalid</c>: a span whose engine message no longer holds its text where the
    /// parser placed it. Without strict, both are warnings and the request is served.
    /// </summary>
    [Fact]
    public void Strict_LocatorDefects_FromTheRealLocator_AreTheError()
    {
        // chunk_template_altered from the locator.
        string marked = "public int Marked => 1; // a line of code long enough ﷐ for a chunk\n";
        string user = "Files:\n" + marked + "Question?";
        var altered = Apply(Body(Conversation(user), "exact", [(1, marked)], """{"strict":true}""")).Plan;
        Assert.Equal(ChunkCacheCodes.TemplateAltered, altered.Error?.Code);
        Assert.Equal(400, altered.Error!.Status);
        Assert.Null(Apply(Body(Conversation(user), "exact", [(1, marked)])).Plan.Error);

        // chunk_span_invalid from the locator: the rendered messages no longer hold the span where the parser put it.
        string body = Body(Conversation(User), "exact", [(1, Chunks[1])], """{"strict":true}""");
        var request = JsonSerializer.Deserialize(body, ServerJsonContext.Default.ChatCompletionRequest)!;
        var parsed = ChunkAnnotationParser.Parse(request, new ChunkCacheOptions(true, 4096), ChunkNamespace.Default)!;
        Assert.Null(parsed.Error);
        var messages = RequestConverter.ToMessages(request.Messages!);
        messages[1] = messages[1] with { Content = messages[1].Content!.Replace(Chunks[1], Chunks[1].ToUpperInvariant(), StringComparison.Ordinal) };
        var options = ReasoningSupport.BuildTemplateOptions(null, null, null, null, false);
        Assert.True(ReasoningSupport.TryApply(_fx.Template, messages, options, out string prompt, out _, out _));
        var located = ChunkSpanLocator.Locate(_fx.Template, messages, options, prompt, parsed.Spans, 0);
        Assert.Equal(new string?[] { ChunkCacheCodes.SpanInvalid }, located.Codes);
        var plan = ChunkCachePipeline.Apply(parsed, located, _fx.Identity, _fx.Tok, prompt, _fx.Tok.Encode(prompt), null, false);
        Assert.Equal(ChunkCacheCodes.SpanInvalid, plan.Error?.Code);
        Assert.Equal(400, plan.Error!.Status);
    }

    /// <summary>Review S1: a failing identity computation is an unsupported model (warning, strict 400), never a throw, and is computed once per model.</summary>
    [Fact]
    public void IdentityFailure_IsModelUnsupported_CachedPerModel_NeverThrows()
    {
        var state = new ServerState // not disposed: it borrows the fixture's model and GGUF
        {
            Options = new ServerOptions { Model = "m", ChunkCache = new ChunkCacheOptions(true, 4096) },
            Model = _fx.Model,
            Config = _fx.Config,
            Tokenizer = _fx.Tok,
            ChatTemplate = _fx.Template,
            CurrentGguf = _fx.Gguf,
        };
        int calls = 0;
        state.ChunkCacheRuntime.IdentityFactory = (_, _, _) => { calls++; throw new KeyNotFoundException("shard"); };
        foreach (bool strict in new[] { false, true, false })
        {
            var request = JsonSerializer.Deserialize(Body(Conversation(User), "shadow", AllChunks(), strict ? """{"strict":true}""" : ""),
                ServerJsonContext.Default.ChatCompletionRequest)!;
            var exchange = ChunkCacheExchange.Begin(ChunkAnnotationParser.Parse(request, state.Options.ChunkCache, ChunkNamespace.Default))!;
            var messages = RequestConverter.ToMessages(request.Messages!);
            var options = ReasoningSupport.BuildTemplateOptions(null, null, null, null, false);
            Assert.True(ReasoningSupport.TryApply(_fx.Template, messages, options, out string prompt, out _, out _));
            var error = exchange.Prepare(state, messages, 0, options, prompt);
            Assert.Equal(strict ? ChunkCacheCodes.ModelUnsupported : null, error?.Code);
            Assert.Contains(exchange.Plan!.Warnings, w => w.Code == ChunkCacheCodes.ModelUnsupported);
        }
        Assert.Equal(1, calls);
    }

    private sealed class CountingStore : IChunkStore
    {
        public int ContainsCalls { get; private set; }
        public bool TryGet(ChunkKey key, out ChunkKvEntry entry) { entry = null!; return false; }
        public void Put(ChunkKey key, ChunkKvEntry entry) => throw new NotSupportedException("read-only");
        public bool Contains(ChunkKey key) { ContainsCalls++; return false; }
        public IEnumerable<ChunkKey> Keys() => [];
        public long TotalBytes => 0;
        public void Evict(ChunkKey key) => throw new NotSupportedException("read-only");
    }

    /// <summary>Encodes text that starts with <c>glue</c> with two extra leading ids when encoded alone: the glue path never matches.</summary>
    private sealed class GlueBreakingTokenizer(ITokenizer inner) : Delegating(inner)
    {
        public override int[] EncodeRaw(string text) =>
            text.StartsWith("glue", StringComparison.Ordinal) ? [7, 7, .. base.EncodeRaw(text)] : base.EncodeRaw(text);
    }

    // ------------------------------------------------------------------------------------------------ tokenizers

    /// <summary>The Task 9 SentencePiece vocabulary plus one merged piece <c>:s</c>, and the Qwen3 template over it.</summary>
    private static (BpeTokenizer, JinjaChatTemplate) MergingTokenizer(bool addBosSpace = false)
    {
        var tokens = new List<string> { "<unk>" };
        var types = new List<int> { 2 };
        for (char c = ' '; c <= '~'; c++)
        {
            tokens.Add(c == ' ' ? "▁" : c.ToString());
            types.Add(1);
        }
        tokens.Add("\n");
        types.Add(1);
        tokens.Add(":s");
        types.Add(1);
        int bos = -1, eos = -1;
        foreach (string s in ServerPinTests.Specials)
        {
            if (s == "<|begin_of_text|>") bos = tokens.Count;
            if (s == "<|im_end|>") eos = tokens.Count;
            tokens.Add(s);
            types.Add(3);
        }
        var scores = tokens.Select(t => t == ":s" ? 10f : 0f).ToArray();
        var tok = BpeTokenizer.CreateSentencePiece(tokens.ToArray(), scores, types.ToArray(), bos, eos, addBosSpace: addBosSpace);
        var data = new GgufTestData(version: 3);
        data.AddString("tokenizer.chat_template", ServerPinTests.FixtureTemplate(ServerPinTests.Qwen3Fixture));
        using var stream = new MemoryStream(data.Build());
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        var metadata = new GgufMetadata(GgufReader.ReadMetadata(reader, header));
        return (tok, GgufChatTemplateFactory.TryCreate(metadata, tok, Architecture.Llama)!);
    }

    private class Delegating(ITokenizer inner) : ITokenizer
    {
        public virtual int[] Encode(string text) => inner.Encode(text);
        public virtual int[] EncodeRaw(string text) => inner.EncodeRaw(text);
        public string MarkUntrustedSpecials(string text, IReadOnlyList<string>? exemptPrefixes = null) => inner.MarkUntrustedSpecials(text, exemptPrefixes);
        public string Decode(ReadOnlySpan<int> tokenIds) => inner.Decode(tokenIds);
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => inner.Decode(tokenIds, stripBosSpace);
        public string DecodeToken(int tokenId) => inner.DecodeToken(tokenId);
        public int VocabSize => inner.VocabSize;
        public int BosTokenId => inner.BosTokenId;
        public int EosTokenId => inner.EosTokenId;
        public int CountTokens(string text) => inner.CountTokens(text);
    }

    private sealed class CountingTokenizer(ITokenizer inner) : Delegating(inner)
    {
        public long EncodedChars { get; private set; }
        public override int[] Encode(string text) { EncodedChars += text.Length; return base.Encode(text); }
        public override int[] EncodeRaw(string text) { EncodedChars += text.Length; return base.EncodeRaw(text); }
    }

    private sealed class ThrowingTokenizer(ITokenizer inner) : Delegating(inner)
    {
        public override int[] Encode(string text) => throw new InvalidOperationException("boom");
    }
}
