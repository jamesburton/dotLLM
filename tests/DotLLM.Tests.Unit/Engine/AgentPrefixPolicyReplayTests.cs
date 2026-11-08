using System.Text;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.PromptCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #1654: a synthetic agent session (one long main conversation, two subagents that share its project instructions, six
/// unrelated short requests) replayed through a real small model on the CPU under <see cref="PrefixCachePolicy.Lru"/> and
/// <see cref="PrefixCachePolicy.Agent"/>. The measure is prefix-hit and prefill TOKEN counts per request (contention-proof);
/// wall time is printed but indicative only. Two caches: the live-KV <see cref="PrefixCache"/> on a dense model (the Vulkan /
/// CUDA / <c>--no-paged</c> serving path) and the single recurrent slot on a hybrid (the Bonsai / Qwen3.x hybrid path).
/// </summary>
public sealed class AgentPrefixPolicyReplayTests
{
    private readonly ITestOutputHelper _out;
    public AgentPrefixPolicyReplayTests(ITestOutputHelper output) => _out = output;

    private const string ProjectInstructions =
        "You are the dotLLM coding agent. Project rules: file-scoped namespaces; readonly record structs for small values; " +
        "Span over arrays in signatures; no managed arrays for tensor data (NativeMemory.AlignedAlloc, 64 bytes); every " +
        "unmanaged buffer wrapped in IDisposable; XML doc comments on public APIs; nullable enabled; prove each correctness " +
        "test can fail (mutation or substitution) and record how in the PR body; benchmark kernels before and after; never " +
        "commit model weights; one issue, one branch, one PR to dev. Reply with the next concrete step and nothing else.";

    private const string OtherInstructions = "You are a terse assistant for quick factual questions. Answer in one sentence.";

    private static readonly string[] MainTurns =
    [
        "Read src/DotLLM.Engine/PromptCache/PrefixCache.cs and tell me what the eviction rule is today.",
        "Tool result: PrefixCache.cs evicts the least recently used entry when at capacity (4 entries). Now design the own-length rule.",
        "Tool result: PrefixRetention.SelectVictim added with unit tests, 11 passing. Next: wire it into the recurrent snapshot cache.",
        "Tool result: RecurrentPrefixSnapshotCache now calls Victim() at four removal sites; build green. What should the docs say?",
        "Tool result: KV_CACHE.md section drafted. Review it for the save-point semantics and list anything missing.",
        "Tool result: review complete, two gaps fixed. Write the PR body with the token-count table and the mutants list.",
    ];

    private static readonly string[][] Subagents =
    [
        [
            "Subagent task: grep tests/ for every test that constructs a PrefixCache and list the files.",
            "Tool result: 4 files found: PrefixCacheTests.cs, TextGeneratorPrefixCacheRecurrentTests.cs, PagedKvCacheIntegrationTests.cs, PromptCacheEndpointTests.cs. Summarise the fixture shapes.",
        ],
        [
            "Subagent task: check docs/TELEMETRY.md for the instruments listed as not yet exposed and quote the line.",
            "Tool result: line 154 lists prefix-cache hit/miss by tier, snapshot bytes per tier, MTP drafted/accepted. Propose instrument names.",
        ],
    ];

    private static readonly string[] Unrelated =
    [
        "What is the capital of Portugal?",
        "How many bytes are in a kibibyte?",
        "Name one noble gas.",
        "What year did the first .NET Framework ship?",
        "Which planet is largest in the solar system?",
        "What does the acronym SIMD stand for?",
    ];

    private const string GenerationSuffix = "<|im_start|>assistant\n";

    private static string Render(string system, IReadOnlyList<(string Role, string Text)> turns, bool generationPrompt)
    {
        var sb = new StringBuilder();
        sb.Append("<|im_start|>system\n").Append(system).Append("<|im_end|>\n");
        foreach (var (role, text) in turns)
            sb.Append("<|im_start|>").Append(role).Append('\n').Append(text).Append("<|im_end|>\n");
        if (generationPrompt) sb.Append(GenerationSuffix);
        return sb.ToString();
    }

    private sealed record Step(string Name, string Thread, string System, string User);

    /// <summary>
    /// A tool result an agent turn carries (about 300 tokens): the asymmetry the own-length rule is built on is a conversation that
    /// is LONG relative to the side traffic between its turns. Without it a 240-token conversation facing 480 tokens of side
    /// requests has, by the rule, expired, and both policies evict it (measured: M3 reused 5 under LRU, 125 under agent). It is
    /// kept to about 300 tokens so that six turns still fit the live cache's 2048-token headroom (#1591): with 1.3k-token
    /// results the third turn no longer fitted the KV and missed on size under both policies (measured).
    /// </summary>
    private static string ToolResult(int turn) => "\n```\n" + string.Join("\n", Enumerable.Range(0, 9).Select(i =>
        $"[{turn}:{i:D2}] PrefixCacheEntry id={turn * 100 + i} length={1000 + i * 37} lastUse={i * 211} savePoint={(i % 3 == 0 ? -1 : i * 19)} tier=live")) + "\n```";

    /// <summary>The session: main turns interleaved with a subagent (2 turns) and 3 unrelated requests after turns 2 and 4.</summary>
    private static IEnumerable<Step> Session()
    {
        int sub = 0, other = 0;
        for (int m = 0; m < MainTurns.Length; m++)
        {
            yield return new($"M{m + 1}", "main", ProjectInstructions, MainTurns[m] + ToolResult(m + 1));
            if (m is 1 or 3)
            {
                for (int t = 0; t < 2; t++) yield return new($"S{sub + 1}{(char)('a' + t)}", $"sub{sub}", ProjectInstructions, Subagents[sub][t]);
                sub++;
                for (int u = 0; u < 3; u++, other++) yield return new($"U{other + 1}", $"other{other}", OtherInstructions, Unrelated[other]);
            }
        }
    }

    private sealed record Row(string Name, int PromptTokens, int Cached, int Prefilled, string? Tier, float FirstLogprob, string Text);

    /// <summary>Replays the session through <paramref name="generate"/>, feeding each thread's reply back as its assistant turn.</summary>
    private List<Row> Replay(Func<string, InferenceOptions, InferenceResponse> generate, DotLLM.Tokenizers.ITokenizer tokenizer, bool savePoints)
    {
        var threads = new Dictionary<string, List<(string Role, string Text)>>();
        var rows = new List<Row>();
        foreach (var step in Session())
        {
            var turns = threads.TryGetValue(step.Thread, out var t) ? t : threads[step.Thread] = [];
            turns.Add(("user", step.User));
            string prompt = Render(step.System, turns, generationPrompt: true);
            var opts = new InferenceOptions
            {
                Temperature = 0f, MaxTokens = 24, Logprobs = true, TopLogprobs = 1,
                GenerationPromptSuffix = GenerationSuffix,
                SavePointPrefix = savePoints && turns.Count > 1 ? Render(step.System, turns.Take(turns.Count - 1).ToList(), generationPrompt: false) : null,
            };
            var r = generate(prompt, opts);
            int promptTokens = tokenizer.Encode(prompt).Length;
            string reply = r.Text.Trim();
            if (reply.Length == 0) reply = "ok";
            turns.Add(("assistant", reply));
            rows.Add(new Row(step.Name, promptTokens, r.Timings.CachedTokenCount, promptTokens - r.Timings.CachedTokenCount,
                r.Timings.CacheTier, r.Logprobs is { Length: > 0 } lp ? lp[0].Logprob : float.NaN, reply));
        }
        return rows;
    }

    private void Print(string title, List<Row> lru, List<Row> agent, double lruSeconds, double agentSeconds)
    {
        _out.WriteLine($"{title}: prefix-hit / prefill tokens per request (LRU vs agent policy). Wall time indicative only.");
        _out.WriteLine($"{"req",-4} {"prompt",6} {"lru-hit",7} {"lru-pf",6} | {"prompt",6} {"ag-hit",6} {"ag-pf",6}  same-text");
        for (int i = 0; i < lru.Count; i++)
        {
            bool same = lru[i].Text == agent[i].Text;
            _out.WriteLine($"{lru[i].Name,-4} {lru[i].PromptTokens,6} {lru[i].Cached,7} {lru[i].Prefilled,6} | {agent[i].PromptTokens,6} {agent[i].Cached,6} {agent[i].Prefilled,6}  {(same ? "yes" : "NO")}");
        }
        _out.WriteLine($"total prefilled: lru {lru.Sum(r => r.Prefilled)} vs agent {agent.Sum(r => r.Prefilled)} tokens; " +
                       $"cached: lru {lru.Sum(r => r.Cached)} vs agent {agent.Sum(r => r.Cached)}; wall (indicative) lru {lruSeconds:F1} s, agent {agentSeconds:F1} s");
    }

    private static void AssertSession(List<Row> lru, List<Row> agent)
    {
        // M3 and M5 are the main conversation's first turns after a block of side requests. Under LRU the side requests took the
        // slot / truncated the entry, so they reuse at most the project instructions; under the agent policy they reuse the whole
        // previous turn.
        foreach (var (name, previous) in new[] { ("M3", "M2"), ("M5", "M4") })
        {
            var l = lru.Single(r => r.Name == name);
            var a = agent.Single(r => r.Name == name);
            int previousPrompt = agent.Single(r => r.Name == previous).PromptTokens;
            Assert.True(a.Cached >= previousPrompt, $"{name}: agent policy reused {a.Cached}/{a.PromptTokens}, less than the previous turn's {previousPrompt}-token prompt");
            Assert.True(l.Cached <= 0.5 * l.PromptTokens, $"{name}: LRU reused {l.Cached}/{l.PromptTokens} (the control arm did not reproduce the problem)");
        }
        Assert.True(agent.Sum(r => r.Prefilled) < lru.Sum(r => r.Prefilled), "the agent policy prefilled at least as many tokens as LRU");
    }

    /// <summary>
    /// Per-request first-token logprob deltas between the arms. Reported, not asserted: the two arms take different reuse paths
    /// for the same prompt (a resumed snapshot against a cold run), and whether THOSE agree is a property of the reuse paths, not
    /// of the policy; <see cref="Recurrent_AProtectedColdRun_MatchesAPlainGenerator"/> pins the policy's own path.
    /// </summary>
    private void ReportLogprobDeltas(List<Row> lru, List<Row> agent)
    {
        var deltas = lru.Zip(agent, (l, a) => (l.Name, Delta: Math.Abs(l.FirstLogprob - a.FirstLogprob), l.FirstLogprob, a.FirstLogprob))
            .Where(d => d.Delta > 1e-3f).ToList();
        _out.WriteLine(deltas.Count == 0
            ? "first-token logprobs agree within 1e-3 on every request"
            : "first-token logprob deltas > 1e-3 (lru vs agent): " + string.Join(", ", deltas.Select(d => $"{d.Name} {d.Item3:F4}/{d.Item4:F4}")));
    }

    [SkippableFact]
    [Trait("Category", "Model")]
    public void Recurrent_AProtectedColdRun_MatchesAPlainGenerator()
    {
        // The policy's only new execution path is the protected cold run (fresh KV, reset state). It must equal a generator with
        // no prefix cache at all. The LRU arm's resumed run of the same prompt is measured alongside for the record.
        string? path = TestFixtureResolver.ResolveFile("DOTLLM_TEV1_08B_GGUF", "DreamBlooms", "Tev1-0.8B-experimental-GGUF", "*Q8_0.gguf").Path;
        Skip.If(path is null, "Tev1-0.8B GGUF not found.");
        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config);
        var opts = new InferenceOptions { Temperature = 0f, MaxTokens = 8, Logprobs = true, TopLogprobs = 1, GenerationPromptSuffix = GenerationSuffix };

        var plain = new TextGenerator(model, tokenizer);
        string m1 = Render(ProjectInstructions, [("user", MainTurns[0] + ToolResult(1))], true);
        string s1a = Render(ProjectInstructions, [("user", Subagents[0][0])], true);
        string s1aReply = plain.Generate(s1a, opts).Text.Trim();
        string s1b = Render(ProjectInstructions, [("user", Subagents[0][0]), ("assistant", s1aReply), ("user", Subagents[0][1])], true);
        float wantLp = plain.Generate(s1b, opts).Logprobs![0].Logprob;
        string wantText = plain.Generate(s1b, opts).Text;

        var agentGen = new TextGenerator(model, tokenizer, recurrentPrefixCache: true, recurrentPrefixCacheOptions: new() { Policy = PrefixCachePolicy.Agent });
        agentGen.Generate(m1, opts);
        agentGen.Generate(s1a, opts);                       // protected: cold
        var protectedRun = agentGen.Generate(s1b, opts);    // protected: cold
        Assert.Equal(2, agentGen.RecurrentPrefixStatistics.ProtectedMisses);
        agentGen.ClearRecurrentPrefixCache();

        var lruGen = new TextGenerator(model, tokenizer, recurrentPrefixCache: true, recurrentPrefixCacheOptions: new() { Policy = PrefixCachePolicy.Lru });
        lruGen.Generate(m1, opts);
        lruGen.Generate(s1a, opts);                         // misses, snapshots the shared system prompt
        var resumed = lruGen.Generate(s1b, opts);           // resumes at the system prompt
        lruGen.ClearRecurrentPrefixCache();

        _out.WriteLine($"S1b first-token logprob: plain {wantLp:F5}, protected cold {protectedRun.Logprobs![0].Logprob:F5}, " +
                       $"LRU resumed at {resumed.Timings.CachedTokenCount} tokens {resumed.Logprobs![0].Logprob:F5}; texts: plain '{wantText}' | cold '{protectedRun.Text}' | resumed '{resumed.Text}'");
        Assert.Equal(0, protectedRun.Timings.CachedTokenCount);
        Assert.True(Math.Abs(wantLp - protectedRun.Logprobs![0].Logprob) < 1e-3f, $"protected cold run {protectedRun.Logprobs![0].Logprob} vs plain {wantLp}");
        Assert.Equal(wantText, protectedRun.Text);
    }

    [SkippableFact]
    [Trait("Category", "Model")]
    public void Dense_LiveKvPrefixCache_AgentPolicy_KeepsTheMainConversationAcrossSideRequests()
    {
        string? path = TestFixtureResolver.ResolveFile("DOTLLM_QWEN25_05B_GGUF", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "*q8_0.gguf").Path;
        Skip.If(path is null, "Qwen2.5-0.5B-Instruct q8_0 GGUF not found.");
        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var lruCache = new PrefixCache(4, PrefixCachePolicy.Lru);
        // Headroom as the Vulkan live path gives (#1591): without it a cache sized to prompt + max_tokens never holds the next turn.
        var lruGen = new TextGenerator(model, tokenizer, prefixCache: lruCache, prefixCacheHeadroomTokens: 2048);
        var lru = Replay((p, o) => lruGen.Generate(p, o), tokenizer, savePoints: false);
        double lruSeconds = sw.Elapsed.TotalSeconds; sw.Restart();
        using var agentCache = new PrefixCache(4, PrefixCachePolicy.Agent);
        var agentGen = new TextGenerator(model, tokenizer, prefixCache: agentCache, prefixCacheHeadroomTokens: 2048);
        var agent = Replay((p, o) => agentGen.Generate(p, o), tokenizer, savePoints: true);
        double agentSeconds = sw.Elapsed.TotalSeconds;

        Print("Qwen2.5-0.5B q8_0, live PrefixCache(4)", lru, agent, lruSeconds, agentSeconds);
        _out.WriteLine($"agent cache: hits {agentCache.Hits}, misses {agentCache.Misses}, protected {agentCache.ProtectedMisses}");
        ReportLogprobDeltas(lru, agent);
        AssertSession(lru, agent);
        Assert.True(agentCache.ProtectedMisses >= 2, "no side request was protected");
    }

    [SkippableFact]
    [Trait("Category", "Model")]
    public void Recurrent_SingleSlot_AgentPolicy_KeepsTheMainConversationAcrossSideRequests()
    {
        string? path = TestFixtureResolver.ResolveFile("DOTLLM_TEV1_08B_GGUF", "DreamBlooms", "Tev1-0.8B-experimental-GGUF", "*Q8_0.gguf").Path;
        Skip.If(path is null, "Tev1-0.8B GGUF not found.");
        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var lruGen = new TextGenerator(model, tokenizer, recurrentPrefixCache: true, recurrentPrefixCacheOptions: new() { Policy = PrefixCachePolicy.Lru });
        var lru = Replay((p, o) => lruGen.Generate(p, o), tokenizer, savePoints: false);
        double lruSeconds = sw.Elapsed.TotalSeconds; sw.Restart();
        lruGen.ClearRecurrentPrefixCache();
        var agentGen = new TextGenerator(model, tokenizer, recurrentPrefixCache: true, recurrentPrefixCacheOptions: new() { Policy = PrefixCachePolicy.Agent });
        var agent = Replay((p, o) => agentGen.Generate(p, o), tokenizer, savePoints: true);
        double agentSeconds = sw.Elapsed.TotalSeconds;

        Print("Tev1-0.8B Q8_0, recurrent single slot", lru, agent, lruSeconds, agentSeconds);
        var stats = agentGen.RecurrentPrefixStatistics;
        _out.WriteLine($"agent slot: hits {stats.Hits}, misses {stats.Misses}, protected {stats.ProtectedMisses}, step checkpoints {stats.StepCheckpoints}");
        agentGen.ClearRecurrentPrefixCache();
        ReportLogprobDeltas(lru, agent);
        AssertSession(lru, agent);
        Assert.True(stats.ProtectedMisses >= 2, "no side request was protected");
    }
}
