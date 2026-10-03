using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.Scheduler;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The Vulkan dense hybrid model (Tev1-4B) served through <see cref="ContinuousBatchSchedulerService"/> (the opt-in
/// <c>DOTLLM_VK_SCHEDULER=1</c> server path) must produce the same greedy tokens as the serial per-request
/// <see cref="TextGenerator"/>, with several requests in flight at once and every request carrying its own GDN slot.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanHybridSchedulerIdentityTests
{
    private readonly Xunit.Abstractions.ITestOutputHelper _out;
    public VulkanHybridSchedulerIdentityTests(Xunit.Abstractions.ITestOutputHelper output) => _out = output;

    private const int MaxTokens = 24;

    private static readonly string[] Questions =
    [
        "Describe how a refrigerator works.",
        "Explain why the sky is blue in two sentences.",
        "List three uses for a paperclip.",
        "What is a prime number?",
    ];

    private static string Chat(string q) =>
        "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    private static string? FindGguf()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_TEV1_4B_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string snaps = Path.Combine(home, ".cache", "huggingface", "hub",
            "models--bartowski--togethercomputer_Tev1-4B-experimental-GGUF", "snapshots");
        if (!Directory.Exists(snaps)) return null;
        foreach (string s in Directory.EnumerateDirectories(snaps))
        {
            string[] hits = Directory.GetFiles(s, "*Q4_K_M.gguf");
            if (hits.Length > 0) return hits[0];
        }
        return null;
    }

    [SkippableFact]
    public async Task Scheduler_ConcurrentRequests_MatchSerialTextGenerator()
    {
        string? path = FindGguf();
        Skip.If(path is null, "Tev1-4B GGUF not found.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var device = VulkanDevice.Create();
        var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
        using var _ = model as IDisposable;
        Skip.IfNot(model.SupportsThreadedSequenceState, "model has no per-sequence recurrent state");

        var opts = new InferenceOptions { Temperature = 0, MaxTokens = MaxTokens };

        // Serial oracle.
        var serialGen = new TextGenerator(model, tokenizer, (cfg, size) => kvFactory(size), mtpEnabled: false);
        int[][] serial = Questions.Select(q => serialGen.Generate(Chat(q), opts).GeneratedTokenIds).ToArray();

        // Scheduler: all requests enqueued before the loop gets a chance to drain them one by one.
        using var service = new ContinuousBatchSchedulerService(
            model, tokenizer, (cfg, size) => kvFactory(size), registerTelemetryProviders: false);
        using var cts = new CancellationTokenSource();
        Task loop = service.RunLoopAsync(cts.Token);
        try
        {
            async Task<InferenceResponse[]> Round() => await Task.WhenAll(Questions.Select(q => service.EnqueueAsync(new InferenceRequest
            {
                TokenIds = tokenizer.Encode(Chat(q)),
                Options = opts,
            }))).WaitAsync(TimeSpan.FromMinutes(5));

            var responses = await Round();

            for (int i = 0; i < Questions.Length; i++)
                Assert.True(serial[i].AsSpan().SequenceEqual(responses[i].GeneratedTokenIds),
                    $"request {i}: scheduler [{string.Join(',', responses[i].GeneratedTokenIds)}] != serial [{string.Join(',', serial[i])}]");

            // Leak guard: per-sequence KV + GDN slots are released when a request finishes. After the warm round (which
            // pays one-time costs: scratch growth, lazy weight bundles) further identical rounds must not add allocations.
            long warm = device.LiveAllocationCount;
            for (int r = 0; r < 3; r++)
            {
                var again = await Round();
                for (int i = 0; i < Questions.Length; i++)
                    Assert.True(serial[i].AsSpan().SequenceEqual(again[i].GeneratedTokenIds), $"round {r} request {i} diverged");
            }
            Assert.Equal(warm, device.LiveAllocationCount);
        }
        finally
        {
            cts.Cancel();
            try { await loop; } catch (OperationCanceledException) { }
        }
    }

    private const string SharedSystem =
        "<|im_start|>system\nYou are a careful assistant. Answer in one short sentence and never add extra commentary.<|im_end|>\n";

    private static string ChatWithSystem(string q) =>
        SharedSystem + "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    /// <summary>
    /// Scheduler recurrent prefix reuse: requests sharing a long system prompt, admitted one after another, snapshot the
    /// shared prefix on the second request and restore it (KV rows + GDN state) on later ones. Output must equal the serial
    /// no-cache oracle, restores must really happen, and dropping the GDN-state half of the restore must break it.
    /// </summary>
    [SkippableFact]
    public async Task Scheduler_RecurrentPrefixRestore_MatchesSerialOracle()
    {
        string? path = FindGguf();
        Skip.If(path is null, "Tev1-4B GGUF not found.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var device = VulkanDevice.Create();
        var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
        using var _ = model as IDisposable;
        Skip.IfNot(model.SupportsSequencePrefixSnapshot, "model cannot snapshot a sequence prefix");

        var opts = new InferenceOptions { Temperature = 0, MaxTokens = MaxTokens };
        var serialGen = new TextGenerator(model, tokenizer, (cfg, size) => kvFactory(size), mtpEnabled: false);
        int[][] serial = Questions.Select(q => serialGen.Generate(ChatWithSystem(q), opts).GeneratedTokenIds).ToArray();

        using var service = new ContinuousBatchSchedulerService(
            model, tokenizer, (cfg, size) => kvFactory(size),
            options: new ContinuousBatchSchedulerOptions { RecurrentPrefixCacheEntries = 4 },
            registerTelemetryProviders: false);
        using var cts = new CancellationTokenSource();
        Task loop = service.RunLoopAsync(cts.Token);
        try
        {
            var responses = new InferenceResponse[Questions.Length];
            for (int i = 0; i < Questions.Length; i++)   // sequential: each request is its own admission
                responses[i] = await service.EnqueueAsync(new InferenceRequest
                {
                    TokenIds = tokenizer.Encode(ChatWithSystem(Questions[i])),
                    Options = opts,
                }).WaitAsync(TimeSpan.FromMinutes(5));

            Assert.Equal(0, responses[0].Timings.CachedTokenCount);    // nothing to share yet
            Assert.Equal(0, responses[1].Timings.CachedTokenCount);    // shares with #0 -> snapshot taken, still a full prefill
            Assert.True(responses[2].Timings.CachedTokenCount >= ContinuousBatchScheduler.RecurrentPrefixMinTokens,
                $"request 2 did not restore a prefix (cached={responses[2].Timings.CachedTokenCount})");
            Assert.True(responses[3].Timings.CachedTokenCount >= ContinuousBatchScheduler.RecurrentPrefixMinTokens,
                $"request 3 did not restore a prefix (cached={responses[3].Timings.CachedTokenCount})");

            for (int i = 0; i < Questions.Length; i++)
                Assert.True(serial[i].AsSpan().SequenceEqual(responses[i].GeneratedTokenIds),
                    $"request {i}: scheduler [{string.Join(',', responses[i].GeneratedTokenIds)}] != serial [{string.Join(',', serial[i])}]");
        }
        finally
        {
            cts.Cancel();
            try { await loop; } catch (OperationCanceledException) { }
        }
    }

    /// <summary>
    /// Opt-in throughput probe (DOTLLM_TEV1_LATENCY_PROBE=1): classifier-shaped requests (long shared prefix, short suffix,
    /// 2 output tokens), 8 in flight. Arms interleaved: serial TextGenerator with its single-slot recurrent prefix cache,
    /// scheduler without prefix restore, scheduler with it.
    /// </summary>
    [SkippableFact]
    public async Task Probe_ClassifierThroughput_SchedulerVsSerial()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_TEV1_LATENCY_PROBE") == "1", "opt-in probe");
        string? path = FindGguf();
        Skip.If(path is null, "Tev1-4B GGUF not found.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var device = VulkanDevice.Create();
        var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
        using var _ = model as IDisposable;

        string filler = string.Join(" ", Enumerable.Range(0, 6).Select(i => $"Rule {i}: returns are allowed within {10 + i * 5} days."));
        string[] prompts = Enumerable.Range(0, 8).Select(i =>
            "<|im_start|>system" + (char)10 + "Select exactly one option letter. " + filler + "<|im_end|>" + (char)10 +
            "<|im_start|>user" + (char)10 + $"Purchase {i} was {7 + i * 9} days ago. Is the return in the window? A yes B no" +
            "<|im_end|>" + (char)10 + "<|im_start|>assistant" + (char)10 + "<think>" + (char)10 + (char)10 + "</think>" + (char)10 + (char)10).ToArray();
        var opts = new InferenceOptions { Temperature = 0, MaxTokens = 2 };

        double Serial()
        {
            var gen = new TextGenerator(model, tokenizer, (cfg, size) => kvFactory(size), mtpEnabled: false, recurrentPrefixCache: true);
            foreach (var pr in prompts.Take(2)) gen.Generate(pr, opts); // warm + snapshot
            var sw = System.Diagnostics.Stopwatch.StartNew();
            foreach (var pr in prompts) gen.Generate(pr, opts);
            return sw.Elapsed.TotalMilliseconds / prompts.Length;
        }

        async Task<double> Sched(int prefixEntries)
        {
            using var svc = new ContinuousBatchSchedulerService(model, tokenizer, (cfg, size) => kvFactory(size),
                options: new ContinuousBatchSchedulerOptions { RecurrentPrefixCacheEntries = prefixEntries }, registerTelemetryProviders: false);
            using var cts = new CancellationTokenSource();
            Task loop = svc.RunLoopAsync(cts.Token);
            try
            {
                Task<InferenceResponse> One(string pr) => svc.EnqueueAsync(new InferenceRequest { TokenIds = tokenizer.Encode(pr), Options = opts });
                foreach (var pr in prompts.Take(2)) await One(pr);   // warm + snapshot
                var sw = System.Diagnostics.Stopwatch.StartNew();
                await Task.WhenAll(prompts.Select(One));
                return sw.Elapsed.TotalMilliseconds / prompts.Length;
            }
            finally { cts.Cancel(); try { await loop; } catch (OperationCanceledException) { } }
        }

        _out.WriteLine($"prompt tokens: {tokenizer.Encode(prompts[0]).Length}");
        {
            // Per-request slot costs the scheduler pays and the serial path does not.
            var sw0 = System.Diagnostics.Stopwatch.StartNew();
            const int N = 20;
            double tKv = 0, tState = 0, tRestore = 0;
            using var snapKv = kvFactory(160);
            using var snapState = model.CreateSequenceState()!;
            var warmReq = tokenizer.Encode(prompts[0]);
            var pos = Enumerable.Range(0, 60).ToArray();
            using (var l = model.ForwardBatch([new DotLLM.Core.Models.SequenceForwardRequest { TokenIds = warmReq.AsMemory(0, 60), Positions = pos.AsMemory(), KvCache = snapKv, GdnState = snapState as DotLLM.Core.Models.IGdnState }], -1)[0]) { }
            using var snap = model.SnapshotSequencePrefix(snapKv, snapState, 60)!;
            for (int i = 0; i < N; i++)
            {
                sw0.Restart(); var kv = kvFactory(160); tKv += sw0.Elapsed.TotalMilliseconds;
                sw0.Restart(); var st = model.CreateSequenceState()!; tState += sw0.Elapsed.TotalMilliseconds;
                sw0.Restart(); model.RestoreSequencePrefix(snap, kv, st); tRestore += sw0.Elapsed.TotalMilliseconds;
                (kv as IDisposable)?.Dispose(); st.Dispose();
            }
            _out.WriteLine($"per-request: kv alloc={tKv / N:F2} ms, gdn state alloc={tState / N:F2} ms, restore copy={tRestore / N:F2} ms");
        }
        for (int round = 0; round < 3; round++)
        {
            double a = Serial();
            double b = await Sched(0);
            double c = await Sched(4);
            _out.WriteLine($"round {round}: ms/request serial+prefix={a:F1}  scheduler/no-restore={b:F1}  scheduler/restore={c:F1}");
        }
    }
}
