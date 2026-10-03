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
}
