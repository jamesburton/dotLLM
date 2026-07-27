using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Exact-token greedy-decode parity harness for the shared MMVQ activation
/// quant (issue #150): runs prefill + N greedy decode steps TWICE on the same
/// GGUF — once with sharing on (default) and once with
/// <c>DOTLLM_VULKAN_MMVQ_NO_SHARE=1</c> — and asserts the generated token-id
/// sequences are IDENTICAL. Env-gated so it does not add a large-model decode
/// to the default sweep; pointed at real 3B/8B GGUFs from the perf wave.
/// </summary>
/// <remarks>
/// Env vars:
/// <list type="bullet">
///   <item><c>DOTLLM_VULKAN_SHARE_GREEDY=1</c> — required to run.</item>
///   <item><c>DOTLLM_VULKAN_SHARE_GREEDY_MODEL</c> — GGUF path (default: the
///     SmolLM-135M.Q8_0 fixture).</item>
///   <item><c>DOTLLM_VULKAN_SHARE_GREEDY_STEPS</c> — greedy decode steps
///     (default 128).</item>
/// </list>
/// </remarks>
[Collection("SmallModel")]
[Trait("Category", "GPU")]
public sealed class VulkanMmvqShareGreedyParityHarness
{
    private readonly SmallModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public VulkanMmvqShareGreedyParityHarness(SmallModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [SkippableFact]
    public void GreedyTokens_IdenticalSharedVsUnshared()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_SHARE_GREEDY") == "1",
            "DOTLLM_VULKAN_SHARE_GREEDY=1 not set.");
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");

        string spvDir = ResolveSpvDir();
        string modelPath = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_SHARE_GREEDY_MODEL") is { Length: > 0 } mp
            ? mp : _fixture.FilePath;
        int steps = int.TryParse(
            Environment.GetEnvironmentVariable("DOTLLM_VULKAN_SHARE_GREEDY_STEPS"), out int s) && s > 0 ? s : 128;
        _output.WriteLine($"model={modelPath} steps={steps}");

        int[] prompt;
        using (var gguf = GgufFile.Open(modelPath))
        {
            var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            prompt = tokenizer.Encode(
                "Write a short story about a lighthouse keeper who discovers a mysterious ship.").ToArray();
        }
        Assert.NotEmpty(prompt);

        int[] shared = RunGreedy(spvDir, modelPath, prompt, steps, noShare: false);
        int[] unshared = RunGreedy(spvDir, modelPath, prompt, steps, noShare: true);

        _output.WriteLine($"shared:   [{string.Join(",", shared)}]");
        _output.WriteLine($"unshared: [{string.Join(",", unshared)}]");
        Assert.Equal(unshared.Length, shared.Length);
        for (int i = 0; i < shared.Length; i++)
            Assert.True(shared[i] == unshared[i],
                $"Greedy token divergence at step {i}: shared={shared[i]}, unshared={unshared[i]}.");
        _output.WriteLine($"exact-token parity over {steps} greedy steps: OK");
    }

    private int[] RunGreedy(string spvDir, string modelPath, int[] prompt, int steps, bool noShare)
    {
        string? original = Environment.GetEnvironmentVariable(VulkanTransformerModel.MmvqNoShareEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(
                VulkanTransformerModel.MmvqNoShareEnvVar, noShare ? "1" : null);

            using var gguf = GgufFile.Open(modelPath);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
            using var cache = model.CreateKvCache(maxSeqLen: prompt.Length + steps + 8);

            var generated = new int[steps];
            int[] positions = new int[prompt.Length];
            for (int i = 0; i < prompt.Length; i++) positions[i] = i;

            int next = ArgmaxLastRow(model, prompt, positions, cache);
            int pos = prompt.Length;
            for (int t = 0; t < steps; t++)
            {
                generated[t] = next;
                next = ArgmaxLastRow(model, new[] { next }, new[] { pos }, cache);
                pos++;
            }
            return generated;
        }
        finally
        {
            Environment.SetEnvironmentVariable(VulkanTransformerModel.MmvqNoShareEnvVar, original);
        }
    }

    private static unsafe int ArgmaxLastRow(
        VulkanTransformerModel model, int[] tokenIds, int[] positions, VulkanKvCache cache)
    {
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, cache);
        int vocabSize = model.Config.VocabSize;
        int seqLen = logits.Shape[0];
        long lastRow = (long)(seqLen - 1) * vocabSize;
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, (int)logits.ElementCount)
            .Slice((int)lastRow, vocabSize);
        int arg = 0;
        for (int i = 1; i < span.Length; i++)
            if (span[i] > span[arg]) arg = i;
        return arg;
    }

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new Xunit.Sdk.XunitException("Compiled SPV shaders not found.");
    }
}
