using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Real-model validation of the Gemma-4 <b>E2B</b> GGUF (#923): its <c>feed_forward_length</c> is a PER-LAYER array
/// (6144 x15, 12288 x20: double-wide MLP on the shared-KV layers) which used to read as 0 and crash the CPU forward
/// (access violation) / fail the Vulkan load. E2B is the repo's fast small test model (~3 GB Q4_K_M, 7/7 on the
/// small-task suite, see ~/.claude/LLMs.md).
/// </summary>
/// <remarks>
/// Gated on <c>DOTLLM_GEMMA4E2B_GGUF</c> (path to the .gguf); falls back to the first
/// <c>gemma-4-E2B-it-Q4_K_M.gguf</c> found in the HF hub cache. Skipped when neither resolves.
/// </remarks>
[Collection(DotLLM.Tests.Integration.GpuCollection.Name)]
public sealed class Gemma4E2BGgufForwardTests
{
    private const string ModelPathEnvVar = "DOTLLM_GEMMA4E2B_GGUF";
    private const string FileName = "gemma-4-E2B-it-Q4_K_M.gguf";

    private readonly ITestOutputHelper _output;

    public Gemma4E2BGgufForwardTests(ITestOutputHelper output) => _output = output;

    private static string? TryResolveModelPath()
    {
        return DotLLM.Tests.Integration.Fixtures.TestFixtureResolver.ResolveFile(ModelPathEnvVar, "unsloth", "gemma-4-E2B-it-GGUF", FileName).Path;
    }

    // "The capital of France is" (llama.cpp ids for the gemma4 vocabulary).
    private static readonly int[] PromptIds = [2, 818, 5279, 529, 7001, 563];

    [SkippableFact]
    public void Gemma4E2B_Cpu_GreedyDecode_CompletesCapitalOfFranceWithParis()
    {
        string? path = TryResolveModelPath();
        Skip.If(path is null, $"Set {ModelPathEnvVar} to {FileName} to run this validation.");

        var (model, gguf, config) = ModelLoader.LoadFromGguf(path!);
        using var _ = gguf;
        using var __ = model;
        Assert.Equal(12288, config.IntermediateSize);

        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] generated = GreedyCpu(model, config, PromptIds, maxNew: 6);
        string completion = tokenizer.Decode(generated);
        _output.WriteLine($"[gemma4 E2B cpu] {completion}");
        Assert.Contains("Paris", completion, StringComparison.OrdinalIgnoreCase);
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void Gemma4E2B_Vulkan_LoadsAndMatchesCpuTop1()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string? path = TryResolveModelPath();
        Skip.If(path is null, $"Set {ModelPathEnvVar} to {FileName} to run this validation.");

        var (cpu, gguf, config) = ModelLoader.LoadFromGguf(path!);
        using var _ = gguf;
        using var __ = cpu;
        // Was: ArgumentOutOfRangeException ("Parameter 'bytes'") at load, from the zero-sized FFN scratch.
        using var vk = VulkanTransformerModel.LoadFromGguf(gguf, config);
        Assert.True(vk.Gemma4PleOnGpu);

        int[] pos = Enumerable.Range(0, PromptIds.Length).ToArray();
        int cpuTop, vkTop;
        using (ITensor l = cpu.Forward(PromptIds, pos, deviceId: -1))
            cpuTop = ArgMaxLastRow(l, config.VocabSize);
        using (ITensor l = vk.Forward(PromptIds, pos, deviceId: -1, kvCache: null))
            vkTop = ArgMaxLastRow(l, config.VocabSize);
        _output.WriteLine($"top-1 cpu={cpuTop} vulkan={vkTop}");
        Assert.Equal(cpuTop, vkTop);
    }

    private static int[] GreedyCpu(DotLLM.Core.Models.IModel model, DotLLM.Core.Models.ModelConfig config, int[] promptIds, int maxNew)
    {
        var generated = new List<int>();
        using var kv = new SimpleKvCache(KvGeometry.FromConfig(config), maxSeqLen: promptIds.Length + maxNew + 2);
        int[] pos = Enumerable.Range(0, promptIds.Length).ToArray();
        int next;
        using (ITensor logits = model.Forward(promptIds, pos, deviceId: -1, kvCache: kv))
            next = ArgMaxLastRow(logits, config.VocabSize);
        generated.Add(next);
        for (int i = 1; i < maxNew; i++)
        {
            using ITensor logits = model.Forward([generated[^1]], [promptIds.Length + i - 1], deviceId: -1, kvCache: kv);
            next = ArgMaxLastRow(logits, config.VocabSize);
            generated.Add(next);
        }
        return generated.ToArray();
    }

    private static unsafe int ArgMaxLastRow(ITensor logits, int vocab)
    {
        float* p = (float*)logits.DataPointer;
        long lastRow = (long)(logits.Shape[0] - 1) * vocab;
        int best = 0;
        float bestV = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            float x = p[lastRow + v];
            if (x > bestV) { bestV = x; best = v; }
        }
        return best;
    }
}
