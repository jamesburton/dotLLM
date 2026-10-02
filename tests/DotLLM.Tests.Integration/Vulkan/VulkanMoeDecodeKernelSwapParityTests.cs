using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #639/#641: real-GGUF decode parity of the Qwen3.6-35B-A3B decode kernel swaps (F32 GEMV for the router / shared-expert matmuls, indexed
/// K-quant MMVQ for the routed experts) against the legacy one-thread-per-cell kernels (<c>DOTLLM_VK_F32_GEMV=0</c>,
/// <c>DOTLLM_VK_MOE_MMVQ=0</c>). Perplexity cannot see decode-path changes (it scores through prefill), so this is the only end-to-end oracle.
/// Self-skips unless the GGUF is staged; never downloads.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class VulkanMoeDecodeKernelSwapParityTests
{
    private const int DecodeSteps = 24;
    private readonly ITestOutputHelper _output;
    public VulkanMoeDecodeKernelSwapParityTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Qwen3MoeHybrid_DecodeKernelSwaps_MatchLegacyKernels_OnRealGguf()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available.");
        string? modelPath = ResolveModelPath();
        Skip.If(modelPath is null, "Qwen3.6-35B-A3B GGUF not staged (set DOTLLM_SPLIT_PARITY_MOE_GGUF); no download is triggered.");

        int[] prompt;
        int vocabSize;
        using (var probe = GgufFile.Open(modelPath!))
        {
            vocabSize = GgufModelConfigExtractor.Extract(probe.Metadata).VocabSize;
            prompt = GgufBpeTokenizerFactory.Load(probe.Metadata).Encode("The history of science is a long and winding road that")[..8];
        }

        var (legacyTokens, legacyLogits) = Run(modelPath!, legacy: true, prompt, vocabSize);
        var (newTokens, newLogits) = Run(modelPath!, legacy: false, prompt, vocabSize);
        _output.WriteLine($"legacy: {string.Join(",", legacyTokens)}");
        _output.WriteLine($"new   : {string.Join(",", newTokens)}");

        int firstDivergence = Array.FindIndex(legacyTokens.Zip(newTokens, (a, b) => a != b).ToArray(), d => d);
        _output.WriteLine(firstDivergence < 0 ? $"token-for-token match over {DecodeSteps} greedy steps." : $"first divergence at step {firstDivergence}.");

        float maxAbs = 0;
        for (int i = 0; i < legacyLogits.Length; i++)
        {
            Assert.True(float.IsFinite(newLogits[i]), $"non-finite logit at {i}");
            maxAbs = MathF.Max(maxAbs, MathF.Abs(newLogits[i] - legacyLogits[i]));
        }
        _output.WriteLine($"final-step logits L_inf = {maxAbs:G6}");
        Assert.True(firstDivergence < 0 || firstDivergence >= 8, $"decode diverged from the legacy kernels at step {firstDivergence}.");
        // Not bit-level: the Q6_K down bank used F32 activations before and now takes the int8 Q8_1 activations every other K-quant path uses,
        // so a sub-1 logit L_inf over a ~250k vocab is expected. This bound is a gross-corruption guard; the token match above is the real check.
        Assert.True(maxAbs <= 2.0f, $"logits L_inf {maxAbs:G6} vs legacy kernels.");
    }

    private static (int[] tokens, float[] lastLogits) Run(string modelPath, bool legacy, int[] prompt, int vocabSize)
    {
        string[] vars = { "DOTLLM_VK_F32_GEMV", "DOTLLM_VK_MOE_MMVQ", "DOTLLM_VK_MOE_SHARED_F16", "DOTLLM_VK_MOE_SHARED_Q8", "DOTLLM_VK_GDN_CONV_DECODE", "DOTLLM_VK_MOE_Q8_MMVQ", "DOTLLM_VK_MOE_DECODE_FUSED" };
        string?[] prev = vars.Select(Environment.GetEnvironmentVariable).ToArray();
        foreach (string v in vars) Environment.SetEnvironmentVariable(v, legacy ? "0" : null);
        try
        {
            using var gguf = GgufFile.Open(modelPath);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(device, gguf, config, ResolveSpvDir());
            using var cache = model.CreateKvCache(prompt.Length + DecodeSteps + 2);

            int[] positions = Enumerable.Range(0, prompt.Length).ToArray();
            int next;
            using (ITensor l = model.Forward(prompt, positions, -1, cache)) next = Argmax(l, vocabSize);

            var tokens = new int[DecodeSteps];
            float[] lastLogits = Array.Empty<float>();
            int pos = prompt.Length;
            for (int i = 0; i < DecodeSteps; i++)
            {
                tokens[i] = next;
                using ITensor l = model.Forward(new[] { next }, new[] { pos }, -1, cache);
                next = Argmax(l, vocabSize);
                if (i == DecodeSteps - 1) lastLogits = ToArray(l, vocabSize);
                pos++;
            }
            return (tokens, lastLogits);
        }
        finally
        {
            for (int i = 0; i < vars.Length; i++) Environment.SetEnvironmentVariable(vars[i], prev[i]);
        }
    }

    private static unsafe int Argmax(ITensor logits, int vocabSize)
    {
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
        int idx = 0; float best = span[0];
        for (int i = 1; i < span.Length; i++) if (span[i] > best) { best = span[i]; idx = i; }
        return idx;
    }

    private static unsafe float[] ToArray(ITensor logits, int vocabSize)
        => new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize).ToArray();

    private static string? ResolveModelPath()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_SPLIT_PARITY_MOE_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string hub = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "huggingface", "hub", "models--unsloth--Qwen3.6-35B-A3B-GGUF", "snapshots");
        return Directory.Exists(hub)
            ? Directory.GetFiles(hub, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", SearchOption.AllDirectories).FirstOrDefault()
            : null;
    }

    private static string ResolveSpvDir()
    {
        foreach (string c in new[]
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        })
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0) return full;
        }
        throw new InvalidOperationException("SPIR-V blobs not found. Run native/vulkan/build.ps1.");
    }
}
