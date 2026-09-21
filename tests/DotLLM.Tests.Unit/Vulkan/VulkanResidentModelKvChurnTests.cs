using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #467: a resident Vulkan model that decodes each request into a fresh KV cache, disposing
/// it afterwards — exactly what the server and <c>ModelResidencyManager</c> do — must return the
/// same greedy tokens every time.
/// </summary>
/// <remarks>
/// Before the fix the freed KV buffers' handles were recycled into the next request's cache and
/// hit stale descriptor sets: 9 of 15 requests diverged on Llama-3.2-1B Q8_0 (gfx1151), and the
/// first token was always right because only the decode path went wrong. At that rate the chance
/// of <see cref="Requests"/> clean requests by luck is below 1e-3. Reusing one cache
/// (<c>Rollback(0)</c>) or never freeing did not reproduce it, which is why this test frees.
/// </remarks>
[Trait("Category", "GPU")]
[Trait("Category", "RealModel")]
[Collection("VulkanKernels")]
public sealed class VulkanResidentModelKvChurnTests
{
    private const int Requests = 8;
    private const int NewTokens = 10;

    private static readonly int[] PromptTokens = [1, 785, 6722, 315, 9625, 374, 12095, 11, 323, 279];

    private readonly ITestOutputHelper _out;

    public VulkanResidentModelKvChurnTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void FreshKvCachePerRequest_DecodesIdentically()
    {
        string? path = FindLlama32_1B_Q8_0();
        Skip.If(path is null, "Llama-3.2-1B-Instruct Q8_0 GGUF not found (set DOTLLM_LLAMA32_1B_Q8_0_GGUF).");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
        int cacheLen = PromptTokens.Length + NewTokens + 8;

        int[] first = DecodeRequest(model, cacheLen, config.VocabSize);
        _out.WriteLine($"request 0: [{string.Join(",", first)}]");

        int diverged = 0;
        for (int r = 1; r <= Requests; r++)
        {
            int[] tokens = DecodeRequest(model, cacheLen, config.VocabSize);
            bool same = tokens.AsSpan().SequenceEqual(first);
            if (!same) diverged++;
            _out.WriteLine($"request {r}: [{string.Join(",", tokens)}]{(same ? "" : "  DIVERGED")}");
        }

        Assert.Equal(0, diverged);
    }

    private static int[] DecodeRequest(VulkanTransformerModel model, int cacheLen, int vocabSize)
    {
        using var kv = model.CreateKvCache(cacheLen);

        int[] positions = new int[PromptTokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        var emitted = new int[NewTokens];
        int next;
        using (ITensor logits = model.Forward(PromptTokens, positions, deviceId: -1, kv))
            next = LastRowArgMax(logits, vocabSize);

        for (int step = 0; step < NewTokens; step++)
        {
            emitted[step] = next;
            if (step == NewTokens - 1) break;
            using ITensor logits = model.Forward([next], [PromptTokens.Length + step], deviceId: -1, kv);
            next = LastRowArgMax(logits, vocabSize);
        }
        return emitted;
    }

    private static unsafe int LastRowArgMax(ITensor logits, int vocabSize)
    {
        float* row = (float*)logits.DataPointer + (long)(logits.Shape[0] - 1) * vocabSize;
        int best = 0;
        for (int i = 1; i < vocabSize; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }

    private static string? FindLlama32_1B_Q8_0()
    {
        string? overridePath = Environment.GetEnvironmentVariable("DOTLLM_LLAMA32_1B_Q8_0_GGUF");
        if (!string.IsNullOrEmpty(overridePath))
            return File.Exists(overridePath) ? overridePath : null;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "test-cache", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
            Path.Combine(home, ".dotllm", "models", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
        ];
        return candidates.FirstOrDefault(File.Exists);
    }
}
