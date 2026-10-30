using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1188 on Vulkan: the <c>qwen35moe</c> MTP head drafts the CPU oracle's tokens (argmax exact, logits to GEMM order), the recurrent
/// checkpoint pair restores the GDN state, and greedy speculation equals plain greedy decode on the device.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3MoeHybridMtpTests : IDisposable
{
    private const int Context = 32;
    private const int Steps = 4;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-qwen35moe-mtp-{Guid.NewGuid():N}");
    private readonly string? _residentEnv = Environment.GetEnvironmentVariable("DOTLLM_VK_MOE_RESIDENT");

    public VulkanQwen3MoeHybridMtpTests()
    {
        Directory.CreateDirectory(_scratch);
        Environment.SetEnvironmentVariable("DOTLLM_VK_MOE_RESIDENT", "1");   // the head needs resident banks
        VulkanQwen3MoeHybridTransformerModel.MtpOptIn = true;
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("DOTLLM_VK_MOE_RESIDENT", _residentEnv);
        VulkanQwen3MoeHybridTransformerModel.MtpOptIn = Environment.GetEnvironmentVariable("DOTLLM_VK_QWEN35MOE_MTP") == "1";
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string Fixture() => SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, "moe-mtp.gguf"), withMtp: true, contextLength: Context);

    /// <summary>
    /// #1224: the resident banks are uploaded by the loader, not by the first forward. The lazy upload made a real model's first prefill pay
    /// a ~19 GiB page-in (11 tok/s against 1,443 warm). <see cref="Drafts_MatchCpuOracle"/> covers the forward on the eagerly uploaded banks.
    /// </summary>
    [SkippableFact]
    public void BuildFromGguf_UploadsEveryResidentBankAtLoad()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var (resident, layers) = WithVulkan(Fixture(), spvDir, m => (m.ResidentMoeLayerCount, m.Config.NumLayers));
        Assert.True(layers > 1);
        Assert.Equal(layers, resident);
    }

    [SkippableFact]
    public void Drafts_MatchCpuOracle()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = Fixture();
        int[] prompt = [1, 5, 3, 7];
        int[] pos = [0, 1, 2, 3];

        var (cpuTokens, cpuLogits) = WithCpu(path, m =>
        {
            using var kv = new SimpleKvCache(m.AttentionLayerCount, m.Config.NumKvHeads, m.Config.HeadDim, m.Config.MaxSequenceLength);
            using var st = m.CreateMtpState()!;
            using (m.Forward(prompt, pos, -1, kv, adapter: null, st)) { }
            return Draft((t, i) => m.ForwardMtp(st, t, prompt.Length + i), m.Config.VocabSize);
        });
        var (vkTokens, vkLogits) = WithVulkan(path, spvDir, m =>
        {
            Assert.True(m.SupportsMtp);
            using var kv = m.CreateKvCache(m.Config.MaxSequenceLength);
            using var st = m.CreateMtpState()!;
            using (m.Forward(prompt, pos, -1, kv, adapter: null, st)) { }
            Assert.Equal(prompt.Length, st.CurrentLength);
            return Draft((t, i) => m.ForwardMtp(st, t, prompt.Length + i), m.Config.VocabSize);
        });

        Assert.Equal(cpuTokens, vkTokens);
        for (int i = 0; i < cpuLogits.Length; i++)
            Assert.True(MathF.Abs(cpuLogits[i] - vkLogits[i]) <= 5e-3f + 1e-3f * MathF.Abs(cpuLogits[i]), $"logit {i}: cpu {cpuLogits[i]} vk {vkLogits[i]}");
    }

    [SkippableFact]
    public void GreedySpeculation_EqualsPlainGreedy()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = Fixture();
        List<int> plain = WithVulkan(path, spvDir, m =>
        {
            using var kv = m.CreateKvCache(m.Config.MaxSequenceLength);
            var ids = new List<int> { 1 };
            for (int p = 0; p < 12; p++)
            {
                using ITensor t = m.Forward([ids[^1]], [p], -1, kv);
                ids.Add(ArgMax(Row(t, t.Shape[0] - 1, m.Config.VocabSize)));
            }
            return ids;
        });
        foreach (int k in new[] { 1, 3 })
        {
            List<int> spec = WithVulkan(path, spvDir, m =>
            {
                int vocab = m.Config.VocabSize;
                var decoder = new MtpSpeculativeDecoder(greedy: true);
                var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
                using var kv = m.CreateKvCache(m.Config.MaxSequenceLength);
                using var st = m.CreateMtpState()!;
                var ids = new List<int> { 1 };
                using (ITensor pre = m.Forward([1], [0], -1, kv, adapter: null, st))
                    ids.Add(ArgMax(Row(pre, pre.Shape[0] - 1, vocab)));
                var buf = new int[k + 1];
                int position = 1;
                while (ids.Count - 1 < 12)
                {
                    var r = decoder.DraftAndVerify(m, kv, st, pipeline, ids, constraint: null, position, vocabSize: vocab, numCandidates: k, buf);
                    Assert.True(r.AcceptedCount > 0);
                    for (int i = 0; i < r.AcceptedCount; i++) ids.Add(buf[i]);
                    position += r.AcceptedCount;
                }
                return ids.Take(13).ToList();
            });
            Assert.Equal(plain, spec);
        }
    }

    // Draft steps chain through their own argmax, from token 4 right after the prompt.
    private static (int[] Tokens, float[] Logits) Draft(Func<int, int, ITensor> step, int vocab)
    {
        var tokens = new int[Steps];
        var logits = new List<float>();
        int token = 4;
        for (int i = 0; i < Steps; i++)
        {
            using ITensor t = step(token, i);
            float[] row = Row(t, 0, vocab);
            logits.AddRange(row);
            token = tokens[i] = ArgMax(row);
        }
        return (tokens, logits.ToArray());
    }

    private static T WithCpu<T>(string path, Func<Qwen3MoeHybridTransformerModel, T> body)
    {
        using var gguf = GgufFile.Open(path);
        using var m = (Qwen3MoeHybridTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, GgufModelConfigExtractor.Extract(gguf.Metadata));
        return body(m);
    }

    private static T WithVulkan<T>(string path, string spvDir, Func<VulkanQwen3MoeHybridTransformerModel, T> body)
    {
        using var gguf = GgufFile.Open(path);
        using var device = VulkanDevice.Create();
        using var m = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(device, gguf, GgufModelConfigExtractor.Extract(gguf.Metadata), spvDir);
        return body(m);
    }

    private static unsafe float[] Row(ITensor t, int row, int vocab) => new ReadOnlySpan<float>((float*)t.DataPointer + (long)row * vocab, vocab).ToArray();

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }
}
