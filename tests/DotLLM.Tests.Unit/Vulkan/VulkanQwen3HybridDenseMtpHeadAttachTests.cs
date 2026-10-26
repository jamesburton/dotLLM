using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.Samplers;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1185 on Vulkan: a separate <c>mtp-*.gguf</c> head attached after the build must draft exactly like the same head embedded in the
/// trunk (same kernels, same weights, so bit-identical), and greedy speculation through it must equal plain greedy decode.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3HybridDenseMtpHeadAttachTests : IDisposable
{
    private const int Context = 32;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-qwen35-mtp-attach-{Guid.NewGuid():N}");

    public VulkanQwen3HybridDenseMtpHeadAttachTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void SeparateHead_MatchesEmbeddedHead_AndSpeculationMatchesPlainGreedy()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string embedded = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "embedded.gguf"),
            withMtp: true, mtpHasOwnHeadTensors: false, contextLength: Context);
        string trunk = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "trunk.gguf"), withMtp: false, contextLength: Context);
        string head = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "mtp-trunk.gguf"),
            withMtp: true, mtpHasOwnHeadTensors: false, contextLength: Context, headOnly: true);

        float[] embDrafts = Run(embedded, null, spvDir, m => Drafts(m));
        float[] splitDrafts = Run(trunk, head, spvDir, m =>
        {
            Assert.True(m.SupportsMtp);
            return Drafts(m);
        });
        Assert.Equal(embDrafts, splitDrafts);

        List<int> spec = Run(trunk, head, spvDir, Speculate);
        List<int> plain = Run(trunk, null, spvDir, Plain);
        Assert.Equal(plain, spec);
    }

    private static T Run<T>(string path, string? headPath, string spvDir, Func<VulkanQwen3HybridDenseTransformerModel, T> body)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        if (headPath is not null)
        {
            Assert.False(model.SupportsMtp);
            model.AttachMtpHead(headPath);
        }
        return body(model);
    }

    private static float[] Drafts(VulkanQwen3HybridDenseTransformerModel model)
    {
        int vocab = model.Config.VocabSize;
        using var kv = model.CreateKvCache(model.Config.MaxSequenceLength);
        using var mtp = model.CreateMtpState()!;
        using (model.Forward([1, 5, 3], [0, 1, 2], deviceId: -1, kv, adapter: null, mtp)) { }
        var all = new List<float>();
        int token = 4;
        for (int step = 0; step < 3; step++)
        {
            using ITensor t = model.ForwardMtp(mtp, token, 3 + step);
            float[] row = Row(t, vocab, 0);
            all.AddRange(row);
            token = ArgMax(row);
        }
        return all.ToArray();
    }

    private static List<int> Speculate(VulkanQwen3HybridDenseTransformerModel model)
    {
        const int NewTokens = 10, K = 3;
        int vocab = model.Config.VocabSize;
        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        using var kv = model.CreateKvCache(model.Config.MaxSequenceLength);
        using var mtp = model.CreateMtpState()!;
        var ids = new List<int> { 1 };
        using (ITensor prefill = model.Forward([1], [0], deviceId: -1, kv, adapter: null, mtp))
            ids.Add(ArgMax(Row(prefill, vocab, prefill.Shape[0] - 1)));
        var buffer = new int[K + 1];
        int position = 1;
        while (ids.Count - 1 < NewTokens)
        {
            var r = decoder.DraftAndVerify(model, kv, mtp, pipeline, ids, constraint: null, position,
                vocabSize: vocab, numCandidates: K, buffer);
            Assert.True(r.AcceptedCount > 0);
            for (int i = 0; i < r.AcceptedCount; i++) ids.Add(buffer[i]);
            position += r.AcceptedCount;
        }
        return ids.Take(NewTokens + 1).ToList();
    }

    private static List<int> Plain(VulkanQwen3HybridDenseTransformerModel model)
    {
        const int NewTokens = 10;
        int vocab = model.Config.VocabSize;
        using var kv = model.CreateKvCache(model.Config.MaxSequenceLength);
        var ids = new List<int> { 1 };
        for (int pos = 0; pos < NewTokens; pos++)
        {
            using ITensor t = model.Forward([ids[^1]], [pos], deviceId: -1, kv);
            ids.Add(ArgMax(Row(t, vocab, t.Shape[0] - 1)));
        }
        return ids;
    }

    private static unsafe float[] Row(ITensor t, int vocab, int row)
        => new ReadOnlySpan<float>((float*)t.DataPointer + (long)row * vocab, vocab).ToArray();

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }
}
