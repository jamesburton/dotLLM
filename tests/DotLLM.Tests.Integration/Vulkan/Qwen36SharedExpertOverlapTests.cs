using System.Security.Cryptography;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #1452: overlapping the shared-expert chain with the routed-expert chain (2..8 rows) only reorders dispatches and barriers, so logits must be
/// bit-identical to the serial chain. One model, one process: teacher-forced N-row chunks with <see cref="VulkanQwen3MoeHybridTransformerModel.SharedOverlapEnabled"/>
/// off then on; the overlap counter proves the path ran only in the ON pass. Skips without a Vulkan device or the Qwen3.6-35B-A3B file. Take the GPU lock first.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class Qwen36SharedExpertOverlapTests
{
    [SkippableTheory]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(8)]
    public unsafe void SharedOverlap_IsBitIdenticalToSerialChain(int chunk)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string? path = DotLLM.Tests.Integration.Fixtures.TestFixtureResolver.ResolveFile(
            "DOTLLM_QWEN36_A3B_Q4_K_M_GGUF", "unsloth", "Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf").Path;
        Skip.If(path is null, "Qwen3.6-35B-A3B UD-Q4_K_M not found (set DOTLLM_QWEN36_A3B_Q4_K_M_GGUF).");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        bool addBos = GgufAddBosResolver.Resolve(gguf.Metadata);
        int[] ids = tokenizer.Encode(string.Concat(Enumerable.Repeat(
            "The history of the Roman Empire spans many centuries, during which the city grew from a small settlement into the capital of a vast realm. ", 3)));
        if (addBos && tokenizer.BosTokenId >= 0) ids = [tokenizer.BosTokenId, .. ids];
        ids = ids.Take(64).ToArray();

        using var device = VulkanDevice.Create();
        var (m, mk) = VulkanModelLoader.CreateFromGguf(device, gguf, config, Path.Combine(AppContext.BaseDirectory, "spv"));
        using var disp = m as IDisposable;
        var core = (VulkanQwen3MoeHybridTransformerModel)m;
        int vocab = config.VocabSize;
        bool saved = VulkanQwen3MoeHybridTransformerModel.SharedOverlapEnabled;
        try
        {
            string[] hashes = new string[3]; long[] counts = new long[3];
            for (int pass = 0; pass < 3; pass++)   // pass 0: discarded warm-up (resident bundles are built lazily)
            {
                VulkanQwen3MoeHybridTransformerModel.SharedOverlapEnabled = pass == 2;
                long before = core.MoePathCounts[(int)VulkanQwen3MoeHybridTransformerModel.MoePath.SharedOverlap];
                core.ResetSequenceState();
                using var cache = mk(ids.Length + 8);
                using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                for (int i = 0; i < ids.Length; i += chunk)
                {
                    int n = Math.Min(chunk, ids.Length - i);
                    using var t = core.Forward(ids.AsSpan(i, n), Enumerable.Range(i, n).ToArray(), 0, cache);
                    h.AppendData(new ReadOnlySpan<byte>((void*)t.DataPointer, (int)t.Shape[0] * vocab * sizeof(float)));
                }
                hashes[pass] = Convert.ToHexString(h.GetHashAndReset());
                counts[pass] = core.MoePathCounts[(int)VulkanQwen3MoeHybridTransformerModel.MoePath.SharedOverlap] - before;
            }
            Assert.Equal(0, counts[1]);
            Assert.True(counts[2] > 0, "the overlapped shared-expert chain never ran (counter did not move)");
            Assert.Equal(hashes[1], hashes[2]);
        }
        finally { VulkanQwen3MoeHybridTransformerModel.SharedOverlapEnabled = saved; }
    }
}
