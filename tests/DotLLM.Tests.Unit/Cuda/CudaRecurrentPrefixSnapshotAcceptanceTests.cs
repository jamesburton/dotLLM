using System.Globalization;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Engine;
using DotLLM.Engine.PromptCache;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>Opt-in, bounded acceptance on the cached real Bonsai checkpoint; no model download.</summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaRecurrentPrefixSnapshotAcceptanceTests(ITestOutputHelper output)
{
    [SkippableFact]
    public void RealBonsai_AlternatingGrowingPrefixes_ReusesEachConversation()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_PREFIX_SNAPSHOT_REAL") != "1", "Set DOTLLM_PREFIX_SNAPSHOT_REAL=1 for the real-model acceptance run.");
        string? path = Environment.GetEnvironmentVariable("DOTLLM_BONSAI_PQ2_0_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "Set DOTLLM_BONSAI_PQ2_0_GGUF to a cached Bonsai checkpoint.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string ptx = Path.Combine(AppContext.BaseDirectory, "ptx");
        Skip.IfNot(Directory.Exists(ptx), "PTX files missing");
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, 0, ptx);
        var tokenizer = new TokenIdsTokenizer(config.VocabSize);
        var cached = new TextGenerator(model, tokenizer, (_, size) => model.CreateKvCache(size), mtpEnabled: false,
            recurrentPrefixCache: true, recurrentPrefixCacheOptions: new() { MaxEntries = 2, HostBudgetBytes = 512L * 1024 * 1024 });
        var cold = new TextGenerator(model, tokenizer, (_, size) => model.CreateKvCache(size), mtpEnabled: false);
        var options = new InferenceOptions { MaxTokens = 1, Temperature = 0, RepetitionPenalty = 1 };
        int[] a = Enumerable.Range(100, 128).ToArray(), b = Enumerable.Range(400, 144).ToArray();
        try
        {
            for (int turn = 0; turn < 3; turn++)
            {
                Check("A", a, turn == 0 ? 0 : a.Length - 21);
                Check("B", b, turn == 0 ? 0 : b.Length - 21);
                a = [.. a, .. Enumerable.Range(1000 + turn * 20, 20)];
                b = [.. b, .. Enumerable.Range(2000 + turn * 20, 20)];
            }
            Assert.Equal(2, cached.RecurrentPrefixStatistics.Entries);
            Assert.Equal(4, cached.RecurrentPrefixStatistics.Hits);
        }
        finally { cached.ClearRecurrentPrefixCache(); }

        void Check(string conversation, int[] tokens, int expectedCached)
        {
            string prompt = string.Join(",", tokens);
            var expected = cold.Generate(prompt, options);
            var actual = cached.Generate(prompt, options);
            output.WriteLine($"{conversation}: prompt={tokens.Length} cached={actual.Timings.CachedTokenCount} suffix={tokens.Length - actual.Timings.CachedTokenCount} tier={actual.Timings.CacheTier ?? "miss"} restore_ms={actual.Timings.CacheRestoreMilliseconds:F3} prefill_ms={actual.Timings.PrefillTimeMs:F3}");
            Assert.Equal(expected.GeneratedTokenIds, actual.GeneratedTokenIds);
            Assert.Equal(expectedCached, actual.Timings.CachedTokenCount);
        }
    }

    private sealed class TokenIdsTokenizer(int vocabSize) : ITokenizer
    {
        public int VocabSize => vocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => vocabSize - 1;
        public int[] Encode(string text) => text.Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();
        public int CountTokens(string text) => Encode(text).Length;
        public string Decode(ReadOnlySpan<int> ids) => string.Join(",", ids.ToArray());
        public string Decode(ReadOnlySpan<int> ids, bool stripBosSpace) => Decode(ids);
        public string DecodeToken(int id) => id.ToString(CultureInfo.InvariantCulture);
    }
}
