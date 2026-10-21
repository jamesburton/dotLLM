using DotLLM.Core.Models;
using DotLLM.Core.Moe;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1070, real weights: the first 9 blocks of Nemotron 3.5 Bolt (see <see cref="CudaNemotronHBoltBlk9RealModelTests"/>)
/// with the routed experts offloaded behind a VRAM cache much smaller than them, held bit for bit to the same file with every
/// expert resident: a 16-token prefill (as one MoE block, and as blocks of 4 so the cache sees several prefill lookups) and
/// four greedy decode steps, KV cache and SSM state carried.
/// </summary>
/// <remarks>Set <c>DOTLLM_BOLT_BLK9_GGUF</c> to the truncated file. Skipped when unset.</remarks>
[Trait("Category", "GPU")]
[Trait("Category", "RealModel")]
[Collection(CudaCollection.Name)]
public sealed class CudaNemotronHBoltBlk9OffloadRealModelTests(ITestOutputHelper output)
{
    private const string EnvVar = "DOTLLM_BOLT_BLK9_GGUF";
    private const int DecodeSteps = 4;
    private const int MaxSeq = 64;
    private const long CacheMib = 256;   // about an eighth of blk9's ~2 GiB of routed experts

    private static readonly int[] Prompt = [1, 1784, 4237, 1278, 6542, 1307, 1278, 2524, 1044, 1531, 1395, 1636, 1317, 2036, 1294, 1278];

    /// <param name="blockTokens">MoE token block (16 = the whole prompt in one lookup per layer).</param>
    /// <param name="pinned">Pinned host store, or the memory-mapped store through the staging ring.</param>
    [SkippableTheory]
    [InlineData(512, true)]
    [InlineData(4, false)]
    public void Blk9_OffloadMatchesResidentBitForBit(int blockTokens, bool pinned)
    {
        string? path = Environment.GetEnvironmentVariable(EnvVar);
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), $"{EnvVar} not set or file missing");
        Skip.IfNot(CudaNemotronMoeTestSupport.IsCudaAvailable(), "No CUDA GPU available");
        string? ptxDir = CudaNemotronMoeTestSupport.FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        int savedBlock = CudaNemotronHTransformerModel.MoeBlockTokens;
        CudaNemotronHTransformerModel.MoeBlockTokens = blockTokens;
        try
        {
            using var gguf = GgufFile.Open(path!);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.Equal(Architecture.NemotronHMoe, config.Architecture);

            var (resident, decodeTokens) = Run(gguf, config, ptxDir!, new MoeOffloadOptions { Mode = MoeOffloadMode.Off }, null, out _);
            CudaNemotronHTransformerModel.MoePathCounters.Reset();
            var options = new MoeOffloadOptions
            {
                Mode = MoeOffloadMode.On,
                CacheMib = CacheMib,
                PinBudgetFraction = pinned ? MoeOffloadOptions.DefaultPinBudgetFraction : 0,
                Log = output.WriteLine,
            };
            var (offload, _) = Run(gguf, config, ptxDir!, options, decodeTokens, out var stats);
            long groups = Interlocked.Read(ref CudaNemotronHTransformerModel.MoePathCounters.Groups) / 2;

            output.WriteLine($"block {blockTokens}, pinned {pinned}: {stats.Info.CacheSlots} slots in {stats.Info.Pools} pools " +
                $"({stats.Info.CacheBytes / (1 << 20)} MiB of {stats.Info.StoreBytes / (1 << 20)} MiB), hits {stats.Hits}, misses " +
                $"{stats.Misses}, evictions {stats.Evictions}, bypasses {stats.Bypasses}");

            Assert.Equal(pinned, stats.Info.PinnedStore);
            Assert.True(stats.Info.CacheBytes <= CacheMib << 20);
            Assert.True(stats.Info.CacheBytes < stats.Info.StoreBytes / 4, "the cache is not much smaller than the experts");
            Assert.Equal(groups, stats.Hits + stats.Misses);
            Assert.True(stats.Misses > 0 && stats.Evictions > 0, "the small cache never missed or evicted");

            Assert.Equal(resident.Count, offload.Count);
            for (int r = 0; r < resident.Count; r++)
                for (int i = 0; i < resident[r].Length; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(resident[r][i]) == BitConverter.SingleToInt32Bits(offload[r][i]),
                        $"row {r} logit {i}: resident {resident[r][i]:R} != offload {offload[r][i]:R}");
        }
        finally
        {
            CudaNemotronHTransformerModel.MoeBlockTokens = savedBlock;
        }
    }

    private sealed record Stats(CudaMoeOffloadInfo Info, long Hits, long Misses, long Evictions, long Bypasses);

    /// <summary>
    /// Prefill then <see cref="DecodeSteps"/> greedy steps. With <paramref name="decodeTokens"/> null the steps take this run's
    /// argmax and return them; otherwise they replay the given tokens (so both runs see the same inputs).
    /// </summary>
    private static (List<float[]> Rows, List<int> DecodeTokens) Run(GgufFile gguf, Core.Models.ModelConfig config, string ptxDir,
        MoeOffloadOptions options, List<int>? decodeTokens, out Stats stats)
    {
        int vocab = config.VocabSize;
        var rows = new List<float[]>();
        var tokens = decodeTokens ?? [];
        var (model, kvFactory) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0, ptxDir, options);
        using (model)
        using (var kv = kvFactory(MaxSeq))
        {
            var cuda = Assert.IsType<CudaNemotronHTransformerModel>(model);
            using (var t = model.Forward(Prompt, [.. Enumerable.Range(0, Prompt.Length)], deviceId: 0, kv)) rows.AddRange(Rows(t, vocab));
            for (int s = 0; s < DecodeSteps; s++)
            {
                if (decodeTokens is null) tokens.Add(ArgMax(rows[^1]));
                using var t = model.Forward([tokens[s]], [Prompt.Length + s], deviceId: 0, kv);
                rows.AddRange(Rows(t, vocab));
            }
            var c = cuda.MoeExpertCache;
            stats = new Stats(cuda.MoeOffload!, c?.Hits ?? 0, c?.Misses ?? 0, c?.Evictions ?? 0, c?.Bypasses ?? 0);
        }
        return (rows, tokens);
    }

    private static unsafe List<float[]> Rows(ITensor logits, int vocab)
    {
        int n = (int)(logits.ElementCount / vocab);
        var rows = new List<float[]>(n);
        float* p = (float*)logits.DataPointer;
        for (int r = 0; r < n; r++) rows.Add(new ReadOnlySpan<float>(p + (long)r * vocab, vocab).ToArray());
        return rows;
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }
}
