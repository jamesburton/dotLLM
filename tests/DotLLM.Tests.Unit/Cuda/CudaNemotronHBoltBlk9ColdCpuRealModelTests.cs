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
/// Issue #1071, real weights: the first 9 blocks of Nemotron 3.5 Bolt (see <see cref="CudaNemotronHBoltBlk9RealModelTests"/>)
/// with the routed experts offloaded and every uncached expert computed on the CPU (<c>--moe-cold cpu</c>, promote-only
/// admission into a small cache), against the same file with every expert resident: a 16-token prefill and four greedy decode
/// steps (the resident run's tokens replayed), KV cache and SSM state carried. The CPU experts round differently from the
/// device's (Q8_K activations), so the bounds are PR A's cross-backend ones: mean KL and top-1 agreement.
/// </summary>
/// <remarks>Set <c>DOTLLM_BOLT_BLK9_GGUF</c> to the truncated file. Skipped when unset.</remarks>
[Trait("Category", "GPU")]
[Trait("Category", "RealModel")]
[Collection(CudaCollection.Name)]
public sealed class CudaNemotronHBoltBlk9ColdCpuRealModelTests(ITestOutputHelper output) : IDisposable
{
    private const string EnvVar = "DOTLLM_BOLT_BLK9_GGUF";
    private const int DecodeSteps = 4;
    private const int MaxSeq = 64;
    private const long CacheMib = 256;   // about an eighth of blk9's ~2 GiB of routed experts

    private static readonly int[] Prompt = [1, 1784, 4237, 1278, 6542, 1307, 1278, 2524, 1044, 1531, 1395, 1636, 1317, 2036, 1294, 1278];

    private readonly string _scratch = Directory.CreateTempSubdirectory("dotllm-blk9-cold-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <param name="blockTokens">MoE token block (512 = the whole prompt in one layer call).</param>
    /// <param name="pinned">Pinned host store, or the memory-mapped store through the staging ring.</param>
    [SkippableTheory]
    [InlineData(512, true)]
    [InlineData(4, false)]
    public void Blk9_ColdCpu_WithinTheCrossBackendBoundOfResident(int blockTokens, bool pinned)
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
            int vocab = config.VocabSize;

            var (resident, decodeTokens) = Run(gguf, config, ptxDir!, new MoeOffloadOptions { Mode = MoeOffloadMode.Off }, null, out _, out _);
            var options = new MoeOffloadOptions
            {
                Mode = MoeOffloadMode.On,
                Cold = MoeColdMode.Cpu,
                CacheMib = CacheMib,
                PinBudgetFraction = pinned ? MoeOffloadOptions.DefaultPinBudgetFraction : 0,
                CalibrationPath = Path.Combine(_scratch, "moe-calibration.json"),
                Log = output.WriteLine,
            };
            var start = MoeTelemetry.Snapshot();
            var (cold, _) = Run(gguf, config, ptxDir!, options, decodeTokens, out var info, out var cache);
            var tel = MoeTelemetry.Snapshot().Since(start);
            output.WriteLine($"block {blockTokens}, pinned {pinned}: {info.CacheSlots} slots, hits {cache.Hits}, cold {cache.ColdMisses}, " +
                $"promotions {cache.Promotions}; {tel.FormatSummary()}");
            Assert.Equal(MoeColdMode.Cpu, info.Cold);
            Assert.True(tel.ColdExpertsCpu > 0, "no expert ran on the CPU");

            int rows = resident.Count, agree = 0;
            double klSum = 0, worstKl = 0;
            for (int r = 0; r < rows; r++)
            {
                double kl = Kl(resident[r], cold[r]);
                klSum += kl;
                worstKl = Math.Max(worstKl, kl);
                bool same = ArgMax(resident[r]) == ArgMax(cold[r]);
                if (same) agree++;
                output.WriteLine($"row {r,2}: KL {kl:E3}, top-1 resident {ArgMax(resident[r])} cold {ArgMax(cold[r])}{(same ? "" : "  <-- differs")}");
            }
            double meanKl = klSum / rows, top1 = (double)agree / rows;
            output.WriteLine($"{Path.GetFileName(path)}: mean KL {meanKl:E4} (worst {worstKl:E3}), top-1 {agree}/{rows} = {top1:P1}, vocab {vocab}");

            // PR A's real-model bounds (CPU Q8_K dots vs CUDA FP32 expert GEMVs).
            Assert.True(meanKl <= 0.02, $"mean KL(resident || cold) {meanKl:E4} > 0.02");
            Assert.True(top1 >= 0.9, $"top-1 agreement {top1:P1} < 90%");
        }
        finally
        {
            CudaNemotronHTransformerModel.MoeBlockTokens = savedBlock;
        }
    }

    private sealed record CacheStats(long Hits, long ColdMisses, long Promotions);

    private static (List<float[]> Rows, List<int> DecodeTokens) Run(GgufFile gguf, ModelConfig config, string ptxDir,
        MoeOffloadOptions options, List<int>? decodeTokens, out CudaMoeOffloadInfo info, out CacheStats cache)
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
            info = cuda.MoeOffload!;
            var c = cuda.MoeExpertCache;
            cache = new CacheStats(c?.Hits ?? 0, c?.ColdMisses ?? 0, c?.Promotions ?? 0);
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

    /// <summary>KL(softmax(p) || softmax(q)) in double.</summary>
    private static double Kl(float[] p, float[] q)
    {
        double mp = p.Max(), mq = q.Max(), zp = 0, zq = 0;
        for (int i = 0; i < p.Length; i++) { zp += Math.Exp(p[i] - mp); zq += Math.Exp(q[i] - mq); }
        double lzp = Math.Log(zp) + mp, lzq = Math.Log(zq) + mq, kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            double lp = p[i] - lzp, lq = q[i] - lzq;
            kl += Math.Exp(lp) * (lp - lq);
        }
        return Math.Max(kl, 0);
    }
}
