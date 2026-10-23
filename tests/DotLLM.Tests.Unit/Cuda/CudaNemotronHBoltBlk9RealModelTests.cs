using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #403, real weights: the first 9 blocks of Nemotron 3.5 Bolt (a truncation of a real GGUF: MoE at layers 1 / 3 / 5 / 8
/// with 240 / 240 / 240 / 304 experts and widths 2048 / 1536 / 2048 / 2304, attention at 7), CUDA vs the CPU model. A
/// 16-token prefill and four greedy decode steps (the CPU's argmax fed to both), with the KV cache and SSM state carried.
/// </summary>
/// <remarks>
/// Set <c>DOTLLM_BOLT_BLK9_GGUF</c> to the truncated file, e.g. <c>bolt-q4km-blk9.gguf</c> (Q4_K / Q5_K banks) or
/// <c>bolt-nvidia-24GB-blk9.gguf</c> (IQ2_XXS banks, NVFP4 SSM). Skipped when unset.
/// </remarks>
[Trait("Category", "GPU")]
[Trait("Category", "RealModel")]
[Collection(CudaCollection.Name)]
public sealed class CudaNemotronHBoltBlk9RealModelTests(ITestOutputHelper output)
{
    private const string EnvVar = "DOTLLM_BOLT_BLK9_GGUF";
    private const int DecodeSteps = 4;
    private const int MaxSeq = 64;

    // A fixed 16-token prompt (ids well inside Bolt's vocabulary); the truncated model's logits are not meaningful text
    // predictions, so the ids only need to be stable.
    private static readonly int[] Prompt = [1, 1784, 4237, 1278, 6542, 1307, 1278, 2524, 1044, 1531, 1395, 1636, 1317, 2036, 1294, 1278];

    [SkippableFact]
    public void Blk9_PrefillAndDecode_CudaMatchesCpu()
    {
        string? path = Environment.GetEnvironmentVariable(EnvVar);
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), $"{EnvVar} not set or file missing");
        Skip.IfNot(CudaNemotronMoeTestSupport.IsCudaAvailable(), "No CUDA GPU available");
        string? ptxDir = CudaNemotronMoeTestSupport.FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.NemotronHMoe, config.Architecture);
        int vocab = config.VocabSize;

        // CPU oracle first, choosing the decode tokens (greedy) that both models then consume.
        var cpuRows = new List<float[]>();
        var decodeTokens = new List<int>();
        using (var cpu = NemotronHTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.Auto))
        using (var kv = new SimpleKvCache(cpu.AttentionLayerCount, config.NumKvHeads, config.HeadDim, MaxSeq))
        {
            using (var t = cpu.Forward(Prompt, Positions(0, Prompt.Length), deviceId: -1, kv)) cpuRows.AddRange(Rows(t, vocab));
            for (int s = 0; s < DecodeSteps; s++)
            {
                int tok = ArgMax(cpuRows[^1]);
                decodeTokens.Add(tok);
                using (var t = cpu.Forward([tok], [Prompt.Length + s], deviceId: -1, kv)) cpuRows.AddRange(Rows(t, vocab));
            }
        }

        CudaNemotronHTransformerModel.MoePathCounters.Reset();
        var cudaRows = new List<float[]>();
        var (model, kvFactory) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0, ptxDir);
        using (model)
        using (var kv = kvFactory(MaxSeq))
        {
            Assert.IsType<CudaNemotronHTransformerModel>(model);
            using (var t = model.Forward(Prompt, Positions(0, Prompt.Length), deviceId: 0, kv)) cudaRows.AddRange(Rows(t, vocab));
            for (int s = 0; s < DecodeSteps; s++)
            {
                using var t = model.Forward([decodeTokens[s]], [Prompt.Length + s], deviceId: 0, kv);
                cudaRows.AddRange(Rows(t, vocab));
            }
        }
        Assert.True(Interlocked.Read(ref CudaNemotronHTransformerModel.MoePathCounters.LayerCalls) >= 4 * (1 + DecodeSteps),
            "the MoE layers did not run on the CUDA expert kernels");

        // Metrics per row: max |logit diff|, KL(cpu || cuda) of the softmaxes, top-1 agreement.
        int rows = cpuRows.Count, agree = 0;
        double klSum = 0, worstKl = 0;
        float maxAbs = 0;
        for (int r = 0; r < rows; r++)
        {
            float[] a = cpuRows[r], b = cudaRows[r];
            for (int i = 0; i < vocab; i++) maxAbs = MathF.Max(maxAbs, MathF.Abs(a[i] - b[i]));
            double kl = Kl(a, b);
            klSum += kl;
            worstKl = Math.Max(worstKl, kl);
            bool same = ArgMax(a) == ArgMax(b);
            if (same) agree++;
            output.WriteLine($"row {r,2} ({(r < Prompt.Length ? "prefill" : "decode")}): KL {kl:E3}, top-1 cpu {ArgMax(a)} cuda {ArgMax(b)}{(same ? "" : "  <-- differs")}");
        }
        double meanKl = klSum / rows;
        double top1 = (double)agree / rows;
        output.WriteLine($"{Path.GetFileName(path)}: max-abs {maxAbs:E3}, mean KL {meanKl:E4} (worst {worstKl:E3}), top-1 {agree}/{rows} = {top1:P1}");

        // Bounds for a cross-backend comparison with different activation rounding (CPU Q8_K dots vs CUDA FP32 expert GEMV and
        // FP16 dense prefill GEMM): a routing or layout defect shows up as KL of order 1 and scattered top-1 flips.
        Assert.True(meanKl <= 0.02, $"mean KL(cpu || cuda) {meanKl:E4} > 0.02");
        Assert.True(top1 >= 0.9, $"top-1 agreement {top1:P1} < 90%");
    }

    private static int[] Positions(int start, int count) => [.. Enumerable.Range(start, count)];

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
