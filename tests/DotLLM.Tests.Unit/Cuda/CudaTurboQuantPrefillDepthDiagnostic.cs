using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Diagnostic, not a gate (#1924), opt-in with <c>DOTLLM_TQ_PREFILL_DEPTH_DIAG=1</c> and <c>DOTLLM_QWEN3_4B_GGUF</c>: does the new
/// TurboQuant prefill entry move the model further from unquantised KV than the legacy entry? A long prompt (default 4096 wikitext
/// tokens, <c>DOTLLM_TQ_PREFILL_DEPTH_TOKENS</c>) is prefilled in 1024-row chunks, so every chunk after the first runs the tensor-core
/// TQ prefill at depth, then 32 teacher-forced decode steps follow (scalar kernel, unchanged). Arms: F16 KV (reference), the scratch
/// oracle, and the in-place cache with the prefill entry forced to Legacy, Specialized (E1) and Staged (E3). Reported per bit width:
/// mean / max KL(f16 || arm) and cos(arm, oracle) over the 33 rows. tq-gate's quality and 1-layer parity modes cannot see this change
/// on CUDA (its forward returns one row, so quality runs one token per forward; one layer cannot move the codes).
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaTurboQuantPrefillDepthDiagnostic
{
    private const ulong Seed = 0x5DEECE66DUL;   // the CLI's default TurboQuant seed
    private readonly ITestOutputHelper _out;
    public CudaTurboQuantPrefillDepthDiagnostic(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void LongPrefill_PerEntry_DistanceToF16AndOracle()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_TQ_PREFILL_DEPTH_DIAG") == "1", "set DOTLLM_TQ_PREFILL_DEPTH_DIAG=1");
        string modelPath = Environment.GetEnvironmentVariable("DOTLLM_QWEN3_4B_GGUF") ?? "";
        Skip.If(!File.Exists(modelPath), "set DOTLLM_QWEN3_4B_GGUF");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        int n = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_TQ_PREFILL_DEPTH_TOKENS"), out int nn) ? nn : 4096;
        const int chunk = 1024, steps = 32;
        string corpus = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "corpora",
                                     "wikitext-2-raw", "wiki.test.lf.raw");
        string ptx = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "ptx"));

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] all = tokenizer.Encode(File.ReadAllText(corpus).Substring(0, 400_000));
        Assert.True(all.Length > n + steps + 1, $"corpus too short: {all.Length}");
        int[] toks = all[..(n + steps + 1)];
        int vocab = config.VocabSize, maxSeq = n + steps + 8;
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptx);

        List<float[]> Run(DotLLM.Core.Attention.IKvCache cache)
        {
            var rows = new List<float[]>();
            float[]? last = null;
            for (int s = 0; s < n; s += chunk)
            {
                int m = Math.Min(chunk, n - s);
                var pos = Enumerable.Range(s, m).ToArray();
                using var l = model.Forward(toks.AsSpan(s, m).ToArray(), pos, 0, cache);
                last = Row(l, vocab);
            }
            rows.Add(last!);
            for (int k = 0; k < steps; k++)
            {
                using var l = model.Forward([toks[n + k]], [n + k], 0, cache);
                rows.Add(Row(l, vocab));
            }
            return rows;
        }

        List<float[]> f16;
        using (var c = model.CreateKvCache(maxSeq)) f16 = Run(c);
        foreach (int bits in new[] { 4, 3 })
        {
            var codecK = new TurboQuantCodec(config.HeadDim, bits, Seed);
            var codecV = new TurboQuantCodec(config.HeadDim, bits, Seed ^ TurboQuantCodec.VSeedMask);
            List<float[]> oracle;
            using (var c = model.CreateTurboQuantKvCache(maxSeq, codecK.MseBits, codecK.Centroids, codecK.RotationSigns,
                                                         codecV.RotationSigns, codecK.InvSqrtD)) oracle = Run(c);
            var kv = new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 0, bits, Seed);
            Report(bits, "oracle", oracle, f16, oracle, 0);
            var arms = new Dictionary<CudaKernels.TqPrefillVariant, List<float[]>>();
            foreach (var v in new[] { CudaKernels.TqPrefillVariant.Legacy, CudaKernels.TqPrefillVariant.Specialized, CudaKernels.TqPrefillVariant.Staged })
            {
                long l0 = CudaKernels.AttentionFlashPrefillKvTqLaunches, s0 = CudaKernels.AttentionFlashPrefillKvTqSpecLaunches,
                     c0 = CudaKernels.AttentionFlashPrefillKvTqCpaLaunches;
                CudaKernels.TqPrefillVariantOverride = v;
                try { using var c = model.CreateKvCache(maxSeq, kv); arms[v] = Run(c); }
                finally { CudaKernels.TqPrefillVariantOverride = null; }
                long flash = CudaKernels.AttentionFlashPrefillKvTqLaunches - l0;
                long spec = CudaKernels.AttentionFlashPrefillKvTqSpecLaunches - s0, cpa = CudaKernels.AttentionFlashPrefillKvTqCpaLaunches - c0;
                Assert.True(flash > 0, "the tensor-core TQ prefill never ran");
                Assert.Equal(v == CudaKernels.TqPrefillVariant.Specialized ? flash : 0, spec);
                Assert.Equal(v == CudaKernels.TqPrefillVariant.Staged ? flash : 0, cpa);
                Report(bits, v.ToString(), arms[v], f16, oracle, flash);
            }
            var (cosLs, _) = CosStats(arms[CudaKernels.TqPrefillVariant.Legacy], arms[CudaKernels.TqPrefillVariant.Staged]);
            var (cosSs, _) = CosStats(arms[CudaKernels.TqPrefillVariant.Specialized], arms[CudaKernels.TqPrefillVariant.Staged]);
            _out.WriteLine($"DEPTH b{bits} n={n}: min cos(Legacy, Staged) {cosLs:F6}; min cos(Specialized, Staged) {cosSs:F6} (E1 and E3 should be bit-identical: 1)");
        }
    }

    private void Report(int bits, string arm, List<float[]> rows, List<float[]> f16, List<float[]> oracle, long flash)
    {
        var kl = rows.Select((r, i) => Kl(f16[i], r)).ToArray();
        var (minCosOr, meanCosOr) = CosStats(rows, oracle);
        int agree = rows.Select((r, i) => Argmax(r) == Argmax(f16[i]) ? 1 : 0).Sum();
        _out.WriteLine($"DEPTH b{bits} {arm,-11}: KL(f16||arm) mean {kl.Average():E4} max {kl.Max():E4} prefill-row {kl[0]:E4}; " +
                       $"cos(arm,oracle) min {minCosOr:F6} mean {meanCosOr:F6}; argmax==f16 {agree}/{rows.Count}; flash launches {flash}");
    }

    private static (double min, double mean) CosStats(List<float[]> a, List<float[]> b)
    {
        var c = a.Select((r, i) => Cos(r, b[i])).ToArray();
        return (c.Min(), c.Average());
    }

    private static double Cos(float[] a, float[] b)
    {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return d / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-30);
    }

    private static double Kl(float[] p, float[] q)
    {
        double mp = p.Max(), mq = q.Max(), zp = 0, zq = 0;
        for (int i = 0; i < p.Length; i++) { zp += Math.Exp(p[i] - mp); zq += Math.Exp(q[i] - mq); }
        double lzp = Math.Log(zp) + mp, lzq = Math.Log(zq) + mq, kl = 0;
        for (int i = 0; i < p.Length; i++) { double lp = p[i] - lzp; kl += Math.Exp(lp) * (lp - (q[i] - lzq)); }
        return kl;
    }

    private static int Argmax(float[] r) { int b = 0; for (int i = 1; i < r.Length; i++) if (r[i] > r[b]) b = i; return b; }

    private static float[] Row(DotLLM.Core.Tensors.ITensor logits, int vocab)
    {
        var row = new float[vocab];
        float* p = (float*)logits.DataPointer;
        int off = (logits.Shape[0] - 1) * vocab;
        for (int i = 0; i < vocab; i++) row[i] = p[off + i];
        return row;
    }
}
