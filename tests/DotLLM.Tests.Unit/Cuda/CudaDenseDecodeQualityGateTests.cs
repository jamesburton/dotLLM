using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1922 E1 generation gate. Opt-in: needs <c>DOTLLM_DENSE_DECODE_GATE_GGUF</c> (e.g. Qwen3-4B Q4_K_M, hd128 32/8) and
/// <c>DOTLLM_DENSE_DECODE_GATE_TEXT</c> (a plain text file of at least prefill + steps + 1 tokens). Optional: <c>_PREFILL</c>
/// (default 6000), <c>_STEPS</c> (1000), <c>_KV</c> (comma list of f16,q8_0,q4_0,tq4,tq3; default f16,q8_0,q4_0), <c>_OUT</c>
/// (append the table to a file).
/// <para>
/// One model, two caches decoded in lockstep and teacher-forced on the text (perplexity cannot see a decode-path change unless
/// the decode path itself scores the tokens): arm A = the scalar kernel, arm B = dense v2 (normalised F16 partials). Per step:
/// NLL of the next text token for each arm, KL(A || B) over the full vocabulary, top-1 agreement. Gate (precedent #222/#988):
/// PPL(B) / PPL(A) - 1 &lt;= +0.30%. Measured 2026-11-21 (framework, depth 6,000, 800 steps): f16 -0.057%, q8_0 -0.061%,
/// q4_0 +0.230%, tq4 -0.172%; an F32-partial control arm measured alongside was indistinguishable and has been removed.
/// </para>
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaDenseDecodeQualityGateTests
{
    private readonly ITestOutputHelper _out;
    public CudaDenseDecodeQualityGateTests(ITestOutputHelper output) => _out = output;

    private static int EnvInt(string n, int d) => int.TryParse(Environment.GetEnvironmentVariable(n), out int v) ? v : d;

    private static KvCacheConfig Kv(string kind) => kind switch
    {
        "f16" => new KvCacheConfig(),
        "q8_0" => new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0),
        "q4_0" => new KvCacheConfig(KvCacheDType.Q4_0, KvCacheDType.Q4_0),
        "tq4" => new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 0, 4),
        "tq3" => new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 0, 3),
        _ => throw new ArgumentOutOfRangeException(nameof(kind)),
    };

    private static void SetArm(int arm)
    {
        CudaDenseDecodeV2.Enabled = arm != 0;
        CudaDenseDecodeV2.MinSeqKv = 1;
    }

    [SkippableFact]
    public void TeacherForced_DenseV2_vs_Scalar()
    {
        string? modelPath = Environment.GetEnvironmentVariable("DOTLLM_DENSE_DECODE_GATE_GGUF");
        string? textPath = Environment.GetEnvironmentVariable("DOTLLM_DENSE_DECODE_GATE_TEXT");
        Skip.If(string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath), "set DOTLLM_DENSE_DECODE_GATE_GGUF");
        Skip.If(string.IsNullOrEmpty(textPath) || !File.Exists(textPath), "set DOTLLM_DENSE_DECODE_GATE_TEXT");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        int prefill = EnvInt("DOTLLM_DENSE_DECODE_GATE_PREFILL", 6000);
        int steps = EnvInt("DOTLLM_DENSE_DECODE_GATE_STEPS", 1000);
        string[] kinds = (Environment.GetEnvironmentVariable("DOTLLM_DENSE_DECODE_GATE_KV") ?? "f16,q8_0,q4_0").Split(',');

        using var gguf = GgufFile.Open(modelPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] text = tok.Encode(File.ReadAllText(textPath!));
        Assert.True(text.Length >= prefill + steps + 1, $"text has {text.Length} tokens, need {prefill + steps + 1}");
        string? ptxDir = new[] { Path.Combine(AppContext.BaseDirectory, "ptx") }.FirstOrDefault(Directory.Exists);

        bool pOn = CudaDenseDecodeV2.Enabled;
        int pMin = CudaDenseDecodeV2.MinSeqKv;
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("| KV | depth | steps | PPL scalar | PPL v2 | dPPL | mean KL | max KL | top1 |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
        try
        {
            using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir!);
            model.UseGraphCapture = false;
            int vocab = config.VocabSize;
            foreach (string kind in kinds)
            {
                var caches = new IKvCache[2];
                for (int a = 0; a < 2; a++) caches[a] = model.CreateKvCache(prefill + steps + 4, Kv(kind));
                try
                {
                    // Prefill (identical in both arms: v2 is decode-only), in 1024-row chunks.
                    for (int a = 0; a < 2; a++)
                    {
                        SetArm(a);
                        for (int start = 0; start < prefill; start += 1024)
                        {
                            int n = Math.Min(1024, prefill - start);
                            var ids = text.AsSpan(start, n).ToArray();
                            var pos = Enumerable.Range(start, n).ToArray();
                            model.Forward(ids, pos, 0, caches[a]).Dispose();
                        }
                    }

                    double[] nll = new double[2];
                    double kl = 0, klMax = 0;
                    int top = 0;
                    var lp = new double[2][];
                    for (int a = 0; a < 2; a++) lp[a] = new double[vocab];
                    long dense0 = CudaKernels.AttentionMmaDecodeDenseLaunches;
                    for (int s = 0; s < steps; s++)
                    {
                        int p = prefill + s;
                        int[] id = [text[p]];
                        int[] ps = [p];
                        int target = text[p + 1];
                        int[] arg = new int[2];
                        for (int a = 0; a < 2; a++)
                        {
                            SetArm(a);
                            using var l = model.Forward(id, ps, 0, caches[a]);
                            float* row = (float*)l.DataPointer;
                            double mx = double.NegativeInfinity;
                            int am = 0;
                            for (int i = 0; i < vocab; i++) { if (row[i] > mx) { mx = row[i]; am = i; } }
                            double sum = 0;
                            for (int i = 0; i < vocab; i++) sum += Math.Exp(row[i] - mx);
                            double lse = mx + Math.Log(sum);
                            for (int i = 0; i < vocab; i++) lp[a][i] = row[i] - lse;
                            nll[a] -= lp[a][target];
                            arg[a] = am;
                        }
                        double k = 0;
                        for (int i = 0; i < vocab; i++) k += Math.Exp(lp[0][i]) * (lp[0][i] - lp[1][i]);
                        kl += k; klMax = Math.Max(klMax, k);
                        if (arg[1] == arg[0]) top++;
                    }
                    long dense = CudaKernels.AttentionMmaDecodeDenseLaunches - dense0;
                    Assert.Equal((long)steps * config.NumLayers, dense);   // arm B ran v2 on every layer, arm A never
                    double pplA = Math.Exp(nll[0] / steps), pplB = Math.Exp(nll[1] / steps);
                    double dB = (pplB / pplA - 1) * 100;
                    sb.AppendLine($"| {kind} | {prefill} | {steps} | {pplA:F4} | {pplB:F4} | {dB:+0.000;-0.000}% | {kl / steps:E2} | {klMax:E2} | {top}/{steps} |");
                    _out.WriteLine(sb.ToString());
                    Assert.True(dB <= 0.30, $"{kind}: dense v2 PPL {pplB:F4} vs scalar {pplA:F4}: {dB:F3}% > +0.30%");
                }
                finally { foreach (var c in caches) c?.Dispose(); }
            }
        }
        finally
        {
            CudaDenseDecodeV2.Enabled = pOn; CudaDenseDecodeV2.MinSeqKv = pMin;
            string? outFile = Environment.GetEnvironmentVariable("DOTLLM_DENSE_DECODE_GATE_OUT");
            if (!string.IsNullOrEmpty(outFile)) File.AppendAllText(outFile, sb.ToString());
        }
    }
}