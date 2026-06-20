using System;
using System.IO;
using DotLLM.Core.Attention;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>
/// Model-level parity benchmark for the TurboQuant KV cache (KV Phase 1 acceptance). The codec /
/// cache unit tests prove per-vector reconstruction and unbiased scores; this measures the effect
/// on a REAL model's next-token distribution end-to-end, which is the lossy-quality acceptance the
/// unit tests cannot give.
///
/// <para>It teacher-forces a fixed text through Llama-3.1-8B-Q4_K_M on CPU with three KV caches —
/// a full-precision <see cref="SimpleKvCache"/> reference, TurboQuant 4-bit (<c>tq4</c>), and
/// TurboQuant 4-bit + QJL (<c>tq4q</c>) — exercising BOTH the prefill and the incremental
/// single-token decode path. It reports per-variant perplexity and, against the F32 reference,
/// the mean/max abs logit delta, top-1 argmax agreement, and mean KL divergence.</para>
///
/// <para><b>Gated, never runs in the normal suite.</b> Set <c>DOTLLM_TURBOQUANT_PARITY_GGUF</c> to a
/// model path to run it (8B on CPU takes minutes). Without the env var it skips — even on a box that
/// happens to have the model cached.</para>
/// </summary>
public sealed unsafe class TurboQuantKvParityTests
{
    private readonly ITestOutputHelper _output;
    public TurboQuantKvParityTests(ITestOutputHelper output) => _output = output;

    // A factual English paragraph — enough tokens to give a stable perplexity over the decode region.
    private const string Text =
        "The Apollo program was a series of human spaceflight missions undertaken by the United States. " +
        "Its goal was to land astronauts on the Moon and bring them safely back to Earth. " +
        "In nineteen sixty-nine, Apollo eleven achieved that goal when Neil Armstrong became the first person " +
        "to walk on the lunar surface, followed shortly afterwards by Buzz Aldrin.";

    [SkippableFact]
    public void TurboQuantKv_NextTokenParity_VsFullPrecision()
    {
        string? modelPath = Environment.GetEnvironmentVariable("DOTLLM_TURBOQUANT_PARITY_GGUF");
        Skip.If(string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath),
            "Set DOTLLM_TURBOQUANT_PARITY_GGUF to a GGUF path to run the TurboQuant model-level parity benchmark.");

        var gguf = GgufFile.Open(modelPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int[] all = tokenizer.Encode(Text);
        // Bound the work: 8 prefill + up to 48 decoded positions keeps the 8B CPU run to a few minutes.
        int total = Math.Min(all.Length, 56);
        int prefillLen = 8;
        Skip.If(total <= prefillLen + 4, $"Tokenized text too short ({total}) for a meaningful decode region.");
        var tokens = new int[total];
        Array.Copy(all, tokens, total);

        int vocab = config.VocabSize;
        ulong seed = 0xC0FFEE_1234_5678UL;
        _output.WriteLine($"model={Path.GetFileName(modelPath)} layers={config.NumLayers} kvHeads={config.NumKvHeads} " +
                          $"headDim={config.HeadDim} vocab={vocab} | prefill={prefillLen} decode={total - prefillLen}");

        using var model = TransformerModel.LoadFromGguf(gguf, config);

        // Reference: full-precision KV. Candidates: TurboQuant 4-bit, and 4-bit + QJL.
        float[][] reference;
        using (var f32 = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, total))
            reference = RunTeacherForced(model, f32, tokens, prefillLen, vocab);

        float[][] tq4, tq4q;
        using (var c = new TurboQuantKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, total, 4, seed, useQjl: false))
            tq4 = RunTeacherForced(model, c, tokens, prefillLen, vocab);
        using (var c = new TurboQuantKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, total, 4, seed, useQjl: true))
            tq4q = RunTeacherForced(model, c, tokens, prefillLen, vocab);

        // Per-variant perplexity over the predicted tokens (positions prefillLen-1 .. total-2 predict
        // tokens prefillLen .. total-1).
        double pplRef = Perplexity(reference, tokens, prefillLen);
        double pplTq4 = Perplexity(tq4, tokens, prefillLen);
        double pplTq4q = Perplexity(tq4q, tokens, prefillLen);

        var mTq4 = CompareToReference(reference, tq4);
        var mTq4q = CompareToReference(reference, tq4q);

        _output.WriteLine("");
        _output.WriteLine("variant   perplexity   top1-agree   mean|Δlogit|   max|Δlogit|   meanKL(ref‖x)");
        _output.WriteLine($"F32 ref   {pplRef,9:F4}   {"—",10}   {"—",12}   {"—",11}   {"—",12}");
        _output.WriteLine($"tq4       {pplTq4,9:F4}   {mTq4.Top1Agreement,9:P1}   {mTq4.MeanAbsLogitDelta,12:F4}   {mTq4.MaxAbsLogitDelta,11:F4}   {mTq4.MeanKl,12:F5}");
        _output.WriteLine($"tq4q      {pplTq4q,9:F4}   {mTq4q.Top1Agreement,9:P1}   {mTq4q.MeanAbsLogitDelta,12:F4}   {mTq4q.MaxAbsLogitDelta,11:F4}   {mTq4q.MeanKl,12:F5}");
        _output.WriteLine("");
        _output.WriteLine($"PPL delta vs F32:  tq4 {(pplTq4 - pplRef):+0.0000;-0.0000}   tq4q {(pplTq4q - pplRef):+0.0000;-0.0000}");

        // Sanity bounds (loose — the benchmark's value is the reported numbers, not a tight gate).
        // 4-bit TurboQuant should track the F32 distribution closely: high argmax agreement and a
        // small, finite logit delta. A broken codec (wrong rotation/centroids) collapses these.
        Assert.True(double.IsFinite(pplTq4) && double.IsFinite(pplTq4q), "perplexity must be finite");
        Assert.True(mTq4.Top1Agreement >= 0.80, $"tq4 top-1 agreement too low: {mTq4.Top1Agreement:P1}");
        Assert.True(mTq4q.Top1Agreement >= 0.80, $"tq4q top-1 agreement too low: {mTq4q.Top1Agreement:P1}");
        Assert.True(mTq4.MeanAbsLogitDelta < 2.0, $"tq4 mean |Δlogit| too high: {mTq4.MeanAbsLogitDelta:F4}");
    }

    // Teacher-forces `tokens` through the model with the given cache: prefill [0,prefillLen), then
    // decode one token at a time. Returns the logit row (length vocab) at each position that predicts
    // a NEXT token, i.e. positions [prefillLen-1, total-1) → predictions for tokens [prefillLen, total).
    private static float[][] RunTeacherForced(TransformerModel model, IKvCache cache, int[] tokens, int prefillLen, int vocab)
    {
        int total = tokens.Length;
        var rows = new float[total - prefillLen][]; // prediction for tokens[prefillLen .. total-1]

        // Prefill.
        var prefillTokens = new int[prefillLen];
        var prefillPos = new int[prefillLen];
        for (int i = 0; i < prefillLen; i++) { prefillTokens[i] = tokens[i]; prefillPos[i] = i; }
        using (var logits = model.Forward(prefillTokens, prefillPos, -1, cache))
        {
            // Last prefill row (position prefillLen-1) predicts tokens[prefillLen].
            rows[0] = CopyRow(logits, prefillLen - 1, vocab);
        }

        // Decode: feed token[t] at position t, output predicts token[t+1].
        for (int t = prefillLen; t < total - 1; t++)
        {
            using var logits = model.Forward(new[] { tokens[t] }, new[] { t }, -1, cache);
            rows[t - prefillLen + 1] = CopyRow(logits, 0, vocab);
        }
        return rows;
    }

    private static float[] CopyRow(DotLLM.Core.Tensors.ITensor logits, int row, int vocab)
    {
        var dst = new float[vocab];
        float* src = (float*)logits.DataPointer + (long)row * vocab;
        for (int i = 0; i < vocab; i++) dst[i] = src[i];
        return dst;
    }

    // Perplexity over the predicted tokens: exp(mean -log softmax(row)[actualNextToken]).
    private static double Perplexity(float[][] rows, int[] tokens, int prefillLen)
    {
        double nll = 0;
        int n = rows.Length;
        for (int i = 0; i < n; i++)
        {
            int actual = tokens[prefillLen + i]; // row i predicts tokens[prefillLen+i]
            nll += -LogSoftmaxAt(rows[i], actual);
        }
        return Math.Exp(nll / n);
    }

    private static double LogSoftmaxAt(float[] row, int idx)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < row.Length; i++) if (row[i] > max) max = row[i];
        double sum = 0;
        for (int i = 0; i < row.Length; i++) sum += Math.Exp(row[i] - max);
        return (row[idx] - max) - Math.Log(sum);
    }

    private readonly record struct RefMetrics(double Top1Agreement, double MeanAbsLogitDelta, double MaxAbsLogitDelta, double MeanKl);

    private static RefMetrics CompareToReference(float[][] reference, float[][] candidate)
    {
        int n = reference.Length;
        int agree = 0;
        double sumAbs = 0, maxAbs = 0, sumKl = 0;
        long count = 0;
        for (int i = 0; i < n; i++)
        {
            float[] r = reference[i], c = candidate[i];
            if (Argmax(r) == Argmax(c)) agree++;
            for (int v = 0; v < r.Length; v++)
            {
                double d = Math.Abs((double)r[v] - c[v]);
                sumAbs += d;
                if (d > maxAbs) maxAbs = d;
            }
            count += r.Length;
            sumKl += KlRefVsCandidate(r, c);
        }
        return new RefMetrics((double)agree / n, sumAbs / count, maxAbs, sumKl / n);
    }

    // KL(softmax(ref) ‖ softmax(cand)) = Σ p_ref (logp_ref - logp_cand).
    private static double KlRefVsCandidate(float[] r, float[] c)
    {
        double rMax = double.NegativeInfinity, cMax = double.NegativeInfinity;
        for (int i = 0; i < r.Length; i++) { if (r[i] > rMax) rMax = r[i]; if (c[i] > cMax) cMax = c[i]; }
        double rSum = 0, cSum = 0;
        for (int i = 0; i < r.Length; i++) { rSum += Math.Exp(r[i] - rMax); cSum += Math.Exp(c[i] - cMax); }
        double logRSum = Math.Log(rSum), logCSum = Math.Log(cSum);
        double kl = 0;
        for (int i = 0; i < r.Length; i++)
        {
            double logPr = (r[i] - rMax) - logRSum;
            double pr = Math.Exp(logPr);
            if (pr <= 0) continue;
            double logPc = (c[i] - cMax) - logCSum;
            kl += pr * (logPr - logPc);
        }
        return kl;
    }

    private static int Argmax(float[] row)
    {
        int best = 0; float bv = row[0];
        for (int i = 1; i < row.Length; i++) if (row[i] > bv) { bv = row[i]; best = i; }
        return best;
    }
}
