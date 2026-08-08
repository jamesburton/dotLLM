using System.Diagnostics;
using System.Numerics.Tensors;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.HuggingFace;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Benchmarks.Profile;

/// <summary>
/// Standalone CPU benchmark harness for MTP (Multi-Token Prediction) self-speculative decoding
/// (issue #253). Drives <see cref="MtpSpeculativeDecoder"/> directly against a real loaded model —
/// there is no CLI/server wiring for MTP yet (tracked as follow-up in docs/SPECULATIVE.md), so this
/// exists purely for real-model measurement without touching <c>TextGenerator</c>'s production
/// decode loops.
///
/// Runs interleaved A/B trials: plain greedy decode (baseline) vs. MTP self-speculative decode,
/// alternating so both share the same thermal/scheduling conditions, then reports median tok/s and
/// the MTP acceptance rate.
///
/// Usage: dotnet run -- mtp-bench --model &lt;path.gguf&gt; [--threads N] [--decode N] [--k N] [--repeats N]
/// </summary>
internal static class MtpBenchProfile
{
    private const string DefaultPrompt =
        "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. " +
        "The capital of Spain is Madrid. The capital of Portugal is Lisbon. Write a short paragraph " +
        "describing how speculative decoding accelerates large language model inference.";

    public static int Run(string[] args)
    {
        string? modelPath = GetOption(args, "--model") ?? Environment.GetEnvironmentVariable("DOTLLM_BENCH_MODEL_PATH");
        if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
        {
            Console.Error.WriteLine("Usage: mtp-bench --model <path.gguf> [--threads N] [--prefill N] [--decode N] [--k N] [--repeats N] [--warmup N]");
            Console.Error.WriteLine($"  (or set DOTLLM_BENCH_MODEL_PATH). Got: '{modelPath}'");
            return 2;
        }

        int threads = GetIntOption(args, "--threads", 0); // 0 = auto (all cores)
        int prefillTokens = GetIntOption(args, "--prefill", 256);
        int decodeTokens = GetIntOption(args, "--decode", 32);
        int k = GetIntOption(args, "--k", 4);
        int repeats = GetIntOption(args, "--repeats", 1);
        int warmup = GetIntOption(args, "--warmup", 0);
        string prompt = GetOption(args, "--prompt") ?? DefaultPrompt;

        var threading = new ThreadingConfig(threads);
        Console.WriteLine($"Model: {modelPath}");
        Console.WriteLine($"Threads: {threading.EffectiveThreadCount} (requested {threads})");

        var loadSw = Stopwatch.StartNew();
        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var typedModel = ModelLoader.CreateCpuModelFromGguf(gguf, config, threading);
        loadSw.Stop();

        Console.WriteLine($"Architecture: {config.Architecture}  Layers={config.NumLayers}  Hidden={config.HiddenSize}  Vocab={config.VocabSize}");
        Console.WriteLine($"Load time: {loadSw.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine($"SupportsMtp: {typedModel.SupportsMtp}");

        if (!typedModel.SupportsMtp)
        {
            Console.Error.WriteLine();
            Console.Error.WriteLine("ERROR: model.SupportsMtp is false — no MTP head (nextn.* tensors) was detected in this GGUF,");
            Console.Error.WriteLine("or the architecture dispatch didn't route to a model that implements it. MTP benchmark cannot run.");
            Console.Error.WriteLine("Baseline-only greedy decode numbers below are still valid for the no-MTP comparison point.");
        }

        int[] promptTokens = tokenizer.Encode(prompt).ToArray();
        int prefillLen = Math.Min(promptTokens.Length, prefillTokens);
        int[] prefill = promptTokens[..prefillLen];
        Console.WriteLine($"Prompt tokens: {promptTokens.Length} (using {prefillLen} for prefill)");
        Console.WriteLine($"Decode tokens per trial: {decodeTokens}  K (MTP candidates): {k}  Repeats: {repeats}  Warmup trials: {warmup}");
        Console.WriteLine();

        int cacheCapacity = prefillLen + decodeTokens + k + 8;

        var baselineResults = new List<TrialResult>();
        var mtpResults = new List<TrialResult>();

        int totalTrials = repeats + warmup;
        for (int trial = 0; trial < totalTrials; trial++)
        {
            bool isWarmup = trial < warmup;
            string tag = isWarmup ? "warmup" : $"trial {trial - warmup + 1}/{repeats}";

            // Interleaved A/B: baseline then MTP each round, same prompt/params, same process.
            var baseline = RunBaseline(typedModel, tokenizer, prefill, prefillLen, decodeTokens, cacheCapacity, config);
            Console.WriteLine($"[{tag}] baseline : {baseline.TokPerSec,7:F2} tok/s  prefill {baseline.PrefillTokPerSec,7:F1} tok/s  ({baseline.DecodeMs:F0} ms / {baseline.TokensGenerated} tok)");

            TrialResult? mtp = null;
            if (typedModel.SupportsMtp)
            {
                mtp = RunMtp(typedModel, tokenizer, prefill, prefillLen, decodeTokens, k, cacheCapacity, config);
                Console.WriteLine($"[{tag}] mtp      : {mtp.Value.TokPerSec,7:F2} tok/s  prefill {mtp.Value.PrefillTokPerSec,7:F1} tok/s  ({mtp.Value.DecodeMs:F0} ms / {mtp.Value.TokensGenerated} tok)  accept={mtp.Value.AcceptanceRate:P0} (drafted {mtp.Value.Drafted}, accepted {mtp.Value.Accepted})");
            }

            if (!isWarmup)
            {
                baselineResults.Add(baseline);
                if (mtp.HasValue) mtpResults.Add(mtp.Value);
            }
        }

        Console.WriteLine();
        Console.WriteLine("──────── SUMMARY (median across non-warmup trials) ────────");
        double baseMedian = Median(baselineResults.Select(r => r.TokPerSec).ToArray());
        double basePrefillMedian = Median(baselineResults.Select(r => r.PrefillTokPerSec).ToArray());
        Console.WriteLine($"  baseline (no MTP) decode  = {baseMedian,7:F2} tok/s   prefill = {basePrefillMedian,7:F1} tok/s");

        if (mtpResults.Count > 0)
        {
            double mtpMedian = Median(mtpResults.Select(r => r.TokPerSec).ToArray());
            double mtpPrefillMedian = Median(mtpResults.Select(r => r.PrefillTokPerSec).ToArray());
            double acceptMedian = Median(mtpResults.Select(r => r.AcceptanceRate).ToArray());
            Console.WriteLine($"  MTP decode                = {mtpMedian,7:F2} tok/s   prefill = {mtpPrefillMedian,7:F1} tok/s   speedup = {mtpMedian / baseMedian,5:F2}x   acceptance = {acceptMedian:P1}");
        }
        else
        {
            Console.WriteLine("  MTP decode                = N/A (model does not support MTP)");
        }
        Console.WriteLine("─────────────────────────────────────────────────────────");

        return 0;
    }

    private readonly struct TrialResult
    {
        public required double TokPerSec { get; init; }
        public required double PrefillTokPerSec { get; init; }
        public required double DecodeMs { get; init; }
        public required int TokensGenerated { get; init; }
        public double AcceptanceRate { get; init; }
        public int Drafted { get; init; }
        public int Accepted { get; init; }
    }

    private static TrialResult RunBaseline(
        DotLLM.Core.Models.IModel model, DotLLM.Tokenizers.ITokenizer tokenizer,
        int[] prefill, int prefillLen, int decodeTokens, int cacheCapacity, DotLLM.Core.Models.ModelConfig config)
    {
        using var kv = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, cacheCapacity);

        int[] prefillPositions = new int[prefillLen];
        for (int i = 0; i < prefillLen; i++) prefillPositions[i] = i;

        var prefillSw = Stopwatch.StartNew();
        int currentToken;
        using (var t = model.Forward(prefill, prefillPositions, deviceId: -1, kv))
            currentToken = ArgmaxFirstRow(t, prefillLen - 1, config.VocabSize);
        prefillSw.Stop();

        int pos = prefillLen;
        var sw = Stopwatch.StartNew();
        int generated = 0;
        int[] tokBuf = new int[1];
        int[] posBuf = new int[1];
        for (int i = 0; i < decodeTokens && pos < cacheCapacity; i++)
        {
            tokBuf[0] = currentToken;
            posBuf[0] = pos;
            using var t = model.Forward(tokBuf, posBuf, deviceId: -1, kv);
            currentToken = ArgmaxFirstRow(t, 0, config.VocabSize);
            pos++;
            generated++;
        }
        sw.Stop();

        return new TrialResult
        {
            TokPerSec = generated / (sw.Elapsed.TotalMilliseconds / 1000.0),
            PrefillTokPerSec = prefillLen / prefillSw.Elapsed.TotalSeconds,
            DecodeMs = sw.Elapsed.TotalMilliseconds,
            TokensGenerated = generated,
        };
    }

    private static TrialResult RunMtp(
        DotLLM.Core.Models.IModel model, DotLLM.Tokenizers.ITokenizer tokenizer,
        int[] prefill, int prefillLen, int decodeTokens, int k, int cacheCapacity, DotLLM.Core.Models.ModelConfig config)
    {
        using var kv = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, cacheCapacity);
        using var mtpState = model.CreateMtpState()!;
        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });

        int[] prefillPositions = new int[prefillLen];
        for (int i = 0; i < prefillLen; i++) prefillPositions[i] = i;

        var prefillSw = Stopwatch.StartNew();
        int lastToken;
        using (var t = model.Forward(prefill, prefillPositions, deviceId: -1, kv))
            lastToken = ArgmaxFirstRow(t, prefillLen - 1, config.VocabSize);
        prefillSw.Stop();

        var generatedIds = new List<int>(prefill) { lastToken };
        int position = prefillLen;

        var sw = Stopwatch.StartNew();
        int generated = 0;
        int totalDrafted = 0, totalAccepted = 0;
        Span<int> outputBuffer = new int[k + 1];
        int guard = 0;
        while (generated < decodeTokens && position < cacheCapacity && guard++ < decodeTokens * 4)
        {
            int remaining = decodeTokens - generated;
            int thisK = Math.Min(k, remaining);
            if (thisK <= 0) break;

            var result = decoder.DraftAndVerify(
                model, kv, mtpState, pipeline, generatedIds,
                constraint: null, position, config.VocabSize, thisK, outputBuffer);

            if (result.AcceptedCount == 0) break;

            totalDrafted += result.DraftedCount;
            totalAccepted += result.AcceptedCount;

            for (int i = 0; i < result.AcceptedCount && generated < decodeTokens; i++)
            {
                generatedIds.Add(outputBuffer[i]);
                generated++;
            }
            position += result.AcceptedCount;
        }
        sw.Stop();

        return new TrialResult
        {
            TokPerSec = generated / (sw.Elapsed.TotalMilliseconds / 1000.0),
            PrefillTokPerSec = prefillLen / prefillSw.Elapsed.TotalSeconds,
            DecodeMs = sw.Elapsed.TotalMilliseconds,
            TokensGenerated = generated,
            Drafted = totalDrafted,
            Accepted = totalAccepted,
            // Approximate acceptance rate: accepted tokens (including any bonus token beyond the
            // K drafted per round) over drafted tokens. Not llama.cpp's exact definition (which
            // excludes the bonus token) but a reasonable single-number proxy for "how often did
            // the MTP head's guesses land" for this harness's purposes.
            AcceptanceRate = totalDrafted > 0 ? (double)totalAccepted / totalDrafted : 0,
        };
    }

    private static unsafe int ArgmaxFirstRow(ITensor logits, int row, int vocabSize)
    {
        var span = new ReadOnlySpan<float>((void*)(logits.DataPointer + (long)row * vocabSize * sizeof(float)), vocabSize);
        return TensorPrimitives.IndexOfMax(span);
    }

    private static double Median(double[] values)
    {
        if (values.Length == 0) return double.NaN;
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    private static string? GetOption(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (args[i] == name) return args[i + 1];
        return null;
    }

    private static int GetIntOption(string[] args, string name, int defaultValue)
    {
        string? s = GetOption(args, name);
        return s is not null && int.TryParse(s, out int v) ? v : defaultValue;
    }
}
