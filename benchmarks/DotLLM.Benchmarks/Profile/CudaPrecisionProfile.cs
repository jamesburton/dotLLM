using System.Diagnostics;
using System.Globalization;
using System.Text;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Engine.Evaluation;
using DotLLM.Models.Gguf;

namespace DotLLM.Benchmarks.Profile;

/// <summary>
/// Issue #1109 CUDA precision study harness (RTX 3060 / Bonsai-2 27B PQ2_0). Loads the model once and runs
/// in-process, interleaved A/B variants of one operation class against an F32-accumulate (or F16-KV) reference:
/// full-vocabulary KL(ref ‖ variant) per scored row, top-1 agreement and the perplexity of the scored rows.
/// </summary>
/// <remarks>
/// <para>Experiments (<c>--exp</c>):</para>
/// <list type="bullet">
/// <item><c>gemm-kl</c>: prefill GEMM accumulation (cuBLAS COMPUTE_32F reference, COMPUTE_16F, COMPUTE_16F with F32
/// promotion every 256 / 1024 elements, and the PQ2_0 W2A8 dp4a MMQ) on 3 fixed prompts plus a wikitext window;
/// every row of each forward is scored (all-row logits).</item>
/// <item><c>gemm-speed</c>: prefill tok/s of the same variants at 512 / 2048 / 8192 tokens, interleaved, median.</item>
/// <item><c>probe</c>: <see cref="GemmPrecisionProbe"/> over one prefill tile: per-projection error by K.</item>
/// <item><c>attn</c>: flash-prefill accumulators (<see cref="CudaFlashAccumulatorExperiment"/>) at long depth.</item>
/// <item><c>kv</c>: KV storage formats at depth, prefill-chunk and decode rows, plus decode tok/s.</item>
/// </list>
/// <para>Depth scoring (<c>attn</c>, <c>kv</c>) uses 16-token forwards with last-row logits: 16 &gt; 8 rows keeps every
/// projection on the prefill GEMM and attention on the flash prefill kernel, while the LM head stays a one-row
/// GEMV, so the 2.5 GB F16 LM-head dequant scratch an all-row forward needs (#1099) never fits next to a long KV.</para>
/// </remarks>
internal static class CudaPrecisionProfile
{
    private const int DepthChunk = 16;

    private static readonly string[] Prompts =
    [
        "<|im_start|>user\nWrite a C# method that returns the n-th Fibonacci number using an iterative loop, "
        + "and explain its time complexity in one sentence.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of Australia, and why was it chosen over Sydney and Melbourne? "
        + "Answer in three short sentences.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nA train leaves at 14:35 and arrives at 18:10. The journey includes a 25 minute stop. "
        + "How long was the train actually moving? Show your working.<|im_end|>\n<|im_start|>assistant\n",
    ];

    public static int Run(string[] args)
    {
        string? modelPath = Opt(args, "--model") ?? Environment.GetEnvironmentVariable("DOTLLM_BONSAI_PQ2_0_GGUF");
        string corpus = Opt(args, "--corpus") ?? ".docs/corpora/wikitext2-test.raw";
        string exp = Opt(args, "--exp") ?? "gemm-kl";
        if (modelPath is null || !File.Exists(modelPath) || !File.Exists(corpus))
        {
            Console.Error.WriteLine("Usage: precision-cuda --exp gemm-kl|gemm-speed|probe|attn|kv --model <gguf> --corpus <wikitext raw> ...");
            return 2;
        }

        Console.WriteLine($"# precision-cuda {exp}  ({DateTime.Now:yyyy-MM-dd HH:mm:ss})");
        Console.WriteLine($"args: {string.Join(' ', args)}");
        GpuSnapshot("start");

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var sw = Stopwatch.StartNew();
        var (loaded, _) = CudaModelLoader.CreateFromGguf(gguf, config, 0);
        using var model = loaded as CudaQwen3HybridDenseTransformerModel
            ?? throw new InvalidOperationException($"Expected the qwen35 hybrid dense CUDA model, got {loaded.GetType().Name}.");
        Console.WriteLine($"loaded in {sw.Elapsed.TotalSeconds:F1}s: hidden={config.HiddenSize} ffn={config.IntermediateSize} "
                          + $"layers={config.NumLayers} heads={config.NumAttentionHeads}/{config.NumKvHeads} hd={config.HeadDim} vocab={config.VocabSize}");
        // Bonsai's CUDA host never calls CudaGemm.ConfigureDefault, so G1 (COMPUTE_16F) is only on when the env forces it.
        Console.WriteLine($"defaults: CudaGemm.Use16FCompute={CudaGemm.Use16FCompute} PQ2_0 MMQ={CudaSmallSGemvDispatch.UseMmq} "
                          + $"dp4a(1..8 rows)={CudaSmallSGemvDispatch.UseDp4a} flash acc={CudaFlashAccumulatorExperiment.Mode} kv={model.KvStorageLabel}");

        int needTokens = Int(args, "--corpus-tokens", 0);
        int[] corpusTokens = Tokenize(corpus, tokenizer, gguf, needTokens > 0 ? needTokens : 140_000);
        Console.WriteLine($"corpus tokens: {corpusTokens.Length}");

        int rc = exp switch
        {
            "gemm-kl" => GemmKl(model, tokenizer, corpusTokens, args),
            "gemm-speed" => GemmSpeed(model, corpusTokens, args),
            "probe" => Probe(model, corpusTokens, args),
            "attn" => Attention(model, corpusTokens, args),
            "kv" => KvFormats(model, corpusTokens, args),
            _ => 2,
        };
        GpuSnapshot("end");
        return rc;
    }

    // ───────────────────────────── experiment 1/4: prefill GEMM ─────────────────────────────

    private sealed record GemmVariant(string Name, bool F16, int PromoteChunk, bool Mmq)
    {
        public void Apply()
        {
            CudaGemm.Use16FCompute = F16;
            CudaGemm.Promote16FChunk = PromoteChunk;
            CudaSmallSGemvDispatch.MmqOverride = Mmq;
        }
    }

    private static readonly GemmVariant GemmRef = new("F32-acc (ref)", false, 0, false);

    private static List<GemmVariant> GemmVariants(string[] args)
    {
        var all = new List<GemmVariant>
        {
            GemmRef,
            new("F16-acc", true, 0, false),
            new("F16-acc, F32 every 1024", true, 1024, false),
            new("F16-acc, F32 every 256", true, 256, false),
            new("W2A8 dp4a MMQ", false, 0, true),
        };
        string? only = Opt(args, "--variants");
        if (only is null) return all;
        var names = only.Split(',');
        return all.Where((v, i) => i == 0 || names.Any(n => v.Name.StartsWith(n, StringComparison.OrdinalIgnoreCase))).ToList();
    }

    private static int GemmKl(CudaQwen3HybridDenseTransformerModel model, DotLLM.Tokenizers.ITokenizer tokenizer,
        int[] corpus, string[] args)
    {
        int windowStart = Int(args, "--window-start", 20_000);
        int prefix = Int(args, "--window-prefix", 1024);
        int scored = Int(args, "--window-scored", 1024);
        int[] window = corpus[windowStart..(windowStart + prefix + scored + 1)];
        var inputs = new List<(string Name, int[] Tokens, int Prefix, int Scored)>();
        for (int i = 0; i < Prompts.Length; i++)
        {
            int[] t = tokenizer.Encode(Prompts[i]);
            inputs.Add(($"prompt{i + 1} ({t.Length} tok)", t, 0, t.Length - 1));
        }
        inputs.Add(($"wikitext [{windowStart}+{prefix}, {scored} rows]", window, prefix, scored));

        var variants = GemmVariants(args);
        // Reference rows per input, as log-softmax.
        var refRows = new List<List<float[]>>();
        GemmRef.Apply();
        foreach (var inp in inputs)
        {
            var rows = new List<float[]>();
            ScoreAllRows(model, inp.Tokens, inp.Prefix, inp.Scored, (_, l) => rows.Add(LogSoftmax(l)));
            refRows.Add(rows);
        }

        var table = new StringBuilder();
        table.AppendLine(KlStats.Header("variant / input"));
        foreach (var v in variants)
        {
            v.Apply();
            var total = new KlStats();
            for (int i = 0; i < inputs.Count; i++)
            {
                var inp = inputs[i];
                var s = new KlStats();
                int r = 0;
                ScoreAllRows(model, inp.Tokens, inp.Prefix, inp.Scored, (pos, l) =>
                {
                    int target = inp.Tokens[pos + 1];
                    s.Add(refRows[i][r], l, target);
                    total.Add(refRows[i][r], l, target);
                    r++;
                });
                table.AppendLine(s.Row($"{v.Name} / {inp.Name}"));
            }
            table.AppendLine(total.Row($"**{v.Name} / all**"));
            Console.WriteLine(table.ToString());
            table.Clear();
        }
        GemmRef.Apply();
        return 0;
    }

    private static int GemmSpeed(CudaQwen3HybridDenseTransformerModel model, int[] corpus, string[] args)
    {
        int reps = Int(args, "--reps", 3);
        int[] lengths = Ints(args, "--lengths", [512, 2048, 8192]);
        var variants = GemmVariants(args);
        var times = new Dictionary<(string, int), List<double>>();
        // Warm-up: every variant once at the shortest length (JIT, cuBLAS heuristics, module loads, scratch growth).
        foreach (var v in variants)
        {
            v.Apply();
            TimedPrefill(model, corpus, lengths.Max());
        }
        for (int rep = 0; rep < reps; rep++)
        {
            foreach (int len in lengths)
            {
                foreach (var v in variants)
                {
                    v.Apply();
                    double s = TimedPrefill(model, corpus, len);
                    (times.TryGetValue((v.Name, len), out var l) ? l : times[(v.Name, len)] = []).Add(s);
                    Console.WriteLine($"rep {rep + 1} len {len} {v.Name}: {len / s:F1} tok/s");
                }
            }
            GpuSnapshot($"after rep {rep + 1}");
        }
        GemmRef.Apply();
        Console.WriteLine();
        Console.WriteLine("| variant | " + string.Join(" | ", lengths.Select(l => $"{l} tok/s (median, min-max)")) + " | " +
                          string.Join(" | ", lengths.Select(l => $"x vs ref @{l}")) + " |");
        Console.WriteLine("|---|" + string.Concat(Enumerable.Repeat("---:|", lengths.Length * 2)));
        foreach (var v in variants)
        {
            var sb = new StringBuilder($"| {v.Name} |");
            foreach (int len in lengths)
            {
                var ts = times[(v.Name, len)].Select(s => len / s).ToList();
                sb.Append(CultureInfo.InvariantCulture, $" {Median(ts):F1} ({ts.Min():F1}-{ts.Max():F1}) |");
            }
            foreach (int len in lengths)
            {
                double refMed = Median(times[(GemmRef.Name, len)].Select(s => len / s).ToList());
                sb.Append(CultureInfo.InvariantCulture, $" {Median(times[(v.Name, len)].Select(s => len / s).ToList()) / refMed:F3} |");
            }
            Console.WriteLine(sb.ToString());
        }
        return 0;
    }

    private static int Probe(CudaQwen3HybridDenseTransformerModel model, int[] corpus, string[] args)
    {
        int len = Int(args, "--probe-len", 512);
        int start = Int(args, "--window-start", 20_000);
        GemmRef.Apply();
        GemmPrecisionProbe.Current.Reset();
        GemmPrecisionProbe.Enabled = true;
        try
        {
            model.ResetSequenceState();
            using var kv = model.CreateKvCache(len + 16);
            int[] t = corpus[start..(start + len)];
            using var _ = model.Forward(t, Positions(0, len), -1, kv, lastTokenLogitsOnly: true);
        }
        finally
        {
            GemmPrecisionProbe.Enabled = false;
        }
        Console.WriteLine($"probed {GemmPrecisionProbe.Current.Calls} GEMM calls over a {len}-token prefill (promote chunk {GemmPrecisionProbe.ProbeChunk})");
        Console.WriteLine(GemmPrecisionProbe.Current.Report(BonsaiProjection));
        return 0;
    }

    private static string BonsaiProjection(int k, int n) => (k, n) switch
    {
        (5120, 10240) => "GDN attn_qkv",
        (5120, 6144) => "GDN attn_gate (z)",
        (5120, 48) => "GDN alpha/beta",
        (6144, 5120) => "GDN ssm_out / attn_output",
        (5120, 12288) => "attn_q (+gate)",
        (5120, 1024) => "attn_k / attn_v",
        (5120, 17408) => "ffn_gate / ffn_up",
        (17408, 5120) => "ffn_down",
        (5120, 248320) => "lm_head",
        _ => "",
    };

    // ───────────────────────────── experiment 2: flash accumulators ─────────────────────────────

    private static int Attention(CudaQwen3HybridDenseTransformerModel model, int[] corpus, string[] args)
    {
        int[] contexts = Ints(args, "--ctx", [4096, 32768]);
        string kvFormat = Opt(args, "--kv") ?? "f16";
        bool full = (Opt(args, "--mode") ?? "shared") == "full";
        int chunks = Int(args, "--chunks", 64);
        int speedTile = Int(args, "--speed-tile", 1024);
        int speedReps = Int(args, "--reps", 3);
        var modes = (Opt(args, "--variants") ?? "f32,qk16,pv16,pv16t,qk16pv16").Split(',')
            .Select(s => s == "f32" ? FlashAccumulatorMode.F32 : CudaFlashAccumulatorExperiment.Parse(s)).ToList();
        if (modes[0] != FlashAccumulatorMode.F32) modes.Insert(0, FlashAccumulatorMode.F32);

        var (kd, vd) = ParseKv(kvFormat);
        model.ReconfigureKvStorageForExperiment(kd, vd);
        GemmRef.Apply();
        Console.WriteLine($"attention accumulators: kv={model.KvStorageLabel} mode={(full ? "full (variant through the whole prefill)" : "shared prefix (variant in the scored rows only)")} scored rows={chunks} x last-of-{DepthChunk}");

        foreach (int ctx in contexts)
        {
            int capacity = ctx + chunks * DepthChunk + speedTile + 16;
            var table = new StringBuilder(KlStats.Header($"ctx {ctx} / variant") + "\n");
            var speed = new Dictionary<FlashAccumulatorMode, List<double>>();
            if (!full)
            {
                CudaFlashAccumulatorExperiment.Mode = FlashAccumulatorMode.F32;
                model.ResetSequenceState();
                using var kv = model.CreateKvCache(capacity);
                double prefillS = Time(() => model.Forward(corpus[..ctx], Positions(0, ctx), -1, kv, lastTokenLogitsOnly: true).Dispose());
                Console.WriteLine($"ctx {ctx}: reference prefill {ctx / prefillS:F1} tok/s ({prefillS:F1}s)");
                GpuSnapshot($"after {ctx} prefill");
                using var ckpt = model.CheckpointRecurrentState() as IDisposable;
                object? state = ckpt;

                List<float[]>? refRows = null;
                foreach (var mode in modes.Append(FlashAccumulatorMode.F32))
                {
                    bool isRepeat = refRows is not null && mode == FlashAccumulatorMode.F32;
                    CudaFlashAccumulatorExperiment.Mode = mode;
                    model.RestoreRecurrentState(state);
                    kv.Rollback(ctx);
                    int xaccBefore = model.DebugFlashXaccCallsForTest;
                    var rows = new List<float[]>();
                    var s = new KlStats();
                    int r = 0;
                    ScoreDepthChunks(model, kv, corpus, ctx, chunks, (pos, l) =>
                    {
                        if (refRows is null) rows.Add(LogSoftmax(l));
                        else s.Add(refRows[r++], l, corpus[pos + 1]);
                    });
                    if (refRows is null) { refRows = rows; continue; }
                    string engaged = mode == FlashAccumulatorMode.F32 ? "" : $" ({model.DebugFlashXaccCallsForTest - xaccBefore} xacc launches)";
                    table.AppendLine(s.Row($"{ctx} / {(isRepeat ? "F32 repeat (determinism)" : mode.ToString())}{engaged}"));
                }
                CudaFlashAccumulatorExperiment.Mode = FlashAccumulatorMode.F32;

                // Speed: one speedTile-token prefill tile at depth ctx, interleaved, median.
                for (int rep = 0; rep < speedReps; rep++)
                {
                    foreach (var mode in modes)
                    {
                        CudaFlashAccumulatorExperiment.Mode = mode;
                        model.RestoreRecurrentState(state);
                        kv.Rollback(ctx);
                        double t = Time(() => model.Forward(corpus[ctx..(ctx + speedTile)], Positions(ctx, speedTile), -1, kv,
                            lastTokenLogitsOnly: true).Dispose());
                        (speed.TryGetValue(mode, out var l) ? l : speed[mode] = []).Add(t);
                    }
                }
                CudaFlashAccumulatorExperiment.Mode = FlashAccumulatorMode.F32;
                model.RestoreRecurrentState(state);
            }
            else
            {
                List<float[]>? refRows = null;
                foreach (var mode in modes)
                {
                    CudaFlashAccumulatorExperiment.Mode = mode;
                    model.ResetSequenceState();
                    using var kv = model.CreateKvCache(capacity);
                    double prefillS = Time(() => model.Forward(corpus[..ctx], Positions(0, ctx), -1, kv, lastTokenLogitsOnly: true).Dispose());
                    (speed.TryGetValue(mode, out var l) ? l : speed[mode] = []).Add(prefillS);
                    var rows = new List<float[]>();
                    var s = new KlStats();
                    int r = 0;
                    ScoreDepthChunks(model, kv, corpus, ctx, chunks, (pos, lg) =>
                    {
                        if (refRows is null) rows.Add(LogSoftmax(lg));
                        else s.Add(refRows[r++], lg, corpus[pos + 1]);
                    });
                    Console.WriteLine($"ctx {ctx} {mode}: whole prefill {ctx / prefillS:F1} tok/s");
                    if (refRows is null) { refRows = rows; continue; }
                    table.AppendLine(s.Row($"{ctx} / {mode} (full)"));
                }
                CudaFlashAccumulatorExperiment.Mode = FlashAccumulatorMode.F32;
            }

            Console.WriteLine(table.ToString());
            Console.WriteLine($"| ctx {ctx} variant | {(full ? $"whole {ctx}-token prefill" : $"{speedTile}-token tile at depth {ctx}")} tok/s (median of {speed.Values.First().Count}) | x vs F32 |");
            Console.WriteLine("|---|---:|---:|");
            int tokens = full ? ctx : speedTile;
            double refMed = Median(speed[FlashAccumulatorMode.F32].Select(t => tokens / t).ToList());
            foreach (var (mode, ts) in speed)
            {
                double med = Median(ts.Select(t => tokens / t).ToList());
                Console.WriteLine($"| {mode} | {med:F1} | {med / refMed:F3} |");
            }
            Console.WriteLine();
            GpuSnapshot($"after ctx {ctx}");
        }
        return 0;
    }

    // ───────────────────────────── experiment 3: KV formats ─────────────────────────────

    private static int KvFormats(CudaQwen3HybridDenseTransformerModel model, int[] corpus, string[] args)
    {
        int[] contexts = Ints(args, "--ctx", [8192, 32768]);
        int chunks = Int(args, "--chunks", 64);
        int decodeSteps = Int(args, "--decode", 48);
        string[] formats = (Opt(args, "--formats") ?? "f16,q8_0,q4_0,k8v4").Split(',');
        GemmRef.Apply();
        CudaFlashAccumulatorExperiment.Mode = FlashAccumulatorMode.F32;

        foreach (int ctx in contexts)
        {
            List<float[]>? refChunkRows = null, refDecodeRows = null;
            var table = new StringBuilder(KlStats.Header($"ctx {ctx} / KV format / rows") + "\n");
            var speedLines = new List<string>();
            foreach (string f in formats)
            {
                var (kd, vd) = ParseKv(f);
                model.ReconfigureKvStorageForExperiment(kd, vd);
                using var kv = model.CreateKvCache(ctx + chunks * DepthChunk + decodeSteps + 16);
                double prefillS = Time(() => model.Forward(corpus[..ctx], Positions(0, ctx), -1, kv, lastTokenLogitsOnly: true).Dispose());

                var chunkRows = new List<float[]>();
                var cs = new KlStats();
                int r = 0;
                ScoreDepthChunks(model, kv, corpus, ctx, chunks, (pos, l) =>
                {
                    if (refChunkRows is null) chunkRows.Add(LogSoftmax(l));
                    else cs.Add(refChunkRows[r++], l, corpus[pos + 1]);
                });

                // Decode: teacher-forced single-token steps after the scored chunks.
                int pos0 = ctx + chunks * DepthChunk;
                var decodeRows = new List<float[]>();
                var ds = new KlStats();
                var stepTimes = new List<double>();
                for (int i = 0; i < decodeSteps; i++)
                {
                    int p = pos0 + i;
                    float[]? logits = null;
                    double t = Time(() =>
                    {
                        using var lt = model.Forward(corpus.AsSpan(p, 1), [p], -1, kv, lastTokenLogitsOnly: true);
                        logits = LastRow(lt, model.Config.VocabSize).ToArray();
                    });
                    if (i >= 2) stepTimes.Add(t);
                    if (refDecodeRows is null) decodeRows.Add(LogSoftmax(logits));
                    else ds.Add(refDecodeRows[i], logits, corpus[p + 1]);
                }
                long kvBytes = model.ComputeKvBytes(1);
                speedLines.Add($"| {f} | {kvBytes / 1024.0:F1} | {ctx / prefillS:F1} | {1.0 / Median(stepTimes):F2} |");
                Console.WriteLine($"ctx {ctx} {f}: prefill {ctx / prefillS:F1} tok/s, decode {1.0 / Median(stepTimes):F2} tok/s");
                if (refChunkRows is null)
                {
                    refChunkRows = chunkRows;
                    refDecodeRows = decodeRows;
                    continue;
                }
                table.AppendLine(cs.Row($"{ctx} / {f} / prefill-chunk rows"));
                table.AppendLine(ds.Row($"{ctx} / {f} / decode rows"));
            }
            Console.WriteLine(table.ToString());
            Console.WriteLine($"| ctx {ctx} KV | KiB/token (all attn layers) | prefill tok/s | decode tok/s (median of {decodeSteps - 2}) |");
            Console.WriteLine("|---|---:|---:|---:|");
            foreach (string line in speedLines) Console.WriteLine(line);
            Console.WriteLine();
            GpuSnapshot($"after ctx {ctx}");
        }
        model.ReconfigureKvStorageForExperiment(KvCacheDType.F32, KvCacheDType.F32);
        return 0;
    }

    private static (KvCacheDType K, KvCacheDType V) ParseKv(string f) => f switch
    {
        "f16" => (KvCacheDType.F32, KvCacheDType.F32),
        "q8_0" => (KvCacheDType.Q8_0, KvCacheDType.Q8_0),
        "q4_0" => (KvCacheDType.Q4_0, KvCacheDType.Q4_0),
        "k8v4" => (KvCacheDType.Q8_0, KvCacheDType.Q4_0),
        "k4v8" => (KvCacheDType.Q4_0, KvCacheDType.Q8_0),
        _ => throw new ArgumentException($"unknown KV format {f}"),
    };

    // ───────────────────────────── scoring ─────────────────────────────

    /// <summary>Prefills <paramref name="prefix"/> tokens, then scores <paramref name="scored"/> rows with all-row logits in 256-row forwards.</summary>
    private static void ScoreAllRows(CudaQwen3HybridDenseTransformerModel model, int[] tokens, int prefix, int scored,
        Action<int, ReadOnlySpan<float>> onRow)
    {
        const int chunk = 256;
        int vocab = model.Config.VocabSize;
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(prefix + scored + 16);
        if (prefix > 0)
            model.Forward(tokens.AsSpan(0, prefix), Positions(0, prefix), -1, kv, lastTokenLogitsOnly: true).Dispose();
        for (int s = prefix; s < prefix + scored; s += chunk)
        {
            int len = Math.Min(chunk, prefix + scored - s);
            using var t = model.Forward(tokens.AsSpan(s, len), Positions(s, len), -1, kv, lastTokenLogitsOnly: false);
            if (t.ElementCount != (long)len * vocab)
                throw new InvalidOperationException($"expected {len} logit rows, got {t.ElementCount / vocab}");
            for (int r = 0; r < len; r++)
                onRow(s + r, Row(t, r, vocab));
        }
    }

    /// <summary>Scores the last row of each of <paramref name="chunks"/> consecutive 16-token forwards starting at <paramref name="start"/>.</summary>
    private static void ScoreDepthChunks(CudaQwen3HybridDenseTransformerModel model, IKvCache kv, int[] tokens, int start, int chunks,
        Action<int, ReadOnlySpan<float>> onRow)
    {
        for (int c = 0; c < chunks; c++)
        {
            int s = start + c * DepthChunk;
            using var t = model.Forward(tokens.AsSpan(s, DepthChunk), Positions(s, DepthChunk), -1, kv, lastTokenLogitsOnly: true);
            onRow(s + DepthChunk - 1, LastRow(t, model.Config.VocabSize));
        }
    }

    private static unsafe ReadOnlySpan<float> Row(ITensor t, int r, int vocab)
        => new((void*)(t.DataPointer + (nint)((long)r * vocab * sizeof(float))), vocab);

    private static ReadOnlySpan<float> LastRow(ITensor t, int vocab) => Row(t, (int)(t.ElementCount / vocab) - 1, vocab);

    private static float[] LogSoftmax(ReadOnlySpan<float> logits)
    {
        double lse = LogSumExp(logits);
        var o = new float[logits.Length];
        for (int i = 0; i < o.Length; i++) o[i] = (float)(logits[i] - lse);
        return o;
    }

    private static double LogSumExp(ReadOnlySpan<float> x)
    {
        float max = float.NegativeInfinity;
        foreach (float v in x) if (v > max) max = v;
        double sum = 0;
        foreach (float v in x) sum += Math.Exp(v - max);
        return max + Math.Log(sum);
    }

    /// <summary>
    /// KL(ref ‖ variant) per row over the full vocabulary, top-1 agreement and both NLLs of the true next token.
    /// Reported as median / p90 / p99 / max plus outlier rates, since a few massive-activation rows dominate a mean.
    /// </summary>
    private sealed class KlStats
    {
        private readonly List<double> _kl = [];
        private int _top1;
        private double _nllRef, _nllVar;
        private int _nonFinite;

        public void Add(float[] refLogProbs, ReadOnlySpan<float> variantLogits, int target)
        {
            double lse = LogSumExp(variantLogits);
            if (!double.IsFinite(lse)) { _nonFinite++; return; }
            double kl = 0;
            int argRef = 0, argVar = 0;
            for (int i = 0; i < refLogProbs.Length; i++)
            {
                double lr = refLogProbs[i];
                double lv = variantLogits[i] - lse;
                kl += Math.Exp(lr) * (lr - lv);
                if (refLogProbs[i] > refLogProbs[argRef]) argRef = i;
                if (variantLogits[i] > variantLogits[argVar]) argVar = i;
            }
            _kl.Add(Math.Max(kl, 0));
            if (argRef == argVar) _top1++;
            _nllRef -= refLogProbs[target];
            _nllVar -= variantLogits[target] - lse;
        }

        public static string Header(string first) =>
            $"| {first} | rows | KL median | KL p90 | KL p99 | KL max | KL mean | rows KL>1e-3 | rows KL>1e-2 | top-1 agree | PPL ref | PPL variant | PPL Δ% |\n"
            + "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|";

        public string Row(string name)
        {
            int n = _kl.Count;
            if (n == 0) return $"| {name} | 0 (non-finite {_nonFinite}) |";
            var s = _kl.OrderBy(x => x).ToList();
            double P(double q) => s[(int)Math.Clamp(Math.Ceiling(q * n) - 1, 0, n - 1)];
            double pplRef = Math.Exp(_nllRef / n), pplVar = Math.Exp(_nllVar / n);
            return string.Create(CultureInfo.InvariantCulture,
                $"| {name} | {n}{(_nonFinite > 0 ? $" (+{_nonFinite} non-finite)" : "")} | {P(0.5):E2} | {P(0.9):E2} | {P(0.99):E2} | {s[^1]:E2} | {s.Average():E2} | "
                + $"{s.Count(x => x > 1e-3) * 100.0 / n:F1}% | {s.Count(x => x > 1e-2) * 100.0 / n:F1}% | {_top1 * 100.0 / n:F1}% | "
                + $"{pplRef:F4} | {pplVar:F4} | {(pplVar / pplRef - 1) * 100:+0.000;-0.000}% |");
        }
    }

    // ───────────────────────────── helpers ─────────────────────────────

    private static double TimedPrefill(CudaQwen3HybridDenseTransformerModel model, int[] corpus, int len)
    {
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(len + 16);
        return Time(() => model.Forward(corpus.AsSpan(1000, len), Positions(0, len), -1, kv, lastTokenLogitsOnly: true).Dispose());
    }

    private static double Time(Action a)
    {
        var sw = Stopwatch.StartNew();
        a();   // the logits D2H copy synchronizes the stream
        return sw.Elapsed.TotalSeconds;
    }

    private static int[] Positions(int start, int len)
    {
        var p = new int[len];
        for (int i = 0; i < len; i++) p[i] = start + i;
        return p;
    }

    private static int[] Tokenize(string corpus, DotLLM.Tokenizers.ITokenizer tokenizer, GgufFile gguf, int maxTokens)
    {
        using var reader = new StreamReader(corpus);
        bool bos = GgufAddBosResolver.Resolve(gguf.Metadata);
        return CorpusReader.StreamTokens(reader, tokenizer, maxTokens, bosTokenId: bos ? tokenizer.BosTokenId : -1).ToArray();
    }

    private static double Median(List<double> xs)
    {
        var s = xs.OrderBy(x => x).ToList();
        return s.Count % 2 == 1 ? s[s.Count / 2] : (s[s.Count / 2 - 1] + s[s.Count / 2]) / 2;
    }

    private static void GpuSnapshot(string label)
    {
        Console.WriteLine($"[gpu {label}] {RunNvidiaSmi("--query-gpu=clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader")}"
                          + $" | apps: {RunNvidiaSmi("--query-compute-apps=pid,process_name,used_memory --format=csv,noheader").Replace('\n', ';')}");
    }

    private static string RunNvidiaSmi(string args)
    {
        try
        {
            using var p = Process.Start(new ProcessStartInfo("nvidia-smi", args) { RedirectStandardOutput = true, UseShellExecute = false })!;
            string o = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit(10_000);
            return o;
        }
        catch (Exception ex)
        {
            return $"(nvidia-smi failed: {ex.Message})";
        }
    }

    private static string? Opt(string[] args, string name)
    {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }

    private static int Int(string[] args, string name, int dflt)
        => int.TryParse(Opt(args, name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) ? v : dflt;

    private static int[] Ints(string[] args, string name, int[] dflt)
        => Opt(args, name)?.Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray() ?? dflt;
}
