using System.Diagnostics;
using System.Globalization;
using System.Text;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Evaluation;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Benchmarks.Profile;

/// <summary>
/// Issue #1109 Vulkan precision study harness (Strix Halo 8060S coopmat, Meteor Lake Arc scalar). Loads a model once and runs
/// in-process, interleaved A/B variants of one operation class against the shipped kernels as reference: full-vocabulary
/// KL(ref ‖ variant) per scored row, top-1 agreement and the perplexity of the scored rows (the CUDA arm's methodology,
/// <c>precision-cuda</c>).
/// </summary>
/// <remarks>
/// <para>Experiments (<c>--exp</c>):</para>
/// <list type="bullet">
/// <item><c>depth</c>: the hybrid-dense host (Bonsai 2, Tev1) grown to each <c>--ctx</c> in turn (one prefill, continued). At each
/// depth, from one GDN checkpoint + KV rollback: the GQA flash-prefill accumulator variants on <c>--chunks</c> 16-token forwards
/// (last row scored), the split-KV decode accumulator variants on <c>--decode</c> teacher-forced decode steps, and the speed of a
/// <c>--speed-tile</c>-token prefill tile at depth per flash variant. With <c>--stats</c> the prefill itself runs the
/// score-statistics flash variant (same arithmetic as the default) and the max |score| / Q-lo histograms are printed per segment.</item>
/// <item><c>gemm-kl</c>: PQ2_0 blocked-coopmat GEMM accumulators (F32, F16, F16 promoted every 256) on 3 chat prompts plus a
/// wikitext window, every row scored with all-row logits in 256-row forwards (the LM head is part of the GEMM under test).</item>
/// <item><c>gemm-speed</c>: whole-prompt prefill tok/s of the same variants at <c>--lengths</c>, interleaved, median.</item>
/// <item><c>mtp-kv</c>: the MTP head's F32 KV rounded to F16 values (<see cref="VulkanPrecisionExperiment.KvRoundF16"/>): KL of the
/// draft logits against the F32 head, teacher-forced.</item>
/// <item><c>dense</c>: a dense GGUF (Llama Q8_0 / Q4_0) loaded once per <c>--arms</c> entry (an arm is a set of environment switches
/// read at load: <c>mmq</c>/<c>nommq</c>, <c>mmvq</c>/<c>nommvq</c>, <c>f16</c>/<c>f16p256</c> GEMM accumulators, <c>kvround</c>), scored
/// on a wikitext window (all rows) and teacher-forced decode rows against the first arm; prefill and decode speed per arm.</item>
/// <item><c>caps</c>: device features (subgroup sizes, cooperative-matrix shapes, integer dot, float16).</item>
/// </list>
/// </remarks>
internal static class VulkanPrecisionProfile
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

    /// <summary>Entry point: <c>precision-vulkan --exp ... --model ... --corpus ...</c>.</summary>
    /// <param name="args">Arguments after the subcommand.</param>
    /// <returns>Process exit code.</returns>
    public static int Run(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        string exp = Opt(args, "--exp") ?? "depth";
        Console.WriteLine($"# precision-vulkan {exp}  ({DateTime.Now:yyyy-MM-dd HH:mm:ss})");
        Console.WriteLine($"args: {string.Join(' ', args)}");
        var device = VulkanModelLoader.SharedDevice;
        Console.WriteLine($"device: {device.DeviceName} vendor=0x{device.VendorId:X4} subgroup={device.SubgroupSize} "
                          + $"[{device.MinSubgroupSize}..{device.MaxSubgroupSize}] coopmat={device.HasCooperativeMatrix} "
                          + $"intdot={device.HasIntegerDotProduct} f16={device.NumericFeatures.ShaderFloat16}");
        if (exp == "caps")
        {
            foreach (var s in device.SupportedCooperativeMatrixProperties)
                Console.WriteLine($"  coopmat {s}");
            return 0;
        }

        if (exp == "kernel") return KernelBench(args);
        string? modelPath = Opt(args, "--model") ?? Environment.GetEnvironmentVariable("DOTLLM_BONSAI_PQ2_0_GGUF");
        string corpus = Opt(args, "--corpus") ?? ".docs/corpora/wikitext2-test.raw";
        if (modelPath is null || !File.Exists(modelPath) || !File.Exists(corpus))
        {
            Console.Error.WriteLine("Usage: precision-vulkan --exp depth|gemm-kl|gemm-speed|mtp-kv|dense|caps --model <gguf> --corpus <wikitext raw> ...");
            return 2;
        }

        if (exp == "dense") return Dense(modelPath, corpus, args);

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var sw = Stopwatch.StartNew();
        var (loaded, _) = VulkanModelLoader.CreateFromGguf(device, gguf, config, VulkanModelLoader.ResolveSpvDir());
        using var model = loaded as VulkanQwen3HybridDenseTransformerModel
            ?? throw new InvalidOperationException($"Expected the qwen35 hybrid dense Vulkan model, got {loaded.GetType().Name}.");
        Console.WriteLine($"loaded in {sw.Elapsed.TotalSeconds:F1}s: hidden={config.HiddenSize} ffn={config.IntermediateSize} "
                          + $"layers={config.NumLayers} (attention {model.AttentionLayerCount}) heads={config.NumAttentionHeads}/{config.NumKvHeads} "
                          + $"hd={config.HeadDim} vocab={config.VocabSize} kv={model.KvStorage} ({model.KvBytesPerPosition / 1024.0:F1} KiB/token)");
        Console.WriteLine($"switches: flash={VulkanPrecisionExperiment.Flash} decode={VulkanPrecisionExperiment.Decode} "
                          + $"gemm={VulkanPrecisionExperiment.Gemm} kvround={VulkanPrecisionExperiment.KvRoundF16}");

        int[] corpusTokens = Tokenize(corpus, tokenizer, gguf, Int(args, "--corpus-tokens", 140_000));
        Console.WriteLine($"corpus tokens: {corpusTokens.Length}");

        return exp switch
        {
            "depth" => Depth(model, corpusTokens, args),
            "gemm-kl" => GemmKl(model, tokenizer, corpusTokens, args),
            "gemm-speed" => GemmSpeed(model, corpusTokens, args),
            "mtp-kv" => MtpKv(model, corpusTokens, args),
            _ => 2,
        };
    }

    // ───────────────────────────── experiments 1 + 2: attention accumulators at depth ─────────────────────────────

    private static int Depth(VulkanQwen3HybridDenseTransformerModel model, int[] corpus, string[] args)
    {
        int[] contexts = Ints(args, "--ctx", [8192, 32768]);
        int chunks = Int(args, "--chunks", 64);
        int decodeSteps = Int(args, "--decode", 48);
        int speedTile = Int(args, "--speed-tile", 1024);
        int reps = Int(args, "--reps", 3);
        int prefillChunk = Int(args, "--prefill-chunk", 2048);
        bool stats = args.Contains("--stats");
        var flashModes = (Opt(args, "--fa-variants") ?? "f32,noqlo,qk16,pv16t,pv16").Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(VulkanPrecisionExperiment.ParseFlash).Distinct().ToList();
        if (flashModes[0] != FlashAccVariant.Default) flashModes.Insert(0, FlashAccVariant.Default);
        var decodeModes = (Opt(args, "--decode-variants") ?? "f32,st16,pv16,st16pv16").Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(VulkanPrecisionExperiment.ParseDecode).Distinct().ToList();
        if (decodeModes.Count > 0 && decodeModes[0] != DecodeAccVariant.Default) decodeModes.Insert(0, DecodeAccVariant.Default);

        int maxCtx = contexts.Max();
        int capacity = maxCtx + Math.Max(chunks * DepthChunk, Math.Max(decodeSteps, speedTile)) + 64;
        if (corpus.Length < capacity + 2) throw new InvalidOperationException($"corpus has {corpus.Length} tokens, need {capacity + 2}");

        using var statsBuf = stats ? VulkanModelLoader.SharedDevice.AllocateHostReadback(VulkanPrecisionExperiment.StatsWords * sizeof(uint)) : null;
        VulkanPrecisionExperiment.StatsBuffer = statsBuf;
        ResetFlags();
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(capacity);
        Console.WriteLine($"depth: contexts={string.Join(',', contexts)} chunks={chunks}x{DepthChunk} decode={decodeSteps} tile={speedTile} reps={reps} "
                          + $"kv={kv.Storage} capacity={capacity}");
        int prefilled = 0;
        foreach (int ctx in contexts)
        {
            // ── grow the shared prefix to ctx (default kernels; the stats variant has the default's arithmetic) ──
            if (stats) ZeroStats(statsBuf!);
            VulkanPrecisionExperiment.Flash = stats ? FlashAccVariant.Stats : FlashAccVariant.Default;
            var psw = Stopwatch.StartNew();
            for (int s = prefilled; s < ctx; s += prefillChunk)
            {
                int len = Math.Min(prefillChunk, ctx - s);
                model.Forward(corpus.AsSpan(s, len), Positions(s, len), -1, kv).Dispose();
            }
            psw.Stop();
            VulkanPrecisionExperiment.Flash = FlashAccVariant.Default;
            Console.WriteLine($"ctx {ctx}: prefill [{prefilled}, {ctx}) {(ctx - prefilled) / psw.Elapsed.TotalSeconds:F1} tok/s ({psw.Elapsed.TotalSeconds:F1}s)"
                              + (stats ? " [stats variant]" : ""));
            if (stats) PrintStats(statsBuf!, $"prefill [{prefilled}, {ctx})");
            prefilled = ctx;
            using var ckpt = model.CheckpointRecurrentState() as IDisposable;
            object? state = ckpt;

            // ── prefill-attention accumulators: 16-token forwards at depth, last row scored ──
            var table = new StringBuilder(KlStats.Header($"ctx {ctx} / flash variant (prefill-chunk rows)") + "\n");
            List<float[]>? refRows = null;
            foreach (var mode in flashModes.Append(FlashAccVariant.Default))
            {
                bool repeat = refRows is not null && mode == FlashAccVariant.Default;
                VulkanPrecisionExperiment.Flash = mode;
                model.RestoreRecurrentState(state);
                kv.Rollback(ctx);
                long before = model.GqaFlashDispatchesForTest;
                var rows = new List<float[]>();
                var st = new KlStats();
                int r = 0;
                var csw = Stopwatch.StartNew();
                for (int c = 0; c < chunks; c++)
                {
                    int s = ctx + c * DepthChunk;
                    using var t = model.Forward(corpus.AsSpan(s, DepthChunk), Positions(s, DepthChunk), -1, kv);
                    // Every row of the chunk went through the flash prefill kernel; all of them are scored (all-row logits <= 16 rows).
                    int nRows = (int)(t.ElementCount / model.Config.VocabSize);
                    for (int i = 0; i < nRows; i++)
                    {
                        var l = Row(t, i, model.Config.VocabSize);
                        int target = corpus[s + DepthChunk - nRows + i + 1];
                        if (refRows is null) rows.Add(LogSoftmax(l));
                        else st.Add(refRows[r++], l, target);
                    }
                }
                Console.WriteLine($"ctx {ctx} {mode}{(repeat ? " (repeat)" : "")}: {chunks} chunks in {csw.Elapsed.TotalSeconds:F1}s, "
                                  + $"{model.GqaFlashDispatchesForTest - before} GQA flash dispatches");
                if (refRows is null) { refRows = rows; continue; }
                table.AppendLine(st.Row($"{ctx} / {(repeat ? "F32 repeat (determinism)" : FlashName(mode))}"));
            }
            VulkanPrecisionExperiment.Flash = FlashAccVariant.Default;
            Console.WriteLine(table.ToString());

            // ── decode accumulators: teacher-forced single-token steps at depth ──
            if (decodeModes.Count > 0 && decodeSteps > 0)
            {
                var dtable = new StringBuilder(KlStats.Header($"ctx {ctx} / decode variant") + "\n");
                var dtimes = new Dictionary<DecodeAccVariant, List<double>>();
                List<float[]>? refDec = null;
                foreach (var mode in decodeModes.Append(DecodeAccVariant.Default))
                {
                    bool repeat = refDec is not null && mode == DecodeAccVariant.Default;
                    VulkanPrecisionExperiment.Decode = mode;
                    model.RestoreRecurrentState(state);
                    kv.Rollback(ctx);
                    var rows = new List<float[]>();
                    var st = new KlStats();
                    for (int i = 0; i < decodeSteps; i++)
                    {
                        int p = ctx + i;
                        using var t = model.Forward(corpus.AsSpan(p, 1), [p], -1, kv);
                        var logits = LastRow(t, model.Config.VocabSize);
                        if (refDec is null) rows.Add(LogSoftmax(logits));
                        else st.Add(refDec[i], logits, corpus[p + 1]);
                    }
                    if (refDec is null) { refDec = rows; continue; }
                    dtable.AppendLine(st.Row($"{ctx} / {(repeat ? "F32 repeat (determinism)" : DecodeName(mode))}"));
                }
                // Timing: 12 steps per arm, arms interleaved and order-reversed per rep (first rep is a warm-up).
                for (int rep = 0; rep < reps + 1; rep++)
                {
                    foreach (var mode in rep % 2 == 0 ? decodeModes : Enumerable.Reverse(decodeModes))
                    {
                        VulkanPrecisionExperiment.Decode = mode;
                        model.RestoreRecurrentState(state);
                        kv.Rollback(ctx);
                        for (int i = 0; i < 12; i++)
                        {
                            int p = ctx + i;
                            double sec = Time(() => model.Forward(corpus.AsSpan(p, 1), [p], -1, kv).Dispose());
                            if (rep > 0 && i >= 2) (dtimes.TryGetValue(mode, out var l) ? l : dtimes[mode] = []).Add(sec);
                        }
                    }
                }
                VulkanPrecisionExperiment.Decode = DecodeAccVariant.Default;
                Console.WriteLine(dtable.ToString());
                Console.WriteLine($"| ctx {ctx} decode variant | ms/token (median of {dtimes.Values.First().Count}) | Δ ms/token vs F32 | Δ ms per attention layer |");
                Console.WriteLine("|---|---:|---:|---:|");
                double dref = Median(dtimes[DecodeAccVariant.Default]) * 1e3;
                foreach (var (mode, ts) in dtimes)
                {
                    double ms = Median(ts) * 1e3;
                    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"| {DecodeName(mode)} | {ms:F2} | {ms - dref:+0.00;-0.00} | {(ms - dref) / model.AttentionLayerCount:+0.000;-0.000} |"));
                }
                Console.WriteLine();
            }

            // ── prefill speed: one tile at depth per flash variant, interleaved ──
            if (speedTile > 0 && reps > 0)
            {
                var speed = new Dictionary<FlashAccVariant, List<double>>();
                for (int rep = 0; rep < reps + 1; rep++)
                {
                    foreach (var mode in rep % 2 == 0 ? flashModes : Enumerable.Reverse(flashModes))
                    {
                        VulkanPrecisionExperiment.Flash = mode;
                        model.RestoreRecurrentState(state);
                        kv.Rollback(ctx);
                        double sec = Time(() => model.Forward(corpus.AsSpan(ctx, speedTile), Positions(ctx, speedTile), -1, kv).Dispose());
                        if (rep > 0) (speed.TryGetValue(mode, out var l) ? l : speed[mode] = []).Add(sec);   // rep 0 = warm-up
                    }
                }
                VulkanPrecisionExperiment.Flash = FlashAccVariant.Default;
                Console.WriteLine($"| ctx {ctx} flash variant | {speedTile}-token tile at depth, tok/s (median of {reps}) | x vs F32 | min-max |");
                Console.WriteLine("|---|---:|---:|---|");
                double refMed = Median(speed[FlashAccVariant.Default].Select(t => speedTile / t).ToList());
                foreach (var (mode, ts) in speed)
                {
                    var tps = ts.Select(t => speedTile / t).ToList();
                    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"| {FlashName(mode)} | {Median(tps):F1} | {Median(tps) / refMed:F3} | {tps.Min():F1}-{tps.Max():F1} |"));
                }
                Console.WriteLine();
            }

            // Back to the shared prefix for the next segment.
            model.RestoreRecurrentState(state);
            kv.Rollback(ctx);
        }
        VulkanPrecisionExperiment.StatsBuffer = null;
        return 0;
    }

    private static string FlashName(FlashAccVariant m) => m switch
    {
        FlashAccVariant.Default => "F32 S, F32 O, Q hi/lo (default)",
        FlashAccVariant.NoQlo => "Q lo off (Q rounded to F16)",
        FlashAccVariant.Qk16 => "qk16: F16 S (Q pre-scaled, hi only)",
        FlashAccVariant.Pv16Tile => "pv16t: F16 P.V within a tile, F32 across",
        FlashAccVariant.Pv16 => "pv16: F16 O over the whole range",
        FlashAccVariant.Stats => "stats (default arithmetic)",
        _ => m.ToString(),
    };

    private static string DecodeName(DecodeAccVariant m) => m switch
    {
        DecodeAccVariant.Default => "F32 max/sum, F32 P.V (default)",
        DecodeAccVariant.Stats16 => "st16: F16 running max/sum",
        DecodeAccVariant.Pv16 => "pv16: F16 P.V accumulators",
        DecodeAccVariant.Stats16Pv16 => "st16pv16: both",
        _ => m.ToString(),
    };

    private static void ResetFlags()
    {
        VulkanPrecisionExperiment.Flash = FlashAccVariant.Default;
        VulkanPrecisionExperiment.Decode = DecodeAccVariant.Default;
        VulkanPrecisionExperiment.Gemm = GemmAccVariant.Default;
        VulkanPrecisionExperiment.KvRoundF16 = false;
    }

    private static unsafe void ZeroStats(VulkanDevice.Buffer buf)
    {
        var zeros = new uint[VulkanPrecisionExperiment.StatsWords];
        VulkanModelLoader.SharedDevice.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes((ReadOnlySpan<uint>)zeros), buf);
    }

    private static void PrintStats(VulkanDevice.Buffer buf, string label)
    {
        var f = new float[VulkanPrecisionExperiment.StatsWords];
        VulkanModelLoader.SharedDevice.Download(buf, f);
        var w = f.Select(BitConverter.SingleToUInt32Bits).ToArray();
        float maxAbs = BitConverter.UInt32BitsToSingle(w[0]);
        float maxLo = BitConverter.UInt32BitsToSingle(w[1]);
        long rows = w[322];
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"[score stats {label}] rows={rows} max|s|={maxAbs:F3} max|Q_lo.K|*scale={maxLo:E3}"));
        var sb = new StringBuilder("  per-row max|s| histogram (bin=[b,b+1)): ");
        long cum = 0;
        for (int b = 0; b < 256; b++)
        {
            if (w[2 + b] == 0) continue;
            cum += w[2 + b];
            sb.Append(CultureInfo.InvariantCulture, $"{b}:{w[2 + b]} ");
        }
        Console.WriteLine(sb.ToString());
        // Quantiles of the per-row max |s|, read off the integer histogram (upper bin edge).
        var q = new StringBuilder("  per-row max|s| quantiles (upper bin edge):");
        foreach (double p in new[] { 0.5, 0.9, 0.99, 0.999, 1.0 })
        {
            long target = (long)Math.Ceiling(p * rows);
            long acc = 0;
            for (int b = 0; b < 256; b++)
            {
                acc += w[2 + b];
                if (acc >= target && target > 0) { q.Append(CultureInfo.InvariantCulture, $" p{p * 100:G4}<{b + 1}"); break; }
            }
        }
        Console.WriteLine(q.ToString());
        var lo = new StringBuilder("  per-row max|Q_lo.K|*scale histogram (log2 bins): ");
        for (int b = 0; b < 64; b++)
        {
            if (w[258 + b] == 0) continue;
            lo.Append(CultureInfo.InvariantCulture, $"2^{b - 40}:{w[258 + b]} ");
        }
        Console.WriteLine(lo.ToString());
    }

    // ───────────────────────────── experiment 4: projection GEMM accumulators ─────────────────────────────

    private static List<GemmAccVariant> GemmVariants(string[] args)
    {
        var list = (Opt(args, "--variants") ?? "f32,f16,f16p256").Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(VulkanPrecisionExperiment.ParseGemm).Distinct().ToList();
        if (list[0] != GemmAccVariant.Default) list.Insert(0, GemmAccVariant.Default);
        return list;
    }

    private static string GemmName(GemmAccVariant v) => v switch
    {
        GemmAccVariant.Default => "F32-acc (ref, default)",
        GemmAccVariant.F16 => "F16-acc",
        GemmAccVariant.F16Promote256 => "F16-acc, F32 every 256",
        _ => v.ToString(),
    };

    private static int GemmKl(VulkanQwen3HybridDenseTransformerModel model, DotLLM.Tokenizers.ITokenizer tokenizer, int[] corpus, string[] args)
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
        if (!model.TrySetAllRowLogitsLimit(256)) throw new InvalidOperationException("all-row logits limit");

        var variants = GemmVariants(args);
        ResetFlags();
        var refRows = new List<List<float[]>>();
        foreach (var inp in inputs)
        {
            var rows = new List<float[]>();
            ScoreAllRows(model, inp.Tokens, inp.Prefix, inp.Scored, (_, l) => rows.Add(LogSoftmax(l)));
            refRows.Add(rows);
        }

        foreach (var v in variants.Append(GemmAccVariant.Default).Skip(1))
        {
            bool repeat = v == GemmAccVariant.Default;
            VulkanPrecisionExperiment.Gemm = v;
            var table = new StringBuilder(KlStats.Header("variant / input") + "\n");
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
                table.AppendLine(s.Row($"{(repeat ? "F32 repeat" : GemmName(v))} / {inp.Name}"));
            }
            table.AppendLine(total.Row($"**{(repeat ? "F32 repeat" : GemmName(v))} / all**"));
            Console.WriteLine(table.ToString());
        }
        ResetFlags();
        return 0;
    }

    private static int GemmSpeed(VulkanQwen3HybridDenseTransformerModel model, int[] corpus, string[] args)
    {
        int reps = Int(args, "--reps", 3);
        int[] lengths = Ints(args, "--lengths", [512, 2048, 8192]);
        int prefillChunk = Int(args, "--prefill-chunk", 2048);
        var variants = GemmVariants(args);
        var times = new Dictionary<(GemmAccVariant, int), List<double>>();
        foreach (var v in variants)
        {
            VulkanPrecisionExperiment.Gemm = v;
            TimedPrefill(model, corpus, lengths.Min(), prefillChunk);   // warm-up: pipelines, scratch growth
        }
        for (int rep = 0; rep < reps; rep++)
        {
            foreach (int len in lengths)
            {
                foreach (var v in rep % 2 == 0 ? variants : Enumerable.Reverse(variants))
                {
                    VulkanPrecisionExperiment.Gemm = v;
                    double s = TimedPrefill(model, corpus, len, prefillChunk);
                    (times.TryGetValue((v, len), out var l) ? l : times[(v, len)] = []).Add(s);
                    Console.WriteLine($"rep {rep + 1} len {len} {GemmName(v)}: {len / s:F1} tok/s");
                }
            }
        }
        ResetFlags();
        Console.WriteLine();
        Console.WriteLine("| variant | " + string.Join(" | ", lengths.Select(l => $"{l} tok/s (median, min-max)")) + " | "
                          + string.Join(" | ", lengths.Select(l => $"x vs ref @{l}")) + " |");
        Console.WriteLine("|---|" + string.Concat(Enumerable.Repeat("---:|", lengths.Length * 2)));
        foreach (var v in variants)
        {
            var sb = new StringBuilder($"| {GemmName(v)} |");
            foreach (int len in lengths)
            {
                var ts = times[(v, len)].Select(s => len / s).ToList();
                sb.Append(CultureInfo.InvariantCulture, $" {Median(ts):F1} ({ts.Min():F1}-{ts.Max():F1}) |");
            }
            foreach (int len in lengths)
            {
                double refMed = Median(times[(GemmAccVariant.Default, len)].Select(s => len / s).ToList());
                sb.Append(CultureInfo.InvariantCulture, $" {Median(times[(v, len)].Select(s => len / s).ToList()) / refMed:F3} |");
            }
            Console.WriteLine(sb.ToString());
        }
        return 0;
    }

    // ───────────────────────────── experiment 3: F32 KV stores rounded to F16 ─────────────────────────────

    private static int MtpKv(VulkanQwen3HybridDenseTransformerModel model, int[] corpus, string[] args)
    {
        if (!model.SupportsMtp) throw new InvalidOperationException("model has no MTP head");
        int start = Int(args, "--window-start", 20_000);
        int prefix = Int(args, "--prefix", 2048);
        int steps = Int(args, "--steps", 128);
        int capacity = prefix + steps + 16;
        List<float[]>? refRows = null;
        var table = new StringBuilder(KlStats.Header("MTP head KV / draft rows") + "\n");
        int[] tok = corpus[start..(start + capacity)];
        foreach (bool round in new[] { false, true, false })
        {
            ResetFlags();
            VulkanPrecisionExperiment.KvRoundF16 = round;
            VulkanKvRoundF16Kernel.InvalidateAll();
            long d0 = VulkanKvRoundF16Kernel.TotalDispatches;
            model.ResetSequenceState();
            using var kv = model.CreateKvCache(capacity);
            using var mtp = model.CreateMtpState(capacity) ?? throw new InvalidOperationException("CreateMtpState returned null");
            model.Forward(tok.AsSpan(0, prefix), Positions(0, prefix), -1, kv, (DotLLM.Core.Lora.ILoraAdapter?)null, mtp).Dispose();
            var rows = new List<float[]>();
            var st = new KlStats();
            int topTruth = 0;
            for (int i = 0; i < steps; i++)
            {
                int p = prefix + i;
                // Draft for p + 1 from the trunk hidden at p - 1 and token p (the head's own KV covers [0, p]).
                using (var d = model.ForwardMtp(mtp, tok[p], p))
                {
                    var l = LastRow(d, model.Config.VocabSize);
                    if (ArgMax(l) == tok[p + 1]) topTruth++;
                    if (refRows is null) rows.Add(LogSoftmax(l));
                    else st.Add(refRows[i], l, tok[p + 1]);
                }
                model.Forward(tok.AsSpan(p, 1), [p], -1, kv, (DotLLM.Core.Lora.ILoraAdapter?)null, mtp).Dispose();
            }
            Console.WriteLine($"mtp kv {(round ? "F16-rounded" : "F32")}: draft top-1 == true next token {topTruth}/{steps}; "
                              + $"round dispatches {VulkanKvRoundF16Kernel.TotalDispatches - d0}");
            if (refRows is null) { refRows = rows; continue; }
            table.AppendLine(st.Row(round ? "F16-rounded MTP KV vs F32" : "F32 repeat (determinism)"));
        }
        ResetFlags();
        Console.WriteLine(table.ToString());
        Console.WriteLine($"MTP head KV bytes per position: F32 {2L * model.Config.NumKvHeads * model.Config.HeadDim * 4} B, "
                          + $"F16 {2L * model.Config.NumKvHeads * model.Config.HeadDim * 2} B");
        return 0;
    }

    // ───────────────────────────── dense models (Q8_0 / Q4_0): arms loaded per environment ─────────────────────────────

    private static readonly Dictionary<string, (string Env, string? Value)[]> ArmEnv = new()
    {
        ["default"] = [],
        ["nommq"] = [("DOTLLM_VULKAN_DISABLE_MMQ", "1")],
        ["nommvq"] = [("DOTLLM_VULKAN_DISABLE_MMVQ", "1")],
        ["f16"] = [("DOTLLM_VULKAN_DISABLE_MMQ", "1"), (VulkanPrecisionExperiment.GemmEnvVar, "f16")],
        ["f16p256"] = [("DOTLLM_VULKAN_DISABLE_MMQ", "1"), (VulkanPrecisionExperiment.GemmEnvVar, "f16p256")],
        ["kvround"] = [(VulkanPrecisionExperiment.KvRoundEnvVar, "1")],
    };

    private static readonly string[] ArmEnvNames =
        ["DOTLLM_VULKAN_DISABLE_MMQ", "DOTLLM_VULKAN_DISABLE_MMVQ", VulkanPrecisionExperiment.GemmEnvVar, VulkanPrecisionExperiment.KvRoundEnvVar];

    private static int Dense(string modelPath, string corpus, string[] args)
    {
        string[] arms = (Opt(args, "--arms") ?? "default,nommq,f16,f16p256").Split(',', StringSplitOptions.RemoveEmptyEntries);
        int windowStart = Int(args, "--window-start", 20_000);
        int prefix = Int(args, "--window-prefix", 1024);
        int scored = Int(args, "--window-scored", 1024);
        int decodeSteps = Int(args, "--decode", 64);
        int[] lengths = Ints(args, "--lengths", [512, 2048]);
        int reps = Int(args, "--reps", 3);
        var device = VulkanModelLoader.SharedDevice;
        string spv = VulkanModelLoader.ResolveSpvDir();

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] tokens = Tokenize(corpus, tokenizer, gguf, windowStart + prefix + scored + decodeSteps + 16 + lengths.Max() + 2000);
        int[] window = tokens[windowStart..(windowStart + prefix + scored + decodeSteps + 2)];
        Console.WriteLine($"dense: {Path.GetFileName(modelPath)} {config.Architecture} hidden={config.HiddenSize} layers={config.NumLayers} "
                          + $"heads={config.NumAttentionHeads}/{config.NumKvHeads} hd={config.HeadDim} vocab={config.VocabSize}");

        var models = new List<(string Arm, IModel Model, Func<int, IKvCache> Kv)>();
        try
        {
            foreach (string arm in arms)
            {
                foreach (string e in ArmEnvNames) Environment.SetEnvironmentVariable(e, null);
                foreach (var (e, v) in ArmEnv[arm]) Environment.SetEnvironmentVariable(e, v);
                VulkanPrecisionExperiment.Gemm = VulkanPrecisionExperiment.ParseGemm(Environment.GetEnvironmentVariable(VulkanPrecisionExperiment.GemmEnvVar));
                var (m, f) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spv);
                VulkanPrecisionExperiment.Gemm = GemmAccVariant.Default;
                models.Add((arm, m, f));
                Console.WriteLine($"arm {arm}: loaded {m.GetType().Name}");
            }
            foreach (string e in ArmEnvNames) Environment.SetEnvironmentVariable(e, null);

            // KL: prefix, then the scored rows in 256-row forwards (all rows), then teacher-forced decode rows.
            List<float[]>? refPre = null, refDec = null;
            var table = new StringBuilder(KlStats.Header("arm / rows") + "\n");
            foreach (var (arm, m, f) in models)
            {
                VulkanPrecisionExperiment.KvRoundF16 = ArmEnv[arm].Any(x => x.Env == VulkanPrecisionExperiment.KvRoundEnvVar);
                VulkanKvRoundF16Kernel.InvalidateAll();
                using var kv = f(prefix + scored + decodeSteps + 16);
                m.Forward(window.AsSpan(0, prefix), Positions(0, prefix), -1, kv).Dispose();
                var pre = new List<float[]>();
                var ps = new KlStats();
                int r = 0;
                for (int s = prefix; s < prefix + scored; s += 256)
                {
                    int len = Math.Min(256, prefix + scored - s);
                    using var t = m.Forward(window.AsSpan(s, len), Positions(s, len), -1, kv);
                    int rowsOut = (int)(t.ElementCount / config.VocabSize);
                    if (rowsOut != len) throw new InvalidOperationException($"{arm}: expected {len} logit rows, got {rowsOut}");
                    for (int i = 0; i < len; i++)
                    {
                        var l = Row(t, i, config.VocabSize);
                        if (refPre is null) pre.Add(LogSoftmax(l));
                        else ps.Add(refPre[r++], l, window[s + i + 1]);
                    }
                }
                var dec = new List<float[]>();
                var ds = new KlStats();
                for (int i = 0; i < decodeSteps; i++)
                {
                    int p = prefix + scored + i;
                    using var t = m.Forward(window.AsSpan(p, 1), [p], -1, kv);
                    var l = LastRow(t, config.VocabSize);
                    if (refDec is null) dec.Add(LogSoftmax(l));
                    else ds.Add(refDec[i], l, window[p + 1]);
                }
                if (refPre is null) { refPre = pre; refDec = dec; continue; }
                table.AppendLine(ps.Row($"{arm} / prefill rows (256-row forwards)"));
                table.AppendLine(ds.Row($"{arm} / decode rows"));
            }
            VulkanPrecisionExperiment.KvRoundF16 = false;
            Console.WriteLine($"reference arm: {models[0].Arm}");
            Console.WriteLine(table.ToString());

            // Speed: prefill tok/s at each length and decode tok/s, arms interleaved, order reversed every rep.
            var pre2 = new Dictionary<(string, int), List<double>>();
            var dec2 = new Dictionary<string, List<double>>();
            for (int rep = 0; rep < reps + 1; rep++)
            {
                foreach (var (arm, m, f) in rep % 2 == 0 ? models : Enumerable.Reverse(models))
                {
                    VulkanPrecisionExperiment.KvRoundF16 = ArmEnv[arm].Any(x => x.Env == VulkanPrecisionExperiment.KvRoundEnvVar);
                    foreach (int len in lengths)
                    {
                        using var kv = f(len + 80);
                        double s = Time(() => m.Forward(tokens.AsSpan(1000, len), Positions(0, len), -1, kv).Dispose());
                        var dt = new List<double>();
                        for (int i = 0; i < 34; i++)
                        {
                            int p = len + i;
                            double ds1 = Time(() => m.Forward(tokens.AsSpan(1000 + p, 1), [p], -1, kv).Dispose());
                            if (i >= 2) dt.Add(ds1);
                        }
                        if (rep == 0) continue;   // warm-up
                        (pre2.TryGetValue((arm, len), out var l) ? l : pre2[(arm, len)] = []).Add(len / s);
                        if (len == lengths.Min()) (dec2.TryGetValue(arm, out var d) ? d : dec2[arm] = []).Add(1.0 / Median(dt));
                    }
                }
            }
            VulkanPrecisionExperiment.KvRoundF16 = false;
            Console.WriteLine("| arm | " + string.Join(" | ", lengths.Select(l => $"prefill {l} tok/s")) + $" | decode tok/s @{lengths.Min()} | "
                              + string.Join(" | ", lengths.Select(l => $"x @{l}")) + " | x decode |");
            Console.WriteLine("|---|" + string.Concat(Enumerable.Repeat("---:|", lengths.Length * 2 + 2)));
            string a0 = models[0].Arm;
            foreach (var (arm, _, _) in models)
            {
                var sb = new StringBuilder($"| {arm} |");
                foreach (int len in lengths) sb.Append(CultureInfo.InvariantCulture, $" {Median(pre2[(arm, len)]):F1} |");
                sb.Append(CultureInfo.InvariantCulture, $" {Median(dec2[arm]):F2} |");
                foreach (int len in lengths) sb.Append(CultureInfo.InvariantCulture, $" {Median(pre2[(arm, len)]) / Median(pre2[(a0, len)]):F3} |");
                sb.Append(CultureInfo.InvariantCulture, $" {Median(dec2[arm]) / Median(dec2[a0]):F3} |");
                Console.WriteLine(sb.ToString());
            }
        }
        finally
        {
            foreach (var (_, m, _) in models) m.Dispose();
        }
        return 0;
    }

    // ───────────────────────────── kernel level: attention variants on synthetic Bonsai-shaped data ─────────────────────────────

    private static int KernelBench(string[] args)
    {
        const int numHeads = 24, numKvHeads = 4, headDim = 256;
        int[] depths = Ints(args, "--depths", [8192, 32768, 131072]);
        int rows = Int(args, "--rows", 1024);
        int rounds = Int(args, "--rounds", 5);
        int layers = Int(args, "--layers", 16);
        double targetSeconds = Int(args, "--target-ms", 800) / 1000.0;
        var device = VulkanModelLoader.SharedDevice;
        string spv = VulkanModelLoader.ResolveSpvDir();
        ResetFlags();
        var flashModes = (Opt(args, "--fa-variants") ?? "f32,noqlo,qk16,pv16t,pv16").Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(VulkanPrecisionExperiment.ParseFlash).Distinct().ToList();
        var decodeModes = (Opt(args, "--decode-variants") ?? "f32,st16,pv16,st16pv16").Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(VulkanPrecisionExperiment.ParseDecode).Distinct().ToList();
        using var gqa = VulkanFlashAttentionGqaKernel.TryCreate(device, spv, VulkanKvStorage.F16)
            ?? throw new InvalidOperationException("no GQA flash kernel");
        if (!gqa.IsCoopmat) flashModes.RemoveAll(m => m is FlashAccVariant.NoQlo or FlashAccVariant.Stats);
        using var split = VulkanSplitKvAttentionKernel.Create(device, spv, VulkanKvStorage.F16);
        split.EnableGqaByDefault();
        Console.WriteLine($"kernel: flash {(gqa.IsCoopmat ? "coopmat" : "scalar")} GQA, decode split-KV GQA={split.UsesGqa(numHeads, numKvHeads, headDim)}; "
                          + $"shape {numHeads}/{numKvHeads}/{headDim}, F16 KV, rows={rows}, rounds={rounds}");

        int maxKv = depths.Max() + rows;
        long kvElems = (long)maxKv * numKvHeads * headDim;
        var rng = new Random(1109);
        float[] q = new float[rows * numHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        ushort[] block = new ushort[(1 << 20) + 7];
        for (int i = 0; i < block.Length; i++) block[i] = BitConverter.HalfToUInt16Bits((Half)(float)((rng.NextDouble() * 2 - 1) * 3));
        using var bufQ = device.Allocate((long)q.Length * sizeof(float));
        using var bufK = device.Allocate(kvElems * sizeof(ushort));
        using var bufV = device.Allocate(kvElems * sizeof(ushort));
        using var bufOut = device.Allocate((long)q.Length * sizeof(float));
        device.Upload(q.AsSpan(), bufQ);
        FillTiled(device, bufK, kvElems, block);
        FillTiled(device, bufV, kvElems, block);

        // Flash prefill: `rows` query positions at depth d (seqKv = d + rows), min over interleaved rounds.
        Console.WriteLine("| depth | flash variant | rows | min ms | attn TFLOP/s | x vs F32 | out relRms vs F32 | out maxAbs vs F32 |");
        Console.WriteLine("|---:|---|---:|---:|---:|---:|---:|---:|");
        double spg = 0;
        foreach (int d in depths)
        {
            int r = rows;
            while (spg > 0 && r > 16 && spg * VulkanPrefillSubmitBudget.AttentionGflops(r, d + r, numHeads, headDim) > targetSeconds) r /= 2;
            var best = flashModes.ToDictionary(m => m, _ => double.MaxValue);
            var outs = new Dictionary<FlashAccVariant, float[]>();
            foreach (var m in flashModes)
            {
                VulkanPrecisionExperiment.Flash = m;
                gqa.Launch(bufQ, bufK, bufV, bufOut, r, d + r, numHeads, numKvHeads, headDim, d);
                var o = new float[r * numHeads * headDim];
                device.Download(bufOut, o);
                outs[m] = o;
            }
            for (int round = 0; round < rounds; round++)
            {
                foreach (var m in round % 2 == 0 ? flashModes : Enumerable.Reverse(flashModes))
                {
                    VulkanPrecisionExperiment.Flash = m;
                    var sw = Stopwatch.StartNew();
                    gqa.Launch(bufQ, bufK, bufV, bufOut, r, d + r, numHeads, numKvHeads, headDim, d);
                    best[m] = Math.Min(best[m], sw.Elapsed.TotalSeconds);
                }
            }
            VulkanPrecisionExperiment.Flash = FlashAccVariant.Default;
            double gflop = VulkanPrefillSubmitBudget.AttentionGflops(r, d + r, numHeads, headDim);
            spg = best[FlashAccVariant.Default] / gflop;
            foreach (var m in flashModes)
            {
                var (rel, maxAbs) = Diff(outs[FlashAccVariant.Default], outs[m]);
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"| {d} | {FlashName(m)} | {r} | {best[m] * 1e3:F1} | {gflop / best[m] / 1000:F2} | {best[FlashAccVariant.Default] / best[m]:F3} | {rel:E2} | {maxAbs:E2} |"));
            }
        }

        // Decode: `layers` split-KV dispatches per submission (one per attention layer), ms per layer.
        Console.WriteLine();
        Console.WriteLine($"| depth | decode variant | ms per layer (min of {rounds}, {layers} layers per submit) | x vs F32 | out relRms vs F32 | out maxAbs vs F32 |");
        Console.WriteLine("|---:|---|---:|---:|---:|---:|");
        foreach (int d in depths)
        {
            var best = decodeModes.ToDictionary(m => m, _ => double.MaxValue);
            var outs = new Dictionary<DecodeAccVariant, float[]>();
            foreach (var m in decodeModes)
            {
                VulkanPrecisionExperiment.Decode = m;
                split.Launch(bufQ, bufK, bufV, bufOut, 1, d, numHeads, numKvHeads, headDim, d - 1);
                var o = new float[numHeads * headDim];
                device.Download(bufOut, o);
                outs[m] = o;
            }
            for (int round = 0; round < rounds; round++)
            {
                foreach (var m in round % 2 == 0 ? decodeModes : Enumerable.Reverse(decodeModes))
                {
                    VulkanPrecisionExperiment.Decode = m;
                    using var ctx = device.CreateSubmitContext();
                    ctx.Begin();
                    for (int l = 0; l < layers; l++)
                        split.Record(ctx.CommandBuffer, bufQ, bufK, bufV, bufOut, 1, d, numHeads, numKvHeads, headDim, d - 1);
                    var sw = Stopwatch.StartNew();
                    ctx.SubmitAndWait();
                    best[m] = Math.Min(best[m], sw.Elapsed.TotalSeconds / layers);
                }
            }
            VulkanPrecisionExperiment.Decode = DecodeAccVariant.Default;
            foreach (var m in decodeModes)
            {
                var (rel, maxAbs) = Diff(outs[DecodeAccVariant.Default], outs[m]);
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"| {d} | {DecodeName(m)} | {best[m] * 1e3:F3} | {best[DecodeAccVariant.Default] / best[m]:F3} | {rel:E2} | {maxAbs:E2} |"));
            }
        }
        return 0;
    }

    private static (double RelRms, double MaxAbs) Diff(float[] a, float[] b)
    {
        double num = 0, den = 0, max = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double e = b[i] - a[i];
            num += e * e;
            den += (double)a[i] * a[i];
            max = Math.Max(max, Math.Abs(e));
        }
        return (Math.Sqrt(num / Math.Max(den, 1e-30)), max);
    }

    private static void FillTiled(VulkanDevice device, VulkanDevice.Buffer buf, long elems, ushort[] block)
    {
        int chunkElems = (int)Math.Min(32 << 20, elems);
        ushort[] chunk = new ushort[chunkElems];
        for (int i = 0; i < chunk.Length; i++) chunk[i] = block[i % block.Length];
        using var staging = device.Allocate((long)chunkElems * sizeof(ushort));
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(chunk.AsSpan()), staging);
        for (long off = 0; off < elems; off += chunkElems)
        {
            long n = Math.Min(chunkElems, elems - off);
            device.CopyBufferRangeSynchronous(staging, buf, srcOffset: 0, dstOffset: (ulong)(off * sizeof(ushort)), size: (ulong)(n * sizeof(ushort)));
        }
    }

    // ───────────────────────────── scoring ─────────────────────────────

    private static void ScoreAllRows(VulkanQwen3HybridDenseTransformerModel model, int[] tokens, int prefix, int scored,
        Action<int, ReadOnlySpan<float>> onRow)
    {
        const int chunk = 256;
        int vocab = model.Config.VocabSize;
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(prefix + scored + 16);
        for (int s = 0; s < prefix; s += 2048)
        {
            int len = Math.Min(2048, prefix - s);
            model.Forward(tokens.AsSpan(s, len), Positions(s, len), -1, kv).Dispose();
        }
        for (int s = prefix; s < prefix + scored; s += chunk)
        {
            int len = Math.Min(chunk, prefix + scored - s);
            using var t = model.Forward(tokens.AsSpan(s, len), Positions(s, len), -1, kv);
            if (t.ElementCount != (long)len * vocab)
                throw new InvalidOperationException($"expected {len} logit rows, got {t.ElementCount / vocab}");
            for (int r = 0; r < len; r++)
                onRow(s + r, Row(t, r, vocab));
        }
    }

    private static unsafe ReadOnlySpan<float> Row(ITensor t, int r, int vocab)
        => new((void*)(t.DataPointer + (nint)((long)r * vocab * sizeof(float))), vocab);

    private static ReadOnlySpan<float> LastRow(ITensor t, int vocab) => Row(t, (int)(t.ElementCount / vocab) - 1, vocab);

    private static int ArgMax(ReadOnlySpan<float> x)
    {
        int a = 0;
        for (int i = 1; i < x.Length; i++) if (x[i] > x[a]) a = i;
        return a;
    }

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
    /// KL(ref ‖ variant) per row over the full vocabulary, top-1 agreement and both NLLs of the true next token, as in
    /// <c>precision-cuda</c>: median / p90 / p99 / max plus outlier rates.
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
                + $"{pplRef:F4} | {pplVar:F4} | {(pplVar / pplRef - 1) * 100:+0.000;-0.000;0.000}% |");
        }
    }

    // ───────────────────────────── helpers ─────────────────────────────

    private static double TimedPrefill(VulkanQwen3HybridDenseTransformerModel model, int[] corpus, int len, int prefillChunk)
    {
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(len + 16);
        return Time(() =>
        {
            for (int s = 0; s < len; s += prefillChunk)
            {
                int n = Math.Min(prefillChunk, len - s);
                model.Forward(corpus.AsSpan(1000 + s, n), Positions(s, n), -1, kv).Dispose();
            }
        });
    }

    private static double Time(Action a)
    {
        var sw = Stopwatch.StartNew();
        a();   // the logits readback waits for the submission
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
