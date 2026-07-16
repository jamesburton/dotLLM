using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Cli.Benchmarking;

/// <summary>
/// llama-bench-equivalent measurement loop: for each repetition, allocates a
/// fresh KV-cache, times one prefill forward over the synthetic prompt, then
/// times <c>n</c> greedy decode steps. The first repetition is a warm-up and
/// is discarded from the reported statistics. Model load time is the caller's
/// concern (excluded here, reported separately by the command).
/// </summary>
public static class BenchRunner
{
    /// <summary>
    /// Runs <paramref name="reps"/> measured repetitions (plus one discarded warm-up)
    /// of prefill + greedy decode against <paramref name="model"/>.
    /// </summary>
    /// <param name="model">Loaded model. Recurrent-state architectures (Nemotron-H SSM,
    /// Qwen3MoeHybrid GDN) reuse the model-owned recurrent state across repetitions —
    /// timings are unaffected but greedy continuations may differ between reps.</param>
    /// <param name="kvCacheFactory">Factory producing a fresh KV-cache of the given
    /// capacity for each repetition; the runner disposes each cache after its rep.</param>
    /// <param name="promptTokens">Synthetic prompt token ids (see <see cref="BenchStats.TilePrompt"/>).</param>
    /// <param name="decodeTokens">Number of greedy decode steps to time per repetition.</param>
    /// <param name="reps">Measured repetition count (warm-up excluded).</param>
    /// <param name="depth">Extra synthetic context tokens fed (untimed) between prefill
    /// and decode, so decode runs at context depth <c>promptTokens.Length + depth</c>.
    /// The extra tokens re-tile the prompt.</param>
    /// <param name="loadMs">Model load wall time to embed in the result.</param>
    public static BenchResult Run(
        IModel model,
        Func<int, IKvCache> kvCacheFactory,
        int[] promptTokens,
        int decodeTokens,
        int reps,
        int depth = 0,
        double loadMs = 0)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(kvCacheFactory);
        ArgumentNullException.ThrowIfNull(promptTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(promptTokens.Length, nameof(promptTokens));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(decodeTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(reps);
        ArgumentOutOfRangeException.ThrowIfNegative(depth);

        int cacheSize = promptTokens.Length + depth + decodeTokens + 8;
        if (cacheSize > model.Config.MaxSequenceLength)
            throw new ArgumentException(
                $"prompt ({promptTokens.Length}) + depth ({depth}) + decode ({decodeTokens}) exceeds the model's " +
                $"max sequence length ({model.Config.MaxSequenceLength}). Reduce -p / -n / --depth.");

        var all = new List<BenchRep>(reps + 1);
        for (int rep = 0; rep < reps + 1; rep++)
            all.Add(RunOneRep(model, kvCacheFactory, promptTokens, decodeTokens, depth, cacheSize));

        return new BenchResult
        {
            Warmup = all[0],
            Reps = BenchStats.DiscardWarmup(all),
            LoadMs = loadMs,
            PromptTokens = promptTokens.Length,
            DecodeTokens = decodeTokens,
            Depth = depth,
        };
    }

    private static BenchRep RunOneRep(
        IModel model, Func<int, IKvCache> kvCacheFactory,
        int[] promptTokens, int decodeTokens, int depth, int cacheSize)
    {
        using IKvCache cache = kvCacheFactory(cacheSize);

        int promptLen = promptTokens.Length;
        int[] positions = new int[promptLen];
        for (int i = 0; i < promptLen; i++) positions[i] = i;

        // Timed prefill (argmax excluded from the stopwatch window).
        int nextToken;
        var prefillSw = Stopwatch.StartNew();
        ITensor prefillLogits = model.Forward(promptTokens, positions, deviceId: -1, cache);
        prefillSw.Stop();
        using (prefillLogits)
            nextToken = ArgmaxLastRow(prefillLogits);

        int nextPos = promptLen;

        // Untimed synthetic context extension: re-tile the prompt for `depth`
        // extra tokens so the timed decode runs at depth promptLen + depth.
        if (depth > 0)
        {
            int[] extra = new int[depth];
            int[] extraPos = new int[depth];
            for (int i = 0; i < depth; i++)
            {
                extra[i] = promptTokens[i % promptLen];
                extraPos[i] = nextPos + i;
            }
            using (var logits = model.Forward(extra, extraPos, deviceId: -1, cache))
                nextToken = ArgmaxLastRow(logits);
            nextPos += depth;
        }

        // Timed greedy decode: n single-token forwards. Only the Forward calls
        // are inside the stopwatch windows; argmax runs between them.
        // DOTLLM_BENCH_DUMP_TOKENS=<path> appends one line of generated token
        // IDs per rep — the exact-token parity gate for pipeline/submit
        // changes (issue #143): diff the dump between two builds/env configs.
        string? dumpPath = Environment.GetEnvironmentVariable("DOTLLM_BENCH_DUMP_TOKENS");
        List<int>? dumped = dumpPath is { Length: > 0 } ? new List<int>(decodeTokens) : null;
        double decodeMs = 0;
        int[] single = new int[1];
        int[] singlePos = new int[1];
        for (int i = 0; i < decodeTokens; i++)
        {
            single[0] = nextToken;
            singlePos[0] = nextPos;
            var sw = Stopwatch.StartNew();
            ITensor logits = model.Forward(single, singlePos, deviceId: -1, cache);
            sw.Stop();
            decodeMs += sw.Elapsed.TotalMilliseconds;
            using (logits)
                nextToken = ArgmaxLastRow(logits);
            dumped?.Add(nextToken);
            nextPos++;
        }

        if (dumped is not null)
        {
            try { File.AppendAllText(dumpPath!, string.Join(' ', dumped) + Environment.NewLine); }
            catch { /* diagnostics only */ }
        }

        return new BenchRep(prefillSw.Elapsed.TotalMilliseconds, decodeMs, promptLen, decodeTokens);
    }

    /// <summary>
    /// Greedy (argmax) token over the LAST row of a <c>[rows, vocab]</c> logits tensor —
    /// CPU models return one row per input token, GPU models return the last row only;
    /// taking the final row handles both.
    /// </summary>
    private static unsafe int ArgmaxLastRow(ITensor logits)
    {
        int rank = logits.Shape.Rank;
        int vocab = logits.Shape[rank - 1];
        long rows = 1;
        for (int d = 0; d < rank - 1; d++) rows *= logits.Shape[d];

        float* basePtr = (float*)logits.DataPointer + (rows - 1) * vocab;
        int best = 0;
        float bestVal = basePtr[0];
        for (int i = 1; i < vocab; i++)
        {
            if (basePtr[i] > bestVal) { bestVal = basePtr[i]; best = i; }
        }
        return best;
    }
}
