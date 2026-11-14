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
    /// Max token count fed to <see cref="IModel.Forward(System.ReadOnlySpan{int}, System.ReadOnlySpan{int}, int, DotLLM.Core.Attention.IKvCache?, bool)"/>
    /// per call while walking the untimed <c>--depth</c> synthetic context extension (issue
    /// #188) -- see the call site's remarks for the full rationale. Matches the
    /// <c>CapacityGranularity</c> the CUDA hybrid models' <c>EnsureCapacity</c> now rounds to
    /// (issue #188) so a chunk never itself crosses a granularity-rounding boundary mid-chunk,
    /// though the two constants are independent and don't need to stay in lockstep.
    /// </summary>
    private const int DepthExtensionChunkSize = 256;

    /// <summary>
    /// Runs <paramref name="reps"/> measured repetitions (plus <paramref name="warmupReps"/>
    /// discarded warm-up reps) of prefill + greedy decode against <paramref name="model"/>.
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
    /// <param name="warmupReps">Number of warm-up repetitions to discard (default 1).
    /// Use 0 to skip warm-up entirely.</param>
    public static BenchResult Run(
        IModel model,
        Func<int, IKvCache> kvCacheFactory,
        int[] promptTokens,
        int decodeTokens,
        int reps,
        int depth = 0,
        double loadMs = 0,
        int warmupReps = 1)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(kvCacheFactory);
        ArgumentNullException.ThrowIfNull(promptTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(promptTokens.Length, nameof(promptTokens));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(decodeTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(reps);
        ArgumentOutOfRangeException.ThrowIfNegative(depth);
        ArgumentOutOfRangeException.ThrowIfNegative(warmupReps);

        int cacheSize = promptTokens.Length + depth + decodeTokens + 8;
        if (cacheSize > model.Config.MaxSequenceLength)
            throw new ArgumentException(
                $"prompt ({promptTokens.Length}) + depth ({depth}) + decode ({decodeTokens}) exceeds the model's " +
                $"max sequence length ({model.Config.MaxSequenceLength}). Reduce -p / -n / --depth.");

        int totalReps = warmupReps + reps;
        var all = new List<BenchRep>(totalReps);
        // #1620: DOTLLM_BENCH_ALT_PLANAR=planar-first|prod-first alternates the planar PQ2_0 decode GEMV per rep inside ONE process
        // (same-session paired A/B, immune to process-start and page-cache noise); each rep reports its mode and planar-call delta on stderr.
        string? alt = Environment.GetEnvironmentVariable("DOTLLM_BENCH_ALT_PLANAR");
        bool altOn = alt is "planar-first" or "prod-first";
        // #1660: DOTLLM_BENCH_ALT_PREFILL_CHUNK=1024,2048,512,256,0 rotates the timed prefill's chunk size per rep inside ONE process
        // (0 = the whole prompt in one Forward call). Each rep reports its chunk, prefill tok/s, fence-wait count and longest fence wait
        // (a lower bound on the longest GPU submission), the prefill's last-row argmax and max |dlogit| against the first arm's first rep,
        // and the greedy continuation, on stderr. Measurement only: the timed region is the chunk loop, as a chunked serving prefill is.
        int[]? chunkArms = ParseChunkArms(Environment.GetEnvironmentVariable("DOTLLM_BENCH_ALT_PREFILL_CHUNK"));
        float[]? refLogits = null;
        for (int rep = 0; rep < totalReps; rep++)
        {
            if (altOn) DotLLM.Cpu.Kernels.MatMul.PQ2_0PlanarEnabled = ((rep & 1) == 0) == (alt == "planar-first");
            long calls0 = DotLLM.Cpu.Kernels.MatMul.PQ2_0PlanarCalls;
            if (chunkArms is not null)
            {
                int chunk = chunkArms[rep % chunkArms.Length];
                all.Add(RunOneRep(model, kvCacheFactory, promptTokens, decodeTokens, depth, cacheSize, chunk, rep, ref refLogits));
                continue;
            }
            all.Add(RunOneRep(model, kvCacheFactory, promptTokens, decodeTokens, depth, cacheSize));
            if (altOn)
                Console.Error.WriteLine($"ALT rep={rep} mode={(DotLLM.Cpu.Kernels.MatMul.PQ2_0PlanarEnabled ? "planar" : "prod")} " +
                    $"decode_tok_s={all[^1].DecodeTokS:F2} prefill_tok_s={all[^1].PrefillTokS:F2} planar_calls={DotLLM.Cpu.Kernels.MatMul.PQ2_0PlanarCalls - calls0}");
        }

        return new BenchResult
        {
            Warmup = warmupReps > 0 ? all[0] : null,
            Reps = warmupReps > 0 ? BenchStats.DiscardWarmup(all, warmupReps) : all,
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

        // Timed prefill (argmax excluded from the stopwatch window). Only the last row is ever
        // read below, so opt in to lastTokenLogitsOnly (issue #185) -- on the CUDA hybrid models
        // this avoids allocating/computing a full [promptLen, vocabSize] logits tensor for a
        // value nothing but the last row consumes; other models ignore the hint and behave as
        // before.
        int nextToken;
        var prefillSw = Stopwatch.StartNew();
        ITensor prefillLogits = model.Forward(promptTokens, positions, deviceId: -1, cache, lastTokenLogitsOnly: true);
        prefillSw.Stop();
        using (prefillLogits)
            nextToken = ArgmaxLastRow(prefillLogits);

        int nextPos = promptLen;

        // Untimed synthetic context extension: re-tile the prompt for `depth`
        // extra tokens so the timed decode runs at depth promptLen + depth. Same
        // lastTokenLogitsOnly reasoning as the prefill call above -- this is the call whose
        // full-seqLen logits tensor was the dominant VRAM cost behind issue #185's
        // `dotllm bench --depth` hang beyond ~650-768 on a 12GB card.
        //
        // Issue #188: fed in bounded DepthExtensionChunkSize-token chunks rather than one single
        // Forward call spanning the entire `depth`. Even after the CUDA hybrid models'
        // EnsureCapacity switched from power-of-two to granularity-bounded rounding (issue #188),
        // a single Forward call of length `depth` still forces those models' per-token scratch
        // buffers to grow to (at minimum) `depth` tokens -- that's a real requirement of a call
        // that size, not rounding waste, so no EnsureCapacity policy can shrink it. Splitting the
        // untimed extension into fixed-size chunks appended incrementally to the same kvCache
        // caps the peak single-call scratch requirement at DepthExtensionChunkSize tokens
        // regardless of how large --depth gets, which is what actually let `--depth 2048` (exact
        // power-of-two, so already zero rounding waste either way) complete cleanly instead of
        // hanging at ~0 MiB free. This mirrors real serving, which chunks large prefills rather
        // than ever submitting one multi-thousand-token Forward call; only intermediate chunks'
        // logits are discarded here (the re-tiled extension tokens are fixed ahead of time, not
        // sampled, so nothing downstream depends on them) -- the last chunk's argmax still seeds
        // the timed decode loop exactly as before.
        if (depth > 0)
        {
            int[] extra = new int[depth];
            int[] extraPos = new int[depth];
            for (int i = 0; i < depth; i++)
            {
                extra[i] = promptTokens[i % promptLen];
                extraPos[i] = nextPos + i;
            }

            int offset = 0;
            while (offset < depth)
            {
                int chunkLen = Math.Min(DepthExtensionChunkSize, depth - offset);
                using var logits = model.Forward(
                    extra.AsSpan(offset, chunkLen), extraPos.AsSpan(offset, chunkLen),
                    deviceId: -1, cache, lastTokenLogitsOnly: true);
                offset += chunkLen;
                if (offset == depth) nextToken = ArgmaxLastRow(logits);
            }
            nextPos += depth;
        }

        // Timed greedy decode: n single-token forwards. Only the Forward calls
        // are inside the stopwatch windows; argmax runs between them.
        // DOTLLM_BENCH_DUMP_TOKENS=<path> appends one line of generated token
        // IDs per rep — the exact-token parity gate for pipeline/submit
        // changes (issue #143): diff the dump between two builds/env configs.
        string? dumpPath = Environment.GetEnvironmentVariable("DOTLLM_BENCH_DUMP_TOKENS");
        List<int>? dumped = dumpPath is { Length: > 0 } ? new List<int>(decodeTokens) : null;
        // DOTLLM_BENCH_STEP_LOG=<path> (#801): append one line per rep holding that rep's per-step decode ms so a
        // run's tok/s can be windowed (drift tracking) without changing the timed region.
        string? stepLogPath = Environment.GetEnvironmentVariable("DOTLLM_BENCH_STEP_LOG");
        List<double>? stepMs = stepLogPath is { Length: > 0 } ? new List<double>(decodeTokens) : null;
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
            stepMs?.Add(sw.Elapsed.TotalMilliseconds);
            using (logits)
                nextToken = ArgmaxLastRow(logits);
            dumped?.Add(nextToken);
            nextPos++;
        }

        if (stepMs is not null)
        {
            try { File.AppendAllText(stepLogPath!, string.Join(' ', stepMs.Select(m => m.ToString("F2", System.Globalization.CultureInfo.InvariantCulture))) + Environment.NewLine); }
            catch { /* diagnostics only */ }
        }

        if (dumped is not null)
        {
            try { File.AppendAllText(dumpPath!, string.Join(' ', dumped) + Environment.NewLine); }
            catch { /* diagnostics only */ }
        }

        return new BenchRep(prefillSw.Elapsed.TotalMilliseconds, decodeMs, promptLen, decodeTokens);
    }

    private static int[]? ParseChunkArms(string? spec)
    {
        if (string.IsNullOrWhiteSpace(spec)) return null;
        var arms = new List<int>();
        foreach (var part in spec.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            if (int.TryParse(part, out int c) && c >= 0) arms.Add(c);
        return arms.Count > 0 ? arms.ToArray() : null;
    }

    /// <summary>#1660: one rep whose timed prefill feeds the prompt in <paramref name="chunk"/>-token Forward calls (0 = one call).</summary>
    private static unsafe BenchRep RunOneRep(
        IModel model, Func<int, IKvCache> kvCacheFactory, int[] promptTokens, int decodeTokens, int depth, int cacheSize,
        int chunk, int rep, ref float[]? refLogits)
    {
        if (depth != 0) throw new ArgumentException("DOTLLM_BENCH_ALT_PREFILL_CHUNK does not support --depth.");
        model.ResetSequenceState();   // untimed: every arm starts from a zero recurrent state, so the parity columns compare like with like
        using IKvCache cache = kvCacheFactory(cacheSize);
        int promptLen = promptTokens.Length;
        int[] positions = new int[promptLen];
        for (int i = 0; i < promptLen; i++) positions[i] = i;
        int step = chunk > 0 ? chunk : promptLen;

        DotLLM.Vulkan.VulkanDevice.SubmitContext.DiagFenceWaits = 0;
        DotLLM.Vulkan.VulkanDevice.SubmitContext.DiagFenceWaitMaxMs = 0;
        int calls = 0;
        ITensor? last = null;
        var prefillSw = Stopwatch.StartNew();
        for (int a = 0; a < promptLen; a += step)
        {
            int n = Math.Min(step, promptLen - a);
            last?.Dispose();
            last = model.Forward(promptTokens.AsSpan(a, n), positions.AsSpan(a, n), deviceId: -1, cache, lastTokenLogitsOnly: true);
            calls++;
        }
        prefillSw.Stop();
        long waits = DotLLM.Vulkan.VulkanDevice.SubmitContext.DiagFenceWaits;
        double maxWait = DotLLM.Vulkan.VulkanDevice.SubmitContext.DiagFenceWaitMaxMs;

        int nextToken;
        float maxAbs = 0, top1 = 0, top2 = float.NegativeInfinity;
        using (last!)
        {
            nextToken = ArgmaxLastRow(last!);
            int rank = last!.Shape.Rank;
            int vocab = last.Shape[rank - 1];
            long rows = 1;
            for (int d = 0; d < rank - 1; d++) rows *= last.Shape[d];
            var row = new ReadOnlySpan<float>((float*)last.DataPointer + (rows - 1) * vocab, vocab);
            top1 = row[nextToken];
            for (int i = 0; i < vocab; i++) if (i != nextToken && row[i] > top2) top2 = row[i];
            if (refLogits is null) refLogits = row.ToArray();
            else for (int i = 0; i < vocab; i++) maxAbs = Math.Max(maxAbs, Math.Abs(row[i] - refLogits[i]));
        }

        double decodeMs = 0;
        var gen = new List<int>(decodeTokens) { nextToken };
        int[] single = new int[1];
        int[] singlePos = new int[1];
        int nextPos = promptLen;
        for (int i = 0; i < decodeTokens; i++)
        {
            single[0] = nextToken;
            singlePos[0] = nextPos++;
            var sw = Stopwatch.StartNew();
            ITensor logits = model.Forward(single, singlePos, deviceId: -1, cache);
            sw.Stop();
            decodeMs += sw.Elapsed.TotalMilliseconds;
            using (logits) nextToken = ArgmaxLastRow(logits);
            gen.Add(nextToken);
        }

        double ms = prefillSw.Elapsed.TotalMilliseconds;
        Console.Error.WriteLine(System.FormattableString.Invariant(
            $"CHUNK rep={rep} chunk={chunk} prompt={promptLen} calls={calls} prefill_ms={ms:F1} prefill_tok_s={promptLen * 1000.0 / ms:F2} ") +
            System.FormattableString.Invariant(
            $"fence_waits={waits} max_fence_ms={maxWait:F1} argmax={gen[0]} top1_top2_gap={top1 - top2:F4} max_abs_dlogit_vs_ref={maxAbs:F4} ") +
            $"gen={string.Join(',', gen)}");
        return new BenchRep(ms, decodeMs, promptLen, decodeTokens);
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
