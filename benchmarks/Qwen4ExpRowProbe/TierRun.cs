using System.Diagnostics;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

// Expert-tiering acceptance driver (#939).
//   tier <gguf> <out.txt> <n> [prompts=6] [warm=8]
// Loads the model once (DOTLLM_VK_EXPERT_SLOTS=N tiers the routed experts, unset = fully resident), then for each prompt prefills TOKEN BY TOKEN (the single-row
// path in BOTH arms, so resident and tiered run identical kernels on identical state; a batched resident prefill sits in a different activation cluster, #873/#909),
// greedily decodes n tokens and writes "<prompt>\t<ids...>" to <out.txt>. Two arms' files diff to prove identity. Prints decode tok/s (after `warm` tokens),
// prefill time and the tier counters per prompt.
internal static class TierRun
{
    private static readonly string[] Prompts =
    {
        "The capital of France is Paris, and the capital of Germany is",
        "Mixture-of-experts models route each token to a small subset of expert networks, which lets the total parameter count grow much faster than the compute per token. Explain in detail how the router is trained and why load balancing matters for throughput.",
        "def is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n\n# Write unit tests for the function above",
        "The French Revolution began in 1789 and transformed the political landscape of Europe. Key events included the storming of the Bastille, the Declaration of the Rights of Man, the Reign of Terror, and the rise of Napoleon Bonaparte, who eventually crowned himself emperor in 1804. Summarize the main causes.",
        "Write a short story about a lighthouse keeper who discovers that the light has been guiding something other than ships.",
        "Explain the difference between a mutex and a semaphore, then show a C# example of a producer-consumer queue using SemaphoreSlim.",
        "Translate to German and explain the grammar: The quick brown fox jumps over the lazy dog while the farmer watches from the barn.",
        "What are the main differences between TCP and UDP, and when would you choose each for a real-time multiplayer game?",
    };

    private static int ArgMax(ReadOnlySpan<float> r) { int b = 0; for (int i = 1; i < r.Length; i++) if (r[i] > r[b]) b = i; return b; }

    public static unsafe void Run(string[] args)
    {
        string gguf = args[1], outPath = args[2];
        int n = int.Parse(args[3]);
        int promptCount = args.Length > 4 ? Math.Min(int.Parse(args[4]), Prompts.Length) : 6;
        int warm = args.Length > 5 ? int.Parse(args[5]) : 8;
        var sw = Stopwatch.StartNew();
        using var file = GgufFile.Open(gguf);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var device = VulkanDevice.Create();
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");
        using var model = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, file, cfg, spv);
        double loadS = sw.Elapsed.TotalSeconds;
        var proc = Process.GetCurrentProcess();
        Console.WriteLine($"[tier] loaded in {loadS:F1} s; tiered={model.ExpertsTiered}; device live {device.LiveBytesTotal() / 1073741824.0:F2} GiB; process WS {proc.WorkingSet64 / 1073741824.0:F2} GiB private {proc.PrivateMemorySize64 / 1073741824.0:F2} GiB");
        var tok = GgufTokenizerFactory.Load(file.Metadata);
        var tier = model.ExpertTier;
        int vocab = cfg.VocabSize;
        var lines = new List<string>();
        double totalDecodeMs = 0; int totalDecodeTokens = 0;
        long hits = 0, misses = 0, bytes = 0;
        double gpuMs = 0, ssdMs = 0, resumeMs = 0;
        for (int pi = 0; pi < promptCount; pi++)
        {
            var ids = tok.Encode(Prompts[pi]);
            int cap = Math.Min(ids.Length + n + 16, model.DenseContextLimit);
            model.ResetSequenceState();
            tier?.Reset();
            using var kv = model.CreateKvCache(cap);
            var psw = Stopwatch.StartNew();
            float[] last = Array.Empty<float>();
            for (int p = 0; p < ids.Length; p++)
            {
                using var lg = model.Forward(ids.AsSpan(p, 1), [p], -1, kv, true);
                if (p == ids.Length - 1) last = new ReadOnlySpan<float>((void*)lg.DataPointer, vocab).ToArray();
            }
            double prefillS = psw.Elapsed.TotalSeconds;
            var gen = new List<int> { ArgMax(last) };
            tier?.ResetCounters();
            var dsw = Stopwatch.StartNew();
            double warmMs = 0;
            while (gen.Count < n)
            {
                if (gen.Count == warm + 1) warmMs = dsw.Elapsed.TotalMilliseconds;
                using var lg = model.Forward([gen[^1]], [ids.Length + gen.Count - 1], -1, kv, true);
                gen.Add(ArgMax(new ReadOnlySpan<float>((void*)lg.DataPointer, vocab)));
            }
            double decodeMs = dsw.Elapsed.TotalMilliseconds;
            int timed = Math.Max(1, gen.Count - 1 - warm);
            double steadyTps = timed * 1000.0 / Math.Max(1e-6, decodeMs - warmMs);
            totalDecodeMs += decodeMs - warmMs; totalDecodeTokens += timed;
            string tierTxt = "";
            if (tier is not null)
            {
                hits += tier.Hits; misses += tier.Misses; bytes += tier.BytesRead;
                double f = 1000.0 / Stopwatch.Frequency;
                gpuMs += tier.GpuWaitTicks * f; ssdMs += tier.ReadTicks * f; resumeMs += tier.ResumeTicks * f;
                tierTxt = " | " + tier.Describe();
            }
            Console.WriteLine($"[tier] prompt {pi}: {ids.Length} tok prefill {prefillS:F1} s ({ids.Length / prefillS:F1} tok/s); decode {gen.Count - 1} tok in {decodeMs / 1000:F1} s = {(gen.Count - 1) * 1000.0 / decodeMs:F2} tok/s overall, {steadyTps:F2} tok/s after {warm} warm{tierTxt}");
            lines.Add($"{pi}\t{string.Join(' ', gen)}");
        }
        File.WriteAllLines(outPath, lines);
        double tps = totalDecodeTokens * 1000.0 / Math.Max(1e-6, totalDecodeMs);
        Console.WriteLine($"[tier] SUMMARY tiered={model.ExpertsTiered} slots={tier?.Slots ?? 0} prompts={promptCount} n={n}: steady decode {tps:F2} tok/s ({totalDecodeMs / Math.Max(1, totalDecodeTokens):F1} ms/token)" +
                          (tier is null ? "" : $"; decode lookups {hits + misses}: {100.0 * hits / Math.Max(1, hits + misses):F1}% hit, {misses} misses, {bytes / 1048576.0:F0} MiB read; gpu-to-router {gpuMs:F0} ms, ssd {ssdMs:F0} ms, resume {resumeMs:F0} ms"));
        proc.Refresh();
        Console.WriteLine($"[tier] end: device live {device.LiveBytesTotal() / 1073741824.0:F2} GiB; process WS {proc.WorkingSet64 / 1073741824.0:F2} GiB private {proc.PrivateMemorySize64 / 1073741824.0:F2} GiB; tokens -> {outPath}");
    }
}
