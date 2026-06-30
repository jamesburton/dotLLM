using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.HuggingFace;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

namespace DotLLM.Benchmarks.Profile;

/// <summary>
/// Real-GGUF demonstration of Vulkan dual-device pipeline-parallel spanning (#366): loads a model split
/// across two Vulkan devices via <see cref="VulkanPipelineTransformerModel"/> and (when it fits on one
/// device) checks its last-token logits against a single-device full model — proving spanning is correct
/// on REAL weights (real quant types, real architecture), not just the synthetic parity fixture.
///
/// Usage: <c>span-vulkan [--gguf PATH] [--split K] [--dev0 I] [--dev1 J] [--tokens a,b,c] [--no-parity] [--gen N]</c>
/// Defaults: downloads TinyLlama-1.1B Q8_0; split at L/2; dev0=0, dev1=1 (or 0 if only one device).
/// </summary>
internal static class VulkanPipelineSpanProfile
{
    private const string DefaultRepoId = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
    private const string DefaultFilename = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf";

    public static int Run(string[] args)
    {
        if (!VulkanDevice.IsAvailable())
        {
            Console.Error.WriteLine("Vulkan device not available — install loader/driver and retry.");
            return 2;
        }

        string? ggufPath = null;
        string repoId = DefaultRepoId;
        string filename = DefaultFilename;
        int? splitArg = null;
        int dev0 = 0, dev1 = -1;
        int genCount = 0;
        int batchCount = 0;
        bool parity = true;
        int[] tokenIds = [1, 15043, 29892, 590, 1024, 338]; // BOS + "Hello, my name is" (Llama BPE ids)

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--gguf" when i + 1 < args.Length: ggufPath = args[++i]; break;
                case "--repo" when i + 1 < args.Length: repoId = args[++i]; break;
                case "--file" when i + 1 < args.Length: filename = args[++i]; break;
                case "--split" when i + 1 < args.Length: splitArg = int.Parse(args[++i]); break;
                case "--dev0" when i + 1 < args.Length: dev0 = int.Parse(args[++i]); break;
                case "--dev1" when i + 1 < args.Length: dev1 = int.Parse(args[++i]); break;
                case "--gen" when i + 1 < args.Length: genCount = int.Parse(args[++i]); break;
                case "--batch" when i + 1 < args.Length: batchCount = int.Parse(args[++i]); break;
                case "--no-parity": parity = false; break;
                case "--tokens" when i + 1 < args.Length:
                    tokenIds = Array.ConvertAll(args[++i].Split(','), s => int.Parse(s.Trim())); break;
            }
        }

        int deviceCount = VulkanDevice.PhysicalDeviceCount();
        Console.WriteLine($"Vulkan physical devices: {deviceCount}");
        if (dev1 < 0) dev1 = deviceCount >= 2 ? 1 : 0;
        if (dev0 >= deviceCount || dev1 >= deviceCount)
        {
            Console.Error.WriteLine($"Requested device indices ({dev0},{dev1}) exceed device count {deviceCount}.");
            return 2;
        }
        if (dev0 == dev1)
            Console.WriteLine("NOTE: both stages on the same physical device (no cross-device transfer; pipeline path still exercised).");

        if (ggufPath is null)
        {
            string cacheDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "dotllm-bench");
            Console.WriteLine($"No --gguf given; downloading {repoId}/{filename} via HuggingFaceDownloader ...");
            using var dl = new HuggingFaceDownloader();
            ggufPath = dl.DownloadFileAsync(repoId, filename, cacheDir).GetAwaiter().GetResult();
        }

        Console.WriteLine($"GGUF: {ggufPath} ({new FileInfo(ggufPath).Length / 1024.0 / 1024.0:F1} MiB)");

        DotLLM.Core.Models.ModelConfig config;
        using (var cfgGguf = GgufFile.Open(ggufPath))
            config = GgufModelConfigExtractor.Extract(cfgGguf.Metadata);
        int numLayers = config.NumLayers;
        int split = splitArg ?? numLayers / 2;
        if (split <= 0 || split >= numLayers)
        {
            Console.Error.WriteLine($"--split must be 1..{numLayers - 1} (model has {numLayers} layers).");
            return 2;
        }
        Console.WriteLine($"Model: {config.Architecture}, {numLayers} layers, hidden={config.HiddenSize}, vocab={config.VocabSize}");
        Console.WriteLine($"Split at layer {split}: stage0 = layers [0..{split}) on device {dev0}, stage1 = layers [{split}..{numLayers}) on device {dev1}");

        string spvDir = Path.Combine(AppContext.BaseDirectory, "spv");
        int[] positions = new int[tokenIds.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        // ── Spanned model ──
        var spanGguf = GgufFile.Open(ggufPath); // owned + disposed by the pipeline model
        using var span = VulkanPipelineTransformerModel.LoadFromGguf(spanGguf, config, split, dev0, dev1, spvDir);
        Console.WriteLine($"Spanned model loaded. ComputeMemoryBytes={span.ComputeMemoryBytes / 1024.0 / 1024.0:F1} MiB");

        var sw = Stopwatch.StartNew();
        float[] spanLogits;
        using (ITensor sl = span.Forward(tokenIds, positions, deviceId: 0, kvCache: null))
            spanLogits = LastRow(sl, config.VocabSize);
        sw.Stop();
        int spanTop1 = ArgMax(spanLogits);
        Console.WriteLine($"Spanned prefill ({tokenIds.Length} tok): {sw.Elapsed.TotalMilliseconds:F1} ms; top-1 token={spanTop1} (logit={spanLogits[spanTop1]:F4})");

        // ── Single-device parity (only if it fits on one device) ──
        if (parity)
        {
            using var fullGguf = GgufFile.Open(ggufPath);
            using var full = VulkanTransformerModel.LoadFromGguf(fullGguf, config, spvDir);
            float[] fullLogits;
            using (ITensor fl = full.Forward(tokenIds, positions, deviceId: 0, kvCache: null))
                fullLogits = LastRow(fl, config.VocabSize);
            int fullTop1 = ArgMax(fullLogits);

            float maxAbs = 0f;
            for (int i = 0; i < fullLogits.Length; i++)
                maxAbs = MathF.Max(maxAbs, MathF.Abs(fullLogits[i] - spanLogits[i]));
            Console.WriteLine();
            Console.WriteLine("─── Parity vs single-device full model ───");
            Console.WriteLine($"  full top-1={fullTop1} (logit={fullLogits[fullTop1]:F4})");
            Console.WriteLine($"  span top-1={spanTop1}");
            Console.WriteLine($"  top-1 match: {(fullTop1 == spanTop1 ? "YES" : "NO")}");
            Console.WriteLine($"  max|diff| over {fullLogits.Length} logits: {maxAbs:E4}");
            Console.WriteLine(fullTop1 == spanTop1
                ? "  RESULT: spanning matches single-device on real weights."
                : "  RESULT: TOP-1 MISMATCH — investigate.");
        }

        // ── Optional greedy generation (token IDs only) ──
        if (genCount > 0)
        {
            Console.WriteLine();
            Console.WriteLine($"─── Greedy generation ({genCount} tokens, token IDs) ───");
            using var kv = span.CreateKvCache(config.MaxSequenceLength);
            var gen = new List<int>(tokenIds);
            // Prefill.
            using (var _ = span.Forward(tokenIds, positions, 0, kv)) { }
            int next = spanTop1;
            var genSw = Stopwatch.StartNew();
            for (int t = 0; t < genCount; t++)
            {
                gen.Add(next);
                int pos = gen.Count - 1;
                using ITensor lg = span.Forward(new[] { next }, new[] { pos }, 0, kv);
                next = ArgMax(LastRow(lg, config.VocabSize));
            }
            genSw.Stop();
            Console.WriteLine($"  generated token IDs: {string.Join(",", gen.GetRange(tokenIds.Length, genCount))}");
            Console.WriteLine($"  decode: {genCount / genSw.Elapsed.TotalSeconds:F1} tok/s");
        }

        // ── Micro-batch overlap: serial loop vs pipelined ForwardBatch ──
        if (batchCount > 1)
        {
            Console.WriteLine();
            Console.WriteLine($"─── Overlap timing: {batchCount} independent prefills (serial loop vs pipelined ForwardBatch) ───");

            // Serial baseline: N independent Forward calls, each its own KV (no overlap).
            var serialKv = new IKvCache[batchCount];
            for (int i = 0; i < batchCount; i++) serialKv[i] = span.CreateKvCache(config.MaxSequenceLength);
            var serialSw = Stopwatch.StartNew();
            for (int i = 0; i < batchCount; i++)
                using (var _ = span.Forward(tokenIds, positions, 0, serialKv[i])) { }
            serialSw.Stop();
            foreach (var kv in serialKv) kv.Dispose();

            // Pipelined: one ForwardBatch over N requests (stage0 on dev0 overlaps stage1 on dev1).
            var batchKv = new IKvCache[batchCount];
            var requests = new List<SequenceForwardRequest>(batchCount);
            for (int i = 0; i < batchCount; i++)
            {
                batchKv[i] = span.CreateKvCache(config.MaxSequenceLength);
                requests.Add(new SequenceForwardRequest { TokenIds = tokenIds, Positions = positions, KvCache = batchKv[i] });
            }
            var batchSw = Stopwatch.StartNew();
            var outputs = span.ForwardBatch(requests, 0);
            batchSw.Stop();
            foreach (var o in outputs) (o as IDisposable)?.Dispose();
            foreach (var kv in batchKv) kv.Dispose();

            double serialMs = serialSw.Elapsed.TotalMilliseconds;
            double batchMs = batchSw.Elapsed.TotalMilliseconds;
            Console.WriteLine($"  serial loop   : {serialMs,8:F1} ms  ({batchCount * 1000.0 / serialMs:F1} seq/s)");
            Console.WriteLine($"  pipelined     : {batchMs,8:F1} ms  ({batchCount * 1000.0 / batchMs:F1} seq/s)");
            Console.WriteLine($"  overlap speed-up: {serialMs / Math.Max(batchMs, 0.001):F2}× ({(dev0 == dev1 ? "same device — expect ~1×" : "cross-device — overlap window")})");
        }

        return 0;
    }

    private static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        int rows = logits.Shape[0];
        int offset = (rows - 1) * vocab;
        return new ReadOnlySpan<float>((void*)(logits.DataPointer + (nint)offset * sizeof(float)), vocab).ToArray();
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }
}
