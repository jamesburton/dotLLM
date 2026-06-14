using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Shared decode thread-scaling sweep, used by the per-model roofline probes
/// (<see cref="Llama32DecodeRooflineProbe"/>, <see cref="SmolLM2DecodeRooflineProbe"/>,
/// <see cref="BielikDecodeRooflineProbe"/>) to settle the decode-thread knee per model size.
/// </summary>
/// <remarks>
/// <para>
/// The threading investigation (<c>.docs/decode-threading-investigation.md</c>) concluded the decode
/// thread cap should be set by <b>per-dispatch work size</b> (driven by the model's hidden/intermediate
/// dimensions), not a single global constant: SmolLM-135M's tiny decode matmuls collapse SpinWait at 32T
/// while Llama-3.2-1B's larger matmuls keep scaling. This sweep is the §5 gating measurement — it runs the
/// <i>same</i> probe across a small/mid/large model so the crossover size (where 32T flips from loss to win)
/// becomes the threshold for any adaptive dispatch gate.
/// </para>
/// <para>
/// Two discriminators beyond the basic knee:
/// <list type="bullet">
///   <item><b>30T vs 32T</b> — if 30T is healthy and only 32T collapses, the trigger is OS oversubscription
///   (no spare core for OS/harness); if 30T also collapses it is pure cache-line contention.</item>
///   <item><b>Short vs long context</b> — longer context grows the attention dispatch relative to the matmul
///   dispatch and can move the knee; the sweep repeats at each requested context length.</item>
/// </list>
/// </para>
/// </remarks>
internal static class DecodeThreadScalingSweep
{
    // Fine grid incl. the 24/30/32 oversubscription-vs-contention discriminators.
    internal static readonly int[] DefaultDecodeThreadCounts = { 2, 4, 8, 16, 24, 30, 32 };

    // Short (attention dispatch small) and long (attention dispatch grows) context points.
    internal static readonly int[] DefaultContexts = { 128, 2048 };

    private const int Steps = 32;   // timed single-token decode steps
    private const int Warmup = 3;

    /// <summary>
    /// Sweeps decode thread count (explicit <see cref="ThreadingConfig.DecodeThreadCount"/>) across the
    /// given context lengths and reports decode throughput, so the knee can be read off per model size.
    /// </summary>
    public static void RunKneeSweep(
        string modelPath,
        int[]? decodeThreadCounts = null,
        int[]? contexts = null)
    {
        decodeThreadCounts ??= DefaultDecodeThreadCounts;
        contexts ??= DefaultContexts;

        int vocab;
        using (var probe = GgufFile.Open(modelPath))
            vocab = GgufModelConfigExtractor.Extract(probe.Metadata).VocabSize;

        Console.WriteLine(
            $"[DecodeKneeSweep] model={Path.GetFileName(modelPath)} cores={Environment.ProcessorCount} " +
            $"steps={Steps} warmup={Warmup}");

        foreach (int context in contexts)
        {
            (int[] ctxTokens, int[] ctxPositions, int decodeToken) = BuildContext(vocab, context);

            Console.WriteLine($"  context={context}");
            double baseTokPerSec = 0;
            foreach (int decodeThreads in decodeThreadCounts)
            {
                var config = new ThreadingConfig(ThreadCount: 0, DecodeThreadCount: decodeThreads);
                double tokPerSec = MeasureDecode(modelPath, config, ctxTokens, ctxPositions, decodeToken, context);

                if (decodeThreads == decodeThreadCounts[0]) baseTokPerSec = tokPerSec;
                double scaling = baseTokPerSec > 0 ? tokPerSec / baseTokPerSec : 1.0;
                Console.WriteLine(
                    $"    decodeThreads={decodeThreads,2}  {tokPerSec,7:F1} tok/s  " +
                    $"scaling-vs-{decodeThreadCounts[0]}T={scaling,5:F2}x");
            }
        }

        Console.WriteLine(
            "  Interpretation: plateau/collapse by ~8T ⇒ small-model regime (dispatch-bound; keep the cap); " +
            "monotonic climb toward 32T ⇒ large-model regime (per-dispatch work amortizes coordination). " +
            "30T healthy but 32T collapses ⇒ OS oversubscription; both collapse ⇒ cache-line contention.");
    }

    /// <summary>
    /// Reports decode throughput under the <i>production configuration paths</i> (not the explicit
    /// <see cref="ThreadingConfig.DecodeThreadCount"/> override the knee sweep uses): the default no-pinning
    /// path, and the NUMA / P-core pinning paths that build a <c>NumaTopology</c>. Validates end-to-end that
    /// enabling pinning no longer drops decode below the default — the cap-2 footgun fix.
    /// </summary>
    public static void RunProductionConfigPaths(string modelPath, int context = 256)
    {
        int vocab;
        using (var probe = GgufFile.Open(modelPath))
            vocab = GgufModelConfigExtractor.Extract(probe.Metadata).VocabSize;

        (int[] ctxTokens, int[] ctxPositions, int decodeToken) = BuildContext(vocab, context);

        Console.WriteLine(
            $"[DecodeProductionPaths] model={Path.GetFileName(modelPath)} cores={Environment.ProcessorCount} " +
            $"context={context} steps={Steps}");

        (string label, ThreadingConfig config)[] paths =
        {
            ("default (no pinning → cap 8)",
                new ThreadingConfig(ThreadCount: 0)),
            ("--numa-pin (topology → floored cap)",
                new ThreadingConfig(ThreadCount: 0, DecodeThreadCount: 0, EnableNumaPinning: true)),
            ("--pcore-only (topology → floored cap)",
                new ThreadingConfig(ThreadCount: 0, DecodeThreadCount: 0, EnablePCorePinning: true)),
        };

        double baseTokPerSec = 0;
        foreach ((string label, ThreadingConfig config) in paths)
        {
            double tokPerSec = MeasureDecode(modelPath, config, ctxTokens, ctxPositions, decodeToken, context);
            if (baseTokPerSec == 0) baseTokPerSec = tokPerSec;
            double ratio = baseTokPerSec > 0 ? tokPerSec / baseTokPerSec : 1.0;
            Console.WriteLine($"  {label,-38}  {tokPerSec,7:F1} tok/s  vs-default={ratio,5:F2}x");
        }

        Console.WriteLine(
            "  Expectation after the footgun fix: pinning paths are within noise of the default (>= ~0.9x), " +
            "NOT the pre-fix ~0.4x (cap-2) regression.");
    }

    private static double MeasureDecode(
        string modelPath, ThreadingConfig config,
        int[] ctxTokens, int[] ctxPositions, int decodeToken, int context)
    {
        using var gguf = GgufFile.Open(modelPath);
        var modelConfig = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, modelConfig, config);
        using var kvCache = new SimpleKvCache(modelConfig.NumLayers, modelConfig.NumKvHeads,
            modelConfig.HeadDim, context + Warmup + Steps + 8);

        int[] one = { decodeToken };
        int[] pos = new int[1];

        kvCache.Rollback(0);
        using (var _ = model.Forward(ctxTokens, ctxPositions, deviceId: -1, kvCache)) { }

        int basePos = context;
        for (int w = 0; w < Warmup; w++)
        {
            pos[0] = basePos + w;
            using var warm = model.Forward(one, pos, deviceId: -1, kvCache);
        }
        kvCache.Rollback(basePos);

        var times = new double[Steps];
        var sw = new Stopwatch();
        for (int s = 0; s < Steps; s++)
        {
            pos[0] = basePos + s;
            sw.Restart();
            using var logits = model.Forward(one, pos, deviceId: -1, kvCache);
            sw.Stop();
            times[s] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(times);
        double median = times[Steps / 2];
        return median > 0 ? 1000.0 / median : 0;
    }

    private static (int[] Tokens, int[] Positions, int DecodeToken) BuildContext(int vocab, int context)
    {
        var rng = new Random(11);
        int[] tokens = new int[context];
        int[] positions = new int[context];
        for (int i = 0; i < context; i++)
        {
            tokens[i] = rng.Next(1, Math.Min(vocab, 32000));
            positions[i] = i;
        }
        int decodeToken = rng.Next(1, Math.Min(vocab, 32000));
        return (tokens, positions, decodeToken);
    }
}
