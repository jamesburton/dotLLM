using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Decode roofline probe: sweeps the decode thread count on Llama-3.2-1B and reports decode throughput,
/// to settle whether decode is memory-bandwidth-bound (no AVX-512 arithmetic kernel can help) or
/// compute-bound (a wider/bf16 decode GEMV could help). The shape of the knee is the answer — no need
/// to know the machine's exact achievable bandwidth:
/// <list type="bullet">
///   <item>tok/s plateaus by ~2–8 threads ⇒ <b>bandwidth/dispatch-bound</b> — the lever is bytes-moved
///   (lower-bit quant / KV quant), NOT ISA width; an AVX-512/VNNI/bf16 decode GEMV would be wasted effort.</item>
///   <item>tok/s keeps climbing toward 32 ⇒ <b>compute-bound</b> — the current decode-thread clamp is too
///   conservative on this host (a config change, not a kernel) and a wider GEMV might also help.</item>
/// </list>
/// </summary>
/// <remarks>
/// <para>
/// Context for the hypothesis: the decode path is already deliberately throttled. <see cref="ThreadingConfig"/>
/// documents "Decode is memory-bandwidth-bound, so more threads than memory channels don't help", and
/// <c>ComputeThreadPool</c> caps decode at <c>min(8, threadCount)</c> (or the memory-channel estimate) because
/// SpinWait dispatch was measured to collapse at 32 threads (10.6 ms vs 32 µs at 8T for the 30-dispatch decode
/// burst). This probe re-measures that knee end-to-end on a real model so the decode-optimization decision rests
/// on a current artifact rather than only the code comments.
/// </para>
/// <para>Opt-in via <c>DOTLLM_RUN_PREFILL_BENCH</c> (loads a ~1.1 GB model; never runs in CI). Decode threads
/// start at 2 — <c>ComputeThreadPool</c> clamps the decode count to <c>[2, threadCount]</c>.</para>
/// </remarks>
[Collection("Llama32Instruct")]
public class Llama32DecodeRooflineProbe
{
    private readonly Llama32InstructFixture _fixture;

    public Llama32DecodeRooflineProbe(Llama32InstructFixture fixture)
    {
        _fixture = fixture;
    }

    // Decode thread counts to sweep (the pool always has all cores; only the decode-phase active-worker
    // count varies). 1 is excluded — the pool clamps the decode count to a minimum of 2.
    private static readonly int[] DecodeThreadCounts = { 2, 4, 8, 16, 32 };

    private const int Context = 256;   // prefill this context into the KV-cache before decoding
    private const int Steps = 32;      // timed single-token decode steps
    private const int Warmup = 3;

    [SkippableFact]
    public void DecodeThreadScaling_RevealsBound()
    {
        Skip.If(string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DOTLLM_RUN_PREFILL_BENCH")),
            "Decode roofline probe is opt-in — set DOTLLM_RUN_PREFILL_BENCH=1 to run.");

        // Tokens are reused across configs; vocab is read once from a throwaway open.
        int vocab;
        using (var probe = GgufFile.Open(_fixture.FilePath))
            vocab = GgufModelConfigExtractor.Extract(probe.Metadata).VocabSize;

        var rng = new Random(11);
        int[] ctxTokens = new int[Context];
        int[] ctxPositions = new int[Context];
        for (int i = 0; i < Context; i++)
        {
            ctxTokens[i] = rng.Next(1, Math.Min(vocab, 32000));
            ctxPositions[i] = i;
        }
        int decodeToken = rng.Next(1, Math.Min(vocab, 32000));

        Console.WriteLine(
            $"[Llama32DecodeRoofline] model={Path.GetFileName(_fixture.FilePath)} cores={Environment.ProcessorCount} " +
            $"context={Context} steps={Steps} warmup={Warmup}");

        double baseTokPerSec = 0;
        foreach (int decodeThreads in DecodeThreadCounts)
        {
            // Full pool (all cores) for prefill; decode phase limited to `decodeThreads` active workers.
            var config = new ThreadingConfig(ThreadCount: 0, DecodeThreadCount: decodeThreads);

            using var gguf = GgufFile.Open(_fixture.FilePath);
            var modelConfig = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = TransformerModel.LoadFromGguf(gguf, modelConfig, config);
            using var kvCache = new SimpleKvCache(modelConfig.NumLayers, modelConfig.NumKvHeads,
                modelConfig.HeadDim, Context + Warmup + Steps + 8);

            int[] one = { decodeToken };
            int[] pos = new int[1];

            kvCache.Rollback(0);
            using (var _ = model.Forward(ctxTokens, ctxPositions, deviceId: -1, kvCache)) { }

            int basePos = Context;
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
            double tokPerSec = 1000.0 / median;
            if (decodeThreads == DecodeThreadCounts[0]) baseTokPerSec = tokPerSec;
            double scaling = baseTokPerSec > 0 ? tokPerSec / baseTokPerSec : 1.0;

            Console.WriteLine(
                $"  decodeThreads={decodeThreads,2}  median={median,7:F2} ms/tok  {tokPerSec,7:F1} tok/s  " +
                $"scaling-vs-{DecodeThreadCounts[0]}T={scaling,5:F2}x");
        }

        Console.WriteLine(
            "  Interpretation: plateau by ~4-8T ⇒ bandwidth/dispatch-bound (ISA-width kernels won't help decode); " +
            "monotonic climb toward 32T ⇒ compute-bound (raise the decode-thread cap; a wider GEMV may help).");
    }
}
