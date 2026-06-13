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
/// End-to-end prefill throughput benchmark for the Q8_0 prefill operators on Llama-3.2-1B-Instruct,
/// comparing the three reduction paths on identical input in one process:
/// <list type="bullet">
///   <item>inner-product / cache-tiled (flag off — the production default)</item>
///   <item>integer R4 outer-product GEMM (<see cref="TransformerModel.UseOuterProductQ8Prefill"/>)</item>
///   <item>bf16 outer-product GEMM (<see cref="TransformerModel.UseBf16OuterProductQ8Prefill"/>, net11 + AVX512-BF16 only)</item>
/// </list>
/// This measures the <em>end-to-end</em> impact of the operators — the per-tile microkernel speedup
/// (~2.3× for bf16, benchmarked separately) is diluted at the model level because attention, RoPE,
/// RMSNorm and softmax are unchanged. The honest end-to-end number is the deliverable.
/// </summary>
/// <remarks>
/// <para>
/// <b>Off by default.</b> This is a slow benchmark (loads a ~1.1 GB model and runs dozens of full
/// forward passes), so it is skipped unless <c>DOTLLM_RUN_PREFILL_BENCH</c> is set — it must never run
/// in normal CI. The bf16 path additionally requires a net11 build on AVX512-BF16 hardware; when that
/// is absent only the inner-vs-integer comparison is produced (still meaningful — the integer
/// outer-product is the safe-to-ship default).
/// </para>
/// <para>
/// <b>Vacuousness guards.</b> Each config asserts the expected kernel actually executed via the global
/// invocation counters — without this, a silent fallback could time the same code path three times and
/// report a false "no impact". Mirrors the parity / accuracy tests' discriminating guards.
/// </para>
/// </remarks>
[Collection("Llama32Instruct")]
public class Llama32PrefillOperatorBenchmark
{
    private readonly Llama32InstructFixture _fixture;

    public Llama32PrefillOperatorBenchmark(Llama32InstructFixture fixture)
    {
        _fixture = fixture;
    }

    private static bool Bf16KernelAvailable =>
#if NET11_0_OR_GREATER
        System.Runtime.Intrinsics.X86.Avx512Bf16.IsSupported;
#else
        false;
#endif

    // Prefill lengths to sweep (llama.cpp pp256 / pp512 equivalents). Prefill is matmul-bound at these
    // sizes — GEMM FLOPs (O(n·d²)) dominate attention (O(n²·d)) for d=2048 — so the operator impact is
    // well-represented; the larger N simply amortizes per-call overhead.
    private static readonly int[] PrefillLengths = { 256, 512 };

    private const int Warmup = 3;   // let tiered JIT / dynamic PGO tier up the kernel before timing
    private const int Timed = 9;    // odd count → well-defined median; report median + min (Strix is noisy)

    [SkippableFact]
    public void PrefillThroughput_InnerVsIntegerVsBf16()
    {
        Skip.If(string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DOTLLM_RUN_PREFILL_BENCH")),
            "Prefill operator benchmark is opt-in — set DOTLLM_RUN_PREFILL_BENCH=1 to run.");

        var gguf = GgufFile.Open(_fixture.FilePath);
        using var _ = gguf;
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.Auto);
        int vocab = config.VocabSize;

        Console.WriteLine(
            $"[Llama32PrefillBench] model={Path.GetFileName(_fixture.FilePath)} arch={config.Architecture} " +
            $"layers={config.NumLayers} hidden={config.HiddenSize} threads={ThreadingConfig.Auto.EffectiveThreadCount} " +
            $"bf16Available={Bf16KernelAvailable} warmup={Warmup} timed={Timed}");

        foreach (int n in PrefillLengths)
        {
            int[] tokenIds = new int[n];
            int[] positions = new int[n];
            var rng = new Random(42);
            for (int i = 0; i < n; i++)
            {
                tokenIds[i] = rng.Next(1, Math.Min(vocab, 32000));
                positions[i] = i;
            }

            var inner = Measure(model, "inner-product", tokenIds, positions, outer: false, bf16: false);
            var integer = Measure(model, "integer-outer", tokenIds, positions, outer: true, bf16: false);
            BenchResult? bf16 = Bf16KernelAvailable
                ? Measure(model, "bf16-outer", tokenIds, positions, outer: true, bf16: true)
                : null;

            Console.WriteLine($"--- prefill n={n} tokens (median of {Timed}, after {Warmup} warmup) ---");
            Report(inner, baseline: inner);
            Report(integer, baseline: inner);
            if (bf16 is { } b)
                Report(b, baseline: inner);
        }

        // ── Decode (single-token, n=1) ─────────────────────────────────────────────────────────────
        // These operators are PREFILL-only (gated on n>1); decode routes to the inner-product GEMV path
        // and never touches the outer-product / bf16 kernel. We measure decode anyway because it is the
        // *blend denominator*: real generation speed is a prompt:completion-weighted mix of the prefill
        // speedup and the (unchanged) decode speed — a reader needs the decode tok/s to compute their
        // workload's blended number. The bf16Tiles==0 delta across the decode loop is the proof the
        // operators do not engage; decode tok/s vs the RAM-bandwidth ceiling indicates whether a future
        // decode GEMV kernel (AVX-512 / VNNI / bf16) could help at all (decode is usually BW-bound).
        const int decodeContext = 256;   // prefill this many tokens into the KV-cache first
        const int decodeSteps = 32;      // then time this many single-token decode steps
        var dRng = new Random(7);
        int[] ctxTokens = new int[decodeContext];
        int[] ctxPositions = new int[decodeContext];
        for (int i = 0; i < decodeContext; i++)
        {
            ctxTokens[i] = dRng.Next(1, Math.Min(vocab, 32000));
            ctxPositions[i] = i;
        }
        int decodeToken = dRng.Next(1, Math.Min(vocab, 32000));

        using var kvCache = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim,
            decodeContext + Warmup + decodeSteps + 8);

        var dInner = MeasureDecode(model, kvCache, "inner-product", ctxTokens, ctxPositions, decodeToken, decodeSteps, outer: false, bf16: false);
        var dInteger = MeasureDecode(model, kvCache, "integer-outer", ctxTokens, ctxPositions, decodeToken, decodeSteps, outer: true, bf16: false);
        BenchResult? dBf16 = Bf16KernelAvailable
            ? MeasureDecode(model, kvCache, "bf16-outer", ctxTokens, ctxPositions, decodeToken, decodeSteps, outer: true, bf16: true)
            : null;

        Console.WriteLine($"--- decode 1 token/step, {decodeContext}-token context (median of {decodeSteps}, after {Warmup} warmup) ---");
        Report(dInner, baseline: dInner);
        Report(dInteger, baseline: dInner);
        if (dBf16 is { } db)
            Report(db, baseline: dInner);
        Console.WriteLine(
            "  (decode is the blend denominator — operators are prefill-only, so decode speed is unchanged; " +
            "bf16Tiles=0 confirms the bf16 kernel never engages on n=1.)");
    }

    private static BenchResult Measure(
        TransformerModel model, string name, int[] tokenIds, int[] positions, bool outer, bool bf16)
    {
        model.UseOuterProductQ8Prefill = outer;
        model.UseBf16OuterProductQ8Prefill = bf16;

        for (int w = 0; w < Warmup; w++)
        {
            using var warm = model.Forward(tokenIds, positions, deviceId: -1);
        }

        long gemmBefore = MatMul.OuterProductGemmQ8_0InvocationCount;
        long bf16Before = MatMul.OuterProductQ8_0Avx512Bf16TileCount;

        var times = new double[Timed];
        var sw = new Stopwatch();
        for (int it = 0; it < Timed; it++)
        {
            sw.Restart();
            using var logits = model.Forward(tokenIds, positions, deviceId: -1);
            sw.Stop();
            times[it] = sw.Elapsed.TotalMilliseconds;
        }

        long gemmDelta = MatMul.OuterProductGemmQ8_0InvocationCount - gemmBefore;
        long bf16Delta = MatMul.OuterProductQ8_0Avx512Bf16TileCount - bf16Before;

        // Vacuousness guards: prove the intended kernel actually ran (and the others did not), so the
        // three timings are genuinely different code paths — not the same path measured three times.
        if (outer)
            Assert.True(gemmDelta > 0, $"{name}: outer-product GEMM never invoked (delta={gemmDelta}) — silent fallback would make this measurement vacuous.");
        else
            Assert.True(gemmDelta == 0, $"{name}: outer-product GEMM ran with the flag OFF (delta={gemmDelta}) — the inner-product baseline is contaminated.");
        if (bf16)
            Assert.True(bf16Delta > 0, $"{name}: bf16 tile never executed (delta={bf16Delta}) — bf16 silently fell back to the integer kernel.");
        else
            Assert.True(bf16Delta == 0, $"{name}: bf16 tile executed with the bf16 flag OFF (delta={bf16Delta}).");

        Array.Sort(times);
        double median = times[Timed / 2];
        double min = times[0];
        int n = tokenIds.Length;
        return new BenchResult(name, median, min, n / (median / 1000.0), gemmDelta, bf16Delta);
    }

    // Realistic decode timing: prefill `context` tokens into the KV-cache, then time `steps` single-token
    // forwards at advancing positions (each attends to the growing cache). Rolls the cache back to the
    // post-prefill length between warmup and timing so every timed step starts from the same state.
    private static BenchResult MeasureDecode(
        TransformerModel model, SimpleKvCache kvCache, string name,
        int[] ctxTokens, int[] ctxPositions, int decodeToken, int steps, bool outer, bool bf16)
    {
        model.UseOuterProductQ8Prefill = outer;
        model.UseBf16OuterProductQ8Prefill = bf16;

        int basePos = ctxTokens.Length;
        int[] one = { decodeToken };
        int[] pos = new int[1];

        // Fresh prefill of the context into the cache (sets CurrentLength = basePos).
        kvCache.Rollback(0);
        using (var _ = model.Forward(ctxTokens, ctxPositions, deviceId: -1, kvCache)) { }

        // Warm the decode path (tier-up applies here too), then reset to the post-prefill state.
        for (int w = 0; w < Warmup; w++)
        {
            pos[0] = basePos + w;
            using var warm = model.Forward(one, pos, deviceId: -1, kvCache);
        }
        kvCache.Rollback(basePos);

        long bf16Before = MatMul.OuterProductQ8_0Avx512Bf16TileCount;
        long gemmBefore = MatMul.OuterProductGemmQ8_0InvocationCount;

        var times = new double[steps];
        var sw = new Stopwatch();
        for (int s = 0; s < steps; s++)
        {
            pos[0] = basePos + s;
            sw.Restart();
            using var logits = model.Forward(one, pos, deviceId: -1, kvCache);
            sw.Stop();
            times[s] = sw.Elapsed.TotalMilliseconds;
        }

        long bf16Delta = MatMul.OuterProductQ8_0Avx512Bf16TileCount - bf16Before;
        long gemmDelta = MatMul.OuterProductGemmQ8_0InvocationCount - gemmBefore;

        // The discriminating fact for decode: the outer-product operators are prefill-only (n>1), so on
        // single-token decode they must NEVER engage regardless of the flags — proving decode speed is
        // genuinely unaffected by these operators (not merely "happened to be the same").
        Assert.True(gemmDelta == 0 && bf16Delta == 0,
            $"{name}: an outer-product/bf16 kernel ran during n=1 decode (gemm={gemmDelta}, bf16={bf16Delta}) — " +
            "the n>1 prefill gate is broken; decode should never use these operators.");

        Array.Sort(times);
        double median = times[steps / 2];
        double min = times[0];
        return new BenchResult(name, median, min, 1.0 / (median / 1000.0), gemmDelta, bf16Delta);
    }

    private static void Report(BenchResult r, BenchResult baseline)
    {
        double speedup = r.MedianMs > 0 ? baseline.MedianMs / r.MedianMs : 0;
        Console.WriteLine(
            $"  {r.Name,-16} median={r.MedianMs,8:F2} ms  min={r.MinMs,8:F2} ms  " +
            $"throughput={r.TokensPerSec,8:F1} tok/s  speedup={speedup,5:F3}x  " +
            $"gemmCalls={r.GemmCalls} bf16Tiles={r.Bf16Tiles}");
    }

    private readonly record struct BenchResult(
        string Name, double MedianMs, double MinMs, double TokensPerSec, long GemmCalls, long Bf16Tiles);
}
