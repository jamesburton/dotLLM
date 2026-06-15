using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu;

/// <summary>
/// Go/no-go probe for the hybrid-dispatch ZP idea (issue #22): does throwing Meteor Lake's E-cores at
/// prefill GEMM actually help throughput, or do they regress it as stragglers? Compares real CPU prefill
/// of the cached Llama-3.2-1B Q8_0 under two threading configs, both using the existing sign-trick kernel:
/// <list type="bullet">
/// <item><b>all-16</b>: <c>ThreadCount=0</c> (all logical cores, unpinned) — workers float over P+E+LP-E.</item>
/// <item><b>pcore-6</b>: <c>ThreadCount=6, EnablePCorePinning=true</c> — pinned to the 6 P-cores only.</item>
/// </list>
/// Both models loaded in ONE process and timed INTERLEAVED (per the campaign's hard rule: a fresh-process
/// A/B is confounded by this laptop's ~2× clock drift). If all-16 already beats pcore-6, E-cores help even as
/// stragglers and ZP-on-E is a near-free win; if pcore-6 wins, capturing the E-core ZP speedup needs
/// proportional (heterogeneous) work partitioning — a much larger change. Opt-in <c>DOTLLM_RUN_PREFILL_BENCH=1</c>.
/// </summary>
[Trait("Category", "CPU")]
public sealed unsafe class CpuPrefillThreadingBench
{
    private readonly ITestOutputHelper _output;

    public CpuPrefillThreadingBench(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void All16VsPCore6_Prefill()
    {
        Skip.If(string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DOTLLM_RUN_PREFILL_BENCH")),
            "Opt-in — set DOTLLM_RUN_PREFILL_BENCH=1.");

        string model = Environment.GetEnvironmentVariable("DOTLLM_CPU_PERF_GGUF")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".dotllm", "test-cache", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.IfNot(File.Exists(model), $"Model not found: {model}");

        int prefillLen = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_PREFILL_LEN"), out int pl) ? pl : 512;
        int reps = 9;

        using var gguf = GgufFile.Open(model);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        using var mAll = TransformerModel.LoadFromGguf(gguf, config, new ThreadingConfig(ThreadCount: 0));
        using var mP = TransformerModel.LoadFromGguf(gguf, config, new ThreadingConfig(ThreadCount: 6, EnablePCorePinning: true));

        var rng = new Random(7);
        int vocab = config.VocabSize;
        int[] prompt = new int[prefillLen];
        int[] pos = new int[prefillLen];
        for (int i = 0; i < prefillLen; i++) { prompt[i] = rng.Next(1, Math.Min(vocab, 32000)); pos[i] = i; }

        double Prefill(TransformerModel m)
        {
            var sw = Stopwatch.StartNew();
            using var logits = m.Forward(prompt, pos, deviceId: 0, kvCache: null);
            _ = logits.Shape; // touch
            sw.Stop();
            return sw.Elapsed.TotalMilliseconds;
        }

        // Warm up both (JIT, page faults, repack-on-first-use).
        for (int w = 0; w < 3; w++) { Prefill(mAll); Prefill(mP); }

        double allMin = double.MaxValue, pMin = double.MaxValue;
        var ratios = new double[reps];
        for (int r = 0; r < reps; r++)
        {
            double a = Prefill(mAll);
            double p = Prefill(mP);
            ratios[r] = a > 0 ? p / a : 0; // >1 ⇒ all-16 faster than pcore-6
            if (a < allMin) allMin = a;
            if (p < pMin) pMin = p;
        }
        Array.Sort(ratios);

        _output.WriteLine($"model={Path.GetFileName(model)} prefillLen={prefillLen} reps={reps} (interleaved; ms = prefill of {prefillLen} tokens)");
        _output.WriteLine($"  all-16 (unpinned)   min={allMin,8:F1} ms   {prefillLen / (allMin / 1000.0),7:F1} tok/s");
        _output.WriteLine($"  pcore-6 (pinned)    min={pMin,8:F1} ms   {prefillLen / (pMin / 1000.0),7:F1} tok/s");
        _output.WriteLine($"  speedup(all-16 vs pcore-6): min={pMin / allMin:F3}x  median-ratio={ratios[reps / 2]:F3}x");
        _output.WriteLine($"  >1 ⇒ E-cores HELP prefill (ZP-on-E is near-free); <1 ⇒ stragglers, need proportional partitioning.");
    }

    /// <summary>
    /// End-to-end correctness gate for the ZP dispatch: a full prefill forward with the ZP path forced on
    /// ALL cores must produce logits matching the baseline (sign-trick) forward. Discriminates a wiring bug
    /// (wrong Σw indexing, scratch sizing, kernel selection) that the kernel-level parity tests can't see.
    /// Model-gated (skips when the cached GGUF is absent) but otherwise runs by default — not opt-in.
    /// </summary>
    [SkippableFact]
    public void ZpDispatch_ForwardMatchesBaseline()
    {
        Skip.IfNot(AvxVnni.IsSupported && !Avx512BW.IsSupported,
            "ZP-on-E path is only active on AVX-VNNI-without-AVX512 hosts (e.g. Meteor Lake).");
        string model = Environment.GetEnvironmentVariable("DOTLLM_CPU_PERF_GGUF")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".dotllm", "test-cache", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.IfNot(File.Exists(model), $"Model not found: {model}");

        int prefillLen = 96;
        using var gguf = GgufFile.Open(model);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var m = TransformerModel.LoadFromGguf(gguf, config, new ThreadingConfig(ThreadCount: 0));

        // The ZP kernel lives in the R4 outer-product prefill path, which is opt-in — without this the
        // forward falls back to the interleaved inner-product path and ZP never runs (a vacuous pass).
        m.UseOuterProductQ8Prefill = true;

        var rng = new Random(7);
        int vocab = config.VocabSize;
        int[] prompt = new int[prefillLen];
        int[] pos = new int[prefillLen];
        for (int i = 0; i < prefillLen; i++) { prompt[i] = rng.Next(1, Math.Min(vocab, 32000)); pos[i] = i; }

        float[] Run(bool zp, bool force)
        {
            MatMul.ZpOnEfficiencyCoresEnabled = zp;
            MatMul.ForceZpAllCoresForTesting = force;
            using var logits = m.Forward(prompt, pos, deviceId: 0, kvCache: null);
            int n = logits.Shape[logits.Shape.Rank - 1];
            var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
            return span.ToArray();
        }

        try
        {
            long invStart = MatMul.OuterProductGemmQ8_0InvocationCount;
            float[] baseLogits = Run(zp: false, force: false);
            float[] zpLogits = Run(zp: true, force: true);
            // Guard against a vacuous pass: the outer-product GEMM (where ZP lives) must have run.
            Assert.True(MatMul.OuterProductGemmQ8_0InvocationCount > invStart,
                "outer-product Q8_0 GEMM did not execute — ZP path was not exercised");

            Assert.Equal(baseLogits.Length, zpLogits.Length);
            float maxAbs = 0, maxDiff = 0;
            for (int i = 0; i < baseLogits.Length; i++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(baseLogits[i]));
                maxDiff = MathF.Max(maxDiff, MathF.Abs(baseLogits[i] - zpLogits[i]));
            }
            _output.WriteLine($"ZP-dispatch forward parity: maxAbs={maxAbs:F4} maxDiff={maxDiff:E3} rel={maxDiff / MathF.Max(maxAbs, 1e-6f):E3}");
            Assert.True(maxAbs > 1e-2f, "baseline logits ~0; parity check would be vacuous");
            // ZP's deferred compensation drifts from the sign trick by ~1e-3 per dot; over a full forward
            // this stays a small fraction of the logit magnitude. Tolerance relative to the logit scale.
            Assert.True(maxDiff < 1e-2f * maxAbs + 1e-2f,
                $"ZP forward diverged: maxDiff={maxDiff}, maxAbs={maxAbs}");
        }
        finally
        {
            MatMul.ZpOnEfficiencyCoresEnabled = false;
            MatMul.ForceZpAllCoresForTesting = false;
        }
    }

    /// <summary>
    /// End-to-end ZP-on-E-cores A/B (issue #22): one all-16-unpinned model, prefill timed with
    /// <see cref="MatMul.ZpOnEfficiencyCoresEnabled"/> OFF vs ON, INTERLEAVED in one process (same thread
    /// layout / memory pattern; only the E-core inner kernel differs, so this cancels clock drift far better
    /// than the cross-config A/B above). Ratio off/on &gt;1 ⇒ ZP-on-E speeds up real prefill. Only meaningful
    /// on AVX-VNNI-without-AVX512 hybrid hosts (Meteor Lake), where the VNNI outer-product branch + E-cores
    /// both exist. Opt-in <c>DOTLLM_RUN_PREFILL_BENCH=1</c>.
    /// </summary>
    [SkippableFact]
    public void ZpOnEcore_VsBaseline_Prefill()
    {
        Skip.If(string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DOTLLM_RUN_PREFILL_BENCH")),
            "Opt-in — set DOTLLM_RUN_PREFILL_BENCH=1.");
        Skip.IfNot(AvxVnni.IsSupported && !Avx512BW.IsSupported,
            "ZP-on-E path is only active on AVX-VNNI-without-AVX512 hosts (e.g. Meteor Lake).");

        string model = Environment.GetEnvironmentVariable("DOTLLM_CPU_PERF_GGUF")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".dotllm", "test-cache", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.IfNot(File.Exists(model), $"Model not found: {model}");

        int prefillLen = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_PREFILL_LEN"), out int pl) ? pl : 512;
        int reps = 11;

        using var gguf = GgufFile.Open(model);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var m = TransformerModel.LoadFromGguf(gguf, config, new ThreadingConfig(ThreadCount: 0));
        m.UseOuterProductQ8Prefill = true; // ZP lives in the outer-product prefill path (opt-in).

        var rng = new Random(7);
        int vocab = config.VocabSize;
        int[] prompt = new int[prefillLen];
        int[] pos = new int[prefillLen];
        for (int i = 0; i < prefillLen; i++) { prompt[i] = rng.Next(1, Math.Min(vocab, 32000)); pos[i] = i; }

        double PrefillWith(bool zp)
        {
            MatMul.ZpOnEfficiencyCoresEnabled = zp;
            var sw = Stopwatch.StartNew();
            using var logits = m.Forward(prompt, pos, deviceId: 0, kvCache: null);
            _ = logits.Shape;
            sw.Stop();
            return sw.Elapsed.TotalMilliseconds;
        }

        for (int w = 0; w < 3; w++) { PrefillWith(false); PrefillWith(true); }

        double offMin = double.MaxValue, onMin = double.MaxValue;
        var ratios = new double[reps];
        for (int r = 0; r < reps; r++)
        {
            double off = PrefillWith(false);
            double on = PrefillWith(true);
            ratios[r] = on > 0 ? off / on : 0; // >1 ⇒ ZP-on-E faster
            if (off < offMin) offMin = off;
            if (on < onMin) onMin = on;
        }
        MatMul.ZpOnEfficiencyCoresEnabled = false;
        Array.Sort(ratios);

        _output.WriteLine($"model={Path.GetFileName(model)} prefillLen={prefillLen} reps={reps} (all-16 unpinned; interleaved)");
        _output.WriteLine($"  ZP-on-E OFF (baseline)  min={offMin,8:F1} ms   {prefillLen / (offMin / 1000.0),7:F1} tok/s");
        _output.WriteLine($"  ZP-on-E ON              min={onMin,8:F1} ms   {prefillLen / (onMin / 1000.0),7:F1} tok/s");
        _output.WriteLine($"  speedup(ON vs OFF): min={offMin / onMin:F3}x  median-ratio={ratios[reps / 2]:F3}x");
        _output.WriteLine($"  >1 ⇒ ZP-on-E speeds up real prefill end-to-end.");
    }
}
