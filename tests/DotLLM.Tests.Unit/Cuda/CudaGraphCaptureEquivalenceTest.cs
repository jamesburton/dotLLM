using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Equivalence test: the CUDA Graphs decode replay path must produce identical
/// argmax (and near-identical logits) to the eager kernel-launch path. Validates
/// the device-resident <c>seq_kv</c> / <c>position_offset</c> mechanism and the
/// device-side KV-cache write kernel against the existing eager forward as oracle.
/// </summary>
/// <remarks>
/// <para><b>Vacuity guard (#338).</b> Every case here used to be able to pass
/// <em>eager-vs-eager</em>. Setting <c>UseGraphCapture = true</c> is not evidence that capture
/// engaged: the constructor clears the flag outright when the kv-write kernel is missing (a stale
/// or absent <c>kv_write</c> PTX — precisely the failure class of #318/#288 — emits one line to
/// stderr and falls back), and the dispatch gate re-checks six further conditions before it
/// replays anything. Any one of them silently routes to the eager body, at which point the suite
/// compared eager against eager and passed green while proving nothing about graph capture.</para>
/// <para>Each case therefore now asserts <see cref="CudaTransformerModel.GraphReplayCount"/> is
/// non-zero on the graph run — and zero on the eager run, so the control arm is a real control.
/// The depth-threshold case additionally asserts it saw BOTH regimes, which is the property that
/// case exists to test.</para>
/// <para><b>All steps are compared (#338).</b> Logits used to be compared numerically only at
/// step 0; steps 1..N were argmax-only, and since decoding is greedy self-feeding those were not
/// independent samples. A drift that only appears after several replays (stale device
/// <c>seq_kv</c>/<c>position_offset</c>, a KV-write off-by-one at a later slot) was invisible
/// unless it happened to flip an argmax. Every step's logits are now bounded.</para>
/// </remarks>
[Trait("Category", "GPU")]
public class CudaGraphCaptureEquivalenceTest
{
    private readonly ITestOutputHelper _out;

    public CudaGraphCaptureEquivalenceTest(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Max-abs logit divergence tolerated between the eager and graph runs.
    /// </summary>
    /// <remarks>
    /// TODO(#338): this bound is ~20-25x the drift actually observed in-suite (0.2-0.25), so the
    /// effective gate is argmax equality and this only catches catastrophic divergence. It was
    /// widened from <c>1e-3f</c> in <c>52034e97</c> on the theory that PTX-JIT cache state left by
    /// earlier tests changes SASS scheduling and hence FP accumulation order. That explanation is
    /// asserted, not demonstrated — "order-dependent, disappears standalone" is equally the
    /// signature of a real cross-test state leak, and we now have independent evidence of such
    /// leaks elsewhere. Re-deriving it needs measured headroom on a CUDA box; it is NOT tightened
    /// here, because nobody can run it on the Strix Halo box to check. See the T5500 runbook in
    /// the PR for #338.
    /// </remarks>
    private const float LogitAbsTolerance = 5.0f;

    /// <summary>One decode run's outputs plus the graph-engagement observables it produced.</summary>
    private sealed record DecodeRun(int[] Tokens, float[][] Logits, int Replays, int Captures, int EagerDecodes);

    [SkippableTheory]
    [InlineData("SmolLM-135M.Q4_K_M.gguf", "DOTLLM_SMOLLM_135M_Q4_K_M_GGUF")]
    [InlineData("SmolLM-135M.Q8_0.gguf", "DOTLLM_SMOLLM_Q8_GGUF")]
    public void EagerVsGraphDecode_Match(string modelFile, string envVar)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        FixtureLocation fixture = ResolveSmolLM(modelFile, envVar);
        Skip.If(!fixture.Found, fixture.SkipMessage($"SmolLM-135M {modelFile}"));

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] prompt = Prompt(gguf);

        const int decodeSteps = 32;
        int kvCap = prompt.Length + decodeSteps + 8;

        DecodeRun eager = RunDecode(gguf, config, prompt, decodeSteps, kvCap, useGraphCapture: false);
        DecodeRun graph = RunDecode(gguf, config, prompt, decodeSteps, kvCap, useGraphCapture: true);

        AssertGraphEngaged(eager, graph, decodeSteps, modelFile);
        AssertEquivalent(eager, graph, modelFile);
    }

    /// <summary>
    /// Same equivalence check as <see cref="EagerVsGraphDecode_Match"/>, but with
    /// the mixed-precision quantized KV-cache (Q8_0 stored region + small FP16 window).
    /// Validates that the device-resident eviction state machine (predicated
    /// quant-on-evict + dyn dequant + window scatter) matches the host-driven
    /// eager path. This is the test that gates the 2x graph-decode speedup landing
    /// for KV-quantized configs.
    /// </summary>
    [SkippableFact]
    public void EagerVsGraphDecode_QuantizedKv_Match()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        FixtureLocation fixture = ResolveSmolLM("SmolLM-135M.Q4_K_M.gguf", "DOTLLM_SMOLLM_135M_Q4_K_M_GGUF");
        Skip.If(!fixture.Found, fixture.SkipMessage("SmolLM-135M Q4_K_M GGUF"));

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] prompt = Prompt(gguf);

        const int decodeSteps = 32;
        int kvCap = prompt.Length + decodeSteps + 8;
        // Window small enough that we exercise both phases (window-only and
        // post-eviction) during the timed decode steps.
        var kvCfg = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0, MixedPrecisionWindowSize: 16);

        DecodeRun eager = RunDecode(gguf, config, prompt, decodeSteps, kvCap, useGraphCapture: false, kvCfg);
        DecodeRun graph = RunDecode(gguf, config, prompt, decodeSteps, kvCap, useGraphCapture: true, kvCfg);

        AssertGraphEngaged(eager, graph, decodeSteps, "quantized-kv");
        AssertEquivalent(eager, graph, "quantized-kv");
    }

    /// <summary>
    /// BitNet (I2_S ternary) variant of <see cref="EagerVsGraphDecode_Match"/>: the eager
    /// path's FP32-residual / Sub-LN / ReLU² branches must be exactly mirrored by
    /// <c>CaptureDecodeGraph</c> — issue #212. BitNet is a strong discriminator (unlike
    /// the dense SmolLM path above, a wrong captured body here can silently overflow FP16
    /// or skip the Sub-LN normalization entirely, producing high-confidence-but-wrong
    /// logits rather than a crash) so this locks in both the real BitNet-2B-4T model
    /// (hidden=2560, 128-aligned) and the ragged bitnet_b1_58-xl model (hidden=2048,
    /// intermediate=5460, not a multiple of 128 — exercises the ragged I2_S GEMV path
    /// inside the graph too, see issue #206).
    /// </summary>
    [SkippableTheory]
    [InlineData("BitNet-2B-4T")]
    [InlineData("bitnet_b1_58-xl")]
    public void EagerVsGraphDecode_BitNet_Match(string label)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        FixtureLocation fixture = ResolveBitNet(label, out string description);
        Skip.If(!fixture.Found, fixture.SkipMessage(description));

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.BitNet, config.Architecture);
        int[] prompt = Prompt(gguf);

        const int decodeSteps = 16;
        int kvCap = prompt.Length + decodeSteps + 8;

        DecodeRun eager = RunDecode(gguf, config, prompt, decodeSteps, kvCap, useGraphCapture: false);
        DecodeRun graph = RunDecode(gguf, config, prompt, decodeSteps, kvCap, useGraphCapture: true);

        AssertGraphEngaged(eager, graph, decodeSteps, label);
        AssertEquivalent(eager, graph, label);
    }

    /// <summary>
    /// Issue #213: BitNet's graph-capture eligibility falls back to eager once the running KV
    /// length reaches <see cref="CudaTransformerModel.BitNetGraphCaptureMaxDepth"/> (default 384).
    /// This decodes PAST that threshold so the "graph" run exercises BOTH halves: graph replay
    /// while shallow, then a mid-generation transition to eager, all within a single
    /// model/KV-cache instance. Must still match the eager oracle — the transition itself must not
    /// corrupt any state (KV-cache length bookkeeping, the cached <c>_decodeGraphExec</c> sitting
    /// unused post-transition, etc.).
    /// </summary>
    /// <remarks>
    /// #338: previously this case would have passed identically had capture never engaged at any
    /// depth — the very thing it was written to characterise. It now asserts that BOTH regimes were
    /// actually visited, which no other case here can establish.
    /// </remarks>
    [SkippableFact]
    public void EagerVsGraphDecode_BitNet_CrossesGraphDepthThreshold_Match()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        FixtureLocation fixture = KnownTestFixtures.BitNetI2S;
        Skip.If(!fixture.Found, fixture.SkipMessage(KnownTestFixtures.BitNetI2SDescription));

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] prompt = Prompt(gguf);

        int threshold = CudaTransformerModel.BitNetGraphCaptureMaxDepth;
        int decodeSteps = Math.Max(32, (threshold - prompt.Length) + 64);
        int kvCap = prompt.Length + decodeSteps + 8;
        _out.WriteLine($"decodeSteps={decodeSteps} (threshold={threshold})");

        DecodeRun eager = RunDecode(gguf, config, prompt, decodeSteps, kvCap, useGraphCapture: false);
        DecodeRun graph = RunDecode(gguf, config, prompt, decodeSteps, kvCap, useGraphCapture: true);

        AssertGraphEngaged(eager, graph, decodeSteps, "depth-threshold");
        Assert.True(graph.EagerDecodes > 0,
            $"the depth-threshold case decoded {decodeSteps} steps past a threshold of {threshold} but "
            + "never fell back to eager, so it exercised only one of the two regimes it exists to "
            + "cover. Either the threshold moved or the fallback did not fire (#338).");
        _out.WriteLine($"regimes visited: {graph.Replays} graph replays, {graph.EagerDecodes} eager decodes");

        AssertEquivalent(eager, graph, "depth-threshold");
    }

    // ════════════════════════════════════════════════════════════════════
    // Helpers
    // ════════════════════════════════════════════════════════════════════

    private static int[] Prompt(GgufFile gguf)
    {
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        return tokenizer.Encode("The capital of France is Paris. The capital of Germany is");
    }

    /// <summary>
    /// Resolves one SmolLM-135M quant.
    /// </summary>
    /// <remarks>
    /// The override variable is PER-QUANT deliberately. `TestFixtureResolver` accepts a
    /// file-valued override and returns it as-is without checking it is the file that was asked
    /// for, so a single shared `DOTLLM_SMOLLM_135M_GGUF` pointing at the Q8_0 blob would make the
    /// Q4_K_M case silently test Q8_0 — a fixture quietly not being what the test says it is, which
    /// is the class of defect this change exists to remove. `DOTLLM_SMOLLM_Q8_GGUF` matches the
    /// name already used by CudaTurboQuantKvEndToEndTests.
    /// </remarks>
    private static FixtureLocation ResolveSmolLM(string modelFile, string envVar)
        => TestFixtureResolver.ResolveFile(envVar, "QuantFactory", "SmolLM-135M-GGUF", modelFile);

    private static FixtureLocation ResolveBitNet(string label, out string description)
    {
        if (string.Equals(label, "BitNet-2B-4T", StringComparison.Ordinal))
        {
            description = KnownTestFixtures.BitNetI2SDescription;
            return KnownTestFixtures.BitNetI2S;
        }

        description = "bitnet_b1_58-xl I2_S GGUF (ragged hidden=2048 / intermediate=5460)";
        return TestFixtureResolver.ResolveFile(
            ["DOTLLM_BITNET_XL_GGUF"],
            "1bitLLM",
            "bitnet_b1_58-xl",
            ["ggml-model-i2_s.gguf"],
            // Legacy conventional path this suite hard-coded before #338.
            ["E:/Development/bitnet-tests/models/bitnet_b1_58-xl",
             "C:/Development/bitnet-tests/models/bitnet_b1_58-xl"]);
    }

    /// <summary>
    /// Prefill + <paramref name="decodeSteps"/> greedy decode steps, capturing every step's full
    /// logit row and the model's graph-engagement counters.
    /// </summary>
    private static unsafe DecodeRun RunDecode(
        GgufFile gguf, ModelConfig config, int[] prompt, int decodeSteps, int kvCap,
        bool useGraphCapture, KvCacheConfig? kvCfg = null)
    {
        int vocab = config.VocabSize;
        var tokens = new int[decodeSteps];
        var logits = new float[decodeSteps][];

        using var model = CudaTransformerModel.LoadFromGguf(gguf, config);
        // Graph capture is default-on, so BOTH arms set the flag explicitly — the eager arm is a
        // control and must be known-eager, not merely default.
        model.UseGraphCapture = useGraphCapture;

        using var kv = kvCfg is { } cfg
            ? (DotLLM.Core.Attention.IKvCache)model.CreateKvCache(kvCap, cfg)
            : (DotLLM.Core.Attention.IKvCache)model.CreateKvCache(kvCap);

        int[] positions = new int[prompt.Length];
        for (int i = 0; i < prompt.Length; i++) positions[i] = i;
        using (var _ = model.Forward(prompt, positions, 0, kv)) { }   // prefill stays eager (multi-token)

        // Counters are read AFTER prefill so only the decode loop is measured.
        int replaysBefore = model.GraphReplayCount;
        int capturesBefore = model.GraphCaptureCount;
        int eagerBefore = model.EagerDecodeCount;

        int curTok = prompt[^1];
        int[] tokBuf = new int[1];
        int[] posBuf = new int[1];
        for (int i = 0; i < decodeSteps; i++)
        {
            tokBuf[0] = curTok;
            posBuf[0] = prompt.Length + i;
            using var t = model.Forward(tokBuf, posBuf, 0, kv);
            var row = new float[vocab];
            new ReadOnlySpan<float>((void*)t.DataPointer, vocab).CopyTo(row);
            logits[i] = row;
            tokens[i] = ArgMax(row);
            curTok = tokens[i];
        }

        return new DecodeRun(
            tokens, logits,
            model.GraphReplayCount - replaysBefore,
            model.GraphCaptureCount - capturesBefore,
            model.EagerDecodeCount - eagerBefore);
    }

    /// <summary>
    /// The vacuity guard (#338). Without it the two runs below can both be eager, and every
    /// assertion in this file passes while testing nothing.
    /// </summary>
    private void AssertGraphEngaged(DecodeRun eager, DecodeRun graph, int decodeSteps, string label)
    {
        _out.WriteLine($"[{label}] eager run: replays={eager.Replays} captures={eager.Captures} "
                        + $"eagerDecodes={eager.EagerDecodes}");
        _out.WriteLine($"[{label}] graph run: replays={graph.Replays} captures={graph.Captures} "
                        + $"eagerDecodes={graph.EagerDecodes}");

        Assert.True(eager.Replays == 0,
            $"[{label}] the CONTROL run replayed {eager.Replays} captured graph(s) despite "
            + "UseGraphCapture=false, so it is not an eager oracle.");
        Assert.True(eager.EagerDecodes == decodeSteps,
            $"[{label}] the control run took the eager decode body {eager.EagerDecodes} times, "
            + $"expected {decodeSteps}.");

        Assert.True(graph.Replays > 0,
            $"[{label}] graph capture never engaged: {graph.Replays} replays across {decodeSteps} "
            + "decode steps with UseGraphCapture=true, so this test compared eager against eager and "
            + "would have passed no matter what the capture path does. Check whether the constructor "
            + "cleared the flag (missing kv-write kernel -> stale/absent kv_write PTX, cf. #318/#288) "
            + "or the dispatch gate declined (batched call, multi-token, profiling enabled, MLA/MoE, "
            + "an active LoRA adapter, or depth past the capture ceiling). See #338.");
        Assert.True(graph.Captures > 0,
            $"[{label}] replays were counted but no capture was: {graph.Captures}.");
    }

    private void AssertEquivalent(DecodeRun eager, DecodeRun graph, string label)
    {
        _out.WriteLine($"[{label}] Eager tokens: [{string.Join(", ", eager.Tokens)}]");
        _out.WriteLine($"[{label}] Graph tokens: [{string.Join(", ", graph.Tokens)}]");

        // Argmax MUST match at every step — the real correctness gate.
        for (int i = 0; i < eager.Tokens.Length; i++)
        {
            Assert.True(eager.Tokens[i] == graph.Tokens[i],
                $"[{label}] Argmax divergence at step {i}: eager={eager.Tokens[i]}, graph={graph.Tokens[i]}");
        }

        // #338: every step's logits, not just step 0. Greedy self-feeding makes the per-step
        // argmaxes highly correlated, so step 0 alone was close to a single sample.
        float worst = 0;
        int worstStep = -1;
        for (int step = 0; step < eager.Logits.Length; step++)
        {
            float[] e = eager.Logits[step];
            float[] g = graph.Logits[step];
            Assert.Equal(e.Length, g.Length);
            float maxDiff = 0;
            double sumDiff = 0;
            for (int i = 0; i < e.Length; i++)
            {
                float d = MathF.Abs(e[i] - g[i]);
                sumDiff += d;
                if (d > maxDiff) maxDiff = d;
            }
            if (maxDiff > worst) { worst = maxDiff; worstStep = step; }
            if (step == 0 || step == eager.Logits.Length - 1)
                _out.WriteLine($"[{label}] step {step} logit max abs diff: {maxDiff:F6}, "
                                + $"mean diff: {sumDiff / e.Length:F6}");

            Assert.True(maxDiff < LogitAbsTolerance,
                $"[{label}] logit divergence too large at step {step}: max abs diff = {maxDiff}");
        }
        _out.WriteLine($"[{label}] worst logit max-abs diff {worst:F6} at step {worstStep} "
                        + $"(tolerance {LogitAbsTolerance}) - see #338 on re-deriving this bound");
    }

    private static int ArgMax(float[] data)
    {
        int best = 0;
        for (int i = 1; i < data.Length; i++)
            if (data[i] > data[best]) best = i;
        return best;
    }
}
