using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// The end-to-end gate for gpt-oss on CUDA: loads ONE synthetic gpt-oss GGUF
/// through both the CPU oracle (<see cref="TransformerModel.LoadFromGguf(GgufFile, ModelConfig, ThreadingConfig)"/>)
/// and the CUDA dispatch point this issue unblocked
/// (<see cref="CudaModelLoader.CreateFromGguf"/>, whose
/// <c>Architecture.GptOss</c> <see cref="NotSupportedException"/> guard #365
/// replaced with the plain <see cref="CudaTransformerModel"/> arm) and compares
/// last-token logits.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this test exists.</b> gpt-oss CUDA support was delivered across three
/// separate issues, each verified only in isolation. This is the first thing
/// that proves they compose. Every gpt-oss feature that differs from a plain
/// Llama-family model is simultaneously live in the single fixture below —
/// four feature groups, five independently measurable behaviours (group 1
/// contributes both the expert biases and the clamped activation):
/// </para>
/// <list type="number">
/// <item><description><b>#348 — MoE per-expert bias + clamped <c>swiglu_oai</c>.</b>
/// Every one of the 4 layers is a routed MoE FFN (4 experts, top-2,
/// softmax-after-top-k) with MXFP4 expert banks and
/// <c>ffn_{gate,up,down}_exps.bias</c>. Exercises
/// <c>CudaMoeWeightsLoader.LoadLayerQuant</c>'s bias upload and
/// <c>CudaMoeFfn.Forward</c>'s <c>LaunchSwiGLUOaiF32</c> branch.</description></item>
/// <item><description><b>#366 — alternating sliding-window/dense attention.</b>
/// <see cref="ModelConfig.SlidingWindowPattern"/> = 2 over
/// <see cref="NumLayers"/> = 4 layers: layers 0/2 windowed to
/// <see cref="SlidingWindow"/> = 8, layers 1/3 dense. <see cref="SeqLen"/> = 24
/// is 3x the window, so a dense layer's true receptive field is 3x what a
/// windowed layer sees — the discriminating shape (a sequence at or below the
/// window makes windowed and dense attention numerically identical, so it
/// would pass by accident). Exercises <c>CudaSlidingWindowResolver</c>.</description></item>
/// <item><description><b>#366 — dense YaRN RoPE mscale.</b> The fixture declares
/// <c>rope.scaling.type = yarn</c> with factor 32 over an original context of 8
/// — the same 32x ratio gpt-oss ships (131072/4096), scaled down. The entire
/// measured effect comes from <c>factor</c> (2→32): it raises the mscale
/// multiplier from ≈1.0693 to ≈1.3466 (scaling cos/sin, and therefore Q, K and
/// every attention score, at every position including position 0), and it
/// divides the interpolated ramp dimensions' frequencies by 32 instead of 2 —
/// 16x more frequency compression on those dimensions. <c>original_context_length</c>
/// (32→8) is kept in step to preserve gpt-oss's real shipped ratio
/// (<c>8 × 32 = 256</c>, mirroring gpt-oss's real <c>4096 × 32 = 131072</c>) but is
/// numerically inert at this fixture's headDim=8 geometry — the YaRN correction
/// range (<c>low</c>/<c>high</c> in
/// <see cref="DotLLM.Cpu.Kernels.RoPE.ComputeYarnInverseFrequencies"/>)
/// clamps to the identical <c>[0, 1]</c> at both settings, so it contributes
/// nothing to the measured effect size; it is a no-op here, not the mechanism.
/// Before #366 CUDA RoPE silently dropped YaRN's mscale term.</description></item>
/// <item><description><b>#365 — per-head attention sinks.</b> Every layer carries
/// <c>attn_sinks.weight</c> with a DISTINCT value per head (1.0, 1.5, 2.0, 2.5
/// across the 4 query heads) under GQA (4 Q heads over 2 KV heads), so a kernel
/// that indexed the sink by KV head, by a flat scalar, or by the wrong head
/// stride would diverge rather than coincide. Exercises the sink epilogue
/// through <c>TransformerLayerWeights.AttnSinksDevice</c>.</description></item>
/// </list>
/// <para>
/// <b>Which kernel this actually hits.</b> <see cref="CudaTransformerModel"/>'s
/// default dense dispatch is the FP16 <c>attention_f16</c> path, not
/// <c>attention_f32</c> — so this test primarily covers Task 4's F16 sink
/// epilogue. That is deliberate: it is the production dispatch a real gpt-oss
/// checkpoint takes. The F32 epilogue is covered by the kernel-level
/// <c>CudaAttentionSinksKernelTests</c>.
/// </para>
/// <para>
/// <b>Fixture provenance.</b> The GGUF writer below is a port of
/// <c>TransformerModelGptOssForwardTests.WriteFixture</c>
/// (tests/DotLLM.Tests.Unit/Models/Architectures/TransformerModelGptOssForwardTests.cs)
/// — the same <see cref="GgufWriter"/> mechanism, same MXFP4 expert-bank
/// layout, same formula-driven sink generation, same
/// <c>post_attention_norm</c> (gpt-oss has no <c>ffn_norm.weight</c>) — widened
/// from that test's window=3/seqLen=6 to window=8/seqLen=24 so the dense layers
/// have a receptive field 3x the window, the discriminating shape
/// <c>CudaAlternatingSwaParityTests</c> established for #366. Feature toggles
/// (<see cref="SinkMode"/>, <see cref="BiasMode"/>) are applied as scalar
/// multipliers AFTER the RNG draw so toggling never perturbs the shared RNG
/// stream: two fixtures built with the same seed are bit-identical on every
/// tensor except the one under test.
/// </para>
/// <para>
/// <b>Tolerance.</b> See <see cref="AbsTol"/>. Calibrated from an observed run
/// on a real RTX 3060, and cross-checked against the measured per-feature
/// effect sizes asserted by
/// <see cref="GptOssFixture_EveryFeatureMovesLogitsFarAboveParityTolerance"/> —
/// a parity tolerance only discriminates a feature whose effect exceeds it, so
/// that second test is what stops this one degenerating into a test that would
/// still pass if CUDA dropped a feature entirely.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class CudaGptOssParitySyntheticTests : IDisposable
{
    // ── Fixture shape ──
    // HiddenSize and MoeIntermediateSize are both pinned to 32: MXFP4 blocks are
    // 32 elements, so every quantized row's K dimension must be a multiple of 32
    // (gate/up have K=HiddenSize, down has K=MoeIntermediateSize).
    private const int HiddenSize = 32;
    private const int NumLayers = 4;      // pattern=2 -> layers 0,2 windowed; 1,3 dense.
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;     // GQA 4Q/2KV: distinct per-head sinks cannot alias KV heads.
    private const int HeadDim = HiddenSize / NumHeads;  // 8
    private const int VocabSize = 12;
    private const int NumExperts = 4;
    private const int TopK = 2;
    private const int MoeIntermediateSize = 32;
    private const int SlidingWindow = 8;
    private const int SeqLen = 24;        // 3x the window: dense layers see 24, windowed layers 8.

    // Dense-YaRN shape. gpt-oss ships factor=32 over an original context of 4096
    // (131072 / 4096 = 32x); this fixture keeps that exact 32x ratio at fixture
    // scale — original context 8, full context 256.
    //
    // The entire measured effect-size increase comes from `factor` (2 -> 32):
    //   • it sets the mscale term to 1 + 0.1*ln(32) = 1.3466 (vs 1.0693 at
    //     factor 2), which multiplies cos AND sin, so Q and K are each scaled
    //     by mscale and the attention scores (Q.K) by ~mscale^2 ~= 1.81 — at
    //     EVERY position, including position 0.
    //   • it also divides the interpolated ramp dimensions' frequencies by 32
    //     instead of 2 (freqInter = 1/(scalingFactor * theta^exponent)) — 16x
    //     more frequency compression on those dimensions.
    // `original_context_length` (32 -> 8) was ALSO changed, to preserve
    // gpt-oss's real shipped ratio (8*32=256, mirroring gpt-oss's real
    // 4096*32=131072) — but at this fixture's small geometry (headDim=8,
    // theta=10000, betaFast=32, betaSlow=1) it is numerically INERT: the YaRN
    // correction range in RoPE.ComputeYarnInverseFrequencies
    // (src/DotLLM.Cpu/Kernels/RoPE.cs) clamps to low=0, high=1 at BOTH
    // original-context settings, so the inverse-frequency table is
    // bit-identical whether original_context_length is 8 or 32. It is kept for
    // shape-fidelity / documentation-of-intent, not because it does anything
    // at this scale — the ramp is a function of DIMENSION index, not of
    // whether a token position exceeds the original context length.
    // Measured consequence: the YaRN effect on the logits rose from 5.515E-003
    // (below AbsTol — the parity gate could not have seen CUDA drop YaRN) to the
    // value pinned in the AbsTol comment below. See task-5-report.md.
    private const int ContextLength = 256;
    private const int YarnOrigContextLength = 8;
    private const float YarnScalingFactor = 32.0f;

    // Parity tolerance: F32 CPU oracle vs the FP16-internal CUDA dense forward
    // (weights upload as F16, cuBLAS HGEMM, attention_f16), so the noise floor is
    // dominated by FP16 GEMM rounding accumulated over 4 layers plus the CPU's
    // Schraudolph exp approximation vs the kernel's expf in both the attention
    // softmax and the MoE router softmax.
    //
    // Empirically calibrated on THIS fixture (2026-09-02, RTX 3060 — raw runs in
    // task-5-report.md). AbsTol is NOT a predicted number; it was set from what
    // was measured, and it is bounded on BOTH sides:
    //
    //   lower bound (noise): observed max |diff| on the as-committed fixture =
    //     1.022E-003. Diffuse across all 12 logits with the worst column being
    //     the one whose CPU logit is nearest zero, i.e. absolute rounding noise,
    //     not a systematic feature error. Note this is ~7x the 1.533E-004
    //     CudaAlternatingSwaParityTests sees on a dense-F32 fixture of similar
    //     depth — the extra comes from MXFP4 expert dequant and the MoE router's
    //     top-k softmax, neither of which that test has.
    //
    //   upper bound (discrimination): the per-feature effect sizes measured by
    //     GptOssFixture_EveryFeatureMovesLogitsFarAboveParityTolerance are
    //     5.856E-001 (swiglu_oai), 4.130E-001 (SWA), 3.164E-001 (sinks),
    //     1.078E-001 (expert biases) and 1.020E-001 (dense YaRN — the binding
    //     one). A tolerance above that smallest effect would let CUDA drop YaRN
    //     entirely and still pass, which is exactly the failure mode
    //     CudaAlternatingSwaParityTests documents hitting when it followed a
    //     "~100x margin" rule of thumb.
    //
    // 8.0E-003 sits near the geometric centre of [1.022E-003, 1.020E-001]
    // (centre 1.02E-002): ~7.8x above the noise floor, ~12.7x below the smallest
    // feature effect.
    private const float AbsTol = 8.0e-3f;
    private const float RelTol = 2.0e-3f;

    /// <summary>
    /// Minimum multiple of <see cref="AbsTol"/> that each individual feature's
    /// effect on the logits must clear for the parity assertion to be able to
    /// detect that feature going missing on the GPU. Set to 5, not 10: the
    /// binding feature (#366 dense YaRN, effect 1.020E-001) clears
    /// <see cref="AbsTol"/> by 12.7x, so a 10x bar would leave only 1.27x
    /// headroom and turn any minor fixture tweak into a spurious failure. At 5x
    /// the bar is 4.0E-002 and the binding feature clears it by 2.55x.
    /// </summary>
    private const float MinEffectSizeMultiple = 5.0f;

    private static readonly int[] Ids = BuildIds();
    private static readonly int[] Pos = BuildPositions();

    private readonly ITestOutputHelper _out;
    private readonly string _scratch;

    public CudaGptOssParitySyntheticTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gptoss-cuda-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  The gate
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Prefill last-token-logit parity, CPU oracle vs CUDA, on a synthetic
    /// gpt-oss GGUF with attention sinks, alternating SWA, dense YaRN RoPE and
    /// biased MXFP4 MoE experts all active at once. Loads the CUDA side through
    /// <see cref="CudaModelLoader.CreateFromGguf"/> (not
    /// <c>CudaTransformerModel.LoadFromGguf</c>) so the architecture dispatch
    /// this issue changed is itself under test.
    /// </summary>
    [SkippableFact]
    public void CudaForward_GptOssAllFeatures_PrefillVsCpu_LastTokenLogitsMatch()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string path = WriteFixture("gptoss-parity", seed: 42);

        using var gguf = GgufFile.Open(path);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        // Assert the fixture really does carry every feature before trusting a
        // parity match: a fixture that quietly lost one would match trivially.
        // Presence only — the EFFECT of each is measured by the sibling test,
        // because presence alone is too weak a check (IsDenseYarnActive is
        // satisfied by any factor > 1.0, including a factor whose effect on the
        // logits is far below AbsTol).
        Assert.Equal(Architecture.GptOss, config.Architecture);
        Assert.Equal(SlidingWindow, config.SlidingWindowSize);
        Assert.Equal(2, config.SlidingWindowPattern);
        // IsDenseYarnActive is precisely the predicate #366's CUDA RoPE mscale
        // fix gates on, so asserting it is what proves that path is live here.
        Assert.NotNull(config.RoPEConfig);
        Assert.True(config.RoPEConfig!.Value.IsDenseYarnActive);
        Assert.NotNull(config.Moe);
        Assert.True(config.Moe!.UseSwiGluOai);
        Assert.True(config.Moe.HasExpertBiases);
        Assert.True(config.Moe.SoftmaxAfterTopK);
        Assert.Equal(NumExperts, config.Moe.NumExperts);
        Assert.Equal(TopK, config.Moe.NumExpertsPerTok);

        float[] cpuLast = RunCpuLastRow(gguf, config);
        float[] cudaLast = RunCudaLastRow(gguf, config);

        AssertLogitsMatch(cpuLast, cudaLast);
    }

    /// <summary>
    /// Guards the gate above from silently degenerating. A CPU-vs-CUDA parity
    /// assertion can only catch a feature the GPU dropped if dropping that
    /// feature moves the logits by more than the tolerance. This measures the
    /// effect of every gpt-oss feature the fixture activates — sinks, MoE
    /// expert biases, alternating SWA, dense YaRN RoPE, clamped
    /// <c>swiglu_oai</c> — on the CPU side alone (no GPU needed) and
    /// asserts every one clears <see cref="AbsTol"/> by
    /// <see cref="MinEffectSizeMultiple"/>x. If a future fixture tweak shrinks
    /// an effect below that bar, this fails loudly instead of leaving the parity
    /// test quietly unable to see that feature.
    /// </summary>
    [Fact]
    public void GptOssFixture_EveryFeatureMovesLogitsFarAboveParityTolerance()
    {
        // #365 sinks: strong per-head sinks vs sinks so negative (-30) they
        // contribute ~0 to the softmax denominator, i.e. effectively no sink.
        // Same probe TransformerModelGptOssForwardTests uses.
        float[] sinksOn = RunCpuLastRow(WriteFixture("eff-sink-on", seed: 7, sinkMode: SinkMode.Strong));
        float[] sinksOff = RunCpuLastRow(WriteFixture("eff-sink-off", seed: 7, sinkMode: SinkMode.Negligible));

        // #348 expert biases: identical weights, ffn_*_exps.bias zeroed.
        float[] biasOn = RunCpuLastRow(WriteFixture("eff-bias-on", seed: 11, biasMode: BiasMode.Normal));
        float[] biasOff = RunCpuLastRow(WriteFixture("eff-bias-off", seed: 11, biasMode: BiasMode.Zeroed));

        // #366 alternating SWA, #366 dense YaRN RoPE and #348 clamped swiglu_oai
        // are all config-driven, so one fixture serves for all three: re-run it
        // with each feature switched off in the ModelConfig.
        string tweakPath = WriteFixture("eff-config", seed: 13);
        using var tweakGguf = GgufFile.Open(tweakPath);
        ModelConfig windowed = GgufModelConfigExtractor.Extract(tweakGguf.Metadata);
        float[] baseline = RunCpuLastRow(tweakGguf, windowed);

        float[] swaOff = RunCpuLastRow(tweakGguf, windowed with { SlidingWindowSize = null });
        float[] oaiOff = RunCpuLastRow(
            tweakGguf, windowed with { Moe = windowed.Moe! with { UseSwiGluOai = false } });

        // #366 dense YaRN RoPE. The parity gate asserts RoPEConfig.IsDenseYarnActive,
        // but that predicate is satisfied by ANY factor > 1.0 — a fixture edited to
        // e.g. factor=1.01 would keep the assert green while the actual effect on
        // the logits collapsed toward zero, leaving the gate unable to see CUDA
        // ignoring YaRN. That is not hypothetical: silently dropping YaRN's mscale
        // is precisely the pre-existing CUDA bug #366 found and fixed. So measure
        // the effect, don't just assert presence. Turning ScalingType off disables
        // the whole dense-YaRN path (both the inverse-frequency ramp and the mscale
        // term folded into cos/sin), so this bounds the full YaRN contribution, of
        // which the mscale term #366 restored is a part.
        float[] yarnOff = RunCpuLastRow(
            tweakGguf,
            windowed with
            {
                RoPEConfig = windowed.RoPEConfig!.Value with { ScalingType = RoPEScalingType.None },
            });

        float bar = AbsTol * MinEffectSizeMultiple;
        AssertEffectSize("#365 per-head attention sinks", sinksOn, sinksOff, bar);
        AssertEffectSize("#348 MoE per-expert biases", biasOn, biasOff, bar);
        AssertEffectSize("#366 alternating sliding-window attention", baseline, swaOff, bar);
        AssertEffectSize("#366 dense YaRN RoPE scaling", baseline, yarnOff, bar);
        AssertEffectSize("#348 clamped swiglu_oai activation", baseline, oaiOff, bar);
    }

    private void AssertEffectSize(string feature, float[] on, float[] off, float bar)
    {
        float effect = MaxAbsDiff(on, off);
        _out.WriteLine($"{feature,-46} effect = {effect:E3}  (bar {bar:E3}, {effect / AbsTol:F1}x AbsTol)");
        Assert.All(on, v => Assert.True(float.IsFinite(v), $"{feature}: non-finite logit"));
        Assert.All(off, v => Assert.True(float.IsFinite(v), $"{feature}: non-finite logit"));
        Assert.True(effect > bar,
            $"{feature} moves the logits by only {effect:E3}, at or below {MinEffectSizeMultiple}x the "
            + $"parity tolerance ({bar:E3}). The CPU-vs-CUDA parity test therefore could not detect "
            + "CUDA dropping this feature. Strengthen the fixture before trusting the parity result.");
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Forward runners
    // ──────────────────────────────────────────────────────────────────────

    private float[] RunCpuLastRow(string path)
    {
        using var gguf = GgufFile.Open(path);
        return RunCpuLastRow(gguf, GgufModelConfigExtractor.Extract(gguf.Metadata));
    }

    private static unsafe float[] RunCpuLastRow(GgufFile gguf, ModelConfig config)
    {
        using var model = TransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        using ITensor logits = model.Forward(Ids, Pos, deviceId: -1);
        // CPU Forward returns [seqLen, vocab]; the CUDA side returns only the
        // last row, so slice to match.
        Assert.Equal(SeqLen, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, SeqLen * VocabSize);
        return span.Slice((SeqLen - 1) * VocabSize, VocabSize).ToArray();
    }

    private static unsafe float[] RunCudaLastRow(GgufFile gguf, ModelConfig config)
    {
        // Through CudaModelLoader.CreateFromGguf specifically: the guard removal
        // in this issue is a change to THIS dispatch, so the test has to go
        // through it rather than construct CudaTransformerModel directly.
        var (model, _) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0);
        using (model)
        {
            Assert.IsType<CudaTransformerModel>(model);
            using ITensor logits = model.Forward(Ids, Pos, deviceId: -1);
            // CUDA prefill returns [1, vocab] by design (saves an LM-head GEMM).
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);
            return new ReadOnlySpan<float>((void*)logits.DataPointer, VocabSize).ToArray();
        }
    }

    private void AssertLogitsMatch(float[] cpu, float[] cuda)
    {
        Assert.Equal(cpu.Length, cuda.Length);

        float maxDiff = 0f;
        _out.WriteLine("col | cpu        | cuda       | |diff|");
        _out.WriteLine("----+------------+------------+----------");
        for (int c = 0; c < cpu.Length; c++)
        {
            float d = MathF.Abs(cpu[c] - cuda[c]);
            maxDiff = MathF.Max(maxDiff, d);
            _out.WriteLine($"{c,3} | {cpu[c],10:F6} | {cuda[c],10:F6} | {d:E3}");
        }
        _out.WriteLine($"max |diff| = {maxDiff:E3}   (AbsTol {AbsTol:E3})");

        for (int c = 0; c < cpu.Length; c++)
        {
            Assert.True(float.IsFinite(cpu[c]), $"col={c}: cpu logit non-finite: {cpu[c]}");
            Assert.True(float.IsFinite(cuda[c]), $"col={c}: cuda logit non-finite: {cuda[c]}");
            float diff = MathF.Abs(cpu[c] - cuda[c]);
            float bar = AbsTol + RelTol * MathF.Abs(cpu[c]);
            Assert.True(diff <= bar,
                $"col={c}: cpu={cpu[c]:F6} vs cuda={cuda[c]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Synthetic gpt-oss GGUF fixture
    //  (ported from TransformerModelGptOssForwardTests.WriteFixture)
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>Attention-sink magnitude, for the #365 effect-size probe.</summary>
    private enum SinkMode
    {
        /// <summary>Distinct moderate positives per head — materially compete in the softmax denominator.</summary>
        Strong,
        /// <summary>Hugely negative — <c>exp(-30)</c> contributes ~0, i.e. effectively no sink.</summary>
        Negligible,
    }

    /// <summary>Per-expert MoE bias presence, for the #348 effect-size probe.</summary>
    private enum BiasMode
    {
        /// <summary>Biases as drawn.</summary>
        Normal,
        /// <summary>Biases zeroed AFTER the RNG draw, so every other tensor stays bit-identical.</summary>
        Zeroed,
    }

    /// <summary>
    /// Writes a synthetic gpt-oss GGUF: token_embd/output/output_norm, and per
    /// layer attn_norm + Q/K/V/O (F32, with biases) + attn_sinks + the
    /// gpt-oss-named post_attention_norm (no ffn_norm.weight — exercises the
    /// loader's fallback, which the CUDA path inherits because
    /// CudaTransformerModel.LoadFromGguf builds on TransformerWeights.LoadFromGguf)
    /// + a routed-MoE block (ffn_gate_inp router + MXFP4 ffn_{gate,up,down}_exps
    /// 3D expert banks + per-expert biases).
    /// </summary>
    private string WriteFixture(
        string name, int seed,
        SinkMode sinkMode = SinkMode.Strong,
        BiasMode biasMode = BiasMode.Normal)
    {
        string path = Path.Combine(_scratch, $"{name}.gguf");
        var rng = new Random(seed);
        var w = new GgufWriter();
        const string arch = "gpt-oss";

        w.AddString("general.architecture", arch);
        w.AddUInt32("general.alignment", 32);
        w.AddUInt32($"{arch}.context_length", ContextLength);
        w.AddUInt32($"{arch}.embedding_length", HiddenSize);
        w.AddUInt32($"{arch}.block_count", NumLayers);
        w.AddUInt32($"{arch}.feed_forward_length", MoeIntermediateSize);
        w.AddUInt32($"{arch}.attention.head_count", NumHeads);
        w.AddUInt32($"{arch}.attention.head_count_kv", NumKvHeads);
        w.AddUInt32($"{arch}.attention.key_length", HeadDim);
        w.AddUInt32($"{arch}.attention.value_length", HeadDim);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f);
        w.AddUInt32($"{arch}.attention.sliding_window", SlidingWindow);
        // sliding_window_pattern intentionally omitted -> extractor defaults to 2
        // (llama.cpp set_swa_pattern(2)): even layers windowed, odd layers dense.
        w.AddUInt32($"{arch}.expert_count", NumExperts);
        w.AddUInt32($"{arch}.expert_used_count", TopK);
        w.AddUInt32($"{arch}.expert_feed_forward_length", MoeIntermediateSize);
        w.AddFloat32($"{arch}.rope.freq_base", 10000.0f);
        // Dense YaRN scaling — the shape of gpt-oss's shipped rope_scaling (same
        // 32x factor), and the path #366 fixed (CUDA RoPE previously dropped
        // YaRN's mscale term entirely).
        w.AddString($"{arch}.rope.scaling.type", "yarn");
        w.AddFloat32($"{arch}.rope.scaling.factor", YarnScalingFactor);
        w.AddUInt32($"{arch}.rope.scaling.original_context_length", YarnOrigContextLength);
        w.AddUInt32($"{arch}.vocab_size", VocabSize);

        AddMatrixF32(w, rng, "token_embd.weight", inK: HiddenSize, outM: VocabSize, 0.05f);

        for (int i = 0; i < NumLayers; i++)
            AddLayer(w, rng, i, sinkMode, biasMode);

        AddNormF32(w, rng, "output_norm.weight", HiddenSize);
        AddMatrixF32(w, rng, "output.weight", inK: HiddenSize, outM: VocabSize, 0.05f);

        File.WriteAllBytes(path, w.Build());
        return path;
    }

    private static void AddLayer(GgufWriter w, Random rng, int layer, SinkMode sinkMode, BiasMode biasMode)
    {
        string p = $"blk.{layer}";
        const int qOut = NumHeads * HeadDim;    // = HiddenSize
        const int kvOut = NumKvHeads * HeadDim;

        AddNormF32(w, rng, $"{p}.attn_norm.weight", HiddenSize);
        AddMatrixF32(w, rng, $"{p}.attn_q.weight", inK: HiddenSize, outM: qOut, 0.1f);
        AddMatrixF32(w, rng, $"{p}.attn_k.weight", inK: HiddenSize, outM: kvOut, 0.1f);
        AddMatrixF32(w, rng, $"{p}.attn_v.weight", inK: HiddenSize, outM: kvOut, 0.1f);
        AddMatrixF32(w, rng, $"{p}.attn_output.weight", inK: qOut, outM: HiddenSize, 0.1f);
        AddVectorF32(w, rng, $"{p}.attn_q.bias", qOut, 0.02f);
        AddVectorF32(w, rng, $"{p}.attn_k.bias", kvOut, 0.02f);
        AddVectorF32(w, rng, $"{p}.attn_v.bias", kvOut, 0.02f);
        AddVectorF32(w, rng, $"{p}.attn_output.bias", HiddenSize, 0.02f);

        // #365 attention sinks — one scalar per QUERY head, deliberately distinct
        // (1.0/1.5/2.0/2.5) under GQA 4Q/2KV so a kernel indexing the sink by KV
        // head, by a flat scalar, or with the wrong head stride diverges instead
        // of coinciding. Formula-driven (no RNG draw), so the Strong/Negligible
        // pair stays bit-identical on every other tensor.
        float[] sinks = new float[NumHeads];
        for (int h = 0; h < NumHeads; h++)
            sinks[h] = sinkMode == SinkMode.Strong ? 1.0f + 0.5f * h : -30.0f;
        w.AddTensor($"{p}.attn_sinks.weight", [NumHeads], (uint)QuantizationType.F32, ToBytes(sinks));

        // gpt-oss names its pre-FFN norm "post_attention_norm" (llama.cpp
        // LLM_TENSOR_ATTN_POST_NORM) — there is no ffn_norm.weight tensor at all,
        // so this exercises the shared loader's fallback lookup on both backends.
        AddNormF32(w, rng, $"{p}.post_attention_norm.weight", HiddenSize);

        // #348 routed MoE — every gpt-oss layer. Router + MXFP4 expert banks.
        AddMatrixF32(w, rng, $"{p}.ffn_gate_inp.weight", inK: HiddenSize, outM: NumExperts, 0.1f);
        AddVectorF32(w, rng, $"{p}.ffn_gate_inp.bias", NumExperts, 0.05f);

        AddMxfp4ExpertBank(w, rng, $"{p}.ffn_gate_exps.weight",
            inK: HiddenSize, midOut: MoeIntermediateSize, experts: NumExperts);
        AddMxfp4ExpertBank(w, rng, $"{p}.ffn_up_exps.weight",
            inK: HiddenSize, midOut: MoeIntermediateSize, experts: NumExperts);
        AddMxfp4ExpertBank(w, rng, $"{p}.ffn_down_exps.weight",
            inK: MoeIntermediateSize, midOut: HiddenSize, experts: NumExperts);

        // Per-expert biases, flat [E x inner] expert-major (matches
        // TransformerWeights.LoadQuantExpertMoeLayer / CudaMoeWeightsLoader).
        // BiasMode.Zeroed scales AFTER the draw so the RNG stream is unchanged.
        //
        // Amplitude 0.2 rather than the 0.05 used for every other bias in this
        // fixture: on an earlier revision of this fixture, 0.05 produced a
        // measured CPU-side effect of only 2.460E-002 against a parity noise
        // floor of 3.233E-003 -- just 7.6x -- too narrow a band to place a
        // tolerance that both clears the noise and stays below the effect.
        // Raising the amplitude widens the feature effect instead of widening
        // the tolerance, which is the direction that preserves discrimination.
        // (The same reasoning later drove the dense-YaRN strengthening at
        // YarnScalingFactor above; both are recorded in task-5-report.md.)
        // Still small in absolute terms (the residual stream is O(0.1-1) here),
        // so nothing saturates the swiglu_oai clamp.
        const float ExpertBiasAmplitude = 0.2f;
        float biasScale = biasMode == BiasMode.Normal ? 1.0f : 0.0f;
        AddVectorF32(w, rng, $"{p}.ffn_gate_exps.bias", NumExperts * MoeIntermediateSize, ExpertBiasAmplitude, biasScale);
        AddVectorF32(w, rng, $"{p}.ffn_up_exps.bias", NumExperts * MoeIntermediateSize, ExpertBiasAmplitude, biasScale);
        AddVectorF32(w, rng, $"{p}.ffn_down_exps.bias", NumExperts * HiddenSize, ExpertBiasAmplitude, biasScale);
    }

    /// <summary>Emits a 2D F32 matrix [ne0=K, ne1=M] of small deterministic weights.</summary>
    private static void AddMatrixF32(GgufWriter w, Random rng, string name, int inK, int outM, float amplitude)
    {
        long count = (long)inK * outM;
        float[] f = new float[count];
        for (long i = 0; i < count; i++) f[i] = (float)(rng.NextDouble() * 2 - 1) * amplitude;
        w.AddTensor(name, [inK, outM], (uint)QuantizationType.F32, ToBytes(f));
    }

    /// <summary>Emits a 1D F32 RMSNorm weight vector [n], centered at 1.0.</summary>
    private static void AddNormF32(GgufWriter w, Random rng, string name, int n)
    {
        float[] f = new float[n];
        for (int i = 0; i < n; i++) f[i] = 1.0f + (float)(rng.NextDouble() * 2 - 1) * 0.05f;
        w.AddTensor(name, [n], (uint)QuantizationType.F32, ToBytes(f));
    }

    /// <summary>
    /// Emits a 1D F32 bias/vector [n]. <paramref name="scale"/> is applied after
    /// the RNG draw (0.0 zeroes the tensor without perturbing the RNG stream).
    /// </summary>
    private static void AddVectorF32(
        GgufWriter w, Random rng, string name, int n, float amplitude, float scale = 1.0f)
    {
        float[] f = new float[n];
        for (int i = 0; i < n; i++) f[i] = (float)(rng.NextDouble() * 2 - 1) * amplitude * scale;
        w.AddTensor(name, [n], (uint)QuantizationType.F32, ToBytes(f));
    }

    /// <summary>
    /// Emits a 3D MXFP4 expert bank [ne0=K, ne1=midOut, ne2=experts]: one
    /// 17-byte block per row (K == 32) — an E8M0 scale byte near unit magnitude
    /// plus 16 nibble-pair bytes, matching <c>GptOssKernelTests.RandomMxfp4</c>.
    /// The values need not reconstruct any particular float (that arithmetic is
    /// pinned by Mxfp4Tests); they only need to be well-formed MXFP4 bytes
    /// flowing through the real GGUF loader on both backends.
    /// </summary>
    private static void AddMxfp4ExpertBank(GgufWriter w, Random rng, string name, int inK, int midOut, int experts)
    {
        if (inK % 32 != 0)
            throw new ArgumentException("MXFP4 K dimension must be a multiple of 32.", nameof(inK));

        long rows = (long)midOut * experts;
        long blocksPerRow = inK / 32;
        byte[] data = new byte[rows * blocksPerRow * 17];
        long o = 0;
        for (long r = 0; r < rows; r++)
        {
            for (long b = 0; b < blocksPerRow; b++)
            {
                data[o++] = (byte)rng.Next(122, 133); // E8M0 scale byte, near unit magnitude.
                for (int j = 0; j < 16; j++) data[o++] = (byte)rng.Next(256);
            }
        }
        w.AddTensor(name, [inK, midOut, experts], (uint)QuantizationType.MXFP4, data);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Small helpers
    // ──────────────────────────────────────────────────────────────────────

    private static int[] BuildIds()
    {
        var rng = new Random(7);
        int[] ids = new int[SeqLen];
        for (int i = 0; i < SeqLen; i++) ids[i] = rng.Next(VocabSize);
        return ids;
    }

    private static int[] BuildPositions()
    {
        int[] pos = new int[SeqLen];
        for (int i = 0; i < SeqLen; i++) pos[i] = i;
        return pos;
    }

    private static byte[] ToBytes(float[] f) => MemoryMarshal.AsBytes(f.AsSpan()).ToArray();

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float maxDiff = 0f;
        for (int i = 0; i < a.Length; i++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(a[i] - b[i]));
        return maxDiff;
    }
}
