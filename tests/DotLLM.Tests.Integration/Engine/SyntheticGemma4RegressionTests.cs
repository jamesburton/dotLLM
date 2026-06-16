using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Self-contained regression + all-features gate for the synthetic Gemma-4 / DiffusionGemma
/// fixture (<see cref="SyntheticGemma4Gguf"/>). Unlike the real-weight tests this generates
/// its OWN tiny GGUF, so it runs in CI with no env var / multi-gig checkpoint. It asserts:
/// <list type="number">
/// <item>the extracted <see cref="ModelConfig"/> carries every Gemma-4 feature (dual head dim,
/// global KV heads, MoE, partial rope 0.25, attn scalar 1.0, softcap 30, dual FFN);</item>
/// <item>a single-thread cacheless causal forward over raw token ids is DETERMINISTIC and
/// STABLE — its last-row logit checksum + argmax equal a hard-coded golden, run-to-run;</item>
/// <item>the diffusion-gemma fixture loads with a non-null DiffusionConfig + self-cond weights
/// and a short deterministic denoise loop produces stable, non-degenerate output;</item>
/// <item>all four quant types (Q8_0, Q4_K, Q5_1, F32) are exercised by the fixture tensors.</item>
/// </list>
/// </summary>
public sealed class SyntheticGemma4RegressionTests
{
    private readonly ITestOutputHelper _out;

    public SyntheticGemma4RegressionTests(ITestOutputHelper output) => _out = output;

    // ── Golden constants ──
    // Captured once from the deterministic fixture (gemma4 tiny, seed 0xC0FFEE, PromptIds below,
    // single-thread cacheless causal forward) on the dev host (AVX2 dequant path). The ARGMAX is
    // the firm cross-environment golden (robust to float-bit jitter). The CHECKSUM is an FNV-1a
    // over the exact float bits of the last logit row — asserted run-to-run for stability, and
    // hard-coded here as the AVX2-path golden; on a non-AVX2 (scalar dequant) host the last-bit
    // rounding can differ, so the checksum assertion is gated on AVX2 being available.
    private const ulong Gemma4LastRowChecksum = 0x2DA986DD36EB50E3UL;
    private const int Gemma4LastRowArgmax = 144;

    private static readonly int[] PromptIds = [2, 17, 42, 99, 123, 200, 7];

    private static ulong Fnv1a(ReadOnlySpan<float> row)
    {
        ulong h = 14695981039346656037UL;
        foreach (float f in row)
        {
            uint bits = (uint)BitConverter.SingleToInt32Bits(f);
            for (int b = 0; b < 4; b++)
            {
                h ^= (byte)(bits >> (b * 8));
                h *= 1099511628211UL;
            }
        }
        return h;
    }

    private static (ulong checksum, int argmax, int vocab) RunGemma4Forward(string path)
    {
        var (model, gguf, config) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
        using var _ = gguf;
        using var __ = model;
        int n = PromptIds.Length;
        int[] positions = new int[n];
        for (int i = 0; i < n; i++) positions[i] = i;
        using ITensor logits = model.Forward(PromptIds, positions, deviceId: -1, kvCache: null,
            adapter: null, AttentionMaskSpec.Causal);
        int vocab = logits.Shape[1];
        unsafe
        {
            float* p = (float*)logits.DataPointer;
            var lastRow = new ReadOnlySpan<float>(p + (long)(n - 1) * vocab, vocab);
            ulong checksum = Fnv1a(lastRow);
            int argmax = 0; float best = float.NegativeInfinity;
            for (int v = 0; v < vocab; v++) if (lastRow[v] > best) { best = lastRow[v]; argmax = v; }
            return (checksum, argmax, vocab);
        }
    }

    /// <summary>
    /// THE GATE: generate the tiny gemma4 fixture, load it, assert all Gemma-4 features are present,
    /// run a deterministic causal forward, and assert the last-row checksum + argmax match the golden
    /// AND are byte-identical across two independent loads (run-to-run stability).
    /// </summary>
    [Fact]
    public void Gemma4_Tiny_AllFeatures_And_DeterministicGolden()
    {
        var cfgDims = SyntheticGemma4Gguf.Tiny;
        string path = Path.Combine(Path.GetTempPath(), $"syn_gemma4_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, cfgDims);
        try
        {
            // ── Load + config feature assertions ──
            var (model, gguf, config) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
            using (gguf)
            using (model)
            {
                Assert.Equal(Architecture.Gemma4, config.Architecture);
                Assert.True(config.Gemma4DualFfn, "Gemma4DualFfn must be set.");
                Assert.NotNull(config.Moe);
                Assert.Equal(cfgDims.ExpertCount, config.Moe!.NumExperts);
                Assert.Equal(cfgDims.ExpertUsedCount, config.Moe.NumExpertsPerTok);
                Assert.Equal(cfgDims.ExpertFeedForward, config.Moe.MoeIntermediateSize);
                // Dual head dim: global != sliding.
                Assert.NotNull(config.GlobalHeadDim);
                Assert.NotEqual(config.HeadDim, config.GlobalHeadDim!.Value);
                Assert.Equal(cfgDims.SlidingHeadDim, config.HeadDim);
                Assert.Equal(cfgDims.GlobalHeadDim, config.GlobalHeadDim!.Value);
                // Dual KV heads.
                Assert.NotNull(config.NumGlobalKvHeads);
                Assert.Equal(cfgDims.GlobalKvHeads, config.NumGlobalKvHeads!.Value);
                Assert.Equal(cfgDims.SlidingKvHeads, config.NumKvHeads);
                // Partial rope, attn scalar, softcap, embedding scale.
                Assert.Equal(0.25f, config.PartialRotaryFactor);
                Assert.Equal(1.0f, config.QueryPreAttnScalar);
                Assert.Equal(30f, config.FinalLogitSoftcap);
                Assert.True(config.TiedEmbeddings, "Gemma ties embeddings.");
                Assert.True(MathF.Abs(MathF.Sqrt(cfgDims.HiddenSize) - (config.EmbeddingScale ?? 0f)) < 1e-3f,
                    $"EmbeddingScale {config.EmbeddingScale} != sqrt(hidden) {MathF.Sqrt(cfgDims.HiddenSize)}.");
                // Per-layer sliding pattern present (last layer global).
                Assert.NotNull(config.PerLayerSlidingWindow);
                Assert.Null(config.PerLayerSlidingWindow![cfgDims.BlockCount - 1]); // global layer
                Assert.NotNull(config.PerLayerSlidingWindow![0]);                   // sliding layer
            }

            // ── Deterministic forward (two independent loads) ──
            var (c1, a1, vocab) = RunGemma4Forward(path);
            var (c2, a2, _) = RunGemma4Forward(path);
            _out.WriteLine($"[gemma4 tiny] vocab={vocab} argmax={a1} checksum=0x{c1:X16}");
            Assert.Equal(c1, c2);   // run-to-run stable
            Assert.Equal(a1, a2);
            Assert.InRange(a1, 0, vocab - 1);

            // Golden: argmax is the firm cross-environment constant; checksum is the AVX2-path
            // golden (gated, since scalar-dequant hosts may differ in the last rounding bit).
            Assert.Equal(Gemma4LastRowArgmax, a1);
            if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
                Assert.Equal(Gemma4LastRowChecksum, c1);
        }
        finally { File.Delete(path); }
    }

    /// <summary>
    /// Confirms the tiny fixture's tensors exercise all four target quant types (Q8_0 attn/dense,
    /// Q4_K gate_up experts, Q5_1 down experts, F32 norms/scales/router) by re-reading the GGUF
    /// tensor descriptors. This is the quant-coverage gate.
    /// </summary>
    [Fact]
    public void Gemma4_Tiny_ExercisesAllFourQuantTypes()
    {
        string path = Path.Combine(Path.GetTempPath(), $"syn_gemma4_q_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path);
        try
        {
            using var gguf = GgufFile.Open(path);
            var byName = gguf.TensorsByName;
            Assert.Equal(QuantizationType.Q8_0, byName["blk.0.attn_q.weight"].QuantizationType);
            Assert.Equal(QuantizationType.Q8_0, byName["blk.0.ffn_gate.weight"].QuantizationType);
            Assert.Equal(QuantizationType.Q4_K, byName["blk.0.ffn_gate_up_exps.weight"].QuantizationType);
            Assert.Equal(QuantizationType.Q5_1, byName["blk.0.ffn_down_exps.weight"].QuantizationType);
            Assert.Equal(QuantizationType.F32, byName["blk.0.attn_norm.weight"].QuantizationType);
            Assert.Equal(QuantizationType.F32, byName["blk.0.ffn_gate_inp.weight"].QuantizationType);
            Assert.Equal(QuantizationType.F32, byName["blk.0.layer_output_scale.weight"].QuantizationType);

            var seen = new HashSet<QuantizationType>(byName.Values.Select(d => d.QuantizationType));
            Assert.Contains(QuantizationType.Q8_0, seen);
            Assert.Contains(QuantizationType.Q4_K, seen);
            Assert.Contains(QuantizationType.Q5_1, seen);
            Assert.Contains(QuantizationType.F32, seen);
            _out.WriteLine($"[gemma4 tiny] quant types present: {string.Join(", ", seen.OrderBy(q => q.ToString()))}");
        }
        finally { File.Delete(path); }
    }

    /// <summary>
    /// The diffusion-gemma fixture: load, assert a non-null DiffusionConfig + self-cond weights,
    /// and run a short deterministic denoise loop (small canvas, greedy sampler) over raw prompt
    /// ids. Asserts the output is non-degenerate and byte-identical across two runs.
    /// </summary>
    [Fact]
    public void DiffusionGemma_Tiny_LoadsDiffusionConfig_And_DeterministicDenoise()
    {
        var dims = SyntheticGemma4Gguf.Tiny;
        string path = Path.Combine(Path.GetTempPath(), $"syn_diffgemma_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteDiffusionGemma(path, dims);
        try
        {
            // Config + self-cond presence.
            {
                var (model, gguf, config) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
                using (gguf)
                using (model)
                {
                    Assert.Equal(Architecture.DiffusionGemma, config.Architecture);
                    Assert.NotNull(config.DiffusionConfig);
                    Assert.Equal(dims.CanvasLength, config.DiffusionConfig!.CanvasLength);
                    Assert.Equal(dims.MaskTokenId, config.DiffusionConfig.MaskTokenId);
                    Assert.Equal(AttentionMaskMode.Hybrid, config.DiffusionConfig.CanvasAttentionMode);
                }
                // The self_cond_* tensors are present in the diffusion fixture (asserted via the
                // GGUF descriptor below) and are wired into the forward by the loader; the denoise
                // loop exercises them on steps > 0.
                using var dgguf = GgufFile.Open(path);
                Assert.Contains("self_cond_pre_norm.weight", dgguf.TensorsByName.Keys);
                Assert.Contains("self_cond_gate.weight", dgguf.TensorsByName.Keys);
                Assert.Contains("self_cond_up.weight", dgguf.TensorsByName.Keys);
                Assert.Contains("self_cond_down.weight", dgguf.TensorsByName.Keys);
                Assert.Contains("blk.0.enc_layer_output_scale.weight", dgguf.TensorsByName.Keys);
            }

            int[] gen1 = RunDenoise(path);
            int[] gen2 = RunDenoise(path);
            int distinct = gen1.Distinct().Count();
            _out.WriteLine($"[diffusion-gemma tiny] gen={gen1.Length} distinct={distinct} ids=[{string.Join(",", gen1)}]");

            // DETERMINISM is the gate: the greedy denoise loop must be byte-identical run-to-run.
            Assert.Equal(gen1, gen2);
            Assert.NotEmpty(gen1);
            foreach (int id in gen1) Assert.InRange(id, 0, dims.VocabSize - 1);
            // NON-DEGENERATE: not a single repeated token, and at least one real (non-mask)
            // content token. NOTE: with random (untrained) fixture weights the denoiser cannot
            // be expected to fully materialise the canvas in a few steps — the trained 26B does
            // (see DiffusionGemmaGgufForwardTests). The synthetic gate is determinism + signal.
            Assert.True(distinct > 1, $"Denoise output degenerate (distinct {distinct}); ids=[{string.Join(",", gen1)}].");
            Assert.Contains(gen1, id => id != dims.MaskTokenId);
        }
        finally { File.Delete(path); }
    }

    private static int[] RunDenoise(string path)
    {
        var (model, gguf, config) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
        using var _ = gguf;
        using var __ = model;
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        // Small canvas + few steps; greedy (default EntropyBoundSampler → argmax, deterministic).
        var diff = config.DiffusionConfig! with
        {
            CanvasLength = 8,
            MaxDenoisingSteps = 6,
            TemperatureMax = 0.0f,
            TemperatureMin = 0.0f,
        };
        var gen = new DiffusionTextGenerator(model, tokenizer, diffusionConfig: diff);
        int[] promptIds = [2, 17, 42, 7]; // raw ids (BOS + arbitrary) — no real tokenizer round-trip needed
        DiffusionResult r = gen.Generate(promptIds);
        return r.GeneratedTokenIds;
    }
}
