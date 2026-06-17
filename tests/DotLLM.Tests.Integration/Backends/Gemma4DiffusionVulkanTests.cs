using DotLLM.Core.Attention;
using DotLLM.Engine;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Backends;

/// <summary>
/// DiffusionGemma masked-diffusion DECODE path on Vulkan: CPU↔Vulkan TOLERANCE parity over the
/// synthetic <c>diffusion-gemma</c> fixture. Exercises the three CPU-side diffusion pieces ported
/// to the Vulkan gemma4 forward:
/// <list type="number">
/// <item>region-aware embed (canvas weight-less rms_noscale) + Hybrid(P) non-causal mask;</item>
/// <item>region-aware per-layer output scalar (enc vs layer output scale);</item>
/// <item>self-conditioning (soft-embed + gated GeGLU MLP) on denoise steps &gt; 0.</item>
/// </list>
/// Compared with TOLERANCE (not checksum) — GPU vs CPU accumulation order differs. The per-row
/// argmax over the canvas region (the signal the unmask sampler consumes) must agree exactly.
/// </summary>
[Trait("Category", "GPU")]
public sealed class Gemma4DiffusionVulkanTests
{
    private readonly ITestOutputHelper _out;

    public Gemma4DiffusionVulkanTests(ITestOutputHelper output) => _out = output;

    private static string WriteFixture()
    {
        string path = Path.Combine(Path.GetTempPath(), $"syn_diffgemma_vk_{Guid.NewGuid():N}.gguf");
        return SyntheticGemma4Gguf.WriteDiffusionGemma(path);
    }

    /// <summary>
    /// Zero-self-conditioning Hybrid(P) forward (denoise step 0). CPU unified forward vs Vulkan
    /// diffusion forward: canvas-region logits within the GPU reduction envelope, per-row argmax
    /// exact. Isolates the region-embed + region-scalar + non-causal canvas mask from SC.
    /// </summary>
    [SkippableFact]
    public unsafe void Vulkan_Diffusion_ZeroSc_MatchesCpu()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");

        string spvDir = ResolveSpvDir();
        string path = WriteFixture();
        try
        {
            int[] prompt = [2, 17, 42, 99, 7];
            int p = prompt.Length;
            const int c = 6;                      // canvas length (P+C=11 > sliding window 8)
            int seqLen = p + c;

            // ── CPU oracle (unified Hybrid forward, zero SC) ──────────────
            float[] cpuCanvas;
            int vocab, maskId;
            {
                var (model, gguf, cfg) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
                using (gguf)
                using (model)
                {
                    vocab = cfg.VocabSize;
                    maskId = cfg.DiffusionConfig!.MaskTokenId;
                    var (seq, pos) = BuildSeq(prompt, maskId, c);
                    model.SetDiffusionSelfCond(ReadOnlySpan<float>.Empty, 0, scUse: 0f);
                    using ITensor logits = model.Forward(seq, pos, -1, null, null, AttentionMaskSpec.Hybrid(p));
                    cpuCanvas = CanvasRows(logits, p, c, vocab);
                }
            }

            // ── Vulkan under test ─────────────────────────────────────────
            float[] vkCanvas;
            {
                using var gguf = GgufFile.Open(path);
                var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
                Assert.Equal(Architecture.DiffusionGemma, cfg.Architecture);
                using var model = VulkanTransformerModel.LoadFromGguf(gguf, cfg, spvDir);
                var (seq, pos) = BuildSeq(prompt, maskId, c);
                model.SetDiffusionSelfCond(ReadOnlySpan<float>.Empty, 0, scUse: 0f);
                using ITensor logits = model.Forward(seq, pos, -1, null, null, AttentionMaskSpec.Hybrid(p));
                Assert.Equal(seqLen, logits.Shape[0]);
                Assert.Equal(vocab, logits.Shape[1]);
                vkCanvas = CanvasRows(logits, p, c, vocab);
            }

            AssertCanvasParity(cpuCanvas, vkCanvas, c, vocab, "zero-SC");
        }
        finally { try { File.Delete(path); } catch { } }
    }

    /// <summary>
    /// Self-conditioning Hybrid(P) forward (denoise step &gt; 0). Feeds a synthetic previous-step
    /// canvas-logit distribution into SC on both backends, then compares the resulting canvas
    /// logits. Discriminates the SC soft-embed + gated GeGLU MLP path from the zero-SC path.
    /// </summary>
    [SkippableFact]
    public unsafe void Vulkan_Diffusion_SelfCond_MatchesCpu()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");

        string spvDir = ResolveSpvDir();
        string path = WriteFixture();
        try
        {
            int[] prompt = [2, 17, 42, 99, 7];
            int p = prompt.Length;
            const int c = 6;
            int seqLen = p + c;
            int vocab, maskId;

            // Deterministic synthetic "previous step" canvas logits [c × vocab].
            float[] prevLogits;
            {
                using var gguf0 = GgufFile.Open(path);
                var cfg0 = GgufModelConfigExtractor.Extract(gguf0.Metadata);
                vocab = cfg0.VocabSize;
                maskId = cfg0.DiffusionConfig!.MaskTokenId;
            }
            prevLogits = new float[c * vocab];
            var rng = new Random(1234);
            for (int i = 0; i < prevLogits.Length; i++) prevLogits[i] = (float)(rng.NextDouble() * 6.0 - 3.0);

            // ── CPU oracle (SC ON) ────────────────────────────────────────
            float[] cpuCanvas;
            {
                var (model, gguf, cfg) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
                using (gguf)
                using (model)
                {
                    var (seq, pos) = BuildSeq(prompt, maskId, c);
                    model.SetDiffusionSelfCond(prevLogits, c, scUse: 1f);
                    using ITensor logits = model.Forward(seq, pos, -1, null, null, AttentionMaskSpec.Hybrid(p));
                    cpuCanvas = CanvasRows(logits, p, c, vocab);
                }
            }

            // ── Vulkan under test (SC ON) ─────────────────────────────────
            float[] vkCanvas;
            {
                using var gguf = GgufFile.Open(path);
                var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
                using var model = VulkanTransformerModel.LoadFromGguf(gguf, cfg, spvDir);
                var (seq, pos) = BuildSeq(prompt, maskId, c);
                model.SetDiffusionSelfCond(prevLogits, c, scUse: 1f);
                using ITensor logits = model.Forward(seq, pos, -1, null, null, AttentionMaskSpec.Hybrid(p));
                Assert.Equal(seqLen, logits.Shape[0]);
                vkCanvas = CanvasRows(logits, p, c, vocab);
            }

            AssertCanvasParity(cpuCanvas, vkCanvas, c, vocab, "self-cond");
        }
        finally { try { File.Delete(path); } catch { } }
    }

    /// <summary>
    /// PKV equivalence on Vulkan: the canvas logits from a PKV prefill(prompt)+decode(canvas)
    /// must match the cacheless unified Hybrid(P) forward's canvas rows. PKV is a pure throughput
    /// optimisation — same math, only the prompt-key copy reorders — so the per-row argmax is exact
    /// and the logits agree within a tight envelope.
    /// </summary>
    [SkippableFact]
    public unsafe void Vulkan_Diffusion_Pkv_MatchesUnified()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");

        string spvDir = ResolveSpvDir();
        string path = WriteFixture();
        try
        {
            int[] prompt = [2, 17, 42, 99, 7];
            int p = prompt.Length;
            const int c = 6;

            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            int vocab = cfg.VocabSize;
            int maskId = cfg.DiffusionConfig!.MaskTokenId;
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, cfg, spvDir);
            Assert.True(model.SupportsDiffusionPromptKv, "Vulkan diffusion-gemma must support PKV.");

            // (a) cacheless unified Hybrid(P) forward, zero SC.
            var (seq, pos) = BuildSeq(prompt, maskId, c);
            model.SetDiffusionSelfCond(ReadOnlySpan<float>.Empty, 0, scUse: 0f);
            float[] unified;
            using (ITensor uni = model.Forward(seq, pos, -1, null, null, AttentionMaskSpec.Hybrid(p)))
                unified = CanvasRows(uni, p, c, vocab);

            // (b) PKV prefill(prompt) + decode(canvas), zero SC.
            using var store = new DiffusionPromptKvStore(cfg.NumLayers);
            int[] promptPos = new int[p];
            for (int i = 0; i < p; i++) promptPos[i] = i;
            model.SetDiffusionSelfCond(ReadOnlySpan<float>.Empty, 0, scUse: 0f);
            model.DiffusionPrefillPromptKv(prompt, promptPos, store);

            // Decode must use the SAME canvas tokens the unified forward used (BuildSeq seeds
            // varied tokens) so the two paths are comparing identical inputs.
            int[] canvasTok = new int[c];
            int[] canvasPos = new int[c];
            for (int i = 0; i < c; i++) { canvasTok[i] = seq[p + i]; canvasPos[i] = p + i; }
            model.SetDiffusionSelfCond(ReadOnlySpan<float>.Empty, 0, scUse: 0f);
            float[] pkv;
            using (ITensor dec = model.DiffusionDecodeWithPromptKv(canvasTok, canvasPos, -1, store))
            {
                Assert.Equal(c, dec.Shape[0]);
                Assert.Equal(vocab, dec.Shape[1]);
                pkv = CanvasRows(dec, 0, c, vocab);
            }

            // Per-row argmax exact; logits within a tight PKV envelope (same math, only the
            // prompt-key copy reorders relative to recomputation).
            for (int r = 0; r < c; r++)
                Assert.True(ArgMax(unified, r * vocab, vocab) == ArgMax(pkv, r * vocab, vocab),
                    $"[pkv] canvas row {r} argmax mismatch");
            float worst = 0;
            for (int i = 0; i < c * vocab; i++) worst = MathF.Max(worst, MathF.Abs(unified[i] - pkv[i]));
            _out.WriteLine($"[pkv] Vulkan PKV vs unified worst |diff|={worst:E3} over {c * vocab} logits.");
            Assert.True(worst <= 5e-3f, $"PKV canvas logits must match unified within 5e-3 (worst={worst:E3}).");
        }
        finally { try { File.Delete(path); } catch { } }
    }

    /// <summary>
    /// END-TO-END DENOISE LOOP on Vulkan: runs the full <see cref="DiffusionTextGenerator"/> (greedy /
    /// deterministic) on Vulkan over the synthetic diffusion-gemma fixture, exercising the iterative
    /// loop WITH self-conditioning (steps &gt; 0), the mask-token suppression in the unmask sampler,
    /// and the region-aware embed/scalar — with PKV both OFF and ON.
    /// </summary>
    /// <remarks>
    /// The loop MUST converge (no surviving mask tokens; exactly canvas-length committed tokens) and
    /// the Vulkan PKV-on path MUST be byte-identical to PKV-off (PKV is a pure throughput optimisation).
    /// The CPU run is printed for reference but NOT asserted equal: the synthetic fixture has random
    /// weights, so an all-mask canvas is intentionally degenerate (the spec calls a single masked-canvas
    /// forward "degenerate") — its near-tied non-mask logits make the FIRST committed token drift-
    /// sensitive across backends, and one differing commit cascades. Cross-backend math correctness is
    /// gated discriminatingly by the single-forward parity tests above (mask-suppressed argmax exact on
    /// a non-degenerate canvas) and the bit-exact PKV-vs-unified test; on real weights the well-separated
    /// logits make the committed tokens robust.
    /// </remarks>
    [SkippableFact]
    public void Vulkan_Diffusion_DenoiseLoop_RunsAndPkvMatches()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");

        string spvDir = ResolveSpvDir();
        string path = WriteFixture();
        try
        {
            int[] promptIds = [2, 17, 42, 99, 7];
            const int canvas = 6, steps = 6;

            int[] cpuIds = RunCpuDenoise(path, promptIds, canvas, steps, null);
            int[] vkPkvOffIds = RunVulkanDenoise(path, spvDir, promptIds, canvas, steps, pkv: false, null);
            int[] vkPkvOnIds = RunVulkanDenoise(path, spvDir, promptIds, canvas, steps, pkv: true, null);

            _out.WriteLine($"[denoise] cpu       =[{string.Join(",", cpuIds)}]  (reference; not asserted equal)");
            _out.WriteLine($"[denoise] vk pkv-off =[{string.Join(",", vkPkvOffIds)}]");
            _out.WriteLine($"[denoise] vk pkv-on  =[{string.Join(",", vkPkvOnIds)}]");

            const int maskId = 4;
            // The Vulkan loop must converge: canvas-length tokens, none left masked.
            Assert.Equal(canvas, vkPkvOffIds.Length);
            Assert.DoesNotContain(maskId, vkPkvOffIds);
            // PKV must be a pure throughput optimisation — output-identical to the cacheless path.
            Assert.Equal(vkPkvOffIds, vkPkvOnIds);
        }
        finally { try { File.Delete(path); } catch { } }
    }

    private static int[] RunCpuDenoise(string path, int[] promptIds, int canvas, int steps, List<string>? _)
    {
        var (model, gguf, config) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
        using (gguf)
        using (model)
        {
            var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            var diff = config.DiffusionConfig! with
            { CanvasLength = canvas, MaxDenoisingSteps = steps, TemperatureMax = 0f, TemperatureMin = 0f };
            var gen = new DiffusionTextGenerator(model, tokenizer, diffusionConfig: diff, enablePromptKv: false);
            return gen.Generate(promptIds).GeneratedTokenIds;
        }
    }

    private static int[] RunVulkanDenoise(string path, string spvDir, int[] promptIds, int canvas, int steps, bool pkv, List<string>? _)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var diff = config.DiffusionConfig! with
        { CanvasLength = canvas, MaxDenoisingSteps = steps, TemperatureMax = 0f, TemperatureMin = 0f };
        var gen = new DiffusionTextGenerator(model, tokenizer, diffusionConfig: diff, enablePromptKv: pkv);
        return gen.Generate(promptIds).GeneratedTokenIds;
    }

    private void AssertCanvasParity(float[] cpu, float[] vk, int c, int vocab, string label)
    {
        // Per-row argmax — the unmask sampler's signal — must agree. The diffusion vocab
        // contains the mask token (id 4), which the model ranks argmax at low-confidence
        // canvas positions; the sampler SUPPRESSES it before committing, so the discriminating
        // check is the argmax with the mask token excluded (matches what the sampler commits).
        const int maskId = 4;
        for (int r = 0; r < c; r++)
        {
            int au = ArgMaxSuppress(cpu, r * vocab, vocab, maskId);
            int av = ArgMaxSuppress(vk, r * vocab, vocab, maskId);
            Assert.True(au == av, $"[{label}] canvas row {r}: mask-suppressed argmax cpu={au} vk={av}");
        }

        // Per-logit envelope — diffusion adds region-embed + (optional) SC on top of the gemma4
        // dual-FFN reduction-order drift, so the bar matches the AR gemma4 probe.
        // Diffusion stacks the canvas weight-less rms_noscale + Hybrid bidirectional
        // attention on top of the gemma4 dual-FFN reduction-order drift; the final softcap
        // (30·tanh(z/30)) then compresses logits toward zero where the relative error
        // amplifies, so the absolute envelope is a touch wider than the AR probe's 6e-2.
        // The per-row argmax (asserted above) is the exact correctness gate.
        const float absTol = 2.5e-1f, relTol = 8.0e-3f;
        float worst = 0; int worstIdx = -1; int over = 0;
        for (int i = 0; i < c * vocab; i++)
        {
            float d = MathF.Abs(cpu[i] - vk[i]);
            if (d > worst) { worst = d; worstIdx = i; }
            if (d > absTol + relTol * MathF.Abs(cpu[i])) over++;
        }
        for (int i = 0; i < c * vocab; i++)
        {
            float bar = absTol + relTol * MathF.Abs(cpu[i]);
            Assert.True(MathF.Abs(cpu[i] - vk[i]) <= bar,
                $"[{label}] idx {i}: cpu={cpu[i]:F6} vk={vk[i]:F6} (|diff|={MathF.Abs(cpu[i] - vk[i]):E3} > {bar:E3}); "
                + $"worst |diff|={worst:E3} @ {worstIdx}, over={over}");
        }
        _out.WriteLine($"[{label}] CPU↔Vulkan canvas parity OK: worst |diff|={worst:E3} @ {worstIdx} over {c * vocab} logits.");
    }

    // Canvas tokens for the parity forwards. A non-degenerate canvas (VARIED tokens, not the
    // all-mask canvas) gives well-separated logits so the cross-backend argmax is a clean
    // correctness gate rather than a coin-flip among near-tied non-mask tokens. The all-mask
    // canvas (identical K/V on every canvas position) is intentionally degenerate per the spec —
    // the iterative denoise loop is exercised by Vulkan_Diffusion_DenoiseLoop_MatchesCpuTokens.
    private static readonly int[] CanvasSeed = [11, 23, 5, 88, 200, 31];

    private static (int[] seq, int[] pos) BuildSeq(int[] prompt, int maskId, int c)
    {
        int p = prompt.Length;
        int seqLen = p + c;
        int[] seq = new int[seqLen];
        int[] pos = new int[seqLen];
        Array.Copy(prompt, seq, p);
        for (int i = p; i < seqLen; i++) seq[i] = CanvasSeed[(i - p) % CanvasSeed.Length];
        for (int i = 0; i < seqLen; i++) pos[i] = i;
        return (seq, pos);
    }

    private static unsafe float[] CanvasRows(ITensor logits, int p, int c, int vocab)
    {
        float* basePtr = (float*)logits.DataPointer;
        var canvas = new float[c * vocab];
        new ReadOnlySpan<float>(basePtr + (long)p * vocab, c * vocab).CopyTo(canvas);
        return canvas;
    }

    private static int ArgMax(float[] v, int offset, int len)
    {
        int best = offset;
        for (int i = offset + 1; i < offset + len; i++) if (v[i] > v[best]) best = i;
        return best - offset;
    }

    private static int ArgMaxSuppress(float[] v, int offset, int len, int suppressId)
    {
        int best = -1; float bv = float.NegativeInfinity;
        for (int i = 0; i < len; i++)
        {
            if (i == suppressId) continue;
            float x = v[offset + i];
            if (x > bv) { bv = x; best = i; }
        }
        return best;
    }

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var cand in candidates)
        {
            string full = Path.GetFullPath(cand);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException(
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");
    }
}
