using DotLLM.Core.Attention;
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

    private void AssertCanvasParity(float[] cpu, float[] vk, int c, int vocab, string label)
    {
        // Per-row argmax — the unmask sampler's signal — must agree exactly.
        for (int r = 0; r < c; r++)
        {
            int au = ArgMax(cpu, r * vocab, vocab);
            int av = ArgMax(vk, r * vocab, vocab);
            Assert.True(au == av, $"[{label}] canvas row {r}: argmax cpu={au} vk={av}");
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

    private static (int[] seq, int[] pos) BuildSeq(int[] prompt, int maskId, int c)
    {
        int p = prompt.Length;
        int seqLen = p + c;
        int[] seq = new int[seqLen];
        int[] pos = new int[seqLen];
        Array.Copy(prompt, seq, p);
        for (int i = p; i < seqLen; i++) seq[i] = maskId;
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
