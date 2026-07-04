using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Backends;

/// <summary>
/// Vulkan coverage for issue #121 over the synthetic <c>diffusion-gemma</c> fixture (vocab 256):
/// <list type="number">
/// <item><b>Sparse SC parity</b> — with <see cref="DiffusionConfig.SelfCondTopK"/> forced into
/// the genuinely-sparse range (K = 32 &lt; vocab) on BOTH backends, the Vulkan canvas logits
/// must track the CPU oracle exactly as the dense SC parity test does (both backends consume
/// the SHARED <c>SelfCondSoftEmbed</c> helper, so any drift is a wiring bug);</item>
/// <item><b>Chunked LM head</b> — the row-chunked diffusion head
/// (<see cref="VulkanTransformerModel.DiffusionHeadChunkRows"/>, the TDR mitigation) must
/// reproduce the monolithic dispatch: chunking is row-partitioned scheduling only, per-row
/// math is untouched;</item>
/// <item><b>End-to-end</b> — the denoise loop converges with sparse SC + a forced-small head
/// chunk (the exact configuration the real-26B canvas-256 run needs).</item>
/// </list>
/// Style mirrors <see cref="Gemma4DiffusionVulkanTests"/>; self-skips without a Vulkan device
/// or under <c>DOTLLM_SKIP_VULKAN=1</c>.
/// </summary>
[Trait("Category", "GPU")]
public sealed class Gemma4DiffusionScTopKVulkanTests
{
    private readonly ITestOutputHelper _out;

    public Gemma4DiffusionScTopKVulkanTests(ITestOutputHelper output) => _out = output;

    private const int MaskId = 4; // SyntheticGemma4Config default mask token id.

    private static string WriteFixture()
    {
        string path = Path.Combine(Path.GetTempPath(), $"syn_diffgemma_vk_sctopk_{Guid.NewGuid():N}.gguf");
        return SyntheticGemma4Gguf.WriteDiffusionGemma(path);
    }

    /// <summary>
    /// SPARSE self-conditioning CPU↔Vulkan parity (K = 32 &lt; vocab 256, peaked prev logits).
    /// Discriminates the sparse SC path specifically: with K in the sparse range both backends
    /// run top-K selection + renormalised softmax + gathered accumulation, and must agree to
    /// the same envelope as the dense SC parity test (the SC signal is computed by the shared
    /// helper on both sides; the remaining drift is the usual GPU reduction-order envelope of
    /// the transformer stack itself).
    /// </summary>
    [SkippableFact]
    public unsafe void Vulkan_Diffusion_SparseSelfCond_MatchesCpu()
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
            const int topK = 32;

            int vocab, maskId;
            ModelConfig cfgSparse;
            {
                using var gguf0 = GgufFile.Open(path);
                var cfg0 = GgufModelConfigExtractor.Extract(gguf0.Metadata);
                vocab = cfg0.VocabSize;
                maskId = cfg0.DiffusionConfig!.MaskTokenId;
                cfgSparse = cfg0 with { DiffusionConfig = cfg0.DiffusionConfig with { SelfCondTopK = topK } };
            }
            Assert.True(topK < vocab, "fixture must make K genuinely sparse");

            // Deterministic PEAKED "previous step" canvas logits [c × vocab] — the regime a
            // trained checkpoint produces and the one where sparse SC is representative.
            var prevLogits = new float[c * vocab];
            var rng = new Random(1234);
            for (int i = 0; i < prevLogits.Length; i++) prevLogits[i] = (float)(rng.NextDouble() - 0.5);
            for (int col = 0; col < c; col++)
                for (int peak = 0; peak < 3; peak++)
                    prevLogits[col * vocab + rng.Next(vocab)] += 20f;

            // ── CPU oracle (sparse SC ON) ─────────────────────────────────
            float[] cpuCanvas;
            {
                using var gguf = GgufFile.Open(path);
                using var model = DotLLM.Models.Architectures.TransformerModel.LoadFromGguf(
                    gguf, cfgSparse, ThreadingConfig.SingleThreaded);
                var (seq, pos) = BuildSeq(prompt, maskId, c);
                model.SetDiffusionSelfCond(prevLogits, c, scUse: 1f);
                using ITensor logits = model.Forward(seq, pos, -1, null, null, AttentionMaskSpec.Hybrid(p));
                cpuCanvas = CanvasRows(logits, p, c, vocab);
            }

            // ── Vulkan under test (sparse SC ON) ──────────────────────────
            float[] vkCanvas;
            {
                using var gguf = GgufFile.Open(path);
                using var model = VulkanTransformerModel.LoadFromGguf(gguf, cfgSparse, spvDir);
                var (seq, pos) = BuildSeq(prompt, maskId, c);
                model.SetDiffusionSelfCond(prevLogits, c, scUse: 1f);
                using ITensor logits = model.Forward(seq, pos, -1, null, null, AttentionMaskSpec.Hybrid(p));
                Assert.Equal(p + c, logits.Shape[0]);
                Assert.Equal(vocab, logits.Shape[1]);
                vkCanvas = CanvasRows(logits, p, c, vocab);
            }

            AssertCanvasParity(cpuCanvas, vkCanvas, c, vocab, "sparse-sc");
        }
        finally { try { File.Delete(path); } catch { } }
    }

    /// <summary>
    /// Row-chunked diffusion LM head vs the monolithic dispatch (zero-SC forward, chunk rows
    /// forced to 2 so the 11-row fixture forward runs 6 chunks). Chunking only partitions the
    /// output rows across dispatches — each row's reduction is computed by the same kernel
    /// over the same inputs — so the full [seqLen × vocab] logits are expected bit-identical;
    /// asserted with per-row exact argmax and a near-zero envelope for driver headroom.
    /// </summary>
    [SkippableFact]
    public unsafe void Vulkan_Diffusion_ChunkedHead_MatchesMonolithic()
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

            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            int vocab = cfg.VocabSize;
            int maskId = cfg.DiffusionConfig!.MaskTokenId;
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, cfg, spvDir);
            var (seq, pos) = BuildSeq(prompt, maskId, c);

            float[] Run(int chunkRows)
            {
                model.DiffusionHeadChunkRows = chunkRows;
                model.SetDiffusionSelfCond(ReadOnlySpan<float>.Empty, 0, scUse: 0f);
                using ITensor logits = model.Forward(seq, pos, -1, null, null, AttentionMaskSpec.Hybrid(p));
                Assert.Equal(seqLen, logits.Shape[0]);
                Assert.Equal(vocab, logits.Shape[1]);
                var all = new float[seqLen * vocab];
                new ReadOnlySpan<float>((float*)logits.DataPointer, all.Length).CopyTo(all);
                return all;
            }

            float[] mono = Run(chunkRows: 0);   // <= 0 ⇒ chunking disabled (monolithic dispatch)
            float[] chunked = Run(chunkRows: 2); // 6 chunks over 11 rows

            float worst = 0f;
            for (int i = 0; i < mono.Length; i++)
                worst = MathF.Max(worst, MathF.Abs(mono[i] - chunked[i]));
            for (int r = 0; r < seqLen; r++)
            {
                int am = ArgMax(mono, r * vocab, vocab);
                int ac = ArgMax(chunked, r * vocab, vocab);
                Assert.True(am == ac, $"row {r}: monolithic argmax {am} != chunked argmax {ac}");
            }
            _out.WriteLine($"[head-chunk] monolithic vs chunked worst |diff|={worst:E3} over {mono.Length} logits.");
            // Same kernel, same per-row reduction order ⇒ expected bit-identical; the tiny
            // envelope only tolerates drivers that reorder identical dispatches internally.
            Assert.True(worst <= 1e-5f, $"chunked head must reproduce the monolithic head (worst |diff|={worst:E3}).");
        }
        finally { try { File.Delete(path); } catch { } }
    }

    /// <summary>
    /// END-TO-END: sparse SC (K = 32) + forced-small head chunk (2 rows) through the full
    /// <see cref="DiffusionTextGenerator"/> denoise loop on Vulkan — the exact shape of the
    /// real-26B canvas-256 configuration this issue unblocks. Must converge (canvas-length
    /// committed tokens, no surviving masks).
    /// </summary>
    [SkippableFact]
    public void Vulkan_Diffusion_SparseScChunkedHead_DenoiseLoopConverges()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");

        string spvDir = ResolveSpvDir();
        string path = WriteFixture();
        try
        {
            int[] promptIds = [2, 17, 42, 99, 7];
            const int canvas = 6, steps = 6;

            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            var cfgSparse = config with
            { DiffusionConfig = config.DiffusionConfig! with { SelfCondTopK = 32 } };
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, cfgSparse, spvDir);
            model.DiffusionHeadChunkRows = 2;
            var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            var diff = cfgSparse.DiffusionConfig! with
            { CanvasLength = canvas, MaxDenoisingSteps = steps, TemperatureMax = 0f, TemperatureMin = 0f };
            var gen = new DiffusionTextGenerator(model, tokenizer, diffusionConfig: diff);
            int[] ids = gen.Generate(promptIds).GeneratedTokenIds;

            _out.WriteLine($"[sparse+chunk] vk =[{string.Join(",", ids)}]");
            Assert.Equal(canvas, ids.Length);
            Assert.DoesNotContain(MaskId, ids);
        }
        finally { try { File.Delete(path); } catch { } }
    }

    // ── helpers (mirroring Gemma4DiffusionVulkanTests) ──────────────────────

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

    private void AssertCanvasParity(float[] cpu, float[] vk, int c, int vocab, string label)
    {
        // Per-row mask-suppressed argmax — the signal the unmask sampler commits — exact.
        for (int r = 0; r < c; r++)
        {
            int au = ArgMaxSuppress(cpu, r * vocab, vocab, MaskId);
            int av = ArgMaxSuppress(vk, r * vocab, vocab, MaskId);
            Assert.True(au == av, $"[{label}] canvas row {r}: mask-suppressed argmax cpu={au} vk={av}");
        }

        // Same envelope as the dense SC parity test (Gemma4DiffusionVulkanTests): the SC
        // signal itself is host-computed by the shared helper on both sides, so the drift
        // budget is the transformer stack's usual GPU reduction-order envelope.
        const float absTol = 2.5e-1f, relTol = 8.0e-3f;
        float worst = 0; int worstIdx = -1;
        for (int i = 0; i < c * vocab; i++)
        {
            float d = MathF.Abs(cpu[i] - vk[i]);
            if (d > worst) { worst = d; worstIdx = i; }
            Assert.True(d <= absTol + relTol * MathF.Abs(cpu[i]),
                $"[{label}] idx {i}: cpu={cpu[i]:F6} vk={vk[i]:F6} |diff|={d:E3}");
        }
        _out.WriteLine($"[{label}] CPU↔Vulkan canvas parity OK: worst |diff|={worst:E3} @ {worstIdx} over {c * vocab} logits.");
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
