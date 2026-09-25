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
/// Equivalence gate for the DiffusionGemma <b>prompt-KV (PKV) prefill/decode</b> optimisation.
/// PKV caches each layer's prompt K/V once per canvas block and reuses them on every denoise
/// step (canvas-only forward over <c>[cached prompt K/V | fresh canvas K/V]</c>) instead of
/// recomputing the whole <c>[prompt | canvas]</c> prefix. It MUST be a pure optimisation: the
/// canvas logits — and therefore the generated token ids — are IDENTICAL to the cacheless
/// unified forward. These tests assert that equivalence on the self-contained synthetic
/// diffusion-gemma fixture (no checkpoint, deterministic, single-thread).
/// </summary>
/// <remarks>
/// The synthetic fixture's sliding window is 8 and the global (last) layer is V-from-K, so a
/// canvas spanning past the window AND a V-less global layer are both exercised — the rectangular
/// PKV mask + the V-from-K cache path are discriminated, not degenerate.
/// </remarks>
public sealed class DiffusionGemmaPromptKvEquivalenceTests
{
    private readonly ITestOutputHelper _out;

    public DiffusionGemmaPromptKvEquivalenceTests(ITestOutputHelper output) => _out = output;

    private static DiffusionConfig DeterministicDiffusion(DiffusionConfig baseCfg, int canvas, int steps) =>
        baseCfg with
        {
            CanvasLength = canvas,
            MaxDenoisingSteps = steps,
            TemperatureMax = 0.0f,   // greedy → deterministic given identical logits
            TemperatureMin = 0.0f,
        };

    /// <summary>
    /// SINGLE-STEP LOGIT EQUIVALENCE: one zero-self-conditioning forward over
    /// <c>[prompt | all-mask canvas]</c>. Compare the canvas-region logits produced by
    /// (a) the unified cacheless Hybrid(P) forward and (b) a PKV prefill(prompt)+decode(canvas).
    /// They must match to within a tight tolerance (ideally exact). This isolates the PKV
    /// attention rewrite from the multi-step denoise loop.
    /// </summary>
    [Fact]
    public unsafe void PkvDecode_SingleStep_CanvasLogitsMatchUnified()
    {
        string path = Path.Combine(Path.GetTempPath(), $"syn_diffgemma_pkv1_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteDiffusionGemma(path);
        try
        {
            var (model, gguf, config) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
            using var _ = gguf;
            using var __ = model;
            Assert.Equal(Architecture.DiffusionGemma, config.Architecture);
            Assert.True(model.SupportsDiffusionPromptKv, "Synthetic diffusion-gemma must support PKV.");

            int maskId = config.DiffusionConfig!.MaskTokenId;
            int vocab = config.VocabSize;
            int[] prompt = [2, 17, 42, 99, 7];     // BOS + arbitrary in-range ids
            int p = prompt.Length;
            const int c = 6;                        // canvas length (P+C = 11 > sliding window 8)
            int seqLen = p + c;

            // Unified [prompt | canvas] tokens & positions.
            int[] seq = new int[seqLen];
            int[] pos = new int[seqLen];
            Array.Copy(prompt, seq, p);
            for (int i = p; i < seqLen; i++) seq[i] = maskId;
            for (int i = 0; i < seqLen; i++) pos[i] = i;

            // (a) unified forward, zero SC.
            model.SetDiffusionSelfCond(ReadOnlySpan<float>.Empty, 0, scUse: 0f);
            float[] unifiedCanvas = new float[c * vocab];
            using (ITensor uni = model.Forward(seq, pos, deviceId: -1, kvCache: null, adapter: null,
                       AttentionMaskSpec.Hybrid(p)))
            {
                float* up = (float*)uni.DataPointer;
                new ReadOnlySpan<float>(up + (long)p * vocab, c * vocab).CopyTo(unifiedCanvas);
            }

            // (b) PKV prefill(prompt) + decode(canvas), zero SC.
            using var store = new DiffusionPromptKvStore(config.NumLayers);
            model.SetDiffusionSelfCond(ReadOnlySpan<float>.Empty, 0, scUse: 0f);
            int[] promptPos = new int[p];
            for (int i = 0; i < p; i++) promptPos[i] = i;
            model.DiffusionPrefillPromptKv(prompt, promptPos, store);

            int[] canvasTok = new int[c];
            int[] canvasPos = new int[c];
            for (int i = 0; i < c; i++) { canvasTok[i] = maskId; canvasPos[i] = p + i; }
            model.SetDiffusionSelfCond(ReadOnlySpan<float>.Empty, 0, scUse: 0f);
            float[] pkvCanvas = new float[c * vocab];
            using (ITensor dec = model.DiffusionDecodeWithPromptKv(canvasTok, canvasPos, deviceId: -1, store))
            {
                Assert.Equal(c, dec.Shape[0]);
                Assert.Equal(vocab, dec.Shape[1]);
                new ReadOnlySpan<float>((float*)dec.DataPointer, c * vocab).CopyTo(pkvCanvas);
            }

            // Max absolute delta over every canvas logit.
            float maxDelta = 0f;
            int exact = 0;
            for (int i = 0; i < c * vocab; i++)
            {
                float d = MathF.Abs(unifiedCanvas[i] - pkvCanvas[i]);
                if (d > maxDelta) maxDelta = d;
                if (d == 0f) exact++;
            }
            _out.WriteLine($"[pkv 1-step] P={p} C={c} vocab={vocab} maxDelta={maxDelta:E3} exactBits={exact}/{c * vocab}");

            // Per-row argmax must be identical (the decode signal the sampler consumes).
            for (int r = 0; r < c; r++)
            {
                int au = ArgMax(unifiedCanvas, r * vocab, vocab);
                int ap = ArgMax(pkvCanvas, r * vocab, vocab);
                Assert.Equal(au, ap);
            }

            // Tight tolerance — the task allows <= 1e-4 for any benign reorder. We typically see
            // exact or near-exact (the only reorder is the prompt-key copy, not the math).
            Assert.True(maxDelta <= 1e-4f,
                $"PKV canvas logits must match unified within 1e-4 (maxDelta={maxDelta:E3}).");
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// END-TO-END EQUIVALENCE: run the full denoise loop with PKV OFF and PKV ON over the same
    /// prompt/canvas/config (greedy, single-thread) and assert the generated token ids are
    /// byte-identical. PKV must not change output — it is a throughput optimisation only.
    /// </summary>
    [Fact]
    public void PkvDecode_FullGeneration_IdenticalIdsToUnified()
    {
        string path = Path.Combine(Path.GetTempPath(), $"syn_diffgemma_pkv2_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteDiffusionGemma(path);
        try
        {
            int[] promptIds = [2, 17, 42, 99, 123, 7];

            int[] offIds = RunGenerate(path, promptIds, pkv: false, out double offMs);
            int[] onIds = RunGenerate(path, promptIds, pkv: true, out double onMs);

            _out.WriteLine($"[pkv e2e] off ids=[{string.Join(",", offIds)}]  ({offMs:F1} ms)");
            _out.WriteLine($"[pkv e2e] on  ids=[{string.Join(",", onIds)}]  ({onMs:F1} ms)");
            _out.WriteLine($"[pkv e2e] speedup x{(onMs > 0 ? offMs / onMs : 0):F2}");

            Assert.Equal(offIds, onIds);   // PKV must be output-identical to the unified path
            Assert.NotEmpty(offIds);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// MULTI-CANVAS (block-AR) EQUIVALENCE: targetLength > canvas forces a second canvas block
    /// whose prompt prefix includes the first finished canvas. PKV must still match the unified
    /// path id-for-id across both blocks (prefill re-runs per block over the grown prefix).
    /// </summary>
    [Fact]
    public void PkvDecode_MultiCanvas_IdenticalIdsToUnified()
    {
        string path = Path.Combine(Path.GetTempPath(), $"syn_diffgemma_pkv3_{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteDiffusionGemma(path);
        try
        {
            int[] promptIds = [2, 31, 58, 7];
            // canvas 6, target 14 ⇒ 3 blocks (6 + 6 + 2).
            int[] offIds = RunGenerate(path, promptIds, pkv: false, out _, canvas: 6, steps: 6, target: 14);
            int[] onIds = RunGenerate(path, promptIds, pkv: true, out _, canvas: 6, steps: 6, target: 14);
            _out.WriteLine($"[pkv multi] off=[{string.Join(",", offIds)}]");
            _out.WriteLine($"[pkv multi] on =[{string.Join(",", onIds)}]");
            Assert.Equal(offIds, onIds);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    private int[] RunGenerate(string path, int[] promptIds, bool pkv, out double ms,
        int canvas = 8, int steps = 6, int? target = null)
    {
        var (model, gguf, config) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
        using var _ = gguf;
        using var __ = model;
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var diff = DeterministicDiffusion(config.DiffusionConfig!, canvas, steps);
        var gen = new DiffusionTextGenerator(model, tokenizer, diffusionConfig: diff, enablePromptKv: pkv);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        DiffusionResult r = gen.Generate(promptIds, target);
        sw.Stop();
        ms = sw.Elapsed.TotalMilliseconds;
        return r.GeneratedTokenIds;
    }

    private static int ArgMax(float[] buf, int offset, int n)
    {
        int best = 0; float bv = float.NegativeInfinity;
        for (int v = 0; v < n; v++) { float x = buf[offset + v]; if (x > bv) { bv = x; best = v; } }
        return best;
    }
}
