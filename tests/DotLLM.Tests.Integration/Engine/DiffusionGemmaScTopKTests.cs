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
/// CPU-only coverage for the top-K sparsified self-conditioning soft-embed (issue #121) over
/// the synthetic <c>diffusion-gemma</c> fixture (vocab 256):
/// <list type="number">
/// <item>dense (K &lt;= 0) vs sparse (K &lt; vocab) SC single-forward comparison — canvas
/// argmax agreement high, logits within a tight envelope when the previous-step distribution
/// is peaked (the regime real checkpoints live in);</item>
/// <item>K &gt;= vocab is BYTE-identical to dense (forward logits and full generation);</item>
/// <item>the sparse end-to-end denoise loop converges (no surviving mask tokens).</item>
/// </list>
/// The CPU dense path is the exact GEMMA4-GRAPH-SPEC reference and the cross-backend oracle;
/// K is plumbed via <see cref="DiffusionConfig.SelfCondTopK"/> (env <c>DOTLLM_DG_SC_TOPK</c>
/// overrides at runtime — not used here so the tests stay parallel-safe).
/// </summary>
public sealed class DiffusionGemmaScTopKTests
{
    private readonly ITestOutputHelper _out;

    public DiffusionGemmaScTopKTests(ITestOutputHelper output) => _out = output;

    private const int MaskId = 4; // SyntheticGemma4Config default mask token id.

    private static string WriteFixture()
    {
        string path = Path.Combine(Path.GetTempPath(), $"syn_diffgemma_sctopk_{Guid.NewGuid():N}.gguf");
        return SyntheticGemma4Gguf.WriteDiffusionGemma(path);
    }

    /// <summary>Loads the fixture as a CPU model with <see cref="DiffusionConfig.SelfCondTopK"/> = <paramref name="topK"/>.</summary>
    private static (IModel Model, GgufFile Gguf, ModelConfig Config) LoadWithTopK(string path, int topK)
    {
        DiffusionConfig baseDiff;
        using (var probe = GgufFile.Open(path))
            baseDiff = GgufModelConfigExtractor.Extract(probe.Metadata).DiffusionConfig!;
        return ModelLoader.LoadFromGguf(
            path, ThreadingConfig.SingleThreaded, baseDiff with { SelfCondTopK = topK });
    }

    private static (int[] seq, int[] pos) BuildSeq(int[] prompt, int c)
    {
        // Varied (non-degenerate) canvas tokens — same rationale as the Vulkan parity tests.
        int[] canvasSeed = [11, 23, 5, 88, 200, 31];
        int p = prompt.Length;
        int[] seq = new int[p + c];
        int[] pos = new int[p + c];
        Array.Copy(prompt, seq, p);
        for (int i = p; i < seq.Length; i++) seq[i] = canvasSeed[(i - p) % canvasSeed.Length];
        for (int i = 0; i < pos.Length; i++) pos[i] = i;
        return (seq, pos);
    }

    private static unsafe float[] CanvasRows(ITensor logits, int p, int c, int vocab)
    {
        var canvas = new float[c * vocab];
        new ReadOnlySpan<float>((float*)logits.DataPointer + (long)p * vocab, c * vocab).CopyTo(canvas);
        return canvas;
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

    /// <summary>
    /// Runs one SC forward (denoise step &gt; 0 semantics) and returns the canvas-region logits.
    /// </summary>
    private static float[] RunScForward(IModel model, int[] prompt, int c, float[] prevLogits, int vocab)
    {
        var (seq, pos) = BuildSeq(prompt, c);
        model.SetDiffusionSelfCond(prevLogits, c, scUse: 1f);
        using ITensor logits = model.Forward(seq, pos, -1, null, null, AttentionMaskSpec.Hybrid(prompt.Length));
        return CanvasRows(logits, prompt.Length, c, vocab);
    }

    /// <summary>
    /// PEAKED previous-step distribution (a few dominant tokens per canvas column, +20 logits ⇒
    /// residual mass outside the peaks &lt; 1e-7): sparse K = 32 must reproduce the dense
    /// canvas logits to a tight envelope and the mask-suppressed per-row argmax — the signal
    /// the unmask sampler commits — must agree on (nearly) every row. Repeated over three
    /// deterministic draws.
    /// </summary>
    [Fact]
    public void SparseSc_PeakedPrevLogits_CanvasArgmaxAgreesWithDense()
    {
        string path = WriteFixture();
        try
        {
            int[] prompt = [2, 17, 42, 99, 7];
            const int c = 6;

            var (denseModel, denseGguf, cfg) = LoadWithTopK(path, topK: 0);
            var (sparseModel, sparseGguf, _) = LoadWithTopK(path, topK: 32);
            int vocab = cfg.VocabSize;
            using (denseGguf)
            using (denseModel)
            using (sparseGguf)
            using (sparseModel)
            {
                int agree = 0, total = 0;
                float worst = 0f;
                for (int draw = 0; draw < 3; draw++)
                {
                    var rng = new Random(1000 + draw);
                    var prev = new float[c * vocab];
                    for (int i = 0; i < prev.Length; i++) prev[i] = (float)(rng.NextDouble() - 0.5);
                    for (int col = 0; col < c; col++)
                        for (int peak = 0; peak < 3; peak++)
                            prev[col * vocab + rng.Next(vocab)] += 20f;

                    float[] dense = RunScForward(denseModel, prompt, c, prev, vocab);
                    float[] sparse = RunScForward(sparseModel, prompt, c, prev, vocab);

                    for (int r = 0; r < c; r++, total++)
                        if (ArgMaxSuppress(dense, r * vocab, vocab, MaskId)
                            == ArgMaxSuppress(sparse, r * vocab, vocab, MaskId))
                            agree++;
                    for (int i = 0; i < dense.Length; i++)
                        worst = MathF.Max(worst, MathF.Abs(dense[i] - sparse[i]));
                }

                _out.WriteLine($"[sc-topk] peaked-prev argmax agreement {agree}/{total}, worst |diff|={worst:E3}.");
                // Sparse truncation on a peaked distribution perturbs the SC signal by < 1e-6
                // relative mass; allow at most one near-tie flip across all draws.
                Assert.True(agree >= total - 1,
                    $"mask-suppressed canvas argmax agreement too low: {agree}/{total}.");
                Assert.True(worst <= 5e-2f,
                    $"dense vs sparse canvas logits diverged (worst |diff|={worst:E3} > 5e-2).");
            }
        }
        finally { try { File.Delete(path); } catch { } }
    }

    /// <summary>
    /// K &gt;= vocab routes to the dense reference path — the canvas logits must be
    /// BYTE-identical to the K = 0 dense run even on a SPREAD previous-step distribution
    /// (the case where genuine sparsification would deviate, discriminating the routing).
    /// </summary>
    [Fact]
    public void KAtLeastVocab_ForwardByteIdenticalToDense()
    {
        string path = WriteFixture();
        try
        {
            int[] prompt = [2, 17, 42, 99, 7];
            const int c = 6;

            var (denseModel, denseGguf, cfg) = LoadWithTopK(path, topK: 0);
            int vocab = cfg.VocabSize;
            var (kModel, kGguf, _) = LoadWithTopK(path, topK: vocab);
            using (denseGguf)
            using (denseModel)
            using (kGguf)
            using (kModel)
            {
                var rng = new Random(4321);
                var prev = new float[c * vocab];
                for (int i = 0; i < prev.Length; i++) prev[i] = (float)(rng.NextDouble() * 6.0 - 3.0);

                float[] dense = RunScForward(denseModel, prompt, c, prev, vocab);
                float[] routed = RunScForward(kModel, prompt, c, prev, vocab);

                for (int i = 0; i < dense.Length; i++)
                    Assert.True(
                        BitConverter.SingleToInt32Bits(dense[i]) == BitConverter.SingleToInt32Bits(routed[i]),
                        $"idx {i}: dense={dense[i]:R} K>=vocab={routed[i]:R} — must be byte-identical.");
                _out.WriteLine($"[sc-topk] K>=vocab byte-identical over {dense.Length} canvas logits.");
            }
        }
        finally { try { File.Delete(path); } catch { } }
    }

    /// <summary>
    /// End-to-end denoise loop, dense (K = 0) vs sparse (K = 32) vs K = vocab, greedy /
    /// deterministic: the K = vocab run must be byte-identical to dense over the whole
    /// trajectory (identical committed ids, step for step); the sparse run must converge
    /// (canvas-length tokens, no surviving masks). Per-step canvases are logged; the step-0
    /// snapshots (zero-SC — sparsification inert) must match across all three runs.
    /// </summary>
    [Fact]
    public void Generation_DenseVsSparse_ConvergesAndKVocabIsByteIdentical()
    {
        string path = WriteFixture();
        try
        {
            int[] prompt = [2, 17, 42, 99, 7];
            const int canvas = 6, steps = 6;

            (int[] ids, List<int[]> stepCanvases) Run(int topK)
            {
                var (model, gguf, config) = LoadWithTopK(path, topK);
                using (gguf)
                using (model)
                {
                    var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
                    var diff = config.DiffusionConfig! with
                    {
                        CanvasLength = canvas,
                        MaxDenoisingSteps = steps,
                        TemperatureMax = 0f,
                        TemperatureMin = 0f,
                    };
                    var gen = new DiffusionTextGenerator(model, tokenizer, diffusionConfig: diff);
                    var snapshots = new List<int[]>();
                    var result = gen.Generate(prompt, onCanvasStep: s => { if (!s.Completed) snapshots.Add(s.Canvas); });
                    return (result.GeneratedTokenIds, snapshots);
                }
            }

            var (denseIds, denseSteps) = Run(topK: 0);
            var (sparseIds, sparseSteps) = Run(topK: 32);
            var (kVocabIds, kVocabSteps) = Run(topK: 256);

            _out.WriteLine($"[gen] dense  =[{string.Join(",", denseIds)}]");
            _out.WriteLine($"[gen] sparse =[{string.Join(",", sparseIds)}]");
            _out.WriteLine($"[gen] k=vocab=[{string.Join(",", kVocabIds)}]");

            // K >= vocab ⇒ dense routing ⇒ the ENTIRE trajectory is byte-identical.
            Assert.Equal(denseIds, kVocabIds);
            Assert.Equal(denseSteps.Count, kVocabSteps.Count);
            for (int s = 0; s < denseSteps.Count; s++)
                Assert.Equal(denseSteps[s], kVocabSteps[s]);

            // Step 0 is zero-SC on every run — sparsification cannot touch it.
            Assert.True(denseSteps.Count > 0 && sparseSteps.Count > 0, "runs must emit step snapshots.");
            Assert.Equal(denseSteps[0], sparseSteps[0]);

            // The sparse loop must converge like the dense one.
            Assert.Equal(canvas, sparseIds.Length);
            Assert.DoesNotContain(MaskId, sparseIds);

            // Per-step canvas agreement (diagnostic bound: the synthetic fixture's random
            // weights make post-divergence steps drift-sensitive; the discriminating gate is
            // the peaked-forward argmax test above).
            int common = Math.Min(denseSteps.Count, sparseSteps.Count);
            int same = 0, cells = 0;
            for (int s = 0; s < common; s++)
                for (int i = 0; i < canvas; i++, cells++)
                    if (denseSteps[s][i] == sparseSteps[s][i]) same++;
            _out.WriteLine($"[gen] dense↔sparse per-step canvas agreement {same}/{cells}.");
            Assert.True(same >= cells / 2,
                $"dense vs sparse per-step canvas agreement collapsed: {same}/{cells}.");
        }
        finally { try { File.Delete(path); } catch { } }
    }
}
