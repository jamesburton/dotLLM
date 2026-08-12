using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// End-to-end integration checks for the split-KV (Flash-Decoding) decode path
/// wired into <see cref="VulkanTransformerModel"/> (issues #345/#346), decoding a
/// real model with the split path ENABLED (the shipping default) vs forced OFF
/// (<c>DOTLLM_VULKAN_DISABLE_SPLIT_DECODE=1</c>).
/// </summary>
/// <remarks>
/// <para>
/// <b>Engagement threshold (issue #331).</b> The split path is <i>not</i> a
/// "long context only" path. <see cref="VulkanSplitKvAttentionKernel.ComputeSplits"/>
/// takes <c>S = min(TargetWorkgroups / numHeads, ceil(seqKv / MinKvPerSplit))</c>
/// and the caller routes to the split kernel when <c>S &gt;= 2</c>. With the
/// shipping defaults (<c>TargetWorkgroups = 256</c>, and <c>MinKvPerSplit = 16</c>
/// since issue #143 lowered it from 256) that is true from <b>seqKv &gt;= 17</b>
/// for any model with <c>numHeads &lt;= 128</c> — i.e. essentially every decode
/// step of every real model, not only past 256. #345's original "short context is
/// bit-identical" argument rested on the old 256 floor and no longer describes the
/// shipping path; see <see cref="EngagementThreshold_IsSeventeen_NotTwoFiftySix"/>,
/// which pins the real threshold executably.
/// </para>
/// <para>
/// <b>Why that needs a #222-class guard.</b> The split kernel differs from the
/// per-token kernel by cross-split online-softmax reassociation (plus a coalesced
/// subgroup Q·K reduction and shifted KV tile alignment), so it is
/// divergence-<i>capable</i> in exactly the class CUDA's #183/#222 investigation
/// characterised: a tiny logit perturbation at a near-tie flips argmax and then
/// compounds through the sampled-token feedback loop. Exact-token equality over a
/// long generation is therefore not a sustainable invariant for <i>any</i>
/// reassociated attention kernel, and this project's precedent (#222 vs the
/// MMA-decode GQA-split kernel) is to decide on the paired <b>perplexity delta</b>:
/// +0.30% kept CUDA's #183 opt-in, −0.173% justified flipping the MMA kernel
/// default-ON. So the load-bearing assertions here are (i) bit-exactness below the
/// engagement threshold and (ii) a bound on the post-engagement decode-perplexity
/// ratio; the first token divergence is reported diagnostically with its top1/top2
/// margins rather than asserted.
/// </para>
/// <para>
/// The kernel itself is validated against the CPU oracle in the unit tests
/// (<c>VulkanSplitKvAttentionKernelTests</c>). These tests cover what those cannot:
/// that the kernel is actually <i>routed</i> at decode, that the per-(head,split)
/// scratch is reused correctly across the model's layers and across a long
/// generation, and that the inter-pass / inter-layer barriers are right — a
/// scratch-corruption or barrier bug shows up here as gross divergence or NaN.
/// </para>
/// <para>
/// Self-skips when the parity GGUF is absent (env override
/// <c>DOTLLM_SPLIT_PARITY_GGUF</c> or the conventional shared-cache path).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class VulkanSplitDecodeParityTests
{
    private const int Context = 384;     // long-context routing guard (S = 10 on a 24-head model)
    private const int DecodeSteps = 24;
    // Vulkan-vs-Vulkan: only the attention reduction order differs, so drift is
    // small, but it accumulates through ~28 layers + lm_head on a 3B model whose
    // raw logits reach ~±20. The exact-token assertion is the load-bearing check;
    // this guards against gross corruption / NaN (which would be many units off).
    private const float LogitsAbsTol = 0.5f;

    // ── Deep (#222-class) run parameters ─────────────────────────────────────
    // Short prompt so the generation crosses the real engagement threshold (~17)
    // inside the run rather than starting above it.
    private const int DeepPromptLen = 8;
    private const int DeepGenSteps = 512;     // final decode depth ~520
    private const int DeepPplTokens = 1024;   // teacher-forced decode-mode PPL corpus length
    // Post-engagement decode-perplexity ratio bound (on == split-KV default).
    // Sized against this project's own precedent: CUDA's #183 was rejected at
    // +0.30%, the MMA-decode GQA-split kernel accepted at −0.173%. Anything inside
    // ±0.10% is well below both and is reduction-order noise, not a quality change.
    private const double PplRatioTol = 0.0010;

    // Deep-context arm: the depth range #345's 2.1x/2.8x win was measured at.
    private const int DeepContextFill = 3840;
    private const int DeepContextScored = 256;   // scored steps run at depth 3840-4096

    private readonly ITestOutputHelper _output;
    public VulkanSplitDecodeParityTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Issue #331: pins the <i>actual</i> engagement threshold of the shipping
    /// heuristic so the stale "bit-identical below 256" claim cannot silently come
    /// back. Pure arithmetic on <see cref="VulkanSplitKvAttentionKernel.ComputeSplits"/>
    /// — no GPU required.
    /// </summary>
    [Fact]
    public void EngagementThreshold_IsSeventeen_NotTwoFiftySix()
    {
        // Skip when the env overrides are set — the thresholds below are the
        // shipping defaults (TargetWorkgroups=256, MinKvPerSplit=16).
        if (Environment.GetEnvironmentVariable(VulkanSplitKvAttentionKernel.TargetWorkgroupsEnvVar) is not null
            || Environment.GetEnvironmentVariable(VulkanSplitKvAttentionKernel.MinKvPerSplitEnvVar) is not null)
            return;

        foreach (int numHeads in new[] { 9, 16, 24, 32, 64, 128 })
        {
            Assert.False(VulkanSplitKvAttentionKernel.WouldSplit(16, numHeads),
                $"seqKv=16 must NOT split (numHeads={numHeads}).");
            Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(17, numHeads),
                $"seqKv=17 MUST split (numHeads={numHeads}) — the effective threshold is 17, not 256.");
            Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(255, numHeads),
                $"seqKv=255 must split (numHeads={numHeads}) — well below the stale 256 claim.");
        }

        // Above 128 heads the occupancy term alone drops S to 1, so the split path
        // never engages regardless of context. Documented, not a regression.
        Assert.False(VulkanSplitKvAttentionKernel.WouldSplit(4096, 256));
    }

    /// <summary>
    /// Long-context routing guard: prefill to <see cref="Context"/>, decode
    /// <see cref="DecodeSteps"/> steps, assert exact greedy tokens and near-identical
    /// logits. Cheap; catches gross scratch/barrier corruption in the multi-split
    /// (S = 10) regime the perf win was measured in.
    /// </summary>
    [SkippableFact]
    public void SplitDecode_MatchesPerTokenKernel_LongContext()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available.");

        string? modelPath = ResolveModelPath();
        Skip.If(modelPath is null, "Parity model GGUF not found (set DOTLLM_SPLIT_PARITY_GGUF).");

        // Sanity: this shape must actually split, or the test proves nothing.
        int numHeads;
        using (var probe = GgufFile.Open(modelPath!))
            numHeads = DotLLM.Models.Gguf.GgufModelConfigExtractor.Extract(probe.Metadata).NumAttentionHeads;
        Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(Context, numHeads),
            $"Context {Context} too short to split for numHeads={numHeads}.");
        _output.WriteLine(
            $"numHeads={numHeads}; splits at ctx {Context} = {VulkanSplitKvAttentionKernel.ComputeSplits(Context, numHeads)}");

        var (tokensOn, logitsOn) = RunDecode(modelPath!, disableSplit: false);
        var (tokensOff, logitsOff) = RunDecode(modelPath!, disableSplit: true);

        _output.WriteLine($"split-on  tokens: {string.Join(",", tokensOn)}");
        _output.WriteLine($"split-off tokens: {string.Join(",", tokensOff)}");

        // Greedy decode must follow the identical path under both kernels.
        Assert.Equal(tokensOff, tokensOn);

        // And the final-step logits must agree within reduction-order noise.
        float maxAbs = 0;
        for (int i = 0; i < logitsOn.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(logitsOn[i] - logitsOff[i]));
        _output.WriteLine($"final-step logits L_inf = {maxAbs:G6}");
        Assert.True(maxAbs <= LogitsAbsTol,
            $"Split vs per-token decode logits diverged: L_inf={maxAbs:G6} > {LogitsAbsTol}");
    }

    /// <summary>
    /// Issue #331 — the #222-class deep guard. From an 8-token real-prose prompt
    /// (so the run crosses the true ~17 engagement threshold), runs
    /// <see cref="DeepGenSteps"/> greedy steps ON vs OFF and reports the first token
    /// divergence with its top1/top2 margins; then runs a teacher-forced decode-mode
    /// perplexity over <see cref="DeepPplTokens"/> real tokens and asserts
    /// (i) bit-exact per-step NLL below the engagement threshold and (ii) a bound on
    /// the post-engagement perplexity ratio. A fresh model instance is loaded for
    /// each arm because the split gate is read at model construction.
    /// </summary>
    [SkippableFact]
    public void SplitDecode_DeepGenerationAndDecodePerplexity_MatchPerTokenKernel()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available.");

        string? modelPath = ResolveModelPath();
        Skip.If(modelPath is null, "Parity model GGUF not found (set DOTLLM_SPLIT_PARITY_GGUF).");

        int numHeads, vocabSize;
        int[] corpus;
        using (var probe = GgufFile.Open(modelPath!))
        {
            var cfg = DotLLM.Models.Gguf.GgufModelConfigExtractor.Extract(probe.Metadata);
            numHeads = cfg.NumAttentionHeads;
            vocabSize = cfg.VocabSize;
            corpus = GgufBpeTokenizerFactory.Load(probe.Metadata).Encode(Corpus);
        }

        // The depth at which the split path first engages for THIS model's shape.
        int gate = FirstSplittingSeqKv(numHeads);
        _output.WriteLine($"numHeads={numHeads}; split path first engages at seqKv={gate} " +
                          $"(S={VulkanSplitKvAttentionKernel.ComputeSplits(gate, numHeads)}); " +
                          $"S at depth {DeepPromptLen + DeepGenSteps} = " +
                          $"{VulkanSplitKvAttentionKernel.ComputeSplits(DeepPromptLen + DeepGenSteps, numHeads)}");
        _output.WriteLine($"corpus: {corpus.Length} real BPE tokens");
        Assert.True(corpus.Length >= DeepPromptLen + 64, "Corpus too short.");

        // ── 1. Greedy generation, ON vs OFF ──────────────────────────────────
        int[] prompt = corpus[..DeepPromptLen];

        _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] generation, split OFF (fresh model)");
        var (tokensOff, marginsOff) = WithSplitDisabled(true,
            () => WithModel(modelPath!, m => GreedyGenerate(m, prompt, DeepGenSteps, vocabSize)));
        _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] generation, split ON (fresh model)");
        var (tokensOn, marginsOn) = WithSplitDisabled(false,
            () => WithModel(modelPath!, m => GreedyGenerate(m, prompt, DeepGenSteps, vocabSize)));

        int firstDivergence = -1;
        for (int i = 0; i < DeepGenSteps; i++)
            if (tokensOff[i] != tokensOn[i]) { firstDivergence = i; break; }

        if (firstDivergence < 0)
        {
            _output.WriteLine(
                $"GENERATION: token-for-token EXACT MATCH across all {DeepGenSteps} greedy steps " +
                $"(final decode depth {DeepPromptLen + DeepGenSteps}).");
        }
        else
        {
            int mismatches = 0;
            for (int i = firstDivergence; i < DeepGenSteps; i++)
                if (tokensOff[i] != tokensOn[i]) mismatches++;
            _output.WriteLine(
                $"GENERATION: FIRST DIVERGENCE at generated-step {firstDivergence} " +
                $"(decode depth {DeepPromptLen + firstDivergence}): off={tokensOff[firstDivergence]} " +
                $"on={tokensOn[firstDivergence]}; margins off={marginsOff[firstDivergence]:E4} " +
                $"on={marginsOn[firstDivergence]:E4}; {mismatches}/{DeepGenSteps - firstDivergence} " +
                "steps differ from there onward (compounding, as expected once the argmax path diverges).");
        }

        float minMarginOff = float.MaxValue, minMarginOn = float.MaxValue;
        for (int i = 0; i < DeepGenSteps; i++)
        {
            minMarginOff = MathF.Min(minMarginOff, marginsOff[i]);
            minMarginOn = MathF.Min(minMarginOn, marginsOn[i]);
        }
        _output.WriteLine($"GENERATION: smallest top1/top2 margin — off {minMarginOff:E4}, on {minMarginOn:E4}.");

        // A divergence is expected-in-principle for a reassociated kernel, but a
        // divergence in the FIRST few steps — i.e. below/at the engagement point,
        // where perturbation is a single split boundary — would mean something much
        // worse than reassociation noise. Guard that specifically.
        Assert.True(firstDivergence < 0 || DeepPromptLen + firstDivergence > gate,
            $"Generation diverged at decode depth {DeepPromptLen + firstDivergence}, at or below the " +
            $"split engagement depth {gate} — that is not reassociation noise.");

        // ── 2. Teacher-forced decode-mode perplexity, ON vs OFF ──────────────
        int[] pplTokens = corpus.Length > DeepPplTokens ? corpus[..DeepPplTokens] : corpus;
        _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] decode-PPL, split OFF ({pplTokens.Length} tokens, fresh model)");
        var (_, perStepOff) = WithSplitDisabled(true,
            () => WithModel(modelPath!, m => DecodeModeNll(m, pplTokens, vocabSize)));
        _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] decode-PPL, split ON (fresh model)");
        var (_, perStepOn) = WithSplitDisabled(false,
            () => WithModel(modelPath!, m => DecodeModeNll(m, pplTokens, vocabSize)));
        _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] all phases complete");

        // perStep[t] scores token t+1 from a forward at position t, i.e. seqKv = t+1.
        double preOff = 0, preOn = 0, postOff = 0, postOn = 0;
        int preCount = 0, postCount = 0;
        double maxPreStepDiff = 0;
        for (int t = 0; t < perStepOff.Length; t++)
        {
            int seqKv = t + 1;
            if (seqKv < gate)
            {
                preOff += perStepOff[t]; preOn += perStepOn[t]; preCount++;
                maxPreStepDiff = Math.Max(maxPreStepDiff, Math.Abs(perStepOff[t] - perStepOn[t]));
            }
            else { postOff += perStepOff[t]; postOn += perStepOn[t]; postCount++; }
        }

        double pplPreOff = Math.Exp(preOff / Math.Max(1, preCount));
        double pplPreOn = Math.Exp(preOn / Math.Max(1, preCount));
        double pplPostOff = Math.Exp(postOff / Math.Max(1, postCount));
        double pplPostOn = Math.Exp(postOn / Math.Max(1, postCount));
        double pplAllOff = Math.Exp((preOff + postOff) / perStepOff.Length);
        double pplAllOn = Math.Exp((preOn + postOn) / perStepOn.Length);

        _output.WriteLine(
            $"PPL pre-engagement  (seqKv<{gate}, {preCount} steps, split cannot engage): " +
            $"off={pplPreOff:F5} on={pplPreOn:F5} ({(pplPreOn / pplPreOff - 1) * 100:+0.0000;-0.0000}%)");
        _output.WriteLine(
            $"PPL post-engagement (seqKv>={gate}, {postCount} steps, split path engaged): " +
            $"off={pplPostOff:F5} on={pplPostOn:F5} ({(pplPostOn / pplPostOff - 1) * 100:+0.0000;-0.0000}%)");
        _output.WriteLine(
            $"PPL overall ({perStepOff.Length} steps): off={pplAllOff:F5} on={pplAllOn:F5} " +
            $"({(pplAllOn / pplAllOff - 1) * 100:+0.0000;-0.0000}%)");
        _output.WriteLine($"Pre-engagement per-step NLL max|off-on| = {maxPreStepDiff:E3} (expect exactly 0).");

        // (i) Below the engagement threshold the split kernel cannot be dispatched,
        //     so the two arms must be bit-identical. A non-zero here is a gate bug,
        //     not a precision question.
        Assert.Equal(0.0, maxPreStepDiff);

        // (ii) The decision metric: post-engagement perplexity ratio.
        double postRatio = pplPostOn / pplPostOff;
        Assert.True(Math.Abs(postRatio - 1.0) <= PplRatioTol,
            $"Post-engagement decode perplexity moved by {(postRatio - 1) * 100:+0.0000;-0.0000}% " +
            $"(off={pplPostOff:F5} on={pplPostOn:F5}), beyond the ±{PplRatioTol * 100:F2}% bound. " +
            "Split-KV is default-ON — re-open issue #331 before shipping this.");
    }

    /// <summary>
    /// Issue #331 — the same perplexity comparison as the deep test, but at the
    /// decode depth the split-KV win was actually measured at (#345: 2.1x at ctx
    /// 2048, 2.8x at ctx 4096). Prefills a filler context to
    /// <see cref="DeepContextFill"/> (chunked, to stay under the gfx1151 watchdog),
    /// then teacher-forces the tail of the real corpus one token at a time so every
    /// scored step runs the decode path at depth ~<see cref="DeepContextFill"/>+.
    /// Scoring only real prose keeps the NLL meaningful even though the filler is
    /// tiled; and because this is an ON-vs-OFF differential, the filler's own
    /// statistics cancel.
    /// </summary>
    [SkippableFact]
    public void SplitDecode_DecodePerplexity_MatchesPerTokenKernel_AtDeepContext()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available.");

        string? modelPath = ResolveModelPath();
        Skip.If(modelPath is null, "Parity model GGUF not found (set DOTLLM_SPLIT_PARITY_GGUF).");

        int numHeads, vocabSize;
        int[] corpus;
        using (var probe = GgufFile.Open(modelPath!))
        {
            var cfg = DotLLM.Models.Gguf.GgufModelConfigExtractor.Extract(probe.Metadata);
            numHeads = cfg.NumAttentionHeads;
            vocabSize = cfg.VocabSize;
            corpus = GgufBpeTokenizerFactory.Load(probe.Metadata).Encode(Corpus);
        }
        Assert.True(corpus.Length > DeepContextScored + 16, "Corpus too short.");

        _output.WriteLine(
            $"numHeads={numHeads}; S at depth {DeepContextFill} = " +
            $"{VulkanSplitKvAttentionKernel.ComputeSplits(DeepContextFill, numHeads)}, at " +
            $"{DeepContextFill + DeepContextScored} = " +
            $"{VulkanSplitKvAttentionKernel.ComputeSplits(DeepContextFill + DeepContextScored, numHeads)}");

        // Filler = the corpus tiled to DeepContextFill positions; scored tail = the
        // last DeepContextScored real corpus tokens.
        int[] filler = new int[DeepContextFill];
        for (int i = 0; i < DeepContextFill; i++) filler[i] = corpus[i % corpus.Length];
        int[] scored = corpus[^(DeepContextScored + 1)..];

        _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] deep-context decode-PPL, split OFF (fresh model)");
        double[] perStepOff = WithSplitDisabled(true,
            () => WithModel(modelPath!, m => DeepContextNll(m, filler, scored, vocabSize)));
        _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] deep-context decode-PPL, split ON (fresh model)");
        double[] perStepOn = WithSplitDisabled(false,
            () => WithModel(modelPath!, m => DeepContextNll(m, filler, scored, vocabSize)));
        _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] complete");

        double sumOff = 0, sumOn = 0;
        for (int i = 0; i < perStepOff.Length; i++) { sumOff += perStepOff[i]; sumOn += perStepOn[i]; }
        double pplOff = Math.Exp(sumOff / perStepOff.Length);
        double pplOn = Math.Exp(sumOn / perStepOn.Length);
        double ratio = pplOn / pplOff;
        _output.WriteLine(
            $"PPL at depth {DeepContextFill}-{DeepContextFill + DeepContextScored} " +
            $"({perStepOff.Length} scored steps): off={pplOff:F5} on={pplOn:F5} " +
            $"({(ratio - 1) * 100:+0.0000;-0.0000}%)");

        foreach (double v in perStepOn) Assert.True(double.IsFinite(v), "non-finite NLL with split-KV on");
        Assert.True(Math.Abs(ratio - 1.0) <= PplRatioTol,
            $"Deep-context decode perplexity moved by {(ratio - 1) * 100:+0.0000;-0.0000}% " +
            $"(off={pplOff:F5} on={pplOn:F5}), beyond the ±{PplRatioTol * 100:F2}% bound. " +
            "Split-KV is default-ON — re-open issue #331 before shipping this.");
    }

    /// <summary>
    /// Chunked prefill of <paramref name="filler"/>, then teacher-forced one-token
    /// decode over <paramref name="scored"/>; returns the per-step NLL for
    /// <c>scored[1..]</c>.
    /// </summary>
    private static unsafe double[] DeepContextNll(
        VulkanTransformerModel model, int[] filler, int[] scored, int vocabSize)
    {
        using var cache = model.CreateKvCache(filler.Length + scored.Length + 1);
        int[] positions = new int[filler.Length];
        for (int i = 0; i < filler.Length; i++) positions[i] = i;

        const int PrefillChunk = 256;   // keeps each dispatch under the gfx1151 watchdog
        for (int off = 0; off < filler.Length; off += PrefillChunk)
        {
            int len = Math.Min(PrefillChunk, filler.Length - off);
            using var _ = model.Forward(filler.AsSpan(off, len), positions.AsSpan(off, len), -1, cache);
        }

        var perStep = new double[scored.Length - 1];
        int pos = filler.Length;
        for (int t = 0; t < scored.Length - 1; t++)
        {
            int[] one = { scored[t] };
            int[] p = { pos };
            using ITensor logits = model.Forward(one, p, -1, cache);
            var span = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
            perStep[t] = -StableLogProb(span, scored[t + 1]);
            pos++;
        }
        return perStep;
    }

    /// <summary>Smallest <c>seqKv</c> at which <see cref="VulkanSplitKvAttentionKernel.WouldSplit"/> is true.</summary>
    private static int FirstSplittingSeqKv(int numHeads)
    {
        for (int s = 1; s <= 8192; s++)
            if (VulkanSplitKvAttentionKernel.WouldSplit(s, numHeads)) return s;
        return int.MaxValue;
    }

    private static T WithSplitDisabled<T>(bool disable, Func<T> body)
    {
        string? prev = Environment.GetEnvironmentVariable(VulkanTransformerModel.DisableSplitDecodeEnvVar);
        Environment.SetEnvironmentVariable(VulkanTransformerModel.DisableSplitDecodeEnvVar, disable ? "1" : null);
        try { return body(); }
        finally { Environment.SetEnvironmentVariable(VulkanTransformerModel.DisableSplitDecodeEnvVar, prev); }
    }

    private static T WithModel<T>(string modelPath, Func<VulkanTransformerModel, T> body)
    {
        string spvDir = ResolveSpvDir();
        using var gguf = GgufFile.Open(modelPath);
        var config = DotLLM.Models.Gguf.GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
        return body(model);
    }

    /// <summary>Greedy (argmax) generation; returns tokens and the top1−top2 logit margin per step.</summary>
    private static unsafe (int[] tokens, float[] margins) GreedyGenerate(
        VulkanTransformerModel model, int[] prompt, int steps, int vocabSize)
    {
        var tokens = new int[steps];
        var margins = new float[steps];
        using var cache = model.CreateKvCache(prompt.Length + steps + 1);

        int[] promptPositions = new int[prompt.Length];
        for (int i = 0; i < prompt.Length; i++) promptPositions[i] = i;

        int nextToken; float margin;
        using (ITensor logits = model.Forward(prompt, promptPositions, -1, cache))
            (nextToken, margin) = ArgmaxAndMargin(logits, vocabSize);

        int pos = prompt.Length;
        for (int step = 0; step < steps; step++)
        {
            tokens[step] = nextToken;
            margins[step] = margin;
            int[] one = { nextToken };
            int[] onePos = { pos };
            using ITensor logits = model.Forward(one, onePos, -1, cache);
            (nextToken, margin) = ArgmaxAndMargin(logits, vocabSize);
            pos++;
        }
        return (tokens, margins);
    }

    /// <summary>
    /// Teacher-forced decode-mode NLL (natural log) per step. Every forward is
    /// <c>seqLen == 1</c>, i.e. the real decode path this issue is about.
    /// </summary>
    private static unsafe (double totalNll, double[] perStep) DecodeModeNll(
        VulkanTransformerModel model, int[] tokenIds, int vocabSize)
    {
        using var cache = model.CreateKvCache(tokenIds.Length + 1);
        var perStep = new double[tokenIds.Length - 1];
        double total = 0;
        for (int t = 0; t < tokenIds.Length - 1; t++)
        {
            int[] one = { tokenIds[t] };
            int[] pos = { t };
            using ITensor logits = model.Forward(one, pos, -1, cache);
            var span = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
            double nll = -StableLogProb(span, tokenIds[t + 1]);
            perStep[t] = nll;
            total += nll;
        }
        return (total, perStep);
    }

    private static double StableLogProb(ReadOnlySpan<float> logitsRow, int target)
    {
        float max = logitsRow[0];
        for (int j = 1; j < logitsRow.Length; j++)
            if (logitsRow[j] > max) max = logitsRow[j];
        double sumExp = 0;
        for (int j = 0; j < logitsRow.Length; j++)
            sumExp += Math.Exp(logitsRow[j] - max);
        return (logitsRow[target] - max) - Math.Log(sumExp);
    }

    private static unsafe (int argmax, float margin) ArgmaxAndMargin(ITensor logits, int vocabSize)
    {
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
        int best = 0; float bestVal = span[0];
        float secondVal = float.NegativeInfinity;
        bool haveSecond = false;
        for (int i = 1; i < span.Length; i++)
        {
            float v = span[i];
            if (v > bestVal) { secondVal = bestVal; haveSecond = true; best = i; bestVal = v; }
            else if (v > secondVal) { secondVal = v; haveSecond = true; }
        }
        return (best, haveSecond ? bestVal - secondVal : float.PositiveInfinity);
    }

    private (int[] tokens, float[] lastLogits) RunDecode(string modelPath, bool disableSplit)
    {
        // IsSplitDecodeDisabled() is read at model construction, so set the env
        // var before LoadFromGguf and clear it afterwards.
        return WithSplitDisabled(disableSplit, () => WithModel(modelPath, model =>
        {
            using var gguf = GgufFile.Open(modelPath);
            var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

            int[] baseTok = tokenizer.Encode("The history of science is a long and winding road that ");
            Assert.NotEmpty(baseTok);
            int[] prompt = new int[Context];
            int[] positions = new int[Context];
            for (int i = 0; i < Context; i++) { prompt[i] = baseTok[i % baseTok.Length]; positions[i] = i; }

            int maxSeq = Context + DecodeSteps + 8;
            using var cache = model.CreateKvCache(maxSeq);

            // Chunked prefill keeps each dispatch under the gfx1151 watchdog.
            const int PrefillChunk = 256;
            int nextToken = 0;
            for (int off = 0; off < Context; off += PrefillChunk)
            {
                int len = Math.Min(PrefillChunk, Context - off);
                using var logits = model.Forward(prompt.AsSpan(off, len), positions.AsSpan(off, len), -1, cache);
                if (off + len >= Context) nextToken = Argmax(logits);
            }

            var tokens = new int[DecodeSteps];
            float[] lastLogits = Array.Empty<float>();
            int pos = Context;
            for (int i = 0; i < DecodeSteps; i++)
            {
                int[] s = { nextToken };
                int[] p = { pos };
                using var l = model.Forward(s, p, -1, cache);
                nextToken = Argmax(l);
                tokens[i] = nextToken;
                if (i == DecodeSteps - 1) lastLogits = ToArray(l);
                pos++;
            }
            return (tokens, lastLogits);
        }));
    }

    private static unsafe int Argmax(ITensor logits)
    {
        int n = logits.Shape[logits.Shape.Rank - 1];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
        int idx = 0; float best = span[0];
        for (int i = 1; i < n; i++) if (span[i] > best) { best = span[i]; idx = i; }
        return idx;
    }

    private static unsafe float[] ToArray(ITensor logits)
    {
        int n = logits.Shape[logits.Shape.Rank - 1];
        return new ReadOnlySpan<float>((void*)logits.DataPointer, n).ToArray();
    }

    private static string? ResolveModelPath()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_SPLIT_PARITY_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string conventional = "C:/Development/gguf-cache/Llama-3.2-3B-Instruct-IQ4_XS.gguf";
        return File.Exists(conventional) ? conventional : null;
    }

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException("SPIR-V blobs not found. Run native/vulkan/build.ps1.");
    }

    // A long, real, public-domain English passage (opening of Pride and Prejudice)
    // — real prose, so BPE token statistics and perplexity are meaningful.
    private const string Corpus =
        "It is a truth universally acknowledged, that a single man in possession of a good "
        + "fortune must be in want of a wife. However little known the feelings or views of "
        + "such a man may be on his first entering a neighbourhood, this truth is so well "
        + "fixed in the minds of the surrounding families, that he is considered as the "
        + "rightful property of some one or other of their daughters. My dear Mr. Bennet, "
        + "said his lady to him one day, have you heard that Netherfield Park is let at last? "
        + "Mr. Bennet replied that he had not. But it is, returned she; for Mrs. Long has just "
        + "been here, and she told me all about it. Mr. Bennet made no answer. Do not you want "
        + "to know who has taken it? cried his wife impatiently. You want to tell me, and I "
        + "have no objection to hearing it. This was invitation enough. Why, my dear, you must "
        + "know, Mrs. Long says that Netherfield is taken by a young man of large fortune from "
        + "the north of England; that he came down on Monday in a chaise and four to see the "
        + "place, and was so much delighted with it, that he agreed with Mr. Morris immediately; "
        + "that he is to take possession before Michaelmas, and some of his servants are to be "
        + "in the house by the end of next week. What is his name? Bingley. Is he married or "
        + "single? Oh! single, my dear, to be sure! A single man of large fortune; four or five "
        + "thousand a year. What a fine thing for our girls! How so? how can it affect them? My "
        + "dear Mr. Bennet, replied his wife, how can you be so tiresome! You must know that I "
        + "am thinking of his marrying one of them. Is that his design in settling here? Design! "
        + "nonsense, how can you talk so! But it is very likely that he may fall in love with one "
        + "of them, and therefore you must visit him as soon as he comes. I see no occasion for "
        + "that. You and the girls may go, or you may send them by themselves, which perhaps will "
        + "be still better; for as you are as handsome as any of them, Mr. Bingley might like you "
        + "the best of the party. My dear, you flatter me. I certainly have had my share of "
        + "beauty, but I do not pretend to be anything extraordinary now. When a woman has five "
        + "grown up daughters, she ought to give over thinking of her own beauty. In such cases, "
        + "a woman has not often much beauty to think of. But, my dear, you must indeed go and "
        + "see Mr. Bingley when he comes into the neighbourhood. It is more than I engage for, I "
        + "assure you. But consider your daughters. Only think what an establishment it would be "
        + "for one of them. Sir William and Lady Lucas are determined to go, merely on that "
        + "account, for in general you know they visit no newcomers. Indeed you must go, for it "
        + "will be impossible for us to visit him if you do not. You are over scrupulous surely. "
        + "I dare say Mr. Bingley will be very glad to see you; and I will send a few lines by "
        + "you to assure him of my hearty consent to his marrying whichever he chuses of the "
        + "girls; though I must throw in a good word for my little Lizzy. I desire you will do "
        + "no such thing. Lizzy is not a bit better than the others; and I am sure she is not "
        + "half so handsome as Jane, nor half so good humoured as Lydia. But you are always "
        + "giving her the preference. They have none of them much to recommend them, replied he; "
        + "they are all silly and ignorant like other girls; but Lizzy has something more of "
        + "quickness than her sisters. Mr. Bennet, how can you abuse your own children in such a "
        + "way? You take delight in vexing me. You have no compassion for my poor nerves. You "
        + "mistake me, my dear. I have a high respect for your nerves. They are my old friends. "
        + "I have heard you mention them with consideration these twenty years at least. Ah, you "
        + "do not know what I suffer. But I hope you will get over it, and live to see many young "
        + "men of four thousand a year come into the neighbourhood. It will be no use to us, if "
        + "twenty such should come, since you will not visit them. Depend upon it, my dear, that "
        + "when there are twenty, I will visit them all. Mr. Bennet was so odd a mixture of quick "
        + "parts, sarcastic humour, reserve, and caprice, that the experience of three and twenty "
        + "years had been insufficient to make his wife understand his character. Her mind was "
        + "less difficult to develope. She was a woman of mean understanding, little information, "
        + "and uncertain temper. When she was discontented she fancied herself nervous. The "
        + "business of her life was to get her daughters married; its solace was visiting and "
        + "news. The rest of the party arrived soon after, and the evening was spent in the "
        + "usual manner. After dinner the card tables were placed, and all the ladies rose to "
        + "leave the room, and every one was surprised, and Elizabeth was catching Miss Bingley's "
        + "eye, was forced to submit to a general inquiry after their families. Nothing was "
        + "spoken of but the ball, and every particular was recollected with pleasure by those "
        + "who had taken most delight in it. It was a large party assembled at Longbourn, and the "
        + "arrival of a fresh person could not fail of giving intelligence to the surrounding "
        + "families, and of continuing the flow of small talk that filled every drawing room in "
        + "the neighbourhood for the following week. The morning conversation which had passed "
        + "between the sisters was resumed with a good deal of animation, as one report followed "
        + "another, until the whole party was in a state of pleasant expectation regarding the "
        + "gentleman who had lately arrived among them.";
}
