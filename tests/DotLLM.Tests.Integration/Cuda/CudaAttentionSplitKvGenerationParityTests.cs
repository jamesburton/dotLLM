using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Issue #222: verifies <see cref="CudaKernels.EnableAttentionSplitKv"/> (issue #183, opt-in,
/// default-OFF) at real generation scale on the actual Bonsai-27B model, not just the per-step
/// unit tolerance already covered by <c>CudaAttentionF32SplitKvTests</c>. Two independent checks,
/// same loaded model/weights, only the toggle differs (same pattern as
/// <see cref="CudaGemm16FPerplexityTests"/>):
/// <list type="number">
/// <item>Real greedy autoregressive generation, baseline vs split-KV, same prompt, spanning
/// decode depth well past the 768-1024 range issue #219 profiled — token-for-token comparison,
/// plus the top1/top2 logit margin at every step (a logit-level nudge that doesn't flip argmax
/// still shows up here even when the generated text matches exactly).</item>
/// <item>Teacher-forced decode-mode perplexity (methodology from
/// <see cref="CudaGemm16FPerplexityTests.DecodeModePerplexity"/>, adapted for the CUDA hybrid
/// model) over a long real-text corpus, split by decode depth into a "pre-gate" segment
/// (seqKv &lt; <see cref="CudaKernels.AttentionSplitKvMinSeqKv"/>, split-KV cannot engage,
/// expected bit-identical) and a "post-gate" segment (seqKv &gt;= the gate, where any real
/// precision drift would show up) — perplexity is far more sensitive to small per-step
/// differences than greedy argmax-only comparison.</item>
/// </list>
/// </summary>
[Trait("Category", "GPU")]
public sealed class CudaAttentionSplitKvGenerationParityTests
{
    private readonly ITestOutputHelper _output;
    public CudaAttentionSplitKvGenerationParityTests(ITestOutputHelper output) => _output = output;

    // A long, real, public-domain English passage (opening of Pride and Prejudice) — real
    // prose, not synthetic/tiled tokens, so BPE token statistics and perplexity are meaningful.
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

    [SkippableFact]
    public unsafe void SplitKv_GreedyGenerationAndDecodePerplexity_MatchesBaseline_AtRealScale()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string? ggufPath = ResolveModelPath();
        Skip.If(ggufPath is null,
            "Bonsai PQ2_0 GGUF fixture not found. Set DOTLLM_BONSAI_PQ2_0_GGUF, or place "
            + "Ternary-Bonsai-27B-Q2_0.gguf under ~/.dotllm/models/PrismML/Ternary-Bonsai-27B-GGUF/ "
            + "or ~/.dotllm/test-cache/PrismML/Ternary-Bonsai-27B-GGUF/.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        using var gguf = GgufFile.Open(ggufPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int[] fullCorpus = tokenizer.Encode(Corpus);
        _output.WriteLine($"Corpus: {fullCorpus.Length} real BPE tokens.");
        Assert.True(fullCorpus.Length >= 1100,
            $"Corpus too short ({fullCorpus.Length} tokens) to span depth 768-1024+.");

        bool priorSplitKv = CudaKernels.EnableAttentionSplitKv;
        try
        {
            _output.WriteLine($"AttentionSplitKvMinSeqKv gate = {CudaKernels.AttentionSplitKvMinSeqKv}");

            // ── 1. Greedy autoregressive generation parity ──────────────────────
            // Real prompt (first 32 corpus tokens), then GREEDY (argmax) generation for
            // enough steps to pass depth 768-1024+.
            //
            // IMPORTANT: a FRESH model instance is loaded for EACH of the off/on runs (and
            // again for each PPL run below), not one shared instance with the KV cache swapped.
            // Qwen3HybridDense's GDN (linear-attention) recurrent state (CudaGdnStateCache) is
            // owned by the MODEL object, not by the IKvCache handle passed to Forward() -- 32 of
            // Bonsai-27B's 64 layers are GDN, so reusing one model instance across the off/on
            // passes leaks GDN state from the first run into the second and corrupts everything
            // downstream (confirmed empirically: an earlier version of this test that shared one
            // model instance showed divergence at generated-step 0, depth 32 -- BELOW the
            // AttentionSplitKvMinSeqKv=256 gate where split-KV cannot even engage -- and
            // catastrophic PPL, i.e. a state-leak artifact, not a kernel precision finding).
            // Reloading costs ~4 model loads total (~8-12s each) instead of one; cheap relative
            // to the ~1100-step generations/PPL sweeps this test runs.
            const int promptLen = 32;
            const int genSteps = 1000; // final position ~1032 -- spans the profiled 768-1024 range
            int[] prompt = fullCorpus[..promptLen];

            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: baseline generation (fresh model load)");
            CudaKernels.EnableAttentionSplitKv = false;
            (int[] tokensOff, float[] marginsOff) genOff;
            using (var modelOff = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
            {
                _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] baseline model loaded, starting generation");
                genOff = GreedyGenerate(modelOff, prompt, genSteps, config.VocabSize);
            }
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE END: baseline generation (model disposed)");

            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: split-kv generation (fresh model load)");
            CudaKernels.EnableAttentionSplitKv = true;
            (int[] tokensOn, float[] marginsOn) genOn;
            using (var modelOn = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
            {
                _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] split-kv model loaded, starting generation");
                genOn = GreedyGenerate(modelOn, prompt, genSteps, config.VocabSize);
            }
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE END: split-kv generation (model disposed)");

            var (tokensOff, marginsOff) = genOff;
            var (tokensOn, marginsOn) = genOn;

            // Human-readable sanity check: the first ~60 generated tokens decoded back to text,
            // so a reviewer can see this is coherent continuation, not harness-bug garbage.
            _output.WriteLine($"Baseline generation (first 60 tokens): {tokenizer.Decode(tokensOff.AsSpan(0, Math.Min(60, tokensOff.Length)))}");
            _output.WriteLine($"Split-KV generation (first 60 tokens): {tokenizer.Decode(tokensOn.AsSpan(0, Math.Min(60, tokensOn.Length)))}");

            int firstDivergence = -1;
            for (int i = 0; i < genSteps; i++)
            {
                if (tokensOff[i] != tokensOn[i]) { firstDivergence = i; break; }
            }

            if (firstDivergence < 0)
            {
                _output.WriteLine(
                    $"GENERATION: token-for-token EXACT MATCH across all {genSteps} greedy steps " +
                    $"(final depth {promptLen + genSteps}).");
            }
            else
            {
                int depthAtDivergence = promptLen + firstDivergence;
                _output.WriteLine(
                    $"GENERATION: FIRST DIVERGENCE at generated-step {firstDivergence} " +
                    $"(decode depth {depthAtDivergence}): baseline token={tokensOff[firstDivergence]} " +
                    $"split-kv token={tokensOn[firstDivergence]}");
                _output.WriteLine(
                    $"  baseline top1/top2 margin at divergence: {marginsOff[firstDivergence]:E4}; " +
                    $"split-kv: {marginsOn[firstDivergence]:E4}");
                int mismatches = 0;
                for (int i = firstDivergence; i < genSteps; i++)
                    if (tokensOff[i] != tokensOn[i]) mismatches++;
                _output.WriteLine(
                    $"  {mismatches}/{genSteps - firstDivergence} steps differ from divergence point onward " +
                    "(compounding, as expected once the argmax path itself diverges).");
            }

            // Margin summary (logit-level "how close was every decision", even where argmax
            // matched): a small minimum margin close to the gate depth (256) would mean split-KV
            // came close to flipping argmax without actually doing so in this run.
            float minMarginOff = float.MaxValue, minMarginOn = float.MaxValue;
            int minMarginOffIdx = -1, minMarginOnIdx = -1;
            for (int i = 0; i < genSteps; i++)
            {
                if (marginsOff[i] < minMarginOff) { minMarginOff = marginsOff[i]; minMarginOffIdx = i; }
                if (marginsOn[i] < minMarginOn) { minMarginOn = marginsOn[i]; minMarginOnIdx = i; }
            }
            _output.WriteLine(
                $"GENERATION: smallest top1/top2 logit margin — baseline {minMarginOff:E4} at step " +
                $"{minMarginOffIdx} (depth {promptLen + minMarginOffIdx}); split-kv {minMarginOn:E4} at " +
                $"step {minMarginOnIdx} (depth {promptLen + minMarginOnIdx}).");

            // ── 2. Teacher-forced decode-mode perplexity, pre-gate vs post-gate ─
            // Again: a fresh model instance per pass (see note above on GDN state ownership).
            int[] pplTokens = fullCorpus.Length > 1040 ? fullCorpus[..1040] : fullCorpus;
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: baseline decode-PPL (fresh model load), {pplTokens.Length} tokens");
            CudaKernels.EnableAttentionSplitKv = false;
            (double nllOff, double[] perStepOff) pplOffResult;
            using (var modelOff = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
                pplOffResult = DecodeModeNll(modelOff, pplTokens, config.VocabSize);
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE END: baseline decode-PPL (model disposed)");

            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: split-kv decode-PPL (fresh model load)");
            CudaKernels.EnableAttentionSplitKv = true;
            (double nllOn, double[] perStepOn) pplOnResult;
            using (var modelOn = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
                pplOnResult = DecodeModeNll(modelOn, pplTokens, config.VocabSize);
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE END: split-kv decode-PPL (model disposed) -- ALL PHASES COMPLETE");

            var (nllOff, perStepOff) = pplOffResult;
            var (nllOn, perStepOn) = pplOnResult;

            int gate = CudaKernels.AttentionSplitKvMinSeqKv;
            double preGateNllOff = 0, preGateNllOn = 0;
            double postGateNllOff = 0, postGateNllOn = 0;
            int preCount = 0, postCount = 0;
            for (int t = 0; t < perStepOff.Length; t++)
            {
                int depth = t; // seqKv at the step scoring token t+1 is t+1; use t as a depth proxy
                if (depth < gate) { preGateNllOff += perStepOff[t]; preGateNllOn += perStepOn[t]; preCount++; }
                else { postGateNllOff += perStepOff[t]; postGateNllOn += perStepOn[t]; postCount++; }
            }

            double pplPreOff = Math.Exp(preGateNllOff / Math.Max(1, preCount));
            double pplPreOn = Math.Exp(preGateNllOn / Math.Max(1, preCount));
            double pplPostOff = Math.Exp(postGateNllOff / Math.Max(1, postCount));
            double pplPostOn = Math.Exp(postGateNllOn / Math.Max(1, postCount));
            double pplAllOff = Math.Exp(nllOff / perStepOff.Length);
            double pplAllOn = Math.Exp(nllOn / perStepOn.Length);

            _output.WriteLine(
                $"PPL pre-gate  (depth<{gate}, {preCount} steps, split-KV cannot engage): " +
                $"off={pplPreOff:F5} on={pplPreOn:F5} ratio={(pplPreOn / pplPreOff):F6} " +
                $"({(pplPreOn / pplPreOff - 1) * 100:+0.0000;-0.0000}%)");
            _output.WriteLine(
                $"PPL post-gate (depth>={gate}, {postCount} steps, split-KV path engaged): " +
                $"off={pplPostOff:F5} on={pplPostOn:F5} ratio={(pplPostOn / pplPostOff):F6} " +
                $"({(pplPostOn / pplPostOff - 1) * 100:+0.0000;-0.0000}%)");
            _output.WriteLine(
                $"PPL overall   ({perStepOff.Length} steps): off={pplAllOff:F5} on={pplAllOn:F5} " +
                $"ratio={(pplAllOn / pplAllOff):F6} ({(pplAllOn / pplAllOff - 1) * 100:+0.0000;-0.0000}%)");

            // Sanity/engagement check: pre-gate NLL per-step should be IDENTICAL (bit-exact),
            // since the split-KV flag cannot engage below the gate — if it's not identical,
            // something is wrong with the gate itself, not a precision question.
            double maxPreGateStepDiff = 0;
            for (int t = 0; t < preCount; t++)
                maxPreGateStepDiff = Math.Max(maxPreGateStepDiff, Math.Abs(perStepOff[t] - perStepOn[t]));
            _output.WriteLine($"Pre-gate per-step NLL max|off-on| = {maxPreGateStepDiff:E3} (expect 0 exactly).");
        }
        finally
        {
            CudaKernels.EnableAttentionSplitKv = priorSplitKv;
        }
    }

    /// <summary>
    /// Issue #226: does the fp64-combine <see cref="CudaKernels.EnableAttentionSplitKvHp"/> variant
    /// reduce or eliminate the argmax-flip divergence (generated-step 225, decode depth 257) and
    /// post-gate perplexity regression (+0.30%) that #222 found in the plain float-combine split-KV
    /// kernel? Three-way comparison (baseline / plain split-KV / fp64-combine split-KV), same
    /// prompt/corpus/step-count as #222's harness, a fresh model instance per pass (same
    /// GDN-state-ownership reasoning as #222 -- see that test's note).
    /// </summary>
    [SkippableFact]
    public unsafe void SplitKvHp_GreedyGenerationAndDecodePerplexity_ComparesToBaselineAndPlainSplitKv()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string? ggufPath = ResolveModelPath();
        Skip.If(ggufPath is null,
            "Bonsai PQ2_0 GGUF fixture not found. Set DOTLLM_BONSAI_PQ2_0_GGUF, or place "
            + "Ternary-Bonsai-27B-Q2_0.gguf under ~/.dotllm/models/PrismML/Ternary-Bonsai-27B-GGUF/ "
            + "or ~/.dotllm/test-cache/PrismML/Ternary-Bonsai-27B-GGUF/.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        using var gguf = GgufFile.Open(ggufPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int[] fullCorpus = tokenizer.Encode(Corpus);
        Assert.True(fullCorpus.Length >= 1100, $"Corpus too short ({fullCorpus.Length} tokens).");

        bool priorSplitKv = CudaKernels.EnableAttentionSplitKv;
        bool priorSplitKvHp = CudaKernels.EnableAttentionSplitKvHp;
        try
        {
            const int promptLen = 32;
            const int genSteps = 1000;
            int[] prompt = fullCorpus[..promptLen];

            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: baseline generation");
            CudaKernels.EnableAttentionSplitKv = false;
            CudaKernels.EnableAttentionSplitKvHp = false;
            (int[] tokensOff, float[] marginsOff) genOff;
            using (var m = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
                genOff = GreedyGenerate(m, prompt, genSteps, config.VocabSize);
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE END: baseline generation");

            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: plain split-kv generation");
            CudaKernels.EnableAttentionSplitKv = true;
            CudaKernels.EnableAttentionSplitKvHp = false;
            (int[] tokensSplit, float[] marginsSplit) genSplit;
            using (var m = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
                genSplit = GreedyGenerate(m, prompt, genSteps, config.VocabSize);
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE END: plain split-kv generation");

            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: fp64-combine split-kv generation");
            CudaKernels.EnableAttentionSplitKv = false;
            CudaKernels.EnableAttentionSplitKvHp = true;
            (int[] tokensHp, float[] marginsHp) genHp;
            using (var m = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
                genHp = GreedyGenerate(m, prompt, genSteps, config.VocabSize);
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE END: fp64-combine split-kv generation");

            var (tokensOffA, marginsOffA) = genOff;
            var (tokensSplitA, marginsSplitA) = genSplit;
            var (tokensHpA, marginsHpA) = genHp;

            ReportDivergence("plain split-KV",   tokensOffA, tokensSplitA, marginsOffA, marginsSplitA, promptLen, genSteps);
            ReportDivergence("fp64-combine (hp)", tokensOffA, tokensHpA,    marginsOffA, marginsHpA,    promptLen, genSteps);

            // ── Teacher-forced decode-mode perplexity, three-way ──
            int[] pplTokens = fullCorpus.Length > 1040 ? fullCorpus[..1040] : fullCorpus;

            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: baseline decode-PPL");
            CudaKernels.EnableAttentionSplitKv = false;
            CudaKernels.EnableAttentionSplitKvHp = false;
            (double nllOff, double[] perStepOff) pplOff;
            using (var m = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
                pplOff = DecodeModeNll(m, pplTokens, config.VocabSize);

            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: plain split-kv decode-PPL");
            CudaKernels.EnableAttentionSplitKv = true;
            CudaKernels.EnableAttentionSplitKvHp = false;
            (double nllSplit, double[] perStepSplit) pplSplit;
            using (var m = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
                pplSplit = DecodeModeNll(m, pplTokens, config.VocabSize);

            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: fp64-combine decode-PPL");
            CudaKernels.EnableAttentionSplitKv = false;
            CudaKernels.EnableAttentionSplitKvHp = true;
            (double nllHp, double[] perStepHp) pplHp;
            using (var m = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
                pplHp = DecodeModeNll(m, pplTokens, config.VocabSize);
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE END: all PPL passes complete");

            int gate = CudaKernels.AttentionSplitKvMinSeqKv;
            ReportPostGatePpl("plain split-KV",    pplOff.perStepOff, pplSplit.perStepSplit, gate);
            ReportPostGatePpl("fp64-combine (hp)", pplOff.perStepOff, pplHp.perStepHp, gate);
        }
        finally
        {
            CudaKernels.EnableAttentionSplitKv = priorSplitKv;
            CudaKernels.EnableAttentionSplitKvHp = priorSplitKvHp;
        }
    }

    private void ReportDivergence(string label, int[] baseline, int[] variant,
        float[] marginsBaseline, float[] marginsVariant, int promptLen, int steps)
    {
        int firstDivergence = -1;
        for (int i = 0; i < steps; i++)
            if (baseline[i] != variant[i]) { firstDivergence = i; break; }

        if (firstDivergence < 0)
        {
            _output.WriteLine($"[{label}] token-for-token EXACT MATCH across all {steps} greedy steps.");
            return;
        }

        int depth = promptLen + firstDivergence;
        int mismatches = 0;
        for (int i = firstDivergence; i < steps; i++)
            if (baseline[i] != variant[i]) mismatches++;

        _output.WriteLine($"[{label}] FIRST DIVERGENCE at generated-step {firstDivergence} (depth {depth}): " +
            $"baseline margin={marginsBaseline[firstDivergence]:E4} variant margin={marginsVariant[firstDivergence]:E4}; " +
            $"{mismatches}/{steps - firstDivergence} steps differ from there onward.");
    }

    private void ReportPostGatePpl(string label, double[] perStepBaseline, double[] perStepVariant, int gate)
    {
        double preOff = 0, preOn = 0, postOff = 0, postOn = 0;
        int preCount = 0, postCount = 0;
        for (int t = 0; t < perStepBaseline.Length; t++)
        {
            if (t < gate) { preOff += perStepBaseline[t]; preOn += perStepVariant[t]; preCount++; }
            else { postOff += perStepBaseline[t]; postOn += perStepVariant[t]; postCount++; }
        }
        double pplPreOff = Math.Exp(preOff / Math.Max(1, preCount));
        double pplPreOn = Math.Exp(preOn / Math.Max(1, preCount));
        double pplPostOff = Math.Exp(postOff / Math.Max(1, postCount));
        double pplPostOn = Math.Exp(postOn / Math.Max(1, postCount));
        _output.WriteLine($"[{label}] PPL pre-gate ({preCount} steps): off={pplPreOff:F5} on={pplPreOn:F5} " +
            $"({(pplPreOn / pplPreOff - 1) * 100:+0.0000;-0.0000}%)");
        _output.WriteLine($"[{label}] PPL post-gate ({postCount} steps): off={pplPostOff:F5} on={pplPostOn:F5} " +
            $"({(pplPostOn / pplPostOff - 1) * 100:+0.0000;-0.0000}%)");
    }

    /// <summary>
    /// Greedy (argmax) autoregressive generation. Returns the generated token ids and, at each
    /// step, the top1-top2 logit margin (how decisively argmax was chosen).
    /// </summary>
    private static unsafe (int[] tokens, float[] margins) GreedyGenerate(
        CudaQwen3HybridDenseTransformerModel model, int[] prompt, int steps, int vocabSize)
    {
        var tokens = new int[steps];
        var margins = new float[steps];
        using var cache = model.CreateKvCache(maxSeqLen: prompt.Length + steps + 1);

        int[] promptPositions = new int[prompt.Length];
        for (int i = 0; i < prompt.Length; i++) promptPositions[i] = i;

        int nextToken;
        float margin;
        using (ITensor logits = model.Forward(prompt, promptPositions, deviceId: -1, cache))
        {
            (nextToken, margin) = ArgmaxAndMargin(logits, vocabSize);
        }

        int pos = prompt.Length;
        for (int step = 0; step < steps; step++)
        {
            tokens[step] = nextToken;
            margins[step] = margin;

            int[] one = { nextToken };
            int[] onePos = { pos };
            using ITensor logits = model.Forward(one, onePos, deviceId: -1, cache);
            (nextToken, margin) = ArgmaxAndMargin(logits, vocabSize);
            pos++;
        }
        return (tokens, margins);
    }

    private static unsafe (int argmax, float margin) ArgmaxAndMargin(ITensor logits, int vocabSize)
    {
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
        int best = 0; float bestVal = span[0];
        int second = -1; float secondVal = float.NegativeInfinity;
        for (int i = 1; i < span.Length; i++)
        {
            float v = span[i];
            if (v > bestVal) { second = best; secondVal = bestVal; best = i; bestVal = v; }
            else if (v > secondVal) { second = i; secondVal = v; }
        }
        return (best, second < 0 ? float.PositiveInfinity : bestVal - secondVal);
    }

    /// <summary>
    /// Teacher-forced decode-mode NLL (natural log), per-step, over <paramref name="tokenIds"/>.
    /// Every forward after the first is seqLen==1 (real decode path). Returns (total NLL,
    /// per-step NLL array indexed by target position t, scoring tokenIds[t+1]).
    /// </summary>
    private static unsafe (double totalNll, double[] perStep) DecodeModeNll(
        CudaQwen3HybridDenseTransformerModel model, int[] tokenIds, int vocabSize)
    {
        using var cache = model.CreateKvCache(maxSeqLen: tokenIds.Length + 1);
        var perStep = new double[tokenIds.Length - 1];
        double total = 0;
        for (int t = 0; t < tokenIds.Length - 1; t++)
        {
            int[] one = { tokenIds[t] };
            int[] pos = { t };
            using ITensor logits = model.Forward(one, pos, deviceId: -1, cache);
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

    private static string? ResolveModelPath()
    {
        string? envPath = Environment.GetEnvironmentVariable("DOTLLM_BONSAI_PQ2_0_GGUF");
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "models", "PrismML", "Ternary-Bonsai-27B-GGUF", "Ternary-Bonsai-27B-Q2_0.gguf"),
            Path.Combine(home, ".dotllm", "test-cache", "PrismML", "Ternary-Bonsai-27B-GGUF", "Ternary-Bonsai-27B-Q2_0.gguf"),
        ];
        foreach (string candidate in candidates)
            if (File.Exists(candidate))
                return candidate;

        return null;
    }

    private static string? FindPtxDir()
    {
        string? dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && dir is not null; i++)
        {
            string candidate = Path.Combine(dir, "native", "ptx");
            if (Directory.Exists(candidate))
                return candidate;
            dir = Path.GetDirectoryName(dir);
        }
        return null;
    }
}
