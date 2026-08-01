using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Issue #199 v2: verifies <see cref="CudaAttentionMmaDecodeGqaSplit.Enabled"/> (the composed
/// tensor-core + GQA-group decode attention kernel, opt-in, default-OFF) at real generation
/// scale on the actual Bonsai-27B model, not just the per-step synthetic-fixture tolerance
/// already covered by <c>CudaAttentionMmaDecodeGqaSplitTests</c>. Same harness shape as
/// <see cref="CudaAttentionSplitKvGenerationParityTests"/> (issue #222) -- that test exists
/// specifically because a kernel that passes synthetic-fixture parity can still carry a real
/// precision problem that only surfaces at real generation scale; the v2 kernel's own writeup
/// explicitly flagged this exact gap as the next required step before it could ship even opt-in.
/// </summary>
[Trait("Category", "GPU")]
public sealed class CudaAttentionMmaDecodeGqaSplitGenerationParityTests
{
    private readonly ITestOutputHelper _output;
    public CudaAttentionMmaDecodeGqaSplitGenerationParityTests(ITestOutputHelper output) => _output = output;

    // Same real, public-domain English passage (opening of Pride and Prejudice) as issue #222's
    // harness -- real prose, not synthetic/tiled tokens, so BPE token statistics and perplexity
    // are meaningful, and results are directly comparable to that investigation's numbers.
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
    public unsafe void MmaDecodeGqaSplit_GreedyGenerationAndDecodePerplexity_MatchesBaseline_AtRealScale()
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

        bool priorEnabled = CudaAttentionMmaDecodeGqaSplit.Enabled;
        try
        {
            _output.WriteLine($"AttentionGqaSplitMinSeqKv gate (shared) = {CudaKernels.AttentionGqaSplitMinSeqKv}");

            // ── 1. Greedy autoregressive generation parity ──────────────────────
            // Same GDN-state-ownership caveat as issue #222's harness: a FRESH model instance
            // per off/on pass (and per PPL pass below), never one shared instance with the KV
            // cache swapped -- 32 of Bonsai-27B's 64 layers are GDN, whose recurrent state is
            // owned by the model object, not the IKvCache handle.
            const int promptLen = 32;
            const int genSteps = 1000; // final position ~1032 -- spans the profiled 512+ range
            int[] prompt = fullCorpus[..promptLen];

            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: baseline generation (fresh model load)");
            CudaAttentionMmaDecodeGqaSplit.Enabled = false;
            (int[] tokensOff, float[] marginsOff) genOff;
            using (var modelOff = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
                genOff = GreedyGenerate(modelOff, prompt, genSteps, config.VocabSize);
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE END: baseline generation (model disposed)");

            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: mma-decode-gqa-split generation (fresh model load)");
            CudaAttentionMmaDecodeGqaSplit.Enabled = true;
            long dispatchBefore = CudaAttentionMmaDecodeGqaSplit.DispatchCount;
            (int[] tokensOn, float[] marginsOn) genOn;
            using (var modelOn = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
                genOn = GreedyGenerate(modelOn, prompt, genSteps, config.VocabSize);
            long dispatched = CudaAttentionMmaDecodeGqaSplit.DispatchCount - dispatchBefore;
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE END: mma-decode-gqa-split generation (model disposed)");
            _output.WriteLine($"Composed tensor-core kernel dispatched {dispatched} times " +
                $"(expect > 0 -- proves the branch actually fired, not a silent fallthrough).");
            Assert.True(dispatched > 0,
                "CudaAttentionMmaDecodeGqaSplit.DispatchCount did not increase -- the kernel never " +
                "actually fired during this generation run; the flag/gate is a no-op for this shape.");

            var (tokensOffA, marginsOffA) = genOff;
            var (tokensOnA, marginsOnA) = genOn;

            _output.WriteLine($"Baseline generation (first 60 tokens): {tokenizer.Decode(tokensOffA.AsSpan(0, Math.Min(60, tokensOffA.Length)))}");
            _output.WriteLine($"MMA-decode-GQA-split generation (first 60 tokens): {tokenizer.Decode(tokensOnA.AsSpan(0, Math.Min(60, tokensOnA.Length)))}");

            int firstDivergence = -1;
            for (int i = 0; i < genSteps; i++)
            {
                if (tokensOffA[i] != tokensOnA[i]) { firstDivergence = i; break; }
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
                int mismatches = 0;
                for (int i = firstDivergence; i < genSteps; i++)
                    if (tokensOffA[i] != tokensOnA[i]) mismatches++;
                _output.WriteLine(
                    $"GENERATION: FIRST DIVERGENCE at generated-step {firstDivergence} " +
                    $"(decode depth {depthAtDivergence}): baseline token={tokensOffA[firstDivergence]} " +
                    $"mma-gqa-split token={tokensOnA[firstDivergence]}; " +
                    $"baseline margin={marginsOffA[firstDivergence]:E4} variant margin={marginsOnA[firstDivergence]:E4}; " +
                    $"{mismatches}/{genSteps - firstDivergence} steps differ from there onward.");
            }

            float minMarginOff = float.MaxValue, minMarginOn = float.MaxValue;
            int minMarginOffIdx = -1, minMarginOnIdx = -1;
            for (int i = 0; i < genSteps; i++)
            {
                if (marginsOffA[i] < minMarginOff) { minMarginOff = marginsOffA[i]; minMarginOffIdx = i; }
                if (marginsOnA[i] < minMarginOn) { minMarginOn = marginsOnA[i]; minMarginOnIdx = i; }
            }
            _output.WriteLine(
                $"GENERATION: smallest top1/top2 logit margin — baseline {minMarginOff:E4} at step " +
                $"{minMarginOffIdx} (depth {promptLen + minMarginOffIdx}); mma-gqa-split {minMarginOn:E4} at " +
                $"step {minMarginOnIdx} (depth {promptLen + minMarginOnIdx}).");

            // ── 2. Teacher-forced decode-mode perplexity, pre-gate vs post-gate ─
            int[] pplTokens = fullCorpus.Length > 1040 ? fullCorpus[..1040] : fullCorpus;
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: baseline decode-PPL (fresh model load), {pplTokens.Length} tokens");
            CudaAttentionMmaDecodeGqaSplit.Enabled = false;
            (double nllOff, double[] perStepOff) pplOffResult;
            using (var modelOff = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
                pplOffResult = DecodeModeNll(modelOff, pplTokens, config.VocabSize);
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE END: baseline decode-PPL (model disposed)");

            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE START: mma-decode-gqa-split decode-PPL (fresh model load)");
            CudaAttentionMmaDecodeGqaSplit.Enabled = true;
            (double nllOn, double[] perStepOn) pplOnResult;
            using (var modelOn = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!))
                pplOnResult = DecodeModeNll(modelOn, pplTokens, config.VocabSize);
            _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] PHASE END: mma-decode-gqa-split decode-PPL (model disposed) -- ALL PHASES COMPLETE");

            var (nllOffA, perStepOffA) = pplOffResult;
            var (nllOnA, perStepOnA) = pplOnResult;

            int gate = CudaKernels.AttentionGqaSplitMinSeqKv;
            double preGateNllOff = 0, preGateNllOn = 0;
            double postGateNllOff = 0, postGateNllOn = 0;
            int preCount = 0, postCount = 0;
            for (int t = 0; t < perStepOffA.Length; t++)
            {
                // seqKv AT the call scoring token t+1 is t+1 (the KV write for position t lands
                // before this step's attention read) -- t alone under-counts by one and
                // misclassifies the boundary step as pre-gate when the kernel was actually
                // already eligible. Confirmed empirically (Diagnostic_TwoFreshBaselineLoads_...):
                // two Enabled=false loads and two Enabled=true loads are each bit-identical to
                // themselves, and an Enabled=false vs Enabled=true comparison is bit-identical
                // through step index gate-2, first diverges at step index gate-1 -- exactly where
                // t+1 first reaches the gate.
                int depth = t + 1;
                if (depth < gate) { preGateNllOff += perStepOffA[t]; preGateNllOn += perStepOnA[t]; preCount++; }
                else { postGateNllOff += perStepOffA[t]; postGateNllOn += perStepOnA[t]; postCount++; }
            }

            double pplPreOff = Math.Exp(preGateNllOff / Math.Max(1, preCount));
            double pplPreOn = Math.Exp(preGateNllOn / Math.Max(1, preCount));
            double pplPostOff = Math.Exp(postGateNllOff / Math.Max(1, postCount));
            double pplPostOn = Math.Exp(postGateNllOn / Math.Max(1, postCount));
            double pplAllOff = Math.Exp(nllOffA / perStepOffA.Length);
            double pplAllOn = Math.Exp(nllOnA / perStepOnA.Length);

            _output.WriteLine(
                $"PPL pre-gate  (depth<{gate}, {preCount} steps, mma-gqa-split cannot engage): " +
                $"off={pplPreOff:F5} on={pplPreOn:F5} ratio={(pplPreOn / pplPreOff):F6} " +
                $"({(pplPreOn / pplPreOff - 1) * 100:+0.0000;-0.0000}%)");
            _output.WriteLine(
                $"PPL post-gate (depth>={gate}, {postCount} steps, mma-gqa-split path engaged): " +
                $"off={pplPostOff:F5} on={pplPostOn:F5} ratio={(pplPostOn / pplPostOff):F6} " +
                $"({(pplPostOn / pplPostOff - 1) * 100:+0.0000;-0.0000}%)");
            _output.WriteLine(
                $"PPL overall   ({perStepOffA.Length} steps): off={pplAllOff:F5} on={pplAllOn:F5} " +
                $"ratio={(pplAllOn / pplAllOff):F6} ({(pplAllOn / pplAllOff - 1) * 100:+0.0000;-0.0000}%)");

            // Sanity/engagement check: pre-gate NLL per-step should be IDENTICAL (bit-exact),
            // since this kernel cannot engage below the gate — if it's not identical, something
            // is wrong with the gate itself, not a precision question.
            double maxPreGateStepDiff = 0;
            for (int t = 0; t < preCount; t++)
                maxPreGateStepDiff = Math.Max(maxPreGateStepDiff, Math.Abs(perStepOffA[t] - perStepOnA[t]));
            _output.WriteLine($"Pre-gate per-step NLL max|off-on| = {maxPreGateStepDiff:E3} (expect 0 exactly).");
            Assert.Equal(0.0, maxPreGateStepDiff);
        }
        finally
        {
            CudaAttentionMmaDecodeGqaSplit.Enabled = priorEnabled;
        }
    }

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
