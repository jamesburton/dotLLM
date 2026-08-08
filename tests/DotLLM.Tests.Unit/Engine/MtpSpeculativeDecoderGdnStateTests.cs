using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Reproduction harness for issue #287: speculative decoding has no rollback mechanism for a
/// rejected draft token's effect on <b>recurrent</b> trunk state (<see cref="IGdnState"/>).
/// </summary>
/// <remarks>
/// Uses the real (if tiny) <see cref="Qwen3HybridDenseTransformerModel"/> loaded from the
/// deterministic <see cref="SyntheticQwen35HybridDenseMtpGguf"/> fixture with its DEFAULT
/// (mixed GDN + attention) layout — deliberately NOT passing <c>fullAttnInterval: 1</c>, which
/// is the exact escape hatch the fixture's own doc comment says other tests use "to isolate a
/// test from the separate, pre-existing 'speculative decoding + recurrent trunk state has no
/// rollback' limitation." This test does the opposite: it targets that limitation directly.
/// </remarks>
public sealed class MtpSpeculativeDecoderGdnStateTests : IDisposable
{
    private readonly string _scratch;
    private readonly ITestOutputHelper _output;

    public MtpSpeculativeDecoderGdnStateTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gdn-rollback-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>
    /// Ground truth: plain token-by-token greedy decode of the target model alone, advancing the
    /// model's own recurrent GDN state exactly once per accepted token, matching
    /// <see cref="MtpSpeculativeDecoder"/>'s documented "the accepted sequence is identical to what
    /// argmax-decoding the target model would produce" claim. The MTP head's guesses are real
    /// (untrained-random) weights independent of the trunk, so they disagree with the trunk's own
    /// argmax often (12-way vocab ⇒ ~1/12 chance of agreement per step) — guaranteeing multiple
    /// genuine rejection rounds without any adversarial mock.
    /// </summary>
    [Fact]
    public void DraftAndVerify_RealGdnModel_MatchesPlainGreedyDecode_AcrossRejections()
    {
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "qwen35-mtp-gdn.gguf"), withMtp: true);

        const int startToken = 3;
        const int totalNewTokens = 10;
        const int k = 3;

        List<int> plain = RunPlainGreedy(path, startToken, totalNewTokens);
        (List<int> speculative, int rejectionRounds) = RunSpeculative(path, startToken, totalNewTokens, k);

        _output.WriteLine($"plain:       [{string.Join(", ", plain)}]");
        _output.WriteLine($"speculative: [{string.Join(", ", speculative)}]");
        _output.WriteLine($"rejectionRounds: {rejectionRounds}");

        // The fixture's default layout includes a GDN layer (layer 0), so unless the target model
        // happens to accept literally every drafted token across the whole run (astronomically
        // unlikely given ~1/12 draft/target agreement odds), at least one round must have rejected
        // a draft token. If NO rejection occurred this run isn't discriminating — fail loudly
        // instead of silently passing on a non-reproduction.
        Assert.True(rejectionRounds > 0,
            "Test setup failure: no rejection rounds occurred, so this run cannot discriminate " +
            "between correct and GDN-state-corrupted behavior. Adjust startToken/k/seed.");

        Assert.Equal(plain, speculative);
    }

    /// <summary>
    /// The strong regression test for issue #287's fix: compares RAW LOGITS (not argmax-derived
    /// tokens, which the earlier test above showed can coincidentally match even when the
    /// underlying recurrent state differs — this tiny model's argmax decisions never happened to
    /// be sensitive to the corruption within a 10-token horizon, see
    /// <c>DotLLM.Tests.Unit.Models.Architectures.GdnStateSpeculativeCorruptionTests</c> for the
    /// direct, argmax-independent proof that the corruption mechanism itself is real). After
    /// running <see cref="MtpSpeculativeDecoder"/> through several rounds against a real GDN model
    /// (hitting at least one genuine rejection), forwarding the LAST accepted/corrected token
    /// against the model's own internal, decoder-managed GDN state must produce BYTE-IDENTICAL
    /// logits to forwarding the exact same accepted-token history, one token at a time, through a
    /// fresh model instance.
    /// </summary>
    /// <remarks>
    /// <b>Why the last accepted token specifically.</b> Per the type's own remarks, a corrected or
    /// bonus token "has, by construction, never been forwarded through the trunk as an input" —
    /// it only gets fed once a NEXT round's catchup call runs. So after N rounds, the trunk's
    /// recurrent state has only ever processed <c>acceptedHistory[0..^2]</c>
    /// (<c>acceptedHistory[^1]</c> is still pending). Forwarding
    /// <c>acceptedHistory[^1]</c> now is exactly the "next round's catchup" the decoder would do,
    /// and is the most direct way to observe whether the decoder's internal state matches a clean
    /// serial replay of the identical token sequence.
    /// </remarks>
    [Fact]
    public void DraftAndVerify_AfterRejection_NextTokenLogitsMatchCleanReplay_RawFloats()
    {
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "qwen35-mtp-gdn-logits.gguf"), withMtp: true);

        const int startToken = 3;
        const int k = 3;
        const int rounds = 3;

        List<int> acceptedHistory;
        int lastTokenPosition;
        float[] nextLogitsFromDecoder;
        bool anyRejection;

        // ── Run the (fixed) decoder through several rounds, then forward the still-pending last
        //    token against the SAME model instance's internal, decoder-managed GDN state — exactly
        //    what a hypothetical next round's catchup call would do. ──
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
            var decoder = new MtpSpeculativeDecoder(greedy: true);
            var pipeline = new SamplerPipeline(new DotLLM.Core.Configuration.InferenceOptions { Temperature = 0f });

            var generatedIds = new List<int> { startToken };
            using var kvCache = new SimpleKvCache(
                model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using var mtpState = model.CreateMtpState()!;

            int position = 0;
            anyRejection = false;
            Span<int> outputBuffer = stackalloc int[k + 1];
            for (int r = 0; r < rounds; r++)
            {
                // Leave one spare slot for the trailing forward below.
                int kThisRound = Math.Min(k, config.MaxSequenceLength - position - 2);
                if (kThisRound <= 0) break;

                var result = decoder.DraftAndVerify(
                    model, kvCache, mtpState, pipeline, generatedIds,
                    constraint: null, position, vocabSize: config.VocabSize, numCandidates: kThisRound, outputBuffer);

                if (result.AcceptedCount <= result.DraftedCount)
                    anyRejection = true;

                _output.WriteLine(
                    $"[logits-test] round {r} @ position={position} k={kThisRound} accepted={result.AcceptedCount} " +
                    $"drafted={result.DraftedCount} out=[{string.Join(",", outputBuffer.Slice(0, result.AcceptedCount).ToArray())}]");

                for (int i = 0; i < result.AcceptedCount; i++)
                    generatedIds.Add(outputBuffer[i]);

                position += result.AcceptedCount;
            }

            Assert.True(anyRejection,
                "Test setup failure: no rejection occurred across the rounds run, so this comparison " +
                "cannot discriminate between correct and GDN-state-corrupted behavior.");

            acceptedHistory = generatedIds;
            lastTokenPosition = position; // == acceptedHistory.Count - 1; the pending last token's own slot.
            _output.WriteLine($"[logits-test] acceptedHistory=[{string.Join(",", acceptedHistory)}] lastTokenPosition={lastTokenPosition}");

            using ITensor nextLogits = model.Forward([acceptedHistory[^1]], [lastTokenPosition], deviceId: -1, kvCache);
            unsafe
            {
                var span = new ReadOnlySpan<float>((void*)nextLogits.DataPointer, config.VocabSize);
                nextLogitsFromDecoder = span.ToArray();
            }
        }

        // ── Clean reference: fresh model instance, forward EXACTLY the same accepted-token history,
        //    one token at a time — no decoder involved at all. ──
        float[] nextLogitsClean;
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
            using var kvCache = new SimpleKvCache(
                model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);

            ITensor? last = null;
            for (int pos = 0; pos < acceptedHistory.Count; pos++)
            {
                last?.Dispose();
                last = model.Forward([acceptedHistory[pos]], [pos], deviceId: -1, kvCache);
            }

            using ITensor probeLogits = last!;
            unsafe
            {
                var span = new ReadOnlySpan<float>((void*)probeLogits.DataPointer, config.VocabSize);
                nextLogitsClean = span.ToArray();
            }
        }

        Assert.Equal(nextLogitsClean.Length, nextLogitsFromDecoder.Length);
        for (int i = 0; i < nextLogitsClean.Length; i++)
            Assert.Equal(nextLogitsClean[i], nextLogitsFromDecoder[i]); // byte-identical float compare
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static List<int> RunPlainGreedy(string path, int startToken, int totalNewTokens)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);

        using var kvCache = new SimpleKvCache(
            model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);

        var seq = new List<int> { startToken };
        int token = startToken;
        for (int pos = 0; pos < totalNewTokens; pos++)
        {
            using ITensor logits = model.Forward([token], [pos], deviceId: -1, kvCache);
            int argmax = ArgMax(logits, config.VocabSize);
            seq.Add(argmax);
            token = argmax;
        }

        return seq;
    }

    private (List<int> sequence, int rejectionRounds) RunSpeculative(
        string path, int startToken, int totalNewTokens, int k)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);

        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new DotLLM.Core.Configuration.InferenceOptions { Temperature = 0f });

        var generatedIds = new List<int> { startToken };
        using var kvCache = new SimpleKvCache(
            model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using var mtpState = model.CreateMtpState()!;

        // NOTE: deliberately NO manual prefill forward here (unlike MtpSpeculativeDecoderTests'
        // mock-model harness). DraftAndVerify's own "catchup" forward already seeds both the
        // KV-cache and the trunk's hidden state for `lastToken` at `position` on round 1 (see the
        // type remarks) — a real recurrent (GDN) model must not be forwarded through the SAME
        // token/position twice, since unlike the position-indexed KV-cache, GDN state has no
        // position addressing and a second pass would double-advance it. The mock-model test's
        // manual prefill is harmless there only because that mock's logits are a pure function of
        // the input token, independent of any accumulated state.
        int position = 0;
        int rejectionRounds = 0;
        Span<int> outputBuffer = stackalloc int[k + 1];
        int guard = 0;
        while (generatedIds.Count - 1 < totalNewTokens && guard++ < totalNewTokens * 4)
        {
            int kThisRound = Math.Min(k, config.MaxSequenceLength - position - 1);
            if (kThisRound <= 0) break;

            var result = decoder.DraftAndVerify(
                model, kvCache, mtpState, pipeline, generatedIds,
                constraint: null, position, vocabSize: config.VocabSize, numCandidates: kThisRound, outputBuffer);

            Assert.True(result.AcceptedCount > 0, "Every round must accept at least the corrected/bonus token.");
            if (result.AcceptedCount <= result.DraftedCount)
                rejectionRounds++;

            _output.WriteLine(
                $"round @ position={position} k={kThisRound} accepted={result.AcceptedCount} " +
                $"drafted={result.DraftedCount} out=[{string.Join(",", outputBuffer.Slice(0, result.AcceptedCount).ToArray())}]");

            for (int i = 0; i < result.AcceptedCount && generatedIds.Count - 1 < totalNewTokens; i++)
                generatedIds.Add(outputBuffer[i]);

            position += result.AcceptedCount;
        }

        var trimmed = generatedIds.Take(totalNewTokens + 1).ToList();
        return (trimmed, rejectionRounds);
    }

    private static int ArgMax(ITensor tensor, int vocabSize)
    {
        unsafe
        {
            var span = new ReadOnlySpan<float>((void*)tensor.DataPointer, vocabSize);
            int best = 0;
            for (int i = 1; i < span.Length; i++)
                if (span[i] > span[best]) best = i;
            return best;
        }
    }
}
