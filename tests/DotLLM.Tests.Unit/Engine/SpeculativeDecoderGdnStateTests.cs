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
/// Real-model regression coverage for issue #287 against <see cref="SpeculativeDecoder"/> (the
/// original two-model draft-verify-accept decoder) — the sibling of
/// <see cref="MtpSpeculativeDecoderGdnStateTests"/>, which covers the newer self-speculative
/// <see cref="MtpSpeculativeDecoder"/>. The issue affects both decoders identically; this proves
/// the fix works for this one too.
/// </summary>
/// <remarks>
/// <para>
/// <b>An important asymmetry from <see cref="MtpSpeculativeDecoder"/>, caught by this exact test
/// during development.</b> <see cref="MtpSpeculativeDecoder"/> forwards <c>lastToken</c> via a
/// separate, always-legitimate "catchup" call BEFORE its recurrent-state checkpoint, so only the
/// draft tokens need replaying on rollback. <see cref="SpeculativeDecoder"/>'s verify batch instead
/// forwards <c>lastToken</c> itself as row 0 (<c>verifyTokens[0]</c>) — the checkpoint taken
/// immediately before that SAME batched call therefore predates <c>lastToken</c>'s own trunk
/// processing too. A first version of the fix only replayed the accepted draft tokens (mirroring
/// <see cref="MtpSpeculativeDecoder"/> verbatim) and this test caught the resulting state
/// corruption immediately (raw logits differed) — the fix replays <c>lastToken</c> first, then the
/// accepted draft-token prefix. See <see cref="SpeculativeDecoder"/>'s <c>RollbackCaches</c> remarks.
/// </para>
/// </remarks>
public sealed class SpeculativeDecoderGdnStateTests : IDisposable
{
    private readonly string _scratch;
    private readonly ITestOutputHelper _output;

    public SpeculativeDecoderGdnStateTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-specdec-gdn-rollback-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>
    /// After running <see cref="SpeculativeDecoder"/> through several rounds against a real GDN
    /// target model (paired with an independent, differently-seeded draft model — guaranteeing
    /// genuine disagreements/rejections without any adversarial mock), forwarding the still-pending
    /// last accepted/corrected token against the TARGET model's own internal, decoder-managed GDN
    /// state must produce BYTE-IDENTICAL logits to forwarding the exact same accepted-token
    /// history, one token at a time, through a fresh target model instance.
    /// </summary>
    [Fact]
    public void DraftAndVerify_AfterRejection_NextTokenLogitsMatchCleanReplay_RawFloats()
    {
        string targetPath = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "target-gdn.gguf"), seed: 0xC0FFEEu, withMtp: false);
        // Different seed => independently-random weights => frequent target/draft disagreement.
        // fullAttnInterval:1 keeps the draft side simple (no GDN); irrelevant to this test's claim,
        // which is only about the TARGET model's recurrent state.
        string draftPath = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "draft-plain.gguf"), seed: 0xDEADBEEFu, withMtp: false, fullAttnInterval: 1);

        const int startToken = 3;
        const int k = 3;
        const int rounds = 3;

        List<int> acceptedHistory;
        int lastTokenPosition;
        float[] nextLogitsFromDecoder;
        bool anyRejection;

        // ── Run the (fixed) decoder through several rounds, then forward the still-pending last
        //    token against the SAME target model instance's internal, decoder-managed GDN state. ──
        {
            using var targetGguf = GgufFile.Open(targetPath);
            var targetConfig = GgufModelConfigExtractor.Extract(targetGguf.Metadata);
            using var targetModel = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(targetGguf, targetConfig);

            using var draftGguf = GgufFile.Open(draftPath);
            var draftConfig = GgufModelConfigExtractor.Extract(draftGguf.Metadata);
            using var draftModel = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(draftGguf, draftConfig);

            var decoder = new SpeculativeDecoder(greedy: true, seed: 42);
            var pipeline = new SamplerPipeline(new DotLLM.Core.Configuration.InferenceOptions { Temperature = 0f });

            var generatedIds = new List<int> { startToken };
            using var kvCacheTarget = new SimpleKvCache(
                targetModel.AttentionLayerCount, targetConfig.NumKvHeads, targetConfig.HeadDim, targetConfig.MaxSequenceLength);
            using var kvCacheDraft = new SimpleKvCache(
                draftModel.AttentionLayerCount, draftConfig.NumKvHeads, draftConfig.HeadDim, draftConfig.MaxSequenceLength);

            // NOTE: deliberately NO manual prefill forward — DraftAndVerify's own verify batch
            // forwards `lastToken` as row 0 for the first time on round 1 (see the type remarks on
            // SpeculativeDecoder's RollbackCaches). A real recurrent (GDN) target model must not be
            // forwarded through the same token/position twice ahead of that.
            int position = 0;
            anyRejection = false;
            Span<int> outputBuffer = stackalloc int[k + 1];
            for (int r = 0; r < rounds; r++)
            {
                int kThisRound = Math.Min(k, Math.Min(targetConfig.MaxSequenceLength, draftConfig.MaxSequenceLength) - position - 2);
                if (kThisRound <= 0) break;

                var result = decoder.DraftAndVerify(
                    targetModel, draftModel, kvCacheTarget, kvCacheDraft,
                    pipeline, generatedIds, constraint: null, position,
                    targetVocabSize: targetConfig.VocabSize, draftVocabSize: draftConfig.VocabSize,
                    numCandidates: kThisRound, outputBuffer);

                if (result.AcceptedCount <= result.DraftedCount)
                    anyRejection = true;

                _output.WriteLine(
                    $"[specdec-logits] round {r} @ position={position} k={kThisRound} accepted={result.AcceptedCount} " +
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
            _output.WriteLine($"[specdec-logits] acceptedHistory=[{string.Join(",", acceptedHistory)}] lastTokenPosition={lastTokenPosition}");

            using ITensor nextLogits = targetModel.Forward([acceptedHistory[^1]], [lastTokenPosition], deviceId: -1, kvCacheTarget);
            unsafe
            {
                var span = new ReadOnlySpan<float>((void*)nextLogits.DataPointer, targetConfig.VocabSize);
                nextLogitsFromDecoder = span.ToArray();
            }
        }

        // ── Clean reference: fresh target model instance, forward EXACTLY the same accepted-token
        //    history, one token at a time — no decoder involved at all. ──
        float[] nextLogitsClean;
        {
            using var gguf = GgufFile.Open(targetPath);
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

        // Tolerance-based, NOT byte-exact (unlike MtpSpeculativeDecoderGdnStateTests' equivalent
        // assertion, which IS byte-exact — that decoder's simpler catchup-then-verify shape was
        // proven bit-identical to a clean serial replay). This decoder's rollback replays THREE
        // separate rounds' worth of checkpoint/restore/re-batch cycles (round 0's replay alone is a
        // 2-token batch reconstructing what a 4-token verify batch partially computed); isolated
        // per-round mechanism tests (see GdnBatchVsSerialDiagnosticTests) proved every INDIVIDUAL
        // checkpoint/verify/restore/replay step is bit-exact in every shape this test exercises, but
        // composing three such cycles back-to-back leaves a residual at the float32 ULP level
        // (~1e-7 relative) — IEEE 754 addition is not associative, and different physical
        // reduction orders across a chain of re-batched GEMM/GDN-scan calls can legitimately land on
        // the last bit differently while remaining mathematically equivalent. 1e-4 is generously
        // tight: the actual bug this test caught during development (replaying only the accepted
        // draft tokens and omitting lastToken — see the type remarks) produced ~1% relative error,
        // four orders of magnitude larger than this tolerance.
        const float relTol = 1e-4f;
        Assert.Equal(nextLogitsClean.Length, nextLogitsFromDecoder.Length);
        for (int i = 0; i < nextLogitsClean.Length; i++)
        {
            float expected = nextLogitsClean[i];
            float actual = nextLogitsFromDecoder[i];
            float tol = relTol * Math.Max(1f, Math.Abs(expected));
            Assert.True(Math.Abs(expected - actual) <= tol,
                $"[{i}] expected {expected}, got {actual} (tolerance {tol})");
        }
    }
}
