using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Byte-identical parity tests for <see cref="TransformerModel.ForwardBatch"/>.
/// Validates that the lm_head-batched override produces per-sequence logits
/// equal to running <see cref="TransformerModel.Forward(System.ReadOnlySpan{int}, System.ReadOnlySpan{int}, int, DotLLM.Core.Attention.IKvCache?, DotLLM.Core.Lora.ILoraAdapter?)"/>
/// per sequence and concatenating the results.
/// </summary>
/// <remarks>
/// Phase 5a only fuses the lm_head — intra-block matmuls still run per-seq.
/// The lm_head <c>GemmInterleaved</c> accumulator order does not depend on the
/// batched seqLen (each output element is an independent dot product over a
/// fixed-length hidden dimension), so per-row results are bit-exact regardless
/// of whether 1 or N sequences flow through the GEMM. Hence the tolerance is
/// strict <see cref="float.Equals(float)"/> rather than an abs/rel envelope.
///
/// Uses the cached SmolLM-135M Q8_0 GGUF if available
/// (<c>~/.dotllm/test-cache/QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q8_0.gguf</c>).
/// Skipped otherwise.
/// </remarks>
public sealed class TransformerModelForwardBatchTests
{
    private static readonly string CachedSmolLmPath =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "test-cache", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");

    [SkippableFact]
    public void ForwardBatch_SingleSequence_EqualsForwardLoop()
    {
        Skip.IfNot(File.Exists(CachedSmolLmPath), $"SmolLM-135M GGUF not cached at {CachedSmolLmPath}");

        using var gguf = GgufFile.Open(CachedSmolLmPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);

        int[] tokens = [10, 11, 12];
        int[] positions = [0, 1, 2];

        // Per-seq Forward → reference logits
        using var kvRef = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using ITensor refLogits = model.Forward(tokens, positions, deviceId: -1, kvRef);
        float[] refFlat = CopyLogits(refLogits);

        // ForwardBatch with a single request → must equal the loop path
        using var kvBatch = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        var request = new SequenceForwardRequest
        {
            TokenIds = tokens,
            Positions = positions,
            KvCache = kvBatch,
        };
        var results = model.ForwardBatch(new[] { request }, deviceId: -1);
        Assert.Single(results);
        try
        {
            float[] batchFlat = CopyLogits(results[0]);
            AssertBitEqual(refFlat, batchFlat, $"SingleSeq: {tokens.Length} tokens × {config.VocabSize} vocab");
        }
        finally
        {
            foreach (var t in results) t.Dispose();
        }
    }

    [SkippableFact]
    public void ForwardBatch_TwoSequences_DifferentPrompts_MatchesPerSeqLoop()
    {
        Skip.IfNot(File.Exists(CachedSmolLmPath), $"SmolLM-135M GGUF not cached at {CachedSmolLmPath}");

        using var gguf = GgufFile.Open(CachedSmolLmPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);

        int[] tokensA = [10, 11, 12, 13];
        int[] positionsA = [0, 1, 2, 3];
        int[] tokensB = [20, 21];
        int[] positionsB = [0, 1];

        // Per-seq references — each call uses a FRESH KV cache because the per-seq
        // ForwardBatch path also uses a fresh cache per sequence.
        float[] refA, refB;
        {
            using var kvA = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using ITensor logitsA = model.Forward(tokensA, positionsA, deviceId: -1, kvA);
            refA = CopyLogits(logitsA);
        }
        {
            using var kvB = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using ITensor logitsB = model.Forward(tokensB, positionsB, deviceId: -1, kvB);
            refB = CopyLogits(logitsB);
        }

        // Batched path
        using var kvA2 = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using var kvB2 = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        var requests = new[]
        {
            new SequenceForwardRequest { TokenIds = tokensA, Positions = positionsA, KvCache = kvA2 },
            new SequenceForwardRequest { TokenIds = tokensB, Positions = positionsB, KvCache = kvB2 },
        };

        var results = model.ForwardBatch(requests, deviceId: -1);
        Assert.Equal(2, results.Count);
        try
        {
            float[] batchA = CopyLogits(results[0]);
            float[] batchB = CopyLogits(results[1]);

            Assert.Equal(tokensA.Length, results[0].Shape[0]);
            Assert.Equal(tokensB.Length, results[1].Shape[0]);

            AssertBitEqual(refA, batchA, $"SeqA: {tokensA.Length} tokens × {config.VocabSize} vocab");
            AssertBitEqual(refB, batchB, $"SeqB: {tokensB.Length} tokens × {config.VocabSize} vocab");
        }
        finally
        {
            foreach (var t in results) t.Dispose();
        }
    }

    [SkippableFact]
    public void ForwardBatch_FourSequences_MixedLengths_MatchesPerSeqLoop()
    {
        Skip.IfNot(File.Exists(CachedSmolLmPath), $"SmolLM-135M GGUF not cached at {CachedSmolLmPath}");

        using var gguf = GgufFile.Open(CachedSmolLmPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);

        // 1 decode-step (1 token), 1 prefill (5 tokens), 1 longer prefill (8 tokens),
        // and 1 medium (3 tokens). Exercises the Σ N_i = 17 stacked lm_head GEMM.
        int[][] tokenSets =
        [
            [30],
            [40, 41, 42, 43, 44],
            [50, 51, 52, 53, 54, 55, 56, 57],
            [60, 61, 62],
        ];

        var refLogits = new float[tokenSets.Length][];
        for (int i = 0; i < tokenSets.Length; i++)
        {
            int[] positions = Enumerable.Range(0, tokenSets[i].Length).ToArray();
            using var kv = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using ITensor logits = model.Forward(tokenSets[i], positions, deviceId: -1, kv);
            refLogits[i] = CopyLogits(logits);
        }

        var caches = new SimpleKvCache[tokenSets.Length];
        var requests = new SequenceForwardRequest[tokenSets.Length];
        try
        {
            for (int i = 0; i < tokenSets.Length; i++)
            {
                caches[i] = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
                requests[i] = new SequenceForwardRequest
                {
                    TokenIds = tokenSets[i],
                    Positions = Enumerable.Range(0, tokenSets[i].Length).ToArray(),
                    KvCache = caches[i],
                };
            }

            var results = model.ForwardBatch(requests, deviceId: -1);
            Assert.Equal(tokenSets.Length, results.Count);

            try
            {
                for (int i = 0; i < tokenSets.Length; i++)
                {
                    Assert.Equal(tokenSets[i].Length, results[i].Shape[0]);
                    float[] batchLogits = CopyLogits(results[i]);
                    AssertBitEqual(refLogits[i], batchLogits,
                        $"Seq[{i}]: {tokenSets[i].Length} tokens × {config.VocabSize} vocab");
                }
            }
            finally
            {
                foreach (var t in results) t.Dispose();
            }
        }
        finally
        {
            foreach (var kv in caches) kv?.Dispose();
        }
    }

    [Fact]
    public void ForwardBatch_EmptyRequests_ReturnsEmpty()
    {
        Skip.IfNot(File.Exists(CachedSmolLmPath), $"SmolLM-135M GGUF not cached at {CachedSmolLmPath}");

        using var gguf = GgufFile.Open(CachedSmolLmPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);

        var results = model.ForwardBatch(System.Array.Empty<SequenceForwardRequest>(), deviceId: -1);
        Assert.Empty(results);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static float[] CopyLogits(ITensor logits)
    {
        int n = checked((int)logits.Shape.ElementCount);
        var dest = new float[n];
        unsafe
        {
            new System.Span<float>((void*)logits.DataPointer, n).CopyTo(dest);
        }
        return dest;
    }

    private static void AssertBitEqual(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        int mismatches = 0;
        float maxAbs = 0;
        int firstBad = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            if (!expected[i].Equals(actual[i]))
            {
                mismatches++;
                float diff = MathF.Abs(expected[i] - actual[i]);
                if (diff > maxAbs) { maxAbs = diff; firstBad = i; }
            }
        }
        Assert.True(mismatches == 0,
            $"[{label}] {mismatches}/{expected.Length} logits diverged from per-seq Forward; "
            + $"maxAbs={maxAbs:G9}, first divergent index={firstBad} "
            + $"(expected={(firstBad >= 0 ? expected[firstBad].ToString("R") : "n/a")}, "
            + $"actual={(firstBad >= 0 ? actual[firstBad].ToString("R") : "n/a")})");
    }
}
