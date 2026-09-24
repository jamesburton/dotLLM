using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Issue #530, end to end: a prompt's logits must not depend on how the prefill was chunked.
/// Splits that contain a one-token chunk used to diverge, because <c>GemmInterleaved</c> ran a
/// different kernel over a different weight layout for <c>n == 1</c> than for <c>n &gt; 1</c>.
/// </summary>
/// <remarks>
/// Compares the full final-position logit vector bit-exactly, not the sampled token: argmax is
/// insensitive to the ~1e-6 drift this produced, which is exactly why the defect survived the
/// greedy-text comparisons that were already in the suite.
/// </remarks>
public static class ChunkedPrefillLogits
{
    /// <summary>Arbitrary ids — the property under test is chunk-invariance, not the prompt.</summary>
    private static readonly int[] Tokens = [1, 337, 6208, 293, 12];

    /// <summary>
    /// Every split of 5 tokens the issue reported, plus [3,2] which was already clean and so acts
    /// as a control that the harness is measuring the right thing.
    /// </summary>
    public static readonly int[] ChunkSizes = [1, 2, 3, 4];

    public static void AssertChunkInvariant(string ggufPath)
    {
        using var gguf = GgufFile.Open(ggufPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);

        float[] singlePass = LastPositionLogits(model, Tokens.Length);

        foreach (int chunk in ChunkSizes)
        {
            float[] chunked = LastPositionLogits(model, chunk);
            Assert.Equal(singlePass.Length, chunked.Length);

            for (int i = 0; i < singlePass.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(singlePass[i]) != BitConverter.SingleToInt32Bits(chunked[i]))
                {
                    Assert.Fail(
                        $"chunk size {chunk}: logit[{i}] = {chunked[i]:R} but the single-pass prefill " +
                        $"gives {singlePass[i]:R} (delta {MathF.Abs(singlePass[i] - chunked[i]):E3}). " +
                        "Prefill chunking must not change a token's logits.");
                }
            }
        }
    }

    /// <summary>
    /// Prefills <see cref="Tokens"/> in chunks of <paramref name="chunkSize"/> through one KV cache
    /// and returns the logit row for the last token.
    /// </summary>
    private static unsafe float[] LastPositionLogits(IModel model, int chunkSize)
    {
        var config = model.Config;
        using var cache = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim,
            maxSeqLen: Tokens.Length + 8);

        float[] last = [];
        for (int start = 0; start < Tokens.Length; start += chunkSize)
        {
            int len = Math.Min(chunkSize, Tokens.Length - start);
            int[] ids = Tokens[start..(start + len)];
            int[] positions = new int[len];
            for (int i = 0; i < len; i++) positions[i] = start + i;

            using ITensor logits = model.Forward(ids, positions, deviceId: -1, cache);

            int vocab = config.VocabSize;
            var span = new ReadOnlySpan<float>((void*)logits.DataPointer, (int)logits.ElementCount);
            last = span.Slice((len - 1) * vocab, vocab).ToArray();
        }

        return last;
    }
}

/// <summary>Q4_K_M — the quantization the issue reported.</summary>
[Collection("Q4KModel")]
public class ChunkedPrefillLogitsQ4KTests(Q4KModelFixture fixture)
{
    [Fact]
    public void ChunkedPrefill_LogitsAreChunkSizeInvariant()
        => ChunkedPrefillLogits.AssertChunkInvariant(fixture.FilePath);
}

/// <summary>Q5_0 — a second, structurally different quantization through the same dispatch.</summary>
[Collection("Q5_0Model")]
public class ChunkedPrefillLogitsQ5_0Tests(Q5_0ModelFixture fixture)
{
    [Fact]
    public void ChunkedPrefill_LogitsAreChunkSizeInvariant()
        => ChunkedPrefillLogits.AssertChunkInvariant(fixture.FilePath);
}
