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

    /// <summary>
    /// Returns the max absolute logit delta against the single-pass prefill, per chunk size.
    /// </summary>
    public static IReadOnlyDictionary<int, float> MeasureChunkDeltas(string ggufPath)
    {
        using var gguf = GgufFile.Open(ggufPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);

        float[] singlePass = LastPositionLogits(model, Tokens.Length);
        var deltas = new Dictionary<int, float>();

        foreach (int chunk in ChunkSizes)
        {
            float[] chunked = LastPositionLogits(model, chunk);
            Assert.Equal(singlePass.Length, chunked.Length);

            float worst = 0;
            for (int i = 0; i < singlePass.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(singlePass[i]) != BitConverter.SingleToInt32Bits(chunked[i]))
                    worst = MathF.Max(worst, MathF.Abs(singlePass[i] - chunked[i]));
            }

            deltas[chunk] = worst;
        }

        return deltas;
    }

    public static void AssertChunkInvariant(string ggufPath)
    {
        var deltas = MeasureChunkDeltas(ggufPath);
        if (deltas.Values.All(d => d == 0f))
            return;

        string report = string.Join(", ", deltas.Select(kv =>
            kv.Value == 0f ? $"[{kv.Key}: clean]" : $"[{kv.Key}: {kv.Value:E3}]"));
        Assert.Fail($"Prefill chunking changed the final-position logits. Max |delta| per chunk size: {report}. " +
                    "A token's logits must not depend on how many tokens shared its forward pass.");
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

/// <summary>
/// Q8_0 carries a gap #530 does not close, but only at kernel level: its R4 and row-major kernels
/// are not bit-identical (see
/// <c>MatMulR4BatchInvarianceTests.Q8_0_RowMajorVsRepacked_StillDiverges</c>), and the fused decode
/// QKV path (<c>FusedDecodeGemv3</c>) reads the original row-major weights while prefill QKV runs
/// the R4 kernels. On SmolLM-135M that never surfaces — measured 0.0 delta at every chunk size —
/// so this arm records the measurement rather than asserting a symptom no model here exhibits.
/// A model whose projection rows clear <c>InterleavedMinRowBytes</c> could still expose it.
/// </summary>
[Collection("SmallModel")]
public class ChunkedPrefillLogitsQ8_0Tests(SmallModelFixture fixture)
{
    [Fact]
    public void ChunkedPrefill_Q8_0_DivergenceIsRecorded()
    {
        var deltas = ChunkedPrefillLogits.MeasureChunkDeltas(fixture.FilePath);
        foreach (var (chunk, delta) in deltas)
            Console.WriteLine($"[#530] Q8_0 chunk {chunk}: max |delta| = {delta:E3}");

        // Bound only — Q8_0 chunk-invariance needs the fused decode path taught the R4 layout.
        foreach (var (chunk, delta) in deltas)
            Assert.True(delta < 1f, $"Q8_0 chunk {chunk} diverged by {delta:E3}, far beyond accumulation noise.");
    }
}
