using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Issue #525: splitting a prefill into chunks must not change the logits. A quantized model used
/// to emit different text for the same prompt depending only on how prefill was batched, because
/// attention reduced its softmax rows over the padded KV-cache length rather than over the causally
/// visible prefix.
/// <para>
/// <b>These tests compare logits, not sampled tokens, on purpose.</b> Argmax insensitivity is what
/// hid the defect for months: the pre-existing guard exercised one 10-token prompt at chunk sizes 3
/// and 5, both of which happened to be clean combinations. The discriminating axis is the KV-cache
/// length at the split, so the sweep is prompt length x chunk size.
/// </para>
/// <para>
/// <b>The F32 arm is a sensitivity control, not extra coverage.</b> The load-bearing arm is Q8_0:
/// the broken form moved its final logits row by <c>4.193E-01</c>, while the F32-decoded model --
/// the same weights dequantized outside dotLLM's lineage -- moved by <c>1.049E-05</c>, four orders
/// of magnitude less, and never changed a token. That separation is the control: it shows the Q8_0
/// number is quantization amplifying a ULP rather than the harness measuring itself. (The F32 arm
/// does sit just over the shared budget on the broken form; its raw-logit deltas are ~5e-7
/// *relative* on logits spanning tens, consistent with the 1.7e-7 logprob spread the issue reported
/// after softmax.) Both arms are bit-exact once the fix is in.
/// </para>
/// </summary>
public sealed class ChunkedPrefillInvarianceTests
{
    /// <summary>
    /// Logit agreement budget. The broken form moved the top-1 logprob by 0.043 on Q8_0 — four
    /// orders of magnitude above this — while float reduction noise on a clean path is ~1e-7.
    /// </summary>
    private const float LogitTolerance = 1e-5f;

    private const string LongText =
        "The capital of France is Paris, a city that has served as the political and cultural " +
        "centre of the country for many centuries, drawing writers, painters and students from " +
        "every corner of the continent and beyond, and it remains one of the most visited places " +
        "in the world today because of its museums, its architecture and its food.";

    private readonly ITestOutputHelper _output;

    public ChunkedPrefillInvarianceTests(ITestOutputHelper output) => _output = output;

    private static FixtureLocation Q8Fixture() => TestFixtureResolver.ResolveFile(
        "DOTLLM_SMOLLM_135M_Q8_0_GGUF", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");

    private static FixtureLocation F32Fixture() => TestFixtureResolver.ResolveFile(
        "DOTLLM_SMOLLM_135M_F32_DECODED_GGUF", "QuantFactory", "SmolLM-135M-GGUF",
        "SmolLM-135M.F32-decoded.gguf");

    /// <summary>
    /// The load-bearing arm. Q8_0, swept over prompt length x chunk size. RED before the fix:
    /// a 5-token prompt at chunk sizes 1 and 3 (the two splits that make position 2 attend with a
    /// 3-entry cache instead of a 5-entry one) moved the top-1 logprob by 0.043.
    /// </summary>
    [SkippableFact]
    public void Q8_0_ChunkedPrefillLogits_MatchSinglePass_AcrossPromptLengthAndChunkSize()
    {
        FixtureLocation fixture = Q8Fixture();
        Skip.If(!fixture.Found, fixture.SkipMessage("SmolLM-135M Q8_0 GGUF"));
        AssertSweep(fixture.Path!, "Q8_0");
    }

    /// <summary>
    /// Sensitivity control: the same sweep on an F32 decode of the same weights, produced outside
    /// dotLLM's lineage. Its worst delta on the broken form was 1.049E-05 against Q8_0's 4.193E-01 —
    /// four orders of magnitude — and no token ever changed. That separation is what shows the Q8_0
    /// number is quantization amplifying a reduction-order ULP, not a harness artefact.
    /// </summary>
    [SkippableFact]
    public void F32Decoded_ChunkedPrefillLogits_MatchSinglePass_Control()
    {
        FixtureLocation fixture = F32Fixture();
        Skip.If(!fixture.Found, fixture.SkipMessage("SmolLM-135M F32-decoded GGUF"));
        AssertSweep(fixture.Path!, "F32-decoded");
    }

    /// <summary>
    /// End-to-end acceptance from the issue: for the reported prompt every prefill chunk size must
    /// produce the same tokens and the same pos-0 top-1 logprob. Before the fix chunk 1 and 3 gave
    /// -0.55704731 while 0/2/4/5 gave -0.51405847. Re-measured on this tree (after #501's precise
    /// exp) the broken form gives -0.55717152 vs -0.51418936, with the token sequences diverging
    /// from position 1.
    /// </summary>
    [SkippableFact]
    public void Q8_0_CapitalOfFrance_AllPrefillChunkSizesAgree()
    {
        FixtureLocation fixture = Q8Fixture();
        Skip.If(!fixture.Found, fixture.SkipMessage("SmolLM-135M Q8_0 GGUF"));

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        const string prompt = "The capital of France is";
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 12, RepetitionPenalty = 1.0f };

        int[]? baselineTokens = null;
        float baselineLogProb = 0f;

        for (int chunk = 0; chunk <= 5; chunk++)
        {
            int[] tokens = new TextGenerator(model, tokenizer, prefillChunkSize: chunk)
                .Generate(prompt, options).GeneratedTokenIds;

            int[] promptIds = tokenizer.Encode(prompt);
            float logProb = TopOneLogProb(PrefillLogits(model, config, promptIds, chunk));

            _output.WriteLine($"chunk={chunk} pos0 top1 logprob={logProb:F8} tokens=[{string.Join(',', tokens)}]");

            if (baselineTokens is null)
            {
                baselineTokens = tokens;
                baselineLogProb = logProb;
                continue;
            }

            Assert.Equal(baselineTokens, tokens);
            Assert.True(MathF.Abs(logProb - baselineLogProb) <= 1e-6f,
                $"chunk={chunk}: pos0 top-1 logprob {logProb:F8} vs baseline {baselineLogProb:F8} " +
                $"(delta {MathF.Abs(logProb - baselineLogProb):E3}).");
        }
    }

    private void AssertSweep(string ggufPath, string label)
    {
        using var gguf = GgufFile.Open(ggufPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int[] allIds = tokenizer.Encode(LongText);
        // Short lengths reproduce the reported divergence; the long one crosses the point where a
        // single-pass prefill used to switch attention kernels (seqQ * seqKv * 4 > 8192).
        int[] promptLengths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 64];

        float worst = 0f;
        string worstCase = "none";

        foreach (int len in promptLengths)
        {
            if (len > allIds.Length) continue;
            int[] promptIds = allIds[..len];
            float[] single = PrefillLogits(model, config, promptIds, chunkSize: 0);

            for (int chunk = 1; chunk <= 8; chunk++)
            {
                float[] chunked = PrefillLogits(model, config, promptIds, chunk);
                float delta = MaxAbsDelta(single, chunked);
                if (delta > worst)
                {
                    worst = delta;
                    worstCase = $"len={len} chunk={chunk}";
                }

                Assert.True(delta <= LogitTolerance,
                    $"{label}: prompt length {len}, chunk size {chunk}: final logits row differs from " +
                    $"single-pass prefill by {delta:E3} (budget {LogitTolerance:E3}). Attention must " +
                    $"reduce over the causally visible prefix, not the padded KV-cache length (#525).");
            }
        }

        _output.WriteLine($"{label}: worst max|delta| over the sweep = {worst:E3} ({worstCase})");
    }

    /// <summary>
    /// Prefills <paramref name="promptIds"/> against a fresh KV cache, optionally in chunks of
    /// <paramref name="chunkSize"/>, and returns the final logits row.
    /// </summary>
    private static unsafe float[] PrefillLogits(IModel model, ModelConfig config, int[] promptIds, int chunkSize)
    {
        using var kv = new SimpleKvCache(KvGeometry.FromConfig(config), promptIds.Length + 8);
        int step = chunkSize > 0 ? Math.Min(chunkSize, promptIds.Length) : promptIds.Length;

        float[] last = [];
        for (int start = 0; start < promptIds.Length; start += step)
        {
            int count = Math.Min(step, promptIds.Length - start);
            int[] ids = promptIds[start..(start + count)];
            int[] positions = [.. Enumerable.Range(start, count)];

            using ITensor logits = model.Forward(ids, positions, deviceId: -1, kv);
            int rows = logits.Shape[0];
            last = new ReadOnlySpan<float>(
                (float*)logits.DataPointer + (long)(rows - 1) * config.VocabSize, config.VocabSize).ToArray();
        }
        return last;
    }

    private static float MaxAbsDelta(float[] a, float[] b)
    {
        float worst = 0f;
        for (int i = 0; i < a.Length; i++)
            worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        return worst;
    }

    /// <summary>log-softmax of the argmax entry — the quantity the issue reported moving by 0.043.</summary>
    private static float TopOneLogProb(float[] logits)
    {
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];

        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
            sum += Math.Exp(logits[i] - max);

        return (float)(-Math.Log(sum));
    }
}
