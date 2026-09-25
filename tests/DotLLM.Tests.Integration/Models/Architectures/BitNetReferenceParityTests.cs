using System.Text;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// External ground-truth gate for the BitNet b1.58 (I2_S ternary) forward: dotLLM's greedy
/// decode must agree with <b>Microsoft's official BitNet reference</b> (bitnet.cpp — the
/// <c>microsoft/BitNet</c> llama.cpp fork) running the <i>identical</i>
/// <c>ggml-model-i2_s.gguf</c> weights.
/// </summary>
/// <remarks>
/// <para>
/// The existing <see cref="BitNetAccuracyTests"/> establishes a self-computed CPU perplexity and
/// CPU↔CUDA agreement — both <i>internal</i> consistency signals (two of our own implementations
/// agreeing). This test adds the missing <i>external</i> corroboration: a fully independent C++
/// implementation (Microsoft's own reference for this model) on the same quantized weights produces
/// the same greedy continuations, confirming the I2_S forward is correct, not just self-consistent.
/// </para>
/// <para>
/// <b>Why greedy-agreement and not a perplexity scalar?</b> A direct perplexity comparison is not
/// clean here: (a) <c>llama-perplexity</c> uses sliding-window second-half scoring and requires
/// ≥2× the context length in tokens, whereas dotLLM's accuracy test scores every token from a
/// single full-context window; and (b) the reference always prepends a BOS while dotLLM's
/// <c>Encode</c> does not. Greedy next-token agreement on the same weights sidesteps both: it is a
/// behavioural equality check robust to the BOS and PPL-definition differences. (The reference's
/// answers were also BOS-insensitive in practice — they held with and without leading BOS.)
/// </para>
/// <para>
/// <b>Reference capture (2026-06-24):</b> Microsoft BitNet fork
/// <c>build/bin/Release/llama-cli.exe -m ggml-model-i2_s.gguf -p "&lt;prompt&gt;" -n 8 --temp 0</c>
/// (greedy, temp 0) on the same cached GGUF this fixture downloads. Captured continuations:
/// "…is Paris", "…is Tokyo", "…hydrogen and oxygen", "…was George Washington", "…hot is cold".
/// The expected substring in each <see cref="Probes"/> entry is the salient answer token(s).
/// Assertions are substring (case-insensitive) rather than token-exact so a benign
/// tokenization/whitespace difference does not flap; the answer content is the correctness signal.
/// </para>
/// <para>
/// The model is loaded once and reused across all probes (the GGUF is ~1.2 GB; per-probe reloads
/// dominate the wall-clock). Each probe greedy-decodes from a fresh full-context prefill.
/// </para>
/// </remarks>
[Collection("BitNetModel")]
public sealed class BitNetReferenceParityTests
{
    private readonly BitNetModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public BitNetReferenceParityTests(BitNetModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    // (prompt, expected answer substring) — answers are Microsoft's BitNet reference greedy output
    // on the identical i2_s GGUF (see remarks for the exact capture command).
    private static readonly (string Prompt, string Expected)[] Probes =
    [
        ("The capital of France is", "Paris"),
        ("The capital of Japan is", "Tokyo"),
        ("Water is made of hydrogen and", "oxygen"),
        ("The first president of the United States was", "George Washington"),
        ("The opposite of hot is", "cold"),
    ];

    [Fact]
    public void GreedyDecode_MatchesMicrosoftBitNetReference()
    {
        var gguf = GgufFile.Open(_fixture.FilePath);
        using var _ = gguf;
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        var mismatches = new List<string>();
        foreach (var (prompt, expected) in Probes)
        {
            // Decode enough new tokens to span the longest reference answer ("George Washington").
            string continuation = GreedyDecode(model, tokenizer, prompt, maxNewTokens: 6);
            bool ok = continuation.Contains(expected, StringComparison.OrdinalIgnoreCase);

            _output.WriteLine($"[{(ok ? "ok" : "MISMATCH")}] \"{prompt}\" -> dotLLM: "
                + $"\"{continuation.Trim()}\"  (MS BitNet ref expects: \"{expected}\")");

            if (!ok)
                mismatches.Add($"\"{prompt}\": expected \"{expected}\", got \"{continuation.Trim()}\"");
        }

        Assert.True(mismatches.Count == 0,
            "dotLLM greedy decode diverged from the Microsoft BitNet reference on the identical "
            + $"i2_s weights for {mismatches.Count}/{Probes.Length} probe(s):\n  "
            + string.Join("\n  ", mismatches));
    }

    private static string GreedyDecode(
        TransformerModel model, BpeTokenizer tokenizer, string prompt, int maxNewTokens)
    {
        int vocabSize = model.Config.VocabSize;
        var ids = new List<int>(tokenizer.Encode(prompt));
        var generated = new StringBuilder();

        for (int step = 0; step < maxNewTokens; step++)
        {
            int[] positions = new int[ids.Count];
            for (int i = 0; i < positions.Length; i++)
                positions[i] = i;

            int next;
            using (ITensor logits = model.Forward(ids.ToArray(), positions, deviceId: -1))
            {
                unsafe
                {
                    float* lastRow = (float*)logits.DataPointer + (long)(ids.Count - 1) * vocabSize;
                    next = ArgMax(new ReadOnlySpan<float>(lastRow, vocabSize));
                }
            }

            ids.Add(next);
            generated.Append(tokenizer.DecodeToken(next));
        }

        return generated.ToString();
    }

    private static int ArgMax(ReadOnlySpan<float> span)
        => System.Numerics.Tensors.TensorPrimitives.IndexOfMax(span);
}
