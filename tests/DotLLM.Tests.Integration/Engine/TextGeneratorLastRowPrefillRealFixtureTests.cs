using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Issue #493 on a real checkpoint: after <c>TextGenerator.ForwardPrefill</c> started passing
/// <c>lastTokenLogitsOnly: true</c>, greedy generation must produce exactly the token sequence a
/// hand-rolled prefill-then-decode loop over the same model produces.
/// </summary>
/// <remarks>
/// The CPU <c>TransformerModel</c> does not honour the hint — it has no VRAM ceiling to defend, so
/// it keeps returning <c>[len, vocab]</c>. That makes this test's value twofold and both halves are
/// worth stating plainly:
/// <list type="bullet">
/// <item>it pins the end-to-end invariance the issue asks for (identical output before and after)
/// on a real fixture, which the engine-level fakes cannot claim to have exercised;</item>
/// <item>it pins the CPU backend's <em>documented</em> answer to "does this backend honour the
/// flag" — <b>no</b> — so a future CPU change that silently starts returning one row shows up here
/// as a deliberate decision rather than as a mysterious serving-path regression.</item>
/// </list>
/// The discriminating shape assertions for a backend that DOES honour it live in the unit suite
/// (<c>TextGeneratorLastRowPrefillTests</c>) and, on real hardware, in
/// <c>CudaQwen3HybridDenseLastTokenLogitsOnlyTest</c>.
/// </remarks>
public sealed class TextGeneratorLastRowPrefillRealFixtureTests
{
    private const string Prompt = "The capital of France is";
    private const int MaxTokens = 12;

    [SkippableFact]
    public void SmolLM2_135M_Q8_0_GreedyGeneration_MatchesManualPrefillDecodeLoop()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_SMOLLM2_135M_INSTRUCT_Q8_GGUF",
            "bartowski", "SmolLM2-135M-Instruct-GGUF",
            "SmolLM2-135M-Instruct-Q8_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("SmolLM2-135M-Instruct Q8_0 GGUF"));

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int[] promptIds = tokenizer.Encode(Prompt);
        int[] expected = ManualGreedy(model, config, promptIds, MaxTokens, out int prefillRows);

        // The CPU backend ignores the hint: one row per input position, as documented.
        Assert.Equal(promptIds.Length, prefillRows);

        var generator = new TextGenerator(model, tokenizer);
        var response = generator.Generate(Prompt,
            new InferenceOptions { MaxTokens = MaxTokens, Temperature = 0f, RepetitionPenalty = 1.0f });

        Assert.Equal(expected.Take(response.GeneratedTokenIds.Length), response.GeneratedTokenIds);
    }

    /// <summary>
    /// Prefill in one forward, then greedy-decode one token at a time, reading row
    /// <c>Shape[0] - 1</c> exactly as the engine does. Stops at EOS.
    /// </summary>
    private static unsafe int[] ManualGreedy(IModel model, ModelConfig config, int[] promptIds,
                                             int maxTokens, out int prefillRows)
    {
        int vocab = config.VocabSize;
        using var kv = new SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(config),
                                         promptIds.Length + maxTokens + 1);
        int[] positions = Enumerable.Range(0, promptIds.Length).ToArray();

        var generated = new List<int>(maxTokens);
        int next;
        using (ITensor logits = model.Forward(promptIds, positions, deviceId: -1, kv))
        {
            prefillRows = logits.Shape[0];
            next = ArgMaxLastRow(logits, vocab);
        }
        generated.Add(next);

        for (int i = 1; i < maxTokens; i++)
        {
            int pos = promptIds.Length + i - 1;
            using ITensor logits = model.Forward([next], [pos], deviceId: -1, kv);
            next = ArgMaxLastRow(logits, vocab);
            generated.Add(next);
        }
        return [.. generated];
    }

    private static unsafe int ArgMaxLastRow(ITensor logits, int vocab)
    {
        int rows = logits.Shape[0];
        var row = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * vocab, vocab);
        int best = 0;
        for (int i = 1; i < vocab; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }
}
