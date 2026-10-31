using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// #1482: <c>--cache-type-k/v q8_0</c> on the CPU Qwen3HybridDense model threw NullReferenceException because the
/// full-attention layers read fp32 pointers from a quantized cache (no such buffers; the pointer is null).
/// </summary>
public sealed class Qwen3HybridDenseQuantizedKvTests
{
    [SkippableFact]
    public void Q8_0_Cache_PrefillAndDecode_TracksF32CacheLogits()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_TEV1_4B_Q4_K_M_GGUF",
            "bartowski", "togethercomputer_Tev1-4B-experimental-GGUF",
            "togethercomputer_Tev1-4B-experimental-Q4_K_M.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Tev1-4B-experimental Q4_K_M GGUF"));

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int[] prompt = tokenizer.Encode("The capital of France is");
        int[] pos = Enumerable.Range(0, prompt.Length).ToArray();
        var geometry = KvGeometry.FromConfig(config);

        using var f32 = new SimpleKvCache(geometry, prompt.Length + 4);
        using var q8 = new QuantizedKvCache(geometry, prompt.Length + 4, KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);

        float[] a = LastRow(model.Forward(prompt, pos, -1, f32), config.VocabSize);
        int next = ArgMax(a);
        float[] a2 = LastRow(model.Forward([next], [prompt.Length], -1, f32), config.VocabSize);

        // The model owns one recurrent (GDN) state; rewind it so the q8 run starts from the same point.
        model.ResetSequenceState();
        float[] b = LastRow(model.Forward(prompt, pos, -1, q8), config.VocabSize);
        float[] b2 = LastRow(model.Forward([next], [prompt.Length], -1, q8), config.VocabSize);

        Assert.True(Cosine(a, b) > 0.99, $"prefill cosine {Cosine(a, b)}");
        Assert.Equal(next, ArgMax(b));
        Assert.True(Cosine(a2, b2) > 0.99, $"decode cosine {Cosine(a2, b2)}");
    }

    private static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        using (logits)
        {
            int rows = logits.Shape[0];
            return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * vocab, vocab).ToArray();
        }
    }

    private static int ArgMax(float[] x)
    {
        int best = 0;
        for (int i = 1; i < x.Length; i++) if (x[i] > x[best]) best = i;
        return best;
    }

    private static double Cosine(float[] a, float[] b)
    {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return d / Math.Sqrt(na * nb);
    }
}
