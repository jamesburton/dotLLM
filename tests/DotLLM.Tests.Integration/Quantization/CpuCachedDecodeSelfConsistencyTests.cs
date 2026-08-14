using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Quantization;

/// <summary>
/// Scores the CPU reference against ITSELF on the cached <c>seqLen == 1</c> step: the same context
/// is evaluated once through the KV cache and once as a full uncached re-prefill. Same device,
/// weights, kernels and tokens, so the two must agree to float noise — a divergence localises a
/// defect to the CPU cached-decode path without any cross-backend comparison to confound it.
/// </summary>
[Collection("QuantLadder")]
public sealed unsafe class CpuCachedDecodeSelfConsistencyTests(QuantLadderFixture ladder, ITestOutputHelper output)
{
    [Fact]
    [Trait("Category", "Fixtures")]
    public void CachedStep_MatchesUncachedReprefill_OnEveryFixture()
    {
        foreach (QuantLadderEntry entry in ladder.Available)
        {
            using GgufFile gguf = GgufFile.Open(entry.FilePath);
            ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            ITokenizer tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            IModel model = ModelLoader.CreateCpuModelFromGguf(gguf, config, new ThreadingConfig(0));

            try
            {
                double worst = 1.0;
                int worstPrompt = -1;
                double worstSeedCos = 1.0;
                int worstSeedPrompt = -1;

                for (int i = 0; i < CrossBackendQuantGateTests.DecodePrompts.Length; i++)
                {
                    int[] tokens = tokenizer.Encode(CrossBackendQuantGateTests.DecodePrompts[i]);
                    var positions = new int[tokens.Length];
                    for (int p = 0; p < positions.Length; p++)
                        positions[p] = p;

                    model.ResetSequenceState();

                    // Arm A — prefill through the cache, then one cached seqLen == 1 step.
                    var cache = new DotLLM.Engine.KvCache.SimpleKvCache(
                        KvGeometry.FromConfig(config), tokens.Length + 1);
                    float[] cached;
                    int seed;
                    try
                    {
                        using (ITensor prefill = model.Forward(tokens, positions, -1, cache))
                            seed = ArgMax(LastRow(prefill, config.VocabSize));

                        using ITensor step = model.Forward([seed], [tokens.Length], -1, cache);
                        cached = LastRow(step, config.VocabSize);
                    }
                    finally
                    {
                        cache.Dispose();
                    }

                    // Arm B — the same context, uncached, in one forward. seqLen > 1, so this takes
                    // the GEMM path where arm A took the fused GEMV one.
                    model.ResetSequenceState();
                    int[] full = [.. tokens, seed];
                    var fullPos = new int[full.Length];
                    for (int p = 0; p < fullPos.Length; p++)
                        fullPos[p] = p;

                    using ITensor reprefill = model.Forward(full, fullPos, -1);
                    float[] uncached = LastRow(reprefill, config.VocabSize);

                    double cos = Cosine(cached, uncached);
                    if (cos < worst)
                    {
                        worst = cos;
                        worstPrompt = i;
                    }

                    // Arm C — the SAME cached step, seeded with the runner-up token instead of the
                    // argmax. This is exactly what happens when two backends disagree about the
                    // prefill argmax: leg 3 then compares logits from two different contexts.
                    model.ResetSequenceState();
                    var cache2 = new DotLLM.Engine.KvCache.SimpleKvCache(
                        KvGeometry.FromConfig(config), tokens.Length + 1);
                    try
                    {
                        int runnerUp;
                        using (ITensor prefill = model.Forward(tokens, positions, -1, cache2))
                            runnerUp = RunnerUp(LastRow(prefill, config.VocabSize), seed);

                        using ITensor step2 = model.Forward([runnerUp], [tokens.Length], -1, cache2);
                        double seedCos = Cosine(cached, LastRow(step2, config.VocabSize));
                        if (seedCos < worstSeedCos)
                        {
                            worstSeedCos = seedCos;
                            worstSeedPrompt = i;
                        }
                    }
                    finally
                    {
                        cache2.Dispose();
                    }
                }

                output.WriteLine(
                    $"SELFCOS {entry.Type,-10} worst={worst:F6} (1-cos={1 - worst:E3}) prompt={worstPrompt} " +
                    $"| SEEDFLIP worst={worstSeedCos:F6} prompt={worstSeedPrompt}");
            }
            finally
            {
                model.Dispose();
            }
        }
    }

    private static int RunnerUp(float[] row, int exclude)
    {
        int best = exclude == 0 ? 1 : 0;
        for (int v = 0; v < row.Length; v++)
        {
            if (v != exclude && row[v] > row[best])
                best = v;
        }

        return best;
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int v = 1; v < row.Length; v++)
        {
            if (row[v] > row[best])
                best = v;
        }

        return best;
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }

        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    private static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        long total = logits.ElementCount;
        var row = new float[vocab];
        var source = new ReadOnlySpan<float>(
            (void*)(logits.DataPointer + (nint)(total - vocab) * sizeof(float)), vocab);
        source.CopyTo(row);
        return row;
    }
}
