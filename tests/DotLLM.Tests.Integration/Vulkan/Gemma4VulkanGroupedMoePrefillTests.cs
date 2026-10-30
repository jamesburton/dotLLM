using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #773: real Gemma-4-26B-A4B on Vulkan, grouped-by-expert coopmat MoE prefill vs the scalar indexed prefill, SAME loaded model
/// (the path is switched at runtime through <see cref="VulkanTransformerModel.Gemma4GroupedMoeEnabled"/>), with the CPU forward as the
/// independent oracle. Reports the prefill speedup and, per prompt length, the last-row logit envelope of grouped-vs-scalar and of each
/// vs CPU, plus top-k agreement. (Wikitext perplexity is NOT used: it is ~1e4-1e5 for this checkpoint on the CPU engine too, so it cannot
/// discriminate.)
/// </summary>
/// <remarks>
/// Pre-existing finding recorded here: on a ~167-token real-prose prompt the SCALAR Vulkan prefill already disagrees with CPU (max|logit
/// diff| ~16, different argmax), so the comparison against CPU is only meaningful on the shorter prompts below, where scalar-Vulkan
/// tracks CPU. The grouped path is therefore gated on grouped-vs-scalar agreement plus (where the baseline is sound) CPU argmax.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class Gemma4VulkanGroupedMoePrefillTests
{
    private static FixtureLocation Gemma4Fixture => KnownTestFixtures.Gemma4_26B_A4B_Q4KM;

    private const string Prompt =
        "The history of the city of Lisbon stretches back more than two thousand years. Founded by the Phoenicians as a trading post, it was "
        + "later ruled by the Romans, the Visigoths and the Moors before being captured by Christian forces in 1147. In 1755 a devastating "
        + "earthquake, followed by a tsunami and fires, destroyed much of the city, and the Marquis of Pombal oversaw its rebuilding on a "
        + "grid of broad streets. Today Lisbon is the capital of Portugal, a centre of finance, tourism and the arts, and its hills, "
        + "tiled facades and historic trams draw visitors from around the world. The river Tagus widens into a broad estuary beside the "
        + "city, and the old quarter of Alfama, with its narrow lanes, still preserves the layout of the medieval town. In summary, the";

    private readonly ITestOutputHelper _output;

    public Gemma4VulkanGroupedMoePrefillTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public unsafe void Gemma4_26B_GroupedMoePrefill_MatchesScalarIndexedPrefill()
    {
        FixtureLocation fixture = Gemma4Fixture;
        Skip.If(!fixture.Found, fixture.SkipMessage(KnownTestFixtures.Gemma4_26BDescription));
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available on this host.");
        string spvDir = ResolveSpvDir();
        bool cpuOracle = Environment.GetEnvironmentVariable("DOTLLM_GEMMA4_CPU_ORACLE") != "0";

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Gemma4, config.Architecture);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
        int vocab = config.VocabSize;

        DotLLM.Core.Models.IModel? cpuModel = null;
        GgufFile? cpuGguf = null;
        if (cpuOracle)
        {
            var loaded = DotLLM.Models.ModelLoader.LoadFromGguf(fixture.Path!);
            cpuModel = loaded.Item1;
            cpuGguf = loaded.Item2;
        }

        try
        {
            // Prompt lengths in characters (cut at a word boundary): ~28, ~60, ~120 tokens and the full ~167.
            foreach (int chars in new[] { 130, 280, 560, Prompt.Length })
            {
                string text = chars >= Prompt.Length ? Prompt : Prompt[..Prompt.LastIndexOf(' ', chars)];
                int[] encoded = tokenizer.Encode(text);
                int[] ids = new int[encoded.Length + 1];
                ids[0] = tokenizer.BosTokenId;
                Array.Copy(encoded, 0, ids, 1, encoded.Length);
                int[] pos = new int[ids.Length];
                for (int i = 0; i < pos.Length; i++) pos[i] = i;

                float[] Prefill(bool grouped, out double ms)
                {
                    model.Gemma4GroupedMoeEnabled = grouped;
                    int before = model.Gemma4GroupedMoeDispatchCount;
                    using var kv = model.CreateKvCache(maxSeqLen: ids.Length + 4);
                    var sw = Stopwatch.StartNew();
                    using ITensor logits = model.Forward(ids, pos, deviceId: -1, kvCache: kv);
                    sw.Stop();
                    ms = sw.Elapsed.TotalMilliseconds;
                    int dispatches = model.Gemma4GroupedMoeDispatchCount - before;
                    if (grouped) Assert.True(dispatches > 0, "grouped path never ran: the comparison would be vacuous.");
                    else Assert.Equal(0, dispatches);
                    return LastRow(logits, vocab);
                }

                float[] scalar = Prefill(false, out double scalarMs);
                _ = Prefill(true, out _);   // warm-up (pipelines / descriptor sets)
                float[] grouped = Prefill(true, out double groupedMs);
                foreach (float v in grouped) Assert.True(float.IsFinite(v), "grouped path produced a non-finite logit (F16 operand overflow?).");

                float[]? cpu = null;
                if (cpuModel is not null)
                {
                    using ITensor cl = cpuModel.Forward(ids, pos, deviceId: -1, kvCache: null);
                    cpu = LastRow(cl, vocab);
                }

                (float max, double rms) Diff(float[] a, float[] b)
                {
                    float m = 0; double ss = 0;
                    for (int i = 0; i < vocab; i++) { float d = MathF.Abs(a[i] - b[i]); m = MathF.Max(m, d); ss += (double)d * d; }
                    return (m, Math.Sqrt(ss / vocab));
                }

                var gs = Diff(grouped, scalar);
                string line = $"[{ids.Length,4} tok] prefill ms scalar={scalarMs:F0} grouped={groupedMs:F0} ({scalarMs / groupedMs:F1}x)  "
                    + $"grouped-vs-scalar max={gs.max:G3} rms={gs.rms:G3}  top5 scalar={string.Join(",", TopK(scalar, 5))} grouped={string.Join(",", TopK(grouped, 5))}";
                if (cpu is not null)
                {
                    var sc = Diff(scalar, cpu);
                    var gc = Diff(grouped, cpu);
                    line += $"  | vs CPU: scalar max={sc.max:G3} rms={sc.rms:G3}, grouped max={gc.max:G3} rms={gc.rms:G3}, cpu top5={string.Join(",", TopK(cpu, 5))}";
                    _output.WriteLine(line);
                    // Where the scalar baseline picks CPU's token, the grouped path must too; and the grouped path may not be
                    // meaningfully FARTHER from CPU than the baseline already is (the baseline disagrees with CPU by rms 0.4-2 here:
                    // quantized-MoE router near-ties amplify any reduction-order change).
                    if (ArgMax(scalar) == ArgMax(cpu)) Assert.Equal(ArgMax(cpu), ArgMax(grouped));
                    Assert.True(gc.rms <= sc.rms * 1.25 + 0.05, $"grouped path is farther from CPU (rms {gc.rms:G3}) than scalar-Vulkan (rms {sc.rms:G3}) at {ids.Length} tokens");
                }
                else _output.WriteLine(line);

                // The 167-token case is report-only: there the scalar baseline itself is >3 rms from CPU (pre-existing, see remarks),
                // so grouped-vs-scalar there measures baseline instability, not the kernels.
                if (cpu is not null && Diff(scalar, cpu).rms > 3.0) continue;
                Assert.Equal(ArgMax(scalar), ArgMax(grouped));
                Assert.True(gs.max < 4f && gs.rms < 0.5,$"grouped-vs-scalar last-row logit drift max={gs.max:G3} rms={gs.rms:G3} at {ids.Length} tokens exceeds the F16-operand envelope");
            }

            // Greedy identity on a factual prompt the baseline answers coherently (>= 16 tokens so the grouped path engages).
            const string Factual = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after "
                + "the engineer Gustave Eiffel, whose company designed and built the tower. The Eiffel Tower is located in the city of";
            int[] fe = tokenizer.Encode(Factual);
            int[] fids = new int[fe.Length + 1];
            fids[0] = tokenizer.BosTokenId;
            Array.Copy(fe, 0, fids, 1, fe.Length);
            string[] texts = new string[2];
            for (int arm = 0; arm < 2; arm++)
            {
                model.Gemma4GroupedMoeEnabled = arm == 1;
                int before = model.Gemma4GroupedMoeDispatchCount;
                using var kv = model.CreateKvCache(maxSeqLen: fids.Length + 40);
                int[] fpos = Enumerable.Range(0, fids.Length).ToArray();
                int next;
                using (ITensor lg = model.Forward(fids, fpos, deviceId: -1, kvCache: kv)) next = ArgMax(LastRow(lg, vocab));
                Assert.Equal(arm == 1, model.Gemma4GroupedMoeDispatchCount > before);
                var gen = new List<int> { next };
                for (int s = 0; s < 24; s++)
                {
                    int[] one = { next }, onePos = { fids.Length + s };
                    using ITensor lg = model.Forward(one, onePos, deviceId: -1, kvCache: kv);
                    next = ArgMax(LastRow(lg, vocab));
                    gen.Add(next);
                }
                texts[arm] = tokenizer.Decode(gen.ToArray());
            }
            _output.WriteLine($"greedy scalar : '{texts[0]}'");
            _output.WriteLine($"greedy grouped: '{texts[1]}'");
            Assert.Contains("Paris", texts[0], StringComparison.OrdinalIgnoreCase);
            Assert.Equal(texts[0], texts[1]);
        }
        finally
        {
            cpuModel?.Dispose();
            cpuGguf?.Dispose();
        }
    }

    private static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        var row = new float[vocab];
        new ReadOnlySpan<float>((float*)logits.DataPointer + (total - vocab), vocab).CopyTo(row);
        return row;
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static int[] TopK(float[] v, int k)
        => Enumerable.Range(0, v.Length).OrderByDescending(i => v[i]).Take(k).ToArray();

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException("SPIR-V blobs not found. Run native/vulkan/build.sh.");
    }
}
