using System.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Diagnostic-only (issue #213): dumps per-token decode wall-clock time for both the
/// eager and CUDA-Graphs decode paths across a full depth sweep, so we can see the
/// SHAPE of the eager-vs-graph divergence (smooth growth vs. step function at TILE_KV=256
/// boundaries) rather than just aggregate tok/s at a handful of sampled depths. Not part
/// of the regular correctness/perf gate — run manually via
/// `dotnet test --filter FullyQualifiedName~CudaGraphDecodeDepthProfileTest`.
/// </summary>
/// <remarks>
/// #338: this is a profiling harness, not a gate. It was nonetheless collected by the default
/// unit run, where it executed 2 x 1100 decode steps on a 2B model plus two full model loads and
/// could only "fail" by timeout or exception. It now carries <c>Category=Benchmark</c> alongside
/// <c>Category=GPU</c>, so the documented exclusions (CI's
/// <c>Category!=GPU&amp;Category!=Benchmark</c>) drop it, and it is run deliberately or not at all.
/// It also no longer mislabels its own output: <c>BitNetGraphCaptureMaxDepth</c> defaults to 384,
/// so every row past that depth was eager-vs-eager while the column said "graph". The sweep now
/// records actual graph engagement per step from <see cref="CudaTransformerModel.GraphReplayCount"/>
/// and reports it per block.
/// </remarks>
[Trait("Category", "GPU")]
[Trait("Category", "Benchmark")]
public class CudaGraphDecodeDepthProfileTest
{
    private readonly ITestOutputHelper _out;

    public CudaGraphDecodeDepthProfileTest(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public unsafe void DepthSweep_EagerVsGraph_PerTokenMs()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        // #338: was a hard-coded E:/ snapshot path (duplicated in the equivalence suite, so the
        // two literals drifted independently) — green-by-skip on every machine but one.
        FixtureLocation fixture = KnownTestFixtures.BitNetI2S;
        Skip.If(!fixture.Found, fixture.SkipMessage(KnownTestFixtures.BitNetI2SDescription));

        using var gguf = GgufFile.Open(fixture.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] prompt = tokenizer.Encode(
            "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. "
            + "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.");
        _out.WriteLine($"Prompt tokens: {prompt.Length}");
        _out.WriteLine($"HiddenSize={config.HiddenSize} NumLayers={config.NumLayers} " +
                        $"NumHeads={config.NumAttentionHeads} NumKvHeads={config.NumKvHeads} HeadDim={config.HeadDim}");

        const int totalDecodeSteps = 1100; // covers depth up to ~1100 past the prompt
        int kvCap = prompt.Length + totalDecodeSteps + 8;

        double[] eagerMs = new double[totalDecodeSteps];
        double[] graphMs = new double[totalDecodeSteps];
        // #338: per-step record of whether graph replay ACTUALLY happened. Beyond
        // BitNetGraphCaptureMaxDepth (384 by default) the dispatch gate falls back to eager, so
        // without this the "graph" column silently becomes a second eager column exactly where
        // the sweep gets interesting.
        bool[] graphEngaged = new bool[totalDecodeSteps];

        // === Eager run ===
        using (var model = CudaTransformerModel.LoadFromGguf(gguf, config))
        using (var kv = model.CreateKvCache(kvCap))
        {
            model.UseGraphCapture = false;
            int[] positions = new int[prompt.Length];
            for (int i = 0; i < prompt.Length; i++) positions[i] = i;
            using (var _ = model.Forward(prompt, positions, 0, kv)) { }

            int curTok = prompt[^1];
            int[] tokBuf = new int[1];
            int[] posBuf = new int[1];
            var sw = new Stopwatch();
            for (int i = 0; i < totalDecodeSteps; i++)
            {
                tokBuf[0] = curTok;
                posBuf[0] = prompt.Length + i;
                sw.Restart();
                using var t = model.Forward(tokBuf, posBuf, 0, kv);
                sw.Stop();
                eagerMs[i] = sw.Elapsed.TotalMilliseconds;
                curTok = ArgMax((float*)t.DataPointer, config.VocabSize);
            }
        }

        // === Graph run ===
        using (var model = CudaTransformerModel.LoadFromGguf(gguf, config))
        using (var kv = model.CreateKvCache(kvCap))
        {
            model.UseGraphCapture = true;
            int[] positions = new int[prompt.Length];
            for (int i = 0; i < prompt.Length; i++) positions[i] = i;
            using (var _ = model.Forward(prompt, positions, 0, kv)) { }

            int curTok = prompt[^1];
            int[] tokBuf = new int[1];
            int[] posBuf = new int[1];
            var sw = new Stopwatch();
            for (int i = 0; i < totalDecodeSteps; i++)
            {
                tokBuf[0] = curTok;
                posBuf[0] = prompt.Length + i;
                int replaysBefore = model.GraphReplayCount;
                sw.Restart();
                using var t = model.Forward(tokBuf, posBuf, 0, kv);
                sw.Stop();
                graphMs[i] = sw.Elapsed.TotalMilliseconds;
                graphEngaged[i] = model.GraphReplayCount > replaysBefore;
                curTok = ArgMax((float*)t.DataPointer, config.VocabSize);
            }
        }

        // Dump per-token ms, grouped in blocks of 32 steps (median per block to denoise),
        // annotated with distance to nearest TILE_KV=256 boundary.
        int engagedSteps = graphEngaged.Count(x => x);
        _out.WriteLine($"graph replay engaged on {engagedSteps}/{totalDecodeSteps} steps "
                        + $"(BitNetGraphCaptureMaxDepth={CudaTransformerModel.BitNetGraphCaptureMaxDepth})");
        Assert.True(engagedSteps > 0,
            "graph replay never engaged on ANY step, so the entire 'graph' column is a second eager "
            + "column and this profile measures nothing. Check the kv-write kernel / PTX and the "
            + "dispatch gate in CudaTransformerModel.Forward (#338).");

        // `graph_engaged` says whether the block's steps actually replayed a captured graph; a
        // block marked eager is eager-vs-eager and its delta_ms is noise, not a graph measurement.
        _out.WriteLine("depth,eager_median_ms,graph_median_ms,delta_ms,dist_to_256_boundary,graph_engaged");
        const int block = 32;
        for (int start = 0; start < totalDecodeSteps; start += block)
        {
            int end = Math.Min(start + block, totalDecodeSteps);
            int n = end - start;
            double[] eBlk = new double[n];
            double[] gBlk = new double[n];
            for (int i = 0; i < n; i++) { eBlk[i] = eagerMs[start + i]; gBlk[i] = graphMs[start + i]; }
            Array.Sort(eBlk); Array.Sort(gBlk);
            double eMed = eBlk[n / 2];
            double gMed = gBlk[n / 2];
            int depth = prompt.Length + start;
            int distTo256 = depth % 256;
            int engagedInBlock = 0;
            for (int i = 0; i < n; i++) if (graphEngaged[start + i]) engagedInBlock++;
            string engaged = engagedInBlock == n ? "graph"
                : engagedInBlock == 0 ? "eager(fallback)"
                : $"mixed({engagedInBlock}/{n})";
            _out.WriteLine($"{depth},{eMed:F4},{gMed:F4},{(gMed - eMed):F4},{distTo256},{engaged}");
        }
    }

    private static unsafe int ArgMax(float* data, int n)
    {
        int best = 0;
        for (int i = 1; i < n; i++)
            if (data[i] > data[best]) best = i;
        return best;
    }
}
