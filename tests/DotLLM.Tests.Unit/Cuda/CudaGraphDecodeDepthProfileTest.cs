using System.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
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
[Trait("Category", "GPU")]
public class CudaGraphDecodeDepthProfileTest
{
    private readonly ITestOutputHelper _out;

    public CudaGraphDecodeDepthProfileTest(ITestOutputHelper output) => _out = output;

    private const string ModelPath =
        "E:/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T-gguf/snapshots/a1f2f1c765812aa8af3f6eda4a313707064bba15/ggml-model-i2_s.gguf";

    [SkippableFact]
    public unsafe void DepthSweep_EagerVsGraph_PerTokenMs()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(!File.Exists(ModelPath), $"BitNet-2B-4T GGUF not found at {ModelPath}");

        using var gguf = GgufFile.Open(ModelPath);
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
                sw.Restart();
                using var t = model.Forward(tokBuf, posBuf, 0, kv);
                sw.Stop();
                graphMs[i] = sw.Elapsed.TotalMilliseconds;
                curTok = ArgMax((float*)t.DataPointer, config.VocabSize);
            }
        }

        // Dump per-token ms, grouped in blocks of 32 steps (median per block to denoise),
        // annotated with distance to nearest TILE_KV=256 boundary.
        _out.WriteLine("depth,eager_median_ms,graph_median_ms,delta_ms,dist_to_256_boundary");
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
            _out.WriteLine($"{depth},{eMed:F4},{gMed:F4},{(gMed - eMed):F4},{distTo256}");
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
