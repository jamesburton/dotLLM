using System.Diagnostics;
using DotLLM.ChunkReuse;
using DotLLM.Core.Attention;

namespace ChunkKvLab;

/// <summary>
/// <c>backend-check</c>: smoke test of the lab's backend wiring on a real model. Loads the model on <c>--backend</c>, prefills
/// <c>--tokens</c> ids into a cache from the chunk backend, asks for all-row logits, places the first half of the rows again at the
/// end with the backend's row IO plus a key shift, and prints shapes and timings. No quality claim; Task 2 measures the floor.
/// </summary>
internal static class BackendCheck
{
    public static int Run(string modelPath, int tokens, int threads)
    {
        using LabModel lab = LabModel.Load(modelPath, threads);
        int[] ids = Enumerable.Range(0, tokens).Select(i => (i * 131 + 17) % 5000 + 100).ToArray();
        int half = tokens / 2;
        using IKvCache cache = lab.NewBackendCache(tokens + half);

        var sw = Stopwatch.StartNew();
        float[] logits = lab.ForwardLogits(cache, ids, 0, allRows: true);
        double fwdMs = sw.Elapsed.TotalMilliseconds;
        int rows = logits.Length / lab.Config.VocabSize;
        Console.WriteLine($"backend {lab.Backend.Tag}: forward {tokens} tokens, {rows} logit rows, cache length {cache.CurrentLength}, {fwdMs:F1} ms");

        // Copy rows [0, half) to [tokens, tokens + half) through the backend and shift their keys by `tokens`.
        sw.Restart();
        for (int l = 0; l < lab.Config.NumLayers; l++)
        {
            int stride = lab.Geometry.KvStrideOf(l);
            var k = new float[half * stride];
            var v = new float[half * stride];
            lab.Backend.ReadRows(cache, l, 0, half, k, v);
            lab.Backend.WriteRows(cache, l, tokens, k, v);
        }
        lab.Backend.ShiftKeys(cache, tokens, half, tokens, RopeParameters.FromConfig(lab.Config));
        Console.WriteLine($"backend {lab.Backend.Tag}: row IO + shift of {half} rows x {lab.Config.NumLayers} layers in {sw.Elapsed.TotalMilliseconds:F1} ms; "
            + $"cache length {cache.CurrentLength}");
        return rows == tokens && cache.CurrentLength == tokens + half ? 0 : 1;
    }
}
