using System.Diagnostics;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

/// <summary>
/// #1362 profile probe for the Qwen3.6-35B-A3B (qwen35moe) host: wall ms of an N-row forward per perturbation arm (interleaved reps),
/// then per-stage split-submit times. Usage: moestage &lt;gguf&gt; &lt;rows,rows,..&gt; &lt;reps&gt; [arms=base,scatter,shared,base]
/// </summary>
internal static class MoeVerifyStage
{
    public static void Run(string[] a)
    {
        string gguf = a[1];
        int[] rowSet = a[2].Split(',').Select(int.Parse).ToArray();
        int reps = int.Parse(a[3]);
        string[] arms = (a.Length > 4 ? a[4] : "base,scatter,shared,base").Split(',');
        using var file = GgufFile.Open(gguf);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var device = VulkanDevice.Create();
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");
        using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(device, file, cfg, spv);
        Console.WriteLine("loaded");
        int V = cfg.VocabSize;
        int[] Ids(int n, int seed) { var r = new Random(seed); return Enumerable.Range(0, n).Select(_ => r.Next(1000, Math.Min(V, 100000))).ToArray(); }
        double Fwd(int rows, int seed)
        {
            var ids = Ids(rows, seed);
            var t0 = Stopwatch.GetTimestamp();
            using var lg = model.Forward(ids, Enumerable.Range(16, rows).ToArray(), 0);
            return Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        }
        for (int i = 0; i < 6; i++) { Fwd(1, i); Fwd(5, i); }   // clock ramp / warm
        foreach (int rows in rowSet)
        {
            var res = new Dictionary<string, List<double>>();
            for (int r = 0; r < reps; r++)
                for (int ai = 0; ai < arms.Length; ai++)
                {
                    // "base" = current default; "ovoff"/"ovon" force the #1452 shared-expert overlap off/on; scatter/shared are skip perturbations.
                    VulkanQwen3MoeHybridTransformerModel.SharedOverlapEnabled = arms[ai] != "ovoff";
                    VulkanQwen3MoeHybridTransformerModel.PerturbSkip = arms[ai] is "base" or "ovoff" or "ovon" ? null : arms[ai];
                    string key = arms[ai] + "#" + ai;
                    if (!res.TryGetValue(key, out var l)) res[key] = l = new();
                    l.Add(Fwd(rows, 100 + r));
                }
            VulkanQwen3MoeHybridTransformerModel.PerturbSkip = null;
            VulkanQwen3MoeHybridTransformerModel.SharedOverlapEnabled = true;
            foreach (var kv in res) { var s = kv.Value.OrderBy(x => x).ToList(); Console.WriteLine($"rows={rows} arm={kv.Key,-10} min {s[0]:F2} med {s[s.Count / 2]:F2} max {s[^1]:F2} ms"); }
            VulkanQwen3MoeHybridTransformerModel.StageProfileEnabled = true;
            model.TakeStageTimes();
            Fwd(rows, 1); model.TakeStageTimes();
            var sums = new Dictionary<string, double>(); double tot = 0;
            for (int r = 0; r < reps; r++)
            {
                tot += Fwd(rows, 200 + r);
                foreach (var kv in model.TakeStageTimes()) sums[kv.Key] = sums.GetValueOrDefault(kv.Key) + kv.Value;
            }
            VulkanQwen3MoeHybridTransformerModel.StageProfileEnabled = false;
            Console.WriteLine($"rows={rows} STAGES forward {tot / reps:F1} ms, stages sum {sums.Values.Sum() / reps:F1}");
            foreach (var kv in sums.OrderByDescending(k => k.Value)) Console.WriteLine($"   {kv.Key,-22} {kv.Value / reps,8:F2} ms");
        }
    }
}
