using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;

// Position-resolved logit dumper for the qwen4exp KL investigation (#873). The CPU oracle and the Vulkan model run the SAME token file in
// chunks of <rows>; for every sampled position the full-vocab logits go to <prefix>.logits.f32 ([samples][vocab]) and for EVERY position
// the target NLL goes to <prefix>.nll.f32 ([N-1], position i scores token i+1). Offline: benchmarks/Qwen4ExpRowProbe/klcmp.py.
internal static class KlCurve
{
    /// <summary>Samples positions p where p % stride == stride - 1 (so a chunk boundary aligned to the stride is not required).</summary>
    public static unsafe void Run(Func<int[], int[], ITensor> forward, int[] ids, string prefix, int n, int rows, int stride, int vocab, Action<string> log)
    {
        n = Math.Min(n, ids.Length);
        var nll = new float[n - 1];
        var samplePos = new List<int>();
        using var lg = new FileStream(prefix + ".logits.f32", FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 20);
        for (int p0 = 0; p0 < n; p0 += rows)
        {
            int m = Math.Min(rows, n - p0);
            var chunk = ids.AsSpan(p0, m).ToArray();
            var pos = Enumerable.Range(p0, m).ToArray();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            using ITensor t = forward(chunk, pos);
            if (t.Shape[0] != m) throw new InvalidOperationException($"forward returned {t.Shape[0]} rows for {m} (all-rows limit not raised?)");
            for (int r = 0; r < m; r++)
            {
                int p = p0 + r;
                var row = new ReadOnlySpan<float>((void*)(t.DataPointer + (nint)((long)r * vocab * 4)), vocab);
                if (p + 1 < n) nll[p] = (float)-TargetLogProb(row, ids[p + 1]);
                if (p % stride == stride - 1)
                {
                    lg.Write(MemoryMarshal.AsBytes(row));
                    samplePos.Add(p);
                }
            }
            log($"curve chunk @{p0} rows={m}: {sw.Elapsed.TotalSeconds:F1} s");
        }
        File.WriteAllBytes(prefix + ".nll.f32", MemoryMarshal.AsBytes(nll.AsSpan()).ToArray());
        File.WriteAllLines(prefix + ".pos.txt", samplePos.Select(x => x.ToString()));
    }

    private static double TargetLogProb(ReadOnlySpan<float> row, int target)
    {
        double max = double.MinValue;
        foreach (var v in row) if (v > max) max = v;
        double z = 0;
        foreach (var v in row) z += Math.Exp(v - max);
        return row[target] - max - Math.Log(z);
    }
}
