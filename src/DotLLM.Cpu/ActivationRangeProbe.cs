using System.Collections.Concurrent;
using System.Globalization;

namespace DotLLM.Cpu;

/// <summary>
/// Env-gated (<c>DOTLLM_AMAX_PROBE=&lt;tsv path&gt;</c>) per-layer, per-site activation range recorder for the CPU forward.
/// Records <c>max|x|</c> (what an F16 copy must hold) and the largest 32-element block sum <c>|Σx|</c> (what a Q8_1 block's
/// F16 <c>s</c> must hold), so GPU paths that store activations as F16 / Q8_1 can be checked against the 65504 ceiling (#1191).
/// Writes one TSV row per (layer, site) at process exit. Disabled cost = one static readonly bool check per site.
/// </summary>
public static class ActivationRangeProbe
{
    /// <summary>True when <c>DOTLLM_AMAX_PROBE</c> names an output file.</summary>
    public static readonly bool On = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DOTLLM_AMAX_PROBE"));

    private sealed class Stat
    {
        public float Amax;
        public float BlockSumMax;
        public long Elements;
        public long NonFinite;
        public long Over16K;
    }

    private static readonly ConcurrentDictionary<(int Layer, string Site), Stat> Stats = new();

    static ActivationRangeProbe() { if (On) AppDomain.CurrentDomain.ProcessExit += (_, _) => Dump(); }

    /// <summary>Records the range of <paramref name="n"/> contiguous values at <paramref name="data"/>.</summary>
    /// <param name="layer">Layer index (-1 for model-level sites).</param>
    /// <param name="site">Site name, e.g. <c>ffn_down_in</c>.</param>
    /// <param name="data">Activation values.</param>
    /// <param name="n">Element count; rows must be multiples of 32 for the block sums to line up with Q8_1 blocks.</param>
    public static unsafe void Record(int layer, string site, float* data, long n)
    {
        if (!On || n <= 0) return;
        float amax = 0f, blockMax = 0f;
        long nonFinite = 0, over = 0;
        for (long b = 0; b < n; b += 32)
        {
            var block = new ReadOnlySpan<float>(data + b, (int)Math.Min(32, n - b));
            float sum = 0f;
            foreach (float v in block)
            {
                if (!float.IsFinite(v)) { nonFinite++; continue; }
                float a = MathF.Abs(v);
                if (a > amax) amax = a;
                if (a > 16384f) over++;
                sum += v;
            }
            blockMax = MathF.Max(blockMax, MathF.Abs(sum));
        }
        var s = Stats.GetOrAdd((layer, site), _ => new Stat());
        lock (s)
        {
            s.Amax = MathF.Max(s.Amax, amax);
            s.BlockSumMax = MathF.Max(s.BlockSumMax, blockMax);
            s.Elements += n;
            s.NonFinite += nonFinite;
            s.Over16K += over;
        }

        // Long runs: rewrite the table every minute so a killed or still-running probe leaves data behind.
        long now = Environment.TickCount64;
        long last = Interlocked.Read(ref _lastDump);
        if (now - last > 60_000 && Interlocked.CompareExchange(ref _lastDump, now, last) == last)
            Dump();
    }

    private static long _lastDump = Environment.TickCount64;

    private static void Dump()
    {
        lock (Stats)
        {
            DumpCore();
        }
    }

    private static void DumpCore()
    {
        string path = Environment.GetEnvironmentVariable("DOTLLM_AMAX_PROBE")!;
        using var w = new StreamWriter(path);
        w.WriteLine("layer\tsite\tamax\tblock_sum_max\telements\tnon_finite\tover_16k");
        foreach (var kv in Stats.OrderBy(k => k.Key.Layer).ThenBy(k => k.Key.Site, StringComparer.Ordinal))
        {
            var s = kv.Value;
            w.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{kv.Key.Layer}\t{kv.Key.Site}\t{s.Amax:G6}\t{s.BlockSumMax:G6}\t{s.Elements}\t{s.NonFinite}\t{s.Over16K}"));
        }
    }
}
