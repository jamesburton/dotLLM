using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Vision;
using DotLLM.Models.Vision;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>Locations and helpers shared by the Inkling-2 vision tests (issue #2035).</summary>
internal static class Inkling2VisionFixtures
{
    internal static string OracleDir => Path.Combine(Inkling2RealModelTests.OracleDirPath, "vision");

    internal static string? FindMmproj()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_MMPROJ_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string hub = Environment.GetEnvironmentVariable("HF_HUB_CACHE")
            ?? Path.Combine(Environment.GetEnvironmentVariable("HF_HOME") is { Length: > 0 } h ? h : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface"), "hub");
        string snaps = Path.Combine(hub, "models--ggml-org--Inkling-2-Mini-GGUF", "snapshots");
        if (!Directory.Exists(snaps)) return null;
        foreach (var d in Directory.GetDirectories(snaps))
        {
            string p = Path.Combine(d, "mmproj-Inkling-2-Mini-Q8_0.gguf");
            if (File.Exists(p)) return p;
        }
        return null;
    }

    /// <summary>The <c>llama-mtmd-debug -p rainbow</c> test image: float RGB in [0,1] (not normalised), square, float32 math like the C++ source.</summary>
    internal static float[] Rainbow(int n)
    {
        var img = new float[n * n * 3];
        float cx = n / 2.0f, cy = n / 2.0f;
        float maxDist = MathF.Sqrt(cx * cx + cy * cy);
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                float dx = x - cx, dy = y - cy;
                float hue = MathF.Atan2(dy, dx) / (2.0f * 3.14159265f);
                if (hue < 0) hue += 1.0f;
                float sat = MathF.Sqrt(dx * dx + dy * dy) / maxDist;
                if (sat > 1.0f) sat = 1.0f;
                float h6 = hue * 6.0f;
                int i6 = (int)h6;
                float f = h6 - i6;
                float p = 1.0f - sat, q = 1.0f - sat * f, t = 1.0f - sat * (1.0f - f);
                float r, g, b;
                switch (i6 % 6)
                {
                    case 0: r = 1; g = t; b = p; break;
                    case 1: r = q; g = 1; b = p; break;
                    case 2: r = p; g = 1; b = t; break;
                    case 3: r = p; g = q; b = 1; break;
                    case 4: r = t; g = p; b = 1; break;
                    default: r = 1; g = p; b = q; break;
                }
                int o = (y * n + x) * 3;
                img[o] = r; img[o + 1] = g; img[o + 2] = b;
            }
        return img;
    }

    internal sealed record Checkpoint(string Name, int[] Ne, double? Sum, Dictionary<string, double> Vals);

    internal static Dictionary<string, Checkpoint>? LoadTrace(string file)
    {
        string path = Path.Combine(OracleDir, file);
        if (!File.Exists(path)) return null;
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var res = new Dictionary<string, Checkpoint>();
        foreach (var e in doc.RootElement.EnumerateArray())
        {
            var ne = e.GetProperty("ne").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            var vals = new Dictionary<string, double>();
            foreach (var p in e.GetProperty("vals").EnumerateObject()) vals[p.Name] = p.Value.GetDouble();
            double? sum = e.GetProperty("sum").ValueKind == JsonValueKind.Null ? null : e.GetProperty("sum").GetDouble();
            res[e.GetProperty("name").GetString()!] = new Checkpoint(e.GetProperty("name").GetString()!, ne, sum, vals);
        }
        return res;
    }

    /// <summary>Per-checkpoint comparison of a captured tensor against the trimmed llama.cpp dump.</summary>
    internal sealed record CheckpointDiff(string Name, double MaxCornerErr, double MaxCornerRel, double SumErr, double SumScale, int Corners);

    internal static CheckpointDiff Compare(Checkpoint oracle, float[] mine)
    {
        long ne0 = oracle.Ne[0], ne1 = oracle.Ne[1], ne2 = oracle.Ne[2];
        double maxAbs = 0, maxRel = 0;
        double scale = 0;
        foreach (var v in oracle.Vals.Values) scale = Math.Max(scale, Math.Abs(v));
        foreach (var (key, want) in oracle.Vals)
        {
            var c = key.Split(',').Select(int.Parse).ToArray();
            long flat = ((c[0] * ne2 + c[1]) * ne1 + c[2]) * ne0 + c[3];
            double err = Math.Abs(mine[flat] - want);
            maxAbs = Math.Max(maxAbs, err);
        }
        maxRel = scale > 0 ? maxAbs / scale : maxAbs;
        double sum = 0, sq = 0;
        foreach (var v in mine) { sum += v; sq += (double)v * v; }
        double sumScale = Math.Sqrt(sq * mine.Length);   // N * rms: the scale a sum of N noisy values can reach
        return new CheckpointDiff(oracle.Name, maxAbs, maxRel, oracle.Sum is { } s ? Math.Abs(sum - s) : double.NaN, sumScale, oracle.Vals.Count);
    }

    internal static Dictionary<string, float[]> Capture(Inkling2VisionEncoder enc, Action run)
    {
        var captured = new Dictionary<string, float[]>();
        enc.Trace = (name, data, rows, cols) => captured[name] = data.ToArray();
        try { run(); } finally { enc.Trace = null; }
        return captured;
    }
}
