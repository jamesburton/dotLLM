using System.Globalization;

namespace ChunkKvLab;

/// <summary>
/// Chunk-KV reuse lab (research harness, CPU).
/// <code>
/// ChunkKvLab e1 --model &lt;gguf&gt; --corpus &lt;dir&gt; --seeds 3 --out results.csv [--sizes 2048,4096,8192] [--pattern *.cs]
///               [--chunk 512] [--continuation 64] [--greedy 16] [--threads 0]
/// ChunkKvLab rope-precision --model &lt;gguf&gt; --corpus &lt;dir&gt; --out rope.csv [--deltas 1024,4096,8192] [--chunks 3]
/// </code>
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.Error.WriteLine("usage: ChunkKvLab e1|rope-precision --model <gguf> --corpus <dir> --out <csv> [options]");
            return 2;
        }
        var opts = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 1; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Length)
            {
                Console.Error.WriteLine($"bad argument '{args[i]}'");
                return 2;
            }
            opts[args[i][2..]] = args[++i];
        }

        string Get(string key, string? fallback = null) =>
            opts.TryGetValue(key, out string? v) ? v : fallback ?? throw new ArgumentException($"--{key} is required");
        int Int(string key, int fallback) => opts.TryGetValue(key, out string? v) ? int.Parse(v, CultureInfo.InvariantCulture) : fallback;
        int[] Ints(string key, string fallback) =>
            Get(key, fallback).Split(',', StringSplitOptions.RemoveEmptyEntries).Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();

        switch (args[0])
        {
            case "e1":
                return E1Runner.Run(new E1Options(
                    Get("model"), Get("corpus"), Get("pattern", "*.cs"), Int("seeds", 3), Ints("sizes", "2048,4096,8192"),
                    Int("chunk", 512), Int("continuation", 64), Int("greedy", 16), Int("threads", 0), Get("out")));
            case "rope-precision":
                return RopePrecision.Run(Get("model"), Get("corpus"), Get("pattern", "*.cs"), Ints("deltas", "1024,4096,8192"),
                    Int("chunks", 3), Int("chunk", 512), Int("threads", 0), Get("out"));
            default:
                Console.Error.WriteLine($"unknown command '{args[0]}'");
                return 2;
        }
    }
}
